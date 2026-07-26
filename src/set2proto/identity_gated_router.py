"""P1-2 identity-directed anchor-residual condition routing.

This module keeps P1-1's permutation-invariant local router, adds a learned
gate around the deterministic quality anchor, and optimizes the deployable
output against a train-only clean teacher gallery.  No teacher or gallery is
needed at inference.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate

from .continuous_residual_oracle import (
    HARD_SCENARIOS,
    quality_anchor,
    teacher_best_frame_oracle,
)
from .continuous_router import (
    BASELINE_METHODS,
    ContinuousLocalRouter,
    evaluate_continuous_router_split,
    measure_continuous_router_latency,
)
from .training import TokenTrainingDataset


GATED_ROUTER_METHOD = "identity_gated_anchor_residual_router"


def _canonical_sum(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    return torch.sort(value, dim=dim).values.sum(dim=dim)


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


@dataclass(frozen=True)
class TrainIdentityGallery:
    vectors: torch.Tensor
    identities: torch.Tensor
    clean_dataset_indices: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.identities.numel())

    def to(self, device: torch.device) -> "TrainIdentityGallery":
        return TrainIdentityGallery(
            vectors=self.vectors.to(device),
            identities=self.identities.to(device),
            clean_dataset_indices=self.clean_dataset_indices,
        )


def build_train_identity_gallery(
    dataset: TokenTrainingDataset,
) -> TrainIdentityGallery:
    """Build one clean disjoint-teacher embedding per train identity."""

    if getattr(dataset.synthetic, "split", None) != "train":
        raise ValueError("identity gallery must be fit on train split")
    if dataset._precomputed is None:
        dataset.precompute()
    assert dataset._precomputed is not None
    scenarios = dataset._precomputed["scenarios"]
    clean_indices = [
        index
        for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    if not clean_indices:
        raise ValueError("train split contains no clean gallery samples")
    identities = dataset._precomputed["target_identity"][clean_indices].long()
    unique, counts = torch.unique(identities, return_counts=True)
    if not bool((counts == 1).all().item()):
        raise ValueError(
            "train clean gallery must contain exactly one sample per identity"
        )
    order = torch.argsort(identities)
    ordered_indices = torch.tensor(clean_indices, dtype=torch.int64)[order]
    ordered_identities = identities[order]
    maps = dataset.targets["continuous_prototypes"][
        ordered_indices
    ].float()
    vectors = F.normalize(maps.mean(dim=1), dim=-1)
    if int(unique.numel()) != int(vectors.shape[0]):
        raise RuntimeError("train identity gallery cardinality mismatch")
    return TrainIdentityGallery(
        vectors=vectors,
        identities=ordered_identities,
        clean_dataset_indices=ordered_indices,
    )


class AnchorResidualIdentityRouter(nn.Module):
    """Quality-anchor-preserving router with a learned local residual gate."""

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        positions: int = 49,
        router_hidden_dim: int = 64,
        gate_hidden_dim: int = 32,
        dropout: float = 0.0,
        gate_initial_bias: float = -3.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.positions = int(positions)
        self.hidden_dim = int(router_hidden_dim)
        self.router = ContinuousLocalRouter(
            feature_dim=feature_dim,
            positions=positions,
            hidden_dim=router_hidden_dim,
            dropout=dropout,
        )
        gate_input_dim = 3 * feature_dim + 5
        self.gate_encoder = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(gate_hidden_dim),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
        )
        self.gate_output = nn.Linear(gate_hidden_dim, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.constant_(self.gate_output.bias, float(gate_initial_bias))

    def forward(
        self,
        condition: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        routed, weights, logits = self.router(condition, quality)
        anchor, _ = quality_anchor(condition, quality)
        frames = int(condition.shape[1])
        local = F.normalize(condition.float(), dim=-1).permute(0, 2, 1, 3)
        local_mean = F.normalize(
            _canonical_sum(local, dim=2) / float(frames),
            dim=-1,
            eps=1e-6,
        )
        local_consensus = (
            local * local_mean.unsqueeze(2)
        ).sum(dim=-1).mean(dim=2)
        normalized_entropy = -(
            weights.clamp_min(1e-8).log() * weights
        ).sum(dim=-1) / math.log(float(frames))
        quality_value = quality.float().clamp_min(1e-6)
        quality_share = quality_value / _canonical_sum(
            quality_value,
            dim=1,
        ).unsqueeze(-1).clamp_min(1e-8)
        quality_concentration = quality_share.max(dim=1).values
        anchor_routed_cosine = (anchor * routed).sum(dim=-1)
        residual_norm = (routed - anchor).norm(dim=-1)
        scalar = torch.stack(
            (
                weights.max(dim=-1).values,
                1.0 - normalized_entropy,
                local_consensus,
                quality_concentration[:, None].expand(
                    -1,
                    self.positions,
                ),
                residual_norm,
            ),
            dim=-1,
        )
        gate_input = torch.cat(
            (
                anchor,
                routed,
                routed - anchor,
                scalar,
            ),
            dim=-1,
        )
        gate_logits = self.gate_output(
            self.gate_encoder(gate_input)
        ).squeeze(-1)
        gate = torch.sigmoid(gate_logits.float())
        output = F.normalize(
            anchor + gate.unsqueeze(-1) * (routed - anchor),
            dim=-1,
            eps=1e-6,
        )
        return (
            _require_finite("gated output", output),
            _require_finite("routing weights", weights),
            _require_finite("routing logits", logits.float()),
            _require_finite("residual gate", gate),
            _require_finite("routed map", routed),
            _require_finite("quality anchor", anchor),
        )


def build_identity_gated_router(
    config: Mapping[str, Any],
) -> AnchorResidualIdentityRouter:
    grid = config["data"]["feature_grid"]
    model = AnchorResidualIdentityRouter(
        feature_dim=int(grid["projected_dim"]),
        positions=int(grid["height"]) * int(grid["width"]),
        router_hidden_dim=int(config["p1_1"]["hidden_dim"]),
        gate_hidden_dim=int(config["p1_2"]["gate_hidden_dim"]),
        dropout=float(config["p1_1"]["dropout"]),
        gate_initial_bias=float(config["p1_2"]["gate_initial_bias"]),
    )
    count = sum(value.numel() for value in model.parameters())
    if count > int(config["p1_2"]["max_parameters"]):
        raise ValueError(
            f"identity gated router has {count} parameters, exceeding cap"
        )
    return model


def load_p1_1_router_initialization(
    *,
    model: AnchorResidualIdentityRouter,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    model.router.load_state_dict(payload["model_state"], strict=True)
    if bool(payload.get("teacher_used_at_inference", True)):
        raise ValueError("P1-1 initialization is not marked deployable")
    return payload


def _gallery_targets(
    identities: torch.Tensor,
    gallery: TrainIdentityGallery,
) -> torch.Tensor:
    targets = torch.searchsorted(gallery.identities, identities.long())
    if bool((targets >= gallery.size).any().item()):
        raise ValueError("training identity is absent from gallery")
    restored = gallery.identities.gather(0, targets)
    if not bool((restored == identities.long()).all().item()):
        raise ValueError("training identity is absent from gallery")
    return targets


def identity_gated_router_loss(
    *,
    model_outputs: Sequence[torch.Tensor],
    condition: torch.Tensor,
    quality: torch.Tensor,
    teacher_map: torch.Tensor,
    identities: torch.Tensor,
    gallery: TrainIdentityGallery,
    recipe: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    output, _, logits, gate, _, anchor = model_outputs
    with torch.no_grad():
        oracle_map, oracle = teacher_best_frame_oracle(
            condition,
            quality,
            teacher_map,
        )
        oracle_weights = oracle["weights"].clone()
        teacher = F.normalize(teacher_map.float(), dim=-1)
        oracle_improvement = (
            (oracle_map * teacher).sum(dim=-1)
            - (anchor.detach() * teacher).sum(dim=-1)
        )
        gate_target = (
            oracle_improvement
            > float(
                config["p1_2"][
                    "gate_teacher_improvement_epsilon"
                ]
            )
        ).float()

    distillation = -(
        oracle_weights * F.log_softmax(logits.float(), dim=-1)
    ).sum(dim=-1).mean()
    gate_logits_for_loss = torch.logit(
        gate.float().clamp(1e-6, 1.0 - 1e-6)
    )
    gate_target_loss = F.binary_cross_entropy_with_logits(
        gate_logits_for_loss,
        gate_target,
    )
    local_teacher = 1.0 - (
        F.normalize(output.float(), dim=-1) * teacher
    ).sum(dim=-1).mean()
    output_embedding = F.normalize(output.float().mean(dim=1), dim=-1)
    teacher_embedding = F.normalize(teacher.mean(dim=1), dim=-1)
    global_teacher = 1.0 - (
        output_embedding * teacher_embedding
    ).sum(dim=-1).mean()

    targets = _gallery_targets(identities, gallery)
    cosine_logits = output_embedding @ gallery.vectors.transpose(0, 1)
    gallery_cross_entropy = F.cross_entropy(
        cosine_logits / float(config["p1_2"]["gallery_temperature"]),
        targets,
    )
    positive = cosine_logits.gather(1, targets[:, None]).squeeze(1)
    negative_mask = torch.ones_like(cosine_logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets[:, None], False)
    strongest_impostor = cosine_logits.masked_fill(
        ~negative_mask,
        float("-inf"),
    ).max(dim=1).values
    gallery_margin = F.relu(
        float(config["p1_2"]["gallery_margin"])
        + strongest_impostor
        - positive
    ).mean()
    anchor_regularization = gate.mean()
    components = {
        "distillation_loss": distillation,
        "gate_target_loss": gate_target_loss,
        "local_teacher_loss": local_teacher,
        "global_teacher_loss": global_teacher,
        "gallery_cross_entropy_loss": gallery_cross_entropy,
        "gallery_margin_loss": gallery_margin,
        "anchor_regularization_loss": anchor_regularization,
    }
    total = sum(
        float(recipe[name.replace("_loss", "_weight")]) * value
        for name, value in components.items()
    )
    _require_finite("identity gated total loss", total)
    metrics = {
        "loss": float(total.detach().item()),
        **{
            name: float(value.detach().item())
            for name, value in components.items()
        },
        "mean_gate": float(gate.detach().mean().item()),
        "gate_target_fraction": float(gate_target.mean().item()),
        "gallery_positive_cosine": float(positive.mean().item()),
        "gallery_strongest_impostor_cosine": float(
            strongest_impostor.mean().item()
        ),
    }
    return total, metrics


@torch.inference_mode()
def evaluate_identity_gated_router_split(
    *,
    split: str,
    dataset: TokenTrainingDataset,
    model: AnchorResidualIdentityRouter,
    config: Mapping[str, Any],
    device: torch.device,
    method_name: str = GATED_ROUTER_METHOD,
) -> dict[str, Any]:
    result = evaluate_continuous_router_split(
        split=split,
        dataset=dataset,
        model=model,
        config=config,
        device=device,
        method_name=method_name,
    )
    batch_size = int(config["evaluation"]["batch_size"])
    scenarios: list[str] = []
    gates: list[torch.Tensor] = []
    residual_cosines: list[torch.Tensor] = []
    gate_target_agreement: list[torch.Tensor] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"].to(device)
        quality = batch["condition_quality"].to(device)
        teacher = dataset.targets["continuous_prototypes"][
            indices
        ].float().to(device)
        output, _, _, gate, _, anchor = model(condition, quality)
        oracle, _ = teacher_best_frame_oracle(
            condition,
            quality,
            teacher,
        )
        normalized_teacher = F.normalize(teacher, dim=-1)
        target = (
            (oracle * normalized_teacher).sum(dim=-1)
            - (anchor * normalized_teacher).sum(dim=-1)
            > float(
                config["p1_2"][
                    "gate_teacher_improvement_epsilon"
                ]
            )
        )
        gates.append(gate.cpu())
        residual_cosines.append(
            F.cosine_similarity(output, anchor, dim=-1).cpu()
        )
        gate_target_agreement.append(
            ((gate > 0.5) == target).float().cpu()
        )
        scenarios.extend(str(value) for value in batch["scenario"])
    all_gates = torch.cat(gates)
    all_anchor_cosines = torch.cat(residual_cosines)
    all_agreement = torch.cat(gate_target_agreement)
    scenario_array = np.asarray(scenarios)
    diagnostics: dict[str, Any] = {}
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        selected_gate = all_gates[mask]
        diagnostics[scenario] = {
            "samples": int(mask.sum().item()),
            "mean_gate": float(selected_gate.mean().item()),
            "gate_std": float(
                selected_gate.std(unbiased=False).item()
            ),
            "gate_p05": float(torch.quantile(selected_gate, 0.05).item()),
            "gate_p50": float(torch.quantile(selected_gate, 0.50).item()),
            "gate_p95": float(torch.quantile(selected_gate, 0.95).item()),
            "gate_over_half_fraction": float(
                (selected_gate > 0.5).float().mean().item()
            ),
            "output_anchor_cosine": float(
                all_anchor_cosines[mask].mean().item()
            ),
            "gate_target_accuracy": float(
                all_agreement[mask].mean().item()
            ),
        }
    diagnostics["all"] = {
        "mean_gate": float(all_gates.mean().item()),
        "gate_over_half_fraction": float(
            (all_gates > 0.5).float().mean().item()
        ),
        "output_anchor_cosine": float(all_anchor_cosines.mean().item()),
        "gate_target_accuracy": float(all_agreement.mean().item()),
    }
    result["gate_diagnostics"] = diagnostics
    result["teacher_used_at_inference"] = False
    result["gallery_used_at_inference"] = False
    return result


def _selection_row(
    *,
    recipe_name: str,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = result["methods"]["quality_pooling"]
    learned = result["methods"][GATED_ROUTER_METHOD]
    base_verification = baseline["verification"]
    verification = learned["verification"]
    thresholds = config["p1_2"]
    deltas = {
        scenario: (
            verification[scenario]["roc_auc"]
            - base_verification[scenario]["roc_auc"]
        )
        for scenario in HARD_SCENARIOS
    }
    hard_gain = (
        verification["hard_average"]["roc_auc"]
        - base_verification["hard_average"]["roc_auc"]
    )
    clean_drop = (
        base_verification["clean"]["roc_auc"]
        - verification["clean"]["roc_auc"]
    )
    rank_drop = (
        base_verification["all"]["rank1_accuracy"]
        - verification["all"]["rank1_accuracy"]
    )
    teacher_gain = (
        learned["teacher_similarity"]["all"]["map_cosine"]
        - baseline["teacher_similarity"]["all"]["map_cosine"]
    )
    wins = sum(delta > 0.0 for delta in deltas.values())
    correctness = result["correctness"]
    checks = {
        "minimum_hard_gain": (
            hard_gain
            >= float(thresholds["selection_min_hard_auc_gain"])
        ),
        "complementary_gain": (
            deltas["complementary_occlusion"]
            > float(
                thresholds["selection_min_complementary_auc_gain"]
            )
        ),
        "hard_scenario_wins": (
            wins
            >= int(thresholds["selection_min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            clean_drop
            <= float(thresholds["selection_clean_auc_max_drop"])
        ),
        "rank1_constraint": (
            rank_drop
            <= float(thresholds["selection_rank1_max_drop"])
        ),
        "teacher_map_headroom": (
            teacher_gain
            >= float(
                thresholds["selection_min_teacher_map_cosine_gain"]
            )
        ),
        "correctness": all(
            (
                correctness["output_shape_valid"],
                correctness["output_unit_norm_valid"],
                correctness["weight_sum_valid"],
                correctness["permutation_equivalent"],
            )
        ),
    }
    return {
        "recipe": recipe_name,
        "method": GATED_ROUTER_METHOD,
        "pooled_all_roc_auc": verification["all"]["roc_auc"],
        "hard_macro_roc_auc": verification["hard_average"]["roc_auc"],
        "clean_roc_auc": verification["clean"]["roc_auc"],
        "eer": verification["all"]["eer"],
        "rank1_accuracy": verification["all"]["rank1_accuracy"],
        "margin": verification["all"]["mean_genuine_impostor_margin"],
        "teacher_map_cosine": learned["teacher_similarity"]["all"][
            "map_cosine"
        ],
        "hard_gain_vs_quality": hard_gain,
        "teacher_map_cosine_gain_vs_quality": teacher_gain,
        "clean_auc_drop_vs_quality": clean_drop,
        "rank1_drop_vs_quality": rank_drop,
        "complementary_gain_vs_quality": deltas[
            "complementary_occlusion"
        ],
        "hard_scenario_wins": wins,
        "scenario_auc_deltas": deltas,
        "mean_gate": result["gate_diagnostics"]["all"]["mean_gate"],
        "checks": checks,
        "constraints_passed": all(checks.values()),
    }


def select_identity_gated_candidate(
    *,
    validation_results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    recipe_order = list(config["p1_2"]["candidate_recipes"])
    if set(validation_results) != set(recipe_order):
        raise ValueError("validation results do not match candidate recipes")
    rows = [
        _selection_row(
            recipe_name=recipe,
            result=validation_results[recipe],
            config=config,
        )
        for recipe in recipe_order
    ]
    feasible = [row for row in rows if row["constraints_passed"]]
    ordered = sorted(
        feasible if feasible else rows,
        key=lambda row: (
            -float(row["hard_macro_roc_auc"]),
            -float(row["pooled_all_roc_auc"]),
            -float(row["teacher_map_cosine"]),
            recipe_order.index(str(row["recipe"])),
        ),
    )
    return {
        "selection_split": "val",
        "test_metrics_consulted": False,
        "candidate_count": len(rows),
        "feasible_candidate_count": len(feasible),
        "test_authorized": bool(feasible),
        "selected": dict(ordered[0]) if feasible else None,
        "best_validation_candidate": dict(ordered[0]),
        "rows": rows,
        "thresholds": {
            key: value
            for key, value in config["p1_2"].items()
            if key.startswith("selection_")
        },
        "selection_rule": (
            "Require every preregistered identity/AUC/correctness gate, then "
            "select recipe and checkpoint by validation hard AUC, pooled-all "
            "AUC, and teacher-map cosine. Never use test for selection."
        ),
    }


def _learning_rate(
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    base: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base * step / float(warmup_steps)
    progress = min(
        1.0,
        (step - warmup_steps) / float(max(1, max_steps - warmup_steps)),
    )
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def _batch(
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    batch = default_collate([dataset[index] for index in indices])
    teacher = dataset.targets["continuous_prototypes"][
        list(indices)
    ].float().to(device)
    return batch, teacher


def save_identity_gated_checkpoint(
    *,
    path: Path,
    model: AnchorResidualIdentityRouter,
    optimizer: torch.optim.Optimizer,
    step: int,
    recipe_name: str,
    validation_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "stage": "P1-2",
            "step": step,
            "recipe": recipe_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "parameter_count": sum(
                value.numel() for value in model.parameters()
            ),
            "validation": validation_result,
            "p1_2_config": dict(config["p1_2"]),
            "teacher_used_at_inference": False,
            "gallery_used_at_inference": False,
        },
        temporary,
    )
    temporary.replace(path)


def load_identity_gated_checkpoint(
    *,
    path: str | Path,
    model: AnchorResidualIdentityRouter,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def train_identity_gated_candidate(
    *,
    recipe_name: str,
    train_dataset: TokenTrainingDataset,
    val_dataset: TokenTrainingDataset,
    gallery: TrainIdentityGallery,
    model: AnchorResidualIdentityRouter,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path_for_step: Callable[[int], Path],
    max_steps_override: int | None = None,
    metric_callback: Callable[
        [str, int, Mapping[str, float]],
        None,
    ]
    | None = None,
) -> dict[str, Any]:
    if recipe_name not in config["p1_2"]["candidate_recipes"]:
        raise ValueError("recipe is not preregistered")
    if getattr(train_dataset.synthetic, "split", None) != "train":
        raise ValueError("training dataset must be train")
    if getattr(val_dataset.synthetic, "split", None) != "val":
        raise ValueError("validation dataset must be val")
    recipe = config["p1_2"]["recipes"][recipe_name]
    section = config["p1_2"]
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else int(section["max_steps"])
    )
    smoke = max_steps_override is not None
    validate_every = (
        max_steps if smoke else int(section["validate_every_steps"])
    )
    minimum_steps = min(
        max_steps,
        1 if smoke else int(section["min_steps"]),
    )
    generator = torch.Generator().manual_seed(
        int(config["project"]["seed"])
        + 1200
        + list(config["p1_2"]["candidate_recipes"]).index(recipe_name)
    )
    model.to(device)
    gallery_device = gallery.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda" and section["precision"] == "fp16"
        ),
    )
    autocast_dtype = (
        torch.bfloat16
        if section["precision"] == "bf16"
        else torch.float16
    )
    permutation = torch.randperm(len(train_dataset), generator=generator)
    cursor = 0
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    best_score: tuple[float, float, float] | None = None
    best_checkpoint: Path | None = None
    best_step = 0
    patience = 0
    stopped_early = False
    start = time.perf_counter()
    batch_size = int(section["batch_size"])

    for step in range(1, max_steps + 1):
        if cursor + batch_size > len(permutation):
            permutation = torch.randperm(
                len(train_dataset),
                generator=generator,
            )
            cursor = 0
        indices = permutation[cursor : cursor + batch_size].tolist()
        cursor += batch_size
        batch, teacher = _batch(train_dataset, indices, device)
        condition = batch["condition_features"].to(device)
        quality = batch["condition_quality"].to(device)
        identities = batch["target_identity"].long().to(device)
        lr = _learning_rate(
            step=step,
            max_steps=max_steps,
            warmup_steps=min(
                int(section["warmup_steps"]),
                max(0, max_steps - 1),
            ),
            base=float(section["learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            outputs = model(condition, quality)
            loss, losses = identity_gated_router_loss(
                model_outputs=outputs,
                condition=condition,
                quality=quality,
                teacher_map=teacher,
                identities=identities,
                gallery=gallery_device,
                recipe=recipe,
                config=config,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(section["max_grad_norm"]),
            ).item()
        )
        scaler.step(optimizer)
        scaler.update()
        row = {
            "recipe": recipe_name,
            "step": step,
            **losses,
            "learning_rate": lr,
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        if metric_callback is not None:
            metric_callback(recipe_name, step, row)

        if step % validate_every == 0 or step == max_steps:
            validation = evaluate_identity_gated_router_split(
                split="val",
                dataset=val_dataset,
                model=model,
                config=config,
                device=device,
            )
            learned = validation["methods"][GATED_ROUTER_METHOD]
            score = (
                float(
                    learned["verification"]["hard_average"]["roc_auc"]
                ),
                float(learned["verification"]["all"]["roc_auc"]),
                float(
                    learned["teacher_similarity"]["all"]["map_cosine"]
                ),
            )
            validations.append(
                {
                    "recipe": recipe_name,
                    "step": step,
                    "score": list(score),
                    "selection": _selection_row(
                        recipe_name=recipe_name,
                        result=validation,
                        config=config,
                    ),
                }
            )
            if best_score is None or score > best_score:
                path = checkpoint_path_for_step(step)
                save_identity_gated_checkpoint(
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    recipe_name=recipe_name,
                    validation_result=validation,
                    config=config,
                )
                best_score = score
                best_checkpoint = path
                best_step = step
                patience = 0
            else:
                patience += 1
            model.train()
            if (
                not smoke
                and step >= minimum_steps
                and patience >= int(section["early_stop_patience"])
            ):
                stopped_early = True
                break
    if best_checkpoint is None:
        raise RuntimeError("candidate training produced no checkpoint")
    return {
        "recipe": recipe_name,
        "steps_completed": int(history[-1]["step"]),
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint),
        "best_score": list(best_score or ()),
        "stopped_early": stopped_early,
        "history": history,
        "validations": validations,
        "elapsed_seconds": time.perf_counter() - start,
        "parameter_count": sum(
            value.numel() for value in model.parameters()
        ),
        "teacher_used_at_inference": False,
        "gallery_used_at_inference": False,
    }


def test_gate(
    *,
    recipe_name: str,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    row = _selection_row(
        recipe_name=recipe_name,
        result=result,
        config=config,
    )
    return {
        "passed": bool(row["constraints_passed"]),
        "checks": row["checks"],
        "hard_gain_vs_quality": row["hard_gain_vs_quality"],
        "scenario_auc_deltas": row["scenario_auc_deltas"],
        "teacher_map_cosine_gain_vs_quality": row[
            "teacher_map_cosine_gain_vs_quality"
        ],
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tar(values: Mapping[str, Any], far: float) -> float:
    return float(values["tar_at_far"][f"{far:g}"]["tar"])


def write_identity_gated_artifacts(
    *,
    output_directory: str | Path,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    training_runs: Mapping[str, Mapping[str, Any]],
    validation_results: Mapping[str, Mapping[str, Any]],
    selected_validation_result: Mapping[str, Any],
    base_validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: Mapping[str, bool],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_recipe = str(
        selection["best_validation_candidate"]["recipe"]
    )
    split_results = [("val", selected_validation_result)]
    if test_result is not None:
        split_results.append(("test", test_result))
    core_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for split, result in split_results:
        for method in (*BASELINE_METHODS, GATED_ROUTER_METHOD):
            values = result["methods"][method]
            verification = values["verification"]
            core_rows.append(
                {
                    "split": split,
                    "recipe": (
                        selected_recipe
                        if method == GATED_ROUTER_METHOD
                        else ""
                    ),
                    "method": method,
                    "pooled_all_roc_auc": verification["all"][
                        "roc_auc"
                    ],
                    "hard_macro_roc_auc": verification["hard_average"][
                        "roc_auc"
                    ],
                    "eer": verification["all"]["eer"],
                    "tar_at_far_1e-2": _tar(
                        verification["all"],
                        1e-2,
                    ),
                    "tar_at_far_1e-3": _tar(
                        verification["all"],
                        1e-3,
                    ),
                    "rank1_accuracy": verification["all"][
                        "rank1_accuracy"
                    ],
                    "margin": verification["all"][
                        "mean_genuine_impostor_margin"
                    ],
                    "teacher_map_cosine": values[
                        "teacher_similarity"
                    ]["all"]["map_cosine"],
                }
            )
            for scenario, scenario_values in verification.items():
                if scenario in {"all", "hard_average"}:
                    continue
                scenario_rows.append(
                    {
                        "split": split,
                        "recipe": (
                            selected_recipe
                            if method == GATED_ROUTER_METHOD
                            else ""
                        ),
                        "method": method,
                        "scenario": scenario,
                        "roc_auc": scenario_values["roc_auc"],
                        "eer": scenario_values["eer"],
                        "tar_at_far_1e-2": _tar(
                            scenario_values,
                            1e-2,
                        ),
                        "tar_at_far_1e-3": _tar(
                            scenario_values,
                            1e-3,
                        ),
                    }
                )
    core_path = output / "core_comparison.csv"
    _write_csv(core_path, core_rows, list(core_rows[0]))
    scenario_path = output / "scenario_metrics.csv"
    _write_csv(scenario_path, scenario_rows, list(scenario_rows[0]))

    search_rows: list[dict[str, Any]] = []
    for row in selection["rows"]:
        search_rows.append(
            {
                "recipe": row["recipe"],
                "hard_macro_roc_auc": row["hard_macro_roc_auc"],
                "pooled_all_roc_auc": row["pooled_all_roc_auc"],
                "hard_gain_vs_quality": row["hard_gain_vs_quality"],
                "complementary_gain_vs_quality": row[
                    "complementary_gain_vs_quality"
                ],
                "hard_scenario_wins": row["hard_scenario_wins"],
                "clean_auc_drop_vs_quality": row[
                    "clean_auc_drop_vs_quality"
                ],
                "rank1_drop_vs_quality": row[
                    "rank1_drop_vs_quality"
                ],
                "teacher_map_cosine_gain_vs_quality": row[
                    "teacher_map_cosine_gain_vs_quality"
                ],
                "mean_gate": row["mean_gate"],
                "constraints_passed": row["constraints_passed"],
            }
        )
    search_path = output / "validation_candidate_search.csv"
    _write_csv(search_path, search_rows, list(search_rows[0]))
    history_rows = [
        dict(row)
        for recipe in config["p1_2"]["candidate_recipes"]
        for row in training_runs[recipe]["history"]
    ]
    history_path = output / "training_history.csv"
    _write_csv(history_path, history_rows, list(history_rows[0]))
    gate_rows: list[dict[str, Any]] = []
    for split, result in split_results:
        for scenario, values in result["gate_diagnostics"].items():
            gate_rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    **values,
                }
            )
    gate_path = output / "gate_diagnostics.csv"
    _write_csv(gate_path, gate_rows, list(gate_rows[0]))

    test_evaluation = (
        test_gate(
            recipe_name=selected_recipe,
            result=test_result,
            config=config,
        )
        if test_result is not None
        else None
    )
    decision = {
        "validation_gate_passed": bool(selection["test_authorized"]),
        "test_evaluated": test_result is not None,
        "test_gate": test_evaluation,
        "go": bool(
            selection["test_authorized"]
            and test_evaluation is not None
            and test_evaluation["passed"]
            and all(checks.values())
        ),
        "recommendation": (
            "proceed_to_new_holdout_confirmation_before_diffusion"
            if (
                selection["test_authorized"]
                and test_evaluation is not None
                and test_evaluation["passed"]
                and all(checks.values())
            )
            else "keep_quality_anchor_do_not_start_diffusion"
        ),
    }
    diagnostics = {
        "schema_version": 1,
        "preregistration": preregistration,
        "selection": selection,
        "training_runs": training_runs,
        "validation_results": validation_results,
        "base_p1_1_validation": base_validation_result,
        "test": test_result,
        "latency": latency,
        "checks": checks,
        "reproducibility": reproducibility,
        "decision": decision,
    }
    diagnostics_path = output / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    quality = selected_validation_result["methods"][
        "quality_pooling"
    ]["verification"]
    gated = selected_validation_result["methods"][
        GATED_ROUTER_METHOD
    ]["verification"]
    base = base_validation_result["methods"][
        "continuous_local_router"
    ]["verification"]
    locked_test_lines = (
        "| 未授权 | - | - | - |"
        if test_result is None
        else "\n".join(
            f"| {method} | "
            f"{test_result['methods'][method]['verification']['all']['roc_auc']:.6f} | "
            f"{test_result['methods'][method]['verification']['hard_average']['roc_auc']:.6f} | "
            f"{test_result['methods'][method]['verification']['all']['rank1_accuracy']:.6f} |"
            for method in ("quality_pooling", GATED_ROUTER_METHOD)
        )
    )
    report = f"""# P1-2 Identity-Gated Anchor Residual Router

## 结论

Validation 选择配方：`{selected_recipe}`。
Validation 门槛：**{'PASS' if selection['test_authorized'] else 'FAIL'}**。
最终结论：**{'GO' if decision['go'] else 'NO-GO'}**。

模型以 quality pooling 为安全锚点，通过逐位置 gate 注入路由残差。
train-only clean disjoint-teacher gallery 仅用于身份交叉熵和
hardest-impostor margin 训练；teacher/gallery 均不参与推理。

## Validation

| 方法 | pooled-all AUC | hard macro AUC | EER | Rank-1 |
|---|---:|---:|---:|---:|
| quality_pooling | {quality['all']['roc_auc']:.6f} | {quality['hard_average']['roc_auc']:.6f} | {quality['all']['eer']:.6f} | {quality['all']['rank1_accuracy']:.6f} |
| P1-1 local router | {base['all']['roc_auc']:.6f} | {base['hard_average']['roc_auc']:.6f} | {base['all']['eer']:.6f} | {base['all']['rank1_accuracy']:.6f} |
| P1-2 gated router | {gated['all']['roc_auc']:.6f} | {gated['hard_average']['roc_auc']:.6f} | {gated['all']['eer']:.6f} | {gated['all']['rank1_accuracy']:.6f} |

P1-2 hard AUC 相对 quality：
`{selection['best_validation_candidate']['hard_gain_vs_quality']:+.6f}`。
P1-2 teacher-map cosine 相对 quality：
`{selection['best_validation_candidate']['teacher_map_cosine_gain_vs_quality']:+.6f}`。
平均 residual gate：
`{selection['best_validation_candidate']['mean_gate']:.6f}`。

## Locked test

| 方法 | pooled-all AUC | hard macro AUC | Rank-1 |
|---|---:|---:|---:|
{locked_test_lines}

## 解释边界

- 两个损失配方和所有门槛在 test 前固定，只用 validation 选择。
- 当前 test 身份在既往阶段已经暴露，不是全新确认性 holdout。
- 单次 pilot 不代表统计显著；即使通过，也应先建立新身份 holdout，
  再决定是否进入 continuous residual diffusion。
"""
    report_path = output / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "core_comparison_csv": str(core_path),
        "scenario_metrics_csv": str(scenario_path),
        "validation_candidate_search_csv": str(search_path),
        "training_history_csv": str(history_path),
        "gate_diagnostics_csv": str(gate_path),
        "diagnostics_json": str(diagnostics_path),
        "decision": decision,
    }
