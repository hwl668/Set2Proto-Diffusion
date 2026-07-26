"""P1-3 bounded identity-agnostic scalar-evidence residual routing."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate

from .continuous_residual_oracle import (
    HARD_SCENARIOS,
    quality_anchor,
    teacher_best_frame_oracle,
)
from .continuous_router import ContinuousLocalRouter
from .identity_gated_router import (
    GATED_ROUTER_METHOD,
    TrainIdentityGallery,
    _gallery_targets,
    evaluate_identity_gated_router_split,
)
from .training import TokenTrainingDataset


SCALAR_EVIDENCE_METHOD = "bounded_scalar_evidence_router"
BASELINE_METHODS = ("mean_pooling", "quality_pooling")


def _canonical_sum(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    return torch.sort(value, dim=dim).values.sum(dim=dim)


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


class BoundedScalarEvidenceRouter(nn.Module):
    """Gate a frozen/slow router using identity-agnostic evidence scalars."""

    evidence_dim = 13

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        positions: int = 49,
        router_hidden_dim: int = 64,
        gate_hidden_dim: int = 32,
        dropout: float = 0.0,
        initial_gate: float = 0.05,
        maximum_gate: float = 0.35,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gate < maximum_gate <= 1.0:
            raise ValueError("gate values must satisfy 0 < initial < max <= 1")
        self.feature_dim = int(feature_dim)
        self.positions = int(positions)
        self.hidden_dim = int(router_hidden_dim)
        self.maximum_gate = float(maximum_gate)
        self.router = ContinuousLocalRouter(
            feature_dim=feature_dim,
            positions=positions,
            hidden_dim=router_hidden_dim,
            dropout=dropout,
        )
        self.gate_network = nn.Sequential(
            nn.Linear(self.evidence_dim, gate_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(gate_hidden_dim),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        final = self.gate_network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        probability = initial_gate / maximum_gate
        nn.init.constant_(
            final.bias,
            math.log(probability / (1.0 - probability)),
        )

    def _evidence(
        self,
        *,
        condition: torch.Tensor,
        quality: torch.Tensor,
        weights: torch.Tensor,
        logits: torch.Tensor,
        routed: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        frames = int(condition.shape[1])
        local = F.normalize(condition.float(), dim=-1).permute(0, 2, 1, 3)
        local_mean = F.normalize(
            _canonical_sum(local, dim=2) / float(frames),
            dim=-1,
            eps=1e-6,
        )
        frame_consensus = (
            local * local_mean.unsqueeze(2)
        ).sum(dim=-1)
        consensus_mean = frame_consensus.mean(dim=-1)
        consensus_std = frame_consensus.std(dim=-1, unbiased=False)
        consensus_range = (
            frame_consensus.max(dim=-1).values
            - frame_consensus.min(dim=-1).values
        )

        quality_value = quality.float().clamp_min(1e-6)
        quality_weights = quality_value / _canonical_sum(
            quality_value,
            dim=1,
        ).unsqueeze(-1).clamp_min(1e-8)
        quality_local = quality_weights[:, None, :].expand_as(weights)
        route_entropy = -(
            weights.clamp_min(1e-8).log() * weights
        ).sum(dim=-1) / math.log(float(frames))
        quality_entropy = -(
            quality_weights.clamp_min(1e-8).log() * quality_weights
        ).sum(dim=-1) / math.log(float(frames))
        ordered_logits = torch.sort(
            logits.float(),
            dim=-1,
            descending=True,
        ).values
        top_logit_gap = ordered_logits[..., 0] - ordered_logits[..., 1]
        route_quality_l1 = (
            weights - quality_local
        ).abs().mean(dim=-1)
        consensus_advantage = (
            (weights * frame_consensus).sum(dim=-1)
            - (quality_local * frame_consensus).sum(dim=-1)
        )

        frame_global = F.normalize(local.mean(dim=1), dim=-1)
        global_mean = F.normalize(
            _canonical_sum(frame_global, dim=1) / float(frames),
            dim=-1,
            eps=1e-6,
        )
        global_consensus = (
            frame_global * global_mean.unsqueeze(1)
        ).sum(dim=-1)
        global_consensus_std = global_consensus.std(
            dim=1,
            unbiased=False,
        )
        evidence = torch.stack(
            (
                weights.max(dim=-1).values,
                top_logit_gap,
                1.0 - route_entropy,
                route_quality_l1,
                consensus_mean,
                consensus_std,
                consensus_range,
                consensus_advantage,
                quality_weights.max(dim=1).values[:, None].expand(
                    -1,
                    self.positions,
                ),
                (1.0 - quality_entropy)[:, None].expand(
                    -1,
                    self.positions,
                ),
                (anchor * routed).sum(dim=-1),
                (routed - anchor).norm(dim=-1),
                global_consensus_std[:, None].expand(
                    -1,
                    self.positions,
                ),
            ),
            dim=-1,
        )
        return _require_finite("scalar gate evidence", evidence)

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
        evidence = self._evidence(
            condition=condition,
            quality=quality,
            weights=weights,
            logits=logits,
            routed=routed,
            anchor=anchor,
        )
        gate = self.maximum_gate * torch.sigmoid(
            self.gate_network(evidence).squeeze(-1).float()
        )
        output = F.normalize(
            anchor + gate.unsqueeze(-1) * (routed - anchor),
            dim=-1,
            eps=1e-6,
        )
        return (
            _require_finite("bounded scalar output", output),
            weights,
            logits.float(),
            _require_finite("bounded scalar gate", gate),
            routed,
            anchor,
        )


def build_scalar_evidence_router(
    config: Mapping[str, Any],
) -> BoundedScalarEvidenceRouter:
    grid = config["data"]["feature_grid"]
    section = config["p1_3"]
    model = BoundedScalarEvidenceRouter(
        feature_dim=int(grid["projected_dim"]),
        positions=int(grid["height"]) * int(grid["width"]),
        router_hidden_dim=int(config["p1_1"]["hidden_dim"]),
        gate_hidden_dim=int(section["gate_hidden_dim"]),
        dropout=float(config["p1_1"]["dropout"]),
        initial_gate=float(section["gate_initial_value"]),
        maximum_gate=float(section["maximum_gate"]),
    )
    count = sum(value.numel() for value in model.parameters())
    if count > int(section["max_parameters"]):
        raise ValueError("scalar evidence router exceeds parameter cap")
    return model


def load_router_initialization(
    *,
    model: BoundedScalarEvidenceRouter,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    state = payload["model_state"]
    prefixed_keys = [
        key for key in state if key.startswith("router.")
    ]
    if prefixed_keys:
        router_state = {
            key[len("router.") :]: value
            for key, value in state.items()
            if key.startswith("router.")
        }
    else:
        router_state = state
    model.router.load_state_dict(router_state, strict=True)
    if bool(payload.get("teacher_used_at_inference", True)):
        raise ValueError("initial router checkpoint is not deployable")
    return payload


def scalar_evidence_loss(
    *,
    outputs: Sequence[torch.Tensor],
    condition: torch.Tensor,
    quality: torch.Tensor,
    teacher_map: torch.Tensor,
    identities: torch.Tensor,
    gallery: TrainIdentityGallery,
    recipe: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    output, _, logits, gate, _, anchor = outputs
    with torch.no_grad():
        oracle_map, oracle = teacher_best_frame_oracle(
            condition,
            quality,
            teacher_map,
        )
        oracle_weights = oracle["weights"].clone()
        teacher = F.normalize(teacher_map.float(), dim=-1)
        improvement = (
            (oracle_map * teacher).sum(dim=-1)
            - (anchor.detach() * teacher).sum(dim=-1)
        ).clamp_min(0.0)
        target_gate = (
            improvement
            / float(config["p1_3"]["gate_target_improvement_scale"])
        ).clamp(0.0, 1.0) * float(config["p1_3"]["maximum_gate"])

    distillation = -(
        oracle_weights * F.log_softmax(logits.float(), dim=-1)
    ).sum(dim=-1).mean()
    gate_target = F.smooth_l1_loss(
        gate.float(),
        target_gate,
        beta=0.02,
    )
    local_teacher = 1.0 - (
        F.normalize(output.float(), dim=-1) * teacher
    ).sum(dim=-1).mean()
    output_embedding = F.normalize(output.float().mean(dim=1), dim=-1)
    anchor_embedding = F.normalize(anchor.float().mean(dim=1), dim=-1)
    teacher_embedding = F.normalize(teacher.mean(dim=1), dim=-1)
    global_teacher = 1.0 - (
        output_embedding * teacher_embedding
    ).sum(dim=-1).mean()

    targets = _gallery_targets(identities, gallery)
    output_scores = output_embedding @ gallery.vectors.transpose(0, 1)
    anchor_scores = anchor_embedding @ gallery.vectors.transpose(0, 1)
    gallery_cross_entropy = F.cross_entropy(
        output_scores / float(config["p1_3"]["gallery_temperature"]),
        targets,
    )
    positive_output = output_scores.gather(
        1,
        targets[:, None],
    ).squeeze(1)
    positive_anchor = anchor_scores.gather(
        1,
        targets[:, None],
    ).squeeze(1)
    negative_mask = torch.ones_like(output_scores, dtype=torch.bool)
    negative_mask.scatter_(1, targets[:, None], False)
    output_impostor = output_scores.masked_fill(
        ~negative_mask,
        float("-inf"),
    ).max(dim=1).values
    anchor_impostor = anchor_scores.masked_fill(
        ~negative_mask,
        float("-inf"),
    ).max(dim=1).values
    output_margin = positive_output - output_impostor
    anchor_margin = positive_anchor - anchor_impostor
    relative_gallery_margin = F.relu(
        anchor_margin
        + float(config["p1_3"]["relative_gallery_margin_target"])
        - output_margin
    ).mean()
    anchor_safety = (
        F.relu(anchor_margin - output_margin).mean()
        + F.relu(positive_anchor - positive_output).mean()
    )
    components = {
        "distillation_loss": distillation,
        "gate_target_loss": gate_target,
        "local_teacher_loss": local_teacher,
        "global_teacher_loss": global_teacher,
        "gallery_cross_entropy_loss": gallery_cross_entropy,
        "relative_gallery_margin_loss": relative_gallery_margin,
        "anchor_safety_loss": anchor_safety,
    }
    total = sum(
        float(recipe[name.replace("_loss", "_weight")]) * value
        for name, value in components.items()
    )
    _require_finite("scalar evidence total loss", total)
    return total, {
        "loss": float(total.detach().item()),
        **{
            name: float(value.detach().item())
            for name, value in components.items()
        },
        "mean_gate": float(gate.detach().mean().item()),
        "gate_std": float(gate.detach().std(unbiased=False).item()),
        "target_gate_mean": float(target_gate.mean().item()),
        "output_gallery_margin": float(output_margin.mean().item()),
        "anchor_gallery_margin": float(anchor_margin.mean().item()),
    }


@torch.inference_mode()
def evaluate_scalar_evidence_split(
    *,
    split: str,
    dataset: TokenTrainingDataset,
    model: BoundedScalarEvidenceRouter,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    result = evaluate_identity_gated_router_split(
        split=split,
        dataset=dataset,
        model=model,  # type: ignore[arg-type]
        config=config,
        device=device,
        method_name=SCALAR_EVIDENCE_METHOD,
    )
    result["gate_bounded_by"] = model.maximum_gate
    result["gate_identity_vector_input"] = False
    return result


def _selection_row(
    *,
    recipe_name: str,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = result["methods"]["quality_pooling"]
    learned = result["methods"][SCALAR_EVIDENCE_METHOD]
    base_verification = baseline["verification"]
    verification = learned["verification"]
    section = config["p1_3"]
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
    wins = sum(value > 0.0 for value in deltas.values())
    correctness = result["correctness"]
    checks = {
        "minimum_hard_gain": (
            hard_gain >= float(section["selection_min_hard_auc_gain"])
        ),
        "complementary_gain": (
            deltas["complementary_occlusion"]
            > float(section["selection_min_complementary_auc_gain"])
        ),
        "hard_scenario_wins": (
            wins >= int(section["selection_min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            clean_drop <= float(section["selection_clean_auc_max_drop"])
        ),
        "rank1_constraint": (
            rank_drop <= float(section["selection_rank1_max_drop"])
        ),
        "teacher_map_headroom": (
            teacher_gain
            >= float(section["selection_min_teacher_map_cosine_gain"])
        ),
        "correctness": all(
            (
                correctness["output_shape_valid"],
                correctness["output_unit_norm_valid"],
                correctness["weight_sum_valid"],
                correctness["permutation_equivalent"],
            )
        ),
        "bounded_gate": (
            result["gate_diagnostics"]["all"]["mean_gate"]
            <= float(section["maximum_gate"])
        ),
    }
    return {
        "recipe": recipe_name,
        "method": SCALAR_EVIDENCE_METHOD,
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


def select_scalar_evidence_candidate(
    *,
    validation_results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    order = list(config["p1_3"]["candidate_recipes"])
    if set(validation_results) != set(order):
        raise ValueError("validation results do not match P1-3 recipes")
    rows = [
        _selection_row(
            recipe_name=recipe,
            result=validation_results[recipe],
            config=config,
        )
        for recipe in order
    ]
    feasible = [row for row in rows if row["constraints_passed"]]
    ranked = sorted(
        feasible if feasible else rows,
        key=lambda row: (
            -float(row["hard_macro_roc_auc"]),
            -float(row["pooled_all_roc_auc"]),
            -float(row["teacher_map_cosine"]),
            order.index(str(row["recipe"])),
        ),
    )
    return {
        "selection_split": "val",
        "test_metrics_consulted": False,
        "candidate_count": len(rows),
        "feasible_candidate_count": len(feasible),
        "test_authorized": bool(feasible),
        "selected": dict(ranked[0]) if feasible else None,
        "best_validation_candidate": dict(ranked[0]),
        "rows": rows,
        "thresholds": {
            key: value
            for key, value in config["p1_3"].items()
            if key.startswith("selection_")
        },
        "selection_rule": (
            "Select the bounded scalar-evidence recipe/checkpoint using "
            "validation only and require every preregistered gate."
        ),
    }


def _learning_rate(
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    base: float,
) -> float:
    if warmup_steps and step <= warmup_steps:
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


def _save_checkpoint(
    *,
    path: Path,
    model: BoundedScalarEvidenceRouter,
    optimizer: torch.optim.Optimizer,
    step: int,
    recipe: str,
    validation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "stage": "P1-3",
            "step": step,
            "recipe": recipe,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "parameter_count": sum(
                value.numel() for value in model.parameters()
            ),
            "validation": validation,
            "p1_3_config": dict(config["p1_3"]),
            "teacher_used_at_inference": False,
            "gallery_used_at_inference": False,
        },
        temporary,
    )
    temporary.replace(path)


def load_scalar_evidence_checkpoint(
    *,
    path: str | Path,
    model: BoundedScalarEvidenceRouter,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def train_scalar_evidence_candidate(
    *,
    recipe_name: str,
    train_dataset: TokenTrainingDataset,
    val_dataset: TokenTrainingDataset,
    gallery: TrainIdentityGallery,
    model: BoundedScalarEvidenceRouter,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path_for_step: Callable[[int], Path],
    max_steps_override: int | None = None,
    metric_callback: Callable[
        [str, int, Mapping[str, Any]],
        None,
    ]
    | None = None,
) -> dict[str, Any]:
    section = config["p1_3"]
    if recipe_name not in section["candidate_recipes"]:
        raise ValueError("recipe is not preregistered")
    recipe = section["recipes"][recipe_name]
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
    router_scale = float(recipe["router_learning_rate_scale"])
    if router_scale == 0.0:
        for parameter in model.router.parameters():
            parameter.requires_grad_(False)
    gate_parameters = list(model.gate_network.parameters())
    optimizer_groups: list[dict[str, Any]] = [
        {
            "params": gate_parameters,
            "lr": float(section["learning_rate"]),
        }
    ]
    if router_scale > 0.0:
        optimizer_groups.append(
            {
                "params": list(model.router.parameters()),
                "lr": float(section["learning_rate"]) * router_scale,
            }
        )
    model.to(device)
    gallery_device = gallery.to(device)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
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
    generator = torch.Generator().manual_seed(
        int(config["project"]["seed"])
        + 1300
        + list(section["candidate_recipes"]).index(recipe_name)
    )
    permutation = torch.randperm(len(train_dataset), generator=generator)
    cursor = 0
    batch_size = int(section["batch_size"])
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    best_score: tuple[float, float, float] | None = None
    best_checkpoint: Path | None = None
    best_step = 0
    patience = 0
    stopped_early = False
    started = time.perf_counter()

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
        base_lr = _learning_rate(
            step=step,
            max_steps=max_steps,
            warmup_steps=min(
                int(section["warmup_steps"]),
                max(0, max_steps - 1),
            ),
            base=float(section["learning_rate"]),
        )
        optimizer.param_groups[0]["lr"] = base_lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = base_lr * router_scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            outputs = model(condition, quality)
            loss, losses = scalar_evidence_loss(
                outputs=outputs,
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
                [parameter for group in optimizer.param_groups
                 for parameter in group["params"]],
                float(section["max_grad_norm"]),
            ).item()
        )
        scaler.step(optimizer)
        scaler.update()
        row = {
            "recipe": recipe_name,
            "step": step,
            **losses,
            "learning_rate": base_lr,
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        if metric_callback is not None:
            metric_callback(recipe_name, step, row)

        if step % validate_every == 0 or step == max_steps:
            validation = evaluate_scalar_evidence_split(
                split="val",
                dataset=val_dataset,
                model=model,
                config=config,
                device=device,
            )
            learned = validation["methods"][SCALAR_EVIDENCE_METHOD]
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
                checkpoint = checkpoint_path_for_step(step)
                _save_checkpoint(
                    path=checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    recipe=recipe_name,
                    validation=validation,
                    config=config,
                )
                best_score = score
                best_checkpoint = checkpoint
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
        raise RuntimeError("P1-3 candidate produced no checkpoint")
    return {
        "recipe": recipe_name,
        "steps_completed": int(history[-1]["step"]),
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint),
        "best_score": list(best_score or ()),
        "stopped_early": stopped_early,
        "history": history,
        "validations": validations,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(
            value.numel() for value in model.parameters()
        ),
        "router_learning_rate_scale": router_scale,
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


def write_scalar_evidence_artifacts(
    *,
    output_directory: str | Path,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    training_runs: Mapping[str, Mapping[str, Any]],
    validation_results: Mapping[str, Mapping[str, Any]],
    selected_validation: Mapping[str, Any],
    p1_2_validation: Mapping[str, Any],
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
    split_results = [("val", selected_validation)]
    if test_result is not None:
        split_results.append(("test", test_result))
    core_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for split, result in split_results:
        for method in (*BASELINE_METHODS, SCALAR_EVIDENCE_METHOD):
            values = result["methods"][method]
            verification = values["verification"]
            core_rows.append(
                {
                    "split": split,
                    "recipe": (
                        selected_recipe
                        if method == SCALAR_EVIDENCE_METHOD
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
                            if method == SCALAR_EVIDENCE_METHOD
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
    search_rows = [
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
            "rank1_drop_vs_quality": row["rank1_drop_vs_quality"],
            "teacher_map_cosine_gain_vs_quality": row[
                "teacher_map_cosine_gain_vs_quality"
            ],
            "mean_gate": row["mean_gate"],
            "constraints_passed": row["constraints_passed"],
        }
        for row in selection["rows"]
    ]
    search_path = output / "validation_candidate_search.csv"
    _write_csv(search_path, search_rows, list(search_rows[0]))
    history_rows = [
        dict(row)
        for recipe in config["p1_3"]["candidate_recipes"]
        for row in training_runs[recipe]["history"]
    ]
    history_path = output / "training_history.csv"
    _write_csv(history_path, history_rows, list(history_rows[0]))
    gate_rows: list[dict[str, Any]] = []
    for split, result in split_results:
        for scenario, values in result["gate_diagnostics"].items():
            gate_rows.append(
                {"split": split, "scenario": scenario, **values}
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
            "new_holdout_confirmation_before_diffusion"
            if (
                selection["test_authorized"]
                and test_evaluation is not None
                and test_evaluation["passed"]
                and all(checks.values())
            )
            else "stop_router_tuning_keep_quality_anchor"
        ),
    }
    diagnostics = {
        "schema_version": 1,
        "preregistration": preregistration,
        "selection": selection,
        "training_runs": training_runs,
        "validation_results": validation_results,
        "p1_2_initial_validation": p1_2_validation,
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

    quality = selected_validation["methods"]["quality_pooling"][
        "verification"
    ]
    p1_2 = p1_2_validation["methods"][GATED_ROUTER_METHOD][
        "verification"
    ]
    p1_3 = selected_validation["methods"][SCALAR_EVIDENCE_METHOD][
        "verification"
    ]
    test_lines = (
        "| 未授权 | - | - | - |"
        if test_result is None
        else "\n".join(
            f"| {method} | "
            f"{test_result['methods'][method]['verification']['all']['roc_auc']:.6f} | "
            f"{test_result['methods'][method]['verification']['hard_average']['roc_auc']:.6f} | "
            f"{test_result['methods'][method]['verification']['all']['rank1_accuracy']:.6f} |"
            for method in ("quality_pooling", SCALAR_EVIDENCE_METHOD)
        )
    )
    report = f"""# P1-3 Bounded Scalar-Evidence Router

## 结论

Validation 选择：`{selected_recipe}`。
Validation 门槛：**{'PASS' if selection['test_authorized'] else 'FAIL'}**。
最终结论：**{'GO' if decision['go'] else 'NO-GO'}**。

P1-3 删除 gate 的身份向量输入，仅保留 13 个置信度、质量和跨帧共识
标量；residual gate 上限为 {config['p1_3']['maximum_gate']}。训练目标改为
相对 quality anchor 的 gallery margin 改善，teacher/gallery 均不参与推理。

## Validation

| 方法 | pooled-all AUC | hard macro AUC | EER | Rank-1 |
|---|---:|---:|---:|---:|
| quality_pooling | {quality['all']['roc_auc']:.6f} | {quality['hard_average']['roc_auc']:.6f} | {quality['all']['eer']:.6f} | {quality['all']['rank1_accuracy']:.6f} |
| P1-2 identity gate | {p1_2['all']['roc_auc']:.6f} | {p1_2['hard_average']['roc_auc']:.6f} | {p1_2['all']['eer']:.6f} | {p1_2['all']['rank1_accuracy']:.6f} |
| P1-3 scalar gate | {p1_3['all']['roc_auc']:.6f} | {p1_3['hard_average']['roc_auc']:.6f} | {p1_3['all']['eer']:.6f} | {p1_3['all']['rank1_accuracy']:.6f} |

P1-3 hard AUC 相对 quality：
`{selection['best_validation_candidate']['hard_gain_vs_quality']:+.6f}`。
互补遮挡 AUC 相对 quality：
`{selection['best_validation_candidate']['complementary_gain_vs_quality']:+.6f}`。
平均 gate：`{selection['best_validation_candidate']['mean_gate']:.6f}`。

## Locked test

| 方法 | pooled-all AUC | hard macro AUC | Rank-1 |
|---|---:|---:|---:|
{test_lines}

## 解释边界

- candidate、checkpoint 和 test 授权仅由 validation 决定。
- 现有 test 身份已在早期诊断暴露，不是全新确认性 holdout。
- 如果仍未达到门槛，应停止继续微调 late-hook router，而不是放宽门槛。
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
