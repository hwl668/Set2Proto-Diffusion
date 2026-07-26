"""Trainable, permutation-invariant continuous local condition router.

The router is distilled from the P1-0 teacher-guided best-frame-or-quality
oracle.  The teacher prototype is used only while constructing the training
loss; inference consumes condition features and their frozen AdaFace quality
norms only.
"""

from __future__ import annotations

import csv
import json
import math
import time
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
from .research_evaluation import continuous_aggregation_baselines
from .training import TokenTrainingDataset
from .visibility_aggregation import (
    _evaluate_collected_maps,
    _weight_summary,
    _wrong_frame_mask,
)


ROUTER_METHOD = "continuous_local_router"
BASELINE_METHODS = ("mean_pooling", "quality_pooling")


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def _canonical_sum(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Use a stable order so frame permutations are numerically equivalent."""

    return torch.sort(value, dim=dim).values.sum(dim=dim)


def _weighted_pool(
    condition: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if condition.ndim != 4:
        raise ValueError("condition must have shape [B,F,P,D]")
    batch, frames, positions, _ = condition.shape
    if tuple(weights.shape) != (batch, positions, frames):
        raise ValueError("weights must have shape [B,P,F]")
    features = F.normalize(condition.float(), dim=-1).permute(0, 2, 1, 3)
    contributions = features * weights.float().unsqueeze(-1)
    return F.normalize(
        _canonical_sum(contributions, dim=2),
        dim=-1,
        eps=1e-6,
    )


class ContinuousLocalRouter(nn.Module):
    """Small DeepSets-style local frame router with no frame-index embedding."""

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        positions: int = 49,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or positions <= 0 or hidden_dim <= 0:
            raise ValueError("router dimensions must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0,1]")
        self.feature_dim = int(feature_dim)
        self.positions = int(positions)
        self.hidden_dim = int(hidden_dim)
        # Four vector channels plus four invariant scalar diagnostics:
        # normalized quality share, log-quality z, local consensus, and
        # global frame-to-set consensus.
        input_dim = 4 * self.feature_dim + 4
        self.position_embedding = nn.Parameter(
            torch.zeros(self.positions, self.hidden_dim)
        )
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        condition: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if condition.ndim != 4:
            raise ValueError("condition must have shape [B,F,P,D]")
        batch, frames, positions, dimension = condition.shape
        if positions != self.positions or dimension != self.feature_dim:
            raise ValueError(
                "condition shape does not match configured positions/dimension"
            )
        if tuple(quality.shape) != (batch, frames):
            raise ValueError("quality must have shape [B,F]")
        if frames < 2:
            raise ValueError("router requires at least two condition frames")

        local = F.normalize(condition.float(), dim=-1).permute(0, 2, 1, 3)
        set_local = F.normalize(
            _canonical_sum(local, dim=2) / float(frames),
            dim=-1,
            eps=1e-6,
        )
        local_sum = _canonical_sum(local, dim=2)
        local_consensus = (
            local * (local_sum.unsqueeze(2) - local)
        ).sum(dim=-1) / float(frames - 1)

        frame_global = F.normalize(local.mean(dim=1), dim=-1, eps=1e-6)
        set_global = F.normalize(
            _canonical_sum(frame_global, dim=1) / float(frames),
            dim=-1,
            eps=1e-6,
        )
        global_consensus = (
            frame_global * set_global.unsqueeze(1)
        ).sum(dim=-1)

        quality_value = quality.float().clamp_min(1e-6)
        quality_share = quality_value / _canonical_sum(
            quality_value,
            dim=1,
        ).unsqueeze(-1).clamp_min(1e-8)
        log_quality = quality_value.log()
        log_quality_z = (
            log_quality - log_quality.mean(dim=1, keepdim=True)
        ) / log_quality.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)

        expanded_mean = set_local.unsqueeze(2).expand_as(local)
        vector_features = torch.cat(
            (
                local,
                expanded_mean,
                local * expanded_mean,
                local - expanded_mean,
            ),
            dim=-1,
        )
        scalar_features = torch.stack(
            (
                quality_share[:, None, :].expand(-1, positions, -1),
                log_quality_z[:, None, :].expand(-1, positions, -1),
                local_consensus,
                global_consensus[:, None, :].expand(-1, positions, -1),
            ),
            dim=-1,
        )
        encoded = self.frame_encoder(
            torch.cat((vector_features, scalar_features), dim=-1)
        )
        encoded = encoded + self.position_embedding[None, :, None, :]
        context = _canonical_sum(encoded, dim=2) / float(frames)
        expanded_context = context.unsqueeze(2).expand_as(encoded)
        logits = self.score_head(
            torch.cat(
                (
                    encoded,
                    expanded_context,
                    encoded * expanded_context,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        weights = torch.softmax(logits.float(), dim=-1)
        pooled = _weighted_pool(condition, weights)
        return (
            _require_finite("router pooled map", pooled),
            _require_finite("router weights", weights),
            _require_finite("router logits", logits.float()),
        )


def build_continuous_router(
    config: Mapping[str, Any],
) -> ContinuousLocalRouter:
    grid = config["data"]["feature_grid"]
    router_config = config["p1_1"]
    model = ContinuousLocalRouter(
        feature_dim=int(grid["projected_dim"]),
        positions=int(grid["height"]) * int(grid["width"]),
        hidden_dim=int(router_config["hidden_dim"]),
        dropout=float(router_config["dropout"]),
    )
    parameters = sum(value.numel() for value in model.parameters())
    if parameters > int(router_config["max_parameters"]):
        raise ValueError(
            f"router has {parameters} parameters, exceeding configured "
            f"maximum {router_config['max_parameters']}"
        )
    return model


def router_parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def continuous_router_loss(
    *,
    logits: torch.Tensor,
    output_map: torch.Tensor,
    oracle_weights: torch.Tensor,
    teacher_map: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    router_config = config["p1_1"]
    # Oracle targets may have been constructed under inference_mode. Clone
    # them into ordinary tensors before an autograd-tracked multiplication.
    target_weights = oracle_weights.float().detach().clone()
    target_map = F.normalize(
        teacher_map.float().detach().clone(),
        dim=-1,
    )
    distillation = -(
        target_weights * F.log_softmax(logits.float(), dim=-1)
    ).sum(dim=-1).mean()
    local_map = 1.0 - (
        F.normalize(output_map.float(), dim=-1) * target_map
    ).sum(dim=-1).mean()
    output_identity = F.normalize(output_map.float().mean(dim=1), dim=-1)
    target_identity = F.normalize(target_map.mean(dim=1), dim=-1)
    global_identity = 1.0 - (
        output_identity * target_identity
    ).sum(dim=-1).mean()
    total = (
        float(router_config["distillation_weight"]) * distillation
        + float(router_config["local_map_cosine_weight"]) * local_map
        + float(router_config["global_identity_weight"]) * global_identity
    )
    _require_finite("router training loss", total)
    return total, {
        "loss": float(total.detach().item()),
        "distillation_loss": float(distillation.detach().item()),
        "local_map_cosine_loss": float(local_map.detach().item()),
        "global_identity_loss": float(global_identity.detach().item()),
    }


def _batch(
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    values = default_collate([dataset[index] for index in indices])
    teacher = dataset.targets["continuous_prototypes"][
        list(indices)
    ].float().to(device)
    return values, teacher


@torch.inference_mode()
def evaluate_continuous_router_split(
    *,
    split: str,
    dataset: TokenTrainingDataset,
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    method_name: str = ROUTER_METHOD,
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if getattr(dataset.synthetic, "split", None) != split:
        raise ValueError("dataset split does not match requested split")
    model.eval()
    positions = int(getattr(model, "positions"))
    feature_dim = int(getattr(model, "feature_dim"))
    batch_size = int(config["evaluation"]["batch_size"])
    map_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in (*BASELINE_METHODS, method_name)
    }
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    weights: list[torch.Tensor] = []
    visibility: list[torch.Tensor] = []
    wrong_frames: list[torch.Tensor] = []
    maximum_norm_error = 0.0
    maximum_weight_error = 0.0
    observed_shapes: set[tuple[int, ...]] = set()
    oracle_l1_chunks: list[torch.Tensor] = []
    oracle_top1_chunks: list[torch.Tensor] = []

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch, teacher = _batch(dataset, indices, device)
        condition = batch["condition_features"].to(device)
        quality = batch["condition_quality"].to(device)
        observed_shapes.add(tuple(condition.shape))
        baselines = continuous_aggregation_baselines(condition, quality)
        forward_values = model(condition, quality)
        output, local_weights, _ = forward_values[:3]
        _, oracle = teacher_best_frame_oracle(
            condition,
            quality,
            teacher,
        )
        oracle_weights = oracle["weights"]
        for name, value in (
            ("mean_pooling", baselines["mean_pooling"]),
            ("quality_pooling", baselines["quality_pooling"]),
            (method_name, output),
        ):
            if tuple(value.shape[1:]) != (
                positions,
                feature_dim,
            ):
                raise RuntimeError(f"{name} returned invalid map shape")
            maximum_norm_error = max(
                maximum_norm_error,
                float((value.norm(dim=-1) - 1.0).abs().max().item()),
            )
            map_chunks[name].append(value.detach().cpu().half())
        maximum_weight_error = max(
            maximum_weight_error,
            float(
                (local_weights.sum(dim=-1) - 1.0)
                .abs()
                .max()
                .item()
            ),
        )
        weights.append(local_weights.cpu())
        oracle_l1_chunks.append(
            (local_weights - oracle_weights).abs().mean(dim=(1, 2)).cpu()
        )
        oracle_top1_chunks.append(
            (
                local_weights.argmax(dim=-1)
                == oracle_weights.argmax(dim=-1)
            )
            .float()
            .mean(dim=1)
            .cpu()
        )
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(str(value) for value in batch["scenario"])
        visibility.append(batch["condition_visibility"].cpu())
        wrong_frames.append(
            _wrong_frame_mask(dataset, indices, condition.shape[1])
        )

    evaluated = _evaluate_collected_maps(
        dataset=dataset,
        map_chunks=map_chunks,
        token_chunks={},
        identities=identities,
        scenarios=scenarios,
        far_targets=[
            float(value) for value in config["evaluation"]["far_targets"]
        ],
    )
    all_weights = torch.cat(weights)
    weight_diagnostics = _weight_summary(
        weights=all_weights,
        visibility=torch.cat(visibility),
        wrong_frames=torch.cat(wrong_frames),
        scenarios=scenarios,
    )
    scenario_array = np.asarray(scenarios)
    imitation: dict[str, Any] = {}
    oracle_l1 = torch.cat(oracle_l1_chunks)
    oracle_top1 = torch.cat(oracle_top1_chunks)
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        imitation[scenario] = {
            "oracle_weight_l1": float(oracle_l1[mask].mean().item()),
            "oracle_top1_accuracy": float(
                oracle_top1[mask].mean().item()
            ),
        }
    imitation["all"] = {
        "oracle_weight_l1": float(oracle_l1.mean().item()),
        "oracle_top1_accuracy": float(oracle_top1.mean().item()),
    }

    first_count = min(batch_size, len(dataset))
    first_indices = list(range(first_count))
    first, _ = _batch(dataset, first_indices, device)
    first_condition = first["condition_features"].to(device)
    first_quality = first["condition_quality"].to(device)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    inverse = torch.argsort(permutation)
    original_values = model(
        first_condition,
        first_quality,
    )
    permuted_values = model(
        first_condition[:, permutation],
        first_quality[:, permutation],
    )
    original_map, original_weights = original_values[:2]
    permuted_map, permuted_weights = permuted_values[:2]
    map_diff = float((original_map - permuted_map).abs().max().item())
    routing_diff = float(
        (
            original_weights - permuted_weights[..., inverse]
        )
        .abs()
        .max()
        .item()
    )
    map_cosine = float(
        F.cosine_similarity(
            original_map.float(),
            permuted_map.float(),
            dim=-1,
        )
        .min()
        .item()
    )
    correctness = {
        "condition_shapes_observed": [
            list(value) for value in sorted(observed_shapes)
        ],
        "output_shape_valid": all(
            sum(chunk.shape[0] for chunk in chunks) == len(dataset)
            for chunks in map_chunks.values()
        ),
        "output_unit_norm_max_abs_error": maximum_norm_error,
        "output_unit_norm_valid": maximum_norm_error <= 1e-5,
        "weight_sum_max_abs_error": maximum_weight_error,
        "weight_sum_valid": maximum_weight_error <= 1e-6,
        "permutation_map_max_abs_diff": map_diff,
        "permutation_map_cosine_min": map_cosine,
        "permutation_routing_max_abs_diff": routing_diff,
        "permutation_equivalent": (
            map_diff <= 1e-5
            and map_cosine >= 0.999999
            and routing_diff <= 1e-5
        ),
    }
    evaluated.update(
        {
            "split": split,
            "router_method": method_name,
            "teacher_used_at_inference": False,
            "deployable_with_cached_condition_features": True,
            "weight_diagnostics": weight_diagnostics,
            "oracle_imitation": imitation,
            "correctness": correctness,
        }
    )
    return evaluated


def _selection_row(
    *,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = result["methods"]["quality_pooling"]
    learned = result["methods"][ROUTER_METHOD]
    baseline_verification = baseline["verification"]
    verification = learned["verification"]
    thresholds = config["p1_1"]
    scenario_deltas = {
        scenario: (
            verification[scenario]["roc_auc"]
            - baseline_verification[scenario]["roc_auc"]
        )
        for scenario in HARD_SCENARIOS
    }
    hard_gain = (
        verification["hard_average"]["roc_auc"]
        - baseline_verification["hard_average"]["roc_auc"]
    )
    clean_drop = (
        baseline_verification["clean"]["roc_auc"]
        - verification["clean"]["roc_auc"]
    )
    rank_drop = (
        baseline_verification["all"]["rank1_accuracy"]
        - verification["all"]["rank1_accuracy"]
    )
    teacher_gain = (
        learned["teacher_similarity"]["all"]["map_cosine"]
        - baseline["teacher_similarity"]["all"]["map_cosine"]
    )
    wins = sum(delta > 0.0 for delta in scenario_deltas.values())
    correctness = result["correctness"]
    checks = {
        "minimum_hard_gain": (
            hard_gain
            >= float(thresholds["selection_min_hard_auc_gain"])
        ),
        "complementary_gain": (
            scenario_deltas["complementary_occlusion"]
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
        "method": ROUTER_METHOD,
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
        "complementary_gain_vs_quality": scenario_deltas[
            "complementary_occlusion"
        ],
        "hard_scenario_wins": wins,
        "scenario_auc_deltas": scenario_deltas,
        "checks": checks,
        "constraints_passed": all(checks.values()),
    }


def select_continuous_router(
    *,
    validation_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if validation_result["split"] != "val":
        raise ValueError("router selection must use validation")
    row = _selection_row(result=validation_result, config=config)
    return {
        "selection_split": "val",
        "test_metrics_consulted": False,
        "test_authorized": bool(row["constraints_passed"]),
        "selected": dict(row) if row["constraints_passed"] else None,
        "best_validation_checkpoint": dict(row),
        "thresholds": {
            key: value
            for key, value in config["p1_1"].items()
            if key.startswith("selection_")
        },
        "selection_rule": (
            "Select the checkpoint by validation hard AUC, pooled-all AUC, "
            "then teacher-map cosine. Authorize one test evaluation only if "
            "all preregistered improvement and correctness gates pass."
        ),
    }


def _learning_rate(
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_learning_rate * step / float(warmup_steps)
    denominator = max(1, max_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / float(denominator))
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_router_checkpoint(
    *,
    path: str | Path,
    model: ContinuousLocalRouter,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "step": int(step),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model": {
                "feature_dim": model.feature_dim,
                "positions": model.positions,
                "hidden_dim": model.hidden_dim,
                "parameter_count": router_parameter_count(model),
            },
            "validation": validation_result,
            "p1_1_config": dict(config["p1_1"]),
            "teacher_used_at_inference": False,
        },
        temporary,
    )
    temporary.replace(target)
    return target


def load_router_checkpoint(
    *,
    path: str | Path,
    model: ContinuousLocalRouter,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def train_continuous_router(
    *,
    train_dataset: TokenTrainingDataset,
    val_dataset: TokenTrainingDataset,
    model: ContinuousLocalRouter,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path_for_step: Callable[[int], Path],
    max_steps_override: int | None = None,
    metric_callback: Callable[[int, Mapping[str, float]], None] | None = None,
) -> dict[str, Any]:
    if getattr(train_dataset.synthetic, "split", None) != "train":
        raise ValueError("training dataset must be train split")
    if getattr(val_dataset.synthetic, "split", None) != "val":
        raise ValueError("validation dataset must be val split")
    router_config = config["p1_1"]
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else int(router_config["max_steps"])
    )
    if max_steps <= 0:
        raise ValueError("training steps must be positive")
    smoke = max_steps_override is not None
    validate_every = (
        max_steps
        if smoke
        else int(router_config["validate_every_steps"])
    )
    min_steps = min(
        max_steps,
        1 if smoke else int(router_config["min_steps"]),
    )
    patience_limit = int(router_config["early_stop_patience"])
    batch_size = int(router_config["batch_size"])
    generator = torch.Generator().manual_seed(
        int(config["project"]["seed"]) + 1101
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(router_config["learning_rate"]),
        weight_decay=float(router_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
            and str(router_config["precision"]) == "fp16"
        ),
    )
    autocast_dtype = (
        torch.bfloat16
        if str(router_config["precision"]) == "bf16"
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
    start_time = time.perf_counter()

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
        with torch.inference_mode():
            _, oracle = teacher_best_frame_oracle(
                condition,
                quality,
                teacher,
            )
            oracle_weights = oracle["weights"]

        learning_rate = _learning_rate(
            step=step,
            max_steps=max_steps,
            warmup_steps=min(
                int(router_config["warmup_steps"]),
                max(0, max_steps - 1),
            ),
            base_learning_rate=float(router_config["learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            output, _, logits = model(condition, quality)
            loss, losses = continuous_router_loss(
                logits=logits,
                output_map=output,
                oracle_weights=oracle_weights,
                teacher_map=teacher,
                config=config,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(router_config["max_grad_norm"]),
            ).item()
        )
        scaler.step(optimizer)
        scaler.update()
        row: dict[str, Any] = {
            "step": step,
            **losses,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        if metric_callback is not None:
            metric_callback(step, row)

        if step % validate_every == 0 or step == max_steps:
            validation = evaluate_continuous_router_split(
                split="val",
                dataset=val_dataset,
                model=model,
                config=config,
                device=device,
            )
            learned = validation["methods"][ROUTER_METHOD]
            verification = learned["verification"]
            score = (
                float(verification["hard_average"]["roc_auc"]),
                float(verification["all"]["roc_auc"]),
                float(
                    learned["teacher_similarity"]["all"]["map_cosine"]
                ),
            )
            validation_record = {
                "step": step,
                "score": list(score),
                "selection": _selection_row(
                    result=validation,
                    config=config,
                ),
            }
            validations.append(validation_record)
            if best_score is None or score > best_score:
                path = checkpoint_path_for_step(step)
                save_router_checkpoint(
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
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
                and step >= min_steps
                and patience >= patience_limit
            ):
                stopped_early = True
                break

    if best_checkpoint is None:
        raise RuntimeError("training completed without a validation checkpoint")
    return {
        "steps_completed": int(history[-1]["step"]),
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint),
        "best_score": list(best_score or ()),
        "stopped_early": stopped_early,
        "history": history,
        "validations": validations,
        "elapsed_seconds": time.perf_counter() - start_time,
        "parameter_count": router_parameter_count(model),
        "teacher_supervision": (
            "teacher_best_frame_or_quality oracle used for training targets"
        ),
        "teacher_used_at_inference": False,
    }


@torch.inference_mode()
def measure_continuous_router_latency(
    *,
    dataset: TokenTrainingDataset,
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    method_name: str = ROUTER_METHOD,
) -> dict[str, Any]:
    batch_size = 16
    if len(dataset) < batch_size:
        raise ValueError("latency measurement requires at least 16 samples")
    batch, _ = _batch(dataset, list(range(batch_size)), device)
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    model.eval()

    def baseline_call() -> torch.Tensor:
        return quality_anchor(condition, quality)[0]

    def router_call() -> torch.Tensor:
        return model(condition, quality)[0]

    warmup = int(config["evaluation"]["latency_warmup_runs"])
    runs = int(config["evaluation"]["latency_measurement_runs"])
    result: dict[str, Any] = {}
    for name, function in (
        ("quality_pooling", baseline_call),
        (method_name, router_call),
    ):
        for _ in range(warmup):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(runs):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        result[name] = {
            "batch_size": batch_size,
            "warmup_runs": warmup,
            "measurement_runs": runs,
            "batch_latency_ms": elapsed * 1000.0 / runs,
            "per_set_latency_ms": (
                elapsed * 1000.0 / (runs * batch_size)
            ),
            "teacher_required": False,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        }
    return result


def test_gate(
    *,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    row = _selection_row(result=result, config=config)
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


def _tar(verification: Mapping[str, Any], far: float) -> float:
    return float(verification["tar_at_far"][f"{far:g}"]["tar"])


def write_continuous_router_artifacts(
    *,
    output_directory: str | Path,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    training: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: Mapping[str, bool],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_results = [("val", validation_result)]
    if test_result is not None:
        split_results.append(("test", test_result))
    core_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for split, result in split_results:
        for method in (*BASELINE_METHODS, ROUTER_METHOD):
            values = result["methods"][method]
            verification = values["verification"]
            core_rows.append(
                {
                    "split": split,
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
                    "mean_genuine_impostor_margin": verification["all"][
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
    training_rows = [dict(value) for value in training["history"]]
    history_path = output / "training_history.csv"
    _write_csv(history_path, training_rows, list(training_rows[0]))

    weight_rows: list[dict[str, Any]] = []
    for split, result in split_results:
        for scenario, values in result["weight_diagnostics"].items():
            if scenario == "global":
                continue
            weight_rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "samples": values["samples"],
                    "mean_effective_frames": values[
                        "mean_effective_frames"
                    ],
                    "frame_0_weight": values["frame_mean_weights"][0],
                    "frame_1_weight": values["frame_mean_weights"][1],
                    "frame_2_weight": values["frame_mean_weights"][2],
                    "frame_3_weight": values["frame_mean_weights"][3],
                    "visible_weight_mean": values[
                        "visible_weight_mean"
                    ],
                    "occluded_weight_mean": values[
                        "occluded_weight_mean"
                    ],
                    "wrong_identity_frame_mean_weight": values.get(
                        "wrong_identity_frame_mean_weight"
                    ),
                    "oracle_weight_l1": result["oracle_imitation"][
                        scenario
                    ]["oracle_weight_l1"],
                    "oracle_top1_accuracy": result["oracle_imitation"][
                        scenario
                    ]["oracle_top1_accuracy"],
                }
            )
    weights_path = output / "weight_diagnostics.csv"
    _write_csv(weights_path, weight_rows, list(weight_rows[0]))

    test_evaluation = (
        test_gate(result=test_result, config=config)
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
            "proceed_to_p1_2_continuous_residual_refinement"
            if (
                selection["test_authorized"]
                and test_evaluation is not None
                and test_evaluation["passed"]
                and all(checks.values())
            )
            else "do_not_start_diffusion_improve_or_reject_router"
        ),
    }
    diagnostics = {
        "schema_version": 1,
        "preregistration": preregistration,
        "selection": selection,
        "training": training,
        "validation": validation_result,
        "test": test_result,
        "latency": latency,
        "checks": checks,
        "decision": decision,
        "reproducibility": reproducibility,
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

    baseline = validation_result["methods"]["quality_pooling"][
        "verification"
    ]
    learned = validation_result["methods"][ROUTER_METHOD]["verification"]
    test_lines: list[str] = []
    if test_result is not None:
        for method in ("quality_pooling", ROUTER_METHOD):
            verification = test_result["methods"][method]["verification"]
            test_lines.append(
                f"| {method} | {verification['all']['roc_auc']:.6f} | "
                f"{verification['hard_average']['roc_auc']:.6f} | "
                f"{verification['all']['eer']:.6f} | "
                f"{verification['all']['rank1_accuracy']:.6f} |"
            )
    else:
        test_lines.append("| 未授权 | - | - | - | - |")
    report = f"""# P1-1 Continuous Local Router

## 结论

Validation 门槛：**{'PASS' if selection['test_authorized'] else 'FAIL'}**。
最终 Go/No-Go：**{'GO' if decision['go'] else 'NO-GO'}**。

这是一个可部署的缓存特征路由器。teacher-guided oracle 仅产生训练监督，
推理时不读取 teacher prototype，也不运行 AdaFace backbone。

## Validation

| 方法 | pooled-all AUC | hard macro AUC | EER | Rank-1 |
|---|---:|---:|---:|---:|
| quality_pooling | {baseline['all']['roc_auc']:.6f} | {baseline['hard_average']['roc_auc']:.6f} | {baseline['all']['eer']:.6f} | {baseline['all']['rank1_accuracy']:.6f} |
| {ROUTER_METHOD} | {learned['all']['roc_auc']:.6f} | {learned['hard_average']['roc_auc']:.6f} | {learned['all']['eer']:.6f} | {learned['all']['rank1_accuracy']:.6f} |

Validation hard AUC 增益：
`{selection['best_validation_checkpoint']['hard_gain_vs_quality']:+.6f}`。
Validation teacher-map cosine 增益：
`{selection['best_validation_checkpoint']['teacher_map_cosine_gain_vs_quality']:+.6f}`。

## Locked test

| 方法 | pooled-all AUC | hard macro AUC | EER | Rank-1 |
|---|---:|---:|---:|---:|
{chr(10).join(test_lines)}

## 训练与成本

- 最佳 step：{training['best_step']}
- 完成 step：{training['steps_completed']}
- 参数量：{training['parameter_count']}
- 训练耗时：{training['elapsed_seconds']:.2f} s
- batch=16 router latency：{latency[ROUTER_METHOD]['batch_latency_ms']:.3f} ms
- router 峰值 reserved 显存：{latency[ROUTER_METHOD]['peak_reserved_mib']:.1f} MiB

## 解释边界

- checkpoint 只由 validation 指标选择；test 若运行，只评估锁定 checkpoint 一次。
- 当前 test 身份已在更早 P0/P1-0 诊断中使用，因此不是全新的确认性 holdout。
- 单次 pilot 不构成统计显著性结论；失败也只否定当前轻量路由器与 late hook，
  不能单独否定所有连续残差或集合原型研究。
"""
    report_path = output / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "core_comparison_csv": str(core_path),
        "scenario_metrics_csv": str(scenario_path),
        "training_history_csv": str(history_path),
        "weight_diagnostics_csv": str(weights_path),
        "diagnostics_json": str(diagnostics_path),
        "decision": decision,
    }
