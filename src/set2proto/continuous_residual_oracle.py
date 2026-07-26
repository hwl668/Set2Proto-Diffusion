"""Teacher-guided continuous residual headroom diagnostics.

These methods are deliberately non-deployable. The teacher prototype is used
only to choose coefficients over information already present in the condition
frames. The experiment asks whether a continuous local residual has headroom
before any shared predictor, diffusion process, or backbone update is trained.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from .research_evaluation import continuous_aggregation_baselines
from .training import TokenTrainingDataset
from .visibility_aggregation import (
    _evaluate_collected_maps,
    _wrong_frame_mask,
)


BASELINE_METHODS = ("mean_pooling", "quality_pooling")
ORACLE_METHODS = (
    "teacher_best_frame_oracle",
    "teacher_convex_residual_oracle",
    "teacher_bounded_span_residual_oracle",
)
HARD_SCENARIOS = (
    "low_quality",
    "complementary_occlusion",
    "common_occlusion",
    "wrong_identity",
)


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(value)
    if not bool(finite.all().item()):
        count = int((~finite).sum().item())
        raise FloatingPointError(
            f"{name} contains {count} non-finite value(s)"
        )
    return value


def _canonical_sum(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    return torch.sort(value, dim=dim).values.sum(dim=dim)


def _quality_weights(
    quality: torch.Tensor,
    *,
    positions: int,
) -> torch.Tensor:
    if quality.ndim != 2:
        raise ValueError("quality must have shape [B,F]")
    values = _require_finite("condition quality", quality.float()).clamp_min(
        1e-6
    )
    denominator = _canonical_sum(values, dim=1).unsqueeze(1)
    normalized = values / denominator.clamp_min(1e-8)
    return normalized[:, None, :].expand(-1, positions, -1)


def _weighted_pool(
    features: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError("features must have shape [B,F,P,D]")
    batch, frames, positions, _ = features.shape
    if tuple(weights.shape) != (batch, positions, frames):
        raise ValueError("weights must have shape [B,P,F]")
    if float(weights.min().item()) < -1e-6:
        raise ValueError("convex weights cannot be negative")
    if float((weights.sum(dim=-1) - 1.0).abs().max().item()) > 1e-5:
        raise ValueError("convex weights must sum to one")
    local = F.normalize(
        _require_finite("condition features", features.float()),
        dim=-1,
    ).permute(0, 2, 1, 3)
    contributions = weights.unsqueeze(-1) * local
    pooled = _canonical_sum(contributions, dim=2)
    return _require_finite(
        "weighted continuous map",
        F.normalize(pooled, dim=-1, eps=1e-6),
    )


def _teacher_cosine(
    maps: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    return _require_finite(
        "teacher cosine",
        (
            F.normalize(maps.float(), dim=-1)
            * F.normalize(teacher.float(), dim=-1)
        ).sum(dim=-1),
    )


def project_simplex(value: torch.Tensor) -> torch.Tensor:
    """Project the final dimension onto the probability simplex."""

    if value.ndim < 1:
        raise ValueError("simplex projection requires at least one dimension")
    _require_finite("simplex input", value)
    size = int(value.shape[-1])
    ordered = torch.sort(value, dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1) - 1.0
    divisor = torch.arange(
        1,
        size + 1,
        dtype=value.dtype,
        device=value.device,
    )
    active = ordered - cumulative / divisor > 0
    rho = active.sum(dim=-1).clamp_min(1)
    theta = cumulative.gather(
        -1,
        (rho - 1).unsqueeze(-1),
    ).squeeze(-1) / rho.to(value.dtype)
    projected = (value - theta.unsqueeze(-1)).clamp_min(0.0)
    projected = projected / projected.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    return _require_finite("simplex projection", projected)


def quality_anchor(
    condition: torch.Tensor,
    quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = int(condition.shape[2])
    weights = _quality_weights(quality, positions=positions)
    return _weighted_pool(condition, weights), weights


def _canonicalize_frames(
    condition: torch.Tensor,
    quality: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Put an unordered frame set in a deterministic teacher-guided order."""

    features = F.normalize(condition.float(), dim=-1)
    target = F.normalize(teacher.float(), dim=-1)
    teacher_key = torch.einsum(
        "bfpd,bpd->bfp",
        features,
        target,
    ).mean(dim=-1)
    quality_scale = quality.float() / quality.float().abs().amax(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)
    fingerprint = features[..., 0].mean(dim=-1)
    key = teacher_key + 1e-3 * quality_scale + 1e-6 * fingerprint
    order = torch.argsort(key, dim=1, stable=True)
    condition_index = order[:, :, None, None].expand_as(condition)
    ordered_condition = condition.gather(1, condition_index)
    ordered_quality = quality.gather(1, order)
    return ordered_condition, ordered_quality, order


def _restore_frame_values(
    ordered_values: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    indices = order[:, None, :].expand_as(ordered_values)
    restored = torch.zeros_like(ordered_values)
    restored.scatter_(-1, indices, ordered_values)
    return restored


def teacher_best_frame_oracle(
    condition: torch.Tensor,
    quality: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Choose the teacher-closest frame or retain the quality anchor."""

    features = F.normalize(condition.float(), dim=-1)
    target = F.normalize(teacher.float(), dim=-1)
    anchor, anchor_weights = quality_anchor(features, quality)
    similarities = torch.einsum("bfpd,bpd->bpf", features, target)
    maximum = similarities.max(dim=-1, keepdim=True).values
    ties = similarities >= maximum - 1e-8
    frame_weights = ties.float()
    frame_weights = frame_weights / frame_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)
    frame_output = _weighted_pool(features, frame_weights)
    use_frame = (
        _teacher_cosine(frame_output, target)
        > _teacher_cosine(anchor, target)
    )
    weights = torch.where(
        use_frame.unsqueeze(-1),
        frame_weights,
        anchor_weights,
    )
    output = _weighted_pool(features, weights)
    return output, {"weights": weights}


def teacher_convex_residual_oracle(
    condition: torch.Tensor,
    quality: torch.Tensor,
    teacher: torch.Tensor,
    *,
    iterations: int,
    step_size: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Approximate the teacher-closest point in each local frame convex hull."""

    if iterations <= 0 or step_size <= 0.0:
        raise ValueError("convex oracle iterations/step_size must be positive")
    ordered_condition, ordered_quality, order = _canonicalize_frames(
        condition,
        quality,
        teacher,
    )
    features = F.normalize(ordered_condition.float(), dim=-1)
    target = F.normalize(teacher.float(), dim=-1)
    anchor, anchor_weights = quality_anchor(features, ordered_quality)
    anchor_cosine = _teacher_cosine(anchor, target)
    frame_output, frame_diagnostics = teacher_best_frame_oracle(
        features,
        ordered_quality,
        target,
    )
    frame_cosine = _teacher_cosine(frame_output, target)
    use_frame = frame_cosine > anchor_cosine
    best_weights = torch.where(
        use_frame.unsqueeze(-1),
        frame_diagnostics["weights"],
        anchor_weights,
    )
    best_cosine = torch.maximum(frame_cosine, anchor_cosine)
    weights = anchor_weights.clone()
    local = features.permute(0, 2, 1, 3)
    for iteration in range(iterations):
        current = _weighted_pool(features, weights)
        current_cosine = _teacher_cosine(current, target)
        current_norm = _canonical_sum(
            weights.unsqueeze(-1) * local,
            dim=2,
        ).norm(dim=-1).clamp_min(1e-6)
        gradient_map = (
            target - current_cosine.unsqueeze(-1) * current
        ) / current_norm.unsqueeze(-1)
        gradient_weights = torch.einsum(
            "bpfd,bpd->bpf",
            local,
            gradient_map,
        )
        learning_rate = step_size / math.sqrt(float(iteration + 1))
        proposal = project_simplex(
            weights + learning_rate * gradient_weights
        )
        proposal_map = _weighted_pool(features, proposal)
        proposal_cosine = _teacher_cosine(proposal_map, target)
        improved = proposal_cosine > best_cosine
        best_weights = torch.where(
            improved.unsqueeze(-1),
            proposal,
            best_weights,
        )
        best_cosine = torch.maximum(best_cosine, proposal_cosine)
        weights = proposal
    output = _weighted_pool(features, best_weights)
    restored_weights = _restore_frame_values(best_weights, order)
    return output, {
        "weights": restored_weights,
        "optimization_teacher_cosine": best_cosine,
    }


def teacher_bounded_span_residual_oracle(
    condition: torch.Tensor,
    quality: torch.Tensor,
    teacher: torch.Tensor,
    *,
    ridge: float,
    residual_norm_cap: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit a teacher-guided residual in the local frame-disagreement span."""

    if ridge <= 0.0 or residual_norm_cap <= 0.0:
        raise ValueError("span ridge and residual_norm_cap must be positive")
    features = F.normalize(condition.float(), dim=-1)
    target = F.normalize(teacher.float(), dim=-1)
    anchor, _ = quality_anchor(features, quality)
    local = features.permute(0, 2, 1, 3)
    directions = local - anchor.unsqueeze(2)
    target_delta = target - anchor
    gram = torch.einsum(
        "bpfd,bpgd->bpfg",
        directions,
        directions,
    )
    identity = torch.eye(
        int(features.shape[1]),
        dtype=gram.dtype,
        device=gram.device,
    )
    gram = gram + ridge * identity
    right = torch.einsum(
        "bpfd,bpd->bpf",
        directions,
        target_delta,
    )
    coefficients = torch.linalg.solve(
        gram,
        right.unsqueeze(-1),
    ).squeeze(-1)
    residual = _canonical_sum(
        coefficients.unsqueeze(-1) * directions,
        dim=2,
    )
    unconstrained_norm = residual.norm(dim=-1)
    scale = (
        residual_norm_cap
        / unconstrained_norm.clamp_min(1e-8)
    ).clamp_max(1.0)
    bounded = residual * scale.unsqueeze(-1)
    candidate = F.normalize(anchor + bounded, dim=-1, eps=1e-6)
    anchor_cosine = _teacher_cosine(anchor, target)
    candidate_cosine = _teacher_cosine(candidate, target)
    improved = candidate_cosine > anchor_cosine
    output = torch.where(improved.unsqueeze(-1), candidate, anchor)
    coefficients = torch.where(
        improved.unsqueeze(-1),
        coefficients * scale.unsqueeze(-1),
        torch.zeros_like(coefficients),
    )
    applied_residual = torch.where(
        improved.unsqueeze(-1),
        bounded,
        torch.zeros_like(bounded),
    )
    return _require_finite("bounded span output", output), {
        "coefficients": coefficients,
        "residual_norm": applied_residual.norm(dim=-1),
        "cap_hit": (
            (unconstrained_norm > residual_norm_cap) & improved
        ).float(),
    }


def apply_oracle_method(
    method: str,
    condition: torch.Tensor,
    quality: torch.Tensor,
    teacher: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if method == "teacher_best_frame_oracle":
        output, diagnostics = teacher_best_frame_oracle(
            condition,
            quality,
            teacher,
        )
    elif method == "teacher_convex_residual_oracle":
        output, diagnostics = teacher_convex_residual_oracle(
            condition,
            quality,
            teacher,
            iterations=int(config["p1_0"]["convex_iterations"]),
            step_size=float(config["p1_0"]["convex_step_size"]),
        )
    elif method == "teacher_bounded_span_residual_oracle":
        output, diagnostics = teacher_bounded_span_residual_oracle(
            condition,
            quality,
            teacher,
            ridge=float(config["p1_0"]["span_ridge"]),
            residual_norm_cap=float(
                config["p1_0"]["span_residual_norm_cap"]
            ),
        )
    else:
        raise ValueError(f"unsupported continuous residual oracle: {method}")
    anchor, _ = quality_anchor(condition, quality)
    teacher_gain = _teacher_cosine(output, teacher) - _teacher_cosine(
        anchor,
        teacher,
    )
    diagnostics = dict(diagnostics)
    diagnostics["residual_norm"] = (
        output.float() - anchor.float()
    ).norm(dim=-1)
    diagnostics["teacher_cosine_gain"] = teacher_gain
    return output, diagnostics


def _append_diagnostics(
    destination: dict[str, dict[str, list[torch.Tensor]]],
    method: str,
    diagnostics: Mapping[str, torch.Tensor],
) -> None:
    for key, value in diagnostics.items():
        destination.setdefault(method, {}).setdefault(key, []).append(
            value.detach().cpu()
        )


def _diagnostic_summary(
    *,
    method: str,
    diagnostics: Mapping[str, torch.Tensor],
    scenarios: Sequence[str],
    visibility: torch.Tensor,
    wrong_frames: torch.Tensor,
) -> dict[str, Any]:
    scenario_array = np.asarray(list(scenarios))
    visibility_by_position = visibility.permute(0, 2, 1).bool()
    result: dict[str, Any] = {}
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        residual_norm = diagnostics["residual_norm"][mask]
        teacher_gain = diagnostics["teacher_cosine_gain"][mask]
        values: dict[str, Any] = {
            "samples": int(mask.sum().item()),
            "residual_norm_mean": float(residual_norm.mean().item()),
            "residual_norm_p95": float(
                torch.quantile(residual_norm.flatten(), 0.95).item()
            ),
            "teacher_cosine_gain_mean": float(teacher_gain.mean().item()),
            "teacher_cosine_gain_min": float(teacher_gain.min().item()),
            "teacher_cosine_improved_fraction": float(
                (teacher_gain > 1e-7).float().mean().item()
            ),
        }
        if "weights" in diagnostics:
            weights = diagnostics["weights"][mask]
            effective = 1.0 / weights.square().sum(
                dim=-1
            ).clamp_min(1e-8)
            selected_visibility = visibility_by_position[mask]
            values.update(
                {
                    "mean_effective_frames": float(
                        effective.mean().item()
                    ),
                    "maximum_frame_weight_mean": float(
                        weights.max(dim=-1).values.mean().item()
                    ),
                    "visible_weight_mean": (
                        float(weights[selected_visibility].mean().item())
                        if bool(selected_visibility.any().item())
                        else None
                    ),
                    "occluded_weight_mean": (
                        float(weights[~selected_visibility].mean().item())
                        if bool((~selected_visibility).any().item())
                        else None
                    ),
                    "frame_mean_weights": [
                        float(weights[..., frame].mean().item())
                        for frame in range(weights.shape[-1])
                    ],
                }
            )
            selected_wrong = wrong_frames[mask]
            if bool(selected_wrong.any().item()):
                expanded_wrong = selected_wrong[:, None, :].expand_as(
                    weights
                )
                values["wrong_identity_frame_mean_weight"] = float(
                    weights[expanded_wrong].mean().item()
                )
        if "coefficients" in diagnostics:
            coefficients = diagnostics["coefficients"][mask]
            values.update(
                {
                    "coefficient_abs_mean": float(
                        coefficients.abs().mean().item()
                    ),
                    "coefficient_abs_max": float(
                        coefficients.abs().max().item()
                    ),
                    "negative_coefficient_fraction": float(
                        (coefficients < 0).float().mean().item()
                    ),
                    "cap_hit_fraction": float(
                        diagnostics["cap_hit"][mask].mean().item()
                    ),
                }
            )
        result[scenario] = values
    result["global"] = {
        "method": method,
        "teacher_supervision_at_inference": True,
        "deployable": False,
    }
    return result


@torch.inference_mode()
def evaluate_continuous_residual_split(
    *,
    split: str,
    dataset: TokenTrainingDataset,
    config: Mapping[str, Any],
    device: torch.device,
    oracle_methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if getattr(dataset.synthetic, "split", None) != split:
        raise ValueError("dataset split does not match requested split")
    configured = [
        str(value) for value in config["p1_0"]["candidate_methods"]
    ]
    methods = configured if oracle_methods is None else list(oracle_methods)
    if not methods or any(method not in configured for method in methods):
        raise ValueError("oracle_methods must be a non-empty configured subset")
    batch_size = int(config["evaluation"]["batch_size"])
    map_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in (*BASELINE_METHODS, *methods)
    }
    diagnostic_chunks: dict[
        str, dict[str, list[torch.Tensor]]
    ] = {}
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    visibility_chunks: list[torch.Tensor] = []
    wrong_chunks: list[torch.Tensor] = []
    maximum_norm_error = 0.0
    maximum_anchor_regression = 0.0
    observed_shapes: set[tuple[int, ...]] = set()

    def append_map(name: str, value: torch.Tensor) -> None:
        nonlocal maximum_norm_error
        _require_finite(f"{name} output", value)
        if tuple(value.shape[1:]) != (49, 128):
            raise RuntimeError(f"{name} returned invalid output shape")
        maximum_norm_error = max(
            maximum_norm_error,
            float((value.norm(dim=-1) - 1.0).abs().max().item()),
        )
        map_chunks[name].append(value.detach().cpu().half())

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"].to(device)
        quality = batch["condition_quality"].to(device)
        teacher = dataset.targets["continuous_prototypes"][
            indices
        ].float().to(device)
        observed_shapes.add(tuple(condition.shape))
        baselines = continuous_aggregation_baselines(condition, quality)
        anchor, _ = quality_anchor(condition, quality)
        maximum_anchor_regression = max(
            maximum_anchor_regression,
            float(
                (anchor - baselines["quality_pooling"])
                .abs()
                .max()
                .item()
            ),
        )
        append_map("mean_pooling", baselines["mean_pooling"])
        append_map("quality_pooling", baselines["quality_pooling"])
        for method in methods:
            output, diagnostics = apply_oracle_method(
                method,
                condition,
                quality,
                teacher,
                config=config,
            )
            append_map(method, output)
            _append_diagnostics(
                diagnostic_chunks,
                method,
                diagnostics,
            )
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(str(value) for value in batch["scenario"])
        visibility_chunks.append(batch["condition_visibility"].cpu())
        wrong_chunks.append(
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
    combined_diagnostics = {
        method: {
            key: torch.cat(chunks)
            for key, chunks in values.items()
        }
        for method, values in diagnostic_chunks.items()
    }
    visibility = torch.cat(visibility_chunks)
    wrong_frames = torch.cat(wrong_chunks)
    summaries = {
        method: _diagnostic_summary(
            method=method,
            diagnostics=values,
            scenarios=scenarios,
            visibility=visibility,
            wrong_frames=wrong_frames,
        )
        for method, values in combined_diagnostics.items()
    }

    first_count = min(batch_size, len(dataset))
    first_indices = list(range(first_count))
    first = default_collate([dataset[index] for index in first_indices])
    first_condition = first["condition_features"].to(device)
    first_quality = first["condition_quality"].to(device)
    first_teacher = dataset.targets["continuous_prototypes"][
        first_indices
    ].float().to(device)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    inverse = torch.argsort(permutation)
    method_correctness: dict[str, Any] = {}
    for method in methods:
        original, original_diagnostics = apply_oracle_method(
            method,
            first_condition,
            first_quality,
            first_teacher,
            config=config,
        )
        permuted, permuted_diagnostics = apply_oracle_method(
            method,
            first_condition[:, permutation],
            first_quality[:, permutation],
            first_teacher,
            config=config,
        )
        map_diff = float((original - permuted).abs().max().item())
        map_cosine = float(
            F.cosine_similarity(original, permuted, dim=-1).min().item()
        )
        routing_diff = 0.0
        routing_key = None
        if "weights" in original_diagnostics:
            routing_key = "weights"
        elif "coefficients" in original_diagnostics:
            routing_key = "coefficients"
        if routing_key is not None:
            routing_diff = float(
                (
                    original_diagnostics[routing_key]
                    - permuted_diagnostics[routing_key][..., inverse]
                )
                .abs()
                .max()
                .item()
            )
        minimum_gain = float(
            combined_diagnostics[method][
                "teacher_cosine_gain"
            ].min().item()
        )
        method_correctness[method] = {
            "permutation_map_max_abs_diff": map_diff,
            "permutation_map_cosine_min": map_cosine,
            "permutation_routing_max_abs_diff": routing_diff,
            "permutation_equivalent": (
                map_diff <= 1e-4
                and map_cosine >= 0.999999
                and routing_diff <= 1e-4
            ),
            "teacher_cosine_gain_min": minimum_gain,
            "teacher_cosine_monotonic": minimum_gain >= -1e-6,
        }
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
        "quality_anchor_regression_max_abs_diff": (
            maximum_anchor_regression
        ),
        "quality_anchor_reproduced": maximum_anchor_regression <= 1e-6,
        "methods": method_correctness,
        "all_oracle_methods_correct": all(
            values["permutation_equivalent"]
            and values["teacher_cosine_monotonic"]
            for values in method_correctness.values()
        ),
    }
    evaluated.update(
        {
            "split": split,
            "oracle_methods": methods,
            "oracle_supervision": (
                "per-probe continuous teacher prototype used only to choose "
                "condition-frame coefficients"
            ),
            "deployable": False,
            "diagnostics": summaries,
            "correctness": correctness,
        }
    )
    return evaluated


def select_continuous_residual_candidate(
    *,
    validation_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if validation_result["split"] != "val":
        raise ValueError("continuous residual selection must use validation")
    baseline = validation_result["methods"]["quality_pooling"]
    baseline_verification = baseline["verification"]
    baseline_teacher_cosine = baseline["teacher_similarity"]["all"][
        "map_cosine"
    ]
    thresholds = config["p1_0"]
    rows: list[dict[str, Any]] = []
    for method in thresholds["candidate_methods"]:
        values = validation_result["methods"][method]
        verification = values["verification"]
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
            values["teacher_similarity"]["all"]["map_cosine"]
            - baseline_teacher_cosine
        )
        wins = sum(delta > 0.0 for delta in scenario_deltas.values())
        method_correctness = validation_result["correctness"]["methods"][
            method
        ]
        checks = {
            "minimum_hard_gain": (
                hard_gain
                >= float(thresholds["selection_min_hard_auc_gain"])
            ),
            "complementary_gain": (
                scenario_deltas["complementary_occlusion"]
                > float(
                    thresholds[
                        "selection_min_complementary_auc_gain"
                    ]
                )
            ),
            "hard_scenario_wins": (
                wins
                >= int(
                    thresholds["selection_min_hard_scenario_wins"]
                )
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
                    thresholds[
                        "selection_min_teacher_map_cosine_gain"
                    ]
                )
            ),
            "correctness": (
                validation_result["correctness"]["output_shape_valid"]
                and validation_result["correctness"][
                    "output_unit_norm_valid"
                ]
                and validation_result["correctness"][
                    "quality_anchor_reproduced"
                ]
                and method_correctness["permutation_equivalent"]
                and method_correctness["teacher_cosine_monotonic"]
            ),
        }
        rows.append(
            {
                "method": method,
                "val_pooled_all_roc_auc": verification["all"]["roc_auc"],
                "val_hard_macro_roc_auc": verification["hard_average"][
                    "roc_auc"
                ],
                "val_clean_roc_auc": verification["clean"]["roc_auc"],
                "val_eer": verification["all"]["eer"],
                "val_rank1_accuracy": verification["all"][
                    "rank1_accuracy"
                ],
                "val_margin": verification["all"][
                    "mean_genuine_impostor_margin"
                ],
                "val_teacher_map_cosine": values["teacher_similarity"][
                    "all"
                ]["map_cosine"],
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
        )
    feasible = [row for row in rows if row["constraints_passed"]]
    ordered = sorted(
        feasible if feasible else rows,
        key=lambda row: (
            -float(row["val_hard_macro_roc_auc"]),
            -float(row["val_pooled_all_roc_auc"]),
            -float(row["val_teacher_map_cosine"]),
            list(thresholds["candidate_methods"]).index(row["method"]),
        ),
    )
    best = dict(ordered[0])
    return {
        "selection_split": "val",
        "test_metrics_consulted": False,
        "candidate_count": len(rows),
        "feasible_candidate_count": len(feasible),
        "test_authorized": bool(feasible),
        "selected": dict(ordered[0]) if feasible else None,
        "best_validation_candidate": best,
        "rows": rows,
        "thresholds": {
            key: value
            for key, value in thresholds.items()
            if key.startswith("selection_")
        },
        "selection_rule": (
            "Require preregistered hard AUC, complementary, hard-scenario "
            "wins, clean, Rank-1, teacher-map headroom, and correctness "
            "gates. Select by validation hard AUC, all AUC, then teacher-map "
            "cosine. If none pass, lock selected=null and do not construct "
            "the test TokenTrainingDataset."
        ),
    }


@torch.inference_mode()
def measure_continuous_residual_latency(
    *,
    dataset: TokenTrainingDataset,
    method: str,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    batch_size = 16
    if len(dataset) < batch_size:
        raise ValueError("latency measurement requires at least 16 samples")
    indices = list(range(batch_size))
    batch = default_collate([dataset[index] for index in indices])
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    teacher = dataset.targets["continuous_prototypes"][
        indices
    ].float().to(device)

    def quality_call() -> torch.Tensor:
        return quality_anchor(condition, quality)[0]

    def oracle_call() -> torch.Tensor:
        return apply_oracle_method(
            method,
            condition,
            quality,
            teacher,
            config=config,
        )[0]

    warmup = int(config["evaluation"]["latency_warmup_runs"])
    runs = int(config["evaluation"]["latency_measurement_runs"])
    results: dict[str, Any] = {}
    for name, function in (
        ("quality_pooling", quality_call),
        (method, oracle_call),
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
        results[name] = {
            "batch_size": batch_size,
            "warmup_runs": warmup,
            "measurement_runs": runs,
            "batch_latency_ms": elapsed * 1000.0 / runs,
            "per_set_latency_ms": elapsed * 1000.0 / (runs * batch_size),
            "teacher_required": name != "quality_pooling",
            "deployable": name == "quality_pooling",
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
    return results


def _test_gate(
    *,
    result: Mapping[str, Any],
    method: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = result["methods"]["quality_pooling"]["verification"]
    oracle = result["methods"][method]["verification"]
    deltas = {
        scenario: oracle[scenario]["roc_auc"] - baseline[scenario]["roc_auc"]
        for scenario in HARD_SCENARIOS
    }
    hard_gain = (
        oracle["hard_average"]["roc_auc"]
        - baseline["hard_average"]["roc_auc"]
    )
    teacher_gain = (
        result["methods"][method]["teacher_similarity"]["all"]["map_cosine"]
        - result["methods"]["quality_pooling"]["teacher_similarity"]["all"][
            "map_cosine"
        ]
    )
    p1 = config["p1_0"]
    checks = {
        "minimum_hard_gain": (
            hard_gain >= float(p1["selection_min_hard_auc_gain"])
        ),
        "complementary_gain": (
            deltas["complementary_occlusion"]
            > float(p1["selection_min_complementary_auc_gain"])
        ),
        "hard_scenario_wins": (
            sum(value > 0.0 for value in deltas.values())
            >= int(p1["selection_min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            baseline["clean"]["roc_auc"] - oracle["clean"]["roc_auc"]
            <= float(p1["selection_clean_auc_max_drop"])
        ),
        "rank1_constraint": (
            baseline["all"]["rank1_accuracy"]
            - oracle["all"]["rank1_accuracy"]
            <= float(p1["selection_rank1_max_drop"])
        ),
        "teacher_map_headroom": (
            teacher_gain
            >= float(p1["selection_min_teacher_map_cosine_gain"])
        ),
        "correctness": (
            result["correctness"]["output_shape_valid"]
            and result["correctness"]["output_unit_norm_valid"]
            and result["correctness"]["quality_anchor_reproduced"]
            and result["correctness"]["methods"][method][
                "permutation_equivalent"
            ]
            and result["correctness"]["methods"][method][
                "teacher_cosine_monotonic"
            ]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "hard_gain_vs_quality": hard_gain,
        "teacher_map_cosine_gain_vs_quality": teacher_gain,
        "scenario_auc_deltas": deltas,
    }


def build_continuous_residual_decision(
    *,
    selection: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    operational_names = (
        "dataset_valid_after_lock",
        "quantization_valid_after_lock",
        "source_artifacts_unchanged",
        "selection_lock_unchanged",
        "selection_used_validation_only",
        "test_policy_followed",
        "validation_outputs_correct",
        "test_outputs_correct_if_evaluated",
        "same_gallery_protocol",
        "no_shared_parameter_training",
        "memory_within_configured_cap",
        "protected_outputs_unchanged_before_artifact_write",
        "checkpoints_unchanged_before_artifact_write",
    )
    missing = [name for name in operational_names if name not in checks]
    if missing:
        raise ValueError(
            "continuous residual decision missing checks: "
            + ", ".join(missing)
        )
    failed = [name for name in operational_names if not checks[name]]
    operational_passed = not failed
    if not selection["test_authorized"]:
        return {
            "validation_headroom_passed": False,
            "test_evaluated": False,
            "test_gate": None,
            "scientific_gate_passed": False,
            "operational_gate_passed": operational_passed,
            "failed_operational_checks": failed,
            "go": False,
            "recommendation": (
                "stop_current_set2proto_representation_direction"
            ),
        }
    if test_result is None or selection["selected"] is None:
        raise ValueError("authorized residual oracle test result is missing")
    method = str(selection["selected"]["method"])
    gate = _test_gate(result=test_result, method=method, config=config)
    scientific = bool(gate["passed"])
    go = scientific and operational_passed
    return {
        "validation_headroom_passed": True,
        "test_evaluated": True,
        "test_gate": gate,
        "scientific_gate_passed": scientific,
        "operational_gate_passed": operational_passed,
        "failed_operational_checks": failed,
        "go": go,
        "recommendation": (
            "train_lightweight_continuous_residual_predictor_on_new_holdout"
            if go
            else "stop_current_set2proto_representation_direction"
        ),
    }


def _result_items(
    *,
    validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
) -> list[tuple[str, Mapping[str, Any]]]:
    values = [("val", validation_result)]
    if test_result is not None:
        values.append(("test", test_result))
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _search_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    selected = (
        None
        if selection["selected"] is None
        else selection["selected"]["method"]
    )
    for value in selection["rows"]:
        row = {
            key: item
            for key, item in value.items()
            if key not in {"checks", "scenario_auc_deltas"}
        }
        row["selected"] = value["method"] == selected
        for key, item in value["checks"].items():
            row[f"check_{key}"] = item
        for key, item in value["scenario_auc_deltas"].items():
            row[f"delta_auc_{key}"] = item
        rows.append(row)
    return rows


def _core_rows(
    *,
    validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
    latency: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, result in _result_items(
        validation_result=validation_result,
        test_result=test_result,
    ):
        baseline = result["methods"]["quality_pooling"]["verification"]
        for method, values in result["methods"].items():
            metrics = values["verification"]["all"]
            hard = values["verification"]["hard_average"]
            latency_values = latency.get(method, {})
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "teacher_guided_oracle": method in ORACLE_METHODS,
                    "deployable": method not in ORACLE_METHODS,
                    "pooled_all_roc_auc": metrics["roc_auc"],
                    "hard_macro_roc_auc": hard["roc_auc"],
                    "delta_hard_auc_vs_quality": (
                        hard["roc_auc"]
                        - baseline["hard_average"]["roc_auc"]
                    ),
                    "clean_roc_auc": values["verification"]["clean"][
                        "roc_auc"
                    ],
                    "eer": metrics["eer"],
                    "tar_at_far_1e-2": metrics["tar_at_far"]["0.01"][
                        "tar"
                    ],
                    "tar_at_far_1e-3": metrics["tar_at_far"]["0.001"][
                        "tar"
                    ],
                    "rank1_identification_accuracy": metrics[
                        "rank1_accuracy"
                    ],
                    "mean_genuine_impostor_margin": metrics[
                        "mean_genuine_impostor_margin"
                    ],
                    "teacher_map_cosine": values["teacher_similarity"][
                        "all"
                    ]["map_cosine"],
                    "batch16_latency_ms": latency_values.get(
                        "batch_latency_ms", ""
                    ),
                    "per_set_latency_ms": latency_values.get(
                        "per_set_latency_ms", ""
                    ),
                }
            )
    return rows


def _scenario_rows(
    *,
    validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, result in _result_items(
        validation_result=validation_result,
        test_result=test_result,
    ):
        baseline = result["methods"]["quality_pooling"]["verification"]
        for method, values in result["methods"].items():
            for scenario in ("clean", *HARD_SCENARIOS):
                metrics = values["verification"][scenario]
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "scenario": scenario,
                        "roc_auc": metrics["roc_auc"],
                        "delta_auc_vs_quality": (
                            metrics["roc_auc"]
                            - baseline[scenario]["roc_auc"]
                        ),
                        "eer": metrics["eer"],
                        "tar_at_far_1e-2": metrics["tar_at_far"][
                            "0.01"
                        ]["tar"],
                        "tar_at_far_1e-3": metrics["tar_at_far"][
                            "0.001"
                        ]["tar"],
                        "rank1_identification_accuracy": metrics[
                            "rank1_accuracy"
                        ],
                        "mean_genuine_impostor_margin": metrics[
                            "mean_genuine_impostor_margin"
                        ],
                        "teacher_map_cosine": values[
                            "teacher_similarity"
                        ][scenario]["map_cosine"],
                    }
                )
    return rows


def _diagnostic_rows(
    *,
    validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, result in _result_items(
        validation_result=validation_result,
        test_result=test_result,
    ):
        for method, scenarios in result["diagnostics"].items():
            for scenario, values in scenarios.items():
                if scenario == "global":
                    continue
                for metric, value in values.items():
                    if isinstance(value, list):
                        for index, item in enumerate(value):
                            rows.append(
                                {
                                    "split": split,
                                    "method": method,
                                    "scenario": scenario,
                                    "metric": f"{metric}_{index}",
                                    "value": item,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "split": split,
                                "method": method,
                                "scenario": scenario,
                                "metric": metric,
                                "value": value,
                            }
                        )
    return rows


def _report_text(
    *,
    selection: Mapping[str, Any],
    decision: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
    latency: Mapping[str, Any],
) -> str:
    baseline = validation_result["methods"]["quality_pooling"]
    rows = []
    for value in selection["rows"]:
        rows.append(
            "| "
            + " | ".join(
                (
                    value["method"],
                    f"{value['val_hard_macro_roc_auc']:.6f}",
                    f"{value['hard_gain_vs_quality']:+.6f}",
                    f"{value['teacher_map_cosine_gain_vs_quality']:+.6f}",
                    f"{value['complementary_gain_vs_quality']:+.6f}",
                    str(value["hard_scenario_wins"]),
                    str(value["constraints_passed"]),
                )
            )
            + " |"
        )
    best = selection["best_validation_candidate"]
    test_section = ""
    if test_result is not None and decision["test_gate"] is not None:
        gate = decision["test_gate"]
        scenario_lines = "\n".join(
            f"- {scenario}: {delta:+.6f}"
            for scenario, delta in gate["scenario_auc_deltas"].items()
        )
        test_section = f"""
## Locked test

- method: `{selection["selected"]["method"]}`
- hard AUC gain vs quality: {gate["hard_gain_vs_quality"]:+.6f}
- teacher-map cosine gain: {gate["teacher_map_cosine_gain_vs_quality"]:+.6f}
- gate passed: {gate["passed"]}

Scenario AUC deltas:

{scenario_lines}
"""
    elif not selection["test_authorized"]:
        test_section = """
## Locked test

Validation 没有候选通过全部门槛，因此 selection lock 为
`selected=null, test_authorized=false`；没有构造 test dataset 或运行 test。
"""
    latency_method = best["method"]
    latency_values = latency[latency_method]
    recommendation = (
        "只允许下一步训练轻量 continuous residual predictor，并在全新身份 "
        "holdout 复验；oracle 结果本身不是可部署性能。"
        if decision["go"]
        else "当前条件特征没有足够、稳定的连续残差 headroom；停止当前 "
        "Set2Proto 表征路线，不进入 residual diffusion 训练。"
    )
    return f"""# P1-0 continuous residual oracle headroom

## 协议

- Validation 样本：{validation_result["samples"]} 个 set，
  {validation_result["identities"]} 个身份。
- Anchor：现有 late `quality_pooling`。
- Oracle 只能重组同位置四帧，或在帧间差异张成的子空间内添加
  norm≤0.5 的连续残差。
- 每个 probe 的 teacher prototype 用于求 oracle 系数，因此所有 oracle
  都不可部署；该实验只测信息 headroom。
- Gallery 保持 clean disjoint-teacher continuous，不重新拟合 PCA/codebook。

## Validation candidate search

Quality pooling hard AUC：
{baseline["verification"]["hard_average"]["roc_auc"]:.6f}。

| Method | Hard AUC | Δ quality | Δ teacher cosine | Δ complementary | Hard wins | Pass |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

最佳 validation 候选：`{best["method"]}`，hard AUC
{best["val_hard_macro_roc_auc"]:.6f}，相对 quality
{best["hard_gain_vs_quality"]:+.6f}。
{test_section}
## Runtime

- measured oracle: `{latency_method}`
- batch=16 latency: {latency_values["batch_latency_ms"]:.3f} ms
- per-set latency: {latency_values["per_set_latency_ms"]:.3f} ms
- peak reserved CUDA memory: {latency_values["peak_reserved_mib"]:.1f} MiB
- teacher required at inference: True

## Go/No-Go

- validation headroom: {decision["validation_headroom_passed"]}
- test evaluated: {decision["test_evaluated"]}
- scientific gate: {decision["scientific_gate_passed"]}
- operational gate: {decision["operational_gate_passed"]}
- final Go: {decision["go"]}
- 建议：{recommendation}

现有 test identities 已在更早阶段被观察；即使本阶段 lock-before-test，
它也不是全项目层面的全新确认性 holdout，不声明统计显著。
"""


def write_continuous_residual_artifacts(
    *,
    output_directory: Path,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: Mapping[str, bool],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    search_path = output_directory / "validation_candidate_search.csv"
    core_path = output_directory / "core_comparison.csv"
    scenario_path = output_directory / "scenario_metrics.csv"
    residual_path = output_directory / "residual_diagnostics.csv"
    diagnostics_path = output_directory / "diagnostics.json"
    report_path = output_directory / "REPORT.md"
    _write_csv(search_path, _search_rows(selection))
    _write_csv(
        core_path,
        _core_rows(
            validation_result=validation_result,
            test_result=test_result,
            latency=latency,
        ),
    )
    _write_csv(
        scenario_path,
        _scenario_rows(
            validation_result=validation_result,
            test_result=test_result,
        ),
    )
    _write_csv(
        residual_path,
        _diagnostic_rows(
            validation_result=validation_result,
            test_result=test_result,
        ),
    )
    decision = build_continuous_residual_decision(
        selection=selection,
        test_result=test_result,
        config=config,
        checks=checks,
    )
    diagnostics = {
        "schema_version": 1,
        "preregistration": dict(preregistration),
        "selection": dict(selection),
        "decision": decision,
        "validation_correctness": validation_result["correctness"],
        "validation_residual_diagnostics": validation_result["diagnostics"],
        "test_correctness": (
            None if test_result is None else test_result["correctness"]
        ),
        "test_residual_diagnostics": (
            None if test_result is None else test_result["diagnostics"]
        ),
        "latency": dict(latency),
        "checks": dict(checks),
        "reproducibility": dict(reproducibility),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _report_text(
            selection=selection,
            decision=decision,
            validation_result=validation_result,
            test_result=test_result,
            latency=latency,
        ),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "validation_candidate_search_csv": str(search_path),
        "core_comparison_csv": str(core_path),
        "scenario_metrics_csv": str(scenario_path),
        "residual_diagnostics_csv": str(residual_path),
        "diagnostics_json": str(diagnostics_path),
        "decision": decision,
    }
