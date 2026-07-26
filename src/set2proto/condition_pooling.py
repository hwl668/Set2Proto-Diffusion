"""P0-1 deterministic robust condition-side pooling evaluation."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from .quantization import quantize_vectors, robust_condition_pool
from .research_evaluation import (
    _evaluate_embeddings,
    continuous_aggregation_baselines,
)
from .training import TokenTrainingDataset


METHOD_ORDER = (
    "mean_pooling",
    "quality_pooling",
    "mean_pooling_quantized",
    "quality_pooling_quantized",
    "condition_robust_pooling",
    "condition_robust_pooling_quantized",
)
QUANTIZED_METHODS = {
    "mean_pooling_quantized",
    "quality_pooling_quantized",
    "condition_robust_pooling_quantized",
}
P0_ROBUST_PARAMETERS = {
    "robust_keep_fraction": 0.75,
    "consensus_weight": 2.0,
    "quality_weight": 1.0,
}


def validate_p0_robust_parameters(config: Mapping[str, Any]) -> dict[str, float]:
    """Require the first P0-1 run to use the existing teacher parameters."""

    teacher = config["teacher"]
    observed = {
        "robust_keep_fraction": float(teacher["robust_keep_fraction"]),
        "consensus_weight": float(teacher["consensus_weight"]),
        "quality_weight": float(teacher["quality_weight"]),
    }
    for name, expected in P0_ROBUST_PARAMETERS.items():
        if not math.isclose(observed[name], expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"P0-1 requires {name}={expected}, got {observed[name]}"
            )
    return observed


def _summarize(values: torch.Tensor) -> dict[str, float | int | None]:
    flattened = values.detach().float().reshape(-1)
    flattened = flattened[torch.isfinite(flattened)]
    if flattened.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(flattened.numel()),
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "min": float(flattened.min().item()),
        "p05": float(torch.quantile(flattened, 0.05).item()),
        "median": float(flattened.median().item()),
        "p95": float(torch.quantile(flattened, 0.95).item()),
        "max": float(flattened.max().item()),
    }


def _scenario_masks(scenarios: Sequence[str]) -> dict[str, torch.Tensor]:
    scenario_array = np.asarray(list(scenarios))
    masks = {
        scenario: torch.from_numpy(scenario_array == scenario)
        for scenario in sorted(set(scenarios))
    }
    masks["all"] = torch.ones(len(scenarios), dtype=torch.bool)
    return masks


def _map_similarity_metrics(
    method_maps: torch.Tensor,
    teacher_maps: torch.Tensor,
    scenarios: Sequence[str],
) -> dict[str, dict[str, float]]:
    method_local = F.normalize(method_maps.float(), dim=-1)
    teacher_local = F.normalize(teacher_maps.float(), dim=-1)
    method_embeddings = F.normalize(method_local.mean(dim=1), dim=-1)
    teacher_embeddings = F.normalize(teacher_local.mean(dim=1), dim=-1)
    map_cosine = F.cosine_similarity(
        method_embeddings,
        teacher_embeddings,
        dim=-1,
    )
    flattened_cosine = F.cosine_similarity(
        method_local.flatten(1),
        teacher_local.flatten(1),
        dim=-1,
    )
    local_cosine = F.cosine_similarity(
        method_local,
        teacher_local,
        dim=-1,
    ).mean(dim=1)
    result: dict[str, dict[str, float]] = {}
    for scenario, mask in _scenario_masks(scenarios).items():
        result[scenario] = {
            "map_cosine": float(map_cosine[mask].mean().item()),
            "flattened_map_cosine": float(
                flattened_cosine[mask].mean().item()
            ),
            "mean_local_cosine": float(local_cosine[mask].mean().item()),
        }
    return result


def _token_accuracy_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scenarios: Sequence[str],
) -> dict[str, float]:
    matches = (predicted.to(torch.int64) == target.to(torch.int64)).float()
    return {
        scenario: float(matches[mask].mean().item())
        for scenario, mask in _scenario_masks(scenarios).items()
    }


def _wrong_frame_mask(
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    frame_count: int,
) -> torch.Tensor:
    rows = getattr(dataset.synthetic, "rows", None)
    if rows is None:
        raise ValueError("P0-1 diagnostics require real-set row metadata")
    result = torch.zeros((len(indices), frame_count), dtype=torch.bool)
    for batch_index, dataset_index in enumerate(indices):
        for position in rows[dataset_index].get("wrong_positions", []):
            result[batch_index, int(position)] = True
    return result


def _weight_diagnostics(
    *,
    split: str,
    weights: torch.Tensor,
    keep_mask: torch.Tensor,
    consensus: torch.Tensor,
    effective_frames: torch.Tensor,
    visibility: torch.Tensor,
    wrong_frames: torch.Tensor,
    quality_only_weights: torch.Tensor,
    scenarios: Sequence[str],
    grid_height: int,
    grid_width: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize [N,P,F] weights and emit long-form diagnostic rows."""

    if tuple(visibility.shape) != (
        weights.shape[0],
        weights.shape[2],
        weights.shape[1],
    ):
        raise ValueError("visibility must have shape [N,F,P]")
    visibility_by_position = visibility.permute(0, 2, 1).bool()
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    def add_row(
        *,
        scenario: str,
        scope: str,
        metric: str,
        value: float | int | None,
        count: int,
        frame_index: int | None = None,
        position: int | None = None,
    ) -> None:
        rows.append(
            {
                "split": split,
                "scenario": scenario,
                "scope": scope,
                "position": "" if position is None else position,
                "grid_row": (
                    "" if position is None else position // grid_width
                ),
                "grid_column": (
                    "" if position is None else position % grid_width
                ),
                "frame_index": "" if frame_index is None else frame_index,
                "metric": metric,
                "value": "" if value is None else value,
                "count": count,
            }
        )

    scenario_array = np.asarray(list(scenarios))
    for scenario in sorted(set(scenarios)):
        sample_mask = torch.from_numpy(scenario_array == scenario)
        selected_weights = weights[sample_mask]
        selected_keep = keep_mask[sample_mask]
        selected_consensus = consensus[sample_mask]
        selected_effective = effective_frames[sample_mask]
        selected_visibility = visibility_by_position[sample_mask]
        selected_wrong = wrong_frames[sample_mask]
        selected_quality_only_weights = quality_only_weights[sample_mask]
        scenario_summary: dict[str, Any] = {
            "samples": int(sample_mask.sum().item()),
            "frame_mean_weights": [
                float(selected_weights[:, :, frame].mean().item())
                for frame in range(weights.shape[-1])
            ],
            "frame_trimmed_fractions": [
                float((~selected_keep[:, :, frame]).float().mean().item())
                for frame in range(weights.shape[-1])
            ],
            "mean_effective_frames": float(selected_effective.mean().item()),
            "trimmed_frame_fraction": float(
                (~selected_keep).float().mean().item()
            ),
            "weight_statistics": _summarize(selected_weights),
            "consensus_statistics": _summarize(selected_consensus),
            "visible_weight_mean": (
                float(selected_weights[selected_visibility].mean().item())
                if bool(selected_visibility.any().item())
                else None
            ),
            "occluded_weight_mean": (
                float(selected_weights[~selected_visibility].mean().item())
                if bool((~selected_visibility).any().item())
                else None
            ),
        }
        for frame in range(weights.shape[-1]):
            add_row(
                scenario=scenario,
                scope="frame",
                metric="mean_weight",
                value=scenario_summary["frame_mean_weights"][frame],
                count=int(selected_weights[:, :, frame].numel()),
                frame_index=frame,
            )
            add_row(
                scenario=scenario,
                scope="frame",
                metric="trimmed_fraction",
                value=scenario_summary["frame_trimmed_fractions"][frame],
                count=int(selected_keep[:, :, frame].numel()),
                frame_index=frame,
            )
        for metric, value in (
            ("mean_effective_frames", scenario_summary["mean_effective_frames"]),
            ("trimmed_frame_fraction", scenario_summary["trimmed_frame_fraction"]),
            ("visible_weight_mean", scenario_summary["visible_weight_mean"]),
            ("occluded_weight_mean", scenario_summary["occluded_weight_mean"]),
        ):
            add_row(
                scenario=scenario,
                scope="scenario",
                metric=metric,
                value=value,
                count=int(selected_weights.numel()),
            )

        if bool(selected_wrong.any().item()):
            wrong_by_position = selected_wrong[:, None, :].expand_as(
                selected_weights
            )
            correct_by_position = ~wrong_by_position
            scenario_summary.update(
                {
                    "wrong_identity_frame_mean_weight": float(
                        selected_weights[wrong_by_position].mean().item()
                    ),
                    "correct_identity_frame_mean_weight": float(
                        selected_weights[correct_by_position].mean().item()
                    ),
                    "wrong_identity_frame_trimmed_fraction": float(
                        (~selected_keep[wrong_by_position]).float().mean().item()
                    ),
                    "correct_identity_frame_trimmed_fraction": float(
                        (~selected_keep[correct_by_position]).float().mean().item()
                    ),
                    "wrong_identity_frame_mean_consensus": float(
                        selected_consensus[wrong_by_position].mean().item()
                    ),
                    "correct_identity_frame_mean_consensus": float(
                        selected_consensus[correct_by_position].mean().item()
                    ),
                    "wrong_identity_quality_only_mean_weight": float(
                        selected_quality_only_weights[
                            wrong_by_position
                        ].mean().item()
                    ),
                    "consensus_effect_on_wrong_identity_weight": float(
                        selected_weights[wrong_by_position].mean().item()
                        - selected_quality_only_weights[
                            wrong_by_position
                        ].mean().item()
                    ),
                }
            )
            for metric, count in (
                (
                    "wrong_identity_frame_mean_weight",
                    int(wrong_by_position.sum().item()),
                ),
                (
                    "correct_identity_frame_mean_weight",
                    int(correct_by_position.sum().item()),
                ),
                (
                    "wrong_identity_frame_trimmed_fraction",
                    int(wrong_by_position.sum().item()),
                ),
                (
                    "correct_identity_frame_trimmed_fraction",
                    int(correct_by_position.sum().item()),
                ),
                (
                    "wrong_identity_frame_mean_consensus",
                    int(wrong_by_position.sum().item()),
                ),
                (
                    "correct_identity_frame_mean_consensus",
                    int(correct_by_position.sum().item()),
                ),
                (
                    "wrong_identity_quality_only_mean_weight",
                    int(wrong_by_position.sum().item()),
                ),
                (
                    "consensus_effect_on_wrong_identity_weight",
                    int(wrong_by_position.sum().item()),
                ),
            ):
                add_row(
                    scenario=scenario,
                    scope="identity_membership",
                    metric=metric,
                    value=scenario_summary[metric],
                    count=count,
                )

        if scenario in {"complementary_occlusion", "common_occlusion"}:
            position_summaries: list[dict[str, Any]] = []
            for position in range(weights.shape[1]):
                position_weights = selected_weights[:, position, :]
                position_visibility = selected_visibility[:, position, :]
                position_stats = {
                    **_summarize(position_weights),
                    "position": position,
                    "grid_row": position // grid_width,
                    "grid_column": position % grid_width,
                    "mean_effective_frames": float(
                        selected_effective[:, position].mean().item()
                    ),
                    "trimmed_fraction": float(
                        (~selected_keep[:, position, :]).float().mean().item()
                    ),
                    "visible_weight_mean": (
                        float(
                            position_weights[position_visibility].mean().item()
                        )
                        if bool(position_visibility.any().item())
                        else None
                    ),
                    "occluded_weight_mean": (
                        float(
                            position_weights[~position_visibility].mean().item()
                        )
                        if bool((~position_visibility).any().item())
                        else None
                    ),
                }
                position_summaries.append(position_stats)
                for metric in (
                    "mean",
                    "std",
                    "min",
                    "p05",
                    "median",
                    "p95",
                    "max",
                    "mean_effective_frames",
                    "trimmed_fraction",
                    "visible_weight_mean",
                    "occluded_weight_mean",
                ):
                    add_row(
                        scenario=scenario,
                        scope="position",
                        position=position,
                        metric=metric,
                        value=position_stats[metric],
                        count=int(position_weights.numel()),
                    )
            scenario_summary["by_position"] = position_summaries
        summary[scenario] = scenario_summary
    summary["global"] = {
        "weight_sum_max_abs_error": float(
            (weights.sum(dim=-1) - 1.0).abs().max().item()
        ),
        "mean_effective_frames": float(effective_frames.mean().item()),
        "trimmed_frame_fraction": float((~keep_mask).float().mean().item()),
    }
    return summary, rows


@torch.inference_mode()
def evaluate_condition_pooling_split(
    *,
    split: str,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the six P0-1 methods against the existing clean gallery."""

    parameters = validate_p0_robust_parameters(config)
    batch_size = int(config["evaluation"]["batch_size"])
    condition_frames = int(config["data"]["condition_frames"])
    grid = config["data"]["feature_grid"]
    expected_shape_tail = (
        int(grid["height"]) * int(grid["width"]),
        int(grid["projected_dim"]),
    )
    codebook_device = F.normalize(codebook.float(), dim=-1).to(device)

    map_chunks: dict[str, list[torch.Tensor]] = {
        method: [] for method in METHOD_ORDER
    }
    token_chunks: dict[str, list[torch.Tensor]] = {
        method: [] for method in QUANTIZED_METHODS
    }
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    weight_chunks: list[torch.Tensor] = []
    keep_chunks: list[torch.Tensor] = []
    consensus_chunks: list[torch.Tensor] = []
    effective_chunks: list[torch.Tensor] = []
    quality_only_weight_chunks: list[torch.Tensor] = []
    visibility_chunks: list[torch.Tensor] = []
    wrong_frame_chunks: list[torch.Tensor] = []
    observed_shapes: set[tuple[int, ...]] = set()
    maximum_norm_error = 0.0
    maximum_weight_sum_error = 0.0

    def append_map(name: str, maps: torch.Tensor) -> None:
        nonlocal maximum_norm_error
        if tuple(maps.shape[1:]) != expected_shape_tail:
            raise RuntimeError(f"{name} returned invalid shape {tuple(maps.shape)}")
        maximum_norm_error = max(
            maximum_norm_error,
            float((maps.float().norm(dim=-1) - 1.0).abs().max().item()),
        )
        map_chunks[name].append(maps.detach().cpu().half())

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"][:, :condition_frames].to(device)
        quality = batch["condition_quality"][:, :condition_frames].to(device)
        observed_shapes.add(tuple(condition.shape))
        if tuple(condition.shape[1:]) != (
            condition_frames,
            *expected_shape_tail,
        ):
            raise RuntimeError(
                "TokenTrainingDataset returned an invalid condition shape: "
                f"{tuple(condition.shape)}"
            )
        baselines = continuous_aggregation_baselines(condition, quality)
        robust_maps, robust_diagnostics = robust_condition_pool(
            condition,
            quality,
            keep_fraction=parameters["robust_keep_fraction"],
            consensus_weight=parameters["consensus_weight"],
            quality_weight=parameters["quality_weight"],
        )
        _, quality_only_diagnostics = robust_condition_pool(
            condition,
            quality,
            keep_fraction=parameters["robust_keep_fraction"],
            consensus_weight=0.0,
            quality_weight=parameters["quality_weight"],
        )
        continuous_maps = {
            "mean_pooling": baselines["mean_pooling"],
            "quality_pooling": baselines["quality_pooling"],
            "condition_robust_pooling": robust_maps,
        }
        for name, maps in continuous_maps.items():
            append_map(name, maps)
            quantized_tokens, _ = quantize_vectors(maps, codebook_device)
            quantized_name = f"{name}_quantized"
            token_chunks[quantized_name].append(
                quantized_tokens.detach().cpu()
            )
            append_map(
                quantized_name,
                codebook_device[quantized_tokens].float(),
            )

        weights = robust_diagnostics["weights"]
        maximum_weight_sum_error = max(
            maximum_weight_sum_error,
            float((weights.sum(dim=-1) - 1.0).abs().max().item()),
        )
        weight_chunks.append(weights.detach().cpu())
        keep_chunks.append(robust_diagnostics["keep_mask"].detach().cpu())
        consensus_chunks.append(
            robust_diagnostics["consensus"].detach().cpu()
        )
        effective_chunks.append(
            robust_diagnostics["effective_frames"].detach().cpu()
        )
        quality_only_weight_chunks.append(
            quality_only_diagnostics["weights"].detach().cpu()
        )
        visibility_chunks.append(
            batch["condition_visibility"][:, :condition_frames].cpu()
        )
        wrong_frame_chunks.append(
            _wrong_frame_mask(dataset, indices, condition_frames)
        )
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(str(value) for value in batch["scenario"])

    all_maps = {
        name: torch.cat(chunks).float()
        for name, chunks in map_chunks.items()
    }
    all_tokens = {
        name: torch.cat(chunks).to(torch.int64)
        for name, chunks in token_chunks.items()
    }
    teacher_maps = dataset.targets["continuous_prototypes"].float()
    target_tokens = dataset.targets["tokens"].to(torch.int64)
    identity_tensor = torch.cat(identities)
    embeddings = {
        name: F.normalize(maps.mean(dim=1), dim=-1)
        for name, maps in all_maps.items()
    }
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identity_tensor[clean_indices]
    gallery_embeddings = F.normalize(
        teacher_maps[clean_indices].mean(dim=1),
        dim=-1,
    )
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    if int(torch.unique(gallery_identities).numel()) != len(clean_indices):
        raise RuntimeError("clean gallery must have one entry per identity")
    far_targets = [float(value) for value in config["evaluation"]["far_targets"]]

    methods: dict[str, Any] = {}
    for name in METHOD_ORDER:
        methods[name] = {
            "verification": _evaluate_embeddings(
                embeddings=embeddings[name],
                gallery_embeddings=gallery_embeddings,
                identities=identity_tensor,
                gallery_identities=gallery_identities,
                scenarios=scenarios,
                far_targets=far_targets,
            ),
            "teacher_similarity": _map_similarity_metrics(
                all_maps[name],
                teacher_maps,
                scenarios,
            ),
            "exact_token_accuracy": (
                _token_accuracy_metrics(
                    all_tokens[name],
                    target_tokens,
                    scenarios,
                )
                if name in QUANTIZED_METHODS
                else None
            ),
        }

    weight_summary, weight_rows = _weight_diagnostics(
        split=split,
        weights=torch.cat(weight_chunks),
        keep_mask=torch.cat(keep_chunks),
        consensus=torch.cat(consensus_chunks),
        effective_frames=torch.cat(effective_chunks),
        visibility=torch.cat(visibility_chunks),
        wrong_frames=torch.cat(wrong_frame_chunks),
        quality_only_weights=torch.cat(quality_only_weight_chunks),
        scenarios=scenarios,
        grid_height=int(grid["height"]),
        grid_width=int(grid["width"]),
    )

    first_batch = default_collate(
        [dataset[index] for index in range(min(batch_size, len(dataset)))]
    )
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    original_condition = first_batch["condition_features"].to(device)
    original_quality = first_batch["condition_quality"].to(device)
    original_pool, original_diagnostics = robust_condition_pool(
        original_condition,
        original_quality,
        keep_fraction=parameters["robust_keep_fraction"],
        consensus_weight=parameters["consensus_weight"],
        quality_weight=parameters["quality_weight"],
    )
    permuted_pool, permuted_diagnostics = robust_condition_pool(
        original_condition[:, permutation],
        original_quality[:, permutation],
        keep_fraction=parameters["robust_keep_fraction"],
        consensus_weight=parameters["consensus_weight"],
        quality_weight=parameters["quality_weight"],
    )
    inverse_permutation = torch.argsort(permutation)
    permutation_pool_max_abs_diff = float(
        (original_pool - permuted_pool).abs().max().item()
    )
    permutation_weight_max_abs_diff = float(
        (
            original_diagnostics["weights"]
            - permuted_diagnostics["weights"][..., inverse_permutation]
        )
        .abs()
        .max()
        .item()
    )
    correctness = {
        "condition_shapes_observed": [list(shape) for shape in observed_shapes],
        "output_shape_tail": list(expected_shape_tail),
        "output_shape_valid": all(
            tuple(maps.shape) == (len(dataset), *expected_shape_tail)
            for maps in all_maps.values()
        ),
        "output_unit_norm_max_abs_error": maximum_norm_error,
        "output_unit_norm_valid": maximum_norm_error <= 1e-5,
        "frame_weight_sum_max_abs_error": maximum_weight_sum_error,
        "frame_weight_sum_valid": maximum_weight_sum_error <= 1e-6,
        "permutation": permutation.tolist(),
        "permutation_pool_max_abs_diff": permutation_pool_max_abs_diff,
        "permutation_weight_max_abs_diff": permutation_weight_max_abs_diff,
        "permutation_equivalent": (
            permutation_pool_max_abs_diff <= 1e-5
            and permutation_weight_max_abs_diff <= 1e-5
        ),
        "kept_frames_per_position": int(
            torch.cat(keep_chunks).sum(dim=-1).unique().item()
        ),
    }
    return {
        "split": split,
        "samples": len(dataset),
        "identities": len(clean_indices),
        "gallery": "clean_disjoint_teacher_continuous",
        "robust_parameters": parameters,
        "methods": methods,
        "weight_diagnostics": weight_summary,
        "weight_rows": weight_rows,
        "correctness": correctness,
    }


@torch.inference_mode()
def measure_condition_pooling_latency(
    *,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Measure end-to-end aggregation latency for one cached batch of 16."""

    parameters = validate_p0_robust_parameters(config)
    batch_size = 16
    if len(dataset) < batch_size:
        raise ValueError("latency measurement requires at least 16 samples")
    batch = default_collate([dataset[index] for index in range(batch_size)])
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    codebook_device = F.normalize(codebook.float(), dim=-1).to(device)

    def mean_pool() -> torch.Tensor:
        return F.normalize(condition.float().mean(dim=1), dim=-1)

    def quality_pool() -> torch.Tensor:
        normalized = quality.float().clamp_min(1e-6)
        normalized = normalized / normalized.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        return F.normalize(
            (normalized[:, :, None, None] * condition.float()).sum(dim=1),
            dim=-1,
        )

    def robust_pool() -> torch.Tensor:
        return robust_condition_pool(
            condition,
            quality,
            keep_fraction=parameters["robust_keep_fraction"],
            consensus_weight=parameters["consensus_weight"],
            quality_weight=parameters["quality_weight"],
        )[0]

    def quantized(function: Callable[[], torch.Tensor]) -> torch.Tensor:
        tokens, _ = quantize_vectors(function(), codebook_device)
        return codebook_device[tokens]

    methods: dict[str, Callable[[], torch.Tensor]] = {
        "mean_pooling": mean_pool,
        "quality_pooling": quality_pool,
        "mean_pooling_quantized": lambda: quantized(mean_pool),
        "quality_pooling_quantized": lambda: quantized(quality_pool),
        "condition_robust_pooling": robust_pool,
        "condition_robust_pooling_quantized": lambda: quantized(robust_pool),
    }
    warmup_runs = int(config["evaluation"]["latency_warmup_runs"])
    measurement_runs = int(config["evaluation"]["latency_measurement_runs"])
    results: dict[str, Any] = {}
    for name in METHOD_ORDER:
        function = methods[name]
        for _ in range(warmup_runs):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(measurement_runs):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        results[name] = {
            "device": str(device),
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "batch_latency_ms": elapsed * 1000.0 / measurement_runs,
            "per_set_latency_ms": (
                elapsed * 1000.0 / (measurement_runs * batch_size)
            ),
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _core_rows(
    split_results: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for method in METHOD_ORDER:
            values = split_results[split]["methods"][method]
            verification = values["verification"]
            all_metrics = verification["all"]
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "pooled_all_roc_auc": all_metrics["roc_auc"],
                    "hard_macro_roc_auc": verification["hard_average"][
                        "roc_auc"
                    ],
                    "eer": all_metrics["eer"],
                    "tar_at_far_1e-2": all_metrics["tar_at_far"]["0.01"][
                        "tar"
                    ],
                    "tar_at_far_1e-3": all_metrics["tar_at_far"]["0.001"][
                        "tar"
                    ],
                    "rank1_identification_accuracy": all_metrics[
                        "rank1_accuracy"
                    ],
                    "mean_genuine_impostor_margin": all_metrics[
                        "mean_genuine_impostor_margin"
                    ],
                    "teacher_map_cosine": values["teacher_similarity"]["all"][
                        "map_cosine"
                    ],
                    "teacher_flattened_map_cosine": values[
                        "teacher_similarity"
                    ]["all"]["flattened_map_cosine"],
                    "teacher_mean_local_cosine": values[
                        "teacher_similarity"
                    ]["all"]["mean_local_cosine"],
                    "exact_token_accuracy": (
                        ""
                        if values["exact_token_accuracy"] is None
                        else values["exact_token_accuracy"]["all"]
                    ),
                    "batch16_latency_ms": latency[method][
                        "batch_latency_ms"
                    ],
                    "per_set_latency_ms": latency[method][
                        "per_set_latency_ms"
                    ],
                }
            )
    return rows


def _scenario_rows(split_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for method in METHOD_ORDER:
            values = split_results[split]["methods"][method]
            for scenario in (
                "clean",
                "low_quality",
                "complementary_occlusion",
                "common_occlusion",
                "wrong_identity",
            ):
                verification = values["verification"][scenario]
                similarity = values["teacher_similarity"][scenario]
                token_accuracy = values["exact_token_accuracy"]
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "scenario": scenario,
                        "roc_auc": verification["roc_auc"],
                        "eer": verification["eer"],
                        "tar_at_far_1e-2": verification["tar_at_far"][
                            "0.01"
                        ]["tar"],
                        "tar_at_far_1e-3": verification["tar_at_far"][
                            "0.001"
                        ]["tar"],
                        "rank1_identification_accuracy": verification[
                            "rank1_accuracy"
                        ],
                        "mean_genuine_score": verification[
                            "mean_genuine_score"
                        ],
                        "mean_strongest_impostor_score": verification[
                            "mean_strongest_impostor_score"
                        ],
                        "mean_genuine_impostor_margin": verification[
                            "mean_genuine_impostor_margin"
                        ],
                        "teacher_map_cosine": similarity["map_cosine"],
                        "teacher_flattened_map_cosine": similarity[
                            "flattened_map_cosine"
                        ],
                        "teacher_mean_local_cosine": similarity[
                            "mean_local_cosine"
                        ],
                        "exact_token_accuracy": (
                            ""
                            if token_accuracy is None
                            else token_accuracy[scenario]
                        ),
                    }
                )
    return rows


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _report_text(
    *,
    split_results: Mapping[str, Any],
    latency: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> tuple[str, dict[str, Any]]:
    hard_scenarios = (
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    )
    comparisons: dict[str, Any] = {}
    for split in ("val", "test"):
        robust = split_results[split]["methods"][
            "condition_robust_pooling"
        ]
        quality = split_results[split]["methods"]["quality_pooling"]
        robust_quantized = split_results[split]["methods"][
            "condition_robust_pooling_quantized"
        ]
        quality_quantized = split_results[split]["methods"][
            "quality_pooling_quantized"
        ]
        comparisons[split] = {
            "robust_minus_quality_pooled_auc": (
                robust["verification"]["all"]["roc_auc"]
                - quality["verification"]["all"]["roc_auc"]
            ),
            "robust_minus_quality_hard_macro_auc": (
                robust["verification"]["hard_average"]["roc_auc"]
                - quality["verification"]["hard_average"]["roc_auc"]
            ),
            "robust_minus_quality_by_scenario_auc": {
                scenario: (
                    robust["verification"][scenario]["roc_auc"]
                    - quality["verification"][scenario]["roc_auc"]
                )
                for scenario in hard_scenarios
            },
            "robust_quantized_minus_quality_quantized_hard_macro_auc": (
                robust_quantized["verification"]["hard_average"]["roc_auc"]
                - quality_quantized["verification"]["hard_average"]["roc_auc"]
            ),
            "robust_quantization_hard_macro_auc_loss": (
                robust["verification"]["hard_average"]["roc_auc"]
                - robust_quantized["verification"]["hard_average"]["roc_auc"]
            ),
            "robust_quantization_map_cosine_loss": (
                robust["teacher_similarity"]["all"]["map_cosine"]
                - robust_quantized["teacher_similarity"]["all"]["map_cosine"]
            ),
        }

    test_weights = split_results["test"]["weight_diagnostics"][
        "wrong_identity"
    ]
    wrong_weight = test_weights["wrong_identity_frame_mean_weight"]
    correct_weight = test_weights["correct_identity_frame_mean_weight"]
    quality_only_wrong_weight = test_weights[
        "wrong_identity_quality_only_mean_weight"
    ]
    consensus_effect = test_weights[
        "consensus_effect_on_wrong_identity_weight"
    ]
    wrong_consensus = test_weights[
        "wrong_identity_frame_mean_consensus"
    ]
    correct_consensus = test_weights[
        "correct_identity_frame_mean_consensus"
    ]
    consensus_downweights_wrong = (
        consensus_effect < 0.0 and wrong_consensus < correct_consensus
    )
    validation_supports = (
        comparisons["val"]["robust_minus_quality_hard_macro_auc"] > 0.0
    )
    test_supports = (
        comparisons["test"]["robust_minus_quality_hard_macro_auc"] > 0.0
    )
    affected_test_wins = sum(
        value > 0.0
        for value in comparisons["test"][
            "robust_minus_quality_by_scenario_auc"
        ].values()
    )
    enter_p0_2 = (
        validation_supports
        and test_supports
        and affected_test_wins >= 2
        and consensus_downweights_wrong
        and all(checks.values())
    )
    decision = {
        "enter_p0_2": enter_p0_2,
        "recommendation": (
            "进入 P0-2：继续确定性 evidence aggregation，但固定 val 选型并保留同一协议。"
            if enter_p0_2
            else (
                "暂不进入 P0-2：当前固定 robust rule 未在 validation/test 的 hard "
                "macro AUC 上同时超过 quality pooling；late AdaFace hook 下应先判定"
                " quality pooling 接近饱和或重新审视更早的局部特征。"
            )
        ),
        "consensus_downweights_wrong_identity": consensus_downweights_wrong,
        "wrong_identity_frame_mean_weight_test": wrong_weight,
        "correct_identity_frame_mean_weight_test": correct_weight,
        "wrong_identity_quality_only_mean_weight_test": (
            quality_only_wrong_weight
        ),
        "consensus_effect_on_wrong_identity_weight_test": consensus_effect,
        "wrong_identity_frame_mean_consensus_test": wrong_consensus,
        "correct_identity_frame_mean_consensus_test": correct_consensus,
        "affected_test_scenario_wins": affected_test_wins,
        "comparisons": comparisons,
    }

    table_lines = [
        "| Split | Method | All AUC | Hard macro AUC | EER | TAR@1e-2 | "
        "TAR@1e-3 | Rank-1 | Margin | Teacher map cos | Token acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("val", "test"):
        for method in METHOD_ORDER:
            values = split_results[split]["methods"][method]
            all_metrics = values["verification"]["all"]
            token_accuracy = values["exact_token_accuracy"]
            table_lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        method,
                        _format_metric(all_metrics["roc_auc"]),
                        _format_metric(
                            values["verification"]["hard_average"]["roc_auc"]
                        ),
                        _format_metric(all_metrics["eer"]),
                        _format_metric(
                            all_metrics["tar_at_far"]["0.01"]["tar"]
                        ),
                        _format_metric(
                            all_metrics["tar_at_far"]["0.001"]["tar"]
                        ),
                        _format_metric(all_metrics["rank1_accuracy"]),
                        _format_metric(
                            all_metrics["mean_genuine_impostor_margin"]
                        ),
                        _format_metric(
                            values["teacher_similarity"]["all"]["map_cosine"]
                        ),
                        (
                            "—"
                            if token_accuracy is None
                            else _format_metric(token_accuracy["all"])
                        ),
                    ]
                )
                + " |"
            )

    scenario_lines = []
    for split in ("val", "test"):
        deltas = comparisons[split]["robust_minus_quality_by_scenario_auc"]
        scenario_lines.append(
            f"- {split}: "
            + ", ".join(
                f"{scenario} {value:+.6f}"
                for scenario, value in deltas.items()
            )
        )
    report = f"""# P0-1 条件侧确定性 robust local aggregation

## 实验约束

- 使用既有 identity-disjoint train/val/test；本阶段只评估 val 和 test。
- S=4、T=4，gallery 仍为 `clean_disjoint_teacher_continuous`。
- PCA(512→128)、K=1024 codebook 和 teacher target 均直接读取 stage10 产物，
  没有重新拟合。
- robust 参数固定为 keep fraction=0.75、consensus weight=2.0、
  quality weight=1.0；每位置保留 3/4 帧。没有使用 test 调参。
- 没有加载或训练 Transformer，没有写入 checkpoint，也没有覆盖 stage15。

## 核心结果

{chr(10).join(table_lines)}

All AUC 是 pooled-all ROC-AUC；Hard macro AUC 是四个受扰场景的非加权均值。
主 `teacher map cos` 沿用项目既有定义：先对 49 个局部 token 空间均值并归一化，
再与 continuous teacher prototype 求 cosine；它是 identity-embedding map cosine，
不是展平 `[49,128]` 的 cosine。后者及 mean-local cosine 同时保存在
`core_comparison.csv` 和 `scenario_metrics.csv`。

## robust 相对 quality pooling

- val pooled-all AUC 差值：
  {comparisons["val"]["robust_minus_quality_pooled_auc"]:+.6f}；hard macro AUC
  差值：{comparisons["val"]["robust_minus_quality_hard_macro_auc"]:+.6f}。
- test pooled-all AUC 差值：
  {comparisons["test"]["robust_minus_quality_pooled_auc"]:+.6f}；hard macro AUC
  差值：{comparisons["test"]["robust_minus_quality_hard_macro_auc"]:+.6f}。
- 分场景 AUC 差值（robust - quality）：
{chr(10).join(scenario_lines)}
- 提升没有稳定集中在目标受扰条件：两 split 的 complementary 都约下降
  0.012；wrong-ID 仅 test 微升、val 微降；common 仅 val 上升、test 微降。
- 结论是“总体未超过”而不是所有指标都退化：robust 的 val EER 为
  {split_results["val"]["methods"]["condition_robust_pooling"]["verification"]["all"]["eer"]:.6f}
  （quality
  {split_results["val"]["methods"]["quality_pooling"]["verification"]["all"]["eer"]:.6f}），
  test TAR@FAR=1e-3 为
  {split_results["test"]["methods"]["condition_robust_pooling"]["verification"]["all"]["tar_at_far"]["0.001"]["tar"]:.6f}
  （quality
  {split_results["test"]["methods"]["quality_pooling"]["verification"]["all"]["tar_at_far"]["0.001"]["tar"]:.6f}），
  val/test 平均 margin 也略高；但 pooled/hard AUC 和 Rank-1 在两 split
  均更低。

## 权重机制是否按预期工作

- test wrong-ID 帧平均权重：{wrong_weight:.6f}；正确身份帧平均权重：
  {correct_weight:.6f}；均匀权重参考值为 0.25。
- wrong/correct 帧的平均 local consensus 分别为 {wrong_consensus:.6f} /
  {correct_consensus:.6f}。仅关闭 consensus 项、保留相同 quality 与 trimming 时，
  wrong-ID 帧平均权重为 {quality_only_wrong_weight:.6f}；加入 consensus 后变化
  {consensus_effect:+.6f}。
- 因而 consensus {'确实' if consensus_downweights_wrong else '没有'}在这个固定规则中
  进一步降低错误身份帧权重。该 counterfactual 只用于机制诊断，不用于 test 选参。
- 但 complementary occlusion 出现反向机制：test 可见帧/被遮挡帧的平均权重为
  {split_results["test"]["weight_diagnostics"]["complementary_occlusion"]["visible_weight_mean"]:.6f}
  /
  {split_results["test"]["weight_diagnostics"]["complementary_occlusion"]["occluded_weight_mean"]:.6f}。
  每位置只有一帧可见时，三帧遮挡特征彼此更有“共识”，反而压低了唯一可见证据；
  这与该场景 AUC 的最大退化一致。
- 逐场景 frame weight、effective frames、trim 比例，以及 complementary/common
  的 49 个位置统计见 `weight_diagnostics.csv` 和 `diagnostics.json`。

## 量化影响

- val robust 量化后的 hard macro AUC 损失：
  {comparisons["val"]["robust_quantization_hard_macro_auc_loss"]:+.6f}；
  teacher map cosine 差：
  {comparisons["val"]["robust_quantization_map_cosine_loss"]:+.6f}。
- test robust 量化后的 hard macro AUC 损失：
  {comparisons["test"]["robust_quantization_hard_macro_auc_loss"]:+.6f}；
  teacher map cosine 差：
  {comparisons["test"]["robust_quantization_map_cosine_loss"]:+.6f}。
- robust-quantized 相对 quality-quantized 的 hard macro AUC：val
  {comparisons["val"]["robust_quantized_minus_quality_quantized_hard_macro_auc"]:+.6f}，
  test
  {comparisons["test"]["robust_quantized_minus_quality_quantized_hard_macro_auc"]:+.6f}。
- robust-quantized 的 exact token accuracy 略高于 quality-quantized（val
  {split_results["val"]["methods"]["condition_robust_pooling_quantized"]["exact_token_accuracy"]["all"]:.6f}
  vs.
  {split_results["val"]["methods"]["quality_pooling_quantized"]["exact_token_accuracy"]["all"]:.6f}；
  test
  {split_results["test"]["methods"]["condition_robust_pooling_quantized"]["exact_token_accuracy"]["all"]:.6f}
  vs.
  {split_results["test"]["methods"]["quality_pooling_quantized"]["exact_token_accuracy"]["all"]:.6f}），
  但 verification hard AUC 明显更低，因而没有达到 quantized quality pooling。

## 延迟

batch=16、缓存 PCA 特征、同一设备；连续方法只计聚合，量化方法包含
nearest-cosine 与 codebook gather。每个方法的 batch/per-set 延迟记录在
`core_comparison.csv` 和 `diagnostics.json`。

## P0-2 建议

{decision["recommendation"]}

这里的“不进入”只否定当前固定 robust rule 的直接延伸，不能证明 quality pooling
已经统计意义上饱和。这是一次 pilot 数据上的确定性对照，不进行显著性声明，也不将
test 结果用于选参。
"""
    return report, decision


def write_condition_pooling_artifacts(
    *,
    output_directory: Path,
    split_results: Mapping[str, Any],
    latency: Mapping[str, Any],
    checks: Mapping[str, bool],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Write all required P0-1 artifacts without touching prior stages."""

    output_directory.mkdir(parents=True, exist_ok=True)
    core_path = output_directory / "core_comparison.csv"
    scenario_path = output_directory / "scenario_metrics.csv"
    weight_path = output_directory / "weight_diagnostics.csv"
    diagnostics_path = output_directory / "diagnostics.json"
    report_path = output_directory / "REPORT.md"
    _write_csv(core_path, _core_rows(split_results, latency))
    _write_csv(scenario_path, _scenario_rows(split_results))
    weight_rows = [
        row
        for split in ("val", "test")
        for row in split_results[split]["weight_rows"]
    ]
    _write_csv(weight_path, weight_rows)
    report, decision = _report_text(
        split_results=split_results,
        latency=latency,
        checks=checks,
    )
    diagnostics_payload = {
        "schema_version": 1,
        "protocol": {
            "gallery": "clean_disjoint_teacher_continuous",
            "methods": list(METHOD_ORDER),
            "robust_parameters": P0_ROBUST_PARAMETERS,
            "test_parameter_tuning": False,
        },
        "checks": dict(checks),
        "latency": latency,
        "correctness": {
            split: split_results[split]["correctness"]
            for split in ("val", "test")
        },
        "weight_diagnostics": {
            split: split_results[split]["weight_diagnostics"]
            for split in ("val", "test")
        },
        "decision": decision,
        "reproducibility": dict(reproducibility),
    }
    diagnostics_path.write_text(
        json.dumps(
            diagnostics_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "core_comparison_csv": str(core_path),
        "scenario_metrics_csv": str(scenario_path),
        "weight_diagnostics_csv": str(weight_path),
        "diagnostics_json": str(diagnostics_path),
        "decision": decision,
    }
