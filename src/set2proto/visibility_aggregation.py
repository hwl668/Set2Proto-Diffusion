"""P0-2 visibility/reliability-aware deterministic local aggregation."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from .condition_pooling import (
    _map_similarity_metrics,
    _token_accuracy_metrics,
)
from .quantization import quantize_vectors, robust_condition_pool
from .research_evaluation import (
    _evaluate_embeddings,
    continuous_aggregation_baselines,
)
from .training import TokenTrainingDataset


FINAL_METHOD_ORDER = (
    "mean_pooling",
    "quality_pooling",
    "quality_pooling_quantized",
    "condition_robust_pooling",
    "oracle_visibility_pooling",
    "identity_gate_only",
    "local_reliability_only",
    "visibility_identity_pooling",
    "visibility_identity_pooling_quantized",
)
QUANTIZED_METHODS = {
    "quality_pooling_quantized",
    "visibility_identity_pooling_quantized",
}


@dataclass(frozen=True)
class CleanReferenceBank:
    """Train-only clean local-feature support and calibration statistics."""

    vectors: torch.Tensor
    centers: dict[int, torch.Tensor]
    scales: dict[int, torch.Tensor]
    teacher_indices: torch.Tensor
    teacher_identities: torch.Tensor
    identity_gap_threshold: float
    identity_gap_scale: float
    clean_condition_sets: int

    @property
    def positions(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def references(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[2])

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fit_split": "train",
            "method": "positionwise_clean_teacher_topk_cosine",
            "vectors": self.vectors.cpu().half(),
            "centers": {
                str(key): value.cpu().float()
                for key, value in self.centers.items()
            },
            "scales": {
                str(key): value.cpu().float()
                for key, value in self.scales.items()
            },
            "teacher_indices": self.teacher_indices.cpu(),
            "teacher_identities": self.teacher_identities.cpu(),
            "identity_gap_threshold": self.identity_gap_threshold,
            "identity_gap_scale": self.identity_gap_scale,
            "clean_condition_sets": self.clean_condition_sets,
        }


def _frame_consensus_and_gaps(
    condition_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-2 frame support and permutation-invariant low gaps."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,F,P,D]")
    frame_embeddings = F.normalize(
        F.normalize(condition_features.float(), dim=-1).mean(dim=2),
        dim=-1,
    )
    pairwise = torch.matmul(
        frame_embeddings,
        frame_embeddings.transpose(-1, -2),
    )
    frames = int(frame_embeddings.shape[1])
    if frames < 2:
        raise ValueError("identity gating requires at least two frames")
    diagonal = torch.eye(
        frames,
        dtype=torch.bool,
        device=pairwise.device,
    ).view(1, frames, frames)
    pairwise = pairwise.masked_fill(diagonal, float("-inf"))
    neighbor_count = min(2, frames - 1)
    consensus = pairwise.topk(
        k=neighbor_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).values.mean(dim=-1)
    sorted_consensus = consensus.sort(dim=1).values
    if frames % 2:
        center = sorted_consensus[:, frames // 2]
    else:
        center = 0.5 * (
            sorted_consensus[:, frames // 2 - 1]
            + sorted_consensus[:, frames // 2]
        )
    gaps = center[:, None] - consensus
    return consensus, gaps


def identity_outlier_penalty(
    condition_features: torch.Tensor,
    *,
    threshold: float,
    scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize only frame-level identity gaps beyond clean calibration."""

    if scale <= 0:
        raise ValueError("identity gap scale must be positive")
    consensus, gaps = _frame_consensus_and_gaps(condition_features)
    penalty = F.relu((gaps - float(threshold)) / float(scale)).clamp_max(4.0)
    return penalty, {
        "frame_consensus": consensus,
        "identity_gap": gaps,
    }


def local_reliability_scores(
    condition_features: torch.Tensor,
    *,
    reference_vectors: torch.Tensor,
    centers: Mapping[int, torch.Tensor],
    scales: Mapping[int, torch.Tensor],
    top_k_values: Sequence[int],
    z_clip: tuple[float, float],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Score local tokens against train-only clean references at each position."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,F,P,D]")
    features = F.normalize(condition_features.float(), dim=-1)
    references = F.normalize(reference_vectors.float(), dim=-1)
    if tuple(references.shape[:1]) != (features.shape[2],):
        raise ValueError("reference bank position count does not match")
    if int(references.shape[-1]) != int(features.shape[-1]):
        raise ValueError("reference bank feature dimension does not match")
    selected_top_k = sorted({int(value) for value in top_k_values})
    if not selected_top_k or selected_top_k[0] <= 0:
        raise ValueError("top_k_values must contain positive integers")
    if selected_top_k[-1] > int(references.shape[1]):
        raise ValueError("top-k exceeds clean reference count")
    similarities = torch.einsum("bfpd,pnd->bfpn", features, references)
    largest = similarities.topk(
        k=selected_top_k[-1],
        dim=-1,
        largest=True,
        sorted=True,
    ).values
    raw: dict[int, torch.Tensor] = {}
    standardized: dict[int, torch.Tensor] = {}
    lower, upper = z_clip
    if lower >= upper:
        raise ValueError("z_clip must be increasing")
    for top_k in selected_top_k:
        values = largest[..., :top_k].mean(dim=-1)
        center = centers[top_k].to(values.device).view(1, 1, -1)
        scale = scales[top_k].to(values.device).view(1, 1, -1)
        raw[top_k] = values
        standardized[top_k] = (
            (values - center) / scale.clamp_min(1e-8)
        ).clamp(min=lower, max=upper)
    return standardized, raw


def reliability_weighted_pool(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    *,
    local_reliability: torch.Tensor | None,
    identity_penalty: torch.Tensor | None,
    local_weight: float,
    identity_weight: float,
    quality_weight: float,
    minimum_frame_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pool with clean typicality and global identity outlier penalties."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,F,P,D]")
    batch, frames, positions, _ = condition_features.shape
    if tuple(condition_quality.shape) != (batch, frames):
        raise ValueError("condition_quality must have shape [B,F]")
    if not 0.0 <= minimum_frame_weight < 1.0 / float(frames):
        raise ValueError("minimum_frame_weight must be in [0, 1/F)")
    if any(value < 0 for value in (local_weight, identity_weight, quality_weight)):
        raise ValueError("aggregation weights must be non-negative")
    features = F.normalize(condition_features.float(), dim=-1)
    quality = condition_quality.float().clamp_min(1e-6)
    quality_score = torch.log(
        quality / quality.mean(dim=1, keepdim=True).clamp_min(1e-6)
    )
    logits = quality_weight * quality_score[:, None, :].expand(
        batch,
        positions,
        frames,
    )
    local_absolute_gate = torch.zeros(
        (batch, positions),
        dtype=logits.dtype,
        device=logits.device,
    )
    if local_reliability is not None:
        if tuple(local_reliability.shape) != (batch, frames, positions):
            raise ValueError("local_reliability must have shape [B,F,P]")
        local = local_reliability.float()
        # The train-calibrated zero point is the fifth percentile of clean
        # cross-identity support. If every frame is below it, local evidence
        # must vanish instead of merely choosing the least-bad missing frame.
        local_absolute_gate = local.max(dim=1).values.clamp(0.0, 1.0)
        local_centered = local - local.mean(dim=1, keepdim=True)
        local_term = (
            local_centered
            * local_absolute_gate[:, None, :]
        )
        logits = logits + local_weight * local_term.permute(0, 2, 1)
    if identity_penalty is not None:
        if tuple(identity_penalty.shape) != (batch, frames):
            raise ValueError("identity_penalty must have shape [B,F]")
        logits = logits - identity_weight * identity_penalty[:, None, :]
    weights = torch.softmax(logits, dim=-1)
    if minimum_frame_weight > 0:
        weights = (
            (1.0 - frames * minimum_frame_weight) * weights
            + minimum_frame_weight
        )
    features_by_position = features.permute(0, 2, 1, 3)
    pooled = torch.sum(
        weights.unsqueeze(-1) * features_by_position,
        dim=2,
    )
    pooled_norm = pooled.norm(dim=-1, keepdim=True)
    # Exact cancellation is pathological but possible. A symmetric
    # second-moment fallback preserves unit norm and frame permutation
    # invariance without selecting an arbitrary frame.
    fallback = torch.sum(
        weights.unsqueeze(-1) * features_by_position.square(),
        dim=2,
    )
    pooled = torch.where(
        pooled_norm > 1e-8,
        pooled,
        fallback,
    )
    pooled = F.normalize(pooled, dim=-1)
    return pooled, {
        "weights": weights,
        "logits": logits,
        "local_absolute_gate": local_absolute_gate,
        "pre_normalization_min_norm": pooled_norm.min(),
        "effective_frames": 1.0
        / weights.square().sum(dim=-1).clamp_min(1e-8),
    }


def oracle_visibility_pool(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    condition_visibility: torch.Tensor,
    same_identity: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Metadata-assisted headroom diagnostic; never a deployable method."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,F,P,D]")
    batch, frames, positions, _ = condition_features.shape
    if tuple(condition_quality.shape) != (batch, frames):
        raise ValueError("condition_quality must have shape [B,F]")
    if tuple(condition_visibility.shape) != (batch, frames, positions):
        raise ValueError("condition_visibility must have shape [B,F,P]")
    if tuple(same_identity.shape) != (batch, frames):
        raise ValueError("same_identity must have shape [B,F]")
    features = F.normalize(condition_features.float(), dim=-1)
    quality = condition_quality.float().clamp_min(1e-6)
    visible = condition_visibility.float().clamp(0.0, 1.0)
    identity_mask = same_identity.float().clamp(0.0, 1.0)
    weighted_visibility = (
        visible * identity_mask[:, :, None] * quality[:, :, None]
    )
    denominator = weighted_visibility.sum(dim=1, keepdim=True)
    visible_weights = weighted_visibility / denominator.clamp_min(1e-8)
    identity_quality = quality * identity_mask
    identity_denominator = identity_quality.sum(dim=1, keepdim=True)
    fallback = identity_quality / identity_denominator.clamp_min(1e-8)
    all_quality = quality / quality.sum(dim=1, keepdim=True).clamp_min(1e-8)
    fallback = torch.where(identity_denominator > 0, fallback, all_quality)
    fallback = fallback[:, :, None].expand(batch, frames, positions)
    weights_by_frame = torch.where(
        denominator > 0,
        visible_weights,
        fallback,
    )
    weights = weights_by_frame.permute(0, 2, 1)
    features_by_position = features.permute(0, 2, 1, 3)
    pooled = torch.sum(
        weights.unsqueeze(-1) * features_by_position,
        dim=2,
    )
    pooled_norm = pooled.norm(dim=-1, keepdim=True)
    fallback_map = torch.sum(
        weights.unsqueeze(-1) * features_by_position.square(),
        dim=2,
    )
    pooled = torch.where(pooled_norm > 1e-8, pooled, fallback_map)
    return F.normalize(pooled, dim=-1), {
        "weights": weights,
        "effective_frames": 1.0
        / weights.square().sum(dim=-1).clamp_min(1e-8),
        "fallback_positions": (denominator[:, 0] <= 0),
        "pre_normalization_min_norm": pooled_norm.min(),
    }


@torch.inference_mode()
def build_train_clean_reference(
    *,
    dataset: TokenTrainingDataset,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[CleanReferenceBank, dict[str, Any]]:
    """Build and calibrate a position-wise bank from unique train teachers."""

    if getattr(dataset.synthetic, "split", None) != "train":
        raise ValueError("clean reference bank must be built from train split")
    rows = getattr(dataset.synthetic, "rows", None)
    clean_features = getattr(dataset.synthetic, "clean_features", None)
    if rows is None or clean_features is None:
        raise ValueError("clean reference bank requires cached real-set data")
    index_to_identity: dict[int, int] = {}
    condition_indices: set[int] = set()
    source_indices: set[int] = set()
    teacher_occurrences = 0
    for row in rows:
        identity = int(row["target_identity"])
        condition_indices.update(int(value) for value in row["condition_indices"])
        source_indices.update(
            int(value)
            for value in row.get("source_indices", row["condition_indices"])
        )
        for index in row["teacher_indices"]:
            teacher_occurrences += 1
            selected = int(index)
            previous = index_to_identity.setdefault(selected, identity)
            if previous != identity:
                raise ValueError("teacher index maps to multiple identities")
    teacher_indices = sorted(index_to_identity)
    if not teacher_indices:
        raise ValueError("train split has no clean teacher references")
    target_condition_overlap = sorted(
        condition_indices.intersection(teacher_indices)
    )
    actual_source_overlap = sorted(source_indices.intersection(teacher_indices))
    if target_condition_overlap:
        raise ValueError("train clean teacher bank overlaps condition images")

    projected_chunks: list[torch.Tensor] = []
    for start in range(0, len(teacher_indices), batch_size):
        selected = teacher_indices[start : start + batch_size]
        raw = torch.from_numpy(
            np.array(clean_features[np.asarray(selected)], copy=True)
        ).float()
        projected_chunks.append(dataset.projector.transform(raw).half())
    references_by_image = torch.cat(projected_chunks, dim=0)
    reference_vectors = references_by_image.permute(1, 0, 2).contiguous()
    p0_config = config["p0_2"]
    top_k_values = sorted(
        {int(value) for value in p0_config["reference_top_k_candidates"]}
    )
    if top_k_values[-1] >= len(teacher_indices):
        raise ValueError("reference top-k must be less than bank size")
    scale_floor = float(p0_config["reliability_scale_floor"])
    centers = {
        top_k: torch.empty(reference_vectors.shape[0])
        for top_k in top_k_values
    }
    scales = {
        top_k: torch.empty(reference_vectors.shape[0])
        for top_k in top_k_values
    }
    maximum_top_k = top_k_values[-1]
    reference_identities_device = torch.tensor(
        [index_to_identity[index] for index in teacher_indices],
        dtype=torch.int64,
        device=device,
    )
    same_reference_identity = (
        reference_identities_device[:, None]
        == reference_identities_device[None, :]
    )
    for position in range(reference_vectors.shape[0]):
        position_vectors = F.normalize(
            reference_vectors[position].float().to(device),
            dim=-1,
        )
        similarities = torch.matmul(
            position_vectors,
            position_vectors.transpose(0, 1),
        )
        similarities.masked_fill_(same_reference_identity, float("-inf"))
        nearest = similarities.topk(
            k=maximum_top_k,
            dim=-1,
            largest=True,
            sorted=True,
        ).values
        for top_k in top_k_values:
            clean_scores = nearest[:, :top_k].mean(dim=-1)
            center = torch.quantile(clean_scores, 0.05)
            q25 = torch.quantile(clean_scores, 0.25)
            q75 = torch.quantile(clean_scores, 0.75)
            robust_scale = ((q75 - q25) / 1.349).clamp_min(scale_floor)
            centers[top_k][position] = center.cpu()
            scales[top_k][position] = robust_scale.cpu()

    if dataset._precomputed is None:
        dataset.precompute()
    assert dataset._precomputed is not None
    clean_set_indices = [
        index
        for index, scenario in enumerate(dataset._precomputed["scenarios"])
        if scenario == "clean"
    ]
    if not clean_set_indices:
        raise ValueError("train split has no clean condition sets")
    clean_condition = dataset._precomputed["condition_features"][
        clean_set_indices
    ]
    gap_chunks: list[torch.Tensor] = []
    for start in range(0, len(clean_condition), batch_size):
        _, gaps = _frame_consensus_and_gaps(
            clean_condition[start : start + batch_size].to(device)
        )
        gap_chunks.append(gaps.cpu())
    clean_gaps = torch.cat(gap_chunks).reshape(-1)
    gap_quantile = float(p0_config["clean_gap_quantile"])
    identity_gap_threshold = float(
        torch.quantile(clean_gaps, gap_quantile).item()
    )
    identity_gap_scale = max(
        float(clean_gaps.std(unbiased=False).item()),
        float(p0_config["identity_scale_floor"]),
    )
    bank = CleanReferenceBank(
        vectors=reference_vectors.half().cpu(),
        centers=centers,
        scales=scales,
        teacher_indices=torch.tensor(teacher_indices, dtype=torch.int64),
        teacher_identities=torch.tensor(
            [index_to_identity[index] for index in teacher_indices],
            dtype=torch.int64,
        ),
        identity_gap_threshold=identity_gap_threshold,
        identity_gap_scale=identity_gap_scale,
        clean_condition_sets=len(clean_set_indices),
    )
    details = {
        "fit_split": "train",
        "unique_teacher_references": bank.references,
        "teacher_occurrences_before_deduplication": teacher_occurrences,
        "deduplication_fraction": (
            1.0 - bank.references / float(teacher_occurrences)
        ),
        "unique_teacher_identities": int(
            torch.unique(bank.teacher_identities).numel()
        ),
        "target_condition_teacher_overlap": len(target_condition_overlap),
        "actual_source_teacher_overlap": len(actual_source_overlap),
        "actual_source_teacher_overlap_indices": actual_source_overlap,
        "shape": list(bank.vectors.shape),
        "top_k_candidates": top_k_values,
        "center_ranges": {
            str(top_k): [
                float(centers[top_k].min().item()),
                float(centers[top_k].max().item()),
            ]
            for top_k in top_k_values
        },
        "scale_ranges": {
            str(top_k): [
                float(scales[top_k].min().item()),
                float(scales[top_k].max().item()),
            ]
            for top_k in top_k_values
        },
        "identity_gap_threshold": identity_gap_threshold,
        "identity_gap_scale": identity_gap_scale,
        "clean_condition_sets": len(clean_set_indices),
    }
    return bank, details


def _same_identity_frame_mask(
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    frame_count: int,
) -> torch.Tensor:
    rows = getattr(dataset.synthetic, "rows", None)
    if rows is None:
        raise ValueError("identity diagnostics require real-set row metadata")
    result = torch.empty((len(indices), frame_count), dtype=torch.bool)
    for batch_index, dataset_index in enumerate(indices):
        row = rows[dataset_index]
        target_identity = int(row["target_identity"])
        source_identities = [
            int(value) for value in row["source_identity_ids"]
        ]
        if len(source_identities) != frame_count:
            raise ValueError("source_identity_ids does not match frame count")
        derived_wrong = {
            position
            for position, identity in enumerate(source_identities)
            if identity != target_identity
        }
        annotated_wrong = {
            int(value) for value in row.get("wrong_positions", [])
        }
        if derived_wrong != annotated_wrong:
            raise ValueError(
                "wrong_positions disagrees with source identity metadata"
            )
        result[batch_index] = torch.tensor(
            [
                identity == target_identity
                for identity in source_identities
            ],
            dtype=torch.bool,
        )
    return result


def _wrong_frame_mask(
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    frame_count: int,
) -> torch.Tensor:
    return ~_same_identity_frame_mask(dataset, indices, frame_count)


def _weight_summary(
    *,
    weights: torch.Tensor,
    visibility: torch.Tensor,
    wrong_frames: torch.Tensor,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    """Summarize [N,P,F] weights using perturbation metadata."""

    visibility_by_position = visibility.permute(0, 2, 1).bool()
    scenario_array = np.asarray(list(scenarios))
    result: dict[str, Any] = {}
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        selected = weights[mask]
        selected_visibility = visibility_by_position[mask]
        selected_wrong = wrong_frames[mask]
        effective = 1.0 / selected.square().sum(dim=-1).clamp_min(1e-8)
        values: dict[str, Any] = {
            "samples": int(mask.sum().item()),
            "frame_mean_weights": [
                float(selected[:, :, frame].mean().item())
                for frame in range(selected.shape[-1])
            ],
            "mean_effective_frames": float(effective.mean().item()),
            "visible_weight_mean": (
                float(selected[selected_visibility].mean().item())
                if bool(selected_visibility.any().item())
                else None
            ),
            "occluded_weight_mean": (
                float(selected[~selected_visibility].mean().item())
                if bool((~selected_visibility).any().item())
                else None
            ),
            "weight_min": float(selected.min().item()),
            "weight_max": float(selected.max().item()),
            "zero_weight_fraction": float((selected == 0).float().mean().item()),
        }
        if bool(selected_wrong.any().item()):
            wrong = selected_wrong[:, None, :].expand_as(selected)
            values["wrong_identity_frame_mean_weight"] = float(
                selected[wrong].mean().item()
            )
            values["correct_identity_frame_mean_weight"] = float(
                selected[~wrong].mean().item()
            )
        if scenario in {"complementary_occlusion", "common_occlusion"}:
            by_position: list[dict[str, Any]] = []
            for position in range(selected.shape[1]):
                position_weights = selected[:, position, :]
                position_visibility = selected_visibility[:, position, :]
                position_effective = (
                    1.0
                    / position_weights.square().sum(dim=-1).clamp_min(1e-8)
                )
                by_position.append(
                    {
                        "position": position,
                        "mean_weight": float(position_weights.mean().item()),
                        "std_weight": float(
                            position_weights.std(unbiased=False).item()
                        ),
                        "min_weight": float(position_weights.min().item()),
                        "max_weight": float(position_weights.max().item()),
                        "mean_effective_frames": float(
                            position_effective.mean().item()
                        ),
                        "visible_weight_mean": (
                            float(
                                position_weights[
                                    position_visibility
                                ].mean().item()
                            )
                            if bool(position_visibility.any().item())
                            else None
                        ),
                        "occluded_weight_mean": (
                            float(
                                position_weights[
                                    ~position_visibility
                                ].mean().item()
                            )
                            if bool((~position_visibility).any().item())
                            else None
                        ),
                    }
                )
            values["by_position"] = by_position
        result[scenario] = values
    result["global"] = {
        "weight_sum_max_abs_error": float(
            (weights.sum(dim=-1) - 1.0).abs().max().item()
        )
    }
    return result


def _evaluate_collected_maps(
    *,
    dataset: TokenTrainingDataset,
    map_chunks: Mapping[str, list[torch.Tensor]],
    token_chunks: Mapping[str, list[torch.Tensor]],
    identities: list[torch.Tensor],
    scenarios: list[str],
    far_targets: list[float],
) -> dict[str, Any]:
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
        raise RuntimeError("clean gallery must contain one entry per identity")
    methods: dict[str, Any] = {}
    for name, maps in all_maps.items():
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
                maps,
                teacher_maps,
                scenarios,
            ),
            "exact_token_accuracy": (
                _token_accuracy_metrics(
                    all_tokens[name],
                    target_tokens,
                    scenarios,
                )
                if name in all_tokens
                else None
            ),
        }
    return {
        "samples": len(dataset),
        "identities": len(clean_indices),
        "gallery": "clean_disjoint_teacher_continuous",
        "methods": methods,
    }


def _candidate_name(top_k: int, identity_weight: float) -> str:
    weight = f"{identity_weight:g}".replace(".", "p")
    return f"combined_k{top_k}_identity{weight}"


@torch.inference_mode()
def evaluate_validation_candidates(
    *,
    dataset: TokenTrainingDataset,
    reference: CleanReferenceBank,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the predeclared combined grid on validation only."""

    if getattr(dataset.synthetic, "split", None) != "val":
        raise ValueError("candidate selection may only use validation split")
    p0_config = config["p0_2"]
    top_k_values = sorted(
        {int(value) for value in p0_config["reference_top_k_candidates"]}
    )
    identity_weights = [
        float(value) for value in p0_config["identity_weight_candidates"]
    ]
    batch_size = int(config["evaluation"]["batch_size"])
    bank_device = reference.vectors.float().to(device)
    centers_device = {
        key: value.to(device) for key, value in reference.centers.items()
    }
    scales_device = {
        key: value.to(device) for key, value in reference.scales.items()
    }
    z_clip = tuple(float(value) for value in p0_config["local_z_clip"])
    map_chunks: dict[str, list[torch.Tensor]] = {
        "quality_pooling": [],
        **{
            _candidate_name(top_k, identity_weight): []
            for top_k in top_k_values
            for identity_weight in identity_weights
        },
    }
    candidate_weight_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in map_chunks if name != "quality_pooling"
    }
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    visibility_chunks: list[torch.Tensor] = []
    wrong_chunks: list[torch.Tensor] = []

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"].to(device)
        quality = batch["condition_quality"].to(device)
        baselines = continuous_aggregation_baselines(condition, quality)
        map_chunks["quality_pooling"].append(
            baselines["quality_pooling"].detach().cpu().half()
        )
        penalty, _ = identity_outlier_penalty(
            condition,
            threshold=reference.identity_gap_threshold,
            scale=reference.identity_gap_scale,
        )
        local_scores, _ = local_reliability_scores(
            condition,
            reference_vectors=bank_device,
            centers=centers_device,
            scales=scales_device,
            top_k_values=top_k_values,
            z_clip=z_clip,
        )
        for top_k in top_k_values:
            for identity_weight in identity_weights:
                name = _candidate_name(top_k, identity_weight)
                pooled, diagnostics = reliability_weighted_pool(
                    condition,
                    quality,
                    local_reliability=local_scores[top_k],
                    identity_penalty=penalty,
                    local_weight=float(
                        p0_config["local_reliability_weight"]
                    ),
                    identity_weight=identity_weight,
                    quality_weight=float(p0_config["quality_weight"]),
                    minimum_frame_weight=float(
                        p0_config["minimum_frame_weight"]
                    ),
                )
                map_chunks[name].append(pooled.detach().cpu().half())
                candidate_weight_chunks[name].append(
                    diagnostics["weights"].detach().cpu()
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
    visibility = torch.cat(visibility_chunks)
    wrong_frames = torch.cat(wrong_chunks)
    quality_metrics = evaluated["methods"]["quality_pooling"]["verification"]
    clean_auc_max_drop = float(
        p0_config["selection_clean_auc_max_drop"]
    )
    rank1_max_drop = float(p0_config["selection_rank1_max_drop"])
    rows: list[dict[str, Any]] = []
    for top_k in top_k_values:
        for identity_weight in identity_weights:
            name = _candidate_name(top_k, identity_weight)
            values = evaluated["methods"][name]
            verification = values["verification"]
            weight_summary = _weight_summary(
                weights=torch.cat(candidate_weight_chunks[name]),
                visibility=visibility,
                wrong_frames=wrong_frames,
                scenarios=scenarios,
            )
            clean_drop = (
                quality_metrics["clean"]["roc_auc"]
                - verification["clean"]["roc_auc"]
            )
            rank1_drop = (
                quality_metrics["all"]["rank1_accuracy"]
                - verification["all"]["rank1_accuracy"]
            )
            rows.append(
                {
                    "candidate": name,
                    "top_k_clean": top_k,
                    "identity_weight": identity_weight,
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
                    "val_teacher_map_cosine": values[
                        "teacher_similarity"
                    ]["all"]["map_cosine"],
                    "clean_auc_drop_vs_quality": clean_drop,
                    "rank1_drop_vs_quality": rank1_drop,
                    "clean_constraint_passed": (
                        clean_drop <= clean_auc_max_drop
                    ),
                    "rank1_constraint_passed": (
                        rank1_drop <= rank1_max_drop
                    ),
                    "constraints_passed": (
                        clean_drop <= clean_auc_max_drop
                        and rank1_drop <= rank1_max_drop
                    ),
                    "complementary_visible_weight": weight_summary[
                        "complementary_occlusion"
                    ]["visible_weight_mean"],
                    "complementary_occluded_weight": weight_summary[
                        "complementary_occlusion"
                    ]["occluded_weight_mean"],
                    "wrong_identity_frame_weight": weight_summary[
                        "wrong_identity"
                    ].get("wrong_identity_frame_mean_weight"),
                }
            )
    feasible = [row for row in rows if row["constraints_passed"]]
    if feasible:
        selected = sorted(
            feasible,
            key=lambda row: (
                -float(row["val_hard_macro_roc_auc"]),
                -float(row["val_pooled_all_roc_auc"]),
                float(row["identity_weight"]),
                abs(int(row["top_k_clean"]) - 8),
            ),
        )[0]
    else:
        selected = sorted(
            rows,
            key=lambda row: (
                max(
                    0.0,
                    float(row["clean_auc_drop_vs_quality"])
                    - clean_auc_max_drop,
                ),
                max(
                    0.0,
                    float(row["rank1_drop_vs_quality"])
                    - rank1_max_drop,
                ),
                -float(row["val_hard_macro_roc_auc"]),
                float(row["identity_weight"]),
                abs(int(row["top_k_clean"]) - 8),
            ),
        )[0]
    return {
        "selection_split": "val",
        "test_metrics_consulted": False,
        "candidate_count": len(rows),
        "feasible_candidate_count": len(feasible),
        "fallback_to_unconstrained": not bool(feasible),
        "selection_rule": (
            "maximize val hard macro AUC among clean/rank1 feasible "
            "candidates; tie-break all AUC, smaller identity weight, top-k "
            "closest to 8. If none are feasible, minimize clean then Rank-1 "
            "constraint violation before hard AUC."
        ),
        "constraints": {
            "clean_auc_max_drop": clean_auc_max_drop,
            "rank1_max_drop": rank1_max_drop,
        },
        "quality_reference": {
            "clean_roc_auc": quality_metrics["clean"]["roc_auc"],
            "rank1_accuracy": quality_metrics["all"]["rank1_accuracy"],
            "hard_macro_roc_auc": quality_metrics["hard_average"]["roc_auc"],
        },
        "selected": dict(selected),
        "rows": rows,
    }


def _signal_summary(
    *,
    local_reliability: torch.Tensor,
    local_raw_score: torch.Tensor,
    local_absolute_gate: torch.Tensor,
    identity_penalty: torch.Tensor,
    frame_consensus: torch.Tensor,
    visibility: torch.Tensor,
    wrong_frames: torch.Tensor,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    scenario_array = np.asarray(list(scenarios))
    result: dict[str, Any] = {}
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        local = local_reliability[mask]
        raw = local_raw_score[mask]
        visible = visibility[mask].bool()
        position_visible = visible.any(dim=1)
        absolute_gate = local_absolute_gate[mask]
        penalty = identity_penalty[mask]
        consensus = frame_consensus[mask]
        wrong = wrong_frames[mask]
        values: dict[str, Any] = {
            "local_reliability_visible_mean": (
                float(local[visible].mean().item())
                if bool(visible.any().item())
                else None
            ),
            "local_reliability_occluded_mean": (
                float(local[~visible].mean().item())
                if bool((~visible).any().item())
                else None
            ),
            "local_raw_visible_mean": (
                float(raw[visible].mean().item())
                if bool(visible.any().item())
                else None
            ),
            "local_raw_occluded_mean": (
                float(raw[~visible].mean().item())
                if bool((~visible).any().item())
                else None
            ),
            "local_absolute_gate_mean": float(
                absolute_gate.mean().item()
            ),
            "local_absolute_gate_visible_position_mean": (
                float(absolute_gate[position_visible].mean().item())
                if bool(position_visible.any().item())
                else None
            ),
            "local_absolute_gate_fully_missing_position_mean": (
                float(absolute_gate[~position_visible].mean().item())
                if bool((~position_visible).any().item())
                else None
            ),
            "identity_penalty_mean": float(penalty.mean().item()),
            "frame_consensus_mean": float(consensus.mean().item()),
        }
        if bool(wrong.any().item()):
            values.update(
                {
                    "wrong_identity_penalty_mean": float(
                        penalty[wrong].mean().item()
                    ),
                    "correct_identity_penalty_mean": float(
                        penalty[~wrong].mean().item()
                    ),
                    "wrong_identity_consensus_mean": float(
                        consensus[wrong].mean().item()
                    ),
                    "correct_identity_consensus_mean": float(
                        consensus[~wrong].mean().item()
                    ),
                }
            )
        result[scenario] = values
    return result


@torch.inference_mode()
def evaluate_locked_split(
    *,
    split: str,
    dataset: TokenTrainingDataset,
    reference: CleanReferenceBank,
    codebook: torch.Tensor,
    selected_top_k: int,
    selected_identity_weight: float,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate fixed P0-2 parameters on one split."""

    if split not in {"val", "test"}:
        raise ValueError("locked evaluation split must be val or test")
    if getattr(dataset.synthetic, "split", None) != split:
        raise ValueError("dataset split does not match locked evaluation split")
    p0_config = config["p0_2"]
    batch_size = int(config["evaluation"]["batch_size"])
    bank_device = reference.vectors.float().to(device)
    centers_device = {
        key: value.to(device) for key, value in reference.centers.items()
    }
    scales_device = {
        key: value.to(device) for key, value in reference.scales.items()
    }
    codebook_device = F.normalize(codebook.float(), dim=-1).to(device)
    z_clip = tuple(float(value) for value in p0_config["local_z_clip"])
    map_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in FINAL_METHOD_ORDER
    }
    token_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in QUANTIZED_METHODS
    }
    method_weight_chunks: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "oracle_visibility_pooling",
            "identity_gate_only",
            "local_reliability_only",
            "visibility_identity_pooling",
        )
    }
    local_chunks: list[torch.Tensor] = []
    local_raw_chunks: list[torch.Tensor] = []
    local_absolute_gate_chunks: list[torch.Tensor] = []
    penalty_chunks: list[torch.Tensor] = []
    consensus_chunks: list[torch.Tensor] = []
    visibility_chunks: list[torch.Tensor] = []
    wrong_chunks: list[torch.Tensor] = []
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    maximum_norm_error = 0.0
    maximum_weight_sum_error = 0.0
    observed_shapes: set[tuple[int, ...]] = set()

    def append_map(name: str, maps: torch.Tensor) -> None:
        nonlocal maximum_norm_error
        expected_tail = (
            int(config["data"]["feature_grid"]["height"])
            * int(config["data"]["feature_grid"]["width"]),
            int(config["data"]["feature_grid"]["projected_dim"]),
        )
        if tuple(maps.shape[1:]) != expected_tail:
            raise RuntimeError(f"{name} returned invalid map shape")
        maximum_norm_error = max(
            maximum_norm_error,
            float((maps.norm(dim=-1) - 1.0).abs().max().item()),
        )
        map_chunks[name].append(maps.detach().cpu().half())

    def append_quantized(name: str, maps: torch.Tensor) -> None:
        tokens, _ = quantize_vectors(maps, codebook_device)
        token_chunks[name].append(tokens.detach().cpu())
        append_map(name, codebook_device[tokens].float())

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"].to(device)
        quality = batch["condition_quality"].to(device)
        visibility = batch["condition_visibility"].to(device)
        same_identity_cpu = _same_identity_frame_mask(
            dataset,
            indices,
            int(condition.shape[1]),
        )
        wrong = ~same_identity_cpu
        same_identity = same_identity_cpu.to(device)
        observed_shapes.add(tuple(condition.shape))
        baselines = continuous_aggregation_baselines(condition, quality)
        append_map("mean_pooling", baselines["mean_pooling"])
        append_map("quality_pooling", baselines["quality_pooling"])
        append_quantized(
            "quality_pooling_quantized",
            baselines["quality_pooling"],
        )
        robust, _ = robust_condition_pool(
            condition,
            quality,
            keep_fraction=float(
                config["teacher"]["robust_keep_fraction"]
            ),
            consensus_weight=float(config["teacher"]["consensus_weight"]),
            quality_weight=float(config["teacher"]["quality_weight"]),
        )
        append_map("condition_robust_pooling", robust)
        oracle, oracle_diagnostics = oracle_visibility_pool(
            condition,
            quality,
            visibility,
            same_identity,
        )
        append_map("oracle_visibility_pooling", oracle)
        penalty, identity_diagnostics = identity_outlier_penalty(
            condition,
            threshold=reference.identity_gap_threshold,
            scale=reference.identity_gap_scale,
        )
        local_scores, local_raw = local_reliability_scores(
            condition,
            reference_vectors=bank_device,
            centers=centers_device,
            scales=scales_device,
            top_k_values=[selected_top_k],
            z_clip=z_clip,
        )
        local = local_scores[selected_top_k]
        identity_only, identity_only_diagnostics = reliability_weighted_pool(
            condition,
            quality,
            local_reliability=None,
            identity_penalty=penalty,
            local_weight=0.0,
            identity_weight=selected_identity_weight,
            quality_weight=float(p0_config["quality_weight"]),
            minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
        )
        local_only, local_only_diagnostics = reliability_weighted_pool(
            condition,
            quality,
            local_reliability=local,
            identity_penalty=None,
            local_weight=float(p0_config["local_reliability_weight"]),
            identity_weight=0.0,
            quality_weight=float(p0_config["quality_weight"]),
            minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
        )
        combined, combined_diagnostics = reliability_weighted_pool(
            condition,
            quality,
            local_reliability=local,
            identity_penalty=penalty,
            local_weight=float(p0_config["local_reliability_weight"]),
            identity_weight=selected_identity_weight,
            quality_weight=float(p0_config["quality_weight"]),
            minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
        )
        append_map("identity_gate_only", identity_only)
        append_map("local_reliability_only", local_only)
        append_map("visibility_identity_pooling", combined)
        append_quantized(
            "visibility_identity_pooling_quantized",
            combined,
        )
        for name, diagnostics in (
            ("oracle_visibility_pooling", oracle_diagnostics),
            ("identity_gate_only", identity_only_diagnostics),
            ("local_reliability_only", local_only_diagnostics),
            ("visibility_identity_pooling", combined_diagnostics),
        ):
            weights = diagnostics["weights"]
            maximum_weight_sum_error = max(
                maximum_weight_sum_error,
                float((weights.sum(dim=-1) - 1.0).abs().max().item()),
            )
            method_weight_chunks[name].append(weights.detach().cpu())
        local_chunks.append(local.detach().cpu())
        local_raw_chunks.append(local_raw[selected_top_k].detach().cpu())
        local_absolute_gate_chunks.append(
            combined_diagnostics["local_absolute_gate"].detach().cpu()
        )
        penalty_chunks.append(penalty.detach().cpu())
        consensus_chunks.append(
            identity_diagnostics["frame_consensus"].detach().cpu()
        )
        visibility_chunks.append(visibility.cpu())
        wrong_chunks.append(wrong)
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(str(value) for value in batch["scenario"])

    evaluated = _evaluate_collected_maps(
        dataset=dataset,
        map_chunks=map_chunks,
        token_chunks=token_chunks,
        identities=identities,
        scenarios=scenarios,
        far_targets=[
            float(value) for value in config["evaluation"]["far_targets"]
        ],
    )
    visibility_all = torch.cat(visibility_chunks)
    wrong_all = torch.cat(wrong_chunks)
    weight_diagnostics = {
        name: _weight_summary(
            weights=torch.cat(chunks),
            visibility=visibility_all,
            wrong_frames=wrong_all,
            scenarios=scenarios,
        )
        for name, chunks in method_weight_chunks.items()
    }
    signal_diagnostics = _signal_summary(
        local_reliability=torch.cat(local_chunks),
        local_raw_score=torch.cat(local_raw_chunks),
        local_absolute_gate=torch.cat(local_absolute_gate_chunks),
        identity_penalty=torch.cat(penalty_chunks),
        frame_consensus=torch.cat(consensus_chunks),
        visibility=visibility_all,
        wrong_frames=wrong_all,
        scenarios=scenarios,
    )

    probe_count = min(batch_size, len(dataset))
    probe = default_collate([dataset[index] for index in range(probe_count)])
    probe_condition = probe["condition_features"].to(device)
    probe_quality = probe["condition_quality"].to(device)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    probe_penalty, _ = identity_outlier_penalty(
        probe_condition,
        threshold=reference.identity_gap_threshold,
        scale=reference.identity_gap_scale,
    )
    probe_local, _ = local_reliability_scores(
        probe_condition,
        reference_vectors=bank_device,
        centers=centers_device,
        scales=scales_device,
        top_k_values=[selected_top_k],
        z_clip=z_clip,
    )
    original_pool, original_diagnostics = reliability_weighted_pool(
        probe_condition,
        probe_quality,
        local_reliability=probe_local[selected_top_k],
        identity_penalty=probe_penalty,
        local_weight=float(p0_config["local_reliability_weight"]),
        identity_weight=selected_identity_weight,
        quality_weight=float(p0_config["quality_weight"]),
        minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
    )
    permuted_condition = probe_condition[:, permutation]
    permuted_quality = probe_quality[:, permutation]
    permuted_penalty, _ = identity_outlier_penalty(
        permuted_condition,
        threshold=reference.identity_gap_threshold,
        scale=reference.identity_gap_scale,
    )
    permuted_local, _ = local_reliability_scores(
        permuted_condition,
        reference_vectors=bank_device,
        centers=centers_device,
        scales=scales_device,
        top_k_values=[selected_top_k],
        z_clip=z_clip,
    )
    permuted_pool, permuted_diagnostics = reliability_weighted_pool(
        permuted_condition,
        permuted_quality,
        local_reliability=permuted_local[selected_top_k],
        identity_penalty=permuted_penalty,
        local_weight=float(p0_config["local_reliability_weight"]),
        identity_weight=selected_identity_weight,
        quality_weight=float(p0_config["quality_weight"]),
        minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
    )
    inverse = torch.argsort(permutation)
    pool_diff = float((original_pool - permuted_pool).abs().max().item())
    weight_diff = float(
        (
            original_diagnostics["weights"]
            - permuted_diagnostics["weights"][..., inverse]
        )
        .abs()
        .max()
        .item()
    )
    evaluated.update(
        {
            "split": split,
            "selected_parameters": {
                "algorithm_version": (
                    "p0_2_top2_identity_absolute_local_gate_v1"
                ),
                "top_k_clean": selected_top_k,
                "identity_weight": selected_identity_weight,
                "local_reliability_weight": float(
                    p0_config["local_reliability_weight"]
                ),
                "quality_weight": float(p0_config["quality_weight"]),
                "minimum_frame_weight": float(
                    p0_config["minimum_frame_weight"]
                ),
                "local_z_clip": [
                    float(value) for value in p0_config["local_z_clip"]
                ],
            },
            "evaluated_configuration_count": 1,
            "weight_diagnostics": weight_diagnostics,
            "signal_diagnostics": signal_diagnostics,
            "correctness": {
                "condition_shapes_observed": [
                    list(shape) for shape in sorted(observed_shapes)
                ],
                "all_output_shapes_valid": all(
                    len(chunks) > 0 for chunks in map_chunks.values()
                ),
                "output_unit_norm_max_abs_error": maximum_norm_error,
                "output_unit_norm_valid": maximum_norm_error <= 1e-5,
                "weight_sum_max_abs_error": maximum_weight_sum_error,
                "weight_sum_valid": maximum_weight_sum_error <= 1e-6,
                "minimum_combined_weight": float(
                    torch.cat(
                        method_weight_chunks[
                            "visibility_identity_pooling"
                        ]
                    ).min().item()
                ),
                "minimum_weight_floor_respected": float(
                    torch.cat(
                        method_weight_chunks[
                            "visibility_identity_pooling"
                        ]
                    ).min().item()
                )
                >= float(p0_config["minimum_frame_weight"]) - 1e-7,
                "permutation": permutation.cpu().tolist(),
                "permutation_pool_max_abs_diff": pool_diff,
                "permutation_weight_max_abs_diff": weight_diff,
                "permutation_equivalent": (
                    pool_diff <= 1e-5 and weight_diff <= 1e-5
                ),
            },
        }
    )
    return evaluated


@torch.inference_mode()
def measure_visibility_aggregation_latency(
    *,
    dataset: TokenTrainingDataset,
    reference: CleanReferenceBank,
    codebook: torch.Tensor,
    selected_top_k: int,
    selected_identity_weight: float,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Measure batch-16 cached-feature latency with the bank preloaded."""

    if len(dataset) < 16:
        raise ValueError("latency measurement requires at least 16 samples")
    batch = default_collate([dataset[index] for index in range(16)])
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    visibility = batch["condition_visibility"].to(device)
    same_identity = _same_identity_frame_mask(
        dataset,
        list(range(16)),
        condition.shape[1],
    ).to(device)
    bank_device = reference.vectors.float().to(device)
    centers_device = {
        key: value.to(device) for key, value in reference.centers.items()
    }
    scales_device = {
        key: value.to(device) for key, value in reference.scales.items()
    }
    codebook_device = F.normalize(codebook.float(), dim=-1).to(device)
    p0_config = config["p0_2"]
    z_clip = tuple(float(value) for value in p0_config["local_z_clip"])

    def mean_pool() -> torch.Tensor:
        return F.normalize(condition.float().mean(dim=1), dim=-1)

    def quality_pool() -> torch.Tensor:
        features = F.normalize(condition.float(), dim=-1)
        normalized = quality.float().clamp_min(1e-6)
        normalized = normalized / normalized.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        return F.normalize(
            (normalized[:, :, None, None] * features).sum(dim=1),
            dim=-1,
        )

    def quality_quantized() -> torch.Tensor:
        tokens, _ = quantize_vectors(quality_pool(), codebook_device)
        return codebook_device[tokens]

    def robust_pool() -> torch.Tensor:
        return robust_condition_pool(
            condition,
            quality,
            keep_fraction=float(
                config["teacher"]["robust_keep_fraction"]
            ),
            consensus_weight=float(config["teacher"]["consensus_weight"]),
            quality_weight=float(config["teacher"]["quality_weight"]),
        )[0]

    def oracle_pool() -> torch.Tensor:
        return oracle_visibility_pool(
            condition,
            quality,
            visibility,
            same_identity,
        )[0]

    def identity_only() -> torch.Tensor:
        penalty, _ = identity_outlier_penalty(
            condition,
            threshold=reference.identity_gap_threshold,
            scale=reference.identity_gap_scale,
        )
        return reliability_weighted_pool(
            condition,
            quality,
            local_reliability=None,
            identity_penalty=penalty,
            local_weight=0.0,
            identity_weight=selected_identity_weight,
            quality_weight=float(p0_config["quality_weight"]),
            minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
        )[0]

    def local_only() -> torch.Tensor:
        scores, _ = local_reliability_scores(
            condition,
            reference_vectors=bank_device,
            centers=centers_device,
            scales=scales_device,
            top_k_values=[selected_top_k],
            z_clip=z_clip,
        )
        return reliability_weighted_pool(
            condition,
            quality,
            local_reliability=scores[selected_top_k],
            identity_penalty=None,
            local_weight=float(p0_config["local_reliability_weight"]),
            identity_weight=0.0,
            quality_weight=float(p0_config["quality_weight"]),
            minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
        )[0]

    def combined() -> torch.Tensor:
        penalty, _ = identity_outlier_penalty(
            condition,
            threshold=reference.identity_gap_threshold,
            scale=reference.identity_gap_scale,
        )
        scores, _ = local_reliability_scores(
            condition,
            reference_vectors=bank_device,
            centers=centers_device,
            scales=scales_device,
            top_k_values=[selected_top_k],
            z_clip=z_clip,
        )
        return reliability_weighted_pool(
            condition,
            quality,
            local_reliability=scores[selected_top_k],
            identity_penalty=penalty,
            local_weight=float(p0_config["local_reliability_weight"]),
            identity_weight=selected_identity_weight,
            quality_weight=float(p0_config["quality_weight"]),
            minimum_frame_weight=float(p0_config["minimum_frame_weight"]),
        )[0]

    def combined_quantized() -> torch.Tensor:
        tokens, _ = quantize_vectors(combined(), codebook_device)
        return codebook_device[tokens]

    methods: dict[str, Callable[[], torch.Tensor]] = {
        "mean_pooling": mean_pool,
        "quality_pooling": quality_pool,
        "quality_pooling_quantized": quality_quantized,
        "condition_robust_pooling": robust_pool,
        "oracle_visibility_pooling": oracle_pool,
        "identity_gate_only": identity_only,
        "local_reliability_only": local_only,
        "visibility_identity_pooling": combined,
        "visibility_identity_pooling_quantized": combined_quantized,
    }
    warmup_runs = int(config["evaluation"]["latency_warmup_runs"])
    measurement_runs = int(config["evaluation"]["latency_measurement_runs"])
    result: dict[str, Any] = {}
    for name in FINAL_METHOD_ORDER:
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
        result[name] = {
            "device": str(device),
            "batch_size": 16,
            "reference_bank_preloaded": True,
            "measurement_scope": (
                "cached PCA features through aggregation; quantized methods "
                "include nearest-cosine and codebook gather"
            ),
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "batch_latency_ms": elapsed * 1000.0 / measurement_runs,
            "per_set_latency_ms": elapsed * 1000.0
            / (measurement_runs * 16),
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_metadata(name: str) -> dict[str, Any]:
    return {
        "method_role": (
            "metadata_headroom"
            if name == "oracle_visibility_pooling"
            else (
                "deployable_candidate"
                if name
                in {
                    "identity_gate_only",
                    "local_reliability_only",
                    "visibility_identity_pooling",
                    "visibility_identity_pooling_quantized",
                }
                else "baseline"
            )
        ),
        "deployable": name != "oracle_visibility_pooling",
        "metadata_oracle": name == "oracle_visibility_pooling",
        "quantized": name in QUANTIZED_METHODS,
    }


def _core_rows(
    *,
    split_results: Mapping[str, Any],
    latency: Mapping[str, Any],
    selected: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        quality = split_results[split]["methods"]["quality_pooling"][
            "verification"
        ]
        for method in FINAL_METHOD_ORDER:
            values = split_results[split]["methods"][method]
            verification = values["verification"]
            all_metrics = verification["all"]
            token_accuracy = values["exact_token_accuracy"]
            rows.append(
                {
                    "split": split,
                    "method": method,
                    **_method_metadata(method),
                    "selection_source": (
                        "validation_lock"
                        if method
                        in {
                            "identity_gate_only",
                            "local_reliability_only",
                            "visibility_identity_pooling",
                            "visibility_identity_pooling_quantized",
                        }
                        else "fixed"
                    ),
                    "locked_top_k_clean": int(selected["top_k_clean"]),
                    "locked_identity_weight": float(
                        selected["identity_weight"]
                    ),
                    "local_weight": float(
                        config["p0_2"]["local_reliability_weight"]
                    ),
                    "minimum_frame_weight": float(
                        config["p0_2"]["minimum_frame_weight"]
                    ),
                    "pooled_all_roc_auc": all_metrics["roc_auc"],
                    "hard_macro_roc_auc": verification["hard_average"][
                        "roc_auc"
                    ],
                    "clean_roc_auc": verification["clean"]["roc_auc"],
                    "delta_hard_auc_vs_quality": (
                        verification["hard_average"]["roc_auc"]
                        - quality["hard_average"]["roc_auc"]
                    ),
                    "delta_clean_auc_vs_quality": (
                        verification["clean"]["roc_auc"]
                        - quality["clean"]["roc_auc"]
                    ),
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
                    "exact_token_accuracy": (
                        ""
                        if token_accuracy is None
                        else token_accuracy["all"]
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
        quality = split_results[split]["methods"]["quality_pooling"][
            "verification"
        ]
        for method in FINAL_METHOD_ORDER:
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
                        "delta_auc_vs_quality": (
                            verification["roc_auc"]
                            - quality[scenario]["roc_auc"]
                        ),
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
                        "mean_genuine_impostor_margin": verification[
                            "mean_genuine_impostor_margin"
                        ],
                        "teacher_map_cosine": similarity["map_cosine"],
                        "teacher_flattened_map_cosine": similarity[
                            "flattened_map_cosine"
                        ],
                        "exact_token_accuracy": (
                            ""
                            if token_accuracy is None
                            else token_accuracy[scenario]
                        ),
                    }
                )
    return rows


def _weight_rows(split_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(
        *,
        split: str,
        method: str,
        scenario: str,
        scope: str,
        metric: str,
        value: Any,
        position: int | str = "",
    ) -> None:
        rows.append(
            {
                "split": split,
                "method": method,
                "scenario": scenario,
                "scope": scope,
                "position": position,
                "grid_row": (
                    "" if position == "" else int(position) // 7
                ),
                "grid_column": (
                    "" if position == "" else int(position) % 7
                ),
                "metric": metric,
                "value": "" if value is None else value,
            }
        )

    for split in ("val", "test"):
        result = split_results[split]
        for method, scenarios in result["weight_diagnostics"].items():
            for scenario, values in scenarios.items():
                if scenario == "global":
                    continue
                for metric, value in values.items():
                    if metric == "by_position":
                        continue
                    if metric == "frame_mean_weights":
                        for frame_index, frame_value in enumerate(value):
                            append(
                                split=split,
                                method=method,
                                scenario=scenario,
                                scope=f"frame_{frame_index}",
                                metric="mean_weight",
                                value=frame_value,
                            )
                        continue
                    if isinstance(value, (int, float)) or value is None:
                        append(
                            split=split,
                            method=method,
                            scenario=scenario,
                            scope="scenario",
                            metric=metric,
                            value=value,
                        )
                for position_values in values.get("by_position", []):
                    position = int(position_values["position"])
                    for metric, value in position_values.items():
                        if metric == "position":
                            continue
                        append(
                            split=split,
                            method=method,
                            scenario=scenario,
                            scope="position",
                            position=position,
                            metric=metric,
                            value=value,
                        )
        for scenario, values in result["signal_diagnostics"].items():
            for metric, value in values.items():
                append(
                    split=split,
                    method="reliability_signals",
                    scenario=scenario,
                    scope="signal",
                    metric=metric,
                    value=value,
                )
    return rows


def _build_decision(
    *,
    split_results: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for split in ("val", "test"):
        methods = split_results[split]["methods"]
        quality = methods["quality_pooling"]["verification"]
        combined = methods["visibility_identity_pooling"]["verification"]
        quality_quantized = methods["quality_pooling_quantized"][
            "verification"
        ]
        combined_quantized = methods[
            "visibility_identity_pooling_quantized"
        ]["verification"]
        oracle = methods["oracle_visibility_pooling"]["verification"]
        weights = split_results[split]["weight_diagnostics"][
            "visibility_identity_pooling"
        ]
        comparisons[split] = {
            "combined_minus_quality_pooled_auc": (
                combined["all"]["roc_auc"] - quality["all"]["roc_auc"]
            ),
            "combined_minus_quality_hard_auc": (
                combined["hard_average"]["roc_auc"]
                - quality["hard_average"]["roc_auc"]
            ),
            "combined_minus_quality_clean_auc": (
                combined["clean"]["roc_auc"] - quality["clean"]["roc_auc"]
            ),
            "combined_minus_quality_rank1": (
                combined["all"]["rank1_accuracy"]
                - quality["all"]["rank1_accuracy"]
            ),
            "combined_minus_quality_by_scenario_auc": {
                scenario: (
                    combined[scenario]["roc_auc"]
                    - quality[scenario]["roc_auc"]
                )
                for scenario in (
                    "low_quality",
                    "complementary_occlusion",
                    "common_occlusion",
                    "wrong_identity",
                )
            },
            "oracle_minus_quality_hard_auc": (
                oracle["hard_average"]["roc_auc"]
                - quality["hard_average"]["roc_auc"]
            ),
            "combined_quantized_minus_quality_quantized_hard_auc": (
                combined_quantized["hard_average"]["roc_auc"]
                - quality_quantized["hard_average"]["roc_auc"]
            ),
            "wrong_identity_frame_weight": weights["wrong_identity"].get(
                "wrong_identity_frame_mean_weight"
            ),
            "complementary_visible_weight": weights[
                "complementary_occlusion"
            ]["visible_weight_mean"],
            "complementary_occluded_weight": weights[
                "complementary_occlusion"
            ]["occluded_weight_mean"],
        }
    clean_limit = float(config["p0_2"]["selection_clean_auc_max_drop"])
    rank1_limit = float(config["p0_2"]["selection_rank1_max_drop"])
    continuous_gate = all(
        condition
        for split in ("val", "test")
        for condition in (
            comparisons[split]["combined_minus_quality_hard_auc"] > 0.0,
            comparisons[split]["combined_minus_quality_clean_auc"]
            >= -clean_limit,
            comparisons[split]["combined_minus_quality_rank1"]
            >= -rank1_limit,
            comparisons[split]["wrong_identity_frame_weight"] < 0.10,
            comparisons[split]["complementary_visible_weight"]
            > comparisons[split]["complementary_occluded_weight"],
        )
    )
    quantized_gate = all(
        comparisons[split][
            "combined_quantized_minus_quality_quantized_hard_auc"
        ]
        >= 0.0
        for split in ("val", "test")
    )
    oracle_headroom = (
        comparisons["val"]["oracle_minus_quality_hard_auc"] > 0.0
    )
    if not oracle_headroom:
        recommendation = (
            "no_go_current_late_hook_move_to_earlier_spatial_hook_or_stop"
        )
    elif not continuous_gate:
        recommendation = (
            "no_go_current_proxy_oracle_has_headroom_improve_reliability"
        )
    elif not quantized_gate:
        recommendation = (
            "go_continuous_only_consider_continuous_residual_prototype"
        )
    else:
        recommendation = "go_deterministic_and_quantized_followup"
    return {
        "selection_feasible": not bool(
            selection["fallback_to_unconstrained"]
        ),
        "oracle_headroom_on_validation": oracle_headroom,
        "continuous_gate_passed": continuous_gate,
        "quantized_gate_passed": quantized_gate,
        "recommendation": recommendation,
        "comparisons": comparisons,
    }


def _report_text(
    *,
    split_results: Mapping[str, Any],
    selection: Mapping[str, Any],
    reference_details: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    selected = selection["selected"]
    val = decision["comparisons"]["val"]
    test = decision["comparisons"]["test"]
    val_methods = split_results["val"]["methods"]
    test_methods = split_results["test"]["methods"]
    val_combined_hard = val_methods["visibility_identity_pooling"][
        "verification"
    ]["hard_average"]["roc_auc"]
    test_combined_hard = test_methods["visibility_identity_pooling"][
        "verification"
    ]["hard_average"]["roc_auc"]
    val_combined_quantized_hard = val_methods[
        "visibility_identity_pooling_quantized"
    ]["verification"]["hard_average"]["roc_auc"]
    test_combined_quantized_hard = test_methods[
        "visibility_identity_pooling_quantized"
    ]["verification"]["hard_average"]["roc_auc"]
    recommendation_cn = {
        "no_go_current_late_hook_move_to_earlier_spatial_hook_or_stop": (
            "当前 visibility metadata oracle 在 validation 上也没有显示可靠 "
            "headroom；不要继续调 late-hook 聚合，应先缓存更早的空间 hook 做同一 "
            "oracle 检查，否则停止该分支。"
        ),
        "no_go_current_proxy_oracle_has_headroom_improve_reliability": (
            "metadata oracle 有 headroom，但当前可部署 reliability proxy 未通过；"
            "下一步只能改进可靠性估计，不能据此重启 diffusion。"
        ),
        "go_continuous_only_consider_continuous_residual_prototype": (
            "连续确定性聚合通过、量化未通过；保留 continuous 路线，若需要迭代模型，"
            "优先 continuous residual prototype。"
        ),
        "go_deterministic_and_quantized_followup": (
            "连续与量化门槛均通过，可继续确定性聚合，并重新评估离散后续。"
        ),
    }[decision["recommendation"]]
    rows = []
    for split in ("val", "test"):
        for method in FINAL_METHOD_ORDER:
            values = split_results[split]["methods"][method]
            verification = values["verification"]
            token_accuracy = values["exact_token_accuracy"]
            rows.append(
                "| "
                + " | ".join(
                    [
                        split,
                        method,
                        f"{verification['all']['roc_auc']:.6f}",
                        f"{verification['hard_average']['roc_auc']:.6f}",
                        f"{verification['all']['eer']:.6f}",
                        f"{verification['all']['rank1_accuracy']:.6f}",
                        f"{verification['all']['mean_genuine_impostor_margin']:.6f}",
                        (
                            "—"
                            if token_accuracy is None
                            else f"{token_accuracy['all']:.6f}"
                        ),
                    ]
                )
                + " |"
            )
    return f"""# P0-2 visibility/reliability-aware deterministic aggregation

## 协议

- Reference bank 仅由 train clean disjoint teachers 构建：
  {reference_details["unique_teacher_references"]} 张、覆盖
  {reference_details["unique_teacher_identities"]} 个训练身份，布局
  `{reference_details["shape"]}`。
- 同目标身份 S/T 图像重叠为
  {reference_details["target_condition_teacher_overlap"]}；另有
  {reference_details["actual_source_teacher_overlap"]} 张 reference 曾在 train
  wrong-ID 场景作为异身份 source 出现，已单独记录。它不进入 clean 校准，
  且 val/test 身份与 train 隔离，因此不是测试泄漏。
- PCA、K=1024 codebook、S/T、identity split 和 clean continuous gallery 均沿用
  既有产物，不重新拟合。
- Local reliability 采用 train-clean cross-identity 第 5 百分位作为绝对零点；
  当某位置四帧均低于该零点时关闭局部排序并回退到 quality/global gate。
- Global identity support 使用每帧最相似的两个邻帧，避免单个 wrong-ID 帧
  同时稀释三个正确帧的共识。
- 只在 validation 搜索 9 个预注册候选；锁定 `top_k={int(selected["top_k_clean"])}`
  与 `identity_weight={float(selected["identity_weight"]):g}` 后才运行一次 test。
- `oracle_visibility_pooling` 使用扰动 visibility 与 source identity metadata，
  只是不可部署的 headroom diagnostic，不是理论上界，也不参与选参。

## 结果

| Split | Method | All AUC | Hard AUC | EER | Rank-1 | Margin | Token acc |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 对 quality pooling 的关键差值

- Combined continuous：val all/hard AUC
  {val["combined_minus_quality_pooled_auc"]:+.6f} /
  {val["combined_minus_quality_hard_auc"]:+.6f}；test
  {test["combined_minus_quality_pooled_auc"]:+.6f} /
  {test["combined_minus_quality_hard_auc"]:+.6f}。
- Metadata oracle hard AUC headroom：val
  {val["oracle_minus_quality_hard_auc"]:+.6f}，test
  {test["oracle_minus_quality_hard_auc"]:+.6f}。
- Combined quantized 相对 quantized quality hard AUC：val
  {val["combined_quantized_minus_quality_quantized_hard_auc"]:+.6f}，test
  {test["combined_quantized_minus_quality_quantized_hard_auc"]:+.6f}。

## 机制检查

- Combined test wrong-ID 帧平均权重：
  {test["wrong_identity_frame_weight"]:.6f}。
- Combined test complementary 可见/遮挡帧平均权重：
  {test["complementary_visible_weight"]:.6f} /
  {test["complementary_occluded_weight"]:.6f}。
- 详细 identity penalty、raw/standardized local reliability、逐场景和 49 位置权重
  统计见 `weight_diagnostics.csv` 与 `diagnostics.json`。

## 解释

- 机制层面成立：wrong-ID 帧被压到 0.10 以下，且 complementary 中可见位置
  权重高于遮挡位置；但“权重方向正确”没有稳定转化为验证性能。
- Combined 的 teacher-map cosine 相对 quality pooling：val
  {val_methods["visibility_identity_pooling"]["teacher_similarity"]["all"]["map_cosine"] - val_methods["quality_pooling"]["teacher_similarity"]["all"]["map_cosine"]:+.6f}，
  test
  {test_methods["visibility_identity_pooling"]["teacher_similarity"]["all"]["map_cosine"] - test_methods["quality_pooling"]["teacher_similarity"]["all"]["map_cosine"]:+.6f}。
  目标图更接近 teacher，却没有在 validation 提升 hard AUC，说明当前 teacher-map
  一致性与身份判别指标并不完全对齐。
- Combined 量化造成 hard AUC 损失：val
  {val_combined_quantized_hard - val_combined_hard:+.6f}，test
  {test_combined_quantized_hard - test_combined_hard:+.6f}；当前 K=1024
  离散表示仍是明显瓶颈。
- Test 上 local-only/combined 的正增益与 validation 上的负增益方向相反；
  这是 split instability，不能把单次 test 提升解释为泛化收益。

## Go/No-Go

- validation 选择约束可行：{decision["selection_feasible"]}。
- metadata oracle headroom：{decision["oracle_headroom_on_validation"]}。
- continuous gate：{decision["continuous_gate_passed"]}。
- quantized gate：{decision["quantized_gate_passed"]}。
- 建议：{recommendation_cn}

本实验 validation/test 仅有 25/50 个身份，不声称统计显著。Reference typicality
也可能编码 CelebA 风格、姿态和人口属性，而不只是可见性。
"""


def write_visibility_aggregation_artifacts(
    *,
    output_directory: Path,
    split_results: Mapping[str, Any],
    selection: Mapping[str, Any],
    reference_details: Mapping[str, Any],
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: Mapping[str, bool],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the complete locked P0-2 result package."""

    output_directory.mkdir(parents=True, exist_ok=True)
    core_path = output_directory / "core_comparison.csv"
    scenario_path = output_directory / "scenario_metrics.csv"
    search_path = output_directory / "validation_search.csv"
    weights_path = output_directory / "weight_diagnostics.csv"
    diagnostics_path = output_directory / "diagnostics.json"
    report_path = output_directory / "REPORT.md"
    selected = selection["selected"]
    _write_csv(
        core_path,
        _core_rows(
            split_results=split_results,
            latency=latency,
            selected=selected,
            config=config,
        ),
    )
    _write_csv(scenario_path, _scenario_rows(split_results))
    search_rows = []
    for row in selection["rows"]:
        search_rows.append(
            {
                **row,
                "selected": row["candidate"] == selected["candidate"],
            }
        )
    _write_csv(search_path, search_rows)
    _write_csv(weights_path, _weight_rows(split_results))
    decision = _build_decision(
        split_results=split_results,
        selection=selection,
        config=config,
    )
    diagnostics = {
        "schema_version": 1,
        "checks": dict(checks),
        "reference": dict(reference_details),
        "selection": dict(selection),
        "latency": dict(latency),
        "correctness": {
            split: split_results[split]["correctness"]
            for split in ("val", "test")
        },
        "weight_diagnostics": {
            split: split_results[split]["weight_diagnostics"]
            for split in ("val", "test")
        },
        "signal_diagnostics": {
            split: split_results[split]["signal_diagnostics"]
            for split in ("val", "test")
        },
        "decision": decision,
        "reproducibility": dict(reproducibility),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _report_text(
            split_results=split_results,
            selection=selection,
            reference_details=reference_details,
            decision=decision,
        ),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "core_comparison_csv": str(core_path),
        "scenario_metrics_csv": str(scenario_path),
        "validation_search_csv": str(search_path),
        "weight_diagnostics_csv": str(weights_path),
        "diagnostics_json": str(diagnostics_path),
        "decision": decision,
    }
