"""PointerRoute development split and teacher-scored oracle utilities.

The route space is sample-relative: route 0 selects the quality anchor and
routes 1..F select the corresponding condition frame at each spatial position.
No codebook is fit or used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .research_evaluation import _evaluate_embeddings


HARD_SCENARIOS = (
    "low_quality",
    "complementary_occlusion",
    "common_occlusion",
    "wrong_identity",
)


@dataclass(frozen=True)
class RouteDevSplit:
    """Identity-disjoint indices carved only from the original train split."""

    identity_ids: dict[str, list[int]]
    sample_indices: dict[str, list[int]]
    checks: dict[str, Any]


def build_route_dev_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train_identities: int,
    calibration_identities: int,
    validation_identities: int,
    required_scenarios: Sequence[str],
) -> RouteDevSplit:
    """Create a deterministic 800/100/100-style identity partition."""

    groups: dict[int, list[int]] = {}
    scenario_sets: dict[int, set[str]] = {}
    for index, row in enumerate(rows):
        identity = int(row["target_identity"])
        groups.setdefault(identity, []).append(index)
        scenario_sets.setdefault(identity, set()).add(str(row["scenario"]))

    counts = {
        "route_train": int(train_identities),
        "route_calibration": int(calibration_identities),
        "route_validation": int(validation_identities),
    }
    requested = sum(counts.values())
    identities = np.asarray(sorted(groups), dtype=np.int64)
    if requested != int(identities.size):
        raise ValueError(
            f"route split requests {requested} identities but source has "
            f"{identities.size}"
        )
    required = set(str(value) for value in required_scenarios)
    incomplete = {
        identity: sorted(required.difference(scenario_sets[identity]))
        for identity in groups
        if not required.issubset(scenario_sets[identity])
    }
    if incomplete:
        raise ValueError(
            "source identities do not cover every required scenario: "
            f"{list(incomplete.items())[:3]}"
        )

    generator = np.random.default_rng(int(seed))
    shuffled = generator.permutation(identities).tolist()
    identity_ids: dict[str, list[int]] = {}
    sample_indices: dict[str, list[int]] = {}
    cursor = 0
    for name, count in counts.items():
        selected = sorted(int(value) for value in shuffled[cursor : cursor + count])
        cursor += count
        identity_ids[name] = selected
        selected_set = set(selected)
        sample_indices[name] = [
            index
            for index, row in enumerate(rows)
            if int(row["target_identity"]) in selected_set
        ]

    identity_sets = {name: set(values) for name, values in identity_ids.items()}
    pairwise_disjoint = all(
        identity_sets[left].isdisjoint(identity_sets[right])
        for left_index, left in enumerate(identity_sets)
        for right in list(identity_sets)[left_index + 1 :]
    )
    union = set().union(*identity_sets.values())
    checks = {
        "source_identities": int(identities.size),
        "requested_identities": requested,
        "identity_counts": {
            name: len(values) for name, values in identity_ids.items()
        },
        "sample_counts": {
            name: len(values) for name, values in sample_indices.items()
        },
        "pairwise_identity_disjoint": pairwise_disjoint,
        "identity_union_exact": union == set(int(value) for value in identities),
        "all_identities_cover_required_scenarios": not incomplete,
        "passed": (
            pairwise_disjoint
            and union == set(int(value) for value in identities)
            and not incomplete
        ),
    }
    return RouteDevSplit(
        identity_ids=identity_ids,
        sample_indices=sample_indices,
        checks=checks,
    )


def build_route_candidates(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
) -> torch.Tensor:
    """Return quality anchor plus frame-local candidates as [B,P,1+F,D]."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,F,P,D]")
    batch, frames, _, _ = condition_features.shape
    if tuple(condition_quality.shape) != (batch, frames):
        raise ValueError("condition_quality must have shape [B,F]")
    features = F.normalize(condition_features.float(), dim=-1, eps=1e-6)
    quality = condition_quality.float().clamp_min(1e-6)
    quality = quality / quality.sum(dim=1, keepdim=True).clamp_min(1e-6)
    anchor = F.normalize(
        (quality[:, :, None, None] * features).sum(dim=1),
        dim=-1,
        eps=1e-6,
    )
    return torch.cat(
        [anchor.unsqueeze(2), features.permute(0, 2, 1, 3)],
        dim=2,
    )


def teacher_route_scores(
    candidates: torch.Tensor,
    teacher_maps: torch.Tensor,
) -> torch.Tensor:
    """Cosine score every candidate against the disjoint teacher position."""

    if candidates.ndim != 4:
        raise ValueError("candidates must have shape [B,P,C,D]")
    if tuple(teacher_maps.shape) != (
        candidates.shape[0],
        candidates.shape[1],
        candidates.shape[3],
    ):
        raise ValueError("teacher_maps must have shape [B,P,D]")
    return torch.sum(
        F.normalize(candidates.float(), dim=-1)
        * F.normalize(teacher_maps.float(), dim=-1).unsqueeze(2),
        dim=-1,
    )


def margin_adjusted_utilities(
    scores: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    if scores.ndim != 3 or scores.shape[-1] < 2:
        raise ValueError("scores must have shape [B,P,1+F]")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    return torch.cat(
        [scores[..., :1], scores[..., 1:] - float(delta)],
        dim=-1,
    )


def hard_route_targets(
    scores: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Select a frame only when its teacher gain clears the anchor margin."""

    return margin_adjusted_utilities(scores, delta=delta).argmax(dim=-1)


def soft_route_targets(
    scores: torch.Tensor,
    *,
    delta: float,
    temperature: float,
) -> torch.Tensor:
    """Return anchor-relative soft route labels."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    anchor_relative = torch.cat(
        [
            torch.zeros_like(scores[..., :1]),
            scores[..., 1:] - scores[..., :1] - float(delta),
        ],
        dim=-1,
    )
    return torch.softmax(anchor_relative / float(temperature), dim=-1)


def reconstruct_hard_routes(
    candidates: torch.Tensor,
    routes: torch.Tensor,
) -> torch.Tensor:
    if tuple(routes.shape) != tuple(candidates.shape[:2]):
        raise ValueError("routes must have shape [B,P]")
    if bool((routes < 0).any()) or bool((routes >= candidates.shape[2]).any()):
        raise ValueError("routes contain an invalid candidate index")
    selected = torch.gather(
        candidates,
        2,
        routes[..., None, None].expand(-1, -1, 1, candidates.shape[-1]),
    ).squeeze(2)
    return F.normalize(selected, dim=-1, eps=1e-6)


def reconstruct_soft_routes(
    candidates: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    if tuple(probabilities.shape) != tuple(candidates.shape[:3]):
        raise ValueError("probabilities must have shape [B,P,C]")
    probabilities_float = probabilities.float()
    if not bool(torch.isfinite(probabilities_float).all().item()):
        raise ValueError("route probabilities must be finite")
    if float(probabilities_float.min().item()) < 0.0:
        raise ValueError("route probabilities must be non-negative")
    probability_sum = probabilities_float.sum(dim=-1)
    if float((probability_sum - 1.0).abs().max().item()) > 2e-3:
        raise ValueError("route probabilities must sum to one")
    normalized_probabilities = (
        probabilities_float
        / probability_sum.unsqueeze(-1).clamp_min(1e-8)
    )
    return F.normalize(
        (
            normalized_probabilities.unsqueeze(-1)
            * candidates.float()
        ).sum(dim=2),
        dim=-1,
        eps=1e-6,
    )


def _scenario_masks(scenarios: Sequence[str]) -> dict[str, torch.Tensor]:
    values = np.asarray(list(scenarios))
    masks = {
        scenario: torch.from_numpy(values == scenario)
        for scenario in sorted(set(values.tolist()))
    }
    masks["all"] = torch.ones(len(values), dtype=torch.bool)
    return masks


def _summarize(values: torch.Tensor) -> dict[str, float]:
    flattened = values.float().reshape(-1)
    return {
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "p05": float(torch.quantile(flattened, 0.05).item()),
        "p50": float(torch.quantile(flattened, 0.50).item()),
        "p95": float(torch.quantile(flattened, 0.95).item()),
    }


def map_similarity_metrics(
    maps: torch.Tensor,
    teacher_maps: torch.Tensor,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    local = F.cosine_similarity(
        F.normalize(maps.float(), dim=-1),
        F.normalize(teacher_maps.float(), dim=-1),
        dim=-1,
    )
    pooled = F.cosine_similarity(
        F.normalize(maps.float().mean(dim=1), dim=-1),
        F.normalize(teacher_maps.float().mean(dim=1), dim=-1),
        dim=-1,
    )
    result: dict[str, Any] = {}
    for scenario, mask in _scenario_masks(scenarios).items():
        result[scenario] = {
            "mean_local_cosine": float(local[mask].mean().item()),
            "mean_map_cosine": float(pooled[mask].mean().item()),
        }
    return result


def route_diagnostics(
    *,
    routes: torch.Tensor,
    probabilities: torch.Tensor,
    scores: torch.Tensor,
    delta: float,
    source_identities: torch.Tensor,
    target_identities: torch.Tensor,
    visibility: torch.Tensor,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    """Summarize route usage, ambiguity, visibility, and wrong-ID choices."""

    batch, positions = routes.shape
    frames = scores.shape[-1] - 1
    if tuple(source_identities.shape) != (batch, frames):
        raise ValueError("source_identities must have shape [B,F]")
    if tuple(visibility.shape) != (batch, frames, positions):
        raise ValueError("visibility must have shape [B,F,P]")
    if tuple(target_identities.shape) != (batch,):
        raise ValueError("target_identities must have shape [B]")

    adjusted = margin_adjusted_utilities(scores, delta=delta)
    sorted_utilities = torch.sort(adjusted, dim=-1, descending=True).values
    top_gap = sorted_utilities[..., 0] - sorted_utilities[..., 1]
    entropy = -(
        probabilities.clamp_min(1e-12)
        * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    selected_scores = torch.gather(scores, 2, routes.unsqueeze(-1)).squeeze(-1)
    raw_regret = scores.max(dim=-1).values - selected_scores
    anchor_gain = selected_scores - scores[..., 0]

    frame_route = routes.clamp_min(1) - 1
    selected_is_frame = routes > 0
    wrong_frames = source_identities != target_identities[:, None]
    expanded_wrong = wrong_frames[:, None, :].expand(-1, positions, -1)
    selected_wrong = torch.gather(
        expanded_wrong,
        2,
        frame_route.unsqueeze(-1),
    ).squeeze(-1) & selected_is_frame
    expanded_visibility = visibility.permute(0, 2, 1).bool()
    selected_visible = torch.gather(
        expanded_visibility,
        2,
        frame_route.unsqueeze(-1),
    ).squeeze(-1) & selected_is_frame

    result: dict[str, Any] = {}
    for scenario, sample_mask in _scenario_masks(scenarios).items():
        selected = routes[sample_mask]
        selected_frame_mask = selected_is_frame[sample_mask]
        selected_wrong_mask = selected_wrong[sample_mask]
        selected_visible_mask = selected_visible[sample_mask]
        frame_count = int(selected_frame_mask.sum().item())
        frequencies = torch.bincount(
            selected.reshape(-1),
            minlength=frames + 1,
        ).float()
        frequencies = frequencies / frequencies.sum().clamp_min(1.0)
        result[scenario] = {
            "anchor_fraction": float((selected == 0).float().mean().item()),
            "non_anchor_fraction": float(
                selected_frame_mask.float().mean().item()
            ),
            "route_frequencies": [
                float(value) for value in frequencies.tolist()
            ],
            "wrong_frame_selection_fraction_all_positions": float(
                selected_wrong_mask.float().mean().item()
            ),
            "wrong_frame_selection_fraction_of_frame_routes": (
                float(selected_wrong_mask.sum().item() / frame_count)
                if frame_count
                else 0.0
            ),
            "selected_frame_visible_fraction": (
                float(selected_visible_mask.sum().item() / frame_count)
                if frame_count
                else None
            ),
            "top1_top2_utility_gap": _summarize(top_gap[sample_mask]),
            "soft_target_entropy": _summarize(entropy[sample_mask]),
            "selected_gain_over_anchor": _summarize(anchor_gain[sample_mask]),
            "raw_oracle_regret": _summarize(raw_regret[sample_mask]),
        }
    return result


def teacher_loo_stability(
    *,
    candidates: torch.Tensor,
    full_routes: torch.Tensor,
    full_teacher_maps: torch.Tensor,
    loo_teacher_maps: torch.Tensor,
    scenarios: Sequence[str],
    delta: float,
    equivalent_epsilon: float,
) -> dict[str, Any]:
    """Compare full-teacher routes to four leave-one-teacher-out targets."""

    if loo_teacher_maps.ndim != 4:
        raise ValueError("loo_teacher_maps must have shape [B,L,P,D]")
    exact_parts: list[torch.Tensor] = []
    equivalent_parts: list[torch.Tensor] = []
    teacher_cosine_parts: list[torch.Tensor] = []
    for leave_index in range(loo_teacher_maps.shape[1]):
        teacher = loo_teacher_maps[:, leave_index]
        scores = teacher_route_scores(candidates, teacher)
        loo_routes = hard_route_targets(scores, delta=delta)
        exact_parts.append(loo_routes == full_routes)
        utilities = margin_adjusted_utilities(scores, delta=delta)
        selected_utility = torch.gather(
            utilities,
            2,
            full_routes.unsqueeze(-1),
        ).squeeze(-1)
        equivalent_parts.append(
            utilities.max(dim=-1).values - selected_utility
            <= float(equivalent_epsilon)
        )
        teacher_cosine_parts.append(
            F.cosine_similarity(
                F.normalize(full_teacher_maps.float(), dim=-1),
                F.normalize(teacher.float(), dim=-1),
                dim=-1,
            )
        )
    exact = torch.stack(exact_parts, dim=1)
    equivalent = torch.stack(equivalent_parts, dim=1)
    teacher_cosine = torch.stack(teacher_cosine_parts, dim=1)
    result: dict[str, Any] = {}
    for scenario, mask in _scenario_masks(scenarios).items():
        result[scenario] = {
            "exact_route_agreement": float(exact[mask].float().mean().item()),
            "equivalent_route_agreement": float(
                equivalent[mask].float().mean().item()
            ),
            "mean_teacher_local_cosine": float(
                teacher_cosine[mask].mean().item()
            ),
        }
    return result


def permutation_check(
    *,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    teacher_maps: torch.Tensor,
    delta: float,
    temperature: float,
    permutation: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Check route-label equivariance and reconstructed-map invariance."""

    frames = condition_features.shape[1]
    if permutation is None:
        permutation = torch.tensor(
            list(reversed(range(frames))),
            dtype=torch.int64,
            device=condition_features.device,
        )
    candidates = build_route_candidates(condition_features, condition_quality)
    scores = teacher_route_scores(candidates, teacher_maps)
    routes = hard_route_targets(scores, delta=delta)
    probabilities = soft_route_targets(
        scores,
        delta=delta,
        temperature=temperature,
    )
    hard_map = reconstruct_hard_routes(candidates, routes)
    soft_map = reconstruct_soft_routes(candidates, probabilities)

    permuted_candidates = build_route_candidates(
        condition_features[:, permutation],
        condition_quality[:, permutation],
    )
    permuted_scores = teacher_route_scores(permuted_candidates, teacher_maps)
    permuted_routes = hard_route_targets(permuted_scores, delta=delta)
    permuted_probabilities = soft_route_targets(
        permuted_scores,
        delta=delta,
        temperature=temperature,
    )
    permuted_hard_map = reconstruct_hard_routes(
        permuted_candidates,
        permuted_routes,
    )
    permuted_soft_map = reconstruct_soft_routes(
        permuted_candidates,
        permuted_probabilities,
    )

    original_route = permuted_routes.clone()
    frame_mask = original_route > 0
    original_route[frame_mask] = permutation[
        original_route[frame_mask] - 1
    ] + 1
    reordered_probabilities = torch.empty_like(permuted_probabilities)
    reordered_probabilities[..., 0] = permuted_probabilities[..., 0]
    for permuted_index, original_index in enumerate(permutation.tolist()):
        reordered_probabilities[..., original_index + 1] = (
            permuted_probabilities[..., permuted_index + 1]
        )

    return {
        "permutation": [int(value) for value in permutation.tolist()],
        "route_disagreement_fraction": float(
            (routes != original_route).float().mean().item()
        ),
        "probability_max_abs_difference": float(
            (probabilities - reordered_probabilities).abs().max().item()
        ),
        "hard_map_max_abs_difference": float(
            (hard_map - permuted_hard_map).abs().max().item()
        ),
        "soft_map_max_abs_difference": float(
            (soft_map - permuted_soft_map).abs().max().item()
        ),
    }


def evaluate_route_oracle(
    *,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    teacher_maps: torch.Tensor,
    loo_teacher_maps: torch.Tensor,
    identities: torch.Tensor,
    source_identities: torch.Tensor,
    visibility: torch.Tensor,
    scenarios: Sequence[str],
    delta: float,
    temperature: float,
    equivalent_epsilon: float,
    far_targets: Sequence[float],
) -> dict[str, Any]:
    """Evaluate quality, hard-route, and soft-route maps on one dev split."""

    candidates = build_route_candidates(condition_features, condition_quality)
    scores = teacher_route_scores(candidates, teacher_maps)
    routes = hard_route_targets(scores, delta=delta)
    probabilities = soft_route_targets(
        scores,
        delta=delta,
        temperature=temperature,
    )
    maps = {
        "quality_anchor": candidates[:, :, 0],
        "hard_route_oracle": reconstruct_hard_routes(candidates, routes),
        "soft_route_oracle": reconstruct_soft_routes(candidates, probabilities),
    }
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identities[clean_indices]
    gallery_embeddings = F.normalize(
        teacher_maps[clean_indices].float().mean(dim=1),
        dim=-1,
    )
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    if int(torch.unique(gallery_identities).numel()) != len(clean_indices):
        raise RuntimeError("route-dev gallery must have one clean item per identity")

    methods: dict[str, Any] = {}
    for name, method_map in maps.items():
        methods[name] = {
            "verification": _evaluate_embeddings(
                embeddings=F.normalize(method_map.float().mean(dim=1), dim=-1),
                gallery_embeddings=gallery_embeddings,
                identities=identities,
                gallery_identities=gallery_identities,
                scenarios=list(scenarios),
                far_targets=[float(value) for value in far_targets],
            ),
            "teacher_similarity": map_similarity_metrics(
                method_map,
                teacher_maps,
                scenarios,
            ),
        }

    return {
        "samples": int(condition_features.shape[0]),
        "identities": len(clean_indices),
        "gallery": "clean_disjoint_teacher_continuous",
        "delta": float(delta),
        "temperature": float(temperature),
        "methods": methods,
        "route_diagnostics": route_diagnostics(
            routes=routes,
            probabilities=probabilities,
            scores=scores,
            delta=delta,
            source_identities=source_identities,
            target_identities=identities,
            visibility=visibility,
            scenarios=scenarios,
        ),
        "teacher_loo_stability": teacher_loo_stability(
            candidates=candidates,
            full_routes=routes,
            full_teacher_maps=teacher_maps,
            loo_teacher_maps=loo_teacher_maps,
            scenarios=scenarios,
            delta=delta,
            equivalent_epsilon=equivalent_epsilon,
        ),
        "permutation_check": permutation_check(
            condition_features=condition_features[: min(32, len(scenarios))],
            condition_quality=condition_quality[: min(32, len(scenarios))],
            teacher_maps=teacher_maps[: min(32, len(scenarios))],
            delta=delta,
            temperature=temperature,
            permutation=torch.tensor(
                [2, 0, 3, 1],
                dtype=torch.int64,
                device=condition_features.device,
            ),
        ),
        "targets": {
            "hard_routes": routes.cpu(),
            "soft_probabilities": probabilities.cpu(),
            "teacher_scores": scores.cpu(),
        },
    }


def select_calibration_pair(
    rows: Sequence[Mapping[str, Any]],
    *,
    auc_tie_tolerance: float,
) -> dict[str, float]:
    """Select a conservative delta and temperature using calibration only."""

    if not rows:
        raise ValueError("calibration rows cannot be empty")
    hard_by_delta: dict[float, float] = {}
    for row in rows:
        delta = float(row["delta"])
        hard_by_delta[delta] = float(row["hard_route_hard_auc"])
    best_hard = max(hard_by_delta.values())
    eligible_delta = [
        delta
        for delta, auc in hard_by_delta.items()
        if best_hard - auc <= float(auc_tie_tolerance)
    ]
    selected_delta = max(eligible_delta)

    selected_rows = [
        row for row in rows if float(row["delta"]) == selected_delta
    ]
    best_soft = max(float(row["soft_route_hard_auc"]) for row in selected_rows)
    eligible_temperature = [
        float(row["temperature"])
        for row in selected_rows
        if best_soft - float(row["soft_route_hard_auc"])
        <= float(auc_tie_tolerance)
    ]
    return {
        "delta": float(selected_delta),
        "temperature": float(max(eligible_temperature)),
        "best_hard_route_hard_auc": float(best_hard),
        "best_soft_route_hard_auc_at_selected_delta": float(best_soft),
    }
