"""Plain confidence-only masked diffusion over sample-relative route tokens."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .decoding import (
    combine_two_level_candidate_support,
    compute_frame_reliability,
)
from .pointer_route_model import PointerRouteTransformer
from .pointer_route_oracle import (
    build_route_candidates,
    map_similarity_metrics,
    margin_adjusted_utilities,
    reconstruct_hard_routes,
    route_diagnostics,
)
from .pointer_route_training import (
    _require_precomputed,
    _route_accuracy,
    _subset_tensors,
    pointer_route_loss,
)
from .research_evaluation import _evaluate_embeddings
from .training import TokenTrainingDataset


@dataclass(frozen=True)
class PointerDecodeResult:
    final_routes: torch.Tensor
    final_probabilities: torch.Tensor
    provisional_routes: tuple[torch.Tensor, ...]
    probabilities: tuple[torch.Tensor, ...]
    newly_committed: tuple[torch.Tensor, ...]
    remasked: tuple[torch.Tensor, ...]
    committed_after_step: tuple[torch.Tensor, ...]
    commit_steps: torch.Tensor
    remask_count: torch.Tensor
    commit_confidence: torch.Tensor
    commit_score: torch.Tensor
    commit_evidence: torch.Tensor
    frame_reliability: torch.Tensor | None
    mean_local_usage: torch.Tensor | None
    effective_frames: torch.Tensor | None


@dataclass(frozen=True)
class PointerRouteEvidenceResult:
    evidence: torch.Tensor
    frame_reliability: torch.Tensor
    identity_consensus: torch.Tensor
    identity_reliability: torch.Tensor
    mean_local_usage: torch.Tensor
    effective_frames: torch.Tensor


@torch.no_grad()
def compute_pointer_route_evidence(
    *,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    candidates: torch.Tensor,
    identity_neighbor_fraction: float = 0.5,
    identity_temperature: float = 3.0,
    quality_weight: float = 0.25,
    local_temperature: float = 4.0,
    reliability_floor: float = 1e-4,
) -> PointerRouteEvidenceResult:
    """Two-level stop-gradient evidence for anchor-plus-frame routes."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,F,P,D]")
    batch, frames, positions, dimension = condition_features.shape
    expected = (batch, positions, frames + 1, dimension)
    if tuple(candidates.shape) != expected:
        raise ValueError(f"candidates must have shape {expected}")
    reliability = compute_frame_reliability(
        condition_features,
        condition_quality,
        identity_neighbor_fraction=identity_neighbor_fraction,
        identity_temperature=identity_temperature,
        quality_weight=quality_weight,
        reliability_floor=reliability_floor,
    )
    condition = F.normalize(condition_features.float(), dim=-1)
    normalized_candidates = F.normalize(candidates.float(), dim=-1)
    per_frame_support = torch.einsum(
        "bfpd,bpkd->bfpk",
        condition,
        normalized_candidates,
    )
    evidence, mean_local_usage = combine_two_level_candidate_support(
        per_frame_support,
        reliability.frame_reliability,
        local_temperature=local_temperature,
        reliability_floor=reliability_floor,
    )
    return PointerRouteEvidenceResult(
        evidence=evidence.detach(),
        frame_reliability=reliability.frame_reliability.detach(),
        identity_consensus=reliability.identity_consensus.detach(),
        identity_reliability=reliability.identity_reliability.detach(),
        mean_local_usage=mean_local_usage.detach(),
        effective_frames=reliability.effective_frames.detach(),
    )


@torch.no_grad()
def build_pointer_rollout_corruption(
    *,
    model: PointerRouteTransformer,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    target_routes: torch.Tensor,
    rollout_probability: float,
    rollout_total_steps: int,
    rollout_depths: Sequence[int],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Create partial states from the model's own irreversible commits.

    Samples not selected for rollout remain fully masked. Rollout choices and
    committed routes are detached by construction; gradients are computed
    only by the subsequent training forward pass on still-masked positions.
    """

    if target_routes.ndim != 2:
        raise ValueError("target_routes must have shape [B,P]")
    if not 0.0 <= rollout_probability <= 1.0:
        raise ValueError("rollout_probability must be in [0,1]")
    if rollout_total_steps < 2:
        raise ValueError("rollout_total_steps must be at least two")
    depths = tuple(int(value) for value in rollout_depths)
    if (
        not depths
        or len(set(depths)) != len(depths)
        or any(not 1 <= value < rollout_total_steps for value in depths)
    ):
        raise ValueError(
            "rollout_depths must be unique values in "
            "[1, rollout_total_steps)"
        )
    batch, positions = target_routes.shape
    if positions != model.spatial_tokens:
        raise ValueError("target route positions do not match the model")
    device = condition_features.device
    if condition_quality.device != device or target_routes.device != device:
        raise ValueError("rollout tensors must share one device")
    model_device = next(model.parameters()).device
    if model_device != device:
        raise ValueError("model and rollout tensors must share one device")

    use_rollout_cpu = (
        torch.rand(batch, generator=generator) < rollout_probability
    )
    depth_choice_cpu = torch.randint(
        0,
        len(depths),
        (batch,),
        generator=generator,
    )
    selected_depth_cpu = torch.tensor(depths, dtype=torch.int64)[
        depth_choice_cpu
    ]
    use_rollout = use_rollout_cpu.to(device)
    rollout_indices = torch.nonzero(
        use_rollout,
        as_tuple=False,
    ).flatten()
    rollout_count = int(rollout_indices.numel())
    corrupted = model.all_mask_input(batch, device=device)
    if rollout_count == 0:
        loss_mask = corrupted == model.mask_token_id
        return corrupted, loss_mask, {
            "rollout_fraction": 0.0,
            "all_mask_fraction": 1.0,
            "mean_rollout_depth": 0.0,
            "committed_fraction": 0.0,
            "committed_error_fraction": 0.0,
        }

    rollout_depth = selected_depth_cpu[use_rollout_cpu].to(device)
    rollout_condition = condition_features.index_select(0, rollout_indices)
    rollout_quality = condition_quality.index_select(0, rollout_indices)
    rollout_targets = target_routes.index_select(0, rollout_indices)
    current = model.all_mask_input(rollout_count, device=device)
    committed = torch.zeros(
        (rollout_count, positions),
        dtype=torch.bool,
        device=device,
    )
    states: dict[int, torch.Tensor] = {}
    maximum_depth = int(rollout_depth.max().item())
    for step in range(1, maximum_depth + 1):
        logits, _ = model(
            current,
            rollout_condition,
            rollout_quality,
        )
        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        cumulative = cosine_commit_count(
            positions=positions,
            step=step,
            total_steps=rollout_total_steps,
        )
        already = int(committed[0].sum().item())
        newly_required = max(0, cumulative - already)
        if newly_required:
            ranking_confidence = confidence.masked_fill(
                committed,
                -torch.inf,
            )
            selected = ranking_confidence.topk(
                newly_required,
                dim=-1,
            ).indices
            newly_committed = torch.zeros_like(committed)
            newly_committed.scatter_(1, selected, True)
            current = torch.where(
                newly_committed,
                prediction,
                current,
            )
            committed = committed | newly_committed
        if step in depths:
            states[step] = current.clone()

    for depth in depths:
        local_mask = rollout_depth == depth
        if bool(local_mask.any()):
            local_indices = rollout_indices[local_mask]
            corrupted[local_indices] = states[depth][local_mask]
    loss_mask = corrupted == model.mask_token_id
    if not bool(loss_mask.any(dim=1).all()):
        raise RuntimeError("rollout corruption left a sample without masks")
    committed_mask = ~loss_mask
    committed_count = int(committed_mask.sum().item())
    wrong_count = int(
        ((corrupted != target_routes) & committed_mask).sum().item()
    )
    diagnostics = {
        "rollout_fraction": float(use_rollout.float().mean().item()),
        "all_mask_fraction": float((~use_rollout).float().mean().item()),
        "mean_rollout_depth": (
            float(rollout_depth.float().mean().item())
        ),
        "committed_fraction": float(committed_mask.float().mean().item()),
        "committed_error_fraction": (
            float(wrong_count / committed_count)
            if committed_count
            else 0.0
        ),
    }
    return corrupted.detach(), loss_mask.detach(), diagnostics


def build_pointer_corruption(
    target_routes: torch.Tensor,
    *,
    mask_token_id: int,
    all_mask_probability: float,
    min_mask_ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix all-mask and random partial-mask training states."""

    if target_routes.ndim != 2:
        raise ValueError("target_routes must have shape [B,P]")
    if not 0.0 <= all_mask_probability <= 1.0:
        raise ValueError("all_mask_probability must be in [0,1]")
    if not 0.0 < min_mask_ratio <= 1.0:
        raise ValueError("min_mask_ratio must be in (0,1]")
    batch, positions = target_routes.shape
    all_mask = torch.rand(batch, generator=generator) < all_mask_probability
    uniform = torch.rand(batch, generator=generator)
    ratios = torch.cos(0.5 * math.pi * uniform).clamp_min(min_mask_ratio)
    counts = torch.ceil(ratios * positions).to(torch.int64)
    if positions > 1:
        counts = counts.clamp(min=1, max=positions - 1)
    counts = torch.where(
        all_mask,
        torch.full_like(counts, positions),
        counts,
    )
    random_scores = torch.rand(
        (batch, positions),
        generator=generator,
    )
    order = torch.argsort(random_scores, dim=-1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(positions)[None].expand(batch, -1),
    )
    mask = ranks < counts[:, None]
    corrupted = target_routes.clone()
    corrupted[mask] = int(mask_token_id)
    return corrupted, mask


def cosine_commit_count(
    *,
    positions: int,
    step: int,
    total_steps: int,
) -> int:
    if positions <= 0 or total_steps <= 0 or not 1 <= step <= total_steps:
        raise ValueError("invalid cosine commit schedule arguments")
    if step == total_steps:
        return positions
    remaining = int(
        math.ceil(
            positions
            * math.cos(0.5 * math.pi * float(step) / float(total_steps))
        )
    )
    return positions - min(positions, max(0, remaining))


@torch.inference_mode()
def decode_pointer_routes(
    *,
    model: PointerRouteTransformer,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    steps: int,
    commit_mode: str = "confidence_only",
    evidence_lambda: float = 0.0,
    evidence_config: Mapping[str, Any] | None = None,
    remask_fraction: float | None = None,
    remask_min_score_gain: float | None = None,
) -> PointerDecodeResult:
    """Decode with irreversible commits or conservative evidence remasking.

    ``evidence_remask`` first follows the unchanged cosine commit schedule.
    At each non-final step it may swap a bounded number of the least-supported
    committed positions for the strongest still-masked candidates. A swap is
    allowed only when the new candidate score exceeds the retained-token score
    by ``remask_min_score_gain``. Remasked positions return to the mask state
    and must be predicted again by a later iteration.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if commit_mode not in {
        "confidence_only",
        "evidence_order_only",
        "evidence_logits",
        "evidence_remask",
    }:
        raise ValueError("unsupported PointerRoute commit_mode")
    if evidence_lambda < 0.0:
        raise ValueError("evidence_lambda must be non-negative")
    parameters = dict(evidence_config or {})
    selected_remask_fraction = float(
        parameters.get("remask_fraction", 0.0)
        if remask_fraction is None
        else remask_fraction
    )
    selected_remask_gain = float(
        parameters.get("remask_min_score_gain", 0.0)
        if remask_min_score_gain is None
        else remask_min_score_gain
    )
    if not 0.0 <= selected_remask_fraction <= 1.0:
        raise ValueError("remask_fraction must be in [0,1]")
    if selected_remask_gain < 0.0:
        raise ValueError("remask_min_score_gain must be non-negative")
    if commit_mode != "evidence_remask" and selected_remask_fraction != 0.0:
        raise ValueError("remask_fraction is only valid for evidence_remask")
    model.eval()
    batch = condition_features.shape[0]
    positions = model.spatial_tokens
    device = condition_features.device
    input_routes = model.all_mask_input(batch, device=device)
    committed = torch.zeros(
        (batch, positions),
        dtype=torch.bool,
        device=device,
    )
    commit_steps = torch.zeros(
        (batch, positions),
        dtype=torch.int64,
        device=device,
    )
    commit_confidence = torch.zeros(
        (batch, positions),
        dtype=torch.float32,
        device=device,
    )
    commit_score = torch.zeros_like(commit_confidence)
    commit_evidence = torch.zeros_like(commit_confidence)
    remask_count = torch.zeros_like(commit_steps)
    evidence_result: PointerRouteEvidenceResult | None = None
    if commit_mode in {
        "evidence_order_only",
        "evidence_logits",
        "evidence_remask",
    }:
        candidates = build_route_candidates(
            condition_features,
            condition_quality,
        )
        evidence_result = compute_pointer_route_evidence(
            condition_features=condition_features,
            condition_quality=condition_quality,
            candidates=candidates,
            identity_neighbor_fraction=float(
                parameters.get("identity_neighbor_fraction", 0.5)
            ),
            identity_temperature=float(
                parameters.get("identity_temperature", 3.0)
            ),
            quality_weight=float(parameters.get("quality_weight", 0.25)),
            local_temperature=float(
                parameters.get("local_temperature", 4.0)
            ),
            reliability_floor=float(
                parameters.get("reliability_floor", 1e-4)
            ),
        )
    provisional_parts: list[torch.Tensor] = []
    probability_parts: list[torch.Tensor] = []
    new_parts: list[torch.Tensor] = []
    remasked_parts: list[torch.Tensor] = []
    committed_parts: list[torch.Tensor] = []

    for step in range(1, steps + 1):
        logits, _ = model(
            input_routes,
            condition_features,
            condition_quality,
        )
        selection_logits = logits.float()
        if commit_mode in {"evidence_logits", "evidence_remask"}:
            assert evidence_result is not None
            selection_logits = (
                selection_logits
                + float(evidence_lambda) * evidence_result.evidence
            )
        probabilities = torch.softmax(selection_logits, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        selected_evidence = (
            torch.zeros_like(confidence)
            if evidence_result is None
            else torch.gather(
                evidence_result.evidence,
                2,
                predicted.unsqueeze(-1),
            ).squeeze(-1)
        )
        ordering_score = (
            (
                torch.log(confidence.clamp_min(1e-8))
                + float(evidence_lambda) * selected_evidence
            )
            if commit_mode == "evidence_order_only"
            else confidence
        )
        provisional = torch.where(committed, input_routes, predicted)
        desired = cosine_commit_count(
            positions=positions,
            step=step,
            total_steps=steps,
        )
        current = int(committed[0].sum().item())
        add_count = desired - current
        if add_count < 0:
            raise RuntimeError("commit schedule cannot remove committed routes")
        eligible_score = ordering_score.masked_fill(
            committed,
            float("-inf"),
        )
        newly_committed = torch.zeros_like(committed)
        remasked = torch.zeros_like(committed)
        if add_count > 0:
            selected = torch.topk(
                eligible_score,
                k=add_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            newly_committed.scatter_(1, selected, True)

        if (
            commit_mode == "evidence_remask"
            and step < steps
            and current > 0
            and selected_remask_fraction > 0.0
        ):
            remaining_after_schedule = positions - desired
            remask_budget = min(
                current,
                remaining_after_schedule,
                max(1, int(math.ceil(current * selected_remask_fraction))),
            )
            if remask_budget > 0:
                retained_route = input_routes.clamp_max(
                    model.mask_token_id - 1
                )
                retained_score = torch.gather(
                    probabilities,
                    2,
                    retained_route.unsqueeze(-1),
                ).squeeze(-1)
                retained_score = retained_score.masked_fill(
                    ~committed,
                    float("inf"),
                )
                weakest_score, weakest_index = torch.topk(
                    retained_score,
                    k=remask_budget,
                    dim=-1,
                    largest=False,
                    sorted=True,
                )
                optional_score = eligible_score.masked_fill(
                    newly_committed,
                    float("-inf"),
                )
                strongest_score, strongest_index = torch.topk(
                    optional_score,
                    k=remask_budget,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
                swap = (
                    strongest_score
                    >= weakest_score + selected_remask_gain
                )
                remasked.scatter_(1, weakest_index, swap)
                optional_new = torch.zeros_like(committed)
                optional_new.scatter_(1, strongest_index, swap)
                newly_committed = newly_committed | optional_new

        if bool(remasked.any().item()):
            input_routes = torch.where(
                remasked,
                torch.full_like(input_routes, model.mask_token_id),
                input_routes,
            )
            commit_steps = torch.where(
                remasked,
                torch.zeros_like(commit_steps),
                commit_steps,
            )
            commit_confidence = torch.where(
                remasked,
                torch.zeros_like(commit_confidence),
                commit_confidence,
            )
            commit_score = torch.where(
                remasked,
                torch.zeros_like(commit_score),
                commit_score,
            )
            commit_evidence = torch.where(
                remasked,
                torch.zeros_like(commit_evidence),
                commit_evidence,
            )
            remask_count = remask_count + remasked.to(remask_count.dtype)
        if bool(newly_committed.any().item()):
            input_routes = torch.where(
                newly_committed,
                predicted,
                input_routes,
            )
            commit_steps = torch.where(
                newly_committed,
                torch.full_like(commit_steps, step),
                commit_steps,
            )
            commit_confidence = torch.where(
                newly_committed,
                confidence,
                commit_confidence,
            )
            commit_score = torch.where(
                newly_committed,
                ordering_score,
                commit_score,
            )
            commit_evidence = torch.where(
                newly_committed,
                selected_evidence,
                commit_evidence,
            )
        committed = (committed | newly_committed) & ~remasked
        if not bool((committed.sum(dim=1) == desired).all().item()):
            raise RuntimeError("remask decoder violated the commit schedule")
        provisional_parts.append(provisional)
        probability_parts.append(probabilities)
        new_parts.append(newly_committed)
        remasked_parts.append(remasked)
        committed_parts.append(committed.clone())

    if not bool(committed.all().item()):
        raise RuntimeError("plain PointerRoute decoder left masked positions")
    return PointerDecodeResult(
        final_routes=input_routes,
        final_probabilities=probability_parts[-1],
        provisional_routes=tuple(provisional_parts),
        probabilities=tuple(probability_parts),
        newly_committed=tuple(new_parts),
        remasked=tuple(remasked_parts),
        committed_after_step=tuple(committed_parts),
        commit_steps=commit_steps,
        remask_count=remask_count,
        commit_confidence=commit_confidence,
        commit_score=commit_score,
        commit_evidence=commit_evidence,
        frame_reliability=(
            None
            if evidence_result is None
            else evidence_result.frame_reliability
        ),
        mean_local_usage=(
            None
            if evidence_result is None
            else evidence_result.mean_local_usage
        ),
        effective_frames=(
            None
            if evidence_result is None
            else evidence_result.effective_frames
        ),
    )


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    count = int(denominator.sum().item())
    if count == 0:
        return 0.0
    return float(numerator[denominator].float().mean().item())


def _correctness_transitions(
    *,
    first: torch.Tensor,
    final: torch.Tensor,
) -> dict[str, float]:
    wrong_first = ~first
    correct_first = first
    corrected = wrong_first & final
    injured = correct_first & ~final
    total = float(first.numel())
    return {
        "ecr": _safe_ratio(final, wrong_first),
        "eir": _safe_ratio(~final, correct_first),
        "corrected_fraction_all_positions": float(
            corrected.float().mean().item()
        ),
        "injured_fraction_all_positions": float(
            injured.float().mean().item()
        ),
        "net_correction_fraction": float(
            (corrected.sum() - injured.sum()).item() / total
        ),
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    if left.numel() < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.item()) <= 1e-12:
        return None
    return float((left * right).sum().div(denominator).item())


def trajectory_diagnostics(
    *,
    provisional_routes: Sequence[torch.Tensor],
    target_routes: torch.Tensor,
    teacher_scores: torch.Tensor,
    delta: float,
    scenarios: Sequence[str],
    teacher_maps: torch.Tensor,
    candidates: torch.Tensor,
    identities: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_identities: torch.Tensor,
    commit_steps: torch.Tensor,
    commit_confidence: torch.Tensor,
    commit_score: torch.Tensor,
    commit_evidence: torch.Tensor,
    remasked: Sequence[torch.Tensor] | None,
    visibility: torch.Tensor,
    equivalent_epsilon: float,
) -> dict[str, Any]:
    utilities = margin_adjusted_utilities(teacher_scores.float(), delta=delta)
    exact_by_step = [
        routes == target_routes for routes in provisional_routes
    ]
    equivalent_by_step: list[torch.Tensor] = []
    for routes in provisional_routes:
        selected = torch.gather(
            utilities,
            2,
            routes.unsqueeze(-1),
        ).squeeze(-1)
        equivalent_by_step.append(
            utilities.max(dim=-1).values - selected
            <= float(equivalent_epsilon)
        )
    scenario_array = np.asarray(list(scenarios))
    masks = {
        scenario: torch.from_numpy(scenario_array == scenario)
        for scenario in sorted(set(scenarios))
    }
    masks["all"] = torch.ones(len(scenarios), dtype=torch.bool)
    transitions: dict[str, Any] = {}
    for scenario, mask in masks.items():
        exact_transition = _correctness_transitions(
            first=exact_by_step[0][mask],
            final=exact_by_step[-1][mask],
        )
        equivalent_transition = _correctness_transitions(
            first=equivalent_by_step[0][mask],
            final=equivalent_by_step[-1][mask],
        )
        transitions[scenario] = {
            "exact": exact_transition,
            "equivalent": equivalent_transition,
            "route_change_fraction": float(
                (
                    provisional_routes[0][mask]
                    != provisional_routes[-1][mask]
                )
                .float()
                .mean()
                .item()
            ),
            "no_op_fraction": float(
                (
                    provisional_routes[0][mask]
                    == provisional_routes[-1][mask]
                )
                .float()
                .mean()
                .item()
            ),
        }

    per_step: list[dict[str, Any]] = []
    labels = identities[:, None] == gallery_identities[None, :]
    target_columns = labels.to(torch.int64).argmax(dim=1)
    row_indices = torch.arange(len(identities))
    for index, routes in enumerate(provisional_routes, start=1):
        method_map = reconstruct_hard_routes(candidates, routes)
        selected_score = torch.gather(
            teacher_scores.float(),
            2,
            routes.unsqueeze(-1),
        ).squeeze(-1)
        regret = teacher_scores.max(dim=-1).values - selected_score
        embeddings = F.normalize(method_map.mean(dim=1), dim=-1)
        gallery_scores = embeddings @ gallery_embeddings.transpose(0, 1)
        genuine = gallery_scores[row_indices, target_columns]
        strongest = gallery_scores.masked_fill(
            labels,
            float("-inf"),
        ).max(dim=1).values
        per_step.append(
            {
                "step": index,
                "exact_route_accuracy": float(
                    exact_by_step[index - 1].float().mean().item()
                ),
                "equivalent_route_accuracy": float(
                    equivalent_by_step[index - 1].float().mean().item()
                ),
                "mean_oracle_regret": float(regret.mean().item()),
                "teacher_map_cosine": map_similarity_metrics(
                    method_map,
                    teacher_maps,
                    scenarios,
                )["all"]["mean_map_cosine"],
                "mean_identity_margin": float(
                    (genuine - strongest).mean().item()
                ),
            }
        )

    final_routes = provisional_routes[-1]
    remask_diagnostics: dict[str, Any] = {
        "used": remasked is not None,
        "total_events": 0,
        "event_fraction": 0.0,
        "per_step_events": [],
        "wrong_when_remasked_fraction": None,
        "equivalent_wrong_when_remasked_fraction": None,
        "eventual_exact_correction_fraction": None,
        "eventual_equivalent_correction_fraction": None,
        "eventual_exact_injury_fraction": None,
        "eventual_equivalent_injury_fraction": None,
    }
    if remasked is not None:
        if len(remasked) != len(provisional_routes):
            raise ValueError("remasked trace must match provisional routes")
        event_mask = torch.stack(list(remasked), dim=0).bool()
        total_events = int(event_mask.sum().item())
        exact_at_event = torch.stack(exact_by_step, dim=0)[event_mask]
        equivalent_at_event = torch.stack(
            equivalent_by_step,
            dim=0,
        )[event_mask]
        final_exact_at_event = final_routes.unsqueeze(0).expand(
            len(remasked),
            -1,
            -1,
        )[event_mask] == target_routes.unsqueeze(0).expand(
            len(remasked),
            -1,
            -1,
        )[event_mask]
        final_selected_utility = torch.gather(
            utilities,
            2,
            final_routes.unsqueeze(-1),
        ).squeeze(-1)
        final_equivalent = (
            utilities.max(dim=-1).values - final_selected_utility
            <= float(equivalent_epsilon)
        )
        final_equivalent_at_event = final_equivalent.unsqueeze(0).expand(
            len(remasked),
            -1,
            -1,
        )[event_mask]
        if total_events:
            exact_wrong = ~exact_at_event
            equivalent_wrong = ~equivalent_at_event
            remask_diagnostics.update(
                {
                    "total_events": total_events,
                    "event_fraction": float(
                        total_events
                        / (
                            len(remasked)
                            * int(target_routes.numel())
                        )
                    ),
                    "wrong_when_remasked_fraction": float(
                        exact_wrong.float().mean().item()
                    ),
                    "equivalent_wrong_when_remasked_fraction": float(
                        equivalent_wrong.float().mean().item()
                    ),
                    "eventual_exact_correction_fraction": _safe_ratio(
                        final_exact_at_event,
                        exact_wrong,
                    ),
                    "eventual_equivalent_correction_fraction": _safe_ratio(
                        final_equivalent_at_event,
                        equivalent_wrong,
                    ),
                    "eventual_exact_injury_fraction": _safe_ratio(
                        ~final_exact_at_event,
                        exact_at_event,
                    ),
                    "eventual_equivalent_injury_fraction": _safe_ratio(
                        ~final_equivalent_at_event,
                        equivalent_at_event,
                    ),
                }
            )
        remask_diagnostics["per_step_events"] = [
            int(value.sum().item()) for value in remasked
        ]
    frame_selected = final_routes > 0
    frame_index = final_routes.clamp_min(1) - 1
    selected_visibility = torch.gather(
        visibility.permute(0, 2, 1).bool(),
        2,
        frame_index.unsqueeze(-1),
    ).squeeze(-1)
    selected_visibility = selected_visibility[frame_selected]
    selected_confidence = commit_confidence[frame_selected]
    selected_commit_score = commit_score[frame_selected]
    selected_commit_evidence = commit_evidence[frame_selected]
    return {
        "transitions": transitions,
        "per_step": per_step,
        "mean_commit_step": float(commit_steps.float().mean().item()),
        "commit_step_histogram": {
            str(step): int((commit_steps == step).sum().item())
            for step in range(1, len(provisional_routes) + 1)
        },
        "commit_confidence_visibility_correlation": _pearson(
            selected_confidence,
            selected_visibility.float(),
        ),
        "commit_score_visibility_correlation": _pearson(
            selected_commit_score,
            selected_visibility.float(),
        ),
        "commit_evidence_visibility_correlation": _pearson(
            selected_commit_evidence,
            selected_visibility.float(),
        ),
        "mean_commit_confidence_visible": (
            float(selected_confidence[selected_visibility].mean().item())
            if bool(selected_visibility.any().item())
            else None
        ),
        "mean_commit_confidence_occluded": (
            float(selected_confidence[~selected_visibility].mean().item())
            if bool((~selected_visibility).any().item())
            else None
        ),
        "mean_commit_evidence_visible": (
            float(
                selected_commit_evidence[selected_visibility].mean().item()
            )
            if bool(selected_visibility.any().item())
            else None
        ),
        "mean_commit_evidence_occluded": (
            float(
                selected_commit_evidence[~selected_visibility].mean().item()
            )
            if bool((~selected_visibility).any().item())
            else None
        ),
        "remask": remask_diagnostics,
    }


@torch.inference_mode()
def evaluate_pointer_diffusion_subset(
    *,
    split_name: str,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    targets: Mapping[str, torch.Tensor],
    one_shot_model: PointerRouteTransformer,
    diffusion_model: PointerRouteTransformer,
    config: Mapping[str, Any],
    device: torch.device,
    delta: float,
    temperature: float,
    evaluation_steps: Sequence[int] | None = None,
    commit_mode: str = "confidence_only",
    evidence_lambda: float = 0.0,
    method_prefix: str = "pointer_diffusion",
) -> dict[str, Any]:
    from .pointer_route_training import evaluate_pointer_route_subset

    base = evaluate_pointer_route_subset(
        split_name=split_name,
        dataset=dataset,
        indices=indices,
        targets=targets,
        model=one_shot_model,
        config=config,
        device=device,
        delta=delta,
        temperature=temperature,
    )
    values = _subset_tensors(dataset=dataset, indices=indices)
    selected_steps = [
        int(value)
        for value in (
            evaluation_steps
            if evaluation_steps is not None
            else config["pointer_route"]["diffusion"]["evaluation_steps"]
        )
    ]
    batch_size = int(config["evaluation"]["batch_size"])
    route_parts: dict[int, list[torch.Tensor]] = {
        steps: [] for steps in selected_steps
    }
    probability_parts: dict[int, list[torch.Tensor]] = {
        steps: [] for steps in selected_steps
    }
    map_parts: dict[int, list[torch.Tensor]] = {
        steps: [] for steps in selected_steps
    }
    primary_steps = int(config["pointer_route"]["diffusion"]["default_steps"])
    primary_provisional: list[list[torch.Tensor]] = [
        [] for _ in range(primary_steps)
    ]
    primary_commit_steps: list[torch.Tensor] = []
    primary_commit_confidence: list[torch.Tensor] = []
    primary_commit_score: list[torch.Tensor] = []
    primary_commit_evidence: list[torch.Tensor] = []
    primary_remasked: list[list[torch.Tensor]] = [
        [] for _ in range(primary_steps)
    ]
    primary_frame_reliability: list[torch.Tensor] = []
    primary_local_usage: list[torch.Tensor] = []
    primary_effective_frames: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        condition = values["condition_features"][start:end].to(device)
        quality = values["condition_quality"][start:end].to(device)
        candidates = diffusion_model(
            diffusion_model.all_mask_input(end - start, device=device),
            condition,
            quality,
        )[1]
        candidate_parts.append(candidates.cpu().half())
        evidence_section = {
            "evidence_logits": "evidence_logits",
            "evidence_remask": "evidence_remask",
        }.get(commit_mode, "evidence_order")
        for steps in selected_steps:
            decoded = decode_pointer_routes(
                model=diffusion_model,
                condition_features=condition,
                condition_quality=quality,
                steps=steps,
                commit_mode=commit_mode,
                evidence_lambda=evidence_lambda,
                evidence_config=config["pointer_route"]["diffusion"].get(
                    evidence_section
                ),
            )
            routes = decoded.final_routes
            route_parts[steps].append(routes.cpu())
            probability_parts[steps].append(
                decoded.final_probabilities.cpu().half()
            )
            map_parts[steps].append(
                reconstruct_hard_routes(candidates, routes).cpu().half()
            )
            if steps == primary_steps:
                for index, provisional in enumerate(
                    decoded.provisional_routes
                ):
                    primary_provisional[index].append(provisional.cpu())
                primary_commit_steps.append(decoded.commit_steps.cpu())
                primary_commit_confidence.append(
                    decoded.commit_confidence.cpu()
                )
                primary_commit_score.append(decoded.commit_score.cpu())
                primary_commit_evidence.append(
                    decoded.commit_evidence.cpu()
                )
                for index, remasked in enumerate(decoded.remasked):
                    primary_remasked[index].append(remasked.cpu())
                if decoded.frame_reliability is not None:
                    primary_frame_reliability.append(
                        decoded.frame_reliability.cpu()
                    )
                    assert decoded.mean_local_usage is not None
                    assert decoded.effective_frames is not None
                    primary_local_usage.append(
                        decoded.mean_local_usage.cpu()
                    )
                    primary_effective_frames.append(
                        decoded.effective_frames.cpu()
                    )

    identities = values["target_identity"].to(torch.int64)
    scenarios = values["scenarios"]
    teacher_maps = values["teacher_maps"].float()
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identities[clean_indices]
    gallery_embeddings = F.normalize(
        teacher_maps[clean_indices].mean(dim=1),
        dim=-1,
    )
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    target_indices = values["global_indices"]
    hard_targets = targets["hard_routes"][target_indices]
    teacher_scores = targets["teacher_scores"][target_indices].float()
    routes_by_steps = {
        steps: torch.cat(parts) for steps, parts in route_parts.items()
    }
    probabilities_by_steps = {
        steps: torch.cat(parts).float()
        for steps, parts in probability_parts.items()
    }
    maps_by_steps = {
        steps: torch.cat(parts).float() for steps, parts in map_parts.items()
    }
    route_accuracy: dict[str, Any] = {}
    route_diagnostic_values: dict[str, Any] = {}
    for steps in selected_steps:
        name = f"{method_prefix}_{steps}step"
        method_map = maps_by_steps[steps]
        base["methods"][name] = {
            "verification": _evaluate_embeddings(
                embeddings=F.normalize(method_map.mean(dim=1), dim=-1),
                gallery_embeddings=gallery_embeddings,
                identities=identities,
                gallery_identities=gallery_identities,
                scenarios=scenarios,
                far_targets=[
                    float(value)
                    for value in config["evaluation"]["far_targets"]
                ],
            ),
            "teacher_similarity": map_similarity_metrics(
                method_map,
                teacher_maps,
                scenarios,
            ),
        }
        route_accuracy[name] = _route_accuracy(
            predicted=routes_by_steps[steps],
            target=hard_targets,
            scores=teacher_scores,
            delta=delta,
            equivalent_epsilon=float(
                config["pointer_route"]["equivalent_epsilon"]
            ),
            scenarios=scenarios,
        )
        route_diagnostic_values[name] = route_diagnostics(
            routes=routes_by_steps[steps],
            probabilities=probabilities_by_steps[steps],
            scores=teacher_scores,
            delta=delta,
            source_identities=values["source_identities"],
            target_identities=identities,
            visibility=values["condition_visibility"],
            scenarios=scenarios,
        )

    provisional = [
        torch.cat(parts) for parts in primary_provisional
    ]
    candidates = torch.cat(candidate_parts).float()
    trajectory = trajectory_diagnostics(
        provisional_routes=provisional,
        target_routes=hard_targets,
        teacher_scores=teacher_scores,
        delta=delta,
        scenarios=scenarios,
        teacher_maps=teacher_maps,
        candidates=candidates,
        identities=identities,
        gallery_embeddings=gallery_embeddings,
        gallery_identities=gallery_identities,
        commit_steps=torch.cat(primary_commit_steps),
        commit_confidence=torch.cat(primary_commit_confidence),
        commit_score=torch.cat(primary_commit_score),
        commit_evidence=torch.cat(primary_commit_evidence),
        remasked=(
            [
                torch.cat(parts)
                for parts in primary_remasked
            ]
            if commit_mode == "evidence_remask"
            else None
        ),
        visibility=values["condition_visibility"],
        equivalent_epsilon=float(
            config["pointer_route"]["equivalent_epsilon"]
        ),
    )

    first_count = min(16, len(indices))
    first_condition = values["condition_features"][:first_count].to(device)
    first_quality = values["condition_quality"][:first_count].to(device)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    original = decode_pointer_routes(
        model=diffusion_model,
        condition_features=first_condition,
        condition_quality=first_quality,
        steps=primary_steps,
        commit_mode=commit_mode,
        evidence_lambda=evidence_lambda,
        evidence_config=config["pointer_route"]["diffusion"].get(
            {
                "evidence_logits": "evidence_logits",
                "evidence_remask": "evidence_remask",
            }.get(commit_mode, "evidence_order")
        ),
    )
    permuted = decode_pointer_routes(
        model=diffusion_model,
        condition_features=first_condition[:, permutation],
        condition_quality=first_quality[:, permutation],
        steps=primary_steps,
        commit_mode=commit_mode,
        evidence_lambda=evidence_lambda,
        evidence_config=config["pointer_route"]["diffusion"].get(
            {
                "evidence_logits": "evidence_logits",
                "evidence_remask": "evidence_remask",
            }.get(commit_mode, "evidence_order")
        ),
    )
    original_candidates = diffusion_model(
        diffusion_model.all_mask_input(first_count, device=device),
        first_condition,
        first_quality,
    )[1]
    permuted_candidates = diffusion_model(
        diffusion_model.all_mask_input(first_count, device=device),
        first_condition[:, permutation],
        first_quality[:, permutation],
    )[1]
    original_map = reconstruct_hard_routes(
        original_candidates,
        original.final_routes,
    )
    permuted_map = reconstruct_hard_routes(
        permuted_candidates,
        permuted.final_routes,
    )
    permutation_check = {
        "permutation": permutation.tolist(),
        "route_map_max_abs_difference": float(
            (original_map - permuted_map).abs().max().item()
        ),
        "route_map_cosine_min": float(
            F.cosine_similarity(original_map, permuted_map, dim=-1)
            .min()
            .item()
        ),
    }
    base["diffusion_route_accuracy"] = route_accuracy
    base["diffusion_route_diagnostics"] = route_diagnostic_values
    base["trajectory"] = trajectory
    base["diffusion_permutation_check"] = permutation_check
    base["diffusion_inference_contract"] = {
        "teacher_used": False,
        "gallery_used": False,
        "evidence_used": commit_mode != "confidence_only",
        "remask_used": commit_mode == "evidence_remask",
        "commit_mode": commit_mode,
        "selection_logits_modified": commit_mode in {
            "evidence_logits",
            "evidence_remask",
        },
        "model_logits_modified": False,
        "evidence_lambda": float(evidence_lambda),
    }
    if primary_frame_reliability:
        frame_reliability = torch.cat(primary_frame_reliability)
        local_usage = torch.cat(primary_local_usage)
        effective_frames = torch.cat(primary_effective_frames)
        source_identities = values["source_identities"].to(torch.int64)
        wrong_identity = (
            source_identities != identities.unsqueeze(1)
        )
        base["evidence_diagnostics"] = {
            "mean_frame_reliability": [
                float(value)
                for value in frame_reliability.mean(dim=0).tolist()
            ],
            "mean_local_usage": [
                float(value) for value in local_usage.mean(dim=0).tolist()
            ],
            "mean_effective_frames": float(
                effective_frames.mean().item()
            ),
            "mean_wrong_identity_frame_reliability": (
                float(frame_reliability[wrong_identity].mean().item())
                if bool(wrong_identity.any().item())
                else None
            ),
            "mean_inlier_frame_reliability": float(
                frame_reliability[~wrong_identity].mean().item()
            ),
            "stop_gradient": True,
            "by_scenario": {},
        }
        scenario_array = np.asarray(list(scenarios))
        for scenario in sorted(set(scenarios)):
            scenario_mask = torch.from_numpy(scenario_array == scenario)
            local_wrong = wrong_identity[scenario_mask]
            local_reliability = frame_reliability[scenario_mask]
            base["evidence_diagnostics"]["by_scenario"][scenario] = {
                "mean_frame_reliability": [
                    float(value)
                    for value in local_reliability.mean(dim=0).tolist()
                ],
                "mean_local_usage": [
                    float(value)
                    for value in local_usage[scenario_mask].mean(
                        dim=0
                    ).tolist()
                ],
                "mean_effective_frames": float(
                    effective_frames[scenario_mask].mean().item()
                ),
                "mean_wrong_identity_frame_reliability": (
                    float(
                        local_reliability[local_wrong].mean().item()
                    )
                    if bool(local_wrong.any().item())
                    else None
                ),
            }
    else:
        base["evidence_diagnostics"] = None
    return base


def _learning_rate(
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    base: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base * float(step) / float(warmup_steps)
    progress = min(
        1.0,
        max(
            0.0,
            float(step - warmup_steps)
            / float(max(1, max_steps - warmup_steps)),
        ),
    )
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def train_pointer_diffusion(
    *,
    dataset: TokenTrainingDataset,
    train_indices: Sequence[int],
    calibration_indices: Sequence[int],
    targets: Mapping[str, torch.Tensor],
    model: PointerRouteTransformer,
    paired_one_shot_model: PointerRouteTransformer,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_directory: str | Path,
    delta: float,
    temperature: float,
    seed: int,
    smoke: bool,
    corruption_mode: str = "random_partial",
    metric_callback: Callable[[int, Mapping[str, float]], None] | None = None,
) -> dict[str, Any]:
    from .pointer_route_training import _next_batch

    section = config["pointer_route"]["one_shot"]
    diffusion = config["pointer_route"]["diffusion"]
    max_steps = (
        int(section["smoke_steps"]) if smoke else int(section["max_steps"])
    )
    min_steps = 1 if smoke else int(section["min_steps"])
    validate_every = max_steps if smoke else int(
        section["validate_every_steps"]
    )
    accumulation = 1 if smoke else int(section["gradient_accumulation"])
    batch_size = min(2, int(section["batch_size"])) if smoke else int(
        section["batch_size"]
    )
    warmup = 0 if smoke else int(section["warmup_steps"])
    checkpoint_root = Path(checkpoint_directory).expanduser().resolve()
    if corruption_mode not in {"random_partial", "self_rollout"}:
        raise ValueError(
            "corruption_mode must be 'random_partial' or 'self_rollout'"
        )
    if corruption_mode == "self_rollout" and device.type != "cuda":
        raise RuntimeError("self-rollout rescue training requires CUDA")
    model.to(device)
    paired_one_shot_model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    use_fp16 = str(section["precision"]) == "fp16"
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and use_fp16,
    )
    batch_generator = torch.Generator().manual_seed(int(seed))
    corruption_generator = torch.Generator().manual_seed(int(seed) + 1)
    sampler_state: dict[str, Any] = {}
    train_tensor = torch.tensor(list(train_indices), dtype=torch.int64)
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    best_score: tuple[float, float, float] | None = None
    best_checkpoint: Path | None = None
    best_step = 0
    patience = 0
    started = time.perf_counter()
    device_cache: dict[str, torch.Tensor] | None = None
    cache_bytes = 0
    if corruption_mode == "self_rollout":
        rescue = diffusion["rollout_rescue"]
        if not bool(rescue["use_gpu_resident_training_cache"]):
            raise RuntimeError("self-rollout rescue requires GPU cache")
        precomputed = _require_precomputed(dataset)
        device_cache = {
            "condition_features": precomputed["condition_features"].to(
                device,
                non_blocking=True,
            ),
            "condition_quality": precomputed["condition_quality"].to(
                device,
                non_blocking=True,
            ),
            "teacher_maps": dataset.targets["continuous_prototypes"].to(
                device,
                non_blocking=True,
            ),
            "hard_routes": targets["hard_routes"].to(
                device,
                non_blocking=True,
            ),
            "soft_probabilities": targets["soft_probabilities"].to(
                device,
                non_blocking=True,
            ),
        }
        cache_bytes = sum(
            value.numel() * value.element_size()
            for value in device_cache.values()
        )
        torch.cuda.synchronize(device)
    device_audit: dict[str, Any] = {
        "requested_device": str(device),
        "model_device": str(next(model.parameters()).device),
        "autocast_dtype": str(autocast_dtype),
        "gpu_resident_training_cache": device_cache is not None,
        "gpu_cache_mib": cache_bytes / 1024**2,
        "batch_residency_verified": False,
        "rollout_inference_device": None,
        "loss_device": None,
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _ in range(accumulation):
            indices = _next_batch(
                train_indices=train_tensor,
                batch_size=batch_size,
                generator=batch_generator,
                state=sampler_state,
            )
            if device_cache is None:
                precomputed = _require_precomputed(dataset)
                condition = precomputed["condition_features"][indices].to(
                    device
                )
                quality = precomputed["condition_quality"][indices].to(device)
                teacher = dataset.targets["continuous_prototypes"][
                    indices
                ].to(device)
                hard_cpu = targets["hard_routes"][indices]
                input_routes_cpu, loss_mask_cpu = build_pointer_corruption(
                    hard_cpu,
                    mask_token_id=model.mask_token_id,
                    all_mask_probability=float(
                        diffusion["all_mask_probability"]
                    ),
                    min_mask_ratio=float(diffusion["min_mask_ratio"]),
                    generator=corruption_generator,
                )
                hard = hard_cpu.to(device)
                soft = targets["soft_probabilities"][indices].to(device)
                input_routes = input_routes_cpu.to(device)
                loss_mask = loss_mask_cpu.to(device)
                corruption_values: dict[str, float] = {}
            else:
                device_indices = indices.to(device, non_blocking=True)
                condition = device_cache["condition_features"].index_select(
                    0,
                    device_indices,
                )
                quality = device_cache["condition_quality"].index_select(
                    0,
                    device_indices,
                )
                teacher = device_cache["teacher_maps"].index_select(
                    0,
                    device_indices,
                )
                hard = device_cache["hard_routes"].index_select(
                    0,
                    device_indices,
                )
                soft = device_cache["soft_probabilities"].index_select(
                    0,
                    device_indices,
                )
                rescue = diffusion["rollout_rescue"]
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=True,
                ):
                    input_routes, loss_mask, corruption_values = (
                        build_pointer_rollout_corruption(
                            model=model,
                            condition_features=condition,
                            condition_quality=quality,
                            target_routes=hard,
                            rollout_probability=float(
                                rescue["rollout_probability"]
                            ),
                            rollout_total_steps=int(
                                rescue["rollout_total_steps"]
                            ),
                            rollout_depths=[
                                int(value)
                                for value in rescue["rollout_depths"]
                            ],
                            generator=corruption_generator,
                        )
                    )
            if not device_audit["batch_residency_verified"]:
                batch_tensors = (
                    condition,
                    quality,
                    teacher,
                    hard,
                    soft,
                    input_routes,
                    loss_mask,
                )
                if any(value.device != device for value in batch_tensors):
                    raise RuntimeError(
                        "PointerRoute training batch escaped the target device"
                    )
                device_audit["batch_residency_verified"] = True
                device_audit["rollout_inference_device"] = str(
                    input_routes.device
                )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                logits, candidates = model(
                    input_routes,
                    condition,
                    quality,
                )
                loss, loss_values = pointer_route_loss(
                    logits=logits,
                    candidates=candidates,
                    hard_targets=hard,
                    soft_targets=soft,
                    teacher_maps=teacher,
                    config=config,
                    loss_mask=loss_mask,
                )
                scaled_loss = loss / float(accumulation)
            if scaled_loss.device != device:
                raise RuntimeError("PointerRoute loss escaped the target device")
            device_audit["loss_device"] = str(scaled_loss.device)
            scaler.scale(scaled_loss).backward()
            for key, value in loss_values.items():
                accumulated[key] = accumulated.get(key, 0.0) + (
                    value / float(accumulation)
                )
            for key, value in corruption_values.items():
                name = f"corruption_{key}"
                accumulated[name] = accumulated.get(name, 0.0) + (
                    value / float(accumulation)
                )
        scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(section["max_grad_norm"]),
            ).item()
        )
        learning_rate = _learning_rate(
            step=step,
            max_steps=max_steps,
            warmup_steps=warmup,
            base=float(section["learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        scaler.step(optimizer)
        scaler.update()
        row = {
            "step": step,
            **accumulated,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        if metric_callback is not None:
            metric_callback(step, row)

        if step % validate_every == 0 or step == max_steps:
            validation = evaluate_pointer_diffusion_subset(
                split_name="route_calibration",
                dataset=dataset,
                indices=calibration_indices,
                targets=targets,
                one_shot_model=paired_one_shot_model,
                diffusion_model=model,
                config=config,
                device=device,
                delta=delta,
                temperature=temperature,
                evaluation_steps=[int(diffusion["default_steps"])],
            )
            name = f"pointer_diffusion_{int(diffusion['default_steps'])}step"
            primary = validation["methods"][name]["verification"]
            net_correction = validation["trajectory"]["transitions"]["all"][
                "equivalent"
            ]["net_correction_fraction"]
            score = (
                float(primary["hard_average"]["roc_auc"]),
                float(primary["all"]["roc_auc"]),
                float(net_correction),
            )
            validations.append(
                {
                    "step": step,
                    "score": list(score),
                    "hard_auc": score[0],
                    "all_auc": score[1],
                    "equivalent_net_correction": score[2],
                }
            )
            if best_score is None or score > best_score:
                prefix = (
                    "rollout_rescue"
                    if corruption_mode == "self_rollout"
                    else "diffusion"
                )
                checkpoint = checkpoint_root / f"{prefix}_step_{step:08d}.pt"
                _atomic_checkpoint(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "stage": (
                            "pointer-route-rollout-rescue"
                            if corruption_mode == "self_rollout"
                            else "pointer-route-plain-diffusion"
                        ),
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "parameter_count": model.parameter_count(),
                        "validation": validation,
                        "delta": float(delta),
                        "temperature": float(temperature),
                        "config": dict(diffusion),
                        "corruption_mode": corruption_mode,
                        "device_audit": device_audit,
                    },
                )
                best_score = score
                best_checkpoint = checkpoint
                best_step = step
                patience = 0
            else:
                patience += 1
            if (
                not smoke
                and not bool(diffusion["paired_force_full_steps"])
                and step >= min_steps
                and patience >= int(section["early_stop_patience"])
            ):
                break
    if best_checkpoint is None:
        raise RuntimeError("plain PointerRoute diffusion produced no checkpoint")
    return {
        "steps_completed": int(history[-1]["step"]),
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint),
        "best_score": list(best_score or ()),
        "history": history,
        "validations": validations,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": model.parameter_count(),
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
        "smoke": smoke,
        "corruption_mode": corruption_mode,
        "device_audit": device_audit,
    }


@torch.inference_mode()
def measure_diffusion_latency(
    *,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    model: PointerRouteTransformer,
    config: Mapping[str, Any],
    device: torch.device,
    commit_mode: str = "confidence_only",
    evidence_lambda: float = 0.0,
) -> dict[str, Any]:
    batch_size = 16
    precomputed = _require_precomputed(dataset)
    selected = torch.tensor(list(indices[:batch_size]), dtype=torch.int64)
    condition = precomputed["condition_features"][selected].to(device)
    quality = precomputed["condition_quality"][selected].to(device)
    warmup = int(config["evaluation"]["latency_warmup_runs"])
    repeats = int(config["evaluation"]["latency_measurement_runs"])
    result: dict[str, Any] = {}
    for steps in config["pointer_route"]["diffusion"]["evaluation_steps"]:
        steps = int(steps)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            decode_pointer_routes(
                model=model,
                condition_features=condition,
                condition_quality=quality,
                steps=steps,
                commit_mode=commit_mode,
                evidence_lambda=evidence_lambda,
                evidence_config=config["pointer_route"]["diffusion"].get(
                    {
                        "evidence_logits": "evidence_logits",
                        "evidence_remask": "evidence_remask",
                    }.get(commit_mode, "evidence_order")
                ),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            decode_pointer_routes(
                model=model,
                condition_features=condition,
                condition_quality=quality,
                steps=steps,
                commit_mode=commit_mode,
                evidence_lambda=evidence_lambda,
                evidence_config=config["pointer_route"]["diffusion"].get(
                    {
                        "evidence_logits": "evidence_logits",
                        "evidence_remask": "evidence_remask",
                    }.get(commit_mode, "evidence_order")
                ),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        result[f"{steps}step"] = {
            "batch_size": batch_size,
            "mean_batch_latency_ms": 1000.0 * elapsed / repeats,
            "mean_sample_latency_ms": (
                1000.0 * elapsed / (repeats * batch_size)
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
            "commit_mode": commit_mode,
            "evidence_lambda": float(evidence_lambda),
        }
    return result
