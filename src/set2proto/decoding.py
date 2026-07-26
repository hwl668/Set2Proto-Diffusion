"""One-shot and MaskGIT-style iterative decoding with local evidence guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .model import ConditionalTokenTransformer


DecodeMode = Literal[
    "confidence",
    "evidence-ordering",
    "evidence-logits",
    "evidence-remask",
]


@dataclass(frozen=True)
class DecodeResult:
    tokens: torch.Tensor
    commit_iteration: torch.Tensor
    commit_score: torch.Tensor
    trace: list[dict[str, Any]]
    provisional_tokens: tuple[torch.Tensor, ...]
    committed_after_iteration: tuple[torch.Tensor, ...]
    remasked: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TwoLevelEvidenceResult:
    """Observation evidence and compact frame-level diagnostics.

    ``frame_reliability`` combines a robust, leave-one-out identity consensus
    with the supplied whole-frame quality. ``mean_local_usage`` reports how
    often each frame contributes after candidate-specific local reweighting.
    All tensors are stop-gradient outputs.
    """

    evidence: torch.Tensor
    frame_reliability: torch.Tensor
    identity_consensus: torch.Tensor
    identity_reliability: torch.Tensor
    quality_standardized: torch.Tensor
    mean_local_usage: torch.Tensor
    effective_frames: torch.Tensor


@dataclass(frozen=True)
class FrameReliabilityResult:
    """Permutation-equivariant global identity/quality frame diagnostics."""

    frame_reliability: torch.Tensor
    identity_consensus: torch.Tensor
    identity_reliability: torch.Tensor
    quality_standardized: torch.Tensor
    effective_frames: torch.Tensor


def _robust_frame_standardize(
    values: torch.Tensor,
    *,
    clip: float = 4.0,
    scale_floor: float = 1e-6,
) -> torch.Tensor:
    """Median/MAD standardization along the frame dimension."""

    # The CUDA median-with-indices kernel is not deterministic. Sorting four
    # frames is cheap, preserves the lower-median convention for even M, and
    # keeps this evidence path reproducible under the project runtime policy.
    median_index = (int(values.shape[1]) - 1) // 2
    center = values.sort(dim=1).values[
        :, median_index : median_index + 1
    ]
    deviation = (values - center).abs()
    scale = 1.4826 * deviation.sort(dim=1).values[
        :, median_index : median_index + 1
    ]
    standardized = (values - center) / scale.clamp_min(scale_floor)
    return standardized.clamp(min=-clip, max=clip)


def compute_frame_reliability(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    *,
    identity_neighbor_fraction: float = 0.5,
    identity_temperature: float = 3.0,
    quality_weight: float = 0.25,
    reliability_floor: float = 1e-4,
) -> FrameReliabilityResult:
    """Compute the global identity-inlier prior shared by evidence variants."""
    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,M,P,D]")
    if condition_quality.ndim != 2:
        raise ValueError("condition_quality must have shape [B,M]")
    batch, frames, _, _ = condition_features.shape
    if tuple(condition_quality.shape) != (batch, frames):
        raise ValueError(
            "condition_quality must match the condition batch and frames"
        )
    if frames < 1:
        raise ValueError("condition_features must contain at least one frame")
    if not 0.0 < identity_neighbor_fraction <= 1.0:
        raise ValueError("identity_neighbor_fraction must be in (0,1]")
    if identity_temperature < 0.0:
        raise ValueError("identity_temperature must be non-negative")
    if quality_weight < 0.0:
        raise ValueError("quality_weight must be non-negative")
    if reliability_floor <= 0.0:
        raise ValueError("reliability_floor must be positive")

    with torch.no_grad():
        normalized_condition = F.normalize(
            condition_features.float(),
            dim=-1,
        )
        frame_descriptors = F.normalize(
            normalized_condition.mean(dim=2),
            dim=-1,
        )
        if frames == 1:
            identity_consensus = torch.ones(
                (batch, 1),
                dtype=torch.float32,
                device=condition_features.device,
            )
        else:
            pairwise = torch.einsum(
                "bmd,bnd->bmn",
                frame_descriptors,
                frame_descriptors,
            )
            diagonal = torch.eye(
                frames,
                dtype=torch.bool,
                device=condition_features.device,
            ).unsqueeze(0)
            pairwise = pairwise.masked_fill(diagonal, float("-inf"))
            neighbors = max(
                1,
                min(
                    frames - 1,
                    int(
                        math.ceil(
                            identity_neighbor_fraction * (frames - 1)
                        )
                    ),
                ),
            )
            identity_consensus = torch.topk(
                pairwise,
                k=neighbors,
                dim=-1,
                largest=True,
                sorted=False,
            ).values.mean(dim=-1)

        identity_standardized = _robust_frame_standardize(
            identity_consensus,
            # Small descriptor changes caused by one complementary region are
            # not identity-outlier evidence. A cosine tolerance prevents a
            # zero MAD among near-duplicate inliers from collapsing that frame.
            scale_floor=0.05,
        )
        identity_reliability = torch.sigmoid(
            identity_temperature * identity_standardized
        )
        quality_standardized = _robust_frame_standardize(
            condition_quality.float()
        )

        # Multiplication, rather than an unconstrained additive identity logit,
        # makes global identity consistency a genuine gate. Quality remains a
        # bounded secondary preference and cannot rescue a strong outlier.
        unnormalized_reliability = (
            identity_reliability
            * torch.exp(quality_weight * quality_standardized)
        ).clamp_min(reliability_floor)
        frame_reliability = unnormalized_reliability / (
            unnormalized_reliability.sum(dim=1, keepdim=True)
        )
        effective_frames = frame_reliability.square().sum(
            dim=1
        ).reciprocal()

    return FrameReliabilityResult(
        frame_reliability=frame_reliability.detach(),
        identity_consensus=identity_consensus.detach(),
        identity_reliability=identity_reliability.detach(),
        quality_standardized=quality_standardized.detach(),
        effective_frames=effective_frames.detach(),
    )


def combine_two_level_candidate_support(
    per_frame_support: torch.Tensor,
    frame_reliability: torch.Tensor,
    *,
    local_temperature: float = 4.0,
    reliability_floor: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse arbitrary candidate support using a global frame-inlier prior.

    ``per_frame_support`` has shape ``[B,M,P,K]``. Keeping this operation
    separate lets residual-token decoding score the sample-dependent
    ``normalize(anchor + residual_code)`` candidates without pretending that
    Euclidean residual centroids are unit absolute prototypes.
    """

    if per_frame_support.ndim != 4:
        raise ValueError("per_frame_support must have shape [B,M,P,K]")
    if frame_reliability.ndim != 2:
        raise ValueError("frame_reliability must have shape [B,M]")
    if tuple(frame_reliability.shape) != tuple(per_frame_support.shape[:2]):
        raise ValueError("frame reliability must match support batch/frames")
    if local_temperature < 0.0:
        raise ValueError("local_temperature must be non-negative")
    if reliability_floor <= 0.0:
        raise ValueError("reliability_floor must be positive")
    if not bool(torch.isfinite(per_frame_support).all().item()):
        raise ValueError("per_frame_support contains non-finite values")

    with torch.no_grad():
        normalized_reliability = frame_reliability.float().clamp_min(
            reliability_floor
        )
        normalized_reliability = normalized_reliability / (
            normalized_reliability.sum(dim=1, keepdim=True)
        )
        candidate_frame_weights = torch.softmax(
            normalized_reliability.log()[:, :, None, None]
            + local_temperature * per_frame_support.float(),
            dim=1,
        )
        evidence = (
            candidate_frame_weights * per_frame_support.float()
        ).sum(dim=1)
        mean_local_usage = candidate_frame_weights.mean(dim=(2, 3))
    return evidence.detach(), mean_local_usage.detach()


def compute_two_level_evidence(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    codebook: torch.Tensor,
    *,
    identity_neighbor_fraction: float = 0.5,
    identity_temperature: float = 3.0,
    quality_weight: float = 0.25,
    local_temperature: float = 4.0,
    reliability_floor: float = 1e-4,
) -> TwoLevelEvidenceResult:
    """Compute global-identity x local-candidate evidence.

    The first level assigns a permutation-equivariant reliability to every
    frame. The second level reweights those priors separately for every local
    token candidate, preserving complementary evidence from an inlier while
    preventing a globally inconsistent frame from dominating.
    """

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,M,P,D]")
    if codebook.ndim != 2 or int(codebook.shape[1]) != int(
        condition_features.shape[-1]
    ):
        raise ValueError("condition/codebook dimensions do not match")
    if local_temperature < 0.0:
        raise ValueError("local_temperature must be non-negative")

    reliability = compute_frame_reliability(
        condition_features,
        condition_quality,
        identity_neighbor_fraction=identity_neighbor_fraction,
        identity_temperature=identity_temperature,
        quality_weight=quality_weight,
        reliability_floor=reliability_floor,
    )
    with torch.no_grad():
        normalized_condition = F.normalize(
            condition_features.float(),
            dim=-1,
        )
        normalized_codebook = F.normalize(codebook.float(), dim=-1)
        per_frame_support = torch.einsum(
            "bmpd,kd->bmpk",
            normalized_condition,
            normalized_codebook,
        )
    evidence, mean_local_usage = combine_two_level_candidate_support(
        per_frame_support,
        reliability.frame_reliability,
        local_temperature=local_temperature,
        reliability_floor=reliability_floor,
    )

    return TwoLevelEvidenceResult(
        evidence=evidence.detach(),
        frame_reliability=reliability.frame_reliability,
        identity_consensus=reliability.identity_consensus,
        identity_reliability=reliability.identity_reliability,
        quality_standardized=reliability.quality_standardized,
        mean_local_usage=mean_local_usage.detach(),
        effective_frames=reliability.effective_frames,
    )


def compute_local_evidence(
    condition_features: torch.Tensor,
    codebook: torch.Tensor,
    *,
    top_k_frames: int,
) -> torch.Tensor:
    """Return stop-gradient [B,P,K] top-frame mean cosine evidence."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,M,P,D]")
    if condition_features.shape[-1] != codebook.shape[-1]:
        raise ValueError("condition/codebook dimensions do not match")
    frames = int(condition_features.shape[1])
    if not 1 <= top_k_frames <= frames:
        raise ValueError("top_k_frames must be in [1, condition_frames]")
    with torch.no_grad():
        normalized_condition = F.normalize(
            condition_features.float(),
            dim=-1,
        )
        normalized_codebook = F.normalize(codebook.float(), dim=-1)
        per_frame = torch.einsum(
            "bmpd,kd->bmpk",
            normalized_condition,
            normalized_codebook,
        )
        evidence = torch.topk(
            per_frame,
            k=top_k_frames,
            dim=1,
            largest=True,
            sorted=False,
        ).values.mean(dim=1)
    return evidence.detach()


def _target_commit_count(
    *,
    iteration: int,
    steps: int,
    positions: int,
) -> int:
    fraction = 1.0 - math.cos(
        0.5 * math.pi * float(iteration + 1) / float(steps)
    )
    return min(positions, max(1, int(math.ceil(positions * fraction))))


@torch.inference_mode()
def one_shot_decode(
    model: ConditionalTokenTransformer,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    evidence_anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = condition_features.shape[0]
    input_tokens = torch.full(
        (batch, model.spatial_tokens),
        model.mask_token_id,
        dtype=torch.int64,
        device=condition_features.device,
    )
    logits = (
        model(input_tokens, condition_features, condition_quality)
        if evidence_anchor is None
        else model(
            input_tokens,
            condition_features,
            condition_quality,
            evidence_anchor,
        )
    )
    return logits.argmax(dim=-1)


@torch.inference_mode()
def maskgit_decode(
    model: ConditionalTokenTransformer,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    *,
    codebook: torch.Tensor,
    steps: int,
    mode: DecodeMode,
    top_k_frames: int,
    evidence_lambda: float,
    evidence_override: torch.Tensor | None = None,
    evidence_anchor: torch.Tensor | None = None,
) -> DecodeResult:
    if steps < 1:
        raise ValueError("steps must be positive")
    if mode not in {
        "confidence",
        "evidence-ordering",
        "evidence-logits",
        "evidence-remask",
    }:
        raise ValueError(f"unsupported decode mode: {mode}")

    batch = condition_features.shape[0]
    positions = model.spatial_tokens
    tokens = torch.full(
        (batch, positions),
        model.mask_token_id,
        dtype=torch.int64,
        device=condition_features.device,
    )
    committed = torch.zeros(
        (batch, positions),
        dtype=torch.bool,
        device=condition_features.device,
    )
    commit_iteration = torch.full(
        (batch, positions),
        -1,
        dtype=torch.int64,
        device=condition_features.device,
    )
    committed_score = torch.full(
        (batch, positions),
        float("nan"),
        dtype=torch.float32,
        device=condition_features.device,
    )
    evidence = None
    if mode != "confidence":
        if evidence_override is None:
            evidence = compute_local_evidence(
                condition_features,
                codebook,
                top_k_frames=top_k_frames,
            )
        else:
            expected_shape = (
                batch,
                positions,
                model.codebook_size,
            )
            if tuple(evidence_override.shape) != expected_shape:
                raise ValueError(
                    "evidence_override must have shape "
                    f"{expected_shape}, got {tuple(evidence_override.shape)}"
                )
            if not bool(torch.isfinite(evidence_override).all().item()):
                raise ValueError("evidence_override contains non-finite values")
            evidence = evidence_override.detach().float().to(
                condition_features.device
            )
    trace: list[dict[str, Any]] = []
    provisional_parts: list[torch.Tensor] = []
    committed_parts: list[torch.Tensor] = []
    remasked_parts: list[torch.Tensor] = []

    for iteration in range(steps):
        logits = (
            model(tokens, condition_features, condition_quality)
            if evidence_anchor is None
            else model(
                tokens,
                condition_features,
                condition_quality,
                evidence_anchor,
            )
        ).float()
        if mode in {"evidence-logits", "evidence-remask"}:
            assert evidence is not None
            selection_logits = logits + evidence_lambda * evidence
        else:
            selection_logits = logits
        probabilities = torch.softmax(selection_logits, dim=-1)
        confidence, candidates = probabilities.max(dim=-1)
        candidate_evidence = (
            torch.zeros_like(confidence)
            if evidence is None
            else evidence.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
        )
        if mode == "evidence-ordering":
            commit_score = (
                torch.log(confidence.clamp_min(1e-8))
                + evidence_lambda * candidate_evidence
            )
        else:
            commit_score = torch.log(confidence.clamp_min(1e-8))

        target_count = _target_commit_count(
            iteration=iteration,
            steps=steps,
            positions=positions,
        )
        remasked_count = 0
        remasked = torch.zeros_like(committed)
        if mode == "evidence-remask":
            selected_indices = torch.topk(
                commit_score,
                k=target_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            selected = torch.zeros_like(committed)
            selected.scatter_(1, selected_indices, True)
            remasked = committed & ~selected
            remasked_count = int(remasked.sum().item())
            tokens.fill_(model.mask_token_id)
            tokens[selected] = candidates[selected]
            commit_iteration[~selected] = -1
            committed_score[~selected] = float("nan")
            newly_selected = selected & ~committed
            commit_iteration[newly_selected] = iteration
            committed_score[selected] = commit_score[selected]
            committed = selected
        else:
            current_count = int(committed[0].sum().item())
            add_count = max(0, target_count - current_count)
            if add_count > 0:
                available_score = commit_score.masked_fill(
                    committed,
                    float("-inf"),
                )
                selected_indices = torch.topk(
                    available_score,
                    k=add_count,
                    dim=-1,
                    largest=True,
                    sorted=False,
                ).indices
                selected = torch.zeros_like(committed)
                selected.scatter_(1, selected_indices, True)
                tokens[selected] = candidates[selected]
                commit_iteration[selected] = iteration
                committed_score[selected] = commit_score[selected]
                committed |= selected

        provisional_parts.append(candidates.detach().clone())
        committed_parts.append(committed.detach().clone())
        remasked_parts.append(remasked.detach().clone())

        trace.append(
            {
                "iteration": iteration + 1,
                "target_committed": target_count,
                "actual_committed": int(committed.sum(dim=1).min().item()),
                "remasked_total_batch": remasked_count,
                "mean_confidence": float(confidence.mean().item()),
                "mean_candidate_evidence": float(
                    candidate_evidence.mean().item()
                ),
            }
        )

    if bool((tokens == model.mask_token_id).any().item()):
        raise RuntimeError("decoder finished with uncommitted mask tokens")
    return DecodeResult(
        tokens=tokens,
        commit_iteration=commit_iteration,
        commit_score=committed_score,
        trace=trace,
        provisional_tokens=tuple(provisional_parts),
        committed_after_iteration=tuple(committed_parts),
        remasked=tuple(remasked_parts),
    )
