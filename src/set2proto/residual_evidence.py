"""Anchor-aware evidence for Euclidean residual prototype tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .decoding import (
    FrameReliabilityResult,
    combine_two_level_candidate_support,
    compute_frame_reliability,
)


@dataclass(frozen=True)
class ResidualEvidenceResult:
    """Anchor-aware evidence plus global frame reliability diagnostics."""

    evidence: torch.Tensor
    frame_reliability: torch.Tensor
    identity_consensus: torch.Tensor
    identity_reliability: torch.Tensor
    quality_standardized: torch.Tensor
    mean_local_usage: torch.Tensor
    effective_frames: torch.Tensor


def quality_weighted_anchor(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
) -> torch.Tensor:
    """Return the existing quality-pooling baseline as a unit local map."""

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,M,P,D]")
    batch, frames, _, _ = condition_features.shape
    if tuple(condition_quality.shape) != (batch, frames):
        raise ValueError("condition_quality must have shape [B,M]")
    if not bool(torch.isfinite(condition_features).all().item()):
        raise ValueError("condition_features contains non-finite values")
    if not bool(torch.isfinite(condition_quality).all().item()):
        raise ValueError("condition_quality contains non-finite values")

    features = F.normalize(condition_features.float(), dim=-1)
    quality = condition_quality.float().clamp_min(1e-6)
    quality = quality / quality.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.normalize(
        (quality[:, :, None, None] * features).sum(dim=1),
        dim=-1,
    )


def residual_candidate_support(
    condition_features: torch.Tensor,
    quality_anchor: torch.Tensor,
    residual_codebook: torch.Tensor,
) -> torch.Tensor:
    """Cosine support for ``normalize(anchor + residual_code)`` candidates.

    The result has shape ``[B,M,P,K]``. The implementation uses dot-product
    identities and never materializes a ``[B,P,K,D]`` candidate tensor.
    """

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,M,P,D]")
    batch, _, positions, dimension = condition_features.shape
    if tuple(quality_anchor.shape) != (batch, positions, dimension):
        raise ValueError("quality_anchor must have shape [B,P,D]")
    if (
        residual_codebook.ndim != 2
        or int(residual_codebook.shape[1]) != dimension
    ):
        raise ValueError("residual_codebook must have shape [K,D]")

    with torch.no_grad():
        condition = F.normalize(condition_features.float(), dim=-1)
        anchor = F.normalize(quality_anchor.float(), dim=-1)
        codebook = residual_codebook.float().to(condition.device)

        condition_anchor = torch.einsum(
            "bmpd,bpd->bmp",
            condition,
            anchor,
        )
        condition_residual = torch.einsum(
            "bmpd,kd->bmpk",
            condition,
            codebook,
        )
        anchor_residual = torch.einsum(
            "bpd,kd->bpk",
            anchor,
            codebook,
        )
        candidate_norm = (
            1.0
            + codebook.square().sum(dim=-1)[None, None, :]
            + 2.0 * anchor_residual
        ).clamp_min(1e-12).sqrt()
        support = (
            condition_anchor.unsqueeze(-1) + condition_residual
        ) / candidate_norm.unsqueeze(1)
    return support.clamp(min=-1.0, max=1.0).detach()


def compute_local_residual_evidence(
    condition_features: torch.Tensor,
    quality_anchor: torch.Tensor,
    residual_codebook: torch.Tensor,
    *,
    top_k_frames: int,
) -> torch.Tensor:
    """Top-k local evidence for sample-dependent residual candidates."""

    frames = int(condition_features.shape[1])
    if not 1 <= int(top_k_frames) <= frames:
        raise ValueError("top_k_frames must be in [1, condition_frames]")
    support = residual_candidate_support(
        condition_features,
        quality_anchor,
        residual_codebook,
    )
    return torch.topk(
        support,
        k=int(top_k_frames),
        dim=1,
        largest=True,
        sorted=False,
    ).values.mean(dim=1).detach()


def compute_two_level_residual_evidence(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    quality_anchor: torch.Tensor,
    residual_codebook: torch.Tensor,
    *,
    identity_neighbor_fraction: float = 0.5,
    identity_temperature: float = 3.0,
    quality_weight: float = 0.25,
    local_temperature: float = 4.0,
    reliability_floor: float = 1e-4,
) -> ResidualEvidenceResult:
    """Global identity-inlier x anchor-aware local candidate evidence."""

    reliability: FrameReliabilityResult = compute_frame_reliability(
        condition_features,
        condition_quality,
        identity_neighbor_fraction=identity_neighbor_fraction,
        identity_temperature=identity_temperature,
        quality_weight=quality_weight,
        reliability_floor=reliability_floor,
    )
    support = residual_candidate_support(
        condition_features,
        quality_anchor,
        residual_codebook,
    )
    evidence, mean_local_usage = combine_two_level_candidate_support(
        support,
        reliability.frame_reliability,
        local_temperature=local_temperature,
        reliability_floor=reliability_floor,
    )
    return ResidualEvidenceResult(
        evidence=evidence,
        frame_reliability=reliability.frame_reliability,
        identity_consensus=reliability.identity_consensus,
        identity_reliability=reliability.identity_reliability,
        quality_standardized=reliability.quality_standardized,
        mean_local_usage=mean_local_usage,
        effective_frames=reliability.effective_frames,
    )
