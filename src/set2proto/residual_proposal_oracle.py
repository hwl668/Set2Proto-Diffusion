"""Oracle audit of top-k residual proposals from a frozen Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as F

from .model import ConditionalTokenTransformer
from .residual_correction_gate import _gather, _two_level_evidence


ProposalOracleMode = Literal["exact_token", "local_teacher"]


@dataclass(frozen=True)
class ProposalOracleDecodeResult:
    initial_tokens: torch.Tensor
    tokens: torch.Tensor
    trace: tuple[dict[str, Any], ...]


def _proposal_mask(priority: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("proposal fraction must be in (0,1]")
    positions = int(priority.shape[1])
    count = max(1, min(positions, int(math.ceil(positions * fraction))))
    indices = torch.topk(
        priority,
        k=count,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    selected = torch.zeros_like(priority, dtype=torch.bool)
    selected.scatter_(1, indices, True)
    return selected


@torch.inference_mode()
def proposal_oracle_decode(
    *,
    model: ConditionalTokenTransformer,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    target_tokens: torch.Tensor,
    teacher_map: torch.Tensor,
    quality_anchor: torch.Tensor,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    proposal_fraction: float,
    top_k: int,
    rounds: int,
    mode: ProposalOracleMode,
) -> ProposalOracleDecodeResult:
    """Decode with target-token or local-teacher access as an upper bound."""

    if mode not in {"exact_token", "local_teacher"}:
        raise ValueError("unsupported proposal oracle mode")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not 1 <= int(top_k) <= model.codebook_size:
        raise ValueError("top_k is outside the codebook")
    batch = int(condition_features.shape[0])
    positions = model.spatial_tokens
    if tuple(target_tokens.shape) != (batch, positions):
        raise ValueError("target token shape mismatch")
    if teacher_map.shape != quality_anchor.shape:
        raise ValueError("teacher/anchor shapes differ")
    device = condition_features.device
    codebook = residual_codebook.float().to(device)
    all_mask = torch.full(
        (batch, positions),
        model.mask_token_id,
        dtype=torch.int64,
        device=device,
    )
    current = model(
        all_mask,
        condition_features,
        condition_quality,
    ).argmax(dim=-1)
    initial = current.clone()
    evidence, _ = _two_level_evidence(
        condition=condition_features,
        quality=condition_quality,
        codebook=codebook,
        config=config,
    )
    evidence_mean = evidence.mean(dim=-1, keepdim=True)
    evidence_std = evidence.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-4)
    evidence_z = (evidence - evidence_mean) / evidence_std
    trace: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        before = current.clone()
        context_logits = model(
            current,
            condition_features,
            condition_quality,
        )
        context_log_probabilities = F.log_softmax(
            context_logits.float(),
            dim=-1,
        )
        priority = _gather(context_log_probabilities, current) + float(
            config["p2_4"]["proposal_evidence_weight"]
        ) * _gather(evidence_z, current)
        proposed = _proposal_mask(priority, proposal_fraction)
        state = current.clone()
        state[proposed] = model.mask_token_id
        proposal_logits = model(
            state,
            condition_features,
            condition_quality,
        )
        candidates = torch.topk(
            proposal_logits,
            k=int(top_k),
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        target_match = candidates == target_tokens.unsqueeze(-1)
        target_available = target_match.any(dim=-1)
        target_rank = torch.where(
            target_available,
            target_match.float().argmax(dim=-1) + 1,
            torch.zeros_like(target_tokens),
        )
        if mode == "exact_token":
            choose_target = proposed & target_available
            current = torch.where(choose_target, target_tokens, current)
        else:
            candidate_set = torch.cat(
                (current.unsqueeze(-1), candidates),
                dim=-1,
            )
            candidate_vectors = codebook[candidate_set]
            candidate_maps = F.normalize(
                quality_anchor.float().unsqueeze(2) + candidate_vectors,
                dim=-1,
            )
            local_cosine = (
                candidate_maps * F.normalize(
                    teacher_map.float(),
                    dim=-1,
                ).unsqueeze(2)
            ).sum(dim=-1)
            best = local_cosine.argmax(dim=-1, keepdim=True)
            local_choice = candidate_set.gather(-1, best).squeeze(-1)
            current = torch.where(proposed, local_choice, current)
        before_correct = before == target_tokens
        after_correct = current == target_tokens
        naive_top1 = candidates[..., 0]
        naive_after = torch.where(proposed, naive_top1, before)
        naive_correct = naive_after == target_tokens
        proposed_wrong = proposed & (~before_correct)
        correctable = proposed_wrong & target_available
        trace.append(
            {
                "round": round_index,
                "proposed": int(proposed.sum().item()),
                "proposed_wrong": int(proposed_wrong.sum().item()),
                "wrong_before": int((~before_correct).sum().item()),
                "target_available_for_proposed_wrong": int(
                    correctable.sum().item()
                ),
                "target_rank_counts": {
                    str(rank): int(
                        (
                            correctable
                            & (target_rank == rank)
                        ).sum().item()
                    )
                    for rank in range(1, int(top_k) + 1)
                },
                "oracle_corrected": int(
                    ((~before_correct) & after_correct).sum().item()
                ),
                "oracle_injured": int(
                    (before_correct & (~after_correct)).sum().item()
                ),
                "oracle_changed": int((before != current).sum().item()),
                "naive_top1_corrected": int(
                    ((~before_correct) & naive_correct).sum().item()
                ),
                "naive_top1_injured": int(
                    (before_correct & (~naive_correct)).sum().item()
                ),
            }
        )
    return ProposalOracleDecodeResult(
        initial_tokens=initial,
        tokens=current,
        trace=tuple(trace),
    )
