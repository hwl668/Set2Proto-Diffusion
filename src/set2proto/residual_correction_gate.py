"""Risk-controlled token refinement for residual prototype decoding."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConditionalTokenTransformer
from .residual_evidence import (
    compute_two_level_residual_evidence,
    quality_weighted_anchor,
)
from .training import TokenTrainingDataset


CORRECTION_CLASSES = ("neutral", "correction", "injury")
GATE_FEATURE_NAMES = (
    "candidate_log_probability",
    "current_log_probability",
    "candidate_logit_margin",
    "current_logit_margin",
    "normalized_entropy",
    "candidate_evidence_z",
    "current_evidence_z",
    "evidence_delta_z",
    "candidate_equals_current",
    "candidate_current_code_cosine",
    "proposal_fraction",
    "round_fraction",
    "effective_frames_fraction",
)


class ResidualCorrectionGate(nn.Module):
    """Small calibrated classifier for neutral/correction/injury proposals."""

    def __init__(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 or feature_std.shape != feature_mean.shape:
            raise ValueError("feature scaler must be one-dimensional")
        if int(feature_mean.numel()) != len(GATE_FEATURE_NAMES):
            raise ValueError("unexpected correction-gate feature dimension")
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer(
            "feature_std",
            feature_std.float().clamp_min(1e-4).clone(),
        )
        dimension = int(feature_mean.numel())
        self.network = nn.Sequential(
            nn.Linear(dimension, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), len(CORRECTION_CLASSES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (
            features.float() - self.feature_mean
        ) / self.feature_std
        return self.network(standardized)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class CorrectionExampleSet:
    features: torch.Tensor
    labels: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CorrectionGateTrainingResult:
    gate: ResidualCorrectionGate
    history: list[dict[str, float]]
    elapsed_seconds: float
    peak_reserved_mib: float
    class_counts: list[int]
    class_weights: list[float]


@dataclass(frozen=True)
class RiskControlledDecodeResult:
    initial_tokens: torch.Tensor
    tokens: torch.Tensor
    trace: tuple[dict[str, Any], ...]
    replacement_masks: tuple[torch.Tensor, ...]
    utilities: tuple[torch.Tensor, ...]


def _gather(values: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return values.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def _token_margin(
    log_probabilities: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    top_values, top_indices = torch.topk(
        log_probabilities,
        k=2,
        dim=-1,
    )
    selected = _gather(log_probabilities, tokens)
    competitor = torch.where(
        top_indices[..., 0] == tokens,
        top_values[..., 1],
        top_values[..., 0],
    )
    return selected - competitor


@torch.no_grad()
def extract_correction_features(
    *,
    proposal_logits: torch.Tensor,
    evidence: torch.Tensor,
    current_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    residual_codebook: torch.Tensor,
    proposal_fraction: float,
    round_fraction: float,
    effective_frames: torch.Tensor,
    condition_frames: int,
) -> torch.Tensor:
    """Create stop-gradient per-position proposal features."""

    if proposal_logits.ndim != 3:
        raise ValueError("proposal_logits must have shape [B,P,K]")
    if evidence.shape != proposal_logits.shape:
        raise ValueError("evidence and proposal logits must have equal shape")
    if current_tokens.shape != candidate_tokens.shape:
        raise ValueError("current and candidate token shapes differ")
    if tuple(current_tokens.shape) != tuple(proposal_logits.shape[:2]):
        raise ValueError("token/proposal shapes are incompatible")
    batch = int(current_tokens.shape[0])
    if tuple(effective_frames.shape) != (batch,):
        raise ValueError("effective_frames must have shape [B]")
    if not 0.0 < float(proposal_fraction) <= 1.0:
        raise ValueError("proposal_fraction must be in (0,1]")
    if not 0.0 <= float(round_fraction) <= 1.0:
        raise ValueError("round_fraction must be in [0,1]")

    log_probabilities = F.log_softmax(proposal_logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(
        probabilities * log_probabilities
    ).sum(dim=-1) / math.log(float(proposal_logits.shape[-1]))
    candidate_log_probability = _gather(
        log_probabilities,
        candidate_tokens,
    )
    current_log_probability = _gather(log_probabilities, current_tokens)
    candidate_margin = _token_margin(log_probabilities, candidate_tokens)
    current_margin = _token_margin(log_probabilities, current_tokens)

    evidence_float = evidence.float()
    evidence_mean = evidence_float.mean(dim=-1, keepdim=True)
    evidence_std = evidence_float.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-4)
    evidence_z = (evidence_float - evidence_mean) / evidence_std
    candidate_evidence = _gather(evidence_z, candidate_tokens)
    current_evidence = _gather(evidence_z, current_tokens)
    codebook = residual_codebook.float().to(proposal_logits.device)
    candidate_vectors = F.normalize(codebook[candidate_tokens], dim=-1)
    current_vectors = F.normalize(codebook[current_tokens], dim=-1)
    code_cosine = (candidate_vectors * current_vectors).sum(dim=-1)
    same = (candidate_tokens == current_tokens).float()
    positions = int(current_tokens.shape[1])
    proposal_value = torch.full(
        (batch, positions),
        float(proposal_fraction),
        device=proposal_logits.device,
    )
    round_value = torch.full(
        (batch, positions),
        float(round_fraction),
        device=proposal_logits.device,
    )
    effective_value = (
        effective_frames.float() / float(condition_frames)
    )[:, None].expand(batch, positions)
    features = torch.stack(
        (
            candidate_log_probability,
            current_log_probability,
            candidate_margin,
            current_margin,
            entropy,
            candidate_evidence,
            current_evidence,
            candidate_evidence - current_evidence,
            same,
            code_cosine,
            proposal_value,
            round_value,
            effective_value,
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("correction-gate features contain non-finite values")
    return features.detach()


def proposal_labels(
    current_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return 0 neutral, 1 correction, 2 injury."""

    if (
        current_tokens.shape != candidate_tokens.shape
        or current_tokens.shape != target_tokens.shape
    ):
        raise ValueError("proposal label token shapes differ")
    current_correct = current_tokens == target_tokens
    candidate_correct = candidate_tokens == target_tokens
    labels = torch.zeros_like(target_tokens)
    labels[(~current_correct) & candidate_correct] = 1
    labels[current_correct & (~candidate_correct)] = 2
    return labels


def _two_level_evidence(
    *,
    condition: torch.Tensor,
    quality: torch.Tensor,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor = quality_weighted_anchor(condition, quality)
    section = config["p2_2"]["evidence"]
    result = compute_two_level_residual_evidence(
        condition,
        quality,
        anchor,
        codebook,
        identity_neighbor_fraction=float(
            section["identity_neighbor_fraction"]
        ),
        identity_temperature=float(section["identity_temperature"]),
        quality_weight=float(section["quality_weight"]),
        local_temperature=float(section["local_temperature"]),
        reliability_floor=float(section["reliability_floor"]),
    )
    return result.evidence, result.effective_frames


def _select_proposal_mask(
    *,
    priority: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
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
def generate_correction_examples(
    *,
    model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    smoke: bool = False,
) -> CorrectionExampleSet:
    """Generate labels from frozen-model proposals on route-train only."""

    if dataset._precomputed is None:
        raise ValueError("correction example generation requires precompute")
    model.eval()
    codebook = residual_codebook.float().to(device)
    section = config["p2_3"]["gate"]
    batch_size = int(section["feature_batch_size"])
    chosen = list(indices[:32] if smoke else indices)
    fractions = [
        float(value) for value in section["training_mask_fractions"]
    ]
    views = int(section["views_per_fraction"])
    random_views = int(round(views * float(section["random_view_fraction"])))
    random_views = max(0, min(views, random_views))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    view_counts = {"low_risk": 0, "random": 0}
    for start in range(0, len(chosen), batch_size):
        batch_indices = torch.tensor(
            chosen[start : start + batch_size],
            dtype=torch.int64,
        )
        condition = dataset._precomputed["condition_features"].index_select(
            0,
            batch_indices,
        ).to(device)
        quality = dataset._precomputed["condition_quality"].index_select(
            0,
            batch_indices,
        ).to(device)
        targets = dataset.targets["tokens"].index_select(
            0,
            batch_indices,
        ).to(device)
        all_mask = torch.full_like(targets, model.mask_token_id)
        one_shot_logits = model(all_mask, condition, quality)
        current = one_shot_logits.argmax(dim=-1)
        evidence, effective_frames = _two_level_evidence(
            condition=condition,
            quality=quality,
            codebook=codebook,
            config=config,
        )
        for fraction in fractions:
            view_current = current.clone()
            for view in range(views):
                context_logits = model(view_current, condition, quality)
                context_log_probabilities = F.log_softmax(
                    context_logits.float(),
                    dim=-1,
                )
                current_probability = _gather(
                    context_log_probabilities,
                    view_current,
                )
                evidence_mean = evidence.mean(dim=-1, keepdim=True)
                evidence_std = evidence.std(
                    dim=-1,
                    keepdim=True,
                    unbiased=False,
                ).clamp_min(1e-4)
                current_evidence = _gather(
                    (evidence - evidence_mean) / evidence_std,
                    view_current,
                )
                risk_priority = current_probability + float(
                    section["proposal_evidence_weight"]
                ) * current_evidence
                if view < random_views:
                    priority = torch.rand(
                        current.shape,
                        device=device,
                        generator=generator,
                    )
                    view_counts["random"] += int(current.shape[0])
                else:
                    priority = risk_priority
                    view_counts["low_risk"] += int(current.shape[0])
                selected = _select_proposal_mask(
                    priority=priority,
                    fraction=fraction,
                )
                state = view_current.clone()
                state[selected] = model.mask_token_id
                proposal_logits = model(state, condition, quality)
                candidates = proposal_logits.argmax(dim=-1)
                features = extract_correction_features(
                    proposal_logits=proposal_logits,
                    evidence=evidence,
                    current_tokens=view_current,
                    candidate_tokens=candidates,
                    residual_codebook=codebook,
                    proposal_fraction=fraction,
                    round_fraction=float(view + 1) / float(views),
                    effective_frames=effective_frames,
                    condition_frames=int(condition.shape[1]),
                )
                labels = proposal_labels(view_current, candidates, targets)
                feature_parts.append(features[selected].cpu())
                label_parts.append(labels[selected].cpu())
                # Expose later views to the frozen model's own unfiltered
                # committed states. The learned gate is not used here.
                view_current = torch.where(
                    selected,
                    candidates,
                    view_current,
                )
    features = torch.cat(feature_parts).float()
    labels = torch.cat(label_parts).long()
    counts = torch.bincount(labels, minlength=len(CORRECTION_CLASSES))
    return CorrectionExampleSet(
        features=features,
        labels=labels,
        diagnostics={
            "samples": len(chosen),
            "examples": int(labels.numel()),
            "class_counts": counts.tolist(),
            "class_fractions": (
                counts.float() / counts.sum().clamp_min(1)
            ).tolist(),
            "views": view_counts,
            "feature_names": list(GATE_FEATURE_NAMES),
            "transformer_trainable": False,
            "split": "route_train",
        },
    )


def train_correction_gate(
    *,
    examples: CorrectionExampleSet,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    smoke: bool = False,
) -> CorrectionGateTrainingResult:
    section = config["p2_3"]["gate"]
    steps = int(section["smoke_steps"] if smoke else section["formal_steps"])
    feature_mean = examples.features.mean(dim=0)
    feature_std = examples.features.std(dim=0, unbiased=False).clamp_min(1e-4)
    torch.manual_seed(int(seed))
    gate = ResidualCorrectionGate(
        feature_mean=feature_mean,
        feature_std=feature_std,
        hidden_dim=int(section["hidden_dim"]),
        dropout=float(section["dropout"]),
    ).to(device)
    counts = torch.bincount(
        examples.labels,
        minlength=len(CORRECTION_CLASSES),
    ).float()
    weights = torch.sqrt(counts.max() / counts.clamp_min(1.0))
    weights = (weights / weights.mean()).clamp(max=10.0).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    batch_size = min(int(section["batch_size"]), len(examples.labels))
    generator = torch.Generator().manual_seed(int(seed) + 1)
    history: list[dict[str, float]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    gate.train()
    for step in range(1, steps + 1):
        selected = torch.randint(
            0,
            len(examples.labels),
            (batch_size,),
            generator=generator,
        )
        features = examples.features.index_select(0, selected).to(device)
        labels = examples.labels.index_select(0, selected).to(device)
        logits = gate(features)
        loss = F.cross_entropy(logits, labels, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            gate.parameters(),
            float(section["max_grad_norm"]),
        )
        optimizer.step()
        predictions = logits.argmax(dim=-1)
        history.append(
            {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "accuracy": float(
                    (predictions == labels).float().mean().item()
                ),
                "gradient_norm": float(gradient_norm.item()),
            }
        )
    gate.eval()
    return CorrectionGateTrainingResult(
        gate=gate,
        history=history,
        elapsed_seconds=time.perf_counter() - started,
        peak_reserved_mib=float(
            torch.cuda.max_memory_reserved(device) / 1024**2
        ),
        class_counts=[int(value) for value in counts.tolist()],
        class_weights=[float(value) for value in weights.cpu().tolist()],
    )


@torch.inference_mode()
def risk_controlled_refine(
    *,
    model: ConditionalTokenTransformer,
    gate: ResidualCorrectionGate,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    proposal_fraction: float,
    utility_threshold: float,
    rounds: int,
    evidence_override: torch.Tensor | None = None,
    effective_frames_override: torch.Tensor | None = None,
    target_tokens: torch.Tensor | None = None,
) -> RiskControlledDecodeResult:
    """Refine one-shot tokens only when expected correction exceeds injury."""

    if rounds < 1:
        raise ValueError("rounds must be positive")
    model.eval()
    gate.eval()
    device = condition_features.device
    codebook = residual_codebook.float().to(device)
    batch = int(condition_features.shape[0])
    all_mask = torch.full(
        (batch, model.spatial_tokens),
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
    if evidence_override is None or effective_frames_override is None:
        evidence, effective_frames = _two_level_evidence(
            condition=condition_features,
            quality=condition_quality,
            codebook=codebook,
            config=config,
        )
    else:
        evidence = evidence_override.to(device).float()
        effective_frames = effective_frames_override.to(device).float()
    section = config["p2_3"]["gate"]
    replacement_masks: list[torch.Tensor] = []
    utilities: list[torch.Tensor] = []
    trace: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        context_logits = model(
            current,
            condition_features,
            condition_quality,
        )
        context_log_probabilities = F.log_softmax(
            context_logits.float(),
            dim=-1,
        )
        current_probability = _gather(context_log_probabilities, current)
        evidence_mean = evidence.mean(dim=-1, keepdim=True)
        evidence_std = evidence.std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-4)
        current_evidence = _gather(
            (evidence - evidence_mean) / evidence_std,
            current,
        )
        priority = current_probability + float(
            section["proposal_evidence_weight"]
        ) * current_evidence
        proposed = _select_proposal_mask(
            priority=priority,
            fraction=float(proposal_fraction),
        )
        state = current.clone()
        state[proposed] = model.mask_token_id
        proposal_logits = model(
            state,
            condition_features,
            condition_quality,
        )
        candidates = proposal_logits.argmax(dim=-1)
        features = extract_correction_features(
            proposal_logits=proposal_logits,
            evidence=evidence,
            current_tokens=current,
            candidate_tokens=candidates,
            residual_codebook=codebook,
            proposal_fraction=float(proposal_fraction),
            round_fraction=float(round_index) / float(rounds),
            effective_frames=effective_frames,
            condition_frames=int(condition_features.shape[1]),
        )
        probabilities = F.softmax(gate(features), dim=-1)
        utility = probabilities[..., 1] - probabilities[..., 2]
        replace = (
            proposed
            & (candidates != current)
            & (utility >= float(utility_threshold))
        )
        corrected = injured = None
        if target_tokens is not None:
            before = current == target_tokens
            after = torch.where(replace, candidates, current) == target_tokens
            corrected = int(((~before) & after).sum().item())
            injured = int((before & (~after)).sum().item())
        current = torch.where(replace, candidates, current)
        replacement_masks.append(replace.detach().clone())
        utilities.append(utility.detach().clone())
        trace.append(
            {
                "round": round_index,
                "proposed": int(proposed.sum().item()),
                "replaced": int(replace.sum().item()),
                "corrected": corrected,
                "injured": injured,
                "mean_utility_replaced": (
                    float(utility[replace].mean().item())
                    if bool(replace.any().item())
                    else None
                ),
            }
        )
    return RiskControlledDecodeResult(
        initial_tokens=initial,
        tokens=current,
        trace=tuple(trace),
        replacement_masks=tuple(replacement_masks),
        utilities=tuple(utilities),
    )
