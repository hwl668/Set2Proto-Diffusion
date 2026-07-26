"""Set-aware, risk-controlled reranking of frozen residual proposals."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConditionalTokenTransformer
from .residual_correction_gate import _two_level_evidence
from .residual_evidence import quality_weighted_anchor
from .residual_listwise_reranker import (
    ListwiseCandidateCache,
    RERANKER_FEATURE_NAMES,
    extract_listwise_features,
)
from .training import TokenTrainingDataset


class SetAwareResidualReranker(nn.Module):
    """Jointly score the top-k lattice and predict safe replacement positions."""

    def __init__(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        positions: int,
        top_k: int,
        hidden_dim: int,
        attention_heads: int,
        layers: int,
        feedforward_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            feature_mean.ndim != 1
            or feature_std.shape != feature_mean.shape
            or int(feature_mean.numel()) != len(RERANKER_FEATURE_NAMES)
        ):
            raise ValueError("invalid set-aware feature scaler")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.positions = int(positions)
        self.top_k = int(top_k)
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer(
            "feature_std",
            feature_std.float().clamp_min(1e-4).clone(),
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(len(RERANKER_FEATURE_NAMES), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rank_embedding = nn.Embedding(top_k, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, positions, hidden_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * feedforward_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.position_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("features must have shape [B,P,C,F]")
        batch, positions, candidates, _ = features.shape
        if positions != self.positions or candidates != self.top_k:
            raise ValueError("feature lattice differs from configured shape")
        standardized = (
            features.float() - self.feature_mean
        ) / self.feature_std
        candidate = self.feature_projection(standardized)
        ranks = torch.arange(candidates, device=features.device)
        candidate = candidate + self.rank_embedding(ranks)[None, None]
        position = candidate.mean(dim=2) + self.position_embedding
        context = self.position_encoder(position)
        global_context = context.mean(dim=1, keepdim=True).expand(
            batch,
            positions,
            -1,
        )
        candidate_context = torch.cat(
            (
                candidate,
                context.unsqueeze(2).expand(-1, -1, candidates, -1),
                global_context.unsqueeze(2).expand(
                    -1,
                    -1,
                    candidates,
                    -1,
                ),
            ),
            dim=-1,
        )
        scores = self.score_head(candidate_context).squeeze(-1)
        gate_logits = self.gate_head(
            torch.cat((context, global_context), dim=-1)
        ).squeeze(-1)
        return scores, gate_logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class SetAwareTrainingResult:
    reranker: SetAwareResidualReranker
    history: list[dict[str, float]]
    elapsed_seconds: float
    peak_reserved_mib: float


@dataclass(frozen=True)
class SetAwareDecodeResult:
    initial_tokens: torch.Tensor
    tokens: torch.Tensor
    candidate_tokens: torch.Tensor
    combined_scores: torch.Tensor
    gate_probabilities: torch.Tensor
    replacement_mask: torch.Tensor


def train_set_aware_reranker(
    *,
    cache: ListwiseCandidateCache,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    smoke: bool = False,
) -> SetAwareTrainingResult:
    section = config["p2_6"]["reranker"]
    steps = int(section["smoke_steps"] if smoke else section["formal_steps"])
    feature_mean = cache.features.float().mean(dim=(0, 1, 2))
    feature_std = cache.features.float().std(
        dim=(0, 1, 2),
        unbiased=False,
    ).clamp_min(1e-4)
    torch.manual_seed(int(seed))
    model = SetAwareResidualReranker(
        feature_mean=feature_mean,
        feature_std=feature_std,
        positions=int(cache.features.shape[1]),
        top_k=int(cache.features.shape[2]),
        hidden_dim=int(section["hidden_dim"]),
        attention_heads=int(section["attention_heads"]),
        layers=int(section["layers"]),
        feedforward_multiplier=int(section["feedforward_multiplier"]),
        dropout=float(section["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    batch_size = min(int(section["batch_size"]), len(cache.sample_indices))
    generator = torch.Generator().manual_seed(int(seed) + 1)
    codebook = residual_codebook.float().to(device)
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        selected = torch.randint(
            0,
            len(cache.sample_indices),
            (batch_size,),
            generator=generator,
        )
        source_indices = cache.sample_indices.index_select(0, selected)
        features = cache.features.index_select(0, selected).to(device)
        candidate_tokens = cache.candidate_tokens.index_select(
            0,
            selected,
        ).to(device)
        target_distribution = cache.target_distribution.index_select(
            0,
            selected,
        ).to(device).float()
        genuine = cache.genuine_gallery_embeddings.index_select(
            0,
            selected,
        ).to(device).float()
        impostors = cache.hard_impostor_embeddings.index_select(
            0,
            selected,
        ).to(device).float()
        condition = dataset._precomputed["condition_features"].index_select(
            0,
            source_indices,
        ).to(device)
        quality = dataset._precomputed["condition_quality"].index_select(
            0,
            source_indices,
        ).to(device)
        teacher = dataset.targets["continuous_prototypes"].index_select(
            0,
            source_indices,
        ).to(device).float()
        anchor = quality_weighted_anchor(condition, quality)
        candidate_maps = F.normalize(
            anchor.unsqueeze(2) + codebook[candidate_tokens],
            dim=-1,
        )
        scores, gate_logits = model(features)
        listwise_loss = -(
            target_distribution * F.log_softmax(scores, dim=-1)
        ).sum(dim=-1).mean()
        candidate_weights = F.softmax(scores, dim=-1)
        gate_probability = torch.sigmoid(gate_logits)
        gated_weights = candidate_weights * gate_probability.unsqueeze(-1)
        gated_weights[..., 0] += 1.0 - gate_probability
        soft_map = F.normalize(
            (gated_weights.unsqueeze(-1) * candidate_maps).sum(dim=2),
            dim=-1,
        )
        embedding = F.normalize(soft_map.mean(dim=1), dim=-1)
        genuine_similarity = (embedding * genuine).sum(dim=-1)
        impostor_similarity = torch.einsum(
            "bd,bhd->bh",
            embedding,
            impostors,
        ).max(dim=-1).values
        identity_loss = F.softplus(
            float(section["identity_margin"])
            + impostor_similarity
            - genuine_similarity
        ).mean()
        local_loss = (
            1.0 - F.cosine_similarity(soft_map, teacher, dim=-1)
        ).mean()
        best_probability = target_distribution.max(dim=-1).values
        top1_probability = target_distribution[..., 0]
        best_index = target_distribution.argmax(dim=-1)
        safe_target = (
            (best_index != 0)
            & (
                best_probability - top1_probability
                >= float(section["safe_replacement_probability_gain"])
            )
        ).float()
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits,
            safe_target,
        )
        expected_utility = (
            candidate_weights * target_distribution
        ).sum(dim=-1)
        expected_injury = (
            gate_probability
            * F.relu(top1_probability - expected_utility)
        ).mean()
        loss = (
            float(section["listwise_loss_weight"]) * listwise_loss
            + float(section["identity_margin_loss_weight"]) * identity_loss
            + float(section["local_map_loss_weight"]) * local_loss
            + float(section["replacement_gate_loss_weight"]) * gate_loss
            + float(section["expected_injury_loss_weight"])
            * expected_injury
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(section["max_grad_norm"]),
        )
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "listwise_loss": float(listwise_loss.detach().item()),
                "identity_margin_loss": float(identity_loss.detach().item()),
                "local_map_loss": float(local_loss.detach().item()),
                "gate_loss": float(gate_loss.detach().item()),
                "expected_injury_loss": float(expected_injury.detach().item()),
                "safe_replacement_fraction": float(
                    safe_target.mean().detach().item()
                ),
                "mean_gate_probability": float(
                    gate_probability.mean().detach().item()
                ),
                "genuine_similarity": float(
                    genuine_similarity.mean().detach().item()
                ),
                "strongest_impostor_similarity": float(
                    impostor_similarity.mean().detach().item()
                ),
                "gradient_norm": float(gradient_norm.item()),
            }
        )
    model.eval()
    peak = (
        float(torch.cuda.max_memory_reserved(device) / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return SetAwareTrainingResult(
        reranker=model,
        history=history,
        elapsed_seconds=time.perf_counter() - started,
        peak_reserved_mib=peak,
    )


@torch.inference_mode()
def set_aware_rerank_decode(
    *,
    model: ConditionalTokenTransformer,
    reranker: SetAwareResidualReranker,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    model_logit_weight: float,
    replacement_threshold: float,
    gate_threshold: float,
    replacement_budget_fraction: float,
) -> SetAwareDecodeResult:
    top_k = int(config["p2_6"]["top_k"])
    batch = int(condition_features.shape[0])
    positions = int(model.spatial_tokens)
    all_mask = torch.full(
        (batch, positions),
        model.mask_token_id,
        dtype=torch.int64,
        device=condition_features.device,
    )
    logits = model(all_mask, condition_features, condition_quality)
    candidates = torch.topk(
        logits,
        k=top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    initial = candidates[..., 0]
    anchor = quality_weighted_anchor(
        condition_features,
        condition_quality,
    )
    evidence, effective_frames = _two_level_evidence(
        condition=condition_features,
        quality=condition_quality,
        codebook=residual_codebook.to(condition_features.device),
        config=config,
    )
    features, _, _ = extract_listwise_features(
        proposal_logits=logits,
        candidate_tokens=candidates,
        evidence=evidence,
        condition_features=condition_features,
        quality_anchor=anchor,
        residual_codebook=residual_codebook,
        effective_frames=effective_frames,
    )
    reranker_scores, gate_logits = reranker(features)
    reranker_z = (
        reranker_scores - reranker_scores.mean(dim=-1, keepdim=True)
    ) / reranker_scores.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-4)
    candidate_log_probability = F.log_softmax(
        logits.float(),
        dim=-1,
    ).gather(-1, candidates)
    model_z = (
        candidate_log_probability
        - candidate_log_probability.mean(dim=-1, keepdim=True)
    ) / candidate_log_probability.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-4)
    combined = reranker_z + float(model_logit_weight) * model_z
    best_index = combined.argmax(dim=-1, keepdim=True)
    best_tokens = candidates.gather(-1, best_index).squeeze(-1)
    best_scores = combined.gather(-1, best_index).squeeze(-1)
    score_gain = best_scores - combined[..., 0]
    gate_probability = torch.sigmoid(gate_logits)
    eligible = (
        (best_tokens != initial)
        & (score_gain >= float(replacement_threshold))
        & (gate_probability >= float(gate_threshold))
    )
    budget = min(
        positions,
        max(0, int(round(positions * float(replacement_budget_fraction)))),
    )
    if budget == 0:
        replace = torch.zeros_like(eligible)
    else:
        priority = (score_gain * gate_probability).masked_fill(
            ~eligible,
            float("-inf"),
        )
        selected = torch.topk(
            priority,
            k=budget,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        replace = torch.zeros_like(eligible)
        replace.scatter_(1, selected, True)
        replace &= eligible
    tokens = torch.where(replace, best_tokens, initial)
    return SetAwareDecodeResult(
        initial_tokens=initial,
        tokens=tokens,
        candidate_tokens=candidates,
        combined_scores=combined,
        gate_probabilities=gate_probability,
        replacement_mask=replace,
    )
