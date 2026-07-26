"""Identity-margin/listwise reranking of frozen residual top-k proposals."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConditionalTokenTransformer
from .residual_correction_gate import _gather, _two_level_evidence
from .residual_evidence import quality_weighted_anchor
from .training import TokenTrainingDataset


RERANKER_FEATURE_NAMES = (
    "model_log_probability",
    "model_log_probability_delta",
    "rank_fraction",
    "two_level_evidence_z",
    "evidence_delta_from_top1",
    "residual_norm",
    "candidate_top1_code_cosine",
    "candidate_anchor_cosine",
    "local_condition_top2_cosine",
    "local_condition_mean_cosine",
    "global_condition_top2_cosine",
    "global_condition_mean_cosine",
    "is_top1",
    "effective_frames_fraction",
)


class ResidualListwiseReranker(nn.Module):
    def __init__(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            feature_mean.ndim != 1
            or feature_std.shape != feature_mean.shape
            or int(feature_mean.numel()) != len(RERANKER_FEATURE_NAMES)
        ):
            raise ValueError("invalid reranker feature scaler")
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
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (
            features.float() - self.feature_mean
        ) / self.feature_std
        return self.network(standardized).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class ListwiseCandidateCache:
    sample_indices: torch.Tensor
    features: torch.Tensor
    candidate_tokens: torch.Tensor
    target_distribution: torch.Tensor
    genuine_gallery_embeddings: torch.Tensor
    hard_impostor_embeddings: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ListwiseTrainingResult:
    reranker: ResidualListwiseReranker
    history: list[dict[str, float]]
    elapsed_seconds: float
    peak_reserved_mib: float


@dataclass(frozen=True)
class ListwiseDecodeResult:
    initial_tokens: torch.Tensor
    tokens: torch.Tensor
    candidate_tokens: torch.Tensor
    combined_scores: torch.Tensor
    replacement_mask: torch.Tensor


@torch.no_grad()
def extract_listwise_features(
    *,
    proposal_logits: torch.Tensor,
    candidate_tokens: torch.Tensor,
    evidence: torch.Tensor,
    condition_features: torch.Tensor,
    quality_anchor: torch.Tensor,
    residual_codebook: torch.Tensor,
    effective_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate features, local maps, and candidate global embeddings."""

    if candidate_tokens.ndim != 3:
        raise ValueError("candidate_tokens must have shape [B,P,C]")
    batch, positions, candidates = candidate_tokens.shape
    if proposal_logits.shape[:2] != (batch, positions):
        raise ValueError("proposal/candidate shapes are incompatible")
    if evidence.shape != proposal_logits.shape:
        raise ValueError("evidence/proposal shapes differ")
    codebook = residual_codebook.float().to(proposal_logits.device)
    log_probabilities = F.log_softmax(proposal_logits.float(), dim=-1)
    candidate_log_probability = log_probabilities.gather(
        -1,
        candidate_tokens,
    )
    top1_log_probability = candidate_log_probability[..., :1]
    rank_fraction = torch.arange(
        candidates,
        device=proposal_logits.device,
        dtype=torch.float32,
    )[None, None, :] / float(max(1, candidates - 1))

    evidence_float = evidence.float()
    evidence_z = (
        evidence_float - evidence_float.mean(dim=-1, keepdim=True)
    ) / evidence_float.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-4)
    candidate_evidence = evidence_z.gather(-1, candidate_tokens)
    top1_evidence = candidate_evidence[..., :1]

    residual_vectors = codebook[candidate_tokens]
    residual_norm = residual_vectors.norm(dim=-1)
    normalized_residual = F.normalize(residual_vectors, dim=-1)
    top1_residual = normalized_residual[..., :1, :]
    residual_cosine = (
        normalized_residual * top1_residual
    ).sum(dim=-1)
    candidate_maps = F.normalize(
        quality_anchor.float().unsqueeze(2) + residual_vectors,
        dim=-1,
    )
    anchor_cosine = (
        candidate_maps * F.normalize(
            quality_anchor.float(),
            dim=-1,
        ).unsqueeze(2)
    ).sum(dim=-1)

    condition = F.normalize(condition_features.float(), dim=-1)
    local_support = torch.einsum(
        "bmpd,bpcd->bmpc",
        condition,
        candidate_maps,
    )
    local_top2 = torch.topk(
        local_support,
        k=min(2, int(condition.shape[1])),
        dim=1,
        largest=True,
        sorted=False,
    ).values.mean(dim=1)
    local_mean = local_support.mean(dim=1)

    current_maps = candidate_maps[..., 0, :]
    current_sum = current_maps.sum(dim=1)
    candidate_global = F.normalize(
        (
            current_sum[:, None, None, :]
            - current_maps[:, :, None, :]
            + candidate_maps
        )
        / float(positions),
        dim=-1,
    )
    frame_global = F.normalize(condition.mean(dim=2), dim=-1)
    global_support = torch.einsum(
        "bmd,bpcd->bmpc",
        frame_global,
        candidate_global,
    )
    global_top2 = torch.topk(
        global_support,
        k=min(2, int(frame_global.shape[1])),
        dim=1,
        largest=True,
        sorted=False,
    ).values.mean(dim=1)
    global_mean = global_support.mean(dim=1)
    is_top1 = torch.zeros(
        (batch, positions, candidates),
        device=proposal_logits.device,
    )
    is_top1[..., 0] = 1.0
    effective = (
        effective_frames.float() / float(condition.shape[1])
    )[:, None, None].expand(batch, positions, candidates)
    features = torch.stack(
        (
            candidate_log_probability,
            candidate_log_probability - top1_log_probability,
            rank_fraction.expand(batch, positions, candidates),
            candidate_evidence,
            candidate_evidence - top1_evidence,
            residual_norm,
            residual_cosine,
            anchor_cosine,
            local_top2,
            local_mean,
            global_top2,
            global_mean,
            is_top1,
            effective,
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("listwise features contain non-finite values")
    return (
        features.detach(),
        candidate_maps.detach(),
        candidate_global.detach(),
    )


def _clean_gallery(
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    scenarios = dataset._precomputed["scenarios"]
    identities = dataset._precomputed["target_identity"]
    clean = [
        int(index)
        for index in indices
        if scenarios[int(index)] == "clean"
    ]
    gallery_ids = identities[clean].long()
    gallery_maps = dataset.targets["continuous_prototypes"][clean].float()
    gallery_embeddings = F.normalize(gallery_maps.mean(dim=1), dim=-1)
    order = torch.argsort(gallery_ids)
    return gallery_ids[order], gallery_embeddings[order]


@torch.inference_mode()
def generate_listwise_candidate_cache(
    *,
    model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    smoke: bool = False,
) -> ListwiseCandidateCache:
    if dataset._precomputed is None:
        raise ValueError("listwise cache generation requires precompute")
    model.eval()
    section = config["p2_5"]
    top_k = int(section["top_k"])
    batch_size = int(section["feature_batch_size"])
    selected_indices = list(indices[:32] if smoke else indices)
    gallery_ids, gallery_embeddings = _clean_gallery(dataset, indices)
    gallery_ids_gpu = gallery_ids.to(device)
    gallery_gpu = gallery_embeddings.to(device)
    feature_parts: list[torch.Tensor] = []
    token_parts: list[torch.Tensor] = []
    target_distribution_parts: list[torch.Tensor] = []
    genuine_parts: list[torch.Tensor] = []
    impostor_parts: list[torch.Tensor] = []
    local_best_is_target = 0
    margin_best_is_target = 0
    target_in_candidates = 0
    total_positions = 0
    utility_config = section["utility"]
    for start in range(0, len(selected_indices), batch_size):
        batch_indices = torch.tensor(
            selected_indices[start : start + batch_size],
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
        teachers = dataset.targets["continuous_prototypes"].index_select(
            0,
            batch_indices,
        ).to(device)
        identities = dataset._precomputed["target_identity"].index_select(
            0,
            batch_indices,
        ).to(device)
        all_mask = torch.full_like(targets, model.mask_token_id)
        proposal_logits = model(all_mask, condition, quality)
        candidates = torch.topk(
            proposal_logits,
            k=top_k,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        anchor = quality_weighted_anchor(condition, quality)
        evidence, effective_frames = _two_level_evidence(
            condition=condition,
            quality=quality,
            codebook=residual_codebook.to(device),
            config=config,
        )
        features, candidate_maps, candidate_global = (
            extract_listwise_features(
                proposal_logits=proposal_logits,
                candidate_tokens=candidates,
                evidence=evidence,
                condition_features=condition,
                quality_anchor=anchor,
                residual_codebook=residual_codebook,
                effective_frames=effective_frames,
            )
        )
        gallery_matches = identities[:, None] == gallery_ids_gpu[None, :]
        if not bool(gallery_matches.any(dim=1).all().item()):
            raise RuntimeError("route-train clean gallery is incomplete")
        genuine_index = gallery_matches.float().argmax(dim=1)
        genuine = gallery_gpu[genuine_index]
        current_global = candidate_global[..., 0, :].mean(dim=1)
        current_global = F.normalize(current_global, dim=-1)
        gallery_similarity = current_global @ gallery_gpu.T
        gallery_similarity = gallery_similarity.masked_fill(
            gallery_matches,
            float("-inf"),
        )
        hard_indices = torch.topk(
            gallery_similarity,
            k=min(int(section["hard_impostors"]), len(gallery_ids) - 1),
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        hard_impostors = gallery_gpu[hard_indices]
        genuine_similarity = torch.einsum(
            "bpcd,bd->bpc",
            candidate_global,
            genuine,
        )
        impostor_similarity = torch.einsum(
            "bpcd,bhd->bpch",
            candidate_global,
            hard_impostors,
        ).max(dim=-1).values
        gallery_margin = genuine_similarity - impostor_similarity
        local_teacher = (
            candidate_maps
            * F.normalize(teachers.float(), dim=-1).unsqueeze(2)
        ).sum(dim=-1)
        exact = (
            candidates == targets.unsqueeze(-1)
        ).float()
        utility = (
            float(utility_config["local_teacher_weight"]) * local_teacher
            + float(utility_config["gallery_margin_weight"]) * gallery_margin
            + float(utility_config["exact_token_weight"]) * exact
        )
        target_distribution = F.softmax(
            utility / float(utility_config["temperature"]),
            dim=-1,
        )
        feature_parts.append(features.cpu().half())
        token_parts.append(candidates.cpu())
        target_distribution_parts.append(target_distribution.cpu().half())
        genuine_parts.append(genuine.cpu().half())
        impostor_parts.append(hard_impostors.cpu().half())
        local_best_is_target += int(
            exact.gather(
                -1,
                local_teacher.argmax(dim=-1, keepdim=True),
            ).sum().item()
        )
        margin_best_is_target += int(
            exact.gather(
                -1,
                gallery_margin.argmax(dim=-1, keepdim=True),
            ).sum().item()
        )
        target_in_candidates += int(exact.any(dim=-1).sum().item())
        total_positions += int(targets.numel())
    features = torch.cat(feature_parts)
    candidate_tokens = torch.cat(token_parts)
    target_distribution = torch.cat(target_distribution_parts)
    return ListwiseCandidateCache(
        sample_indices=torch.tensor(selected_indices, dtype=torch.int64),
        features=features,
        candidate_tokens=candidate_tokens,
        target_distribution=target_distribution,
        genuine_gallery_embeddings=torch.cat(genuine_parts),
        hard_impostor_embeddings=torch.cat(impostor_parts),
        diagnostics={
            "split": "route_train",
            "samples": len(selected_indices),
            "positions": total_positions,
            "top_k": top_k,
            "feature_names": list(RERANKER_FEATURE_NAMES),
            "target_in_top_k_fraction": (
                target_in_candidates / total_positions
            ),
            "local_best_is_target_fraction": (
                local_best_is_target / total_positions
            ),
            "margin_best_is_target_fraction": (
                margin_best_is_target / total_positions
            ),
            "transformer_trainable": False,
            "gallery_identities": int(len(gallery_ids)),
        },
    )


def train_listwise_reranker(
    *,
    cache: ListwiseCandidateCache,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    smoke: bool = False,
) -> ListwiseTrainingResult:
    section = config["p2_5"]["reranker"]
    steps = int(section["smoke_steps"] if smoke else section["formal_steps"])
    feature_mean = cache.features.float().mean(dim=(0, 1, 2))
    feature_std = cache.features.float().std(
        dim=(0, 1, 2),
        unbiased=False,
    ).clamp_min(1e-4)
    torch.manual_seed(int(seed))
    reranker = ResidualListwiseReranker(
        feature_mean=feature_mean,
        feature_std=feature_std,
        hidden_dim=int(section["hidden_dim"]),
        dropout=float(section["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        reranker.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    batch_size = min(int(section["batch_size"]), len(cache.sample_indices))
    generator = torch.Generator().manual_seed(int(seed) + 1)
    codebook = residual_codebook.float().to(device)
    history: list[dict[str, float]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    reranker.train()
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
        scores = reranker(features)
        listwise_loss = -(
            target_distribution * F.log_softmax(scores, dim=-1)
        ).sum(dim=-1).mean()
        soft_weights = F.softmax(scores, dim=-1)
        soft_map = (
            soft_weights.unsqueeze(-1) * candidate_maps
        ).sum(dim=2)
        soft_map = F.normalize(soft_map, dim=-1)
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
            1.0
            - F.cosine_similarity(soft_map, teacher, dim=-1)
        ).mean()
        loss = (
            float(section["listwise_loss_weight"]) * listwise_loss
            + float(section["identity_margin_loss_weight"]) * identity_loss
            + float(section["local_map_loss_weight"]) * local_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            reranker.parameters(),
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
                "genuine_similarity": float(
                    genuine_similarity.mean().detach().item()
                ),
                "strongest_impostor_similarity": float(
                    impostor_similarity.mean().detach().item()
                ),
                "gradient_norm": float(gradient_norm.item()),
            }
        )
    reranker.eval()
    return ListwiseTrainingResult(
        reranker=reranker,
        history=history,
        elapsed_seconds=time.perf_counter() - started,
        peak_reserved_mib=float(
            torch.cuda.max_memory_reserved(device) / 1024**2
        ),
    )


@torch.inference_mode()
def listwise_rerank_decode(
    *,
    model: ConditionalTokenTransformer,
    reranker: ResidualListwiseReranker,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    model_logit_weight: float,
    replacement_threshold: float,
) -> ListwiseDecodeResult:
    top_k = int(config["p2_5"]["top_k"])
    batch = int(condition_features.shape[0])
    all_mask = torch.full(
        (batch, model.spatial_tokens),
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
    reranker_score = reranker(features)
    reranker_z = (
        reranker_score - reranker_score.mean(dim=-1, keepdim=True)
    ) / reranker_score.std(
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
    current_scores = combined[..., 0]
    replace = (
        (best_tokens != initial)
        & (
            best_scores - current_scores
            >= float(replacement_threshold)
        )
    )
    tokens = torch.where(replace, best_tokens, initial)
    return ListwiseDecodeResult(
        initial_tokens=initial,
        tokens=tokens,
        candidate_tokens=candidates,
        combined_scores=combined,
        replacement_mask=replace,
    )
