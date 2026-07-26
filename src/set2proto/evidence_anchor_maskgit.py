"""Evidence-anchored residual-token dataset, model, training and evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate

from .decoding import maskgit_decode, one_shot_decode
from .evidence_anchor_quantization import infer_anchor_maps
from .environment import config_sha256
from .model import ConditionalTokenTransformer
from .research_evaluation import _evaluate_embeddings
from .residual_evaluation import _map_fidelity
from .residual_evidence import compute_two_level_residual_evidence
from .residual_quantization import (
    encode_teacher_residuals,
    reconstruct_from_residual_tokens,
)
from .scalar_evidence_router import BoundedScalarEvidenceRouter
from .training import (
    TokenTrainingDataset,
    build_corrupted_tokens,
    masked_cross_entropy,
)


def canonicalize_condition_set(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put an unordered frame set into a deterministic content-based order.

    Set attention is permutation invariant in exact arithmetic, but BF16
    reduction order can perturb nearly tied token logits.  Sorting frames by
    a fixed, parameter-free content signature makes the reduction order
    identical for every joint permutation.  No identity or target information
    is used and gradients through the frozen condition features are not
    required.
    """

    if condition_features.ndim != 4:
        raise ValueError("condition_features must have shape [B,M,P,D]")
    batch, frames, _, dimension = condition_features.shape
    if tuple(condition_quality.shape) != (batch, frames):
        raise ValueError("condition_quality must have shape [B,M]")
    weights = torch.sin(
        torch.arange(
            1,
            dimension + 1,
            device=condition_features.device,
            dtype=torch.float32,
        )
        * 1.61803398875
    )
    signature = torch.einsum(
        "bmd,d->bm",
        condition_features.float().mean(dim=2),
        weights,
    )
    signature = signature + 0.03125 * torch.log(
        condition_quality.float().clamp_min(1e-6)
    )
    order = torch.argsort(signature, dim=1, stable=True)
    feature_index = order[:, :, None, None].expand_as(condition_features)
    return (
        condition_features.gather(1, feature_index),
        condition_quality.gather(1, order),
    )


class EvidenceResidualDataset:
    """Read-only target override with a frozen per-sample evidence anchor."""

    def __init__(
        self,
        *,
        base: TokenTrainingDataset,
        evidence_anchors: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> None:
        if base._precomputed is None:
            raise ValueError("base dataset must be precomputed")
        expected_map = tuple(base.targets["continuous_prototypes"].shape)
        if tuple(evidence_anchors.shape) != expected_map:
            raise ValueError("evidence anchors have invalid shape")
        if tuple(target_tokens.shape) != expected_map[:-1]:
            raise ValueError("residual targets have invalid shape")
        self.base = base
        self.evidence_anchors = evidence_anchors.cpu().half()
        self.target_tokens = target_tokens.cpu().long()
        self.synthetic = base.synthetic
        self._precomputed = base._precomputed
        self.targets = dict(base.targets)
        self.targets["tokens"] = self.target_tokens

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.base[index])
        row["target_tokens"] = self.target_tokens[index]
        row["evidence_anchor"] = self.evidence_anchors[index]
        return row


@torch.inference_mode()
def build_evidence_residual_dataset(
    *,
    base: TokenTrainingDataset,
    router: BoundedScalarEvidenceRouter,
    residual_codebook: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> EvidenceResidualDataset:
    inference = infer_anchor_maps(
        dataset=base,
        router=router,
        device=device,
        batch_size=batch_size,
    )
    tokens, _ = encode_teacher_residuals(
        inference.teacher.to(device),
        inference.evidence_anchor.to(device),
        residual_codebook.to(device),
        chunk_size=8192,
    )
    return EvidenceResidualDataset(
        base=base,
        evidence_anchors=inference.evidence_anchor,
        target_tokens=tokens.cpu(),
    )


class EvidenceAnchoredTokenTransformer(ConditionalTokenTransformer):
    """Conditional Transformer with explicit frozen-anchor context tokens."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.anchor_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.anchor_position = nn.Parameter(
            torch.empty(1, self.spatial_tokens, self.hidden_dim)
        )
        self.anchor_type = nn.Parameter(
            torch.empty(1, 1, self.hidden_dim)
        )
        nn.init.xavier_uniform_(self.anchor_projection.weight)
        nn.init.zeros_(self.anchor_projection.bias)
        nn.init.trunc_normal_(self.anchor_position, std=0.02)
        nn.init.trunc_normal_(self.anchor_type, std=0.02)

    def encode_condition(
        self,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
    ) -> torch.Tensor:
        condition_features, condition_quality = canonicalize_condition_set(
            condition_features,
            condition_quality,
        )
        return super().encode_condition(
            condition_features,
            condition_quality,
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
        evidence_anchor: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(evidence_anchor.shape) != (
            int(input_tokens.shape[0]),
            self.spatial_tokens,
            self.feature_dim,
        ):
            raise ValueError("evidence_anchor has invalid shape")
        context = self.encode_condition(
            condition_features,
            condition_quality,
        )
        anchor_context = (
            self.anchor_projection(evidence_anchor.float())
            + self.anchor_position
            + self.anchor_type
        )
        context = torch.cat((context, anchor_context), dim=1)
        target = self.token_embedding(input_tokens) + self.target_position
        for block in self.blocks:
            target = block(target, context)
        return self.output_head(self.output_norm(target))


def build_evidence_anchored_model(
    config: Mapping[str, Any],
) -> EvidenceAnchoredTokenTransformer:
    grid = config["data"]["feature_grid"]
    section = config["p3_1"]["model"]
    return EvidenceAnchoredTokenTransformer(
        feature_dim=int(grid["projected_dim"]),
        spatial_tokens=int(grid["height"]) * int(grid["width"]),
        codebook_size=int(config["p3_1"]["codebook_size"]),
        hidden_dim=int(section["hidden_dim"]),
        layers=int(section["layers"]),
        attention_heads=int(section["attention_heads"]),
        mlp_ratio=float(section["mlp_ratio"]),
        dropout=float(section["dropout"]),
    )


@dataclass(frozen=True)
class AnchoredTrainingResult:
    model: EvidenceAnchoredTokenTransformer
    mode: str
    steps: int
    history: list[dict[str, float]]
    checkpoint: Path
    elapsed_seconds: float
    peak_reserved_mib: float


def train_anchored_model(
    *,
    model: EvidenceAnchoredTokenTransformer,
    dataset: EvidenceResidualDataset,
    config: Mapping[str, Any],
    mode: str,
    device: torch.device,
    checkpoint_directory: Path,
    seed: int,
    smoke: bool,
) -> AnchoredTrainingResult:
    if mode not in {"one-shot", "maskgit"}:
        raise ValueError("unsupported training mode")
    section = config["p3_1"]["training"]
    steps = int(section["smoke_steps"] if smoke else section["formal_steps"])
    batch_size = int(section["batch_size"])
    accumulation = int(section["gradient_accumulation"])
    precision = str(section["precision"])
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed) + 101)
    torch.manual_seed(int(seed) + 202)
    model.to(device).train()
    condition_cache = dataset._precomputed["condition_features"].to(device)
    quality_cache = dataset._precomputed["condition_quality"].to(device)
    anchor_cache = dataset.evidence_anchors.to(device)
    token_cache = dataset.target_tokens.to(device)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_accuracy = 0.0
        for _ in range(accumulation):
            indices = torch.randint(
                0,
                len(dataset),
                (batch_size,),
                generator=generator,
            ).to(device)
            condition = condition_cache.index_select(0, indices)
            quality = quality_cache.index_select(0, indices)
            anchor = anchor_cache.index_select(0, indices)
            targets = token_cache.index_select(0, indices)
            corrupted, mask = build_corrupted_tokens(
                targets,
                mask_token_id=model.mask_token_id,
                mode=mode,
                min_mask_ratio=float(section["maskgit_min_mask_ratio"]),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and precision != "fp32",
            ):
                logits = model(corrupted, condition, quality, anchor)
                loss, accuracy = masked_cross_entropy(
                    logits,
                    targets,
                    mask,
                    label_smoothing=float(section["label_smoothing"]),
                )
            scaler.scale(loss / accumulation).backward()
            total_loss += float(loss.detach().item())
            total_accuracy += float(accuracy.detach().item())
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(section["max_grad_norm"]),
        )
        warmup = int(section["warmup_steps"])
        if step <= warmup:
            learning_rate = float(section["learning_rate"]) * step / max(
                1,
                warmup,
            )
        else:
            progress = (step - warmup) / max(1, steps - warmup)
            learning_rate = float(section["learning_rate"]) * 0.5 * (
                1.0 + np.cos(np.pi * progress)
            )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        scaler.step(optimizer)
        scaler.update()
        history.append(
            {
                "step": float(step),
                "loss": total_loss / accumulation,
                "accuracy": total_accuracy / accumulation,
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm.item()),
            }
        )
        if (
            step % int(section["checkpoint_every_steps"]) == 0
            or step == steps
        ):
            path = checkpoint_directory / f"{mode}_{step:08d}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "schema_version": 1,
                    "stage": "p3-1",
                    "mode": mode,
                    "step": step,
                    "model_state": model.state_dict(),
                    "parameter_count": model.parameter_count(),
                    "config_sha256": config_sha256(config),
                    "teacher_used_at_inference": False,
                },
                temporary,
            )
            os.replace(temporary, path)
    checkpoint = checkpoint_directory / f"{mode}_{steps:08d}.pt"
    return AnchoredTrainingResult(
        model=model.eval(),
        mode=mode,
        steps=steps,
        history=history,
        checkpoint=checkpoint,
        elapsed_seconds=time.perf_counter() - started,
        peak_reserved_mib=(
            float(torch.cuda.max_memory_reserved(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    )


@torch.inference_mode()
def evaluate_anchored_models(
    *,
    one_shot_model: EvidenceAnchoredTokenTransformer,
    maskgit_model: EvidenceAnchoredTokenTransformer,
    dataset: EvidenceResidualDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    evidence_mode: str,
    evidence_lambda: float,
) -> dict[str, Any]:
    section = config["p3_1"]["decoding"]
    batch_size = int(config["evaluation"]["batch_size"])
    maps: dict[str, list[torch.Tensor]] = {}
    tokens: dict[str, list[torch.Tensor]] = {}
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    codebook = residual_codebook.float().to(device)

    def append(name: str, token: torch.Tensor, anchor: torch.Tensor) -> None:
        tokens.setdefault(name, []).append(token.cpu())
        maps.setdefault(name, []).append(
            reconstruct_from_residual_tokens(
                anchor,
                token,
                codebook,
            ).cpu().half()
        )

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = default_collate([dataset[i] for i in range(start, end)])
        condition = batch["condition_features"].to(device)
        quality = batch["condition_quality"].to(device)
        anchor = batch["evidence_anchor"].to(device)
        target = batch["target_tokens"].to(device)
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(list(batch["scenario"]))
        maps.setdefault("evidence_anchor", []).append(anchor.cpu().half())
        maps.setdefault("teacher_continuous", []).append(
            dataset.targets["continuous_prototypes"][start:end].cpu().half()
        )
        append("teacher_residual_quantized", target, anchor)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            one_shot = one_shot_decode(
                one_shot_model,
                condition,
                quality,
                anchor,
            )
        append("one_shot", one_shot, anchor)
        confidence_by_step: dict[int, torch.Tensor] = {}
        for steps in section["step_candidates"]:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                decoded = maskgit_decode(
                    maskgit_model,
                    condition,
                    quality,
                    codebook=codebook,
                    steps=int(steps),
                    mode="confidence",
                    top_k_frames=int(section["top_k_frames"]),
                    evidence_lambda=0.0,
                    evidence_anchor=anchor,
                )
            confidence_by_step[int(steps)] = decoded.tokens
            append(f"maskgit_confidence_{int(steps)}step", decoded.tokens, anchor)
        canonical_condition, canonical_quality = canonicalize_condition_set(
            condition,
            quality,
        )
        evidence = compute_two_level_residual_evidence(
            canonical_condition,
            canonical_quality,
            anchor,
            codebook,
        ).evidence
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            guided = maskgit_decode(
                maskgit_model,
                condition,
                quality,
                codebook=codebook,
                steps=4,
                mode=evidence_mode,
                top_k_frames=int(section["top_k_frames"]),
                evidence_lambda=float(evidence_lambda),
                evidence_override=evidence,
                evidence_anchor=anchor,
            )
        append("maskgit_evidence_guided_4step", guided.tokens, anchor)
    all_maps = {
        name: torch.cat(parts).float() for name, parts in maps.items()
    }
    all_tokens = {name: torch.cat(parts) for name, parts in tokens.items()}
    identity = torch.cat(identities)
    embeddings = {
        name: F.normalize(value.mean(dim=1), dim=-1)
        for name, value in all_maps.items()
    }
    clean = torch.tensor([value == "clean" for value in scenarios])
    gallery_ids = identity[clean]
    gallery = embeddings["teacher_continuous"][clean]
    order = torch.argsort(gallery_ids)
    gallery_ids = gallery_ids[order]
    gallery = gallery[order]
    far_targets = [
        float(value) for value in config["evaluation"]["far_targets"]
    ]
    return {
        "samples": len(dataset),
        "verification": {
            name: _evaluate_embeddings(
                embeddings=value,
                gallery_embeddings=gallery,
                identities=identity,
                gallery_identities=gallery_ids,
                scenarios=scenarios,
                far_targets=far_targets,
            )
            for name, value in embeddings.items()
        },
        "map_fidelity": {
            name: _map_fidelity(
                value,
                all_maps["teacher_continuous"],
                scenarios,
            )
            for name, value in all_maps.items()
        },
        "exact_token_accuracy": {
            name: float(
                (value == dataset.target_tokens).float().mean().item()
            )
            for name, value in all_tokens.items()
        },
        "evidence_mode": evidence_mode,
        "evidence_lambda": float(evidence_lambda),
    }
