"""Quantization diagnostics for a frozen evidence-routed prototype anchor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .continuous_residual_oracle import quality_anchor
from .research_evaluation import _evaluate_embeddings
from .residual_correction_evaluation import _map_fidelity
from .residual_quantization import (
    ResidualCodebook,
    compute_residual_quantization_metrics,
    encode_teacher_residuals,
    reconstruct_from_residual_tokens,
)
from .scalar_evidence_router import BoundedScalarEvidenceRouter
from .training import TokenTrainingDataset


@dataclass(frozen=True)
class AnchorInference:
    quality_anchor: torch.Tensor
    evidence_anchor: torch.Tensor
    teacher: torch.Tensor
    gates: torch.Tensor
    scenarios: list[str]
    identities: torch.Tensor


@torch.inference_mode()
def infer_anchor_maps(
    *,
    dataset: TokenTrainingDataset,
    router: BoundedScalarEvidenceRouter,
    device: torch.device,
    batch_size: int,
) -> AnchorInference:
    if dataset._precomputed is None:
        raise ValueError("anchor inference requires precomputed features")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    quality_parts: list[torch.Tensor] = []
    evidence_parts: list[torch.Tensor] = []
    gate_parts: list[torch.Tensor] = []
    router.eval()
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        condition = dataset._precomputed["condition_features"][
            start:end
        ].to(device)
        quality = dataset._precomputed["condition_quality"][start:end].to(
            device
        )
        quality_map, _ = quality_anchor(condition, quality)
        evidence_map, _, _, gate, _, returned_anchor = router(
            condition,
            quality,
        )
        if not torch.allclose(
            quality_map.float(),
            returned_anchor.float(),
            atol=2e-5,
            rtol=2e-5,
        ):
            raise RuntimeError("router quality anchor differs from baseline")
        quality_parts.append(quality_map.cpu().half())
        evidence_parts.append(evidence_map.cpu().half())
        gate_parts.append(gate.cpu().half())
    quality_maps = torch.cat(quality_parts)
    evidence_maps = torch.cat(evidence_parts)
    gates = torch.cat(gate_parts)
    expected = (
        len(dataset),
        int(dataset.targets["continuous_prototypes"].shape[1]),
        int(dataset.targets["continuous_prototypes"].shape[2]),
    )
    if tuple(quality_maps.shape) != expected:
        raise RuntimeError("quality anchor has invalid shape")
    if tuple(evidence_maps.shape) != expected:
        raise RuntimeError("evidence anchor has invalid shape")
    norms = evidence_maps.float().norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=2e-3,
        rtol=2e-3,
    ):
        raise RuntimeError("evidence anchor is not unit normalized")
    return AnchorInference(
        quality_anchor=quality_maps,
        evidence_anchor=evidence_maps,
        teacher=dataset.targets["continuous_prototypes"].cpu(),
        gates=gates,
        scenarios=list(dataset._precomputed["scenarios"]),
        identities=dataset._precomputed["target_identity"].cpu(),
    )


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.float().reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p05": float(torch.quantile(flat, 0.05).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
    }


def paired_residual_diagnostics(
    inference: AnchorInference,
) -> dict[str, Any]:
    teacher = F.normalize(inference.teacher.float(), dim=-1)
    quality = F.normalize(inference.quality_anchor.float(), dim=-1)
    evidence = F.normalize(inference.evidence_anchor.float(), dim=-1)
    quality_norm = (teacher - quality).norm(dim=-1)
    evidence_norm = (teacher - evidence).norm(dim=-1)
    quality_cosine = (teacher * quality).sum(dim=-1)
    evidence_cosine = (teacher * evidence).sum(dim=-1)

    def metrics(mask: torch.Tensor) -> dict[str, Any]:
        q_norm = quality_norm[mask]
        e_norm = evidence_norm[mask]
        q_cos = quality_cosine[mask]
        e_cos = evidence_cosine[mask]
        return {
            "quality_residual_norm": _summary(q_norm),
            "evidence_residual_norm": _summary(e_norm),
            "mean_residual_norm_reduction_fraction": float(
                (q_norm.mean() - e_norm.mean()).item()
                / q_norm.mean().clamp_min(1e-8).item()
            ),
            "mean_teacher_cosine_gain": float(
                (e_cos - q_cos).mean().item()
            ),
            "evidence_anchor_better_fraction": float(
                (e_cos > q_cos).float().mean().item()
            ),
        }

    all_mask = torch.ones(
        len(inference.scenarios),
        dtype=torch.bool,
    )
    scenario_array = np.asarray(inference.scenarios, dtype=object)
    return {
        "all": metrics(all_mask),
        "by_scenario": {
            scenario: metrics(
                torch.from_numpy(scenario_array == scenario)
            )
            for scenario in sorted(set(inference.scenarios))
        },
        "gate": {
            "all": _summary(inference.gates),
            "by_scenario": {
                scenario: _summary(
                    inference.gates[
                        torch.from_numpy(scenario_array == scenario)
                    ]
                )
                for scenario in sorted(set(inference.scenarios))
            },
        },
    }


@torch.inference_mode()
def evaluate_quantized_anchors(
    *,
    inference: AnchorInference,
    quality_codebook: ResidualCodebook,
    evidence_codebook: ResidualCodebook,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    section = config["p3_0"]
    chunk_size = int(section["assignment_chunk_size"])
    teacher = inference.teacher.float().to(device)
    quality_anchor_map = inference.quality_anchor.float().to(device)
    evidence_anchor_map = inference.evidence_anchor.float().to(device)
    quality_vectors = quality_codebook.vectors.float().to(device)
    evidence_vectors = evidence_codebook.vectors.float().to(device)
    quality_tokens, _ = encode_teacher_residuals(
        teacher,
        quality_anchor_map,
        quality_vectors,
        chunk_size=chunk_size,
    )
    evidence_cross_tokens, _ = encode_teacher_residuals(
        teacher,
        evidence_anchor_map,
        quality_vectors,
        chunk_size=chunk_size,
    )
    evidence_tokens, _ = encode_teacher_residuals(
        teacher,
        evidence_anchor_map,
        evidence_vectors,
        chunk_size=chunk_size,
    )
    maps = {
        "quality_anchor": quality_anchor_map,
        "evidence_anchor": evidence_anchor_map,
        "quality_residual_quantized": reconstruct_from_residual_tokens(
            quality_anchor_map,
            quality_tokens,
            quality_vectors,
        ),
        "evidence_anchor_quality_codebook": (
            reconstruct_from_residual_tokens(
                evidence_anchor_map,
                evidence_cross_tokens,
                quality_vectors,
            )
        ),
        "evidence_residual_quantized": reconstruct_from_residual_tokens(
            evidence_anchor_map,
            evidence_tokens,
            evidence_vectors,
        ),
        "teacher_continuous": F.normalize(teacher, dim=-1),
    }
    embeddings = {
        name: F.normalize(value.mean(dim=1), dim=-1).cpu()
        for name, value in maps.items()
    }
    scenarios = inference.scenarios
    identities = inference.identities.cpu()
    clean = torch.tensor(
        [scenario == "clean" for scenario in scenarios],
        dtype=torch.bool,
    )
    gallery_ids = identities[clean]
    gallery = embeddings["teacher_continuous"][clean]
    order = torch.argsort(gallery_ids)
    gallery_ids = gallery_ids[order]
    gallery = gallery[order]
    far_targets = [
        float(value) for value in config["evaluation"]["far_targets"]
    ]
    verification = {
        name: _evaluate_embeddings(
            embeddings=value,
            gallery_embeddings=gallery,
            identities=identities,
            gallery_identities=gallery_ids,
            scenarios=scenarios,
            far_targets=far_targets,
        )
        for name, value in embeddings.items()
    }
    map_fidelity = {
        name: _map_fidelity(value.cpu(), teacher.cpu(), scenarios)
        for name, value in maps.items()
    }
    quantization = {
        "quality_residual_quantized": (
            compute_residual_quantization_metrics(
                teacher_map=teacher,
                quality_anchor=quality_anchor_map,
                tokens=quality_tokens,
                codebook=quality_vectors,
                scenarios=scenarios,
            )
        ),
        "evidence_anchor_quality_codebook": (
            compute_residual_quantization_metrics(
                teacher_map=teacher,
                quality_anchor=evidence_anchor_map,
                tokens=evidence_cross_tokens,
                codebook=quality_vectors,
                scenarios=scenarios,
            )
        ),
        "evidence_residual_quantized": (
            compute_residual_quantization_metrics(
                teacher_map=teacher,
                quality_anchor=evidence_anchor_map,
                tokens=evidence_tokens,
                codebook=evidence_vectors,
                scenarios=scenarios,
            )
        ),
    }
    teacher_cosine = F.cosine_similarity(
        maps["teacher_continuous"],
        maps["evidence_residual_quantized"],
        dim=-1,
    )
    quality_cosine = F.cosine_similarity(
        maps["teacher_continuous"],
        maps["quality_residual_quantized"],
        dim=-1,
    )
    return {
        "samples": len(scenarios),
        "gallery": "clean_disjoint_teacher_continuous",
        "verification": verification,
        "map_fidelity": map_fidelity,
        "quantization": quantization,
        "paired_reconstruction": {
            "evidence_better_token_fraction": float(
                (teacher_cosine > quality_cosine).float().mean().item()
            ),
            "mean_token_cosine_delta": float(
                (teacher_cosine - quality_cosine).mean().item()
            ),
        },
    }


@torch.inference_mode()
def permutation_check(
    *,
    router: BoundedScalarEvidenceRouter,
    dataset: TokenTrainingDataset,
    device: torch.device,
) -> dict[str, float]:
    condition = dataset._precomputed["condition_features"][:16].to(device)
    quality = dataset._precomputed["condition_quality"][:16].to(device)
    first, _, _, first_gate, _, _ = router(condition, quality)
    order = torch.tensor([2, 0, 3, 1], device=device)
    second, _, _, second_gate, _, _ = router(
        condition[:, order],
        quality[:, order],
    )
    return {
        "maximum_map_difference": float(
            (first - second).abs().max().item()
        ),
        "maximum_gate_difference": float(
            (first_gate - second_gate).abs().max().item()
        ),
    }


@torch.inference_mode()
def measure_anchor_latency(
    *,
    router: BoundedScalarEvidenceRouter,
    dataset: TokenTrainingDataset,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    condition = dataset._precomputed["condition_features"][:16].to(device)
    quality = dataset._precomputed["condition_quality"][:16].to(device)

    def run() -> None:
        router(condition, quality)

    for _ in range(10):
        run()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    values = []
    for _ in range(100):
        started = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        values.append((time.perf_counter() - started) * 1000.0)
    tensor = torch.tensor(values)
    return {
        "batch_size": 16,
        "runs": 100,
        "mean_batch_ms": float(tensor.mean().item()),
        "mean_per_set_ms": float(tensor.mean().item() / 16.0),
        "p50_batch_ms": float(tensor.median().item()),
        "peak_reserved_mib": float(
            torch.cuda.max_memory_reserved(device) / 1024**2
        ),
    }
