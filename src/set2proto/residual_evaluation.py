"""Validation-only evaluation for anchor-relative residual token models."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from .decoding import maskgit_decode, one_shot_decode
from .model import ConditionalTokenTransformer
from .research_evaluation import _evaluate_embeddings
from .residual_evidence import (
    compute_local_residual_evidence,
    compute_two_level_residual_evidence,
    quality_weighted_anchor,
)
from .residual_quantization import reconstruct_from_residual_tokens
from .training import TokenTrainingDataset


def _map_fidelity(
    maps: torch.Tensor,
    teachers: torch.Tensor,
    scenarios: list[str],
) -> dict[str, Any]:
    local = F.cosine_similarity(
        F.normalize(maps.float(), dim=-1),
        F.normalize(teachers.float(), dim=-1),
        dim=-1,
    ).mean(dim=1)
    global_maps = F.normalize(maps.float().mean(dim=1), dim=-1)
    global_teachers = F.normalize(teachers.float().mean(dim=1), dim=-1)
    global_cosine = F.cosine_similarity(
        global_maps,
        global_teachers,
        dim=-1,
    )
    scenario_array = np.asarray(scenarios)

    def summarize(mask: torch.Tensor | None = None) -> dict[str, float]:
        selected_local = local if mask is None else local[mask]
        selected_global = global_cosine if mask is None else global_cosine[mask]
        return {
            "mean_local_token_cosine": float(selected_local.mean().item()),
            "mean_map_cosine": float(selected_global.mean().item()),
        }

    result: dict[str, Any] = summarize()
    result["by_scenario"] = {
        scenario: summarize(torch.from_numpy(scenario_array == scenario))
        for scenario in sorted(set(scenarios))
    }
    return result


@torch.inference_mode()
def evaluate_residual_quantization_oracle(
    *,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    absolute_codebook: torch.Tensor,
    absolute_target_tokens: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the representation gate before any Transformer training."""

    if str(dataset.targets.get("split")) != "val":
        raise ValueError("P2-1 representation gate is validation-only")
    if dataset._precomputed is None:
        dataset.precompute()
    assert dataset._precomputed is not None
    condition = dataset._precomputed["condition_features"].to(device)
    quality = dataset._precomputed["condition_quality"].to(device)
    anchors = quality_weighted_anchor(condition, quality)
    teachers = dataset.targets["continuous_prototypes"].to(device).float()
    residual_tokens = dataset.targets["tokens"].to(device)
    residual_maps = reconstruct_from_residual_tokens(
        anchors,
        residual_tokens,
        residual_codebook.to(device),
    )
    absolute_maps = F.normalize(
        absolute_codebook.float().to(device),
        dim=-1,
    )[absolute_target_tokens.to(device).long()]
    maps = {
        "quality_anchor": anchors,
        "teacher_absolute_quantized": absolute_maps,
        "teacher_residual_quantized": residual_maps,
        "teacher_continuous": teachers,
    }
    embeddings = {
        name: F.normalize(value.mean(dim=1), dim=-1)
        for name, value in maps.items()
    }
    identities = dataset._precomputed["target_identity"].cpu()
    scenarios = list(dataset._precomputed["scenarios"])
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identities[clean_indices]
    gallery_embeddings = embeddings["teacher_continuous"][
        clean_indices
    ].cpu()
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    far_targets = [
        float(value) for value in config["evaluation"]["far_targets"]
    ]
    return {
        "split": "val",
        "gallery": "clean_disjoint_teacher_continuous",
        "test_dataset_constructed": False,
        "verification": {
            name: _evaluate_embeddings(
                embeddings=value.cpu(),
                gallery_embeddings=gallery_embeddings,
                identities=identities,
                gallery_identities=gallery_identities,
                scenarios=scenarios,
                far_targets=far_targets,
            )
            for name, value in embeddings.items()
        },
        "map_fidelity": {
            name: _map_fidelity(
                value.cpu(),
                teachers.cpu(),
                scenarios,
            )
            for name, value in maps.items()
        },
    }


@torch.inference_mode()
def evaluate_residual_validation(
    *,
    one_shot_model: ConditionalTokenTransformer,
    maskgit_model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    absolute_codebook: torch.Tensor,
    absolute_target_tokens: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    condition_frames: int,
) -> dict[str, Any]:
    """Evaluate residual models with the unchanged clean teacher gallery."""

    if str(dataset.targets.get("split")) != "val":
        raise ValueError("P2-1 model evaluation is validation-only")
    if absolute_target_tokens.shape != dataset.targets["tokens"].shape:
        raise ValueError("absolute/residual target shapes do not match")
    one_shot_model.eval()
    maskgit_model.eval()
    residual_codebook = residual_codebook.float().to(device)
    absolute_codebook = F.normalize(
        absolute_codebook.float(),
        dim=-1,
    ).to(device)
    p2_evidence = config["p2_1"]["evidence"]
    top_k = min(int(p2_evidence["top_k_frames"]), condition_frames)
    evidence_lambda = float(p2_evidence["lambda"])
    batch_size = int(config["evaluation"]["batch_size"])
    steps_values = [
        int(value) for value in config["decoding"]["evaluation_steps"]
    ]

    map_chunks: dict[str, list[torch.Tensor]] = {}
    token_chunks: dict[str, list[torch.Tensor]] = {}
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    frame_reliability_chunks: list[torch.Tensor] = []
    local_usage_chunks: list[torch.Tensor] = []
    effective_frames_chunks: list[torch.Tensor] = []
    commit_trace: dict[str, list[dict[str, Any]]] = {}

    def append_map(name: str, value: torch.Tensor) -> None:
        map_chunks.setdefault(name, []).append(value.detach().cpu().half())

    def append_residual_tokens(
        name: str,
        value: torch.Tensor,
        anchor: torch.Tensor,
    ) -> None:
        token_chunks.setdefault(name, []).append(value.detach().cpu())
        append_map(
            name,
            reconstruct_from_residual_tokens(
                anchor,
                value,
                residual_codebook,
            ),
        )

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"][:, :condition_frames].to(
            device
        )
        quality = batch["condition_quality"][:, :condition_frames].to(device)
        anchor = quality_weighted_anchor(condition, quality)
        target_tokens = batch["target_tokens"].to(device)
        teacher = dataset.targets["continuous_prototypes"][indices].to(device)
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(list(batch["scenario"]))

        append_map("quality_anchor", anchor)
        append_map("teacher_continuous", teacher)
        append_map(
            "teacher_absolute_quantized",
            absolute_codebook[
                absolute_target_tokens[indices].to(device).long()
            ],
        )
        append_residual_tokens(
            "teacher_residual_quantized",
            target_tokens,
            anchor,
        )

        one_shot = one_shot_decode(one_shot_model, condition, quality)
        append_residual_tokens("one_shot_transformer", one_shot, anchor)

        for steps in steps_values:
            decoded = maskgit_decode(
                maskgit_model,
                condition,
                quality,
                codebook=residual_codebook,
                steps=steps,
                mode="confidence",
                top_k_frames=top_k,
                evidence_lambda=evidence_lambda,
            )
            name = f"maskgit_confidence_{steps}step"
            append_residual_tokens(name, decoded.tokens, anchor)
            commit_trace.setdefault(name, []).append(decoded.trace[-1])

        local_evidence = compute_local_residual_evidence(
            condition,
            anchor,
            residual_codebook,
            top_k_frames=top_k,
        )
        two_level = compute_two_level_residual_evidence(
            condition,
            quality,
            anchor,
            residual_codebook,
            identity_neighbor_fraction=float(
                p2_evidence["identity_neighbor_fraction"]
            ),
            identity_temperature=float(
                p2_evidence["identity_temperature"]
            ),
            quality_weight=float(p2_evidence["quality_weight"]),
            local_temperature=float(p2_evidence["local_temperature"]),
            reliability_floor=float(p2_evidence["reliability_floor"]),
        )
        frame_reliability_chunks.append(
            two_level.frame_reliability.detach().cpu()
        )
        local_usage_chunks.append(two_level.mean_local_usage.detach().cpu())
        effective_frames_chunks.append(
            two_level.effective_frames.detach().cpu()
        )
        for prefix, evidence_tensor, modes in (
            (
                "local",
                local_evidence,
                ("evidence-logits", "evidence-remask"),
            ),
            (
                "two_level",
                two_level.evidence,
                (
                    "evidence-ordering",
                    "evidence-logits",
                    "evidence-remask",
                ),
            ),
        ):
            for mode in modes:
                decoded = maskgit_decode(
                    maskgit_model,
                    condition,
                    quality,
                    codebook=residual_codebook,
                    steps=int(config["decoding"]["default_steps"]),
                    mode=mode,
                    top_k_frames=top_k,
                    evidence_lambda=evidence_lambda,
                    evidence_override=evidence_tensor,
                )
                suffix = mode.removeprefix("evidence-").replace("-", "_")
                name = f"maskgit_{prefix}_{suffix}"
                append_residual_tokens(name, decoded.tokens, anchor)
                commit_trace.setdefault(name, []).append(decoded.trace[-1])

    identity_tensor = torch.cat(identities)
    all_maps = {
        name: torch.cat(chunks).float()
        for name, chunks in map_chunks.items()
    }
    embeddings = {
        name: F.normalize(maps.mean(dim=1), dim=-1)
        for name, maps in all_maps.items()
    }
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identity_tensor[clean_indices]
    gallery_embeddings = embeddings["teacher_continuous"][clean_indices]
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    far_targets = [
        float(value) for value in config["evaluation"]["far_targets"]
    ]
    verification = {
        name: _evaluate_embeddings(
            embeddings=value,
            gallery_embeddings=gallery_embeddings,
            identities=identity_tensor,
            gallery_identities=gallery_identities,
            scenarios=scenarios,
            far_targets=far_targets,
        )
        for name, value in embeddings.items()
    }
    fidelity = {
        name: _map_fidelity(value, all_maps["teacher_continuous"], scenarios)
        for name, value in all_maps.items()
    }
    reference_tokens = dataset.targets["tokens"]
    scenario_array = np.asarray(scenarios)
    exact_token_accuracy: dict[str, Any] = {}
    for name, chunks in token_chunks.items():
        predicted = torch.cat(chunks)
        exact_token_accuracy[name] = {
            "all": float(
                (predicted == reference_tokens).float().mean().item()
            ),
            "by_scenario": {
                scenario: float(
                    (
                        predicted[
                            torch.from_numpy(scenario_array == scenario)
                        ]
                        == reference_tokens[
                            torch.from_numpy(scenario_array == scenario)
                        ]
                    ).float().mean().item()
                )
                for scenario in sorted(set(scenarios))
            },
        }

    frame_reliability = torch.cat(frame_reliability_chunks)
    local_usage = torch.cat(local_usage_chunks)
    effective_frames = torch.cat(effective_frames_chunks)
    frame_diagnostics: dict[str, Any] = {
        "all": {
            "mean_frame_weights": frame_reliability.mean(dim=0).tolist(),
            "mean_local_usage": local_usage.mean(dim=0).tolist(),
            "mean_effective_frames": float(effective_frames.mean().item()),
        },
        "by_scenario": {},
    }
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        frame_diagnostics["by_scenario"][scenario] = {
            "mean_frame_weights": frame_reliability[mask].mean(dim=0).tolist(),
            "mean_local_usage": local_usage[mask].mean(dim=0).tolist(),
            "mean_effective_frames": float(
                effective_frames[mask].mean().item()
            ),
        }
    wrong_weights: list[float] = []
    for index, row in enumerate(dataset.synthetic.rows):
        for frame_index in row.get("wrong_positions", []):
            wrong_weights.append(
                float(frame_reliability[index, int(frame_index)].item())
            )
    frame_diagnostics["wrong_identity_frame_mean_weight"] = (
        float(np.mean(wrong_weights)) if wrong_weights else None
    )
    return {
        "split": "val",
        "samples": len(dataset),
        "identities": len(clean_indices),
        "gallery": "clean_disjoint_teacher_continuous",
        "test_dataset_constructed": False,
        "verification": verification,
        "map_fidelity": fidelity,
        "exact_token_accuracy": exact_token_accuracy,
        "frame_diagnostics": frame_diagnostics,
        "commit_trace_final": commit_trace,
    }


@torch.inference_mode()
def measure_residual_inference_latency(
    *,
    one_shot_model: ConditionalTokenTransformer,
    maskgit_model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Measure complete cached-feature pipelines, including evidence."""

    if str(dataset.targets.get("split")) != "val":
        raise ValueError("latency dataset must be validation")
    batch = default_collate(
        [dataset[index] for index in range(min(batch_size, len(dataset)))]
    )
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    codebook = residual_codebook.float().to(device)
    evidence_config = config["p2_1"]["evidence"]
    top_k = int(evidence_config["top_k_frames"])
    evidence_lambda = float(evidence_config["lambda"])
    steps = int(config["decoding"]["default_steps"])

    def confidence() -> None:
        maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=steps,
            mode="confidence",
            top_k_frames=top_k,
            evidence_lambda=evidence_lambda,
        )

    def local_logits() -> None:
        anchor = quality_weighted_anchor(condition, quality)
        evidence = compute_local_residual_evidence(
            condition,
            anchor,
            codebook,
            top_k_frames=top_k,
        )
        maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=steps,
            mode="evidence-logits",
            top_k_frames=top_k,
            evidence_lambda=evidence_lambda,
            evidence_override=evidence,
        )

    def two_level_logits() -> None:
        anchor = quality_weighted_anchor(condition, quality)
        evidence = compute_two_level_residual_evidence(
            condition,
            quality,
            anchor,
            codebook,
            identity_neighbor_fraction=float(
                evidence_config["identity_neighbor_fraction"]
            ),
            identity_temperature=float(
                evidence_config["identity_temperature"]
            ),
            quality_weight=float(evidence_config["quality_weight"]),
            local_temperature=float(evidence_config["local_temperature"]),
            reliability_floor=float(evidence_config["reliability_floor"]),
        ).evidence
        maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=steps,
            mode="evidence-logits",
            top_k_frames=top_k,
            evidence_lambda=evidence_lambda,
            evidence_override=evidence,
        )

    def two_level_remask() -> None:
        anchor = quality_weighted_anchor(condition, quality)
        evidence = compute_two_level_residual_evidence(
            condition,
            quality,
            anchor,
            codebook,
            identity_neighbor_fraction=float(
                evidence_config["identity_neighbor_fraction"]
            ),
            identity_temperature=float(
                evidence_config["identity_temperature"]
            ),
            quality_weight=float(evidence_config["quality_weight"]),
            local_temperature=float(evidence_config["local_temperature"]),
            reliability_floor=float(evidence_config["reliability_floor"]),
        ).evidence
        maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=steps,
            mode="evidence-remask",
            top_k_frames=top_k,
            evidence_lambda=evidence_lambda,
            evidence_override=evidence,
        )

    methods = {
        "one_shot_transformer": lambda: one_shot_decode(
            one_shot_model,
            condition,
            quality,
        ),
        "maskgit_confidence_4step": confidence,
        "maskgit_local_logits": local_logits,
        "maskgit_two_level_logits": two_level_logits,
        "maskgit_two_level_remask": two_level_remask,
    }
    warmups = int(config["evaluation"]["latency_warmup_runs"])
    measurements = int(config["evaluation"]["latency_measurement_runs"])
    results: dict[str, Any] = {}
    for name, function in methods.items():
        for _ in range(warmups):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elapsed: list[float] = []
        for _ in range(measurements):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            function()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - started) * 1000.0)
        results[name] = {
            "batch_size": int(condition.shape[0]),
            "runs": measurements,
            "mean_batch_ms": float(np.mean(elapsed)),
            "p50_batch_ms": float(np.median(elapsed)),
            "mean_per_set_ms": float(
                np.mean(elapsed) / float(condition.shape[0])
            ),
            "peak_allocated_mib": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_mib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
    return results


@torch.inference_mode()
def residual_permutation_check(
    *,
    one_shot_model: ConditionalTokenTransformer,
    maskgit_model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Verify frame-set permutation invariance on one cached validation batch."""

    batch = default_collate([dataset[index] for index in range(min(16, len(dataset)))])
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    codebook = residual_codebook.float().to(device)
    top_k = int(config["p2_1"]["evidence"]["top_k_frames"])
    evidence_lambda = float(config["p2_1"]["evidence"]["lambda"])
    steps = int(config["decoding"]["default_steps"])

    one_a = one_shot_decode(one_shot_model, condition, quality)
    one_b = one_shot_decode(
        one_shot_model,
        condition[:, permutation],
        quality[:, permutation],
    )
    confidence_a = maskgit_decode(
        maskgit_model,
        condition,
        quality,
        codebook=codebook,
        steps=steps,
        mode="confidence",
        top_k_frames=top_k,
        evidence_lambda=evidence_lambda,
    ).tokens
    confidence_b = maskgit_decode(
        maskgit_model,
        condition[:, permutation],
        quality[:, permutation],
        codebook=codebook,
        steps=steps,
        mode="confidence",
        top_k_frames=top_k,
        evidence_lambda=evidence_lambda,
    ).tokens
    anchor_a = quality_weighted_anchor(condition, quality)
    anchor_b = quality_weighted_anchor(
        condition[:, permutation],
        quality[:, permutation],
    )
    evidence_config = config["p2_1"]["evidence"]
    evidence_a = compute_two_level_residual_evidence(
        condition,
        quality,
        anchor_a,
        codebook,
        identity_neighbor_fraction=float(
            evidence_config["identity_neighbor_fraction"]
        ),
        identity_temperature=float(evidence_config["identity_temperature"]),
        quality_weight=float(evidence_config["quality_weight"]),
        local_temperature=float(evidence_config["local_temperature"]),
        reliability_floor=float(evidence_config["reliability_floor"]),
    ).evidence
    evidence_b = compute_two_level_residual_evidence(
        condition[:, permutation],
        quality[:, permutation],
        anchor_b,
        codebook,
        identity_neighbor_fraction=float(
            evidence_config["identity_neighbor_fraction"]
        ),
        identity_temperature=float(evidence_config["identity_temperature"]),
        quality_weight=float(evidence_config["quality_weight"]),
        local_temperature=float(evidence_config["local_temperature"]),
        reliability_floor=float(evidence_config["reliability_floor"]),
    ).evidence
    return {
        "permutation": permutation.cpu().tolist(),
        "one_shot_token_disagreement": float(
            (one_a != one_b).float().mean().item()
        ),
        "maskgit_confidence_token_disagreement": float(
            (confidence_a != confidence_b).float().mean().item()
        ),
        "two_level_evidence_max_abs_difference": float(
            (evidence_a - evidence_b).abs().max().item()
        ),
        "anchor_max_abs_difference": float(
            (anchor_a - anchor_b).abs().max().item()
        ),
    }
