"""Evaluation helpers for risk-controlled residual token refinement."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from .decoding import one_shot_decode
from .model import ConditionalTokenTransformer
from .research_evaluation import _evaluate_embeddings
from .residual_correction_gate import (
    ResidualCorrectionGate,
    risk_controlled_refine,
)
from .residual_evaluation import _map_fidelity
from .residual_evidence import quality_weighted_anchor
from .residual_quantization import reconstruct_from_residual_tokens
from .training import TokenTrainingDataset


def transition_statistics(
    *,
    initial_tokens: torch.Tensor,
    final_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    scenarios: list[str],
) -> dict[str, Any]:
    if not (
        initial_tokens.shape == final_tokens.shape == target_tokens.shape
    ):
        raise ValueError("transition token shapes differ")

    def summarize(mask: torch.Tensor | None = None) -> dict[str, float]:
        initial = initial_tokens if mask is None else initial_tokens[mask]
        final = final_tokens if mask is None else final_tokens[mask]
        target = target_tokens if mask is None else target_tokens[mask]
        before = initial == target
        after = final == target
        corrected = (~before) & after
        injured = before & (~after)
        changed = initial != final
        return {
            "initial_accuracy": float(before.float().mean().item()),
            "final_accuracy": float(after.float().mean().item()),
            "change_fraction": float(changed.float().mean().item()),
            "corrected_fraction": float(corrected.float().mean().item()),
            "injured_fraction": float(injured.float().mean().item()),
            "net_correction_fraction": float(
                (corrected.sum() - injured.sum()).item() / before.numel()
            ),
            "ecr": (
                float(corrected.sum().item() / (~before).sum().item())
                if bool((~before).any().item())
                else 0.0
            ),
            "eir": (
                float(injured.sum().item() / before.sum().item())
                if bool(before.any().item())
                else 0.0
            ),
            "replacement_precision": (
                float(corrected.sum().item() / changed.sum().item())
                if bool(changed.any().item())
                else 0.0
            ),
        }

    scenario_array = np.asarray(scenarios)
    return {
        "all": summarize(),
        "by_scenario": {
            scenario: summarize(
                torch.from_numpy(scenario_array == scenario)
            )
            for scenario in sorted(set(scenarios))
        },
    }


@torch.inference_mode()
def decode_risk_controlled_dataset(
    *,
    model: ConditionalTokenTransformer,
    gate: ResidualCorrectionGate,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    proposal_fraction: float,
    utility_threshold: float,
    rounds: int,
) -> dict[str, Any]:
    if dataset._precomputed is None:
        raise ValueError("risk-controlled evaluation requires precompute")
    batch_size = int(config["evaluation"]["batch_size"])
    initial_parts: list[torch.Tensor] = []
    final_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    trace_totals = [
        {"proposed": 0, "replaced": 0, "corrected": 0, "injured": 0}
        for _ in range(rounds)
    ]
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        condition = dataset._precomputed["condition_features"][
            start:end
        ].to(device)
        quality = dataset._precomputed["condition_quality"][start:end].to(
            device
        )
        targets = dataset.targets["tokens"][start:end].to(device)
        result = risk_controlled_refine(
            model=model,
            gate=gate,
            condition_features=condition,
            condition_quality=quality,
            residual_codebook=residual_codebook,
            config=config,
            proposal_fraction=proposal_fraction,
            utility_threshold=utility_threshold,
            rounds=rounds,
            target_tokens=targets,
        )
        initial_parts.append(result.initial_tokens.cpu())
        final_parts.append(result.tokens.cpu())
        target_parts.append(targets.cpu())
        for index, row in enumerate(result.trace):
            for key in trace_totals[index]:
                trace_totals[index][key] += int(row[key] or 0)
    initial = torch.cat(initial_parts)
    final = torch.cat(final_parts)
    targets = torch.cat(target_parts)
    scenarios = list(dataset._precomputed["scenarios"])
    return {
        "initial_tokens": initial,
        "tokens": final,
        "trace": trace_totals,
        "transitions": transition_statistics(
            initial_tokens=initial,
            final_tokens=final,
            target_tokens=targets,
            scenarios=scenarios,
        ),
    }


@torch.inference_mode()
def decode_one_shot_dataset(
    *,
    model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        condition = dataset._precomputed["condition_features"][
            start:end
        ].to(device)
        quality = dataset._precomputed["condition_quality"][start:end].to(
            device
        )
        parts.append(
            one_shot_decode(model, condition, quality).detach().cpu()
        )
    return torch.cat(parts)


@torch.inference_mode()
def evaluate_residual_token_predictions(
    *,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate supplied residual tokens with the unchanged clean gallery."""

    if str(dataset.targets.get("split")) != "val":
        raise ValueError("residual token prediction evaluation is val-only")
    condition = dataset._precomputed["condition_features"].to(device)
    quality = dataset._precomputed["condition_quality"].to(device)
    anchor = quality_weighted_anchor(condition, quality)
    teacher = dataset.targets["continuous_prototypes"].float().to(device)
    target_tokens = dataset.targets["tokens"].long().to(device)
    codebook = residual_codebook.float().to(device)
    maps = {
        "quality_anchor": anchor,
        "teacher_continuous": teacher,
        "teacher_residual_quantized": reconstruct_from_residual_tokens(
            anchor,
            target_tokens,
            codebook,
        ),
    }
    token_values: dict[str, torch.Tensor] = {}
    for name, tokens in predictions.items():
        if tokens.shape != dataset.targets["tokens"].shape:
            raise ValueError(f"prediction shape mismatch for {name}")
        token_values[name] = tokens.long().cpu()
        maps[name] = reconstruct_from_residual_tokens(
            anchor,
            tokens.long().to(device),
            codebook,
        )
    embeddings = {
        name: F.normalize(value.mean(dim=1), dim=-1).cpu()
        for name, value in maps.items()
    }
    identities = dataset._precomputed["target_identity"].cpu()
    scenarios = list(dataset._precomputed["scenarios"])
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identities[clean_indices]
    gallery_embeddings = embeddings["teacher_continuous"][clean_indices]
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    far_targets = [
        float(value) for value in config["evaluation"]["far_targets"]
    ]
    return {
        "split": "val",
        "samples": len(dataset),
        "gallery": "clean_disjoint_teacher_continuous",
        "test_dataset_constructed": False,
        "verification": {
            name: _evaluate_embeddings(
                embeddings=value,
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
                teacher.cpu(),
                scenarios,
            )
            for name, value in maps.items()
        },
        "exact_token_accuracy": {
            name: {
                "all": float(
                    (
                        tokens == dataset.targets["tokens"]
                    ).float().mean().item()
                ),
                "by_scenario": {
                    scenario: float(
                        (
                            tokens[
                                torch.from_numpy(
                                    np.asarray(scenarios) == scenario
                                )
                            ]
                            == dataset.targets["tokens"][
                                torch.from_numpy(
                                    np.asarray(scenarios) == scenario
                                )
                            ]
                        ).float().mean().item()
                    )
                    for scenario in sorted(set(scenarios))
                },
            }
            for name, tokens in token_values.items()
        },
    }


@torch.inference_mode()
def correction_permutation_check(
    *,
    model: ConditionalTokenTransformer,
    gate: ResidualCorrectionGate,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    proposal_fraction: float,
    utility_threshold: float,
    rounds: int,
) -> dict[str, Any]:
    batch = default_collate(
        [dataset[index] for index in range(min(16, len(dataset)))]
    )
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    first = risk_controlled_refine(
        model=model,
        gate=gate,
        condition_features=condition,
        condition_quality=quality,
        residual_codebook=residual_codebook,
        config=config,
        proposal_fraction=proposal_fraction,
        utility_threshold=utility_threshold,
        rounds=rounds,
    )
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    second = risk_controlled_refine(
        model=model,
        gate=gate,
        condition_features=condition[:, permutation],
        condition_quality=quality[:, permutation],
        residual_codebook=residual_codebook,
        config=config,
        proposal_fraction=proposal_fraction,
        utility_threshold=utility_threshold,
        rounds=rounds,
    )
    return {
        "permutation": permutation.cpu().tolist(),
        "initial_token_disagreement": float(
            (first.initial_tokens != second.initial_tokens).float().mean().item()
        ),
        "final_token_disagreement": float(
            (first.tokens != second.tokens).float().mean().item()
        ),
        "max_utility_difference": max(
            float((left - right).abs().max().item())
            for left, right in zip(first.utilities, second.utilities)
        ),
    }


@torch.inference_mode()
def measure_correction_latency(
    *,
    model: ConditionalTokenTransformer,
    gate: ResidualCorrectionGate,
    dataset: TokenTrainingDataset,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    proposal_fraction: float,
    utility_threshold: float,
    rounds: int,
    runs: int = 100,
) -> dict[str, Any]:
    batch = default_collate(
        [dataset[index] for index in range(min(16, len(dataset)))]
    )
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)

    def one_shot() -> None:
        one_shot_decode(model, condition, quality)

    def refined() -> None:
        risk_controlled_refine(
            model=model,
            gate=gate,
            condition_features=condition,
            condition_quality=quality,
            residual_codebook=residual_codebook,
            config=config,
            proposal_fraction=proposal_fraction,
            utility_threshold=utility_threshold,
            rounds=rounds,
        )

    result = {}
    for name, function in (("rollout_one_shot", one_shot), ("refined", refined)):
        for _ in range(10):
            function()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        values = []
        for _ in range(runs):
            started = time.perf_counter()
            function()
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)
        tensor = torch.tensor(values)
        result[name] = {
            "batch_size": int(condition.shape[0]),
            "runs": runs,
            "mean_batch_ms": float(tensor.mean().item()),
            "p50_batch_ms": float(tensor.median().item()),
            "mean_per_set_ms": float(
                tensor.mean().item() / condition.shape[0]
            ),
            "peak_reserved_mib": float(
                torch.cuda.max_memory_reserved(device) / 1024**2
            ),
        }
    return result
