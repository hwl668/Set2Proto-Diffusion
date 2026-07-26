"""Fast functional evaluation for newly trained one-shot and MaskGIT models."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.utils.data import default_collate

from .decoding import maskgit_decode, one_shot_decode
from .model import ConditionalTokenTransformer
from .training import TokenTrainingDataset


def _token_accuracy(tokens: torch.Tensor, targets: torch.Tensor) -> float:
    return float((tokens == targets).float().mean().item())


def _scenario_accuracy(
    tokens: torch.Tensor,
    targets: torch.Tensor,
    scenarios: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for scenario in sorted(set(scenarios)):
        indices = [
            index for index, value in enumerate(scenarios) if value == scenario
        ]
        result[scenario] = _token_accuracy(
            tokens[indices],
            targets[indices],
        )
    return result


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    if left.numel() < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    return float(torch.corrcoef(torch.stack([left_rank, right_rank]))[0, 1])


def _representative_indices(dataset: TokenTrainingDataset) -> list[int]:
    selected: dict[str, int] = {}
    for index, row in enumerate(dataset.synthetic.rows):
        selected.setdefault(str(row["scenario"]), index)
    return [selected[name] for name in sorted(selected)]


@torch.inference_mode()
def evaluate_training_smoke(
    *,
    one_shot_model: ConditionalTokenTransformer,
    maskgit_model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    condition_frames: int,
) -> dict[str, Any]:
    indices = _representative_indices(dataset)
    batch = default_collate([dataset[index] for index in indices])
    condition = batch["condition_features"][:, :condition_frames].to(device)
    quality = batch["condition_quality"][:, :condition_frames].to(device)
    visibility = batch["condition_visibility"][:, :condition_frames].to(device)
    targets = batch["target_tokens"].to(device)
    scenarios = list(batch["scenario"])
    codebook = codebook.to(device)
    one_shot_model.eval()
    maskgit_model.eval()

    one_shot_tokens = one_shot_decode(
        one_shot_model,
        condition,
        quality,
    )
    evaluation_steps = [
        int(value) for value in config["decoding"]["evaluation_steps"]
    ]
    evidence_config = config["decoding"]["evidence"]
    top_k_frames = min(
        int(evidence_config["top_k_frames"]),
        condition_frames,
    )
    evidence_lambda = float(evidence_config["default_lambda"])

    confidence_by_steps: dict[str, Any] = {}
    confidence_default = None
    default_steps = int(config["decoding"]["default_steps"])
    for steps in evaluation_steps:
        decoded = maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=steps,
            mode="confidence",
            top_k_frames=top_k_frames,
            evidence_lambda=evidence_lambda,
        )
        confidence_by_steps[str(steps)] = {
            "token_accuracy": _token_accuracy(decoded.tokens, targets),
            "scenario_accuracy": _scenario_accuracy(
                decoded.tokens,
                targets,
                scenarios,
            ),
            "trace": decoded.trace,
        }
        if steps == default_steps:
            confidence_default = decoded
    assert confidence_default is not None

    mode_results: dict[str, Any] = {}
    decoded_by_mode = {"confidence": confidence_default}
    for mode in (
        "evidence-ordering",
        "evidence-logits",
        "evidence-remask",
    ):
        decoded = maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=default_steps,
            mode=mode,
            top_k_frames=top_k_frames,
            evidence_lambda=evidence_lambda,
        )
        decoded_by_mode[mode] = decoded
    for mode, decoded in decoded_by_mode.items():
        mode_results[mode] = {
            "token_accuracy": _token_accuracy(decoded.tokens, targets),
            "scenario_accuracy": _scenario_accuracy(
                decoded.tokens,
                targets,
                scenarios,
            ),
            "token_difference_from_confidence": float(
                (
                    decoded.tokens != confidence_default.tokens
                ).float().mean().item()
            ),
            "trace": decoded.trace,
        }

    visible_support = visibility.mean(dim=1)
    mode_results["evidence-remask"][
        "commit_score_visibility_rank_correlation"
    ] = _rank_correlation(
        decoded_by_mode["evidence-remask"].commit_score,
        visible_support,
    )

    all_mask = torch.full(
        (condition.shape[0], maskgit_model.spatial_tokens),
        maskgit_model.mask_token_id,
        dtype=torch.int64,
        device=device,
    )
    original_logits = maskgit_model(all_mask, condition, quality).float()
    permutation = torch.arange(
        condition_frames - 1,
        -1,
        -1,
        device=device,
    )
    permuted_logits = maskgit_model(
        all_mask,
        condition[:, permutation],
        quality[:, permutation],
    ).float()
    permutation_metrics = {
        "max_absolute_logit_difference": float(
            (original_logits - permuted_logits).abs().max().item()
        ),
        "argmax_token_disagreement": float(
            (
                original_logits.argmax(dim=-1)
                != permuted_logits.argmax(dim=-1)
            )
            .float()
            .mean()
            .item()
        ),
    }
    return {
        "samples": len(indices),
        "scenarios": scenarios,
        "one_shot": {
            "token_accuracy": _token_accuracy(one_shot_tokens, targets),
            "scenario_accuracy": _scenario_accuracy(
                one_shot_tokens,
                targets,
                scenarios,
            ),
        },
        "maskgit_confidence_by_steps": confidence_by_steps,
        "maskgit_modes": mode_results,
        "permutation": permutation_metrics,
    }

