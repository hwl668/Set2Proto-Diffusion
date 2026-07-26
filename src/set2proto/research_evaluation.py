"""Continuous baselines and identity verification for synthetic pilot runs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import default_collate

from .decoding import maskgit_decode, one_shot_decode
from .model import ConditionalTokenTransformer
from .quantization import quantize_vectors
from .training import TokenTrainingDataset


def continuous_aggregation_baselines(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return four required continuous local aggregation baselines."""

    features = F.normalize(condition_features.float(), dim=-1)
    batch, _, positions, dimension = features.shape
    best_indices = condition_quality.argmax(dim=1)
    batch_indices = torch.arange(batch, device=features.device)
    best_single = features[batch_indices, best_indices]
    mean_pooling = F.normalize(features.mean(dim=1), dim=-1)
    max_pooling = F.normalize(features.max(dim=1).values, dim=-1)
    quality = condition_quality.float().clamp_min(1e-6)
    quality = quality / quality.sum(dim=1, keepdim=True).clamp_min(1e-6)
    quality_pooling = F.normalize(
        (
            quality[:, :, None, None]
            * features
        ).sum(dim=1),
        dim=-1,
    )
    expected_shape = (batch, positions, dimension)
    outputs = {
        "best_single_frame": best_single,
        "mean_pooling": mean_pooling,
        "max_pooling": max_pooling,
        "quality_pooling": quality_pooling,
    }
    if any(tuple(value.shape) != expected_shape for value in outputs.values()):
        raise RuntimeError("aggregation baseline returned an invalid shape")
    return outputs


def _verification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    far_targets: list[float],
) -> dict[str, Any]:
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float(0.5 * (fpr[eer_index] + fnr[eer_index]))
    negative_count = int((labels == 0).sum())
    tar: dict[str, Any] = {}
    for far in far_targets:
        valid = np.where(fpr <= far)[0]
        value = float(tpr[valid].max()) if len(valid) else 0.0
        tar[f"{far:g}"] = {
            "tar": value,
            "negative_pairs": negative_count,
            "expected_false_accepts_at_far": negative_count * far,
            "reliable": negative_count * far >= 1.0,
        }
    return {
        "roc_auc": auc,
        "eer": eer,
        "tar_at_far": tar,
        "positive_pairs": int((labels == 1).sum()),
        "negative_pairs": negative_count,
    }


def _identification_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute closed-set Rank-1 and genuine/impostor score separation."""

    if scores.ndim != 2 or labels.shape != scores.shape:
        raise ValueError("scores and labels must have matching [N,G] shapes")
    positives_per_row = labels.sum(dim=1)
    if not bool((positives_per_row == 1).all().item()):
        raise ValueError("gallery must contain exactly one target per probe")
    target_columns = labels.to(torch.int64).argmax(dim=1)
    row_indices = torch.arange(scores.shape[0], device=scores.device)
    genuine_scores = scores[row_indices, target_columns]
    impostor_scores = scores.masked_fill(labels, float("-inf"))
    strongest_impostor = impostor_scores.max(dim=1).values
    predicted_columns = scores.argmax(dim=1)
    margins = genuine_scores - strongest_impostor
    return {
        "rank1_accuracy": float(
            (predicted_columns == target_columns).float().mean().item()
        ),
        "mean_genuine_score": float(genuine_scores.mean().item()),
        "mean_strongest_impostor_score": float(
            strongest_impostor.mean().item()
        ),
        "mean_genuine_impostor_margin": float(margins.mean().item()),
    }


def _evaluate_embeddings(
    *,
    embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    identities: torch.Tensor,
    gallery_identities: torch.Tensor,
    scenarios: list[str],
    far_targets: list[float],
) -> dict[str, Any]:
    scores = torch.matmul(
        F.normalize(embeddings.float(), dim=-1),
        F.normalize(gallery_embeddings.float(), dim=-1).transpose(0, 1),
    )
    labels = identities[:, None] == gallery_identities[None, :]
    results: dict[str, Any] = {}
    scenario_array = np.asarray(scenarios)
    for scenario in sorted(set(scenarios)):
        mask = scenario_array == scenario
        tensor_mask = torch.from_numpy(mask).to(labels.device)
        selected_scores = scores[tensor_mask]
        selected_labels = labels[tensor_mask]
        results[scenario] = {
            **_verification_metrics(
                selected_labels.reshape(-1).cpu().numpy().astype(int),
                selected_scores.reshape(-1).cpu().numpy(),
                far_targets,
            ),
            **_identification_metrics(selected_scores, selected_labels),
        }
    results["all"] = {
        **_verification_metrics(
        labels.reshape(-1).cpu().numpy().astype(int),
        scores.reshape(-1).cpu().numpy(),
        far_targets,
        ),
        **_identification_metrics(scores, labels),
    }
    hard_scenarios = [
        scenario
        for scenario in sorted(set(scenarios))
        if scenario != "clean"
    ]
    results["hard_average"] = {
        "roc_auc": float(
            np.mean([results[name]["roc_auc"] for name in hard_scenarios])
        ),
        "eer": float(
            np.mean([results[name]["eer"] for name in hard_scenarios])
        ),
        "rank1_accuracy": float(
            np.mean(
                [results[name]["rank1_accuracy"] for name in hard_scenarios]
            )
        ),
        "mean_genuine_impostor_margin": float(
            np.mean(
                [
                    results[name]["mean_genuine_impostor_margin"]
                    for name in hard_scenarios
                ]
            )
        ),
    }
    return results


@torch.inference_mode()
def evaluate_research_split(
    *,
    one_shot_model: ConditionalTokenTransformer,
    maskgit_model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    condition_frames: int,
) -> dict[str, Any]:
    """Evaluate required aggregation/model methods against one clean gallery."""

    one_shot_model.eval()
    maskgit_model.eval()
    codebook_device = F.normalize(codebook.float(), dim=-1).to(device)
    batch_size = int(config["evaluation"]["batch_size"])
    default_steps = int(config["decoding"]["default_steps"])
    evidence = config["decoding"]["evidence"]
    top_k_frames = min(int(evidence["top_k_frames"]), condition_frames)
    evidence_lambda = float(evidence["default_lambda"])
    evaluation_steps = [
        int(value) for value in config["decoding"]["evaluation_steps"]
    ]

    map_chunks: dict[str, list[torch.Tensor]] = {}
    token_chunks: dict[str, list[torch.Tensor]] = {}
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []

    def append_map(name: str, value: torch.Tensor) -> None:
        map_chunks.setdefault(name, []).append(value.detach().cpu().half())

    def append_tokens(name: str, value: torch.Tensor) -> None:
        token_chunks.setdefault(name, []).append(value.detach().cpu())
        append_map(name, codebook_device[value].float())

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"][:, :condition_frames].to(
            device
        )
        quality = batch["condition_quality"][:, :condition_frames].to(device)
        targets = batch["target_tokens"].to(device)
        continuous_targets = dataset.targets["continuous_prototypes"][
            indices
        ].to(device)
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(list(batch["scenario"]))

        baselines = continuous_aggregation_baselines(condition, quality)
        for name, maps in baselines.items():
            append_map(name, maps)
            quantized_tokens, _ = quantize_vectors(
                maps,
                codebook_device,
            )
            append_tokens(f"{name}_quantized", quantized_tokens)
        append_map("teacher_continuous", continuous_targets)
        append_tokens("teacher_quantized", targets)

        one_shot_tokens = one_shot_decode(
            one_shot_model,
            condition,
            quality,
        )
        append_tokens("one_shot_transformer", one_shot_tokens)
        for steps in evaluation_steps:
            decoded = maskgit_decode(
                maskgit_model,
                condition,
                quality,
                codebook=codebook_device,
                steps=steps,
                mode="confidence",
                top_k_frames=top_k_frames,
                evidence_lambda=evidence_lambda,
            )
            name = f"maskgit_confidence_{steps}step"
            append_tokens(name, decoded.tokens)
            if steps == default_steps:
                append_tokens("maskgit_confidence", decoded.tokens)
        for mode, name in (
            ("evidence-ordering", "maskgit_evidence_ordering"),
            ("evidence-logits", "maskgit_evidence_logits"),
            ("evidence-remask", "maskgit_evidence_remask"),
        ):
            decoded = maskgit_decode(
                maskgit_model,
                condition,
                quality,
                codebook=codebook_device,
                steps=default_steps,
                mode=mode,
                top_k_frames=top_k_frames,
                evidence_lambda=evidence_lambda,
            )
            append_tokens(name, decoded.tokens)

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
        index for index, scenario in enumerate(scenarios) if scenario == "clean"
    ]
    gallery_identities = identity_tensor[clean_indices]
    gallery_embeddings = embeddings["teacher_continuous"][clean_indices]
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    far_targets = [float(value) for value in config["evaluation"]["far_targets"]]

    verification = {
        name: _evaluate_embeddings(
            embeddings=method_embeddings,
            gallery_embeddings=gallery_embeddings,
            identities=identity_tensor,
            gallery_identities=gallery_identities,
            scenarios=scenarios,
            far_targets=far_targets,
        )
        for name, method_embeddings in embeddings.items()
    }
    token_accuracy: dict[str, Any] = {}
    target_tokens = dataset.targets["tokens"]
    scenario_array = np.asarray(scenarios)
    for name, chunks in token_chunks.items():
        predicted = torch.cat(chunks)
        overall = float((predicted == target_tokens).float().mean().item())
        by_scenario = {}
        for scenario in sorted(set(scenarios)):
            mask = torch.from_numpy(scenario_array == scenario)
            by_scenario[scenario] = float(
                (predicted[mask] == target_tokens[mask]).float().mean().item()
            )
        token_accuracy[name] = {
            "all": overall,
            "by_scenario": by_scenario,
        }
    return {
        "samples": len(dataset),
        "identities": len(clean_indices),
        "gallery": "clean_disjoint_teacher_continuous",
        "verification": verification,
        "token_accuracy": token_accuracy,
    }
