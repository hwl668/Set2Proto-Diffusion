"""Post-training latency, commit-evidence, and failure-case diagnostics."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import default_collate

from .decoding import maskgit_decode, one_shot_decode
from .model import build_model_from_config
from .research_evaluation import continuous_aggregation_baselines
from .smoke_evaluation import _rank_correlation
from .training import TokenTrainingDataset, load_codebook


def _load_model(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
):
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = build_model_from_config(
        config,
        hidden_dim=int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model


def _method_embedding(
    tokens: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    maps = codebook[tokens]
    return functional.normalize(maps.float().mean(dim=1), dim=-1)


def _scenario_correlations(
    scores: torch.Tensor,
    visibility: torch.Tensor,
    scenarios: list[str],
) -> dict[str, float | None]:
    def correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
        value = _rank_correlation(left, right)
        return value if math.isfinite(value) else None

    result = {
        "all": correlation(scores, visibility),
    }
    scenario_array = np.asarray(scenarios)
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        result[scenario] = correlation(
            scores[mask],
            visibility[mask],
        )
    return result


@torch.inference_mode()
def evaluate_post_training_diagnostics(
    *,
    config: Mapping[str, Any],
    dataset_root: Path,
    quantization_root: Path,
    one_shot_checkpoint: Path,
    maskgit_checkpoint: Path,
    device: torch.device,
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="test",
        precompute=True,
    )
    codebook = load_codebook(quantization_root).to(device)
    one_shot_model = _load_model(one_shot_checkpoint, config, device)
    maskgit_model = _load_model(maskgit_checkpoint, config, device)
    batch_size = int(config["evaluation"]["batch_size"])
    condition_frames = int(config["data"]["condition_frames"])
    evidence = config["decoding"]["evidence"]
    top_k_frames = min(
        int(evidence["top_k_frames"]),
        condition_frames,
    )
    evidence_lambda = float(evidence["default_lambda"])
    default_steps = int(config["decoding"]["default_steps"])

    embedding_chunks: dict[str, list[torch.Tensor]] = {}
    commit_chunks: dict[str, list[torch.Tensor]] = {}
    visibility_chunks: list[torch.Tensor] = []
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    sample_ids: list[str] = []

    def append_embedding(name: str, value: torch.Tensor) -> None:
        embedding_chunks.setdefault(name, []).append(value.detach().cpu())

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in indices])
        condition = batch["condition_features"][:, :condition_frames].to(
            device
        )
        quality = batch["condition_quality"][:, :condition_frames].to(device)
        visibility = batch["condition_visibility"][
            :, :condition_frames
        ].to(device)
        baselines = continuous_aggregation_baselines(condition, quality)
        append_embedding(
            "mean_pooling",
            functional.normalize(
                baselines["mean_pooling"].mean(dim=1),
                dim=-1,
            ),
        )
        append_embedding(
            "quality_pooling",
            functional.normalize(
                baselines["quality_pooling"].mean(dim=1),
                dim=-1,
            ),
        )
        one_shot_tokens = one_shot_decode(
            one_shot_model,
            condition,
            quality,
        )
        append_embedding(
            "one_shot_transformer",
            _method_embedding(one_shot_tokens, codebook),
        )
        for mode, name in (
            ("confidence", "maskgit_confidence"),
            ("evidence-ordering", "maskgit_evidence_ordering"),
            ("evidence-logits", "maskgit_evidence_logits"),
            ("evidence-remask", "maskgit_evidence_remask"),
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
            append_embedding(
                name,
                _method_embedding(decoded.tokens, codebook),
            )
            commit_chunks.setdefault(name, []).append(
                decoded.commit_score.detach().cpu()
            )
        visibility_chunks.append(visibility.mean(dim=1).cpu())
        identities.append(batch["target_identity"].cpu())
        scenarios.extend(str(value) for value in batch["scenario"])
        sample_ids.extend(str(value) for value in batch["sample_id"])

    all_embeddings = {
        name: torch.cat(chunks)
        for name, chunks in embedding_chunks.items()
    }
    identity_tensor = torch.cat(identities)
    visible_support = torch.cat(visibility_chunks)
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identity_tensor[clean_indices]
    gallery_embeddings = functional.normalize(
        dataset.targets["continuous_prototypes"][clean_indices]
        .float()
        .mean(dim=1),
        dim=-1,
    )
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    identity_to_gallery = {
        int(identity.item()): index
        for index, identity in enumerate(gallery_identities)
    }

    failure_rows: list[dict[str, Any]] = []
    for method, embeddings in all_embeddings.items():
        scores = torch.matmul(embeddings, gallery_embeddings.transpose(0, 1))
        for index in range(len(dataset)):
            target_identity = int(identity_tensor[index].item())
            target_column = identity_to_gallery[target_identity]
            genuine_score = float(scores[index, target_column].item())
            impostor_scores = scores[index].clone()
            impostor_scores[target_column] = -float("inf")
            best_impostor_score, best_impostor_column = torch.max(
                impostor_scores,
                dim=0,
            )
            rank = int(
                (scores[index] > scores[index, target_column]).sum().item()
            ) + 1
            predicted_column = int(scores[index].argmax().item())
            failure_rows.append(
                {
                    "method": method,
                    "sample_id": sample_ids[index],
                    "scenario": scenarios[index],
                    "target_identity": target_identity,
                    "predicted_identity": int(
                        gallery_identities[predicted_column].item()
                    ),
                    "best_impostor_identity": int(
                        gallery_identities[
                            int(best_impostor_column.item())
                        ].item()
                    ),
                    "genuine_rank": rank,
                    "genuine_score": genuine_score,
                    "best_impostor_score": float(
                        best_impostor_score.item()
                    ),
                    "genuine_margin": (
                        genuine_score - float(best_impostor_score.item())
                    ),
                    "correct_top1": predicted_column == target_column,
                }
            )
    failure_rows.sort(
        key=lambda row: (
            row["method"],
            -int(row["genuine_rank"]),
            float(row["genuine_margin"]),
        )
    )
    failure_path = output_directory / "failure_cases.csv"
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(failure_rows[0]),
        )
        writer.writeheader()
        writer.writerows(failure_rows)

    commit_correlations = {
        name: _scenario_correlations(
            torch.cat(chunks),
            visible_support,
            scenarios,
        )
        for name, chunks in commit_chunks.items()
    }

    latency_batch = default_collate(
        [dataset[index] for index in range(min(batch_size, len(dataset)))]
    )
    latency_condition = latency_batch["condition_features"][
        :, :condition_frames
    ].to(device)
    latency_quality = latency_batch["condition_quality"][
        :, :condition_frames
    ].to(device)
    warmup_runs = int(config["evaluation"]["latency_warmup_runs"])
    measurement_runs = int(
        config["evaluation"]["latency_measurement_runs"]
    )

    methods: dict[str, Callable[[], Any]] = {
        "mean_pooling": lambda: continuous_aggregation_baselines(
            latency_condition,
            latency_quality,
        )["mean_pooling"],
        "quality_pooling": lambda: continuous_aggregation_baselines(
            latency_condition,
            latency_quality,
        )["quality_pooling"],
        "one_shot_transformer": lambda: one_shot_decode(
            one_shot_model,
            latency_condition,
            latency_quality,
        )
    }
    for steps in (1, 2, 4, 8):
        methods[f"maskgit_confidence_{steps}step"] = (
            lambda selected_steps=steps: maskgit_decode(
                maskgit_model,
                latency_condition,
                latency_quality,
                codebook=codebook,
                steps=selected_steps,
                mode="confidence",
                top_k_frames=top_k_frames,
                evidence_lambda=evidence_lambda,
            )
        )
    for mode, name in (
        ("evidence-ordering", "maskgit_evidence_ordering"),
        ("evidence-logits", "maskgit_evidence_logits"),
        ("evidence-remask", "maskgit_evidence_remask"),
    ):
        methods[name] = (
            lambda selected_mode=mode: maskgit_decode(
                maskgit_model,
                latency_condition,
                latency_quality,
                codebook=codebook,
                steps=default_steps,
                mode=selected_mode,
                top_k_frames=top_k_frames,
                evidence_lambda=evidence_lambda,
            )
        )
    latency: dict[str, Any] = {}
    for name, function in methods.items():
        for _ in range(warmup_runs):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        for _ in range(measurement_runs):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        measured_batch = int(latency_condition.shape[0])
        latency[name] = {
            "batch_size": measured_batch,
            "measurement_runs": measurement_runs,
            "batch_latency_ms": (
                elapsed * 1000.0 / measurement_runs
            ),
            "per_set_latency_ms": (
                elapsed * 1000.0
                / (measurement_runs * measured_batch)
            ),
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        }

    worst_cases = {}
    for method in all_embeddings:
        selected = [row for row in failure_rows if row["method"] == method]
        worst_cases[method] = selected[:20]
    result = {
        "samples": len(dataset),
        "identities": len(gallery_identities),
        "device": str(device),
        "commit_visibility_rank_correlation": commit_correlations,
        "latency": latency,
        "failure_cases_csv": str(failure_path),
        "worst_cases": worst_cases,
    }
    diagnostics_path = output_directory / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result["diagnostics_json"] = str(diagnostics_path)
    return result
