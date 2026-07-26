"""Train-only PCA, robust teacher prototypes, and spherical K-means targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# joblib's Windows physical-core probe can fail in restricted shells. Its
# processes are not needed here, so force one loky worker before importing it;
# BLAS/OpenMP kernels used by PCA and K-means remain independently available.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .datasets import (
    load_train_feature_bank,
    make_set_dataset,
    validate_set_dataset,
)
from .environment import config_sha256
from .synthetic import (
    SPLITS,
    _load_tensor_file,
)


QUANTIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PCAProjection:
    mean: torch.Tensor
    components: torch.Tensor
    explained_variance_ratio: torch.Tensor
    fit_tokens: int
    fit_split: str = "train"

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(
                f"expected feature dim {self.input_dim}, "
                f"got {features.shape[-1]}"
            )
        projected = torch.matmul(
            features.float() - self.mean,
            self.components.transpose(0, 1),
        )
        return F.normalize(projected, p=2.0, dim=-1, eps=1e-6)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": QUANTIZATION_SCHEMA_VERSION,
            "method": "pca",
            "fit_split": self.fit_split,
            "fit_tokens": self.fit_tokens,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "mean": self.mean.cpu(),
            "components": self.components.cpu(),
            "explained_variance_ratio": self.explained_variance_ratio.cpu(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PCAProjection":
        if payload.get("method") != "pca":
            raise ValueError("projection payload is not PCA")
        return cls(
            mean=payload["mean"].float(),
            components=payload["components"].float(),
            explained_variance_ratio=payload[
                "explained_variance_ratio"
            ].float(),
            fit_tokens=int(payload["fit_tokens"]),
            fit_split=str(payload["fit_split"]),
        )


@dataclass(frozen=True)
class QuantizationPreparationResult:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    validation: dict[str, Any]
    reused: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fit_pca_projection(
    train_features: torch.Tensor,
    *,
    output_dim: int,
    max_fit_tokens: int,
    seed: int,
    svd_solver: str,
) -> PCAProjection:
    """Fit PCA on a deterministic train-only token sample."""

    if train_features.ndim != 3:
        raise ValueError("train_features must have shape [images, tokens, dim]")
    flattened = train_features.reshape(-1, train_features.shape[-1])
    available = int(flattened.shape[0])
    if output_dim > min(available, int(flattened.shape[1])):
        raise ValueError(
            "PCA output_dim cannot exceed available tokens or input dimension"
        )
    selected_count = min(available, max_fit_tokens)
    if selected_count < available:
        generator = torch.Generator().manual_seed(seed)
        selected_indices = torch.randperm(
            available,
            generator=generator,
        )[:selected_count]
        selected = flattened[selected_indices].float()
    else:
        selected = flattened.float()

    estimator = PCA(
        n_components=output_dim,
        whiten=False,
        svd_solver=svd_solver,
        random_state=seed if svd_solver == "randomized" else None,
        copy=True,
    )
    estimator.fit(selected.numpy())
    return PCAProjection(
        mean=torch.from_numpy(np.asarray(estimator.mean_)).float(),
        components=torch.from_numpy(
            np.asarray(estimator.components_)
        ).float(),
        explained_variance_ratio=torch.from_numpy(
            np.asarray(estimator.explained_variance_ratio_)
        ).float(),
        fit_tokens=selected_count,
    )


def robust_teacher_pool(
    projected_features: torch.Tensor,
    quality_norm: torch.Tensor,
    *,
    keep_fraction: float,
    consensus_weight: float,
    quality_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pool [B,T,P,D] teachers using local consensus and quality trimming."""

    squeeze_batch = False
    if projected_features.ndim == 3:
        projected_features = projected_features.unsqueeze(0)
        quality_norm = quality_norm.unsqueeze(0)
        squeeze_batch = True
    if projected_features.ndim != 4:
        raise ValueError(
            "projected_features must have shape [B,T,P,D] or [T,P,D]"
        )
    batch, teachers, positions, _ = projected_features.shape
    if tuple(quality_norm.shape) != (batch, teachers):
        raise ValueError("quality_norm must have shape [B,T]")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")

    features = F.normalize(projected_features.float(), dim=-1, eps=1e-6)
    by_position = features.permute(0, 2, 1, 3)
    pairwise = torch.matmul(by_position, by_position.transpose(-1, -2))
    if teachers > 1:
        consensus = (
            pairwise.sum(dim=-1) - pairwise.diagonal(dim1=-2, dim2=-1)
        ) / float(teachers - 1)
    else:
        consensus = torch.ones(
            (batch, positions, teachers),
            dtype=features.dtype,
            device=features.device,
        )

    normalized_quality = quality_norm.float().clamp_min(1e-6)
    normalized_quality = normalized_quality / normalized_quality.mean(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)
    quality_score = torch.log(normalized_quality).unsqueeze(1)
    combined_score = (
        consensus_weight * consensus + quality_weight * quality_score
    )
    keep_count = max(1, int(math.floor(teachers * keep_fraction)))
    kept_indices = torch.topk(
        combined_score,
        k=keep_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    keep_mask = torch.zeros_like(combined_score, dtype=torch.bool)
    keep_mask.scatter_(-1, kept_indices, True)
    masked_score = combined_score.masked_fill(~keep_mask, float("-inf"))
    weights = torch.softmax(masked_score, dim=-1)
    pooled = torch.sum(weights.unsqueeze(-1) * by_position, dim=2)
    pooled = F.normalize(pooled, dim=-1, eps=1e-6)
    effective_frames = 1.0 / weights.square().sum(dim=-1).clamp_min(1e-6)
    diagnostics = {
        "weights": weights,
        "consensus": consensus,
        "keep_mask": keep_mask,
        "effective_frames": effective_frames,
    }
    if squeeze_batch:
        pooled = pooled[0]
        diagnostics = {key: value[0] for key, value in diagnostics.items()}
    return pooled, diagnostics


def robust_condition_pool(
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    *,
    keep_fraction: float,
    consensus_weight: float,
    quality_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the teacher robust-pooling rule unchanged to condition frames.

    The semantic wrapper intentionally delegates all scoring, trimming, and
    softmax weighting to :func:`robust_teacher_pool` so the condition-side
    experiment cannot silently drift from the teacher construction.
    """

    return robust_teacher_pool(
        condition_features,
        condition_quality,
        keep_fraction=keep_fraction,
        consensus_weight=consensus_weight,
        quality_weight=quality_weight,
    )


def quantize_vectors(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-cosine quantization for arbitrary leading vector dimensions."""

    if vectors.shape[-1] != codebook.shape[-1]:
        raise ValueError("vectors and codebook dimensions do not match")
    original_shape = vectors.shape[:-1]
    flat = F.normalize(vectors.reshape(-1, vectors.shape[-1]).float(), dim=-1)
    normalized_codebook = F.normalize(codebook.float(), dim=-1)
    token_chunks: list[torch.Tensor] = []
    cosine_chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], chunk_size):
        scores = torch.matmul(
            flat[start : start + chunk_size],
            normalized_codebook.transpose(0, 1),
        )
        cosine, tokens = torch.max(scores, dim=-1)
        token_chunks.append(tokens)
        cosine_chunks.append(cosine)
    tokens = torch.cat(token_chunks).reshape(original_shape)
    cosine = torch.cat(cosine_chunks).reshape(original_shape)
    return tokens.to(torch.int64), cosine


def _build_split_prototypes(
    *,
    dataset_root: Path,
    split: str,
    projector: PCAProjection,
    teacher_config: Mapping[str, Any],
    batch_size: int = 64,
) -> dict[str, Any]:
    teacher_mode = str(teacher_config["mode"])
    dataset = make_set_dataset(
        dataset_root,
        split,
        teacher_mode=teacher_mode,
    )
    all_prototypes: list[torch.Tensor] = []
    effective_frames: list[torch.Tensor] = []
    consensus_values: list[torch.Tensor] = []
    target_identities: list[int] = []
    sample_ids: list[str] = []
    scenarios: list[str] = []

    for start in range(0, len(dataset), batch_size):
        samples = [
            dataset[index]
            for index in range(
                start,
                min(start + batch_size, len(dataset)),
            )
        ]
        raw_features = torch.stack(
            [sample["teacher_features"] for sample in samples]
        ).float()
        quality = torch.stack(
            [sample["teacher_quality"] for sample in samples]
        ).float()
        projected = projector.transform(raw_features)
        prototypes, diagnostics = robust_teacher_pool(
            projected,
            quality,
            keep_fraction=float(
                teacher_config["robust_keep_fraction"]
            ),
            consensus_weight=float(teacher_config["consensus_weight"]),
            quality_weight=float(teacher_config["quality_weight"]),
        )
        all_prototypes.append(prototypes.to(torch.float16))
        effective_frames.append(
            diagnostics["effective_frames"].mean(dim=1).cpu()
        )
        consensus_values.append(
            diagnostics["consensus"].mean(dim=(1, 2)).cpu()
        )
        target_identities.extend(
            int(sample["target_identity"].item()) for sample in samples
        )
        sample_ids.extend(str(sample["sample_id"]) for sample in samples)
        scenarios.extend(str(sample["scenario"]) for sample in samples)

    return {
        "continuous_prototypes": torch.cat(all_prototypes, dim=0),
        "target_identity": torch.tensor(target_identities, dtype=torch.int64),
        "sample_ids": sample_ids,
        "scenarios": scenarios,
        "teacher_mode": teacher_mode,
        "mean_effective_frames": float(
            torch.cat(effective_frames).mean().item()
        ),
        "mean_local_consensus": float(
            torch.cat(consensus_values).mean().item()
        ),
    }


def fit_spherical_codebook(
    train_prototypes: torch.Tensor,
    *,
    codebook_size: int,
    max_fit_tokens: int,
    batch_size: int,
    iterations: int,
    n_init: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    flattened = F.normalize(
        train_prototypes.reshape(-1, train_prototypes.shape[-1]).float(),
        dim=-1,
    )
    available = int(flattened.shape[0])
    selected_count = min(available, max_fit_tokens)
    if selected_count < available:
        generator = torch.Generator().manual_seed(seed + 1)
        indices = torch.randperm(available, generator=generator)[
            :selected_count
        ]
        selected = flattened[indices]
    else:
        selected = flattened
    if selected_count < codebook_size:
        raise ValueError(
            f"codebook K={codebook_size} exceeds fit tokens={selected_count}"
        )

    estimator = MiniBatchKMeans(
        n_clusters=codebook_size,
        init="k-means++",
        n_init=n_init,
        max_iter=iterations,
        batch_size=min(batch_size, selected_count),
        random_state=seed,
        reassignment_ratio=0.01,
        verbose=0,
    )
    estimator.fit(selected.numpy())
    codebook = F.normalize(
        torch.from_numpy(np.asarray(estimator.cluster_centers_)).float(),
        dim=-1,
    )
    details = {
        "fit_split": "train",
        "fit_tokens": selected_count,
        "available_train_tokens": available,
        "inertia": float(estimator.inertia_),
        "n_steps": int(getattr(estimator, "n_steps_", 0)),
        "n_iter": int(getattr(estimator, "n_iter_", 0)),
    }
    return codebook, details


def compute_quantization_metrics(
    *,
    continuous: torch.Tensor,
    tokens: torch.Tensor,
    codebook: torch.Tensor,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    reconstruction = codebook[tokens].float()
    normalized_continuous = F.normalize(continuous.float(), dim=-1)
    token_cosine = F.cosine_similarity(
        normalized_continuous,
        reconstruction,
        dim=-1,
    )
    continuous_map = F.normalize(
        normalized_continuous.mean(dim=1),
        dim=-1,
    )
    reconstructed_map = F.normalize(reconstruction.mean(dim=1), dim=-1)
    map_cosine = F.cosine_similarity(
        continuous_map,
        reconstructed_map,
        dim=-1,
    )
    counts = torch.bincount(
        tokens.reshape(-1),
        minlength=codebook.shape[0],
    ).float()
    probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
    perplexity = float(
        torch.exp(
            -(probabilities * torch.log(probabilities)).sum()
        ).item()
    )

    def summarize(mask: torch.Tensor | None = None) -> dict[str, float]:
        if mask is None:
            selected_token_cosine = token_cosine.reshape(-1)
            selected_map_cosine = map_cosine
            selected_tokens = tokens.reshape(-1)
        else:
            selected_token_cosine = token_cosine[mask].reshape(-1)
            selected_map_cosine = map_cosine[mask]
            selected_tokens = tokens[mask].reshape(-1)
        used = int(torch.unique(selected_tokens).numel())
        return {
            "mean_token_cosine": float(
                selected_token_cosine.mean().item()
            ),
            "median_token_cosine": float(
                selected_token_cosine.median().item()
            ),
            "p05_token_cosine": float(
                torch.quantile(selected_token_cosine, 0.05).item()
            ),
            "mean_map_cosine": float(selected_map_cosine.mean().item()),
            "codebook_utilization": used / float(codebook.shape[0]),
            "used_codes": used,
        }

    metrics: dict[str, Any] = summarize()
    metrics["token_perplexity"] = perplexity
    metrics["normalized_perplexity"] = perplexity / float(codebook.shape[0])
    metrics["token_mse"] = float(
        F.mse_loss(reconstruction, normalized_continuous).item()
    )
    scenario_metrics: dict[str, Any] = {}
    scenario_array = np.asarray(list(scenarios))
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        scenario_metrics[scenario] = summarize(mask)
    metrics["by_scenario"] = scenario_metrics
    return metrics


def prepare_quantization_artifacts(
    *,
    config: Mapping[str, Any],
    profile: str,
    run_id: str,
    dataset_root: str | Path,
    resume: bool = False,
) -> QuantizationPreparationResult:
    """Fit train-only artifacts and generate quantized targets for all splits."""

    source_root = Path(dataset_root).expanduser().resolve()
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"synthetic dataset manifest does not exist: {source_manifest_path}"
        )
    source_validation = validate_set_dataset(source_root)
    if not source_validation["passed"]:
        raise ValueError("source set dataset failed validation")
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    source_data_config = source_manifest["data_config"]
    current_data_config = config["data"]
    if source_data_config["feature_grid"] != current_data_config["feature_grid"]:
        raise ValueError("dataset feature grid does not match current config")
    if int(source_manifest["seed"]) != int(config["project"]["seed"]):
        raise ValueError("dataset seed does not match current config")

    artifact_root = (
        Path(config["paths"]["cache_dir"]).resolve()
        / "quantization"
        / run_id
    )
    manifest_path = artifact_root / "manifest.json"
    source_manifest_hash = _file_sha256(source_manifest_path)
    if artifact_root.exists():
        if not resume:
            raise FileExistsError(
                "quantization artifacts already exist and will not be "
                f"overwritten: {artifact_root}"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "cannot resume incomplete quantization artifacts without "
                f"manifest: {artifact_root}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("config_sha256") != config_sha256(config):
            raise ValueError(
                "cannot resume quantization artifacts from a different config"
            )
        if manifest.get("source_manifest_sha256") != source_manifest_hash:
            raise ValueError(
                "cannot resume after source dataset manifest changed"
            )
        validation = validate_quantization_artifacts(artifact_root)
        return QuantizationPreparationResult(
            root=artifact_root,
            manifest_path=manifest_path,
            manifest=manifest,
            validation=validation,
            reused=True,
        )

    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = artifact_root.parent / (
        f".{artifact_root.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary_root.mkdir(parents=False, exist_ok=False)

    train_features = load_train_feature_bank(source_root)
    projection_config = config["projection"]
    grid = config["data"]["feature_grid"]
    projector = fit_pca_projection(
        train_features,
        output_dim=int(grid["projected_dim"]),
        max_fit_tokens=int(projection_config["max_fit_tokens"]),
        seed=int(config["project"]["seed"]),
        svd_solver=str(projection_config["svd_solver"]),
    )
    projection_path = temporary_root / "projection.pt"
    torch.save(projector.to_payload(), projection_path)

    split_prototypes = {
        split: _build_split_prototypes(
            dataset_root=source_root,
            split=split,
            projector=projector,
            teacher_config=config["teacher"],
        )
        for split in SPLITS
    }
    quantization_config = config["quantization"]
    codebook, codebook_details = fit_spherical_codebook(
        split_prototypes["train"]["continuous_prototypes"],
        codebook_size=int(quantization_config["codebook_size"]),
        max_fit_tokens=int(quantization_config["max_fit_tokens"]),
        batch_size=int(quantization_config["batch_size"]),
        iterations=int(quantization_config["iterations"]),
        n_init=int(quantization_config["n_init"]),
        seed=int(config["project"]["seed"]),
    )
    codebook_path = temporary_root / "codebook.pt"
    torch.save(
        {
            "schema_version": QUANTIZATION_SCHEMA_VERSION,
            "method": "spherical_kmeans",
            "vectors": codebook,
            "codebook_size": int(codebook.shape[0]),
            "dimension": int(codebook.shape[1]),
            **codebook_details,
        },
        codebook_path,
    )

    metrics: dict[str, Any] = {}
    teacher_diagnostics: dict[str, Any] = {}
    target_paths: dict[str, Path] = {}
    for split in SPLITS:
        split_data = split_prototypes[split]
        continuous = split_data["continuous_prototypes"]
        tokens, _ = quantize_vectors(continuous, codebook)
        split_metrics = compute_quantization_metrics(
            continuous=continuous,
            tokens=tokens,
            codebook=codebook,
            scenarios=split_data["scenarios"],
        )
        metrics[split] = split_metrics
        teacher_diagnostics[split] = {
            "mean_effective_frames": split_data[
                "mean_effective_frames"
            ],
            "mean_local_consensus": split_data["mean_local_consensus"],
        }
        target_path = temporary_root / f"{split}_targets.pt"
        target_paths[split] = target_path
        torch.save(
            {
                "schema_version": QUANTIZATION_SCHEMA_VERSION,
                "split": split,
                "teacher_mode": split_data["teacher_mode"],
                "continuous_prototypes": continuous,
                "tokens": tokens,
                "target_identity": split_data["target_identity"],
                "sample_ids": split_data["sample_ids"],
                "scenarios": split_data["scenarios"],
            },
            target_path,
        )

    files: dict[str, Any] = {}
    for path in (projection_path, codebook_path, *target_paths.values()):
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    gate_min_cosine = float(quantization_config["gate_min_cosine"])
    gate_min_utilization = float(
        quantization_config["gate_min_utilization"]
    )
    research_gate = {
        split: {
            "cosine_passed": metrics[split]["mean_token_cosine"]
            >= gate_min_cosine,
            "utilization_passed": metrics[split]["codebook_utilization"]
            >= gate_min_utilization,
            "passed": (
                metrics[split]["mean_token_cosine"] >= gate_min_cosine
                and metrics[split]["codebook_utilization"]
                >= gate_min_utilization
            ),
        }
        for split in SPLITS
    }
    manifest = {
        "schema_version": QUANTIZATION_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        ),
        "profile": profile,
        "run_id": run_id,
        "config_sha256": config_sha256(config),
        "source_dataset": str(source_root),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_hash,
        "fit_split": "train",
        "projection": {
            "method": "pca",
            "input_dim": projector.input_dim,
            "output_dim": projector.output_dim,
            "fit_tokens": projector.fit_tokens,
            "explained_variance_ratio_sum": float(
                projector.explained_variance_ratio.sum().item()
            ),
        },
        "teacher": {
            **dict(config["teacher"]),
            "diagnostics": teacher_diagnostics,
        },
        "codebook": {
            "method": "spherical_kmeans",
            "size": int(codebook.shape[0]),
            "dimension": int(codebook.shape[1]),
            **codebook_details,
        },
        "metrics": metrics,
        "research_gate": research_gate,
        "files": files,
    }
    _atomic_json(temporary_root / "manifest.json", manifest)
    os.replace(temporary_root, artifact_root)

    validation = validate_quantization_artifacts(artifact_root)
    return QuantizationPreparationResult(
        root=artifact_root,
        manifest_path=artifact_root / "manifest.json",
        manifest=manifest,
        validation=validation,
        reused=False,
    )


def validate_quantization_artifacts(root: str | Path) -> dict[str, Any]:
    artifact_root = Path(root).expanduser().resolve()
    with (artifact_root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    projection_payload = _load_tensor_file(artifact_root / "projection.pt")
    codebook_payload = _load_tensor_file(artifact_root / "codebook.pt")
    codebook = codebook_payload["vectors"].float()
    checks: dict[str, bool] = {
        "fit_split_is_train": manifest.get("fit_split") == "train"
        and projection_payload.get("fit_split") == "train"
        and codebook_payload.get("fit_split") == "train",
        "projection_shape": tuple(projection_payload["components"].shape)
        == (
            int(manifest["projection"]["output_dim"]),
            int(manifest["projection"]["input_dim"]),
        ),
        "projection_finite": bool(
            torch.isfinite(projection_payload["components"]).all().item()
        ),
        "codebook_shape": tuple(codebook.shape)
        == (
            int(manifest["codebook"]["size"]),
            int(manifest["codebook"]["dimension"]),
        ),
        "codebook_finite": bool(torch.isfinite(codebook).all().item()),
        "codebook_unit_norm": bool(
            torch.allclose(
                codebook.norm(dim=-1),
                torch.ones(codebook.shape[0]),
                atol=1e-4,
                rtol=1e-4,
            )
        ),
        "file_checksums": True,
    }
    for filename, file_info in manifest["files"].items():
        path = artifact_root / filename
        if not path.is_file() or _file_sha256(path) != file_info["sha256"]:
            checks["file_checksums"] = False

    identity_sets: dict[str, set[int]] = {}
    for split in SPLITS:
        payload = _load_tensor_file(artifact_root / f"{split}_targets.pt")
        continuous = payload["continuous_prototypes"]
        tokens = payload["tokens"]
        identities = payload["target_identity"]
        identity_sets[split] = set(int(value) for value in identities.tolist())
        checks[f"{split}.target_shapes"] = (
            continuous.ndim == 3
            and tokens.shape == continuous.shape[:2]
            and identities.shape[0] == continuous.shape[0]
        )
        checks[f"{split}.continuous_finite"] = bool(
            torch.isfinite(continuous).all().item()
        )
        checks[f"{split}.token_range"] = bool(
            tokens.numel() > 0
            and int(tokens.min().item()) >= 0
            and int(tokens.max().item()) < codebook.shape[0]
        )
    checks["identity_disjoint.train_val"] = identity_sets[
        "train"
    ].isdisjoint(identity_sets["val"])
    checks["identity_disjoint.train_test"] = identity_sets[
        "train"
    ].isdisjoint(identity_sets["test"])
    checks["identity_disjoint.val_test"] = identity_sets["val"].isdisjoint(
        identity_sets["test"]
    )
    return {"passed": all(checks.values()), "checks": checks}
