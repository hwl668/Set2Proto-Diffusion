"""Collision-safe train/validation artifacts for residual-token experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from .environment import config_sha256
from .residual_evidence import quality_weighted_anchor
from .residual_quantization import (
    ResidualCodebook,
    compute_residual_quantization_metrics,
    encode_teacher_residuals,
    fit_residual_codebook,
)
from .training import TokenTrainingDataset


@dataclass(frozen=True)
class ResidualArtifactResult:
    root: Path
    manifest: dict[str, Any]
    validation: dict[str, Any]
    reused: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _split_inputs(
    *,
    dataset_root: Path,
    absolute_quantization_root: Path,
    split: str,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError("P2-1 artifact construction is train/validation only")
    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=absolute_quantization_root,
        split=split,
        precompute=True,
    )
    assert dataset._precomputed is not None
    condition = dataset._precomputed["condition_features"]
    quality = dataset._precomputed["condition_quality"]
    anchor_chunks = [
        quality_weighted_anchor(
            condition[start : start + batch_size],
            quality[start : start + batch_size],
        ).cpu()
        for start in range(0, len(dataset), batch_size)
    ]
    anchors = torch.cat(anchor_chunks)
    teachers = dataset.targets["continuous_prototypes"].float()
    metadata = {
        "sample_ids": list(dataset.targets["sample_ids"]),
        "target_identity": dataset.targets["target_identity"].clone(),
        "scenarios": list(dataset.targets["scenarios"]),
        "teacher_mode": str(dataset.targets["teacher_mode"]),
    }
    if tuple(anchors.shape) != tuple(teachers.shape):
        raise RuntimeError("quality anchors and teacher maps are misaligned")
    return teachers, anchors, metadata


def prepare_residual_artifacts(
    *,
    config: Mapping[str, Any],
    dataset_root: str | Path,
    absolute_quantization_root: str | Path,
    artifact_root: str | Path,
    resume: bool = False,
) -> ResidualArtifactResult:
    """Fit one raw Euclidean codebook on train and encode train/val targets."""

    source_dataset = Path(dataset_root).expanduser().resolve()
    source_quantization = Path(
        absolute_quantization_root
    ).expanduser().resolve()
    destination = Path(artifact_root).expanduser().resolve()
    source_dataset_manifest = source_dataset / "manifest.json"
    source_quantization_manifest = source_quantization / "manifest.json"
    source_projection = source_quantization / "projection.pt"
    for required in (
        source_dataset_manifest,
        source_quantization_manifest,
        source_projection,
        source_quantization / "train_targets.pt",
        source_quantization / "val_targets.pt",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"required P2-1 input is missing: {required}")

    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"residual artifacts already exist: {destination}"
            )
        manifest = json.loads(
            (destination / "manifest.json").read_text(encoding="utf-8")
        )
        validation = validate_residual_artifacts(
            destination,
            dataset_root=source_dataset,
        )
        if not validation["passed"]:
            raise ValueError("existing residual artifacts failed validation")
        return ResidualArtifactResult(
            root=destination,
            manifest=manifest,
            validation=validation,
            reused=True,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        train_teacher, train_anchor, train_metadata = _split_inputs(
            dataset_root=source_dataset,
            absolute_quantization_root=source_quantization,
            split="train",
        )
        p2 = config["p2_1"]
        codebook, fit_details = fit_residual_codebook(
            train_teacher,
            train_anchor,
            codebook_size=int(p2["codebook_size"]),
            max_fit_tokens=int(p2["max_fit_tokens"]),
            batch_size=int(p2["batch_size"]),
            iterations=int(p2["iterations"]),
            n_init=int(p2["n_init"]),
            seed=int(config["project"]["seed"]),
            fit_split="train",
            device=str(p2["fit_device"]),
        )
        val_teacher, val_anchor, val_metadata = _split_inputs(
            dataset_root=source_dataset,
            absolute_quantization_root=source_quantization,
            split="val",
        )

        shutil.copy2(source_projection, temporary / "projection.pt")
        torch.save(codebook.to_payload(), temporary / "codebook.pt")
        metrics: dict[str, Any] = {}
        split_values = {
            "train": (train_teacher, train_anchor, train_metadata),
            "val": (val_teacher, val_anchor, val_metadata),
        }
        for split, (teacher, anchor, metadata) in split_values.items():
            tokens, squared_distance = encode_teacher_residuals(
                teacher,
                anchor,
                codebook,
                chunk_size=int(p2["assignment_chunk_size"]),
            )
            metrics[split] = compute_residual_quantization_metrics(
                teacher_map=teacher,
                quality_anchor=anchor,
                tokens=tokens,
                codebook=codebook,
                scenarios=metadata["scenarios"],
            )
            metrics[split]["mean_assignment_squared_distance"] = float(
                squared_distance.mean().item()
            )
            torch.save(
                {
                    "schema_version": 1,
                    "split": split,
                    "teacher_mode": metadata["teacher_mode"],
                    "representation": "quality_anchor_plus_raw_residual",
                    "continuous_prototypes": teacher,
                    "tokens": tokens.cpu(),
                    "target_identity": metadata["target_identity"],
                    "sample_ids": metadata["sample_ids"],
                    "scenarios": metadata["scenarios"],
                },
                temporary / f"{split}_targets.pt",
            )

        files: dict[str, Any] = {}
        for name in (
            "projection.pt",
            "codebook.pt",
            "train_targets.pt",
            "val_targets.pt",
        ):
            path = temporary / name
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "stage": "p2-1-residual-quantization",
            "protocol": "validation_only",
            "selection_split": "val",
            "test_dataset_constructed": False,
            "test_artifacts_constructed": False,
            "fit_split": "train",
            "config_sha256": config_sha256(config),
            "source_dataset": str(source_dataset),
            "source_dataset_manifest_sha256": _sha256(
                source_dataset_manifest
            ),
            "source_absolute_quantization": str(source_quantization),
            "source_absolute_manifest_sha256": _sha256(
                source_quantization_manifest
            ),
            "source_projection_sha256": _sha256(source_projection),
            "anchor": "quality_pooling",
            "residual": "raw_teacher_minus_anchor",
            "codebook": {
                **fit_details,
                "size": codebook.codebook_size,
                "dimension": codebook.dimension,
            },
            "metrics": metrics,
            "files": files,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    validation = validate_residual_artifacts(
        destination,
        dataset_root=source_dataset,
    )
    if not validation["passed"]:
        raise RuntimeError("new residual artifacts failed validation")
    return ResidualArtifactResult(
        root=destination,
        manifest=manifest,
        validation=validation,
        reused=False,
    )


def validate_residual_artifacts(
    root: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate raw codebook semantics, hashes, target alignment, and policy."""

    artifact_root = Path(root).expanduser().resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"residual manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    codebook = ResidualCodebook.from_payload(
        torch.load(
            artifact_root / "codebook.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    vector_norms = codebook.vectors.norm(dim=-1)
    checks: dict[str, bool] = {
        "fit_split_is_train": manifest.get("fit_split") == "train"
        and codebook.fit_split == "train",
        "validation_only": manifest.get("protocol") == "validation_only",
        "test_dataset_not_constructed": (
            manifest.get("test_dataset_constructed") is False
        ),
        "test_artifacts_not_constructed": (
            manifest.get("test_artifacts_constructed") is False
            and not (artifact_root / "test_targets.pt").exists()
        ),
        "raw_codebook_finite": bool(
            torch.isfinite(codebook.vectors).all().item()
        ),
        "raw_codebook_magnitude_preserved": bool(
            (vector_norms.sub(1.0).abs() > 1e-3).any().item()
        ),
        "projection_is_exact_source_copy": (
            _sha256(artifact_root / "projection.pt")
            == str(manifest["source_projection_sha256"])
        ),
    }
    file_checks: dict[str, bool] = {}
    for name, details in manifest["files"].items():
        path = artifact_root / name
        file_checks[name] = (
            path.is_file()
            and path.stat().st_size == int(details["size_bytes"])
            and _sha256(path) == str(details["sha256"])
        )
    checks["file_hashes"] = all(file_checks.values())

    target_details: dict[str, Any] = {}
    for split in ("train", "val"):
        payload = torch.load(
            artifact_root / f"{split}_targets.pt",
            map_location="cpu",
            weights_only=True,
        )
        tokens = payload["tokens"]
        split_checks = {
            "split": payload.get("split") == split,
            "representation": (
                payload.get("representation")
                == "quality_anchor_plus_raw_residual"
            ),
            "shape": tokens.ndim == 2
            and int(tokens.shape[1]) == 49
            and tuple(payload["continuous_prototypes"].shape)
            == (int(tokens.shape[0]), 49, codebook.dimension),
            "token_range": int(tokens.min().item()) >= 0
            and int(tokens.max().item()) < codebook.codebook_size,
            "sample_count": len(payload["sample_ids"]) == int(tokens.shape[0]),
        }
        if dataset_root is not None:
            dataset = TokenTrainingDataset(
                dataset_root=dataset_root,
                quantization_root=artifact_root,
                split=split,
                precompute=False,
            )
            split_checks["dataset_alignment"] = len(dataset) == int(
                tokens.shape[0]
            )
        target_details[split] = {
            "checks": split_checks,
            "samples": int(tokens.shape[0]),
        }
        checks[f"{split}_targets"] = all(split_checks.values())

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "file_checks": file_checks,
        "targets": target_details,
        "codebook_norm": {
            "mean": float(vector_norms.mean().item()),
            "minimum": float(vector_norms.min().item()),
            "maximum": float(vector_norms.max().item()),
        },
    }
