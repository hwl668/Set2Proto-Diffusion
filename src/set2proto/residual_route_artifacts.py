"""Identity-carved residual codebook artifacts for P2-2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .residual_evidence import quality_weighted_anchor
from .residual_quantization import (
    ResidualCodebook,
    compute_residual_quantization_metrics,
    encode_teacher_residuals,
    fit_residual_codebook,
)
from .training import TokenTrainingDataset


ROUTE_SPLITS = (
    "route_train",
    "route_calibration",
    "route_validation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_split_indices(
    *,
    indices: Mapping[str, Sequence[int]],
    dataset: TokenTrainingDataset,
) -> dict[str, Any]:
    if set(indices) != set(ROUTE_SPLITS):
        raise ValueError("P2-2 route split names are incomplete")
    normalized = {
        split: [int(value) for value in indices[split]]
        for split in ROUTE_SPLITS
    }
    flattened = [value for split in ROUTE_SPLITS for value in normalized[split]]
    if (
        len(flattened) != len(dataset)
        or len(set(flattened)) != len(dataset)
        or set(flattened) != set(range(len(dataset)))
    ):
        raise ValueError("P2-2 route indices must partition the train dataset")
    identities = dataset.targets["target_identity"].to(torch.int64)
    identity_sets = {
        split: set(identities[normalized[split]].tolist())
        for split in ROUTE_SPLITS
    }
    intersections = {
        f"{left}__{right}": sorted(identity_sets[left] & identity_sets[right])
        for index, left in enumerate(ROUTE_SPLITS)
        for right in ROUTE_SPLITS[index + 1 :]
    }
    if any(intersections.values()):
        raise ValueError("P2-2 route identities are not disjoint")
    scenarios = list(dataset.targets["scenarios"])
    scenario_counts = {
        split: {
            scenario: sum(
                scenarios[index] == scenario for index in normalized[split]
            )
            for scenario in sorted(set(scenarios))
        }
        for split in ROUTE_SPLITS
    }
    if any(
        count <= 0
        for split_counts in scenario_counts.values()
        for count in split_counts.values()
    ):
        raise ValueError("every P2-2 split must contain every scenario")
    return {
        "indices": normalized,
        "identity_counts": {
            split: len(values) for split, values in identity_sets.items()
        },
        "sample_counts": {
            split: len(normalized[split]) for split in ROUTE_SPLITS
        },
        "identity_intersections": intersections,
        "scenario_counts": scenario_counts,
    }


def validate_route_residual_artifacts(
    root: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    artifact_root = Path(root).expanduser().resolve()
    manifest_path = artifact_root / "manifest.json"
    required = (
        manifest_path,
        artifact_root / "projection.pt",
        artifact_root / "codebook.pt",
        artifact_root / "train_targets.pt",
        artifact_root / "route_split.json",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("P2-2 residual route artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    codebook = ResidualCodebook.from_payload(
        torch.load(
            artifact_root / "codebook.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    targets = torch.load(
        artifact_root / "train_targets.pt",
        map_location="cpu",
        weights_only=True,
    )
    split = json.loads(
        (artifact_root / "route_split.json").read_text(encoding="utf-8")
    )
    checks = {
        "protocol": (
            manifest.get("protocol")
            == "route_identity_carveout_rollout_stability"
        ),
        "fit_scope": (
            manifest.get("codebook_fit_split") == "route_train"
            and codebook.fit_split == "train"
        ),
        "test_absent": (
            manifest.get("test_artifacts_constructed") is False
            and not (artifact_root / "test_targets.pt").exists()
        ),
        "target_representation": (
            targets.get("representation")
            == "quality_anchor_plus_raw_residual"
            and targets.get("split") == "train"
        ),
        "target_shape": (
            targets["tokens"].ndim == 2
            and int(targets["tokens"].shape[1]) == 49
            and int(targets["continuous_prototypes"].shape[0])
            == int(targets["tokens"].shape[0])
        ),
        "route_partition": (
            set(split["sample_indices"]) == set(ROUTE_SPLITS)
            and sum(
                len(values) for values in split["sample_indices"].values()
            )
            == int(targets["tokens"].shape[0])
        ),
        "identity_disjoint": all(
            not values
            for values in split["identity_intersections"].values()
        ),
        "raw_codebook": (
            bool(torch.isfinite(codebook.vectors).all().item())
            and bool(
                (
                    codebook.vectors.norm(dim=-1).sub(1.0).abs() > 1e-3
                ).any().item()
            )
        ),
    }
    file_checks = {
        name: (
            (artifact_root / name).is_file()
            and _sha256(artifact_root / name) == details["sha256"]
            and (artifact_root / name).stat().st_size
            == int(details["size_bytes"])
        )
        for name, details in manifest["files"].items()
    }
    checks["file_hashes"] = all(file_checks.values())
    if dataset_root is not None:
        source_manifest = (
            Path(dataset_root).expanduser().resolve() / "manifest.json"
        )
        checks["source_dataset"] = (
            source_manifest.is_file()
            and _sha256(source_manifest)
            == manifest["source_dataset_manifest_sha256"]
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "file_checks": file_checks,
        "codebook_shape": list(codebook.vectors.shape),
        "target_shape": list(targets["tokens"].shape),
    }


def prepare_route_residual_artifacts(
    *,
    config: Mapping[str, Any],
    dataset_root: str | Path,
    absolute_quantization_root: str | Path,
    route_split: Mapping[str, Any],
    artifact_root: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Fit the residual codebook only on route-train identities."""

    destination = Path(artifact_root).expanduser().resolve()
    source_dataset = Path(dataset_root).expanduser().resolve()
    source_absolute = Path(
        absolute_quantization_root
    ).expanduser().resolve()
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"P2-2 residual artifacts already exist: {destination}"
            )
        validation = validate_route_residual_artifacts(
            destination,
            dataset_root=source_dataset,
        )
        if not validation["passed"]:
            raise RuntimeError("reused P2-2 residual artifacts failed checks")
        return {
            "root": destination,
            "manifest": json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            ),
            "validation": validation,
            "reused": True,
        }

    dataset = TokenTrainingDataset(
        dataset_root=source_dataset,
        quantization_root=source_absolute,
        split="train",
        precompute=True,
    )
    assert dataset._precomputed is not None
    split_audit = _validate_split_indices(
        indices=route_split["sample_indices"],
        dataset=dataset,
    )
    expected_ids = {
        split: [
            dataset.targets["sample_ids"][index]
            for index in split_audit["indices"][split]
        ]
        for split in ROUTE_SPLITS
    }
    if expected_ids != route_split["sample_ids"]:
        raise RuntimeError("P2-2 route sample IDs changed")
    condition = dataset._precomputed["condition_features"]
    quality = dataset._precomputed["condition_quality"]
    anchor_chunks = [
        quality_weighted_anchor(
            condition[start : start + 64],
            quality[start : start + 64],
        ).cpu()
        for start in range(0, len(dataset), 64)
    ]
    anchors = torch.cat(anchor_chunks)
    teachers = dataset.targets["continuous_prototypes"].float()
    train_indices = torch.tensor(
        split_audit["indices"]["route_train"],
        dtype=torch.int64,
    )
    section = config["p2_2"]
    codebook, fit_details = fit_residual_codebook(
        teachers.index_select(0, train_indices),
        anchors.index_select(0, train_indices),
        codebook_size=int(section["codebook_size"]),
        max_fit_tokens=int(section["max_fit_tokens"]),
        batch_size=int(section["codebook_batch_size"]),
        iterations=int(section["codebook_iterations"]),
        n_init=int(section["codebook_n_init"]),
        seed=int(config["project"]["seed"]) + 72001,
        fit_split="train",
        device=str(section["fit_device"]),
    )
    tokens, distances = encode_teacher_residuals(
        teachers,
        anchors,
        codebook,
        chunk_size=int(section["assignment_chunk_size"]),
    )
    scenario_values = list(dataset.targets["scenarios"])
    metrics: dict[str, Any] = {}
    for split in ROUTE_SPLITS:
        indices = torch.tensor(
            split_audit["indices"][split],
            dtype=torch.int64,
        )
        metrics[split] = compute_residual_quantization_metrics(
            teacher_map=teachers.index_select(0, indices),
            quality_anchor=anchors.index_select(0, indices),
            tokens=tokens.index_select(0, indices),
            codebook=codebook,
            scenarios=[scenario_values[index] for index in indices.tolist()],
        )
        metrics[split]["mean_assignment_squared_distance"] = float(
            distances.index_select(0, indices).mean().item()
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        shutil.copy2(
            source_absolute / "projection.pt",
            temporary / "projection.pt",
        )
        torch.save(codebook.to_payload(), temporary / "codebook.pt")
        torch.save(
            {
                "schema_version": 1,
                "split": "train",
                "teacher_mode": dataset.targets["teacher_mode"],
                "representation": "quality_anchor_plus_raw_residual",
                "continuous_prototypes": teachers,
                "tokens": tokens.cpu(),
                "target_identity": dataset.targets["target_identity"],
                "sample_ids": list(dataset.targets["sample_ids"]),
                "scenarios": scenario_values,
            },
            temporary / "train_targets.pt",
        )
        route_payload = {
            "schema_version": 1,
            "sample_indices": split_audit["indices"],
            "sample_ids": expected_ids,
            "identity_counts": split_audit["identity_counts"],
            "sample_counts": split_audit["sample_counts"],
            "identity_intersections": split_audit[
                "identity_intersections"
            ],
            "scenario_counts": split_audit["scenario_counts"],
        }
        _atomic_json(temporary / "route_split.json", route_payload)
        files: dict[str, Any] = {}
        for name in (
            "projection.pt",
            "codebook.pt",
            "train_targets.pt",
            "route_split.json",
        ):
            path = temporary / name
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        source_manifest = source_dataset / "manifest.json"
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "stage": "p2-2-route-residual-quantization",
            "protocol": "route_identity_carveout_rollout_stability",
            "source_dataset": str(source_dataset),
            "source_dataset_manifest_sha256": _sha256(source_manifest),
            "source_absolute_quantization": str(source_absolute),
            "source_projection_sha256": _sha256(
                source_absolute / "projection.pt"
            ),
            "codebook_fit_split": "route_train",
            "test_artifacts_constructed": False,
            "anchor": "quality_pooling",
            "residual": "raw_teacher_minus_anchor",
            "codebook": {
                **fit_details,
                "size": codebook.codebook_size,
                "dimension": codebook.dimension,
            },
            "split_audit": {
                key: value
                for key, value in split_audit.items()
                if key != "indices"
            },
            "metrics": metrics,
            "files": files,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validation = validate_route_residual_artifacts(
        destination,
        dataset_root=source_dataset,
    )
    if not validation["passed"]:
        raise RuntimeError("new P2-2 residual artifacts failed validation")
    return {
        "root": destination,
        "manifest": manifest,
        "validation": validation,
        "reused": False,
    }
