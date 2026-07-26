"""Identity-disjoint synthetic spatial-feature data for end-to-end MVP smoke tests."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .environment import config_sha256
from .reproducibility import make_dataloader_generator, seed_dataloader_worker


SYNTHETIC_SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SyntheticPreparationResult:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    validation: dict[str, Any]
    reused: bool


def _unit_normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), p=2.0, dim=-1, eps=1e-6)


def _random_unit(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return _unit_normalize(torch.randn(shape, generator=generator))


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


def _load_tensor_file(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid synthetic feature bank: {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected an object at {path}:{line_number}"
                )
            yield value


def _generate_feature_bank(
    *,
    split: str,
    identity_count: int,
    identity_offset: int,
    images_per_identity: int,
    tokens: int,
    feature_dim: int,
    seed: int,
    position_basis: torch.Tensor,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    identity_vectors = _random_unit(
        (identity_count, feature_dim),
        generator=generator,
    )
    identity_local = _random_unit(
        (identity_count, tokens, feature_dim),
        generator=generator,
    )
    identity_maps = _unit_normalize(
        identity_vectors[:, None, :]
        + 0.35 * identity_local
        + 0.15 * position_basis[None, :, :]
    )

    image_count = identity_count * images_per_identity
    local_identity_ids = torch.arange(identity_count).repeat_interleave(
        images_per_identity
    )
    identity_ids = local_identity_ids + identity_offset
    image_indices = torch.arange(image_count, dtype=torch.int64)
    qualities = 0.35 + 0.65 * torch.rand(image_count, generator=generator)
    quality_noise = 1.5 * torch.randn(image_count, generator=generator)
    quality_norms = (25.0 + 75.0 * qualities + quality_noise).clamp_min(1.0)
    features = torch.empty(
        (image_count, tokens, feature_dim),
        dtype=torch.float16,
    )

    chunk_size = 256
    for start in range(0, image_count, chunk_size):
        end = min(start + chunk_size, image_count)
        chunk_ids = local_identity_ids[start:end]
        chunk_quality = qualities[start:end, None, None]
        nuisance = _random_unit(
            (end - start, 1, feature_dim),
            generator=generator,
        )
        local_noise = _random_unit(
            (end - start, tokens, feature_dim),
            generator=generator,
        )
        noise_scale = 0.035 + 0.32 * (1.0 - chunk_quality)
        chunk = (
            identity_maps[chunk_ids]
            + 0.18 * (1.0 - chunk_quality) * nuisance
            + noise_scale * local_noise
        )
        features[start:end] = _unit_normalize(chunk).to(torch.float16)

    return {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "source": "synthetic_feature_generator",
        "split": split,
        "features": features,
        "identity_ids": identity_ids.to(torch.int64),
        "image_indices": image_indices,
        "quality": qualities.to(torch.float32),
        "quality_norm": quality_norms.to(torch.float32),
    }


def _build_set_rows(
    *,
    split: str,
    bank: Mapping[str, Any],
    identity_count: int,
    identity_offset: int,
    images_per_identity: int,
    condition_frames: int,
    teacher_frames: int,
    sets_per_identity: int,
    scenarios: list[str],
    wrong_identity_probability: float,
    seed: int,
) -> list[dict[str, Any]]:
    quality = bank["quality"]
    rows: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)

    for local_identity in range(identity_count):
        target_identity = identity_offset + local_identity
        first = local_identity * images_per_identity
        indices = torch.arange(first, first + images_per_identity)
        ranked = indices[torch.argsort(quality[indices], descending=True)]
        teacher_pool = ranked[:teacher_frames]
        condition_pool = ranked[teacher_frames:]

        for set_index in range(sets_per_identity):
            condition_order = torch.randperm(
                len(condition_pool),
                generator=generator,
            )[:condition_frames]
            teacher_order = torch.randperm(
                len(teacher_pool),
                generator=generator,
            )
            condition_indices = condition_pool[condition_order].tolist()
            teacher_indices = teacher_pool[teacher_order].tolist()
            scenario = scenarios[set_index % len(scenarios)]
            sample_seed = int(
                torch.randint(
                    0,
                    2**31 - 1,
                    (1,),
                    generator=generator,
                ).item()
            )

            distractor_indices = [-1] * condition_frames
            wrong_positions: list[int] = []
            if scenario == "wrong_identity" and wrong_identity_probability > 0:
                wrong_count = max(
                    1,
                    int(round(condition_frames * wrong_identity_probability)),
                )
                wrong_count = min(wrong_count, condition_frames)
                wrong_positions = (
                    torch.randperm(
                        condition_frames,
                        generator=generator,
                    )[:wrong_count]
                    .tolist()
                )
                for frame_position in wrong_positions:
                    foreign_offset = int(
                        torch.randint(
                            1,
                            identity_count,
                            (1,),
                            generator=generator,
                        ).item()
                    )
                    foreign_local_identity = (
                        local_identity + foreign_offset
                    ) % identity_count
                    foreign_image = int(
                        torch.randint(
                            0,
                            images_per_identity,
                            (1,),
                            generator=generator,
                        ).item()
                    )
                    distractor_indices[frame_position] = (
                        foreign_local_identity * images_per_identity
                        + foreign_image
                    )

            rows.append(
                {
                    "schema_version": SYNTHETIC_SCHEMA_VERSION,
                    "sample_id": (
                        f"{split}-id{target_identity:05d}-set{set_index:03d}"
                    ),
                    "split": split,
                    "target_identity": target_identity,
                    "scenario": scenario,
                    "sample_seed": sample_seed,
                    "condition_indices": condition_indices,
                    "teacher_indices": teacher_indices,
                    "wrong_positions": wrong_positions,
                    "distractor_indices": distractor_indices,
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _rectangle_visibility(
    *,
    height: int,
    width: int,
    generator: torch.Generator,
    min_fraction: float,
    max_fraction: float,
) -> torch.Tensor:
    min_height = max(1, int(round(height * min_fraction)))
    max_height = max(min_height, int(round(height * max_fraction)))
    min_width = max(1, int(round(width * min_fraction)))
    max_width = max(min_width, int(round(width * max_fraction)))
    rectangle_height = int(
        torch.randint(
            min_height,
            min(max_height, height) + 1,
            (1,),
            generator=generator,
        ).item()
    )
    rectangle_width = int(
        torch.randint(
            min_width,
            min(max_width, width) + 1,
            (1,),
            generator=generator,
        ).item()
    )
    top = int(
        torch.randint(
            0,
            height - rectangle_height + 1,
            (1,),
            generator=generator,
        ).item()
    )
    left = int(
        torch.randint(
            0,
            width - rectangle_width + 1,
            (1,),
            generator=generator,
        ).item()
    )
    visible = torch.ones((height, width), dtype=torch.bool)
    visible[
        top : top + rectangle_height,
        left : left + rectangle_width,
    ] = False
    return visible.flatten()


def _complementary_visibility(
    frame_index: int,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    row_boundary = (height + 1) // 2
    column_boundary = (width + 1) // 2
    quadrant = frame_index % 4
    row_slice = (
        slice(0, row_boundary)
        if quadrant < 2
        else slice(row_boundary, height)
    )
    column_slice = (
        slice(0, column_boundary)
        if quadrant % 2 == 0
        else slice(column_boundary, width)
    )
    visible = torch.zeros((height, width), dtype=torch.bool)
    visible[row_slice, column_slice] = True
    return visible.flatten()


def _replace_invisible(
    features: torch.Tensor,
    visible: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    result = features.clone().float()
    invisible = ~visible
    if invisible.any():
        result[invisible] = _random_unit(
            (int(invisible.sum().item()), result.shape[-1]),
            generator=generator,
        )
    return _unit_normalize(result)


def _spatial_average(
    features: torch.Tensor,
    *,
    height: int,
    width: int,
    kernel: tuple[int, int],
) -> torch.Tensor:
    grid = features.reshape(height, width, -1).permute(2, 0, 1)[None]
    pad_height = kernel[0] // 2
    pad_width = kernel[1] // 2
    padded = F.pad(
        grid,
        (pad_width, pad_width, pad_height, pad_height),
        mode="replicate",
    )
    blurred = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    return _unit_normalize(
        blurred[0].permute(1, 2, 0).reshape(height * width, -1)
    )


def _apply_low_quality(
    features: torch.Tensor,
    frame_index: int,
    *,
    height: int,
    width: int,
    corruptions: Mapping[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float, str]:
    degradation = ("gaussian_blur", "motion_blur", "low_light", "jpeg")[
        frame_index % 4
    ]
    result = features.float()
    quality_multiplier = 1.0

    if degradation == "gaussian_blur" and corruptions["gaussian_blur"]:
        result = _spatial_average(
            result,
            height=height,
            width=width,
            kernel=(3, 3),
        )
        quality_multiplier = 0.75
    elif degradation == "motion_blur" and corruptions["motion_blur"]:
        kernel = (1, 3) if frame_index % 2 == 0 else (3, 1)
        result = _spatial_average(
            result,
            height=height,
            width=width,
            kernel=kernel,
        )
        quality_multiplier = 0.70
    elif degradation == "low_light" and corruptions["low_light"]:
        noise = _random_unit(tuple(result.shape), generator=generator)
        result = _unit_normalize(0.62 * result + 0.38 * noise)
        quality_multiplier = 0.55
    elif degradation == "jpeg" and corruptions["jpeg"]:
        result = _unit_normalize(torch.round(result * 24.0) / 24.0)
        quality_multiplier = 0.80

    visible = torch.ones(height * width, dtype=torch.bool)
    if corruptions["random_occlusion"]:
        visible = _rectangle_visibility(
            height=height,
            width=width,
            generator=generator,
            min_fraction=0.15,
            max_fraction=0.35,
        )
        result = _replace_invisible(
            result,
            visible,
            generator=generator,
        )
        quality_multiplier *= float(visible.float().mean().item())
    return result, visible, quality_multiplier, degradation


class SyntheticSetDataset(Dataset[dict[str, Any]]):
    """Load a clean feature bank and deterministically synthesize set corruption."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        teacher_mode: str = "disjoint",
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if teacher_mode not in {"disjoint", "self"}:
            raise ValueError("teacher_mode must be 'disjoint' or 'self'")
        self.root = Path(root).expanduser().resolve()
        with (self.root / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.split = split
        self.teacher_mode = teacher_mode
        self.bank = _load_tensor_file(self.root / f"{split}_features.pt")
        self.rows = list(_iter_jsonl(self.root / f"{split}_sets.jsonl"))

        data_config = self.manifest["data_config"]
        grid = data_config["feature_grid"]
        self.height = int(grid["height"])
        self.width = int(grid["width"])
        self.tokens = self.height * self.width
        self.condition_frames = int(data_config["condition_frames"])
        self.corruptions = data_config["corruptions"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        generator = torch.Generator().manual_seed(int(row["sample_seed"]))
        condition_indices = torch.tensor(
            row["condition_indices"],
            dtype=torch.int64,
        )
        clean_condition = self.bank["features"][condition_indices].float()
        condition = clean_condition.clone()
        condition_quality = self.bank["quality_norm"][condition_indices].clone()
        source_identity_ids = self.bank["identity_ids"][
            condition_indices
        ].clone()
        visibility = torch.ones(
            (self.condition_frames, self.tokens),
            dtype=torch.float32,
        )
        degradation_labels = ["none"] * self.condition_frames
        scenario = str(row["scenario"])

        if scenario == "low_quality":
            for frame_index in range(self.condition_frames):
                (
                    condition[frame_index],
                    frame_visibility,
                    quality_multiplier,
                    label,
                ) = _apply_low_quality(
                    condition[frame_index],
                    frame_index,
                    height=self.height,
                    width=self.width,
                    corruptions=self.corruptions,
                    generator=generator,
                )
                visibility[frame_index] = frame_visibility.float()
                condition_quality[frame_index] *= quality_multiplier
                degradation_labels[frame_index] = label

        elif scenario == "complementary_occlusion":
            if self.corruptions["complementary_occlusion"]:
                for frame_index in range(self.condition_frames):
                    frame_visibility = _complementary_visibility(
                        frame_index,
                        height=self.height,
                        width=self.width,
                    )
                    condition[frame_index] = _replace_invisible(
                        condition[frame_index],
                        frame_visibility,
                        generator=generator,
                    )
                    visibility[frame_index] = frame_visibility.float()
                    condition_quality[frame_index] *= float(
                        frame_visibility.float().mean().item()
                    )
                    degradation_labels[frame_index] = (
                        "complementary_occlusion"
                    )

        elif scenario == "common_occlusion":
            if self.corruptions["common_occlusion"]:
                shared_visibility = _rectangle_visibility(
                    height=self.height,
                    width=self.width,
                    generator=generator,
                    min_fraction=0.30,
                    max_fraction=0.55,
                )
                for frame_index in range(self.condition_frames):
                    condition[frame_index] = _replace_invisible(
                        condition[frame_index],
                        shared_visibility,
                        generator=generator,
                    )
                    visibility[frame_index] = shared_visibility.float()
                    condition_quality[frame_index] *= float(
                        shared_visibility.float().mean().item()
                    )
                    degradation_labels[frame_index] = "common_occlusion"

        elif scenario == "wrong_identity":
            for frame_position in row["wrong_positions"]:
                distractor_index = int(
                    row["distractor_indices"][frame_position]
                )
                condition[frame_position] = self.bank["features"][
                    distractor_index
                ].float()
                condition_quality[frame_position] = self.bank["quality_norm"][
                    distractor_index
                ]
                source_identity_ids[frame_position] = self.bank["identity_ids"][
                    distractor_index
                ]
                visibility[frame_position].zero_()
                degradation_labels[frame_position] = "wrong_identity"

        elif scenario != "clean":
            raise ValueError(f"unsupported synthetic scenario: {scenario}")

        if self.teacher_mode == "disjoint":
            teacher_indices = torch.tensor(
                row["teacher_indices"],
                dtype=torch.int64,
            )
            teacher_source = "disjoint"
        else:
            teacher_indices = condition_indices
            teacher_source = "self"

        teacher_features = self.bank["features"][teacher_indices].float()
        teacher_quality = self.bank["quality_norm"][teacher_indices].clone()
        return {
            "sample_id": row["sample_id"],
            "split": self.split,
            "scenario": scenario,
            "target_identity": torch.tensor(
                row["target_identity"],
                dtype=torch.int64,
            ),
            "condition_indices": condition_indices,
            "condition_features": _unit_normalize(condition),
            "condition_quality": condition_quality.float(),
            "condition_visibility": visibility,
            "condition_source_identity": source_identity_ids.to(torch.int64),
            "condition_degradation": degradation_labels,
            "teacher_indices": teacher_indices,
            "teacher_features": _unit_normalize(teacher_features),
            "teacher_quality": teacher_quality.float(),
            "teacher_source": teacher_source,
        }


def _create_manifest(
    *,
    root: Path,
    config: Mapping[str, Any],
    profile: str,
    run_id: str,
    split_summaries: Mapping[str, Any],
    files: Mapping[str, Any],
) -> dict[str, Any]:
    data_config = config["data"]
    return {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "generator": "set2proto.synthetic",
        "created_at": datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        ),
        "profile": profile,
        "run_id": run_id,
        "seed": int(config["project"]["seed"]),
        "config_sha256": config_sha256(config),
        "root": str(root),
        "data_config": {
            "feature_grid": data_config["feature_grid"],
            "condition_frames": data_config["condition_frames"],
            "teacher_frames": data_config["teacher_frames"],
            "input_size": data_config["input_size"],
            "synthetic": data_config["synthetic"],
            "corruptions": data_config["corruptions"],
        },
        "scenarios": list(config["evaluation"]["scenarios"]),
        "splits": dict(split_summaries),
        "files": dict(files),
    }


def prepare_synthetic_dataset(
    *,
    config: Mapping[str, Any],
    profile: str,
    run_id: str,
    resume: bool = False,
) -> SyntheticPreparationResult:
    """Generate clean banks and deterministic S/T set manifests."""

    dataset_root = (
        Path(config["paths"]["data_dir"]).resolve()
        / "synthetic"
        / run_id
    )
    manifest_path = dataset_root / "manifest.json"
    if dataset_root.exists():
        if not resume:
            raise FileExistsError(
                "synthetic dataset already exists and will not be overwritten: "
                f"{dataset_root}"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"cannot resume incomplete dataset without manifest: {dataset_root}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("config_sha256") != config_sha256(config):
            raise ValueError(
                "cannot resume synthetic data created from a different "
                "resolved configuration"
            )
        validation = validate_synthetic_dataset(dataset_root)
        return SyntheticPreparationResult(
            root=dataset_root,
            manifest_path=manifest_path,
            manifest=manifest,
            validation=validation,
            reused=True,
        )

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = dataset_root.parent / (
        f".{dataset_root.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary_root.mkdir(parents=False, exist_ok=False)

    data_config = config["data"]
    synthetic_config = data_config["synthetic"]
    grid = data_config["feature_grid"]
    height = int(grid["height"])
    width = int(grid["width"])
    tokens = height * width
    feature_dim = int(grid["backbone_dim"])
    base_seed = int(config["project"]["seed"])
    position_generator = torch.Generator().manual_seed(base_seed + 991)
    position_basis = _random_unit(
        (tokens, feature_dim),
        generator=position_generator,
    )
    scenarios = list(config["evaluation"]["scenarios"])

    split_identity_counts = {
        "train": int(synthetic_config["train_identities"]),
        "val": int(synthetic_config["val_identities"]),
        "test": int(synthetic_config["test_identities"]),
    }
    split_summaries: dict[str, Any] = {}
    files: dict[str, Any] = {}
    identity_offset = 0

    for split_index, split in enumerate(SPLITS):
        identity_count = split_identity_counts[split]
        bank = _generate_feature_bank(
            split=split,
            identity_count=identity_count,
            identity_offset=identity_offset,
            images_per_identity=int(
                synthetic_config["images_per_identity"]
            ),
            tokens=tokens,
            feature_dim=feature_dim,
            seed=base_seed + 1000 * (split_index + 1),
            position_basis=position_basis,
        )
        feature_filename = f"{split}_features.pt"
        feature_path = temporary_root / feature_filename
        torch.save(bank, feature_path)

        rows = _build_set_rows(
            split=split,
            bank=bank,
            identity_count=identity_count,
            identity_offset=identity_offset,
            images_per_identity=int(
                synthetic_config["images_per_identity"]
            ),
            condition_frames=int(data_config["condition_frames"]),
            teacher_frames=int(data_config["teacher_frames"]),
            sets_per_identity=int(
                synthetic_config["sets_per_identity"]
            ),
            scenarios=scenarios,
            wrong_identity_probability=float(
                data_config["corruptions"][
                    "wrong_identity_probability"
                ]
            ),
            seed=base_seed + 1000 * (split_index + 1) + 17,
        )
        sets_filename = f"{split}_sets.jsonl"
        sets_path = temporary_root / sets_filename
        _write_jsonl(sets_path, rows)

        identity_min = identity_offset
        identity_max = identity_offset + identity_count - 1
        scenario_counts = {
            scenario: sum(row["scenario"] == scenario for row in rows)
            for scenario in scenarios
        }
        split_summaries[split] = {
            "identities": identity_count,
            "identity_min": identity_min,
            "identity_max": identity_max,
            "images": int(bank["features"].shape[0]),
            "sets": len(rows),
            "feature_shape": list(bank["features"].shape),
            "feature_dtype": str(bank["features"].dtype).removeprefix(
                "torch."
            ),
            "scenario_counts": scenario_counts,
        }
        files[feature_filename] = {
            "size_bytes": feature_path.stat().st_size,
            "sha256": _file_sha256(feature_path),
        }
        files[sets_filename] = {
            "size_bytes": sets_path.stat().st_size,
            "sha256": _file_sha256(sets_path),
        }
        identity_offset += identity_count

    future_root = dataset_root
    manifest = _create_manifest(
        root=future_root,
        config=config,
        profile=profile,
        run_id=run_id,
        split_summaries=split_summaries,
        files=files,
    )
    _atomic_json(temporary_root / "manifest.json", manifest)
    os.replace(temporary_root, dataset_root)

    validation = validate_synthetic_dataset(dataset_root)
    return SyntheticPreparationResult(
        root=dataset_root,
        manifest_path=dataset_root / "manifest.json",
        manifest=manifest,
        validation=validation,
        reused=False,
    )


def validate_synthetic_dataset(root: str | Path) -> dict[str, Any]:
    """Validate identity splits, S/T separation, shapes, and scenario semantics."""

    dataset_root = Path(root).expanduser().resolve()
    with (dataset_root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    data_config = manifest["data_config"]
    grid = data_config["feature_grid"]
    tokens = int(grid["height"]) * int(grid["width"])
    feature_dim = int(grid["backbone_dim"])
    condition_frames = int(data_config["condition_frames"])
    teacher_frames = int(data_config["teacher_frames"])

    identity_sets: dict[str, set[int]] = {}
    split_results: dict[str, Any] = {}
    all_checks: dict[str, bool] = {}

    for split in SPLITS:
        bank = _load_tensor_file(dataset_root / f"{split}_features.pt")
        rows = list(_iter_jsonl(dataset_root / f"{split}_sets.jsonl"))
        features = bank["features"]
        identities = set(int(value) for value in bank["identity_ids"].tolist())
        identity_sets[split] = identities
        split_checks = {
            "feature_rank": features.ndim == 3,
            "feature_tokens": int(features.shape[1]) == tokens,
            "feature_dim": int(features.shape[2]) == feature_dim,
            "feature_dtype_fp16": features.dtype == torch.float16,
            "features_finite": bool(torch.isfinite(features).all().item()),
            "set_count_matches_manifest": len(rows)
            == int(manifest["splits"][split]["sets"]),
            "condition_teacher_disjoint": True,
            "indices_match_target_identity": True,
            "teacher_quality_not_lower": True,
        }
        scenario_counts = {scenario: 0 for scenario in manifest["scenarios"]}
        representative_rows: dict[str, int] = {}

        for row_index, row in enumerate(rows):
            scenario = str(row["scenario"])
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            representative_rows.setdefault(scenario, row_index)
            condition_indices = set(int(value) for value in row["condition_indices"])
            teacher_indices = set(int(value) for value in row["teacher_indices"])
            if condition_indices.intersection(teacher_indices):
                split_checks["condition_teacher_disjoint"] = False
            if (
                len(condition_indices) != condition_frames
                or len(teacher_indices) != teacher_frames
            ):
                split_checks["condition_teacher_disjoint"] = False
            target = int(row["target_identity"])
            selected = list(condition_indices | teacher_indices)
            if any(
                int(bank["identity_ids"][selected_index].item()) != target
                for selected_index in selected
            ):
                split_checks["indices_match_target_identity"] = False
            condition_mean = float(
                bank["quality_norm"][list(condition_indices)].mean().item()
            )
            teacher_mean = float(
                bank["quality_norm"][list(teacher_indices)].mean().item()
            )
            if teacher_mean + 1e-6 < condition_mean:
                split_checks["teacher_quality_not_lower"] = False

        split_checks["all_scenarios_present"] = all(
            scenario_counts.get(scenario, 0) > 0
            for scenario in manifest["scenarios"]
        )

        dataset = SyntheticSetDataset(dataset_root, split)
        scenario_semantics: dict[str, bool] = {}
        for scenario, row_index in representative_rows.items():
            sample = dataset[row_index]
            base_shape_ok = (
                tuple(sample["condition_features"].shape)
                == (condition_frames, tokens, feature_dim)
                and tuple(sample["teacher_features"].shape)
                == (teacher_frames, tokens, feature_dim)
                and bool(
                    torch.isfinite(sample["condition_features"]).all().item()
                )
            )
            semantic_ok = base_shape_ok
            visibility = sample["condition_visibility"]
            if scenario == "complementary_occlusion":
                semantic_ok = semantic_ok and bool(
                    (visibility.sum(dim=0) > 0).all().item()
                )
                semantic_ok = semantic_ok and bool(
                    (visibility < 1).any().item()
                )
            elif scenario == "common_occlusion":
                semantic_ok = semantic_ok and bool(
                    torch.equal(
                        visibility,
                        visibility[0:1].expand_as(visibility),
                    )
                )
                semantic_ok = semantic_ok and bool(
                    (visibility[0] == 0).any().item()
                )
            elif scenario == "wrong_identity":
                source_ids = sample["condition_source_identity"]
                target = sample["target_identity"]
                expected_wrong = len(rows[row_index]["wrong_positions"])
                actual_wrong = int((source_ids != target).sum().item())
                semantic_ok = semantic_ok and actual_wrong == expected_wrong
            elif scenario == "low_quality":
                semantic_ok = semantic_ok and bool(
                    (visibility < 1).any().item()
                )
                semantic_ok = semantic_ok and all(
                    label != "none"
                    for label in sample["condition_degradation"]
                )
            scenario_semantics[scenario] = semantic_ok
        split_checks["scenario_semantics"] = all(
            scenario_semantics.values()
        )

        split_results[split] = {
            "checks": split_checks,
            "scenario_counts": scenario_counts,
            "scenario_semantics": scenario_semantics,
        }
        for key, value in split_checks.items():
            all_checks[f"{split}.{key}"] = bool(value)

    all_checks["identity_disjoint.train_val"] = identity_sets[
        "train"
    ].isdisjoint(identity_sets["val"])
    all_checks["identity_disjoint.train_test"] = identity_sets[
        "train"
    ].isdisjoint(identity_sets["test"])
    all_checks["identity_disjoint.val_test"] = identity_sets["val"].isdisjoint(
        identity_sets["test"]
    )
    passed = all(all_checks.values())
    return {
        "passed": passed,
        "checks": all_checks,
        "splits": split_results,
    }


def probe_synthetic_dataloader(
    *,
    root: str | Path,
    batch_size: int,
    seed: int,
    precision: str,
) -> dict[str, Any]:
    """Load a real batch, transfer it to CUDA, and run a finite pooling probe."""

    dataset = SyntheticSetDataset(root, "train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=make_dataloader_generator(seed),
        worker_init_fn=seed_dataloader_worker,
    )
    batch = next(iter(loader))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        condition = batch["condition_features"].to(
            device=device,
            dtype=dtype,
            non_blocking=device.type == "cuda",
        )
        teacher = batch["teacher_features"].to(
            device=device,
            dtype=dtype,
            non_blocking=device.type == "cuda",
        )
        condition_prototype = F.normalize(
            condition.float().mean(dim=(1, 2)),
            dim=-1,
        )
        teacher_prototype = F.normalize(
            teacher.float().mean(dim=(1, 2)),
            dim=-1,
        )
        cosine = F.cosine_similarity(
            condition_prototype,
            teacher_prototype,
            dim=-1,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {
            "success": bool(torch.isfinite(cosine).all().item()),
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "condition_shape": list(condition.shape),
            "teacher_shape": list(teacher.shape),
            "mean_condition_teacher_cosine": float(
                cosine.float().mean().item()
            ),
            "peak_allocated_mib": (
                round(torch.cuda.max_memory_allocated(device) / 1024**2, 3)
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_mib": (
                round(torch.cuda.max_memory_reserved(device) / 1024**2, 3)
                if device.type == "cuda"
                else 0.0
            ),
        }
        del condition, teacher, condition_prototype, teacher_prototype, cosine
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result

