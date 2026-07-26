"""Identity-disjoint real-image S/T sets with image-space perturbations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from .adaface_backbone import (
    align_and_normalize_bgr,
    discover_spatial_hook,
    extract_spatial_and_embedding,
    load_adaface_ir50,
)
from .environment import config_sha256
from .real_features import load_selected_landmarks
from .synthetic import SPLITS, _iter_jsonl


REAL_SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RealSetPreparationResult:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    validation: dict[str, Any]
    reused: bool


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def _read_cache_index(cache_root: Path, split: str) -> list[dict[str, str]]:
    path = cache_root / split / "index.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row_index, row in enumerate(rows):
        if int(row["row_index"]) != row_index or row["split"] != split:
            raise ValueError(f"invalid feature-cache index ordering: {path}")
    return rows


def _build_rows(
    *,
    split: str,
    records: list[dict[str, str]],
    quality_norms: np.ndarray,
    condition_frames: int,
    teacher_frames: int,
    sets_per_identity: int,
    scenarios: list[str],
    wrong_identity_probability: float,
    seed: int,
    max_identities: int,
) -> list[dict[str, Any]]:
    groups: dict[int, list[int]] = {}
    for row_index, row in enumerate(records):
        groups.setdefault(int(row["identity_id"]), []).append(row_index)
    identity_ids = sorted(groups)
    if max_identities > 0:
        identity_ids = identity_ids[:max_identities]
    if len(identity_ids) < 2:
        raise ValueError(f"{split} needs at least two identities for real sets")
    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    required = condition_frames + teacher_frames

    for identity_id in identity_ids:
        indices = np.asarray(groups[identity_id], dtype=np.int64)
        if len(indices) < required:
            raise ValueError(
                f"identity {identity_id} has {len(indices)} images, "
                f"needs at least {required}"
            )
        order = indices[
            np.argsort(quality_norms[indices], kind="stable")[::-1]
        ]
        teacher_pool = order[:teacher_frames]
        condition_pool = order[teacher_frames:]
        for set_index in range(sets_per_identity):
            condition_indices = generator.choice(
                condition_pool,
                size=condition_frames,
                replace=False,
            )
            teacher_indices = generator.permutation(teacher_pool)
            scenario = scenarios[set_index % len(scenarios)]
            sample_seed = int(generator.integers(0, 2**31 - 1))
            source_indices = condition_indices.copy()
            wrong_positions: list[int] = []
            distractor_indices = [-1] * condition_frames
            if scenario == "wrong_identity" and wrong_identity_probability > 0:
                wrong_count = max(
                    1,
                    int(round(condition_frames * wrong_identity_probability)),
                )
                wrong_count = min(wrong_count, condition_frames)
                wrong_positions = [
                    int(value)
                    for value in generator.choice(
                        condition_frames,
                        size=wrong_count,
                        replace=False,
                    )
                ]
                foreign_identities = [
                    value for value in identity_ids if value != identity_id
                ]
                for position in wrong_positions:
                    foreign_identity = int(generator.choice(foreign_identities))
                    distractor = int(generator.choice(groups[foreign_identity]))
                    distractor_indices[position] = distractor
                    source_indices[position] = distractor

            source_identity_ids = [
                int(records[int(index)]["identity_id"])
                for index in source_indices
            ]
            if scenario == "low_quality":
                degradation = [
                    "gaussian_blur",
                    "motion_blur",
                    "low_light",
                    "jpeg",
                ][:condition_frames]
            else:
                degradation = [scenario] * condition_frames
                if scenario == "clean":
                    degradation = ["none"] * condition_frames
            rows.append(
                {
                    "schema_version": REAL_SET_SCHEMA_VERSION,
                    "sample_id": (
                        f"{split}-id{identity_id:05d}-set{set_index:03d}"
                    ),
                    "split": split,
                    "target_identity": identity_id,
                    "scenario": scenario,
                    "sample_seed": sample_seed,
                    "condition_indices": [
                        int(value) for value in condition_indices
                    ],
                    "teacher_indices": [
                        int(value) for value in teacher_indices
                    ],
                    "source_indices": [
                        int(value) for value in source_indices
                    ],
                    "source_identity_ids": source_identity_ids,
                    "wrong_positions": wrong_positions,
                    "distractor_indices": distractor_indices,
                    "condition_degradation": degradation,
                }
            )
    return rows


def _rectangle_visibility(
    *,
    height: int,
    width: int,
    generator: np.random.Generator,
    min_fraction: float,
    max_fraction: float,
) -> torch.Tensor:
    rectangle_height = int(
        generator.integers(
            max(1, math.ceil(height * min_fraction)),
            min(height, max(1, math.ceil(height * max_fraction))) + 1,
        )
    )
    rectangle_width = int(
        generator.integers(
            max(1, math.ceil(width * min_fraction)),
            min(width, max(1, math.ceil(width * max_fraction))) + 1,
        )
    )
    top = int(generator.integers(0, height - rectangle_height + 1))
    left = int(generator.integers(0, width - rectangle_width + 1))
    visible = torch.ones((height, width), dtype=torch.bool)
    visible[
        top : top + rectangle_height,
        left : left + rectangle_width,
    ] = False
    return visible


def _complementary_visibility(
    frame_index: int,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    row_boundary = (height + 1) // 2
    column_boundary = (width + 1) // 2
    visible = torch.zeros((height, width), dtype=torch.bool)
    row_slice = (
        slice(0, row_boundary)
        if frame_index < 2
        else slice(row_boundary, height)
    )
    column_slice = (
        slice(0, column_boundary)
        if frame_index % 2 == 0
        else slice(column_boundary, width)
    )
    visible[row_slice, column_slice] = True
    return visible


def _apply_visibility(
    image: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    height_scale = image.shape[-2] // visibility.shape[0]
    width_scale = image.shape[-1] // visibility.shape[1]
    pixel_visibility = visibility.repeat_interleave(
        height_scale,
        dim=0,
    ).repeat_interleave(width_scale, dim=1)
    result = image.clone()
    result[:, ~pixel_visibility] = -1.0
    return result


def _to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    return (
        ((image.permute(1, 2, 0).numpy() + 1.0) * 127.5)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )


def _from_bgr_uint8(image: np.ndarray) -> torch.Tensor:
    normalized = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(normalized.transpose(2, 0, 1)).contiguous()


def _apply_degradation(
    image: torch.Tensor,
    label: str,
    *,
    generator: np.random.Generator,
) -> torch.Tensor:
    bgr = _to_bgr_uint8(image)
    if label == "gaussian_blur":
        bgr = cv2.GaussianBlur(bgr, (9, 9), sigmaX=2.0)
    elif label == "motion_blur":
        kernel = np.zeros((9, 9), dtype=np.float32)
        if bool(generator.integers(0, 2)):
            kernel[4, :] = 1.0 / 9.0
        else:
            kernel[:, 4] = 1.0 / 9.0
        bgr = cv2.filter2D(bgr, -1, kernel)
    elif label == "low_light":
        factor = float(generator.uniform(0.18, 0.42))
        normalized = np.power(bgr.astype(np.float32) / 255.0, 1.5)
        bgr = np.clip(normalized * factor * 255.0, 0, 255).astype(np.uint8)
    elif label == "jpeg":
        quality = int(generator.integers(10, 36))
        success, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not success:
            raise RuntimeError("OpenCV failed to encode JPEG perturbation")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("OpenCV failed to decode JPEG perturbation")
        bgr = decoded
    else:
        raise ValueError(f"unsupported degradation: {label}")
    return _from_bgr_uint8(bgr)


class _PerturbedSetImageDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, int]]
):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        records: list[dict[str, str]],
        image_directory: Path,
        landmarks: Mapping[str, np.ndarray],
        height: int,
        width: int,
        corruptions: Mapping[str, Any],
        start_index: int,
    ) -> None:
        self.rows = rows
        self.records = records
        self.image_directory = image_directory
        self.landmarks = landmarks
        self.height = height
        self.width = width
        self.corruptions = corruptions
        self.start_index = start_index

    def __len__(self) -> int:
        return len(self.rows) - self.start_index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        absolute_index = self.start_index + index
        row = self.rows[absolute_index]
        generator = np.random.default_rng(int(row["sample_seed"]))
        images = []
        for source_index in row["source_indices"]:
            image_name = self.records[int(source_index)]["image_name"]
            images.append(
                align_and_normalize_bgr(
                    self.image_directory / image_name,
                    self.landmarks[image_name],
                )
            )
        visibility = torch.ones(
            (len(images), self.height, self.width),
            dtype=torch.bool,
        )
        scenario = str(row["scenario"])
        if scenario == "low_quality":
            for frame_index, label in enumerate(
                row["condition_degradation"]
            ):
                if bool(self.corruptions.get(label, False)):
                    images[frame_index] = _apply_degradation(
                        images[frame_index],
                        label,
                        generator=generator,
                    )
                if bool(self.corruptions["random_occlusion"]):
                    frame_visibility = _rectangle_visibility(
                        height=self.height,
                        width=self.width,
                        generator=generator,
                        min_fraction=0.15,
                        max_fraction=0.35,
                    )
                    visibility[frame_index] = frame_visibility
                    images[frame_index] = _apply_visibility(
                        images[frame_index],
                        frame_visibility,
                    )
        elif (
            scenario == "complementary_occlusion"
            and bool(self.corruptions["complementary_occlusion"])
        ):
            for frame_index in range(len(images)):
                frame_visibility = _complementary_visibility(
                    frame_index,
                    height=self.height,
                    width=self.width,
                )
                visibility[frame_index] = frame_visibility
                images[frame_index] = _apply_visibility(
                    images[frame_index],
                    frame_visibility,
                )
        elif (
            scenario == "common_occlusion"
            and bool(self.corruptions["common_occlusion"])
        ):
            shared_visibility = _rectangle_visibility(
                height=self.height,
                width=self.width,
                generator=generator,
                min_fraction=0.30,
                max_fraction=0.55,
            )
            for frame_index in range(len(images)):
                visibility[frame_index] = shared_visibility
                images[frame_index] = _apply_visibility(
                    images[frame_index],
                    shared_visibility,
                )
        elif scenario == "wrong_identity":
            for frame_index in row["wrong_positions"]:
                visibility[int(frame_index)].zero_()
        elif scenario != "clean":
            raise ValueError(f"unsupported real-set scenario: {scenario}")
        return (
            torch.stack(images),
            visibility.flatten(1).to(torch.uint8),
            absolute_index,
        )


class RealSetDataset(Dataset[dict[str, Any]]):
    """Read precomputed image-space corrupted conditions and clean teachers."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        teacher_mode: str = "disjoint",
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if teacher_mode not in {"disjoint", "self"}:
            raise ValueError("teacher_mode must be 'disjoint' or 'self'")
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.split = split
        self.teacher_mode = teacher_mode
        split_directory = self.root / split
        self.rows = list(_iter_jsonl(split_directory / "sets.jsonl"))
        self.condition = np.load(
            split_directory / "condition_features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.condition_quality = np.load(
            split_directory / "condition_quality.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.visibility = np.load(
            split_directory / "condition_visibility.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        cache_root = Path(
            self.manifest["source_feature_cache"]["root"]
        ).expanduser().resolve()
        self.clean_features = np.load(
            cache_root / split / "local_features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.clean_quality = np.load(
            cache_root / split / "feature_norms.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        condition = torch.from_numpy(
            np.array(self.condition[index], copy=True)
        ).float()
        condition_quality = torch.from_numpy(
            np.array(self.condition_quality[index], copy=True)
        ).float()
        if self.teacher_mode == "disjoint":
            teacher_indices = np.asarray(
                row["teacher_indices"],
                dtype=np.int64,
            )
            teacher_features = torch.from_numpy(
                np.array(self.clean_features[teacher_indices], copy=True)
            ).float()
            teacher_quality = torch.from_numpy(
                np.array(self.clean_quality[teacher_indices], copy=True)
            ).float()
            teacher_source = "disjoint"
        else:
            teacher_indices = np.asarray(
                row["condition_indices"],
                dtype=np.int64,
            )
            teacher_features = condition.clone()
            teacher_quality = condition_quality.clone()
            teacher_source = "self"
        return {
            "sample_id": row["sample_id"],
            "split": self.split,
            "scenario": row["scenario"],
            "target_identity": torch.tensor(
                row["target_identity"],
                dtype=torch.int64,
            ),
            "condition_indices": torch.tensor(
                row["condition_indices"],
                dtype=torch.int64,
            ),
            "condition_features": condition,
            "condition_quality": condition_quality,
            "condition_visibility": torch.from_numpy(
                np.array(self.visibility[index], copy=True)
            ).float(),
            "condition_source_identity": torch.tensor(
                row["source_identity_ids"],
                dtype=torch.int64,
            ),
            "condition_degradation": row["condition_degradation"],
            "teacher_indices": torch.from_numpy(teacher_indices.copy()),
            "teacher_features": teacher_features,
            "teacher_quality": teacher_quality,
            "teacher_source": teacher_source,
        }


def _open_output_arrays(
    split_directory: Path,
    *,
    count: int,
    condition_frames: int,
    tokens: int,
    channels: int,
    resume: bool,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    mode = "r+" if resume else "w+"
    condition = np.lib.format.open_memmap(
        split_directory / "condition_features.npy",
        mode=mode,
        dtype=np.float16,
        shape=(count, condition_frames, tokens, channels),
    )
    quality = np.lib.format.open_memmap(
        split_directory / "condition_quality.npy",
        mode=mode,
        dtype=np.float32,
        shape=(count, condition_frames),
    )
    visibility = np.lib.format.open_memmap(
        split_directory / "condition_visibility.npy",
        mode=mode,
        dtype=np.uint8,
        shape=(count, condition_frames, tokens),
    )
    return condition, quality, visibility


def prepare_real_set_dataset(
    *,
    config: Mapping[str, Any],
    profile: str,
    run_id: str,
    dataset_root: Path,
    feature_cache_root: Path,
    checkpoint: Path,
    resume: bool,
    device: torch.device,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> RealSetPreparationResult:
    """Build actual image-perturbed condition sets with clean cached teachers."""

    output_root = (
        Path(config["paths"]["data_dir"]).resolve() / "real_sets" / run_id
    )
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = validate_real_set_dataset(output_root)
        return RealSetPreparationResult(
            root=output_root,
            manifest_path=manifest_path,
            manifest=manifest,
            validation=validation,
            reused=True,
        )
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(
            f"real set dataset already exists; pass --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    cache_manifest_path = feature_cache_root / "manifest.json"
    cache_manifest = json.loads(
        cache_manifest_path.read_text(encoding="utf-8")
    )
    if not bool(cache_manifest.get("complete")):
        raise ValueError("source feature cache is incomplete")
    condition_frames = int(config["data"]["condition_frames"])
    teacher_frames = int(config["data"]["teacher_frames"])
    height = int(config["data"]["feature_grid"]["height"])
    width = int(config["data"]["feature_grid"]["width"])
    channels = int(config["data"]["feature_grid"]["backbone_dim"])
    tokens = height * width
    real_config = config["data"]["real"]
    sets_per_identity = int(real_config["sets_per_identity"])
    max_identities = int(real_config["set_max_identities_per_split"])
    scenarios = [str(value) for value in config["evaluation"]["scenarios"]]
    base_seed = int(config["project"]["seed"])

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    records_by_split: dict[str, list[dict[str, str]]] = {}
    plan_summaries: dict[str, Any] = {}
    selected_image_names: set[str] = set()
    for split_index, split in enumerate(SPLITS):
        records = _read_cache_index(feature_cache_root, split)
        norms = np.load(
            feature_cache_root / split / "feature_norms.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        rows = _build_rows(
            split=split,
            records=records,
            quality_norms=norms,
            condition_frames=condition_frames,
            teacher_frames=teacher_frames,
            sets_per_identity=sets_per_identity,
            scenarios=scenarios,
            wrong_identity_probability=float(
                config["data"]["corruptions"][
                    "wrong_identity_probability"
                ]
            ),
            seed=base_seed + 1000 * (split_index + 1),
            max_identities=max_identities,
        )
        rows_by_split[split] = rows
        records_by_split[split] = records
        plan_summaries[split] = {
            "sets": len(rows),
            "identities": len(
                {int(row["target_identity"]) for row in rows}
            ),
        }
        for row in rows:
            for source_index in row["source_indices"]:
                selected_image_names.add(
                    records[int(source_index)]["image_name"]
                )
        split_directory = output_root / split
        split_directory.mkdir(parents=True, exist_ok=True)
        rows_path = split_directory / "sets.jsonl"
        if not rows_path.is_file():
            _write_jsonl(rows_path, rows)

    plan = {
        "schema_version": REAL_SET_SCHEMA_VERSION,
        "profile": profile,
        "seed": base_seed,
        "config_sha256": config_sha256(config),
        "source_feature_cache_manifest_sha256": _sha256(
            cache_manifest_path
        ),
        "splits": plan_summaries,
    }
    plan_path = output_root / "plan.json"
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise ValueError("resume real-set plan differs from current inputs")
    else:
        _atomic_json(plan_path, plan)
    progress_path = output_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if resume and progress_path.is_file()
        else {"completed": {split: 0 for split in SPLITS}}
    )
    _atomic_json(progress_path, progress)

    landmarks = load_selected_landmarks(
        dataset_root / "Anno" / "list_landmarks_align_celeba.txt",
        selected_image_names,
    )
    image_directory = dataset_root / "Img" / "img_align_celeba"
    model, checkpoint_report = load_adaface_ir50(checkpoint)
    model.to(device)
    hook_info = discover_spatial_hook(model, device=device)
    observed = tuple(hook_info.shape[1:])
    if observed != (channels, height, width):
        raise ValueError(
            f"real set hook shape {observed} does not match config"
        )
    precision = str(config["training"]["precision"])
    autocast_dtype = (
        torch.bfloat16 if precision == "bf16" else torch.float16
    )
    image_batch_size = int(config["backbone"]["cache_batch_size"])
    set_batch_size = max(1, image_batch_size // condition_frames)
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0
    file_summaries: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        rows = rows_by_split[split]
        split_directory = output_root / split
        completed = int(progress["completed"][split])
        arrays_exist = all(
            (split_directory / filename).is_file()
            for filename in (
                "condition_features.npy",
                "condition_quality.npy",
                "condition_visibility.npy",
            )
        )
        condition_array, quality_array, visibility_array = (
            _open_output_arrays(
                split_directory,
                count=len(rows),
                condition_frames=condition_frames,
                tokens=tokens,
                channels=channels,
                resume=resume and arrays_exist,
            )
        )
        dataset = _PerturbedSetImageDataset(
            rows=rows,
            records=records_by_split[split],
            image_directory=image_directory,
            landmarks=landmarks,
            height=height,
            width=width,
            corruptions=config["data"]["corruptions"],
            start_index=completed,
        )
        loader = DataLoader(
            dataset,
            batch_size=set_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=bool(config["runtime"]["pin_memory"]),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_index, (images, visibility, absolute_indices) in enumerate(
            loader
        ):
            batch_sets, frames = images.shape[:2]
            flat_images = images.flatten(0, 1).to(
                device,
                non_blocking=True,
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                spatial, _, norms = extract_spatial_and_embedding(
                    model,
                    flat_images,
                    module_name=hook_info.module_name,
                )
            local = spatial.permute(0, 2, 3, 1).reshape(
                batch_sets,
                frames,
                tokens,
                channels,
            )
            local = functional.normalize(local.float(), dim=-1)
            batch_indices = absolute_indices.numpy()
            condition_array[batch_indices] = (
                local.cpu().numpy().astype(np.float16)
            )
            quality_array[batch_indices] = (
                norms.float()
                .reshape(batch_sets, frames)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            visibility_array[batch_indices] = visibility.numpy().astype(
                np.uint8
            )
            condition_array.flush()
            quality_array.flush()
            visibility_array.flush()
            progress["completed"][split] = int(batch_indices[-1]) + 1
            _atomic_json(progress_path, progress)
            if event_callback is not None:
                event_callback(
                    "real_set_batch",
                    {
                        "split": split,
                        "batch_index": batch_index,
                        "completed": progress["completed"][split],
                        "total": len(rows),
                    },
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated_mib = max(
                peak_allocated_mib,
                torch.cuda.max_memory_allocated(device) / 1024**2,
            )
            peak_reserved_mib = max(
                peak_reserved_mib,
                torch.cuda.max_memory_reserved(device) / 1024**2,
            )
        del condition_array, quality_array, visibility_array
        split_files: dict[str, Any] = {}
        for filename in (
            "condition_features.npy",
            "condition_quality.npy",
            "condition_visibility.npy",
            "sets.jsonl",
        ):
            path = split_directory / filename
            split_files[filename] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        file_summaries[split] = split_files

    split_summaries = {}
    for split, rows in rows_by_split.items():
        scenario_counts = {
            scenario: sum(row["scenario"] == scenario for row in rows)
            for scenario in scenarios
        }
        split_summaries[split] = {
            **plan_summaries[split],
            "condition_shape": [
                len(rows),
                condition_frames,
                tokens,
                channels,
            ],
            "scenario_counts": scenario_counts,
            "files": file_summaries[split],
        }
    manifest = {
        "schema_version": REAL_SET_SCHEMA_VERSION,
        "generator": "set2proto.real_sets",
        "profile": profile,
        "run_id": run_id,
        "seed": base_seed,
        "config_sha256": config_sha256(config),
        "data_config": {
            "feature_grid": dict(config["data"]["feature_grid"]),
            "condition_frames": condition_frames,
            "teacher_frames": teacher_frames,
            "input_size": int(config["data"]["input_size"]),
            "real": dict(config["data"]["real"]),
            "corruptions": dict(config["data"]["corruptions"]),
        },
        "scenarios": scenarios,
        "source_images": str(dataset_root),
        "source_feature_cache": {
            "root": str(feature_cache_root),
            "manifest": str(cache_manifest_path),
            "manifest_sha256": _sha256(cache_manifest_path),
        },
        "checkpoint": {
            **asdict(checkpoint_report),
            "path": str(checkpoint_report.path),
        },
        "spatial_hook": asdict(hook_info),
        "dataloader": {
            "image_batch_size": image_batch_size,
            "set_batch_size": set_batch_size,
            "num_workers": 0,
        },
        "splits": split_summaries,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "complete": all(
            int(progress["completed"][split]) == len(rows_by_split[split])
            for split in SPLITS
        ),
    }
    _atomic_json(manifest_path, manifest)
    validation = validate_real_set_dataset(output_root)
    return RealSetPreparationResult(
        root=output_root,
        manifest_path=manifest_path,
        manifest=manifest,
        validation=validation,
        reused=False,
    )


def validate_real_set_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    condition_frames = int(manifest["data_config"]["condition_frames"])
    teacher_frames = int(manifest["data_config"]["teacher_frames"])
    grid = manifest["data_config"]["feature_grid"]
    tokens = int(grid["height"]) * int(grid["width"])
    channels = int(grid["backbone_dim"])
    cache_root = Path(
        manifest["source_feature_cache"]["root"]
    ).expanduser().resolve()
    all_checks: dict[str, bool] = {
        "manifest_complete": bool(manifest.get("complete")),
        "source_cache_manifest_unchanged": (
            _sha256(cache_root / "manifest.json")
            == manifest["source_feature_cache"]["manifest_sha256"]
        ),
    }
    identity_sets: dict[str, set[int]] = {}
    split_results: dict[str, Any] = {}

    for split in SPLITS:
        rows = list(_iter_jsonl(dataset_root / split / "sets.jsonl"))
        condition = np.load(
            dataset_root / split / "condition_features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        quality = np.load(
            dataset_root / split / "condition_quality.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        visibility = np.load(
            dataset_root / split / "condition_visibility.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        clean_norms = np.load(
            cache_root / split / "feature_norms.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        records = _read_cache_index(cache_root, split)
        identity_sets[split] = {
            int(row["target_identity"]) for row in rows
        }
        expected_shape = (
            len(rows),
            condition_frames,
            tokens,
            channels,
        )
        split_checks = {
            "shape": (
                tuple(condition.shape) == expected_shape
                and tuple(quality.shape)
                == (len(rows), condition_frames)
                and tuple(visibility.shape)
                == (len(rows), condition_frames, tokens)
            ),
            "dtype": (
                condition.dtype == np.float16
                and quality.dtype == np.float32
                and visibility.dtype == np.uint8
            ),
            "finite": True,
            "unit_normalized": True,
            "positive_quality": True,
            "condition_teacher_disjoint": True,
            "indices_match_identity": True,
            "teacher_quality_not_lower": True,
            "scenario_semantics": True,
            "file_hashes": True,
        }
        maximum_unit_error = 0.0
        for start in range(0, len(rows), 128):
            chunk = np.asarray(condition[start : start + 128], dtype=np.float32)
            quality_chunk = np.asarray(
                quality[start : start + 128],
                dtype=np.float32,
            )
            split_checks["finite"] = split_checks["finite"] and bool(
                np.isfinite(chunk).all() and np.isfinite(quality_chunk).all()
            )
            split_checks["positive_quality"] = (
                split_checks["positive_quality"]
                and bool((quality_chunk > 0).all())
            )
            if chunk.size:
                maximum_unit_error = max(
                    maximum_unit_error,
                    float(
                        np.max(
                            np.abs(np.linalg.norm(chunk, axis=-1) - 1.0)
                        )
                    ),
                )
        split_checks["unit_normalized"] = maximum_unit_error < 2e-3
        scenario_counts = {
            scenario: 0 for scenario in manifest["scenarios"]
        }
        for row_index, row in enumerate(rows):
            scenario = str(row["scenario"])
            scenario_counts[scenario] += 1
            condition_indices = {
                int(value) for value in row["condition_indices"]
            }
            teacher_indices = {
                int(value) for value in row["teacher_indices"]
            }
            split_checks["condition_teacher_disjoint"] = (
                split_checks["condition_teacher_disjoint"]
                and not condition_indices.intersection(teacher_indices)
                and len(condition_indices) == condition_frames
                and len(teacher_indices) == teacher_frames
            )
            target = int(row["target_identity"])
            selected = condition_indices | teacher_indices
            split_checks["indices_match_identity"] = (
                split_checks["indices_match_identity"]
                and all(
                    int(records[index]["identity_id"]) == target
                    for index in selected
                )
            )
            split_checks["teacher_quality_not_lower"] = (
                split_checks["teacher_quality_not_lower"]
                and float(clean_norms[list(teacher_indices)].mean())
                >= float(clean_norms[list(condition_indices)].mean()) - 1e-6
            )
            sample_visibility = visibility[row_index]
            if scenario == "clean":
                semantic = bool((sample_visibility == 1).all())
            elif scenario == "low_quality":
                semantic = bool((sample_visibility == 0).any())
            elif scenario == "complementary_occlusion":
                semantic = bool(
                    (sample_visibility.sum(axis=0) > 0).all()
                    and (sample_visibility == 0).any()
                )
            elif scenario == "common_occlusion":
                semantic = bool(
                    np.all(sample_visibility == sample_visibility[0:1])
                    and (sample_visibility[0] == 0).any()
                )
            elif scenario == "wrong_identity":
                wrong = [int(value) for value in row["wrong_positions"]]
                semantic = bool(
                    wrong
                    and all((sample_visibility[position] == 0).all()
                            for position in wrong)
                    and all(
                        int(row["source_identity_ids"][position]) != target
                        for position in wrong
                    )
                )
            else:
                semantic = False
            split_checks["scenario_semantics"] = (
                split_checks["scenario_semantics"] and semantic
            )
        split_checks["all_scenarios_present"] = all(
            scenario_counts[scenario] > 0
            for scenario in manifest["scenarios"]
        )
        for filename, file_summary in manifest["splits"][split][
            "files"
        ].items():
            path = dataset_root / split / filename
            split_checks["file_hashes"] = (
                split_checks["file_hashes"]
                and path.stat().st_size == int(file_summary["size_bytes"])
                and _sha256(path) == file_summary["sha256"]
            )
        split_results[split] = {
            "checks": split_checks,
            "sets": len(rows),
            "identities": len(identity_sets[split]),
            "scenario_counts": scenario_counts,
            "maximum_unit_norm_error": maximum_unit_error,
        }
        for name, value in split_checks.items():
            all_checks[f"{split}.{name}"] = bool(value)

    all_checks["identity_disjoint.train_val"] = identity_sets[
        "train"
    ].isdisjoint(identity_sets["val"])
    all_checks["identity_disjoint.train_test"] = identity_sets[
        "train"
    ].isdisjoint(identity_sets["test"])
    all_checks["identity_disjoint.val_test"] = identity_sets[
        "val"
    ].isdisjoint(identity_sets["test"])
    return {
        "passed": all(all_checks.values()),
        "checks": all_checks,
        "splits": split_results,
    }
