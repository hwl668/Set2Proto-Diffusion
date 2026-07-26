"""Resumable offline cache for frozen AdaFace spatial and global features."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

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
from .real_features import load_selected_landmarks
from .reproducibility import seed_dataloader_worker


@dataclass(frozen=True)
class FeatureCacheResult:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    reused: bool


def validate_real_feature_cache(
    cache_root: Path,
    *,
    verify_hashes: bool = True,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Validate shapes, dtypes, finiteness, normalization, and file hashes."""

    cache_root = cache_root.expanduser().resolve()
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature cache manifest not found: {manifest_path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_results: dict[str, dict[str, Any]] = {}
    all_hashes_match = True
    all_shapes_match = True
    all_dtypes_match = True
    all_finite = True
    all_norms_positive = True
    maximum_local_unit_norm_error = 0.0
    maximum_embedding_unit_norm_error = 0.0

    for split in ("train", "val", "test"):
        summary = manifest["splits"][split]
        split_directory = cache_root / split
        local = np.load(
            split_directory / "local_features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        embedding = np.load(
            split_directory / "embeddings.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        norms = np.load(
            split_directory / "feature_norms.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        expected_local_shape = tuple(summary["local_shape"])
        expected_embedding_shape = tuple(summary["embedding_shape"])
        shape_matches = (
            tuple(local.shape) == expected_local_shape
            and tuple(embedding.shape) == expected_embedding_shape
            and tuple(norms.shape) == (int(summary["images"]),)
        )
        dtype_matches = (
            local.dtype == np.float16
            and embedding.dtype == np.float16
            and norms.dtype == np.float32
        )
        split_finite = True
        split_norms_positive = True
        split_local_error = 0.0
        split_embedding_error = 0.0
        for start in range(0, int(summary["images"]), chunk_size):
            stop = min(start + chunk_size, int(summary["images"]))
            local_chunk = np.asarray(local[start:stop], dtype=np.float32)
            embedding_chunk = np.asarray(
                embedding[start:stop],
                dtype=np.float32,
            )
            norm_chunk = np.asarray(norms[start:stop], dtype=np.float32)
            split_finite = split_finite and bool(
                np.isfinite(local_chunk).all()
                and np.isfinite(embedding_chunk).all()
                and np.isfinite(norm_chunk).all()
            )
            split_norms_positive = split_norms_positive and bool(
                (norm_chunk > 0).all()
            )
            if local_chunk.size:
                split_local_error = max(
                    split_local_error,
                    float(
                        np.max(
                            np.abs(
                                np.linalg.norm(local_chunk, axis=-1) - 1.0
                            )
                        )
                    ),
                )
            if embedding_chunk.size:
                split_embedding_error = max(
                    split_embedding_error,
                    float(
                        np.max(
                            np.abs(
                                np.linalg.norm(embedding_chunk, axis=-1) - 1.0
                            )
                        )
                    ),
                )
        index_path = split_directory / "index.csv"
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            index_rows = sum(1 for _ in csv.DictReader(handle))
        hashes_match = True
        if verify_hashes:
            for filename, file_summary in summary["files"].items():
                path = split_directory / filename
                hashes_match = hashes_match and (
                    path.is_file()
                    and path.stat().st_size == int(file_summary["size_bytes"])
                    and _sha256(path) == file_summary["sha256"]
                )
        split_results[split] = {
            "images": int(summary["images"]),
            "shape_matches": shape_matches,
            "dtype_matches": dtype_matches,
            "finite": split_finite,
            "feature_norms_positive": split_norms_positive,
            "index_rows": index_rows,
            "index_rows_match": index_rows == int(summary["images"]),
            "hashes_match": hashes_match,
            "maximum_local_unit_norm_error": split_local_error,
            "maximum_embedding_unit_norm_error": split_embedding_error,
        }
        all_hashes_match = all_hashes_match and hashes_match
        all_shapes_match = all_shapes_match and shape_matches
        all_dtypes_match = all_dtypes_match and dtype_matches
        all_finite = all_finite and split_finite
        all_norms_positive = all_norms_positive and split_norms_positive
        maximum_local_unit_norm_error = max(
            maximum_local_unit_norm_error,
            split_local_error,
        )
        maximum_embedding_unit_norm_error = max(
            maximum_embedding_unit_norm_error,
            split_embedding_error,
        )

    checks = {
        "manifest_complete": bool(manifest.get("complete")),
        "shapes_match_manifest": all_shapes_match,
        "dtypes_match_manifest": all_dtypes_match,
        "all_values_finite": all_finite,
        "feature_norms_positive": all_norms_positive,
        "index_rows_match": all(
            result["index_rows_match"] for result in split_results.values()
        ),
        "file_hashes_match": all_hashes_match,
        "local_features_unit_normalized": (
            maximum_local_unit_norm_error < 2e-3
        ),
        "embeddings_unit_normalized": (
            maximum_embedding_unit_norm_error < 2e-3
        ),
    }
    return {
        "checks": checks,
        "splits": split_results,
        "maximum_local_unit_norm_error": maximum_local_unit_norm_error,
        "maximum_embedding_unit_norm_error": (
            maximum_embedding_unit_norm_error
        ),
    }


class CelebAAlignedDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        *,
        records: list[dict[str, str]],
        image_directory: Path,
        landmarks: Mapping[str, np.ndarray],
        start_index: int = 0,
    ) -> None:
        self.records = records
        self.image_directory = image_directory
        self.landmarks = landmarks
        self.start_index = start_index

    def __len__(self) -> int:
        return len(self.records) - self.start_index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        absolute_index = self.start_index + index
        image_name = self.records[absolute_index]["image_name"]
        tensor = align_and_normalize_bgr(
            self.image_directory / image_name,
            self.landmarks[image_name],
        )
        return tensor, absolute_index


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
    temporary.replace(path)


def _read_records(split_root: Path) -> dict[str, list[dict[str, str]]]:
    records = {"train": [], "val": [], "test": []}
    with (split_root / "images.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in records:
                raise ValueError(f"unexpected split in images.csv: {split}")
            records[split].append(dict(row))
    return records


def _records_hash(records: Mapping[str, list[dict[str, str]]]) -> str:
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        for row in records[split]:
            digest.update(split.encode())
            digest.update(row["identity_id"].encode())
            digest.update(row["image_name"].encode())
    return digest.hexdigest()


def _write_index(path: Path, records: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("row_index", "split", "identity_id", "image_name"),
        )
        writer.writeheader()
        for row_index, row in enumerate(records):
            writer.writerow(
                {
                    "row_index": row_index,
                    "split": row["split"],
                    "identity_id": row["identity_id"],
                    "image_name": row["image_name"],
                }
            )
    temporary.replace(path)


def _open_arrays(
    split_directory: Path,
    *,
    count: int,
    spatial_tokens: int,
    channels: int,
    resume: bool,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    mode = "r+" if resume else "w+"
    local = np.lib.format.open_memmap(
        split_directory / "local_features.npy",
        mode=mode,
        dtype=np.float16,
        shape=(count, spatial_tokens, channels),
    )
    embedding = np.lib.format.open_memmap(
        split_directory / "embeddings.npy",
        mode=mode,
        dtype=np.float16,
        shape=(count, 512),
    )
    norms = np.lib.format.open_memmap(
        split_directory / "feature_norms.npy",
        mode=mode,
        dtype=np.float32,
        shape=(count,),
    )
    return local, embedding, norms


def prepare_real_feature_cache(
    *,
    config: Mapping[str, Any],
    profile: str,
    run_id: str,
    dataset_root: Path,
    split_root: Path,
    checkpoint: Path,
    resume: bool,
    device: torch.device,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> FeatureCacheResult:
    """Cache normalized 49x512 local features with batch-level resume."""

    cache_root = (
        Path(config["paths"]["cache_dir"]) / "real_features" / run_id
    )
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file() and resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return FeatureCacheResult(
            root=cache_root,
            manifest_path=manifest_path,
            manifest=manifest,
            reused=True,
        )
    if cache_root.exists() and any(cache_root.iterdir()) and not resume:
        raise FileExistsError(
            f"feature cache already exists; pass --resume: {cache_root}"
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    all_records = _read_records(split_root)
    limit = int(config["backbone"]["cache_max_images_per_split"])
    records = {
        split: rows if limit == 0 else rows[:limit]
        for split, rows in all_records.items()
    }
    plan = {
        "schema_version": 1,
        "profile": profile,
        "seed": int(config["project"]["seed"]),
        "records_sha256": _records_hash(records),
        "counts": {split: len(rows) for split, rows in records.items()},
    }
    plan_path = cache_root / "cache_plan.json"
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise ValueError("resume cache plan does not match current split")
    else:
        _atomic_json(plan_path, plan)

    progress_path = cache_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if resume and progress_path.is_file()
        else {"completed": {"train": 0, "val": 0, "test": 0}}
    )
    _atomic_json(progress_path, progress)

    model, checkpoint_report = load_adaface_ir50(checkpoint)
    model.to(device)
    hook_info = discover_spatial_hook(model, device=device)
    _, channels, height, width = hook_info.shape
    expected_grid = config["data"]["feature_grid"]
    expected_shape = (
        int(expected_grid["backbone_dim"]),
        int(expected_grid["height"]),
        int(expected_grid["width"]),
    )
    if (channels, height, width) != expected_shape:
        raise ValueError(
            "observed AdaFace spatial feature differs from config: "
            f"observed={(channels, height, width)}, expected={expected_shape}"
        )
    spatial_tokens = height * width
    selected_names = {
        row["image_name"] for rows in records.values() for row in rows
    }
    landmarks = load_selected_landmarks(
        dataset_root / "Anno" / "list_landmarks_align_celeba.txt",
        selected_names,
    )
    image_directory = dataset_root / "Img" / "img_align_celeba"
    batch_size = int(config["backbone"]["cache_batch_size"])
    requested_num_workers = int(config["runtime"]["num_workers"])
    effective_num_workers = requested_num_workers
    worker_fallback_reason: str | None = None
    if os.name == "nt" and requested_num_workers > 0:
        try:
            queue = multiprocessing.get_context("spawn").Queue()
        except OSError as exc:
            effective_num_workers = 0
            worker_fallback_reason = f"{type(exc).__name__}: {exc}"
            if event_callback is not None:
                event_callback(
                    "dataloader_worker_fallback",
                    {
                        "split": "all",
                        "requested_num_workers": requested_num_workers,
                        "selected_num_workers": 0,
                        "reason": worker_fallback_reason,
                    },
                )
        else:
            queue.close()
            queue.join_thread()
    precision = str(config["training"]["precision"])
    autocast_dtype = (
        torch.bfloat16 if precision == "bf16" else torch.float16
    )
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0
    split_summaries: dict[str, dict[str, Any]] = {}

    for split in ("train", "val", "test"):
        split_directory = cache_root / split
        split_directory.mkdir(parents=True, exist_ok=True)
        split_records = records[split]
        completed = int(progress["completed"][split])
        if completed < 0 or completed > len(split_records):
            raise ValueError(f"invalid cache progress for {split}: {completed}")
        arrays_exist = all(
            (split_directory / filename).is_file()
            for filename in (
                "local_features.npy",
                "embeddings.npy",
                "feature_norms.npy",
            )
        )
        use_existing = resume and arrays_exist
        local_array, embedding_array, norm_array = _open_arrays(
            split_directory,
            count=len(split_records),
            spatial_tokens=spatial_tokens,
            channels=channels,
            resume=use_existing,
        )
        _write_index(split_directory / "index.csv", split_records)
        dataset = CelebAAlignedDataset(
            records=split_records,
            image_directory=image_directory,
            landmarks=landmarks,
            start_index=completed,
        )
        generator = torch.Generator()
        generator.manual_seed(int(config["project"]["seed"]))
        def build_loader(num_workers: int) -> DataLoader:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=bool(config["runtime"]["pin_memory"]),
                worker_init_fn=(
                    seed_dataloader_worker if num_workers > 0 else None
                ),
                generator=generator,
                persistent_workers=num_workers > 0,
            )

        loader = build_loader(effective_num_workers)
        try:
            loader_iterator = iter(loader)
        except PermissionError as exc:
            if effective_num_workers == 0:
                raise
            if event_callback is not None:
                event_callback(
                    "dataloader_worker_fallback",
                    {
                        "split": split,
                        "requested_num_workers": effective_num_workers,
                        "selected_num_workers": 0,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                )
            effective_num_workers = 0
            worker_fallback_reason = f"{type(exc).__name__}: {exc}"
            loader = build_loader(effective_num_workers)
            loader_iterator = iter(loader)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_index, (images, absolute_indices) in enumerate(
            loader_iterator
        ):
            images = images.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                spatial, embeddings, norms = extract_spatial_and_embedding(
                    model,
                    images,
                    module_name=hook_info.module_name,
                )
            local = spatial.permute(0, 2, 3, 1).reshape(
                spatial.shape[0],
                spatial_tokens,
                channels,
            )
            local = functional.normalize(local.float(), dim=-1)
            embeddings = functional.normalize(embeddings.float(), dim=-1)
            indices = absolute_indices.numpy()
            local_array[indices] = local.cpu().numpy().astype(np.float16)
            embedding_array[indices] = (
                embeddings.cpu().numpy().astype(np.float16)
            )
            norm_array[indices] = norms.float().flatten().cpu().numpy()
            local_array.flush()
            embedding_array.flush()
            norm_array.flush()
            progress["completed"][split] = int(indices[-1]) + 1
            _atomic_json(progress_path, progress)
            if event_callback is not None:
                event_callback(
                    "feature_cache_batch",
                    {
                        "split": split,
                        "batch_index": batch_index,
                        "completed": progress["completed"][split],
                        "total": len(split_records),
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
        del local_array, embedding_array, norm_array
        files = {}
        for name in (
            "local_features.npy",
            "embeddings.npy",
            "feature_norms.npy",
            "index.csv",
        ):
            path = split_directory / name
            files[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        split_summaries[split] = {
            "images": len(split_records),
            "local_shape": [
                len(split_records),
                spatial_tokens,
                channels,
            ],
            "embedding_shape": [len(split_records), 512],
            "files": files,
        }

    manifest = {
        "schema_version": 1,
        "profile": profile,
        "run_id": run_id,
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "checkpoint": {
            **asdict(checkpoint_report),
            "path": str(checkpoint_report.path),
        },
        "spatial_hook": asdict(hook_info),
        "cache_dtype": "float16",
        "dataloader": {
            "requested_num_workers": requested_num_workers,
            "effective_num_workers": effective_num_workers,
            "fallback_reason": worker_fallback_reason,
        },
        "splits": split_summaries,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "complete": all(
            int(progress["completed"][split]) == len(records[split])
            for split in ("train", "val", "test")
        ),
    }
    _atomic_json(manifest_path, manifest)
    return FeatureCacheResult(
        root=cache_root,
        manifest_path=manifest_path,
        manifest=manifest,
        reused=False,
    )
