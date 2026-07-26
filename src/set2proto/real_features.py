"""Real-image AdaFace preprocessing and frozen-backbone smoke probing."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adaface_backbone import (
    align_and_normalize_bgr,
    discover_spatial_hook,
    extract_spatial_and_embedding,
    load_adaface_ir50,
)


def load_selected_landmarks(
    path: Path,
    image_names: set[str],
) -> dict[str, np.ndarray]:
    """Load only requested aligned CelebA five-point landmarks."""

    selected: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        count_line = handle.readline().strip()
        header = handle.readline().split()
        if int(count_line) <= 0 or len(header) != 10:
            raise ValueError(f"invalid CelebA landmark header: {path}")
        for line_number, line in enumerate(handle, start=3):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 11:
                raise ValueError(
                    f"{path} line {line_number} must contain 11 fields"
                )
            image_name = fields[0]
            if image_name in image_names:
                values = np.asarray(fields[1:], dtype=np.float32)
                selected[image_name] = values.reshape(5, 2)
                if len(selected) == len(image_names):
                    break
    missing = image_names - set(selected)
    if missing:
        raise ValueError(f"missing landmarks for {len(missing)} images")
    return selected


def resolve_real_split_root(
    dataset_root: Path,
    *,
    profile: str,
    seed: int,
) -> Path:
    preferred = dataset_root / "splits" / f"{profile}-seed{seed}"
    if (preferred / "images.csv").is_file():
        return preferred
    candidates = sorted(
        path
        for path in (dataset_root / "splits").glob(f"*-seed{seed}")
        if (path / "images.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no real split images.csv found under {dataset_root / 'splits'}"
        )
    return candidates[0]


def _sample_image_names(split_root: Path, batch_size: int) -> list[str]:
    names: list[str] = []
    with (split_root / "images.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train":
                continue
            names.append(row["image_name"])
            if len(names) == batch_size:
                break
    if len(names) != batch_size:
        raise ValueError(
            f"requested {batch_size} images but found only {len(names)}"
        )
    return names


def probe_adaface_backbone(
    *,
    checkpoint: Path,
    dataset_root: Path,
    split_root: Path,
    batch_size: int,
    precision: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run a frozen real-image batch and report actual hook shape/memory."""

    image_names = _sample_image_names(split_root, batch_size)
    landmarks = load_selected_landmarks(
        dataset_root / "Anno" / "list_landmarks_align_celeba.txt",
        set(image_names),
    )
    image_directory = dataset_root / "Img" / "img_align_celeba"
    batch = torch.stack(
        [
            align_and_normalize_bgr(
                image_directory / image_name,
                landmarks[image_name],
            )
            for image_name in image_names
        ]
    )
    if tuple(batch.shape[1:]) != (3, 112, 112):
        raise RuntimeError(f"unexpected AdaFace input shape: {tuple(batch.shape)}")
    if not torch.isfinite(batch).all():
        raise RuntimeError("AdaFace input contains non-finite values")

    model, checkpoint_report = load_adaface_ir50(checkpoint)
    model.to(device)
    hook_info = discover_spatial_hook(model, device=device)
    batch = batch.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = (
        torch.bfloat16 if precision == "bf16" else torch.float16
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=enabled,
    ):
        spatial, embedding, norm = extract_spatial_and_embedding(
            model,
            batch,
            module_name=hook_info.module_name,
        )
    if not torch.isfinite(spatial).all():
        raise RuntimeError("AdaFace spatial features contain non-finite values")
    if not torch.isfinite(embedding).all() or not torch.isfinite(norm).all():
        raise RuntimeError("AdaFace embeddings contain non-finite values")
    unit_norm_error = (
        torch.linalg.vector_norm(embedding.float(), dim=1)
        .sub(1.0)
        .abs()
        .max()
        .item()
    )
    peak_allocated = 0.0
    peak_reserved = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    return {
        "checkpoint": {
            **asdict(checkpoint_report),
            "path": str(checkpoint_report.path),
        },
        "split_root": str(split_root),
        "images": image_names,
        "input": {
            "shape": list(batch.shape),
            "dtype": str(batch.dtype),
            "color_order": "bgr",
            "minimum": float(batch.min().item()),
            "maximum": float(batch.max().item()),
        },
        "spatial_hook": asdict(hook_info),
        "spatial_feature": {
            "shape": list(spatial.shape),
            "dtype": str(spatial.dtype),
        },
        "embedding": {
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
            "maximum_unit_norm_error": float(unit_norm_error),
            "norm_mean": float(norm.float().mean().item()),
            "norm_minimum": float(norm.float().min().item()),
            "norm_maximum": float(norm.float().max().item()),
        },
        "frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "training": model.training,
        "device": str(device),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }
