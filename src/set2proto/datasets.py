"""Dispatch helpers for synthetic and cached-real Set2Proto datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .synthetic import SyntheticSetDataset, _load_tensor_file


def load_set_manifest(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"dataset manifest must be an object: {manifest_path}")
    return value


def make_set_dataset(
    root: str | Path,
    split: str,
    *,
    teacher_mode: str = "disjoint",
):
    manifest = load_set_manifest(root)
    generator = str(manifest.get("generator", ""))
    if generator == "set2proto.synthetic":
        return SyntheticSetDataset(root, split, teacher_mode=teacher_mode)
    if generator == "set2proto.real_sets":
        from .real_sets import RealSetDataset

        return RealSetDataset(root, split, teacher_mode=teacher_mode)
    raise ValueError(f"unsupported dataset generator: {generator!r}")


def validate_set_dataset(root: str | Path) -> dict[str, Any]:
    manifest = load_set_manifest(root)
    generator = str(manifest.get("generator", ""))
    if generator == "set2proto.synthetic":
        from .synthetic import validate_synthetic_dataset

        return validate_synthetic_dataset(root)
    if generator == "set2proto.real_sets":
        from .real_sets import validate_real_set_dataset

        return validate_real_set_dataset(root)
    raise ValueError(f"unsupported dataset generator: {generator!r}")


def load_train_feature_bank(root: str | Path) -> torch.Tensor:
    dataset_root = Path(root).expanduser().resolve()
    manifest = load_set_manifest(dataset_root)
    generator = str(manifest.get("generator", ""))
    if generator == "set2proto.synthetic":
        return _load_tensor_file(
            dataset_root / "train_features.pt"
        )["features"]
    if generator == "set2proto.real_sets":
        cache_root = Path(
            manifest["source_feature_cache"]["root"]
        ).expanduser().resolve()
        array = np.load(
            cache_root / "train" / "local_features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        return torch.from_numpy(np.array(array, copy=True))
    raise ValueError(f"unsupported dataset generator: {generator!r}")
