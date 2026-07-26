"""P0-3 earlier-hook metadata-oracle headroom experiment.

The experiment pools raw intermediate AdaFace activations, runs the unchanged
frozen suffix back to the existing 7x7 hook, and then applies the already-fitted
PCA projection. No adapter, codebook, Transformer, or diffusion model is fit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adaface_backbone import (
    AdaFaceIRBackbone,
    extract_named_spatial_and_embedding,
    run_body_suffix_to_spatial,
)
from .quantization import PCAProjection
from .real_features import load_selected_landmarks
from .real_sets import (
    _PerturbedSetImageDataset,
    _read_cache_index,
)
from .research_evaluation import continuous_aggregation_baselines
from .synthetic import _iter_jsonl
from .training import TokenTrainingDataset
from .visibility_aggregation import (
    _evaluate_collected_maps,
    _same_identity_frame_mask,
    oracle_visibility_pool,
)


LATE_METHODS = (
    "late_mean_pooling",
    "late_quality_pooling",
    "late_metadata_oracle_pooling",
)
EARLY_METHODS = (
    "suffix_replay_quality",
    "pre_suffix_mean_pooling",
    "pre_suffix_quality_pooling",
    "pre_suffix_metadata_oracle_pooling",
)
ALL_METHODS = (*LATE_METHODS, *EARLY_METHODS)
HARD_SCENARIOS = (
    "low_quality",
    "complementary_occlusion",
    "common_occlusion",
    "wrong_identity",
)
_FLOAT16_MAX = float(np.finfo(np.float16).max)


def _require_finite_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(value)
    if not bool(finite.all().item()):
        invalid = int((~finite).sum().item())
        raise FloatingPointError(
            f"{name} contains {invalid} non-finite value(s)"
        )
    return value


def _require_finite_array(name: str, value: np.ndarray) -> np.ndarray:
    finite = np.isfinite(value)
    if not bool(finite.all()):
        invalid = int(np.size(value) - np.count_nonzero(finite))
        raise FloatingPointError(
            f"{name} contains {invalid} non-finite value(s)"
        )
    return value


def _canonical_sum(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Sum a small unordered set through a permutation-invariant FP path."""

    ordered = torch.sort(value, dim=dim).values
    return ordered.sum(dim=dim)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _project_late_spatial(
    late_spatial: torch.Tensor,
    projector: PCAProjection,
) -> torch.Tensor:
    if late_spatial.ndim != 4 or int(late_spatial.shape[1]) != 512:
        raise ValueError("late_spatial must have shape [B,512,7,7]")
    _require_finite_tensor("late_spatial", late_spatial)
    local = late_spatial.permute(0, 2, 3, 1).reshape(
        late_spatial.shape[0],
        -1,
        late_spatial.shape[1],
    )
    local = F.normalize(local.float(), dim=-1)
    _require_finite_tensor("normalized late_spatial", local)
    mean = projector.mean.to(local.device)
    components = projector.components.to(local.device)
    projected = torch.matmul(
        local - mean,
        components.transpose(0, 1),
    )
    _require_finite_tensor("projected late_spatial", projected)
    normalized = F.normalize(projected, dim=-1, eps=1e-6)
    return _require_finite_tensor(
        "normalized projected late_spatial",
        normalized,
    )


def resize_visibility_to_hook(
    visibility: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Nearest-resize the original 7x7 input-cell metadata to an early grid."""

    if visibility.ndim != 3:
        raise ValueError("visibility must have shape [B,F,P]")
    source_tokens = int(visibility.shape[-1])
    source_side = int(round(source_tokens**0.5))
    if source_side * source_side != source_tokens:
        raise ValueError("visibility token count must be a square grid")
    batch, frames = visibility.shape[:2]
    resized = F.interpolate(
        visibility.float().reshape(batch * frames, 1, source_side, source_side),
        size=(height, width),
        mode="nearest",
    )
    return resized.reshape(batch, frames, height, width)


def metadata_oracle_weights(
    condition_quality: torch.Tensor,
    visibility: torch.Tensor,
    same_identity: torch.Tensor,
) -> torch.Tensor:
    """Return metadata routing weights with shape [B,H,W,F]."""

    if visibility.ndim != 4:
        raise ValueError("visibility must have shape [B,F,H,W]")
    batch, frames, height, width = visibility.shape
    if tuple(condition_quality.shape) != (batch, frames):
        raise ValueError("condition_quality must have shape [B,F]")
    if tuple(same_identity.shape) != (batch, frames):
        raise ValueError("same_identity must have shape [B,F]")
    quality = _require_finite_tensor(
        "condition_quality",
        condition_quality.float(),
    ).clamp_min(1e-6)
    identity = same_identity.float().clamp(0.0, 1.0)
    evidence = _require_finite_tensor(
        "visibility",
        visibility.float(),
    ).clamp(0.0, 1.0)
    evidence = evidence * identity[:, :, None, None] * quality[:, :, None, None]
    denominator = _canonical_sum(evidence, dim=1).unsqueeze(1)
    routed = evidence / denominator.clamp_min(1e-8)
    identity_quality = quality * identity
    identity_denominator = _canonical_sum(
        identity_quality,
        dim=1,
    ).unsqueeze(1)
    fallback = identity_quality / identity_denominator.clamp_min(1e-8)
    all_denominator = _canonical_sum(quality, dim=1).unsqueeze(1)
    all_quality = quality / all_denominator.clamp_min(1e-8)
    fallback = torch.where(identity_denominator > 0, fallback, all_quality)
    fallback = fallback[:, :, None, None].expand(
        batch,
        frames,
        height,
        width,
    )
    weights_by_frame = torch.where(denominator > 0, routed, fallback)
    return _require_finite_tensor(
        "metadata oracle weights",
        weights_by_frame.permute(0, 2, 3, 1),
    )


def quality_spatial_weights(
    condition_quality: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if condition_quality.ndim != 2:
        raise ValueError("condition_quality must have shape [B,F]")
    quality = _require_finite_tensor(
        "condition_quality",
        condition_quality.float(),
    ).clamp_min(1e-6)
    denominator = _canonical_sum(quality, dim=1).unsqueeze(1)
    quality = quality / denominator.clamp_min(1e-8)
    return _require_finite_tensor(
        "quality spatial weights",
        quality[:, None, None, :].expand(-1, height, width, -1),
    )


def mean_spatial_weights(
    *,
    batch: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    return _require_finite_tensor(
        "mean spatial weights",
        torch.full(
            (batch, height, width, frames),
            1.0 / float(frames),
            dtype=torch.float32,
            device=device,
        ),
    )


def pool_raw_early_activation(
    early_features: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Pool raw [B,F,C,H,W] activations without feature normalization."""

    if early_features.ndim != 5:
        raise ValueError("early_features must have shape [B,F,C,H,W]")
    batch, frames, _, height, width = early_features.shape
    if tuple(weights.shape) != (batch, height, width, frames):
        raise ValueError("weights must have shape [B,H,W,F]")
    _require_finite_tensor("early features", early_features)
    _require_finite_tensor("frame weights", weights)
    if float((weights.sum(dim=-1) - 1.0).abs().max().item()) > 1e-5:
        raise ValueError("frame weights must sum to one")
    features = early_features.float().permute(0, 3, 4, 1, 2)
    contributions = weights.unsqueeze(-1) * features
    pooled = _canonical_sum(contributions, dim=3)
    return _require_finite_tensor(
        "pooled early activation",
        pooled.permute(0, 3, 1, 2).contiguous(),
    )


def _load_replay_image_dataset(
    *,
    set_dataset_root: Path,
    split: str,
    set_manifest: Mapping[str, Any],
) -> tuple[_PerturbedSetImageDataset, list[dict[str, Any]]]:
    rows = list(_iter_jsonl(set_dataset_root / split / "sets.jsonl"))
    feature_cache_root = Path(
        set_manifest["source_feature_cache"]["root"]
    ).resolve()
    records = _read_cache_index(feature_cache_root, split)
    selected_names = {
        records[int(source_index)]["image_name"]
        for row in rows
        for source_index in row["source_indices"]
    }
    image_root = Path(set_manifest["source_images"]).resolve()
    landmarks = load_selected_landmarks(
        image_root / "Anno" / "list_landmarks_align_celeba.txt",
        selected_names,
    )
    dataset = _PerturbedSetImageDataset(
        rows=rows,
        records=records,
        image_directory=image_root / "Img" / "img_align_celeba",
        landmarks=landmarks,
        height=7,
        width=7,
        corruptions=set_manifest["data_config"]["corruptions"],
        start_index=0,
    )
    return dataset, rows


@torch.inference_mode()
def cache_early_hook_features(
    *,
    split: str,
    hook_names: Sequence[str],
    hook_shapes: Mapping[str, Sequence[int]],
    model: AdaFaceIRBackbone,
    token_dataset: TokenTrainingDataset,
    set_dataset_root: Path,
    set_manifest: Mapping[str, Any],
    output_directory: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Replay deterministic perturbations and persist float16 early maps."""

    if split not in {"val", "test"}:
        raise ValueError("P0-3 cache split must be val or test")
    if getattr(token_dataset.synthetic, "split", None) != split:
        raise ValueError("token dataset split does not match cache split")
    names = list(hook_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("hook_names must be non-empty and unique")
    output_directory.mkdir(parents=True, exist_ok=False)
    replay_dataset, rows = _load_replay_image_dataset(
        set_dataset_root=set_dataset_root,
        split=split,
        set_manifest=set_manifest,
    )
    if len(replay_dataset) != len(token_dataset):
        raise ValueError("replay and token datasets are not aligned")
    frames = int(config["data"]["condition_frames"])
    temporary_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    arrays: dict[str, np.memmap] = {}
    for name in names:
        channels, height, width = [
            int(value) for value in hook_shapes[name]
        ]
        safe_name = name.replace(".", "_")
        temporary = output_directory / f"{split}_{safe_name}.tmp.npy"
        final = output_directory / f"{split}_{safe_name}.npy"
        if temporary.exists() or final.exists():
            raise FileExistsError(f"early cache already exists for {name}")
        arrays[name] = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(replay_dataset),
                frames,
                channels,
                height,
                width,
            ),
        )
        temporary_paths[name] = temporary
        final_paths[name] = final

    loader = DataLoader(
        replay_dataset,
        batch_size=int(config["p0_3"]["set_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(config["runtime"]["pin_memory"]),
    )
    autocast_dtype = (
        torch.bfloat16
        if str(config["training"]["precision"]) == "bf16"
        else torch.float16
    )
    late_name = str(set_manifest["spatial_hook"]["module_name"])
    capture_names = [*names, late_name]
    projected_cosine_values: list[torch.Tensor] = []
    projected_max_abs = 0.0
    quality_max_abs = 0.0
    suffix_cosines: dict[str, list[torch.Tensor]] = {
        name: [] for name in names
    }
    suffix_max_abs: dict[str, float] = {name: 0.0 for name in names}
    raw_sums = {name: 0.0 for name in names}
    raw_square_sums = {name: 0.0 for name in names}
    raw_counts = {name: 0 for name in names}
    raw_mins = {name: float("inf") for name in names}
    raw_maxs = {name: float("-inf") for name in names}
    raw_max_abs = {name: 0.0 for name in names}
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for images, _visibility, absolute_indices in loader:
        batch_sets, batch_frames = images.shape[:2]
        if batch_frames != frames:
            raise RuntimeError("replayed frame count changed")
        flat_images = images.flatten(0, 1).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            captured, _embedding, norms = (
                extract_named_spatial_and_embedding(
                    model,
                    flat_images,
                    module_names=capture_names,
                )
            )
        batch_indices = absolute_indices.numpy()
        late_projected = _project_late_spatial(
            captured[late_name],
            token_dataset.projector,
        ).reshape(batch_sets, frames, 49, -1)
        expected_features = torch.stack(
            [
                token_dataset[int(index)]["condition_features"]
                for index in batch_indices
            ]
        ).to(device)
        _require_finite_tensor(
            "cached condition projected features",
            expected_features,
        )
        projected_cosine = F.cosine_similarity(
            late_projected.float(),
            expected_features.float(),
            dim=-1,
        )
        projected_cosine_values.append(
            _require_finite_tensor(
                "cache-build projected cosine",
                projected_cosine,
            ).detach().cpu()
        )
        projected_max_abs = max(
            projected_max_abs,
            float(
                (late_projected - expected_features)
                .abs()
                .max()
                .item()
            ),
        )
        expected_quality = torch.stack(
            [
                token_dataset[int(index)]["condition_quality"]
                for index in batch_indices
            ]
        ).to(device)
        _require_finite_tensor("cached condition quality", expected_quality)
        observed_quality = _require_finite_tensor(
            "replayed AdaFace quality",
            norms.float().reshape(batch_sets, frames),
        )
        quality_max_abs = max(
            quality_max_abs,
            float(
                (observed_quality - expected_quality).abs().max().item()
            ),
        )
        for name in names:
            value = captured[name]
            expected_shape = tuple(int(item) for item in hook_shapes[name])
            if tuple(value.shape[1:]) != expected_shape:
                raise RuntimeError(f"hook shape changed for {name}")
            reshaped = value.reshape(
                batch_sets,
                frames,
                *expected_shape,
            )
            raw = _require_finite_tensor(
                f"{name} raw early activation",
                reshaped.detach().float(),
            )
            batch_min = float(raw.amin().item())
            batch_max = float(raw.amax().item())
            batch_max_abs = max(abs(batch_min), abs(batch_max))
            if batch_max_abs > _FLOAT16_MAX:
                raise OverflowError(
                    f"{name} raw early activation exceeds float16 range: "
                    f"max_abs={batch_max_abs}, limit={_FLOAT16_MAX}"
                )
            raw_cpu = _require_finite_array(
                f"{name} pre-cast early activation",
                raw.cpu().numpy(),
            )
            stored = _require_finite_array(
                f"{name} float16 early activation",
                raw_cpu.astype(np.float16),
            )
            arrays[name][batch_indices] = stored
            _require_finite_array(
                f"{name} written float16 cache",
                np.asarray(arrays[name][batch_indices]),
            )
            raw_sums[name] += float(raw.sum().item())
            raw_square_sums[name] += float(raw.square().sum().item())
            raw_counts[name] += int(raw.numel())
            raw_mins[name] = min(raw_mins[name], batch_min)
            raw_maxs[name] = max(raw_maxs[name], batch_max)
            raw_max_abs[name] = max(raw_max_abs[name], batch_max_abs)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                replayed = run_body_suffix_to_spatial(
                    model,
                    value,
                    module_name=name,
                )
            _require_finite_tensor(
                f"{name} direct suffix replay",
                replayed,
            )
            suffix_cosine = F.cosine_similarity(
                replayed.float().flatten(1),
                captured[late_name].float().flatten(1),
                dim=-1,
            )
            suffix_cosines[name].append(
                _require_finite_tensor(
                    f"{name} direct suffix cosine",
                    suffix_cosine,
                ).detach().cpu()
            )
            suffix_max_abs[name] = max(
                suffix_max_abs[name],
                float(
                    (
                        replayed.float() - captured[late_name].float()
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
        for array in arrays.values():
            array.flush()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - start_time
    del array
    del arrays
    for name in names:
        os.replace(temporary_paths[name], final_paths[name])
    projected_cosine = _require_finite_tensor(
        "cache-build projected cosine",
        torch.cat(projected_cosine_values),
    )
    files = {
        name: {
            "path": str(final_paths[name]),
            "shape": [
                len(replay_dataset),
                frames,
                *[int(value) for value in hook_shapes[name]],
            ],
            "dtype": "float16",
            "size_bytes": final_paths[name].stat().st_size,
            "sha256": _file_sha256(final_paths[name]),
        }
        for name in names
    }
    raw_statistics = {}
    for name in names:
        mean = raw_sums[name] / float(raw_counts[name])
        variance = max(
            0.0,
            raw_square_sums[name] / float(raw_counts[name]) - mean * mean,
        )
        raw_statistics[name] = {
            "mean": mean,
            "std": variance**0.5,
            "min": raw_mins[name],
            "max": raw_maxs[name],
            "max_abs": raw_max_abs[name],
        }
    return {
        "split": split,
        "sets": len(rows),
        "frames_per_set": frames,
        "hook_names": names,
        "files": files,
        "build_seconds": build_seconds,
        "pipeline_regression": {
            "projected_map_cosine_mean": float(
                projected_cosine.mean().item()
            ),
            "projected_map_cosine_min": float(
                projected_cosine.min().item()
            ),
            "projected_map_max_abs_diff": projected_max_abs,
            "quality_max_abs_diff": quality_max_abs,
            "suffix_replay": {
                name: {
                    "late_spatial_cosine_mean": float(
                        torch.cat(suffix_cosines[name]).mean().item()
                    ),
                    "late_spatial_cosine_min": float(
                        torch.cat(suffix_cosines[name]).min().item()
                    ),
                    "late_spatial_max_abs_diff": suffix_max_abs[name],
                }
                for name in names
            },
        },
        "raw_statistics": raw_statistics,
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


def _oracle_weight_diagnostics(
    *,
    weights: torch.Tensor,
    visibility: torch.Tensor,
    same_identity: torch.Tensor,
) -> dict[str, Any]:
    visible = visibility.permute(0, 2, 3, 1).bool()
    wrong = (~same_identity)[:, None, None, :].expand_as(weights)
    return {
        "weight_sum_max_abs_error": float(
            (weights.sum(dim=-1) - 1.0).abs().max().item()
        ),
        "visible_weight_mean": (
            float(weights[visible].mean().item())
            if bool(visible.any().item())
            else None
        ),
        "occluded_weight_mean": (
            float(weights[~visible].mean().item())
            if bool((~visible).any().item())
            else None
        ),
        "wrong_identity_weight_mean": (
            float(weights[wrong].mean().item())
            if bool(wrong.any().item())
            else None
        ),
        "minimum_weight": float(weights.min().item()),
        "maximum_weight": float(weights.max().item()),
    }


@torch.inference_mode()
def evaluate_early_hook_split(
    *,
    split: str,
    hook_name: str,
    hook_shape: Sequence[int],
    cache_path: Path,
    dataset: TokenTrainingDataset,
    model: AdaFaceIRBackbone,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one preregistered hook under the unchanged gallery protocol."""

    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if getattr(dataset.synthetic, "split", None) != split:
        raise ValueError("dataset split does not match requested split")
    channels, height, width = [int(value) for value in hook_shape]
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    expected_cache_shape = (
        len(dataset),
        int(config["data"]["condition_frames"]),
        channels,
        height,
        width,
    )
    if tuple(cache.shape) != expected_cache_shape:
        raise ValueError("early cache shape does not match protocol")
    batch_size = int(config["evaluation"]["batch_size"])
    autocast_dtype = (
        torch.bfloat16
        if str(config["training"]["precision"]) == "bf16"
        else torch.float16
    )
    map_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in ALL_METHODS
    }
    identities: list[torch.Tensor] = []
    scenarios: list[str] = []
    oracle_diagnostics: dict[str, list[float]] = {}
    pooled_raw_stats = {
        name: {"sum": 0.0, "square_sum": 0.0, "count": 0}
        for name in (
            "pre_suffix_mean_pooling",
            "pre_suffix_quality_pooling",
            "pre_suffix_metadata_oracle_pooling",
        )
    }
    maximum_norm_error = 0.0
    maximum_permutation_diff = 0.0
    maximum_permutation_relative_diff = 0.0
    maximum_permutation_weight_diff = 0.0
    maximum_permutation_map_diff = 0.0
    minimum_permutation_map_cosine = 1.0
    cache_roundtrip_cosine_chunks: list[torch.Tensor] = []
    cache_roundtrip_max_abs = 0.0

    def append_map(name: str, maps: torch.Tensor) -> None:
        nonlocal maximum_norm_error
        if tuple(maps.shape[1:]) != (49, 128):
            raise RuntimeError(f"{name} returned invalid projected shape")
        _require_finite_tensor(f"{name} projected map", maps)
        maximum_norm_error = max(
            maximum_norm_error,
            float((maps.norm(dim=-1) - 1.0).abs().max().item()),
        )
        map_chunks[name].append(maps.detach().cpu().half())

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        samples = [dataset[index] for index in indices]
        condition = torch.stack(
            [sample["condition_features"] for sample in samples]
        ).to(device)
        _require_finite_tensor("condition projected features", condition)
        quality = torch.stack(
            [sample["condition_quality"] for sample in samples]
        ).to(device)
        visibility = torch.stack(
            [sample["condition_visibility"] for sample in samples]
        ).to(device)
        same_identity_cpu = _same_identity_frame_mask(
            dataset,
            indices,
            condition.shape[1],
        )
        same_identity = same_identity_cpu.to(device)
        baselines = continuous_aggregation_baselines(condition, quality)
        late_oracle, _ = oracle_visibility_pool(
            condition,
            quality,
            visibility,
            same_identity,
        )
        append_map("late_mean_pooling", baselines["mean_pooling"])
        append_map("late_quality_pooling", baselines["quality_pooling"])
        append_map("late_metadata_oracle_pooling", late_oracle)

        early = torch.from_numpy(
            np.array(cache[indices], copy=True)
        ).to(device)
        _require_finite_tensor("cached early features", early)
        resized_visibility = resize_visibility_to_hook(
            visibility,
            height=height,
            width=width,
        )
        mean_weights = mean_spatial_weights(
            batch=early.shape[0],
            frames=early.shape[1],
            height=height,
            width=width,
            device=device,
        )
        quality_weights = quality_spatial_weights(
            quality,
            height=height,
            width=width,
        )
        oracle_weights = metadata_oracle_weights(
            quality,
            resized_visibility,
            same_identity,
        )
        batch_oracle = _oracle_weight_diagnostics(
            weights=oracle_weights,
            visibility=resized_visibility,
            same_identity=same_identity,
        )
        for key, value in batch_oracle.items():
            if value is not None:
                oracle_diagnostics.setdefault(key, []).append(float(value))
        pooled_early = {
            "pre_suffix_mean_pooling": pool_raw_early_activation(
                early,
                mean_weights,
            ),
            "pre_suffix_quality_pooling": pool_raw_early_activation(
                early,
                quality_weights,
            ),
            "pre_suffix_metadata_oracle_pooling": (
                pool_raw_early_activation(early, oracle_weights)
            ),
        }
        for name, value in pooled_early.items():
            stats = pooled_raw_stats[name]
            stats["sum"] += float(value.float().sum().item())
            stats["square_sum"] += float(value.float().square().sum().item())
            stats["count"] += int(value.numel())
        stacked = torch.cat(list(pooled_early.values()), dim=0)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            suffix_maps = run_body_suffix_to_spatial(
                model,
                stacked,
                module_name=hook_name,
            )
        projected = _project_late_spatial(
            suffix_maps,
            dataset.projector,
        )
        projected_by_method = dict(
            zip(
                pooled_early,
                projected.chunk(len(pooled_early), dim=0),
            )
        )
        for name, value in projected_by_method.items():
            append_map(name, value)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            replay_late = run_body_suffix_to_spatial(
                model,
                early.flatten(0, 1),
                module_name=hook_name,
            )
        replay_projected = _project_late_spatial(
            replay_late,
            dataset.projector,
        ).reshape(condition.shape)
        cache_roundtrip_cosine = F.cosine_similarity(
            replay_projected.float(),
            condition.float(),
            dim=-1,
        )
        cache_roundtrip_cosine_chunks.append(
            _require_finite_tensor(
                "per-token cache roundtrip cosine",
                cache_roundtrip_cosine,
            ).detach().cpu()
        )
        cache_roundtrip_max_abs = max(
            cache_roundtrip_max_abs,
            float(
                (replay_projected.float() - condition.float())
                .abs()
                .max()
                .item()
            ),
        )
        replay_quality = continuous_aggregation_baselines(
            replay_projected,
            quality,
        )["quality_pooling"]
        append_map("suffix_replay_quality", replay_quality)

        if start == 0:
            permutation = torch.tensor([2, 0, 3, 1], device=device)
            permuted_weights = metadata_oracle_weights(
                quality[:, permutation],
                resized_visibility[:, permutation],
                same_identity[:, permutation],
            )
            permuted_pool = pool_raw_early_activation(
                early[:, permutation],
                permuted_weights,
            )
            original_pool = pooled_early[
                "pre_suffix_metadata_oracle_pooling"
            ]
            maximum_permutation_diff = float(
                (original_pool - permuted_pool).abs().max().item()
            )
            maximum_permutation_relative_diff = (
                maximum_permutation_diff
                / max(1.0, float(original_pool.abs().max().item()))
            )
            inverse = torch.argsort(permutation)
            maximum_permutation_weight_diff = float(
                (
                    oracle_weights
                    - permuted_weights[..., inverse]
                )
                .abs()
                .max()
                .item()
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                paired_suffix = run_body_suffix_to_spatial(
                    model,
                    torch.cat((original_pool, permuted_pool), dim=0),
                    module_name=hook_name,
                )
            paired_projected = _project_late_spatial(
                paired_suffix,
                dataset.projector,
            )
            original_projected, permuted_projected = (
                paired_projected.chunk(2, dim=0)
            )
            maximum_permutation_map_diff = float(
                (original_projected - permuted_projected)
                .abs()
                .max()
                .item()
            )
            minimum_permutation_map_cosine = float(
                F.cosine_similarity(
                    original_projected,
                    permuted_projected,
                    dim=-1,
                )
                .min()
                .item()
            )
        identities.append(
            torch.stack(
                [sample["target_identity"] for sample in samples]
            ).cpu()
        )
        scenarios.extend(str(sample["scenario"]) for sample in samples)

    evaluated = _evaluate_collected_maps(
        dataset=dataset,
        map_chunks=map_chunks,
        token_chunks={},
        identities=identities,
        scenarios=scenarios,
        far_targets=[
            float(value) for value in config["evaluation"]["far_targets"]
        ],
    )
    replay_maps = torch.cat(map_chunks["suffix_replay_quality"]).float()
    late_quality_maps = torch.cat(map_chunks["late_quality_pooling"]).float()
    replay_cosine = F.cosine_similarity(
        replay_maps,
        late_quality_maps,
        dim=-1,
    )
    _require_finite_tensor("pooled suffix replay cosine", replay_cosine)
    cache_roundtrip_cosine = _require_finite_tensor(
        "per-token cache roundtrip cosine",
        torch.cat(cache_roundtrip_cosine_chunks),
    )
    pooled_statistics = {}
    for name, stats in pooled_raw_stats.items():
        mean = stats["sum"] / float(stats["count"])
        variance = max(
            0.0,
            stats["square_sum"] / float(stats["count"]) - mean * mean,
        )
        pooled_statistics[name] = {"mean": mean, "std": variance**0.5}
    evaluated.update(
        {
            "split": split,
            "hook_name": hook_name,
            "hook_shape": [channels, height, width],
            "cache_path": str(cache_path),
            "evaluated_configuration_count": 1,
            "correctness": {
                "all_output_shapes_valid": all(
                    bool(chunks) for chunks in map_chunks.values()
                ),
                "output_unit_norm_max_abs_error": maximum_norm_error,
                "output_unit_norm_valid": maximum_norm_error <= 1e-5,
                "suffix_replay_map_cosine_mean": float(
                    replay_cosine.mean().item()
                ),
                "suffix_replay_map_cosine_min": float(
                    replay_cosine.min().item()
                ),
                "suffix_replay_map_max_abs_diff": float(
                    (replay_maps - late_quality_maps).abs().max().item()
                ),
                "cache_roundtrip_token_cosine_mean": float(
                    cache_roundtrip_cosine.mean().item()
                ),
                "cache_roundtrip_token_cosine_min": float(
                    cache_roundtrip_cosine.min().item()
                ),
                "cache_roundtrip_token_max_abs_diff": (
                    cache_roundtrip_max_abs
                ),
                "permutation_pool_max_abs_diff": (
                    maximum_permutation_diff
                ),
                "permutation_pool_max_relative_diff": (
                    maximum_permutation_relative_diff
                ),
                "permutation_weight_max_abs_diff": (
                    maximum_permutation_weight_diff
                ),
                "permutation_output_map_max_abs_diff": (
                    maximum_permutation_map_diff
                ),
                "permutation_output_map_cosine_min": (
                    minimum_permutation_map_cosine
                ),
                "permutation_equivalent": (
                    maximum_permutation_relative_diff <= 1e-6
                    and maximum_permutation_weight_diff <= 1e-6
                    and maximum_permutation_map_diff <= 1e-5
                    and minimum_permutation_map_cosine >= 0.999999
                ),
            },
            "oracle_weight_diagnostics": {
                key: float(np.mean(values))
                for key, values in oracle_diagnostics.items()
            },
            "pooled_raw_statistics": pooled_statistics,
        }
    )
    return evaluated


def select_validation_hook(
    *,
    candidate_results: Mapping[str, Mapping[str, Any]],
    cache_details: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered validation-only hook selection gates."""

    if not candidate_results:
        raise ValueError("candidate_results cannot be empty")
    p0 = config["p0_3"]
    rows: list[dict[str, Any]] = []
    for hook_name in config["p0_3"]["candidate_hooks"]:
        result = candidate_results[hook_name]
        if result["split"] != "val":
            raise ValueError("hook selection may only use validation results")
        methods = result["methods"]
        late_quality = methods["late_quality_pooling"]["verification"]
        early_quality = methods["pre_suffix_quality_pooling"]["verification"]
        oracle = methods[
            "pre_suffix_metadata_oracle_pooling"
        ]["verification"]
        scenario_deltas = {
            scenario: (
                oracle[scenario]["roc_auc"]
                - late_quality[scenario]["roc_auc"]
            )
            for scenario in HARD_SCENARIOS
        }
        hard_wins = sum(value > 0.0 for value in scenario_deltas.values())
        hard_gain = (
            oracle["hard_average"]["roc_auc"]
            - late_quality["hard_average"]["roc_auc"]
        )
        early_quality_gain = (
            oracle["hard_average"]["roc_auc"]
            - early_quality["hard_average"]["roc_auc"]
        )
        clean_drop = (
            late_quality["clean"]["roc_auc"]
            - oracle["clean"]["roc_auc"]
        )
        rank1_drop = (
            late_quality["all"]["rank1_accuracy"]
            - oracle["all"]["rank1_accuracy"]
        )
        cache_replay = cache_details["pipeline_regression"][
            "suffix_replay"
        ][hook_name]["late_spatial_cosine_min"]
        cache_roundtrip_min = result["correctness"][
            "cache_roundtrip_token_cosine_min"
        ]
        checks = {
            "minimum_hard_gain": (
                hard_gain
                >= float(p0["selection_min_hard_auc_gain"])
            ),
            "beats_same_hook_quality": early_quality_gain > 0.0,
            "complementary_gain": (
                scenario_deltas["complementary_occlusion"]
                > float(p0["selection_min_complementary_auc_gain"])
            ),
            "hard_scenario_wins": (
                hard_wins
                >= int(p0["selection_min_hard_scenario_wins"])
            ),
            "clean_constraint": (
                clean_drop <= float(p0["selection_clean_auc_max_drop"])
            ),
            "rank1_constraint": (
                rank1_drop <= float(p0["selection_rank1_max_drop"])
            ),
            "suffix_replay": (
                cache_replay
                >= float(p0["suffix_replay_min_map_cosine"])
                and result["correctness"][
                    "suffix_replay_map_cosine_min"
                ]
                >= float(p0["suffix_replay_min_map_cosine"])
                and cache_roundtrip_min
                >= float(p0["suffix_replay_min_map_cosine"])
            ),
            "shape_norm_permutation": (
                result["correctness"]["all_output_shapes_valid"]
                and result["correctness"]["output_unit_norm_valid"]
                and result["correctness"]["permutation_equivalent"]
            ),
        }
        rows.append(
            {
                "hook_name": hook_name,
                "hook_shape": result["hook_shape"],
                "val_pooled_all_roc_auc": oracle["all"]["roc_auc"],
                "val_hard_macro_roc_auc": oracle["hard_average"]["roc_auc"],
                "val_clean_roc_auc": oracle["clean"]["roc_auc"],
                "val_rank1_accuracy": oracle["all"]["rank1_accuracy"],
                "val_eer": oracle["all"]["eer"],
                "val_margin": oracle["all"][
                    "mean_genuine_impostor_margin"
                ],
                "hard_gain_vs_late_quality": hard_gain,
                "hard_gain_vs_same_hook_quality": early_quality_gain,
                "clean_auc_drop_vs_late_quality": clean_drop,
                "rank1_drop_vs_late_quality": rank1_drop,
                "complementary_gain_vs_late_quality": scenario_deltas[
                    "complementary_occlusion"
                ],
                "hard_scenario_wins": hard_wins,
                "scenario_auc_deltas": scenario_deltas,
                "suffix_replay_map_cosine_min": result["correctness"][
                    "suffix_replay_map_cosine_min"
                ],
                "cache_roundtrip_token_cosine_min": (
                    cache_roundtrip_min
                ),
                "cache_roundtrip_token_max_abs_diff": (
                    result["correctness"].get(
                        "cache_roundtrip_token_max_abs_diff"
                    )
                ),
                "cache_suffix_cosine_min": cache_replay,
                "checks": checks,
                "constraints_passed": all(checks.values()),
            }
        )
    feasible = [row for row in rows if row["constraints_passed"]]
    preference = {
        name: index
        for index, name in enumerate(
            reversed(list(config["p0_3"]["candidate_hooks"]))
        )
    }
    ordered = sorted(
        feasible if feasible else rows,
        key=lambda row: (
            -float(row["val_hard_macro_roc_auc"]),
            -float(row["val_pooled_all_roc_auc"]),
            -float(row["val_clean_roc_auc"]),
            (
                preference[str(row["hook_name"])]
                if bool(p0["tie_break_prefer_later"])
                else 0
            ),
        ),
    )
    best = dict(ordered[0])
    selected = dict(ordered[0]) if feasible else None
    return {
        "selection_split": "val",
        "test_metrics_consulted": False,
        "candidate_count": len(rows),
        "feasible_candidate_count": len(feasible),
        "test_authorized": bool(feasible),
        "selection_rule": (
            "Require preregistered hard-gain, same-hook quality, "
            "complementary, scenario-win, clean, Rank-1, replay, shape, "
            "norm, and permutation gates. Among feasible hooks maximize val "
            "hard AUC, all AUC, clean AUC, then prefer the later/lower-cost "
            "candidate. If none pass, lock selected_hook=null and do not "
            "construct or evaluate test early features."
        ),
        "thresholds": {
            key: value
            for key, value in p0.items()
            if key.startswith("selection_")
            or key == "suffix_replay_min_map_cosine"
        },
        "selected": selected,
        "best_validation_candidate": best,
        "rows": rows,
    }


@torch.inference_mode()
def measure_early_hook_latency(
    *,
    hook_name: str,
    hook_shape: Sequence[int],
    cache_path: Path,
    dataset: TokenTrainingDataset,
    model: AdaFaceIRBackbone,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Measure batch-16 cached-activation pooling, suffix, and PCA latency."""

    if len(dataset) < 16:
        raise ValueError("latency measurement requires at least 16 sets")
    channels, height, width = [int(value) for value in hook_shape]
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    early = torch.from_numpy(np.array(cache[:16], copy=True)).to(device)
    samples = [dataset[index] for index in range(16)]
    quality = torch.stack(
        [sample["condition_quality"] for sample in samples]
    ).to(device)
    visibility = torch.stack(
        [sample["condition_visibility"] for sample in samples]
    ).to(device)
    same_identity = _same_identity_frame_mask(
        dataset,
        list(range(16)),
        early.shape[1],
    ).to(device)
    resized = resize_visibility_to_hook(
        visibility,
        height=height,
        width=width,
    )
    quality_weights = quality_spatial_weights(
        quality,
        height=height,
        width=width,
    )
    oracle_weights = metadata_oracle_weights(
        quality,
        resized,
        same_identity,
    )
    autocast_dtype = (
        torch.bfloat16
        if str(config["training"]["precision"]) == "bf16"
        else torch.float16
    )

    def run(weights: torch.Tensor) -> torch.Tensor:
        pooled = pool_raw_early_activation(early, weights)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            late = run_body_suffix_to_spatial(
                model,
                pooled,
                module_name=hook_name,
            )
        return _project_late_spatial(late, dataset.projector)

    functions = {
        "pre_suffix_quality_pooling": lambda: run(quality_weights),
        "pre_suffix_metadata_oracle_pooling": lambda: run(oracle_weights),
    }
    warmup = int(config["evaluation"]["latency_warmup_runs"])
    repeats = int(config["evaluation"]["latency_measurement_runs"])
    result: dict[str, Any] = {}
    for name, function in functions.items():
        for _ in range(warmup):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        result[name] = {
            "hook_name": hook_name,
            "hook_shape": [channels, height, width],
            "batch_size": 16,
            "measurement_scope": (
                "cached float16 early activation through raw pooling, "
                "frozen suffix, output_layer.1, and existing PCA"
            ),
            "warmup_runs": warmup,
            "measurement_runs": repeats,
            "batch_latency_ms": elapsed * 1000.0 / repeats,
            "per_set_latency_ms": elapsed * 1000.0 / (repeats * 16),
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
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _search_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected_name = (
        selection["selected"]["hook_name"]
        if selection["selected"] is not None
        else None
    )
    rows = []
    for value in selection["rows"]:
        rows.append(
            {
                "hook_name": value["hook_name"],
                "hook_shape": json.dumps(value["hook_shape"]),
                "selected": value["hook_name"] == selected_name,
                "constraints_passed": value["constraints_passed"],
                "val_pooled_all_roc_auc": value[
                    "val_pooled_all_roc_auc"
                ],
                "val_hard_macro_roc_auc": value[
                    "val_hard_macro_roc_auc"
                ],
                "val_clean_roc_auc": value["val_clean_roc_auc"],
                "val_rank1_accuracy": value["val_rank1_accuracy"],
                "val_eer": value["val_eer"],
                "val_margin": value["val_margin"],
                "hard_gain_vs_late_quality": value[
                    "hard_gain_vs_late_quality"
                ],
                "hard_gain_vs_same_hook_quality": value[
                    "hard_gain_vs_same_hook_quality"
                ],
                "complementary_gain_vs_late_quality": value[
                    "complementary_gain_vs_late_quality"
                ],
                "hard_scenario_wins": value["hard_scenario_wins"],
                "clean_auc_drop_vs_late_quality": value[
                    "clean_auc_drop_vs_late_quality"
                ],
                "rank1_drop_vs_late_quality": value[
                    "rank1_drop_vs_late_quality"
                ],
                "suffix_replay_map_cosine_min": value[
                    "suffix_replay_map_cosine_min"
                ],
                "cache_roundtrip_token_cosine_min": value[
                    "cache_roundtrip_token_cosine_min"
                ],
                "cache_roundtrip_token_max_abs_diff": value[
                    "cache_roundtrip_token_max_abs_diff"
                ],
                **{
                    f"check_{key}": flag
                    for key, flag in value["checks"].items()
                },
                **{
                    f"delta_auc_{scenario}": delta
                    for scenario, delta in value[
                        "scenario_auc_deltas"
                    ].items()
                },
            }
        )
    return rows


def _result_items(
    *,
    validation_results: Mapping[str, Mapping[str, Any]],
    test_result: Mapping[str, Any] | None,
) -> list[tuple[str, str, Mapping[str, Any]]]:
    items = [
        ("val", hook_name, result)
        for hook_name, result in validation_results.items()
    ]
    if test_result is not None:
        items.append(
            ("test", str(test_result["hook_name"]), test_result)
        )
    return items


def _core_rows(
    *,
    validation_results: Mapping[str, Mapping[str, Any]],
    test_result: Mapping[str, Any] | None,
    latency: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, hook_name, result in _result_items(
        validation_results=validation_results,
        test_result=test_result,
    ):
        late_quality = result["methods"]["late_quality_pooling"][
            "verification"
        ]
        for method in ALL_METHODS:
            values = result["methods"][method]
            verification = values["verification"]
            all_metrics = verification["all"]
            latency_values = latency.get(method, {})
            if latency_values.get("hook_name") != hook_name:
                latency_values = {}
            rows.append(
                {
                    "split": split,
                    "hook_name": hook_name,
                    "method": method,
                    "metadata_oracle": (
                        method
                        in {
                            "late_metadata_oracle_pooling",
                            "pre_suffix_metadata_oracle_pooling",
                        }
                    ),
                    "deployable": (
                        method
                        not in {
                            "late_metadata_oracle_pooling",
                            "pre_suffix_metadata_oracle_pooling",
                        }
                    ),
                    "pooled_all_roc_auc": all_metrics["roc_auc"],
                    "hard_macro_roc_auc": verification["hard_average"][
                        "roc_auc"
                    ],
                    "delta_hard_auc_vs_late_quality": (
                        verification["hard_average"]["roc_auc"]
                        - late_quality["hard_average"]["roc_auc"]
                    ),
                    "clean_roc_auc": verification["clean"]["roc_auc"],
                    "eer": all_metrics["eer"],
                    "tar_at_far_1e-2": all_metrics["tar_at_far"]["0.01"][
                        "tar"
                    ],
                    "tar_at_far_1e-3": all_metrics["tar_at_far"]["0.001"][
                        "tar"
                    ],
                    "rank1_identification_accuracy": all_metrics[
                        "rank1_accuracy"
                    ],
                    "mean_genuine_impostor_margin": all_metrics[
                        "mean_genuine_impostor_margin"
                    ],
                    "teacher_map_cosine": values["teacher_similarity"]["all"][
                        "map_cosine"
                    ],
                    "batch16_cached_suffix_latency_ms": latency_values.get(
                        "batch_latency_ms",
                        "",
                    ),
                    "per_set_cached_suffix_latency_ms": latency_values.get(
                        "per_set_latency_ms",
                        "",
                    ),
                }
            )
    return rows


def _scenario_rows(
    *,
    validation_results: Mapping[str, Mapping[str, Any]],
    test_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, hook_name, result in _result_items(
        validation_results=validation_results,
        test_result=test_result,
    ):
        late_quality = result["methods"]["late_quality_pooling"][
            "verification"
        ]
        for method in ALL_METHODS:
            values = result["methods"][method]
            for scenario in (
                "clean",
                *HARD_SCENARIOS,
            ):
                metric = values["verification"][scenario]
                similarity = values["teacher_similarity"][scenario]
                rows.append(
                    {
                        "split": split,
                        "hook_name": hook_name,
                        "method": method,
                        "scenario": scenario,
                        "roc_auc": metric["roc_auc"],
                        "delta_auc_vs_late_quality": (
                            metric["roc_auc"]
                            - late_quality[scenario]["roc_auc"]
                        ),
                        "eer": metric["eer"],
                        "tar_at_far_1e-2": metric["tar_at_far"]["0.01"][
                            "tar"
                        ],
                        "tar_at_far_1e-3": metric["tar_at_far"]["0.001"][
                            "tar"
                        ],
                        "rank1_identification_accuracy": metric[
                            "rank1_accuracy"
                        ],
                        "mean_genuine_impostor_margin": metric[
                            "mean_genuine_impostor_margin"
                        ],
                        "teacher_map_cosine": similarity["map_cosine"],
                    }
                )
    return rows


def _diagnostic_rows(
    *,
    validation_results: Mapping[str, Mapping[str, Any]],
    test_result: Mapping[str, Any] | None,
    cache_details: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(
        split: str,
        hook: str,
        category: str,
        metric: str,
        value: Any,
    ) -> None:
        rows.append(
            {
                "split": split,
                "hook_name": hook,
                "category": category,
                "metric": metric,
                "value": value,
            }
        )

    for split, hook_name, result in _result_items(
        validation_results=validation_results,
        test_result=test_result,
    ):
        for category in (
            "correctness",
            "oracle_weight_diagnostics",
        ):
            for metric, value in result[category].items():
                append(split, hook_name, category, metric, value)
        for method, values in result["pooled_raw_statistics"].items():
            for metric, value in values.items():
                append(
                    split,
                    hook_name,
                    f"raw_activation_{method}",
                    metric,
                    value,
                )
    for split, values in cache_details.items():
        for hook_name, replay in values["pipeline_regression"][
            "suffix_replay"
        ].items():
            for metric, value in replay.items():
                append(split, hook_name, "cache_suffix_replay", metric, value)
        for metric, value in values["pipeline_regression"].items():
            if metric != "suffix_replay":
                append(split, "all", "cache_pipeline", metric, value)
    return rows


def _test_gate(
    *,
    test_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = test_result["methods"]
    late = methods["late_quality_pooling"]["verification"]
    early_quality = methods["pre_suffix_quality_pooling"]["verification"]
    oracle = methods["pre_suffix_metadata_oracle_pooling"]["verification"]
    deltas = {
        scenario: oracle[scenario]["roc_auc"] - late[scenario]["roc_auc"]
        for scenario in HARD_SCENARIOS
    }
    cache_roundtrip_min = test_result["correctness"][
        "cache_roundtrip_token_cosine_min"
    ]
    checks = {
        "hard_auc_gain": (
            oracle["hard_average"]["roc_auc"]
            > late["hard_average"]["roc_auc"]
        ),
        "beats_same_hook_quality": (
            oracle["hard_average"]["roc_auc"]
            > early_quality["hard_average"]["roc_auc"]
        ),
        "complementary_gain": deltas["complementary_occlusion"] > 0.0,
        "hard_scenario_wins": (
            sum(value > 0.0 for value in deltas.values())
            >= int(config["p0_3"]["selection_min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            late["clean"]["roc_auc"] - oracle["clean"]["roc_auc"]
            <= float(config["p0_3"]["selection_clean_auc_max_drop"])
        ),
        "rank1_constraint": (
            late["all"]["rank1_accuracy"] - oracle["all"]["rank1_accuracy"]
            <= float(config["p0_3"]["selection_rank1_max_drop"])
        ),
        "correctness": all(
            (
                test_result["correctness"]["all_output_shapes_valid"],
                test_result["correctness"]["output_unit_norm_valid"],
                test_result["correctness"]["permutation_equivalent"],
                test_result["correctness"][
                    "suffix_replay_map_cosine_min"
                ]
                >= float(
                    config["p0_3"]["suffix_replay_min_map_cosine"]
                ),
                cache_roundtrip_min
                >= float(
                    config["p0_3"]["suffix_replay_min_map_cosine"]
                ),
            )
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "scenario_auc_deltas": deltas,
        "hard_gain_vs_late_quality": (
            oracle["hard_average"]["roc_auc"]
            - late["hard_average"]["roc_auc"]
        ),
        "hard_gain_vs_same_hook_quality": (
            oracle["hard_average"]["roc_auc"]
            - early_quality["hard_average"]["roc_auc"]
        ),
    }


def build_p0_3_decision(
    *,
    selection: Mapping[str, Any],
    test_result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    correctness_names = (
        "candidate_hooks_and_shapes_preregistered",
        "selection_used_validation_only",
        "selection_lock_reloaded_before_test",
        "selection_lock_unchanged",
        "test_policy_followed",
        "validation_pipeline_reproduced",
        "validation_outputs_correct",
        "test_outputs_correct_if_evaluated",
        "same_gallery_protocol",
    )
    resource_names = (
        "dataset_valid_after_lock",
        "quantization_valid_after_lock",
        "quantization_source_link_matches",
        "set_manifest_unchanged",
        "quantization_manifest_unchanged",
        "projection_unchanged",
        "codebook_unchanged",
        "checkpoint_unchanged",
        "backbone_frozen_eval",
        "no_training_or_quantized_early_method",
        "memory_within_configured_cap",
        "protected_outputs_unchanged_before_artifact_write",
        "checkpoints_unchanged_before_artifact_write",
    )
    missing = [
        name
        for name in (*correctness_names, *resource_names)
        if name not in checks
    ]
    if missing:
        raise ValueError(
            "P0-3 decision is missing required runtime checks: "
            + ", ".join(missing)
        )
    failed_correctness = [
        name for name in correctness_names if not bool(checks[name])
    ]
    failed_resource = [
        name for name in resource_names if not bool(checks[name])
    ]
    correctness_passed = not failed_correctness
    resource_passed = not failed_resource
    if not selection["test_authorized"]:
        return {
            "validation_headroom_passed": False,
            "test_evaluated": False,
            "test_gate": None,
            "scientific_gate_passed": False,
            "correctness_gate_passed": correctness_passed,
            "resource_gate_passed": resource_passed,
            "failed_correctness_checks": failed_correctness,
            "failed_resource_checks": failed_resource,
            "go": False,
            "recommendation": (
                "stop_current_adaface_hook_local_aggregation_keep_quality_pooling"
            ),
        }
    if test_result is None:
        raise ValueError("authorized test result is missing")
    test_gate = _test_gate(test_result=test_result, config=config)
    scientific_passed = bool(test_gate["passed"])
    go = scientific_passed and correctness_passed and resource_passed
    if go:
        recommendation = (
            "develop_locked_hook_visibility_estimator_on_new_holdout"
        )
    elif not correctness_passed or not resource_passed:
        recommendation = "no_go_operational_or_correctness_gate_failed"
    else:
        recommendation = "no_go_split_instability_keep_quality_pooling"
    return {
        "validation_headroom_passed": True,
        "test_evaluated": True,
        "test_gate": test_gate,
        "scientific_gate_passed": scientific_passed,
        "correctness_gate_passed": correctness_passed,
        "resource_gate_passed": resource_passed,
        "failed_correctness_checks": failed_correctness,
        "failed_resource_checks": failed_resource,
        "go": go,
        "recommendation": recommendation,
    }


def _report_text(
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    decision: Mapping[str, Any],
    validation_results: Mapping[str, Mapping[str, Any]],
    test_result: Mapping[str, Any] | None,
) -> str:
    best = selection["best_validation_candidate"]
    candidate_text = "、".join(
        f"`{name}`" for name in preregistration["candidate_hooks"]
    )
    rows = []
    for value in selection["rows"]:
        rows.append(
            "| "
            + " | ".join(
                (
                    str(value["hook_name"]),
                    str(value["hook_shape"]),
                    f"{value['val_hard_macro_roc_auc']:.6f}",
                    f"{value['hard_gain_vs_late_quality']:+.6f}",
                    f"{value['hard_gain_vs_same_hook_quality']:+.6f}",
                    f"{value['complementary_gain_vs_late_quality']:+.6f}",
                    str(value["hard_scenario_wins"]),
                    str(value["constraints_passed"]),
                )
            )
            + " |"
        )
    if selection["test_authorized"]:
        selected = selection["selected"]
        selection_text = (
            f"Validation 锁定 `{selected['hook_name']}`，随后仅对该层运行一次 "
            "test。"
        )
    else:
        selection_text = (
            "没有候选通过 validation 预注册门槛；lock 中 "
            "`selected=null, test_authorized=false`，因此没有构造或评估 "
            "test early features。"
        )
    if decision["recommendation"] == (
        "stop_current_adaface_hook_local_aggregation_keep_quality_pooling"
    ):
        recommendation = (
            "停止当前 AdaFace hook 上的局部聚合分支，保留 late quality "
            "pooling。不要训练 visibility estimator、Transformer 或 diffusion。"
        )
    elif decision["recommendation"] == (
        "no_go_split_instability_keep_quality_pooling"
    ):
        recommendation = (
            "Validation 有 headroom 但 locked test 未复现，判定 split "
            "instability；保留 quality pooling，不进入训练阶段。"
        )
    elif decision["recommendation"] == (
        "no_go_operational_or_correctness_gate_failed"
    ):
        recommendation = (
            "科学指标不能覆盖运行正确性或资源门失败；保留 quality pooling，"
            "先修复并重跑，不进入 visibility estimator 训练。"
        )
    else:
        recommendation = (
            "Earlier hook 的 metadata oracle 在 val/test 均通过；下一步只能在"
            "锁定层开发可部署 visibility estimator，并使用未被 P0-1/P0-2 "
            "观察过的新身份 holdout 复验。"
        )
    test_section = ""
    if test_result is not None:
        test_gate = decision["test_gate"]
        scenario_lines = "\n".join(
            f"- {scenario}: {delta:+.6f}"
            for scenario, delta in test_gate[
                "scenario_auc_deltas"
            ].items()
        )
        check_lines = "\n".join(
            f"- {name}: {passed}"
            for name, passed in test_gate["checks"].items()
        )
        test_section = f"""
## Locked test

- hard AUC gain vs late quality:
  {test_gate["hard_gain_vs_late_quality"]:+.6f}
- hard AUC gain vs same-hook quality:
  {test_gate["hard_gain_vs_same_hook_quality"]:+.6f}
- gate passed: {test_gate["passed"]}

Scenario AUC deltas vs late quality:

{scenario_lines}

Test gate checks:

{check_lines}
"""
    return f"""# P0-3 earlier-hook metadata-oracle headroom

## 协议

- 候选层固定为 {candidate_text}；不扫描其他 block。
- 在 raw early activation 上聚合，再运行冻结 AdaFace suffix 到现有
  `output_layer.1`，最后使用原 stage10 PCA 和 clean disjoint-teacher gallery。
- Visibility 只做 7×7→early-grid nearest resize；不搜索膨胀、腐蚀或阈值。
- Metadata oracle 使用 input visibility 与 source identity，只是 headroom
  diagnostic，不是可部署方法或理论上界。
- {selection_text}

## Validation layer search

| Hook | Shape | Oracle hard AUC | Δ late quality | Δ same-hook quality | Δ complementary | Hard wins | Pass |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Validation 最佳候选是 `{best["hook_name"]}`，hard AUC
{best["val_hard_macro_roc_auc"]:.6f}，相对 late quality
{best["hard_gain_vs_late_quality"]:+.6f}。
{test_section}
## Go/No-Go

- validation headroom: {decision["validation_headroom_passed"]}
- test evaluated: {decision["test_evaluated"]}
- scientific gate: {decision["scientific_gate_passed"]}
- correctness gate: {decision["correctness_gate_passed"]}
- resource/memory gate: {decision["resource_gate_passed"]}
- failed correctness checks: {decision["failed_correctness_checks"]}
- failed resource checks: {decision["failed_resource_checks"]}
- final Go: {decision["go"]}
- 建议：{recommendation}

现有 test identities 已在 P0-1/P0-2 中被观察。因此即使本阶段做到
lock-before-test，它也只是 P0-3 内部的 test-blind 锁参流程，不是全项目层面的
全新确认性 holdout。
"""


def write_p0_3_artifacts(
    *,
    output_directory: Path,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    validation_results: Mapping[str, Mapping[str, Any]],
    test_result: Mapping[str, Any] | None,
    cache_details: Mapping[str, Any],
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: Mapping[str, bool],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    search_path = output_directory / "validation_layer_search.csv"
    core_path = output_directory / "core_comparison.csv"
    scenario_path = output_directory / "scenario_metrics.csv"
    layer_path = output_directory / "layer_diagnostics.csv"
    diagnostics_path = output_directory / "diagnostics.json"
    report_path = output_directory / "REPORT.md"
    _write_csv(search_path, _search_rows(selection))
    _write_csv(
        core_path,
        _core_rows(
            validation_results=validation_results,
            test_result=test_result,
            latency=latency,
        ),
    )
    _write_csv(
        scenario_path,
        _scenario_rows(
            validation_results=validation_results,
            test_result=test_result,
        ),
    )
    _write_csv(
        layer_path,
        _diagnostic_rows(
            validation_results=validation_results,
            test_result=test_result,
            cache_details=cache_details,
        ),
    )
    decision = build_p0_3_decision(
        selection=selection,
        test_result=test_result,
        config=config,
        checks=checks,
    )
    diagnostics = {
        "schema_version": 1,
        "preregistration": dict(preregistration),
        "selection": dict(selection),
        "decision": decision,
        "cache": dict(cache_details),
        "latency": dict(latency),
        "validation_correctness": {
            hook: result["correctness"]
            for hook, result in validation_results.items()
        },
        "validation_oracle_weights": {
            hook: result["oracle_weight_diagnostics"]
            for hook, result in validation_results.items()
        },
        "test_correctness": (
            None if test_result is None else test_result["correctness"]
        ),
        "test_oracle_weights": (
            None
            if test_result is None
            else test_result["oracle_weight_diagnostics"]
        ),
        "checks": dict(checks),
        "reproducibility": dict(reproducibility),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _report_text(
            preregistration=preregistration,
            selection=selection,
            decision=decision,
            validation_results=validation_results,
            test_result=test_result,
        ),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "validation_layer_search_csv": str(search_path),
        "core_comparison_csv": str(core_path),
        "scenario_metrics_csv": str(scenario_path),
        "layer_diagnostics_csv": str(layer_path),
        "diagnostics_json": str(diagnostics_path),
        "decision": decision,
    }
