"""Frozen AdaFace IR backbone and automatic spatial-feature hook discovery.

The module layout intentionally matches ``mk-minchul/AdaFace`` ``net.py`` so
official checkpoints load without key remapping beyond their Lightning prefix.
The reference implementation is MIT licensed.
"""

from __future__ import annotations

import hashlib
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torch import nn


ARCFACE_TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
Bottleneck = namedtuple("Block", ["in_channel", "depth", "stride"])


class Flatten(nn.Module):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.view(input_tensor.size(0), -1)


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    depth,
                    (1, 1),
                    stride,
                    bias=False,
                ),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(
                in_channel,
                depth,
                (3, 3),
                (1, 1),
                1,
                bias=False,
            ),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.res_layer(input_tensor) + self.shortcut_layer(input_tensor)


def _get_block(
    in_channel: int,
    depth: int,
    num_units: int,
    stride: int = 2,
) -> list[Bottleneck]:
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def _get_blocks(num_layers: int) -> list[list[Bottleneck]]:
    units = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 14, 3),
        100: (3, 13, 30, 3),
    }
    if num_layers not in units:
        raise ValueError(f"unsupported AdaFace IR depth: {num_layers}")
    counts = units[num_layers]
    return [
        _get_block(64, 64, counts[0]),
        _get_block(64, 128, counts[1]),
        _get_block(128, 256, counts[2]),
        _get_block(256, 512, counts[3]),
    ]


class AdaFaceIRBackbone(nn.Module):
    """IR-18/34/50/100 with official AdaFace parameter names."""

    def __init__(
        self,
        *,
        input_size: tuple[int, int] = (112, 112),
        num_layers: int = 50,
    ) -> None:
        super().__init__()
        if input_size not in {(112, 112), (224, 224)}:
            raise ValueError("AdaFace input_size must be 112x112 or 224x224")
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        modules = []
        for block in _get_blocks(num_layers):
            for unit in block:
                modules.append(
                    BasicBlockIR(
                        unit.in_channel,
                        unit.depth,
                        unit.stride,
                    )
                )
        self.body = nn.Sequential(*modules)
        spatial_size = 7 if input_size[0] == 112 else 14
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * spatial_size * spatial_size, 512),
            nn.BatchNorm1d(512, affine=False),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_layer(input_tensor)
        hidden = self.body(hidden)
        embedding = self.output_layer(hidden)
        norm = torch.norm(embedding, 2, 1, True)
        normalized = torch.div(embedding, norm.clamp_min(1e-12))
        return normalized, norm


@dataclass(frozen=True)
class CheckpointLoadReport:
    path: Path
    sha256: str
    source_keys: int
    model_keys: int
    ignored_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class SpatialHookInfo:
    module_name: str
    shape: tuple[int, int, int, int]
    dtype: str


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(
    payload: Any,
) -> tuple[Mapping[str, torch.Tensor], tuple[str, ...]]:
    if isinstance(payload, Mapping) and isinstance(
        payload.get("state_dict"),
        Mapping,
    ):
        state = payload["state_dict"]
    elif isinstance(payload, Mapping):
        state = payload
    else:
        raise ValueError("AdaFace checkpoint must contain a state dictionary")

    tensor_items = {
        str(key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    model_prefixes = ("module.model.", "model.")
    has_model_prefix = any(
        key.startswith(model_prefixes) for key in tensor_items
    )
    normalized: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    for key, value in tensor_items.items():
        original_key = key
        if has_model_prefix and not key.startswith(model_prefixes):
            ignored.append(original_key)
            continue
        if not isinstance(value, torch.Tensor):
            continue
        key = str(key)
        for prefix in ("module.model.", "model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        normalized[key] = value
    return normalized, tuple(sorted(ignored))


def load_adaface_ir50(
    checkpoint: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[AdaFaceIRBackbone, CheckpointLoadReport]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AdaFace checkpoint not found: {checkpoint}")
    model = AdaFaceIRBackbone(num_layers=50)
    payload = torch.load(
        checkpoint,
        map_location=map_location,
        weights_only=True,
    )
    state, ignored_keys = _checkpoint_state(payload)
    incompatible = model.load_state_dict(state, strict=False)
    report = CheckpointLoadReport(
        path=checkpoint,
        sha256=_sha256(checkpoint),
        source_keys=len(state),
        model_keys=len(model.state_dict()),
        ignored_keys=ignored_keys,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
    if report.missing_keys or report.unexpected_keys:
        raise ValueError(
            "AdaFace checkpoint is incompatible: "
            f"missing={report.missing_keys[:5]}, "
            f"unexpected={report.unexpected_keys[:5]}"
        )
    model.eval()
    model.requires_grad_(False)
    return model, report


def discover_spatial_hook(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 112, 112),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SpatialHookInfo:
    """Find the last leaf-module 4D feature before embedding flattening."""

    execution: list[tuple[str, tuple[int, ...], torch.dtype]] = []
    handles = []

    def record(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: Any,
        ) -> None:
            if isinstance(output, torch.Tensor):
                execution.append((name, tuple(output.shape), output.dtype))

        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(record(name)))
    try:
        probe = torch.zeros(input_shape, device=device, dtype=dtype)
        with torch.inference_mode():
            model(probe)
    finally:
        for handle in handles:
            handle.remove()

    candidates = [
        item
        for item in execution
        if len(item[1]) == 4 and item[1][-2] > 1 and item[1][-1] > 1
    ]
    if not candidates:
        raise RuntimeError("no spatial feature module was observed")
    name, shape, observed_dtype = candidates[-1]
    return SpatialHookInfo(
        module_name=name,
        shape=(
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
            int(shape[3]),
        ),
        dtype=str(observed_dtype),
    )


def extract_spatial_and_embedding(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"spatial hook module not found: {module_name}")
    captured: list[torch.Tensor] = []

    def capture(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("spatial hook output must be a tensor")
        captured.append(output.detach())

    handle = modules[module_name].register_forward_hook(capture)
    try:
        embedding, norm = model(input_tensor)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one spatial capture, got {len(captured)}")
    return captured[0], embedding, norm


def probe_named_spatial_hooks(
    model: nn.Module,
    module_names: list[str] | tuple[str, ...],
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 112, 112),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, SpatialHookInfo]:
    """Probe a preregistered set of parent-module spatial endpoints."""

    if not module_names or len(set(module_names)) != len(module_names):
        raise ValueError("module_names must be non-empty and unique")
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise KeyError(f"spatial hook modules not found: {missing}")
    captured: dict[str, list[torch.Tensor]] = {
        name: [] for name in module_names
    }
    handles = []

    def capture(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: Any,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"spatial hook {name} output is not a tensor")
            captured[name].append(output.detach())

        return hook

    try:
        for name in module_names:
            handles.append(
                modules[name].register_forward_hook(capture(name))
            )
        with torch.inference_mode():
            model(torch.zeros(input_shape, device=device, dtype=dtype))
    finally:
        for handle in handles:
            handle.remove()
    result: dict[str, SpatialHookInfo] = {}
    for name in module_names:
        if len(captured[name]) != 1:
            raise RuntimeError(
                f"expected one capture for {name}, got {len(captured[name])}"
            )
        value = captured[name][0]
        if value.ndim != 4 or value.shape[-2] <= 1 or value.shape[-1] <= 1:
            raise ValueError(f"hook {name} is not a spatial feature map")
        result[name] = SpatialHookInfo(
            module_name=name,
            shape=tuple(int(item) for item in value.shape),
            dtype=str(value.dtype),
        )
    return result


def extract_named_spatial_and_embedding(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    module_names: list[str] | tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Capture multiple preregistered endpoints in one frozen forward."""

    if not module_names or len(set(module_names)) != len(module_names):
        raise ValueError("module_names must be non-empty and unique")
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise KeyError(f"spatial hook modules not found: {missing}")
    captured: dict[str, list[torch.Tensor]] = {
        name: [] for name in module_names
    }
    handles = []

    def capture(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: Any,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"spatial hook {name} output is not a tensor")
            captured[name].append(output.detach())

        return hook

    try:
        for name in module_names:
            handles.append(
                modules[name].register_forward_hook(capture(name))
            )
        embedding, norm = model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()
    values: dict[str, torch.Tensor] = {}
    for name in module_names:
        if len(captured[name]) != 1:
            raise RuntimeError(
                f"expected one capture for {name}, got {len(captured[name])}"
            )
        values[name] = captured[name][0]
    return values, embedding, norm


def run_body_suffix_to_spatial(
    model: AdaFaceIRBackbone,
    activation: torch.Tensor,
    *,
    module_name: str,
) -> torch.Tensor:
    """Run a raw ``body.<index>`` activation to ``output_layer.1``."""

    prefix = "body."
    if not module_name.startswith(prefix):
        raise ValueError("suffix cut must be a body.<index> module")
    try:
        body_index = int(module_name[len(prefix) :])
    except ValueError as exc:
        raise ValueError("suffix cut must be a body.<index> module") from exc
    blocks = list(model.body.children())
    if body_index < 0 or body_index >= len(blocks):
        raise ValueError(f"body index is out of range: {body_index}")
    hidden = activation
    for block in blocks[body_index + 1 :]:
        hidden = block(hidden)
    hidden = model.output_layer[0](hidden)
    hidden = model.output_layer[1](hidden)
    return hidden


def similarity_transform(
    source_points: np.ndarray,
    destination_points: np.ndarray = ARCFACE_TEMPLATE_112,
) -> np.ndarray:
    """Return a deterministic least-squares 2D similarity transform."""

    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    if source.shape != (5, 2) or destination.shape != (5, 2):
        raise ValueError("face landmarks must have shape [5, 2]")
    design = np.zeros((10, 4), dtype=np.float64)
    target = destination.reshape(-1)
    for index, (x_coord, y_coord) in enumerate(source):
        design[2 * index] = (x_coord, -y_coord, 1.0, 0.0)
        design[2 * index + 1] = (y_coord, x_coord, 0.0, 1.0)
    a_value, b_value, x_offset, y_offset = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )[0]
    return np.asarray(
        [
            [a_value, -b_value, x_offset],
            [b_value, a_value, y_offset],
        ],
        dtype=np.float32,
    )


def align_and_normalize_bgr(
    image_path: Path,
    landmarks: np.ndarray,
    *,
    output_size: int = 112,
) -> torch.Tensor:
    try:
        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"failed to read image: {image_path}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {image_path}")
    matrix = similarity_transform(landmarks)
    aligned = cv2.warpAffine(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    normalized = (aligned.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(normalized.transpose(2, 0, 1)).contiguous()
