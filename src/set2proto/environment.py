"""Runtime, Git, GPU, and native SDPA environment inspection."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


def config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_read_only_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 5.0,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def collect_git_info(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    git_executable = shutil.which("git")
    if git_executable is None:
        return {
            "available": False,
            "repository": False,
            "commit": None,
            "dirty": None,
            "branch": None,
            "status": "git-not-installed",
        }

    probe = _run_read_only_command(
        [git_executable, "rev-parse", "--is-inside-work-tree"],
        cwd=root,
    )
    if probe is None or probe.returncode != 0 or probe.stdout.strip() != "true":
        return {
            "available": True,
            "repository": False,
            "commit": None,
            "dirty": None,
            "branch": None,
            "status": "no-repository",
        }

    commit_result = _run_read_only_command(
        [git_executable, "rev-parse", "HEAD"],
        cwd=root,
    )
    branch_result = _run_read_only_command(
        [git_executable, "branch", "--show-current"],
        cwd=root,
    )
    status_result = _run_read_only_command(
        [git_executable, "status", "--porcelain"],
        cwd=root,
    )
    commit = (
        commit_result.stdout.strip()
        if commit_result is not None and commit_result.returncode == 0
        else None
    )
    branch = (
        branch_result.stdout.strip()
        if branch_result is not None and branch_result.returncode == 0
        else None
    )
    dirty = (
        bool(status_result.stdout.strip())
        if status_result is not None and status_result.returncode == 0
        else None
    )
    return {
        "available": True,
        "repository": True,
        "commit": commit,
        "dirty": dirty,
        "branch": branch or None,
        "status": "ok",
    }


def collect_nvidia_smi(project_root: str | Path) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "gpus": [], "error": "nvidia-smi-not-found"}
    result = _run_read_only_command(
        [
            executable,
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path(project_root).resolve(),
    )
    if result is None or result.returncode != 0:
        return {
            "available": False,
            "gpus": [],
            "error": "nvidia-smi-query-failed",
        }

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, driver, total, free = parts
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
            }
        )
    return {"available": True, "gpus": gpus, "error": None}


def collect_torch_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                    "total_memory_mib": round(total_bytes / 1024**2, 2),
                    "free_memory_mib": round(free_bytes / 1024**2, 2),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    return {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": devices,
        "native_sdpa_available": hasattr(F, "scaled_dot_product_attention"),
    }


def probe_native_sdpa(
    *,
    require_cuda: bool,
    precision: str,
) -> dict[str, Any]:
    """Run a tiny attention operation and report its memory footprint."""

    result: dict[str, Any] = {
        "success": False,
        "device": None,
        "dtype": precision,
        "output_shape": None,
        "all_finite": False,
        "peak_allocated_mib": 0.0,
        "peak_reserved_mib": 0.0,
        "error": None,
    }
    if not hasattr(F, "scaled_dot_product_attention"):
        result["error"] = "native-scaled-dot-product-attention-unavailable"
        return result
    if require_cuda and not torch.cuda.is_available():
        result["error"] = "cuda-required-but-unavailable"
        return result

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype_lookup = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_lookup[precision]
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    result["device"] = str(device)
    result["dtype"] = str(dtype).removeprefix("torch.")

    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            generator = torch.Generator(device=device)
            generator.manual_seed(17)
            shape = (1, 8, 49, 32)
            query = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            key = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            value = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            result["output_shape"] = list(output.shape)
            result["all_finite"] = bool(torch.isfinite(output).all().item())
            if device.type == "cuda":
                result["peak_allocated_mib"] = round(
                    torch.cuda.max_memory_allocated(device) / 1024**2,
                    3,
                )
                result["peak_reserved_mib"] = round(
                    torch.cuda.max_memory_reserved(device) / 1024**2,
                    3,
                )
            result["success"] = (
                result["all_finite"] and result["output_shape"] == list(shape)
            )
            del query, key, value, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # The diagnostic must preserve the actual error.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def collect_environment(
    *,
    project_root: str | Path,
    config: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "command": list(command),
        "working_directory": str(Path.cwd().resolve()),
        "project_root": str(root),
        "config_sha256": config_sha256(config),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "timezone": datetime.now().astimezone().tzname(),
        },
        "process": {
            "pid": os.getpid(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "torch": collect_torch_info(),
        "nvidia_smi": collect_nvidia_smi(root),
        "git": collect_git_info(root),
    }

