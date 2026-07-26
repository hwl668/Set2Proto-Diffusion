"""Reproducible RNG setup for Python, NumPy, PyTorch, and DataLoader workers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic: bool
    python_hash_seed: str


def seed_everything(seed: int, deterministic: bool = True) -> SeedState:
    """Seed all RNGs used by the MVP and configure deterministic CUDA behavior."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if torch.cuda.is_available():
        # Flash/memory-efficient SDPA backward can be nondeterministic on
        # Windows. Math SDPA is still the native PyTorch implementation and
        # gives exact repeatability for deterministic research runs.
        torch.backends.cuda.enable_flash_sdp(not deterministic)
        torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    return SeedState(
        seed=seed,
        deterministic=deterministic,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
    )


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return a separately seeded generator for deterministic sample ordering."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the worker-specific PyTorch initial seed."""

    del worker_id  # The worker id is already folded into torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
