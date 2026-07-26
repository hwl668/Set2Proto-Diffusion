"""Collision-safe experiment directory and checkpoint naming helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    logs: Path
    artifacts: Path

    def checkpoint_path(self, step: int) -> Path:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        return self.checkpoints / f"checkpoint_step_{step:08d}.pt"


def default_run_id(profile: str, seed: int, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S-%f")
    return f"{timestamp}_{profile}_s{seed}"


def create_run_paths(
    output_root: str | Path,
    *,
    profile: str,
    seed: int,
    run_id: str | None = None,
    resume: bool = False,
) -> RunPaths:
    """Create a run directory, refusing to overwrite an existing run by default."""

    output_path = Path(output_root).expanduser().resolve()
    selected_run_id = run_id or default_run_id(profile, seed)
    if not _SAFE_RUN_ID.fullmatch(selected_run_id):
        raise ValueError(
            "run_id must use only letters, numbers, dots, underscores, and hyphens"
        )

    run_root = output_path / selected_run_id
    if run_root.exists() and not resume:
        raise FileExistsError(
            f"run directory already exists and will not be overwritten: {run_root}"
        )
    if resume and not run_root.is_dir():
        raise FileNotFoundError(f"cannot resume missing run directory: {run_root}")

    checkpoints = run_root / "checkpoints"
    logs = run_root / "logs"
    artifacts = run_root / "artifacts"
    for path in (checkpoints, logs, artifacts):
        path.mkdir(parents=True, exist_ok=resume)

    return RunPaths(
        root=run_root,
        checkpoints=checkpoints,
        logs=logs,
        artifacts=artifacts,
    )


def ensure_workspace_directories(paths: dict[str, str]) -> dict[str, Path]:
    """Create the configured local data/artifact roots without deleting content."""

    required = ("data_dir", "weights_dir", "cache_dir", "output_dir")
    resolved: dict[str, Path] = {}
    for key in required:
        if key not in paths:
            raise KeyError(f"missing configured workspace path: {key}")
        path = Path(paths[key]).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        resolved[key] = path
    return resolved

