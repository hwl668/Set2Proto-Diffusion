"""Structured JSONL/CSV experiment logging with atomic run manifests."""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


class ExperimentLogger:
    """Append-only event/metric logs plus an atomically written manifest."""

    _METRIC_COLUMNS = (
        "timestamp",
        "step",
        "split",
        "scenario",
        "name",
        "value",
        "unit",
    )

    def __init__(
        self,
        log_dir: str | Path,
        *,
        jsonl_filename: str = "events.jsonl",
        metrics_filename: str = "metrics.csv",
        manifest_filename: str = "run_manifest.json",
    ) -> None:
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / jsonl_filename
        self.metrics_path = self.log_dir / metrics_filename
        self.manifest_path = self.log_dir / manifest_filename
        self._lock = threading.Lock()

    def write_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        if self.manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"manifest already exists and will not be overwritten: "
                f"{self.manifest_path}"
            )
        payload = dict(manifest)
        payload.setdefault("timestamp", utc_timestamp())
        temporary_path = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".tmp"
        )
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self.manifest_path)

    def log_event(
        self,
        event: str,
        *,
        level: str = "info",
        **fields: Any,
    ) -> None:
        if not event:
            raise ValueError("event name must be non-empty")
        record = {
            "timestamp": utc_timestamp(),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        with self._lock:
            with self.events_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(line + "\n")

    def log_metric(
        self,
        *,
        name: str,
        value: float,
        step: int,
        split: str,
        scenario: str = "",
        unit: str = "",
    ) -> None:
        if not name:
            raise ValueError("metric name must be non-empty")
        record = {
            "timestamp": utc_timestamp(),
            "step": int(step),
            "split": split,
            "scenario": scenario,
            "name": name,
            "value": float(value),
            "unit": unit,
        }
        with self._lock:
            needs_header = (
                not self.metrics_path.exists()
                or self.metrics_path.stat().st_size == 0
            )
            with self.metrics_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=self._METRIC_COLUMNS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(record)

