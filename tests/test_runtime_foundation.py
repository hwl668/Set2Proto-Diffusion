from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.environment import collect_git_info, config_sha256
from set2proto.experiment_logging import ExperimentLogger
from set2proto.run_paths import create_run_paths, default_run_id


class RunPathTests(unittest.TestCase):
    def test_run_layout_and_checkpoint_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_run_paths(
                temporary,
                profile="smoke",
                seed=7,
                run_id="unit-run",
            )
            self.assertTrue(paths.checkpoints.is_dir())
            self.assertTrue(paths.logs.is_dir())
            self.assertTrue(paths.artifacts.is_dir())
            self.assertEqual(
                paths.checkpoint_path(125).name,
                "checkpoint_step_00000125.pt",
            )

    def test_existing_run_is_not_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            create_run_paths(
                temporary,
                profile="smoke",
                seed=7,
                run_id="unit-run",
            )
            with self.assertRaises(FileExistsError):
                create_run_paths(
                    temporary,
                    profile="smoke",
                    seed=7,
                    run_id="unit-run",
                )
            resumed = create_run_paths(
                temporary,
                profile="smoke",
                seed=7,
                run_id="unit-run",
                resume=True,
            )
            self.assertEqual(resumed.root.name, "unit-run")

    def test_invalid_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                create_run_paths(
                    temporary,
                    profile="smoke",
                    seed=7,
                    run_id="../unsafe",
                )

    def test_default_run_id_contains_profile_and_seed(self) -> None:
        fixed = datetime(2026, 7, 25, 16, 45, 3, 123456)
        run_id = default_run_id("pilot", 99, fixed)
        self.assertEqual(run_id, "20260725-164503-123456_pilot_s99")


class LoggingTests(unittest.TestCase):
    def test_manifest_events_and_metrics_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = ExperimentLogger(temporary)
            logger.write_manifest({"status": "passed", "path": Path(temporary)})
            logger.log_event("unit_event", answer=42)
            logger.log_metric(
                name="loss",
                value=0.25,
                step=3,
                split="train",
            )

            with logger.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["status"], "passed")
            self.assertIn("timestamp", manifest)

            with logger.events_path.open("r", encoding="utf-8") as handle:
                event = json.loads(handle.readline())
            self.assertEqual(event["event"], "unit_event")
            self.assertEqual(event["answer"], 42)

            with logger.metrics_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0]["name"], "loss")
            self.assertEqual(metrics[0]["step"], "3")

            with self.assertRaises(FileExistsError):
                logger.write_manifest({"status": "would-overwrite"})

    def test_config_hash_is_key_order_independent(self) -> None:
        self.assertEqual(
            config_sha256({"a": 1, "b": 2}),
            config_sha256({"b": 2, "a": 1}),
        )

    def test_non_repository_git_state_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            git = collect_git_info(temporary)
        self.assertTrue(git["available"])
        self.assertFalse(git["repository"])
        self.assertEqual(git["status"], "no-repository")
        self.assertIsNone(git["commit"])


if __name__ == "__main__":
    unittest.main()

