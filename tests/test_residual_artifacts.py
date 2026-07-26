from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.residual_artifacts import validate_residual_artifacts


class ResidualArtifactValidationTests(unittest.TestCase):
    def test_validator_preserves_raw_norm_and_rejects_test_target(self) -> None:
        # The full preparation path is exercised by the real P2-1 run. This
        # focused fixture verifies the two policy checks that differ from the
        # legacy spherical validator.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection = root / "projection.pt"
            torch.save({"method": "pca"}, projection)
            import hashlib
            import json

            projection_hash = hashlib.sha256(projection.read_bytes()).hexdigest()
            codebook = {
                "schema_version": 1,
                "method": "euclidean_residual_kmeans",
                "fit_split": "train",
                "fit_tokens": 10,
                "available_train_tokens": 10,
                "seed": 1,
                "codebook_size": 2,
                "dimension": 3,
                "vectors": torch.tensor(
                    [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]
                ),
            }
            torch.save(codebook, root / "codebook.pt")
            for split, count in (("train", 3), ("val", 2)):
                torch.save(
                    {
                        "split": split,
                        "representation": (
                            "quality_anchor_plus_raw_residual"
                        ),
                        "tokens": torch.zeros(count, 49, dtype=torch.int64),
                        "continuous_prototypes": torch.randn(count, 49, 3),
                        "sample_ids": [f"{split}-{i}" for i in range(count)],
                    },
                    root / f"{split}_targets.pt",
                )

            files = {}
            for name in (
                "projection.pt",
                "codebook.pt",
                "train_targets.pt",
                "val_targets.pt",
            ):
                path = root / name
                files[name] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            manifest = {
                "fit_split": "train",
                "protocol": "validation_only",
                "test_dataset_constructed": False,
                "test_artifacts_constructed": False,
                "source_projection_sha256": projection_hash,
                "files": files,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            passed = validate_residual_artifacts(root)
            self.assertTrue(passed["passed"])
            self.assertAlmostEqual(
                passed["codebook_norm"]["maximum"],
                0.2,
                places=6,
            )

            torch.save({}, root / "test_targets.pt")
            failed = validate_residual_artifacts(root)
            self.assertFalse(failed["passed"])
            self.assertFalse(
                failed["checks"]["test_artifacts_not_constructed"]
            )


if __name__ == "__main__":
    unittest.main()
