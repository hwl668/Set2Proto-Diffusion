from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.feature_cache import validate_real_feature_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureCacheValidationTests(unittest.TestCase):
    def test_valid_cache_checks_shapes_hashes_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_summaries = {}
            for split in ("train", "val", "test"):
                split_directory = root / split
                split_directory.mkdir()
                local = np.zeros((2, 3, 4), dtype=np.float16)
                local[..., 0] = 1.0
                embeddings = np.zeros((2, 512), dtype=np.float16)
                embeddings[..., 0] = 1.0
                norms = np.asarray([10.0, 20.0], dtype=np.float32)
                np.save(split_directory / "local_features.npy", local)
                np.save(split_directory / "embeddings.npy", embeddings)
                np.save(split_directory / "feature_norms.npy", norms)
                with (split_directory / "index.csv").open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=(
                            "row_index",
                            "split",
                            "identity_id",
                            "image_name",
                        ),
                    )
                    writer.writeheader()
                    for row_index in range(2):
                        writer.writerow(
                            {
                                "row_index": row_index,
                                "split": split,
                                "identity_id": row_index + 1,
                                "image_name": f"{row_index + 1:06d}.jpg",
                            }
                        )
                files = {}
                for filename in (
                    "local_features.npy",
                    "embeddings.npy",
                    "feature_norms.npy",
                    "index.csv",
                ):
                    path = split_directory / filename
                    files[filename] = {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                split_summaries[split] = {
                    "images": 2,
                    "local_shape": [2, 3, 4],
                    "embedding_shape": [2, 512],
                    "files": files,
                }
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "splits": split_summaries,
                    }
                ),
                encoding="utf-8",
            )

            validation = validate_real_feature_cache(root, chunk_size=1)

            self.assertTrue(all(validation["checks"].values()))
            self.assertEqual(
                validation["maximum_local_unit_norm_error"],
                0.0,
            )
            self.assertEqual(
                validation["maximum_embedding_unit_norm_error"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
