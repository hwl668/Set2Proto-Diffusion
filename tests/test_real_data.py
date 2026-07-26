from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.real_data import build_identity_split, load_split_identities


class RealDataSplitTests(unittest.TestCase):
    def _records(self) -> tuple[dict[str, int], dict[str, int]]:
        identities: dict[str, int] = {}
        partitions: dict[str, int] = {}
        identity_id = 1
        for partition in range(3):
            for _ in range(5):
                for image_index in range(10):
                    name = f"{identity_id:04d}_{image_index:02d}.jpg"
                    identities[name] = identity_id
                    partitions[name] = partition
                identity_id += 1
        return identities, partitions

    def test_split_is_reproducible_and_identity_disjoint(self) -> None:
        identities, partitions = self._records()
        requested = {"train": 3, "val": 2, "test": 2}
        first, images = build_identity_split(
            identities,
            partitions,
            requested_identities=requested,
            min_images_per_identity=8,
            seed=20260725,
        )
        second, _ = build_identity_split(
            identities,
            partitions,
            requested_identities=requested,
            min_images_per_identity=8,
            seed=20260725,
        )
        self.assertEqual(first, second)
        self.assertFalse(set(first["train"]) & set(first["val"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["val"]) & set(first["test"]))
        self.assertTrue(
            all(
                len(images[identity_id]) == 10
                for ids in first.values()
                for identity_id in ids
            )
        )

    def test_insufficient_images_are_excluded(self) -> None:
        identities, partitions = self._records()
        short_identity = min(identities.values())
        removed = 0
        for image_name in list(identities):
            if identities[image_name] == short_identity and removed < 3:
                identities.pop(image_name)
                partitions.pop(image_name)
                removed += 1
        selected, _ = build_identity_split(
            identities,
            partitions,
            requested_identities={"train": 4, "val": 2, "test": 2},
            min_images_per_identity=8,
            seed=7,
        )
        self.assertNotIn(short_identity, selected["train"])

    def test_identity_crossing_partitions_is_rejected(self) -> None:
        identities, partitions = self._records()
        image_name = next(iter(identities))
        partitions[image_name] = 1
        with self.assertRaisesRegex(ValueError, "cross official partitions"):
            build_identity_split(
                identities,
                partitions,
                requested_identities={"train": 2, "val": 2, "test": 2},
                min_images_per_identity=8,
                seed=1,
            )

    def test_reference_split_identities_are_excluded_per_partition(
        self,
    ) -> None:
        identities, partitions = self._records()
        reference = {
            "train": {1, 2},
            "val": {6},
            "test": {11, 12},
        }
        selected, _ = build_identity_split(
            identities,
            partitions,
            requested_identities={"train": 3, "val": 3, "test": 3},
            min_images_per_identity=8,
            seed=23,
            excluded_identities=reference,
        )
        for split in ("train", "val", "test"):
            self.assertFalse(set(selected[split]) & reference[split])

    def test_exclusion_is_reproducible(self) -> None:
        identities, partitions = self._records()
        kwargs = {
            "requested_identities": {
                "train": 2,
                "val": 2,
                "test": 2,
            },
            "min_images_per_identity": 8,
            "seed": 9,
            "excluded_identities": {
                "train": {1},
                "val": {6},
                "test": {11},
            },
        }
        first, _ = build_identity_split(
            identities,
            partitions,
            **kwargs,
        )
        second, _ = build_identity_split(
            identities,
            partitions,
            **kwargs,
        )
        self.assertEqual(first, second)

    def test_materialized_reference_identities_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "identities.csv").write_text(
                "split,identity_id,image_count\n"
                "train,11,10\n"
                "val,22,12\n"
                "test,33,14\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_split_identities(root),
                {
                    "train": {11},
                    "val": {22},
                    "test": {33},
                },
            )


if __name__ == "__main__":
    unittest.main()
