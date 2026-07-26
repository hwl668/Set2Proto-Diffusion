from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.real_sets import (
    _apply_degradation,
    _build_rows,
    _complementary_visibility,
)


class RealSetConstructionTests(unittest.TestCase):
    def test_rows_have_disjoint_higher_quality_teachers_and_distractor(self) -> None:
        records = []
        quality = []
        for identity in (10, 20, 30):
            for image_index in range(10):
                records.append(
                    {
                        "row_index": str(len(records)),
                        "split": "train",
                        "identity_id": str(identity),
                        "image_name": f"{len(records) + 1:06d}.jpg",
                    }
                )
                quality.append(float(image_index + 1))
        rows = _build_rows(
            split="train",
            records=records,
            quality_norms=np.asarray(quality, dtype=np.float32),
            condition_frames=4,
            teacher_frames=4,
            sets_per_identity=5,
            scenarios=[
                "clean",
                "low_quality",
                "complementary_occlusion",
                "common_occlusion",
                "wrong_identity",
            ],
            wrong_identity_probability=0.25,
            seed=123,
            max_identities=2,
        )

        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["target_identity"] for row in rows},
            {10, 20},
        )
        for row in rows:
            condition = set(row["condition_indices"])
            teacher = set(row["teacher_indices"])
            self.assertTrue(condition.isdisjoint(teacher))
            self.assertGreaterEqual(
                np.mean([quality[index] for index in teacher]),
                np.mean([quality[index] for index in condition]),
            )
            if row["scenario"] == "wrong_identity":
                self.assertEqual(len(row["wrong_positions"]), 1)
                position = row["wrong_positions"][0]
                self.assertNotEqual(
                    row["source_identity_ids"][position],
                    row["target_identity"],
                )

    def test_complementary_visibility_covers_every_token(self) -> None:
        masks = torch.stack(
            [
                _complementary_visibility(index, height=7, width=7)
                for index in range(4)
            ]
        )
        self.assertTrue(bool((masks.sum(dim=0) > 0).all().item()))
        self.assertTrue(bool((masks.sum(dim=0) == 1).all().item()))

    def test_image_degradations_preserve_shape_and_change_pixels(self) -> None:
        image = torch.linspace(-1.0, 1.0, 3 * 112 * 112).reshape(
            3,
            112,
            112,
        )
        for label in (
            "gaussian_blur",
            "motion_blur",
            "low_light",
            "jpeg",
        ):
            degraded = _apply_degradation(
                image,
                label,
                generator=np.random.default_rng(42),
            )
            self.assertEqual(tuple(degraded.shape), (3, 112, 112))
            self.assertTrue(bool(torch.isfinite(degraded).all().item()))
            self.assertFalse(torch.equal(image, degraded))


if __name__ == "__main__":
    unittest.main()
