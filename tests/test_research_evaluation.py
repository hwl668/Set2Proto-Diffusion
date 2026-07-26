from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.research_evaluation import (
    _verification_metrics,
    continuous_aggregation_baselines,
)


class AggregationBaselineTests(unittest.TestCase):
    def test_required_baselines_have_expected_shape_and_quality_choice(self) -> None:
        condition = torch.zeros(2, 4, 3, 2)
        for frame in range(4):
            condition[:, frame, :, frame % 2] = float(frame + 1)
        quality = torch.tensor(
            [[1.0, 2.0, 9.0, 3.0], [8.0, 2.0, 1.0, 3.0]]
        )
        outputs = continuous_aggregation_baselines(condition, quality)

        self.assertEqual(
            set(outputs),
            {
                "best_single_frame",
                "mean_pooling",
                "max_pooling",
                "quality_pooling",
            },
        )
        for value in outputs.values():
            self.assertEqual(tuple(value.shape), (2, 3, 2))
            torch.testing.assert_close(
                value.norm(dim=-1),
                torch.ones(2, 3),
                atol=1e-6,
                rtol=1e-6,
            )
        expected_first = torch.tensor([1.0, 0.0]).expand(3, 2)
        expected_second = torch.tensor([1.0, 0.0]).expand(3, 2)
        torch.testing.assert_close(
            outputs["best_single_frame"][0],
            expected_first,
        )
        torch.testing.assert_close(
            outputs["best_single_frame"][1],
            expected_second,
        )

    def test_verification_metrics_for_perfect_separation(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0, 0])
        scores = np.asarray([0.95, 0.9, 0.2, 0.1, -0.1, -0.2])
        metrics = _verification_metrics(
            labels,
            scores,
            far_targets=[0.01, 0.001],
        )
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["eer"], 0.0)
        self.assertAlmostEqual(
            metrics["tar_at_far"]["0.01"]["tar"],
            1.0,
        )
        self.assertFalse(metrics["tar_at_far"]["0.001"]["reliable"])


if __name__ == "__main__":
    unittest.main()

