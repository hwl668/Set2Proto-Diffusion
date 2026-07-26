from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.pointer_route_oracle import (
    build_route_candidates,
    build_route_dev_split,
    hard_route_targets,
    permutation_check,
    reconstruct_hard_routes,
    reconstruct_soft_routes,
    select_calibration_pair,
    soft_route_targets,
    teacher_route_scores,
)


class RouteDevelopmentSplitTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_scenario_complete(self) -> None:
        scenarios = ("clean", "low_quality")
        rows = [
            {
                "target_identity": identity,
                "scenario": scenario,
                "sample_id": f"{identity}-{scenario}",
            }
            for identity in range(6)
            for scenario in scenarios
        ]
        first = build_route_dev_split(
            rows,
            seed=91,
            train_identities=2,
            calibration_identities=2,
            validation_identities=2,
            required_scenarios=scenarios,
        )
        second = build_route_dev_split(
            rows,
            seed=91,
            train_identities=2,
            calibration_identities=2,
            validation_identities=2,
            required_scenarios=scenarios,
        )
        self.assertEqual(first.identity_ids, second.identity_ids)
        self.assertEqual(first.sample_indices, second.sample_indices)
        self.assertTrue(first.checks["passed"])
        self.assertEqual(
            first.checks["sample_counts"],
            {
                "route_train": 4,
                "route_calibration": 4,
                "route_validation": 4,
            },
        )

    def test_split_rejects_missing_scenario(self) -> None:
        rows = [
            {"target_identity": 0, "scenario": "clean"},
            {"target_identity": 1, "scenario": "clean"},
        ]
        with self.assertRaisesRegex(ValueError, "cover every required"):
            build_route_dev_split(
                rows,
                seed=1,
                train_identities=0,
                calibration_identities=1,
                validation_identities=1,
                required_scenarios=("clean", "wrong_identity"),
            )


class PointerRouteOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)
        self.condition = torch.randn(3, 4, 7, 5)
        self.quality = torch.rand(3, 4) + 0.2

    def test_candidate_maps_and_reconstructions_are_unit_normalized(self) -> None:
        candidates = build_route_candidates(self.condition, self.quality)
        self.assertEqual(tuple(candidates.shape), (3, 7, 5, 5))
        self.assertLess(
            float((candidates.norm(dim=-1) - 1.0).abs().max()),
            1e-5,
        )
        teacher = candidates[:, :, 3].clone()
        scores = teacher_route_scores(candidates, teacher)
        routes = hard_route_targets(scores, delta=0.01)
        probabilities = soft_route_targets(
            scores,
            delta=0.01,
            temperature=0.02,
        )
        self.assertTrue(bool((routes == 3).all()))
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities[..., 0]),
                atol=1e-6,
            )
        )
        hard = reconstruct_hard_routes(candidates, routes)
        soft = reconstruct_soft_routes(candidates, probabilities)
        self.assertLess(float((hard.norm(dim=-1) - 1.0).abs().max()), 1e-5)
        self.assertLess(float((soft.norm(dim=-1) - 1.0).abs().max()), 1e-5)

    def test_anchor_margin_preserves_anchor_for_negligible_gain(self) -> None:
        scores = torch.tensor([[[0.80, 0.805, 0.70, 0.60, 0.50]]])
        self.assertEqual(
            int(hard_route_targets(scores, delta=0.01).item()),
            0,
        )
        self.assertEqual(
            int(hard_route_targets(scores, delta=0.001).item()),
            1,
        )

    def test_soft_reconstruction_accepts_fp16_rounding_not_invalid_mass(
        self,
    ) -> None:
        candidates = build_route_candidates(self.condition, self.quality)
        probabilities = torch.softmax(
            torch.randn(3, 7, 5),
            dim=-1,
        ).half()
        reconstructed = reconstruct_soft_routes(candidates, probabilities)
        self.assertEqual(tuple(reconstructed.shape), (3, 7, 5))
        invalid = probabilities.float()
        invalid[..., 0] += 0.1
        with self.assertRaisesRegex(ValueError, "sum to one"):
            reconstruct_soft_routes(candidates, invalid)

    def test_joint_frame_permutation_is_equivariant(self) -> None:
        candidates = build_route_candidates(self.condition, self.quality)
        teacher = candidates[:, :, 2].clone()
        result = permutation_check(
            condition_features=self.condition,
            condition_quality=self.quality,
            teacher_maps=teacher,
            delta=0.01,
            temperature=0.02,
            permutation=torch.tensor([2, 0, 3, 1]),
        )
        self.assertEqual(result["route_disagreement_fraction"], 0.0)
        self.assertLess(result["probability_max_abs_difference"], 1e-6)
        self.assertLess(result["hard_map_max_abs_difference"], 1e-6)
        self.assertLess(result["soft_map_max_abs_difference"], 1e-6)

    def test_calibration_selection_uses_conservative_ties(self) -> None:
        rows = [
            {
                "delta": delta,
                "temperature": temperature,
                "hard_route_hard_auc": hard,
                "soft_route_hard_auc": soft,
            }
            for delta, hard in ((0.005, 0.9100), (0.01, 0.9098))
            for temperature, soft in ((0.01, 0.9200), (0.05, 0.9197))
        ]
        selected = select_calibration_pair(rows, auc_tie_tolerance=0.0005)
        self.assertEqual(selected["delta"], 0.01)
        self.assertEqual(selected["temperature"], 0.05)


if __name__ == "__main__":
    unittest.main()
