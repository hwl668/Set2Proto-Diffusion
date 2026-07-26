from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.continuous_residual_oracle import (
    apply_oracle_method,
    project_simplex,
    quality_anchor,
    select_continuous_residual_candidate,
    teacher_best_frame_oracle,
    teacher_bounded_span_residual_oracle,
    teacher_convex_residual_oracle,
)


class ContinuousResidualOracleTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "p1_0": {
                "candidate_methods": [
                    "teacher_best_frame_oracle",
                    "teacher_convex_residual_oracle",
                    "teacher_bounded_span_residual_oracle",
                ],
                "convex_iterations": 32,
                "convex_step_size": 0.5,
                "span_ridge": 0.01,
                "span_residual_norm_cap": 0.5,
                "selection_min_hard_auc_gain": 0.002,
                "selection_min_complementary_auc_gain": 0.0,
                "selection_min_hard_scenario_wins": 2,
                "selection_clean_auc_max_drop": 0.001,
                "selection_rank1_max_drop": 0.01,
                "selection_min_teacher_map_cosine_gain": 0.01,
            }
        }

    @staticmethod
    def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(19)
        condition = F.normalize(
            torch.randn((2, 4, 3, 8), generator=generator),
            dim=-1,
        )
        quality = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]]
        )
        teacher = F.normalize(
            0.7 * condition[:, 1] + 0.3 * condition[:, 3],
            dim=-1,
        )
        return condition, quality, teacher

    def test_simplex_projection_is_nonnegative_and_sums_to_one(self) -> None:
        value = torch.tensor(
            [[[1.2, -0.3, 0.7, 2.0], [-2.0, -1.0, 4.0, 0.1]]]
        )
        projected = project_simplex(value)
        self.assertTrue(bool((projected >= 0).all().item()))
        torch.testing.assert_close(
            projected.sum(dim=-1),
            torch.ones((1, 2)),
        )

    def test_best_frame_selects_teacher_matching_observation(self) -> None:
        condition, quality, _ = self._toy()
        teacher = condition[:, 2].clone()
        output, diagnostics = teacher_best_frame_oracle(
            condition,
            quality,
            teacher,
        )
        torch.testing.assert_close(output, teacher)
        self.assertTrue(
            bool((diagnostics["weights"][..., 2] == 1).all().item())
        )

    def test_best_frame_oracle_never_loses_to_quality_anchor(self) -> None:
        condition, quality, teacher = self._toy()
        anchor, _ = quality_anchor(condition, quality)
        output, _ = teacher_best_frame_oracle(
            condition,
            quality,
            teacher,
        )
        self.assertTrue(
            bool(
                (
                    (output * teacher).sum(dim=-1)
                    >= (anchor * teacher).sum(dim=-1) - 1e-6
                )
                .all()
                .item()
            )
        )

    def test_convex_oracle_never_reduces_teacher_cosine(self) -> None:
        condition, quality, teacher = self._toy()
        anchor, _ = quality_anchor(condition, quality)
        output, diagnostics = teacher_convex_residual_oracle(
            condition,
            quality,
            teacher,
            iterations=32,
            step_size=0.5,
        )
        anchor_cosine = (anchor * teacher).sum(dim=-1)
        output_cosine = (output * teacher).sum(dim=-1)
        self.assertTrue(
            bool((output_cosine >= anchor_cosine - 1e-6).all().item())
        )
        torch.testing.assert_close(
            diagnostics["weights"].sum(dim=-1),
            torch.ones_like(anchor_cosine),
            atol=1e-6,
            rtol=0.0,
        )

    def test_span_residual_respects_norm_cap(self) -> None:
        condition, quality, teacher = self._toy()
        output, diagnostics = teacher_bounded_span_residual_oracle(
            condition,
            quality,
            teacher,
            ridge=0.01,
            residual_norm_cap=0.25,
        )
        self.assertEqual(tuple(output.shape), (2, 3, 8))
        self.assertLessEqual(
            float(diagnostics["residual_norm"].max().item()),
            0.250001,
        )
        torch.testing.assert_close(
            output.norm(dim=-1),
            torch.ones((2, 3)),
            atol=1e-6,
            rtol=0.0,
        )

    def test_all_oracles_are_jointly_frame_permutation_equivalent(self) -> None:
        condition, quality, teacher = self._toy()
        permutation = torch.tensor([2, 0, 3, 1])
        for method in self._config()["p1_0"]["candidate_methods"]:
            with self.subTest(method=method):
                original, _ = apply_oracle_method(
                    method,
                    condition,
                    quality,
                    teacher,
                    config=self._config(),
                )
                permuted, _ = apply_oracle_method(
                    method,
                    condition[:, permutation],
                    quality[:, permutation],
                    teacher,
                    config=self._config(),
                )
                torch.testing.assert_close(
                    original,
                    permuted,
                    atol=1e-4,
                    rtol=1e-4,
                )

    @staticmethod
    def _verification(
        hard_auc: float,
        *,
        complementary: float,
        clean: float = 0.95,
    ) -> dict[str, object]:
        scenarios = (
            "clean",
            "low_quality",
            "complementary_occlusion",
            "common_occlusion",
            "wrong_identity",
        )
        values = {
            scenario: {
                "roc_auc": (
                    clean
                    if scenario == "clean"
                    else (
                        complementary
                        if scenario == "complementary_occlusion"
                        else hard_auc
                    )
                )
            }
            for scenario in scenarios
        }
        values["all"] = {
            "roc_auc": hard_auc,
            "rank1_accuracy": 0.90,
            "eer": 0.05,
            "mean_genuine_impostor_margin": 0.2,
        }
        values["hard_average"] = {"roc_auc": hard_auc}
        return values

    def test_validation_selection_authorizes_only_passing_oracle(self) -> None:
        methods: dict[str, object] = {
            "quality_pooling": {
                "verification": self._verification(
                    0.900,
                    complementary=0.900,
                ),
                "teacher_similarity": {"all": {"map_cosine": 0.60}},
            }
        }
        oracle_aucs = {
            "teacher_best_frame_oracle": 0.901,
            "teacher_convex_residual_oracle": 0.905,
            "teacher_bounded_span_residual_oracle": 0.900,
        }
        for method, auc in oracle_aucs.items():
            methods[method] = {
                "verification": self._verification(
                    auc,
                    complementary=auc,
                ),
                "teacher_similarity": {"all": {"map_cosine": 0.63}},
            }
        correctness = {
            "output_shape_valid": True,
            "output_unit_norm_valid": True,
            "quality_anchor_reproduced": True,
            "methods": {
                method: {
                    "permutation_equivalent": True,
                    "teacher_cosine_monotonic": True,
                }
                for method in oracle_aucs
            },
        }
        selection = select_continuous_residual_candidate(
            validation_result={
                "split": "val",
                "methods": methods,
                "correctness": correctness,
            },
            config=self._config(),
        )
        self.assertTrue(selection["test_authorized"])
        self.assertEqual(
            selection["selected"]["method"],
            "teacher_convex_residual_oracle",
        )


if __name__ == "__main__":
    unittest.main()
