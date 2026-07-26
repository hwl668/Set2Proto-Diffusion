from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.continuous_residual_oracle import (
    teacher_best_frame_oracle,
)
from set2proto.continuous_router import (
    ROUTER_METHOD,
    ContinuousLocalRouter,
    continuous_router_loss,
    router_parameter_count,
    select_continuous_router,
)


class ContinuousRouterTests(unittest.TestCase):
    @staticmethod
    def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(113)
        condition = F.normalize(
            torch.randn((3, 4, 49, 128), generator=generator),
            dim=-1,
        )
        quality = torch.rand((3, 4), generator=generator) + 0.5
        teacher = F.normalize(
            0.8 * condition[:, 1] + 0.2 * condition[:, 3],
            dim=-1,
        )
        return condition, quality, teacher

    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "p1_1": {
                "distillation_weight": 1.0,
                "local_map_cosine_weight": 1.0,
                "global_identity_weight": 1.0,
                "selection_min_hard_auc_gain": 0.002,
                "selection_min_complementary_auc_gain": 0.0,
                "selection_min_hard_scenario_wins": 2,
                "selection_clean_auc_max_drop": 0.001,
                "selection_rank1_max_drop": 0.01,
                "selection_min_teacher_map_cosine_gain": 0.005,
            }
        }

    def test_forward_shape_norm_weight_sum_and_parameter_cap(self) -> None:
        condition, quality, _ = self._toy()
        model = ContinuousLocalRouter()
        output, weights, logits = model(condition, quality)
        self.assertEqual(tuple(output.shape), (3, 49, 128))
        self.assertEqual(tuple(weights.shape), (3, 49, 4))
        self.assertEqual(tuple(logits.shape), (3, 49, 4))
        torch.testing.assert_close(
            output.norm(dim=-1),
            torch.ones((3, 49)),
            atol=1e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones((3, 49)),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertLess(router_parameter_count(model), 250_000)

    def test_joint_frame_permutation_is_equivalent(self) -> None:
        condition, quality, _ = self._toy()
        model = ContinuousLocalRouter()
        model.eval()
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        original_map, original_weights, _ = model(condition, quality)
        permuted_map, permuted_weights, _ = model(
            condition[:, permutation],
            quality[:, permutation],
        )
        torch.testing.assert_close(
            original_map,
            permuted_map,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            original_weights,
            permuted_weights[..., inverse],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_distillation_loss_is_finite_and_backpropagates(self) -> None:
        condition, quality, teacher = self._toy()
        model = ContinuousLocalRouter()
        output, _, logits = model(condition, quality)
        with torch.inference_mode():
            _, diagnostics = teacher_best_frame_oracle(
                condition,
                quality,
                teacher,
            )
        loss, values = continuous_router_loss(
            logits=logits,
            output_map=output,
            oracle_weights=diagnostics["weights"],
            teacher_map=teacher,
            config=self._config(),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            )
        )
        self.assertGreater(values["loss"], 0.0)

    @staticmethod
    def _verification(
        hard_auc: float,
        *,
        complementary: float,
    ) -> dict[str, object]:
        scenarios = (
            "clean",
            "low_quality",
            "complementary_occlusion",
            "common_occlusion",
            "wrong_identity",
        )
        result = {
            scenario: {
                "roc_auc": (
                    0.95
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
        result["all"] = {
            "roc_auc": hard_auc,
            "eer": 0.05,
            "rank1_accuracy": 0.90,
            "mean_genuine_impostor_margin": 0.2,
        }
        result["hard_average"] = {"roc_auc": hard_auc}
        return result

    def test_selection_uses_validation_and_requires_all_gates(self) -> None:
        methods = {
            "quality_pooling": {
                "verification": self._verification(
                    0.900,
                    complementary=0.900,
                ),
                "teacher_similarity": {"all": {"map_cosine": 0.60}},
            },
            ROUTER_METHOD: {
                "verification": self._verification(
                    0.905,
                    complementary=0.906,
                ),
                "teacher_similarity": {"all": {"map_cosine": 0.62}},
            },
        }
        selection = select_continuous_router(
            validation_result={
                "split": "val",
                "methods": methods,
                "correctness": {
                    "output_shape_valid": True,
                    "output_unit_norm_valid": True,
                    "weight_sum_valid": True,
                    "permutation_equivalent": True,
                },
            },
            config=self._config(),
        )
        self.assertTrue(selection["test_authorized"])
        self.assertFalse(selection["test_metrics_consulted"])
        self.assertEqual(selection["selected"]["method"], ROUTER_METHOD)


if __name__ == "__main__":
    unittest.main()
