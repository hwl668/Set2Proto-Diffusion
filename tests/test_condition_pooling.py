from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.condition_pooling import validate_p0_robust_parameters
from set2proto.quantization import (
    quantize_vectors,
    robust_condition_pool,
    robust_teacher_pool,
)
from set2proto.research_evaluation import _identification_metrics


class ConditionRobustPoolingTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(91)
        identity_signal = F.normalize(
            torch.randn((2, 49, 8), generator=generator),
            dim=-1,
        )
        noise = 0.08 * torch.randn((2, 4, 49, 8), generator=generator)
        self.features = F.normalize(
            identity_signal[:, None] + noise,
            dim=-1,
        )
        self.quality = torch.tensor(
            [[0.7, 1.1, 0.9, 1.4], [1.2, 0.8, 1.5, 1.0]],
            dtype=torch.float32,
        )
        self.parameters = {
            "keep_fraction": 0.75,
            "consensus_weight": 2.0,
            "quality_weight": 1.0,
        }

    def test_wrapper_reuses_teacher_math_and_satisfies_invariants(self) -> None:
        condition_output, condition_diagnostics = robust_condition_pool(
            self.features,
            self.quality,
            **self.parameters,
        )
        teacher_output, teacher_diagnostics = robust_teacher_pool(
            self.features,
            self.quality,
            **self.parameters,
        )
        torch.testing.assert_close(condition_output, teacher_output)
        torch.testing.assert_close(
            condition_diagnostics["weights"],
            teacher_diagnostics["weights"],
        )
        self.assertEqual(tuple(condition_output.shape), (2, 49, 8))
        torch.testing.assert_close(
            condition_output.norm(dim=-1),
            torch.ones((2, 49)),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            condition_diagnostics["weights"].sum(dim=-1),
            torch.ones((2, 49)),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(
            bool(
                (
                    condition_diagnostics["keep_mask"].sum(dim=-1) == 3
                )
                .all()
                .item()
            )
        )

    def test_joint_frame_permutation_is_numerically_equivalent(self) -> None:
        original, original_diagnostics = robust_condition_pool(
            self.features,
            self.quality,
            **self.parameters,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted, permuted_diagnostics = robust_condition_pool(
            self.features[:, permutation],
            self.quality[:, permutation],
            **self.parameters,
        )
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            original_diagnostics["weights"],
            permuted_diagnostics["weights"][..., inverse],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_nearest_cosine_reconstruction_has_map_shape_and_unit_norm(self) -> None:
        pooled, _ = robust_condition_pool(
            self.features,
            self.quality,
            **self.parameters,
        )
        generator = torch.Generator().manual_seed(17)
        codebook = F.normalize(
            torch.randn((32, 8), generator=generator),
            dim=-1,
        )
        tokens, _ = quantize_vectors(pooled, codebook)
        reconstruction = codebook[tokens]
        self.assertEqual(tuple(tokens.shape), (2, 49))
        self.assertEqual(tuple(reconstruction.shape), (2, 49, 8))
        torch.testing.assert_close(
            reconstruction.norm(dim=-1),
            torch.ones((2, 49)),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_main_run_parameters_are_fixed(self) -> None:
        valid = {
            "teacher": {
                "robust_keep_fraction": 0.75,
                "consensus_weight": 2.0,
                "quality_weight": 1.0,
            }
        }
        self.assertEqual(
            validate_p0_robust_parameters(valid),
            {
                "robust_keep_fraction": 0.75,
                "consensus_weight": 2.0,
                "quality_weight": 1.0,
            },
        )
        invalid = {
            "teacher": {
                "robust_keep_fraction": 1.0,
                "consensus_weight": 2.0,
                "quality_weight": 1.0,
            }
        }
        with self.assertRaises(ValueError):
            validate_p0_robust_parameters(invalid)


class IdentificationMetricTests(unittest.TestCase):
    def test_rank1_and_margin_use_strongest_impostor(self) -> None:
        scores = torch.tensor(
            [[0.9, 0.4, 0.2], [0.5, 0.6, 0.7]],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [[True, False, False], [False, True, False]]
        )
        metrics = _identification_metrics(scores, labels)
        self.assertAlmostEqual(metrics["rank1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_genuine_score"], 0.75)
        self.assertAlmostEqual(
            metrics["mean_strongest_impostor_score"],
            0.55,
        )
        self.assertAlmostEqual(
            metrics["mean_genuine_impostor_margin"],
            0.20,
        )

    def test_gallery_requires_one_target_per_probe(self) -> None:
        scores = torch.zeros((1, 2))
        with self.assertRaises(ValueError):
            _identification_metrics(
                scores,
                torch.tensor([[True, True]]),
            )


if __name__ == "__main__":
    unittest.main()
