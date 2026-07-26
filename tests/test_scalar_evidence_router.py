from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.identity_gated_router import TrainIdentityGallery
from set2proto.scalar_evidence_router import (
    SCALAR_EVIDENCE_METHOD,
    BoundedScalarEvidenceRouter,
    scalar_evidence_loss,
    select_scalar_evidence_candidate,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


class ScalarEvidenceRouterTests(unittest.TestCase):
    @staticmethod
    def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(313)
        condition = F.normalize(
            torch.randn((3, 4, 49, 128), generator=generator),
            dim=-1,
        )
        quality = torch.rand((3, 4), generator=generator) + 0.5
        teacher = F.normalize(
            0.6 * condition[:, 0] + 0.4 * condition[:, 2],
            dim=-1,
        )
        return condition, quality, teacher

    def test_gate_uses_only_scalars_and_respects_bound(self) -> None:
        condition, quality, _ = self._toy()
        model = BoundedScalarEvidenceRouter(
            initial_gate=0.05,
            maximum_gate=0.35,
        )
        first = model.gate_network[0]
        self.assertIsInstance(first, nn.Linear)
        self.assertEqual(first.in_features, model.evidence_dim)
        output, weights, logits, gate, _, _ = model(condition, quality)
        self.assertEqual(tuple(output.shape), (3, 49, 128))
        self.assertEqual(tuple(weights.shape), (3, 49, 4))
        self.assertEqual(tuple(logits.shape), (3, 49, 4))
        self.assertEqual(tuple(gate.shape), (3, 49))
        torch.testing.assert_close(
            gate,
            torch.full_like(gate, 0.05),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertLessEqual(float(gate.max().item()), 0.35)

    def test_joint_frame_permutation_is_equivalent(self) -> None:
        condition, quality, _ = self._toy()
        model = BoundedScalarEvidenceRouter()
        model.eval()
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        original = model(condition, quality)
        permuted = model(
            condition[:, permutation],
            quality[:, permutation],
        )
        torch.testing.assert_close(
            original[0],
            permuted[0],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            original[1],
            permuted[1][..., inverse],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            original[3],
            permuted[3],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_relative_margin_loss_is_finite_and_backpropagates(self) -> None:
        condition, quality, teacher = self._toy()
        model = BoundedScalarEvidenceRouter()
        outputs = model(condition, quality)
        gallery = TrainIdentityGallery(
            vectors=F.normalize(teacher.mean(dim=1), dim=-1),
            identities=torch.tensor([10, 20, 30]),
            clean_dataset_indices=torch.arange(3),
        )
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        recipe = config["p1_3"]["recipes"]["scalar_low_lr"]
        loss, metrics = scalar_evidence_loss(
            outputs=outputs,
            condition=condition,
            quality=quality,
            teacher_map=teacher,
            identities=torch.tensor([10, 20, 30]),
            gallery=gallery,
            recipe=recipe,
            config=config,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreaterEqual(
            metrics["relative_gallery_margin_loss"],
            0.0,
        )
        self.assertGreaterEqual(metrics["anchor_safety_loss"], 0.0)
        self.assertLessEqual(metrics["mean_gate"], 0.35)

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
            "eer": 0.04,
            "rank1_accuracy": 0.91,
            "mean_genuine_impostor_margin": 0.2,
        }
        result["hard_average"] = {"roc_auc": hard_auc}
        return result

    def test_selection_uses_validation_only_and_checks_bound(self) -> None:
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        results: dict[str, object] = {}
        for recipe, auc in (
            ("scalar_frozen", 0.903),
            ("scalar_low_lr", 0.907),
        ):
            results[recipe] = {
                "split": "val",
                "methods": {
                    "quality_pooling": {
                        "verification": self._verification(
                            0.900,
                            complementary=0.900,
                        ),
                        "teacher_similarity": {
                            "all": {"map_cosine": 0.60}
                        },
                    },
                    SCALAR_EVIDENCE_METHOD: {
                        "verification": self._verification(
                            auc,
                            complementary=auc,
                        ),
                        "teacher_similarity": {
                            "all": {"map_cosine": 0.62}
                        },
                    },
                },
                "gate_diagnostics": {"all": {"mean_gate": 0.2}},
                "correctness": {
                    "output_shape_valid": True,
                    "output_unit_norm_valid": True,
                    "weight_sum_valid": True,
                    "permutation_equivalent": True,
                },
            }
        selection = select_scalar_evidence_candidate(
            validation_results=results,
            config=config,
        )
        self.assertTrue(selection["test_authorized"])
        self.assertFalse(selection["test_metrics_consulted"])
        self.assertEqual(
            selection["selected"]["recipe"],
            "scalar_low_lr",
        )


if __name__ == "__main__":
    unittest.main()
