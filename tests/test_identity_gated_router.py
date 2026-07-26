from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.continuous_residual_oracle import quality_anchor
from set2proto.identity_gated_router import (
    GATED_ROUTER_METHOD,
    AnchorResidualIdentityRouter,
    TrainIdentityGallery,
    build_train_identity_gallery,
    identity_gated_router_loss,
    select_identity_gated_candidate,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


class IdentityGatedRouterTests(unittest.TestCase):
    @staticmethod
    def _toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(212)
        condition = F.normalize(
            torch.randn((3, 4, 49, 128), generator=generator),
            dim=-1,
        )
        quality = torch.rand((3, 4), generator=generator) + 0.5
        teacher = F.normalize(
            0.65 * condition[:, 1] + 0.35 * condition[:, 2],
            dim=-1,
        )
        return condition, quality, teacher

    def test_gate_initializes_near_quality_anchor_and_shapes_are_valid(
        self,
    ) -> None:
        condition, quality, _ = self._toy()
        model = AnchorResidualIdentityRouter(gate_initial_bias=-3.0)
        output, weights, logits, gate, routed, anchor = model(
            condition,
            quality,
        )
        self.assertEqual(tuple(output.shape), (3, 49, 128))
        self.assertEqual(tuple(weights.shape), (3, 49, 4))
        self.assertEqual(tuple(logits.shape), (3, 49, 4))
        self.assertEqual(tuple(gate.shape), (3, 49))
        self.assertEqual(tuple(routed.shape), (3, 49, 128))
        self.assertEqual(tuple(anchor.shape), (3, 49, 128))
        self.assertLess(float(gate.mean().item()), 0.06)
        torch.testing.assert_close(
            output.norm(dim=-1),
            torch.ones((3, 49)),
            atol=1e-5,
            rtol=0.0,
        )
        expected_anchor, _ = quality_anchor(condition, quality)
        torch.testing.assert_close(anchor, expected_anchor)

    def test_joint_frame_permutation_is_equivalent(self) -> None:
        condition, quality, _ = self._toy()
        model = AnchorResidualIdentityRouter()
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

    def test_train_gallery_uses_one_clean_teacher_per_identity(self) -> None:
        generator = torch.Generator().manual_seed(9)
        identities = torch.tensor([20, 10, 20, 10])
        maps = F.normalize(
            torch.randn((4, 49, 128), generator=generator),
            dim=-1,
        )
        dataset = SimpleNamespace(
            synthetic=SimpleNamespace(split="train"),
            _precomputed={
                "scenarios": [
                    "clean",
                    "clean",
                    "low_quality",
                    "wrong_identity",
                ],
                "target_identity": identities,
            },
            targets={"continuous_prototypes": maps},
        )
        gallery = build_train_identity_gallery(dataset)
        self.assertEqual(gallery.size, 2)
        self.assertEqual(gallery.identities.tolist(), [10, 20])
        self.assertEqual(gallery.clean_dataset_indices.tolist(), [1, 0])
        torch.testing.assert_close(
            gallery.vectors.norm(dim=-1),
            torch.ones(2),
        )

    def test_identity_gallery_losses_backpropagate(self) -> None:
        condition, quality, teacher = self._toy()
        model = AnchorResidualIdentityRouter()
        outputs = model(condition, quality)
        gallery = TrainIdentityGallery(
            vectors=F.normalize(
                teacher.mean(dim=1),
                dim=-1,
            ),
            identities=torch.tensor([10, 20, 30]),
            clean_dataset_indices=torch.arange(3),
        )
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        recipe = config["p1_2"]["recipes"]["identity_balanced"]
        loss, metrics = identity_gated_router_loss(
            model_outputs=outputs,
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
        self.assertGreater(metrics["gallery_cross_entropy_loss"], 0.0)
        self.assertGreaterEqual(metrics["gallery_margin_loss"], 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            )
        )

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

    def test_candidate_selection_is_validation_only(self) -> None:
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        results: dict[str, object] = {}
        for recipe, auc in (
            ("anchor_safe", 0.903),
            ("identity_balanced", 0.907),
        ):
            methods = {
                "quality_pooling": {
                    "verification": self._verification(
                        0.900,
                        complementary=0.900,
                    ),
                    "teacher_similarity": {
                        "all": {"map_cosine": 0.60}
                    },
                },
                GATED_ROUTER_METHOD: {
                    "verification": self._verification(
                        auc,
                        complementary=auc,
                    ),
                    "teacher_similarity": {
                        "all": {"map_cosine": 0.62}
                    },
                },
            }
            results[recipe] = {
                "split": "val",
                "methods": methods,
                "gate_diagnostics": {"all": {"mean_gate": 0.2}},
                "correctness": {
                    "output_shape_valid": True,
                    "output_unit_norm_valid": True,
                    "weight_sum_valid": True,
                    "permutation_equivalent": True,
                },
            }
        selection = select_identity_gated_candidate(
            validation_results=results,
            config=config,
        )
        self.assertTrue(selection["test_authorized"])
        self.assertFalse(selection["test_metrics_consulted"])
        self.assertEqual(
            selection["selected"]["recipe"],
            "identity_balanced",
        )


if __name__ == "__main__":
    unittest.main()
