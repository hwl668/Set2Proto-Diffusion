from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.earlier_hook_oracle import (
    build_p0_3_decision,
    metadata_oracle_weights,
    pool_raw_early_activation,
    quality_spatial_weights,
    resize_visibility_to_hook,
    select_validation_hook,
)


class EarlierHookOracleTests(unittest.TestCase):
    _SCENARIOS = (
        "clean",
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    )

    @classmethod
    def _verification(
        cls,
        hard_auc: float,
        *,
        complementary: float,
    ) -> dict[str, object]:
        values = {
            scenario: {
                "roc_auc": (
                    complementary
                    if scenario == "complementary_occlusion"
                    else (0.95 if scenario == "clean" else hard_auc)
                )
            }
            for scenario in cls._SCENARIOS
        }
        values["all"] = {
            "roc_auc": hard_auc,
            "rank1_accuracy": 0.90,
            "eer": 0.05,
            "mean_genuine_impostor_margin": 0.2,
        }
        values["hard_average"] = {"roc_auc": hard_auc}
        return values

    @classmethod
    def _validation_result(
        cls,
        hook: str,
        oracle_auc: float,
    ) -> dict[str, object]:
        return {
            "split": "val",
            "hook_name": hook,
            "hook_shape": [128, 28, 28],
            "methods": {
                "late_quality_pooling": {
                    "verification": cls._verification(
                        0.900,
                        complementary=0.900,
                    )
                },
                "pre_suffix_quality_pooling": {
                    "verification": cls._verification(
                        0.901,
                        complementary=0.901,
                    )
                },
                "pre_suffix_metadata_oracle_pooling": {
                    "verification": cls._verification(
                        oracle_auc,
                        complementary=oracle_auc,
                    )
                },
            },
            "correctness": {
                "suffix_replay_map_cosine_min": 0.9999,
                "cache_roundtrip_token_cosine_min": 0.9999,
                "all_output_shapes_valid": True,
                "output_unit_norm_valid": True,
                "permutation_equivalent": True,
            },
        }

    @staticmethod
    def _selection_config() -> dict[str, object]:
        return {
            "p0_3": {
                "candidate_hooks": ["body.3", "body.7"],
                "selection_min_hard_auc_gain": 0.002,
                "selection_min_complementary_auc_gain": 0.0,
                "selection_min_hard_scenario_wins": 2,
                "selection_clean_auc_max_drop": 0.001,
                "selection_rank1_max_drop": 0.01,
                "suffix_replay_min_map_cosine": 0.999,
                "tie_break_prefer_later": True,
            }
        }

    @staticmethod
    def _cache_details() -> dict[str, object]:
        return {
            "pipeline_regression": {
                "suffix_replay": {
                    "body.3": {"late_spatial_cosine_min": 1.0},
                    "body.7": {"late_spatial_cosine_min": 1.0},
                }
            }
        }

    @staticmethod
    def _decision_checks() -> dict[str, bool]:
        names = (
            "candidate_hooks_and_shapes_preregistered",
            "selection_used_validation_only",
            "selection_lock_reloaded_before_test",
            "selection_lock_unchanged",
            "test_policy_followed",
            "validation_pipeline_reproduced",
            "validation_outputs_correct",
            "test_outputs_correct_if_evaluated",
            "same_gallery_protocol",
            "dataset_valid_after_lock",
            "quantization_valid_after_lock",
            "quantization_source_link_matches",
            "set_manifest_unchanged",
            "quantization_manifest_unchanged",
            "projection_unchanged",
            "codebook_unchanged",
            "checkpoint_unchanged",
            "backbone_frozen_eval",
            "no_training_or_quantized_early_method",
            "memory_within_configured_cap",
            "protected_outputs_unchanged_before_artifact_write",
            "checkpoints_unchanged_before_artifact_write",
        )
        return {name: True for name in names}

    def test_visibility_resize_preserves_seven_cell_boundaries(self) -> None:
        visibility = torch.zeros((1, 1, 49))
        visibility[0, 0, 0] = 1
        resized = resize_visibility_to_hook(
            visibility,
            height=28,
            width=28,
        )
        self.assertEqual(tuple(resized.shape), (1, 1, 28, 28))
        self.assertEqual(float(resized.sum().item()), 16.0)
        self.assertTrue(bool((resized[0, 0, :4, :4] == 1).all().item()))

    def test_oracle_excludes_wrong_identity_and_falls_back_on_missing(self) -> None:
        quality = torch.tensor([[1.0, 3.0, 100.0, 1.0]])
        same_identity = torch.tensor([[True, True, False, True]])
        visible = torch.ones((1, 4, 2, 2))
        weights = metadata_oracle_weights(
            quality,
            visible,
            same_identity,
        )
        torch.testing.assert_close(
            weights[..., 2],
            torch.zeros((1, 2, 2)),
        )
        missing = metadata_oracle_weights(
            quality,
            torch.zeros_like(visible),
            same_identity,
        )
        expected = torch.tensor([0.2, 0.6, 0.0, 0.2])
        torch.testing.assert_close(missing[0, 0, 0], expected)
        torch.testing.assert_close(
            missing.sum(dim=-1),
            torch.ones((1, 2, 2)),
        )

    def test_raw_pool_is_jointly_permutation_invariant(self) -> None:
        generator = torch.Generator().manual_seed(13)
        early = torch.randn((2, 4, 8, 3, 3), generator=generator)
        quality = torch.rand((2, 4), generator=generator) + 0.1
        weights = quality_spatial_weights(quality, height=3, width=3)
        pooled = pool_raw_early_activation(early, weights)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted_weights = quality_spatial_weights(
            quality[:, permutation],
            height=3,
            width=3,
        )
        permuted = pool_raw_early_activation(
            early[:, permutation],
            permuted_weights,
        )
        self.assertEqual(tuple(pooled.shape), (2, 8, 3, 3))
        torch.testing.assert_close(pooled, permuted)
        self.assertTrue(torch.equal(pooled, permuted))
        inverse = torch.argsort(permutation)
        self.assertTrue(
            torch.equal(weights, permuted_weights[..., inverse])
        )

    def test_validation_gate_authorizes_only_a_passing_hook(self) -> None:
        selection = select_validation_hook(
            candidate_results={
                "body.3": self._validation_result("body.3", 0.904),
                "body.7": self._validation_result("body.7", 0.901),
            },
            cache_details=self._cache_details(),
            config=self._selection_config(),
        )
        self.assertTrue(selection["test_authorized"])
        self.assertEqual(selection["selected"]["hook_name"], "body.3")
        self.assertEqual(selection["feasible_candidate_count"], 1)

    def test_validation_gate_blocks_test_when_no_candidate_passes(self) -> None:
        selection = select_validation_hook(
            candidate_results={
                "body.3": self._validation_result("body.3", 0.901),
                "body.7": self._validation_result("body.7", 0.900),
            },
            cache_details=self._cache_details(),
            config=self._selection_config(),
        )
        self.assertIsNone(selection["selected"])
        self.assertFalse(selection["test_authorized"])
        self.assertEqual(selection["feasible_candidate_count"], 0)

    def test_decision_requires_memory_and_correctness_gates(self) -> None:
        test_result = self._validation_result("body.3", 0.904)
        test_result["split"] = "test"
        selection = {"test_authorized": True}
        checks = self._decision_checks()
        decision = build_p0_3_decision(
            selection=selection,
            test_result=test_result,
            config=self._selection_config(),
            checks=checks,
        )
        self.assertTrue(decision["go"])
        checks["memory_within_configured_cap"] = False
        blocked = build_p0_3_decision(
            selection=selection,
            test_result=test_result,
            config=self._selection_config(),
            checks=checks,
        )
        self.assertFalse(blocked["go"])
        self.assertFalse(blocked["resource_gate_passed"])
        self.assertIn(
            "memory_within_configured_cap",
            blocked["failed_resource_checks"],
        )


if __name__ == "__main__":
    unittest.main()
