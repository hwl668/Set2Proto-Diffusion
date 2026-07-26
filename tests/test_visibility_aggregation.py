from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.visibility_aggregation import (
    _same_identity_frame_mask,
    identity_outlier_penalty,
    local_reliability_scores,
    oracle_visibility_pool,
    reliability_weighted_pool,
)


class OracleVisibilityPoolingTests(unittest.TestCase):
    def test_complementary_positions_select_the_only_visible_frame(self) -> None:
        features = torch.zeros((1, 4, 4, 4))
        for frame in range(4):
            features[0, frame, :, frame] = 1.0
        quality = torch.ones((1, 4))
        visibility = torch.eye(4).unsqueeze(0)
        same_identity = torch.ones((1, 4), dtype=torch.bool)
        pooled, diagnostics = oracle_visibility_pool(
            features,
            quality,
            visibility,
            same_identity,
        )
        torch.testing.assert_close(pooled[0], torch.eye(4))
        torch.testing.assert_close(
            diagnostics["weights"].sum(dim=-1),
            torch.ones((1, 4)),
        )

    def test_common_missing_falls_back_to_same_identity_quality(self) -> None:
        features = F.normalize(
            torch.tensor(
                [
                    [
                        [[1.0, 0.0]],
                        [[0.0, 1.0]],
                        [[-1.0, 0.0]],
                        [[0.0, -1.0]],
                    ]
                ]
            ),
            dim=-1,
        )
        quality = torch.tensor([[1.0, 3.0, 100.0, 1.0]])
        visibility = torch.zeros((1, 4, 1))
        same_identity = torch.tensor([[True, True, False, True]])
        _, diagnostics = oracle_visibility_pool(
            features,
            quality,
            visibility,
            same_identity,
        )
        expected = torch.tensor([0.2, 0.6, 0.0, 0.2])
        torch.testing.assert_close(
            diagnostics["weights"][0, 0],
            expected,
        )

    def test_visible_wrong_identity_receives_zero_oracle_weight(self) -> None:
        features = F.normalize(torch.randn((1, 4, 2, 3)), dim=-1)
        quality = torch.tensor([[1.0, 1.0, 100.0, 1.0]])
        visibility = torch.ones((1, 4, 2))
        same_identity = torch.tensor([[True, True, False, True]])
        _, diagnostics = oracle_visibility_pool(
            features,
            quality,
            visibility,
            same_identity,
        )
        torch.testing.assert_close(
            diagnostics["weights"][0, :, 2],
            torch.zeros(2),
        )

    def test_source_identity_metadata_is_checked_against_annotations(self) -> None:
        class Synthetic:
            rows = [
                {
                    "target_identity": 7,
                    "source_identity_ids": [7, 9, 7, 7],
                    "wrong_positions": [1],
                }
            ]

        class Dataset:
            synthetic = Synthetic()

        mask = _same_identity_frame_mask(Dataset(), [0], 4)
        torch.testing.assert_close(
            mask,
            torch.tensor([[True, False, True, True]]),
        )
        Dataset.synthetic.rows[0]["wrong_positions"] = []
        with self.assertRaisesRegex(ValueError, "disagrees"):
            _same_identity_frame_mask(Dataset(), [0], 4)


class ReliabilityAggregationTests(unittest.TestCase):
    def test_identity_penalty_detects_one_global_outlier(self) -> None:
        consensus = torch.tensor([1.0, 0.0, 0.0])
        outlier = torch.tensor([0.0, 1.0, 0.0])
        frames = torch.stack(
            [consensus, consensus, consensus, outlier]
        )[None, :, None, :].expand(1, 4, 5, 3)
        penalty, diagnostics = identity_outlier_penalty(
            frames,
            threshold=0.0,
            scale=0.1,
        )
        self.assertGreater(
            float(penalty[0, 3].item()),
            float(penalty[0, :3].max().item()),
        )
        self.assertLess(
            float(diagnostics["frame_consensus"][0, 3].item()),
            float(diagnostics["frame_consensus"][0, :3].min().item()),
        )
        torch.testing.assert_close(
            diagnostics["frame_consensus"][0, :3],
            torch.ones(3),
        )

    def test_local_reference_score_prefers_exact_clean_support(self) -> None:
        references = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.7, 0.3, 0.0],
                ]
            ]
        )
        condition = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 1.0]],
                ]
            ]
        )
        scores, raw = local_reliability_scores(
            condition,
            reference_vectors=references,
            centers={1: torch.zeros(1)},
            scales={1: torch.ones(1)},
            top_k_values=[1],
            z_clip=(-4.0, 2.0),
        )
        self.assertGreater(
            float(raw[1][0, 0, 0].item()),
            float(raw[1][0, 1, 0].item()),
        )
        self.assertGreater(
            float(scores[1][0, 0, 0].item()),
            float(scores[1][0, 1, 0].item()),
        )

    def test_pool_is_normalized_positive_and_permutation_equivalent(self) -> None:
        generator = torch.Generator().manual_seed(123)
        features = F.normalize(
            torch.randn((2, 4, 7, 8), generator=generator),
            dim=-1,
        )
        quality = torch.rand((2, 4), generator=generator) + 0.5
        local = torch.randn((2, 4, 7), generator=generator)
        penalty = torch.rand((2, 4), generator=generator)
        pooled, diagnostics = reliability_weighted_pool(
            features,
            quality,
            local_reliability=local,
            identity_penalty=penalty,
            local_weight=1.0,
            identity_weight=2.0,
            quality_weight=1.0,
            minimum_frame_weight=0.02,
        )
        self.assertEqual(tuple(pooled.shape), (2, 7, 8))
        torch.testing.assert_close(
            pooled.norm(dim=-1),
            torch.ones((2, 7)),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            diagnostics["weights"].sum(dim=-1),
            torch.ones((2, 7)),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertGreaterEqual(
            float(diagnostics["weights"].min().item()),
            0.02 - 1e-7,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted, permuted_diagnostics = reliability_weighted_pool(
            features[:, permutation],
            quality[:, permutation],
            local_reliability=local[:, permutation],
            identity_penalty=penalty[:, permutation],
            local_weight=1.0,
            identity_weight=2.0,
            quality_weight=1.0,
            minimum_frame_weight=0.02,
        )
        torch.testing.assert_close(pooled, permuted, atol=1e-6, rtol=1e-6)
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(
            diagnostics["weights"],
            permuted_diagnostics["weights"][..., inverse],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_all_unreliable_local_scores_fall_back_to_quality(self) -> None:
        generator = torch.Generator().manual_seed(9)
        features = F.normalize(
            torch.randn((2, 4, 3, 5), generator=generator),
            dim=-1,
        )
        quality = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).expand(2, -1)
        all_missing = -torch.tensor(
            [1.0, 2.0, 3.0, 4.0]
        )[None, :, None].expand(2, 4, 3)
        with_local, diagnostics = reliability_weighted_pool(
            features,
            quality,
            local_reliability=all_missing,
            identity_penalty=None,
            local_weight=1.0,
            identity_weight=0.0,
            quality_weight=1.0,
            minimum_frame_weight=0.02,
        )
        quality_only, quality_diagnostics = reliability_weighted_pool(
            features,
            quality,
            local_reliability=None,
            identity_penalty=None,
            local_weight=0.0,
            identity_weight=0.0,
            quality_weight=1.0,
            minimum_frame_weight=0.02,
        )
        torch.testing.assert_close(with_local, quality_only)
        torch.testing.assert_close(
            diagnostics["weights"],
            quality_diagnostics["weights"],
        )
        torch.testing.assert_close(
            diagnostics["local_absolute_gate"],
            torch.zeros((2, 3)),
        )

    def test_exact_cancellation_uses_unit_norm_symmetric_fallback(self) -> None:
        feature = torch.tensor([1.0, 0.0, 0.0])
        features = torch.stack(
            [feature, -feature, feature, -feature]
        )[None, :, None, :]
        pooled, diagnostics = reliability_weighted_pool(
            features,
            torch.ones((1, 4)),
            local_reliability=None,
            identity_penalty=None,
            local_weight=0.0,
            identity_weight=0.0,
            quality_weight=1.0,
            minimum_frame_weight=0.02,
        )
        torch.testing.assert_close(
            pooled.norm(dim=-1),
            torch.ones((1, 1)),
        )
        self.assertEqual(
            float(diagnostics["pre_normalization_min_norm"].item()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
