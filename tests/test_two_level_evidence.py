from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.decoding import compute_two_level_evidence


class TwoLevelEvidenceTests(unittest.TestCase):
    def test_shapes_normalization_and_stop_gradient(self) -> None:
        torch.manual_seed(19)
        condition = torch.randn(2, 4, 7, 8, requires_grad=True)
        quality = torch.rand(2, 4, requires_grad=True)
        codebook = torch.randn(11, 8, requires_grad=True)

        result = compute_two_level_evidence(
            condition,
            quality,
            codebook,
        )

        self.assertEqual(tuple(result.evidence.shape), (2, 7, 11))
        self.assertEqual(tuple(result.frame_reliability.shape), (2, 4))
        self.assertEqual(tuple(result.mean_local_usage.shape), (2, 4))
        torch.testing.assert_close(
            result.frame_reliability.sum(dim=1),
            torch.ones(2),
        )
        for value in (
            result.evidence,
            result.frame_reliability,
            result.identity_consensus,
            result.identity_reliability,
            result.quality_standardized,
            result.mean_local_usage,
            result.effective_frames,
        ):
            self.assertFalse(value.requires_grad)

    def test_is_permutation_invariant_and_diagnostics_are_equivariant(
        self,
    ) -> None:
        torch.manual_seed(23)
        condition = torch.randn(2, 4, 9, 6)
        quality = torch.rand(2, 4) + 1.0
        codebook = torch.randn(13, 6)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)

        original = compute_two_level_evidence(
            condition,
            quality,
            codebook,
        )
        permuted = compute_two_level_evidence(
            condition[:, permutation],
            quality[:, permutation],
            codebook,
        )

        torch.testing.assert_close(
            original.evidence,
            permuted.evidence,
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            original.frame_reliability,
            permuted.frame_reliability[:, inverse],
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            original.mean_local_usage,
            permuted.mean_local_usage[:, inverse],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_wrong_identity_is_downweighted_despite_high_quality(self) -> None:
        # Three inliers agree globally. The fourth frame is a sharp but
        # deliberately orthogonal identity and receives the highest quality.
        condition = torch.zeros(1, 4, 9, 4)
        condition[:, :3, :, 0] = 1.0
        condition[:, 3, :, 1] = 1.0
        quality = torch.tensor([[1.0, 1.0, 1.0, 4.0]])
        codebook = torch.eye(4)

        result = compute_two_level_evidence(
            condition,
            quality,
            codebook,
        )

        wrong_weight = float(result.frame_reliability[0, 3])
        inlier_weight = float(result.frame_reliability[0, :3].mean())
        self.assertLess(wrong_weight, 0.01)
        self.assertGreater(inlier_weight, 0.32)
        self.assertGreater(
            float(result.evidence[0, 0, 0]),
            float(result.evidence[0, 0, 1]) + 0.8,
        )

    def test_single_inlier_can_supply_complementary_local_evidence(self) -> None:
        # All four frames share the same global identity. At position zero only
        # frame zero contains the complementary e1 evidence. The local stage
        # should retain it instead of averaging it away.
        condition = torch.zeros(1, 4, 9, 4)
        condition[:, :, :, 0] = 1.0
        condition[:, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        quality = torch.ones(1, 4)
        codebook = torch.eye(4)

        result = compute_two_level_evidence(
            condition,
            quality,
            codebook,
        )

        self.assertGreater(float(result.evidence[0, 0, 1]), 0.90)
        self.assertLess(float(result.evidence[0, 1, 1]), 0.01)
        self.assertGreater(float(result.frame_reliability[0, 0]), 0.20)


if __name__ == "__main__":
    unittest.main()
