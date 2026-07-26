from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.residual_evidence import (
    compute_local_residual_evidence,
    compute_two_level_residual_evidence,
    quality_weighted_anchor,
    residual_candidate_support,
)


class ResidualEvidenceTests(unittest.TestCase):
    def test_quality_anchor_is_unit_and_permutation_invariant(self) -> None:
        torch.manual_seed(31)
        condition = torch.randn(2, 4, 7, 8)
        quality = torch.rand(2, 4) + 0.2
        permutation = torch.tensor([3, 1, 0, 2])
        original = quality_weighted_anchor(condition, quality)
        permuted = quality_weighted_anchor(
            condition[:, permutation],
            quality[:, permutation],
        )
        self.assertEqual(tuple(original.shape), (2, 7, 8))
        torch.testing.assert_close(
            original.norm(dim=-1),
            torch.ones(2, 7),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)

    def test_analytic_candidate_support_matches_explicit_reconstruction(
        self,
    ) -> None:
        torch.manual_seed(37)
        condition = F.normalize(torch.randn(2, 4, 3, 5), dim=-1)
        quality = torch.rand(2, 4) + 0.5
        anchor = quality_weighted_anchor(condition, quality)
        codebook = 0.4 * torch.randn(11, 5)
        actual = residual_candidate_support(condition, anchor, codebook)

        candidate = F.normalize(
            anchor[:, :, None, :] + codebook[None, None, :, :],
            dim=-1,
        )
        expected = torch.einsum("bmpd,bpkd->bmpk", condition, candidate)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertFalse(actual.requires_grad)

    def test_two_level_residual_evidence_downweights_wrong_identity(
        self,
    ) -> None:
        condition = torch.zeros(1, 4, 5, 4)
        condition[:, :3, :, 0] = 1.0
        condition[:, 3, :, 1] = 1.0
        quality = torch.tensor([[1.0, 1.0, 1.0, 4.0]])
        anchor = quality_weighted_anchor(condition, quality)
        codebook = torch.tensor(
            [
                [0.5, 0.0, 0.0, 0.0],
                [-0.5, 1.0, 0.0, 0.0],
            ]
        )
        result = compute_two_level_residual_evidence(
            condition,
            quality,
            anchor,
            codebook,
        )
        self.assertEqual(tuple(result.evidence.shape), (1, 5, 2))
        self.assertLess(float(result.frame_reliability[0, 3]), 0.01)
        self.assertGreater(
            float(result.evidence[0, 0, 0]),
            float(result.evidence[0, 0, 1]),
        )

    def test_local_and_two_level_are_permutation_invariant(self) -> None:
        torch.manual_seed(41)
        condition = torch.randn(2, 4, 6, 7, requires_grad=True)
        quality = torch.rand(2, 4, requires_grad=True) + 0.5
        anchor = quality_weighted_anchor(condition, quality)
        codebook = torch.randn(13, 7, requires_grad=True) * 0.3
        permutation = torch.tensor([2, 0, 3, 1])

        local = compute_local_residual_evidence(
            condition,
            anchor,
            codebook,
            top_k_frames=3,
        )
        two_level = compute_two_level_residual_evidence(
            condition,
            quality,
            anchor,
            codebook,
        )
        permuted_anchor = quality_weighted_anchor(
            condition[:, permutation],
            quality[:, permutation],
        )
        permuted_local = compute_local_residual_evidence(
            condition[:, permutation],
            permuted_anchor,
            codebook,
            top_k_frames=3,
        )
        permuted_two_level = compute_two_level_residual_evidence(
            condition[:, permutation],
            quality[:, permutation],
            permuted_anchor,
            codebook,
        )
        torch.testing.assert_close(local, permuted_local, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            two_level.evidence,
            permuted_two_level.evidence,
            atol=2e-6,
            rtol=2e-6,
        )
        self.assertFalse(local.requires_grad)
        self.assertFalse(two_level.evidence.requires_grad)


if __name__ == "__main__":
    unittest.main()
