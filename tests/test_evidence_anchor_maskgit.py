from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.decoding import maskgit_decode, one_shot_decode
from set2proto.evidence_anchor_maskgit import (
    EvidenceAnchoredTokenTransformer,
    canonicalize_condition_set,
)


class EvidenceAnchorMaskGITTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = EvidenceAnchoredTokenTransformer(
            feature_dim=8,
            spatial_tokens=7,
            codebook_size=16,
            hidden_dim=32,
            layers=1,
            attention_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        ).eval()
        self.condition = torch.randn(3, 4, 7, 8)
        self.quality = torch.rand(3, 4) + 0.1
        self.anchor = torch.nn.functional.normalize(
            torch.randn(3, 7, 8),
            dim=-1,
        )
        self.codebook = torch.randn(16, 8) * 0.2

    def test_explicit_anchor_forward_shape(self) -> None:
        tokens = torch.full((3, 7), 16, dtype=torch.int64)
        logits = self.model(
            tokens,
            self.condition,
            self.quality,
            self.anchor,
        )
        self.assertEqual(tuple(logits.shape), (3, 7, 16))

    def test_one_shot_and_maskgit_accept_anchor(self) -> None:
        one_shot = one_shot_decode(
            self.model,
            self.condition,
            self.quality,
            self.anchor,
        )
        decoded = maskgit_decode(
            self.model,
            self.condition,
            self.quality,
            codebook=self.codebook,
            steps=4,
            mode="confidence",
            top_k_frames=3,
            evidence_lambda=0.0,
            evidence_anchor=self.anchor,
        )
        self.assertEqual(tuple(one_shot.shape), (3, 7))
        self.assertEqual(tuple(decoded.tokens.shape), (3, 7))
        self.assertFalse(bool((decoded.tokens == 16).any()))

    def test_joint_frame_permutation_is_equivalent(self) -> None:
        first = one_shot_decode(
            self.model,
            self.condition,
            self.quality,
            self.anchor,
        )
        order = torch.tensor([2, 0, 3, 1])
        second = one_shot_decode(
            self.model,
            self.condition[:, order],
            self.quality[:, order],
            self.anchor,
        )
        self.assertTrue(torch.equal(first, second))

    def test_canonical_order_is_exact_for_joint_permutation(self) -> None:
        order = torch.tensor([2, 0, 3, 1])
        first_features, first_quality = canonicalize_condition_set(
            self.condition,
            self.quality,
        )
        second_features, second_quality = canonicalize_condition_set(
            self.condition[:, order],
            self.quality[:, order],
        )
        self.assertTrue(torch.equal(first_features, second_features))
        self.assertTrue(torch.equal(first_quality, second_quality))


if __name__ == "__main__":
    unittest.main()
