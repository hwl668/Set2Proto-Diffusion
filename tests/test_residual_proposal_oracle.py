from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.model import ConditionalTokenTransformer
from set2proto.residual_evidence import quality_weighted_anchor
from set2proto.residual_proposal_oracle import proposal_oracle_decode


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


class ResidualProposalOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(91)
        self.config = load_config(CONFIG_PATH, "expanded").to_dict()
        self.model = ConditionalTokenTransformer(
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
        self.codebook = torch.randn(16, 8) * 0.2
        self.target = torch.randint(0, 16, (3, 7))
        self.anchor = quality_weighted_anchor(self.condition, self.quality)
        self.teacher = F.normalize(
            self.anchor + self.codebook[self.target],
            dim=-1,
        )

    def _decode(self, mode: str, condition=None, quality=None):
        return proposal_oracle_decode(
            model=self.model,
            condition_features=(
                self.condition if condition is None else condition
            ),
            condition_quality=self.quality if quality is None else quality,
            target_tokens=self.target,
            teacher_map=self.teacher,
            quality_anchor=self.anchor,
            residual_codebook=self.codebook,
            config=self.config,
            proposal_fraction=1.0,
            top_k=16,
            rounds=1,
            mode=mode,
        )

    def test_exact_oracle_corrects_all_available_targets_without_injury(self):
        result = self._decode("exact_token")
        self.assertTrue(torch.equal(result.tokens, self.target))
        self.assertEqual(result.trace[0]["oracle_injured"], 0)
        self.assertGreaterEqual(
            int((result.tokens == self.target).sum()),
            int((result.initial_tokens == self.target).sum()),
        )

    def test_local_teacher_oracle_never_reduces_local_teacher_cosine(self):
        result = self._decode("local_teacher")
        initial_map = F.normalize(
            self.anchor + self.codebook[result.initial_tokens],
            dim=-1,
        )
        final_map = F.normalize(
            self.anchor + self.codebook[result.tokens],
            dim=-1,
        )
        initial_cosine = F.cosine_similarity(initial_map, self.teacher, dim=-1)
        final_cosine = F.cosine_similarity(final_map, self.teacher, dim=-1)
        self.assertTrue(bool((final_cosine >= initial_cosine - 1e-6).all()))

    def test_exact_oracle_is_jointly_frame_permutation_invariant(self):
        first = self._decode("exact_token")
        permutation = torch.tensor([2, 0, 3, 1])
        second = self._decode(
            "exact_token",
            condition=self.condition[:, permutation],
            quality=self.quality[:, permutation],
        )
        self.assertTrue(torch.equal(first.tokens, second.tokens))


if __name__ == "__main__":
    unittest.main()
