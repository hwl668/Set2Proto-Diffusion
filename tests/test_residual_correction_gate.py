from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.model import ConditionalTokenTransformer
from set2proto.residual_correction_gate import (
    GATE_FEATURE_NAMES,
    ResidualCorrectionGate,
    extract_correction_features,
    proposal_labels,
    risk_controlled_refine,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


class ResidualCorrectionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
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
        self.codebook = torch.randn(16, 8)
        self.condition = torch.randn(3, 4, 7, 8)
        self.quality = torch.rand(3, 4) + 0.1
        self.gate = ResidualCorrectionGate(
            feature_mean=torch.zeros(len(GATE_FEATURE_NAMES)),
            feature_std=torch.ones(len(GATE_FEATURE_NAMES)),
            hidden_dim=16,
            dropout=0.0,
        ).eval()

    def test_feature_shape_finite_and_stop_gradient(self) -> None:
        current = torch.randint(0, 16, (3, 7))
        logits = torch.randn(3, 7, 16, requires_grad=True)
        candidates = logits.argmax(dim=-1)
        evidence = torch.randn(3, 7, 16)
        features = extract_correction_features(
            proposal_logits=logits,
            evidence=evidence,
            current_tokens=current,
            candidate_tokens=candidates,
            residual_codebook=self.codebook,
            proposal_fraction=0.25,
            round_fraction=0.5,
            effective_frames=torch.full((3,), 2.5),
            condition_frames=4,
        )
        self.assertEqual(
            tuple(features.shape),
            (3, 7, len(GATE_FEATURE_NAMES)),
        )
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertFalse(features.requires_grad)

    def test_proposal_labels_distinguish_correction_and_injury(self) -> None:
        current = torch.tensor([[0, 1, 2, 3]])
        candidate = torch.tensor([[0, 2, 1, 4]])
        target = torch.tensor([[0, 2, 2, 3]])
        labels = proposal_labels(current, candidate, target)
        self.assertTrue(
            torch.equal(labels, torch.tensor([[0, 1, 2, 2]]))
        )

    def test_high_threshold_is_exact_one_shot_fallback(self) -> None:
        decoded = risk_controlled_refine(
            model=self.model,
            gate=self.gate,
            condition_features=self.condition,
            condition_quality=self.quality,
            residual_codebook=self.codebook,
            config=self.config,
            proposal_fraction=0.5,
            utility_threshold=1.0,
            rounds=3,
        )
        self.assertTrue(torch.equal(decoded.initial_tokens, decoded.tokens))
        self.assertTrue(
            all(not bool(mask.any()) for mask in decoded.replacement_masks)
        )

    def test_joint_frame_permutation_is_equivalent(self) -> None:
        first = risk_controlled_refine(
            model=self.model,
            gate=self.gate,
            condition_features=self.condition,
            condition_quality=self.quality,
            residual_codebook=self.codebook,
            config=self.config,
            proposal_fraction=0.3,
            utility_threshold=0.0,
            rounds=2,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        second = risk_controlled_refine(
            model=self.model,
            gate=self.gate,
            condition_features=self.condition[:, permutation],
            condition_quality=self.quality[:, permutation],
            residual_codebook=self.codebook,
            config=self.config,
            proposal_fraction=0.3,
            utility_threshold=0.0,
            rounds=2,
        )
        self.assertTrue(torch.equal(first.tokens, second.tokens))


if __name__ == "__main__":
    unittest.main()
