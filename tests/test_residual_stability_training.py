from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.decoding import maskgit_decode
from set2proto.model import ConditionalTokenTransformer
from set2proto.residual_stability_training import (
    build_exposure_matched_corruption,
    build_residual_self_rollout,
    weighted_token_cross_entropy,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


def _model() -> ConditionalTokenTransformer:
    return ConditionalTokenTransformer(
        feature_dim=8,
        spatial_tokens=7,
        codebook_size=16,
        hidden_dim=32,
        layers=1,
        attention_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )


class ResidualStabilityTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(77)
        self.model = _model()
        self.condition = torch.randn(6, 4, 7, 8)
        self.quality = torch.rand(6, 4) + 0.1
        self.targets = torch.randint(0, 16, (6, 7))
        self.config = load_config(CONFIG_PATH, "expanded").to_dict()

    def test_self_rollout_is_reproducible_detached_and_partial(self) -> None:
        first = build_residual_self_rollout(
            model=self.model,
            condition_features=self.condition,
            condition_quality=self.quality,
            rollout_total_steps=4,
            rollout_depths=[1, 2, 3],
            generator=torch.Generator().manual_seed(8),
        )
        second = build_residual_self_rollout(
            model=self.model,
            condition_features=self.condition,
            condition_quality=self.quality,
            rollout_total_steps=4,
            rollout_depths=[1, 2, 3],
            generator=torch.Generator().manual_seed(8),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first.requires_grad)
        self.assertTrue(bool((first == self.model.mask_token_id).any(dim=1).all()))
        # With only seven positions, the first cosine-schedule rollout depth
        # can legitimately commit zero positions after integer rounding.
        # The mixed-depth batch must still contain committed predictions.
        self.assertTrue(bool((first != self.model.mask_token_id).any()))

    def test_exposure_corruption_has_weights_and_all_three_sources(self) -> None:
        config = self.config
        config["p2_2"]["corruption"].update(
            {
                "all_mask_probability": 0.0,
                "teacher_partial_probability": 0.0,
                "self_rollout_probability": 1.0,
            }
        )
        state, weights, diagnostics = build_exposure_matched_corruption(
            model=self.model,
            condition_features=self.condition,
            condition_quality=self.quality,
            target_tokens=self.targets,
            config=config,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertEqual(tuple(state.shape), (6, 7))
        self.assertEqual(tuple(weights.shape), (6, 7))
        self.assertTrue(bool((weights > 0).all()))
        self.assertEqual(diagnostics["self_rollout_fraction"], 1.0)
        self.assertGreater(diagnostics["wrong_committed_fraction"], 0.0)
        self.assertFalse(state.requires_grad)
        self.assertFalse(weights.requires_grad)

    def test_weighted_loss_is_finite_and_backpropagates(self) -> None:
        inputs = torch.full_like(self.targets, self.model.mask_token_id)
        logits = self.model(inputs, self.condition, self.quality)
        weights = torch.ones_like(self.targets, dtype=torch.float32)
        loss, diagnostics = weighted_token_cross_entropy(
            logits,
            self.targets,
            weights,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(diagnostics["weighted_accuracy"], 0.0)
        self.assertTrue(
            any(value.grad is not None for value in self.model.parameters())
        )

    def test_decoder_exposes_iteration_and_remask_traces(self) -> None:
        codebook = torch.randn(16, 8)
        result = maskgit_decode(
            self.model.eval(),
            self.condition,
            self.quality,
            codebook=codebook,
            steps=4,
            mode="evidence-remask",
            top_k_frames=3,
            evidence_lambda=0.5,
        )
        self.assertEqual(len(result.provisional_tokens), 4)
        self.assertEqual(len(result.committed_after_iteration), 4)
        self.assertEqual(len(result.remasked), 4)
        self.assertTrue(bool(result.committed_after_iteration[-1].all()))
        self.assertFalse(bool(result.remasked[-1].any()))


if __name__ == "__main__":
    unittest.main()
