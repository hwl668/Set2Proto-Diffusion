from __future__ import annotations

import unittest
from pathlib import Path

import torch

from set2proto.config import load_config
from set2proto.model import ConditionalTokenTransformer
from set2proto.set_aware_residual_reranker import (
    SetAwareResidualReranker,
    set_aware_rerank_decode,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml"


class SetAwareResidualRerankerTests(unittest.TestCase):
    def _model(self) -> SetAwareResidualReranker:
        return SetAwareResidualReranker(
            feature_mean=torch.zeros(14),
            feature_std=torch.ones(14),
            positions=49,
            top_k=8,
            hidden_dim=32,
            attention_heads=4,
            layers=1,
            feedforward_multiplier=2,
            dropout=0.0,
        )

    def test_joint_lattice_shapes_and_backpropagation(self) -> None:
        model = self._model()
        features = torch.randn(3, 49, 8, 14)
        scores, gates = model(features)
        self.assertEqual(scores.shape, (3, 49, 8))
        self.assertEqual(gates.shape, (3, 49))
        loss = scores.square().mean() + gates.square().mean()
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_other_positions_change_candidate_scores(self) -> None:
        model = self._model().eval()
        features = torch.zeros(1, 49, 8, 14)
        with torch.no_grad():
            first, _ = model(features)
            features[:, 1:, :, 0] = 2.0
            second, _ = model(features)
        self.assertGreater(
            float((first[:, 0] - second[:, 0]).abs().max()),
            0.0,
        )

    def test_invalid_lattice_is_rejected(self) -> None:
        model = self._model()
        with self.assertRaisesRegex(ValueError, "lattice"):
            model(torch.randn(2, 48, 8, 14))

    def _decode_fixture(self):
        torch.manual_seed(31)
        config = load_config(CONFIG_PATH, "expanded").to_dict()
        config["p2_6"]["top_k"] = 4
        token_model = ConditionalTokenTransformer(
            feature_dim=8,
            spatial_tokens=7,
            codebook_size=16,
            hidden_dim=32,
            layers=1,
            attention_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        ).eval()
        reranker = SetAwareResidualReranker(
            feature_mean=torch.zeros(14),
            feature_std=torch.ones(14),
            positions=7,
            top_k=4,
            hidden_dim=32,
            attention_heads=4,
            layers=1,
            feedforward_multiplier=2,
            dropout=0.0,
        ).eval()
        condition = torch.randn(3, 4, 7, 8)
        quality = torch.rand(3, 4) + 0.1
        codebook = torch.randn(16, 8) * 0.2
        return config, token_model, reranker, condition, quality, codebook

    def test_zero_budget_is_exact_top1_fallback(self) -> None:
        config, model, reranker, condition, quality, codebook = (
            self._decode_fixture()
        )
        result = set_aware_rerank_decode(
            model=model,
            reranker=reranker,
            condition_features=condition,
            condition_quality=quality,
            residual_codebook=codebook,
            config=config,
            model_logit_weight=0.0,
            replacement_threshold=0.0,
            gate_threshold=0.0,
            replacement_budget_fraction=0.0,
        )
        self.assertTrue(torch.equal(result.initial_tokens, result.tokens))
        self.assertFalse(bool(result.replacement_mask.any()))

    def test_replacement_budget_is_enforced_per_set(self) -> None:
        config, model, reranker, condition, quality, codebook = (
            self._decode_fixture()
        )
        result = set_aware_rerank_decode(
            model=model,
            reranker=reranker,
            condition_features=condition,
            condition_quality=quality,
            residual_codebook=codebook,
            config=config,
            model_logit_weight=0.0,
            replacement_threshold=0.0,
            gate_threshold=0.0,
            replacement_budget_fraction=0.3,
        )
        self.assertTrue(
            bool((result.replacement_mask.sum(dim=1) <= 2).all())
        )

    def test_joint_frame_permutation_is_token_equivalent(self) -> None:
        config, model, reranker, condition, quality, codebook = (
            self._decode_fixture()
        )
        kwargs = {
            "model": model,
            "reranker": reranker,
            "residual_codebook": codebook,
            "config": config,
            "model_logit_weight": 0.25,
            "replacement_threshold": 0.2,
            "gate_threshold": 0.5,
            "replacement_budget_fraction": 0.2,
        }
        first = set_aware_rerank_decode(
            condition_features=condition,
            condition_quality=quality,
            **kwargs,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        second = set_aware_rerank_decode(
            condition_features=condition[:, permutation],
            condition_quality=quality[:, permutation],
            **kwargs,
        )
        self.assertTrue(torch.equal(first.tokens, second.tokens))


if __name__ == "__main__":
    unittest.main()
