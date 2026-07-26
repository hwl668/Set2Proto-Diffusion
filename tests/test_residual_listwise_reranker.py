from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.model import ConditionalTokenTransformer
from set2proto.residual_evidence import quality_weighted_anchor
from set2proto.residual_listwise_reranker import (
    RERANKER_FEATURE_NAMES,
    ResidualListwiseReranker,
    extract_listwise_features,
    listwise_rerank_decode,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


class ResidualListwiseRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
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
        self.config["p2_5"]["top_k"] = 4
        self.condition = torch.randn(3, 4, 7, 8)
        self.quality = torch.rand(3, 4) + 0.1
        self.codebook = torch.randn(16, 8) * 0.2
        self.reranker = ResidualListwiseReranker(
            feature_mean=torch.zeros(len(RERANKER_FEATURE_NAMES)),
            feature_std=torch.ones(len(RERANKER_FEATURE_NAMES)),
            hidden_dim=16,
            dropout=0.0,
        ).eval()

    def test_feature_shapes_are_finite_and_stop_gradient(self) -> None:
        mask = torch.full((3, 7), 16, dtype=torch.int64)
        logits = self.model(mask, self.condition, self.quality)
        candidates = torch.topk(logits, k=4, dim=-1).indices
        evidence = torch.randn_like(logits)
        anchor = quality_weighted_anchor(self.condition, self.quality)
        features, maps, global_embeddings = extract_listwise_features(
            proposal_logits=logits,
            candidate_tokens=candidates,
            evidence=evidence,
            condition_features=self.condition,
            quality_anchor=anchor,
            residual_codebook=self.codebook,
            effective_frames=torch.full((3,), 2.5),
        )
        self.assertEqual(
            tuple(features.shape),
            (3, 7, 4, len(RERANKER_FEATURE_NAMES)),
        )
        self.assertEqual(tuple(maps.shape), (3, 7, 4, 8))
        self.assertEqual(tuple(global_embeddings.shape), (3, 7, 4, 8))
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertFalse(features.requires_grad)

    def test_listwise_loss_backpropagates(self) -> None:
        features = torch.randn(5, 7, 4, len(RERANKER_FEATURE_NAMES))
        target = torch.softmax(torch.randn(5, 7, 4), dim=-1)
        reranker = ResidualListwiseReranker(
            feature_mean=torch.zeros(len(RERANKER_FEATURE_NAMES)),
            feature_std=torch.ones(len(RERANKER_FEATURE_NAMES)),
            hidden_dim=16,
            dropout=0.0,
        )
        scores = reranker(features)
        loss = -(target * torch.log_softmax(scores, dim=-1)).sum(-1).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(parameter.grad is not None for parameter in reranker.parameters())
        )

    def test_high_threshold_is_exact_top1_fallback(self) -> None:
        result = listwise_rerank_decode(
            model=self.model,
            reranker=self.reranker,
            condition_features=self.condition,
            condition_quality=self.quality,
            residual_codebook=self.codebook,
            config=self.config,
            model_logit_weight=0.0,
            replacement_threshold=100.0,
        )
        self.assertTrue(torch.equal(result.initial_tokens, result.tokens))
        self.assertFalse(bool(result.replacement_mask.any()))

    def test_joint_frame_permutation_is_equivalent(self) -> None:
        first = listwise_rerank_decode(
            model=self.model,
            reranker=self.reranker,
            condition_features=self.condition,
            condition_quality=self.quality,
            residual_codebook=self.codebook,
            config=self.config,
            model_logit_weight=0.25,
            replacement_threshold=0.1,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        second = listwise_rerank_decode(
            model=self.model,
            reranker=self.reranker,
            condition_features=self.condition[:, permutation],
            condition_quality=self.quality[:, permutation],
            residual_codebook=self.codebook,
            config=self.config,
            model_logit_weight=0.25,
            replacement_threshold=0.1,
        )
        self.assertTrue(torch.equal(first.tokens, second.tokens))


if __name__ == "__main__":
    unittest.main()
