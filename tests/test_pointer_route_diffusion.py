from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.pointer_route_diffusion import (
    build_pointer_corruption,
    build_pointer_rollout_corruption,
    compute_pointer_route_evidence,
    cosine_commit_count,
    decode_pointer_routes,
)
from set2proto.pointer_route_model import PointerRouteTransformer
from set2proto.pointer_route_oracle import (
    build_route_candidates,
    hard_route_targets,
    soft_route_targets,
    teacher_route_scores,
)
from set2proto.pointer_route_training import pointer_route_loss


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


def _model() -> PointerRouteTransformer:
    return PointerRouteTransformer(
        feature_dim=8,
        spatial_tokens=7,
        hidden_dim=32,
        layers=1,
        attention_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )


class _ChangingRetentionModel(torch.nn.Module):
    """Small deterministic decoder fixture that forces a useful remask."""

    spatial_tokens = 7
    mask_token_id = 5

    def all_mask_input(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size, self.spatial_tokens),
            self.mask_token_id,
            dtype=torch.int64,
            device=device,
        )

    def forward(
        self,
        input_routes: torch.Tensor,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del condition_quality
        batch = input_routes.shape[0]
        device = input_routes.device
        position = torch.arange(
            self.spatial_tokens,
            dtype=torch.float32,
            device=device,
        )
        logits = torch.zeros(
            batch,
            self.spatial_tokens,
            self.mask_token_id,
            device=device,
        )
        logits[..., 0] = position[None] * 0.35
        committed = input_routes != self.mask_token_id
        logits[..., 0] = torch.where(
            committed,
            torch.full_like(logits[..., 0], -8.0),
            logits[..., 0],
        )
        logits[..., 1] = torch.where(
            committed,
            torch.full_like(logits[..., 1], 8.0),
            logits[..., 1],
        )
        candidates = torch.zeros(
            batch,
            self.spatial_tokens,
            self.mask_token_id,
            condition_features.shape[-1],
            device=device,
        )
        candidates[..., 0] = 1.0
        return logits, candidates


class PointerRouteDiffusionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.condition = torch.randn(4, 4, 7, 8)
        self.quality = torch.rand(4, 4) + 0.2

    def test_corruption_is_reproducible_and_never_empty(self) -> None:
        routes = torch.randint(0, 5, (32, 49))
        first = build_pointer_corruption(
            routes,
            mask_token_id=5,
            all_mask_probability=0.5,
            min_mask_ratio=0.05,
            generator=torch.Generator().manual_seed(7),
        )
        second = build_pointer_corruption(
            routes,
            mask_token_id=5,
            all_mask_probability=0.5,
            min_mask_ratio=0.05,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(bool(first[1].any(dim=1).all()))
        self.assertTrue(
            torch.equal(first[0][~first[1]], routes[~first[1]])
        )

    def test_cosine_schedule_is_monotonic_and_finishes(self) -> None:
        counts = [
            cosine_commit_count(positions=49, step=step, total_steps=4)
            for step in range(1, 5)
        ]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 49)
        self.assertLess(counts[0], counts[-1])

    def test_self_rollout_corruption_is_detached_and_reproducible(
        self,
    ) -> None:
        model = _model().train()
        target = torch.randint(0, 5, (4, 7))
        first = build_pointer_rollout_corruption(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            target_routes=target,
            rollout_probability=1.0,
            rollout_total_steps=4,
            rollout_depths=[2],
            generator=torch.Generator().manual_seed(17),
        )
        second = build_pointer_rollout_corruption(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            target_routes=target,
            rollout_probability=1.0,
            rollout_total_steps=4,
            rollout_depths=[2],
            generator=torch.Generator().manual_seed(17),
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(bool((~first[1]).any(dim=1).all()))
        self.assertTrue(bool(first[1].any(dim=1).all()))
        self.assertFalse(first[0].requires_grad)
        self.assertFalse(first[1].requires_grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )
        self.assertEqual(first[2]["rollout_fraction"], 1.0)
        self.assertGreater(first[2]["committed_fraction"], 0.0)

    def test_self_rollout_probability_zero_stays_all_masked(self) -> None:
        model = _model().eval()
        target = torch.randint(0, 5, (4, 7))
        corrupted, loss_mask, diagnostics = (
            build_pointer_rollout_corruption(
                model=model,
                condition_features=self.condition,
                condition_quality=self.quality,
                target_routes=target,
                rollout_probability=0.0,
                rollout_total_steps=4,
                rollout_depths=[1, 2, 3],
                generator=torch.Generator().manual_seed(19),
            )
        )
        self.assertTrue(bool((corrupted == model.mask_token_id).all()))
        self.assertTrue(bool(loss_mask.all()))
        self.assertEqual(diagnostics["all_mask_fraction"], 1.0)

    def test_decoder_commits_every_position_without_remask(self) -> None:
        model = _model().eval()
        result = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
        )
        self.assertEqual(tuple(result.final_routes.shape), (4, 7))
        self.assertTrue(bool((result.final_routes < 5).all()))
        self.assertTrue(bool((result.commit_steps > 0).all()))
        self.assertTrue(bool(result.committed_after_step[-1].all()))
        cumulative = [
            int(value[0].sum().item())
            for value in result.committed_after_step
        ]
        self.assertEqual(cumulative, sorted(cumulative))
        for previous, current in zip(
            result.committed_after_step,
            result.committed_after_step[1:],
        ):
            self.assertTrue(bool((previous <= current).all()))

    def test_evidence_order_does_not_modify_route_logits(self) -> None:
        model = _model().eval()
        confidence = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
        )
        guided = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
            commit_mode="evidence_order_only",
            evidence_lambda=1.0,
        )
        torch.testing.assert_close(
            confidence.probabilities[0],
            guided.probabilities[0],
        )
        self.assertTrue(
            torch.equal(
                confidence.provisional_routes[0],
                guided.provisional_routes[0],
            )
        )
        self.assertTrue(bool(guided.committed_after_step[-1].all()))
        self.assertFalse(guided.commit_score.requires_grad)
        self.assertFalse(guided.commit_evidence.requires_grad)
        self.assertIsNotNone(guided.frame_reliability)

    def test_evidence_order_is_identical_at_one_step(self) -> None:
        model = _model().eval()
        confidence = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=1,
        )
        guided = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=1,
            commit_mode="evidence_order_only",
            evidence_lambda=2.0,
        )
        self.assertTrue(
            torch.equal(confidence.final_routes, guided.final_routes)
        )

    def test_evidence_logits_changes_selection_distribution_only(self) -> None:
        model = _model().eval()
        confidence = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
        )
        guided = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
            commit_mode="evidence_logits",
            evidence_lambda=1.0,
        )
        self.assertGreater(
            float(
                (
                    confidence.probabilities[0]
                    - guided.probabilities[0]
                )
                .abs()
                .max()
            ),
            1e-6,
        )
        self.assertTrue(bool(guided.committed_after_step[-1].all()))
        for previous, current in zip(
            guided.committed_after_step,
            guided.committed_after_step[1:],
        ):
            self.assertTrue(bool((previous <= current).all()))
        self.assertFalse(guided.commit_evidence.requires_grad)

    def test_budgeted_remask_is_active_and_finishes_full_commit(self) -> None:
        model = _ChangingRetentionModel().eval()
        result = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
            commit_mode="evidence_remask",
            evidence_lambda=0.5,
            remask_fraction=1.0,
            remask_min_score_gain=0.0,
        )
        self.assertTrue(bool(result.committed_after_step[-1].all()))
        self.assertTrue(bool((result.final_routes < 5).all()))
        self.assertGreater(int(result.remask_count.sum().item()), 0)
        self.assertGreater(
            sum(int(value.sum().item()) for value in result.remasked),
            0,
        )
        expected = [
            cosine_commit_count(positions=7, step=step, total_steps=4)
            for step in range(1, 5)
        ]
        actual = [
            int(value[0].sum().item())
            for value in result.committed_after_step
        ]
        self.assertEqual(actual, expected)
        self.assertFalse(bool(result.remasked[-1].any()))

    def test_budgeted_remask_respects_replacement_margin(self) -> None:
        model = _ChangingRetentionModel().eval()
        result = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
            commit_mode="evidence_remask",
            evidence_lambda=0.5,
            remask_fraction=1.0,
            remask_min_score_gain=2.0,
        )
        self.assertEqual(int(result.remask_count.sum().item()), 0)
        self.assertTrue(bool(result.committed_after_step[-1].all()))

    def test_two_level_route_evidence_downweights_wrong_identity(self) -> None:
        condition = torch.zeros(1, 4, 7, 4)
        condition[:, :3, :, 0] = 1.0
        condition[:, 3, :, 1] = 1.0
        condition.requires_grad_(True)
        quality = torch.tensor(
            [[1.0, 1.0, 1.0, 4.0]],
            requires_grad=True,
        )
        candidates = build_route_candidates(condition, quality)
        result = compute_pointer_route_evidence(
            condition_features=condition,
            condition_quality=quality,
            candidates=candidates,
        )
        self.assertEqual(tuple(result.evidence.shape), (1, 7, 5))
        self.assertLess(float(result.frame_reliability[0, 3]), 0.01)
        self.assertGreater(
            float(result.evidence[0, 0, 1]),
            float(result.evidence[0, 0, 4]) + 0.5,
        )
        for value in (
            result.evidence,
            result.frame_reliability,
            result.mean_local_usage,
            result.effective_frames,
        ):
            self.assertFalse(value.requires_grad)

    def test_masked_loss_backpropagates_only_with_valid_mask(self) -> None:
        model = _model()
        targets = torch.randint(0, 5, (4, 7))
        corrupted, mask = build_pointer_corruption(
            targets,
            mask_token_id=5,
            all_mask_probability=0.0,
            min_mask_ratio=0.2,
            generator=torch.Generator().manual_seed(9),
        )
        logits, candidates = model(
            corrupted,
            self.condition,
            self.quality,
        )
        teacher = candidates[:, :, 2].detach()
        scores = teacher_route_scores(candidates.detach(), teacher)
        hard = hard_route_targets(scores, delta=0.02)
        soft = soft_route_targets(
            scores,
            delta=0.02,
            temperature=0.02,
        )
        config = load_config(CONFIG_PATH, "expanded").to_dict()
        loss, metrics = pointer_route_loss(
            logits=logits,
            candidates=candidates,
            hard_targets=hard,
            soft_targets=soft,
            teacher_maps=teacher,
            config=config,
            loss_mask=mask,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(
            metrics["masked_fraction"],
            float(mask.float().mean()),
            places=6,
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_four_step_decode_is_jointly_frame_permutation_equivariant(
        self,
    ) -> None:
        model = _model().eval()
        permutation = torch.tensor([2, 0, 3, 1])
        original = decode_pointer_routes(
            model=model,
            condition_features=self.condition,
            condition_quality=self.quality,
            steps=4,
        )
        permuted = decode_pointer_routes(
            model=model,
            condition_features=self.condition[:, permutation],
            condition_quality=self.quality[:, permutation],
            steps=4,
        )
        mask = model.all_mask_input(4, device=torch.device("cpu"))
        original_candidates = model(
            mask,
            self.condition,
            self.quality,
        )[1]
        permuted_candidates = model(
            mask,
            self.condition[:, permutation],
            self.quality[:, permutation],
        )[1]
        original_map = F.normalize(
            torch.gather(
                original_candidates,
                2,
                original.final_routes[..., None, None].expand(
                    -1,
                    -1,
                    1,
                    8,
                ),
            ).squeeze(2),
            dim=-1,
        )
        permuted_map = F.normalize(
            torch.gather(
                permuted_candidates,
                2,
                permuted.final_routes[..., None, None].expand(
                    -1,
                    -1,
                    1,
                    8,
                ),
            ).squeeze(2),
            dim=-1,
        )
        self.assertLess(
            float((original_map - permuted_map).abs().max()),
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
