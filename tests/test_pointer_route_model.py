from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.pointer_route_model import PointerRouteTransformer
from set2proto.pointer_route_oracle import (
    build_route_candidates,
    hard_route_targets,
    soft_route_targets,
    teacher_route_scores,
)
from set2proto.pointer_route_training import (
    load_pointer_checkpoint,
    pointer_route_loss,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


def _small_model() -> PointerRouteTransformer:
    return PointerRouteTransformer(
        feature_dim=8,
        spatial_tokens=7,
        hidden_dim=32,
        layers=1,
        attention_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )


class PointerRouteModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.condition = torch.randn(3, 4, 7, 8)
        self.quality = torch.rand(3, 4) + 0.5

    def test_forward_shapes_candidate_norms_and_parameter_count(self) -> None:
        model = _small_model()
        routes = model.all_mask_input(3, device=torch.device("cpu"))
        logits, candidates = model(routes, self.condition, self.quality)
        self.assertEqual(tuple(logits.shape), (3, 7, 5))
        self.assertEqual(tuple(candidates.shape), (3, 7, 5, 8))
        self.assertLess(
            float((candidates.norm(dim=-1) - 1.0).abs().max()),
            1e-5,
        )
        self.assertGreater(model.parameter_count(), 0)

    def test_all_mask_logits_are_frame_permutation_equivariant(self) -> None:
        model = _small_model().eval()
        permutation = torch.tensor([2, 0, 3, 1])
        mask = model.all_mask_input(3, device=torch.device("cpu"))
        original, _ = model(mask, self.condition, self.quality)
        permuted, _ = model(
            mask,
            self.condition[:, permutation],
            self.quality[:, permutation],
        )
        reordered = torch.empty_like(permuted)
        reordered[..., 0] = permuted[..., 0]
        for permuted_index, original_index in enumerate(permutation.tolist()):
            reordered[..., original_index + 1] = (
                permuted[..., permuted_index + 1]
            )
        self.assertLess(
            float((original - reordered).abs().max().detach()),
            1e-5,
        )

    def test_soft_route_loss_is_finite_and_backpropagates(self) -> None:
        model = _small_model()
        mask = model.all_mask_input(3, device=torch.device("cpu"))
        logits, candidates = model(mask, self.condition, self.quality)
        teacher = build_route_candidates(
            self.condition,
            self.quality,
        )[:, :, 2]
        scores = teacher_route_scores(candidates, teacher)
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
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["loss"], 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )

    def test_checkpoint_reload_restores_logits(self) -> None:
        model = _small_model().eval()
        mask = model.all_mask_input(3, device=torch.device("cpu"))
        expected, _ = model(mask, self.condition, self.quality)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pointer.pt"
            torch.save({"model_state": model.state_dict(), "step": 1}, path)
            restored = _small_model().eval()
            payload = load_pointer_checkpoint(
                path=path,
                model=restored,
                device=torch.device("cpu"),
            )
            actual, _ = restored(mask, self.condition, self.quality)
        self.assertEqual(payload["step"], 1)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
