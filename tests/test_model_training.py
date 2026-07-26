from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config, validate_config
from set2proto.decoding import (
    compute_local_evidence,
    maskgit_decode,
)
from set2proto.model import ConditionalTokenTransformer
from set2proto.quantization import prepare_quantization_artifacts
from set2proto.synthetic import prepare_synthetic_dataset
from set2proto.training import (
    TokenTrainingDataset,
    build_corrupted_tokens,
    train_single_model,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


def _small_model() -> ConditionalTokenTransformer:
    return ConditionalTokenTransformer(
        feature_dim=8,
        spatial_tokens=9,
        codebook_size=16,
        hidden_dim=32,
        layers=2,
        attention_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )


def _tiny_training_config(root: Path) -> dict:
    config = load_config(CONFIG_PATH, "smoke").to_dict()
    config["paths"].update(
        {
            "data_dir": str(root / "data"),
            "cache_dir": str(root / "cache"),
            "output_dir": str(root / "outputs"),
            "weights_dir": str(root / "weights"),
        }
    )
    config["data"]["feature_grid"] = {
        "height": 3,
        "width": 3,
        "backbone_dim": 16,
        "projected_dim": 8,
    }
    config["data"]["synthetic"].update(
        {
            "train_identities": 2,
            "val_identities": 2,
            "test_identities": 2,
            "images_per_identity": 8,
            "sets_per_identity": 5,
        }
    )
    config["projection"]["max_fit_tokens"] = 128
    config["quantization"].update(
        {
            "codebook_size": 8,
            "max_fit_tokens": 256,
            "batch_size": 64,
            "iterations": 20,
            "n_init": 1,
        }
    )
    config["model"].update(
        {
            "hidden_dim": 32,
            "layers": 1,
            "attention_heads": 4,
            "mlp_ratio": 2.0,
        }
    )
    config["training"].update(
        {
            "batch_size": 2,
            "gradient_accumulation": 1,
            "min_steps": 1,
            "max_steps": 1,
            "warmup_steps": 0,
            "checkpoint_every_steps": 1,
            "precision": "fp32",
        }
    )
    config["runtime"].update(
        {
            "require_cuda": False,
            "num_workers": 0,
            "oom_batch_fallback": [2, 1],
            "oom_condition_frames_fallback": [4, 3, 2],
            "oom_hidden_dim_fallback": [32, 16, 8],
        }
    )
    validate_config(config)
    return config


class ModelAndDecodingTests(unittest.TestCase):
    def test_forward_shape_and_parameter_match(self) -> None:
        first = _small_model()
        second = _small_model()
        condition = torch.randn(2, 4, 9, 8)
        quality = torch.ones(2, 4)
        tokens = torch.full((2, 9), first.mask_token_id)
        logits = first(tokens, condition, quality)

        self.assertEqual(tuple(logits.shape), (2, 9, 16))
        self.assertEqual(first.parameter_count(), second.parameter_count())

    def test_condition_frame_permutation_is_invariant(self) -> None:
        torch.manual_seed(3)
        model = _small_model().eval()
        condition = torch.randn(2, 4, 9, 8)
        quality = torch.rand(2, 4) + 1.0
        tokens = torch.full((2, 9), model.mask_token_id)
        original = model(tokens, condition, quality)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = model(
            tokens,
            condition[:, permutation],
            quality[:, permutation],
        )
        torch.testing.assert_close(
            original,
            permuted,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_all_decode_modes_finish_without_masks(self) -> None:
        torch.manual_seed(4)
        model = _small_model().eval()
        condition = torch.randn(2, 4, 9, 8)
        quality = torch.rand(2, 4) + 1.0
        codebook = torch.randn(16, 8)
        for mode in (
            "confidence",
            "evidence-ordering",
            "evidence-logits",
            "evidence-remask",
        ):
            decoded = maskgit_decode(
                model,
                condition,
                quality,
                codebook=codebook,
                steps=4,
                mode=mode,
                top_k_frames=3,
                evidence_lambda=1.0,
            )
            self.assertFalse(
                bool((decoded.tokens == model.mask_token_id).any().item())
            )
            self.assertEqual(decoded.trace[-1]["actual_committed"], 9)
            self.assertFalse(decoded.commit_score.requires_grad)

    def test_evidence_top_three_rejects_one_conflicting_frame(self) -> None:
        condition = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        codebook = torch.eye(2)
        evidence = compute_local_evidence(
            condition,
            codebook,
            top_k_frames=3,
        )
        self.assertAlmostEqual(float(evidence[0, 0, 0]), 1.0, places=6)
        self.assertLess(float(evidence[0, 0, 1]), 0.34)
        self.assertFalse(evidence.requires_grad)

    def test_training_corruption_modes(self) -> None:
        targets = torch.arange(18).reshape(2, 9) % 8
        one_shot, one_shot_mask = build_corrupted_tokens(
            targets,
            mask_token_id=8,
            mode="one-shot",
            min_mask_ratio=0.05,
        )
        self.assertTrue(bool(one_shot_mask.all().item()))
        self.assertTrue(bool((one_shot == 8).all().item()))
        maskgit, maskgit_mask = build_corrupted_tokens(
            targets,
            mask_token_id=8,
            mode="maskgit",
            min_mask_ratio=0.05,
        )
        self.assertTrue(bool(maskgit_mask.any(dim=1).all().item()))
        self.assertTrue(torch.equal(maskgit[~maskgit_mask], targets[~maskgit_mask]))


class TrainingCheckpointTests(unittest.TestCase):
    def test_one_step_training_checkpoint_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _tiny_training_config(root)
            synthetic = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="data",
            )
            quantization = prepare_quantization_artifacts(
                config=config,
                profile="smoke",
                run_id="quant",
                dataset_root=synthetic.root,
            )
            dataset = TokenTrainingDataset(
                dataset_root=synthetic.root,
                quantization_root=quantization.root,
                split="train",
            )
            cached_dataset = TokenTrainingDataset(
                dataset_root=synthetic.root,
                quantization_root=quantization.root,
                split="train",
                precompute=True,
            )
            dynamic_sample = dataset[1]
            cached_sample = cached_dataset[1]
            torch.testing.assert_close(
                dynamic_sample["condition_features"],
                cached_sample["condition_features"],
            )
            torch.testing.assert_close(
                dynamic_sample["target_tokens"],
                cached_sample["target_tokens"],
            )
            model = ConditionalTokenTransformer(
                feature_dim=8,
                spatial_tokens=9,
                codebook_size=8,
                hidden_dim=32,
                layers=1,
                attention_heads=4,
                mlp_ratio=2.0,
                dropout=0.0,
            )
            first = train_single_model(
                model=model,
                dataset=dataset,
                config=config,
                mode="maskgit",
                device=torch.device("cpu"),
                checkpoint_dir=root / "checkpoints",
                batch_size=2,
                condition_frames=4,
                seed=17,
            )
            self.assertTrue(first.checkpoint.is_file())
            resumed_model = ConditionalTokenTransformer(
                feature_dim=8,
                spatial_tokens=9,
                codebook_size=8,
                hidden_dim=32,
                layers=1,
                attention_heads=4,
                mlp_ratio=2.0,
                dropout=0.0,
            )
            resumed = train_single_model(
                model=resumed_model,
                dataset=dataset,
                config=config,
                mode="maskgit",
                device=torch.device("cpu"),
                checkpoint_dir=root / "checkpoints",
                batch_size=2,
                condition_frames=4,
                seed=17,
                resume_checkpoint=first.checkpoint,
            )
            self.assertEqual(resumed.steps, 1)
            self.assertEqual(resumed.losses, first.losses)
            self.assertIsNotNone(resumed.resumed_from)


if __name__ == "__main__":
    unittest.main()
