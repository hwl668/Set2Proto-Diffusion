from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config, validate_config
from set2proto.quantization import (
    fit_pca_projection,
    prepare_quantization_artifacts,
    quantize_vectors,
    robust_teacher_pool,
)
from set2proto.synthetic import prepare_synthetic_dataset


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


def _tiny_config(root: Path) -> dict:
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
    validate_config(config)
    return config


class ProjectionAndTeacherTests(unittest.TestCase):
    def test_pca_projection_is_deterministic_and_normalized(self) -> None:
        generator = torch.Generator().manual_seed(5)
        features = torch.randn((8, 9, 16), generator=generator)
        first = fit_pca_projection(
            features,
            output_dim=8,
            max_fit_tokens=64,
            seed=11,
            svd_solver="randomized",
        )
        second = fit_pca_projection(
            features,
            output_dim=8,
            max_fit_tokens=64,
            seed=11,
            svd_solver="randomized",
        )
        torch.testing.assert_close(
            first.components,
            second.components,
            rtol=0,
            atol=0,
        )
        projected = first.transform(features)
        self.assertEqual(tuple(projected.shape), (8, 9, 8))
        torch.testing.assert_close(
            projected.norm(dim=-1),
            torch.ones((8, 9)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_robust_teacher_pool_rejects_local_outlier(self) -> None:
        consensus = torch.tensor([1.0, 0.0, 0.0])
        outlier = torch.tensor([0.0, 1.0, 0.0])
        features = torch.stack(
            [consensus, consensus, consensus, outlier],
            dim=0,
        )[:, None, :].expand(4, 3, 3)
        quality = torch.ones(4)
        pooled, diagnostics = robust_teacher_pool(
            features,
            quality,
            keep_fraction=0.75,
            consensus_weight=2.0,
            quality_weight=1.0,
        )
        expected = F.normalize(consensus, dim=0).expand(3, 3)
        torch.testing.assert_close(pooled, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(
            int(diagnostics["keep_mask"].sum(dim=-1).unique().item()),
            3,
        )

    def test_quantize_exact_codebook_vectors(self) -> None:
        codebook = torch.eye(4)
        vectors = codebook[[3, 1, 2]]
        tokens, cosine = quantize_vectors(vectors, codebook)
        torch.testing.assert_close(tokens, torch.tensor([3, 1, 2]))
        torch.testing.assert_close(cosine, torch.ones(3))


class QuantizationIntegrationTests(unittest.TestCase):
    def test_train_only_artifacts_and_targets_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _tiny_config(Path(temporary))
            dataset = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="dataset",
            )
            result = prepare_quantization_artifacts(
                config=config,
                profile="smoke",
                run_id="quant",
                dataset_root=dataset.root,
            )

            self.assertTrue(result.validation["passed"])
            self.assertEqual(result.manifest["fit_split"], "train")
            self.assertEqual(result.manifest["projection"]["output_dim"], 8)
            self.assertEqual(result.manifest["codebook"]["size"], 8)
            self.assertIn("mean_token_cosine", result.manifest["metrics"]["test"])
            targets = torch.load(
                result.root / "test_targets.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(tuple(targets["tokens"].shape), (10, 9))
            self.assertLess(int(targets["tokens"].max().item()), 8)

            with self.assertRaises(FileExistsError):
                prepare_quantization_artifacts(
                    config=config,
                    profile="smoke",
                    run_id="quant",
                    dataset_root=dataset.root,
                )
            resumed = prepare_quantization_artifacts(
                config=config,
                profile="smoke",
                run_id="quant",
                dataset_root=dataset.root,
                resume=True,
            )
            self.assertTrue(resumed.reused)
            self.assertTrue(resumed.validation["passed"])


if __name__ == "__main__":
    unittest.main()

