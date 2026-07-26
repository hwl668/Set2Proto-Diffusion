from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config, validate_config
from set2proto.synthetic import (
    SyntheticSetDataset,
    prepare_synthetic_dataset,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


def _tiny_config(root: Path) -> dict:
    config = load_config(CONFIG_PATH, "smoke").to_dict()
    config["paths"]["data_dir"] = str(root / "data")
    config["paths"]["cache_dir"] = str(root / "cache")
    config["paths"]["output_dir"] = str(root / "outputs")
    config["paths"]["weights_dir"] = str(root / "weights")
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
    config["quantization"]["codebook_size"] = 8
    validate_config(config)
    return config


class SyntheticPreparationTests(unittest.TestCase):
    def test_preparation_has_disjoint_identities_and_valid_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _tiny_config(Path(temporary))
            result = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="tiny-a",
            )

            self.assertTrue(result.validation["passed"])
            self.assertFalse(result.reused)
            self.assertTrue(result.manifest_path.is_file())
            self.assertEqual(
                result.manifest["splits"]["train"]["feature_shape"],
                [16, 9, 16],
            )
            self.assertEqual(result.manifest["splits"]["train"]["sets"], 10)

    def test_dataset_outputs_all_scenarios_and_expected_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _tiny_config(Path(temporary))
            result = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="tiny-b",
            )
            dataset = SyntheticSetDataset(result.root, "train")
            samples = {dataset[index]["scenario"]: dataset[index] for index in range(5)}

            self.assertEqual(set(samples), set(config["evaluation"]["scenarios"]))
            for sample in samples.values():
                self.assertEqual(
                    tuple(sample["condition_features"].shape),
                    (4, 9, 16),
                )
                self.assertEqual(
                    tuple(sample["teacher_features"].shape),
                    (4, 9, 16),
                )
                self.assertTrue(
                    set(sample["condition_indices"].tolist()).isdisjoint(
                        sample["teacher_indices"].tolist()
                    )
                )

            complementary = samples["complementary_occlusion"][
                "condition_visibility"
            ]
            self.assertTrue(bool((complementary.sum(dim=0) > 0).all().item()))
            common = samples["common_occlusion"]["condition_visibility"]
            self.assertTrue(
                torch.equal(common, common[0:1].expand_as(common))
            )
            wrong = samples["wrong_identity"]
            self.assertEqual(
                int(
                    (
                        wrong["condition_source_identity"]
                        != wrong["target_identity"]
                    )
                    .sum()
                    .item()
                ),
                1,
            )

    def test_self_teacher_mode_uses_clean_condition_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _tiny_config(Path(temporary))
            result = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="tiny-c",
            )
            dataset = SyntheticSetDataset(
                result.root,
                "train",
                teacher_mode="self",
            )
            sample = dataset[0]
            torch.testing.assert_close(
                sample["teacher_indices"],
                sample["condition_indices"],
            )
            self.assertEqual(sample["teacher_source"], "self")

    def test_same_seed_reproduces_feature_banks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _tiny_config(Path(temporary))
            first = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="repeat-a",
            )
            second = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="repeat-b",
            )
            first_bank = torch.load(
                first.root / "train_features.pt",
                map_location="cpu",
                weights_only=True,
            )
            second_bank = torch.load(
                second.root / "train_features.pt",
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(
                first_bank["features"],
                second_bank["features"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                first_bank["quality"],
                second_bank["quality"],
                rtol=0,
                atol=0,
            )

    def test_existing_dataset_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _tiny_config(Path(temporary))
            prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="resume-me",
            )
            with self.assertRaises(FileExistsError):
                prepare_synthetic_dataset(
                    config=config,
                    profile="smoke",
                    run_id="resume-me",
                )
            resumed = prepare_synthetic_dataset(
                config=config,
                profile="smoke",
                run_id="resume-me",
                resume=True,
            )
            self.assertTrue(resumed.reused)
            self.assertTrue(resumed.validation["passed"])


if __name__ == "__main__":
    unittest.main()
