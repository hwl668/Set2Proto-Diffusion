from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_p2_2_residual_stability import _IndexedTokenDataset


class IndexedTokenDatasetTests(unittest.TestCase):
    def test_metadata_and_tensors_follow_the_same_carved_order(self) -> None:
        base = SimpleNamespace()
        base._precomputed = {
            "sample_ids": ["s0", "s1", "s2"],
            "scenarios": ["clean", "wrong_identity", "low_quality"],
            "target_identity": torch.tensor([10, 11, 12]),
            "condition_features": torch.randn(3, 4, 49, 8),
            "condition_quality": torch.randn(3, 4),
            "condition_visibility": torch.ones(3, 4, 49),
        }
        base.targets = {
            "split": "train",
            "tokens": torch.arange(3 * 49).reshape(3, 49),
        }
        base.synthetic = SimpleNamespace(
            split="train",
            rows=[
                {"sample_id": "s0", "wrong_positions": []},
                {"sample_id": "s1", "wrong_positions": [2]},
                {"sample_id": "s2", "wrong_positions": []},
            ],
        )
        base.__len__ = lambda: 3

        class BaseAdapter:
            def __init__(self, value: SimpleNamespace) -> None:
                self.__dict__.update(value.__dict__)

            def __len__(self) -> int:
                return 3

        view = _IndexedTokenDataset(BaseAdapter(base), [2, 1])
        self.assertEqual(view.targets["split"], "val")
        self.assertEqual(view.synthetic.split, "val")
        self.assertEqual(
            [row["sample_id"] for row in view.synthetic.rows],
            ["s2", "s1"],
        )
        self.assertEqual(view[0]["sample_id"], "s2")
        self.assertEqual(view.synthetic.rows[1]["wrong_positions"], [2])
        self.assertTrue(torch.equal(view.targets["tokens"][0], base.targets["tokens"][2]))


if __name__ == "__main__":
    unittest.main()
