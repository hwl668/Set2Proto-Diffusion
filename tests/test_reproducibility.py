from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.reproducibility import (
    make_dataloader_generator,
    seed_everything,
)


def _rng_sample(seed: int) -> tuple[float, np.ndarray, torch.Tensor]:
    seed_everything(seed, deterministic=True)
    return random.random(), np.random.randn(4), torch.randn(4)


def _loader_order(seed: int) -> list[int]:
    dataset = TensorDataset(torch.arange(20))
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=make_dataloader_generator(seed),
    )
    return [int(value) for batch in loader for value in batch[0]]


class ReproducibilityTests(unittest.TestCase):
    def test_all_rngs_repeat_for_same_seed(self) -> None:
        first = _rng_sample(123)
        second = _rng_sample(123)

        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_different_seed_changes_torch_sequence(self) -> None:
        first = _rng_sample(123)
        second = _rng_sample(124)
        self.assertFalse(torch.equal(first[2], second[2]))

    def test_dataloader_shuffle_is_repeatable(self) -> None:
        self.assertEqual(_loader_order(77), _loader_order(77))
        self.assertNotEqual(_loader_order(77), _loader_order(78))

    def test_invalid_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seed_everything(-1)
        with self.assertRaises(ValueError):
            make_dataloader_generator(-1)


if __name__ == "__main__":
    unittest.main()

