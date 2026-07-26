from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.residual_quantization import (
    ResidualCodebook,
    compute_residual_quantization_metrics,
    compute_residuals,
    encode_residuals,
    encode_teacher_residuals,
    fit_residual_codebook,
    reconstruct_from_residual_tokens,
)


def _training_maps() -> tuple[torch.Tensor, torch.Tensor]:
    anchors = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    residuals = torch.tensor(
        [
            [[0.00, 0.20], [0.20, 0.00]],
            [[0.01, 0.19], [0.19, 0.01]],
            [[0.20, 0.00], [0.00, 0.20]],
            [[0.19, 0.01], [0.01, 0.19]],
        ]
    )
    teachers = anchors + residuals
    return teachers, anchors


class ResidualQuantizationTests(unittest.TestCase):
    def test_shapes_assignments_and_unit_reconstruction(self) -> None:
        teacher, anchor = _training_maps()
        residual = compute_residuals(teacher, anchor)
        self.assertEqual(tuple(residual.shape), (4, 2, 2))
        torch.testing.assert_close(residual, teacher - anchor)

        codebook = torch.tensor([[0.0, 0.2], [0.2, 0.0]])
        tokens, squared_distance = encode_residuals(residual, codebook)
        self.assertEqual(tuple(tokens.shape), (4, 2))
        self.assertEqual(tuple(squared_distance.shape), (4, 2))
        reconstructed = reconstruct_from_residual_tokens(
            anchor,
            tokens,
            codebook,
        )
        self.assertEqual(tuple(reconstructed.shape), (4, 2, 2))
        torch.testing.assert_close(
            reconstructed.norm(dim=-1),
            torch.ones((4, 2)),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_exact_centroids_reconstruct_normalized_teacher(self) -> None:
        anchor = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        residual = torch.tensor([[[0.0, 0.5], [0.5, 0.0]]])
        teacher = anchor + residual
        codebook = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
        tokens, distances = encode_teacher_residuals(
            teacher,
            anchor,
            codebook,
        )
        torch.testing.assert_close(distances, torch.zeros_like(distances))
        reconstructed = reconstruct_from_residual_tokens(
            anchor,
            tokens,
            codebook,
        )
        torch.testing.assert_close(
            reconstructed,
            F.normalize(teacher, dim=-1),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_fit_is_deterministic_and_centroids_keep_magnitude(self) -> None:
        teacher, anchor = _training_maps()
        arguments = {
            "codebook_size": 2,
            "max_fit_tokens": 6,
            "batch_size": 6,
            "iterations": 50,
            "n_init": 3,
            "seed": 17,
            "device": "cpu",
        }
        first, first_details = fit_residual_codebook(
            teacher,
            anchor,
            **arguments,
        )
        second, second_details = fit_residual_codebook(
            teacher,
            anchor,
            **arguments,
        )
        torch.testing.assert_close(
            first.vectors,
            second.vectors,
            atol=0,
            rtol=0,
        )
        self.assertEqual(first_details, second_details)
        self.assertEqual(first.fit_split, "train")
        self.assertEqual(first.fit_tokens, 6)
        self.assertEqual(first.available_train_tokens, 8)
        # Euclidean residual centroids are intentionally not unit-normalized.
        self.assertTrue(bool((first.vectors.norm(dim=-1) < 0.5).all().item()))

        restored = ResidualCodebook.from_payload(first.to_payload())
        torch.testing.assert_close(restored.vectors, first.vectors)
        self.assertEqual(restored.fit_split, "train")

    def test_fit_rejects_non_train_split(self) -> None:
        teacher, anchor = _training_maps()
        for split in ("val", "test"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "train-only"):
                    fit_residual_codebook(
                        teacher,
                        anchor,
                        codebook_size=2,
                        max_fit_tokens=8,
                        batch_size=8,
                        iterations=10,
                        n_init=1,
                        seed=3,
                        fit_split=split,
                        device="cpu",
                    )
        with self.assertRaisesRegex(ValueError, "train-only"):
            ResidualCodebook(
                vectors=torch.zeros((2, 2)),
                fit_tokens=2,
                available_train_tokens=2,
                seed=3,
                fit_split="val",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fit_backend_is_chunked_and_usable(self) -> None:
        teacher, anchor = _training_maps()
        codebook, details = fit_residual_codebook(
            teacher,
            anchor,
            codebook_size=2,
            max_fit_tokens=8,
            batch_size=4,
            iterations=2,
            n_init=1,
            seed=23,
            device="cuda",
        )
        self.assertEqual(details["backend"], "torch_minibatch_kmeans")
        self.assertTrue(details["device"].startswith("cuda"))
        tokens, distance = encode_teacher_residuals(
            teacher.cuda(),
            anchor.cuda(),
            codebook,
            chunk_size=2,
        )
        self.assertEqual(tuple(tokens.shape), (4, 2))
        self.assertTrue(bool(torch.isfinite(distance).all().item()))

    def test_metrics_report_usage_accuracy_and_scenarios(self) -> None:
        teacher, anchor = _training_maps()
        codebook = torch.tensor([[0.0, 0.2], [0.2, 0.0]])
        tokens, _ = encode_teacher_residuals(teacher, anchor, codebook)
        metrics = compute_residual_quantization_metrics(
            teacher_map=teacher,
            quality_anchor=anchor,
            tokens=tokens,
            codebook=codebook,
            reference_tokens=tokens.clone(),
            scenarios=["clean", "clean", "hard", "hard"],
        )
        self.assertEqual(metrics["used_codes"], 2)
        self.assertEqual(metrics["codebook_utilization"], 1.0)
        self.assertEqual(metrics["exact_token_accuracy"], 1.0)
        self.assertEqual(set(metrics["by_scenario"]), {"clean", "hard"})
        self.assertGreater(metrics["mean_token_cosine"], 0.99)
        self.assertEqual(len(metrics["token_counts"]), 2)


if __name__ == "__main__":
    unittest.main()
