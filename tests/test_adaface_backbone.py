from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.adaface_backbone import (
    ARCFACE_TEMPLATE_112,
    AdaFaceIRBackbone,
    align_and_normalize_bgr,
    discover_spatial_hook,
    extract_named_spatial_and_embedding,
    extract_spatial_and_embedding,
    probe_named_spatial_hooks,
    run_body_suffix_to_spatial,
    similarity_transform,
)


class AdaFaceBackboneTests(unittest.TestCase):
    def test_auto_hook_discovers_pre_embedding_spatial_map(self) -> None:
        model = AdaFaceIRBackbone(num_layers=18).eval()
        hook = discover_spatial_hook(model)
        spatial, embedding, norm = extract_spatial_and_embedding(
            model,
            torch.zeros(1, 3, 112, 112),
            module_name=hook.module_name,
        )
        self.assertEqual(tuple(spatial.shape), (1, 512, 7, 7))
        self.assertEqual(tuple(embedding.shape), (1, 512))
        self.assertEqual(tuple(norm.shape), (1, 1))

    def test_named_parent_hook_suffix_exactly_replays_final_spatial(self) -> None:
        model = AdaFaceIRBackbone(num_layers=18).eval()
        hooks = probe_named_spatial_hooks(
            model,
            ("body.3", "body.5"),
        )
        self.assertEqual(hooks["body.3"].shape[1:], (128, 28, 28))
        self.assertEqual(hooks["body.5"].shape[1:], (256, 14, 14))
        captured, embedding, norm = extract_named_spatial_and_embedding(
            model,
            torch.randn((2, 3, 112, 112)),
            module_names=("body.3", "body.5", "output_layer.1"),
        )
        self.assertEqual(tuple(embedding.shape), (2, 512))
        self.assertEqual(tuple(norm.shape), (2, 1))
        for name in ("body.3", "body.5"):
            replay = run_body_suffix_to_spatial(
                model,
                captured[name],
                module_name=name,
            )
            torch.testing.assert_close(
                replay,
                captured["output_layer.1"],
                atol=0.0,
                rtol=0.0,
            )

    def test_similarity_transform_is_identity_for_template(self) -> None:
        matrix = similarity_transform(ARCFACE_TEMPLATE_112)
        np.testing.assert_allclose(
            matrix,
            np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            atol=1e-5,
        )

    def test_alignment_outputs_normalized_bgr_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            unicode_directory = Path(temporary) / "中文路径"
            unicode_directory.mkdir()
            path = unicode_directory / "face.jpg"
            image = np.zeros((112, 112, 3), dtype=np.uint8)
            image[:, :, 0] = 255
            encoded, buffer = cv2.imencode(".jpg", image)
            self.assertTrue(encoded)
            path.write_bytes(buffer.tobytes())
            tensor = align_and_normalize_bgr(
                path,
                ARCFACE_TEMPLATE_112,
            )
            self.assertEqual(tuple(tensor.shape), (3, 112, 112))
            self.assertGreater(float(tensor[0].mean()), 0.9)
            self.assertLess(float(tensor[2].mean()), -0.9)


if __name__ == "__main__":
    unittest.main()
