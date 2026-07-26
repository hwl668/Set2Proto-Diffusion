from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.evidence_anchor_quantization import (
    AnchorInference,
    paired_residual_diagnostics,
)


class EvidenceAnchorQuantizationTests(unittest.TestCase):
    def test_paired_diagnostics_detect_smaller_evidence_residual(self) -> None:
        teacher = torch.zeros(2, 3, 4)
        teacher[..., 0] = 1.0
        quality = torch.zeros_like(teacher)
        quality[..., 1] = 1.0
        evidence = teacher.clone()
        result = paired_residual_diagnostics(
            AnchorInference(
                quality_anchor=quality,
                evidence_anchor=evidence,
                teacher=teacher,
                gates=torch.full((2, 3), 0.2),
                scenarios=["clean", "wrong_identity"],
                identities=torch.tensor([1, 2]),
            )
        )
        self.assertGreater(
            result["all"]["mean_residual_norm_reduction_fraction"],
            0.99,
        )
        self.assertEqual(
            result["all"]["evidence_anchor_better_fraction"],
            1.0,
        )

    def test_diagnostics_preserve_all_scenarios(self) -> None:
        torch.manual_seed(4)
        teacher = torch.nn.functional.normalize(
            torch.randn(4, 5, 6),
            dim=-1,
        )
        quality = torch.nn.functional.normalize(
            torch.randn(4, 5, 6),
            dim=-1,
        )
        result = paired_residual_diagnostics(
            AnchorInference(
                quality_anchor=quality,
                evidence_anchor=teacher,
                teacher=teacher,
                gates=torch.rand(4, 5),
                scenarios=["a", "a", "b", "b"],
                identities=torch.arange(4),
            )
        )
        self.assertEqual(set(result["by_scenario"]), {"a", "b"})
        self.assertEqual(set(result["gate"]["by_scenario"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
