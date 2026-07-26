"""Audit the post-hoc canonical frame-order numerical stability fix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_p2_2_residual_stability import _IndexedTokenDataset
from run_p2_5_listwise_identity_reranker import _write_json
from run_p3_0_evidence_anchor_quantization import _load_sources
from run_p3_1_evidence_anchor_maskgit import _permutation_check
from set2proto.config import load_config
from set2proto.evidence_anchor_maskgit import (
    build_evidence_anchored_model,
    build_evidence_residual_dataset,
)
from set2proto.residual_quantization import ResidualCodebook
from set2proto.scalar_evidence_router import (
    build_scalar_evidence_router,
    load_scalar_evidence_checkpoint,
)
from set2proto.training import TokenTrainingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    parser.add_argument("--profile", choices=("expanded",), default="expanded")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--quality-residual-root", type=Path, required=True)
    parser.add_argument("--p1-3-root", type=Path, required=True)
    parser.add_argument("--p3-1-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.profile).to_dict()
    p3_root = args.p3_1_root.expanduser().resolve()
    selection_path = p3_root / "artifacts" / "selection_lock.json"
    original_path = p3_root / "artifacts" / "permutation_check.json"
    maskgit_path = (
        p3_root
        / "checkpoints"
        / "maskgit"
        / "maskgit_00004000.pt"
    )
    for path in (selection_path, original_path, maskgit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))[
        "selected"
    ]
    original = json.loads(original_path.read_text(encoding="utf-8"))
    sources = _load_sources(
        quality_root=args.quality_residual_root.expanduser().resolve(),
        p1_3_root=args.p1_3_root.expanduser().resolve(),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("permutation audit requires CUDA")
    router = build_scalar_evidence_router(config).to(device).eval()
    load_scalar_evidence_checkpoint(
        path=sources["router_checkpoint"],
        model=router,
        device=device,
    )
    codebook_object = ResidualCodebook.from_payload(
        torch.load(
            sources["quality_codebook"],
            map_location="cpu",
            weights_only=True,
        )
    )
    codebook = codebook_object.vectors.float().to(device)
    base = TokenTrainingDataset(
        dataset_root=args.dataset_root.expanduser().resolve(),
        quantization_root=sources["absolute_quantization_root"],
        split="test",
        precompute=True,
    )
    subset = _IndexedTokenDataset(base, list(range(16)))
    dataset = build_evidence_residual_dataset(
        base=subset,
        router=router,
        residual_codebook=codebook,
        device=device,
        batch_size=16,
    )
    model = build_evidence_anchored_model(config).to(device).eval()
    checkpoint = torch.load(
        maskgit_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    fixed = _permutation_check(
        router=router,
        model=model,
        dataset=dataset,
        codebook=codebook,
        evidence_mode=str(selection["evidence_mode"]),
        evidence_lambda=float(selection["evidence_lambda"]),
        config=config,
        device=device,
    )
    result = {
        "schema_version": 1,
        "scope": (
            "post-hoc correctness audit only; locked validation/test metrics "
            "and decision are unchanged"
        ),
        "fix": "parameter-free deterministic content-based frame ordering",
        "checkpoint_retrained": False,
        "selection_changed": False,
        "test_metrics_recomputed": False,
        "before": original,
        "after": fixed,
        "gate_threshold": config["p3_1"]["gates"][
            "max_permutation_difference"
        ],
        "fixed_gate_passed": (
            float(fixed["maximum_numeric_difference"])
            <= float(
                config["p3_1"]["gates"]["max_permutation_difference"]
            )
            and float(fixed["token_disagreement_fraction"]) == 0.0
        ),
    }
    output = (
        p3_root / "artifacts" / "canonical_permutation_diagnostic.json"
    )
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
