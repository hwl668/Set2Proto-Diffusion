"""Train the P3-1 evidence-anchored one-shot and MaskGIT comparison."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_p2_2_residual_stability import _IndexedTokenDataset
from run_p2_5_listwise_identity_reranker import (
    HARD_SCENARIOS,
    _sha256,
    _write_csv,
    _write_json,
)
from run_p3_0_evidence_anchor_quantization import _load_sources
from run_pointer_route_rollout_rescue import CudaActivityMonitor
from set2proto.config import load_config
from set2proto.decoding import maskgit_decode, one_shot_decode
from set2proto.environment import collect_environment, config_sha256
from set2proto.evidence_anchor_maskgit import (
    EvidenceResidualDataset,
    build_evidence_anchored_model,
    build_evidence_residual_dataset,
    canonicalize_condition_set,
    evaluate_anchored_models,
    train_anchored_model,
)
from set2proto.experiment_logging import ExperimentLogger
from set2proto.reproducibility import seed_everything
from set2proto.residual_evidence import compute_two_level_residual_evidence
from set2proto.residual_quantization import (
    ResidualCodebook,
    reconstruct_from_residual_tokens,
)
from set2proto.run_paths import create_run_paths
from set2proto.scalar_evidence_router import (
    build_scalar_evidence_router,
    load_scalar_evidence_checkpoint,
)
from set2proto.training import TokenTrainingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    parser.add_argument("--profile", choices=("expanded",), default="expanded")
    parser.add_argument("--stage", choices=("smoke", "all"), default="all")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--absolute-quantization-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--quality-residual-root", type=Path, required=True)
    parser.add_argument("--p1-3-root", type=Path, required=True)
    parser.add_argument("--p3-0-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _load_locked_sources(
    *,
    absolute_root: Path,
    quality_root: Path,
    p1_root: Path,
    p3_root: Path,
) -> dict[str, Any]:
    sources = _load_sources(
        quality_root=quality_root,
        p1_3_root=p1_root,
    )
    expected_absolute = sources["absolute_quantization_root"].resolve()
    if absolute_root.resolve() != expected_absolute:
        raise RuntimeError(
            "requested absolute quantization root differs from the P3-0 source"
        )
    p3_paths = {
        "decision": p3_root / "artifacts" / "decision.json",
        "validation_lock": p3_root / "artifacts" / "validation_lock.json",
        "manifest": p3_root / "logs" / "run_manifest.json",
        "report": p3_root / "artifacts" / "REPORT.md",
    }
    for name, path in p3_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing P3-0 {name}: {path}")
    decision = json.loads(
        p3_paths["decision"].read_text(encoding="utf-8")
    )
    validation_lock = json.loads(
        p3_paths["validation_lock"].read_text(encoding="utf-8")
    )
    if decision.get("classification") != (
        "GO_EVIDENCE_ANCHORED_DISCRETE_RESIDUAL"
    ):
        raise RuntimeError("P3-0 did not authorize evidence-anchored modeling")
    if not validation_lock.get("test_authorized"):
        raise RuntimeError("P3-0 validation lock was not authorized")
    sources["p3_0_decision"] = decision
    sources["hashes"].update(
        {f"p3_0_{name}": _sha256(path) for name, path in p3_paths.items()}
    )
    return sources


def _target_diagnostics(
    dataset: EvidenceResidualDataset,
    codebook_size: int,
) -> dict[str, Any]:
    tokens = dataset.target_tokens.reshape(-1)
    counts = torch.bincount(tokens, minlength=codebook_size).float()
    probability = counts / counts.sum().clamp_min(1.0)
    nonzero = probability > 0
    perplexity = torch.exp(
        -(probability[nonzero] * probability[nonzero].log()).sum()
    )
    teacher = dataset.targets["continuous_prototypes"].float()
    anchor = dataset.evidence_anchors.float()
    anchor_norm = anchor.norm(dim=-1)
    return {
        "samples": len(dataset),
        "tokens": int(tokens.numel()),
        "shape": list(dataset.evidence_anchors.shape),
        "codebook_utilization": float(nonzero.float().mean().item()),
        "token_perplexity": float(perplexity.item()),
        "normalized_perplexity": float(perplexity.item() / codebook_size),
        "mean_teacher_residual_norm": float(
            (teacher - anchor).norm(dim=-1).mean().item()
        ),
        "anchor_norm_mean": float(anchor_norm.mean().item()),
        "anchor_norm_max_error": float(
            (anchor_norm - 1.0).abs().max().item()
        ),
    }


def _state_identical(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name].cpu(), right[name].cpu()) for name in left
    )


def _hard(result: Mapping[str, Any], method: str) -> float:
    return float(result["verification"][method]["hard_average"]["roc_auc"])


def _auc(
    result: Mapping[str, Any],
    method: str,
    scenario: str,
) -> float:
    return float(result["verification"][method][scenario]["roc_auc"])


def _candidate_row(
    result: Mapping[str, Any],
    mode: str,
    evidence_lambda: float,
) -> dict[str, Any]:
    guided = "maskgit_evidence_guided_4step"
    confidence = "maskgit_confidence_4step"
    one_shot = "one_shot"
    row: dict[str, Any] = {
        "evidence_mode": mode,
        "evidence_lambda": float(evidence_lambda),
        "anchor_hard_auc": _hard(result, "evidence_anchor"),
        "one_shot_hard_auc": _hard(result, one_shot),
        "confidence_4step_hard_auc": _hard(result, confidence),
        "guided_4step_hard_auc": _hard(result, guided),
        "maskgit_gain_over_one_shot": (
            _hard(result, confidence) - _hard(result, one_shot)
        ),
        "evidence_gain_over_confidence": (
            _hard(result, guided) - _hard(result, confidence)
        ),
        "guided_gain_over_one_shot": (
            _hard(result, guided) - _hard(result, one_shot)
        ),
        "guided_map_cosine": float(
            result["map_fidelity"][guided]["mean_map_cosine"]
        ),
        "guided_exact_token_accuracy": float(
            result["exact_token_accuracy"][guided]
        ),
    }
    for scenario in ("clean", *HARD_SCENARIOS):
        row[f"guided_{scenario}_auc"] = _auc(result, guided, scenario)
        row[f"guided_vs_one_shot_{scenario}_auc"] = (
            _auc(result, guided, scenario)
            - _auc(result, one_shot, scenario)
        )
    return row


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no validation candidates")
    best_auc = max(float(row["guided_4step_hard_auc"]) for row in candidates)
    tied = [
        row
        for row in candidates
        if best_auc - float(row["guided_4step_hard_auc"]) <= tolerance
    ]
    selected = max(
        tied,
        key=lambda row: (
            float(row["evidence_gain_over_confidence"]),
            float(row["guided_map_cosine"]),
            -float(row["evidence_lambda"]),
            str(row["evidence_mode"]) == "evidence-ordering",
        ),
    )
    return dict(selected)


def _validation_lock(
    *,
    selected: Mapping[str, Any],
    parameter_match: Mapping[str, Any],
    gpu_activity: Mapping[str, Any],
) -> dict[str, Any]:
    values = [
        float(value)
        for key, value in selected.items()
        if key.endswith("_auc") or key.endswith("_cosine")
    ]
    checks = {
        "metrics_finite": bool(values) and bool(np.isfinite(values).all()),
        "parameter_count_matched": bool(
            parameter_match["parameter_count_matched"]
        ),
        "initialization_exactly_matched": bool(
            parameter_match["initialization_exactly_matched"]
        ),
        "gpu_training_observed": (
            float(gpu_activity.get("max_utilization_percent", 0.0)) > 0.0
        ),
    }
    return {
        "selection_split": "val",
        "selected_evidence_mode": selected["evidence_mode"],
        "selected_evidence_lambda": selected["evidence_lambda"],
        "test_metrics_consulted": False,
        "test_authorized": all(checks.values()),
        "checks": checks,
    }


def _decision(
    *,
    test: Mapping[str, Any],
    selected: Mapping[str, Any],
    parameter_match: Mapping[str, Any],
    permutation: Mapping[str, Any],
    validation_lock: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["p3_1"]["gates"]
    anchor = "evidence_anchor"
    one_shot = "one_shot"
    confidence = "maskgit_confidence_4step"
    guided = "maskgit_evidence_guided_4step"
    one_shot_gain = _hard(test, one_shot) - _hard(test, anchor)
    maskgit_gain = _hard(test, confidence) - _hard(test, one_shot)
    evidence_gain = _hard(test, guided) - _hard(test, confidence)
    scenario_deltas = {
        scenario: _auc(test, guided, scenario)
        - _auc(test, one_shot, scenario)
        for scenario in ("clean", *HARD_SCENARIOS)
    }
    hard_wins = sum(
        scenario_deltas[scenario] > 0.0 for scenario in HARD_SCENARIOS
    )
    clean_drop = -scenario_deltas["clean"]
    checks = {
        "one_shot_improves_anchor": (
            one_shot_gain
            >= float(gates["min_one_shot_hard_auc_gain_over_anchor"])
        ),
        "four_step_improves_one_shot": (
            maskgit_gain >= float(gates["min_maskgit_gain_over_one_shot"])
        ),
        "evidence_improves_confidence": (
            evidence_gain
            >= float(gates["min_evidence_gain_over_confidence"])
        ),
        "hard_scenario_wins": (
            hard_wins
            >= int(gates["min_hard_scenario_wins_over_one_shot"])
        ),
        "clean_constraint": (
            clean_drop <= float(gates["max_clean_auc_drop"])
        ),
        "permutation_invariant": (
            float(permutation["maximum_numeric_difference"])
            <= float(gates["max_permutation_difference"])
        ),
        "parameter_count_matched": bool(
            parameter_match["parameter_count_matched"]
        ),
        "initialization_exactly_matched": bool(
            parameter_match["initialization_exactly_matched"]
        ),
        "test_loaded_after_validation_lock": bool(
            validation_lock["test_authorized"]
            and not validation_lock["test_metrics_consulted"]
        ),
    }
    go = all(checks.values())
    return {
        "classification": (
            "GO_EVIDENCE_ANCHORED_MASKGIT"
            if go
            else "NO_GO_P3_1_MASKGIT_ADVANTAGE"
        ),
        "checks": checks,
        "selected_on_validation": {
            "evidence_mode": selected["evidence_mode"],
            "evidence_lambda": selected["evidence_lambda"],
        },
        "test_one_shot_gain_over_anchor": one_shot_gain,
        "test_maskgit_gain_over_one_shot": maskgit_gain,
        "test_evidence_gain_over_confidence": evidence_gain,
        "test_guided_gain_over_one_shot": (
            _hard(test, guided) - _hard(test, one_shot)
        ),
        "test_scenario_auc_deltas_guided_vs_one_shot": scenario_deltas,
        "hard_scenario_wins": hard_wins,
        "clean_auc_drop": clean_drop,
        "recommendation": (
            "repeat across seeds and a fresh project-wide holdout"
            if go
            else (
                "retain the evidence anchor, but do not claim that the current "
                "masked discrete iteration adds value"
            )
        ),
        "claim_scope": (
            "mechanism feasibility on train-disjoint identities already "
            "observed by prior project stages; not statistical significance"
        ),
    }


def _core_rows(
    split: str,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, verification in result["verification"].items():
        metrics = verification["all"]
        rows.append(
            {
                "split": split,
                "method": method,
                "pooled_all_roc_auc": metrics["roc_auc"],
                "hard_macro_auc": verification["hard_average"]["roc_auc"],
                "eer": metrics["eer"],
                "tar_at_far_1e-2": metrics["tar_at_far"]["0.01"]["tar"],
                "tar_at_far_1e-3": metrics["tar_at_far"]["0.001"]["tar"],
                "rank1": metrics["rank1_accuracy"],
                "margin": metrics["mean_genuine_impostor_margin"],
                "map_cosine": result["map_fidelity"][method][
                    "mean_map_cosine"
                ],
                "exact_token_accuracy": result[
                    "exact_token_accuracy"
                ].get(method, ""),
            }
        )
    return rows


def _scenario_rows(
    split: str,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, verification in result["verification"].items():
        for scenario, metrics in verification.items():
            if scenario in {"all", "hard_average"}:
                continue
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "scenario": scenario,
                    "roc_auc": metrics["roc_auc"],
                    "eer": metrics["eer"],
                    "rank1": metrics["rank1_accuracy"],
                    "margin": metrics["mean_genuine_impostor_margin"],
                    "map_cosine": result["map_fidelity"][method][
                        "by_scenario"
                    ][scenario]["mean_map_cosine"],
                }
            )
    return rows


@torch.inference_mode()
def _permutation_check(
    *,
    router: torch.nn.Module,
    model: torch.nn.Module,
    dataset: EvidenceResidualDataset,
    codebook: torch.Tensor,
    evidence_mode: str,
    evidence_lambda: float,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    count = min(16, len(dataset))
    condition = dataset._precomputed["condition_features"][:count].to(device)
    quality = dataset._precomputed["condition_quality"][:count].to(device)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    condition_permuted = condition.index_select(1, permutation)
    quality_permuted = quality.index_select(1, permutation)
    condition, quality = canonicalize_condition_set(condition, quality)
    condition_permuted, quality_permuted = canonicalize_condition_set(
        condition_permuted,
        quality_permuted,
    )

    anchor = router(condition, quality)[0]
    anchor_permuted = router(condition_permuted, quality_permuted)[0]
    evidence = compute_two_level_residual_evidence(
        condition,
        quality,
        anchor,
        codebook,
    ).evidence
    evidence_permuted = compute_two_level_residual_evidence(
        condition_permuted,
        quality_permuted,
        anchor_permuted,
        codebook,
    ).evidence
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        original = maskgit_decode(
            model,
            condition,
            quality,
            codebook=codebook,
            steps=4,
            mode=evidence_mode,
            top_k_frames=int(config["p3_1"]["decoding"]["top_k_frames"]),
            evidence_lambda=evidence_lambda,
            evidence_override=evidence,
            evidence_anchor=anchor,
        ).tokens
        permuted = maskgit_decode(
            model,
            condition_permuted,
            quality_permuted,
            codebook=codebook,
            steps=4,
            mode=evidence_mode,
            top_k_frames=int(config["p3_1"]["decoding"]["top_k_frames"]),
            evidence_lambda=evidence_lambda,
            evidence_override=evidence_permuted,
            evidence_anchor=anchor_permuted,
        ).tokens
    original_map = reconstruct_from_residual_tokens(
        anchor,
        original,
        codebook,
    )
    permuted_map = reconstruct_from_residual_tokens(
        anchor_permuted,
        permuted,
        codebook,
    )
    values = {
        "anchor_max_abs_difference": float(
            (anchor - anchor_permuted).abs().max().item()
        ),
        "evidence_max_abs_difference": float(
            (evidence - evidence_permuted).abs().max().item()
        ),
        "output_map_max_abs_difference": float(
            (original_map - permuted_map).abs().max().item()
        ),
        "token_disagreement_fraction": float(
            (original != permuted).float().mean().item()
        ),
    }
    values["maximum_numeric_difference"] = max(
        values["anchor_max_abs_difference"],
        values["evidence_max_abs_difference"],
        values["output_map_max_abs_difference"],
    )
    values["permutation"] = permutation.cpu().tolist()
    return values


@torch.inference_mode()
def _latency(
    *,
    router: torch.nn.Module,
    one_shot_model: torch.nn.Module,
    maskgit_model: torch.nn.Module,
    dataset: EvidenceResidualDataset,
    codebook: torch.Tensor,
    evidence_mode: str,
    evidence_lambda: float,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    count = min(16, len(dataset))
    condition = dataset._precomputed["condition_features"][:count].to(device)
    quality = dataset._precomputed["condition_quality"][:count].to(device)
    top_k = int(config["p3_1"]["decoding"]["top_k_frames"])

    def one_shot_pipeline() -> None:
        ordered, ordered_quality = canonicalize_condition_set(
            condition,
            quality,
        )
        anchor = router(ordered, ordered_quality)[0]
        one_shot_decode(one_shot_model, ordered, ordered_quality, anchor)

    def confidence_pipeline() -> None:
        ordered, ordered_quality = canonicalize_condition_set(
            condition,
            quality,
        )
        anchor = router(ordered, ordered_quality)[0]
        maskgit_decode(
            maskgit_model,
            ordered,
            ordered_quality,
            codebook=codebook,
            steps=4,
            mode="confidence",
            top_k_frames=top_k,
            evidence_lambda=0.0,
            evidence_anchor=anchor,
        )

    def guided_pipeline() -> None:
        ordered, ordered_quality = canonicalize_condition_set(
            condition,
            quality,
        )
        anchor = router(ordered, ordered_quality)[0]
        evidence = compute_two_level_residual_evidence(
            ordered,
            ordered_quality,
            anchor,
            codebook,
        ).evidence
        maskgit_decode(
            maskgit_model,
            ordered,
            ordered_quality,
            codebook=codebook,
            steps=4,
            mode=evidence_mode,
            top_k_frames=top_k,
            evidence_lambda=evidence_lambda,
            evidence_override=evidence,
            evidence_anchor=anchor,
        )

    def measure(function: Callable[[], None]) -> dict[str, float]:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            for _ in range(5):
                function()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            values = []
            for _ in range(20):
                started = time.perf_counter()
                function()
                torch.cuda.synchronize(device)
                values.append((time.perf_counter() - started) * 1000.0)
        return {
            "mean_batch_ms": float(np.mean(values)),
            "p95_batch_ms": float(np.quantile(values, 0.95)),
            "mean_per_set_ms": float(np.mean(values) / count),
            "peak_reserved_mib": float(
                torch.cuda.max_memory_reserved(device) / 1024**2
            ),
        }

    return {
        "device": str(device),
        "batch_size": count,
        "cached_feature_protocol": True,
        "includes_frozen_anchor_router": True,
        "one_shot": measure(one_shot_pipeline),
        "maskgit_confidence_4step": measure(confidence_pipeline),
        "maskgit_evidence_guided_4step": measure(guided_pipeline),
    }


def _report(
    *,
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    decision: Mapping[str, Any],
    selected: Mapping[str, Any],
    training: Mapping[str, Any],
    latency: Mapping[str, Any],
    target_diagnostics: Mapping[str, Any],
) -> str:
    methods = (
        "evidence_anchor",
        "teacher_residual_quantized",
        "one_shot",
        "maskgit_confidence_1step",
        "maskgit_confidence_2step",
        "maskgit_confidence_4step",
        "maskgit_confidence_8step",
        "maskgit_evidence_guided_4step",
    )

    def table_row(result: Mapping[str, Any], method: str) -> str:
        metrics = result["verification"][method]
        return (
            f"| {method} | {metrics['all']['roc_auc']:.6f} | "
            f"{metrics['hard_average']['roc_auc']:.6f} | "
            f"{metrics['all']['eer']:.6f} | "
            f"{metrics['all']['rank1_accuracy']:.4f} | "
            f"{result['map_fidelity'][method]['mean_map_cosine']:.6f} | "
            f"{result['exact_token_accuracy'].get(method, float('nan')):.4f} |"
        )

    val_rows = "\n".join(table_row(validation, method) for method in methods)
    test_rows = "\n".join(table_row(test, method) for method in methods)
    scenario_lines = "\n".join(
        f"- {name}: {value:+.6f}"
        for name, value in decision[
            "test_scenario_auc_deltas_guided_vs_one_shot"
        ].items()
    )
    failed = [
        name for name, passed in decision["checks"].items() if not passed
    ]
    return f"""# P3-1 Evidence-Anchored Parameter-Matched MaskGIT

## Locked protocol

- Frozen P1-3 scalar evidence router at validation-selected step 550.
- Frozen train-only K=1024 quality-residual codebook; PCA, identities, S/T
  pairs and clean disjoint-teacher gallery are unchanged.
- One-shot and MaskGIT use exactly the same architecture, parameter count,
  initialization, optimization budget and training samples.
- Evidence mode and lambda were selected on validation only. Selected:
  `{selected['evidence_mode']}`, lambda={selected['evidence_lambda']}.
- Test artifacts were constructed only after the validation lock.

## Training

- Parameters per model: {training['parameter_count']:,}.
- Steps per model: {training['steps']:,}; effective batch:
  {training['effective_batch_size']}.
- One-shot / MaskGIT time:
  {training['one_shot_seconds']:.1f}s / {training['maskgit_seconds']:.1f}s.
- Peak reserved memory:
  {training['peak_reserved_mib']:.1f} MiB.
- Residual target utilization / normalized perplexity:
  {target_diagnostics['train']['codebook_utilization']:.4f} /
  {target_diagnostics['train']['normalized_perplexity']:.4f}.

## Validation

| Method | All AUC | Hard AUC | EER | Rank-1 | Map cosine | Token acc. |
|---|---:|---:|---:|---:|---:|---:|
{val_rows}

## Locked test

| Method | All AUC | Hard AUC | EER | Rank-1 | Map cosine | Token acc. |
|---|---:|---:|---:|---:|---:|---:|
{test_rows}

## Paired test conclusions

- One-shot minus evidence anchor hard AUC:
  {decision['test_one_shot_gain_over_anchor']:+.6f}.
- Confidence 4-step minus matched one-shot:
  {decision['test_maskgit_gain_over_one_shot']:+.6f}.
- Evidence-guided 4-step minus confidence-only 4-step:
  {decision['test_evidence_gain_over_confidence']:+.6f}.
- Evidence-guided 4-step minus matched one-shot:
  {decision['test_guided_gain_over_one_shot']:+.6f}.
- Hard-scenario wins over one-shot: {decision['hard_scenario_wins']}/4.

Scenario AUC deltas, guided 4-step minus one-shot:

{scenario_lines}

## Runtime and decision

- Cached-feature batch=16 latency including frozen anchor router:
  one-shot {latency['one_shot']['mean_batch_ms']:.3f} ms,
  confidence 4-step
  {latency['maskgit_confidence_4step']['mean_batch_ms']:.3f} ms,
  guided 4-step
  {latency['maskgit_evidence_guided_4step']['mean_batch_ms']:.3f} ms.
- Decision: **{decision['classification']}**.
- Failed gates: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

This is a mechanism result, not a statistical-significance claim: test
identities are disjoint from training identities but were observed by earlier
project stages.
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.profile).to_dict()
    base_seed = int(config["project"]["seed"])
    seed_everything(base_seed, deterministic=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    absolute_root = args.absolute_quantization_root.expanduser().resolve()
    quality_root = args.quality_residual_root.expanduser().resolve()
    p1_root = args.p1_3_root.expanduser().resolve()
    p3_root = args.p3_0_root.expanduser().resolve()
    sources = _load_locked_sources(
        absolute_root=absolute_root,
        quality_root=quality_root,
        p1_root=p1_root,
        p3_root=p3_root,
    )
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=base_seed,
        run_id=args.run_id,
    )
    logger = ExperimentLogger(run_paths.logs)
    environment = collect_environment(
        project_root=PROJECT_ROOT,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    preregistration = {
        "schema_version": 1,
        "stage": "p3-1-evidence-anchor-parameter-matched-maskgit",
        "mode": args.stage,
        "protocol": config["p3_1"],
        "source_hashes": sources["hashes"],
        "candidate_selection_split": "val",
        "test_loaded": False,
        "same_initialization_required": True,
        "same_parameter_count_required": True,
        "config_sha256": config_sha256(config),
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event("p3_1_started", mode=args.stage, test_loaded=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P3-1 formal comparison requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("P3-1 requires CUDA BF16 support")

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
    if codebook_object.codebook_size != int(config["p3_1"]["codebook_size"]):
        raise RuntimeError("residual codebook size differs from P3-1")
    codebook = codebook_object.vectors.float().to(device)

    smoke = args.stage == "smoke"
    with CudaActivityMonitor(interval_seconds=1.0) as monitor:
        train_base: Any = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=absolute_root,
            split="train",
            precompute=True,
        )
        if smoke:
            train_base = _IndexedTokenDataset(
                train_base,
                list(range(min(64, len(train_base)))),
            )
        train_dataset = build_evidence_residual_dataset(
            base=train_base,
            router=router,
            residual_codebook=codebook,
            device=device,
            batch_size=32,
        )
        train_target_diagnostics = _target_diagnostics(
            train_dataset,
            codebook_object.codebook_size,
        )

        initialization_seed = base_seed + 95001
        seed_everything(initialization_seed, deterministic=True)
        initialization_model = build_evidence_anchored_model(config)
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in initialization_model.state_dict().items()
        }
        initialization_path = (
            run_paths.artifacts / "matched_initialization.pt"
        )
        torch.save(
            {
                "schema_version": 1,
                "seed": initialization_seed,
                "model_state": initial_state,
            },
            initialization_path,
        )
        one_shot_model = build_evidence_anchored_model(config)
        maskgit_model = build_evidence_anchored_model(config)
        one_shot_model.load_state_dict(initial_state, strict=True)
        maskgit_model.load_state_dict(initial_state, strict=True)
        parameter_match = {
            "architecture": type(one_shot_model).__name__,
            "one_shot_parameter_count": one_shot_model.parameter_count(),
            "maskgit_parameter_count": maskgit_model.parameter_count(),
            "parameter_count_matched": (
                one_shot_model.parameter_count()
                == maskgit_model.parameter_count()
            ),
            "initialization_exactly_matched": _state_identical(
                one_shot_model.state_dict(),
                maskgit_model.state_dict(),
            ),
            "initialization_seed": initialization_seed,
            "initialization_sha256": _sha256(initialization_path),
            "same_training_sample_seed": initialization_seed + 17,
        }
        if not parameter_match["parameter_count_matched"]:
            raise RuntimeError("one-shot and MaskGIT parameter counts differ")
        if not parameter_match["initialization_exactly_matched"]:
            raise RuntimeError("one-shot and MaskGIT initial states differ")
        _write_json(
            run_paths.artifacts / "parameter_match.json",
            parameter_match,
        )

        training_seed = initialization_seed + 17
        one_shot_training = train_anchored_model(
            model=one_shot_model,
            dataset=train_dataset,
            config=config,
            mode="one-shot",
            device=device,
            checkpoint_directory=run_paths.checkpoints / "one_shot",
            seed=training_seed,
            smoke=smoke,
        )
        maskgit_training = train_anchored_model(
            model=maskgit_model,
            dataset=train_dataset,
            config=config,
            mode="maskgit",
            device=device,
            checkpoint_directory=run_paths.checkpoints / "maskgit",
            seed=training_seed,
            smoke=smoke,
        )
        _write_csv(
            run_paths.artifacts / "training_history_one_shot.csv",
            one_shot_training.history,
        )
        _write_csv(
            run_paths.artifacts / "training_history_maskgit.csv",
            maskgit_training.history,
        )
        steps = one_shot_training.steps
        training_summary = {
            "device": str(device),
            "precision": config["p3_1"]["training"]["precision"],
            "steps": steps,
            "batch_size": int(config["p3_1"]["training"]["batch_size"]),
            "gradient_accumulation": int(
                config["p3_1"]["training"]["gradient_accumulation"]
            ),
            "effective_batch_size": int(
                config["p3_1"]["training"]["batch_size"]
            )
            * int(config["p3_1"]["training"]["gradient_accumulation"]),
            "parameter_count": one_shot_model.parameter_count(),
            "one_shot_seconds": one_shot_training.elapsed_seconds,
            "maskgit_seconds": maskgit_training.elapsed_seconds,
            "one_shot_final_loss": one_shot_training.history[-1]["loss"],
            "maskgit_final_loss": maskgit_training.history[-1]["loss"],
            "one_shot_final_accuracy": one_shot_training.history[-1][
                "accuracy"
            ],
            "maskgit_final_accuracy": maskgit_training.history[-1][
                "accuracy"
            ],
            "peak_reserved_mib": max(
                one_shot_training.peak_reserved_mib,
                maskgit_training.peak_reserved_mib,
            ),
            "one_shot_checkpoint": str(one_shot_training.checkpoint),
            "maskgit_checkpoint": str(maskgit_training.checkpoint),
        }
        _write_json(
            run_paths.artifacts / "training_summary.json",
            training_summary,
        )

        val_base: Any = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=absolute_root,
            split="val",
            precompute=True,
        )
        if smoke:
            val_base = _IndexedTokenDataset(
                val_base,
                list(range(min(16, len(val_base)))),
            )
        validation_dataset = build_evidence_residual_dataset(
            base=val_base,
            router=router,
            residual_codebook=codebook,
            device=device,
            batch_size=32,
        )
        val_target_diagnostics = _target_diagnostics(
            validation_dataset,
            codebook_object.codebook_size,
        )
        target_diagnostics = {
            "train": train_target_diagnostics,
            "validation": val_target_diagnostics,
            "test_constructed": False,
        }
        _write_json(
            run_paths.artifacts / "target_diagnostics.json",
            target_diagnostics,
        )

        candidate_rows: list[dict[str, Any]] = []
        candidate_results: dict[tuple[str, float], dict[str, Any]] = {}
        modes = (
            [config["p3_1"]["decoding"]["evidence_modes"][0]]
            if smoke
            else config["p3_1"]["decoding"]["evidence_modes"]
        )
        lambdas = (
            [config["p3_1"]["decoding"]["evidence_lambda_candidates"][0]]
            if smoke
            else config["p3_1"]["decoding"]["evidence_lambda_candidates"]
        )
        for mode in modes:
            for evidence_lambda in lambdas:
                result = evaluate_anchored_models(
                    one_shot_model=one_shot_training.model,
                    maskgit_model=maskgit_training.model,
                    dataset=validation_dataset,
                    residual_codebook=codebook,
                    config=config,
                    device=device,
                    evidence_mode=str(mode),
                    evidence_lambda=float(evidence_lambda),
                )
                key = (str(mode), float(evidence_lambda))
                candidate_results[key] = result
                candidate_rows.append(
                    _candidate_row(result, key[0], key[1])
                )
        selected = _select_candidate(
            candidate_rows,
            float(config["p3_1"]["decoding"]["auc_tie_tolerance"]),
        )
        selected_key = (
            str(selected["evidence_mode"]),
            float(selected["evidence_lambda"]),
        )
        validation = candidate_results[selected_key]
        _write_csv(
            run_paths.artifacts / "calibration_candidates.csv",
            candidate_rows,
        )
        _write_json(
            run_paths.artifacts / "selection_lock.json",
            {
                "selection_split": "val",
                "selected": selected,
                "test_metrics_consulted": False,
            },
        )
        _write_json(
            run_paths.artifacts / "validation_evaluation.json",
            validation,
        )

    gpu_activity = monitor.summary()
    _write_json(run_paths.artifacts / "gpu_activity.json", gpu_activity)
    validation_lock = _validation_lock(
        selected=selected,
        parameter_match=parameter_match,
        gpu_activity=gpu_activity,
    )
    _write_json(
        run_paths.artifacts / "validation_lock.json",
        validation_lock,
    )
    if smoke:
        smoke_result = {
            "status": "passed",
            "device": str(device),
            "bf16": torch.cuda.is_bf16_supported(),
            "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "output_shape": list(
                validation_dataset.evidence_anchors.shape[1:]
            ),
            "parameter_match": parameter_match,
            "training": training_summary,
            "selection": selected,
            "validation_lock": validation_lock,
            "gpu_activity": gpu_activity,
            "test_loaded": False,
        }
        _write_json(
            run_paths.artifacts / "smoke_result.json",
            smoke_result,
        )
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p3-1-smoke",
                "environment": environment,
                "resolved_config": config,
                "protocol_preregistration": preregistration,
                "smoke": smoke_result,
                "test_loaded": False,
            }
        )
        print(json.dumps(smoke_result, ensure_ascii=False, indent=2))
        return 0
    if not validation_lock["test_authorized"]:
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p3-1-evidence-anchor-parameter-matched-maskgit",
                "environment": environment,
                "resolved_config": config,
                "validation_lock": validation_lock,
                "classification": "NO_GO_P3_1_VALIDATION_INTEGRITY",
                "test_loaded": False,
            }
        )
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "classification": "NO_GO_P3_1_VALIDATION_INTEGRITY",
                    "validation_lock": validation_lock,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    locked = json.loads(
        (run_paths.artifacts / "validation_lock.json").read_text(
            encoding="utf-8"
        )
    )
    if not locked["test_authorized"] or locked["test_metrics_consulted"]:
        raise RuntimeError("validation lock does not authorize test")
    with CudaActivityMonitor(interval_seconds=1.0) as test_monitor:
        test_base = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=absolute_root,
            split="test",
            precompute=True,
        )
        test_dataset = build_evidence_residual_dataset(
            base=test_base,
            router=router,
            residual_codebook=codebook,
            device=device,
            batch_size=32,
        )
        test_target_diagnostics = _target_diagnostics(
            test_dataset,
            codebook_object.codebook_size,
        )
        target_diagnostics["test"] = test_target_diagnostics
        target_diagnostics["test_constructed"] = True
        _write_json(
            run_paths.artifacts / "target_diagnostics.json",
            target_diagnostics,
        )
        test = evaluate_anchored_models(
            one_shot_model=one_shot_training.model,
            maskgit_model=maskgit_training.model,
            dataset=test_dataset,
            residual_codebook=codebook,
            config=config,
            device=device,
            evidence_mode=str(selected["evidence_mode"]),
            evidence_lambda=float(selected["evidence_lambda"]),
        )
        permutation = _permutation_check(
            router=router,
            model=maskgit_training.model,
            dataset=test_dataset,
            codebook=codebook,
            evidence_mode=str(selected["evidence_mode"]),
            evidence_lambda=float(selected["evidence_lambda"]),
            config=config,
            device=device,
        )
        latency = _latency(
            router=router,
            one_shot_model=one_shot_training.model,
            maskgit_model=maskgit_training.model,
            dataset=test_dataset,
            codebook=codebook,
            evidence_mode=str(selected["evidence_mode"]),
            evidence_lambda=float(selected["evidence_lambda"]),
            config=config,
            device=device,
        )
    test_gpu_activity = test_monitor.summary()
    decision = _decision(
        test=test,
        selected=selected,
        parameter_match=parameter_match,
        permutation=permutation,
        validation_lock=validation_lock,
        config=config,
    )
    _write_json(run_paths.artifacts / "test_evaluation.json", test)
    _write_json(run_paths.artifacts / "permutation_check.json", permutation)
    _write_json(run_paths.artifacts / "latency.json", latency)
    _write_json(
        run_paths.artifacts / "test_gpu_activity.json",
        test_gpu_activity,
    )
    _write_json(run_paths.artifacts / "decision.json", decision)
    _write_csv(
        run_paths.artifacts / "core_comparison.csv",
        _core_rows("val", validation) + _core_rows("test", test),
    )
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        _scenario_rows("val", validation)
        + _scenario_rows("test", test),
    )
    report = _report(
        validation=validation,
        test=test,
        decision=decision,
        selected=selected,
        training=training_summary,
        latency=latency,
        target_diagnostics=target_diagnostics,
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    logger.write_manifest(
        {
            "schema_version": 1,
            "stage": "p3-1-evidence-anchor-parameter-matched-maskgit",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "source_hashes": sources["hashes"],
            "parameter_match": parameter_match,
            "training": training_summary,
            "target_diagnostics": target_diagnostics,
            "validation_candidates": candidate_rows,
            "selection": selected,
            "validation_lock": validation_lock,
            "validation": validation,
            "test": test,
            "permutation": permutation,
            "latency": latency,
            "decision": decision,
            "gpu_activity": gpu_activity,
            "test_gpu_activity": test_gpu_activity,
            "test_loaded": True,
        }
    )
    logger.log_event(
        "p3_1_completed",
        classification=decision["classification"],
        test_loaded=True,
    )
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "selection": selected,
                "validation_lock": validation_lock,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
