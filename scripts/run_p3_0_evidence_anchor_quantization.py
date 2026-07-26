"""Validate evidence-routed anchors as the base for discrete residual modeling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

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
from run_pointer_route_rollout_rescue import CudaActivityMonitor
from set2proto.config import load_config
from set2proto.environment import collect_environment, config_sha256
from set2proto.evidence_anchor_quantization import (
    evaluate_quantized_anchors,
    infer_anchor_maps,
    measure_anchor_latency,
    paired_residual_diagnostics,
    permutation_check,
)
from set2proto.experiment_logging import ExperimentLogger
from set2proto.reproducibility import seed_everything
from set2proto.residual_quantization import (
    ResidualCodebook,
    fit_residual_codebook,
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
    parser.add_argument("--quality-residual-root", type=Path, required=True)
    parser.add_argument("--p1-3-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _load_sources(
    *,
    quality_root: Path,
    p1_3_root: Path,
) -> dict[str, Any]:
    quality_paths = {
        "manifest": quality_root / "manifest.json",
        "codebook": quality_root / "codebook.pt",
        "projection": quality_root / "projection.pt",
    }
    p1_paths = {
        "selection": p1_3_root / "artifacts" / "selection_lock.json",
        "protocol": (
            p1_3_root / "artifacts" / "protocol_preregistration.json"
        ),
        "checkpoint": (
            p1_3_root
            / "checkpoints"
            / "scalar_frozen"
            / "checkpoint_step_00000550.pt"
        ),
        "manifest": p1_3_root / "logs" / "run_manifest.json",
    }
    for name, path in {**quality_paths, **p1_paths}.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing P3-0 source {name}: {path}")
    quality_manifest = json.loads(
        quality_paths["manifest"].read_text(encoding="utf-8")
    )
    selection = json.loads(
        p1_paths["selection"].read_text(encoding="utf-8")
    )
    protocol = json.loads(
        p1_paths["protocol"].read_text(encoding="utf-8")
    )
    if quality_manifest.get("anchor") != "quality_pooling":
        raise RuntimeError("matched residual codebook has the wrong anchor")
    if int(quality_manifest["codebook"]["fit_tokens"]) != 245000:
        raise RuntimeError("matched codebook was not fit on all train tokens")
    if selection.get("chosen_recipe") != "scalar_frozen":
        raise RuntimeError("P1-3 source is not scalar_frozen")
    if int(selection.get("checkpoint_step", -1)) != 550:
        raise RuntimeError("P1-3 source is not validation-selected step 550")
    if _sha256(p1_paths["checkpoint"]) != selection["checkpoint_sha256"]:
        raise RuntimeError("P1-3 checkpoint hash differs from selection lock")
    projection_hash = _sha256(quality_paths["projection"])
    if (
        selection["source_hashes"]["projection"] != projection_hash
        or quality_manifest["source_projection_sha256"] != projection_hash
    ):
        raise RuntimeError("P1-3 and quality codebook projections differ")
    absolute_quantization_root = Path(
        quality_manifest["source_absolute_quantization"]
    ).expanduser().resolve()
    if not (absolute_quantization_root / "test_targets.pt").is_file():
        raise FileNotFoundError(
            "source absolute quantization has no test targets: "
            f"{absolute_quantization_root}"
        )
    if _sha256(
        absolute_quantization_root / "projection.pt"
    ) != projection_hash:
        raise RuntimeError("source absolute projection differs from residual")
    return {
        "quality_manifest": quality_manifest,
        "selection": selection,
        "protocol": protocol,
        "quality_codebook": quality_paths["codebook"],
        "absolute_quantization_root": absolute_quantization_root,
        "router_checkpoint": p1_paths["checkpoint"],
        "hashes": {
            f"quality_{name}": _sha256(path)
            for name, path in quality_paths.items()
        }
        | {
            f"p1_3_{name}": _sha256(path)
            for name, path in p1_paths.items()
        },
    }


def _hard(result: Mapping[str, Any], method: str) -> float:
    return float(
        result["verification"][method]["hard_average"]["roc_auc"]
    )


def _map_cosine(result: Mapping[str, Any], method: str) -> float:
    return float(result["map_fidelity"][method]["mean_map_cosine"])


def _validation_lock(
    *,
    train_diagnostics: Mapping[str, Any],
    validation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["p3_0"]["gates"]
    anchor_gain = (
        _hard(validation, "evidence_anchor")
        - _hard(validation, "quality_anchor")
    )
    map_gain = (
        _map_cosine(validation, "evidence_residual_quantized")
        - _map_cosine(validation, "quality_residual_quantized")
    )
    quantized_auc_delta = (
        _hard(validation, "evidence_residual_quantized")
        - _hard(validation, "quality_residual_quantized")
    )
    checks = {
        "train_residual_norm_reduction": (
            float(
                train_diagnostics["all"][
                    "mean_residual_norm_reduction_fraction"
                ]
            )
            >= float(gates["min_train_residual_norm_reduction"])
        ),
        "validation_anchor_identity_gain": (
            anchor_gain >= float(gates["min_val_anchor_hard_auc_gain"])
        ),
        "validation_quantized_map_gain": (
            map_gain
            >= float(gates["min_val_quantized_map_cosine_gain"])
        ),
        "validation_quantized_identity_constraint": (
            quantized_auc_delta
            >= -float(gates["max_val_quantized_hard_auc_drop"])
        ),
    }
    return {
        "selection_split": "val",
        "no_hyperparameter_search": True,
        "test_metrics_consulted": False,
        "test_authorized": all(checks.values()),
        "checks": checks,
        "anchor_hard_auc_gain": anchor_gain,
        "quantized_map_cosine_gain": map_gain,
        "quantized_hard_auc_delta": quantized_auc_delta,
    }


def _decision(
    *,
    train_diagnostics: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    permutation: Mapping[str, float],
    quality_codebook_fit: Mapping[str, Any],
    evidence_codebook_fit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["p3_0"]["gates"]
    test_anchor_gain = (
        _hard(test, "evidence_anchor") - _hard(test, "quality_anchor")
    )
    val_map_gain = (
        _map_cosine(validation, "evidence_residual_quantized")
        - _map_cosine(validation, "quality_residual_quantized")
    )
    test_map_gain = (
        _map_cosine(test, "evidence_residual_quantized")
        - _map_cosine(test, "quality_residual_quantized")
    )
    val_auc_delta = (
        _hard(validation, "evidence_residual_quantized")
        - _hard(validation, "quality_residual_quantized")
    )
    test_auc_delta = (
        _hard(test, "evidence_residual_quantized")
        - _hard(test, "quality_residual_quantized")
    )
    scenario_deltas = {
        scenario: (
            test["map_fidelity"]["evidence_residual_quantized"][
                "by_scenario"
            ][scenario]["mean_map_cosine"]
            - test["map_fidelity"]["quality_residual_quantized"][
                "by_scenario"
            ][scenario]["mean_map_cosine"]
        )
        for scenario in sorted(set(test["map_fidelity"][
            "evidence_residual_quantized"
        ]["by_scenario"]))
    }
    scenario_wins = sum(
        scenario_deltas[scenario] > 0.0 for scenario in HARD_SCENARIOS
    )
    maximum_permutation_difference = max(
        float(value) for value in permutation.values()
    )
    checks = {
        "train_residual_norm_reduction": (
            float(
                train_diagnostics["all"][
                    "mean_residual_norm_reduction_fraction"
                ]
            )
            >= float(gates["min_train_residual_norm_reduction"])
        ),
        "validation_anchor_gain": (
            _hard(validation, "evidence_anchor")
            - _hard(validation, "quality_anchor")
            >= float(gates["min_val_anchor_hard_auc_gain"])
        ),
        "test_anchor_gain": (
            test_anchor_gain >= float(gates["min_test_anchor_hard_auc_gain"])
        ),
        "validation_quantized_map_gain": (
            val_map_gain
            >= float(gates["min_val_quantized_map_cosine_gain"])
        ),
        "test_quantized_map_gain": (
            test_map_gain
            >= float(gates["min_test_quantized_map_cosine_gain"])
        ),
        "validation_quantized_identity_constraint": (
            val_auc_delta
            >= -float(gates["max_val_quantized_hard_auc_drop"])
        ),
        "test_quantized_identity_constraint": (
            test_auc_delta
            >= -float(gates["max_test_quantized_hard_auc_drop"])
        ),
        "scenario_map_cosine_wins": (
            scenario_wins >= int(gates["min_scenario_map_cosine_wins"])
        ),
        "permutation_invariant": (
            maximum_permutation_difference
            <= float(gates["max_permutation_difference"])
        ),
        "test_loaded_after_validation_lock": True,
    }
    passed = all(checks.values())
    quality_inertia = float(quality_codebook_fit["inertia"])
    evidence_inertia = float(evidence_codebook_fit["inertia"])
    return {
        "classification": (
            "GO_EVIDENCE_ANCHORED_DISCRETE_RESIDUAL"
            if passed
            else "NO_GO_P3_0"
        ),
        "checks": checks,
        "train_residual_norm_reduction_fraction": float(
            train_diagnostics["all"][
                "mean_residual_norm_reduction_fraction"
            ]
        ),
        "validation_anchor_hard_auc_gain": (
            _hard(validation, "evidence_anchor")
            - _hard(validation, "quality_anchor")
        ),
        "test_anchor_hard_auc_gain": test_anchor_gain,
        "validation_quantized_map_cosine_gain": val_map_gain,
        "test_quantized_map_cosine_gain": test_map_gain,
        "validation_quantized_hard_auc_delta": val_auc_delta,
        "test_quantized_hard_auc_delta": test_auc_delta,
        "test_transferred_codebook_map_cosine_gain": (
            _map_cosine(test, "evidence_anchor_quality_codebook")
            - _map_cosine(test, "quality_residual_quantized")
        ),
        "test_transferred_codebook_hard_auc_delta": (
            _hard(test, "evidence_anchor_quality_codebook")
            - _hard(test, "quality_residual_quantized")
        ),
        "codebook_inertia_reduction_fraction": (
            (quality_inertia - evidence_inertia) / quality_inertia
        ),
        "test_scenario_map_cosine_deltas": scenario_deltas,
        "hard_scenario_map_cosine_wins": scenario_wins,
        "maximum_permutation_difference": maximum_permutation_difference,
        "recommendation": (
            "train masked discrete residual denoising around the frozen "
            "evidence anchor"
            if passed
            else "do not train diffusion until anchor/codebook mismatch is fixed"
        ),
        "project_wide_holdout_caveat": (
            "The test identities are disjoint from train but were observed by "
            "earlier project stages; this is not a fresh significance claim."
        ),
    }


def _core_rows(
    split: str,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for method, verification in result["verification"].items():
        all_metrics = verification["all"]
        rows.append(
            {
                "split": split,
                "method": method,
                "pooled_all_auc": all_metrics["roc_auc"],
                "hard_macro_auc": verification["hard_average"]["roc_auc"],
                "eer": all_metrics["eer"],
                "tar_at_far_1e-2": all_metrics["tar_at_far"]["0.01"]["tar"],
                "tar_at_far_1e-3": all_metrics["tar_at_far"]["0.001"]["tar"],
                "rank1": all_metrics["rank1_accuracy"],
                "margin": all_metrics[
                    "mean_genuine_impostor_margin"
                ],
                "map_cosine": result["map_fidelity"][method][
                    "mean_map_cosine"
                ],
            }
        )
    return rows


def _scenario_rows(
    split: str,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for method, verification in result["verification"].items():
        for scenario in sorted(
            result["map_fidelity"][method]["by_scenario"]
        ):
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "scenario": scenario,
                    "roc_auc": verification[scenario]["roc_auc"],
                    "eer": verification[scenario]["eer"],
                    "rank1": verification[scenario]["rank1_accuracy"],
                    "margin": verification[scenario][
                        "mean_genuine_impostor_margin"
                    ],
                    "map_cosine": result["map_fidelity"][method][
                        "by_scenario"
                    ][scenario]["mean_map_cosine"],
                }
            )
    return rows


def _report(
    *,
    fit: Mapping[str, Any],
    train_diagnostics: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    decision: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    def row(split: Mapping[str, Any], method: str) -> str:
        verification = split["verification"][method]
        return (
            f"| {method} | {verification['all']['roc_auc']:.6f} | "
            f"{verification['hard_average']['roc_auc']:.6f} | "
            f"{verification['all']['eer']:.6f} | "
            f"{verification['all']['rank1_accuracy']:.4f} | "
            f"{split['map_fidelity'][method]['mean_map_cosine']:.6f} |"
        )

    failed = [
        key for key, value in decision["checks"].items() if not value
    ]
    methods = (
        "quality_anchor",
        "evidence_anchor",
        "quality_residual_quantized",
        "evidence_anchor_quality_codebook",
        "evidence_residual_quantized",
    )
    val_rows = "\n".join(row(validation, method) for method in methods)
    test_rows = "\n".join(row(test, method) for method in methods)
    return f"""# P3-0 Evidence-Anchored Discrete Residual Feasibility

## Protocol

- The validation-selected P1-3 scalar router is frozen.
- PCA, train/val/test identity split, condition sets and teachers are unchanged.
- The matched quality and evidence codebooks both use K=1024, all 245,000
  train tokens, seed 20260725, 100 iterations and three initializations.
- No architecture or hyperparameter is selected in this stage.
- Test is loaded only after the validation lock authorizes it.

## Train geometry

- Residual norm reduction:
  {decision['train_residual_norm_reduction_fraction']:+.6f}.
- Evidence codebook fit backend: {fit['backend']}.
- Fit inertia: {fit['inertia']:.3f}.

## Validation

| Method | All AUC | Hard AUC | EER | Rank-1 | Map cosine |
|---|---:|---:|---:|---:|---:|
{val_rows}

## Locked test

| Method | All AUC | Hard AUC | EER | Rank-1 | Map cosine |
|---|---:|---:|---:|---:|---:|
{test_rows}

## Paired gains

- Test continuous evidence-anchor hard AUC gain:
  {decision['test_anchor_hard_auc_gain']:+.6f}.
- Validation/test quantized map-cosine gain:
  {decision['validation_quantized_map_cosine_gain']:+.6f} /
  {decision['test_quantized_map_cosine_gain']:+.6f}.
- Validation/test quantized hard-AUC delta:
  {decision['validation_quantized_hard_auc_delta']:+.6f} /
  {decision['test_quantized_hard_auc_delta']:+.6f}.
- Without refitting the quality codebook, test map-cosine / hard-AUC gain:
  {decision['test_transferred_codebook_map_cosine_gain']:+.6f} /
  {decision['test_transferred_codebook_hard_auc_delta']:+.6f}.
- Matched K-means inertia reduction after evidence anchoring:
  {decision['codebook_inertia_reduction_fraction']:+.6f}.
- Hard-scenario map-cosine wins: {decision['hard_scenario_map_cosine_wins']}/4.

## Runtime and decision

- Frozen anchor latency: {latency['mean_batch_ms']:.3f} ms/batch16
  ({latency['mean_per_set_ms']:.3f} ms/set).
- Peak reserved memory: {latency['peak_reserved_mib']:.1f} MiB.
- Decision: **{decision['classification']}**.
- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

The test identities are train-disjoint but have been observed in earlier
project stages. These results support mechanism feasibility, not statistical
significance.
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.profile).to_dict()
    seed_everything(int(config["project"]["seed"]), deterministic=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    quality_root = args.quality_residual_root.expanduser().resolve()
    p1_root = args.p1_3_root.expanduser().resolve()
    sources = _load_sources(
        quality_root=quality_root,
        p1_3_root=p1_root,
    )
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=int(config["project"]["seed"]),
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
        "stage": "p3-0-evidence-anchor-residual-quantization",
        "mode": args.stage,
        "protocol": config["p3_0"],
        "source_hashes": sources["hashes"],
        "hyperparameter_search": False,
        "test_loaded": False,
        "config_sha256": config_sha256(config),
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event("p3_0_started", mode=args.stage, test_loaded=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P3-0 requires CUDA")
    router = build_scalar_evidence_router(config).to(device).eval()
    load_scalar_evidence_checkpoint(
        path=sources["router_checkpoint"],
        model=router,
        device=device,
    )
    quality_codebook = ResidualCodebook.from_payload(
        torch.load(
            sources["quality_codebook"],
            map_location="cpu",
            weights_only=True,
        )
    )
    train_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=sources["absolute_quantization_root"],
        split="train",
        precompute=True,
    )
    fit_dataset = (
        _IndexedTokenDataset(train_dataset, list(range(64)))
        if args.stage == "smoke"
        else train_dataset
    )
    with CudaActivityMonitor() as monitor:
        train_inference = infer_anchor_maps(
            dataset=fit_dataset,
            router=router,
            device=device,
            batch_size=int(config["p3_0"]["inference_batch_size"]),
        )
        train_diagnostics = paired_residual_diagnostics(train_inference)
        fit_iterations = (
            2 if args.stage == "smoke" else int(config["p3_0"]["iterations"])
        )
        evidence_codebook, fit = fit_residual_codebook(
            train_inference.teacher,
            train_inference.evidence_anchor,
            codebook_size=int(config["p3_0"]["codebook_size"]),
            max_fit_tokens=int(config["p3_0"]["max_fit_tokens"]),
            batch_size=int(config["p3_0"]["batch_size"]),
            iterations=fit_iterations,
            n_init=(
                1 if args.stage == "smoke" else int(config["p3_0"]["n_init"])
            ),
            seed=int(config["p3_0"]["seed"]),
            fit_split="train",
            device=config["p3_0"]["fit_device"],
        )
    checkpoint = run_paths.checkpoints / "evidence_residual_codebook.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(evidence_codebook.to_payload(), temporary)
    os.replace(temporary, checkpoint)
    _write_json(
        run_paths.artifacts / "train_residual_diagnostics.json",
        train_diagnostics,
    )
    _write_json(run_paths.artifacts / "codebook_fit.json", fit)
    if args.stage == "smoke":
        validation_dataset = _IndexedTokenDataset(
            TokenTrainingDataset(
                dataset_root=dataset_root,
                quantization_root=sources["absolute_quantization_root"],
                split="val",
                precompute=True,
            ),
            list(range(16)),
        )
        validation_inference = infer_anchor_maps(
            dataset=validation_dataset,
            router=router,
            device=device,
            batch_size=16,
        )
        smoke_evaluation = evaluate_quantized_anchors(
            inference=validation_inference,
            quality_codebook=quality_codebook,
            evidence_codebook=evidence_codebook,
            config=config,
            device=device,
        )
        smoke = {
            "status": "passed",
            "device": str(device),
            "train_samples": len(fit_dataset),
            "validation_samples": len(validation_dataset),
            "codebook_size": evidence_codebook.codebook_size,
            "train_residual_norm_reduction": train_diagnostics["all"][
                "mean_residual_norm_reduction_fraction"
            ],
            "evaluation_methods": list(smoke_evaluation["verification"]),
            "gpu_activity": monitor.summary(),
            "test_loaded": False,
        }
        _write_json(run_paths.artifacts / "smoke_result.json", smoke)
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p3-0-smoke",
                "environment": environment,
                "resolved_config": config,
                "smoke": smoke,
                "test_loaded": False,
            }
        )
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return 0

    validation_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=sources["absolute_quantization_root"],
        split="val",
        precompute=True,
    )
    validation_inference = infer_anchor_maps(
        dataset=validation_dataset,
        router=router,
        device=device,
        batch_size=int(config["p3_0"]["inference_batch_size"]),
    )
    validation = evaluate_quantized_anchors(
        inference=validation_inference,
        quality_codebook=quality_codebook,
        evidence_codebook=evidence_codebook,
        config=config,
        device=device,
    )
    validation_lock = _validation_lock(
        train_diagnostics=train_diagnostics,
        validation=validation,
        config=config,
    )
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(
        run_paths.artifacts / "validation_lock.json",
        validation_lock,
    )
    if not validation_lock["test_authorized"]:
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p3-0-evidence-anchor-residual-quantization",
                "environment": environment,
                "resolved_config": config,
                "validation_lock": validation_lock,
                "test_loaded": False,
                "classification": "NO_GO_P3_0_VALIDATION",
            }
        )
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "classification": "NO_GO_P3_0_VALIDATION",
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
    test_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=sources["absolute_quantization_root"],
        split="test",
        precompute=True,
    )
    test_inference = infer_anchor_maps(
        dataset=test_dataset,
        router=router,
        device=device,
        batch_size=int(config["p3_0"]["inference_batch_size"]),
    )
    test_diagnostics = paired_residual_diagnostics(test_inference)
    test = evaluate_quantized_anchors(
        inference=test_inference,
        quality_codebook=quality_codebook,
        evidence_codebook=evidence_codebook,
        config=config,
        device=device,
    )
    permutation = permutation_check(
        router=router,
        dataset=test_dataset,
        device=device,
    )
    latency = measure_anchor_latency(
        router=router,
        dataset=test_dataset,
        config=config,
        device=device,
    )
    decision = _decision(
        train_diagnostics=train_diagnostics,
        validation=validation,
        test=test,
        permutation=permutation,
        quality_codebook_fit=sources["quality_manifest"]["codebook"],
        evidence_codebook_fit=fit,
        config=config,
    )
    _write_json(run_paths.artifacts / "test_evaluation.json", test)
    _write_json(
        run_paths.artifacts / "test_residual_diagnostics.json",
        test_diagnostics,
    )
    _write_json(run_paths.artifacts / "permutation_check.json", permutation)
    _write_json(run_paths.artifacts / "latency.json", latency)
    _write_json(
        run_paths.artifacts / "gpu_activity.json",
        monitor.summary(),
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
        fit=fit,
        train_diagnostics=train_diagnostics,
        validation=validation,
        test=test,
        decision=decision,
        latency=latency,
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    logger.write_manifest(
        {
            "schema_version": 1,
            "stage": "p3-0-evidence-anchor-residual-quantization",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "source_hashes": sources["hashes"],
            "codebook_fit": fit,
            "validation_lock": validation_lock,
            "validation": validation,
            "test": test,
            "decision": decision,
            "latency": latency,
            "gpu_activity": monitor.summary(),
            "test_loaded": True,
        }
    )
    logger.log_event(
        "p3_0_completed",
        classification=decision["classification"],
        test_loaded=True,
    )
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
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
