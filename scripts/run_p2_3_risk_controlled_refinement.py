"""Train and validate the P2-3 risk-controlled residual correction gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_p2_2_residual_stability import _IndexedTokenDataset, _load_model
from run_p2_residual_evidence import _core_rows, _scenario_rows
from run_pointer_route_phase3 import _phase01_lock
from run_pointer_route_rollout_rescue import CudaActivityMonitor
from set2proto.config import load_config
from set2proto.environment import collect_environment, config_sha256
from set2proto.experiment_logging import ExperimentLogger
from set2proto.reproducibility import seed_everything
from set2proto.residual_correction_evaluation import (
    correction_permutation_check,
    decode_one_shot_dataset,
    decode_risk_controlled_dataset,
    evaluate_residual_token_predictions,
    measure_correction_latency,
)
from set2proto.residual_correction_gate import (
    CORRECTION_CLASSES,
    generate_correction_examples,
    train_correction_gate,
)
from set2proto.residual_quantization import ResidualCodebook
from set2proto.run_paths import create_run_paths
from set2proto.training import TokenTrainingDataset


HARD_SCENARIOS = (
    "low_quality",
    "complementary_occlusion",
    "common_occlusion",
    "wrong_identity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    parser.add_argument("--profile", choices=("expanded",), default="expanded")
    parser.add_argument("--stage", choices=("smoke", "all"), default="all")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--residual-artifact-root", type=Path, required=True)
    parser.add_argument("--phase01-root", type=Path, required=True)
    parser.add_argument("--p2-2-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_p2_2_lock(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    required = {
        "protocol": artifacts / "protocol_preregistration.json",
        "checkpoint_selection": artifacts / "checkpoint_selection.json",
        "training": artifacts / "training_summary.json",
        "decision": artifacts / "decision.json",
        "manifest": root / "logs" / "run_manifest.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing P2-2 {name}: {path}")
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
    }
    protocol = values["protocol"]
    manifest = values["manifest"]
    if protocol.get("test_loaded") is not False:
        raise RuntimeError("P2-2 protocol loaded test")
    if manifest.get("test_loaded") is not False:
        raise RuntimeError("P2-2 manifest loaded test")
    step = int(
        values["checkpoint_selection"]["selected_candidate"]["step"]
    )
    checkpoints = {
        "matched_one_shot": Path(
            values["training"]["one_shot_control"]["checkpoints"][str(step)]
        ).expanduser().resolve(),
        "rollout": Path(
            values["training"]["rollout_maskgit"]["checkpoints"][str(step)]
        ).expanduser().resolve(),
    }
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing P2-2 {name} checkpoint: {path}")
    values["selected_step"] = step
    values["checkpoints"] = checkpoints
    values["checkpoint_hashes"] = {
        name: _sha256(path) for name, path in checkpoints.items()
    }
    values["artifact_hashes"] = {
        name: _sha256(path) for name, path in required.items()
    }
    return values


def _candidate_row(
    *,
    proposal_fraction: float,
    utility_threshold: float,
    rounds: int,
    evaluation: Mapping[str, Any],
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["verification"]
    matched = methods["matched_one_shot"]["hard_average"]["roc_auc"]
    rollout = methods["rollout_one_shot"]["hard_average"]["roc_auc"]
    refined = methods["risk_controlled_refinement"]["hard_average"]["roc_auc"]
    transition = decoded["transitions"]["all"]
    return {
        "proposal_fraction": proposal_fraction,
        "utility_threshold": utility_threshold,
        "rounds": rounds,
        "matched_one_shot_hard_auc": matched,
        "rollout_one_shot_hard_auc": rollout,
        "refined_hard_auc": refined,
        "gain_over_matched_one_shot": refined - matched,
        "gain_over_rollout_one_shot": refined - rollout,
        "refined_all_auc": methods["risk_controlled_refinement"]["all"][
            "roc_auc"
        ],
        "net_correction_fraction": transition["net_correction_fraction"],
        "ecr": transition["ecr"],
        "eir": transition["eir"],
        "change_fraction": transition["change_fraction"],
        "replacement_precision": transition["replacement_precision"],
    }


def _select_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    best = max(float(row["refined_hard_auc"]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if float(row["refined_hard_auc"]) >= best - tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["gain_over_matched_one_shot"]),
            float(row["net_correction_fraction"]),
            -int(row["rounds"]),
            -float(row["proposal_fraction"]),
            float(row["utility_threshold"]),
        ),
    )
    return {
        "selection_split": "route_calibration",
        "objective": (
            "maximize refined hard AUC; within tolerance maximize gain over "
            "matched one-shot and net correction, then minimize rounds and "
            "proposal fraction and prefer the safer threshold"
        ),
        "auc_tie_tolerance": tolerance,
        "best_hard_auc": best,
        "eligible_candidates": len(eligible),
        "selected_candidate": selected,
        "validation_used": False,
        "test_used": False,
    }


def _decision(
    *,
    evaluation: Mapping[str, Any],
    decoded: Mapping[str, Any],
    selection: Mapping[str, Any],
    permutation: Mapping[str, Any],
    latency: Mapping[str, Any],
    transformer_hash_before: str,
    transformer_hash_after: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["verification"]
    matched = methods["matched_one_shot"]
    rollout = methods["rollout_one_shot"]
    refined = methods["risk_controlled_refinement"]
    matched_hard = matched["hard_average"]["roc_auc"]
    rollout_hard = rollout["hard_average"]["roc_auc"]
    refined_hard = refined["hard_average"]["roc_auc"]
    scenario_deltas = {
        scenario: (
            refined[scenario]["roc_auc"] - matched[scenario]["roc_auc"]
        )
        for scenario in HARD_SCENARIOS
    }
    scenario_wins = sum(value > 1e-12 for value in scenario_deltas.values())
    transition = decoded["transitions"]["all"]
    gates = config["p2_3"]["gates"]
    maximum_permutation_difference = max(
        float(permutation["initial_token_disagreement"]),
        float(permutation["final_token_disagreement"]),
        float(permutation["max_utility_difference"]),
    )
    peak = max(
        float(latency["rollout_one_shot"]["peak_reserved_mib"]),
        float(latency["refined"]["peak_reserved_mib"]),
    )
    checks = {
        "gate_trained_on_route_train_only": True,
        "selection_on_route_calibration_only": (
            selection["validation_used"] is False
            and selection["test_used"] is False
        ),
        "transformer_frozen": transformer_hash_before == transformer_hash_after,
        "beats_rollout_one_shot": (
            refined_hard - rollout_hard
            >= float(gates["min_hard_auc_gain_over_rollout_one_shot"])
        ),
        "beats_matched_one_shot": (
            refined_hard - matched_hard
            >= float(gates["min_hard_auc_gain_over_matched_one_shot"])
        ),
        "hard_scenario_wins": (
            scenario_wins >= int(gates["min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            matched["clean"]["roc_auc"] - refined["clean"]["roc_auc"]
            <= float(gates["max_clean_auc_drop"])
        ),
        "positive_net_correction": (
            transition["net_correction_fraction"] > 0.0
            if bool(gates["require_positive_net_correction"])
            else True
        ),
        "permutation_invariant": (
            maximum_permutation_difference
            <= float(gates["max_permutation_difference"])
        ),
        "memory_within_7_2_gib": (
            peak <= float(config["runtime"]["max_peak_memory_gib"]) * 1024
        ),
        "test_not_loaded": True,
    }
    passed = all(checks.values())
    return {
        "classification": (
            "GO_RISK_CONTROLLED_REFINEMENT" if passed else "NO_GO_P2_3"
        ),
        "checks": checks,
        "matched_one_shot_hard_auc": matched_hard,
        "rollout_one_shot_hard_auc": rollout_hard,
        "refined_hard_auc": refined_hard,
        "gain_over_matched_one_shot": refined_hard - matched_hard,
        "gain_over_rollout_one_shot": refined_hard - rollout_hard,
        "scenario_deltas_vs_matched_one_shot": scenario_deltas,
        "hard_scenario_wins": scenario_wins,
        "transition": transition,
        "maximum_permutation_difference": maximum_permutation_difference,
        "peak_reserved_mib": peak,
        "recommendation": (
            "freeze_refinement_and_create_new_identity_holdout_before_d3pm"
            if passed
            else "do_not_expand_to_d3pm; inspect proposal and gate errors"
        ),
    }


def _report(
    *,
    examples: Mapping[str, Any],
    training: Mapping[str, Any],
    selection: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    decoded: Mapping[str, Any],
    decision: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    methods = evaluation["verification"]

    def hard(name: str) -> float:
        return float(methods[name]["hard_average"]["roc_auc"])

    scenario_rows = "\n".join(
        f"| {scenario} | "
        f"{methods['matched_one_shot'][scenario]['roc_auc']:.6f} | "
        f"{methods['risk_controlled_refinement'][scenario]['roc_auc']:.6f} | "
        f"{decision['scenario_deltas_vs_matched_one_shot'][scenario]:+.6f} |"
        for scenario in HARD_SCENARIOS
    )
    selected = selection["selected_candidate"]
    failed = [
        key for key, value in decision["checks"].items() if not value
    ]
    transition = decoded["transitions"]["all"]
    return f"""# P2-3 Risk-Controlled Residual Refinement

## Locked protocol

- The P2-2 selected rollout Transformer is frozen byte-for-byte.
- Correction-gate examples and labels use route-train identities only.
- Gate labels are neutral / correction / injury under the residual target.
- Decoding starts from rollout one-shot tokens and only accepts proposals when
  `P(correction) - P(injury)` exceeds a calibrated threshold.
- Proposal fraction, threshold, and rounds are selected on route-calibration.
  Route-validation is evaluated once after locking. Test is not loaded.

## Gate training

- Examples: {examples['examples']}.
- Class counts ({', '.join(CORRECTION_CLASSES)}):
  {examples['class_counts']}.
- Gate parameters: {training['parameter_count']}.
- Training time: {training['elapsed_seconds']:.3f} seconds.
- Peak reserved memory: {training['peak_reserved_mib']:.1f} MiB.

## Calibration selection

- Proposal fraction: {selected['proposal_fraction']}.
- Utility threshold: {selected['utility_threshold']}.
- Refinement rounds: {selected['rounds']}.
- Calibration hard AUC: {selected['refined_hard_auc']:.6f}.
- Calibration gain over matched one-shot:
  {selected['gain_over_matched_one_shot']:+.6f}.

## Route-validation

| Method | Hard AUC |
|---|---:|
| Quality anchor | {hard('quality_anchor'):.6f} |
| Residual quantization oracle | {hard('teacher_residual_quantized'):.6f} |
| Matched one-shot control | {hard('matched_one_shot'):.6f} |
| Rollout-model one-shot | {hard('rollout_one_shot'):.6f} |
| Risk-controlled refinement | {hard('risk_controlled_refinement'):.6f} |

| Scenario | Matched one-shot | Refined | Delta |
|---|---:|---:|---:|
{scenario_rows}

## Transition diagnostics

- Changed fraction: {transition['change_fraction']:.6f}.
- Corrected / injured fraction:
  {transition['corrected_fraction']:.6f} /
  {transition['injured_fraction']:.6f}.
- ECR / EIR:
  {transition['ecr']:.6f} /
  {transition['eir']:.6f}.
- Net correction fraction: {transition['net_correction_fraction']:+.6f}.
- Replacement precision: {transition['replacement_precision']:.6f}.

## Runtime

- Rollout one-shot: {latency['rollout_one_shot']['mean_batch_ms']:.3f}
  ms/batch16.
- Refined: {latency['refined']['mean_batch_ms']:.3f} ms/batch16.

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Gain over rollout one-shot:
  {decision['gain_over_rollout_one_shot']:+.6f}.
- Gain over matched one-shot:
  {decision['gain_over_matched_one_shot']:+.6f}.
- Recommendation: `{decision['recommendation']}`.

This is a development result on the existing route-validation split, not a
statistical-significance claim or a new untouched holdout result.
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.profile).to_dict()
    seed = int(config["project"]["seed"])
    seed_everything(seed, deterministic=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    residual_root = args.residual_artifact_root.expanduser().resolve()
    p2_2_root = args.p2_2_root.expanduser().resolve()
    phase01 = _phase01_lock(
        root=args.phase01_root.expanduser().resolve(),
        dataset_root=dataset_root,
        quantization_root=Path(
            json.loads(
                (
                    residual_root / "manifest.json"
                ).read_text(encoding="utf-8")
            )["source_absolute_quantization"]
        ),
    )
    p2_2 = _load_p2_2_lock(p2_2_root)
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
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
        "stage": "p2-3-risk-controlled-residual-refinement",
        "mode": args.stage,
        "protocol": config["p2_3"],
        "p2_2_selected_step": p2_2["selected_step"],
        "p2_2_checkpoint_hashes": p2_2["checkpoint_hashes"],
        "route_split_hashes": phase01["artifact_hashes"],
        "selection_split": "route_calibration",
        "evaluation_split": (
            None if args.stage == "smoke" else "route_validation"
        ),
        "official_val_loaded": False,
        "test_loaded": False,
        "config_sha256": config_sha256(config),
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "p2_3_started",
        mode=args.stage,
        selected_p2_2_step=p2_2["selected_step"],
        test_loaded=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P2-3 requires CUDA")
    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=residual_root,
        split="train",
        precompute=True,
    )
    route_indices = phase01["split"]["sample_indices"]
    codebook = ResidualCodebook.from_payload(
        torch.load(
            residual_root / "codebook.pt",
            map_location="cpu",
            weights_only=True,
        )
    ).vectors
    rollout_model = _load_model(
        config=config,
        checkpoint=p2_2["checkpoints"]["rollout"],
        device=device,
    )
    for parameter in rollout_model.parameters():
        parameter.requires_grad_(False)
    transformer_hash_before = _sha256(p2_2["checkpoints"]["rollout"])
    examples = generate_correction_examples(
        model=rollout_model,
        dataset=dataset,
        indices=route_indices["route_train"],
        residual_codebook=codebook,
        config=config,
        device=device,
        seed=seed + 83001,
        smoke=args.stage == "smoke",
    )
    _write_json(
        run_paths.artifacts / "gate_example_diagnostics.json",
        examples.diagnostics,
    )
    with CudaActivityMonitor() as monitor:
        training_result = train_correction_gate(
            examples=examples,
            config=config,
            device=device,
            seed=seed + 83002,
            smoke=args.stage == "smoke",
        )
    gate = training_result.gate
    training = {
        "steps": len(training_result.history),
        "parameter_count": gate.parameter_count(),
        "elapsed_seconds": training_result.elapsed_seconds,
        "peak_reserved_mib": training_result.peak_reserved_mib,
        "class_counts": training_result.class_counts,
        "class_weights": training_result.class_weights,
        "device": str(device),
        "transformer_trainable_parameters": sum(
            parameter.numel()
            for parameter in rollout_model.parameters()
            if parameter.requires_grad
        ),
    }
    _write_json(run_paths.artifacts / "gate_training_summary.json", training)
    _write_csv(
        run_paths.artifacts / "gate_training_history.csv",
        training_result.history,
    )
    gate_checkpoint = run_paths.checkpoints / "correction_gate.pt"
    temporary = gate_checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "stage": "p2-3-risk-controlled-residual-refinement",
            "model_state": gate.state_dict(),
            "feature_names": examples.diagnostics["feature_names"],
            "parameter_count": gate.parameter_count(),
            "p2_2_checkpoint_hash": transformer_hash_before,
        },
        temporary,
    )
    os.replace(temporary, gate_checkpoint)
    calibration_dataset = _IndexedTokenDataset(
        dataset,
        route_indices["route_calibration"],
    )
    if args.stage == "smoke":
        result = decode_risk_controlled_dataset(
            model=rollout_model,
            gate=gate,
            dataset=_IndexedTokenDataset(
                dataset,
                route_indices["route_calibration"][:16],
            ),
            residual_codebook=codebook,
            config=config,
            device=device,
            proposal_fraction=0.2,
            utility_threshold=0.1,
            rounds=2,
        )
        smoke_result = {
            "status": "passed",
            "device": str(device),
            "examples": examples.diagnostics,
            "training": training,
            "decode_shape": list(result["tokens"].shape),
            "transition": result["transitions"]["all"],
            "gpu_activity": monitor.summary(),
            "test_loaded": False,
        }
        _write_json(run_paths.artifacts / "smoke_result.json", smoke_result)
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p2-3-smoke",
                "status": "passed",
                "environment": environment,
                "resolved_config": config,
                "protocol_preregistration": preregistration,
                "smoke": smoke_result,
                "test_loaded": False,
            }
        )
        print(json.dumps(smoke_result, ensure_ascii=False, indent=2))
        return 0

    matched_model = _load_model(
        config=config,
        checkpoint=p2_2["checkpoints"]["matched_one_shot"],
        device=device,
    )
    matched_calibration_tokens = decode_one_shot_dataset(
        model=matched_model,
        dataset=calibration_dataset,
        device=device,
        batch_size=int(config["evaluation"]["batch_size"]),
    )
    calibration_rows: list[dict[str, Any]] = []
    calibration_evaluations: dict[str, Any] = {}
    for proposal_fraction in config["p2_3"]["calibration"][
        "proposal_fraction_candidates"
    ]:
        for utility_threshold in config["p2_3"]["calibration"][
            "utility_threshold_candidates"
        ]:
            for rounds in config["p2_3"]["calibration"]["round_candidates"]:
                decoded = decode_risk_controlled_dataset(
                    model=rollout_model,
                    gate=gate,
                    dataset=calibration_dataset,
                    residual_codebook=codebook,
                    config=config,
                    device=device,
                    proposal_fraction=float(proposal_fraction),
                    utility_threshold=float(utility_threshold),
                    rounds=int(rounds),
                )
                evaluation = evaluate_residual_token_predictions(
                    dataset=calibration_dataset,
                    residual_codebook=codebook,
                    predictions={
                        "matched_one_shot": matched_calibration_tokens,
                        "rollout_one_shot": decoded["initial_tokens"],
                        "risk_controlled_refinement": decoded["tokens"],
                    },
                    config=config,
                    device=device,
                )
                row = _candidate_row(
                    proposal_fraction=float(proposal_fraction),
                    utility_threshold=float(utility_threshold),
                    rounds=int(rounds),
                    evaluation=evaluation,
                    decoded=decoded,
                )
                calibration_rows.append(row)
                key = (
                    f"f{float(proposal_fraction):g}_"
                    f"t{float(utility_threshold):g}_r{int(rounds)}"
                )
                calibration_evaluations[key] = {
                    "candidate": row,
                    "transitions": decoded["transitions"],
                }
    selection = _select_candidate(
        calibration_rows,
        tolerance=float(
            config["p2_3"]["calibration"]["auc_tie_tolerance"]
        ),
    )
    selected = selection["selected_candidate"]
    # Construct the validation view only after the complete decoder is locked.
    validation_dataset = _IndexedTokenDataset(
        dataset,
        route_indices["route_validation"],
    )
    matched_validation_tokens = decode_one_shot_dataset(
        model=matched_model,
        dataset=validation_dataset,
        device=device,
        batch_size=int(config["evaluation"]["batch_size"]),
    )
    validation_decoded = decode_risk_controlled_dataset(
        model=rollout_model,
        gate=gate,
        dataset=validation_dataset,
        residual_codebook=codebook,
        config=config,
        device=device,
        proposal_fraction=float(selected["proposal_fraction"]),
        utility_threshold=float(selected["utility_threshold"]),
        rounds=int(selected["rounds"]),
    )
    validation = evaluate_residual_token_predictions(
        dataset=validation_dataset,
        residual_codebook=codebook,
        predictions={
            "matched_one_shot": matched_validation_tokens,
            "rollout_one_shot": validation_decoded["initial_tokens"],
            "risk_controlled_refinement": validation_decoded["tokens"],
        },
        config=config,
        device=device,
    )
    permutation = correction_permutation_check(
        model=rollout_model,
        gate=gate,
        dataset=validation_dataset,
        residual_codebook=codebook,
        config=config,
        device=device,
        proposal_fraction=float(selected["proposal_fraction"]),
        utility_threshold=float(selected["utility_threshold"]),
        rounds=int(selected["rounds"]),
    )
    latency = measure_correction_latency(
        model=rollout_model,
        gate=gate,
        dataset=validation_dataset,
        residual_codebook=codebook,
        config=config,
        device=device,
        proposal_fraction=float(selected["proposal_fraction"]),
        utility_threshold=float(selected["utility_threshold"]),
        rounds=int(selected["rounds"]),
    )
    transformer_hash_after = _sha256(p2_2["checkpoints"]["rollout"])
    decision = _decision(
        evaluation=validation,
        decoded=validation_decoded,
        selection=selection,
        permutation=permutation,
        latency=latency,
        transformer_hash_before=transformer_hash_before,
        transformer_hash_after=transformer_hash_after,
        config=config,
    )
    _write_csv(
        run_paths.artifacts / "calibration_candidates.csv",
        calibration_rows,
    )
    _write_json(
        run_paths.artifacts / "calibration_diagnostics.json",
        calibration_evaluations,
    )
    _write_json(run_paths.artifacts / "selection.json", selection)
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(
        run_paths.artifacts / "transition_diagnostics.json",
        {
            "trace": validation_decoded["trace"],
            "transitions": validation_decoded["transitions"],
        },
    )
    _write_json(run_paths.artifacts / "permutation_check.json", permutation)
    _write_json(run_paths.artifacts / "latency.json", latency)
    gpu_activity = monitor.summary()
    _write_json(run_paths.artifacts / "gpu_activity.json", gpu_activity)
    _write_json(run_paths.artifacts / "decision.json", decision)
    _write_csv(
        run_paths.artifacts / "core_comparison.csv",
        _core_rows(validation),
    )
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        _scenario_rows(validation),
    )
    report = _report(
        examples=examples.diagnostics,
        training=training,
        selection=selection,
        evaluation=validation,
        decoded=validation_decoded,
        decision=decision,
        latency=latency,
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "p2-3-risk-controlled-residual-refinement",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "p2_2_lock": {
            "selected_step": p2_2["selected_step"],
            "checkpoint_hashes": p2_2["checkpoint_hashes"],
        },
        "examples": examples.diagnostics,
        "training": training,
        "selection": selection,
        "validation": validation,
        "transitions": validation_decoded["transitions"],
        "permutation": permutation,
        "latency": latency,
        "gpu_activity": gpu_activity,
        "decision": decision,
        "official_val_loaded": False,
        "test_loaded": False,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "p2_3_completed",
        classification=decision["classification"],
        selected_candidate=selected,
        test_loaded=False,
    )
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "selection": selection,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
