"""Train and evaluate the one-shot dynamic PointerRoute Transformer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config
from set2proto.environment import collect_environment, config_sha256
from set2proto.experiment_logging import ExperimentLogger
from set2proto.pointer_route_model import build_pointer_route_model
from set2proto.pointer_route_training import (
    evaluate_pointer_route_subset,
    load_pointer_checkpoint,
    measure_pointer_latency,
    prepare_route_targets,
    train_one_shot_pointer,
)
from set2proto.reproducibility import seed_everything
from set2proto.run_paths import create_run_paths
from set2proto.training import TokenTrainingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    parser.add_argument("--profile", choices=("expanded",), default="expanded")
    parser.add_argument("--stage", choices=("smoke", "all"), default="all")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--quantization-root", type=Path, required=True)
    parser.add_argument("--phase01-root", type=Path, required=True)
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
    fieldnames: list[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_phase01_lock(
    *,
    phase01_root: Path,
    dataset_root: Path,
    quantization_root: Path,
) -> dict[str, Any]:
    artifacts = phase01_root / "artifacts"
    selection_path = artifacts / "selection.json"
    split_path = artifacts / "route_dev_split.json"
    integrity_path = artifacts / "source_integrity.json"
    for path in (selection_path, split_path, integrity_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase 0/1 lock artifact: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    current = {
        "dataset_manifest": _sha256(dataset_root / "manifest.json"),
        "sets_jsonl": _sha256(dataset_root / "train" / "sets.jsonl"),
        "projection": _sha256(quantization_root / "projection.pt"),
        "codebook": _sha256(quantization_root / "codebook.pt"),
        "train_targets": _sha256(quantization_root / "train_targets.pt"),
    }
    expected = {
        "dataset_manifest": integrity["dataset_manifest"]["sha256"],
        "sets_jsonl": integrity["sets_jsonl"]["sha256"],
        "projection": integrity["projection"]["sha256"],
        "codebook": integrity["codebook"]["sha256"],
        "train_targets": integrity["train_targets"]["sha256"],
    }
    if current != expected:
        raise RuntimeError("Phase 0/1 source assets changed after protocol lock")
    if not bool(split["checks"]["passed"]):
        raise RuntimeError("locked route-dev split did not pass integrity checks")
    return {
        "selection": selection,
        "split": split,
        "source_hashes": current,
        "phase01_artifact_hashes": {
            "selection.json": _sha256(selection_path),
            "route_dev_split.json": _sha256(split_path),
            "source_integrity.json": _sha256(integrity_path),
        },
        "pca_refit": False,
        "codebook_refit": False,
        "condition_or_teacher_sets_modified": False,
    }


def _core_rows(
    split: str,
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in evaluation["methods"].items():
        all_metrics = values["verification"]["all"]
        hard = values["verification"]["hard_average"]
        rows.append(
            {
                "split": split,
                "method": method,
                "pooled_all_auc": all_metrics["roc_auc"],
                "hard_macro_auc": hard["roc_auc"],
                "eer_all": all_metrics["eer"],
                "tar_at_far_1e-2": all_metrics["tar_at_far"]["0.01"]["tar"],
                "tar_at_far_1e-3": all_metrics["tar_at_far"]["0.001"]["tar"],
                "rank1_all": all_metrics["rank1_accuracy"],
                "mean_margin_all": all_metrics[
                    "mean_genuine_impostor_margin"
                ],
                "teacher_map_cosine_all": values["teacher_similarity"]["all"][
                    "mean_map_cosine"
                ],
                "teacher_local_cosine_all": values["teacher_similarity"]["all"][
                    "mean_local_cosine"
                ],
            }
        )
    return rows


def _scenario_rows(
    split: str,
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in evaluation["methods"].items():
        for scenario, metrics in values["verification"].items():
            if scenario == "hard_average":
                continue
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "scenario": scenario,
                    "roc_auc": metrics["roc_auc"],
                    "eer": metrics["eer"],
                    "tar_at_far_1e-2": metrics["tar_at_far"]["0.01"]["tar"],
                    "tar_at_far_1e-3": metrics["tar_at_far"]["0.001"]["tar"],
                    "rank1": metrics["rank1_accuracy"],
                    "mean_margin": metrics[
                        "mean_genuine_impostor_margin"
                    ],
                    "teacher_map_cosine": values["teacher_similarity"][
                        scenario
                    ]["mean_map_cosine"],
                }
            )
    return rows


def _decision(
    *,
    evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["methods"]
    quality = methods["quality_anchor"]["verification"]
    oracle = methods["hard_route_oracle"]["verification"]
    pointer = methods["one_shot_pointer"]["verification"]
    quality_hard = float(quality["hard_average"]["roc_auc"])
    oracle_hard = float(oracle["hard_average"]["roc_auc"])
    pointer_hard = float(pointer["hard_average"]["roc_auc"])
    model_gain = pointer_hard - quality_hard
    oracle_gain = oracle_hard - quality_hard
    gap_recovery = model_gain / oracle_gain if oracle_gain > 0.0 else 0.0
    scenario_deltas = {
        scenario: (
            float(pointer[scenario]["roc_auc"])
            - float(quality[scenario]["roc_auc"])
        )
        for scenario in (
            "low_quality",
            "complementary_occlusion",
            "common_occlusion",
            "wrong_identity",
        )
    }
    wins = sum(value > 0.0 for value in scenario_deltas.values())
    gates = config["pointer_route"]["one_shot"]["gates"]
    permutation = evaluation["permutation_check"]
    peak_reserved = max(
        float(training["peak_reserved_mib"]),
        float(latency["peak_reserved_mib"]),
    )
    checks = {
        "pointer_not_below_quality": (
            model_gain
            >= -float(gates["max_hard_auc_drop_vs_quality"])
        ),
        "oracle_gap_recovery": (
            gap_recovery >= float(gates["min_oracle_gap_recovery"])
        ),
        "hard_scenario_wins": (
            wins >= int(gates["min_hard_scenario_wins"])
        ),
        "frame_permutation_equivariant": (
            permutation["logit_max_abs_difference"]
            <= float(gates["max_permutation_logit_difference"])
            and permutation["route_map_max_abs_difference"] <= 1e-5
        ),
        "memory_within_7_2_gib": peak_reserved <= 7.2 * 1024.0,
        "teacher_not_used_at_inference": not bool(
            evaluation["inference_contract"]["teacher_used"]
        ),
        "gallery_not_used_at_inference": not bool(
            evaluation["inference_contract"]["gallery_used"]
        ),
        "test_features_not_loaded": True,
        "test_targets_not_loaded": True,
        "test_gallery_not_constructed": True,
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "go_to_plain_pointer_diffusion": passed,
        "pointer_minus_quality_hard_auc": model_gain,
        "oracle_minus_quality_hard_auc": oracle_gain,
        "oracle_gap_recovery": gap_recovery,
        "hard_scenario_wins": wins,
        "scenario_auc_deltas": scenario_deltas,
        "peak_reserved_mib": peak_reserved,
        "recommendation": (
            "proceed_to_phase3_plain_pointer_diffusion"
            if passed
            else "stop_before_diffusion_and_revisit_pointer_learnability"
        ),
    }


def _report(
    *,
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    latency: Mapping[str, Any],
    decision: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> str:
    rows = _core_rows("route_validation", validation)
    core_lines = [
        "| Method | All AUC | Hard AUC | EER | Rank-1 | Margin | Map cos |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        core_lines.append(
            f"| {row['method']} | {row['pooled_all_auc']:.6f} | "
            f"{row['hard_macro_auc']:.6f} | {row['eer_all']:.6f} | "
            f"{row['rank1_all']:.6f} | {row['mean_margin_all']:.6f} | "
            f"{row['teacher_map_cosine_all']:.6f} |"
        )
    pointer = validation["methods"]["one_shot_pointer"]["verification"]
    quality = validation["methods"]["quality_anchor"]["verification"]
    scenario_lines = [
        "| Scenario | Quality AUC | Pointer AUC | Delta |",
        "|---|---:|---:|---:|",
    ]
    for scenario in (
        "clean",
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    ):
        difference = (
            pointer[scenario]["roc_auc"] - quality[scenario]["roc_auc"]
        )
        scenario_lines.append(
            f"| {scenario} | {quality[scenario]['roc_auc']:.6f} | "
            f"{pointer[scenario]['roc_auc']:.6f} | {difference:+.6f} |"
        )
    failed = [
        name for name, passed in decision["checks"].items() if not passed
    ]
    status = (
        "GO"
        if decision["go_to_plain_pointer_diffusion"]
        else "NO-GO"
    )
    accuracy = validation["route_accuracy"]["all"]
    diagnostics = validation["route_diagnostics"]
    return f"""# PointerRoute Phase 2 One-Shot Training

## Locked protocol

- Phase 0/1 delta: {lock['selection']['delta']:.6f}.
- Phase 0/1 temperature: {lock['selection']['temperature']:.6f}.
- Training: 800 route-train identities only.
- Checkpoint selection: 100 route-calibration identities only.
- Final evaluation: one locked checkpoint on 100 route-validation identities.
- PCA, codebook, S/T sets, and split indices were hash-checked and unchanged.
- Existing P1-3 checkpoints are not reported as a route-validation baseline
  because those checkpoints were trained on all Stage16 train identities,
  including this carve-out.

## Training

- Parameters: {training['parameter_count']:,}.
- Steps completed: {training['steps_completed']}.
- Selected step: {training['best_step']}.
- Elapsed: {training['elapsed_seconds']:.2f} seconds.
- Peak reserved CUDA memory: {training['peak_reserved_mib']:.2f} MiB.

## Route-validation

{chr(10).join(core_lines)}

{chr(10).join(scenario_lines)}

## Route diagnostics

- Exact route accuracy: {accuracy['exact_route_accuracy']:.6f}.
- Epsilon-equivalent route accuracy:
  {accuracy['equivalent_route_accuracy']:.6f}.
- Non-anchor prediction fraction:
  {diagnostics['all']['non_anchor_fraction']:.6f}.
- Wrong-ID frame share among selected frame routes:
  {diagnostics['wrong_identity']['wrong_frame_selection_fraction_of_frame_routes']:.6f}.
- Permutation logit max difference:
  {validation['permutation_check']['logit_max_abs_difference']:.8f}.
- Batch-16 latency: {latency['mean_batch_latency_ms']:.3f} ms.

## Gate decision: {status}

- Pointer minus quality hard AUC:
  {decision['pointer_minus_quality_hard_auc']:+.6f}.
- Recovered oracle gap: {decision['oracle_gap_recovery']:.2%}.
- Hard-scenario wins: {decision['hard_scenario_wins']} / 4.
- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

This run tests whether the five-way teacher route is learnable from S. It does
not yet test iterative denoising and makes no statistical-significance claim.
"""


def main() -> int:
    args = parse_args()
    resolved = load_config(args.config, args.profile)
    config = resolved.to_dict()
    seed = int(config["project"]["seed"])
    seed_everything(seed, deterministic=bool(config["runtime"]["deterministic"]))
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    phase01_root = args.phase01_root.expanduser().resolve()
    lock = _validate_phase01_lock(
        phase01_root=phase01_root,
        dataset_root=dataset_root,
        quantization_root=quantization_root,
    )
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
        "stage": "pointer-route-phase2-one-shot",
        "mode": args.stage,
        "selection_split": "route_calibration",
        "evaluation_split": (
            None if args.stage == "smoke" else "route_validation"
        ),
        "test_features_loaded": False,
        "test_targets_loaded": False,
        "test_gallery_constructed": False,
        "one_shot": config["pointer_route"]["one_shot"],
        "phase01_selection": lock["selection"],
        "phase01_artifact_hashes": lock["phase01_artifact_hashes"],
        "config_sha256": config_sha256(config),
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "pointer_route_phase2_started",
        mode=args.stage,
        test_features_loaded=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("expanded PointerRoute training requires CUDA")
    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=True,
    )
    if dataset.targets["sample_ids"] != lock["split"]["sample_ids"][
        "route_train"
    ] + lock["split"]["sample_ids"]["route_calibration"] + lock["split"][
        "sample_ids"
    ]["route_validation"]:
        # The locked subsets intentionally retain source order within each
        # subset, but concatenating partitions changes global source order.
        locked_union = set().union(
            *(
                set(values)
                for values in lock["split"]["sample_ids"].values()
            )
        )
        if locked_union != set(dataset.targets["sample_ids"]):
            raise RuntimeError("locked route split sample IDs changed")
    delta = float(lock["selection"]["delta"])
    temperature = float(lock["selection"]["temperature"])
    targets = prepare_route_targets(
        dataset=dataset,
        delta=delta,
        temperature=temperature,
        device=device,
    )
    model = build_pointer_route_model(config)
    split_indices = lock["split"]["sample_indices"]

    def metric_callback(step: int, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            if name == "step":
                continue
            logger.log_metric(
                name=name,
                value=float(value),
                step=step,
                split="route_train",
            )

    training = train_one_shot_pointer(
        dataset=dataset,
        train_indices=split_indices["route_train"],
        calibration_indices=split_indices["route_calibration"],
        targets=targets,
        model=model,
        config=config,
        device=device,
        checkpoint_directory=run_paths.checkpoints,
        delta=delta,
        temperature=temperature,
        seed=seed + 52001,
        smoke=args.stage == "smoke",
        metric_callback=metric_callback,
    )
    load_pointer_checkpoint(
        path=training["best_checkpoint"],
        model=model,
        device=device,
    )
    calibration = evaluate_pointer_route_subset(
        split_name="route_calibration",
        dataset=dataset,
        indices=split_indices["route_calibration"],
        targets=targets,
        model=model,
        config=config,
        device=device,
        delta=delta,
        temperature=temperature,
    )
    _write_json(
        run_paths.artifacts / "calibration_evaluation.json",
        calibration,
    )
    _write_csv(
        run_paths.artifacts / "training_history.csv",
        training["history"],
    )
    _write_json(run_paths.artifacts / "training_summary.json", training)

    if args.stage == "smoke":
        manifest = {
            "schema_version": 1,
            "stage": "pointer-route-phase2-one-shot-smoke",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "phase01_lock": lock,
            "training": training,
            "calibration": calibration,
            "parameter_count": model.parameter_count(),
            "test_features_loaded": False,
            "test_targets_loaded": False,
            "test_gallery_constructed": False,
        }
        logger.write_manifest(manifest)
        logger.log_event("pointer_route_phase2_smoke_completed")
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "status": "smoke_passed",
                    "training": {
                        key: training[key]
                        for key in (
                            "steps_completed",
                            "best_step",
                            "parameter_count",
                            "peak_reserved_mib",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    validation = evaluate_pointer_route_subset(
        split_name="route_validation",
        dataset=dataset,
        indices=split_indices["route_validation"],
        targets=targets,
        model=model,
        config=config,
        device=device,
        delta=delta,
        temperature=temperature,
    )
    latency = measure_pointer_latency(
        dataset=dataset,
        indices=split_indices["route_validation"],
        model=model,
        config=config,
        device=device,
    )
    decision = _decision(
        evaluation=validation,
        training=training,
        latency=latency,
        config=config,
    )
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(run_paths.artifacts / "latency.json", latency)
    _write_json(run_paths.artifacts / "decision.json", decision)
    _write_csv(
        run_paths.artifacts / "core_comparison.csv",
        [
            *_core_rows("route_calibration", calibration),
            *_core_rows("route_validation", validation),
        ],
    )
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        [
            *_scenario_rows("route_calibration", calibration),
            *_scenario_rows("route_validation", validation),
        ],
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        _report(
            training=training,
            validation=validation,
            latency=latency,
            decision=decision,
            lock=lock,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "pointer-route-phase2-one-shot",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "phase01_lock": lock,
        "training": training,
        "calibration": calibration,
        "validation": validation,
        "latency": latency,
        "decision": decision,
        "test_features_loaded": False,
        "test_targets_loaded": False,
        "test_gallery_constructed": False,
    }
    logger.write_manifest(manifest)
    logger.log_event("pointer_route_phase2_completed", decision=decision)
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "training": {
                    key: training[key]
                    for key in (
                        "steps_completed",
                        "best_step",
                        "elapsed_seconds",
                        "parameter_count",
                        "peak_reserved_mib",
                    )
                },
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
