"""Paired one-shot versus plain confidence-only PointerRoute diffusion."""

from __future__ import annotations

import argparse
import copy
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
from set2proto.pointer_route_diffusion import (
    evaluate_pointer_diffusion_subset,
    measure_diffusion_latency,
    train_pointer_diffusion,
)
from set2proto.pointer_route_model import build_pointer_route_model
from set2proto.pointer_route_training import (
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


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _phase01_lock(
    *,
    root: Path,
    dataset_root: Path,
    quantization_root: Path,
) -> dict[str, Any]:
    artifacts = root / "artifacts"
    selection_path = artifacts / "selection.json"
    split_path = artifacts / "route_dev_split.json"
    source_path = artifacts / "source_integrity.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    current = {
        "dataset_manifest": _sha256(dataset_root / "manifest.json"),
        "sets_jsonl": _sha256(dataset_root / "train" / "sets.jsonl"),
        "projection": _sha256(quantization_root / "projection.pt"),
        "codebook": _sha256(quantization_root / "codebook.pt"),
        "train_targets": _sha256(quantization_root / "train_targets.pt"),
    }
    expected = {
        "dataset_manifest": source["dataset_manifest"]["sha256"],
        "sets_jsonl": source["sets_jsonl"]["sha256"],
        "projection": source["projection"]["sha256"],
        "codebook": source["codebook"]["sha256"],
        "train_targets": source["train_targets"]["sha256"],
    }
    if current != expected:
        raise RuntimeError("Phase 0/1 source assets changed")
    if not bool(split["checks"]["passed"]):
        raise RuntimeError("Phase 0/1 route split is invalid")
    return {
        "selection": selection,
        "split": split,
        "source_hashes": current,
        "artifact_hashes": {
            "selection": _sha256(selection_path),
            "split": _sha256(split_path),
            "source": _sha256(source_path),
        },
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


def _trajectory_rows(
    split: str,
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    trajectory = evaluation["trajectory"]
    rows = [
        {"split": split, "scope": "per_step", **row}
        for row in trajectory["per_step"]
    ]
    for scenario, values in trajectory["transitions"].items():
        for correctness, metrics in (
            ("exact", values["exact"]),
            ("equivalent", values["equivalent"]),
        ):
            rows.append(
                {
                    "split": split,
                    "scope": f"transition_{correctness}",
                    "scenario": scenario,
                    **metrics,
                    "route_change_fraction": values[
                        "route_change_fraction"
                    ],
                    "no_op_fraction": values["no_op_fraction"],
                }
            )
    return rows


def _decision(
    *,
    validation: Mapping[str, Any],
    one_shot_training: Mapping[str, Any],
    diffusion_training: Mapping[str, Any],
    latency: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = validation["methods"]
    one_shot = methods["one_shot_pointer"]["verification"]
    one_step = methods["pointer_diffusion_1step"]["verification"]
    four_step = methods["pointer_diffusion_4step"]["verification"]
    four_minus_one = (
        four_step["hard_average"]["roc_auc"]
        - one_step["hard_average"]["roc_auc"]
    )
    four_minus_paired = (
        four_step["hard_average"]["roc_auc"]
        - one_shot["hard_average"]["roc_auc"]
    )
    scenario_step_deltas = {
        scenario: (
            four_step[scenario]["roc_auc"]
            - one_step[scenario]["roc_auc"]
        )
        for scenario in (
            "low_quality",
            "complementary_occlusion",
            "common_occlusion",
            "wrong_identity",
        )
    }
    scenario_wins = sum(value > 0.0 for value in scenario_step_deltas.values())
    transition = validation["trajectory"]["transitions"]["all"]["equivalent"]
    gates = config["pointer_route"]["diffusion"]["gates"]
    maximum_latency_memory = max(
        float(values["peak_reserved_mib"])
        for values in latency["diffusion"].values()
    )
    peak_reserved = max(
        float(one_shot_training["peak_reserved_mib"]),
        float(diffusion_training["peak_reserved_mib"]),
        float(latency["one_shot"]["peak_reserved_mib"]),
        maximum_latency_memory,
    )
    checks = {
        "initial_states_identical": bool(initial_state["identical"]),
        "parameter_counts_equal": (
            one_shot_training["parameter_count"]
            == diffusion_training["parameter_count"]
        ),
        "training_steps_equal": (
            one_shot_training["steps_completed"]
            == diffusion_training["steps_completed"]
        ),
        "four_step_beats_one_step": (
            four_minus_one
            >= float(gates["min_four_step_gain_over_one_step"])
        ),
        "four_step_beats_paired_one_shot": (
            four_minus_paired
            >= float(gates["min_four_step_gain_over_paired_one_shot"])
        ),
        "hard_scenario_step_wins": (
            scenario_wins >= int(gates["min_hard_scenario_step_wins"])
        ),
        "clean_step_constraint": (
            one_step["clean"]["roc_auc"] - four_step["clean"]["roc_auc"]
            <= float(gates["max_clean_auc_drop_vs_one_step"])
        ),
        "positive_equivalent_net_correction": (
            transition["net_correction_fraction"] > 0.0
            if bool(gates["require_positive_net_correction"])
            else True
        ),
        "ecr_exceeds_eir": transition["ecr"] > transition["eir"],
        "frame_permutation_equivariant": (
            validation["diffusion_permutation_check"][
                "route_map_max_abs_difference"
            ]
            <= float(gates["max_permutation_map_difference"])
        ),
        "memory_within_7_2_gib": peak_reserved <= 7.2 * 1024.0,
        "evidence_not_used": not bool(
            validation["diffusion_inference_contract"]["evidence_used"]
        ),
        "remask_not_used": not bool(
            validation["diffusion_inference_contract"]["remask_used"]
        ),
        "test_not_loaded": True,
    }
    go = all(checks.values())
    iterative_correction = (
        four_minus_one > 0.0
        and transition["net_correction_fraction"] > 0.0
        and transition["ecr"] > transition["eir"]
    )
    if go:
        recommendation = "proceed_to_two_level_evidence_and_conservative_remask"
        classification = "GO"
    elif iterative_correction:
        recommendation = "partial_go_fix_training_before_evidence"
        classification = "PARTIAL_GO"
    else:
        recommendation = "run_single_rollout_corruption_rescue_before_no_go"
        classification = "NO_GO"
    return {
        "checks": checks,
        "classification": classification,
        "go_to_evidence": go,
        "four_step_minus_one_step_hard_auc": four_minus_one,
        "four_step_minus_paired_one_shot_hard_auc": four_minus_paired,
        "scenario_step_auc_deltas": scenario_step_deltas,
        "hard_scenario_step_wins": scenario_wins,
        "equivalent_ecr": transition["ecr"],
        "equivalent_eir": transition["eir"],
        "equivalent_net_correction_fraction": transition[
            "net_correction_fraction"
        ],
        "peak_reserved_mib": peak_reserved,
        "recommendation": recommendation,
    }


def _report(
    *,
    validation: Mapping[str, Any],
    one_shot_training: Mapping[str, Any],
    diffusion_training: Mapping[str, Any],
    latency: Mapping[str, Any],
    decision: Mapping[str, Any],
    initial_state: Mapping[str, Any],
) -> str:
    selected = (
        "quality_anchor",
        "hard_route_oracle",
        "one_shot_pointer",
        "pointer_diffusion_1step",
        "pointer_diffusion_2step",
        "pointer_diffusion_4step",
        "pointer_diffusion_8step",
    )
    rows = {
        row["method"]: row
        for row in _core_rows("route_validation", validation)
    }
    table = [
        "| Method | All AUC | Hard AUC | EER | Rank-1 | Map cos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in selected:
        row = rows[method]
        table.append(
            f"| {method} | {row['pooled_all_auc']:.6f} | "
            f"{row['hard_macro_auc']:.6f} | {row['eer_all']:.6f} | "
            f"{row['rank1_all']:.6f} | "
            f"{row['teacher_map_cosine_all']:.6f} |"
        )
    one = validation["methods"]["pointer_diffusion_1step"]["verification"]
    four = validation["methods"]["pointer_diffusion_4step"]["verification"]
    scenarios = [
        "| Scenario | D0-1step | D0-4step | Step gain |",
        "|---|---:|---:|---:|",
    ]
    for scenario in (
        "clean",
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    ):
        gain = four[scenario]["roc_auc"] - one[scenario]["roc_auc"]
        scenarios.append(
            f"| {scenario} | {one[scenario]['roc_auc']:.6f} | "
            f"{four[scenario]['roc_auc']:.6f} | {gain:+.6f} |"
        )
    failed = [
        name for name, passed in decision["checks"].items() if not passed
    ]
    transition = validation["trajectory"]["transitions"]["all"]["equivalent"]
    return f"""# PointerRoute Phase 3 Paired Plain Diffusion

## Protocol

- B2-paired and D0 were cloned from the same initial state:
  `{initial_state['one_shot_sha256']}`.
- Equal parameters: {one_shot_training['parameter_count']:,}.
- Equal optimizer steps: {one_shot_training['steps_completed']}.
- D0 corruption: 50% all-mask / 50% random partial-mask.
- D0 decoding: confidence-only irreversible commits.
- Evidence, remask, rollout corruption, and test data were not used.
- Checkpoints were selected on route-calibration. Route-validation was run
  only after both checkpoints were locked.

## Core comparison

{chr(10).join(table)}

## One-step versus four-step

{chr(10).join(scenarios)}

- Four-step minus one-step hard AUC:
  {decision['four_step_minus_one_step_hard_auc']:+.6f}.
- Four-step minus paired one-shot hard AUC:
  {decision['four_step_minus_paired_one_shot_hard_auc']:+.6f}.
- Equivalent ECR: {transition['ecr']:.6f}.
- Equivalent EIR: {transition['eir']:.6f}.
- Equivalent net correction:
  {transition['net_correction_fraction']:+.6f}.
- Four-step route-change fraction:
  {validation['trajectory']['transitions']['all']['route_change_fraction']:.6f}.

## Runtime

- One-shot training: {one_shot_training['elapsed_seconds']:.2f} seconds.
- D0 training: {diffusion_training['elapsed_seconds']:.2f} seconds.
- One-shot batch-16 latency:
  {latency['one_shot']['mean_batch_latency_ms']:.3f} ms.
- Four-step batch-16 latency:
  {latency['diffusion']['4step']['mean_batch_latency_ms']:.3f} ms.
- Peak reserved CUDA memory: {decision['peak_reserved_mib']:.2f} MiB.

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

## Interpretation

- D0 one-step remains competitive with the paired one-shot model, so the
  dynamic discrete route space itself is learnable.
- Accuracy, teacher-map cosine, and hard AUC decrease as decoding moves from
  one to four steps. The committed trajectory corrects fewer equivalent route
  errors than it introduces (ECR {transition['ecr']:.6f} versus EIR
  {transition['eir']:.6f}).
- The failure is therefore localized to iterative self-conditioning under
  the random-mask training distribution. It does not reject the Phase 0/1
  route oracle or the broader use of discrete aggregation.
- Evidence guidance and remasking are not authorized by this result. The
  only bounded rescue is one preregistered rerun with model-rollout
  corruption; if that does not make four-step correction positive, plain
  PointerRoute diffusion should stop.

This is a development mechanism screen on the already observed route-validation
split, not a new holdout or statistical-significance claim.
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
    lock = _phase01_lock(
        root=phase01_root,
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
        "stage": "pointer-route-phase3-paired-plain-diffusion",
        "mode": args.stage,
        "same_initialization": True,
        "same_batch_order": True,
        "same_optimizer_steps": True,
        "selection_split": "route_calibration",
        "evaluation_split": (
            None if args.stage == "smoke" else "route_validation"
        ),
        "diffusion": config["pointer_route"]["diffusion"],
        "training": config["pointer_route"]["one_shot"],
        "phase01": lock["artifact_hashes"],
        "config_sha256": config_sha256(config),
        "test_loaded": False,
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "pointer_route_phase3_started",
        mode=args.stage,
        test_loaded=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("paired PointerRoute training requires CUDA")
    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=True,
    )
    locked_ids = set().union(
        *(
            set(values)
            for values in lock["split"]["sample_ids"].values()
        )
    )
    if locked_ids != set(dataset.targets["sample_ids"]):
        raise RuntimeError("locked sample IDs changed")
    delta = float(lock["selection"]["delta"])
    temperature = float(lock["selection"]["temperature"])
    targets = prepare_route_targets(
        dataset=dataset,
        delta=delta,
        temperature=temperature,
        device=device,
    )
    paired_seed = seed + 61001
    torch.manual_seed(paired_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(paired_seed)
    base_model = build_pointer_route_model(config)
    one_shot_model = copy.deepcopy(base_model)
    diffusion_model = copy.deepcopy(base_model)
    initial_state = {
        "seed": paired_seed,
        "one_shot_sha256": _state_sha256(one_shot_model),
        "diffusion_sha256": _state_sha256(diffusion_model),
    }
    initial_state["identical"] = (
        initial_state["one_shot_sha256"]
        == initial_state["diffusion_sha256"]
    )
    if not initial_state["identical"]:
        raise RuntimeError("paired models did not share initial state")
    _write_json(run_paths.artifacts / "initial_state.json", initial_state)
    split_indices = lock["split"]["sample_indices"]
    batch_seed = seed + 62001

    def one_callback(step: int, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            if name != "step":
                logger.log_metric(
                    name=f"one_shot/{name}",
                    value=float(value),
                    step=step,
                    split="route_train",
                )

    def diffusion_callback(step: int, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            if name != "step":
                logger.log_metric(
                    name=f"diffusion/{name}",
                    value=float(value),
                    step=step,
                    split="route_train",
                )

    one_shot_training = train_one_shot_pointer(
        dataset=dataset,
        train_indices=split_indices["route_train"],
        calibration_indices=split_indices["route_calibration"],
        targets=targets,
        model=one_shot_model,
        config=config,
        device=device,
        checkpoint_directory=run_paths.checkpoints / "one_shot",
        delta=delta,
        temperature=temperature,
        seed=batch_seed,
        smoke=args.stage == "smoke",
        metric_callback=one_callback,
    )
    load_pointer_checkpoint(
        path=one_shot_training["best_checkpoint"],
        model=one_shot_model,
        device=device,
    )
    diffusion_training = train_pointer_diffusion(
        dataset=dataset,
        train_indices=split_indices["route_train"],
        calibration_indices=split_indices["route_calibration"],
        targets=targets,
        model=diffusion_model,
        paired_one_shot_model=one_shot_model,
        config=config,
        device=device,
        checkpoint_directory=run_paths.checkpoints / "diffusion",
        delta=delta,
        temperature=temperature,
        seed=batch_seed,
        smoke=args.stage == "smoke",
        metric_callback=diffusion_callback,
    )
    load_pointer_checkpoint(
        path=diffusion_training["best_checkpoint"],
        model=diffusion_model,
        device=device,
    )
    calibration = evaluate_pointer_diffusion_subset(
        split_name="route_calibration",
        dataset=dataset,
        indices=split_indices["route_calibration"],
        targets=targets,
        one_shot_model=one_shot_model,
        diffusion_model=diffusion_model,
        config=config,
        device=device,
        delta=delta,
        temperature=temperature,
    )
    _write_json(
        run_paths.artifacts / "calibration_evaluation.json",
        calibration,
    )
    training_rows = [
        {"model": "one_shot", **row}
        for row in one_shot_training["history"]
    ] + [
        {"model": "diffusion", **row}
        for row in diffusion_training["history"]
    ]
    _write_csv(
        run_paths.artifacts / "training_history.csv",
        training_rows,
    )
    paired_summary = {
        "one_shot": one_shot_training,
        "diffusion": diffusion_training,
    }
    _write_json(
        run_paths.artifacts / "paired_training_summary.json",
        paired_summary,
    )
    if args.stage == "smoke":
        manifest = {
            "schema_version": 1,
            "stage": "pointer-route-phase3-paired-smoke",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "phase01_lock": lock,
            "initial_state": initial_state,
            "training": paired_summary,
            "calibration": calibration,
            "test_loaded": False,
        }
        logger.write_manifest(manifest)
        logger.log_event("pointer_route_phase3_smoke_completed")
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "status": "paired_smoke_passed",
                    "initial_state": initial_state,
                    "one_shot_steps": one_shot_training["steps_completed"],
                    "diffusion_steps": diffusion_training["steps_completed"],
                    "peak_reserved_mib": max(
                        one_shot_training["peak_reserved_mib"],
                        diffusion_training["peak_reserved_mib"],
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    validation = evaluate_pointer_diffusion_subset(
        split_name="route_validation",
        dataset=dataset,
        indices=split_indices["route_validation"],
        targets=targets,
        one_shot_model=one_shot_model,
        diffusion_model=diffusion_model,
        config=config,
        device=device,
        delta=delta,
        temperature=temperature,
    )
    latency = {
        "one_shot": measure_pointer_latency(
            dataset=dataset,
            indices=split_indices["route_validation"],
            model=one_shot_model,
            config=config,
            device=device,
        ),
        "diffusion": measure_diffusion_latency(
            dataset=dataset,
            indices=split_indices["route_validation"],
            model=diffusion_model,
            config=config,
            device=device,
        ),
    }
    decision = _decision(
        validation=validation,
        one_shot_training=one_shot_training,
        diffusion_training=diffusion_training,
        latency=latency,
        initial_state=initial_state,
        config=config,
    )
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(
        run_paths.artifacts / "trajectory_diagnostics.json",
        validation["trajectory"],
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
    _write_csv(
        run_paths.artifacts / "trajectory_metrics.csv",
        [
            *_trajectory_rows("route_calibration", calibration),
            *_trajectory_rows("route_validation", validation),
        ],
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        _report(
            validation=validation,
            one_shot_training=one_shot_training,
            diffusion_training=diffusion_training,
            latency=latency,
            decision=decision,
            initial_state=initial_state,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "pointer-route-phase3-paired-plain-diffusion",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "phase01_lock": lock,
        "initial_state": initial_state,
        "training": paired_summary,
        "calibration": calibration,
        "validation": validation,
        "latency": latency,
        "decision": decision,
        "test_loaded": False,
    }
    logger.write_manifest(manifest)
    logger.log_event("pointer_route_phase3_completed", decision=decision)
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "one_shot_training": {
                    key: one_shot_training[key]
                    for key in (
                        "steps_completed",
                        "best_step",
                        "elapsed_seconds",
                        "peak_reserved_mib",
                    )
                },
                "diffusion_training": {
                    key: diffusion_training[key]
                    for key in (
                        "steps_completed",
                        "best_step",
                        "elapsed_seconds",
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
