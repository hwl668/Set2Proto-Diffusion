"""Single preregistered self-rollout rescue for PointerRoute diffusion."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_pointer_route_phase3 import (
    _core_rows,
    _phase01_lock,
    _scenario_rows,
    _sha256,
    _state_sha256,
    _trajectory_rows,
    _write_csv,
    _write_json,
)
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
    prepare_route_targets,
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
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


class CudaActivityMonitor:
    """Sample nvidia-smi without adding a Python package dependency."""

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, float]] = []
        self._errors: list[str] = []

    def _run(self) -> None:
        creation_flags = int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                    creationflags=creation_flags,
                )
                values = [
                    float(value.strip())
                    for value in result.stdout.strip().split(",")
                ]
                if len(values) == 3:
                    self._samples.append(
                        {
                            "utilization_percent": values[0],
                            "memory_used_mib": values[1],
                            "power_watts": values[2],
                        }
                    )
            except Exception as error:  # diagnostics must not hide training
                if len(self._errors) < 5:
                    self._errors.append(
                        f"{type(error).__name__}: {error}"
                    )
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "CudaActivityMonitor":
        self._thread = threading.Thread(
            target=self._run,
            name="cuda-activity-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        def values(name: str) -> list[float]:
            return [sample[name] for sample in self._samples]

        def percentile(items: list[float], fraction: float) -> float:
            if not items:
                return 0.0
            ordered = sorted(items)
            index = round((len(ordered) - 1) * fraction)
            return float(ordered[index])

        utilization = values("utilization_percent")
        memory = values("memory_used_mib")
        power = values("power_watts")
        return {
            "backend": "nvidia-smi",
            "interval_seconds": self.interval_seconds,
            "samples": len(self._samples),
            "mean_utilization_percent": (
                statistics.fmean(utilization) if utilization else 0.0
            ),
            "p95_utilization_percent": percentile(utilization, 0.95),
            "max_utilization_percent": max(utilization, default=0.0),
            "max_memory_used_mib": max(memory, default=0.0),
            "mean_power_watts": (
                statistics.fmean(power) if power else 0.0
            ),
            "errors": self._errors,
        }


def _load_phase3_lock(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    required = {
        "initial_state": artifacts / "initial_state.json",
        "validation": artifacts / "validation_evaluation.json",
        "training": artifacts / "paired_training_summary.json",
        "decision": artifacts / "decision.json",
        "protocol": artifacts / "protocol_preregistration.json",
        "latency": artifacts / "latency.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase 3 {name}: {path}")
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
    }
    if (
        values["decision"]["recommendation"]
        != "run_single_rollout_corruption_rescue_before_no_go"
    ):
        raise RuntimeError("Phase 3 did not authorize the bounded rescue")
    if values["protocol"].get("test_loaded") is not False:
        raise RuntimeError("Phase 3 protocol unexpectedly loaded test data")
    return {
        **values,
        "artifact_hashes": {
            name: _sha256(path) for name, path in required.items()
        },
    }


def _renamed_rows(
    *,
    split: str,
    evaluation: Mapping[str, Any],
    family: str,
    scenarios: bool,
) -> list[dict[str, Any]]:
    rows = (
        _scenario_rows(split, evaluation)
        if scenarios
        else _core_rows(split, evaluation)
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        method = str(value["method"])
        if method.startswith("pointer_diffusion_"):
            value["method"] = method.replace(
                "pointer_diffusion_",
                f"{family}_",
                1,
            )
        elif family == "rollout_d1":
            # Shared baselines are emitted only from the locked D0 artifact.
            continue
        result.append(value)
    return result


def _decision(
    *,
    validation: Mapping[str, Any],
    phase3: Mapping[str, Any],
    training: Mapping[str, Any],
    latency: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    gpu_activity: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rescue_methods = validation["methods"]
    previous_methods = phase3["validation"]["methods"]
    one = rescue_methods["pointer_diffusion_1step"]["verification"]
    four = rescue_methods["pointer_diffusion_4step"]["verification"]
    paired = rescue_methods["one_shot_pointer"]["verification"]
    previous_four = previous_methods["pointer_diffusion_4step"][
        "verification"
    ]
    hard_gain_one = (
        four["hard_average"]["roc_auc"]
        - one["hard_average"]["roc_auc"]
    )
    hard_gain_paired = (
        four["hard_average"]["roc_auc"]
        - paired["hard_average"]["roc_auc"]
    )
    hard_gain_d0 = (
        four["hard_average"]["roc_auc"]
        - previous_four["hard_average"]["roc_auc"]
    )
    scenario_deltas = {
        scenario: four[scenario]["roc_auc"] - one[scenario]["roc_auc"]
        for scenario in (
            "low_quality",
            "complementary_occlusion",
            "common_occlusion",
            "wrong_identity",
        )
    }
    scenario_wins = sum(value > 0.0 for value in scenario_deltas.values())
    transition = validation["trajectory"]["transitions"]["all"]["equivalent"]
    gates = config["pointer_route"]["diffusion"]["rollout_rescue"]["gates"]
    peak_reserved = max(
        float(training["peak_reserved_mib"]),
        *(
            float(value["peak_reserved_mib"])
            for value in latency.values()
        ),
    )
    device_audit = training["device_audit"]
    checks = {
        "initial_state_matches_phase3": bool(initial_state["matches_phase3"]),
        "parameter_count_matches_d0": (
            int(training["parameter_count"])
            == int(phase3["training"]["diffusion"]["parameter_count"])
        ),
        "optimizer_steps_match_d0": (
            int(training["steps_completed"])
            == int(phase3["training"]["diffusion"]["steps_completed"])
        ),
        "four_step_beats_one_step": (
            hard_gain_one
            >= float(gates["min_four_step_gain_over_one_step"])
        ),
        "four_step_beats_paired_one_shot": (
            hard_gain_paired
            >= float(gates["min_four_step_gain_over_paired_one_shot"])
        ),
        "four_step_beats_plain_d0": (
            hard_gain_d0
            >= float(gates["min_four_step_gain_over_plain_d0"])
        ),
        "hard_scenario_step_wins": (
            scenario_wins >= int(gates["min_hard_scenario_step_wins"])
        ),
        "clean_step_constraint": (
            one["clean"]["roc_auc"] - four["clean"]["roc_auc"]
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
        "training_model_on_cuda": str(
            device_audit["model_device"]
        ).startswith("cuda"),
        "training_batches_on_cuda": bool(
            device_audit["batch_residency_verified"]
        ),
        "loss_on_cuda": str(device_audit["loss_device"]).startswith("cuda"),
        "gpu_cache_enabled": bool(
            device_audit["gpu_resident_training_cache"]
        ),
        "gpu_activity_observed": (
            int(gpu_activity["samples"]) > 0
            and float(gpu_activity["max_utilization_percent"]) > 0.0
        ),
        "memory_within_7_2_gib": peak_reserved <= 7.2 * 1024.0,
        "evidence_not_used": not bool(
            config["pointer_route"]["diffusion"]["rollout_rescue"][
                "evidence_used"
            ]
        ),
        "remask_not_used": not bool(
            config["pointer_route"]["diffusion"]["rollout_rescue"][
                "remask_used"
            ]
        ),
        "test_not_loaded": True,
    }
    passed = all(checks.values())
    return {
        "classification": (
            "GO_TO_EVIDENCE"
            if passed
            else "NO_GO_STOP_PLAIN_DIFFUSION"
        ),
        "checks": checks,
        "four_step_minus_one_step_hard_auc": hard_gain_one,
        "four_step_minus_paired_one_shot_hard_auc": hard_gain_paired,
        "four_step_minus_plain_d0_hard_auc": hard_gain_d0,
        "scenario_step_auc_deltas": scenario_deltas,
        "hard_scenario_step_wins": scenario_wins,
        "equivalent_ecr": transition["ecr"],
        "equivalent_eir": transition["eir"],
        "equivalent_net_correction_fraction": transition[
            "net_correction_fraction"
        ],
        "peak_reserved_mib": peak_reserved,
        "recommendation": (
            "proceed_to_preregistered_two_level_evidence"
            if passed
            else "stop_plain_pointerroute_diffusion"
        ),
    }


def _report(
    *,
    validation: Mapping[str, Any],
    phase3: Mapping[str, Any],
    training: Mapping[str, Any],
    latency: Mapping[str, Any],
    decision: Mapping[str, Any],
    gpu_activity: Mapping[str, Any],
) -> str:
    current = validation["methods"]
    previous = phase3["validation"]["methods"]

    def hard(methods: Mapping[str, Any], name: str) -> float:
        return float(
            methods[name]["verification"]["hard_average"]["roc_auc"]
        )

    rows = [
        ("quality_anchor", hard(current, "quality_anchor")),
        ("paired_one_shot", hard(current, "one_shot_pointer")),
        ("plain_d0_1step", hard(previous, "pointer_diffusion_1step")),
        ("plain_d0_4step", hard(previous, "pointer_diffusion_4step")),
        ("rollout_d1_1step", hard(current, "pointer_diffusion_1step")),
        ("rollout_d1_2step", hard(current, "pointer_diffusion_2step")),
        ("rollout_d1_4step", hard(current, "pointer_diffusion_4step")),
        ("rollout_d1_8step", hard(current, "pointer_diffusion_8step")),
    ]
    comparison = "\n".join(
        ["| Method | Hard AUC |", "|---|---:|"]
        + [f"| {name} | {value:.6f} |" for name, value in rows]
    )
    scenarios = "\n".join(
        ["| Scenario | D1 1-step | D1 4-step | Gain |", "|---|---:|---:|---:|"]
        + [
            (
                f"| {scenario} | "
                f"{current['pointer_diffusion_1step']['verification'][scenario]['roc_auc']:.6f} | "
                f"{current['pointer_diffusion_4step']['verification'][scenario]['roc_auc']:.6f} | "
                f"{decision['scenario_step_auc_deltas'][scenario]:+.6f} |"
            )
            for scenario in (
                "low_quality",
                "complementary_occlusion",
                "common_occlusion",
                "wrong_identity",
            )
        ]
    )
    failed = [
        name for name, passed in decision["checks"].items() if not passed
    ]
    audit = training["device_audit"]
    history = training["history"]

    def history_mean(name: str) -> float:
        return statistics.fmean(float(row[name]) for row in history)

    previous_penalty = float(
        phase3["decision"]["four_step_minus_one_step_hard_auc"]
    )
    current_penalty = float(
        decision["four_step_minus_one_step_hard_auc"]
    )
    penalty_reduction = (
        1.0 - abs(current_penalty) / abs(previous_penalty)
        if previous_penalty
        else 0.0
    )
    return f"""# PointerRoute rollout-corruption rescue

## Locked protocol

- This is the single rescue authorized by the Phase 3 NO-GO.
- Corruption changed from 50% all-mask / 50% random teacher partial to
  50% all-mask / 50% stop-gradient self-rollout partial.
- Rollout depth is uniformly sampled from 1/2/3 of a four-step cosine
  confidence-only irreversible trajectory.
- Initialization, train identities, batch order, optimizer, losses, 2,000
  optimizer steps, decoder, and selection split are unchanged.
- Evidence, remasking, and test data were not used.

## Core comparison

{comparison}

## Iterative correction

{scenarios}

- D1 four-step minus D1 one-step hard AUC:
  {decision['four_step_minus_one_step_hard_auc']:+.6f}.
- D1 four-step minus paired one-shot hard AUC:
  {decision['four_step_minus_paired_one_shot_hard_auc']:+.6f}.
- D1 four-step minus plain D0 four-step hard AUC:
  {decision['four_step_minus_plain_d0_hard_auc']:+.6f}.
- Equivalent ECR/EIR:
  {decision['equivalent_ecr']:.6f}/{decision['equivalent_eir']:.6f}.
- Equivalent net correction:
  {decision['equivalent_net_correction_fraction']:+.6f}.

## GPU audit

- Model device: `{audit['model_device']}`.
- Loss device: `{audit['loss_device']}`.
- GPU-resident cached tensors: {audit['gpu_cache_mib']:.2f} MiB.
- Training peak reserved CUDA memory:
  {training['peak_reserved_mib']:.2f} MiB.
- nvidia-smi samples: {gpu_activity['samples']}.
- Mean/P95/max sampled GPU utilization:
  {gpu_activity['mean_utilization_percent']:.2f}% /
  {gpu_activity['p95_utilization_percent']:.2f}% /
  {gpu_activity['max_utilization_percent']:.2f}%.
- D1 four-step batch-16 latency:
  {latency['4step']['mean_batch_latency_ms']:.3f} ms.
- Training time: {training['elapsed_seconds']:.2f} seconds.

## Corruption audit

- Mean self-rollout/all-mask fractions:
  {history_mean('corruption_rollout_fraction'):.6f} /
  {history_mean('corruption_all_mask_fraction'):.6f}.
- Mean sampled rollout depth:
  {history_mean('corruption_mean_rollout_depth'):.6f}.
- Mean committed fraction in training inputs:
  {history_mean('corruption_committed_fraction'):.6f}.
- Mean error fraction among self-committed training routes:
  {history_mean('corruption_committed_error_fraction'):.6f}.

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

## Interpretation

- Rollout training reduced the absolute four-step penalty relative to one
  step by {100.0 * penalty_reduction:.2f}% and changed equivalent net
  correction from negative to positive. The Phase 3 distribution-shift
  diagnosis was therefore real.
- D1 four-step now exceeds both the paired one-shot model and plain D0
  four-step, but it still does not exceed its own stronger D1 one-step
  prediction. Only one hard scenario has a positive step delta.
- Confidence-only irreversible ordering is therefore the remaining failed
  mechanism. Continuing to tune plain masked diffusion is not justified.
  Any evidence-guided experiment must be framed as a new isolated hypothesis
  and compared against D1 one-step, not reported as if plain multi-step had
  passed.

This is a mechanism-development result on the already observed
route-validation split, not a statistical-significance or untouched-holdout
claim.
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
    phase3_root = args.phase3_root.expanduser().resolve()
    phase01 = _phase01_lock(
        root=phase01_root,
        dataset_root=dataset_root,
        quantization_root=quantization_root,
    )
    phase3 = _load_phase3_lock(phase3_root)
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
    rescue_config = config["pointer_route"]["diffusion"]["rollout_rescue"]
    preregistration = {
        "schema_version": 1,
        "stage": "pointer-route-rollout-corruption-rescue",
        "mode": args.stage,
        "single_authorized_rescue": True,
        "selection_split": "route_calibration",
        "evaluation_split": (
            None if args.stage == "smoke" else "route_validation"
        ),
        "changed_variable": (
            "random_teacher_partial_to_stop_gradient_self_rollout_partial"
        ),
        "rescue": rescue_config,
        "unchanged_training": config["pointer_route"]["one_shot"],
        "same_batch_seed_as_phase3": seed + 62001,
        "same_initial_seed_as_phase3": seed + 61001,
        "phase01_artifact_hashes": phase01["artifact_hashes"],
        "phase3_artifact_hashes": phase3["artifact_hashes"],
        "config_sha256": config_sha256(config),
        "test_loaded": False,
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "pointer_route_rollout_rescue_started",
        mode=args.stage,
        test_loaded=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("rollout rescue requires CUDA")
    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=True,
    )
    locked_ids = set().union(
        *(
            set(values)
            for values in phase01["split"]["sample_ids"].values()
        )
    )
    if locked_ids != set(dataset.targets["sample_ids"]):
        raise RuntimeError("locked sample IDs changed")
    delta = float(phase01["selection"]["delta"])
    temperature = float(phase01["selection"]["temperature"])
    targets = prepare_route_targets(
        dataset=dataset,
        delta=delta,
        temperature=temperature,
        device=device,
    )

    paired_seed = seed + 61001
    torch.manual_seed(paired_seed)
    torch.cuda.manual_seed_all(paired_seed)
    rescue_model = build_pointer_route_model(config)
    rescue_hash = _state_sha256(rescue_model)
    expected_hash = str(phase3["initial_state"]["diffusion_sha256"])
    initial_state = {
        "seed": paired_seed,
        "rescue_sha256": rescue_hash,
        "phase3_sha256": expected_hash,
        "matches_phase3": rescue_hash == expected_hash,
    }
    if not initial_state["matches_phase3"]:
        raise RuntimeError("rescue initialization does not match Phase 3")
    _write_json(run_paths.artifacts / "initial_state.json", initial_state)

    paired_model = build_pointer_route_model(config)
    paired_checkpoint = Path(
        phase3["training"]["one_shot"]["best_checkpoint"]
    )
    load_pointer_checkpoint(
        path=paired_checkpoint,
        model=paired_model,
        device=device,
    )
    split_indices = phase01["split"]["sample_indices"]

    def callback(step: int, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            if name != "step":
                logger.log_metric(
                    name=f"rollout_rescue/{name}",
                    value=float(value),
                    step=step,
                    split="route_train",
                )

    with CudaActivityMonitor() as monitor:
        training = train_pointer_diffusion(
            dataset=dataset,
            train_indices=split_indices["route_train"],
            calibration_indices=split_indices["route_calibration"],
            targets=targets,
            model=rescue_model,
            paired_one_shot_model=paired_model,
            config=config,
            device=device,
            checkpoint_directory=run_paths.checkpoints / "rollout_rescue",
            delta=delta,
            temperature=temperature,
            seed=seed + 62001,
            smoke=args.stage == "smoke",
            corruption_mode="self_rollout",
            metric_callback=callback,
        )
    gpu_activity = monitor.summary()
    load_pointer_checkpoint(
        path=training["best_checkpoint"],
        model=rescue_model,
        device=device,
    )
    calibration = evaluate_pointer_diffusion_subset(
        split_name="route_calibration",
        dataset=dataset,
        indices=split_indices["route_calibration"],
        targets=targets,
        one_shot_model=paired_model,
        diffusion_model=rescue_model,
        config=config,
        device=device,
        delta=delta,
        temperature=temperature,
    )
    _write_json(
        run_paths.artifacts / "training_summary.json",
        training,
    )
    _write_json(
        run_paths.artifacts / "calibration_evaluation.json",
        calibration,
    )
    _write_json(run_paths.artifacts / "gpu_activity.json", gpu_activity)
    _write_csv(
        run_paths.artifacts / "training_history.csv",
        training["history"],
    )
    if args.stage == "smoke":
        manifest = {
            "schema_version": 1,
            "stage": "pointer-route-rollout-rescue-smoke",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "initial_state": initial_state,
            "training": training,
            "gpu_activity": gpu_activity,
            "calibration": calibration,
            "test_loaded": False,
        }
        logger.write_manifest(manifest)
        logger.log_event("pointer_route_rollout_rescue_smoke_completed")
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "status": "rollout_rescue_smoke_passed",
                    "initial_state": initial_state,
                    "steps": training["steps_completed"],
                    "device_audit": training["device_audit"],
                    "gpu_activity": gpu_activity,
                    "peak_reserved_mib": training["peak_reserved_mib"],
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
        one_shot_model=paired_model,
        diffusion_model=rescue_model,
        config=config,
        device=device,
        delta=delta,
        temperature=temperature,
    )
    latency = measure_diffusion_latency(
        dataset=dataset,
        indices=split_indices["route_validation"],
        model=rescue_model,
        config=config,
        device=device,
    )
    decision = _decision(
        validation=validation,
        phase3=phase3,
        training=training,
        latency=latency,
        initial_state=initial_state,
        gpu_activity=gpu_activity,
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
            *_renamed_rows(
                split="route_validation",
                evaluation=phase3["validation"],
                family="plain_d0",
                scenarios=False,
            ),
            *_renamed_rows(
                split="route_validation",
                evaluation=validation,
                family="rollout_d1",
                scenarios=False,
            ),
        ],
    )
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        [
            *_renamed_rows(
                split="route_validation",
                evaluation=phase3["validation"],
                family="plain_d0",
                scenarios=True,
            ),
            *_renamed_rows(
                split="route_validation",
                evaluation=validation,
                family="rollout_d1",
                scenarios=True,
            ),
        ],
    )
    _write_csv(
        run_paths.artifacts / "trajectory_metrics.csv",
        _trajectory_rows("route_validation", validation),
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        _report(
            validation=validation,
            phase3=phase3,
            training=training,
            latency=latency,
            decision=decision,
            gpu_activity=gpu_activity,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "pointer-route-rollout-corruption-rescue",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "phase01_lock": phase01,
        "phase3_lock": {
            "artifact_hashes": phase3["artifact_hashes"],
            "decision": phase3["decision"],
        },
        "initial_state": initial_state,
        "training": training,
        "gpu_activity": gpu_activity,
        "calibration": calibration,
        "validation": validation,
        "latency": latency,
        "decision": decision,
        "test_loaded": False,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "pointer_route_rollout_rescue_completed",
        classification=decision["classification"],
        test_loaded=False,
    )
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "training": {
                    "steps_completed": training["steps_completed"],
                    "best_step": training["best_step"],
                    "elapsed_seconds": training["elapsed_seconds"],
                    "peak_reserved_mib": training["peak_reserved_mib"],
                },
                "gpu_activity": gpu_activity,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
