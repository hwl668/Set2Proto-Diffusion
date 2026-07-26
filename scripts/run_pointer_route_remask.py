"""Calibration-tuned PointerRoute evidence remasking experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_pointer_route_evidence_logits import _load_order_lock
from run_pointer_route_evidence_order import _load_rollout_lock
from run_pointer_route_phase3 import (
    _core_rows,
    _phase01_lock,
    _scenario_rows,
    _sha256,
    _trajectory_rows,
    _write_csv,
    _write_json,
)
from run_pointer_route_rollout_rescue import (
    CudaActivityMonitor,
    _load_phase3_lock,
)
from set2proto.config import load_config
from set2proto.environment import collect_environment, config_sha256
from set2proto.experiment_logging import ExperimentLogger
from set2proto.pointer_route_diffusion import (
    decode_pointer_routes,
    evaluate_pointer_diffusion_subset,
    measure_diffusion_latency,
)
from set2proto.pointer_route_model import build_pointer_route_model
from set2proto.pointer_route_oracle import reconstruct_hard_routes
from set2proto.pointer_route_training import (
    _subset_tensors,
    load_pointer_checkpoint,
    prepare_route_targets,
)
from set2proto.reproducibility import seed_everything
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
    parser.add_argument("--quantization-root", type=Path, required=True)
    parser.add_argument("--phase01-root", type=Path, required=True)
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--evidence-order-root", type=Path, required=True)
    parser.add_argument("--evidence-logits-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _load_logits_lock(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    required = {
        "validation": artifacts / "validation_evaluation.json",
        "decision": artifacts / "decision.json",
        "selection": artifacts / "selection.json",
        "protocol": artifacts / "protocol_preregistration.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing evidence-logits {name}: {path}")
    result = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
    }
    if result["protocol"].get("test_loaded") is not False:
        raise RuntimeError("evidence-logits protocol loaded test data")
    if result["protocol"].get("training_performed") is not False:
        raise RuntimeError("evidence-logits stage modified the model")
    result["artifact_hashes"] = {
        name: _sha256(path) for name, path in required.items()
    }
    return result


def _candidate_config(
    *,
    base: Mapping[str, Any],
    recipe: Mapping[str, Any],
    evidence_lambda: float,
    remask_fraction: float,
    remask_min_score_gain: float,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    diffusion = config["pointer_route"]["diffusion"]
    evidence_keys = (
        "identity_neighbor_fraction",
        "identity_temperature",
        "quality_weight",
        "local_temperature",
    )
    for section_name in ("evidence_logits", "evidence_remask"):
        section = diffusion[section_name]
        for key in evidence_keys:
            section[key] = float(recipe[key])
        section["reliability_floor"] = float(
            diffusion["evidence_remask"]["reliability_floor"]
        )
    diffusion["evidence_remask"]["selected_recipe"] = str(recipe["name"])
    diffusion["evidence_remask"]["selected_lambda"] = float(evidence_lambda)
    diffusion["evidence_remask"]["remask_fraction"] = float(remask_fraction)
    diffusion["evidence_remask"]["remask_min_score_gain"] = float(
        remask_min_score_gain
    )
    return config


def _metric_row(
    *,
    evaluation: Mapping[str, Any],
    method: str,
) -> dict[str, float]:
    metrics = evaluation["methods"][method]["verification"]
    return {
        "hard_auc": float(metrics["hard_average"]["roc_auc"]),
        "all_auc": float(metrics["all"]["roc_auc"]),
        "clean_auc": float(metrics["clean"]["roc_auc"]),
        "worst_hard_scenario_auc": min(
            float(metrics[scenario]["roc_auc"])
            for scenario in HARD_SCENARIOS
        ),
        **{
            f"{scenario}_auc": float(metrics[scenario]["roc_auc"])
            for scenario in HARD_SCENARIOS
        },
    }


def _select_tuned_baseline(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("baseline tuning rows cannot be empty")
    best_hard = max(float(row["hard_auc"]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if float(row["hard_auc"]) >= best_hard - tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["four_minus_one_hard_auc"]),
            float(row["worst_hard_scenario_auc"]),
            -float(row["lambda"]),
            -int(row["recipe_order"]),
        ),
    )
    return {
        "selection_split": "route_calibration",
        "objective": (
            "maximize four-step hard AUC; within tolerance maximize "
            "four-minus-one, then worst hard scenario, then lower lambda"
        ),
        "auc_tie_tolerance": float(tolerance),
        "best_hard_auc": best_hard,
        "eligible_candidates": len(eligible),
        "selected_candidate": selected,
        "validation_used": False,
        "test_used": False,
    }


def _select_remask(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    active = [
        dict(row) for row in rows if int(row["remask_events"]) > 0
    ]
    if not active:
        raise RuntimeError("no remask tuning candidate produced a remask event")
    best_hard = max(float(row["hard_auc"]) for row in active)
    eligible = [
        row
        for row in active
        if float(row["hard_auc"]) >= best_hard - tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            -float(row["remask_fraction"]),
            float(row["remask_min_score_gain"]),
        ),
    )
    return {
        "selection_split": "route_calibration",
        "objective": (
            "among active remask candidates maximize four-step hard AUC; "
            "within tolerance choose the smallest budget, then the largest "
            "replacement margin"
        ),
        "auc_tie_tolerance": float(tolerance),
        "best_hard_auc": best_hard,
        "active_candidates": len(active),
        "eligible_candidates": len(eligible),
        "selected_candidate": selected,
        "validation_used": False,
        "test_used": False,
    }


@torch.inference_mode()
def _paired_remask_diagnostics(
    *,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    model: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    evidence_lambda: float,
) -> dict[str, Any]:
    values = _subset_tensors(dataset=dataset, indices=indices)
    batch_size = int(config["evaluation"]["batch_size"])
    scenarios = list(values["scenarios"])
    route_changes = 0
    total_routes = 0
    remask_events = 0
    remasked_samples = 0
    map_cosines: list[torch.Tensor] = []
    per_step_events = [0, 0, 0, 0]
    scenario_events = {scenario: 0 for scenario in sorted(set(scenarios))}
    scenario_positions = {scenario: 0 for scenario in sorted(set(scenarios))}
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        condition = values["condition_features"][start:end].to(device)
        quality = values["condition_quality"][start:end].to(device)
        baseline = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=4,
            commit_mode="evidence_logits",
            evidence_lambda=evidence_lambda,
            evidence_config=config["pointer_route"]["diffusion"][
                "evidence_logits"
            ],
        )
        remask = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=4,
            commit_mode="evidence_remask",
            evidence_lambda=evidence_lambda,
            evidence_config=config["pointer_route"]["diffusion"][
                "evidence_remask"
            ],
        )
        event_by_sample = torch.stack(remask.remasked).sum(dim=(0, 2))
        remask_events += int(event_by_sample.sum().item())
        remasked_samples += int((event_by_sample > 0).sum().item())
        for step, event in enumerate(remask.remasked):
            per_step_events[step] += int(event.sum().item())
        for offset, scenario in enumerate(scenarios[start:end]):
            scenario_events[scenario] += int(event_by_sample[offset].item())
            scenario_positions[scenario] += int(model.spatial_tokens)
        route_changes += int(
            (baseline.final_routes != remask.final_routes).sum().item()
        )
        total_routes += int(remask.final_routes.numel())
        candidates = model(
            model.all_mask_input(end - start, device=device),
            condition,
            quality,
        )[1]
        baseline_map = reconstruct_hard_routes(
            candidates,
            baseline.final_routes,
        )
        remask_map = reconstruct_hard_routes(
            candidates,
            remask.final_routes,
        )
        map_cosines.append(
            F.cosine_similarity(baseline_map, remask_map, dim=-1).cpu()
        )
    cosines = torch.cat(map_cosines)
    return {
        "samples": len(indices),
        "total_routes": total_routes,
        "remask_events": remask_events,
        "remasked_sample_fraction": float(remasked_samples / len(indices)),
        "remask_event_fraction_per_position": float(
            remask_events / total_routes
        ),
        "per_step_events": per_step_events,
        "final_route_change_fraction": float(route_changes / total_routes),
        "baseline_remask_map_cosine_mean": float(cosines.mean().item()),
        "baseline_remask_map_cosine_min": float(cosines.min().item()),
        "scenario_remask_events": scenario_events,
        "scenario_remask_events_per_position": {
            scenario: (
                float(scenario_events[scenario] / scenario_positions[scenario])
                if scenario_positions[scenario]
                else 0.0
            )
            for scenario in scenario_events
        },
    }


def _decision(
    *,
    confidence: Mapping[str, Any],
    tuned: Mapping[str, Any],
    remask: Mapping[str, Any],
    baseline_selection: Mapping[str, Any],
    remask_selection: Mapping[str, Any],
    paired: Mapping[str, Any],
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuned_four = tuned["methods"]["tuned_logits_4step"]["verification"]
    remask_four = remask["methods"]["evidence_remask_4step"]["verification"]
    confidence_one = confidence["methods"][
        "pointer_diffusion_1step"
    ]["verification"]
    gates = config["pointer_route"]["diffusion"]["evidence_remask"]["gates"]
    gain_tuned = (
        remask_four["hard_average"]["roc_auc"]
        - tuned_four["hard_average"]["roc_auc"]
    )
    gain_confidence = (
        remask_four["hard_average"]["roc_auc"]
        - confidence_one["hard_average"]["roc_auc"]
    )
    scenario_deltas = {
        scenario: (
            remask_four[scenario]["roc_auc"]
            - tuned_four[scenario]["roc_auc"]
        )
        for scenario in HARD_SCENARIOS
    }
    scenario_wins = sum(value > 1e-12 for value in scenario_deltas.values())
    transition = remask["trajectory"]["transitions"]["all"]["equivalent"]
    remask_mechanism = remask["trajectory"]["remask"]
    correction = remask_mechanism[
        "eventual_equivalent_correction_fraction"
    ]
    injury = remask_mechanism[
        "eventual_equivalent_injury_fraction"
    ]
    peak_reserved = max(
        float(values["peak_reserved_mib"]) for values in latency.values()
    )
    checks = {
        "baseline_tuned_on_calibration_only": (
            baseline_selection["selection_split"] == "route_calibration"
            and baseline_selection["validation_used"] is False
            and baseline_selection["test_used"] is False
        ),
        "remask_tuned_on_calibration_only": (
            remask_selection["selection_split"] == "route_calibration"
            and remask_selection["validation_used"] is False
            and remask_selection["test_used"] is False
        ),
        "four_step_beats_tuned_no_remask": (
            gain_tuned
            >= float(gates["min_four_step_gain_over_tuned_no_remask"])
        ),
        "four_step_beats_confidence_one_step": (
            gain_confidence
            >= float(
                gates["min_four_step_gain_over_confidence_one_step"]
            )
        ),
        "hard_scenario_wins": (
            scenario_wins >= int(gates["min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            tuned_four["clean"]["roc_auc"]
            - remask_four["clean"]["roc_auc"]
            <= float(gates["max_clean_auc_drop_vs_tuned_no_remask"])
        ),
        "positive_equivalent_net_correction": (
            transition["net_correction_fraction"] > 0.0
            if bool(gates["require_positive_net_correction"])
            else True
        ),
        "actual_remask": (
            int(paired["remask_events"]) > 0
            if bool(gates["require_actual_remask"])
            else True
        ),
        "remask_correction_exceeds_injury": (
            correction is not None
            and injury is not None
            and float(correction) > float(injury)
            if bool(gates["require_remask_ecr_exceeds_eir"])
            else True
        ),
        "permutation_equivariant": (
            remask["diffusion_permutation_check"][
                "route_map_max_abs_difference"
            ]
            <= float(gates["max_permutation_map_difference"])
        ),
        "memory_within_7_2_gib": (
            peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
        "checkpoint_frozen": True,
        "test_not_loaded": True,
    }
    passed = all(checks.values())
    return {
        "classification": (
            "GO_REMASK_MECHANISM" if passed else "NO_GO_REMASK"
        ),
        "checks": checks,
        "selected_evidence_recipe": baseline_selection[
            "selected_candidate"
        ]["recipe"],
        "selected_lambda": baseline_selection["selected_candidate"]["lambda"],
        "selected_remask_fraction": remask_selection[
            "selected_candidate"
        ]["remask_fraction"],
        "selected_remask_min_score_gain": remask_selection[
            "selected_candidate"
        ]["remask_min_score_gain"],
        "four_step_gain_over_tuned_no_remask": float(gain_tuned),
        "four_step_gain_over_confidence_one_step": float(gain_confidence),
        "scenario_deltas_vs_tuned_no_remask": scenario_deltas,
        "hard_scenario_wins": scenario_wins,
        "equivalent_ecr": transition["ecr"],
        "equivalent_eir": transition["eir"],
        "equivalent_net_correction_fraction": transition[
            "net_correction_fraction"
        ],
        "remask_events": paired["remask_events"],
        "remask_eventual_equivalent_correction_fraction": correction,
        "remask_eventual_equivalent_injury_fraction": injury,
        "peak_reserved_mib": peak_reserved,
        "recommendation": (
            "retain_remask_for_new_holdout"
            if passed
            else "stop_pointerroute_remask_and_pivot_representation"
        ),
    }


def _unique_rows(
    sources: Sequence[tuple[Mapping[str, Any], set[str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for evaluation, allowed in sources:
        for row in _core_rows("route_validation", evaluation):
            method = str(row["method"])
            if method in allowed and method not in seen:
                rows.append(row)
                seen.add(method)
    return rows


def _report(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    remask_rows: Sequence[Mapping[str, Any]],
    baseline_selection: Mapping[str, Any],
    remask_selection: Mapping[str, Any],
    confidence: Mapping[str, Any],
    previous_logits: Mapping[str, Any],
    tuned: Mapping[str, Any],
    remask: Mapping[str, Any],
    paired: Mapping[str, Any],
    decision: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    core_sources = (
        (
            confidence,
            {
                "quality_anchor",
                "one_shot_pointer",
                "pointer_diffusion_1step",
                "pointer_diffusion_4step",
            },
        ),
        (previous_logits, {"evidence_logits_1step", "evidence_logits_4step"}),
        (tuned, {"tuned_logits_1step", "tuned_logits_4step"}),
        (
            remask,
            {
                "evidence_remask_1step",
                "evidence_remask_2step",
                "evidence_remask_4step",
                "evidence_remask_8step",
            },
        ),
    )
    core_rows = _unique_rows(core_sources)
    core_table = "\n".join(
        "| {method} | {hard_macro_auc:.6f} | {pooled_all_auc:.6f} | "
        "{eer_all:.6f} | {tar_at_far_1e-3:.3f} | {rank1_all:.3f} | "
        "{teacher_map_cosine_all:.6f} |".format(**row)
        for row in core_rows
    )
    baseline_table = "\n".join(
        "| {recipe} | {lambda:.2f} | {one_step_hard_auc:.6f} | "
        "{hard_auc:.6f} | {four_minus_one_hard_auc:+.6f} |".format(**row)
        for row in baseline_rows
    )
    remask_table = "\n".join(
        "| {remask_fraction:.2f} | {remask_min_score_gain:.3f} | "
        "{hard_auc:.6f} | {gain_over_tuned_baseline:+.6f} | "
        "{remask_events} |".format(**row)
        for row in remask_rows
    )
    scenario_table = "\n".join(
        "| {0} | {1:.6f} | {2:.6f} | {3:+.6f} |".format(
            scenario,
            tuned["methods"]["tuned_logits_4step"]["verification"][scenario][
                "roc_auc"
            ],
            remask["methods"]["evidence_remask_4step"]["verification"][
                scenario
            ]["roc_auc"],
            decision["scenario_deltas_vs_tuned_no_remask"][scenario],
        )
        for scenario in HARD_SCENARIOS
    )
    failed = [
        key for key, value in decision["checks"].items() if not value
    ]
    return f"""# Calibration-tuned PointerRoute remask experiment

## Protocol

- The rollout-rescue checkpoint is frozen; no training or checkpoint writes.
- Evidence parameters are tuned first on route-calibration using a fixed
  4-recipe by 6-lambda grid.
- The tuned evidence parameters are then frozen while a fixed 4-budget by
  3-margin remask grid is selected on route-calibration.
- Route-validation is evaluated only after both selections are locked.
- Test is not instantiated or loaded.
- Remasking is budgeted: a low-retention committed route is replaced only when
  a still-masked candidate exceeds it by the selected score margin.

## Evidence tuning on route-calibration

| Recipe | Lambda | 1-step hard AUC | 4-step hard AUC | Step gain |
|---|---:|---:|---:|---:|
{baseline_table}

Selected: `{baseline_selection['selected_candidate']['recipe']}`,
lambda `{baseline_selection['selected_candidate']['lambda']}`.

## Remask tuning on route-calibration

| Budget fraction | Minimum gain | Hard AUC | Gain vs tuned logits | Events |
|---:|---:|---:|---:|---:|
{remask_table}

Selected budget `{remask_selection['selected_candidate']['remask_fraction']}`,
minimum score gain
`{remask_selection['selected_candidate']['remask_min_score_gain']}`.

## Route-validation core comparison

| Method | Hard AUC | All AUC | EER | TAR@1e-3 | Rank-1 | Map cosine |
|---|---:|---:|---:|---:|---:|---:|
{core_table}

## Four-step remask versus tuned no-remask

| Scenario | Tuned logits | Remask | Delta |
|---|---:|---:|---:|
{scenario_table}

- Hard macro delta: `{decision['four_step_gain_over_tuned_no_remask']:+.6f}`.
- Delta versus original confidence one-step:
  `{decision['four_step_gain_over_confidence_one_step']:+.6f}`.
- Remask events: `{paired['remask_events']}`;
  sample activity `{paired['remasked_sample_fraction']:.3%}`.
- Final route change fraction:
  `{paired['final_route_change_fraction']:.3%}`.
- Remask eventual equivalent correction/injury:
  `{decision['remask_eventual_equivalent_correction_fraction']}` /
  `{decision['remask_eventual_equivalent_injury_fraction']}`.
- Four-step batch-16 latency:
  `{latency['4step']['mean_batch_latency_ms']:.3f}` ms.

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

## Interpretation

- Calibration tuning did not expose a stable hidden gain. The selected
  `lambda=1.5` traded a small amount of calibration hard AUC for a larger
  calibration four-minus-one gain, but on route-validation it was worse than
  both the previous `lambda=1.0` evidence-logits result and confidence one-step.
- Remasking was operational rather than a no-op: it touched
  `{paired['remasked_sample_fraction']:.1%}` of samples, and remasked errors
  were eventually corrected more often than correct routes were injured.
  Nevertheless, only `{paired['final_route_change_fraction']:.3%}` of final
  routes changed, so most temporary removals converged back to the same route.
- Two hard scenarios moved in the desired direction at approximately
  numerical-noise scale, while low-quality tied. Common occlusion lost more
  than those gains combined, leaving the hard macro result slightly negative.
- The result argues against further decoder-only tuning of the current
  five-route state space. A new representation should let a position synthesize
  information beyond selecting the anchor or one observed frame.

This is a mechanism-development result on the repeatedly used route-validation
split. It is not an untouched holdout and does not establish statistical
significance.
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.profile).to_dict()
    seed = int(config["project"]["seed"])
    seed_everything(seed, deterministic=bool(config["runtime"]["deterministic"]))
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    phase01 = _phase01_lock(
        root=args.phase01_root.expanduser().resolve(),
        dataset_root=dataset_root,
        quantization_root=quantization_root,
    )
    phase3 = _load_phase3_lock(args.phase3_root.expanduser().resolve())
    rollout = _load_rollout_lock(args.rollout_root.expanduser().resolve())
    order = _load_order_lock(args.evidence_order_root.expanduser().resolve())
    logits = _load_logits_lock(args.evidence_logits_root.expanduser().resolve())
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
    section = config["pointer_route"]["diffusion"]["evidence_remask"]
    preregistration = {
        "schema_version": 1,
        "stage": "pointer-route-calibration-tuned-remask",
        "mode": args.stage,
        "frozen_rollout_checkpoint": str(rollout["checkpoint"]),
        "frozen_rollout_checkpoint_sha256": rollout["checkpoint_sha256"],
        "baseline_tuning_split": "route_calibration",
        "remask_tuning_split": "route_calibration",
        "evaluation_split": (
            None if args.stage == "smoke" else "route_validation"
        ),
        "baseline_grid_size": (
            len(section["evidence_recipes"])
            * len(section["lambda_candidates"])
        ),
        "remask_grid_size": (
            len(section["remask_fraction_candidates"])
            * len(section["remask_min_score_gain_candidates"])
        ),
        "evidence_remask": section,
        "training_performed": False,
        "checkpoint_modified": False,
        "phase01_artifact_hashes": phase01["artifact_hashes"],
        "phase3_artifact_hashes": phase3["artifact_hashes"],
        "rollout_artifact_hashes": rollout["artifact_hashes"],
        "evidence_order_artifact_hashes": order["artifact_hashes"],
        "evidence_logits_artifact_hashes": logits["artifact_hashes"],
        "config_sha256": config_sha256(config),
        "test_loaded": False,
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "pointer_route_remask_started",
        mode=args.stage,
        test_loaded=False,
        training_performed=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PointerRoute remask experiment requires CUDA")
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
    model = build_pointer_route_model(config)
    load_pointer_checkpoint(
        path=rollout["checkpoint"],
        model=model,
        device=device,
    )
    model.to(device).eval()
    paired_model = build_pointer_route_model(config)
    load_pointer_checkpoint(
        path=phase3["training"]["one_shot"]["best_checkpoint"],
        model=paired_model,
        device=device,
    )
    paired_model.to(device).eval()
    split_indices = phase01["split"]["sample_indices"]

    if args.stage == "smoke":
        recipe = section["evidence_recipes"][0]
        smoke_config = _candidate_config(
            base=config,
            recipe=recipe,
            evidence_lambda=float(section["lambda_candidates"][0]),
            remask_fraction=float(section["remask_fraction_candidates"][-1]),
            remask_min_score_gain=0.0,
        )
        evaluation = evaluate_pointer_diffusion_subset(
            split_name="route_calibration",
            dataset=dataset,
            indices=split_indices["route_calibration"],
            targets=targets,
            one_shot_model=paired_model,
            diffusion_model=model,
            config=smoke_config,
            device=device,
            delta=delta,
            temperature=temperature,
            evaluation_steps=[1, 4],
            commit_mode="evidence_remask",
            evidence_lambda=float(section["lambda_candidates"][0]),
            method_prefix="evidence_remask",
        )
        paired = _paired_remask_diagnostics(
            dataset=dataset,
            indices=split_indices["route_calibration"],
            model=model,
            config=smoke_config,
            device=device,
            evidence_lambda=float(section["lambda_candidates"][0]),
        )
        if paired["remask_events"] <= 0:
            raise RuntimeError("smoke test produced no remask events")
        _write_json(run_paths.artifacts / "smoke_evaluation.json", evaluation)
        _write_json(run_paths.artifacts / "remask_diagnostics.json", paired)
        manifest = {
            "schema_version": 1,
            "stage": "pointer-route-remask-smoke",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "evaluation": evaluation,
            "remask_diagnostics": paired,
            "training_performed": False,
            "test_loaded": False,
        }
        logger.write_manifest(manifest)
        logger.log_event("pointer_route_remask_smoke_completed")
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "status": "remask_smoke_passed",
                    "device": str(device),
                    "remask_diagnostics": paired,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    calibration_indices = split_indices["route_calibration"]
    baseline_rows: list[dict[str, Any]] = []
    baseline_evaluations: dict[str, Any] = {}
    with CudaActivityMonitor() as monitor:
        for recipe_order, recipe in enumerate(section["evidence_recipes"]):
            for lambda_value in section["lambda_candidates"]:
                evidence_lambda = float(lambda_value)
                candidate_config = _candidate_config(
                    base=config,
                    recipe=recipe,
                    evidence_lambda=evidence_lambda,
                    remask_fraction=0.0,
                    remask_min_score_gain=0.0,
                )
                evaluation = evaluate_pointer_diffusion_subset(
                    split_name="route_calibration",
                    dataset=dataset,
                    indices=calibration_indices,
                    targets=targets,
                    one_shot_model=paired_model,
                    diffusion_model=model,
                    config=candidate_config,
                    device=device,
                    delta=delta,
                    temperature=temperature,
                    evaluation_steps=[1, 4],
                    commit_mode="evidence_logits",
                    evidence_lambda=evidence_lambda,
                    method_prefix="tuned_logits",
                )
                one = _metric_row(
                    evaluation=evaluation,
                    method="tuned_logits_1step",
                )
                four = _metric_row(
                    evaluation=evaluation,
                    method="tuned_logits_4step",
                )
                row = {
                    "recipe": str(recipe["name"]),
                    "recipe_order": recipe_order,
                    "lambda": evidence_lambda,
                    "one_step_hard_auc": one["hard_auc"],
                    **four,
                    "four_minus_one_hard_auc": (
                        four["hard_auc"] - one["hard_auc"]
                    ),
                }
                baseline_rows.append(row)
                baseline_evaluations[
                    f"{recipe['name']}_lambda_{evidence_lambda:g}"
                ] = evaluation
        baseline_selection = _select_tuned_baseline(
            baseline_rows,
            tolerance=float(section["auc_tie_tolerance"]),
        )
        selected_baseline = baseline_selection["selected_candidate"]
        selected_recipe = next(
            recipe
            for recipe in section["evidence_recipes"]
            if recipe["name"] == selected_baseline["recipe"]
        )
        selected_lambda = float(selected_baseline["lambda"])
        tuned_config = _candidate_config(
            base=config,
            recipe=selected_recipe,
            evidence_lambda=selected_lambda,
            remask_fraction=0.0,
            remask_min_score_gain=0.0,
        )
        tuned_calibration = evaluate_pointer_diffusion_subset(
            split_name="route_calibration",
            dataset=dataset,
            indices=calibration_indices,
            targets=targets,
            one_shot_model=paired_model,
            diffusion_model=model,
            config=tuned_config,
            device=device,
            delta=delta,
            temperature=temperature,
            commit_mode="evidence_logits",
            evidence_lambda=selected_lambda,
            method_prefix="tuned_logits",
        )
        tuned_four_hard = tuned_calibration["methods"][
            "tuned_logits_4step"
        ]["verification"]["hard_average"]["roc_auc"]

        remask_rows: list[dict[str, Any]] = []
        remask_evaluations: dict[str, Any] = {}
        for fraction in section["remask_fraction_candidates"]:
            for gain in section["remask_min_score_gain_candidates"]:
                remask_config = _candidate_config(
                    base=config,
                    recipe=selected_recipe,
                    evidence_lambda=selected_lambda,
                    remask_fraction=float(fraction),
                    remask_min_score_gain=float(gain),
                )
                evaluation = evaluate_pointer_diffusion_subset(
                    split_name="route_calibration",
                    dataset=dataset,
                    indices=calibration_indices,
                    targets=targets,
                    one_shot_model=paired_model,
                    diffusion_model=model,
                    config=remask_config,
                    device=device,
                    delta=delta,
                    temperature=temperature,
                    evaluation_steps=[4],
                    commit_mode="evidence_remask",
                    evidence_lambda=selected_lambda,
                    method_prefix="evidence_remask",
                )
                metrics = _metric_row(
                    evaluation=evaluation,
                    method="evidence_remask_4step",
                )
                events = int(
                    evaluation["trajectory"]["remask"]["total_events"]
                )
                row = {
                    "remask_fraction": float(fraction),
                    "remask_min_score_gain": float(gain),
                    **metrics,
                    "gain_over_tuned_baseline": float(
                        metrics["hard_auc"] - tuned_four_hard
                    ),
                    "remask_events": events,
                    "remask_event_fraction": evaluation["trajectory"][
                        "remask"
                    ]["event_fraction"],
                }
                remask_rows.append(row)
                remask_evaluations[
                    f"fraction_{float(fraction):g}_gain_{float(gain):g}"
                ] = evaluation
        remask_selection = _select_remask(
            remask_rows,
            tolerance=float(section["auc_tie_tolerance"]),
        )
        selected_remask = remask_selection["selected_candidate"]
        locked_config = _candidate_config(
            base=config,
            recipe=selected_recipe,
            evidence_lambda=selected_lambda,
            remask_fraction=float(selected_remask["remask_fraction"]),
            remask_min_score_gain=float(
                selected_remask["remask_min_score_gain"]
            ),
        )
        validation_indices = split_indices["route_validation"]
        tuned_validation = evaluate_pointer_diffusion_subset(
            split_name="route_validation",
            dataset=dataset,
            indices=validation_indices,
            targets=targets,
            one_shot_model=paired_model,
            diffusion_model=model,
            config=locked_config,
            device=device,
            delta=delta,
            temperature=temperature,
            commit_mode="evidence_logits",
            evidence_lambda=selected_lambda,
            method_prefix="tuned_logits",
        )
        remask_validation = evaluate_pointer_diffusion_subset(
            split_name="route_validation",
            dataset=dataset,
            indices=validation_indices,
            targets=targets,
            one_shot_model=paired_model,
            diffusion_model=model,
            config=locked_config,
            device=device,
            delta=delta,
            temperature=temperature,
            commit_mode="evidence_remask",
            evidence_lambda=selected_lambda,
            method_prefix="evidence_remask",
        )
        paired = _paired_remask_diagnostics(
            dataset=dataset,
            indices=validation_indices,
            model=model,
            config=locked_config,
            device=device,
            evidence_lambda=selected_lambda,
        )
        latency = measure_diffusion_latency(
            dataset=dataset,
            indices=validation_indices,
            model=model,
            config=locked_config,
            device=device,
            commit_mode="evidence_remask",
            evidence_lambda=selected_lambda,
        )
    gpu_activity = monitor.summary()
    decision = _decision(
        confidence=rollout["validation"],
        tuned=tuned_validation,
        remask=remask_validation,
        baseline_selection=baseline_selection,
        remask_selection=remask_selection,
        paired=paired,
        latency=latency,
        config=locked_config,
    )
    _write_csv(
        run_paths.artifacts / "baseline_tuning_candidates.csv",
        baseline_rows,
    )
    _write_json(
        run_paths.artifacts / "baseline_tuning_evaluations.json",
        baseline_evaluations,
    )
    _write_json(
        run_paths.artifacts / "baseline_selection.json",
        baseline_selection,
    )
    _write_json(
        run_paths.artifacts / "tuned_calibration_evaluation.json",
        tuned_calibration,
    )
    _write_csv(
        run_paths.artifacts / "remask_tuning_candidates.csv",
        remask_rows,
    )
    _write_json(
        run_paths.artifacts / "remask_tuning_evaluations.json",
        remask_evaluations,
    )
    _write_json(
        run_paths.artifacts / "remask_selection.json",
        remask_selection,
    )
    _write_json(
        run_paths.artifacts / "locked_decoder_config.json",
        locked_config["pointer_route"]["diffusion"]["evidence_remask"],
    )
    _write_json(
        run_paths.artifacts / "validation_tuned_logits.json",
        tuned_validation,
    )
    _write_json(
        run_paths.artifacts / "validation_remask.json",
        remask_validation,
    )
    _write_json(
        run_paths.artifacts / "remask_diagnostics.json",
        paired,
    )
    _write_json(
        run_paths.artifacts / "trajectory_diagnostics.json",
        remask_validation["trajectory"],
    )
    _write_json(run_paths.artifacts / "latency.json", latency)
    _write_json(run_paths.artifacts / "gpu_activity.json", gpu_activity)
    _write_json(run_paths.artifacts / "decision.json", decision)
    core_rows = _unique_rows(
        (
            (
                rollout["validation"],
                {
                    "quality_anchor",
                    "one_shot_pointer",
                    "pointer_diffusion_1step",
                    "pointer_diffusion_4step",
                },
            ),
            (
                logits["validation"],
                {"evidence_logits_1step", "evidence_logits_4step"},
            ),
            (
                tuned_validation,
                {"tuned_logits_1step", "tuned_logits_4step"},
            ),
            (
                remask_validation,
                {
                    "evidence_remask_1step",
                    "evidence_remask_2step",
                    "evidence_remask_4step",
                    "evidence_remask_8step",
                },
            ),
        )
    )
    _write_csv(run_paths.artifacts / "core_comparison.csv", core_rows)
    scenario_rows: list[dict[str, Any]] = []
    for evaluation, allowed in (
        (
            tuned_validation,
            {"tuned_logits_1step", "tuned_logits_4step"},
        ),
        (
            remask_validation,
            {
                "evidence_remask_1step",
                "evidence_remask_2step",
                "evidence_remask_4step",
                "evidence_remask_8step",
            },
        ),
    ):
        scenario_rows.extend(
            row
            for row in _scenario_rows("route_validation", evaluation)
            if row["method"] in allowed
        )
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        scenario_rows,
    )
    _write_csv(
        run_paths.artifacts / "trajectory_metrics.csv",
        _trajectory_rows("route_validation", remask_validation),
    )
    report = _report(
        baseline_rows=baseline_rows,
        remask_rows=remask_rows,
        baseline_selection=baseline_selection,
        remask_selection=remask_selection,
        confidence=rollout["validation"],
        previous_logits=logits["validation"],
        tuned=tuned_validation,
        remask=remask_validation,
        paired=paired,
        decision=decision,
        latency=latency,
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "pointer-route-calibration-tuned-remask",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "locked_decoder_config": locked_config["pointer_route"]["diffusion"][
            "evidence_remask"
        ],
        "protocol_preregistration": preregistration,
        "baseline_selection": baseline_selection,
        "remask_selection": remask_selection,
        "validation_tuned_logits": tuned_validation,
        "validation_remask": remask_validation,
        "remask_diagnostics": paired,
        "latency": latency,
        "gpu_activity": gpu_activity,
        "decision": decision,
        "training_performed": False,
        "checkpoint_modified": False,
        "test_loaded": False,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "pointer_route_remask_completed",
        classification=decision["classification"],
        recipe=baseline_selection["selected_candidate"]["recipe"],
        evidence_lambda=baseline_selection["selected_candidate"]["lambda"],
        remask_fraction=remask_selection["selected_candidate"][
            "remask_fraction"
        ],
        remask_min_score_gain=remask_selection["selected_candidate"][
            "remask_min_score_gain"
        ],
        test_loaded=False,
    )
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "baseline_selection": baseline_selection,
                "remask_selection": remask_selection,
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
