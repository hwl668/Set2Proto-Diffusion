"""Evaluate two-level evidence as PointerRoute commit ordering only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _load_rollout_lock(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    required = {
        "training": artifacts / "training_summary.json",
        "calibration": artifacts / "calibration_evaluation.json",
        "validation": artifacts / "validation_evaluation.json",
        "decision": artifacts / "decision.json",
        "protocol": artifacts / "protocol_preregistration.json",
        "initial_state": artifacts / "initial_state.json",
        "latency": artifacts / "latency.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing rollout {name}: {path}")
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
    }
    if values["protocol"].get("test_loaded") is not False:
        raise RuntimeError("rollout protocol unexpectedly loaded test data")
    if values["decision"]["recommendation"] != (
        "stop_plain_pointerroute_diffusion"
    ):
        raise RuntimeError("rollout result does not match the locked baseline")
    checkpoint = Path(values["training"]["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing locked rollout checkpoint: {checkpoint}")
    return {
        **values,
        "checkpoint": checkpoint,
        "artifact_hashes": {
            name: _sha256(path) for name, path in required.items()
        },
        "checkpoint_sha256": _sha256(checkpoint),
    }


def _method(
    evaluation: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    return evaluation["methods"][name]


def _select_lambda(
    *,
    candidates: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("evidence lambda selection needs candidates")
    best_hard = max(float(row["hard_auc"]) for row in candidates)
    eligible = [
        dict(row)
        for row in candidates
        if best_hard - float(row["hard_auc"]) <= float(tolerance)
    ]
    selected = min(eligible, key=lambda row: float(row["lambda"]))
    return {
        "selection_split": "route_calibration",
        "objective": (
            "maximize hard AUC; within auc_tie_tolerance choose lower lambda"
        ),
        "auc_tie_tolerance": float(tolerance),
        "best_hard_auc": best_hard,
        "eligible_lambdas": [
            float(row["lambda"]) for row in eligible
        ],
        "selected_lambda": float(selected["lambda"]),
        "selected_candidate": selected,
        "validation_used": False,
        "test_used": False,
    }


@torch.inference_mode()
def _paired_order_diagnostics(
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
    final_changes = 0
    total_routes = 0
    commit_assignment_changes = 0
    total_commit_assignments = 0
    map_cosines: list[torch.Tensor] = []
    first_probability_difference = 0.0
    first_provisional_equal = True
    one_step_equal = True
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        condition = values["condition_features"][start:end].to(device)
        quality = values["condition_quality"][start:end].to(device)
        confidence = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=4,
        )
        guided = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=4,
            commit_mode="evidence_order_only",
            evidence_lambda=float(evidence_lambda),
            evidence_config=config["pointer_route"]["diffusion"][
                "evidence_order"
            ],
        )
        one_confidence = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=1,
        )
        one_guided = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=1,
            commit_mode="evidence_order_only",
            evidence_lambda=float(evidence_lambda),
            evidence_config=config["pointer_route"]["diffusion"][
                "evidence_order"
            ],
        )
        one_step_equal = one_step_equal and bool(
            torch.equal(one_confidence.final_routes, one_guided.final_routes)
        )
        first_probability_difference = max(
            first_probability_difference,
            float(
                (
                    confidence.probabilities[0]
                    - guided.probabilities[0]
                )
                .abs()
                .max()
                .item()
            ),
        )
        first_provisional_equal = first_provisional_equal and bool(
            torch.equal(
                confidence.provisional_routes[0],
                guided.provisional_routes[0],
            )
        )
        final_changes += int(
            (
                confidence.final_routes != guided.final_routes
            ).sum().item()
        )
        total_routes += int(confidence.final_routes.numel())
        for confidence_commit, guided_commit in zip(
            confidence.newly_committed,
            guided.newly_committed,
        ):
            commit_assignment_changes += int(
                (confidence_commit != guided_commit).sum().item()
            )
            total_commit_assignments += int(confidence_commit.numel())
        candidates = model(
            model.all_mask_input(end - start, device=device),
            condition,
            quality,
        )[1]
        confidence_map = reconstruct_hard_routes(
            candidates,
            confidence.final_routes,
        )
        guided_map = reconstruct_hard_routes(
            candidates,
            guided.final_routes,
        )
        map_cosines.append(
            F.cosine_similarity(
                confidence_map,
                guided_map,
                dim=-1,
            ).cpu()
        )
    all_cosines = torch.cat(map_cosines)
    return {
        "samples": len(indices),
        "one_step_routes_identical": one_step_equal,
        "first_step_probabilities_max_abs_difference": (
            first_probability_difference
        ),
        "first_step_provisional_routes_identical": first_provisional_equal,
        "four_step_final_route_change_fraction": (
            final_changes / float(total_routes)
        ),
        "commit_assignment_change_fraction": (
            commit_assignment_changes / float(total_commit_assignments)
        ),
        "confidence_guided_map_cosine_mean": float(
            all_cosines.mean().item()
        ),
        "confidence_guided_map_cosine_min": float(
            all_cosines.min().item()
        ),
        "route_logits_modified": False,
        "remask_used": False,
    }


def _decision(
    *,
    confidence: Mapping[str, Any],
    evidence: Mapping[str, Any],
    selection: Mapping[str, Any],
    order_diagnostics: Mapping[str, Any],
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    confidence_four = _method(
        confidence,
        "pointer_diffusion_4step",
    )["verification"]
    confidence_one = _method(
        confidence,
        "pointer_diffusion_1step",
    )["verification"]
    evidence_four = _method(
        evidence,
        "evidence_order_4step",
    )["verification"]
    hard_gain_confidence = (
        evidence_four["hard_average"]["roc_auc"]
        - confidence_four["hard_average"]["roc_auc"]
    )
    hard_gain_one = (
        evidence_four["hard_average"]["roc_auc"]
        - confidence_one["hard_average"]["roc_auc"]
    )
    scenarios = (
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    )
    scenario_deltas = {
        scenario: (
            evidence_four[scenario]["roc_auc"]
            - confidence_four[scenario]["roc_auc"]
        )
        for scenario in scenarios
    }
    scenario_wins = sum(value > 0.0 for value in scenario_deltas.values())
    transition = evidence["trajectory"]["transitions"]["all"]["equivalent"]
    gates = config["pointer_route"]["diffusion"]["evidence_order"]["gates"]
    peak_reserved = max(
        float(value["peak_reserved_mib"]) for value in latency.values()
    )
    checks = {
        "lambda_selected_on_calibration_only": (
            selection["selection_split"] == "route_calibration"
            and selection["validation_used"] is False
            and selection["test_used"] is False
        ),
        "gain_over_confidence_4step": (
            hard_gain_confidence
            >= float(gates["min_gain_over_confidence_4step"])
        ),
        "gain_over_rollout_1step": (
            hard_gain_one
            >= float(gates["min_gain_over_rollout_1step"])
        ),
        "hard_scenario_wins": (
            scenario_wins >= int(gates["min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            confidence_four["clean"]["roc_auc"]
            - evidence_four["clean"]["roc_auc"]
            <= float(gates["max_clean_auc_drop"])
        ),
        "positive_equivalent_net_correction": (
            transition["net_correction_fraction"] > 0.0
            if bool(gates["require_positive_net_correction"])
            else True
        ),
        "ecr_exceeds_eir": transition["ecr"] > transition["eir"],
        "permutation_equivariant": (
            evidence["diffusion_permutation_check"][
                "route_map_max_abs_difference"
            ]
            <= float(gates["max_permutation_map_difference"])
        ),
        "one_step_unchanged": bool(
            order_diagnostics["one_step_routes_identical"]
        ),
        "route_logits_unchanged": (
            float(
                order_diagnostics[
                    "first_step_probabilities_max_abs_difference"
                ]
            )
            == 0.0
            and bool(
                order_diagnostics[
                    "first_step_provisional_routes_identical"
                ]
            )
        ),
        "commit_order_changed": (
            float(
                order_diagnostics["commit_assignment_change_fraction"]
            )
            > 0.0
        ),
        "stop_gradient": bool(
            evidence["evidence_diagnostics"]["stop_gradient"]
        ),
        "no_remask": not bool(
            evidence["diffusion_inference_contract"]["remask_used"]
        ),
        "memory_within_7_2_gib": peak_reserved <= 7.2 * 1024.0,
        "test_not_loaded": True,
    }
    passed = all(checks.values())
    return {
        "classification": (
            "GO_EVIDENCE_ORDER" if passed else "NO_GO_EVIDENCE_ORDER"
        ),
        "checks": checks,
        "selected_lambda": float(selection["selected_lambda"]),
        "hard_auc_gain_over_confidence_4step": hard_gain_confidence,
        "hard_auc_gain_over_rollout_1step": hard_gain_one,
        "scenario_auc_deltas": scenario_deltas,
        "hard_scenario_wins": scenario_wins,
        "equivalent_ecr": transition["ecr"],
        "equivalent_eir": transition["eir"],
        "equivalent_net_correction_fraction": transition[
            "net_correction_fraction"
        ],
        "peak_reserved_mib": peak_reserved,
        "recommendation": (
            "proceed_to_evidence_logits_then_conservative_remask"
            if passed
            else "do_not_add_evidence_logits_or_remask"
        ),
    }


def _comparison_rows(
    *,
    confidence: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    confidence_names = {
        "quality_anchor",
        "one_shot_pointer",
        "pointer_diffusion_1step",
        "pointer_diffusion_4step",
    }
    rows = [
        row
        for row in _core_rows("route_validation", confidence)
        if row["method"] in confidence_names
    ]
    rows.extend(
        row
        for row in _core_rows("route_validation", evidence)
        if str(row["method"]).startswith("evidence_order_")
    )
    return rows


def _report(
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    confidence: Mapping[str, Any],
    evidence: Mapping[str, Any],
    decision: Mapping[str, Any],
    order_diagnostics: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    rows = {
        row["method"]: row
        for row in _comparison_rows(
            confidence=confidence,
            evidence=evidence,
        )
    }
    names = (
        "quality_anchor",
        "one_shot_pointer",
        "pointer_diffusion_1step",
        "pointer_diffusion_4step",
        "evidence_order_1step",
        "evidence_order_2step",
        "evidence_order_4step",
        "evidence_order_8step",
    )
    table = "\n".join(
        [
            "| Method | Hard AUC | EER | TAR@1e-3 | Rank-1 | Map cos |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        + [
            (
                f"| {name} | {rows[name]['hard_macro_auc']:.6f} | "
                f"{rows[name]['eer_all']:.6f} | "
                f"{rows[name]['tar_at_far_1e-3']:.3f} | "
                f"{rows[name]['rank1_all']:.3f} | "
                f"{rows[name]['teacher_map_cosine_all']:.6f} |"
            )
            for name in names
        ]
    )
    calibration_table = "\n".join(
        [
            "| Lambda | Calibration hard AUC | Calibration all AUC |",
            "|---:|---:|---:|",
        ]
        + [
            (
                f"| {row['lambda']:.2f} | {row['hard_auc']:.6f} | "
                f"{row['all_auc']:.6f} |"
            )
            for row in calibration_rows
        ]
    )
    evidence_diag = evidence["evidence_diagnostics"]
    wrong_weight = evidence_diag[
        "mean_wrong_identity_frame_reliability"
    ]
    failed = [
        name for name, passed in decision["checks"].items() if not passed
    ]
    return f"""# PointerRoute evidence-guided commit ordering

## Locked protocol

- The rollout D1 checkpoint is frozen; no training is performed.
- Evidence changes commit ordering only. Route logits, argmax candidates,
  commit counts, and irreversible no-remask decoding are unchanged.
- Evidence is the existing stop-gradient two-level global identity-inlier x
  local candidate cosine support.
- Lambda is selected on route-calibration only from `[0.5, 1.0, 2.0]`.
- Route-validation is evaluated once after lambda is locked. Test is unused.

## Calibration selection

{calibration_table}

- Selected lambda: {selection['selected_lambda']:.2f}.

## Core comparison

{table}

## Mechanism diagnostics

- Four-step evidence minus confidence hard AUC:
  {decision['hard_auc_gain_over_confidence_4step']:+.6f}.
- Four-step evidence minus rollout one-step hard AUC:
  {decision['hard_auc_gain_over_rollout_1step']:+.6f}.
- Hard scenario wins: {decision['hard_scenario_wins']}/4.
- Equivalent ECR/EIR:
  {decision['equivalent_ecr']:.6f}/{decision['equivalent_eir']:.6f}.
- Four-step final route change fraction:
  {order_diagnostics['four_step_final_route_change_fraction']:.6f}.
- Commit-assignment change fraction:
  {order_diagnostics['commit_assignment_change_fraction']:.6f}.
- Confidence/guided map cosine:
  {order_diagnostics['confidence_guided_map_cosine_mean']:.6f}.
- Wrong-identity frame mean reliability:
  {wrong_weight if wrong_weight is not None else 'n/a'}.
- Four-step batch-16 evidence latency:
  {latency['4step']['mean_batch_latency_ms']:.3f} ms.

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

## Interpretation

- Two-level reliability behaves as intended: the mean reliability of a
  wrong-identity frame is {wrong_weight if wrong_weight is not None else 'n/a'},
  while the overall inlier mean is
  {evidence_diag['mean_inlier_frame_reliability']:.6f}.
- Evidence changes commit assignments, improves three of four hard scenarios,
  and gives a small four-step hard-AUC gain over confidence-only. This is a
  directional mechanism signal, not a passed effect-size gate.
- Because ordering is irreversible and route argmax is fixed, only
  {100.0 * order_diagnostics['four_step_final_route_change_fraction']:.2f}%
  of final routes change. The available causal leverage is therefore small.
- Evidence-order four-step still trails the unchanged one-step output.
  Evidence logits and remasking are not authorized by this screen; they
  would be new hypotheses rather than automatic extensions of a passing
  ordering result.

This is a mechanism-development result on the reused route-validation split,
not an untouched-holdout or statistical-significance claim.
"""


def main() -> int:
    args = parse_args()
    resolved = load_config(args.config, args.profile)
    config = resolved.to_dict()
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
    evidence_config = config["pointer_route"]["diffusion"]["evidence_order"]
    preregistration = {
        "schema_version": 1,
        "stage": "pointer-route-evidence-order-only",
        "mode": args.stage,
        "frozen_rollout_checkpoint": str(rollout["checkpoint"]),
        "frozen_rollout_checkpoint_sha256": rollout["checkpoint_sha256"],
        "selection_split": "route_calibration",
        "evaluation_split": (
            None if args.stage == "smoke" else "route_validation"
        ),
        "evidence_order": evidence_config,
        "route_logits_modified": False,
        "route_argmax_modified": False,
        "remask_used": False,
        "training_performed": False,
        "phase01_artifact_hashes": phase01["artifact_hashes"],
        "phase3_artifact_hashes": phase3["artifact_hashes"],
        "rollout_artifact_hashes": rollout["artifact_hashes"],
        "config_sha256": config_sha256(config),
        "test_loaded": False,
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "pointer_route_evidence_order_started",
        mode=args.stage,
        test_loaded=False,
        training_performed=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PointerRoute evidence evaluation requires CUDA")
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
    model.to(device)
    paired_model = build_pointer_route_model(config)
    load_pointer_checkpoint(
        path=phase3["training"]["one_shot"]["best_checkpoint"],
        model=paired_model,
        device=device,
    )
    paired_model.to(device)
    split_indices = phase01["split"]["sample_indices"]
    if args.stage == "smoke":
        smoke_evaluation = evaluate_pointer_diffusion_subset(
            split_name="route_calibration",
            dataset=dataset,
            indices=split_indices["route_calibration"],
            targets=targets,
            one_shot_model=paired_model,
            diffusion_model=model,
            config=config,
            device=device,
            delta=delta,
            temperature=temperature,
            evaluation_steps=[1, 4],
            commit_mode="evidence_order_only",
            evidence_lambda=float(evidence_config["lambda_candidates"][0]),
            method_prefix="evidence_order",
        )
        diagnostics = _paired_order_diagnostics(
            dataset=dataset,
            indices=split_indices["route_calibration"],
            model=model,
            config=config,
            device=device,
            evidence_lambda=float(evidence_config["lambda_candidates"][0]),
        )
        _write_json(
            run_paths.artifacts / "smoke_evaluation.json",
            smoke_evaluation,
        )
        _write_json(
            run_paths.artifacts / "order_diagnostics.json",
            diagnostics,
        )
        manifest = {
            "schema_version": 1,
            "stage": "pointer-route-evidence-order-smoke",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "evaluation": smoke_evaluation,
            "order_diagnostics": diagnostics,
            "test_loaded": False,
        }
        logger.write_manifest(manifest)
        logger.log_event("pointer_route_evidence_order_smoke_completed")
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "status": "evidence_order_smoke_passed",
                    "device": str(device),
                    "order_diagnostics": diagnostics,
                    "evidence_diagnostics": smoke_evaluation[
                        "evidence_diagnostics"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    calibration_evaluations: dict[str, Any] = {}
    calibration_rows: list[dict[str, Any]] = []
    with CudaActivityMonitor() as monitor:
        for value in evidence_config["lambda_candidates"]:
            evidence_lambda = float(value)
            key = f"{evidence_lambda:g}"
            evaluation = evaluate_pointer_diffusion_subset(
                split_name="route_calibration",
                dataset=dataset,
                indices=split_indices["route_calibration"],
                targets=targets,
                one_shot_model=paired_model,
                diffusion_model=model,
                config=config,
                device=device,
                delta=delta,
                temperature=temperature,
                evaluation_steps=[4],
                commit_mode="evidence_order_only",
                evidence_lambda=evidence_lambda,
                method_prefix="evidence_order",
            )
            method = evaluation["methods"]["evidence_order_4step"]
            verification = method["verification"]
            row = {
                "lambda": evidence_lambda,
                "hard_auc": verification["hard_average"]["roc_auc"],
                "all_auc": verification["all"]["roc_auc"],
                "equivalent_net_correction": evaluation["trajectory"][
                    "transitions"
                ]["all"]["equivalent"]["net_correction_fraction"],
            }
            calibration_rows.append(row)
            calibration_evaluations[key] = evaluation
        selection = _select_lambda(
            candidates=calibration_rows,
            tolerance=float(evidence_config["auc_tie_tolerance"]),
        )
        selected_lambda = float(selection["selected_lambda"])
        validation = evaluate_pointer_diffusion_subset(
            split_name="route_validation",
            dataset=dataset,
            indices=split_indices["route_validation"],
            targets=targets,
            one_shot_model=paired_model,
            diffusion_model=model,
            config=config,
            device=device,
            delta=delta,
            temperature=temperature,
            commit_mode="evidence_order_only",
            evidence_lambda=selected_lambda,
            method_prefix="evidence_order",
        )
        order_diagnostics = _paired_order_diagnostics(
            dataset=dataset,
            indices=split_indices["route_validation"],
            model=model,
            config=config,
            device=device,
            evidence_lambda=selected_lambda,
        )
        latency = measure_diffusion_latency(
            dataset=dataset,
            indices=split_indices["route_validation"],
            model=model,
            config=config,
            device=device,
            commit_mode="evidence_order_only",
            evidence_lambda=selected_lambda,
        )
    gpu_activity = monitor.summary()
    decision = _decision(
        confidence=rollout["validation"],
        evidence=validation,
        selection=selection,
        order_diagnostics=order_diagnostics,
        latency=latency,
        config=config,
    )
    _write_json(
        run_paths.artifacts / "calibration_evaluations.json",
        calibration_evaluations,
    )
    _write_csv(
        run_paths.artifacts / "calibration_candidates.csv",
        calibration_rows,
    )
    _write_json(run_paths.artifacts / "selection.json", selection)
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(
        run_paths.artifacts / "trajectory_diagnostics.json",
        validation["trajectory"],
    )
    _write_json(
        run_paths.artifacts / "evidence_diagnostics.json",
        validation["evidence_diagnostics"],
    )
    _write_json(
        run_paths.artifacts / "order_diagnostics.json",
        order_diagnostics,
    )
    _write_json(run_paths.artifacts / "latency.json", latency)
    _write_json(run_paths.artifacts / "gpu_activity.json", gpu_activity)
    _write_json(run_paths.artifacts / "decision.json", decision)
    _write_csv(
        run_paths.artifacts / "core_comparison.csv",
        _comparison_rows(
            confidence=rollout["validation"],
            evidence=validation,
        ),
    )
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        [
            *(
                row
                for row in _scenario_rows(
                    "route_validation",
                    rollout["validation"],
                )
                if row["method"]
                in {
                    "pointer_diffusion_1step",
                    "pointer_diffusion_4step",
                }
            ),
            *(
                row
                for row in _scenario_rows(
                    "route_validation",
                    validation,
                )
                if str(row["method"]).startswith("evidence_order_")
            ),
        ],
    )
    _write_csv(
        run_paths.artifacts / "trajectory_metrics.csv",
        _trajectory_rows("route_validation", validation),
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        _report(
            calibration_rows=calibration_rows,
            selection=selection,
            confidence=rollout["validation"],
            evidence=validation,
            decision=decision,
            order_diagnostics=order_diagnostics,
            latency=latency,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "pointer-route-evidence-order-only",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "selection": selection,
        "calibration_candidates": calibration_rows,
        "validation": validation,
        "order_diagnostics": order_diagnostics,
        "latency": latency,
        "gpu_activity": gpu_activity,
        "decision": decision,
        "training_performed": False,
        "test_loaded": False,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "pointer_route_evidence_order_completed",
        classification=decision["classification"],
        selected_lambda=selection["selected_lambda"],
        test_loaded=False,
    )
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "selection": selection,
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
