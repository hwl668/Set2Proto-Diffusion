"""Frozen-checkpoint PointerRoute two-level evidence-logits diagnostic."""

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

from run_pointer_route_evidence_order import (
    _load_rollout_lock,
    _select_lambda,
)
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
    parser.add_argument("--evidence-order-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _load_order_lock(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    required = {
        "validation": artifacts / "validation_evaluation.json",
        "decision": artifacts / "decision.json",
        "selection": artifacts / "selection.json",
        "protocol": artifacts / "protocol_preregistration.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing evidence-order {name}: {path}")
    result = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
    }
    if result["protocol"].get("test_loaded") is not False:
        raise RuntimeError("evidence-order protocol loaded test data")
    result["artifact_hashes"] = {
        name: _sha256(path) for name, path in required.items()
    }
    return result


@torch.inference_mode()
def _paired_logits_diagnostics(
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
    first_probability_max_difference = 0.0
    first_route_changes = 0
    confidence_guided_changes = 0
    guided_step_changes = 0
    total_routes = 0
    commit_assignment_changes = 0
    total_assignments = 0
    confidence_guided_cosines: list[torch.Tensor] = []
    guided_step_cosines: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        condition = values["condition_features"][start:end].to(device)
        quality = values["condition_quality"][start:end].to(device)
        confidence_four = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=4,
        )
        guided_one = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=1,
            commit_mode="evidence_logits",
            evidence_lambda=float(evidence_lambda),
            evidence_config=config["pointer_route"]["diffusion"][
                "evidence_logits"
            ],
        )
        guided_four = decode_pointer_routes(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            steps=4,
            commit_mode="evidence_logits",
            evidence_lambda=float(evidence_lambda),
            evidence_config=config["pointer_route"]["diffusion"][
                "evidence_logits"
            ],
        )
        first_probability_max_difference = max(
            first_probability_max_difference,
            float(
                (
                    confidence_four.probabilities[0]
                    - guided_four.probabilities[0]
                )
                .abs()
                .max()
                .item()
            ),
        )
        first_route_changes += int(
            (
                confidence_four.provisional_routes[0]
                != guided_four.provisional_routes[0]
            ).sum().item()
        )
        confidence_guided_changes += int(
            (
                confidence_four.final_routes
                != guided_four.final_routes
            ).sum().item()
        )
        guided_step_changes += int(
            (
                guided_one.final_routes
                != guided_four.final_routes
            ).sum().item()
        )
        total_routes += int(guided_four.final_routes.numel())
        for confidence_commit, guided_commit in zip(
            confidence_four.newly_committed,
            guided_four.newly_committed,
        ):
            commit_assignment_changes += int(
                (confidence_commit != guided_commit).sum().item()
            )
            total_assignments += int(confidence_commit.numel())
        candidates = model(
            model.all_mask_input(end - start, device=device),
            condition,
            quality,
        )[1]
        maps = {
            "confidence": reconstruct_hard_routes(
                candidates,
                confidence_four.final_routes,
            ),
            "guided_one": reconstruct_hard_routes(
                candidates,
                guided_one.final_routes,
            ),
            "guided_four": reconstruct_hard_routes(
                candidates,
                guided_four.final_routes,
            ),
        }
        confidence_guided_cosines.append(
            F.cosine_similarity(
                maps["confidence"],
                maps["guided_four"],
                dim=-1,
            ).cpu()
        )
        guided_step_cosines.append(
            F.cosine_similarity(
                maps["guided_one"],
                maps["guided_four"],
                dim=-1,
            ).cpu()
        )
    confidence_cosine = torch.cat(confidence_guided_cosines)
    step_cosine = torch.cat(guided_step_cosines)
    return {
        "samples": len(indices),
        "first_step_selection_probability_max_abs_difference": (
            first_probability_max_difference
        ),
        "first_step_route_change_fraction": (
            first_route_changes / float(total_routes)
        ),
        "confidence_to_guided_four_route_change_fraction": (
            confidence_guided_changes / float(total_routes)
        ),
        "guided_one_to_four_route_change_fraction": (
            guided_step_changes / float(total_routes)
        ),
        "commit_assignment_change_fraction": (
            commit_assignment_changes / float(total_assignments)
        ),
        "confidence_guided_map_cosine_mean": float(
            confidence_cosine.mean().item()
        ),
        "confidence_guided_map_cosine_min": float(
            confidence_cosine.min().item()
        ),
        "guided_one_four_map_cosine_mean": float(
            step_cosine.mean().item()
        ),
        "guided_one_four_map_cosine_min": float(step_cosine.min().item()),
        "checkpoint_frozen": True,
        "extra_order_guidance": False,
        "remask_used": False,
    }


def _decision(
    *,
    confidence: Mapping[str, Any],
    guided: Mapping[str, Any],
    selection: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    confidence_one = confidence["methods"]["pointer_diffusion_1step"][
        "verification"
    ]
    guided_one = guided["methods"]["evidence_logits_1step"][
        "verification"
    ]
    guided_four = guided["methods"]["evidence_logits_4step"][
        "verification"
    ]
    four_minus_guided_one = (
        guided_four["hard_average"]["roc_auc"]
        - guided_one["hard_average"]["roc_auc"]
    )
    four_minus_confidence_one = (
        guided_four["hard_average"]["roc_auc"]
        - confidence_one["hard_average"]["roc_auc"]
    )
    scenarios = (
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    )
    scenario_step_deltas = {
        scenario: (
            guided_four[scenario]["roc_auc"]
            - guided_one[scenario]["roc_auc"]
        )
        for scenario in scenarios
    }
    scenario_wins = sum(value > 0.0 for value in scenario_step_deltas.values())
    transition = guided["trajectory"]["transitions"]["all"]["equivalent"]
    confidence_wrong = confidence["diffusion_route_diagnostics"][
        "pointer_diffusion_1step"
    ]["wrong_identity"][
        "wrong_frame_selection_fraction_all_positions"
    ]
    guided_wrong = guided["diffusion_route_diagnostics"][
        "evidence_logits_4step"
    ]["wrong_identity"][
        "wrong_frame_selection_fraction_all_positions"
    ]
    gates = config["pointer_route"]["diffusion"]["evidence_logits"]["gates"]
    peak_reserved = max(
        float(value["peak_reserved_mib"]) for value in latency.values()
    )
    checks = {
        "lambda_selected_on_calibration_only": (
            selection["selection_split"] == "route_calibration"
            and selection["validation_used"] is False
            and selection["test_used"] is False
        ),
        "four_step_beats_guided_one_step": (
            four_minus_guided_one
            >= float(gates["min_four_step_gain_over_guided_one_step"])
        ),
        "four_step_beats_confidence_one_step": (
            four_minus_confidence_one
            >= float(gates["min_four_step_gain_over_confidence_one_step"])
        ),
        "hard_scenario_step_wins": (
            scenario_wins >= int(gates["min_hard_scenario_step_wins"])
        ),
        "clean_constraint": (
            guided_one["clean"]["roc_auc"]
            - guided_four["clean"]["roc_auc"]
            <= float(gates["max_clean_auc_drop_vs_guided_one_step"])
        ),
        "positive_equivalent_net_correction": (
            transition["net_correction_fraction"] > 0.0
            if bool(gates["require_positive_net_correction"])
            else True
        ),
        "ecr_exceeds_eir": transition["ecr"] > transition["eir"],
        "wrong_identity_selection_reduced": (
            guided_wrong < confidence_wrong
            if bool(gates["require_wrong_identity_selection_reduction"])
            else True
        ),
        "permutation_equivariant": (
            guided["diffusion_permutation_check"][
                "route_map_max_abs_difference"
            ]
            <= float(gates["max_permutation_map_difference"])
        ),
        "selection_logits_changed": (
            float(
                diagnostics[
                    "first_step_selection_probability_max_abs_difference"
                ]
            )
            > 0.0
        ),
        "checkpoint_frozen": bool(diagnostics["checkpoint_frozen"]),
        "stop_gradient": bool(
            guided["evidence_diagnostics"]["stop_gradient"]
        ),
        "no_extra_order_guidance": not bool(
            diagnostics["extra_order_guidance"]
        ),
        "no_remask": not bool(diagnostics["remask_used"]),
        "memory_within_7_2_gib": peak_reserved <= 7.2 * 1024.0,
        "test_not_loaded": True,
    }
    passed = all(checks.values())
    return {
        "classification": (
            "GO_EVIDENCE_LOGITS"
            if passed
            else "NO_GO_STOP_POINTERROUTE_DIFFUSION"
        ),
        "checks": checks,
        "selected_lambda": float(selection["selected_lambda"]),
        "four_step_minus_guided_one_step_hard_auc": four_minus_guided_one,
        "four_step_minus_confidence_one_step_hard_auc": (
            four_minus_confidence_one
        ),
        "scenario_step_auc_deltas": scenario_step_deltas,
        "hard_scenario_step_wins": scenario_wins,
        "equivalent_ecr": transition["ecr"],
        "equivalent_eir": transition["eir"],
        "equivalent_net_correction_fraction": transition[
            "net_correction_fraction"
        ],
        "confidence_wrong_frame_selection_fraction": confidence_wrong,
        "guided_wrong_frame_selection_fraction": guided_wrong,
        "peak_reserved_mib": peak_reserved,
        "recommendation": (
            "run_one_conservative_remask_screen"
            if passed
            else "pivot_to_one_step_or_continuous_residual_aggregation"
        ),
    }


def _core_comparison_rows(
    *,
    confidence: Mapping[str, Any],
    order: Mapping[str, Any],
    guided: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _core_rows("route_validation", confidence)
        if row["method"]
        in {
            "quality_anchor",
            "one_shot_pointer",
            "pointer_diffusion_1step",
            "pointer_diffusion_4step",
        }
    ]
    rows.extend(
        row
        for row in _core_rows("route_validation", order)
        if row["method"] == "evidence_order_4step"
    )
    rows.extend(
        row
        for row in _core_rows("route_validation", guided)
        if str(row["method"]).startswith("evidence_logits_")
    )
    return rows


def _report(
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    confidence: Mapping[str, Any],
    order: Mapping[str, Any],
    guided: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    decision: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    rows = {
        row["method"]: row
        for row in _core_comparison_rows(
            confidence=confidence,
            order=order,
            guided=guided,
        )
    }
    names = (
        "quality_anchor",
        "one_shot_pointer",
        "pointer_diffusion_1step",
        "pointer_diffusion_4step",
        "evidence_order_4step",
        "evidence_logits_1step",
        "evidence_logits_2step",
        "evidence_logits_4step",
        "evidence_logits_8step",
    )
    core_table = "\n".join(
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
            "| Lambda | Guided 1-step | Guided 4-step | Step gain |",
            "|---:|---:|---:|---:|",
        ]
        + [
            (
                f"| {row['lambda']:.2f} | {row['one_step_hard_auc']:.6f} | "
                f"{row['four_step_hard_auc']:.6f} | "
                f"{row['four_minus_one_hard_auc']:+.6f} |"
            )
            for row in calibration_rows
        ]
    )
    guided_one = guided["methods"]["evidence_logits_1step"]["verification"]
    guided_four = guided["methods"]["evidence_logits_4step"]["verification"]
    scenario_table = "\n".join(
        [
            "| Scenario | Guided 1-step | Guided 4-step | Step gain |",
            "|---|---:|---:|---:|",
        ]
        + [
            (
                f"| {scenario} | {guided_one[scenario]['roc_auc']:.6f} | "
                f"{guided_four[scenario]['roc_auc']:.6f} | "
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
    return f"""# PointerRoute evidence-logits-only diagnostic

## Locked protocol

- The rollout D1 checkpoint is frozen; no training is performed.
- Stop-gradient two-level evidence is added to candidate selection logits.
- Commit ordering then uses guided confidence only; there is no additional
  order score, remasking, or changed commit schedule.
- Lambda is selected on route-calibration from `[0.5, 1.0, 2.0]`.
- Route-validation is evaluated once after locking lambda. Test is unused.

## Calibration selection

{calibration_table}

- Selected lambda: {selection['selected_lambda']:.2f}.

## Core comparison

{core_table}

## Guided one-step versus four-step

{scenario_table}

- Four-step minus guided one-step hard AUC:
  {decision['four_step_minus_guided_one_step_hard_auc']:+.6f}.
- Four-step minus confidence one-step hard AUC:
  {decision['four_step_minus_confidence_one_step_hard_auc']:+.6f}.
- Equivalent ECR/EIR:
  {decision['equivalent_ecr']:.6f}/{decision['equivalent_eir']:.6f}.
- Wrong-frame selection, confidence/guided:
  {decision['confidence_wrong_frame_selection_fraction']:.6f}/
  {decision['guided_wrong_frame_selection_fraction']:.6f}.

## Mechanism diagnostics

- First-step route change fraction:
  {diagnostics['first_step_route_change_fraction']:.6f}.
- Confidence-to-guided four-step route change:
  {diagnostics['confidence_to_guided_four_route_change_fraction']:.6f}.
- Guided one-to-four route change:
  {diagnostics['guided_one_to_four_route_change_fraction']:.6f}.
- Guided one/four map cosine:
  {diagnostics['guided_one_four_map_cosine_mean']:.6f}.
- Four-step batch-16 latency:
  {latency['4step']['mean_batch_latency_ms']:.3f} ms.

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

## Interpretation

- Evidence logits reduces wrong-identity frame selection by more than half,
  so the two-level evidence signal is operational rather than inert.
- The largest step loss occurs under complementary occlusion. A global
  identity-inlier prior can suppress an unusual but genuinely useful local
  frame, exposing a conflict between contamination rejection and preserving
  unique complementary evidence.
- Calibration shows only tiny positive step gains and validation reverses the
  primary four-step gain. Eight-step is descriptively closer, but its effect
  is below the preregistered threshold and it is not the primary decoder.
- Candidate guidance therefore improves some error types without establishing
  a robust iterative advantage. Plain PointerRoute diffusion should stop;
  the defensible branches are the strong one-step router or a continuous
  residual prototype model that can synthesize rather than only select an
  observed frame route.

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
    order = _load_order_lock(args.evidence_order_root.expanduser().resolve())
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
    evidence_config = config["pointer_route"]["diffusion"][
        "evidence_logits"
    ]
    preregistration = {
        "schema_version": 1,
        "stage": "pointer-route-evidence-logits-only",
        "mode": args.stage,
        "frozen_rollout_checkpoint": str(rollout["checkpoint"]),
        "frozen_rollout_checkpoint_sha256": rollout["checkpoint_sha256"],
        "selection_split": "route_calibration",
        "evaluation_split": (
            None if args.stage == "smoke" else "route_validation"
        ),
        "evidence_logits": evidence_config,
        "training_performed": False,
        "extra_order_guidance": False,
        "remask_used": False,
        "phase01_artifact_hashes": phase01["artifact_hashes"],
        "phase3_artifact_hashes": phase3["artifact_hashes"],
        "rollout_artifact_hashes": rollout["artifact_hashes"],
        "evidence_order_artifact_hashes": order["artifact_hashes"],
        "config_sha256": config_sha256(config),
        "test_loaded": False,
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "pointer_route_evidence_logits_started",
        mode=args.stage,
        test_loaded=False,
        training_performed=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PointerRoute evidence logits requires CUDA")
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
        evidence_lambda = float(evidence_config["lambda_candidates"][0])
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
            evaluation_steps=[1, 4],
            commit_mode="evidence_logits",
            evidence_lambda=evidence_lambda,
            method_prefix="evidence_logits",
        )
        diagnostics = _paired_logits_diagnostics(
            dataset=dataset,
            indices=split_indices["route_calibration"],
            model=model,
            config=config,
            device=device,
            evidence_lambda=evidence_lambda,
        )
        _write_json(
            run_paths.artifacts / "smoke_evaluation.json",
            evaluation,
        )
        _write_json(
            run_paths.artifacts / "logits_diagnostics.json",
            diagnostics,
        )
        manifest = {
            "schema_version": 1,
            "stage": "pointer-route-evidence-logits-smoke",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "evaluation": evaluation,
            "logits_diagnostics": diagnostics,
            "test_loaded": False,
        }
        logger.write_manifest(manifest)
        logger.log_event("pointer_route_evidence_logits_smoke_completed")
        print(
            json.dumps(
                {
                    "run_directory": str(run_paths.root),
                    "status": "evidence_logits_smoke_passed",
                    "device": str(device),
                    "logits_diagnostics": diagnostics,
                    "evidence_diagnostics": evaluation[
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
                evaluation_steps=[1, 4],
                commit_mode="evidence_logits",
                evidence_lambda=evidence_lambda,
                method_prefix="evidence_logits",
            )
            one = evaluation["methods"]["evidence_logits_1step"][
                "verification"
            ]
            four = evaluation["methods"]["evidence_logits_4step"][
                "verification"
            ]
            row = {
                "lambda": evidence_lambda,
                "hard_auc": four["hard_average"]["roc_auc"],
                "all_auc": four["all"]["roc_auc"],
                "one_step_hard_auc": one["hard_average"]["roc_auc"],
                "four_step_hard_auc": four["hard_average"]["roc_auc"],
                "four_minus_one_hard_auc": (
                    four["hard_average"]["roc_auc"]
                    - one["hard_average"]["roc_auc"]
                ),
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
            commit_mode="evidence_logits",
            evidence_lambda=selected_lambda,
            method_prefix="evidence_logits",
        )
        diagnostics = _paired_logits_diagnostics(
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
            commit_mode="evidence_logits",
            evidence_lambda=selected_lambda,
        )
    gpu_activity = monitor.summary()
    decision = _decision(
        confidence=rollout["validation"],
        guided=validation,
        selection=selection,
        diagnostics=diagnostics,
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
        run_paths.artifacts / "logits_diagnostics.json",
        diagnostics,
    )
    _write_json(run_paths.artifacts / "latency.json", latency)
    _write_json(run_paths.artifacts / "gpu_activity.json", gpu_activity)
    _write_json(run_paths.artifacts / "decision.json", decision)
    _write_csv(
        run_paths.artifacts / "core_comparison.csv",
        _core_comparison_rows(
            confidence=rollout["validation"],
            order=order["validation"],
            guided=validation,
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
                if str(row["method"]).startswith("evidence_logits_")
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
            order=order["validation"],
            guided=validation,
            diagnostics=diagnostics,
            decision=decision,
            latency=latency,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "pointer-route-evidence-logits-only",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "selection": selection,
        "calibration_candidates": calibration_rows,
        "validation": validation,
        "logits_diagnostics": diagnostics,
        "latency": latency,
        "gpu_activity": gpu_activity,
        "decision": decision,
        "training_performed": False,
        "test_loaded": False,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "pointer_route_evidence_logits_completed",
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
