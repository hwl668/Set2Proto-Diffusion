"""Run PointerRoute Phase 0 protocol freeze and Phase 1 oracle audit."""

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
from set2proto.pointer_route_oracle import (
    HARD_SCENARIOS,
    build_route_dev_split,
    evaluate_route_oracle,
    select_calibration_pair,
)
from set2proto.quantization import PCAProjection, robust_teacher_pool
from set2proto.reproducibility import seed_everything
from set2proto.run_paths import create_run_paths
from set2proto.training import TokenTrainingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    parser.add_argument("--profile", choices=("expanded",), default="expanded")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--quantization-root", type=Path, required=True)
    parser.add_argument("--official-split-root", type=Path, required=True)
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


def _canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _official_identity_audit(
    official_split_root: Path,
    route_identity_ids: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    path = official_split_root / "identities.csv"
    official: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            official[str(row["split"])].add(int(row["identity_id"]))
    route_union = set().union(
        *(set(int(value) for value in values)
          for values in route_identity_ids.values())
    )
    return {
        "metadata_file": str(path.resolve()),
        "metadata_sha256": _sha256(path),
        "official_counts": {
            split: len(values) for split, values in official.items()
        },
        "route_union_is_subset_of_official_train": route_union.issubset(
            official["train"]
        ),
        "route_union_disjoint_from_official_val": route_union.isdisjoint(
            official["val"]
        ),
        "route_union_disjoint_from_official_test": route_union.isdisjoint(
            official["test"]
        ),
        "test_features_loaded": False,
        "test_targets_loaded": False,
        "test_gallery_constructed": False,
    }


@torch.inference_mode()
def _load_subset(
    *,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    projector = PCAProjection(
        mean=dataset.projector.mean.to(device),
        components=dataset.projector.components.to(device),
        explained_variance_ratio=dataset.projector.explained_variance_ratio,
        fit_tokens=dataset.projector.fit_tokens,
        fit_split=dataset.projector.fit_split,
    )
    teacher_config = config["teacher"]
    condition_parts: list[torch.Tensor] = []
    quality_parts: list[torch.Tensor] = []
    teacher_parts: list[torch.Tensor] = []
    loo_parts: list[torch.Tensor] = []
    identity_parts: list[torch.Tensor] = []
    source_identity_parts: list[torch.Tensor] = []
    visibility_parts: list[torch.Tensor] = []
    scenarios: list[str] = []
    sample_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    teacher_replay_max_abs_difference = 0.0
    maximum_cuda_allocated_mib = 0.0

    for start in range(0, len(indices), batch_size):
        batch_indices = list(indices[start : start + batch_size])
        samples = [dataset.synthetic[index] for index in batch_indices]
        rows = [dataset.synthetic.rows[index] for index in batch_indices]
        raw_condition = torch.stack(
            [sample["condition_features"] for sample in samples]
        ).to(device)
        raw_teacher = torch.stack(
            [sample["teacher_features"] for sample in samples]
        ).to(device)
        teacher_quality = torch.stack(
            [sample["teacher_quality"] for sample in samples]
        ).to(device)
        projected_condition = projector.transform(raw_condition)
        projected_teacher = projector.transform(raw_teacher)
        replayed_teacher, _ = robust_teacher_pool(
            projected_teacher,
            teacher_quality,
            keep_fraction=float(teacher_config["robust_keep_fraction"]),
            consensus_weight=float(teacher_config["consensus_weight"]),
            quality_weight=float(teacher_config["quality_weight"]),
        )
        cached_teacher = dataset.targets["continuous_prototypes"][
            batch_indices
        ].float().to(device)
        teacher_replay_max_abs_difference = max(
            teacher_replay_max_abs_difference,
            float((replayed_teacher - cached_teacher).abs().max().item()),
        )

        leave_one_out: list[torch.Tensor] = []
        teacher_frames = projected_teacher.shape[1]
        for leave_index in range(teacher_frames):
            keep = [
                index for index in range(teacher_frames)
                if index != leave_index
            ]
            pooled, _ = robust_teacher_pool(
                projected_teacher[:, keep],
                teacher_quality[:, keep],
                keep_fraction=float(teacher_config["robust_keep_fraction"]),
                consensus_weight=float(teacher_config["consensus_weight"]),
                quality_weight=float(teacher_config["quality_weight"]),
            )
            leave_one_out.append(pooled)

        condition_parts.append(projected_condition.cpu().half())
        quality_parts.append(
            torch.stack(
                [sample["condition_quality"] for sample in samples]
            ).cpu()
        )
        teacher_parts.append(cached_teacher.cpu().half())
        loo_parts.append(torch.stack(leave_one_out, dim=1).cpu().half())
        identity_parts.append(
            torch.stack([sample["target_identity"] for sample in samples])
        )
        source_identity_parts.append(
            torch.stack(
                [sample["condition_source_identity"] for sample in samples]
            )
        )
        visibility_parts.append(
            torch.stack(
                [sample["condition_visibility"] for sample in samples]
            )
        )
        scenarios.extend(str(sample["scenario"]) for sample in samples)
        sample_ids.extend(str(sample["sample_id"]) for sample in samples)
        selected_rows.extend(dict(row) for row in rows)
        if device.type == "cuda":
            maximum_cuda_allocated_mib = max(
                maximum_cuda_allocated_mib,
                torch.cuda.max_memory_allocated(device) / 1024**2,
            )

    return {
        "condition_features": torch.cat(condition_parts).float(),
        "condition_quality": torch.cat(quality_parts).float(),
        "teacher_maps": torch.cat(teacher_parts).float(),
        "loo_teacher_maps": torch.cat(loo_parts).float(),
        "identities": torch.cat(identity_parts).to(torch.int64),
        "source_identities": torch.cat(source_identity_parts).to(torch.int64),
        "visibility": torch.cat(visibility_parts).float(),
        "scenarios": scenarios,
        "sample_ids": sample_ids,
        "selected_rows_sha256": _canonical_rows_sha256(selected_rows),
        "teacher_replay_max_abs_difference": (
            teacher_replay_max_abs_difference
        ),
        "peak_cuda_allocated_mib": maximum_cuda_allocated_mib,
    }


def _without_targets(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in evaluation.items() if key != "targets"
    }


def _calibration_row(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    methods = evaluation["methods"]
    diagnostics = evaluation["route_diagnostics"]["all"]
    stability = evaluation["teacher_loo_stability"]["all"]
    return {
        "delta": evaluation["delta"],
        "temperature": evaluation["temperature"],
        "quality_hard_auc": methods["quality_anchor"]["verification"][
            "hard_average"
        ]["roc_auc"],
        "hard_route_hard_auc": methods["hard_route_oracle"]["verification"][
            "hard_average"
        ]["roc_auc"],
        "soft_route_hard_auc": methods["soft_route_oracle"]["verification"][
            "hard_average"
        ]["roc_auc"],
        "hard_gain_vs_quality": (
            methods["hard_route_oracle"]["verification"]["hard_average"][
                "roc_auc"
            ]
            - methods["quality_anchor"]["verification"]["hard_average"][
                "roc_auc"
            ]
        ),
        "soft_gain_vs_quality": (
            methods["soft_route_oracle"]["verification"]["hard_average"][
                "roc_auc"
            ]
            - methods["quality_anchor"]["verification"]["hard_average"][
                "roc_auc"
            ]
        ),
        "non_anchor_fraction": diagnostics["non_anchor_fraction"],
        "wrong_frame_selection_fraction": evaluation["route_diagnostics"][
            "wrong_identity"
        ]["wrong_frame_selection_fraction_of_frame_routes"],
        "teacher_loo_exact_agreement": stability["exact_route_agreement"],
        "teacher_loo_equivalent_agreement": stability[
            "equivalent_route_agreement"
        ],
    }


def _core_rows(
    split: str,
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in evaluation["methods"].items():
        verification = values["verification"]
        all_metrics = verification["all"]
        hard = verification["hard_average"]
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


def _diagnostic_rows(
    split: str,
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, values in evaluation["route_diagnostics"].items():
        for key, value in values.items():
            if isinstance(value, (int, float)) or value is None:
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "scope": "route",
                        "metric": key,
                        "value": value,
                    }
                )
        for name, summary in values.items():
            if isinstance(summary, Mapping):
                for statistic, value in summary.items():
                    rows.append(
                        {
                            "split": split,
                            "scenario": scenario,
                            "scope": name,
                            "metric": statistic,
                            "value": value,
                        }
                    )
    for scenario, values in evaluation["teacher_loo_stability"].items():
        for key, value in values.items():
            rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "scope": "teacher_loo",
                    "metric": key,
                    "value": value,
                }
            )
    return rows


def _decision(
    *,
    split_checks: Mapping[str, Any],
    official_audit: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    gates: Mapping[str, Any],
    teacher_replay_max_abs_difference: float,
) -> dict[str, Any]:
    quality = evaluation["methods"]["quality_anchor"]["verification"]
    oracle = evaluation["methods"]["hard_route_oracle"]["verification"]
    hard_gain = (
        oracle["hard_average"]["roc_auc"]
        - quality["hard_average"]["roc_auc"]
    )
    scenario_deltas = {
        scenario: (
            oracle[scenario]["roc_auc"] - quality[scenario]["roc_auc"]
        )
        for scenario in HARD_SCENARIOS
    }
    wins = sum(value > 0.0 for value in scenario_deltas.values())
    non_anchor = evaluation["route_diagnostics"]["all"][
        "non_anchor_fraction"
    ]
    wrong_selection = evaluation["route_diagnostics"]["wrong_identity"][
        "wrong_frame_selection_fraction_of_frame_routes"
    ]
    loo_equivalent = evaluation["teacher_loo_stability"]["all"][
        "equivalent_route_agreement"
    ]
    permutation = evaluation["permutation_check"]
    checks = {
        "route_split_integrity": bool(split_checks["passed"]),
        "route_identities_subset_of_official_train": bool(
            official_audit["route_union_is_subset_of_official_train"]
        ),
        "route_identities_disjoint_from_official_val": bool(
            official_audit["route_union_disjoint_from_official_val"]
        ),
        "route_identities_disjoint_from_official_test": bool(
            official_audit["route_union_disjoint_from_official_test"]
        ),
        "teacher_target_replay_exact_enough": (
            teacher_replay_max_abs_difference <= 5e-4
        ),
        "oracle_hard_auc_gain": (
            hard_gain >= float(gates["min_oracle_hard_auc_gain"])
        ),
        "hard_scenario_wins": (
            wins >= int(gates["min_hard_scenario_wins"])
        ),
        "non_anchor_route_mass": (
            non_anchor >= float(gates["min_non_anchor_fraction"])
        ),
        "teacher_loo_equivalent_stability": (
            loo_equivalent
            >= float(gates["min_teacher_loo_equivalent_agreement"])
        ),
        "wrong_frame_selection_controlled": (
            wrong_selection
            <= float(gates["max_wrong_frame_selection_fraction"])
        ),
        "frame_permutation_equivariant": (
            permutation["route_disagreement_fraction"] == 0.0
            and permutation["hard_map_max_abs_difference"] <= 1e-5
            and permutation["soft_map_max_abs_difference"] <= 1e-5
        ),
        "test_features_not_loaded": not bool(
            official_audit["test_features_loaded"]
        ),
        "test_targets_not_loaded": not bool(
            official_audit["test_targets_loaded"]
        ),
        "test_gallery_not_constructed": not bool(
            official_audit["test_gallery_constructed"]
        ),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "go_to_one_shot_pointer_training": passed,
        "hard_macro_auc_gain_vs_quality": hard_gain,
        "hard_scenario_wins": wins,
        "scenario_auc_deltas": scenario_deltas,
        "non_anchor_fraction": non_anchor,
        "wrong_frame_selection_fraction": wrong_selection,
        "teacher_loo_equivalent_agreement": loo_equivalent,
        "teacher_replay_max_abs_difference": teacher_replay_max_abs_difference,
        "recommendation": (
            "proceed_to_phase2_one_shot_pointer"
            if passed
            else "stop_before_training_and_revisit_route_target"
        ),
    }


def _report(
    *,
    split_checks: Mapping[str, Any],
    official_audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    calibration: Mapping[str, Any],
    validation: Mapping[str, Any],
    decision: Mapping[str, Any],
    source_integrity: Mapping[str, Any],
) -> str:
    core = _core_rows("route_validation", validation)
    core_lines = [
        "| Method | All AUC | Hard AUC | EER | Rank-1 | Margin | Teacher map cos |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in core:
        core_lines.append(
            f"| {row['method']} | {row['pooled_all_auc']:.6f} | "
            f"{row['hard_macro_auc']:.6f} | {row['eer_all']:.6f} | "
            f"{row['rank1_all']:.6f} | {row['mean_margin_all']:.6f} | "
            f"{row['teacher_map_cosine_all']:.6f} |"
        )
    quality = validation["methods"]["quality_anchor"]["verification"]
    oracle = validation["methods"]["hard_route_oracle"]["verification"]
    scenario_lines = [
        "| Scenario | Quality AUC | Hard route AUC | Delta |",
        "|---|---:|---:|---:|",
    ]
    for scenario in ("clean", *HARD_SCENARIOS):
        delta = oracle[scenario]["roc_auc"] - quality[scenario]["roc_auc"]
        scenario_lines.append(
            f"| {scenario} | {quality[scenario]['roc_auc']:.6f} | "
            f"{oracle[scenario]['roc_auc']:.6f} | {delta:+.6f} |"
        )
    diagnostics = validation["route_diagnostics"]
    stability = validation["teacher_loo_stability"]
    failed = [
        name for name, passed in decision["checks"].items() if not passed
    ]
    status = "GO" if decision["go_to_one_shot_pointer_training"] else "NO-GO"
    return f"""# PointerRoute Phase 0/1 Report

## Protocol freeze

- Source: the existing Stage16 train split only.
- Identity carve-out: 800 route-train / 100 route-calibration /
  100 route-validation identities, with five unchanged scenarios per identity.
- Selection used route-calibration only. Route-validation was evaluated once
  with the selected delta and temperature.
- Existing PCA, codebook, condition sets, teacher sets, and gallery protocol
  were not refit or changed.
- Official val/test identity metadata was used only for disjointness auditing.
  Test features, targets, and gallery were not loaded.
- Split integrity passed: {split_checks['passed']}.
- Train/val/test metadata disjointness passed:
  {official_audit['route_union_disjoint_from_official_val'] and official_audit['route_union_disjoint_from_official_test']}.

## Calibration

- Selected anchor margin delta: {selection['delta']:.6f}.
- Selected soft-target temperature: {selection['temperature']:.6f}.
- Calibration hard-route hard AUC:
  {calibration['methods']['hard_route_oracle']['verification']['hard_average']['roc_auc']:.6f}.
- Calibration quality hard AUC:
  {calibration['methods']['quality_anchor']['verification']['hard_average']['roc_auc']:.6f}.

## Route-validation core comparison

{chr(10).join(core_lines)}

## Scenario comparison

{chr(10).join(scenario_lines)}

## Route diagnostics

- Non-anchor route fraction: {diagnostics['all']['non_anchor_fraction']:.6f}.
- Wrong-ID frame share among selected frame routes:
  {diagnostics['wrong_identity']['wrong_frame_selection_fraction_of_frame_routes']:.6f}.
- Selected-frame visible fraction in complementary occlusion:
  {diagnostics['complementary_occlusion']['selected_frame_visible_fraction']:.6f}.
- Full-vs-leave-one-teacher exact route agreement:
  {stability['all']['exact_route_agreement']:.6f}.
- Full-vs-leave-one-teacher epsilon-equivalent agreement:
  {stability['all']['equivalent_route_agreement']:.6f}.
- Cached teacher target replay max absolute difference:
  {source_integrity['route_validation']['teacher_replay_max_abs_difference']:.8f}.
- Permuted-frame route disagreement:
  {validation['permutation_check']['route_disagreement_fraction']:.8f}.

## Gate decision: {status}

- Hard macro AUC gain over quality:
  {decision['hard_macro_auc_gain_vs_quality']:+.6f}.
- Hard-scenario wins: {decision['hard_scenario_wins']} / 4.
- Failed checks: {', '.join(failed) if failed else 'none'}.
- Recommendation: `{decision['recommendation']}`.

This is a deterministic development oracle audit, not a statistical-
significance claim. The oracle uses disjoint teacher features and is an upper
bound unavailable at inference time.
"""


def main() -> int:
    args = parse_args()
    resolved = load_config(args.config, args.profile)
    config = resolved.to_dict()
    route_config = config["pointer_route"]
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    official_split_root = args.official_split_root.expanduser().resolve()
    seed = int(config["project"]["seed"])
    seed_everything(seed, deterministic=bool(config["runtime"]["deterministic"]))

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
        "stage": "pointer-route-phase01",
        "source_split": "train",
        "selection_split": "route_calibration",
        "evaluation_split": "route_validation",
        "test_features_loaded": False,
        "test_targets_loaded": False,
        "test_gallery_constructed": False,
        "pointer_route": route_config,
        "teacher": config["teacher"],
        "far_targets": config["evaluation"]["far_targets"],
        "config_sha256": config_sha256(config),
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "pointer_route_phase01_started",
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        official_split_root=official_split_root,
        test_features_loaded=False,
    )

    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=False,
    )
    scenarios = [str(value) for value in config["evaluation"]["scenarios"]]
    route_split = build_route_dev_split(
        dataset.synthetic.rows,
        seed=seed + int(route_config["seed_offset"]),
        train_identities=int(route_config["route_train_identities"]),
        calibration_identities=int(
            route_config["route_calibration_identities"]
        ),
        validation_identities=int(
            route_config["route_validation_identities"]
        ),
        required_scenarios=scenarios,
    )
    official_audit = _official_identity_audit(
        official_split_root,
        route_split.identity_ids,
    )
    split_payload = {
        "schema_version": 1,
        "seed": seed + int(route_config["seed_offset"]),
        "source_split": "train",
        "identity_ids": route_split.identity_ids,
        "sample_indices": route_split.sample_indices,
        "sample_ids": {
            name: [
                str(dataset.synthetic.rows[index]["sample_id"])
                for index in indices
            ]
            for name, indices in route_split.sample_indices.items()
        },
        "checks": route_split.checks,
        "official_identity_audit": official_audit,
    }
    _write_json(run_paths.artifacts / "route_dev_split.json", split_payload)
    identity_rows = [
        {"route_split": split, "identity_id": identity}
        for split, identities in route_split.identity_ids.items()
        for identity in identities
    ]
    _write_csv(
        run_paths.artifacts / "route_dev_identities.csv",
        identity_rows,
    )
    if not route_split.checks["passed"]:
        raise RuntimeError("route development split failed integrity checks")
    if not all(
        (
            official_audit["route_union_is_subset_of_official_train"],
            official_audit["route_union_disjoint_from_official_val"],
            official_audit["route_union_disjoint_from_official_test"],
        )
    ):
        raise RuntimeError("route development identities cross official splits")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("expanded PointerRoute audit requires CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    calibration_data = _load_subset(
        dataset=dataset,
        indices=route_split.sample_indices["route_calibration"],
        config=config,
        device=device,
    )
    validation_data = _load_subset(
        dataset=dataset,
        indices=route_split.sample_indices["route_validation"],
        config=config,
        device=device,
    )
    source_integrity = {
        "dataset_manifest": {
            "path": str((dataset_root / "manifest.json").resolve()),
            "sha256": _sha256(dataset_root / "manifest.json"),
        },
        "sets_jsonl": {
            "path": str((dataset_root / "train" / "sets.jsonl").resolve()),
            "sha256": _sha256(dataset_root / "train" / "sets.jsonl"),
        },
        "projection": {
            "path": str((quantization_root / "projection.pt").resolve()),
            "sha256": _sha256(quantization_root / "projection.pt"),
        },
        "codebook": {
            "path": str((quantization_root / "codebook.pt").resolve()),
            "sha256": _sha256(quantization_root / "codebook.pt"),
            "refit": False,
            "used_by_route_oracle": False,
        },
        "train_targets": {
            "path": str((quantization_root / "train_targets.pt").resolve()),
            "sha256": _sha256(quantization_root / "train_targets.pt"),
        },
        "route_calibration": {
            "selected_rows_sha256": calibration_data[
                "selected_rows_sha256"
            ],
            "teacher_replay_max_abs_difference": calibration_data[
                "teacher_replay_max_abs_difference"
            ],
        },
        "route_validation": {
            "selected_rows_sha256": validation_data[
                "selected_rows_sha256"
            ],
            "teacher_replay_max_abs_difference": validation_data[
                "teacher_replay_max_abs_difference"
            ],
        },
        "pca_refit": False,
        "codebook_refit": False,
        "condition_or_teacher_sets_modified": False,
        "test_features_loaded": False,
        "test_targets_loaded": False,
    }
    _write_json(run_paths.artifacts / "source_integrity.json", source_integrity)

    evaluation_kwargs = {
        "equivalent_epsilon": float(route_config["equivalent_epsilon"]),
        "far_targets": config["evaluation"]["far_targets"],
    }
    calibration_grid: list[dict[str, Any]] = []
    calibration_evaluations: dict[str, dict[str, Any]] = {}
    for delta in route_config["delta_candidates"]:
        for temperature in route_config["temperature_candidates"]:
            evaluation = evaluate_route_oracle(
                condition_features=calibration_data["condition_features"],
                condition_quality=calibration_data["condition_quality"],
                teacher_maps=calibration_data["teacher_maps"],
                loo_teacher_maps=calibration_data["loo_teacher_maps"],
                identities=calibration_data["identities"],
                source_identities=calibration_data["source_identities"],
                visibility=calibration_data["visibility"],
                scenarios=calibration_data["scenarios"],
                delta=float(delta),
                temperature=float(temperature),
                **evaluation_kwargs,
            )
            row = _calibration_row(evaluation)
            calibration_grid.append(row)
            calibration_evaluations[
                f"d{float(delta):g}_t{float(temperature):g}"
            ] = _without_targets(evaluation)
    selection = select_calibration_pair(
        calibration_grid,
        auc_tie_tolerance=float(route_config["auc_tie_tolerance"]),
    )
    calibration = evaluate_route_oracle(
        condition_features=calibration_data["condition_features"],
        condition_quality=calibration_data["condition_quality"],
        teacher_maps=calibration_data["teacher_maps"],
        loo_teacher_maps=calibration_data["loo_teacher_maps"],
        identities=calibration_data["identities"],
        source_identities=calibration_data["source_identities"],
        visibility=calibration_data["visibility"],
        scenarios=calibration_data["scenarios"],
        delta=selection["delta"],
        temperature=selection["temperature"],
        **evaluation_kwargs,
    )
    validation = evaluate_route_oracle(
        condition_features=validation_data["condition_features"],
        condition_quality=validation_data["condition_quality"],
        teacher_maps=validation_data["teacher_maps"],
        loo_teacher_maps=validation_data["loo_teacher_maps"],
        identities=validation_data["identities"],
        source_identities=validation_data["source_identities"],
        visibility=validation_data["visibility"],
        scenarios=validation_data["scenarios"],
        delta=selection["delta"],
        temperature=selection["temperature"],
        **evaluation_kwargs,
    )
    torch.save(
        {
            "schema_version": 1,
            "split": "route_calibration",
            "sample_ids": calibration_data["sample_ids"],
            "delta": selection["delta"],
            "temperature": selection["temperature"],
            **calibration["targets"],
        },
        run_paths.artifacts / "calibration_route_targets.pt",
    )
    torch.save(
        {
            "schema_version": 1,
            "split": "route_validation",
            "sample_ids": validation_data["sample_ids"],
            "delta": selection["delta"],
            "temperature": selection["temperature"],
            **validation["targets"],
        },
        run_paths.artifacts / "validation_route_targets.pt",
    )
    calibration_json = _without_targets(calibration)
    validation_json = _without_targets(validation)
    _write_csv(
        run_paths.artifacts / "calibration_grid.csv",
        calibration_grid,
    )
    _write_json(
        run_paths.artifacts / "calibration_grid.json",
        calibration_evaluations,
    )
    _write_json(run_paths.artifacts / "selection.json", selection)
    _write_json(
        run_paths.artifacts / "calibration_evaluation.json",
        calibration_json,
    )
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation_json,
    )
    _write_csv(
        run_paths.artifacts / "core_comparison.csv",
        [
            *_core_rows("route_calibration", calibration_json),
            *_core_rows("route_validation", validation_json),
        ],
    )
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        [
            *_scenario_rows("route_calibration", calibration_json),
            *_scenario_rows("route_validation", validation_json),
        ],
    )
    _write_csv(
        run_paths.artifacts / "route_diagnostics.csv",
        [
            *_diagnostic_rows("route_calibration", calibration_json),
            *_diagnostic_rows("route_validation", validation_json),
        ],
    )

    decision = _decision(
        split_checks=route_split.checks,
        official_audit=official_audit,
        evaluation=validation_json,
        gates=route_config["gates"],
        teacher_replay_max_abs_difference=validation_data[
            "teacher_replay_max_abs_difference"
        ],
    )
    _write_json(run_paths.artifacts / "decision.json", decision)
    (run_paths.artifacts / "REPORT.md").write_text(
        _report(
            split_checks=route_split.checks,
            official_audit=official_audit,
            selection=selection,
            calibration=calibration_json,
            validation=validation_json,
            decision=decision,
            source_integrity=source_integrity,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "pointer-route-phase01",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "official_split_root": str(official_split_root),
        "route_split": split_payload,
        "source_integrity": source_integrity,
        "selection": selection,
        "calibration": calibration_json,
        "validation": validation_json,
        "decision": decision,
        "peak_cuda_allocated_mib": max(
            calibration_data["peak_cuda_allocated_mib"],
            validation_data["peak_cuda_allocated_mib"],
        ),
        "test_features_loaded": False,
        "test_targets_loaded": False,
        "test_gallery_constructed": False,
    }
    logger.write_manifest(manifest)
    logger.log_event("pointer_route_phase01_completed", decision=decision)
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
