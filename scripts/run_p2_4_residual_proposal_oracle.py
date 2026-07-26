"""Audit frozen residual top-k proposals with exact and local oracles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_p2_2_residual_stability import _IndexedTokenDataset, _load_model
from run_p2_3_risk_controlled_refinement import _load_p2_2_lock
from run_p2_residual_evidence import _core_rows, _scenario_rows
from run_pointer_route_phase3 import _phase01_lock
from run_pointer_route_rollout_rescue import CudaActivityMonitor
from set2proto.config import load_config
from set2proto.environment import collect_environment, config_sha256
from set2proto.experiment_logging import ExperimentLogger
from set2proto.reproducibility import seed_everything
from set2proto.residual_correction_evaluation import (
    decode_one_shot_dataset,
    evaluate_residual_token_predictions,
    transition_statistics,
)
from set2proto.residual_evidence import quality_weighted_anchor
from set2proto.residual_proposal_oracle import proposal_oracle_decode
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
    parser.add_argument("--p2-3-root", type=Path, required=True)
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


def _load_p2_3_lock(root: Path) -> dict[str, Any]:
    required = {
        "protocol": root / "artifacts" / "protocol_preregistration.json",
        "selection": root / "artifacts" / "selection.json",
        "decision": root / "artifacts" / "decision.json",
        "manifest": root / "logs" / "run_manifest.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing P2-3 {name}: {path}")
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
    }
    if values["protocol"].get("test_loaded") is not False:
        raise RuntimeError("P2-3 protocol loaded test")
    if values["manifest"].get("test_loaded") is not False:
        raise RuntimeError("P2-3 manifest loaded test")
    values["artifact_hashes"] = {
        name: _sha256(path) for name, path in required.items()
    }
    return values


def _empty_trace(rounds: int, top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "round": index + 1,
            "proposed": 0,
            "proposed_wrong": 0,
            "wrong_before": 0,
            "target_available_for_proposed_wrong": 0,
            "target_rank_counts": {
                str(rank): 0 for rank in range(1, top_k + 1)
            },
            "oracle_corrected": 0,
            "oracle_injured": 0,
            "oracle_changed": 0,
            "naive_top1_corrected": 0,
            "naive_top1_injured": 0,
        }
        for index in range(rounds)
    ]


@torch.inference_mode()
def _decode_dataset(
    *,
    model: torch.nn.Module,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    proposal_fraction: float,
    top_k: int,
    rounds: int,
    mode: str,
) -> dict[str, Any]:
    batch_size = int(config["evaluation"]["batch_size"])
    initial_parts: list[torch.Tensor] = []
    token_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    trace = _empty_trace(rounds, top_k)
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        condition = dataset._precomputed["condition_features"][
            start:end
        ].to(device)
        quality = dataset._precomputed["condition_quality"][start:end].to(
            device
        )
        target = dataset.targets["tokens"][start:end].to(device)
        teacher = dataset.targets["continuous_prototypes"][start:end].to(
            device
        )
        anchor = quality_weighted_anchor(condition, quality)
        result = proposal_oracle_decode(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            target_tokens=target,
            teacher_map=teacher,
            quality_anchor=anchor,
            residual_codebook=codebook,
            config=config,
            proposal_fraction=proposal_fraction,
            top_k=top_k,
            rounds=rounds,
            mode=mode,
        )
        initial_parts.append(result.initial_tokens.cpu())
        token_parts.append(result.tokens.cpu())
        target_parts.append(target.cpu())
        for index, row in enumerate(result.trace):
            for key in trace[index]:
                if key in {"round", "target_rank_counts"}:
                    continue
                trace[index][key] += int(row[key])
            for rank, count in row["target_rank_counts"].items():
                trace[index]["target_rank_counts"][rank] += int(count)
    initial = torch.cat(initial_parts)
    tokens = torch.cat(token_parts)
    targets = torch.cat(target_parts)
    transitions = transition_statistics(
        initial_tokens=initial,
        final_tokens=tokens,
        target_tokens=targets,
        scenarios=list(dataset._precomputed["scenarios"]),
    )
    return {
        "initial_tokens": initial,
        "tokens": tokens,
        "trace": trace,
        "transitions": transitions,
    }


def _candidate_row(
    *,
    proposal_fraction: float,
    top_k: int,
    rounds: int,
    evaluation: Mapping[str, Any],
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["verification"]
    matched = methods["matched_one_shot"]["hard_average"]["roc_auc"]
    rollout = methods["rollout_one_shot"]["hard_average"]["roc_auc"]
    oracle = methods["exact_proposal_oracle"]["hard_average"]["roc_auc"]
    transition = decoded["transitions"]["all"]
    return {
        "proposal_fraction": proposal_fraction,
        "top_k": top_k,
        "rounds": rounds,
        "matched_one_shot_hard_auc": matched,
        "rollout_one_shot_hard_auc": rollout,
        "exact_oracle_hard_auc": oracle,
        "gain_over_matched_one_shot": oracle - matched,
        "gain_over_rollout_one_shot": oracle - rollout,
        "exact_oracle_all_auc": methods["exact_proposal_oracle"]["all"][
            "roc_auc"
        ],
        "correctable_wrong_fraction": transition["ecr"],
        "net_correction_fraction": transition["net_correction_fraction"],
        "change_fraction": transition["change_fraction"],
    }


def _select_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    best = max(float(row["exact_oracle_hard_auc"]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if float(row["exact_oracle_hard_auc"]) >= best - tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["gain_over_matched_one_shot"]),
            float(row["correctable_wrong_fraction"]),
            -int(row["rounds"]),
            -int(row["top_k"]),
            -float(row["proposal_fraction"]),
        ),
    )
    return {
        "selection_split": "route_calibration",
        "objective": (
            "maximize exact proposal-oracle hard AUC; within tolerance "
            "maximize gain over matched one-shot and correctable-wrong "
            "fraction, then prefer the smaller oracle budget"
        ),
        "auc_tie_tolerance": tolerance,
        "best_hard_auc": best,
        "eligible_candidates": len(eligible),
        "selected_candidate": selected,
        "validation_used": False,
        "test_used": False,
    }


@torch.inference_mode()
def _permutation_check(
    *,
    model: torch.nn.Module,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    batch = default_collate(
        [dataset[index] for index in range(min(16, len(dataset)))]
    )
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    targets = batch["target_tokens"].to(device)
    teacher = dataset.targets["continuous_prototypes"][: len(targets)].to(
        device
    )
    anchor = quality_weighted_anchor(condition, quality)

    def decode(condition_value, quality_value):
        return proposal_oracle_decode(
            model=model,
            condition_features=condition_value,
            condition_quality=quality_value,
            target_tokens=targets,
            teacher_map=teacher,
            quality_anchor=anchor,
            residual_codebook=codebook,
            config=config,
            proposal_fraction=float(selected["proposal_fraction"]),
            top_k=int(selected["top_k"]),
            rounds=int(selected["rounds"]),
            mode="exact_token",
        )

    first = decode(condition, quality)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    second = decode(condition[:, permutation], quality[:, permutation])
    return {
        "permutation": permutation.cpu().tolist(),
        "initial_token_disagreement": float(
            (first.initial_tokens != second.initial_tokens).float().mean().item()
        ),
        "oracle_token_disagreement": float(
            (first.tokens != second.tokens).float().mean().item()
        ),
    }


@torch.inference_mode()
def _measure_latency(
    *,
    model: torch.nn.Module,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    batch = default_collate(
        [dataset[index] for index in range(min(16, len(dataset)))]
    )
    condition = batch["condition_features"].to(device)
    quality = batch["condition_quality"].to(device)
    target = batch["target_tokens"].to(device)
    teacher = dataset.targets["continuous_prototypes"][: len(target)].to(
        device
    )
    anchor = quality_weighted_anchor(condition, quality)

    def run(mode: str) -> None:
        proposal_oracle_decode(
            model=model,
            condition_features=condition,
            condition_quality=quality,
            target_tokens=target,
            teacher_map=teacher,
            quality_anchor=anchor,
            residual_codebook=codebook,
            config=config,
            proposal_fraction=float(selected["proposal_fraction"]),
            top_k=int(selected["top_k"]),
            rounds=int(selected["rounds"]),
            mode=mode,
        )

    result = {}
    for mode in ("exact_token", "local_teacher"):
        for _ in range(10):
            run(mode)
        torch.cuda.synchronize(device)
        values = []
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(100):
            started = time.perf_counter()
            run(mode)
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)
        tensor = torch.tensor(values)
        result[mode] = {
            "batch_size": int(condition.shape[0]),
            "runs": 100,
            "mean_batch_ms": float(tensor.mean().item()),
            "mean_per_set_ms": float(
                tensor.mean().item() / condition.shape[0]
            ),
            "p50_batch_ms": float(tensor.median().item()),
            "peak_reserved_mib": float(
                torch.cuda.max_memory_reserved(device) / 1024**2
            ),
        }
    return result


def _decision(
    *,
    evaluation: Mapping[str, Any],
    exact_decoded: Mapping[str, Any],
    selection: Mapping[str, Any],
    permutation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["verification"]
    matched = methods["matched_one_shot"]
    exact = methods["exact_proposal_oracle"]
    local = methods["local_teacher_proposal_oracle"]
    matched_hard = matched["hard_average"]["roc_auc"]
    exact_hard = exact["hard_average"]["roc_auc"]
    local_hard = local["hard_average"]["roc_auc"]
    exact_deltas = {
        scenario: exact[scenario]["roc_auc"] - matched[scenario]["roc_auc"]
        for scenario in HARD_SCENARIOS
    }
    scenario_wins = sum(value > 1e-12 for value in exact_deltas.values())
    correctable = exact_decoded["transitions"]["all"]["ecr"]
    gates = config["p2_4"]["gates"]
    checks = {
        "selection_on_calibration_only": (
            selection["validation_used"] is False
            and selection["test_used"] is False
        ),
        "exact_oracle_has_headroom": (
            exact_hard - matched_hard
            >= float(
                gates[
                    "min_exact_oracle_hard_auc_gain_over_matched_one_shot"
                ]
            )
        ),
        "local_oracle_has_headroom": (
            local_hard - matched_hard
            >= float(
                gates[
                    "min_local_oracle_hard_auc_gain_over_matched_one_shot"
                ]
            )
        ),
        "hard_scenario_wins": (
            scenario_wins >= int(gates["min_hard_scenario_wins"])
        ),
        "wrong_token_candidate_recall": (
            correctable
            >= float(gates["min_wrong_token_correctable_fraction"])
        ),
        "permutation_invariant": (
            max(
                float(permutation["initial_token_disagreement"]),
                float(permutation["oracle_token_disagreement"]),
            )
            <= float(gates["max_permutation_difference"])
        ),
        "test_not_loaded": True,
    }
    exact_headroom = checks["exact_oracle_has_headroom"]
    local_headroom = checks["local_oracle_has_headroom"]
    if exact_headroom and checks["hard_scenario_wins"]:
        classification = "GO_PROPOSAL_HEADROOM"
        recommendation = (
            "proposal candidates are sufficient; replace the current gate "
            "with identity-margin/listwise candidate ranking"
        )
    elif local_headroom:
        classification = "GO_TARGET_REDESIGN"
        recommendation = (
            "continuous local candidates help but exact residual targets do "
            "not; redesign identity-aligned targets or use continuous residuals"
        )
    else:
        classification = "NO_GO_PROPOSAL_MODEL"
        recommendation = (
            "frozen top-k proposals lack headroom; retrain the proposal model "
            "or switch to continuous residual aggregation before D3PM"
        )
    return {
        "classification": classification,
        "checks": checks,
        "matched_one_shot_hard_auc": matched_hard,
        "exact_oracle_hard_auc": exact_hard,
        "local_oracle_hard_auc": local_hard,
        "exact_gain_over_matched_one_shot": exact_hard - matched_hard,
        "local_gain_over_matched_one_shot": local_hard - matched_hard,
        "exact_scenario_deltas_vs_matched": exact_deltas,
        "hard_scenario_wins": scenario_wins,
        "correctable_wrong_fraction": correctable,
        "exact_transition": exact_decoded["transitions"]["all"],
        "recommendation": recommendation,
    }


def _report(
    *,
    selection: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    exact_decoded: Mapping[str, Any],
    local_decoded: Mapping[str, Any],
    decision: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    methods = evaluation["verification"]

    def hard(name: str) -> float:
        return float(methods[name]["hard_average"]["roc_auc"])

    selected = selection["selected_candidate"]
    exact_transition = exact_decoded["transitions"]["all"]
    local_transition = local_decoded["transitions"]["all"]
    scenario_table = "\n".join(
        f"| {scenario} | "
        f"{methods['matched_one_shot'][scenario]['roc_auc']:.6f} | "
        f"{methods['exact_proposal_oracle'][scenario]['roc_auc']:.6f} | "
        f"{methods['local_teacher_proposal_oracle'][scenario]['roc_auc']:.6f} |"
        for scenario in HARD_SCENARIOS
    )
    return f"""# P2-4 Residual Proposal Oracle Audit

## Locked protocol

- The P2-2 selected rollout Transformer and all residual artifacts are frozen.
- Proposal fraction, top-k, and round count are selected on route-calibration
  using the exact-token proposal oracle.
- Exact oracle may accept a proposal only when the target residual token is in
  the frozen model's top-k candidates.
- Local-teacher oracle chooses among current plus top-k candidates by local
  continuous-teacher cosine and is evaluated only after the protocol is locked.
- Route-validation is used once; test and official val are not loaded.

## Selected proposal budget

- Proposal fraction: {selected['proposal_fraction']}.
- Top-k: {selected['top_k']}.
- Rounds: {selected['rounds']}.
- Calibration exact-oracle gain over matched one-shot:
  {selected['gain_over_matched_one_shot']:+.6f}.

## Route-validation

| Method | Hard AUC |
|---|---:|
| Quality anchor | {hard('quality_anchor'):.6f} |
| Residual quantization oracle | {hard('teacher_residual_quantized'):.6f} |
| Matched one-shot | {hard('matched_one_shot'):.6f} |
| Rollout one-shot | {hard('rollout_one_shot'):.6f} |
| Exact-token proposal oracle | {hard('exact_proposal_oracle'):.6f} |
| Local-teacher proposal oracle | {hard('local_teacher_proposal_oracle'):.6f} |

| Scenario | Matched | Exact oracle | Local oracle |
|---|---:|---:|---:|
{scenario_table}

## Proposal recall and transition

- Exact correctable fraction among initially wrong tokens:
  {exact_transition['ecr']:.6f}.
- Exact changed / net-corrected fraction:
  {exact_transition['change_fraction']:.6f} /
  {exact_transition['net_correction_fraction']:+.6f}.
- Local changed / net-corrected fraction:
  {local_transition['change_fraction']:.6f} /
  {local_transition['net_correction_fraction']:+.6f}.

## Oracle runtime

- Exact: {latency['exact_token']['mean_batch_ms']:.3f} ms/batch16.
- Local: {latency['local_teacher']['mean_batch_ms']:.3f} ms/batch16.
  These are diagnostic oracle costs, not deployable inference latencies.

## Decision: {decision['classification']}

- Exact gain over matched one-shot:
  {decision['exact_gain_over_matched_one_shot']:+.6f}.
- Local gain over matched one-shot:
  {decision['local_gain_over_matched_one_shot']:+.6f}.
- Hard scenario wins for exact oracle: {decision['hard_scenario_wins']}.
- Recommendation: {decision['recommendation']}.

This is a mechanism audit on an existing development validation split, not a
statistical-significance claim.
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.profile).to_dict()
    seed = int(config["project"]["seed"])
    seed_everything(seed, deterministic=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    residual_root = args.residual_artifact_root.expanduser().resolve()
    residual_manifest = json.loads(
        (residual_root / "manifest.json").read_text(encoding="utf-8")
    )
    phase01 = _phase01_lock(
        root=args.phase01_root.expanduser().resolve(),
        dataset_root=dataset_root,
        quantization_root=Path(
            residual_manifest["source_absolute_quantization"]
        ),
    )
    p2_2 = _load_p2_2_lock(args.p2_2_root.expanduser().resolve())
    p2_3 = _load_p2_3_lock(args.p2_3_root.expanduser().resolve())
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
        "stage": "p2-4-residual-proposal-oracle",
        "mode": args.stage,
        "protocol": config["p2_4"],
        "p2_2_selected_step": p2_2["selected_step"],
        "p2_2_checkpoint_hashes": p2_2["checkpoint_hashes"],
        "p2_3_artifact_hashes": p2_3["artifact_hashes"],
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
    logger.log_event("p2_4_started", mode=args.stage, test_loaded=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P2-4 requires CUDA")
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
    matched_model = _load_model(
        config=config,
        checkpoint=p2_2["checkpoints"]["matched_one_shot"],
        device=device,
    )
    for model in (rollout_model, matched_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    calibration_dataset = _IndexedTokenDataset(
        dataset,
        (
            route_indices["route_calibration"][:16]
            if args.stage == "smoke"
            else route_indices["route_calibration"]
        ),
    )
    if args.stage == "smoke":
        with CudaActivityMonitor() as monitor:
            exact = _decode_dataset(
                model=rollout_model,
                dataset=calibration_dataset,
                codebook=codebook,
                config=config,
                device=device,
                proposal_fraction=0.3,
                top_k=2,
                rounds=1,
                mode="exact_token",
            )
            local = _decode_dataset(
                model=rollout_model,
                dataset=calibration_dataset,
                codebook=codebook,
                config=config,
                device=device,
                proposal_fraction=0.3,
                top_k=2,
                rounds=1,
                mode="local_teacher",
            )
        smoke = {
            "status": "passed",
            "device": str(device),
            "samples": len(calibration_dataset),
            "exact_transition": exact["transitions"]["all"],
            "local_transition": local["transitions"]["all"],
            "exact_injuries": sum(
                int(row["oracle_injured"]) for row in exact["trace"]
            ),
            "gpu_activity": monitor.summary(),
            "test_loaded": False,
        }
        if smoke["exact_injuries"] != 0:
            raise RuntimeError("exact proposal oracle caused token injury")
        _write_json(run_paths.artifacts / "smoke_result.json", smoke)
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p2-4-smoke",
                "status": "passed",
                "environment": environment,
                "resolved_config": config,
                "protocol_preregistration": preregistration,
                "smoke": smoke,
                "test_loaded": False,
            }
        )
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return 0

    matched_calibration = decode_one_shot_dataset(
        model=matched_model,
        dataset=calibration_dataset,
        device=device,
        batch_size=int(config["evaluation"]["batch_size"]),
    )
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    with CudaActivityMonitor() as monitor:
        for fraction in config["p2_4"]["proposal_fraction_candidates"]:
            for top_k in config["p2_4"]["top_k_candidates"]:
                for rounds in config["p2_4"]["round_candidates"]:
                    decoded = _decode_dataset(
                        model=rollout_model,
                        dataset=calibration_dataset,
                        codebook=codebook,
                        config=config,
                        device=device,
                        proposal_fraction=float(fraction),
                        top_k=int(top_k),
                        rounds=int(rounds),
                        mode="exact_token",
                    )
                    evaluation = evaluate_residual_token_predictions(
                        dataset=calibration_dataset,
                        residual_codebook=codebook,
                        predictions={
                            "matched_one_shot": matched_calibration,
                            "rollout_one_shot": decoded["initial_tokens"],
                            "exact_proposal_oracle": decoded["tokens"],
                        },
                        config=config,
                        device=device,
                    )
                    row = _candidate_row(
                        proposal_fraction=float(fraction),
                        top_k=int(top_k),
                        rounds=int(rounds),
                        evaluation=evaluation,
                        decoded=decoded,
                    )
                    rows.append(row)
                    key = f"f{float(fraction):g}_k{int(top_k)}_r{int(rounds)}"
                    diagnostics[key] = {
                        "candidate": row,
                        "trace": decoded["trace"],
                        "transitions": decoded["transitions"],
                    }
        selection = _select_candidate(
            rows,
            tolerance=float(config["p2_4"]["auc_tie_tolerance"]),
        )
        selected = selection["selected_candidate"]
        validation_dataset = _IndexedTokenDataset(
            dataset,
            route_indices["route_validation"],
        )
        matched_validation = decode_one_shot_dataset(
            model=matched_model,
            dataset=validation_dataset,
            device=device,
            batch_size=int(config["evaluation"]["batch_size"]),
        )
        exact = _decode_dataset(
            model=rollout_model,
            dataset=validation_dataset,
            codebook=codebook,
            config=config,
            device=device,
            proposal_fraction=float(selected["proposal_fraction"]),
            top_k=int(selected["top_k"]),
            rounds=int(selected["rounds"]),
            mode="exact_token",
        )
        local = _decode_dataset(
            model=rollout_model,
            dataset=validation_dataset,
            codebook=codebook,
            config=config,
            device=device,
            proposal_fraction=float(selected["proposal_fraction"]),
            top_k=int(selected["top_k"]),
            rounds=int(selected["rounds"]),
            mode="local_teacher",
        )
        validation = evaluate_residual_token_predictions(
            dataset=validation_dataset,
            residual_codebook=codebook,
            predictions={
                "matched_one_shot": matched_validation,
                "rollout_one_shot": exact["initial_tokens"],
                "exact_proposal_oracle": exact["tokens"],
                "local_teacher_proposal_oracle": local["tokens"],
            },
            config=config,
            device=device,
        )
        permutation = _permutation_check(
            model=rollout_model,
            dataset=validation_dataset,
            codebook=codebook,
            config=config,
            device=device,
            selected=selected,
        )
        latency = _measure_latency(
            model=rollout_model,
            dataset=validation_dataset,
            codebook=codebook,
            config=config,
            device=device,
            selected=selected,
        )
    gpu_activity = monitor.summary()
    decision = _decision(
        evaluation=validation,
        exact_decoded=exact,
        selection=selection,
        permutation=permutation,
        config=config,
    )
    _write_csv(run_paths.artifacts / "calibration_candidates.csv", rows)
    _write_json(
        run_paths.artifacts / "calibration_diagnostics.json",
        diagnostics,
    )
    _write_json(run_paths.artifacts / "selection.json", selection)
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(
        run_paths.artifacts / "proposal_diagnostics.json",
        {
            "exact": {
                "trace": exact["trace"],
                "transitions": exact["transitions"],
            },
            "local_teacher": {
                "trace": local["trace"],
                "transitions": local["transitions"],
            },
        },
    )
    _write_json(run_paths.artifacts / "permutation_check.json", permutation)
    _write_json(run_paths.artifacts / "latency.json", latency)
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
        selection=selection,
        evaluation=validation,
        exact_decoded=exact,
        local_decoded=local,
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
            "stage": "p2-4-residual-proposal-oracle",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "selection": selection,
            "validation": validation,
            "proposal_diagnostics": {
                "exact": exact["transitions"],
                "local_teacher": local["transitions"],
            },
            "permutation": permutation,
            "latency": latency,
            "gpu_activity": gpu_activity,
            "decision": decision,
            "official_val_loaded": False,
            "test_loaded": False,
        }
    )
    logger.log_event(
        "p2_4_completed",
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
