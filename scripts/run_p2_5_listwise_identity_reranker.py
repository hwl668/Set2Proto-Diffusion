"""Train and validate the P2-5 listwise identity-aware top-k reranker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from run_p2_4_residual_proposal_oracle import _load_p2_3_lock
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
from set2proto.residual_listwise_reranker import (
    ResidualListwiseReranker,
    generate_listwise_candidate_cache,
    listwise_rerank_decode,
    train_listwise_reranker,
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
    parser.add_argument("--p2-3-root", type=Path, required=True)
    parser.add_argument("--p2-4-root", type=Path, required=True)
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


def _load_p2_4_lock(root: Path) -> dict[str, Any]:
    required = {
        "protocol": root / "artifacts" / "protocol_preregistration.json",
        "selection": root / "artifacts" / "selection.json",
        "decision": root / "artifacts" / "decision.json",
        "manifest": root / "logs" / "run_manifest.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing P2-4 {name}: {path}")
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
    }
    if values["decision"].get("classification") != "GO_PROPOSAL_HEADROOM":
        raise RuntimeError("P2-4 did not authorize listwise reranking")
    if values["protocol"].get("test_loaded") is not False:
        raise RuntimeError("P2-4 protocol loaded test")
    if values["manifest"].get("test_loaded") is not False:
        raise RuntimeError("P2-4 manifest loaded test")
    values["artifact_hashes"] = {
        name: _sha256(path) for name, path in required.items()
    }
    return values


@torch.inference_mode()
def _decode_dataset(
    *,
    model: torch.nn.Module,
    reranker: ResidualListwiseReranker,
    dataset: TokenTrainingDataset,
    codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    model_logit_weight: float,
    replacement_threshold: float,
) -> dict[str, Any]:
    batch_size = int(config["evaluation"]["batch_size"])
    initial_parts: list[torch.Tensor] = []
    token_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    replacements = 0
    candidate_target_hits = 0
    total_positions = 0
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        condition = dataset._precomputed["condition_features"][
            start:end
        ].to(device)
        quality = dataset._precomputed["condition_quality"][start:end].to(
            device
        )
        targets = dataset.targets["tokens"][start:end].to(device)
        result = listwise_rerank_decode(
            model=model,
            reranker=reranker,
            condition_features=condition,
            condition_quality=quality,
            residual_codebook=codebook,
            config=config,
            model_logit_weight=model_logit_weight,
            replacement_threshold=replacement_threshold,
        )
        initial_parts.append(result.initial_tokens.cpu())
        token_parts.append(result.tokens.cpu())
        target_parts.append(targets.cpu())
        replacements += int(result.replacement_mask.sum().item())
        candidate_target_hits += int(
            (
                result.candidate_tokens == targets.unsqueeze(-1)
            ).any(dim=-1).sum().item()
        )
        total_positions += int(targets.numel())
    initial = torch.cat(initial_parts)
    tokens = torch.cat(token_parts)
    targets = torch.cat(target_parts)
    return {
        "initial_tokens": initial,
        "tokens": tokens,
        "transitions": transition_statistics(
            initial_tokens=initial,
            final_tokens=tokens,
            target_tokens=targets,
            scenarios=list(dataset._precomputed["scenarios"]),
        ),
        "replacements": replacements,
        "replacement_fraction": replacements / total_positions,
        "target_in_top_k_fraction": candidate_target_hits / total_positions,
    }


def _candidate_row(
    *,
    model_logit_weight: float,
    replacement_threshold: float,
    evaluation: Mapping[str, Any],
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["verification"]
    matched = methods["matched_one_shot"]
    rollout = methods["rollout_one_shot"]
    reranked = methods["listwise_identity_reranker"]
    matched_hard = matched["hard_average"]["roc_auc"]
    rollout_hard = rollout["hard_average"]["roc_auc"]
    reranked_hard = reranked["hard_average"]["roc_auc"]
    scenario_wins = sum(
        reranked[scenario]["roc_auc"] > matched[scenario]["roc_auc"]
        for scenario in HARD_SCENARIOS
    )
    transition = decoded["transitions"]["all"]
    return {
        "model_logit_weight": model_logit_weight,
        "replacement_threshold": replacement_threshold,
        "matched_one_shot_hard_auc": matched_hard,
        "rollout_one_shot_hard_auc": rollout_hard,
        "reranked_hard_auc": reranked_hard,
        "gain_over_matched_one_shot": reranked_hard - matched_hard,
        "gain_over_rollout_one_shot": reranked_hard - rollout_hard,
        "reranked_all_auc": reranked["all"]["roc_auc"],
        "hard_scenario_wins": scenario_wins,
        "clean_delta_vs_matched": (
            reranked["clean"]["roc_auc"] - matched["clean"]["roc_auc"]
        ),
        "net_correction_fraction": transition["net_correction_fraction"],
        "corrected_fraction": transition["corrected_fraction"],
        "injured_fraction": transition["injured_fraction"],
        "replacement_fraction": decoded["replacement_fraction"],
    }


def _select_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    best = max(float(row["reranked_hard_auc"]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if float(row["reranked_hard_auc"]) >= best - tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["gain_over_matched_one_shot"]),
            float(row["net_correction_fraction"]),
            int(row["hard_scenario_wins"]),
            -float(row["replacement_fraction"]),
            float(row["replacement_threshold"]),
            -float(row["model_logit_weight"]),
        ),
    )
    return {
        "selection_split": "route_calibration",
        "objective": (
            "maximize listwise-reranked hard AUC; within tolerance maximize "
            "gain over matched one-shot, net correction and scenario wins, "
            "then minimize replacement fraction"
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
    reranker: ResidualListwiseReranker,
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

    def decode(condition_value, quality_value):
        return listwise_rerank_decode(
            model=model,
            reranker=reranker,
            condition_features=condition_value,
            condition_quality=quality_value,
            residual_codebook=codebook,
            config=config,
            model_logit_weight=float(selected["model_logit_weight"]),
            replacement_threshold=float(
                selected["replacement_threshold"]
            ),
        )

    first = decode(condition, quality)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    second = decode(condition[:, permutation], quality[:, permutation])
    return {
        "permutation": permutation.cpu().tolist(),
        "initial_token_disagreement": float(
            (first.initial_tokens != second.initial_tokens).float().mean().item()
        ),
        "final_token_disagreement": float(
            (first.tokens != second.tokens).float().mean().item()
        ),
        "maximum_score_difference": float(
            (first.combined_scores - second.combined_scores).abs().max().item()
        ),
    }


@torch.inference_mode()
def _measure_latency(
    *,
    model: torch.nn.Module,
    reranker: ResidualListwiseReranker,
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

    def run() -> None:
        listwise_rerank_decode(
            model=model,
            reranker=reranker,
            condition_features=condition,
            condition_quality=quality,
            residual_codebook=codebook,
            config=config,
            model_logit_weight=float(selected["model_logit_weight"]),
            replacement_threshold=float(
                selected["replacement_threshold"]
            ),
        )

    for _ in range(10):
        run()
    torch.cuda.synchronize(device)
    values = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(100):
        started = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        values.append((time.perf_counter() - started) * 1000.0)
    tensor = torch.tensor(values)
    return {
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


def _decision(
    *,
    evaluation: Mapping[str, Any],
    decoded: Mapping[str, Any],
    selection: Mapping[str, Any],
    permutation: Mapping[str, Any],
    latency: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["verification"]
    matched = methods["matched_one_shot"]
    rollout = methods["rollout_one_shot"]
    reranked = methods["listwise_identity_reranker"]
    matched_hard = matched["hard_average"]["roc_auc"]
    rollout_hard = rollout["hard_average"]["roc_auc"]
    reranked_hard = reranked["hard_average"]["roc_auc"]
    scenario_deltas = {
        scenario: reranked[scenario]["roc_auc"] - matched[scenario]["roc_auc"]
        for scenario in HARD_SCENARIOS
    }
    scenario_wins = sum(value > 1e-12 for value in scenario_deltas.values())
    transition = decoded["transitions"]["all"]
    gates = config["p2_5"]["gates"]
    maximum_permutation_difference = max(
        float(permutation["initial_token_disagreement"]),
        float(permutation["final_token_disagreement"]),
        float(permutation["maximum_score_difference"]),
    )
    checks = {
        "reranker_trained_on_route_train_only": True,
        "selection_on_route_calibration_only": (
            selection["validation_used"] is False
            and selection["test_used"] is False
        ),
        "beats_rollout_one_shot": (
            reranked_hard - rollout_hard
            >= float(gates["min_hard_auc_gain_over_rollout_one_shot"])
        ),
        "beats_matched_one_shot": (
            reranked_hard - matched_hard
            >= float(gates["min_hard_auc_gain_over_matched_one_shot"])
        ),
        "hard_scenario_wins": (
            scenario_wins >= int(gates["min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            matched["clean"]["roc_auc"] - reranked["clean"]["roc_auc"]
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
            float(latency["peak_reserved_mib"])
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024
        ),
        "test_not_loaded": True,
    }
    passed = all(checks.values())
    return {
        "classification": (
            "GO_LISTWISE_RERANKER" if passed else "NO_GO_P2_5"
        ),
        "checks": checks,
        "matched_one_shot_hard_auc": matched_hard,
        "rollout_one_shot_hard_auc": rollout_hard,
        "reranked_hard_auc": reranked_hard,
        "gain_over_matched_one_shot": reranked_hard - matched_hard,
        "gain_over_rollout_one_shot": reranked_hard - rollout_hard,
        "scenario_deltas_vs_matched": scenario_deltas,
        "hard_scenario_wins": scenario_wins,
        "transition": transition,
        "replacement_fraction": decoded["replacement_fraction"],
        "target_in_top_k_fraction": decoded["target_in_top_k_fraction"],
        "maximum_permutation_difference": maximum_permutation_difference,
        "peak_reserved_mib": latency["peak_reserved_mib"],
        "recommendation": (
            "freeze reranker and test it inside multi-step masked decoding"
            if passed
            else "inspect listwise ranking calibration before multi-step use"
        ),
    }


def _report(
    *,
    cache_diagnostics: Mapping[str, Any],
    training: Mapping[str, Any],
    selection: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    decision: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    methods = evaluation["verification"]

    def hard(name: str) -> float:
        return float(methods[name]["hard_average"]["roc_auc"])

    selected = selection["selected_candidate"]
    transition = decision["transition"]
    scenario_table = "\n".join(
        f"| {scenario} | "
        f"{methods['matched_one_shot'][scenario]['roc_auc']:.6f} | "
        f"{methods['listwise_identity_reranker'][scenario]['roc_auc']:.6f} | "
        f"{decision['scenario_deltas_vs_matched'][scenario]:+.6f} |"
        for scenario in HARD_SCENARIOS
    )
    failed = [
        key for key, value in decision["checks"].items() if not value
    ]
    return f"""# P2-5 Listwise Identity-Margin Top-8 Reranker

## Locked protocol

- The P2-2 rollout Transformer and residual codebook are frozen.
- Dense top-8 supervision uses route-train only.
- Training combines listwise candidate utility, clean-gallery
  genuine/impostor margin, and continuous local map loss.
- Teacher and gallery labels are never reranker inputs.
- Model-logit mixing and replacement threshold are selected on
  route-calibration before one route-validation evaluation.

## Cache and training

- Route-train samples: {cache_diagnostics['samples']}.
- Target-in-top-8 fraction: {cache_diagnostics['target_in_top_k_fraction']:.6f}.
- Clean gallery identities: {cache_diagnostics['gallery_identities']}.
- Reranker parameters: {training['parameter_count']}.
- Training time / peak memory:
  {training['elapsed_seconds']:.3f} s /
  {training['peak_reserved_mib']:.1f} MiB.

## Calibration selection

- Model-logit weight: {selected['model_logit_weight']}.
- Replacement threshold: {selected['replacement_threshold']}.
- Calibration gain over matched one-shot:
  {selected['gain_over_matched_one_shot']:+.6f}.

## Route-validation

| Method | Hard AUC |
|---|---:|
| Quality anchor | {hard('quality_anchor'):.6f} |
| Residual quantization oracle | {hard('teacher_residual_quantized'):.6f} |
| Matched one-shot | {hard('matched_one_shot'):.6f} |
| Rollout one-shot | {hard('rollout_one_shot'):.6f} |
| Listwise identity reranker | {hard('listwise_identity_reranker'):.6f} |

| Scenario | Matched | Reranked | Delta |
|---|---:|---:|---:|
{scenario_table}

## Transition

- Replacement fraction: {decision['replacement_fraction']:.6f}.
- Corrected / injured:
  {transition['corrected_fraction']:.6f} /
  {transition['injured_fraction']:.6f}.
- Net correction: {transition['net_correction_fraction']:+.6f}.
- ECR / EIR: {transition['ecr']:.6f} / {transition['eir']:.6f}.

## Runtime

- {latency['mean_batch_ms']:.3f} ms/batch16,
  {latency['mean_per_set_ms']:.3f} ms/set.
- Peak reserved memory: {latency['peak_reserved_mib']:.1f} MiB.

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Gain over rollout one-shot:
  {decision['gain_over_rollout_one_shot']:+.6f}.
- Gain over matched one-shot:
  {decision['gain_over_matched_one_shot']:+.6f}.
- Recommendation: `{decision['recommendation']}`.

This is a development result on the existing route-validation split, not a
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
    p2_4 = _load_p2_4_lock(args.p2_4_root.expanduser().resolve())
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
        "stage": "p2-5-listwise-identity-reranker",
        "mode": args.stage,
        "protocol": config["p2_5"],
        "p2_2_selected_step": p2_2["selected_step"],
        "p2_2_checkpoint_hashes": p2_2["checkpoint_hashes"],
        "p2_3_artifact_hashes": p2_3["artifact_hashes"],
        "p2_4_artifact_hashes": p2_4["artifact_hashes"],
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
    logger.log_event("p2_5_started", mode=args.stage, test_loaded=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P2-5 requires CUDA")
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
    cache = generate_listwise_candidate_cache(
        model=rollout_model,
        dataset=dataset,
        indices=route_indices["route_train"],
        residual_codebook=codebook,
        config=config,
        device=device,
        smoke=args.stage == "smoke",
    )
    _write_json(
        run_paths.artifacts / "candidate_cache_diagnostics.json",
        cache.diagnostics,
    )
    cache_path = run_paths.artifacts / "candidate_cache.pt"
    torch.save(
        {
            "sample_indices": cache.sample_indices,
            "features": cache.features,
            "candidate_tokens": cache.candidate_tokens,
            "target_distribution": cache.target_distribution,
            "genuine_gallery_embeddings": cache.genuine_gallery_embeddings,
            "hard_impostor_embeddings": cache.hard_impostor_embeddings,
            "diagnostics": cache.diagnostics,
        },
        cache_path,
    )
    with CudaActivityMonitor() as monitor:
        training_result = train_listwise_reranker(
            cache=cache,
            dataset=dataset,
            residual_codebook=codebook,
            config=config,
            device=device,
            seed=seed + 93001,
            smoke=args.stage == "smoke",
        )
    reranker = training_result.reranker
    training = {
        "steps": len(training_result.history),
        "parameter_count": reranker.parameter_count(),
        "elapsed_seconds": training_result.elapsed_seconds,
        "peak_reserved_mib": training_result.peak_reserved_mib,
        "transformer_trainable_parameters": sum(
            parameter.numel()
            for parameter in rollout_model.parameters()
            if parameter.requires_grad
        ),
        "device": str(device),
    }
    _write_json(
        run_paths.artifacts / "reranker_training_summary.json",
        training,
    )
    _write_csv(
        run_paths.artifacts / "reranker_training_history.csv",
        training_result.history,
    )
    checkpoint = run_paths.checkpoints / "listwise_reranker.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "stage": "p2-5-listwise-identity-reranker",
            "model_state": reranker.state_dict(),
            "parameter_count": reranker.parameter_count(),
            "p2_2_checkpoint_hash": p2_2["checkpoint_hashes"]["rollout"],
            "cache_sha256": _sha256(cache_path),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    calibration_dataset = _IndexedTokenDataset(
        dataset,
        (
            route_indices["route_calibration"][:16]
            if args.stage == "smoke"
            else route_indices["route_calibration"]
        ),
    )
    if args.stage == "smoke":
        decoded = _decode_dataset(
            model=rollout_model,
            reranker=reranker,
            dataset=calibration_dataset,
            codebook=codebook,
            config=config,
            device=device,
            model_logit_weight=0.25,
            replacement_threshold=0.1,
        )
        smoke = {
            "status": "passed",
            "device": str(device),
            "cache": cache.diagnostics,
            "training": training,
            "decode_shape": list(decoded["tokens"].shape),
            "transition": decoded["transitions"]["all"],
            "gpu_activity": monitor.summary(),
            "test_loaded": False,
        }
        _write_json(run_paths.artifacts / "smoke_result.json", smoke)
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p2-5-smoke",
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

    matched_model = _load_model(
        config=config,
        checkpoint=p2_2["checkpoints"]["matched_one_shot"],
        device=device,
    )
    matched_calibration = decode_one_shot_dataset(
        model=matched_model,
        dataset=calibration_dataset,
        device=device,
        batch_size=int(config["evaluation"]["batch_size"]),
    )
    calibration_rows: list[dict[str, Any]] = []
    calibration_diagnostics: dict[str, Any] = {}
    for model_weight in config["p2_5"]["calibration"][
        "model_logit_weight_candidates"
    ]:
        for threshold in config["p2_5"]["calibration"][
            "replacement_threshold_candidates"
        ]:
            decoded = _decode_dataset(
                model=rollout_model,
                reranker=reranker,
                dataset=calibration_dataset,
                codebook=codebook,
                config=config,
                device=device,
                model_logit_weight=float(model_weight),
                replacement_threshold=float(threshold),
            )
            evaluation = evaluate_residual_token_predictions(
                dataset=calibration_dataset,
                residual_codebook=codebook,
                predictions={
                    "matched_one_shot": matched_calibration,
                    "rollout_one_shot": decoded["initial_tokens"],
                    "listwise_identity_reranker": decoded["tokens"],
                },
                config=config,
                device=device,
            )
            row = _candidate_row(
                model_logit_weight=float(model_weight),
                replacement_threshold=float(threshold),
                evaluation=evaluation,
                decoded=decoded,
            )
            calibration_rows.append(row)
            key = f"w{float(model_weight):g}_t{float(threshold):g}"
            calibration_diagnostics[key] = {
                "candidate": row,
                "transitions": decoded["transitions"],
            }
    selection = _select_candidate(
        calibration_rows,
        tolerance=float(
            config["p2_5"]["calibration"]["auc_tie_tolerance"]
        ),
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
    validation_decoded = _decode_dataset(
        model=rollout_model,
        reranker=reranker,
        dataset=validation_dataset,
        codebook=codebook,
        config=config,
        device=device,
        model_logit_weight=float(selected["model_logit_weight"]),
        replacement_threshold=float(selected["replacement_threshold"]),
    )
    validation = evaluate_residual_token_predictions(
        dataset=validation_dataset,
        residual_codebook=codebook,
        predictions={
            "matched_one_shot": matched_validation,
            "rollout_one_shot": validation_decoded["initial_tokens"],
            "listwise_identity_reranker": validation_decoded["tokens"],
        },
        config=config,
        device=device,
    )
    permutation = _permutation_check(
        model=rollout_model,
        reranker=reranker,
        dataset=validation_dataset,
        codebook=codebook,
        config=config,
        device=device,
        selected=selected,
    )
    latency = _measure_latency(
        model=rollout_model,
        reranker=reranker,
        dataset=validation_dataset,
        codebook=codebook,
        config=config,
        device=device,
        selected=selected,
    )
    gpu_activity = monitor.summary()
    decision = _decision(
        evaluation=validation,
        decoded=validation_decoded,
        selection=selection,
        permutation=permutation,
        latency=latency,
        config=config,
    )
    _write_csv(
        run_paths.artifacts / "calibration_candidates.csv",
        calibration_rows,
    )
    _write_json(
        run_paths.artifacts / "calibration_diagnostics.json",
        calibration_diagnostics,
    )
    _write_json(run_paths.artifacts / "selection.json", selection)
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(
        run_paths.artifacts / "transition_diagnostics.json",
        {
            "transitions": validation_decoded["transitions"],
            "replacements": validation_decoded["replacements"],
            "replacement_fraction": validation_decoded[
                "replacement_fraction"
            ],
            "target_in_top_k_fraction": validation_decoded[
                "target_in_top_k_fraction"
            ],
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
        cache_diagnostics=cache.diagnostics,
        training=training,
        selection=selection,
        evaluation=validation,
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
            "stage": "p2-5-listwise-identity-reranker",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "protocol_preregistration": preregistration,
            "cache": cache.diagnostics,
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
    )
    logger.log_event(
        "p2_5_completed",
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
