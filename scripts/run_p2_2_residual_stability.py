"""P2-2 residual-token rollout/stability training on identity carveouts."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_pointer_route_phase3 import _phase01_lock
from run_pointer_route_rollout_rescue import CudaActivityMonitor
from run_p2_residual_evidence import _core_rows, _scenario_rows
from set2proto.config import load_config
from set2proto.decoding import maskgit_decode
from set2proto.environment import collect_environment, config_sha256
from set2proto.experiment_logging import ExperimentLogger
from set2proto.model import build_model_from_config
from set2proto.reproducibility import seed_everything
from set2proto.residual_evaluation import (
    evaluate_residual_quantization_oracle,
    evaluate_residual_validation,
    measure_residual_inference_latency,
    residual_permutation_check,
)
from set2proto.residual_evidence import (
    compute_two_level_residual_evidence,
    quality_weighted_anchor,
)
from set2proto.residual_quantization import (
    ResidualCodebook,
    reconstruct_from_residual_tokens,
)
from set2proto.residual_route_artifacts import (
    prepare_route_residual_artifacts,
)
from set2proto.residual_stability_training import (
    train_residual_stability_model,
)
from set2proto.run_paths import create_run_paths
from set2proto.training import TokenTrainingDataset, load_codebook


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
    parser.add_argument("--absolute-quantization-root", type=Path, required=True)
    parser.add_argument("--phase01-root", type=Path, required=True)
    parser.add_argument("--p2-1-root", type=Path, required=True)
    parser.add_argument("--residual-artifact-root", type=Path, required=True)
    parser.add_argument("--reuse-residual-artifacts", action="store_true")
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


def _load_p2_1_lock(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    manifest_path = root / "logs" / "run_manifest.json"
    required = {
        "decision": artifacts / "decision.json",
        "report": artifacts / "REPORT.md",
        "evaluation": artifacts / "validation_evaluation.json",
        "protocol": artifacts / "protocol_preregistration.json",
        "manifest": manifest_path,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing P2-1 {name}: {path}")
    values = {
        name: (
            json.loads(path.read_text(encoding="utf-8"))
            if path.suffix == ".json"
            else path.read_text(encoding="utf-8")
        )
        for name, path in required.items()
    }
    if values["protocol"].get("test_dataset_constructed") is not False:
        raise RuntimeError("P2-1 loaded test data")
    values["artifact_hashes"] = {
        name: _sha256(path) for name, path in required.items()
    }
    return values


class _IndexedTokenDataset:
    """Read-only identity-carved view accepted by residual evaluation."""

    def __init__(
        self,
        base: TokenTrainingDataset,
        indices: Sequence[int],
    ) -> None:
        if base._precomputed is None:
            raise ValueError("indexed residual dataset requires precomputation")
        self.base = base
        self.indices = [int(value) for value in indices]
        selected = torch.tensor(self.indices, dtype=torch.int64)
        self._precomputed = {
            key: (
                value.index_select(0, selected)
                if isinstance(value, torch.Tensor)
                else [value[index] for index in self.indices]
            )
            for key, value in base._precomputed.items()
        }
        self.targets = {
            key: (
                value.index_select(0, selected)
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.shape[0]) == len(base)
                else (
                    [value[index] for index in self.indices]
                    if isinstance(value, list) and len(value) == len(base)
                    else value
                )
            )
            for key, value in base.targets.items()
        }
        self.targets["split"] = "val"
        # Residual evaluation also consumes per-sample metadata (notably the
        # wrong-frame positions).  Preserve the exact carved order instead of
        # exposing the unfiltered base rows.
        self.synthetic = _IndexedSetMetadata(
            rows=[base.synthetic.rows[index] for index in self.indices],
            split="val",
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "sample_id": self._precomputed["sample_ids"][index],
            "scenario": self._precomputed["scenarios"][index],
            "target_identity": self._precomputed["target_identity"][index],
            "condition_features": self._precomputed[
                "condition_features"
            ][index],
            "condition_quality": self._precomputed[
                "condition_quality"
            ][index],
            "condition_visibility": self._precomputed[
                "condition_visibility"
            ][index],
            "target_tokens": self.targets["tokens"][index].to(torch.int64),
        }


class _IndexedSetMetadata:
    """Minimal read-only set metadata carried by an indexed token view."""

    def __init__(self, *, rows: Sequence[Mapping[str, Any]], split: str) -> None:
        self.rows = [dict(row) for row in rows]
        self.split = str(split)

    def __len__(self) -> int:
        return len(self.rows)


def _evaluation_config(
    config: Mapping[str, Any],
    *,
    evidence_lambda: float,
) -> dict[str, Any]:
    selected = copy.deepcopy(dict(config))
    source = selected["p2_2"]["evidence"]
    target = selected["p2_1"]["evidence"]
    for key in (
        "top_k_frames",
        "identity_neighbor_fraction",
        "identity_temperature",
        "quality_weight",
        "local_temperature",
        "reliability_floor",
    ):
        target[key] = source[key]
    target["lambda"] = float(evidence_lambda)
    return selected


def _load_model(
    *,
    config: Mapping[str, Any],
    checkpoint: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model_from_config(config)
    payload = torch.load(
        Path(checkpoint),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model


def _select_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    best = max(float(row["maskgit_four_hard_auc"]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if float(row["maskgit_four_hard_auc"]) >= best - tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["four_minus_one_shot"]),
            float(row["maskgit_four_all_auc"]),
            -int(row["step"]),
        ),
    )
    return {
        "selection_split": "route_calibration",
        "objective": (
            "maximize confidence four-step hard AUC; within tolerance "
            "maximize four-minus-one-shot, then all AUC, then lower step"
        ),
        "auc_tie_tolerance": tolerance,
        "best_hard_auc": best,
        "eligible_candidates": len(eligible),
        "selected_candidate": selected,
        "validation_used": False,
        "test_used": False,
    }


def _select_lambda(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    best = max(float(row["guided_hard_auc"]) for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if float(row["guided_hard_auc"]) >= best - tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["guided_minus_confidence"]),
            -float(row["lambda"]),
        ),
    )
    return {
        "selection_split": "route_calibration",
        "objective": (
            "maximize two-level-remask hard AUC; within tolerance maximize "
            "gain over confidence, then lower lambda"
        ),
        "auc_tie_tolerance": tolerance,
        "best_hard_auc": best,
        "eligible_candidates": len(eligible),
        "selected_candidate": selected,
        "validation_used": False,
        "test_used": False,
    }


def _transition(first: torch.Tensor, final: torch.Tensor) -> dict[str, float]:
    wrong = ~first
    correct = first
    corrected = wrong & final
    injured = correct & ~final
    return {
        "ecr": (
            float(corrected.sum().item() / wrong.sum().item())
            if bool(wrong.any().item())
            else 0.0
        ),
        "eir": (
            float(injured.sum().item() / correct.sum().item())
            if bool(correct.any().item())
            else 0.0
        ),
        "corrected_fraction": float(corrected.float().mean().item()),
        "injured_fraction": float(injured.float().mean().item()),
        "net_correction_fraction": float(
            (corrected.sum() - injured.sum()).item() / first.numel()
        ),
    }


@torch.inference_mode()
def _trajectory_and_synthesis(
    *,
    dataset: _IndexedTokenDataset,
    maskgit_model: torch.nn.Module,
    residual_codebook: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    section = config["p2_1"]["evidence"]
    codebook = residual_codebook.float().to(device)
    batch_size = int(config["evaluation"]["batch_size"])
    guided_one_parts: list[torch.Tensor] = []
    guided_four_parts: list[torch.Tensor] = []
    confidence_four_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    synthesis_cosines: list[torch.Tensor] = []
    teacher_cosines: list[torch.Tensor] = []
    residual_norms: list[torch.Tensor] = []
    remask_events = [0, 0, 0, 0]
    scenario_events = {
        scenario: 0
        for scenario in sorted(set(dataset._precomputed["scenarios"]))
    }
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        condition = dataset._precomputed["condition_features"][
            start:end
        ].to(device)
        quality = dataset._precomputed["condition_quality"][start:end].to(
            device
        )
        targets = dataset.targets["tokens"][start:end].to(device)
        anchor = quality_weighted_anchor(condition, quality)
        evidence = compute_two_level_residual_evidence(
            condition,
            quality,
            anchor,
            codebook,
            identity_neighbor_fraction=float(
                section["identity_neighbor_fraction"]
            ),
            identity_temperature=float(section["identity_temperature"]),
            quality_weight=float(section["quality_weight"]),
            local_temperature=float(section["local_temperature"]),
            reliability_floor=float(section["reliability_floor"]),
        ).evidence
        guided_one = maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=1,
            mode="evidence-remask",
            top_k_frames=int(section["top_k_frames"]),
            evidence_lambda=float(section["lambda"]),
            evidence_override=evidence,
        )
        guided_four = maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=4,
            mode="evidence-remask",
            top_k_frames=int(section["top_k_frames"]),
            evidence_lambda=float(section["lambda"]),
            evidence_override=evidence,
        )
        confidence_four = maskgit_decode(
            maskgit_model,
            condition,
            quality,
            codebook=codebook,
            steps=4,
            mode="confidence",
            top_k_frames=int(section["top_k_frames"]),
            evidence_lambda=float(section["lambda"]),
        )
        guided_one_parts.append(guided_one.tokens.cpu())
        guided_four_parts.append(guided_four.tokens.cpu())
        confidence_four_parts.append(confidence_four.tokens.cpu())
        target_parts.append(targets.cpu())
        for step, remasked in enumerate(guided_four.remasked):
            remask_events[step] += int(remasked.sum().item())
            per_sample = remasked.sum(dim=1)
            for offset, scenario in enumerate(
                dataset._precomputed["scenarios"][start:end]
            ):
                scenario_events[scenario] += int(per_sample[offset].item())
        prototype = reconstruct_from_residual_tokens(
            anchor,
            guided_four.tokens,
            codebook,
        )
        normalized_condition = F.normalize(condition.float(), dim=-1)
        max_observation_cosine = torch.einsum(
            "bpd,bfpd->bpf",
            prototype,
            normalized_condition,
        ).max(dim=-1).values
        synthesis_cosines.append(max_observation_cosine.cpu())
        teacher = dataset.targets["continuous_prototypes"][start:end].to(
            device
        )
        teacher_cosines.append(
            F.cosine_similarity(prototype, teacher.float(), dim=-1).cpu()
        )
        residual_norms.append(codebook[guided_four.tokens].norm(dim=-1).cpu())
    guided_one = torch.cat(guided_one_parts)
    guided_four = torch.cat(guided_four_parts)
    confidence_four = torch.cat(confidence_four_parts)
    targets = torch.cat(target_parts)
    scenarios = list(dataset._precomputed["scenarios"])
    first_correct = guided_one == targets
    final_correct = guided_four == targets
    scenario_transitions = {}
    for scenario in sorted(set(scenarios)):
        mask = torch.tensor([value == scenario for value in scenarios])
        scenario_transitions[scenario] = _transition(
            first_correct[mask],
            final_correct[mask],
        )
    scenario_transitions["all"] = _transition(first_correct, final_correct)
    counts = torch.bincount(
        guided_four.flatten(),
        minlength=int(codebook.shape[0]),
    ).float()
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    perplexity = torch.exp(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    )
    observation_cosine = torch.cat(synthesis_cosines)
    teacher_cosine = torch.cat(teacher_cosines)
    residual_norm = torch.cat(residual_norms)
    return {
        "guided_one_to_four": {
            "transitions": scenario_transitions,
            "token_change_fraction": float(
                (guided_one != guided_four).float().mean().item()
            ),
        },
        "confidence_to_guided_four_token_change_fraction": float(
            (confidence_four != guided_four).float().mean().item()
        ),
        "remask": {
            "per_step_events": remask_events,
            "total_events": sum(remask_events),
            "scenario_events": scenario_events,
        },
        "synthesis": {
            "mean_max_cosine_to_any_observed_frame": float(
                observation_cosine.mean().item()
            ),
            "fraction_not_near_copy_of_observed_frame": float(
                (observation_cosine < 0.999).float().mean().item()
            ),
            "mean_teacher_map_cosine": float(teacher_cosine.mean().item()),
            "mean_residual_norm": float(residual_norm.mean().item()),
            "predicted_codebook_utilization": float(
                (counts > 0).float().mean().item()
            ),
            "predicted_token_perplexity": float(perplexity.item()),
            "normalized_predicted_token_perplexity": float(
                perplexity.item() / codebook.shape[0]
            ),
        },
    }


def _decision(
    *,
    oracle: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    checkpoint_selection: Mapping[str, Any],
    lambda_selection: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    permutation: Mapping[str, Any],
    latency: Mapping[str, Any],
    training: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = evaluation["verification"]
    quality = methods["quality_anchor"]["hard_average"]["roc_auc"]
    oracle_hard = methods["teacher_residual_quantized"]["hard_average"][
        "roc_auc"
    ]
    one = methods["one_shot_transformer"]["hard_average"]["roc_auc"]
    confidence = methods["maskgit_confidence_4step"]["hard_average"]["roc_auc"]
    guided = methods["maskgit_two_level_remask"]["hard_average"]["roc_auc"]
    scenario_deltas = {
        scenario: (
            methods["maskgit_two_level_remask"][scenario]["roc_auc"]
            - methods["maskgit_confidence_4step"][scenario]["roc_auc"]
        )
        for scenario in HARD_SCENARIOS
    }
    scenario_wins = sum(value > 1e-12 for value in scenario_deltas.values())
    transition = trajectory["guided_one_to_four"]["transitions"]["all"]
    gates = config["p2_2"]["gates"]
    peak = max(
        max(
            float(training[mode]["peak_reserved_mib"])
            for mode in training
        ),
        max(float(values["peak_reserved_mib"]) for values in latency.values()),
    )
    checks = {
        "identity_disjoint_representation_fit": True,
        "checkpoint_selected_on_calibration_only": (
            checkpoint_selection["validation_used"] is False
            and checkpoint_selection["test_used"] is False
        ),
        "lambda_selected_on_calibration_only": (
            lambda_selection["validation_used"] is False
            and lambda_selection["test_used"] is False
        ),
        "residual_oracle_beats_quality": (
            oracle_hard - quality
            >= float(
                gates["min_residual_oracle_hard_auc_gain_over_quality"]
            )
        ),
        "confidence_four_beats_one_shot": (
            confidence - one
            >= float(gates["min_four_step_gain_over_one_shot"])
        ),
        "guided_beats_confidence": (
            guided - confidence
            >= float(
                gates["min_guided_gain_over_confidence_four_step"]
            )
        ),
        "guided_beats_one_shot": (
            guided - one
            >= float(gates["min_guided_gain_over_one_shot"])
        ),
        "hard_scenario_wins": (
            scenario_wins >= int(gates["min_hard_scenario_wins"])
        ),
        "clean_constraint": (
            methods["maskgit_confidence_4step"]["clean"]["roc_auc"]
            - methods["maskgit_two_level_remask"]["clean"]["roc_auc"]
            <= float(gates["max_clean_auc_drop_vs_confidence"])
        ),
        "positive_step_net_correction": (
            transition["net_correction_fraction"] > 0.0
            if bool(gates["require_positive_step_net_correction"])
            else True
        ),
        "permutation_invariant": (
            max(
                float(permutation["one_shot_token_disagreement"]),
                float(permutation["maskgit_confidence_token_disagreement"]),
                float(permutation["two_level_evidence_max_abs_difference"]),
                float(permutation["anchor_max_abs_difference"]),
            )
            <= float(gates["max_permutation_difference"])
        ),
        "parameter_counts_equal": (
            int(training["one_shot_control"]["parameter_count"])
            == int(training["rollout_maskgit"]["parameter_count"])
        ),
        "training_steps_equal": (
            int(training["one_shot_control"]["steps"])
            == int(training["rollout_maskgit"]["steps"])
        ),
        "memory_within_7_2_gib": (
            peak <= float(config["runtime"]["max_peak_memory_gib"]) * 1024
        ),
        "test_not_loaded": True,
    }
    return {
        "classification": (
            "GO_RESIDUAL_MULTISTEP" if all(checks.values()) else "NO_GO_P2_2"
        ),
        "checks": checks,
        "quality_hard_auc": quality,
        "residual_oracle_hard_auc": oracle_hard,
        "one_shot_hard_auc": one,
        "confidence_four_hard_auc": confidence,
        "guided_four_hard_auc": guided,
        "confidence_minus_one_shot": confidence - one,
        "guided_minus_confidence": guided - confidence,
        "guided_minus_one_shot": guided - one,
        "scenario_deltas_guided_vs_confidence": scenario_deltas,
        "hard_scenario_wins": scenario_wins,
        "step_ecr": transition["ecr"],
        "step_eir": transition["eir"],
        "step_net_correction_fraction": transition[
            "net_correction_fraction"
        ],
        "peak_reserved_mib": peak,
        "recommendation": (
            "lock_design_and_create_new_untouched_holdout"
            if all(checks.values())
            else "inspect_failed_gates_before_any_d3pm_expansion"
        ),
    }


def _report(
    *,
    artifact: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    checkpoint_selection: Mapping[str, Any],
    lambda_rows: Sequence[Mapping[str, Any]],
    lambda_selection: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    decision: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> str:
    methods = evaluation["verification"]

    def hard(name: str) -> float:
        return float(methods[name]["hard_average"]["roc_auc"])

    def all_auc(name: str) -> float:
        return float(methods[name]["all"]["roc_auc"])

    checkpoint_table = "\n".join(
        "| {step} | {one_shot_hard_auc:.6f} | "
        "{maskgit_four_hard_auc:.6f} | {four_minus_one_shot:+.6f} |".format(
            **row
        )
        for row in checkpoint_rows
    )
    lambda_table = "\n".join(
        "| {lambda:.2f} | {confidence_hard_auc:.6f} | "
        "{guided_hard_auc:.6f} | {guided_minus_confidence:+.6f} |".format(
            **row
        )
        for row in lambda_rows
    )
    scenario_table = "\n".join(
        f"| {scenario} | "
        f"{methods['maskgit_confidence_4step'][scenario]['roc_auc']:.6f} | "
        f"{methods['maskgit_two_level_remask'][scenario]['roc_auc']:.6f} | "
        f"{decision['scenario_deltas_guided_vs_confidence'][scenario]:+.6f} |"
        for scenario in HARD_SCENARIOS
    )
    failed = [
        key for key, value in decision["checks"].items() if not value
    ]
    synth = trajectory["synthesis"]
    return f"""# P2-2 Residual Rollout-Stability Upgrade

## Locked protocol

- The existing Stage16 PCA is reused byte-for-byte.
- Residual K=1024 is refit only on 800 route-train identities.
- Model training uses the same 800 identities; route-calibration and
  route-validation each contain 100 disjoint identities.
- One-shot and rollout-MaskGIT have identical initialization,
  {int(evaluation['parameter_count']):,} parameters, update count, batch size,
  and optimizer schedule.
- MaskGIT training mixes 50% all-mask, 25% teacher-partial, and 25%
  stop-gradient self-rollout states. Wrong committed tokens receive correction
  loss; correct committed tokens receive a smaller stability loss.
- Checkpoint and evidence lambda are selected on route-calibration only.
  Route-validation is evaluated after locking both. Test is never loaded.

## Train-only residual representation

- Route-train fit tokens:
  {artifact['manifest']['codebook']['fit_tokens']}.
- Route-validation codebook utilization:
  {artifact['manifest']['metrics']['route_validation']['codebook_utilization']:.4f}.
- Route-validation residual reconstruction map cosine:
  {artifact['manifest']['metrics']['route_validation']['mean_map_cosine']:.6f}.

## Calibration checkpoint selection

| Step | One-shot hard AUC | Confidence 4-step | Delta |
|---:|---:|---:|---:|
{checkpoint_table}

Selected shared step:
`{checkpoint_selection['selected_candidate']['step']}`.

## Calibration evidence selection

| Lambda | Confidence 4-step | Two-level remask | Delta |
|---:|---:|---:|---:|
{lambda_table}

Selected lambda: `{lambda_selection['selected_candidate']['lambda']}`.

## Route-validation results

| Method | All AUC | Hard AUC |
|---|---:|---:|
| Quality anchor | {all_auc('quality_anchor'):.6f} | {hard('quality_anchor'):.6f} |
| Residual quantization oracle | {all_auc('teacher_residual_quantized'):.6f} | {hard('teacher_residual_quantized'):.6f} |
| One-shot control | {all_auc('one_shot_transformer'):.6f} | {hard('one_shot_transformer'):.6f} |
| Confidence 1-step | {all_auc('maskgit_confidence_1step'):.6f} | {hard('maskgit_confidence_1step'):.6f} |
| Confidence 2-step | {all_auc('maskgit_confidence_2step'):.6f} | {hard('maskgit_confidence_2step'):.6f} |
| Confidence 4-step | {all_auc('maskgit_confidence_4step'):.6f} | {hard('maskgit_confidence_4step'):.6f} |
| Confidence 8-step | {all_auc('maskgit_confidence_8step'):.6f} | {hard('maskgit_confidence_8step'):.6f} |
| Two-level logits | {all_auc('maskgit_two_level_logits'):.6f} | {hard('maskgit_two_level_logits'):.6f} |
| Two-level remask | {all_auc('maskgit_two_level_remask'):.6f} | {hard('maskgit_two_level_remask'):.6f} |

## Guided versus confidence, by scenario

| Scenario | Confidence | Guided remask | Delta |
|---|---:|---:|---:|
{scenario_table}

## Iteration and synthesis diagnostics

- Guided one-to-four token change:
  {trajectory['guided_one_to_four']['token_change_fraction']:.3%}.
- Step ECR/EIR/net:
  {decision['step_ecr']:.6f} /
  {decision['step_eir']:.6f} /
  {decision['step_net_correction_fraction']:+.6f}.
- Remask events:
  {trajectory['remask']['total_events']}.
- Mean maximum cosine to any observed frame:
  {synth['mean_max_cosine_to_any_observed_frame']:.6f}.
- Fraction not near-copying any observed frame:
  {synth['fraction_not_near_copy_of_observed_frame']:.3%}.
- Predicted codebook utilization / normalized perplexity:
  {synth['predicted_codebook_utilization']:.4f} /
  {synth['normalized_predicted_token_perplexity']:.4f}.

## Runtime

{chr(10).join(f"- `{name}`: {values['mean_batch_ms']:.3f} ms/batch, {values['mean_per_set_ms']:.3f} ms/set, peak reserved {values['peak_reserved_mib']:.1f} MiB" for name, values in latency.items())}

## Decision: {decision['classification']}

- Failed checks: {', '.join(failed) if failed else 'none'}.
- Confidence four-step minus one-shot:
  {decision['confidence_minus_one_shot']:+.6f}.
- Guided remask minus confidence four-step:
  {decision['guided_minus_confidence']:+.6f}.
- Guided remask minus one-shot:
  {decision['guided_minus_one_shot']:+.6f}.
- Recommendation: `{decision['recommendation']}`.

This is a mechanism-development result on a newly carved validation split,
not a statistical-significance claim. The official Stage16 val/test identities
remain unused by P2-2 selection and evaluation.
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.profile).to_dict()
    seed = int(config["project"]["seed"])
    seed_everything(seed, deterministic=bool(config["runtime"]["deterministic"]))
    dataset_root = args.dataset_root.expanduser().resolve()
    absolute_root = args.absolute_quantization_root.expanduser().resolve()
    residual_root = args.residual_artifact_root.expanduser().resolve()
    phase01 = _phase01_lock(
        root=args.phase01_root.expanduser().resolve(),
        dataset_root=dataset_root,
        quantization_root=absolute_root,
    )
    p2_1 = _load_p2_1_lock(args.p2_1_root.expanduser().resolve())
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
    section = config["p2_2"]
    preregistration = {
        "schema_version": 1,
        "stage": "p2-2-residual-rollout-stability",
        "mode": args.stage,
        "protocol": section,
        "route_split_hashes": phase01["artifact_hashes"],
        "p2_1_artifact_hashes": p2_1["artifact_hashes"],
        "checkpoint_selection_split": "route_calibration",
        "lambda_selection_split": "route_calibration",
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
        "p2_2_started",
        mode=args.stage,
        official_val_loaded=False,
        test_loaded=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P2-2 requires CUDA")
    artifact = prepare_route_residual_artifacts(
        config=config,
        dataset_root=dataset_root,
        absolute_quantization_root=absolute_root,
        route_split=phase01["split"],
        artifact_root=residual_root,
        resume=bool(args.reuse_residual_artifacts),
    )
    _write_json(
        run_paths.artifacts / "residual_artifact_validation.json",
        artifact["validation"],
    )
    if not artifact["validation"]["passed"]:
        raise RuntimeError("P2-2 residual artifact validation failed")
    dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=residual_root,
        split="train",
        precompute=True,
    )
    route_indices = phase01["split"]["sample_indices"]
    calibration_dataset = _IndexedTokenDataset(
        dataset,
        route_indices["route_calibration"],
    )
    validation_dataset = _IndexedTokenDataset(
        dataset,
        route_indices["route_validation"],
    )
    codebook = ResidualCodebook.from_payload(
        torch.load(
            residual_root / "codebook.pt",
            map_location="cpu",
            weights_only=True,
        )
    ).vectors
    absolute_codebook = load_codebook(absolute_root)
    absolute_targets_payload = torch.load(
        absolute_root / "train_targets.pt",
        map_location="cpu",
        weights_only=True,
    )
    absolute_validation_tokens = absolute_targets_payload["tokens"][
        route_indices["route_validation"]
    ]

    seed_everything(seed + 73001, deterministic=True)
    template = build_model_from_config(config)
    initial_state = copy.deepcopy(template.state_dict())
    parameter_count = template.parameter_count()
    del template

    def train(mode: str, *, smoke: bool) -> tuple[Any, torch.nn.Module]:
        seed_everything(seed + 73002, deterministic=True)
        model = build_model_from_config(config)
        model.load_state_dict(initial_state)

        def callback(row: dict[str, Any]) -> None:
            step = int(row["step"])
            if step == 1 or step % 50 == 0:
                logger.log_event("p2_2_training_step", **row)
                logger.log_metric(
                    name="loss",
                    value=float(row["loss"]),
                    step=step,
                    split="train",
                    scenario=mode,
                )

        result = train_residual_stability_model(
            model=model,
            dataset=dataset,
            train_indices=route_indices["route_train"],
            config=config,
            mode=mode,
            device=device,
            checkpoint_root=run_paths.checkpoints / mode,
            seed=seed + 73003,
            smoke=smoke,
            event_callback=callback,
        )
        return result, model

    if args.stage == "smoke":
        one_result, one_model = train("one_shot_control", smoke=True)
        one_model.cpu()
        torch.cuda.empty_cache()
        mask_result, mask_model = train("rollout_maskgit", smoke=True)
        first = [calibration_dataset[index] for index in range(4)]
        condition = torch.stack(
            [value["condition_features"] for value in first]
        ).to(device)
        quality = torch.stack(
            [value["condition_quality"] for value in first]
        ).to(device)
        decoded = maskgit_decode(
            mask_model,
            condition,
            quality,
            codebook=codebook.to(device),
            steps=4,
            mode="confidence",
            top_k_frames=3,
            evidence_lambda=0.0,
        )
        if bool((decoded.tokens == mask_model.mask_token_id).any().item()):
            raise RuntimeError("P2-2 smoke decoding left masks")
        result = {
            "status": "passed",
            "device": str(device),
            "parameter_count": parameter_count,
            "one_shot": {
                "steps": one_result.steps,
                "peak_reserved_mib": one_result.peak_reserved_mib,
                "device_audit": one_result.device_audit,
            },
            "rollout_maskgit": {
                "steps": mask_result.steps,
                "peak_reserved_mib": mask_result.peak_reserved_mib,
                "device_audit": mask_result.device_audit,
                "final_training_row": mask_result.history[-1],
            },
            "decode_shape": list(decoded.tokens.shape),
            "test_loaded": False,
        }
        _write_json(run_paths.artifacts / "smoke_result.json", result)
        logger.write_manifest(
            {
                "schema_version": 1,
                "stage": "p2-2-smoke",
                "status": "passed",
                "environment": environment,
                "resolved_config": config,
                "protocol_preregistration": preregistration,
                "smoke": result,
                "test_loaded": False,
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    with CudaActivityMonitor() as monitor:
        one_result, one_model = train("one_shot_control", smoke=False)
        one_model.cpu()
        del one_model
        torch.cuda.empty_cache()
        mask_result, mask_model = train("rollout_maskgit", smoke=False)
        mask_model.cpu()
        del mask_model
        torch.cuda.empty_cache()

        training = {
            "one_shot_control": {
                "steps": one_result.steps,
                "parameter_count": one_result.parameter_count,
                "elapsed_seconds": one_result.elapsed_seconds,
                "peak_allocated_mib": one_result.peak_allocated_mib,
                "peak_reserved_mib": one_result.peak_reserved_mib,
                "device_audit": one_result.device_audit,
                "checkpoints": one_result.checkpoints,
                "final_row": one_result.history[-1],
            },
            "rollout_maskgit": {
                "steps": mask_result.steps,
                "parameter_count": mask_result.parameter_count,
                "elapsed_seconds": mask_result.elapsed_seconds,
                "peak_allocated_mib": mask_result.peak_allocated_mib,
                "peak_reserved_mib": mask_result.peak_reserved_mib,
                "device_audit": mask_result.device_audit,
                "checkpoints": mask_result.checkpoints,
                "final_row": mask_result.history[-1],
            },
        }
        checkpoint_rows: list[dict[str, Any]] = []
        checkpoint_evaluations: dict[str, Any] = {}
        base_lambda = float(section["evidence"]["lambda_candidates"][0])
        calibration_config = _evaluation_config(
            config,
            evidence_lambda=base_lambda,
        )
        for step_value in section["calibration_checkpoint_steps"]:
            step = int(step_value)
            one = _load_model(
                config=config,
                checkpoint=one_result.checkpoints[step],
                device=device,
            )
            mask = _load_model(
                config=config,
                checkpoint=mask_result.checkpoints[step],
                device=device,
            )
            evaluation = evaluate_residual_validation(
                one_shot_model=one,
                maskgit_model=mask,
                dataset=calibration_dataset,
                residual_codebook=codebook,
                absolute_codebook=absolute_codebook,
                absolute_target_tokens=absolute_targets_payload["tokens"][
                    route_indices["route_calibration"]
                ],
                config=calibration_config,
                device=device,
                condition_frames=int(config["data"]["condition_frames"]),
            )
            methods = evaluation["verification"]
            row = {
                "step": step,
                "one_shot_hard_auc": methods["one_shot_transformer"][
                    "hard_average"
                ]["roc_auc"],
                "maskgit_four_hard_auc": methods[
                    "maskgit_confidence_4step"
                ]["hard_average"]["roc_auc"],
                "maskgit_four_all_auc": methods[
                    "maskgit_confidence_4step"
                ]["all"]["roc_auc"],
            }
            row["four_minus_one_shot"] = (
                row["maskgit_four_hard_auc"] - row["one_shot_hard_auc"]
            )
            checkpoint_rows.append(row)
            checkpoint_evaluations[str(step)] = evaluation
            del one, mask
            torch.cuda.empty_cache()
        checkpoint_selection = _select_checkpoint(
            checkpoint_rows,
            tolerance=float(section["evidence"]["auc_tie_tolerance"]),
        )
        selected_step = int(
            checkpoint_selection["selected_candidate"]["step"]
        )
        selected_one = _load_model(
            config=config,
            checkpoint=one_result.checkpoints[selected_step],
            device=device,
        )
        selected_mask = _load_model(
            config=config,
            checkpoint=mask_result.checkpoints[selected_step],
            device=device,
        )
        lambda_rows: list[dict[str, Any]] = []
        lambda_evaluations: dict[str, Any] = {}
        for value in section["evidence"]["lambda_candidates"]:
            evidence_lambda = float(value)
            candidate_config = _evaluation_config(
                config,
                evidence_lambda=evidence_lambda,
            )
            evaluation = evaluate_residual_validation(
                one_shot_model=selected_one,
                maskgit_model=selected_mask,
                dataset=calibration_dataset,
                residual_codebook=codebook,
                absolute_codebook=absolute_codebook,
                absolute_target_tokens=absolute_targets_payload["tokens"][
                    route_indices["route_calibration"]
                ],
                config=candidate_config,
                device=device,
                condition_frames=int(config["data"]["condition_frames"]),
            )
            methods = evaluation["verification"]
            confidence_hard = methods["maskgit_confidence_4step"][
                "hard_average"
            ]["roc_auc"]
            guided_hard = methods["maskgit_two_level_remask"][
                "hard_average"
            ]["roc_auc"]
            lambda_rows.append(
                {
                    "lambda": evidence_lambda,
                    "confidence_hard_auc": confidence_hard,
                    "guided_hard_auc": guided_hard,
                    "guided_minus_confidence": (
                        guided_hard - confidence_hard
                    ),
                }
            )
            lambda_evaluations[f"{evidence_lambda:g}"] = evaluation
        lambda_selection = _select_lambda(
            lambda_rows,
            tolerance=float(section["evidence"]["auc_tie_tolerance"]),
        )
        selected_lambda = float(
            lambda_selection["selected_candidate"]["lambda"]
        )
        locked_config = _evaluation_config(
            config,
            evidence_lambda=selected_lambda,
        )
        oracle = evaluate_residual_quantization_oracle(
            dataset=validation_dataset,
            residual_codebook=codebook,
            absolute_codebook=absolute_codebook,
            absolute_target_tokens=absolute_validation_tokens,
            config=locked_config,
            device=device,
        )
        validation = evaluate_residual_validation(
            one_shot_model=selected_one,
            maskgit_model=selected_mask,
            dataset=validation_dataset,
            residual_codebook=codebook,
            absolute_codebook=absolute_codebook,
            absolute_target_tokens=absolute_validation_tokens,
            config=locked_config,
            device=device,
            condition_frames=int(config["data"]["condition_frames"]),
        )
        validation["parameter_count"] = parameter_count
        trajectory = _trajectory_and_synthesis(
            dataset=validation_dataset,
            maskgit_model=selected_mask,
            residual_codebook=codebook,
            config=locked_config,
            device=device,
        )
        permutation = residual_permutation_check(
            one_shot_model=selected_one,
            maskgit_model=selected_mask,
            dataset=validation_dataset,
            residual_codebook=codebook,
            config=locked_config,
            device=device,
        )
        latency = measure_residual_inference_latency(
            one_shot_model=selected_one,
            maskgit_model=selected_mask,
            dataset=validation_dataset,
            residual_codebook=codebook,
            config=locked_config,
            device=device,
            batch_size=16,
        )
    gpu_activity = monitor.summary()
    decision = _decision(
        oracle=oracle,
        evaluation=validation,
        checkpoint_selection=checkpoint_selection,
        lambda_selection=lambda_selection,
        trajectory=trajectory,
        permutation=permutation,
        latency=latency,
        training=training,
        config=locked_config,
    )
    _write_json(run_paths.artifacts / "training_summary.json", training)
    _write_csv(
        run_paths.artifacts / "one_shot_training_history.csv",
        one_result.history,
    )
    _write_csv(
        run_paths.artifacts / "rollout_maskgit_training_history.csv",
        mask_result.history,
    )
    _write_csv(
        run_paths.artifacts / "checkpoint_candidates.csv",
        checkpoint_rows,
    )
    _write_json(
        run_paths.artifacts / "checkpoint_evaluations.json",
        checkpoint_evaluations,
    )
    _write_json(
        run_paths.artifacts / "checkpoint_selection.json",
        checkpoint_selection,
    )
    _write_csv(
        run_paths.artifacts / "lambda_candidates.csv",
        lambda_rows,
    )
    _write_json(
        run_paths.artifacts / "lambda_evaluations.json",
        lambda_evaluations,
    )
    _write_json(
        run_paths.artifacts / "lambda_selection.json",
        lambda_selection,
    )
    _write_json(run_paths.artifacts / "oracle_evaluation.json", oracle)
    _write_json(
        run_paths.artifacts / "validation_evaluation.json",
        validation,
    )
    _write_json(
        run_paths.artifacts / "trajectory_synthesis.json",
        trajectory,
    )
    _write_json(
        run_paths.artifacts / "permutation_check.json",
        permutation,
    )
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
        artifact=artifact,
        checkpoint_rows=checkpoint_rows,
        checkpoint_selection=checkpoint_selection,
        lambda_rows=lambda_rows,
        lambda_selection=lambda_selection,
        evaluation=validation,
        trajectory=trajectory,
        decision=decision,
        latency=latency,
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "p2-2-residual-rollout-stability",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "artifact_manifest": artifact["manifest"],
        "training": training,
        "checkpoint_selection": checkpoint_selection,
        "lambda_selection": lambda_selection,
        "validation": validation,
        "trajectory": trajectory,
        "permutation": permutation,
        "latency": latency,
        "gpu_activity": gpu_activity,
        "decision": decision,
        "official_val_loaded": False,
        "test_loaded": False,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "p2_2_completed",
        classification=decision["classification"],
        selected_step=selected_step,
        selected_lambda=selected_lambda,
        official_val_loaded=False,
        test_loaded=False,
    )
    print(
        json.dumps(
            {
                "run_directory": str(run_paths.root),
                "artifact_reused": artifact["reused"],
                "checkpoint_selection": checkpoint_selection,
                "lambda_selection": lambda_selection,
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
