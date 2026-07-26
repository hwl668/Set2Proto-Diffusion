"""Run P2-1 residual quantization, fair training, and validation-only evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import load_config, validate_config
from set2proto.decoding import maskgit_decode
from set2proto.environment import collect_environment, config_sha256
from set2proto.experiment_logging import ExperimentLogger
from set2proto.model import build_model_from_config
from set2proto.reproducibility import seed_everything
from set2proto.residual_artifacts import (
    prepare_residual_artifacts,
    validate_residual_artifacts,
)
from set2proto.residual_evaluation import (
    evaluate_residual_quantization_oracle,
    evaluate_residual_validation,
    measure_residual_inference_latency,
    residual_permutation_check,
)
from set2proto.residual_quantization import ResidualCodebook
from set2proto.run_paths import create_run_paths
from set2proto.training import (
    TokenTrainingDataset,
    load_codebook,
    train_single_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    parser.add_argument("--profile", choices=("expanded",), default="expanded")
    parser.add_argument("--stage", choices=("prepare", "all"), default="all")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--absolute-quantization-root", type=Path, required=True)
    parser.add_argument("--residual-artifact-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reuse-residual-artifacts", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_config(config: dict[str, Any], *, steps: int, smoke: bool) -> dict[str, Any]:
    selected = copy.deepcopy(config)
    selected["training"].update(
        {
            "batch_size": 2 if smoke else int(config["training"]["batch_size"]),
            "gradient_accumulation": (
                1
                if smoke
                else int(config["training"]["gradient_accumulation"])
            ),
            "min_steps": 1 if smoke else steps,
            "max_steps": steps,
            "validate_every_steps": 1 if smoke else min(500, steps),
            "checkpoint_every_steps": (
                1
                if smoke
                else int(config["p2_1"]["checkpoint_every_steps"])
            ),
            "warmup_steps": 0 if smoke else min(200, max(0, steps - 1)),
            "early_stop_patience": 1 if smoke else 5,
        }
    )
    if smoke:
        selected["runtime"]["oom_batch_fallback"] = [2, 1]
    validate_config(selected)
    return selected


def _model_candidates(config: dict[str, Any]) -> list[tuple[int, int, int]]:
    batches = [int(value) for value in config["runtime"]["oom_batch_fallback"]]
    frames = [
        int(value)
        for value in config["runtime"]["oom_condition_frames_fallback"]
    ]
    hidden = [
        int(value) for value in config["runtime"]["oom_hidden_dim_fallback"]
    ]
    result = [(batch, frames[0], hidden[0]) for batch in batches]
    result.extend((batches[-1], value, hidden[0]) for value in frames[1:])
    result.extend(
        (batches[-1], frames[-1], value) for value in hidden[1:]
    )
    return list(dict.fromkeys(result))


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        "steps": result.steps,
        "initial_loss": float(result.losses[0]),
        "final_loss": float(result.losses[-1]),
        "final_masked_accuracy": float(result.accuracies[-1]),
        "losses": [float(value) for value in result.losses],
        "masked_accuracies": [float(value) for value in result.accuracies],
        "learning_rates": [float(value) for value in result.learning_rates],
        "checkpoint": str(result.checkpoint),
        "peak_allocated_mib": result.peak_allocated_mib,
        "peak_reserved_mib": result.peak_reserved_mib,
    }


def _run_pair(
    *,
    dataset: TokenTrainingDataset,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_root: Path,
    batch_size: int,
    condition_frames: int,
    hidden_dim: int,
    seed: int,
    logger: ExperimentLogger,
    event_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    seed_everything(seed + 10, deterministic=True)
    template = build_model_from_config(config, hidden_dim=hidden_dim)
    initial_state = copy.deepcopy(template.state_dict())
    parameter_count = template.parameter_count()
    del template
    models: dict[str, Any] = {}
    results: dict[str, Any] = {}

    def callback(event: str, fields: dict[str, Any]) -> None:
        logger.log_event(f"{event_prefix}_{event}", **fields)
        if event == "training_step":
            logger.log_metric(
                name="loss",
                value=float(fields["loss"]),
                step=int(fields["step"]),
                split="train",
                scenario=f"{event_prefix}:{fields['mode']}",
            )

    for mode in ("one-shot", "maskgit"):
        seed_everything(seed + 20, deterministic=True)
        model = build_model_from_config(config, hidden_dim=hidden_dim)
        model.load_state_dict(initial_state)
        result = train_single_model(
            model=model,
            dataset=dataset,
            config=config,
            mode=mode,
            device=device,
            checkpoint_dir=checkpoint_root,
            batch_size=batch_size,
            condition_frames=condition_frames,
            seed=seed + 30,
            event_callback=callback,
        )
        result.model.cpu()
        models[mode] = result.model
        results[mode] = result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if models["one-shot"].parameter_count() != models["maskgit"].parameter_count():
        raise RuntimeError("parameter-matched models have different sizes")
    return models, results, parameter_count


def _oracle_gate(
    *,
    artifact_manifest: dict[str, Any],
    absolute_manifest: dict[str, Any],
    oracle: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["p2_1"]["gates"]
    residual_map = float(
        artifact_manifest["metrics"]["val"]["mean_map_cosine"]
    )
    absolute_map = float(
        absolute_manifest["metrics"]["val"]["mean_map_cosine"]
    )
    utilization = float(
        artifact_manifest["metrics"]["val"]["codebook_utilization"]
    )
    quality_hard = float(
        oracle["verification"]["quality_anchor"]["hard_average"]["roc_auc"]
    )
    residual_hard = float(
        oracle["verification"]["teacher_residual_quantized"][
            "hard_average"
        ]["roc_auc"]
    )
    checks = {
        "map_cosine_gain": (
            residual_map - absolute_map
            >= float(gates["min_map_cosine_gain_over_absolute"])
        ),
        "codebook_utilization": (
            utilization >= float(gates["min_codebook_utilization"])
        ),
        "identity_hard_auc_preserved": (
            residual_hard
            >= quality_hard
            - float(gates["max_hard_auc_drop_vs_quality_anchor"])
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "residual_map_cosine": residual_map,
        "absolute_map_cosine": absolute_map,
        "map_cosine_gain": residual_map - absolute_map,
        "val_codebook_utilization": utilization,
        "quality_anchor_hard_auc": quality_hard,
        "residual_quantized_hard_auc": residual_hard,
        "hard_auc_delta": residual_hard - quality_hard,
    }


def _core_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in evaluation["verification"].items():
        all_values = values["all"]
        hard = values["hard_average"]
        fidelity = evaluation["map_fidelity"][method]
        token = evaluation["exact_token_accuracy"].get(method, {})
        rows.append(
            {
                "method": method,
                "pooled_all_roc_auc": all_values["roc_auc"],
                "hard_macro_roc_auc": hard["roc_auc"],
                "all_eer": all_values["eer"],
                "tar_at_far_1e-2": all_values["tar_at_far"]["0.01"]["tar"],
                "tar_at_far_1e-3": all_values["tar_at_far"]["0.001"]["tar"],
                "rank1_accuracy": all_values["rank1_accuracy"],
                "mean_genuine_impostor_margin": all_values[
                    "mean_genuine_impostor_margin"
                ],
                "teacher_map_cosine": fidelity["mean_map_cosine"],
                "teacher_local_token_cosine": fidelity[
                    "mean_local_token_cosine"
                ],
                "exact_residual_token_accuracy": token.get("all", ""),
            }
        )
    return rows


def _scenario_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in evaluation["verification"].items():
        for scenario, metrics in values.items():
            if scenario in {"all", "hard_average"}:
                continue
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "roc_auc": metrics["roc_auc"],
                    "eer": metrics["eer"],
                    "tar_at_far_1e-2": metrics["tar_at_far"]["0.01"]["tar"],
                    "tar_at_far_1e-3": metrics["tar_at_far"]["0.001"]["tar"],
                    "rank1_accuracy": metrics["rank1_accuracy"],
                    "mean_genuine_impostor_margin": metrics[
                        "mean_genuine_impostor_margin"
                    ],
                    "teacher_map_cosine": evaluation["map_fidelity"][method][
                        "by_scenario"
                    ][scenario]["mean_map_cosine"],
                }
            )
    return rows


def _training_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, result in results.items():
        for index, (loss, accuracy, learning_rate) in enumerate(
            zip(result.losses, result.accuracies, result.learning_rates),
            start=1,
        ):
            rows.append(
                {
                    "mode": mode,
                    "step": index,
                    "loss": loss,
                    "masked_token_accuracy": accuracy,
                    "learning_rate": learning_rate,
                }
            )
    return rows


def _report(
    *,
    quant_gate: dict[str, Any],
    evaluation: dict[str, Any],
    latency: dict[str, Any],
    decision: dict[str, Any],
    selected: dict[str, Any],
) -> str:
    methods = evaluation["verification"]

    def hard(name: str) -> float:
        return float(methods[name]["hard_average"]["roc_auc"])

    def all_auc(name: str) -> float:
        return float(methods[name]["all"]["roc_auc"])

    absolute_hard = hard("teacher_absolute_quantized")
    residual_hard = hard("teacher_residual_quantized")
    two_step_gain = (
        hard("maskgit_confidence_2step") - hard("one_shot_transformer")
    )
    remask_vs_confidence = (
        hard("maskgit_two_level_remask")
        - hard("maskgit_confidence_4step")
    )
    remask_vs_one_shot = (
        hard("maskgit_two_level_remask") - hard("one_shot_transformer")
    )
    scenario_lines = []
    for scenario in (
        "clean",
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    ):
        confidence = methods["maskgit_confidence_4step"][scenario]["roc_auc"]
        guided = methods["maskgit_two_level_logits"][scenario]["roc_auc"]
        scenario_lines.append(
            f"| {scenario} | {confidence:.6f} | {guided:.6f} | "
            f"{guided - confidence:+.6f} |"
        )
    return f"""# P2-1 Residual Token + Two-Level Evidence Validation

## Protocol

- Expanded identity-disjoint train/validation only: 5,000 train sets / 1,000
  identities and 500 validation sets / 100 identities.
- No `test` dataset was instantiated and no residual `test_targets.pt` was
  constructed. The previously used Stage16 test remains outside model/parameter
  selection; a future paper claim still needs a newly created holdout.
- PCA and clean disjoint-teacher targets are byte-locked to Stage16. PCA was not
  refit. Residual K-means was fit only on train.
- Anchor: quality pooling. Residual: raw `teacher - anchor`. Reconstruction:
  `normalize(anchor + residual_code[token])`.
- One-shot and MaskGIT have the same {selected['parameter_count']:,} parameters,
  identical initialization, data, batch protocol, and {selected['steps']} steps.

## Representation gate

| Metric | Absolute K=1024 | Residual K=1024 | Delta |
|---|---:|---:|---:|
| Validation teacher-map cosine | {quant_gate['absolute_map_cosine']:.6f} | {quant_gate['residual_map_cosine']:.6f} | {quant_gate['map_cosine_gain']:+.6f} |
| Hard macro ROC-AUC | {absolute_hard:.6f} | {residual_hard:.6f} | {residual_hard - absolute_hard:+.6f} |

Validation residual codebook utilization: {quant_gate['val_codebook_utilization']:.4f}.
Representation gate: **{'PASS' if quant_gate['passed'] else 'FAIL'}**.
Relative to the continuous quality anchor, residual quantization gains
{quant_gate['hard_auc_delta']:+.6f} hard AUC. It does **not** beat the
absolute teacher quantization oracle; higher teacher-map cosine and higher
identity discrimination are different objectives here.

## Parameter-matched validation

| Method | All AUC | Hard AUC |
|---|---:|---:|
| quality anchor | {all_auc('quality_anchor'):.6f} | {hard('quality_anchor'):.6f} |
| residual quantization oracle | {all_auc('teacher_residual_quantized'):.6f} | {hard('teacher_residual_quantized'):.6f} |
| one-shot residual Transformer | {all_auc('one_shot_transformer'):.6f} | {hard('one_shot_transformer'):.6f} |
| 4-step confidence MaskGIT | {all_auc('maskgit_confidence_4step'):.6f} | {hard('maskgit_confidence_4step'):.6f} |
| 4-step local-logits MaskGIT | {all_auc('maskgit_local_logits'):.6f} | {hard('maskgit_local_logits'):.6f} |
| 4-step two-level-logits MaskGIT | {all_auc('maskgit_two_level_logits'):.6f} | {hard('maskgit_two_level_logits'):.6f} |
| 4-step two-level-remask MaskGIT | {all_auc('maskgit_two_level_remask'):.6f} | {hard('maskgit_two_level_remask'):.6f} |

Two-level logits minus confidence, by scenario:

| Scenario | Confidence AUC | Two-level AUC | Delta |
|---|---:|---:|---:|
{chr(10).join(scenario_lines)}

Wrong-identity frame mean global weight:
{evaluation['frame_diagnostics']['wrong_identity_frame_mean_weight']:.6f}.

## Runtime

Cached-feature batch=16 mean latency:

{chr(10).join(f"- `{name}`: {value['mean_batch_ms']:.3f} ms/batch, {value['mean_per_set_ms']:.3f} ms/set, peak reserved {value['peak_reserved_mib']:.1f} MiB" for name, value in latency.items())}

## Decision

P2-1 overall decision: **{'GO' if decision['go'] else 'NO-GO'}**.

- Residual representation gate: {'pass' if decision['checks']['representation_gate'] else 'fail'}.
- 4-step confidence vs parameter-matched one-shot hard-AUC delta:
  {decision['maskgit_minus_one_shot_hard_auc']:+.6f}.
- Pre-registered two-level logits vs confidence hard-AUC delta:
  {decision['two_level_minus_confidence_hard_auc']:+.6f}.

Secondary, non-selection observations: 2-step confidence vs one-shot is
{two_step_gain:+.6f}; two-level remask vs 4-step confidence is
{remask_vs_confidence:+.6f}, and vs one-shot is {remask_vs_one_shot:+.6f}.
These near-zero deltas are useful routing signals, not a post-hoc Go decision.

This is a single validation experiment, not a statistical-significance claim.
The representation result supports continuing residual-token work, but the
pre-registered iterative superiority gate did not pass. Before a new holdout,
P2-2 should target iterative exposure/commit stability, then lock the design
and train longer. The current evidence calibration should not be promoted to
D3PM unchanged.
"""


def main() -> int:
    args = parse_args()
    resolved = load_config(args.config, args.profile)
    config = resolved.to_dict()
    if int(config["p2_1"]["codebook_size"]) != int(
        config["quantization"]["codebook_size"]
    ):
        raise ValueError(
            "P2-1 residual codebook size must match the Transformer vocabulary"
        )
    seed = int(config["project"]["seed"])
    dataset_root = args.dataset_root.expanduser().resolve()
    absolute_root = args.absolute_quantization_root.expanduser().resolve()
    residual_root = args.residual_artifact_root.expanduser().resolve()
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
        "stage": "p2-1-residual-evidence",
        "selection_split": "val",
        "test_dataset_constructed": False,
        "test_artifacts_constructed": False,
        "formal_steps": int(config["p2_1"]["formal_steps"]),
        "primary_evidence_mode": "maskgit_two_level_logits",
        "gates": config["p2_1"]["gates"],
        "codebook": {
            key: config["p2_1"][key]
            for key in (
                "codebook_method",
                "codebook_size",
                "max_fit_tokens",
                "batch_size",
                "iterations",
                "n_init",
            )
        },
        "evidence": config["p2_1"]["evidence"],
        "config_sha256": config_sha256(config),
    }
    _write_json(
        run_paths.artifacts / "protocol_preregistration.json",
        preregistration,
    )
    logger.log_event(
        "p2_1_started",
        dataset_root=dataset_root,
        absolute_quantization_root=absolute_root,
        residual_artifact_root=residual_root,
        test_dataset_constructed=False,
    )
    seed_everything(seed, deterministic=bool(config["runtime"]["deterministic"]))

    artifacts = prepare_residual_artifacts(
        config=config,
        dataset_root=dataset_root,
        absolute_quantization_root=absolute_root,
        artifact_root=residual_root,
        resume=args.reuse_residual_artifacts,
    )
    if not artifacts.validation["passed"]:
        raise RuntimeError("residual artifact validation failed")
    _write_json(
        run_paths.artifacts / "residual_quantization_metrics.json",
        artifacts.manifest["metrics"],
    )
    if args.stage == "prepare":
        manifest = {
            "schema_version": 1,
            "stage": "p2-1-residual-prepare",
            "status": "passed",
            "environment": environment,
            "resolved_config": config,
            "dataset_root": str(dataset_root),
            "absolute_quantization_root": str(absolute_root),
            "residual_artifact_root": str(residual_root),
            "residual_validation": artifacts.validation,
            "test_dataset_constructed": False,
        }
        logger.write_manifest(manifest)
        return 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("P2-1 requires CUDA under the expanded profile")
    residual_codebook = ResidualCodebook.from_payload(
        torch.load(
            residual_root / "codebook.pt",
            map_location="cpu",
            weights_only=True,
        )
    ).vectors
    absolute_codebook = load_codebook(absolute_root)
    absolute_targets = torch.load(
        absolute_root / "val_targets.pt",
        map_location="cpu",
        weights_only=True,
    )
    residual_targets = torch.load(
        residual_root / "val_targets.pt",
        map_location="cpu",
        weights_only=True,
    )
    if absolute_targets["sample_ids"] != residual_targets["sample_ids"]:
        raise RuntimeError("absolute and residual validation samples differ")

    val_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=residual_root,
        split="val",
        precompute=True,
    )
    oracle = evaluate_residual_quantization_oracle(
        dataset=val_dataset,
        residual_codebook=residual_codebook,
        absolute_codebook=absolute_codebook,
        absolute_target_tokens=absolute_targets["tokens"],
        config=config,
        device=device,
    )
    absolute_manifest = json.loads(
        (absolute_root / "manifest.json").read_text(encoding="utf-8")
    )
    quant_gate = _oracle_gate(
        artifact_manifest=artifacts.manifest,
        absolute_manifest=absolute_manifest,
        oracle=oracle,
        config=config,
    )
    _write_json(run_paths.artifacts / "representation_gate.json", quant_gate)
    logger.log_event("representation_gate_completed", **quant_gate)

    train_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=residual_root,
        split="train",
        precompute=True,
    )
    smoke_config = _training_config(
        config,
        steps=int(config["p2_1"]["smoke_steps"]),
        smoke=True,
    )
    formal_config = _training_config(
        config,
        steps=int(config["p2_1"]["formal_steps"]),
        smoke=False,
    )

    selected_candidate: tuple[int, int, int] | None = None
    attempts: list[dict[str, Any]] = []
    for index, (batch_size, condition_frames, hidden_dim) in enumerate(
        _model_candidates(formal_config)
    ):
        name = f"attempt_{index:02d}_b{batch_size}_m{condition_frames}_h{hidden_dim}"
        try:
            smoke_local = copy.deepcopy(smoke_config)
            smoke_local["training"]["batch_size"] = min(2, batch_size)
            smoke_models, smoke_results, parameter_count = _run_pair(
                dataset=train_dataset,
                config=smoke_local,
                device=device,
                checkpoint_root=run_paths.checkpoints / name / "smoke",
                batch_size=min(2, batch_size),
                condition_frames=condition_frames,
                hidden_dim=hidden_dim,
                seed=seed + 1000,
                logger=logger,
                event_prefix=f"{name}_smoke",
            )
            smoke_maskgit = smoke_models["maskgit"].to(device)
            sample = val_dataset[0]
            smoke_decode = maskgit_decode(
                smoke_maskgit,
                sample["condition_features"][None, :condition_frames].to(device),
                sample["condition_quality"][None, :condition_frames].to(device),
                codebook=residual_codebook.to(device),
                steps=4,
                mode="confidence",
                top_k_frames=min(3, condition_frames),
                evidence_lambda=1.0,
            )
            smoke_maskgit.cpu()
            del smoke_models
            if bool((smoke_decode.tokens >= int(config["p2_1"]["codebook_size"])).any()):
                raise RuntimeError("smoke decoder returned an invalid token")
            attempts.append(
                {
                    "attempt": name,
                    "status": "smoke_passed",
                    "batch_size": batch_size,
                    "condition_frames": condition_frames,
                    "hidden_dim": hidden_dim,
                    "parameter_count": parameter_count,
                    "one_shot": _result_summary(smoke_results["one-shot"]),
                    "maskgit": _result_summary(smoke_results["maskgit"]),
                }
            )
            selected_candidate = (batch_size, condition_frames, hidden_dim)
            break
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            attempts.append(
                {
                    "attempt": name,
                    "status": "oom" if is_oom else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logger.log_event(
                "smoke_attempt_failed",
                level="warning" if is_oom else "error",
                **attempts[-1],
            )
            if not is_oom:
                raise
            gc.collect()
            torch.cuda.empty_cache()
    if selected_candidate is None:
        raise RuntimeError("all P2-1 OOM fallback smoke attempts failed")

    batch_size, condition_frames, hidden_dim = selected_candidate
    formal_models, formal_results, parameter_count = _run_pair(
        dataset=train_dataset,
        config=formal_config,
        device=device,
        checkpoint_root=run_paths.checkpoints / "formal",
        batch_size=batch_size,
        condition_frames=condition_frames,
        hidden_dim=hidden_dim,
        seed=seed + 2000,
        logger=logger,
        event_prefix="formal",
    )
    one_shot_model = formal_models["one-shot"].to(device)
    maskgit_model = formal_models["maskgit"].to(device)
    evaluation = evaluate_residual_validation(
        one_shot_model=one_shot_model,
        maskgit_model=maskgit_model,
        dataset=val_dataset,
        residual_codebook=residual_codebook,
        absolute_codebook=absolute_codebook,
        absolute_target_tokens=absolute_targets["tokens"],
        config=config,
        device=device,
        condition_frames=condition_frames,
    )
    latency = measure_residual_inference_latency(
        one_shot_model=one_shot_model,
        maskgit_model=maskgit_model,
        dataset=val_dataset,
        residual_codebook=residual_codebook,
        config=config,
        device=device,
        batch_size=16,
    )
    permutation = residual_permutation_check(
        one_shot_model=one_shot_model,
        maskgit_model=maskgit_model,
        dataset=val_dataset,
        residual_codebook=residual_codebook,
        config=config,
        device=device,
    )
    one_shot_model.cpu()
    maskgit_model.cpu()
    torch.cuda.empty_cache()

    verification = evaluation["verification"]
    maskgit_gain = (
        verification["maskgit_confidence_4step"]["hard_average"]["roc_auc"]
        - verification["one_shot_transformer"]["hard_average"]["roc_auc"]
    )
    two_level_gain = (
        verification["maskgit_two_level_logits"]["hard_average"]["roc_auc"]
        - verification["maskgit_confidence_4step"]["hard_average"]["roc_auc"]
    )
    gates = config["p2_1"]["gates"]
    decision_checks = {
        "representation_gate": bool(quant_gate["passed"]),
        "parameter_counts_equal": (
            formal_models["one-shot"].parameter_count()
            == formal_models["maskgit"].parameter_count()
        ),
        "maskgit_beats_one_shot": (
            maskgit_gain
            >= float(gates["min_maskgit_hard_auc_gain_over_one_shot"])
        ),
        "two_level_beats_confidence": (
            two_level_gain
            >= float(gates["min_two_level_hard_auc_gain_over_confidence"])
        ),
        "permutation_invariant": (
            permutation["one_shot_token_disagreement"] == 0.0
            and permutation["maskgit_confidence_token_disagreement"] == 0.0
            and permutation["two_level_evidence_max_abs_difference"] < 1e-5
        ),
        "memory_within_7_2_gib": max(
            formal_results["one-shot"].peak_reserved_mib,
            formal_results["maskgit"].peak_reserved_mib,
            *(value["peak_reserved_mib"] for value in latency.values()),
        )
        <= 7.2 * 1024.0,
        "test_dataset_not_constructed": True,
    }
    decision = {
        "checks": decision_checks,
        "go": all(decision_checks.values()),
        "maskgit_minus_one_shot_hard_auc": float(maskgit_gain),
        "two_level_minus_confidence_hard_auc": float(two_level_gain),
        "recommendation": (
            "proceed_to_p2_2_new_holdout"
            if all(decision_checks.values())
            else "retain_residual_results_fix_failed_gate_before_p2_2"
        ),
    }
    selected = {
        "batch_size": batch_size,
        "condition_frames": condition_frames,
        "hidden_dim": hidden_dim,
        "parameter_count": parameter_count,
        "steps": int(formal_config["training"]["max_steps"]),
    }
    _write_json(run_paths.artifacts / "oracle_evaluation.json", oracle)
    _write_json(run_paths.artifacts / "validation_evaluation.json", evaluation)
    _write_json(run_paths.artifacts / "latency.json", latency)
    _write_json(run_paths.artifacts / "permutation_check.json", permutation)
    _write_json(run_paths.artifacts / "decision.json", decision)
    _write_csv(run_paths.artifacts / "core_comparison.csv", _core_rows(evaluation))
    _write_csv(
        run_paths.artifacts / "scenario_metrics.csv",
        _scenario_rows(evaluation),
    )
    _write_csv(
        run_paths.artifacts / "training_history.csv",
        _training_rows(formal_results),
    )
    (run_paths.artifacts / "REPORT.md").write_text(
        _report(
            quant_gate=quant_gate,
            evaluation=evaluation,
            latency=latency,
            decision=decision,
            selected=selected,
        ),
        encoding="utf-8",
    )
    residual_validation_after = validate_residual_artifacts(
        residual_root,
        dataset_root=dataset_root,
    )
    manifest = {
        "schema_version": 1,
        "stage": "p2-1-residual-evidence",
        "status": "passed",
        "environment": environment,
        "resolved_config": config,
        "protocol_preregistration": preregistration,
        "dataset_root": str(dataset_root),
        "absolute_quantization_root": str(absolute_root),
        "residual_artifact_root": str(residual_root),
        "test_dataset_constructed": False,
        "test_artifacts_constructed": False,
        "selection_split": "val",
        "attempts": attempts,
        "selected_configuration": selected,
        "representation_gate": quant_gate,
        "training": {
            mode: _result_summary(result)
            for mode, result in formal_results.items()
        },
        "evaluation": evaluation,
        "latency": latency,
        "permutation_check": permutation,
        "decision": decision,
        "residual_validation_after": residual_validation_after,
    }
    logger.write_manifest(manifest)
    logger.log_event("p2_1_completed", decision=decision)
    print(json.dumps(
        {
            "run_directory": str(run_paths.root),
            "residual_artifact_root": str(residual_root),
            "representation_gate": quant_gate,
            "decision": decision,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
