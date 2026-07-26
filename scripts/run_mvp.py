"""Unified command-line entry point for the Set2Proto-Diffusion MVP."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from set2proto.config import (  # noqa: E402
    ConfigError,
    ResolvedConfig,
    load_config,
    validate_config,
)
from set2proto.condition_pooling import (  # noqa: E402
    METHOD_ORDER as CONDITION_POOLING_METHODS,
    evaluate_condition_pooling_split,
    measure_condition_pooling_latency,
    validate_p0_robust_parameters,
    write_condition_pooling_artifacts,
)
from set2proto.visibility_aggregation import (  # noqa: E402
    FINAL_METHOD_ORDER as VISIBILITY_AGGREGATION_METHODS,
    build_train_clean_reference,
    evaluate_locked_split,
    evaluate_validation_candidates,
    measure_visibility_aggregation_latency,
    write_visibility_aggregation_artifacts,
)
from set2proto.adaface_backbone import (  # noqa: E402
    load_adaface_ir50,
    probe_named_spatial_hooks,
)
from set2proto.earlier_hook_oracle import (  # noqa: E402
    cache_early_hook_features,
    evaluate_early_hook_split,
    measure_early_hook_latency,
    select_validation_hook,
    write_p0_3_artifacts,
)
from set2proto.continuous_residual_oracle import (  # noqa: E402
    evaluate_continuous_residual_split,
    measure_continuous_residual_latency,
    select_continuous_residual_candidate,
    write_continuous_residual_artifacts,
)
from set2proto.continuous_router import (  # noqa: E402
    ROUTER_METHOD,
    build_continuous_router,
    evaluate_continuous_router_split,
    load_router_checkpoint,
    measure_continuous_router_latency,
    select_continuous_router,
    test_gate as continuous_router_test_gate,
    train_continuous_router,
    write_continuous_router_artifacts,
)
from set2proto.identity_gated_router import (  # noqa: E402
    GATED_ROUTER_METHOD,
    build_identity_gated_router,
    build_train_identity_gallery,
    evaluate_identity_gated_router_split,
    load_identity_gated_checkpoint,
    load_p1_1_router_initialization,
    select_identity_gated_candidate,
    test_gate as identity_gated_test_gate,
    train_identity_gated_candidate,
    write_identity_gated_artifacts,
)
from set2proto.scalar_evidence_router import (  # noqa: E402
    SCALAR_EVIDENCE_METHOD,
    build_scalar_evidence_router,
    evaluate_scalar_evidence_split,
    load_router_initialization as load_scalar_router_initialization,
    load_scalar_evidence_checkpoint,
    select_scalar_evidence_candidate,
    test_gate as scalar_evidence_test_gate,
    train_scalar_evidence_candidate,
    write_scalar_evidence_artifacts,
)
from set2proto.environment import (  # noqa: E402
    collect_environment,
    config_sha256,
    probe_native_sdpa,
)
from set2proto.diagnostics import (  # noqa: E402
    evaluate_post_training_diagnostics,
)
from set2proto.datasets import validate_set_dataset  # noqa: E402
from set2proto.experiment_logging import ExperimentLogger  # noqa: E402
from set2proto.feature_cache import (  # noqa: E402
    prepare_real_feature_cache,
    validate_real_feature_cache,
)
from set2proto.reproducibility import seed_everything  # noqa: E402
from set2proto.quantization import (  # noqa: E402
    prepare_quantization_artifacts,
    validate_quantization_artifacts,
)
from set2proto.reporting import build_final_report  # noqa: E402
from set2proto.model import build_model_from_config  # noqa: E402
from set2proto.smoke_evaluation import evaluate_training_smoke  # noqa: E402
from set2proto.research_evaluation import (  # noqa: E402
    evaluate_research_split,
)
from set2proto.real_data import prepare_celeba_metadata  # noqa: E402
from set2proto.real_features import (  # noqa: E402
    probe_adaface_backbone,
    resolve_real_split_root,
)
from set2proto.real_sets import prepare_real_set_dataset  # noqa: E402
from set2proto.training import (  # noqa: E402
    TokenTrainingDataset,
    find_latest_checkpoint,
    load_codebook,
    train_single_model,
)
from set2proto.run_paths import (  # noqa: E402
    create_run_paths,
    ensure_workspace_directories,
)
from set2proto.synthetic import (  # noqa: E402
    prepare_synthetic_dataset,
    probe_synthetic_dataloader,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set2Proto-Diffusion reproducible MVP runner",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mvp.yaml",
        help="Path to the single MVP YAML configuration.",
    )
    parser.add_argument(
        "--profile",
        default="smoke",
        help="Configuration profile defined in configs/mvp.yaml.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "env-check",
            "prepare-real",
            "backbone-smoke",
            "cache-real-features",
            "prepare-real-sets",
            "prepare-synthetic",
            "synthetic-smoke",
            "fit-codebook",
            "train-smoke",
            "train-real-smoke",
            "train-pilot",
            "evaluate-diagnostics",
            "evaluate-condition-pooling",
            "evaluate-visibility-aggregation",
            "evaluate-earlier-hook-oracle",
            "evaluate-continuous-residual-oracle",
            "train-continuous-router-smoke",
            "train-continuous-router",
            "train-identity-gated-router-smoke",
            "train-identity-gated-router",
            "train-scalar-evidence-router-smoke",
            "train-scalar-evidence-router",
            "build-report",
        ),
        help="MVP stage to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducibility seed override.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional collision-checked run identifier.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory override, useful for isolated checks.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Prepared synthetic dataset root required by fit-codebook.",
    )
    parser.add_argument(
        "--quantization-root",
        type=Path,
        default=None,
        help="Projection/codebook/target artifact root required by training.",
    )
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=None,
        help="Frozen real-feature cache required by prepare-real-sets.",
    )
    parser.add_argument(
        "--exclude-split-root",
        type=Path,
        default=None,
        help=(
            "Existing split whose train/val/test identities must be excluded "
            "when creating a new real-data split."
        ),
    )
    parser.add_argument(
        "--training-run",
        type=Path,
        default=None,
        help="Completed training run required by evaluate-diagnostics.",
    )
    parser.add_argument(
        "--initial-router-checkpoint",
        type=Path,
        default=None,
        help="Previous-stage router checkpoint used for initialization.",
    )
    parser.add_argument(
        "--diagnostics-run",
        type=Path,
        default=None,
        help="Completed diagnostics run required by build-report.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional AdaFace IR50 checkpoint path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an explicitly named existing run instead of creating one.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Lock model selection on validation and never construct the test "
            "dataset; intended for intermediate router stages."
        ),
    )
    return parser


def _apply_cli_overrides(
    resolved: ResolvedConfig,
    *,
    seed: int | None,
    output_root: Path | None,
) -> dict[str, Any]:
    config = resolved.to_dict()
    if seed is not None:
        config["project"]["seed"] = seed
    if output_root is not None:
        config["paths"]["output_dir"] = str(output_root.expanduser().resolve())
    validate_config(config)
    return config


def _build_checks(
    *,
    config: dict[str, Any],
    environment: dict[str, Any],
    sdpa_probe: dict[str, Any],
) -> dict[str, bool]:
    runtime = config["runtime"]
    training = config["training"]
    torch_info = environment["torch"]
    require_cuda = runtime["require_cuda"]
    cuda_available = bool(torch_info["cuda_available"])
    devices = torch_info["devices"]

    if training["precision"] == "bf16":
        bf16_supported = bool(devices) and all(
            bool(device["bf16_supported"]) for device in devices
        )
    else:
        bf16_supported = True

    if require_cuda:
        memory_budget_fits_device = bool(devices) and all(
            runtime["max_peak_memory_gib"]
            <= float(device["total_memory_mib"]) / 1024.0
            for device in devices
        )
    else:
        memory_budget_fits_device = True

    return {
        "configuration_valid": True,
        "cuda_requirement_met": (not require_cuda) or cuda_available,
        "bf16_requirement_met": bf16_supported,
        "native_sdpa_available": bool(torch_info["native_sdpa_available"]),
        "native_sdpa_probe_passed": bool(sdpa_probe["success"]),
        "memory_budget_fits_device": memory_budget_fits_device,
        "memory_budget_within_7_2_gib_cap": (
            float(runtime["max_peak_memory_gib"]) <= 7.2
        ),
    }


def run_env_check(args: argparse.Namespace) -> int:
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )

    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    command = [sys.executable, *sys.argv]
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=command,
    )
    sdpa_probe = probe_native_sdpa(
        require_cuda=bool(config["runtime"]["require_cuda"]),
        precision=str(config["training"]["precision"]),
    )
    checks = _build_checks(
        config=config,
        environment=environment,
        sdpa_probe=sdpa_probe,
    )
    passed = all(checks.values())

    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "sdpa_probe": sdpa_probe,
        "environment": environment,
        "resolved_config": config,
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "env_check_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        sdpa_probe=sdpa_probe,
    )
    logger.log_metric(
        name="sdpa_peak_allocated_memory",
        value=float(sdpa_probe["peak_allocated_mib"]),
        step=0,
        split="environment",
        unit="MiB",
    )
    logger.log_metric(
        name="sdpa_peak_reserved_memory",
        value=float(sdpa_probe["peak_reserved_mib"]),
        step=0,
        split="environment",
        unit="MiB",
    )

    summary = {
        "status": manifest["status"],
        "run_directory": str(run_paths.root),
        "checks": checks,
        "gpu": (
            environment["torch"]["devices"][0]
            if environment["torch"]["devices"]
            else None
        ),
        "sdpa_probe": sdpa_probe,
        "git": environment["git"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def run_prepare_synthetic(args: argparse.Namespace) -> int:
    if args.stage == "synthetic-smoke" and args.profile != "smoke":
        raise ValueError("synthetic-smoke requires --profile smoke")

    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    command = [sys.executable, *sys.argv]
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=command,
    )
    logger.log_event(
        "synthetic_preparation_started",
        profile=args.profile,
        run_id=run_paths.root.name,
    )

    preparation = prepare_synthetic_dataset(
        config=config,
        profile=args.profile,
        run_id=run_paths.root.name,
        resume=args.resume,
    )
    gpu_probe = None
    if args.stage == "synthetic-smoke":
        gpu_probe = probe_synthetic_dataloader(
            root=preparation.root,
            batch_size=int(config["training"]["batch_size"]),
            seed=seed,
            precision=str(config["training"]["precision"]),
        )

    checks = {
        "synthetic_validation_passed": bool(
            preparation.validation["passed"]
        ),
        "identity_splits_disjoint": all(
            bool(value)
            for key, value in preparation.validation["checks"].items()
            if key.startswith("identity_disjoint.")
        ),
        "gpu_dataloader_probe_passed": (
            True if gpu_probe is None else bool(gpu_probe["success"])
        ),
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "synthetic_dataset": {
            "root": str(preparation.root),
            "manifest": str(preparation.manifest_path),
            "reused": preparation.reused,
            "summary": preparation.manifest["splits"],
        },
        "synthetic_validation": preparation.validation,
        "gpu_dataloader_probe": gpu_probe,
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "synthetic_preparation_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        dataset_root=preparation.root,
        reused=preparation.reused,
        gpu_probe=gpu_probe,
    )

    total_bytes = 0
    for file_info in preparation.manifest["files"].values():
        total_bytes += int(file_info["size_bytes"])
    logger.log_metric(
        name="synthetic_dataset_size",
        value=total_bytes / 1024**2,
        step=0,
        split="all",
        unit="MiB",
    )
    for split, summary in preparation.manifest["splits"].items():
        logger.log_metric(
            name="synthetic_identities",
            value=float(summary["identities"]),
            step=0,
            split=split,
            unit="count",
        )
        logger.log_metric(
            name="synthetic_sets",
            value=float(summary["sets"]),
            step=0,
            split=split,
            unit="count",
        )
    if gpu_probe is not None:
        logger.log_metric(
            name="synthetic_batch_peak_reserved_memory",
            value=float(gpu_probe["peak_reserved_mib"]),
            step=0,
            split="environment",
            unit="MiB",
        )
        logger.log_metric(
            name="condition_teacher_cosine",
            value=float(gpu_probe["mean_condition_teacher_cosine"]),
            step=0,
            split="train",
            scenario="sampled_batch",
        )

    summary = {
        "status": manifest["status"],
        "run_directory": str(run_paths.root),
        "dataset_root": str(preparation.root),
        "dataset_size_mib": round(total_bytes / 1024**2, 3),
        "checks": checks,
        "splits": preparation.manifest["splits"],
        "gpu_dataloader_probe": gpu_probe,
        "git": environment["git"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def run_prepare_real(args: argparse.Namespace) -> int:
    if args.dataset_root is None:
        raise ValueError("prepare-real requires --dataset-root")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    logger.log_event(
        "real_data_preparation_started",
        dataset_root=args.dataset_root.resolve(),
        profile=args.profile,
        exclude_split_root=(
            args.exclude_split_root.resolve()
            if args.exclude_split_root is not None
            else None
        ),
    )
    preparation = prepare_celeba_metadata(
        root=args.dataset_root,
        config=config,
        profile=args.profile,
        resume=args.resume,
        exclude_split_root=args.exclude_split_root,
    )
    checks = preparation.manifest["checks"]
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "blocked",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "real_dataset": {
            "root": str(preparation.root),
            "split_root": str(preparation.split_root),
            "manifest": str(preparation.manifest_path),
            "reused": preparation.reused,
            "source": preparation.manifest["source"],
            "selection": preparation.manifest["selection"],
            "assets": preparation.manifest["assets"],
        },
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "real_data_preparation_completed",
        level="info" if passed else "warning",
        passed=passed,
        checks=checks,
        split_root=preparation.split_root,
    )
    for split, split_summary in preparation.manifest["selection"][
        "splits"
    ].items():
        logger.log_metric(
            name="real_identities",
            value=float(split_summary["identities"]),
            step=0,
            split=split,
            unit="count",
        )
        logger.log_metric(
            name="real_images",
            value=float(split_summary["images"]),
            step=0,
            split=split,
            unit="count",
        )
    summary = {
        "status": manifest["status"],
        "run_directory": str(run_paths.root),
        "dataset_root": str(preparation.root),
        "split_root": str(preparation.split_root),
        "checks": checks,
        "source": preparation.manifest["source"],
        "selection": preparation.manifest["selection"],
        "assets": preparation.manifest["assets"],
        "git": environment["git"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def run_backbone_smoke(args: argparse.Namespace) -> int:
    if args.profile != "smoke":
        raise ValueError("backbone-smoke requires --profile smoke")
    if args.dataset_root is None:
        raise ValueError("backbone-smoke requires --dataset-root")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    checkpoint = (
        args.weights.expanduser().resolve()
        if args.weights is not None
        else (
            Path(config["paths"]["weights_dir"])
            / "adaface"
            / config["backbone"]["checkpoint_filename"]
        )
    )
    dataset_root = args.dataset_root.expanduser().resolve()
    split_root = resolve_real_split_root(
        dataset_root,
        profile=args.profile,
        seed=seed,
    )
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("backbone-smoke requires CUDA")
    logger.log_event(
        "backbone_smoke_started",
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        split_root=split_root,
        device=str(device),
    )
    probe = probe_adaface_backbone(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        split_root=split_root,
        batch_size=int(config["backbone"]["cache_batch_size"]),
        precision=str(config["training"]["precision"]),
        device=device,
    )
    spatial_shape = probe["spatial_feature"]["shape"]
    expected_grid = config["data"]["feature_grid"]
    checks = {
        "checkpoint_strictly_compatible": (
            not probe["checkpoint"]["missing_keys"]
            and not probe["checkpoint"]["unexpected_keys"]
        ),
        "backbone_frozen": bool(probe["frozen"]),
        "backbone_eval_mode": not bool(probe["training"]),
        "actual_spatial_shape_matches_mvp": (
            spatial_shape[1]
            == int(expected_grid["backbone_dim"])
            and spatial_shape[2] == int(expected_grid["height"])
            and spatial_shape[3] == int(expected_grid["width"])
        ),
        "embedding_is_unit_normalized": (
            float(probe["embedding"]["maximum_unit_norm_error"]) < 1e-3
        ),
        "memory_within_configured_cap": (
            float(probe["peak_reserved_mib"])
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "probe": probe,
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "backbone_smoke_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        spatial_hook=probe["spatial_hook"],
        peak_reserved_mib=probe["peak_reserved_mib"],
    )
    logger.log_metric(
        name="backbone_peak_reserved_memory",
        value=float(probe["peak_reserved_mib"]),
        step=0,
        split="environment",
        unit="MiB",
    )
    logger.log_metric(
        name="adaface_feature_norm",
        value=float(probe["embedding"]["norm_mean"]),
        step=0,
        split="train",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "checks": checks,
                "checkpoint": probe["checkpoint"],
                "input": probe["input"],
                "spatial_hook": probe["spatial_hook"],
                "spatial_feature": probe["spatial_feature"],
                "embedding": probe["embedding"],
                "peak_reserved_mib": probe["peak_reserved_mib"],
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_cache_real_features(args: argparse.Namespace) -> int:
    if args.dataset_root is None:
        raise ValueError("cache-real-features requires --dataset-root")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    checkpoint = (
        args.weights.expanduser().resolve()
        if args.weights is not None
        else (
            Path(config["paths"]["weights_dir"])
            / "adaface"
            / config["backbone"]["checkpoint_filename"]
        )
    )
    dataset_root = args.dataset_root.expanduser().resolve()
    split_root = resolve_real_split_root(
        dataset_root,
        profile=args.profile,
        seed=seed,
    )
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("cache-real-features requires CUDA")
    logger.log_event(
        "real_feature_cache_started",
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        split_root=split_root,
        device=str(device),
        resume=args.resume,
    )

    def log_cache_event(event: str, fields: dict[str, Any]) -> None:
        logger.log_event(event, **fields)

    cache_result = prepare_real_feature_cache(
        config=config,
        profile=args.profile,
        run_id=run_paths.root.name,
        dataset_root=dataset_root,
        split_root=split_root,
        checkpoint=checkpoint,
        resume=args.resume,
        device=device,
        event_callback=log_cache_event,
    )
    validation = validate_real_feature_cache(
        cache_result.root,
        verify_hashes=True,
    )
    peak_reserved_mib = float(
        cache_result.manifest["peak_reserved_mib"]
    )
    checks = {
        **validation["checks"],
        "checkpoint_strictly_compatible": (
            not cache_result.manifest["checkpoint"]["missing_keys"]
            and not cache_result.manifest["checkpoint"]["unexpected_keys"]
        ),
        "spatial_shape_matches_mvp": (
            tuple(cache_result.manifest["spatial_hook"]["shape"][1:])
            == (
                int(config["data"]["feature_grid"]["backbone_dim"]),
                int(config["data"]["feature_grid"]["height"]),
                int(config["data"]["feature_grid"]["width"]),
            )
        ),
        "memory_within_configured_cap": (
            peak_reserved_mib
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "feature_cache": {
            "root": str(cache_result.root),
            "manifest": str(cache_result.manifest_path),
            "reused": cache_result.reused,
            "summary": cache_result.manifest,
            "validation": validation,
        },
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "real_feature_cache_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        cache_root=cache_result.root,
        reused=cache_result.reused,
        peak_reserved_mib=peak_reserved_mib,
    )
    logger.log_metric(
        name="feature_cache_peak_reserved_memory",
        value=peak_reserved_mib,
        step=0,
        split="environment",
        unit="MiB",
    )
    logger.log_metric(
        name="local_feature_unit_norm_error",
        value=float(validation["maximum_local_unit_norm_error"]),
        step=0,
        split="all",
    )
    logger.log_metric(
        name="embedding_unit_norm_error",
        value=float(validation["maximum_embedding_unit_norm_error"]),
        step=0,
        split="all",
    )
    for split, split_summary in cache_result.manifest["splits"].items():
        logger.log_metric(
            name="cached_images",
            value=float(split_summary["images"]),
            step=0,
            split=split,
            unit="count",
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "cache_root": str(cache_result.root),
                "reused": cache_result.reused,
                "checks": checks,
                "splits": cache_result.manifest["splits"],
                "validation": validation,
                "spatial_hook": cache_result.manifest["spatial_hook"],
                "peak_reserved_mib": peak_reserved_mib,
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_prepare_real_sets(args: argparse.Namespace) -> int:
    if args.dataset_root is None:
        raise ValueError("prepare-real-sets requires --dataset-root")
    if args.feature_cache_root is None:
        raise ValueError("prepare-real-sets requires --feature-cache-root")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    checkpoint = (
        args.weights.expanduser().resolve()
        if args.weights is not None
        else (
            Path(config["paths"]["weights_dir"])
            / "adaface"
            / config["backbone"]["checkpoint_filename"]
        )
    )
    dataset_root = args.dataset_root.expanduser().resolve()
    feature_cache_root = args.feature_cache_root.expanduser().resolve()
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("prepare-real-sets requires CUDA")
    logger.log_event(
        "real_set_preparation_started",
        dataset_root=dataset_root,
        feature_cache_root=feature_cache_root,
        checkpoint=checkpoint,
        device=str(device),
        resume=args.resume,
    )

    def log_set_event(event: str, fields: dict[str, Any]) -> None:
        logger.log_event(event, **fields)

    result = prepare_real_set_dataset(
        config=config,
        profile=args.profile,
        run_id=run_paths.root.name,
        dataset_root=dataset_root,
        feature_cache_root=feature_cache_root,
        checkpoint=checkpoint,
        resume=args.resume,
        device=device,
        event_callback=log_set_event,
    )
    peak_reserved_mib = float(result.manifest["peak_reserved_mib"])
    checks = {
        **result.validation["checks"],
        "dataset_validation_passed": bool(result.validation["passed"]),
        "checkpoint_strictly_compatible": (
            not result.manifest["checkpoint"]["missing_keys"]
            and not result.manifest["checkpoint"]["unexpected_keys"]
        ),
        "memory_within_configured_cap": (
            peak_reserved_mib
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "real_sets": {
            "root": str(result.root),
            "manifest": str(result.manifest_path),
            "reused": result.reused,
            "summary": result.manifest,
            "validation": result.validation,
        },
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "real_set_preparation_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        root=result.root,
        reused=result.reused,
        peak_reserved_mib=peak_reserved_mib,
    )
    logger.log_metric(
        name="real_set_peak_reserved_memory",
        value=peak_reserved_mib,
        step=0,
        split="environment",
        unit="MiB",
    )
    for split, split_summary in result.manifest["splits"].items():
        logger.log_metric(
            name="real_sets",
            value=float(split_summary["sets"]),
            step=0,
            split=split,
            unit="count",
        )
        logger.log_metric(
            name="real_set_identities",
            value=float(split_summary["identities"]),
            step=0,
            split=split,
            unit="count",
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "dataset_root": str(result.root),
                "reused": result.reused,
                "checks": checks,
                "splits": result.manifest["splits"],
                "validation": result.validation,
                "peak_reserved_mib": peak_reserved_mib,
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_fit_codebook(args: argparse.Namespace) -> int:
    if args.dataset_root is None:
        raise ValueError("fit-codebook requires --dataset-root")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    command = [sys.executable, *sys.argv]
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=command,
    )
    logger.log_event(
        "quantization_fit_started",
        profile=args.profile,
        run_id=run_paths.root.name,
        dataset_root=args.dataset_root.resolve(),
    )
    preparation = prepare_quantization_artifacts(
        config=config,
        profile=args.profile,
        run_id=run_paths.root.name,
        dataset_root=args.dataset_root,
        resume=args.resume,
    )
    checks = {
        "artifact_validation_passed": bool(
            preparation.validation["passed"]
        ),
        "projection_fit_is_train_only": bool(
            preparation.validation["checks"]["fit_split_is_train"]
        ),
        "target_identity_splits_disjoint": all(
            bool(value)
            for key, value in preparation.validation["checks"].items()
            if key.startswith("identity_disjoint.")
        ),
    }
    passed = all(checks.values())
    artifact_manifest = preparation.manifest
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "quantization_artifacts": {
            "root": str(preparation.root),
            "manifest": str(preparation.manifest_path),
            "reused": preparation.reused,
            "projection": artifact_manifest["projection"],
            "codebook": artifact_manifest["codebook"],
        },
        "quantization_metrics": artifact_manifest["metrics"],
        "research_gate": artifact_manifest["research_gate"],
        "artifact_validation": preparation.validation,
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "quantization_fit_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        artifact_root=preparation.root,
        reused=preparation.reused,
        research_gate=artifact_manifest["research_gate"],
    )
    for split, split_metrics in artifact_manifest["metrics"].items():
        for name in (
            "mean_token_cosine",
            "mean_map_cosine",
            "codebook_utilization",
            "token_perplexity",
            "normalized_perplexity",
        ):
            logger.log_metric(
                name=name,
                value=float(split_metrics[name]),
                step=0,
                split=split,
            )
        for scenario, scenario_metrics in split_metrics[
            "by_scenario"
        ].items():
            logger.log_metric(
                name="mean_token_cosine",
                value=float(scenario_metrics["mean_token_cosine"]),
                step=0,
                split=split,
                scenario=scenario,
            )

    summary_metrics = {
        split: {
            key: value
            for key, value in metrics.items()
            if key
            in {
                "mean_token_cosine",
                "mean_map_cosine",
                "codebook_utilization",
                "token_perplexity",
                "normalized_perplexity",
            }
        }
        for split, metrics in artifact_manifest["metrics"].items()
    }
    summary = {
        "status": manifest["status"],
        "run_directory": str(run_paths.root),
        "artifact_root": str(preparation.root),
        "checks": checks,
        "projection": artifact_manifest["projection"],
        "codebook": artifact_manifest["codebook"],
        "metrics": summary_metrics,
        "research_gate": artifact_manifest["research_gate"],
        "git": environment["git"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _oom_candidates(config: dict[str, Any]) -> list[tuple[int, int, int]]:
    runtime = config["runtime"]
    batch_values = [int(value) for value in runtime["oom_batch_fallback"]]
    frame_values = [
        int(value) for value in runtime["oom_condition_frames_fallback"]
    ]
    hidden_values = [
        int(value) for value in runtime["oom_hidden_dim_fallback"]
    ]
    candidates = [
        (batch, frame_values[0], hidden_values[0])
        for batch in batch_values
    ]
    candidates.extend(
        (batch_values[-1], frames, hidden_values[0])
        for frames in frame_values[1:]
    )
    candidates.extend(
        (batch_values[-1], frame_values[-1], hidden)
        for hidden in hidden_values[1:]
    )
    deduplicated: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if candidate not in deduplicated:
            deduplicated.append(candidate)
    return deduplicated


def run_train_smoke(args: argparse.Namespace) -> int:
    if args.stage == "train-smoke" and args.profile != "smoke":
        raise ValueError("train-smoke requires --profile smoke")
    if args.stage == "train-real-smoke" and args.profile != "pilot":
        raise ValueError("train-real-smoke requires --profile pilot")
    if args.stage == "train-pilot" and args.profile != "pilot":
        raise ValueError("train-pilot requires --profile pilot")
    if args.dataset_root is None:
        raise ValueError("train-smoke requires --dataset-root")
    if args.quantization_root is None:
        raise ValueError("train-smoke requires --quantization-root")

    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    if args.stage == "train-real-smoke":
        config["training"].update(
            {
                "batch_size": 2,
                "gradient_accumulation": 1,
                "min_steps": 1,
                "max_steps": 2,
                "validate_every_steps": 1,
                "checkpoint_every_steps": 1,
                "warmup_steps": 0,
                "early_stop_patience": 1,
            }
        )
        config["runtime"]["oom_batch_fallback"] = [2, 1]
        validate_config(config)
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    quantization_validation = validate_quantization_artifacts(
        args.quantization_root
    )
    if not quantization_validation["passed"]:
        raise ValueError("quantization artifacts failed validation")
    train_dataset = TokenTrainingDataset(
        dataset_root=args.dataset_root,
        quantization_root=args.quantization_root,
        split="train",
        precompute=True,
    )
    val_dataset = TokenTrainingDataset(
        dataset_root=args.dataset_root,
        quantization_root=args.quantization_root,
        split="val",
        precompute=True,
    )
    test_dataset = (
        TokenTrainingDataset(
            dataset_root=args.dataset_root,
            quantization_root=args.quantization_root,
            split="test",
            precompute=True,
        )
        if args.stage == "train-pilot"
        else None
    )
    codebook = load_codebook(args.quantization_root)

    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    logger.log_event(
        "training_smoke_started",
        dataset_root=args.dataset_root.resolve(),
        quantization_root=args.quantization_root.resolve(),
        device=str(device),
    )

    def event_callback(event: str, fields: dict[str, Any]) -> None:
        logger.log_event(event, **fields)
        if event == "training_step":
            logger.log_metric(
                name="loss",
                value=float(fields["loss"]),
                step=int(fields["step"]),
                split="train",
                scenario=str(fields["mode"]),
            )
            logger.log_metric(
                name="masked_token_accuracy",
                value=float(fields["accuracy"]),
                step=int(fields["step"]),
                split="train",
                scenario=str(fields["mode"]),
            )

    attempts: list[dict[str, Any]] = []
    successful = None
    one_shot_result = None
    maskgit_result = None
    for attempt_index, (batch_size, frames, hidden_dim) in enumerate(
        _oom_candidates(config)
    ):
        attempt_name = (
            f"attempt_{attempt_index:02d}_b{batch_size}_m{frames}_h{hidden_dim}"
        )
        attempt_checkpoints = run_paths.checkpoints / attempt_name
        logger.log_event(
            "training_attempt_started",
            attempt=attempt_name,
            batch_size=batch_size,
            condition_frames=frames,
            hidden_dim=hidden_dim,
        )
        try:
            seed_everything(
                seed + 1000,
                deterministic=bool(config["runtime"]["deterministic"]),
            )
            base_model = build_model_from_config(
                config,
                hidden_dim=hidden_dim,
            )
            initial_state = copy.deepcopy(base_model.state_dict())
            parameter_count = base_model.parameter_count()
            del base_model

            models = {}
            results = {}
            for mode in ("one-shot", "maskgit"):
                seed_everything(
                    seed + 2000,
                    deterministic=bool(config["runtime"]["deterministic"]),
                )
                model = build_model_from_config(
                    config,
                    hidden_dim=hidden_dim,
                )
                model.load_state_dict(initial_state)
                latest = (
                    find_latest_checkpoint(
                        attempt_checkpoints,
                        mode=mode,
                    )
                    if args.resume
                    else None
                )
                result = train_single_model(
                    model=model,
                    dataset=train_dataset,
                    config=config,
                    mode=mode,
                    device=device,
                    checkpoint_dir=attempt_checkpoints,
                    batch_size=batch_size,
                    condition_frames=frames,
                    seed=seed + 3000,
                    resume_checkpoint=latest,
                    event_callback=event_callback,
                )
                result.model.cpu()
                models[mode] = result.model
                results[mode] = result
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            one_shot_result = results["one-shot"]
            maskgit_result = results["maskgit"]
            if (
                models["one-shot"].parameter_count()
                != models["maskgit"].parameter_count()
            ):
                raise RuntimeError("parameter-matched models differ in size")
            successful = {
                "attempt": attempt_name,
                "batch_size": batch_size,
                "condition_frames": frames,
                "hidden_dim": hidden_dim,
                "parameter_count": parameter_count,
                "models": models,
            }
            attempts.append(
                {
                    "attempt": attempt_name,
                    "status": "passed",
                    "batch_size": batch_size,
                    "condition_frames": frames,
                    "hidden_dim": hidden_dim,
                }
            )
            break
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            attempts.append(
                {
                    "attempt": attempt_name,
                    "status": "oom" if is_oom else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "batch_size": batch_size,
                    "condition_frames": frames,
                    "hidden_dim": hidden_dim,
                }
            )
            logger.log_event(
                "training_attempt_failed",
                level="warning" if is_oom else "error",
                **attempts[-1],
            )
            if not is_oom:
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if successful is None or one_shot_result is None or maskgit_result is None:
        raise RuntimeError("all configured OOM fallback attempts failed")

    one_shot_model = successful["models"]["one-shot"].to(device)
    maskgit_model = successful["models"]["maskgit"].to(device)
    evaluation = evaluate_training_smoke(
        one_shot_model=one_shot_model,
        maskgit_model=maskgit_model,
        dataset=val_dataset,
        codebook=codebook,
        config=config,
        device=device,
        condition_frames=int(successful["condition_frames"]),
    )
    research_evaluation = None
    if args.stage == "train-pilot":
        assert test_dataset is not None
        research_evaluation = {
            "val": evaluate_research_split(
                one_shot_model=one_shot_model,
                maskgit_model=maskgit_model,
                dataset=val_dataset,
                codebook=codebook,
                config=config,
                device=device,
                condition_frames=int(successful["condition_frames"]),
            ),
            "test": evaluate_research_split(
                one_shot_model=one_shot_model,
                maskgit_model=maskgit_model,
                dataset=test_dataset,
                codebook=codebook,
                config=config,
                device=device,
                condition_frames=int(successful["condition_frames"]),
            ),
        }
    one_shot_model.cpu()
    maskgit_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    peak_reserved = max(
        one_shot_result.peak_reserved_mib,
        maskgit_result.peak_reserved_mib,
    )
    checks = {
        "quantization_artifacts_valid": True,
        "parameter_counts_equal": (
            successful["models"]["one-shot"].parameter_count()
            == successful["models"]["maskgit"].parameter_count()
        ),
        "one_shot_checkpoint_exists": one_shot_result.checkpoint.is_file(),
        "maskgit_checkpoint_exists": maskgit_result.checkpoint.is_file(),
        "all_decoders_return_complete_tokens": all(
            value["trace"][-1]["actual_committed"]
            == int(config["data"]["feature_grid"]["height"])
            * int(config["data"]["feature_grid"]["width"])
            for value in evaluation["maskgit_modes"].values()
        ),
        "memory_within_configured_cap": peak_reserved
        <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0,
        "pilot_research_evaluation_complete": (
            True
            if args.stage != "train-pilot"
            else research_evaluation is not None
        ),
    }
    passed = all(checks.values())

    def result_summary(result: Any) -> dict[str, Any]:
        return {
            "steps": result.steps,
            "losses": result.losses,
            "masked_accuracies": result.accuracies,
            "learning_rates": result.learning_rates,
            "checkpoint": str(result.checkpoint),
            "peak_allocated_mib": result.peak_allocated_mib,
            "peak_reserved_mib": result.peak_reserved_mib,
            "resumed_from": result.resumed_from,
        }

    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "dataset_root": str(args.dataset_root.resolve()),
        "quantization_root": str(args.quantization_root.resolve()),
        "attempts": attempts,
        "selected_configuration": {
            key: value
            for key, value in successful.items()
            if key != "models"
        },
        "training": {
            "one-shot": result_summary(one_shot_result),
            "maskgit": result_summary(maskgit_result),
        },
        "smoke_evaluation": evaluation,
        "research_evaluation": research_evaluation,
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "training_smoke_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        selected_configuration=manifest["selected_configuration"],
    )
    if research_evaluation is not None:
        for split, split_results in research_evaluation.items():
            for method, method_results in split_results[
                "verification"
            ].items():
                for scenario, scenario_results in method_results.items():
                    if "roc_auc" not in scenario_results:
                        continue
                    logger.log_metric(
                        name="roc_auc",
                        value=float(scenario_results["roc_auc"]),
                        step=int(config["training"]["max_steps"]),
                        split=split,
                        scenario=f"{method}:{scenario}",
                    )
                    logger.log_metric(
                        name="eer",
                        value=float(scenario_results["eer"]),
                        step=int(config["training"]["max_steps"]),
                        split=split,
                        scenario=f"{method}:{scenario}",
                    )

    compact_training = {
        mode: {
            "steps": values["steps"],
            "initial_loss": values["losses"][0],
            "final_loss": values["losses"][-1],
            "final_masked_accuracy": values["masked_accuracies"][-1],
            "checkpoint": values["checkpoint"],
            "peak_reserved_mib": values["peak_reserved_mib"],
        }
        for mode, values in manifest["training"].items()
    }
    research_summary = None
    if research_evaluation is not None:
        key_methods = (
            "best_single_frame",
            "mean_pooling",
            "max_pooling",
            "quality_pooling",
            "one_shot_transformer",
            "maskgit_confidence",
            "maskgit_evidence_logits",
            "maskgit_evidence_remask",
        )
        research_summary = {
            split: {
                method: results["verification"][method]["hard_average"]
                for method in key_methods
            }
            for split, results in research_evaluation.items()
        }
    summary = {
        "status": manifest["status"],
        "run_directory": str(run_paths.root),
        "checks": checks,
        "selected_configuration": manifest["selected_configuration"],
        "training": compact_training,
        "smoke_evaluation": (
            evaluation
            if args.stage != "train-pilot"
            else {
                "permutation": evaluation["permutation"],
                "evidence_remask_commit_visibility_correlation": (
                    evaluation["maskgit_modes"]["evidence-remask"][
                        "commit_score_visibility_rank_correlation"
                    ]
                ),
            }
        ),
        "research_summary": research_summary,
        "git": environment["git"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }


def _checkpoint_metadata_snapshot(
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for checkpoints in sorted(
        path for path in output_root.glob("*/checkpoints") if path.is_dir()
    ):
        for path in sorted(
            value for value in checkpoints.rglob("*") if value.is_file()
        ):
            stat = path.stat()
            result[path.relative_to(output_root).as_posix()] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _file_sha256(path),
            }
    return result


def _reference_baseline_delta(
    *,
    split_results: dict[str, Any],
    reference_manifest: dict[str, Any],
    methods: tuple[str, ...] = (
        "mean_pooling",
        "quality_pooling",
        "mean_pooling_quantized",
        "quality_pooling_quantized",
    ),
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    for split in ("val", "test"):
        reference_split = reference_manifest["research_evaluation"][split]
        if (
            reference_split["gallery"]
            != "clean_disjoint_teacher_continuous"
        ):
            raise ValueError("reference run used an unexpected gallery")
        for method in methods:
            observed = split_results[split]["methods"][method][
                "verification"
            ]
            expected = reference_split["verification"][method]
            for scenario in (
                "clean",
                "low_quality",
                "complementary_occlusion",
                "common_occlusion",
                "wrong_identity",
                "all",
                "hard_average",
            ):
                for metric in ("roc_auc", "eer"):
                    delta = abs(
                        float(observed[scenario][metric])
                        - float(expected[scenario][metric])
                    )
                    differences.append(
                        {
                            "split": split,
                            "method": method,
                            "scenario": scenario,
                            "metric": metric,
                            "absolute_difference": delta,
                        }
                    )
                if scenario in {"hard_average"}:
                    continue
                for far in ("0.01", "0.001"):
                    delta = abs(
                        float(
                            observed[scenario]["tar_at_far"][far]["tar"]
                        )
                        - float(
                            expected[scenario]["tar_at_far"][far]["tar"]
                        )
                    )
                    differences.append(
                        {
                            "split": split,
                            "method": method,
                            "scenario": scenario,
                            "metric": f"tar_at_far_{far}",
                            "absolute_difference": delta,
                        }
                    )
    worst = max(differences, key=lambda row: row["absolute_difference"])
    return {
        "maximum_absolute_difference": worst["absolute_difference"],
        "worst_comparison": worst,
        "compared_values": len(differences),
    }


def _p0_core_baseline_delta(
    *,
    split_results: dict[str, Any],
    reference_csv: Path,
) -> dict[str, Any]:
    """Regress P0-2 fixed baselines against the reviewed P0-1 table."""

    if not reference_csv.is_file():
        raise FileNotFoundError(
            f"P0-1 reference table not found: {reference_csv}"
        )
    with reference_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {
        (row["split"], row["method"]): row
        for row in rows
    }
    methods = (
        "mean_pooling",
        "quality_pooling",
        "quality_pooling_quantized",
        "condition_robust_pooling",
    )
    differences: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for method in methods:
            expected = indexed[(split, method)]
            observed = split_results[split]["methods"][method]
            verification = observed["verification"]
            comparisons = {
                "pooled_all_roc_auc": verification["all"]["roc_auc"],
                "hard_macro_roc_auc": verification["hard_average"][
                    "roc_auc"
                ],
                "eer": verification["all"]["eer"],
                "tar_at_far_1e-2": verification["all"]["tar_at_far"][
                    "0.01"
                ]["tar"],
                "tar_at_far_1e-3": verification["all"]["tar_at_far"][
                    "0.001"
                ]["tar"],
                "rank1_identification_accuracy": verification["all"][
                    "rank1_accuracy"
                ],
                "mean_genuine_impostor_margin": verification["all"][
                    "mean_genuine_impostor_margin"
                ],
                "teacher_map_cosine": observed["teacher_similarity"]["all"][
                    "map_cosine"
                ],
            }
            for metric, observed_value in comparisons.items():
                delta = abs(
                    float(observed_value) - float(expected[metric])
                )
                differences.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "absolute_difference": delta,
                    }
                )
    worst = max(differences, key=lambda row: row["absolute_difference"])
    return {
        "reference_csv": str(reference_csv),
        "maximum_absolute_difference": worst["absolute_difference"],
        "worst_comparison": worst,
        "compared_values": len(differences),
    }


def run_evaluate_condition_pooling(args: argparse.Namespace) -> int:
    if args.profile != "pilot":
        raise ValueError("evaluate-condition-pooling requires --profile pilot")
    if args.dataset_root is None:
        raise ValueError(
            "evaluate-condition-pooling requires --dataset-root"
        )
    if args.quantization_root is None:
        raise ValueError(
            "evaluate-condition-pooling requires --quantization-root"
        )
    if args.training_run is None:
        raise ValueError(
            "evaluate-condition-pooling requires --training-run for protocol "
            "regression only; no checkpoint will be loaded"
        )
    if args.resume:
        raise ValueError(
            "evaluate-condition-pooling is collision-safe and does not "
            "overwrite or resume an existing run"
        )

    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    robust_parameters = validate_p0_robust_parameters(config)
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    reference_run = args.training_run.expanduser().resolve()
    reference_manifest_path = reference_run / "logs" / "run_manifest.json"
    if not reference_manifest_path.is_file():
        raise FileNotFoundError(
            f"reference training manifest not found: {reference_manifest_path}"
        )
    # The reference manifest contains historical test metrics. Its contents
    # are deliberately not loaded until validation parameters are locked.

    dataset_manifest_path = dataset_root / "manifest.json"
    quantization_manifest_path = quantization_root / "manifest.json"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(
            f"set dataset manifest not found: {dataset_manifest_path}"
        )
    if not quantization_manifest_path.is_file():
        raise FileNotFoundError(
            f"quantization manifest not found: {quantization_manifest_path}"
        )
    dataset_manifest_sha_before = _file_sha256(dataset_manifest_path)
    quantization_manifest = json.loads(
        quantization_manifest_path.read_text(encoding="utf-8")
    )
    source_link_matches = (
        Path(quantization_manifest["source_dataset"]).resolve()
        == dataset_root
        and quantization_manifest["source_manifest_sha256"]
        == dataset_manifest_sha_before
    )
    projection_path = quantization_root / "projection.pt"
    codebook_path = quantization_root / "codebook.pt"
    projection_sha_before = _file_sha256(projection_path)
    codebook_sha_before = _file_sha256(codebook_path)
    dataset_validation_before = validate_set_dataset(dataset_root)
    quantization_validation_before = validate_quantization_artifacts(
        quantization_root
    )
    if not dataset_validation_before["passed"]:
        raise ValueError("source set dataset failed pre-run validation")
    if not quantization_validation_before["passed"]:
        raise ValueError(
            "source quantization artifacts failed pre-run validation"
        )

    output_root = Path(config["paths"]["output_dir"]).resolve()
    stage15_root = output_root / "stage15-final-report"
    stage15_before = _tree_hash_snapshot(stage15_root)
    checkpoint_metadata_before = _checkpoint_metadata_snapshot(output_root)
    run_paths = create_run_paths(
        output_root,
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=False,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    validate_p0_robust_parameters(config)
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("evaluate-condition-pooling requires CUDA")
    logger.log_event(
        "condition_pooling_evaluation_started",
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        reference_training_run=reference_run,
        device=str(device),
        robust_parameters=robust_parameters,
    )

    datasets = {
        split: TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=quantization_root,
            split=split,
            precompute=True,
        )
        for split in ("val", "test")
    }
    codebook = load_codebook(quantization_root)
    split_results = {
        split: evaluate_condition_pooling_split(
            split=split,
            dataset=datasets[split],
            codebook=codebook,
            config=config,
            device=device,
        )
        for split in ("val", "test")
    }
    latency = measure_condition_pooling_latency(
        dataset=datasets["test"],
        codebook=codebook,
        config=config,
        device=device,
    )
    protocol_regression = _reference_baseline_delta(
        split_results=split_results,
        reference_manifest=reference_manifest,
    )

    dataset_validation_after = validate_set_dataset(dataset_root)
    quantization_validation_after = validate_quantization_artifacts(
        quantization_root
    )
    dataset_manifest_sha_after = _file_sha256(dataset_manifest_path)
    projection_sha_after = _file_sha256(projection_path)
    codebook_sha_after = _file_sha256(codebook_path)
    stage15_after = _tree_hash_snapshot(stage15_root)
    checkpoint_metadata_after = _checkpoint_metadata_snapshot(output_root)

    checks = {
        "source_dataset_valid_before": bool(
            dataset_validation_before["passed"]
        ),
        "source_dataset_valid_after": bool(
            dataset_validation_after["passed"]
        ),
        "quantization_valid_before": bool(
            quantization_validation_before["passed"]
        ),
        "quantization_valid_after": bool(
            quantization_validation_after["passed"]
        ),
        "quantization_source_link_matches": source_link_matches,
        "dataset_manifest_unchanged": (
            dataset_manifest_sha_before == dataset_manifest_sha_after
        ),
        "projection_unchanged": projection_sha_before == projection_sha_after,
        "codebook_unchanged": codebook_sha_before == codebook_sha_after,
        "stage15_unchanged": stage15_before == stage15_after,
        "checkpoints_unchanged": (
            checkpoint_metadata_before == checkpoint_metadata_after
        ),
        "reference_gallery_matches": all(
            reference_manifest["research_evaluation"][split]["gallery"]
            == "clean_disjoint_teacher_continuous"
            for split in ("val", "test")
        ),
        "baseline_protocol_reproduced": (
            protocol_regression["maximum_absolute_difference"] <= 1e-7
        ),
        "all_output_shapes_valid": all(
            split_results[split]["correctness"]["output_shape_valid"]
            for split in ("val", "test")
        ),
        "all_outputs_unit_normalized": all(
            split_results[split]["correctness"]["output_unit_norm_valid"]
            for split in ("val", "test")
        ),
        "all_frame_weights_sum_to_one": all(
            split_results[split]["correctness"]["frame_weight_sum_valid"]
            for split in ("val", "test")
        ),
        "condition_permutation_equivalent": all(
            split_results[split]["correctness"]["permutation_equivalent"]
            for split in ("val", "test")
        ),
        "three_of_four_frames_kept": all(
            split_results[split]["correctness"][
                "kept_frames_per_position"
            ]
            == 3
            for split in ("val", "test")
        ),
        "no_test_parameter_tuning": robust_parameters
        == {
            "robust_keep_fraction": 0.75,
            "consensus_weight": 2.0,
            "quality_weight": 1.0,
        },
        "no_transformer_loaded_or_trained": True,
    }
    maximum_peak_reserved = max(
        float(values["peak_reserved_mib"]) for values in latency.values()
    )
    checks["memory_within_configured_cap"] = (
        maximum_peak_reserved
        <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
    )
    passed = all(checks.values())
    reproducibility = {
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "reference_training_run": str(reference_run),
        "dataset_manifest_sha256_before": dataset_manifest_sha_before,
        "dataset_manifest_sha256_after": dataset_manifest_sha_after,
        "projection_sha256_before": projection_sha_before,
        "projection_sha256_after": projection_sha_after,
        "codebook_sha256_before": codebook_sha_before,
        "codebook_sha256_after": codebook_sha_after,
        "stage15_file_snapshot_before": stage15_before,
        "stage15_file_snapshot_after": stage15_after,
        "checkpoint_metadata_before": checkpoint_metadata_before,
        "checkpoint_metadata_after": checkpoint_metadata_after,
        "protocol_regression": protocol_regression,
        "dataset_validation_before": dataset_validation_before,
        "dataset_validation_after": dataset_validation_after,
        "quantization_validation_before": quantization_validation_before,
        "quantization_validation_after": quantization_validation_after,
    }
    artifacts = write_condition_pooling_artifacts(
        output_directory=run_paths.artifacts,
        split_results=split_results,
        latency=latency,
        checks=checks,
        reproducibility=reproducibility,
    )
    artifact_checks = {
        f"artifact_{name}_exists": Path(path).is_file()
        and Path(path).stat().st_size > 0
        for name, path in artifacts.items()
        if isinstance(path, str)
    }
    checks.update(artifact_checks)
    passed = all(checks.values())
    core_metrics = {
        split: {
            method: {
                "pooled_all_roc_auc": split_results[split]["methods"][
                    method
                ]["verification"]["all"]["roc_auc"],
                "hard_macro_roc_auc": split_results[split]["methods"][
                    method
                ]["verification"]["hard_average"]["roc_auc"],
                "eer": split_results[split]["methods"][method][
                    "verification"
                ]["all"]["eer"],
                "rank1_accuracy": split_results[split]["methods"][method][
                    "verification"
                ]["all"]["rank1_accuracy"],
                "teacher_map_cosine": split_results[split]["methods"][
                    method
                ]["teacher_similarity"]["all"]["map_cosine"],
                "exact_token_accuracy": split_results[split]["methods"][
                    method
                ]["exact_token_accuracy"],
            }
            for method in CONDITION_POOLING_METHODS
        }
        for split in ("val", "test")
    }
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "reference_training_run": str(reference_run),
        "protocol": {
            "gallery": "clean_disjoint_teacher_continuous",
            "robust_parameters": robust_parameters,
            "test_parameter_tuning": False,
            "transformer_loaded_or_trained": False,
        },
        "core_metrics": core_metrics,
        "latency": latency,
        "correctness": {
            split: split_results[split]["correctness"]
            for split in ("val", "test")
        },
        "artifacts": artifacts,
        "reproducibility": reproducibility,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "condition_pooling_evaluation_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        maximum_peak_reserved_mib=maximum_peak_reserved,
        decision=artifacts["decision"],
    )
    for split in ("val", "test"):
        for method in CONDITION_POOLING_METHODS:
            values = core_metrics[split][method]
            logger.log_metric(
                name="pooled_all_roc_auc",
                value=float(values["pooled_all_roc_auc"]),
                step=0,
                split=split,
                scenario=method,
            )
            logger.log_metric(
                name="hard_macro_roc_auc",
                value=float(values["hard_macro_roc_auc"]),
                step=0,
                split=split,
                scenario=method,
            )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "checks": checks,
                "core_metrics": core_metrics,
                "latency_ms_per_set": {
                    method: values["per_set_latency_ms"]
                    for method, values in latency.items()
                },
                "decision": artifacts["decision"],
                "artifacts": {
                    key: value
                    for key, value in artifacts.items()
                    if key != "decision"
                },
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_evaluate_visibility_aggregation(args: argparse.Namespace) -> int:
    if args.profile != "pilot":
        raise ValueError(
            "evaluate-visibility-aggregation requires --profile pilot"
        )
    if args.dataset_root is None or args.quantization_root is None:
        raise ValueError(
            "evaluate-visibility-aggregation requires --dataset-root and "
            "--quantization-root"
        )
    if args.training_run is None:
        raise ValueError(
            "evaluate-visibility-aggregation requires --training-run for "
            "read-only protocol regression"
        )
    if args.resume:
        raise ValueError(
            "evaluate-visibility-aggregation does not overwrite or resume"
        )
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    reference_run = args.training_run.expanduser().resolve()
    reference_manifest_path = reference_run / "logs" / "run_manifest.json"
    if not reference_manifest_path.is_file():
        raise FileNotFoundError(
            f"reference training manifest not found: {reference_manifest_path}"
        )
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    if reference_manifest.get("status") != "passed":
        raise ValueError("reference training run did not pass")
    if Path(reference_manifest["dataset_root"]).resolve() != dataset_root:
        raise ValueError("reference run used a different set dataset")
    if (
        Path(reference_manifest["quantization_root"]).resolve()
        != quantization_root
    ):
        raise ValueError("reference run used different quantization artifacts")

    dataset_manifest_path = dataset_root / "manifest.json"
    quantization_manifest_path = quantization_root / "manifest.json"
    dataset_manifest_sha_before = _file_sha256(dataset_manifest_path)
    quantization_manifest = json.loads(
        quantization_manifest_path.read_text(encoding="utf-8")
    )
    source_link_matches = (
        Path(quantization_manifest["source_dataset"]).resolve()
        == dataset_root
        and quantization_manifest["source_manifest_sha256"]
        == dataset_manifest_sha_before
    )
    projection_path = quantization_root / "projection.pt"
    codebook_path = quantization_root / "codebook.pt"
    projection_sha_before = _file_sha256(projection_path)
    codebook_sha_before = _file_sha256(codebook_path)
    dataset_validation_before = validate_set_dataset(dataset_root)
    quantization_validation_before = validate_quantization_artifacts(
        quantization_root
    )
    if not dataset_validation_before["passed"]:
        raise ValueError("source set dataset failed validation")
    if not quantization_validation_before["passed"]:
        raise ValueError("source quantization artifacts failed validation")

    output_root = Path(config["paths"]["output_dir"]).resolve()
    stage15_root = output_root / "stage15-final-report"
    p0_reference_root = (
        output_root / "p0-condition-robust-pooling-reviewed"
    )
    p0_reference_csv = (
        p0_reference_root / "artifacts" / "core_comparison.csv"
    )
    if not p0_reference_csv.is_file():
        raise FileNotFoundError(
            f"reviewed P0-1 reference not found: {p0_reference_csv}"
        )
    stage15_before = _tree_hash_snapshot(stage15_root)
    p0_reference_before = _tree_hash_snapshot(p0_reference_root)
    checkpoint_metadata_before = _checkpoint_metadata_snapshot(output_root)
    run_paths = create_run_paths(
        output_root,
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=False,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("evaluate-visibility-aggregation requires CUDA")
    logger.log_event(
        "visibility_aggregation_started",
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        reference_training_run=reference_run,
        device=str(device),
    )

    train_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    reference_start = time.perf_counter()
    clean_reference, reference_details = build_train_clean_reference(
        dataset=train_dataset,
        config=config,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reference_build_seconds = time.perf_counter() - reference_start
    reference_details = {
        **reference_details,
        "build_seconds": reference_build_seconds,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "source_dataset_manifest_sha256": dataset_manifest_sha_before,
        "source_projection_sha256": projection_sha_before,
    }
    reference_bank_path = run_paths.artifacts / "clean_reference_bank.pt"
    torch.save(clean_reference.to_payload(), reference_bank_path)
    reference_details["artifact"] = str(reference_bank_path)
    reference_details["artifact_sha256"] = _file_sha256(reference_bank_path)
    reference_manifest_path_output = (
        run_paths.artifacts / "reference_bank_manifest.json"
    )
    reference_manifest_path_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit_split": "train",
                "dataset_root": str(dataset_root),
                "quantization_root": str(quantization_root),
                **reference_details,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.log_event(
        "train_reference_bank_built",
        **reference_details,
    )
    del train_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    val_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="val",
        precompute=True,
    )
    selection = evaluate_validation_candidates(
        dataset=val_dataset,
        reference=clean_reference,
        config=config,
        device=device,
    )
    selected_in_memory = selection["selected"]
    aggregation_parameters = {
        "algorithm_version": "p0_2_top2_identity_absolute_local_gate_v1",
        "top_k_clean": int(selected_in_memory["top_k_clean"]),
        "identity_weight": float(selected_in_memory["identity_weight"]),
        "local_reliability_weight": float(
            config["p0_2"]["local_reliability_weight"]
        ),
        "quality_weight": float(config["p0_2"]["quality_weight"]),
        "minimum_frame_weight": float(
            config["p0_2"]["minimum_frame_weight"]
        ),
        "local_z_clip": [
            float(value) for value in config["p0_2"]["local_z_clip"]
        ],
    }
    selection_lock_path = run_paths.artifacts / "selection_lock.json"
    selection_lock_payload = {
        "schema_version": 1,
        "selection_split": "val",
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "selected": selected_in_memory,
        "aggregation_parameters": aggregation_parameters,
        "candidate_count": selection["candidate_count"],
        "feasible_candidate_count": selection["feasible_candidate_count"],
        "fallback_to_unconstrained": selection[
            "fallback_to_unconstrained"
        ],
        "selection_rule": selection["selection_rule"],
        "constraints": selection["constraints"],
        "config_sha256": config_sha256(config),
        "dataset_manifest_sha256": dataset_manifest_sha_before,
        "reference_bank_sha256": reference_details["artifact_sha256"],
    }
    selection_lock_path.write_text(
        json.dumps(
            selection_lock_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selection_lock_sha_before_test = _file_sha256(selection_lock_path)
    locked_selection = json.loads(
        selection_lock_path.read_text(encoding="utf-8")
    )
    if locked_selection["config_sha256"] != config_sha256(config):
        raise RuntimeError("selection lock config hash mismatch")
    if (
        locked_selection["reference_bank_sha256"]
        != reference_details["artifact_sha256"]
    ):
        raise RuntimeError("selection lock reference bank hash mismatch")
    if (
        locked_selection["dataset_manifest_sha256"]
        != dataset_manifest_sha_before
    ):
        raise RuntimeError("selection lock dataset hash mismatch")
    locked_parameters = locked_selection["aggregation_parameters"]
    if locked_parameters != aggregation_parameters:
        raise RuntimeError("selection lock aggregation parameters mismatch")
    locked_config = copy.deepcopy(config)
    for key in (
        "local_reliability_weight",
        "quality_weight",
        "minimum_frame_weight",
        "local_z_clip",
    ):
        locked_config["p0_2"][key] = copy.deepcopy(locked_parameters[key])
    lock_reloaded_before_test = True
    logger.log_event(
        "validation_parameters_locked",
        selection_lock=selection_lock_path,
        selection_lock_sha256=selection_lock_sha_before_test,
        selected=locked_selection["selected"],
        aggregation_parameters=locked_parameters,
        lock_reloaded=True,
        test_dataset_loaded=False,
    )

    selected_top_k = int(locked_parameters["top_k_clean"])
    selected_identity_weight = float(
        locked_parameters["identity_weight"]
    )
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    if reference_manifest.get("status") != "passed":
        raise ValueError("reference training run did not pass")
    if Path(reference_manifest["dataset_root"]).resolve() != dataset_root:
        raise ValueError("reference run used a different set dataset")
    if (
        Path(reference_manifest["quantization_root"]).resolve()
        != quantization_root
    ):
        raise ValueError("reference run used different quantization artifacts")
    codebook = load_codebook(quantization_root)
    val_results = evaluate_locked_split(
        split="val",
        dataset=val_dataset,
        reference=clean_reference,
        codebook=codebook,
        selected_top_k=selected_top_k,
        selected_identity_weight=selected_identity_weight,
        config=locked_config,
        device=device,
    )
    lock_verified_before_test_construction = (
        lock_reloaded_before_test
        and selection_lock_path.is_file()
        and _file_sha256(selection_lock_path)
        == selection_lock_sha_before_test
        and not bool(locked_selection["test_token_dataset_constructed"])
    )
    test_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="test",
        precompute=True,
    )
    logger.log_event(
        "locked_test_evaluation_started",
        selection_lock_sha256=selection_lock_sha_before_test,
        selected=locked_selection["selected"],
        aggregation_parameters=locked_parameters,
    )
    test_results = evaluate_locked_split(
        split="test",
        dataset=test_dataset,
        reference=clean_reference,
        codebook=codebook,
        selected_top_k=selected_top_k,
        selected_identity_weight=selected_identity_weight,
        config=locked_config,
        device=device,
    )
    logger.log_event(
        "locked_test_evaluation_completed",
        evaluated_configuration_count=test_results[
            "evaluated_configuration_count"
        ],
        evaluated_parameters=test_results["selected_parameters"],
    )
    split_results = {"val": val_results, "test": test_results}
    selection_lock_sha_after_test = _file_sha256(selection_lock_path)
    latency = measure_visibility_aggregation_latency(
        dataset=test_dataset,
        reference=clean_reference,
        codebook=codebook,
        selected_top_k=selected_top_k,
        selected_identity_weight=selected_identity_weight,
        config=locked_config,
        device=device,
    )
    protocol_regression = _reference_baseline_delta(
        split_results=split_results,
        reference_manifest=reference_manifest,
        methods=(
            "mean_pooling",
            "quality_pooling",
            "quality_pooling_quantized",
        ),
    )
    p0_core_regression = _p0_core_baseline_delta(
        split_results=split_results,
        reference_csv=p0_reference_csv,
    )

    dataset_validation_after = validate_set_dataset(dataset_root)
    quantization_validation_after = validate_quantization_artifacts(
        quantization_root
    )
    dataset_manifest_sha_after = _file_sha256(dataset_manifest_path)
    projection_sha_after = _file_sha256(projection_path)
    codebook_sha_after = _file_sha256(codebook_path)
    stage15_after_evaluation = _tree_hash_snapshot(stage15_root)
    p0_reference_after_evaluation = _tree_hash_snapshot(p0_reference_root)
    checkpoint_metadata_after_evaluation = _checkpoint_metadata_snapshot(
        output_root
    )
    maximum_peak_reserved = max(
        float(values["peak_reserved_mib"]) for values in latency.values()
    )
    checks = {
        "source_dataset_valid_before": bool(
            dataset_validation_before["passed"]
        ),
        "source_dataset_valid_after": bool(
            dataset_validation_after["passed"]
        ),
        "quantization_valid_before": bool(
            quantization_validation_before["passed"]
        ),
        "quantization_valid_after": bool(
            quantization_validation_after["passed"]
        ),
        "quantization_source_link_matches": source_link_matches,
        "dataset_manifest_unchanged": (
            dataset_manifest_sha_before == dataset_manifest_sha_after
        ),
        "projection_unchanged": projection_sha_before == projection_sha_after,
        "codebook_unchanged": codebook_sha_before == codebook_sha_after,
        "stage15_unchanged_before_artifact_write": (
            stage15_before == stage15_after_evaluation
        ),
        "checkpoints_unchanged_before_artifact_write": (
            checkpoint_metadata_before
            == checkpoint_metadata_after_evaluation
        ),
        "p0_1_reference_unchanged_before_artifact_write": (
            p0_reference_before == p0_reference_after_evaluation
        ),
        "train_only_reference_bank": (
            reference_details["fit_split"] == "train"
        ),
        "reference_bank_shape": (
            reference_details["shape"] == [49, 800, 128]
        ),
        "reference_bank_identity_count": (
            reference_details["unique_teacher_identities"] == 200
        ),
        "reference_target_condition_overlap_zero": (
            reference_details["target_condition_teacher_overlap"] == 0
        ),
        "reference_actual_source_overlap_documented": (
            reference_details["actual_source_teacher_overlap"]
            == len(
                reference_details[
                    "actual_source_teacher_overlap_indices"
                ]
            )
        ),
        "selection_used_validation_only": (
            selection["selection_split"] == "val"
            and not selection["test_metrics_consulted"]
        ),
        "selection_lock_written_and_reloaded_before_test": (
            lock_verified_before_test_construction
        ),
        "selection_lock_unchanged_during_test": (
            selection_lock_sha_before_test == selection_lock_sha_after_test
        ),
        "locked_val_parameters_match_lock": (
            val_results["selected_parameters"] == locked_parameters
        ),
        "locked_test_parameters_match_lock": (
            test_results["selected_parameters"] == locked_parameters
        ),
        "locked_test_configuration_count_one": (
            test_results["evaluated_configuration_count"] == 1
        ),
        "test_grid_not_evaluated": (
            test_results["evaluated_configuration_count"] == 1
            and not any(
                name.startswith("combined_k")
                for name in test_results["methods"]
            )
        ),
        "reference_gallery_matches": all(
            split_results[split]["gallery"]
            == "clean_disjoint_teacher_continuous"
            for split in ("val", "test")
        ),
        "baseline_protocol_reproduced": (
            protocol_regression["maximum_absolute_difference"] <= 1e-7
        ),
        "p0_1_core_protocol_reproduced": (
            p0_core_regression["maximum_absolute_difference"] <= 1e-7
        ),
        "all_output_norms_valid": all(
            split_results[split]["correctness"][
                "output_unit_norm_valid"
            ]
            for split in ("val", "test")
        ),
        "all_frame_weights_sum_to_one": all(
            split_results[split]["correctness"]["weight_sum_valid"]
            for split in ("val", "test")
        ),
        "minimum_weight_floor_respected": all(
            split_results[split]["correctness"][
                "minimum_weight_floor_respected"
            ]
            for split in ("val", "test")
        ),
        "condition_permutation_equivalent": all(
            split_results[split]["correctness"]["permutation_equivalent"]
            for split in ("val", "test")
        ),
        "no_transformer_loaded_or_trained": True,
        "memory_within_configured_cap": (
            maximum_peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
    }
    reproducibility = {
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "reference_training_run": str(reference_run),
        "dataset_manifest_sha256_before": dataset_manifest_sha_before,
        "dataset_manifest_sha256_after": dataset_manifest_sha_after,
        "projection_sha256_before": projection_sha_before,
        "projection_sha256_after": projection_sha_after,
        "codebook_sha256_before": codebook_sha_before,
        "codebook_sha256_after": codebook_sha_after,
        "stage15_file_snapshot_before": stage15_before,
        "stage15_file_snapshot_after_evaluation": (
            stage15_after_evaluation
        ),
        "p0_1_reference_snapshot_before": p0_reference_before,
        "p0_1_reference_snapshot_after_evaluation": (
            p0_reference_after_evaluation
        ),
        "checkpoint_metadata_before": checkpoint_metadata_before,
        "checkpoint_metadata_after_evaluation": (
            checkpoint_metadata_after_evaluation
        ),
        "selection_lock": str(selection_lock_path),
        "selection_lock_sha256_before_test": (
            selection_lock_sha_before_test
        ),
        "selection_lock_sha256_after_test": selection_lock_sha_after_test,
        "protocol_regression": protocol_regression,
        "p0_1_core_protocol_regression": p0_core_regression,
        "dataset_validation_before": dataset_validation_before,
        "dataset_validation_after": dataset_validation_after,
        "quantization_validation_before": quantization_validation_before,
        "quantization_validation_after": quantization_validation_after,
    }
    artifacts = write_visibility_aggregation_artifacts(
        output_directory=run_paths.artifacts,
        split_results=split_results,
        selection=selection,
        reference_details=reference_details,
        latency=latency,
        config=locked_config,
        checks=checks,
        reproducibility=reproducibility,
    )
    artifacts["selection_lock_json"] = str(selection_lock_path)
    artifacts["reference_bank_manifest_json"] = str(
        reference_manifest_path_output
    )
    artifacts["reference_bank_pt"] = str(reference_bank_path)
    artifact_checks = {
        f"artifact_{name}_exists": Path(path).is_file()
        and Path(path).stat().st_size > 0
        for name, path in artifacts.items()
        if isinstance(path, str)
    }
    checks.update(artifact_checks)
    stage15_after_final_artifacts = _tree_hash_snapshot(stage15_root)
    p0_reference_after_final_artifacts = _tree_hash_snapshot(
        p0_reference_root
    )
    checkpoint_metadata_after_final_artifacts = (
        _checkpoint_metadata_snapshot(output_root)
    )
    checks.update(
        {
            "stage15_unchanged_after_all_artifact_writes": (
                stage15_before == stage15_after_final_artifacts
            ),
            "checkpoints_unchanged_after_all_artifact_writes": (
                checkpoint_metadata_before
                == checkpoint_metadata_after_final_artifacts
            ),
            "p0_1_reference_unchanged_after_all_artifact_writes": (
                p0_reference_before
                == p0_reference_after_final_artifacts
            ),
        }
    )
    reproducibility.update(
        {
            "stage15_file_snapshot_after_final_artifacts": (
                stage15_after_final_artifacts
            ),
            "checkpoint_metadata_after_final_artifacts": (
                checkpoint_metadata_after_final_artifacts
            ),
            "p0_1_reference_snapshot_after_final_artifacts": (
                p0_reference_after_final_artifacts
            ),
        }
    )
    diagnostics_path = Path(artifacts["diagnostics_json"])
    diagnostics_payload = json.loads(
        diagnostics_path.read_text(encoding="utf-8")
    )
    diagnostics_payload["checks"] = checks
    diagnostics_payload["reproducibility"] = reproducibility
    diagnostics_path.write_text(
        json.dumps(
            diagnostics_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": locked_config,
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "reference_training_run": str(reference_run),
        "reference_bank": reference_details,
        "selection": {
            key: value for key, value in selection.items() if key != "rows"
        },
        "selection_lock": locked_selection,
        "locked_test_evaluated_configuration_count": test_results[
            "evaluated_configuration_count"
        ],
        "locked_test_evaluated_parameters": test_results[
            "selected_parameters"
        ],
        "latency": latency,
        "correctness": {
            split: split_results[split]["correctness"]
            for split in ("val", "test")
        },
        "artifacts": artifacts,
        "reproducibility": reproducibility,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "visibility_aggregation_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        decision=artifacts["decision"],
    )
    for split in ("val", "test"):
        for method in VISIBILITY_AGGREGATION_METHODS:
            values = split_results[split]["methods"][method]["verification"]
            logger.log_metric(
                name="pooled_all_roc_auc",
                value=float(values["all"]["roc_auc"]),
                step=0,
                split=split,
                scenario=method,
            )
            logger.log_metric(
                name="hard_macro_roc_auc",
                value=float(values["hard_average"]["roc_auc"]),
                step=0,
                split=split,
                scenario=method,
            )
    compact_metrics = {
        split: {
            method: {
                "pooled_all_roc_auc": split_results[split]["methods"][
                    method
                ]["verification"]["all"]["roc_auc"],
                "hard_macro_roc_auc": split_results[split]["methods"][
                    method
                ]["verification"]["hard_average"]["roc_auc"],
            }
            for method in VISIBILITY_AGGREGATION_METHODS
        }
        for split in ("val", "test")
    }
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "checks": checks,
                "reference_bank": reference_details,
                "selection": manifest["selection"],
                "core_metrics": compact_metrics,
                "latency_ms_per_set": {
                    name: values["per_set_latency_ms"]
                    for name, values in latency.items()
                },
                "decision": artifacts["decision"],
                "artifacts": {
                    key: value
                    for key, value in artifacts.items()
                    if key != "decision"
                },
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_evaluate_earlier_hook_oracle(args: argparse.Namespace) -> int:
    """Run the P0-3 validation-gated earlier-hook oracle experiment."""

    if args.profile != "pilot":
        raise ValueError(
            "evaluate-earlier-hook-oracle requires --profile pilot"
        )
    if args.dataset_root is None or args.quantization_root is None:
        raise ValueError(
            "evaluate-earlier-hook-oracle requires --dataset-root and "
            "--quantization-root"
        )
    if args.resume:
        raise ValueError(
            "evaluate-earlier-hook-oracle is collision-safe and cannot resume"
        )
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    set_manifest_path = dataset_root / "manifest.json"
    quantization_manifest_path = quantization_root / "manifest.json"
    if not set_manifest_path.is_file():
        raise FileNotFoundError(
            f"set dataset manifest not found: {set_manifest_path}"
        )
    if not quantization_manifest_path.is_file():
        raise FileNotFoundError(
            f"quantization manifest not found: {quantization_manifest_path}"
        )
    set_manifest = json.loads(
        set_manifest_path.read_text(encoding="utf-8")
    )
    set_manifest_sha_before = _file_sha256(set_manifest_path)
    quantization_manifest_sha_before = _file_sha256(
        quantization_manifest_path
    )
    projection_path = quantization_root / "projection.pt"
    codebook_path = quantization_root / "codebook.pt"
    projection_sha_before = _file_sha256(projection_path)
    codebook_sha_before = _file_sha256(codebook_path)
    checkpoint = (
        args.weights.expanduser().resolve()
        if args.weights is not None
        else Path(set_manifest["checkpoint"]["path"]).resolve()
    )
    checkpoint_sha_before = _file_sha256(checkpoint)
    if checkpoint_sha_before != set_manifest["checkpoint"]["sha256"]:
        raise ValueError("AdaFace checkpoint differs from stage9 provenance")

    output_root = Path(config["paths"]["output_dir"]).resolve()
    protected_roots = {
        "stage15": output_root / "stage15-final-report",
        "p0_1": output_root / "p0-condition-robust-pooling-reviewed",
        "p0_2": (
            output_root
            / "p0-2-visibility-reliability-pooling-reviewed"
        ),
    }
    protected_before = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    checkpoint_metadata_before = _checkpoint_metadata_snapshot(output_root)
    run_paths = create_run_paths(
        output_root,
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=False,
    )
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=config["logging"]["jsonl_filename"],
        metrics_filename=config["logging"]["metrics_filename"],
        manifest_filename=config["logging"]["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("evaluate-earlier-hook-oracle requires CUDA")
    logger.log_event(
        "earlier_hook_oracle_started",
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        checkpoint=checkpoint,
        test_metrics_loaded=False,
    )

    model, checkpoint_report = load_adaface_ir50(
        checkpoint,
        map_location="cpu",
    )
    model.to(device).eval()
    model.requires_grad_(False)
    candidate_hooks = [
        str(value) for value in config["p0_3"]["candidate_hooks"]
    ]
    probed = probe_named_spatial_hooks(
        model,
        candidate_hooks,
        device=device,
    )
    observed_shapes = {
        name: list(probed[name].shape[1:])
        for name in candidate_hooks
    }
    expected_shapes = {
        name: [
            int(value)
            for value in config["p0_3"]["expected_shapes"][name]
        ]
        for name in candidate_hooks
    }
    if observed_shapes != expected_shapes:
        raise ValueError(
            f"preregistered hook shapes changed: {observed_shapes}"
        )

    preregistration = {
        "schema_version": 1,
        "stage": "P0-3",
        "objective": (
            "earlier-hook metadata-oracle headroom before any learned "
            "visibility estimator"
        ),
        "candidate_hooks": candidate_hooks,
        "observed_shapes": observed_shapes,
        "late_control_hook": set_manifest["spatial_hook"],
        "pooling_space": "raw_early_activation",
        "suffix": (
            "body[index+1:] then output_layer[0:2], existing PCA to 128"
        ),
        "visibility_resize": "nearest_from_existing_7x7_metadata",
        "gallery": "clean_disjoint_teacher_continuous",
        "selection_thresholds": {
            key: value
            for key, value in config["p0_3"].items()
            if key.startswith("selection_")
            or key == "suffix_replay_min_map_cosine"
        },
        "test_policy": (
            "If no validation candidate passes every gate, write "
            "selected=null/test_authorized=false and do not construct test "
            "TokenTrainingDataset, images, or early cache."
        ),
        "scope_exclusions": [
            "no adapter training",
            "no new PCA",
            "no quantization",
            "no Transformer",
            "no diffusion",
        ],
        "project_wide_holdout_caveat": (
            "P0-1/P0-2 already exposed the current test identities; this is "
            "only an internally locked P0-3 evaluation."
        ),
        "source_hashes": {
            "set_manifest": set_manifest_sha_before,
            "quantization_manifest": quantization_manifest_sha_before,
            "projection": projection_sha_before,
            "codebook": codebook_sha_before,
            "checkpoint": checkpoint_sha_before,
        },
        "config_sha256": config_sha256(config),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "test_images_loaded": False,
    }
    preregistration_path = (
        run_paths.artifacts / "protocol_preregistration.json"
    )
    preregistration_path.write_text(
        json.dumps(
            preregistration,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preregistration_sha = _file_sha256(preregistration_path)
    logger.log_event(
        "p0_3_protocol_preregistered",
        path=preregistration_path,
        sha256=preregistration_sha,
        candidate_hooks=candidate_hooks,
        test_dataset_loaded=False,
    )

    val_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="val",
        precompute=True,
    )
    val_cache_directory = (
        run_paths.artifacts / "early_feature_cache" / "val"
    )
    val_cache = cache_early_hook_features(
        split="val",
        hook_names=candidate_hooks,
        hook_shapes=observed_shapes,
        model=model,
        token_dataset=val_dataset,
        set_dataset_root=dataset_root,
        set_manifest=set_manifest,
        output_directory=val_cache_directory,
        config=config,
        device=device,
    )
    val_cache_manifest_path = (
        run_paths.artifacts / "val_early_cache_manifest.json"
    )
    val_cache_manifest_path.write_text(
        json.dumps(
            val_cache,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_results = {}
    for hook_name in candidate_hooks:
        validation_results[hook_name] = evaluate_early_hook_split(
            split="val",
            hook_name=hook_name,
            hook_shape=observed_shapes[hook_name],
            cache_path=Path(val_cache["files"][hook_name]["path"]),
            dataset=val_dataset,
            model=model,
            config=config,
            device=device,
        )
    selection = select_validation_hook(
        candidate_results=validation_results,
        cache_details=val_cache,
        config=config,
    )
    selection_lock_payload = {
        "schema_version": 1,
        "selection_split": "val",
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "test_images_loaded": False,
        "test_authorized": selection["test_authorized"],
        "selected": selection["selected"],
        "best_validation_candidate": selection[
            "best_validation_candidate"
        ],
        "selection_rule": selection["selection_rule"],
        "thresholds": selection["thresholds"],
        "candidate_count": selection["candidate_count"],
        "feasible_candidate_count": selection[
            "feasible_candidate_count"
        ],
        "protocol_preregistration_sha256": preregistration_sha,
        "config_sha256": config_sha256(config),
        "source_hashes": preregistration["source_hashes"],
        "checkpoint_module_keys": checkpoint_report.model_keys,
        "pooling_formula": (
            "raw sum_f weight[b,y,x,f]*activation[b,f,c,y,x]"
        ),
        "visibility_resize": "nearest",
        "suffix": "body[index+1:] + output_layer[0:2] + existing PCA",
    }
    selection_lock_path = run_paths.artifacts / "selection_lock.json"
    selection_lock_path.write_text(
        json.dumps(
            selection_lock_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selection_lock_sha_before_test = _file_sha256(selection_lock_path)
    locked_selection = json.loads(
        selection_lock_path.read_text(encoding="utf-8")
    )
    if (
        locked_selection["protocol_preregistration_sha256"]
        != preregistration_sha
        or locked_selection["config_sha256"] != config_sha256(config)
        or locked_selection["source_hashes"]
        != preregistration["source_hashes"]
    ):
        raise RuntimeError("P0-3 selection lock failed hash validation")
    lock_reloaded_before_test = True
    logger.log_event(
        "p0_3_validation_selection_locked",
        path=selection_lock_path,
        sha256=selection_lock_sha_before_test,
        test_authorized=locked_selection["test_authorized"],
        selected=locked_selection["selected"],
        test_dataset_loaded=False,
    )
    # The quantization manifest contains historical test metrics. Hashing the
    # file is harmless, but parsing it is deliberately delayed until the
    # validation-only selection has been written and reloaded.
    quantization_manifest = json.loads(
        quantization_manifest_path.read_text(encoding="utf-8")
    )
    quantization_source_link_matches = (
        Path(quantization_manifest["source_dataset"]).resolve()
        == dataset_root
        and quantization_manifest["source_manifest_sha256"]
        == set_manifest_sha_before
    )
    if not quantization_source_link_matches:
        raise ValueError("quantization artifacts do not link to this dataset")
    logger.log_event(
        "p0_3_quantization_manifest_loaded_after_selection_lock",
        path=quantization_manifest_path,
        selection_lock_sha256=selection_lock_sha_before_test,
        historical_test_metrics_not_used_for_selection=True,
    )

    best_hook = str(
        selection["best_validation_candidate"]["hook_name"]
    )
    latency = measure_early_hook_latency(
        hook_name=best_hook,
        hook_shape=observed_shapes[best_hook],
        cache_path=Path(val_cache["files"][best_hook]["path"]),
        dataset=val_dataset,
        model=model,
        config=config,
        device=device,
    )
    cache_details: dict[str, Any] = {"val": val_cache}
    test_result = None
    test_cache = None
    test_dataset_constructed = False
    if bool(locked_selection["test_authorized"]):
        selected = locked_selection["selected"]
        if selected is None:
            raise RuntimeError("authorized test lock has no selected hook")
        selected_hook = str(selected["hook_name"])
        if selected_hook not in candidate_hooks:
            raise RuntimeError("locked hook is not preregistered")
        if list(selected["hook_shape"]) != observed_shapes[selected_hook]:
            raise RuntimeError("locked hook shape differs from probe")
        test_dataset = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=quantization_root,
            split="test",
            precompute=True,
        )
        test_dataset_constructed = True
        test_cache_directory = (
            run_paths.artifacts / "early_feature_cache" / "test"
        )
        test_cache = cache_early_hook_features(
            split="test",
            hook_names=[selected_hook],
            hook_shapes=observed_shapes,
            model=model,
            token_dataset=test_dataset,
            set_dataset_root=dataset_root,
            set_manifest=set_manifest,
            output_directory=test_cache_directory,
            config=config,
            device=device,
        )
        test_cache_manifest_path = (
            run_paths.artifacts / "test_early_cache_manifest.json"
        )
        test_cache_manifest_path.write_text(
            json.dumps(
                test_cache,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        cache_details["test"] = test_cache
        test_result = evaluate_early_hook_split(
            split="test",
            hook_name=selected_hook,
            hook_shape=observed_shapes[selected_hook],
            cache_path=Path(test_cache["files"][selected_hook]["path"]),
            dataset=test_dataset,
            model=model,
            config=config,
            device=device,
        )
        logger.log_event(
            "p0_3_locked_test_completed",
            evaluated_configuration_count=test_result[
                "evaluated_configuration_count"
            ],
            hook_name=selected_hook,
        )
    else:
        logger.log_event(
            "p0_3_test_skipped_by_validation_lock",
            selected=None,
            test_authorized=False,
        )
    selection_lock_sha_after_test = _file_sha256(selection_lock_path)

    # Full source validators are deliberately delayed until after the
    # validation selection lock because they inspect every split artifact.
    dataset_validation = validate_set_dataset(dataset_root)
    quantization_validation = validate_quantization_artifacts(
        quantization_root
    )
    set_manifest_sha_after = _file_sha256(set_manifest_path)
    quantization_manifest_sha_after = _file_sha256(
        quantization_manifest_path
    )
    projection_sha_after = _file_sha256(projection_path)
    codebook_sha_after = _file_sha256(codebook_path)
    checkpoint_sha_after = _file_sha256(checkpoint)
    protected_after_evaluation = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    checkpoint_metadata_after_evaluation = _checkpoint_metadata_snapshot(
        output_root
    )
    replay_threshold = float(
        config["p0_3"]["suffix_replay_min_map_cosine"]
    )
    validation_correctness = all(
        (
            result["correctness"]["all_output_shapes_valid"]
            and result["correctness"]["output_unit_norm_valid"]
            and result["correctness"]["permutation_equivalent"]
            and result["correctness"]["suffix_replay_map_cosine_min"]
            >= replay_threshold
            and result["correctness"][
                "cache_roundtrip_token_cosine_min"
            ]
            >= replay_threshold
        )
        for result in validation_results.values()
    )
    cache_replay_valid = all(
        replay["late_spatial_cosine_min"] >= replay_threshold
        for replay in val_cache["pipeline_regression"][
            "suffix_replay"
        ].values()
    )
    maximum_peak_reserved = max(
        [
            float(val_cache["peak_reserved_mib"]),
            *[
                float(values["peak_reserved_mib"])
                for values in latency.values()
            ],
            *(
                [float(test_cache["peak_reserved_mib"])]
                if test_cache is not None
                else []
            ),
        ]
    )
    expected_test_policy = (
        (
            bool(locked_selection["test_authorized"])
            and test_dataset_constructed
            and test_result is not None
            and test_result["evaluated_configuration_count"] == 1
            and test_result["hook_name"]
            == locked_selection["selected"]["hook_name"]
        )
        if bool(locked_selection["test_authorized"])
        else (
            not test_dataset_constructed
            and test_result is None
            and test_cache is None
            and locked_selection["selected"] is None
        )
    )
    checks = {
        "dataset_valid_after_lock": bool(dataset_validation["passed"]),
        "quantization_valid_after_lock": bool(
            quantization_validation["passed"]
        ),
        "set_manifest_unchanged": (
            set_manifest_sha_before == set_manifest_sha_after
        ),
        "quantization_manifest_unchanged": (
            quantization_manifest_sha_before
            == quantization_manifest_sha_after
        ),
        "quantization_source_link_matches": (
            quantization_source_link_matches
        ),
        "projection_unchanged": projection_sha_before == projection_sha_after,
        "codebook_unchanged": codebook_sha_before == codebook_sha_after,
        "checkpoint_unchanged": (
            checkpoint_sha_before == checkpoint_sha_after
        ),
        "backbone_frozen_eval": (
            not model.training
            and not any(parameter.requires_grad for parameter in model.parameters())
        ),
        "candidate_hooks_and_shapes_preregistered": (
            observed_shapes == expected_shapes
            and selection["candidate_count"] == len(candidate_hooks)
        ),
        "protocol_written_before_validation": (
            preregistration_path.is_file()
            and preregistration_sha == _file_sha256(preregistration_path)
        ),
        "selection_used_validation_only": (
            selection["selection_split"] == "val"
            and not selection["test_metrics_consulted"]
        ),
        "selection_lock_reloaded_before_test": lock_reloaded_before_test,
        "selection_lock_unchanged": (
            selection_lock_sha_before_test
            == selection_lock_sha_after_test
        ),
        "test_policy_followed": expected_test_policy,
        "validation_pipeline_reproduced": (
            val_cache["pipeline_regression"][
                "projected_map_cosine_min"
            ]
            >= replay_threshold
            and cache_replay_valid
        ),
        "validation_outputs_correct": validation_correctness,
        "test_outputs_correct_if_evaluated": (
            True
            if test_result is None
            else (
                test_result["correctness"]["all_output_shapes_valid"]
                and test_result["correctness"]["output_unit_norm_valid"]
                and test_result["correctness"]["permutation_equivalent"]
                and test_result["correctness"][
                    "suffix_replay_map_cosine_min"
                ]
                >= replay_threshold
                and test_result["correctness"][
                    "cache_roundtrip_token_cosine_min"
                ]
                >= replay_threshold
            )
        ),
        "same_gallery_protocol": all(
            result["gallery"] == "clean_disjoint_teacher_continuous"
            for result in [
                *validation_results.values(),
                *([test_result] if test_result is not None else []),
            ]
        ),
        "no_training_or_quantized_early_method": True,
        "memory_within_configured_cap": (
            maximum_peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
        "protected_outputs_unchanged_before_artifact_write": (
            protected_before == protected_after_evaluation
        ),
        "checkpoints_unchanged_before_artifact_write": (
            checkpoint_metadata_before
            == checkpoint_metadata_after_evaluation
        ),
    }
    reproducibility = {
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "checkpoint": str(checkpoint),
        "source_hashes_before": preregistration["source_hashes"],
        "source_hashes_after": {
            "set_manifest": set_manifest_sha_after,
            "quantization_manifest": quantization_manifest_sha_after,
            "projection": projection_sha_after,
            "codebook": codebook_sha_after,
            "checkpoint": checkpoint_sha_after,
        },
        "protocol_preregistration": str(preregistration_path),
        "protocol_preregistration_sha256": preregistration_sha,
        "selection_lock": str(selection_lock_path),
        "selection_lock_sha256_before_test": (
            selection_lock_sha_before_test
        ),
        "selection_lock_sha256_after_test": selection_lock_sha_after_test,
        "protected_snapshots_before": protected_before,
        "protected_snapshots_after_evaluation": (
            protected_after_evaluation
        ),
        "checkpoint_metadata_before": checkpoint_metadata_before,
        "checkpoint_metadata_after_evaluation": (
            checkpoint_metadata_after_evaluation
        ),
        "dataset_validation_after_lock": dataset_validation,
        "quantization_validation_after_lock": quantization_validation,
    }
    artifacts = write_p0_3_artifacts(
        output_directory=run_paths.artifacts,
        preregistration=preregistration,
        selection=selection,
        validation_results=validation_results,
        test_result=test_result,
        cache_details=cache_details,
        latency=latency,
        config=config,
        checks=checks,
        reproducibility=reproducibility,
    )
    artifacts.update(
        {
            "protocol_preregistration_json": str(preregistration_path),
            "selection_lock_json": str(selection_lock_path),
            "val_early_cache_manifest_json": str(
                val_cache_manifest_path
            ),
        }
    )
    if test_cache is not None:
        artifacts["test_early_cache_manifest_json"] = str(
            run_paths.artifacts / "test_early_cache_manifest.json"
        )
    artifact_checks = {
        f"artifact_{name}_exists": (
            Path(path).is_file() and Path(path).stat().st_size > 0
        )
        for name, path in artifacts.items()
        if isinstance(path, str)
    }
    cache_file_checks = {
        f"cache_{split}_{hook.replace('.', '_')}_exists": (
            Path(values["path"]).is_file()
            and Path(values["path"]).stat().st_size > 0
        )
        for split, details in cache_details.items()
        for hook, values in details["files"].items()
    }
    checks.update(artifact_checks)
    checks.update(cache_file_checks)
    protected_after_artifacts = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    checkpoint_metadata_after_artifacts = _checkpoint_metadata_snapshot(
        output_root
    )
    checks.update(
        {
            "protected_outputs_unchanged_after_artifacts": (
                protected_before == protected_after_artifacts
            ),
            "checkpoints_unchanged_after_artifacts": (
                checkpoint_metadata_before
                == checkpoint_metadata_after_artifacts
            ),
        }
    )
    reproducibility.update(
        {
            "protected_snapshots_after_artifacts": (
                protected_after_artifacts
            ),
            "checkpoint_metadata_after_artifacts": (
                checkpoint_metadata_after_artifacts
            ),
        }
    )
    diagnostics_path = Path(artifacts["diagnostics_json"])
    diagnostics = json.loads(
        diagnostics_path.read_text(encoding="utf-8")
    )
    diagnostics["checks"] = checks
    diagnostics["reproducibility"] = reproducibility
    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "checkpoint": {
            "path": str(checkpoint_report.path),
            "sha256": checkpoint_report.sha256,
            "model_keys": checkpoint_report.model_keys,
            "source_keys": checkpoint_report.source_keys,
            "missing_keys": checkpoint_report.missing_keys,
            "unexpected_keys": checkpoint_report.unexpected_keys,
        },
        "preregistration": preregistration,
        "selection": {
            key: value for key, value in selection.items() if key != "rows"
        },
        "selection_lock": locked_selection,
        "test_dataset_constructed": test_dataset_constructed,
        "test_evaluated_configuration_count": (
            0
            if test_result is None
            else test_result["evaluated_configuration_count"]
        ),
        "cache": cache_details,
        "latency": latency,
        "artifacts": artifacts,
        "decision": artifacts["decision"],
        "reproducibility": reproducibility,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "earlier_hook_oracle_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        decision=artifacts["decision"],
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "candidate_shapes": observed_shapes,
                "selection": manifest["selection"],
                "test_dataset_constructed": test_dataset_constructed,
                "decision": artifacts["decision"],
                "checks": checks,
                "cache_summary": {
                    split: {
                        "build_seconds": details["build_seconds"],
                        "peak_reserved_mib": details["peak_reserved_mib"],
                        "files": details["files"],
                    }
                    for split, details in cache_details.items()
                },
                "artifacts": {
                    key: value
                    for key, value in artifacts.items()
                    if key != "decision"
                },
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_evaluate_continuous_residual_oracle(
    args: argparse.Namespace,
) -> int:
    """Run the validation-gated continuous residual headroom experiment."""

    if args.profile != "pilot":
        raise ValueError(
            "evaluate-continuous-residual-oracle requires --profile pilot"
        )
    if args.dataset_root is None or args.quantization_root is None:
        raise ValueError(
            "evaluate-continuous-residual-oracle requires --dataset-root "
            "and --quantization-root"
        )
    if args.resume:
        raise ValueError(
            "evaluate-continuous-residual-oracle is collision-safe and "
            "cannot resume"
        )
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    set_manifest_path = dataset_root / "manifest.json"
    quantization_manifest_path = quantization_root / "manifest.json"
    if not set_manifest_path.is_file():
        raise FileNotFoundError(
            f"set dataset manifest not found: {set_manifest_path}"
        )
    if not quantization_manifest_path.is_file():
        raise FileNotFoundError(
            f"quantization manifest not found: {quantization_manifest_path}"
        )
    set_manifest = json.loads(
        set_manifest_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "set_manifest": set_manifest_path,
        "quantization_manifest": quantization_manifest_path,
        "projection": quantization_root / "projection.pt",
        "codebook": quantization_root / "codebook.pt",
        "train_targets": quantization_root / "train_targets.pt",
        "val_targets": quantization_root / "val_targets.pt",
        "test_targets": quantization_root / "test_targets.pt",
        "adaface_checkpoint": Path(
            set_manifest["checkpoint"]["path"]
        ).resolve(),
    }
    missing_sources = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "continuous residual source artifact(s) missing: "
            + ", ".join(missing_sources)
        )
    source_hashes_before = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }

    output_root = Path(config["paths"]["output_dir"]).resolve()
    protected_roots = {
        "stage15": output_root / "stage15-final-report",
        "p0_1": output_root / "p0-condition-robust-pooling-reviewed",
        "p0_2": (
            output_root
            / "p0-2-visibility-reliability-pooling-reviewed"
        ),
        "p0_3": (
            output_root
            / "p0-3-earlier-hook-oracle-reviewed-v4"
        ),
    }
    protected_before = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    checkpoint_metadata_before = _checkpoint_metadata_snapshot(output_root)
    run_paths = create_run_paths(
        output_root,
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=False,
    )
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=config["logging"]["jsonl_filename"],
        metrics_filename=config["logging"]["metrics_filename"],
        manifest_filename=config["logging"]["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError(
            "evaluate-continuous-residual-oracle requires CUDA"
        )
    configured_methods = [
        str(value) for value in config["p1_0"]["candidate_methods"]
    ]
    preregistration = {
        "schema_version": 1,
        "stage": "P1-0",
        "objective": (
            "teacher-guided continuous local residual information headroom"
        ),
        "anchor": "late_quality_pooling",
        "candidate_methods": configured_methods,
        "oracle_parameters": {
            key: value
            for key, value in config["p1_0"].items()
            if not key.startswith("selection_")
        },
        "selection_thresholds": {
            key: value
            for key, value in config["p1_0"].items()
            if key.startswith("selection_")
        },
        "teacher_use": (
            "Per-probe continuous teacher map selects coefficients over "
            "condition features. This is non-deployable oracle supervision."
        ),
        "residual_constraint": (
            "Observed-frame convex hull or residual in the local frame-"
            "disagreement span with configured norm cap."
        ),
        "gallery": "clean_disjoint_teacher_continuous",
        "test_policy": (
            "If no validation candidate passes every gate, lock "
            "selected=null/test_authorized=false and do not construct the "
            "test TokenTrainingDataset. Otherwise evaluate exactly one "
            "locked oracle configuration on test."
        ),
        "scope_exclusions": [
            "no shared-parameter training",
            "no Transformer",
            "no diffusion",
            "no new PCA",
            "no codebook fitting or quantized residual",
            "no backbone execution or checkpoint update",
        ],
        "project_wide_holdout_caveat": (
            "Current test identities were exposed in earlier P0 stages; "
            "this remains an internally locked diagnostic, not a fresh "
            "confirmatory holdout."
        ),
        "source_hashes": source_hashes_before,
        "config_sha256": config_sha256(config),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
    }
    preregistration_path = (
        run_paths.artifacts / "protocol_preregistration.json"
    )
    preregistration_path.write_text(
        json.dumps(
            preregistration,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preregistration_sha = _file_sha256(preregistration_path)
    logger.log_event(
        "continuous_residual_protocol_preregistered",
        path=preregistration_path,
        sha256=preregistration_sha,
        candidate_methods=configured_methods,
        quantization_manifest_metrics_loaded=False,
        test_dataset_constructed=False,
    )

    val_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="val",
        precompute=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    validation_result = evaluate_continuous_residual_split(
        split="val",
        dataset=val_dataset,
        config=config,
        device=device,
    )
    validation_peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    selection = select_continuous_residual_candidate(
        validation_result=validation_result,
        config=config,
    )
    selection_lock_payload = {
        "schema_version": 1,
        "selection_split": "val",
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "test_authorized": selection["test_authorized"],
        "selected": selection["selected"],
        "best_validation_candidate": selection[
            "best_validation_candidate"
        ],
        "candidate_count": selection["candidate_count"],
        "feasible_candidate_count": selection[
            "feasible_candidate_count"
        ],
        "selection_rule": selection["selection_rule"],
        "thresholds": selection["thresholds"],
        "protocol_preregistration_sha256": preregistration_sha,
        "config_sha256": config_sha256(config),
        "source_hashes": source_hashes_before,
        "teacher_supervision_at_inference": True,
        "deployable": False,
    }
    selection_lock_path = run_paths.artifacts / "selection_lock.json"
    selection_lock_path.write_text(
        json.dumps(
            selection_lock_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selection_lock_sha_before_test = _file_sha256(selection_lock_path)
    locked_selection = json.loads(
        selection_lock_path.read_text(encoding="utf-8")
    )
    if (
        locked_selection["protocol_preregistration_sha256"]
        != preregistration_sha
        or locked_selection["config_sha256"] != config_sha256(config)
        or locked_selection["source_hashes"] != source_hashes_before
    ):
        raise RuntimeError(
            "continuous residual selection lock failed validation"
        )
    logger.log_event(
        "continuous_residual_validation_selection_locked",
        path=selection_lock_path,
        sha256=selection_lock_sha_before_test,
        selected=locked_selection["selected"],
        test_authorized=locked_selection["test_authorized"],
        test_dataset_constructed=False,
    )

    # The quantization manifest contains historical test metrics and is not
    # parsed until validation selection has been persisted and reloaded.
    quantization_manifest = json.loads(
        quantization_manifest_path.read_text(encoding="utf-8")
    )
    quantization_source_link_matches = (
        Path(quantization_manifest["source_dataset"]).resolve()
        == dataset_root
        and quantization_manifest["source_manifest_sha256"]
        == source_hashes_before["set_manifest"]
    )
    if not quantization_source_link_matches:
        raise ValueError("quantization artifacts do not link to this dataset")
    logger.log_event(
        "continuous_residual_quantization_manifest_loaded_after_lock",
        selection_lock_sha256=selection_lock_sha_before_test,
        historical_test_metrics_not_used_for_selection=True,
    )

    best_method = str(
        selection["best_validation_candidate"]["method"]
    )
    latency = measure_continuous_residual_latency(
        dataset=val_dataset,
        method=best_method,
        config=config,
        device=device,
    )
    test_result = None
    test_dataset_constructed = False
    test_peak_reserved = 0.0
    if bool(locked_selection["test_authorized"]):
        selected = locked_selection["selected"]
        if selected is None:
            raise RuntimeError("authorized residual test has no method")
        selected_method = str(selected["method"])
        if selected_method not in configured_methods:
            raise RuntimeError("locked residual method is not preregistered")
        test_dataset = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=quantization_root,
            split="test",
            precompute=True,
        )
        test_dataset_constructed = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        test_result = evaluate_continuous_residual_split(
            split="test",
            dataset=test_dataset,
            config=config,
            device=device,
            oracle_methods=[selected_method],
        )
        test_peak_reserved = (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        logger.log_event(
            "continuous_residual_locked_test_completed",
            method=selected_method,
            evaluated_oracle_configuration_count=1,
        )
    else:
        logger.log_event(
            "continuous_residual_test_skipped_by_validation_lock",
            selected=None,
            test_authorized=False,
        )
    selection_lock_sha_after_test = _file_sha256(selection_lock_path)

    dataset_validation = validate_set_dataset(dataset_root)
    quantization_validation = validate_quantization_artifacts(
        quantization_root
    )
    source_hashes_after = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    protected_after_evaluation = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    checkpoint_metadata_after_evaluation = _checkpoint_metadata_snapshot(
        output_root
    )
    validation_correctness = validation_result["correctness"]
    test_policy_followed = (
        (
            bool(locked_selection["test_authorized"])
            and test_dataset_constructed
            and test_result is not None
            and len(test_result["oracle_methods"]) == 1
            and test_result["oracle_methods"][0]
            == locked_selection["selected"]["method"]
        )
        if bool(locked_selection["test_authorized"])
        else (
            not test_dataset_constructed
            and test_result is None
            and locked_selection["selected"] is None
        )
    )
    latency_peak = max(
        float(values["peak_reserved_mib"])
        for values in latency.values()
    )
    maximum_peak_reserved = max(
        validation_peak_reserved,
        test_peak_reserved,
        latency_peak,
    )
    checks = {
        "dataset_valid_after_lock": bool(dataset_validation["passed"]),
        "quantization_valid_after_lock": bool(
            quantization_validation["passed"]
        ),
        "quantization_source_link_matches": (
            quantization_source_link_matches
        ),
        "source_artifacts_unchanged": (
            source_hashes_before == source_hashes_after
        ),
        "selection_lock_unchanged": (
            selection_lock_sha_before_test
            == selection_lock_sha_after_test
        ),
        "selection_used_validation_only": (
            selection["selection_split"] == "val"
            and not selection["test_metrics_consulted"]
        ),
        "test_policy_followed": test_policy_followed,
        "validation_outputs_correct": (
            validation_correctness["output_shape_valid"]
            and validation_correctness["output_unit_norm_valid"]
            and validation_correctness["quality_anchor_reproduced"]
            and validation_correctness["all_oracle_methods_correct"]
        ),
        "test_outputs_correct_if_evaluated": (
            True
            if test_result is None
            else (
                test_result["correctness"]["output_shape_valid"]
                and test_result["correctness"]["output_unit_norm_valid"]
                and test_result["correctness"][
                    "quality_anchor_reproduced"
                ]
                and test_result["correctness"][
                    "all_oracle_methods_correct"
                ]
            )
        ),
        "same_gallery_protocol": (
            validation_result["gallery"]
            == "clean_disjoint_teacher_continuous"
            and (
                test_result is None
                or test_result["gallery"]
                == "clean_disjoint_teacher_continuous"
            )
        ),
        "no_shared_parameter_training": True,
        "teacher_supervision_disclosed": (
            not validation_result["deployable"]
            and "teacher" in validation_result["oracle_supervision"]
        ),
        "memory_within_configured_cap": (
            maximum_peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
        "protected_outputs_unchanged_before_artifact_write": (
            protected_before == protected_after_evaluation
        ),
        "checkpoints_unchanged_before_artifact_write": (
            checkpoint_metadata_before
            == checkpoint_metadata_after_evaluation
        ),
    }
    reproducibility = {
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "protocol_preregistration": str(preregistration_path),
        "protocol_preregistration_sha256": preregistration_sha,
        "selection_lock": str(selection_lock_path),
        "selection_lock_sha256_before_test": (
            selection_lock_sha_before_test
        ),
        "selection_lock_sha256_after_test": selection_lock_sha_after_test,
        "dataset_validation_after_lock": dataset_validation,
        "quantization_validation_after_lock": quantization_validation,
        "protected_snapshots_before": protected_before,
        "protected_snapshots_after_evaluation": (
            protected_after_evaluation
        ),
        "checkpoint_metadata_before": checkpoint_metadata_before,
        "checkpoint_metadata_after_evaluation": (
            checkpoint_metadata_after_evaluation
        ),
        "validation_peak_reserved_mib": validation_peak_reserved,
        "test_peak_reserved_mib": test_peak_reserved,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
    }
    artifacts = write_continuous_residual_artifacts(
        output_directory=run_paths.artifacts,
        preregistration=preregistration,
        selection=selection,
        validation_result=validation_result,
        test_result=test_result,
        latency=latency,
        config=config,
        checks=checks,
        reproducibility=reproducibility,
    )
    artifacts.update(
        {
            "protocol_preregistration_json": str(preregistration_path),
            "selection_lock_json": str(selection_lock_path),
        }
    )
    artifact_checks = {
        f"artifact_{name}_exists": (
            Path(path).is_file() and Path(path).stat().st_size > 0
        )
        for name, path in artifacts.items()
        if isinstance(path, str)
    }
    checks.update(artifact_checks)
    protected_after_artifacts = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    checkpoint_metadata_after_artifacts = _checkpoint_metadata_snapshot(
        output_root
    )
    checks.update(
        {
            "protected_outputs_unchanged_after_artifacts": (
                protected_before == protected_after_artifacts
            ),
            "checkpoints_unchanged_after_artifacts": (
                checkpoint_metadata_before
                == checkpoint_metadata_after_artifacts
            ),
        }
    )
    reproducibility.update(
        {
            "protected_snapshots_after_artifacts": (
                protected_after_artifacts
            ),
            "checkpoint_metadata_after_artifacts": (
                checkpoint_metadata_after_artifacts
            ),
        }
    )
    diagnostics_path = Path(artifacts["diagnostics_json"])
    diagnostics = json.loads(
        diagnostics_path.read_text(encoding="utf-8")
    )
    diagnostics["checks"] = checks
    diagnostics["reproducibility"] = reproducibility
    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "preregistration": preregistration,
        "selection": {
            key: value for key, value in selection.items() if key != "rows"
        },
        "selection_lock": locked_selection,
        "test_dataset_constructed": test_dataset_constructed,
        "test_evaluated_oracle_configuration_count": (
            0 if test_result is None else len(test_result["oracle_methods"])
        ),
        "validation_peak_reserved_mib": validation_peak_reserved,
        "test_peak_reserved_mib": test_peak_reserved,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
        "latency": latency,
        "artifacts": artifacts,
        "decision": artifacts["decision"],
        "reproducibility": reproducibility,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "continuous_residual_oracle_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        decision=artifacts["decision"],
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "selection": manifest["selection"],
                "test_dataset_constructed": test_dataset_constructed,
                "decision": artifacts["decision"],
                "maximum_peak_reserved_mib": maximum_peak_reserved,
                "latency": latency,
                "checks": checks,
                "artifacts": {
                    key: value
                    for key, value in artifacts.items()
                    if key != "decision"
                },
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_train_continuous_router(args: argparse.Namespace) -> int:
    """Train and validation-lock the deployable P1-1 local router."""

    smoke = args.stage == "train-continuous-router-smoke"
    if args.profile not in {"pilot", "expanded"}:
        raise ValueError(
            f"{args.stage} requires --profile pilot or expanded"
        )
    if args.dataset_root is None or args.quantization_root is None:
        raise ValueError(
            f"{args.stage} requires --dataset-root and --quantization-root"
        )
    if args.resume:
        raise ValueError(
            f"{args.stage} is collision-safe and cannot resume"
        )
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    set_manifest_path = dataset_root / "manifest.json"
    quantization_manifest_path = quantization_root / "manifest.json"
    if not set_manifest_path.is_file():
        raise FileNotFoundError(
            f"set dataset manifest not found: {set_manifest_path}"
        )
    if not quantization_manifest_path.is_file():
        raise FileNotFoundError(
            "quantization manifest not found: "
            f"{quantization_manifest_path}"
        )
    set_manifest = json.loads(
        set_manifest_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "set_manifest": set_manifest_path,
        "quantization_manifest": quantization_manifest_path,
        "projection": quantization_root / "projection.pt",
        "codebook": quantization_root / "codebook.pt",
        "train_targets": quantization_root / "train_targets.pt",
        "val_targets": quantization_root / "val_targets.pt",
        "test_targets": quantization_root / "test_targets.pt",
        "router_source": (
            PROJECT_ROOT / "src" / "set2proto" / "continuous_router.py"
        ),
        "run_script": Path(__file__).resolve(),
        "config": args.config.expanduser().resolve(),
        "adaface_checkpoint": Path(
            set_manifest["checkpoint"]["path"]
        ).resolve(),
    }
    missing = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "continuous router source artifact(s) missing: "
            + ", ".join(missing)
        )
    source_hashes_before = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    output_root = Path(config["paths"]["output_dir"]).resolve()
    protected_roots = {
        "stage15": output_root / "stage15-final-report",
        "p0_1": output_root / "p0-condition-robust-pooling-reviewed",
        "p0_2": (
            output_root
            / "p0-2-visibility-reliability-pooling-reviewed"
        ),
        "p0_3": (
            output_root
            / "p0-3-earlier-hook-oracle-reviewed-v4"
        ),
        "p1_0": (
            output_root
            / "p1-0-continuous-residual-oracle-reviewed-v2"
        ),
        "stage13_checkpoint": (
            output_root / "stage13-real-training-pilot-2k"
        ),
    }
    protected_before = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    run_paths = create_run_paths(
        output_root,
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=False,
    )
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=config["logging"]["jsonl_filename"],
        metrics_filename=config["logging"]["metrics_filename"],
        manifest_filename=config["logging"]["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError(f"{args.stage} requires CUDA")
    if (
        device.type == "cuda"
        and config["p1_1"]["precision"] == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("configured P1-1 BF16 is unavailable")

    preregistration = {
        "schema_version": 1,
        "stage": "P1-1-smoke" if smoke else "P1-1",
        "objective": (
            "train a deployable permutation-invariant continuous local "
            "condition router by distilling the P1-0 teacher-guided oracle"
        ),
        "model": {
            "architecture": "DeepSets-style local frame router",
            "inputs": (
                "cached projected condition features, AdaFace quality, "
                "local/global consensus"
            ),
            "frame_index_embedding": False,
            "teacher_used_at_inference": False,
            "maximum_parameters": config["p1_1"]["max_parameters"],
        },
        "supervision": {
            "oracle_method": config["p1_1"]["oracle_method"],
            "teacher_use": "training targets only",
            "losses": {
                "soft_weight_distillation": config["p1_1"][
                    "distillation_weight"
                ],
                "local_teacher_map_cosine": config["p1_1"][
                    "local_map_cosine_weight"
                ],
                "global_identity_cosine": config["p1_1"][
                    "global_identity_weight"
                ],
            },
        },
        "training": {
            key: value
            for key, value in config["p1_1"].items()
            if not key.startswith("selection_")
        },
        "selection_thresholds": {
            key: value
            for key, value in config["p1_1"].items()
            if key.startswith("selection_")
        },
        "selection_split": "val",
        "test_policy": (
            "Smoke never constructs test. Formal training evaluates exactly "
            "one validation-locked checkpoint on test only if every gate "
            "passes."
        ),
        "gallery": "clean_disjoint_teacher_continuous",
        "data_scope": {
            "dataset_root": str(dataset_root),
            "quantization_root": str(quantization_root),
            "pca_refit": False,
            "codebook_refit": False,
            "split_changes": False,
            "backbone_execution": False,
        },
        "project_wide_holdout_caveat": (
            "Test identities were used in earlier P0/P1-0 diagnostics and "
            "are not a fresh confirmatory holdout."
        ),
        "source_hashes": source_hashes_before,
        "config_sha256": config_sha256(config),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
    }
    preregistration_path = (
        run_paths.artifacts / "protocol_preregistration.json"
    )
    preregistration_path.write_text(
        json.dumps(
            preregistration,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preregistration_sha = _file_sha256(preregistration_path)
    logger.log_event(
        "continuous_router_protocol_preregistered",
        path=preregistration_path,
        sha256=preregistration_sha,
        smoke=smoke,
        test_dataset_constructed=False,
    )

    train_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=True,
    )
    val_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="val",
        precompute=True,
    )
    model = build_continuous_router(config)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def metric_callback(
        step: int,
        values: Mapping[str, float],
    ) -> None:
        for name, value in values.items():
            if name == "step":
                continue
            logger.log_metric(
                name=str(name),
                value=float(value),
                step=step,
                split="train",
            )
        if step == 1 or step % 100 == 0:
            logger.log_event(
                "continuous_router_training_progress",
                step=step,
                loss=values["loss"],
                learning_rate=values["learning_rate"],
            )
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "step": step,
                        "loss": values["loss"],
                        "learning_rate": values["learning_rate"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    training = train_continuous_router(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        config=config,
        device=device,
        checkpoint_path_for_step=run_paths.checkpoint_path,
        max_steps_override=(
            int(config["p1_1"]["smoke_steps"]) if smoke else None
        ),
        metric_callback=metric_callback,
    )
    training_peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    checkpoint_path = Path(training["best_checkpoint"]).resolve()
    checkpoint_sha = _file_sha256(checkpoint_path)
    checkpoint_payload = load_router_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )
    validation_result = evaluate_continuous_router_split(
        split="val",
        dataset=val_dataset,
        model=model,
        config=config,
        device=device,
    )
    selection = select_continuous_router(
        validation_result=validation_result,
        config=config,
    )
    if smoke or args.validation_only:
        selection = copy.deepcopy(selection)
        selection["test_authorized"] = False
        selection["selected"] = None
        selection["test_policy_override"] = (
            "test disabled by smoke or --validation-only"
        )
    selection_lock = {
        "schema_version": 1,
        "selection_split": "val",
        "smoke": smoke,
        "validation_only": bool(args.validation_only),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "test_authorized": selection["test_authorized"],
        "selected": selection["selected"],
        "best_validation_checkpoint": selection[
            "best_validation_checkpoint"
        ],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_payload["step"],
        "protocol_preregistration_sha256": preregistration_sha,
        "config_sha256": config_sha256(config),
        "source_hashes": source_hashes_before,
        "teacher_used_at_inference": False,
    }
    selection_lock_path = run_paths.artifacts / "selection_lock.json"
    selection_lock_path.write_text(
        json.dumps(
            selection_lock,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selection_lock_sha_before_test = _file_sha256(selection_lock_path)
    locked = json.loads(
        selection_lock_path.read_text(encoding="utf-8")
    )
    if (
        locked["protocol_preregistration_sha256"] != preregistration_sha
        or locked["checkpoint_sha256"] != checkpoint_sha
        or locked["source_hashes"] != source_hashes_before
    ):
        raise RuntimeError("continuous router selection lock is invalid")
    logger.log_event(
        "continuous_router_validation_checkpoint_locked",
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        test_authorized=locked["test_authorized"],
        test_dataset_constructed=False,
    )

    # Source manifests may contain historical test summaries. Full artifact
    # validation deliberately occurs only after the checkpoint lock exists.
    dataset_validation = validate_set_dataset(dataset_root)
    quantization_validation = validate_quantization_artifacts(
        quantization_root
    )
    test_result = None
    test_dataset_constructed = False
    test_peak_reserved = 0.0
    if bool(locked["test_authorized"]):
        test_dataset = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=quantization_root,
            split="test",
            precompute=True,
        )
        test_dataset_constructed = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        test_result = evaluate_continuous_router_split(
            split="test",
            dataset=test_dataset,
            model=model,
            config=config,
            device=device,
        )
        test_peak_reserved = (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        logger.log_event(
            "continuous_router_locked_test_completed",
            checkpoint=checkpoint_path,
            evaluated_checkpoint_count=1,
        )
    else:
        logger.log_event(
            "continuous_router_test_skipped_by_lock",
            smoke=smoke,
            test_dataset_constructed=False,
        )
    selection_lock_sha_after_test = _file_sha256(selection_lock_path)
    latency = measure_continuous_router_latency(
        dataset=val_dataset,
        model=model,
        config=config,
        device=device,
    )
    latency_peak = max(
        float(value["peak_reserved_mib"])
        for value in latency.values()
    )
    maximum_peak_reserved = max(
        training_peak_reserved,
        test_peak_reserved,
        latency_peak,
    )
    source_hashes_after = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    protected_after = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    test_policy_followed = (
        (
            not test_dataset_constructed
            and test_result is None
            and locked["selected"] is None
        )
        if not bool(locked["test_authorized"])
        else (
            test_dataset_constructed
            and test_result is not None
            and locked["selected"]["method"] == ROUTER_METHOD
        )
    )
    validation_correctness = validation_result["correctness"]
    checks = {
        "dataset_valid_after_lock": bool(dataset_validation["passed"]),
        "quantization_valid_after_lock": bool(
            quantization_validation["passed"]
        ),
        "source_artifacts_unchanged": (
            source_hashes_before == source_hashes_after
        ),
        "protected_prior_outputs_unchanged": (
            protected_before == protected_after
        ),
        "selection_lock_unchanged": (
            selection_lock_sha_before_test
            == selection_lock_sha_after_test
        ),
        "selection_used_validation_only": (
            selection["selection_split"] == "val"
            and not selection["test_metrics_consulted"]
        ),
        "test_policy_followed": test_policy_followed,
        "validation_outputs_correct": all(
            (
                validation_correctness["output_shape_valid"],
                validation_correctness["output_unit_norm_valid"],
                validation_correctness["weight_sum_valid"],
                validation_correctness["permutation_equivalent"],
            )
        ),
        "test_outputs_correct_if_evaluated": (
            True
            if test_result is None
            else all(
                (
                    test_result["correctness"]["output_shape_valid"],
                    test_result["correctness"][
                        "output_unit_norm_valid"
                    ],
                    test_result["correctness"]["weight_sum_valid"],
                    test_result["correctness"][
                        "permutation_equivalent"
                    ],
                )
            )
        ),
        "same_gallery_protocol": (
            validation_result["gallery"]
            == "clean_disjoint_teacher_continuous"
            and (
                test_result is None
                or test_result["gallery"]
                == "clean_disjoint_teacher_continuous"
            )
        ),
        "parameter_cap_respected": (
            int(training["parameter_count"])
            <= int(config["p1_1"]["max_parameters"])
        ),
        "teacher_absent_at_inference": (
            not validation_result["teacher_used_at_inference"]
            and (
                test_result is None
                or not test_result["teacher_used_at_inference"]
            )
        ),
        "checkpoint_reloaded_at_locked_step": (
            int(checkpoint_payload["step"])
            == int(training["best_step"])
        ),
        "training_losses_finite": all(
            all(
                math.isfinite(float(row[key]))
                for key in (
                    "loss",
                    "distillation_loss",
                    "local_map_cosine_loss",
                    "global_identity_loss",
                    "gradient_norm",
                )
            )
            for row in training["history"]
        ),
        "memory_within_configured_cap": (
            maximum_peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
    }
    reproducibility = {
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "protected_snapshots_before": protected_before,
        "protected_snapshots_after": protected_after,
        "protocol_preregistration": str(preregistration_path),
        "protocol_preregistration_sha256": preregistration_sha,
        "selection_lock": str(selection_lock_path),
        "selection_lock_sha256_before_test": (
            selection_lock_sha_before_test
        ),
        "selection_lock_sha256_after_test": (
            selection_lock_sha_after_test
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "training_peak_reserved_mib": training_peak_reserved,
        "test_peak_reserved_mib": test_peak_reserved,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
    }
    artifacts = write_continuous_router_artifacts(
        output_directory=run_paths.artifacts,
        preregistration=preregistration,
        selection=selection,
        training=training,
        validation_result=validation_result,
        test_result=test_result,
        latency=latency,
        config=config,
        checks=checks,
        reproducibility=reproducibility,
    )
    artifacts.update(
        {
            "protocol_preregistration_json": str(
                preregistration_path
            ),
            "selection_lock_json": str(selection_lock_path),
            "best_checkpoint": str(checkpoint_path),
        }
    )
    artifact_checks = {
        f"artifact_{name}_exists": (
            Path(path).is_file() and Path(path).stat().st_size > 0
        )
        for name, path in artifacts.items()
        if isinstance(path, str)
    }
    checks.update(artifact_checks)
    passed = all(checks.values())
    scientific_test_gate = (
        continuous_router_test_gate(
            result=test_result,
            config=config,
        )
        if test_result is not None
        else None
    )
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "smoke": smoke,
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "environment": environment,
        "resolved_config": config,
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "preregistration": preregistration,
        "training": {
            key: value
            for key, value in training.items()
            if key != "history"
        },
        "selection": selection,
        "selection_lock": locked,
        "test_dataset_constructed": test_dataset_constructed,
        "test_evaluated_checkpoint_count": (
            1 if test_result is not None else 0
        ),
        "test_gate": scientific_test_gate,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
        "latency": latency,
        "checks": checks,
        "artifacts": artifacts,
        "decision": artifacts["decision"],
        "reproducibility": reproducibility,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "continuous_router_completed",
        level="info" if passed else "error",
        passed=passed,
        smoke=smoke,
        steps=training["steps_completed"],
        best_step=training["best_step"],
        test_dataset_constructed=test_dataset_constructed,
        decision=artifacts["decision"],
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "smoke": smoke,
                "training": manifest["training"],
                "selection": selection,
                "test_dataset_constructed": test_dataset_constructed,
                "test_gate": scientific_test_gate,
                "decision": artifacts["decision"],
                "maximum_peak_reserved_mib": maximum_peak_reserved,
                "latency": latency,
                "checks": checks,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_train_identity_gated_router(args: argparse.Namespace) -> int:
    """Train P1-2 anchor-residual candidates with train-only identity loss."""

    smoke = args.stage == "train-identity-gated-router-smoke"
    if args.profile not in {"pilot", "expanded"}:
        raise ValueError(
            f"{args.stage} requires --profile pilot or expanded"
        )
    if args.dataset_root is None or args.quantization_root is None:
        raise ValueError(
            f"{args.stage} requires --dataset-root and --quantization-root"
        )
    if args.initial_router_checkpoint is None:
        raise ValueError(
            f"{args.stage} requires --initial-router-checkpoint"
        )
    if args.resume:
        raise ValueError(f"{args.stage} is collision-safe and cannot resume")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    initial_checkpoint = (
        args.initial_router_checkpoint.expanduser().resolve()
    )
    set_manifest_path = dataset_root / "manifest.json"
    quantization_manifest_path = quantization_root / "manifest.json"
    if not set_manifest_path.is_file():
        raise FileNotFoundError(
            f"set dataset manifest not found: {set_manifest_path}"
        )
    if not quantization_manifest_path.is_file():
        raise FileNotFoundError(
            "quantization manifest not found: "
            f"{quantization_manifest_path}"
        )
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"P1-1 checkpoint not found: {initial_checkpoint}"
        )
    set_manifest = json.loads(
        set_manifest_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "set_manifest": set_manifest_path,
        "quantization_manifest": quantization_manifest_path,
        "projection": quantization_root / "projection.pt",
        "codebook": quantization_root / "codebook.pt",
        "train_targets": quantization_root / "train_targets.pt",
        "val_targets": quantization_root / "val_targets.pt",
        "test_targets": quantization_root / "test_targets.pt",
        "p1_1_initial_checkpoint": initial_checkpoint,
        "p1_2_source": (
            PROJECT_ROOT
            / "src"
            / "set2proto"
            / "identity_gated_router.py"
        ),
        "p1_1_source": (
            PROJECT_ROOT / "src" / "set2proto" / "continuous_router.py"
        ),
        "run_script": Path(__file__).resolve(),
        "config": args.config.expanduser().resolve(),
        "adaface_checkpoint": Path(
            set_manifest["checkpoint"]["path"]
        ).resolve(),
    }
    missing = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "identity gated source artifact(s) missing: "
            + ", ".join(missing)
        )
    source_hashes_before = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    output_root = Path(config["paths"]["output_dir"]).resolve()
    protected_roots = {
        "stage15": output_root / "stage15-final-report",
        "p0_1": output_root / "p0-condition-robust-pooling-reviewed",
        "p0_2": (
            output_root
            / "p0-2-visibility-reliability-pooling-reviewed"
        ),
        "p0_3": (
            output_root
            / "p0-3-earlier-hook-oracle-reviewed-v4"
        ),
        "p1_0": (
            output_root
            / "p1-0-continuous-residual-oracle-reviewed-v2"
        ),
        "p1_1": initial_checkpoint.parents[1],
    }
    protected_before = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    run_paths = create_run_paths(
        output_root,
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=False,
    )
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=config["logging"]["jsonl_filename"],
        metrics_filename=config["logging"]["metrics_filename"],
        manifest_filename=config["logging"]["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError(f"{args.stage} requires CUDA")
    if (
        device.type == "cuda"
        and config["p1_2"]["precision"] == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("configured P1-2 BF16 is unavailable")
    recipes = [
        str(value) for value in config["p1_2"]["candidate_recipes"]
    ]
    preregistration = {
        "schema_version": 1,
        "stage": "P1-2-smoke" if smoke else "P1-2",
        "objective": (
            "optimize P1-1 with a quality anchor-residual gate and direct "
            "train-only gallery identity discrimination"
        ),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": source_hashes_before[
            "p1_1_initial_checkpoint"
        ],
        "candidate_recipes": recipes,
        "recipes": config["p1_2"]["recipes"],
        "model": {
            "base": "P1-1 permutation-invariant local router",
            "anchor": "AdaFace feature-norm quality pooling",
            "residual": "normalized gated routed-minus-anchor map",
            "gate": "shared per-position MLP initialized near anchor",
            "frame_index_embedding": False,
            "maximum_parameters": config["p1_2"]["max_parameters"],
        },
        "identity_objective": {
            "gallery": "train-only clean disjoint-teacher continuous",
            "cross_entropy": True,
            "hardest_impostor_margin": config["p1_2"][
                "gallery_margin"
            ],
            "temperature": config["p1_2"]["gallery_temperature"],
            "gallery_used_at_inference": False,
        },
        "teacher_use": (
            "training-only route/gate/map targets; absent at inference"
        ),
        "selection_thresholds": {
            key: value
            for key, value in config["p1_2"].items()
            if key.startswith("selection_")
        },
        "selection_split": "val",
        "test_policy": (
            "Smoke never constructs test. Formal training selects one recipe "
            "and checkpoint on validation. Test is constructed only if all "
            "gates pass and evaluates exactly that lock once."
        ),
        "data_scope": {
            "dataset_root": str(dataset_root),
            "quantization_root": str(quantization_root),
            "pca_refit": False,
            "codebook_refit": False,
            "split_changes": False,
            "backbone_execution": False,
        },
        "project_wide_holdout_caveat": (
            "Current test identities were used in prior diagnostics and are "
            "not a fresh confirmatory holdout."
        ),
        "source_hashes": source_hashes_before,
        "config_sha256": config_sha256(config),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
    }
    preregistration_path = (
        run_paths.artifacts / "protocol_preregistration.json"
    )
    preregistration_path.write_text(
        json.dumps(
            preregistration,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preregistration_sha = _file_sha256(preregistration_path)
    logger.log_event(
        "identity_gated_protocol_preregistered",
        path=preregistration_path,
        sha256=preregistration_sha,
        recipes=recipes,
        smoke=smoke,
        test_dataset_constructed=False,
    )

    train_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=True,
    )
    val_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="val",
        precompute=True,
    )
    gallery = build_train_identity_gallery(train_dataset)
    training_runs: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    checkpoint_by_recipe: dict[str, Path] = {}
    base_validation_result = None
    maximum_training_peak = 0.0

    def metric_callback(
        recipe: str,
        step: int,
        values: Mapping[str, Any],
    ) -> None:
        for name, value in values.items():
            if name in {"recipe", "step"}:
                continue
            logger.log_metric(
                name=str(name),
                value=float(value),
                step=step,
                split="train",
                scenario=recipe,
            )
        if step == 1 or step % 100 == 0:
            logger.log_event(
                "identity_gated_training_progress",
                recipe=recipe,
                step=step,
                loss=values["loss"],
                mean_gate=values["mean_gate"],
            )
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "recipe": recipe,
                        "step": step,
                        "loss": values["loss"],
                        "mean_gate": values["mean_gate"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for recipe_index, recipe in enumerate(recipes):
        seed_everything(
            seed + recipe_index,
            deterministic=bool(config["runtime"]["deterministic"]),
        )
        model = build_identity_gated_router(config)
        base_payload = load_p1_1_router_initialization(
            model=model,
            checkpoint_path=initial_checkpoint,
            device=device,
        )
        if base_validation_result is None:
            base_validation_result = base_payload["validation"]
        candidate_directory = run_paths.checkpoints / recipe
        candidate_directory.mkdir(parents=True, exist_ok=False)

        def candidate_checkpoint_path(
            step: int,
            *,
            directory: Path = candidate_directory,
        ) -> Path:
            return directory / f"checkpoint_step_{step:08d}.pt"

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        training = train_identity_gated_candidate(
            recipe_name=recipe,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            gallery=gallery,
            model=model,
            config=config,
            device=device,
            checkpoint_path_for_step=candidate_checkpoint_path,
            max_steps_override=(
                int(config["p1_2"]["smoke_steps"]) if smoke else None
            ),
            metric_callback=metric_callback,
        )
        maximum_training_peak = max(
            maximum_training_peak,
            (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        )
        checkpoint = Path(training["best_checkpoint"]).resolve()
        checkpoint_by_recipe[recipe] = checkpoint
        load_identity_gated_checkpoint(
            path=checkpoint,
            model=model,
            device=device,
        )
        validation_results[recipe] = (
            evaluate_identity_gated_router_split(
                split="val",
                dataset=val_dataset,
                model=model,
                config=config,
                device=device,
            )
        )
        training_runs[recipe] = training
        logger.log_event(
            "identity_gated_candidate_completed",
            recipe=recipe,
            best_step=training["best_step"],
            checkpoint=checkpoint,
            checkpoint_sha256=_file_sha256(checkpoint),
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert base_validation_result is not None

    selection = select_identity_gated_candidate(
        validation_results=validation_results,
        config=config,
    )
    if smoke or args.validation_only:
        selection = copy.deepcopy(selection)
        selection["test_authorized"] = False
        selection["selected"] = None
        selection["test_policy_override"] = (
            "test disabled by smoke or --validation-only"
        )
    chosen_recipe = str(
        selection["best_validation_candidate"]["recipe"]
    )
    chosen_checkpoint = checkpoint_by_recipe[chosen_recipe]
    chosen_checkpoint_sha = _file_sha256(chosen_checkpoint)
    chosen_model = build_identity_gated_router(config)
    chosen_payload = load_identity_gated_checkpoint(
        path=chosen_checkpoint,
        model=chosen_model,
        device=device,
    )
    chosen_model.to(device)
    selection_lock = {
        "schema_version": 1,
        "selection_split": "val",
        "smoke": smoke,
        "validation_only": bool(args.validation_only),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "test_authorized": selection["test_authorized"],
        "selected": selection["selected"],
        "best_validation_candidate": selection[
            "best_validation_candidate"
        ],
        "chosen_recipe": chosen_recipe,
        "checkpoint": str(chosen_checkpoint),
        "checkpoint_sha256": chosen_checkpoint_sha,
        "checkpoint_step": chosen_payload["step"],
        "candidate_checkpoint_sha256": {
            recipe: _file_sha256(checkpoint)
            for recipe, checkpoint in checkpoint_by_recipe.items()
        },
        "protocol_preregistration_sha256": preregistration_sha,
        "config_sha256": config_sha256(config),
        "source_hashes": source_hashes_before,
        "teacher_used_at_inference": False,
        "gallery_used_at_inference": False,
    }
    selection_lock_path = run_paths.artifacts / "selection_lock.json"
    selection_lock_path.write_text(
        json.dumps(
            selection_lock,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lock_sha_before_test = _file_sha256(selection_lock_path)
    locked = json.loads(
        selection_lock_path.read_text(encoding="utf-8")
    )
    if (
        locked["protocol_preregistration_sha256"] != preregistration_sha
        or locked["checkpoint_sha256"] != chosen_checkpoint_sha
        or locked["source_hashes"] != source_hashes_before
    ):
        raise RuntimeError("identity gated selection lock is invalid")
    logger.log_event(
        "identity_gated_validation_selection_locked",
        recipe=chosen_recipe,
        checkpoint=chosen_checkpoint,
        test_authorized=locked["test_authorized"],
        test_dataset_constructed=False,
    )

    dataset_validation = validate_set_dataset(dataset_root)
    quantization_validation = validate_quantization_artifacts(
        quantization_root
    )
    test_result = None
    test_dataset_constructed = False
    test_peak = 0.0
    if bool(locked["test_authorized"]):
        if locked["selected"]["recipe"] != chosen_recipe:
            raise RuntimeError("locked recipe/checkpoint mismatch")
        test_dataset = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=quantization_root,
            split="test",
            precompute=True,
        )
        test_dataset_constructed = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        test_result = evaluate_identity_gated_router_split(
            split="test",
            dataset=test_dataset,
            model=chosen_model,
            config=config,
            device=device,
        )
        test_peak = (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        logger.log_event(
            "identity_gated_locked_test_completed",
            recipe=chosen_recipe,
            evaluated_checkpoint_count=1,
        )
    else:
        logger.log_event(
            "identity_gated_test_skipped_by_lock",
            smoke=smoke,
            test_dataset_constructed=False,
        )
    lock_sha_after_test = _file_sha256(selection_lock_path)
    latency = measure_continuous_router_latency(
        dataset=val_dataset,
        model=chosen_model,
        config=config,
        device=device,
        method_name=GATED_ROUTER_METHOD,
    )
    maximum_peak_reserved = max(
        maximum_training_peak,
        test_peak,
        max(
            float(value["peak_reserved_mib"])
            for value in latency.values()
        ),
    )
    source_hashes_after = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    protected_after = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    selected_validation = validation_results[chosen_recipe]
    val_correctness = selected_validation["correctness"]
    test_policy_followed = (
        (
            test_dataset_constructed
            and test_result is not None
            and locked["selected"]["recipe"] == chosen_recipe
        )
        if bool(locked["test_authorized"])
        else (
            not test_dataset_constructed
            and test_result is None
            and locked["selected"] is None
        )
    )
    checks = {
        "dataset_valid_after_lock": bool(dataset_validation["passed"]),
        "quantization_valid_after_lock": bool(
            quantization_validation["passed"]
        ),
        "source_artifacts_unchanged": (
            source_hashes_before == source_hashes_after
        ),
        "protected_prior_outputs_unchanged": (
            protected_before == protected_after
        ),
        "selection_lock_unchanged": (
            lock_sha_before_test == lock_sha_after_test
        ),
        "selection_used_validation_only": (
            selection["selection_split"] == "val"
            and not selection["test_metrics_consulted"]
        ),
        "candidate_recipes_all_trained": (
            set(training_runs) == set(recipes)
            and set(validation_results) == set(recipes)
        ),
        "train_gallery_is_train_only_and_complete": (
            gallery.size
            == int(
                set_manifest["splits"]["train"]["identities"]
            )
        ),
        "test_policy_followed": test_policy_followed,
        "validation_outputs_correct": all(
            (
                val_correctness["output_shape_valid"],
                val_correctness["output_unit_norm_valid"],
                val_correctness["weight_sum_valid"],
                val_correctness["permutation_equivalent"],
            )
        ),
        "test_outputs_correct_if_evaluated": (
            True
            if test_result is None
            else all(
                (
                    test_result["correctness"]["output_shape_valid"],
                    test_result["correctness"][
                        "output_unit_norm_valid"
                    ],
                    test_result["correctness"]["weight_sum_valid"],
                    test_result["correctness"][
                        "permutation_equivalent"
                    ],
                )
            )
        ),
        "same_gallery_protocol": (
            selected_validation["gallery"]
            == "clean_disjoint_teacher_continuous"
            and (
                test_result is None
                or test_result["gallery"]
                == "clean_disjoint_teacher_continuous"
            )
        ),
        "parameter_cap_respected": all(
            int(training["parameter_count"])
            <= int(config["p1_2"]["max_parameters"])
            for training in training_runs.values()
        ),
        "teacher_and_train_gallery_absent_at_inference": (
            not selected_validation["teacher_used_at_inference"]
            and not selected_validation["gallery_used_at_inference"]
            and (
                test_result is None
                or (
                    not test_result["teacher_used_at_inference"]
                    and not test_result["gallery_used_at_inference"]
                )
            )
        ),
        "checkpoint_reloaded_at_locked_step": (
            int(chosen_payload["step"])
            == int(training_runs[chosen_recipe]["best_step"])
        ),
        "training_losses_finite": all(
            all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key not in {"recipe", "step"}
            )
            for training in training_runs.values()
            for row in training["history"]
        ),
        "memory_within_configured_cap": (
            maximum_peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
    }
    reproducibility = {
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "initial_router_checkpoint": str(initial_checkpoint),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "protected_snapshots_before": protected_before,
        "protected_snapshots_after": protected_after,
        "protocol_preregistration": str(preregistration_path),
        "protocol_preregistration_sha256": preregistration_sha,
        "selection_lock": str(selection_lock_path),
        "selection_lock_sha256_before_test": lock_sha_before_test,
        "selection_lock_sha256_after_test": lock_sha_after_test,
        "chosen_checkpoint": str(chosen_checkpoint),
        "chosen_checkpoint_sha256": chosen_checkpoint_sha,
        "train_gallery_size": gallery.size,
        "train_gallery_identity_min": int(gallery.identities.min().item()),
        "train_gallery_identity_max": int(gallery.identities.max().item()),
        "maximum_training_peak_reserved_mib": maximum_training_peak,
        "test_peak_reserved_mib": test_peak,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
    }
    artifacts = write_identity_gated_artifacts(
        output_directory=run_paths.artifacts,
        preregistration=preregistration,
        selection=selection,
        training_runs=training_runs,
        validation_results=validation_results,
        selected_validation_result=selected_validation,
        base_validation_result=base_validation_result,
        test_result=test_result,
        latency=latency,
        config=config,
        checks=checks,
        reproducibility=reproducibility,
    )
    artifacts.update(
        {
            "protocol_preregistration_json": str(
                preregistration_path
            ),
            "selection_lock_json": str(selection_lock_path),
            "chosen_checkpoint": str(chosen_checkpoint),
        }
    )
    artifact_checks = {
        f"artifact_{name}_exists": (
            Path(path).is_file() and Path(path).stat().st_size > 0
        )
        for name, path in artifacts.items()
        if isinstance(path, str)
    }
    checks.update(artifact_checks)
    passed = all(checks.values())
    scientific_test_gate = (
        identity_gated_test_gate(
            recipe_name=chosen_recipe,
            result=test_result,
            config=config,
        )
        if test_result is not None
        else None
    )
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "smoke": smoke,
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "environment": environment,
        "resolved_config": config,
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "initial_router_checkpoint": str(initial_checkpoint),
        "train_gallery": {
            "split": "train",
            "size": gallery.size,
            "source": "clean_disjoint_teacher_continuous",
            "used_at_inference": False,
        },
        "training_runs": {
            recipe: {
                key: value
                for key, value in training.items()
                if key != "history"
            }
            for recipe, training in training_runs.items()
        },
        "selection": selection,
        "selection_lock": locked,
        "test_dataset_constructed": test_dataset_constructed,
        "test_evaluated_checkpoint_count": (
            1 if test_result is not None else 0
        ),
        "test_gate": scientific_test_gate,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
        "latency": latency,
        "checks": checks,
        "artifacts": artifacts,
        "decision": artifacts["decision"],
        "reproducibility": reproducibility,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "identity_gated_router_completed",
        level="info" if passed else "error",
        passed=passed,
        smoke=smoke,
        chosen_recipe=chosen_recipe,
        test_dataset_constructed=test_dataset_constructed,
        decision=artifacts["decision"],
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "smoke": smoke,
                "training_runs": manifest["training_runs"],
                "selection": selection,
                "test_dataset_constructed": test_dataset_constructed,
                "test_gate": scientific_test_gate,
                "decision": artifacts["decision"],
                "maximum_peak_reserved_mib": maximum_peak_reserved,
                "latency": latency,
                "checks": checks,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_train_scalar_evidence_router(args: argparse.Namespace) -> int:
    """Train P1-3 bounded identity-agnostic evidence gates."""

    smoke = args.stage == "train-scalar-evidence-router-smoke"
    if args.profile not in {"pilot", "expanded"}:
        raise ValueError(
            f"{args.stage} requires --profile pilot or expanded"
        )
    if args.dataset_root is None or args.quantization_root is None:
        raise ValueError(
            f"{args.stage} requires --dataset-root and --quantization-root"
        )
    if args.initial_router_checkpoint is None:
        raise ValueError(
            f"{args.stage} requires --initial-router-checkpoint"
        )
    if args.resume:
        raise ValueError(f"{args.stage} is collision-safe and cannot resume")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    dataset_root = args.dataset_root.expanduser().resolve()
    quantization_root = args.quantization_root.expanduser().resolve()
    initial_checkpoint = (
        args.initial_router_checkpoint.expanduser().resolve()
    )
    set_manifest_path = dataset_root / "manifest.json"
    quantization_manifest_path = quantization_root / "manifest.json"
    if not set_manifest_path.is_file():
        raise FileNotFoundError(
            f"set dataset manifest not found: {set_manifest_path}"
        )
    if not quantization_manifest_path.is_file():
        raise FileNotFoundError(
            f"quantization manifest not found: {quantization_manifest_path}"
        )
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"P1-2 checkpoint not found: {initial_checkpoint}"
        )
    set_manifest = json.loads(
        set_manifest_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "set_manifest": set_manifest_path,
        "quantization_manifest": quantization_manifest_path,
        "projection": quantization_root / "projection.pt",
        "codebook": quantization_root / "codebook.pt",
        "train_targets": quantization_root / "train_targets.pt",
        "val_targets": quantization_root / "val_targets.pt",
        "test_targets": quantization_root / "test_targets.pt",
        "p1_2_initial_checkpoint": initial_checkpoint,
        "p1_3_source": (
            PROJECT_ROOT
            / "src"
            / "set2proto"
            / "scalar_evidence_router.py"
        ),
        "run_script": Path(__file__).resolve(),
        "config": args.config.expanduser().resolve(),
        "adaface_checkpoint": Path(
            set_manifest["checkpoint"]["path"]
        ).resolve(),
    }
    missing = [
        str(path) for path in source_paths.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "P1-3 source artifact(s) missing: " + ", ".join(missing)
        )
    source_hashes_before = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    output_root = Path(config["paths"]["output_dir"]).resolve()
    protected_roots = {
        "stage15": output_root / "stage15-final-report",
        "p1_0": (
            output_root
            / "p1-0-continuous-residual-oracle-reviewed-v2"
        ),
        "p1_1": output_root / "p1-1-continuous-router-pilot-reviewed",
        "p1_2": initial_checkpoint.parents[2],
    }
    protected_before = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    run_paths = create_run_paths(
        output_root,
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=False,
    )
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=config["logging"]["jsonl_filename"],
        metrics_filename=config["logging"]["metrics_filename"],
        manifest_filename=config["logging"]["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError(f"{args.stage} requires CUDA")
    recipes = [
        str(value) for value in config["p1_3"]["candidate_recipes"]
    ]
    preregistration = {
        "schema_version": 1,
        "stage": "P1-3-smoke" if smoke else "P1-3",
        "objective": (
            "replace identity-vector gate inputs with bounded identity-"
            "agnostic scalar evidence and anchor-relative gallery margin"
        ),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": source_hashes_before[
            "p1_2_initial_checkpoint"
        ],
        "candidate_recipes": recipes,
        "recipes": config["p1_3"]["recipes"],
        "gate": {
            "input": (
                "13 identity-agnostic route confidence, entropy, quality, "
                "consensus and anchor-disagreement scalars"
            ),
            "initial_value": config["p1_3"]["gate_initial_value"],
            "maximum": config["p1_3"]["maximum_gate"],
            "raw_identity_vectors": False,
        },
        "identity_objective": (
            "train-only clean gallery; improve margin relative to each "
            "sample's quality anchor rather than memorize absolute identity"
        ),
        "selection_thresholds": {
            key: value
            for key, value in config["p1_3"].items()
            if key.startswith("selection_")
        },
        "selection_split": "val",
        "test_policy": (
            "Smoke never constructs test. Formal test evaluates exactly one "
            "validation-locked candidate only if every gate passes."
        ),
        "source_hashes": source_hashes_before,
        "config_sha256": config_sha256(config),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "project_wide_holdout_caveat": (
            "Current test identities are not a fresh confirmatory holdout."
        ),
    }
    preregistration_path = (
        run_paths.artifacts / "protocol_preregistration.json"
    )
    preregistration_path.write_text(
        json.dumps(
            preregistration,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preregistration_sha = _file_sha256(preregistration_path)
    logger.log_event(
        "scalar_evidence_protocol_preregistered",
        recipes=recipes,
        smoke=smoke,
        sha256=preregistration_sha,
        test_dataset_constructed=False,
    )

    train_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="train",
        precompute=True,
    )
    val_dataset = TokenTrainingDataset(
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        split="val",
        precompute=True,
    )
    gallery = build_train_identity_gallery(train_dataset)
    training_runs: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    checkpoints: dict[str, Path] = {}
    p1_2_validation = None
    maximum_training_peak = 0.0

    def metric_callback(
        recipe: str,
        step: int,
        values: Mapping[str, Any],
    ) -> None:
        for name, value in values.items():
            if name in {"recipe", "step"}:
                continue
            logger.log_metric(
                name=str(name),
                value=float(value),
                step=step,
                split="train",
                scenario=recipe,
            )
        if step == 1 or step % 50 == 0:
            logger.log_event(
                "scalar_evidence_training_progress",
                recipe=recipe,
                step=step,
                loss=values["loss"],
                mean_gate=values["mean_gate"],
                gate_std=values["gate_std"],
            )
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "recipe": recipe,
                        "step": step,
                        "loss": values["loss"],
                        "mean_gate": values["mean_gate"],
                        "gate_std": values["gate_std"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for recipe_index, recipe in enumerate(recipes):
        seed_everything(
            seed + recipe_index,
            deterministic=bool(config["runtime"]["deterministic"]),
        )
        model = build_scalar_evidence_router(config)
        initial_payload = load_scalar_router_initialization(
            model=model,
            checkpoint_path=initial_checkpoint,
            device=device,
        )
        if p1_2_validation is None:
            p1_2_validation = initial_payload["validation"]
        candidate_directory = run_paths.checkpoints / recipe
        candidate_directory.mkdir(parents=True, exist_ok=False)

        def candidate_checkpoint(
            step: int,
            *,
            directory: Path = candidate_directory,
        ) -> Path:
            return directory / f"checkpoint_step_{step:08d}.pt"

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        training = train_scalar_evidence_candidate(
            recipe_name=recipe,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            gallery=gallery,
            model=model,
            config=config,
            device=device,
            checkpoint_path_for_step=candidate_checkpoint,
            max_steps_override=(
                int(config["p1_3"]["smoke_steps"]) if smoke else None
            ),
            metric_callback=metric_callback,
        )
        maximum_training_peak = max(
            maximum_training_peak,
            (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        )
        checkpoint = Path(training["best_checkpoint"]).resolve()
        checkpoints[recipe] = checkpoint
        load_scalar_evidence_checkpoint(
            path=checkpoint,
            model=model,
            device=device,
        )
        validation_results[recipe] = evaluate_scalar_evidence_split(
            split="val",
            dataset=val_dataset,
            model=model,
            config=config,
            device=device,
        )
        training_runs[recipe] = training
        logger.log_event(
            "scalar_evidence_candidate_completed",
            recipe=recipe,
            best_step=training["best_step"],
            checkpoint_sha256=_file_sha256(checkpoint),
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert p1_2_validation is not None

    selection = select_scalar_evidence_candidate(
        validation_results=validation_results,
        config=config,
    )
    if smoke or args.validation_only:
        selection = copy.deepcopy(selection)
        selection["test_authorized"] = False
        selection["selected"] = None
        selection["test_policy_override"] = (
            "test disabled by smoke or --validation-only"
        )
    chosen_recipe = str(
        selection["best_validation_candidate"]["recipe"]
    )
    chosen_checkpoint = checkpoints[chosen_recipe]
    chosen_checkpoint_sha = _file_sha256(chosen_checkpoint)
    chosen_model = build_scalar_evidence_router(config)
    chosen_payload = load_scalar_evidence_checkpoint(
        path=chosen_checkpoint,
        model=chosen_model,
        device=device,
    )
    chosen_model.to(device)
    selection_lock = {
        "schema_version": 1,
        "selection_split": "val",
        "smoke": smoke,
        "validation_only": bool(args.validation_only),
        "test_metrics_consulted": False,
        "test_token_dataset_constructed": False,
        "test_authorized": selection["test_authorized"],
        "selected": selection["selected"],
        "best_validation_candidate": selection[
            "best_validation_candidate"
        ],
        "chosen_recipe": chosen_recipe,
        "checkpoint": str(chosen_checkpoint),
        "checkpoint_sha256": chosen_checkpoint_sha,
        "checkpoint_step": chosen_payload["step"],
        "candidate_checkpoint_sha256": {
            recipe: _file_sha256(path)
            for recipe, path in checkpoints.items()
        },
        "protocol_preregistration_sha256": preregistration_sha,
        "config_sha256": config_sha256(config),
        "source_hashes": source_hashes_before,
        "teacher_used_at_inference": False,
        "gallery_used_at_inference": False,
    }
    selection_lock_path = run_paths.artifacts / "selection_lock.json"
    selection_lock_path.write_text(
        json.dumps(
            selection_lock,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lock_sha_before_test = _file_sha256(selection_lock_path)
    locked = json.loads(
        selection_lock_path.read_text(encoding="utf-8")
    )
    if (
        locked["checkpoint_sha256"] != chosen_checkpoint_sha
        or locked["source_hashes"] != source_hashes_before
    ):
        raise RuntimeError("P1-3 selection lock is invalid")
    logger.log_event(
        "scalar_evidence_validation_selection_locked",
        recipe=chosen_recipe,
        test_authorized=locked["test_authorized"],
        test_dataset_constructed=False,
    )

    dataset_validation = validate_set_dataset(dataset_root)
    quantization_validation = validate_quantization_artifacts(
        quantization_root
    )
    test_result = None
    test_dataset_constructed = False
    test_peak = 0.0
    if bool(locked["test_authorized"]):
        test_dataset = TokenTrainingDataset(
            dataset_root=dataset_root,
            quantization_root=quantization_root,
            split="test",
            precompute=True,
        )
        test_dataset_constructed = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        test_result = evaluate_scalar_evidence_split(
            split="test",
            dataset=test_dataset,
            model=chosen_model,
            config=config,
            device=device,
        )
        test_peak = (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        logger.log_event(
            "scalar_evidence_locked_test_completed",
            recipe=chosen_recipe,
            evaluated_checkpoint_count=1,
        )
    else:
        logger.log_event(
            "scalar_evidence_test_skipped_by_lock",
            test_dataset_constructed=False,
        )
    lock_sha_after_test = _file_sha256(selection_lock_path)
    latency = measure_continuous_router_latency(
        dataset=val_dataset,
        model=chosen_model,
        config=config,
        device=device,
        method_name=SCALAR_EVIDENCE_METHOD,
    )
    maximum_peak_reserved = max(
        maximum_training_peak,
        test_peak,
        max(
            float(value["peak_reserved_mib"])
            for value in latency.values()
        ),
    )
    source_hashes_after = {
        name: _file_sha256(path) for name, path in source_paths.items()
    }
    protected_after = {
        name: _tree_hash_snapshot(path)
        for name, path in protected_roots.items()
    }
    selected_validation = validation_results[chosen_recipe]
    correctness = selected_validation["correctness"]
    test_policy_followed = (
        (
            test_dataset_constructed
            and test_result is not None
            and locked["selected"]["recipe"] == chosen_recipe
        )
        if bool(locked["test_authorized"])
        else (
            not test_dataset_constructed
            and test_result is None
            and locked["selected"] is None
        )
    )
    checks = {
        "dataset_valid_after_lock": bool(dataset_validation["passed"]),
        "quantization_valid_after_lock": bool(
            quantization_validation["passed"]
        ),
        "source_artifacts_unchanged": (
            source_hashes_before == source_hashes_after
        ),
        "protected_prior_outputs_unchanged": (
            protected_before == protected_after
        ),
        "selection_lock_unchanged": (
            lock_sha_before_test == lock_sha_after_test
        ),
        "selection_used_validation_only": (
            selection["selection_split"] == "val"
            and not selection["test_metrics_consulted"]
        ),
        "candidate_recipes_all_trained": (
            set(training_runs) == set(recipes)
        ),
        "train_gallery_complete": (
            gallery.size
            == int(set_manifest["splits"]["train"]["identities"])
        ),
        "test_policy_followed": test_policy_followed,
        "validation_outputs_correct": all(
            (
                correctness["output_shape_valid"],
                correctness["output_unit_norm_valid"],
                correctness["weight_sum_valid"],
                correctness["permutation_equivalent"],
            )
        ),
        "test_outputs_correct_if_evaluated": (
            True
            if test_result is None
            else all(
                (
                    test_result["correctness"]["output_shape_valid"],
                    test_result["correctness"][
                        "output_unit_norm_valid"
                    ],
                    test_result["correctness"]["weight_sum_valid"],
                    test_result["correctness"][
                        "permutation_equivalent"
                    ],
                )
            )
        ),
        "gate_bound_respected": all(
            float(result["gate_diagnostics"]["all"]["mean_gate"])
            <= float(config["p1_3"]["maximum_gate"])
            for result in validation_results.values()
        ),
        "identity_vectors_absent_from_gate": all(
            not result["gate_identity_vector_input"]
            for result in validation_results.values()
        ),
        "parameter_cap_respected": all(
            int(run["parameter_count"])
            <= int(config["p1_3"]["max_parameters"])
            for run in training_runs.values()
        ),
        "teacher_and_gallery_absent_at_inference": (
            not selected_validation["teacher_used_at_inference"]
            and not selected_validation["gallery_used_at_inference"]
        ),
        "checkpoint_reloaded_at_locked_step": (
            int(chosen_payload["step"])
            == int(training_runs[chosen_recipe]["best_step"])
        ),
        "training_losses_finite": all(
            all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key not in {"recipe", "step"}
            )
            for run in training_runs.values()
            for row in run["history"]
        ),
        "memory_within_configured_cap": (
            maximum_peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
    }
    reproducibility = {
        "dataset_root": str(dataset_root),
        "quantization_root": str(quantization_root),
        "initial_checkpoint": str(initial_checkpoint),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "preregistration_sha256": preregistration_sha,
        "selection_lock_sha256_before_test": lock_sha_before_test,
        "selection_lock_sha256_after_test": lock_sha_after_test,
        "chosen_checkpoint": str(chosen_checkpoint),
        "chosen_checkpoint_sha256": chosen_checkpoint_sha,
        "train_gallery_size": gallery.size,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
    }
    artifacts = write_scalar_evidence_artifacts(
        output_directory=run_paths.artifacts,
        preregistration=preregistration,
        selection=selection,
        training_runs=training_runs,
        validation_results=validation_results,
        selected_validation=selected_validation,
        p1_2_validation=p1_2_validation,
        test_result=test_result,
        latency=latency,
        config=config,
        checks=checks,
        reproducibility=reproducibility,
    )
    artifacts.update(
        {
            "protocol_preregistration_json": str(
                preregistration_path
            ),
            "selection_lock_json": str(selection_lock_path),
            "chosen_checkpoint": str(chosen_checkpoint),
        }
    )
    checks.update(
        {
            f"artifact_{name}_exists": (
                Path(path).is_file() and Path(path).stat().st_size > 0
            )
            for name, path in artifacts.items()
            if isinstance(path, str)
        }
    )
    passed = all(checks.values())
    scientific_test_gate = (
        scalar_evidence_test_gate(
            recipe_name=chosen_recipe,
            result=test_result,
            config=config,
        )
        if test_result is not None
        else None
    )
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "smoke": smoke,
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "environment": environment,
        "resolved_config": config,
        "initial_checkpoint": str(initial_checkpoint),
        "training_runs": {
            recipe: {
                key: value
                for key, value in run.items()
                if key != "history"
            }
            for recipe, run in training_runs.items()
        },
        "selection": selection,
        "selection_lock": locked,
        "test_dataset_constructed": test_dataset_constructed,
        "test_gate": scientific_test_gate,
        "maximum_peak_reserved_mib": maximum_peak_reserved,
        "latency": latency,
        "checks": checks,
        "artifacts": artifacts,
        "decision": artifacts["decision"],
        "reproducibility": reproducibility,
    }
    logger.write_manifest(manifest)
    logger.log_event(
        "scalar_evidence_router_completed",
        level="info" if passed else "error",
        passed=passed,
        chosen_recipe=chosen_recipe,
        test_dataset_constructed=test_dataset_constructed,
        decision=artifacts["decision"],
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "smoke": smoke,
                "training_runs": manifest["training_runs"],
                "selection": selection,
                "test_dataset_constructed": test_dataset_constructed,
                "test_gate": scientific_test_gate,
                "decision": artifacts["decision"],
                "maximum_peak_reserved_mib": maximum_peak_reserved,
                "latency": latency,
                "checks": checks,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def run_evaluate_diagnostics(args: argparse.Namespace) -> int:
    if args.profile != "pilot":
        raise ValueError("evaluate-diagnostics requires --profile pilot")
    if args.training_run is None:
        raise ValueError("evaluate-diagnostics requires --training-run")
    training_run = args.training_run.expanduser().resolve()
    training_manifest_path = training_run / "logs" / "run_manifest.json"
    if not training_manifest_path.is_file():
        raise FileNotFoundError(
            f"training run manifest not found: {training_manifest_path}"
        )
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    if training_manifest.get("status") != "passed":
        raise ValueError("diagnostics require a passed training run")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(config["runtime"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("evaluate-diagnostics requires CUDA")
    one_shot_checkpoint = Path(
        training_manifest["training"]["one-shot"]["checkpoint"]
    ).expanduser().resolve()
    maskgit_checkpoint = Path(
        training_manifest["training"]["maskgit"]["checkpoint"]
    ).expanduser().resolve()
    dataset_root = Path(
        training_manifest["dataset_root"]
    ).expanduser().resolve()
    quantization_root = Path(
        training_manifest["quantization_root"]
    ).expanduser().resolve()
    logger.log_event(
        "post_training_diagnostics_started",
        training_run=training_run,
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        device=str(device),
    )
    diagnostics = evaluate_post_training_diagnostics(
        config=config,
        dataset_root=dataset_root,
        quantization_root=quantization_root,
        one_shot_checkpoint=one_shot_checkpoint,
        maskgit_checkpoint=maskgit_checkpoint,
        device=device,
        output_directory=run_paths.artifacts,
    )
    maximum_peak_reserved = max(
        float(value["peak_reserved_mib"])
        for value in diagnostics["latency"].values()
    )
    checks = {
        "diagnostics_json_exists": Path(
            diagnostics["diagnostics_json"]
        ).is_file(),
        "failure_cases_csv_exists": Path(
            diagnostics["failure_cases_csv"]
        ).is_file(),
        "all_required_latency_modes_present": {
            "mean_pooling",
            "quality_pooling",
            "one_shot_transformer",
            "maskgit_confidence_1step",
            "maskgit_confidence_2step",
            "maskgit_confidence_4step",
            "maskgit_confidence_8step",
            "maskgit_evidence_ordering",
            "maskgit_evidence_logits",
            "maskgit_evidence_remask",
        }.issubset(diagnostics["latency"]),
        "memory_within_configured_cap": (
            maximum_peak_reserved
            <= float(config["runtime"]["max_peak_memory_gib"]) * 1024.0
        ),
        "all_test_samples_diagnosed": (
            int(diagnostics["samples"]) == 250
        ),
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "source_training_run": str(training_run),
        "diagnostics": diagnostics,
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "post_training_diagnostics_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        maximum_peak_reserved_mib=maximum_peak_reserved,
    )
    for method, values in diagnostics["latency"].items():
        logger.log_metric(
            name="inference_latency",
            value=float(values["per_set_latency_ms"]),
            step=int(config["training"]["max_steps"]),
            split="test",
            scenario=method,
            unit="ms/set",
        )
        logger.log_metric(
            name="inference_peak_reserved_memory",
            value=float(values["peak_reserved_mib"]),
            step=int(config["training"]["max_steps"]),
            split="test",
            scenario=method,
            unit="MiB",
        )
    compact = {
        "status": manifest["status"],
        "run_directory": str(run_paths.root),
        "checks": checks,
        "latency_ms_per_set": {
            method: values["per_set_latency_ms"]
            for method, values in diagnostics["latency"].items()
        },
        "commit_visibility_rank_correlation": diagnostics[
            "commit_visibility_rank_correlation"
        ],
        "failure_cases_csv": diagnostics["failure_cases_csv"],
        "maximum_peak_reserved_mib": maximum_peak_reserved,
        "git": environment["git"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def run_build_report(args: argparse.Namespace) -> int:
    if args.profile != "pilot":
        raise ValueError("build-report requires --profile pilot")
    if args.training_run is None:
        raise ValueError("build-report requires --training-run")
    if args.diagnostics_run is None:
        raise ValueError("build-report requires --diagnostics-run")
    resolved = load_config(args.config, args.profile)
    config = _apply_cli_overrides(
        resolved,
        seed=args.seed,
        output_root=args.output_root,
    )
    seed = int(config["project"]["seed"])
    ensure_workspace_directories(config["paths"])
    run_paths = create_run_paths(
        config["paths"]["output_dir"],
        profile=args.profile,
        seed=seed,
        run_id=args.run_id,
        resume=args.resume,
    )
    log_config = config["logging"]
    logger = ExperimentLogger(
        run_paths.logs,
        jsonl_filename=log_config["jsonl_filename"],
        metrics_filename=log_config["metrics_filename"],
        manifest_filename=log_config["manifest_filename"],
    )
    seed_state = seed_everything(
        seed,
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    environment = collect_environment(
        project_root=resolved.project_root,
        config=config,
        command=[sys.executable, *sys.argv],
    )
    training_run = args.training_run.expanduser().resolve()
    diagnostics_run = args.diagnostics_run.expanduser().resolve()
    logger.log_event(
        "final_report_started",
        training_run=training_run,
        diagnostics_run=diagnostics_run,
    )
    report = build_final_report(
        training_run=training_run,
        diagnostics_run=diagnostics_run,
        output_directory=run_paths.artifacts,
    )
    expected_files = [
        Path(report["report"]),
        Path(report["decision"]),
        *[Path(value) for value in report["tables"].values()],
        *[Path(value) for value in report["figures"].values()],
    ]
    checks = {
        "all_report_files_exist": all(path.is_file() for path in expected_files),
        "all_report_files_nonempty": all(
            path.stat().st_size > 0 for path in expected_files
        ),
        "go_no_go_decision_recorded": bool(
            report["go_no_go"].get("decision")
        ),
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": 1,
        "project": config["project"]["name"],
        "stage": args.stage,
        "profile": args.profile,
        "run_id": run_paths.root.name,
        "run_directory": str(run_paths.root),
        "status": "passed" if passed else "failed",
        "seed_state": {
            "seed": seed_state.seed,
            "deterministic": seed_state.deterministic,
            "python_hash_seed": seed_state.python_hash_seed,
        },
        "checks": checks,
        "environment": environment,
        "resolved_config": config,
        "source_training_run": str(training_run),
        "source_diagnostics_run": str(diagnostics_run),
        "report": report,
    }
    logger.write_manifest(manifest, overwrite=args.resume)
    logger.log_event(
        "final_report_completed",
        level="info" if passed else "error",
        passed=passed,
        checks=checks,
        decision=report["go_no_go"]["decision"],
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(run_paths.root),
                "checks": checks,
                **report,
                "git": environment["git"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.stage == "env-check":
            return run_env_check(args)
        if args.stage == "prepare-real":
            return run_prepare_real(args)
        if args.stage == "backbone-smoke":
            return run_backbone_smoke(args)
        if args.stage == "cache-real-features":
            return run_cache_real_features(args)
        if args.stage == "prepare-real-sets":
            return run_prepare_real_sets(args)
        if args.stage in {"prepare-synthetic", "synthetic-smoke"}:
            return run_prepare_synthetic(args)
        if args.stage == "fit-codebook":
            return run_fit_codebook(args)
        if args.stage in {
            "train-smoke",
            "train-real-smoke",
            "train-pilot",
        }:
            return run_train_smoke(args)
        if args.stage == "evaluate-diagnostics":
            return run_evaluate_diagnostics(args)
        if args.stage == "evaluate-condition-pooling":
            return run_evaluate_condition_pooling(args)
        if args.stage == "evaluate-visibility-aggregation":
            return run_evaluate_visibility_aggregation(args)
        if args.stage == "evaluate-earlier-hook-oracle":
            return run_evaluate_earlier_hook_oracle(args)
        if args.stage == "evaluate-continuous-residual-oracle":
            return run_evaluate_continuous_residual_oracle(args)
        if args.stage in {
            "train-continuous-router-smoke",
            "train-continuous-router",
        }:
            return run_train_continuous_router(args)
        if args.stage in {
            "train-identity-gated-router-smoke",
            "train-identity-gated-router",
        }:
            return run_train_identity_gated_router(args)
        if args.stage in {
            "train-scalar-evidence-router-smoke",
            "train-scalar-evidence-router",
        }:
            return run_train_scalar_evidence_router(args)
        if args.stage == "build-report":
            return run_build_report(args)
        parser.error(f"stage is not implemented: {args.stage}")
    except (ConfigError, FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
