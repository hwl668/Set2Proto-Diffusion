"""One-shot PointerRoute targets, loss, training, and evaluation."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .pointer_route_model import PointerRouteTransformer
from .pointer_route_oracle import (
    HARD_SCENARIOS,
    build_route_candidates,
    hard_route_targets,
    map_similarity_metrics,
    margin_adjusted_utilities,
    reconstruct_hard_routes,
    reconstruct_soft_routes,
    route_diagnostics,
    soft_route_targets,
    teacher_route_scores,
)
from .research_evaluation import _evaluate_embeddings
from .training import TokenTrainingDataset


def _require_precomputed(dataset: TokenTrainingDataset) -> dict[str, Any]:
    values = dataset._precomputed
    if values is None:
        raise ValueError("PointerRoute training requires precomputed conditions")
    return values


@torch.inference_mode()
def prepare_route_targets(
    *,
    dataset: TokenTrainingDataset,
    delta: float,
    temperature: float,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Construct fixed targets for the unchanged source train samples."""

    precomputed = _require_precomputed(dataset)
    hard_parts: list[torch.Tensor] = []
    soft_parts: list[torch.Tensor] = []
    score_parts: list[torch.Tensor] = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        condition = precomputed["condition_features"][start:end].to(device)
        quality = precomputed["condition_quality"][start:end].to(device)
        teacher = dataset.targets["continuous_prototypes"][start:end].to(device)
        candidates = build_route_candidates(condition, quality)
        scores = teacher_route_scores(candidates, teacher)
        hard_parts.append(
            hard_route_targets(scores, delta=delta).cpu().to(torch.int8)
        )
        soft_parts.append(
            soft_route_targets(
                scores,
                delta=delta,
                temperature=temperature,
            ).cpu().half()
        )
        score_parts.append(scores.cpu().half())
    return {
        "hard_routes": torch.cat(hard_parts).to(torch.int64),
        "soft_probabilities": torch.cat(soft_parts),
        "teacher_scores": torch.cat(score_parts),
    }


def pointer_route_loss(
    *,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    hard_targets: torch.Tensor,
    soft_targets: torch.Tensor,
    teacher_maps: torch.Tensor,
    config: Mapping[str, Any],
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    section = config["pointer_route"]["one_shot"]
    if tuple(logits.shape) != tuple(soft_targets.shape):
        raise ValueError("logits and soft_targets must have matching shapes")
    if tuple(hard_targets.shape) != tuple(logits.shape[:2]):
        raise ValueError("hard_targets must have shape [B,P]")
    if loss_mask is None:
        loss_mask = torch.ones_like(hard_targets, dtype=torch.bool)
    if tuple(loss_mask.shape) != tuple(hard_targets.shape):
        raise ValueError("loss_mask must have shape [B,P]")
    if not bool(loss_mask.any().item()):
        raise ValueError("loss_mask cannot be empty")
    target_distribution = soft_targets.float().detach().clone()
    teacher = F.normalize(
        teacher_maps.float().detach().clone(),
        dim=-1,
    )
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    per_position_soft_ce = -(
        target_distribution * log_probabilities
    ).sum(dim=-1)
    soft_ce = per_position_soft_ce[loss_mask].mean()
    hard_ce = F.cross_entropy(
        logits[loss_mask].float(),
        hard_targets[loss_mask],
    )
    committed_distribution = F.one_hot(
        hard_targets,
        num_classes=logits.shape[-1],
    ).float()
    map_distribution = torch.where(
        loss_mask.unsqueeze(-1),
        probabilities,
        committed_distribution,
    )
    predicted_map = reconstruct_soft_routes(candidates, map_distribution)
    local_map = 1.0 - (
        F.normalize(predicted_map.float(), dim=-1) * teacher
    ).sum(dim=-1).mean()
    predicted_identity = F.normalize(
        predicted_map.float().mean(dim=1),
        dim=-1,
    )
    teacher_identity = F.normalize(teacher.mean(dim=1), dim=-1)
    identity_map = 1.0 - (
        predicted_identity * teacher_identity
    ).sum(dim=-1).mean()
    total = (
        float(section["route_soft_ce_weight"]) * soft_ce
        + float(section["route_hard_ce_weight"]) * hard_ce
        + float(section["local_map_cosine_weight"]) * local_map
        + float(section["identity_map_cosine_weight"]) * identity_map
    )
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("PointerRoute loss is non-finite")
    exact = (
        logits.argmax(dim=-1)[loss_mask] == hard_targets[loss_mask]
    ).float().mean()
    return total, {
        "loss": float(total.detach().item()),
        "route_soft_ce": float(soft_ce.detach().item()),
        "route_hard_ce": float(hard_ce.detach().item()),
        "local_map_cosine_loss": float(local_map.detach().item()),
        "identity_map_cosine_loss": float(identity_map.detach().item()),
        "hard_route_accuracy": float(exact.detach().item()),
        "masked_fraction": float(loss_mask.float().mean().detach().item()),
    }


def _subset_tensors(
    *,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
) -> dict[str, Any]:
    precomputed = _require_precomputed(dataset)
    index_tensor = torch.tensor(list(indices), dtype=torch.int64)
    rows = [dataset.synthetic.rows[index] for index in indices]
    return {
        "condition_features": precomputed["condition_features"][index_tensor],
        "condition_quality": precomputed["condition_quality"][index_tensor],
        "condition_visibility": precomputed["condition_visibility"][index_tensor],
        "target_identity": precomputed["target_identity"][index_tensor],
        "scenarios": [str(precomputed["scenarios"][index]) for index in indices],
        "teacher_maps": dataset.targets["continuous_prototypes"][index_tensor],
        "source_identities": torch.tensor(
            [row["source_identity_ids"] for row in rows],
            dtype=torch.int64,
        ),
        "sample_ids": [str(precomputed["sample_ids"][index]) for index in indices],
        "global_indices": index_tensor,
    }


def _route_accuracy(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    scores: torch.Tensor,
    delta: float,
    equivalent_epsilon: float,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    utilities = margin_adjusted_utilities(scores.float(), delta=delta)
    selected = torch.gather(
        utilities,
        2,
        predicted.unsqueeze(-1),
    ).squeeze(-1)
    equivalent = (
        utilities.max(dim=-1).values - selected
        <= float(equivalent_epsilon)
    )
    exact = predicted == target
    scenario_array = np.asarray(list(scenarios))
    result: dict[str, Any] = {}
    for scenario in sorted(set(scenarios)):
        mask = torch.from_numpy(scenario_array == scenario)
        result[scenario] = {
            "exact_route_accuracy": float(exact[mask].float().mean().item()),
            "equivalent_route_accuracy": float(
                equivalent[mask].float().mean().item()
            ),
        }
    result["all"] = {
        "exact_route_accuracy": float(exact.float().mean().item()),
        "equivalent_route_accuracy": float(equivalent.float().mean().item()),
    }
    return result


@torch.inference_mode()
def evaluate_pointer_route_subset(
    *,
    split_name: str,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    targets: Mapping[str, torch.Tensor],
    model: PointerRouteTransformer,
    config: Mapping[str, Any],
    device: torch.device,
    delta: float,
    temperature: float,
) -> dict[str, Any]:
    model.eval()
    values = _subset_tensors(dataset=dataset, indices=indices)
    batch_size = int(config["evaluation"]["batch_size"])
    map_parts: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "quality_anchor",
            "hard_route_oracle",
            "soft_route_oracle",
            "one_shot_pointer",
            "one_shot_pointer_soft",
        )
    }
    probability_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []
    local_target_indices = values["global_indices"]
    hard_targets = targets["hard_routes"][local_target_indices]
    soft_targets = targets["soft_probabilities"][local_target_indices]
    scores = targets["teacher_scores"][local_target_indices].float()

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        condition = values["condition_features"][start:end].to(device)
        quality = values["condition_quality"][start:end].to(device)
        input_routes = model.all_mask_input(end - start, device=device)
        logits, candidates = model(input_routes, condition, quality)
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted = probabilities.argmax(dim=-1)
        local_hard = hard_targets[start:end].to(device)
        local_soft = soft_targets[start:end].to(device)
        maps = {
            "quality_anchor": candidates[:, :, 0],
            "hard_route_oracle": reconstruct_hard_routes(
                candidates,
                local_hard,
            ),
            "soft_route_oracle": reconstruct_soft_routes(
                candidates,
                local_soft,
            ),
            "one_shot_pointer": reconstruct_hard_routes(
                candidates,
                predicted,
            ),
            "one_shot_pointer_soft": reconstruct_soft_routes(
                candidates,
                probabilities,
            ),
        }
        for name, method_map in maps.items():
            map_parts[name].append(method_map.cpu().half())
        probability_parts.append(probabilities.cpu().half())
        prediction_parts.append(predicted.cpu())

    all_maps = {
        name: torch.cat(parts).float() for name, parts in map_parts.items()
    }
    probabilities = torch.cat(probability_parts).float()
    predictions = torch.cat(prediction_parts)
    identities = values["target_identity"].to(torch.int64)
    teacher_maps = values["teacher_maps"].float()
    scenarios = values["scenarios"]
    clean_indices = [
        index for index, scenario in enumerate(scenarios)
        if scenario == "clean"
    ]
    gallery_identities = identities[clean_indices]
    gallery_embeddings = F.normalize(
        teacher_maps[clean_indices].mean(dim=1),
        dim=-1,
    )
    order = torch.argsort(gallery_identities)
    gallery_identities = gallery_identities[order]
    gallery_embeddings = gallery_embeddings[order]
    methods: dict[str, Any] = {}
    for name, method_map in all_maps.items():
        methods[name] = {
            "verification": _evaluate_embeddings(
                embeddings=F.normalize(method_map.mean(dim=1), dim=-1),
                gallery_embeddings=gallery_embeddings,
                identities=identities,
                gallery_identities=gallery_identities,
                scenarios=scenarios,
                far_targets=[
                    float(value)
                    for value in config["evaluation"]["far_targets"]
                ],
            ),
            "teacher_similarity": map_similarity_metrics(
                method_map,
                teacher_maps,
                scenarios,
            ),
        }

    first_count = min(16, len(indices))
    first_condition = values["condition_features"][:first_count].to(device)
    first_quality = values["condition_quality"][:first_count].to(device)
    original_logits, original_candidates = model(
        model.all_mask_input(first_count, device=device),
        first_condition,
        first_quality,
    )
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    permuted_logits, permuted_candidates = model(
        model.all_mask_input(first_count, device=device),
        first_condition[:, permutation],
        first_quality[:, permutation],
    )
    reordered_logits = torch.empty_like(permuted_logits)
    reordered_logits[..., 0] = permuted_logits[..., 0]
    for permuted_index, original_index in enumerate(permutation.tolist()):
        reordered_logits[..., original_index + 1] = (
            permuted_logits[..., permuted_index + 1]
        )
    original_prediction = original_logits.argmax(dim=-1)
    permuted_prediction = permuted_logits.argmax(dim=-1)
    original_map = reconstruct_hard_routes(
        original_candidates,
        original_prediction,
    )
    permuted_map = reconstruct_hard_routes(
        permuted_candidates,
        permuted_prediction,
    )
    permutation_check = {
        "permutation": permutation.tolist(),
        "logit_max_abs_difference": float(
            (original_logits - reordered_logits).abs().max().item()
        ),
        "route_map_max_abs_difference": float(
            (original_map - permuted_map).abs().max().item()
        ),
        "route_map_cosine_min": float(
            F.cosine_similarity(original_map, permuted_map, dim=-1)
            .min()
            .item()
        ),
    }
    return {
        "split": split_name,
        "samples": len(indices),
        "identities": len(clean_indices),
        "gallery": "clean_disjoint_teacher_continuous",
        "methods": methods,
        "route_accuracy": _route_accuracy(
            predicted=predictions,
            target=hard_targets,
            scores=scores,
            delta=delta,
            equivalent_epsilon=float(
                config["pointer_route"]["equivalent_epsilon"]
            ),
            scenarios=scenarios,
        ),
        "route_diagnostics": route_diagnostics(
            routes=predictions,
            probabilities=probabilities,
            scores=scores,
            delta=delta,
            source_identities=values["source_identities"],
            target_identities=identities,
            visibility=values["condition_visibility"],
            scenarios=scenarios,
        ),
        "permutation_check": permutation_check,
        "inference_contract": {
            "teacher_used": False,
            "gallery_used": False,
            "condition_features_used": True,
            "condition_quality_used": True,
        },
    }


def _learning_rate(
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    base: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base * float(step) / float(warmup_steps)
    progress = min(
        1.0,
        max(
            0.0,
            float(step - warmup_steps)
            / float(max(1, max_steps - warmup_steps)),
        ),
    )
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def load_pointer_checkpoint(
    *,
    path: str | Path,
    model: PointerRouteTransformer,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def _next_batch(
    *,
    train_indices: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    state: dict[str, Any],
) -> torch.Tensor:
    order = state.get("order")
    cursor = int(state.get("cursor", 0))
    if order is None or cursor + batch_size > int(order.numel()):
        order = torch.randperm(
            int(train_indices.numel()),
            generator=generator,
        )
        cursor = 0
    selected = train_indices[order[cursor : cursor + batch_size]]
    state["order"] = order
    state["cursor"] = cursor + batch_size
    return selected


def train_one_shot_pointer(
    *,
    dataset: TokenTrainingDataset,
    train_indices: Sequence[int],
    calibration_indices: Sequence[int],
    targets: Mapping[str, torch.Tensor],
    model: PointerRouteTransformer,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_directory: str | Path,
    delta: float,
    temperature: float,
    seed: int,
    smoke: bool,
    metric_callback: Callable[[int, Mapping[str, float]], None] | None = None,
) -> dict[str, Any]:
    section = config["pointer_route"]["one_shot"]
    max_steps = (
        int(section["smoke_steps"]) if smoke else int(section["max_steps"])
    )
    min_steps = 1 if smoke else int(section["min_steps"])
    validate_every = max_steps if smoke else int(
        section["validate_every_steps"]
    )
    accumulation = 1 if smoke else int(section["gradient_accumulation"])
    batch_size = min(2, int(section["batch_size"])) if smoke else int(
        section["batch_size"]
    )
    warmup = 0 if smoke else int(section["warmup_steps"])
    checkpoint_root = Path(checkpoint_directory).expanduser().resolve()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    use_fp16 = str(section["precision"]) == "fp16"
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and use_fp16,
    )
    generator = torch.Generator().manual_seed(int(seed))
    sampler_state: dict[str, Any] = {}
    train_tensor = torch.tensor(list(train_indices), dtype=torch.int64)
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    best_score: tuple[float, float, float] | None = None
    best_checkpoint: Path | None = None
    best_step = 0
    patience = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _ in range(accumulation):
            indices = _next_batch(
                train_indices=train_tensor,
                batch_size=batch_size,
                generator=generator,
                state=sampler_state,
            )
            precomputed = _require_precomputed(dataset)
            condition = precomputed["condition_features"][indices].to(device)
            quality = precomputed["condition_quality"][indices].to(device)
            teacher = dataset.targets["continuous_prototypes"][indices].to(
                device
            )
            hard = targets["hard_routes"][indices].to(device)
            soft = targets["soft_probabilities"][indices].to(device)
            input_routes = model.all_mask_input(len(indices), device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                logits, candidates = model(
                    input_routes,
                    condition,
                    quality,
                )
                loss, loss_values = pointer_route_loss(
                    logits=logits,
                    candidates=candidates,
                    hard_targets=hard,
                    soft_targets=soft,
                    teacher_maps=teacher,
                    config=config,
                )
                scaled_loss = loss / float(accumulation)
            scaler.scale(scaled_loss).backward()
            for key, value in loss_values.items():
                accumulated[key] = accumulated.get(key, 0.0) + (
                    value / float(accumulation)
                )
        scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(section["max_grad_norm"]),
            ).item()
        )
        learning_rate = _learning_rate(
            step=step,
            max_steps=max_steps,
            warmup_steps=warmup,
            base=float(section["learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        scaler.step(optimizer)
        scaler.update()
        row = {
            "step": step,
            **accumulated,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        if metric_callback is not None:
            metric_callback(step, row)

        if step % validate_every == 0 or step == max_steps:
            validation = evaluate_pointer_route_subset(
                split_name="route_calibration",
                dataset=dataset,
                indices=calibration_indices,
                targets=targets,
                model=model,
                config=config,
                device=device,
                delta=delta,
                temperature=temperature,
            )
            primary = validation["methods"]["one_shot_pointer"][
                "verification"
            ]
            score = (
                float(primary["hard_average"]["roc_auc"]),
                float(primary["all"]["roc_auc"]),
                float(
                    validation["route_accuracy"]["all"][
                        "equivalent_route_accuracy"
                    ]
                ),
            )
            validations.append(
                {
                    "step": step,
                    "score": list(score),
                    "hard_auc": score[0],
                    "all_auc": score[1],
                    "equivalent_route_accuracy": score[2],
                }
            )
            if best_score is None or score > best_score:
                checkpoint = (
                    checkpoint_root / f"one_shot_step_{step:08d}.pt"
                )
                _atomic_checkpoint(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "stage": "pointer-route-one-shot",
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "parameter_count": model.parameter_count(),
                        "validation": validation,
                        "delta": float(delta),
                        "temperature": float(temperature),
                        "config": dict(config["pointer_route"]["one_shot"]),
                    },
                )
                best_score = score
                best_checkpoint = checkpoint
                best_step = step
                patience = 0
            else:
                patience += 1
            if (
                not smoke
                and not bool(
                    config["pointer_route"]["diffusion"][
                        "paired_force_full_steps"
                    ]
                )
                and step >= min_steps
                and patience >= int(section["early_stop_patience"])
            ):
                break

    if best_checkpoint is None:
        raise RuntimeError("PointerRoute training produced no checkpoint")
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    return {
        "steps_completed": int(history[-1]["step"]),
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint),
        "best_score": list(best_score or ()),
        "history": history,
        "validations": validations,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": model.parameter_count(),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "smoke": smoke,
    }


@torch.inference_mode()
def measure_pointer_latency(
    *,
    dataset: TokenTrainingDataset,
    indices: Sequence[int],
    model: PointerRouteTransformer,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    batch_size = 16
    if len(indices) < batch_size:
        raise ValueError("latency measurement needs at least 16 samples")
    precomputed = _require_precomputed(dataset)
    selected = torch.tensor(list(indices[:batch_size]), dtype=torch.int64)
    condition = precomputed["condition_features"][selected].to(device)
    quality = precomputed["condition_quality"][selected].to(device)
    input_routes = model.all_mask_input(batch_size, device=device)
    model.eval()
    warmup = int(config["evaluation"]["latency_warmup_runs"])
    repeats = int(config["evaluation"]["latency_measurement_runs"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        logits, candidates = model(input_routes, condition, quality)
        reconstruct_hard_routes(candidates, logits.argmax(dim=-1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        logits, candidates = model(input_routes, condition, quality)
        reconstruct_hard_routes(candidates, logits.argmax(dim=-1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "warmup_runs": warmup,
        "measurement_runs": repeats,
        "mean_batch_latency_ms": 1000.0 * elapsed / float(repeats),
        "mean_sample_latency_ms": (
            1000.0 * elapsed / float(repeats * batch_size)
        ),
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
    }
