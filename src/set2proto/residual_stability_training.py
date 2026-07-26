"""Exposure-matched training for synthesis-capable residual MaskGIT."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from .model import ConditionalTokenTransformer
from .training import TokenTrainingDataset


StabilityTrainingMode = Literal["one_shot_control", "rollout_maskgit"]


@dataclass(frozen=True)
class ResidualStabilityTrainingResult:
    mode: StabilityTrainingMode
    steps: int
    history: list[dict[str, float]]
    checkpoints: dict[int, str]
    parameter_count: int
    elapsed_seconds: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    device_audit: dict[str, Any]


def _cosine_commit_count(
    *,
    positions: int,
    step: int,
    total_steps: int,
) -> int:
    if positions <= 0 or not 1 <= step <= total_steps:
        raise ValueError("invalid residual rollout commit schedule")
    if step == total_steps:
        return positions
    remaining = int(
        math.ceil(
            positions
            * math.cos(0.5 * math.pi * float(step) / float(total_steps))
        )
    )
    return positions - min(positions, max(0, remaining))


@torch.no_grad()
def build_residual_self_rollout(
    *,
    model: ConditionalTokenTransformer,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    rollout_total_steps: int,
    rollout_depths: Sequence[int],
    generator: torch.Generator,
) -> torch.Tensor:
    """Return partial model-generated states with gradients stopped."""

    depths = tuple(int(value) for value in rollout_depths)
    if (
        not depths
        or len(set(depths)) != len(depths)
        or any(not 1 <= value < rollout_total_steps for value in depths)
    ):
        raise ValueError("rollout depths must be unique intermediate steps")
    batch = int(condition_features.shape[0])
    positions = int(model.spatial_tokens)
    device = condition_features.device
    depth_choice = torch.randint(
        0,
        len(depths),
        (batch,),
        generator=generator,
    )
    selected_depth = torch.tensor(depths, dtype=torch.int64)[depth_choice].to(
        device
    )
    current = torch.full(
        (batch, positions),
        model.mask_token_id,
        dtype=torch.int64,
        device=device,
    )
    committed = torch.zeros_like(current, dtype=torch.bool)
    states: dict[int, torch.Tensor] = {}
    was_training = model.training
    model.eval()
    for step in range(1, max(depths) + 1):
        logits = model(current, condition_features, condition_quality)
        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        desired = _cosine_commit_count(
            positions=positions,
            step=step,
            total_steps=rollout_total_steps,
        )
        current_count = int(committed[0].sum().item())
        add_count = desired - current_count
        if add_count > 0:
            selected = confidence.masked_fill(
                committed,
                float("-inf"),
            ).topk(add_count, dim=-1).indices
            newly_committed = torch.zeros_like(committed)
            newly_committed.scatter_(1, selected, True)
            current = torch.where(newly_committed, prediction, current)
            committed = committed | newly_committed
        if step in depths:
            states[step] = current.clone()
    if was_training:
        model.train()
    result = torch.empty_like(current)
    for depth in depths:
        mask = selected_depth == depth
        if bool(mask.any().item()):
            result[mask] = states[depth][mask]
    if not bool((result == model.mask_token_id).any(dim=1).all().item()):
        raise RuntimeError("self rollout must leave masked positions")
    return result.detach()


def _teacher_partial_state(
    *,
    target_tokens: torch.Tensor,
    mask_token_id: int,
    min_mask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    batch, positions = target_tokens.shape
    uniform = torch.rand(batch, generator=generator)
    ratios = torch.cos(0.5 * math.pi * uniform).clamp_min(min_mask_ratio)
    counts = torch.ceil(ratios * positions).to(torch.int64).clamp(
        min=1,
        max=positions,
    )
    random_scores = torch.rand(
        batch,
        positions,
        generator=generator,
    )
    ordering = torch.argsort(random_scores, dim=-1)
    ranks = torch.empty_like(ordering)
    ranks.scatter_(
        1,
        ordering,
        torch.arange(positions)[None].expand(batch, -1),
    )
    mask = (ranks < counts[:, None]).to(target_tokens.device)
    state = target_tokens.clone()
    state[mask] = int(mask_token_id)
    return state


@torch.no_grad()
def build_exposure_matched_corruption(
    *,
    model: ConditionalTokenTransformer,
    condition_features: torch.Tensor,
    condition_quality: torch.Tensor,
    target_tokens: torch.Tensor,
    config: Mapping[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Mix all-mask, teacher-partial, and model-rollout states."""

    section = config["p2_2"]["corruption"]
    batch, positions = target_tokens.shape
    all_threshold = float(section["all_mask_probability"])
    teacher_threshold = all_threshold + float(
        section["teacher_partial_probability"]
    )
    choice = torch.rand(batch, generator=generator)
    all_mask = choice < all_threshold
    teacher_partial = (choice >= all_threshold) & (choice < teacher_threshold)
    self_rollout = choice >= teacher_threshold
    state = torch.full_like(target_tokens, model.mask_token_id)
    if bool(teacher_partial.any().item()):
        teacher_indices = torch.nonzero(
            teacher_partial,
            as_tuple=False,
        ).flatten().to(target_tokens.device)
        partial = _teacher_partial_state(
            target_tokens=target_tokens.index_select(0, teacher_indices),
            mask_token_id=model.mask_token_id,
            min_mask_ratio=float(section["min_mask_ratio"]),
            generator=generator,
        )
        state.index_copy_(0, teacher_indices, partial)
    if bool(self_rollout.any().item()):
        rollout_indices = torch.nonzero(
            self_rollout,
            as_tuple=False,
        ).flatten().to(target_tokens.device)
        rollout = build_residual_self_rollout(
            model=model,
            condition_features=condition_features.index_select(
                0,
                rollout_indices,
            ),
            condition_quality=condition_quality.index_select(
                0,
                rollout_indices,
            ),
            rollout_total_steps=int(section["rollout_total_steps"]),
            rollout_depths=[
                int(value) for value in section["rollout_depths"]
            ],
            generator=generator,
        )
        state.index_copy_(0, rollout_indices, rollout)
    masked = state == model.mask_token_id
    committed = ~masked
    correct_committed = committed & (state == target_tokens)
    wrong_committed = committed & (state != target_tokens)
    weights = torch.full(
        (batch, positions),
        float(section["masked_token_weight"]),
        dtype=torch.float32,
        device=target_tokens.device,
    )
    weights[correct_committed] = float(
        section["correct_committed_weight"]
    )
    weights[wrong_committed] = float(section["wrong_committed_weight"])
    if not bool((weights.sum(dim=1) > 0).all().item()):
        raise RuntimeError("exposure corruption produced an empty loss")
    diagnostics = {
        "all_mask_fraction": float(all_mask.float().mean().item()),
        "teacher_partial_fraction": float(
            teacher_partial.float().mean().item()
        ),
        "self_rollout_fraction": float(self_rollout.float().mean().item()),
        "masked_fraction": float(masked.float().mean().item()),
        "correct_committed_fraction": float(
            correct_committed.float().mean().item()
        ),
        "wrong_committed_fraction": float(
            wrong_committed.float().mean().item()
        ),
    }
    return state.detach(), weights.detach(), diagnostics


def weighted_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tuple(logits.shape[:2]) != tuple(targets.shape):
        raise ValueError("logits and targets have incompatible shapes")
    if tuple(weights.shape) != tuple(targets.shape):
        raise ValueError("weights and targets have incompatible shapes")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    denominator = weights.sum().clamp_min(1e-8)
    loss = (losses * weights).sum() / denominator
    prediction = logits.argmax(dim=-1)
    accuracy = (prediction == targets).float()
    active = weights > 0
    return loss, {
        "weighted_accuracy": float(
            (accuracy * weights).sum().div(denominator).detach().item()
        ),
        "active_accuracy": float(accuracy[active].mean().detach().item()),
    }


def _learning_rate(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base: float,
) -> float:
    if warmup_steps and step <= warmup_steps:
        return base * float(step) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def train_residual_stability_model(
    *,
    model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    train_indices: Sequence[int],
    config: Mapping[str, Any],
    mode: StabilityTrainingMode,
    device: torch.device,
    checkpoint_root: str | Path,
    seed: int,
    smoke: bool,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ResidualStabilityTrainingResult:
    if mode not in {"one_shot_control", "rollout_maskgit"}:
        raise ValueError("unsupported P2-2 training mode")
    if device.type != "cuda":
        raise RuntimeError("P2-2 stability training requires CUDA")
    if dataset._precomputed is None:
        raise ValueError("P2-2 training requires precomputed conditions")
    section = config["p2_2"]
    steps = int(section["smoke_steps"] if smoke else section["formal_steps"])
    batch_size = 2 if smoke else int(section["batch_size"])
    accumulation = 1 if smoke else int(section["gradient_accumulation"])
    precision = str(section["precision"])
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=precision == "fp16",
    )
    model.to(device)
    model.train()
    selected = torch.tensor(list(train_indices), dtype=torch.int64)
    condition_cache = dataset._precomputed["condition_features"].index_select(
        0,
        selected,
    ).to(device, dtype=torch.float16)
    quality_cache = dataset._precomputed["condition_quality"].index_select(
        0,
        selected,
    ).to(device, dtype=torch.float16)
    token_cache = dataset.targets["tokens"].index_select(
        0,
        selected,
    ).to(device, dtype=torch.int64)
    cache_bytes = sum(
        value.numel() * value.element_size()
        for value in (condition_cache, quality_cache, token_cache)
    )
    batch_generator = torch.Generator().manual_seed(int(seed))
    corruption_generator = torch.Generator().manual_seed(int(seed) + 1)
    checkpoint_directory = Path(checkpoint_root).expanduser().resolve()
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    checkpoints: dict[int, str] = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    checkpoint_every = 1 if smoke else int(section["checkpoint_every_steps"])
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _ in range(accumulation):
            local_indices = torch.randint(
                0,
                len(selected),
                (batch_size,),
                generator=batch_generator,
            ).to(device)
            condition = condition_cache.index_select(0, local_indices)
            quality = quality_cache.index_select(0, local_indices)
            targets = token_cache.index_select(0, local_indices)
            if mode == "one_shot_control":
                state = torch.full_like(targets, model.mask_token_id)
                weights = torch.ones_like(targets, dtype=torch.float32)
                corruption_values = {
                    "all_mask_fraction": 1.0,
                    "teacher_partial_fraction": 0.0,
                    "self_rollout_fraction": 0.0,
                    "masked_fraction": 1.0,
                    "correct_committed_fraction": 0.0,
                    "wrong_committed_fraction": 0.0,
                }
            else:
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                ):
                    state, weights, corruption_values = (
                        build_exposure_matched_corruption(
                            model=model,
                            condition_features=condition,
                            condition_quality=quality,
                            target_tokens=targets,
                            config=config,
                            generator=corruption_generator,
                        )
                    )
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            ):
                logits = model(state, condition, quality)
                loss, loss_values = weighted_token_cross_entropy(
                    logits,
                    targets,
                    weights,
                )
                scaled_loss = loss / float(accumulation)
            scaler.scale(scaled_loss).backward()
            values = {
                "loss": float(loss.detach().item()),
                **loss_values,
                **{
                    f"corruption_{key}": float(value)
                    for key, value in corruption_values.items()
                },
            }
            for key, value in values.items():
                accumulated[key] = accumulated.get(key, 0.0) + (
                    value / float(accumulation)
                )
        scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(section["max_grad_norm"]),
            ).item()
        )
        learning_rate = _learning_rate(
            step=step,
            total_steps=steps,
            warmup_steps=(0 if smoke else int(section["warmup_steps"])),
            base=float(section["learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        scaler.step(optimizer)
        scaler.update()
        row = {
            "step": float(step),
            **accumulated,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
        }
        history.append(row)
        if event_callback is not None:
            event_callback({"mode": mode, **row})
        if step % checkpoint_every == 0 or step == steps:
            checkpoint = checkpoint_directory / (
                f"{mode}_step_{step:08d}.pt"
            )
            _atomic_checkpoint(
                checkpoint,
                {
                    "schema_version": 1,
                    "stage": "p2-2-residual-rollout-stability",
                    "mode": mode,
                    "step": step,
                    "model_state": model.state_dict(),
                    "parameter_count": model.parameter_count(),
                    "history_tail": history[-100:],
                    "device": str(device),
                    "cache_mib": cache_bytes / 1024**2,
                },
            )
            checkpoints[step] = str(checkpoint)
    return ResidualStabilityTrainingResult(
        mode=mode,
        steps=steps,
        history=history,
        checkpoints=checkpoints,
        parameter_count=model.parameter_count(),
        elapsed_seconds=time.perf_counter() - started,
        peak_allocated_mib=float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        peak_reserved_mib=float(
            torch.cuda.max_memory_reserved(device) / 1024**2
        ),
        device_audit={
            "device": str(device),
            "model_device": str(next(model.parameters()).device),
            "condition_cache_device": str(condition_cache.device),
            "quality_cache_device": str(quality_cache.device),
            "token_cache_device": str(token_cache.device),
            "cache_mib": cache_bytes / 1024**2,
            "autocast_dtype": str(autocast_dtype),
        },
    )
