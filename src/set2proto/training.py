"""Dataset adaptation, fair one-shot/MaskGIT training, and checkpointing."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate

from .datasets import make_set_dataset
from .environment import config_sha256
from .model import ConditionalTokenTransformer
from .quantization import PCAProjection
from .synthetic import _load_tensor_file


TrainingMode = Literal["one-shot", "maskgit"]


class TokenTrainingDataset(Dataset[dict[str, Any]]):
    """Join corrupted conditions with projected, quantized teacher targets."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        quantization_root: str | Path,
        split: str,
        precompute: bool = False,
    ) -> None:
        self.synthetic = make_set_dataset(dataset_root, split)
        quantization_path = Path(quantization_root).expanduser().resolve()
        projection_payload = _load_tensor_file(
            quantization_path / "projection.pt"
        )
        self.projector = PCAProjection.from_payload(projection_payload)
        self.targets = _load_tensor_file(
            quantization_path / f"{split}_targets.pt"
        )
        if self.targets["sample_ids"] != [
            row["sample_id"] for row in self.synthetic.rows
        ]:
            raise ValueError(
                "synthetic samples and quantized targets are not aligned"
            )
        if len(self.synthetic) != int(self.targets["tokens"].shape[0]):
            raise ValueError("synthetic/target sample counts do not match")
        self._precomputed: dict[str, Any] | None = None
        if precompute:
            self.precompute()

    def precompute(self, batch_size: int = 64) -> None:
        """Project every deterministic condition set once for repeated epochs."""

        if self._precomputed is not None:
            return
        condition_features: list[torch.Tensor] = []
        condition_quality: list[torch.Tensor] = []
        condition_visibility: list[torch.Tensor] = []
        target_identities: list[torch.Tensor] = []
        sample_ids: list[str] = []
        scenarios: list[str] = []
        for start in range(0, len(self.synthetic), batch_size):
            samples = [
                self.synthetic[index]
                for index in range(
                    start,
                    min(start + batch_size, len(self.synthetic)),
                )
            ]
            raw = torch.stack(
                [sample["condition_features"] for sample in samples]
            )
            condition_features.append(self.projector.transform(raw))
            condition_quality.append(
                torch.stack(
                    [sample["condition_quality"] for sample in samples]
                )
            )
            condition_visibility.append(
                torch.stack(
                    [sample["condition_visibility"] for sample in samples]
                )
            )
            target_identities.append(
                torch.stack(
                    [sample["target_identity"] for sample in samples]
                )
            )
            sample_ids.extend(str(sample["sample_id"]) for sample in samples)
            scenarios.extend(str(sample["scenario"]) for sample in samples)
        self._precomputed = {
            "condition_features": torch.cat(condition_features, dim=0),
            "condition_quality": torch.cat(condition_quality, dim=0),
            "condition_visibility": torch.cat(
                condition_visibility,
                dim=0,
            ),
            "target_identity": torch.cat(target_identities, dim=0),
            "sample_ids": sample_ids,
            "scenarios": scenarios,
        }

    def __len__(self) -> int:
        return len(self.synthetic)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._precomputed is not None:
            return {
                "sample_id": self._precomputed["sample_ids"][index],
                "scenario": self._precomputed["scenarios"][index],
                "target_identity": self._precomputed["target_identity"][
                    index
                ],
                "condition_features": self._precomputed[
                    "condition_features"
                ][index],
                "condition_quality": self._precomputed[
                    "condition_quality"
                ][index],
                "condition_visibility": self._precomputed[
                    "condition_visibility"
                ][index],
                "target_tokens": self.targets["tokens"][index].to(
                    torch.int64
                ),
            }
        sample = self.synthetic[index]
        return {
            "sample_id": sample["sample_id"],
            "scenario": sample["scenario"],
            "target_identity": sample["target_identity"],
            "condition_features": self.projector.transform(
                sample["condition_features"]
            ),
            "condition_quality": sample["condition_quality"],
            "condition_visibility": sample["condition_visibility"],
            "target_tokens": self.targets["tokens"][index].to(torch.int64),
        }


def load_codebook(quantization_root: str | Path) -> torch.Tensor:
    payload = _load_tensor_file(
        Path(quantization_root).expanduser().resolve() / "codebook.pt"
    )
    return F.normalize(payload["vectors"].float(), dim=-1)


def build_corrupted_tokens(
    target_tokens: torch.Tensor,
    *,
    mask_token_id: int,
    mode: TrainingMode,
    min_mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create all-mask one-shot inputs or random-ratio MaskGIT inputs."""

    if mode == "one-shot":
        mask = torch.ones_like(target_tokens, dtype=torch.bool)
    elif mode == "maskgit":
        batch, positions = target_tokens.shape
        uniform = torch.rand(batch, device=target_tokens.device)
        ratios = torch.cos(0.5 * math.pi * uniform).clamp_min(
            min_mask_ratio
        )
        counts = torch.ceil(ratios * positions).to(torch.int64).clamp(
            min=1,
            max=positions,
        )
        random_scores = torch.rand(
            target_tokens.shape,
            device=target_tokens.device,
        )
        ordering = torch.argsort(random_scores, dim=-1)
        ranks = torch.empty_like(ordering)
        rank_values = torch.arange(
            positions,
            device=target_tokens.device,
        )[None].expand(batch, -1)
        ranks.scatter_(1, ordering, rank_values)
        mask = ranks < counts[:, None]
    else:
        raise ValueError(f"unsupported training mode: {mode}")
    corrupted = target_tokens.clone()
    corrupted[mask] = mask_token_id
    return corrupted, mask


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(mask.any().item()):
        raise ValueError("training mask cannot be empty")
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    loss = F.cross_entropy(
        selected_logits,
        selected_targets,
        label_smoothing=label_smoothing,
    )
    accuracy = (
        selected_logits.argmax(dim=-1) == selected_targets
    ).float().mean()
    return loss, accuracy


def _learning_rate(
    *,
    completed_steps: int,
    max_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
) -> float:
    if warmup_steps > 0 and completed_steps < warmup_steps:
        return base_learning_rate * float(completed_steps + 1) / float(
            warmup_steps
        )
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(
        1.0,
        max(0.0, float(completed_steps - warmup_steps) / decay_steps),
    )
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def checkpoint_path(
    checkpoint_dir: str | Path,
    *,
    mode: TrainingMode,
    step: int,
) -> Path:
    safe_mode = mode.replace("-", "_")
    return (
        Path(checkpoint_dir).expanduser().resolve()
        / f"{safe_mode}_step_{step:08d}.pt"
    )


def find_latest_checkpoint(
    checkpoint_dir: str | Path,
    *,
    mode: TrainingMode,
) -> Path | None:
    safe_mode = mode.replace("-", "_")
    candidates = sorted(
        Path(checkpoint_dir).expanduser().resolve().glob(
            f"{safe_mode}_step_*.pt"
        )
    )
    return candidates[-1] if candidates else None


@dataclass
class TrainingResult:
    mode: TrainingMode
    model: ConditionalTokenTransformer
    steps: int
    losses: list[float]
    accuracies: list[float]
    learning_rates: list[float]
    checkpoint: Path
    peak_allocated_mib: float
    peak_reserved_mib: float
    resumed_from: str | None


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_single_model(
    *,
    model: ConditionalTokenTransformer,
    dataset: TokenTrainingDataset,
    config: Mapping[str, Any],
    mode: TrainingMode,
    device: torch.device,
    checkpoint_dir: str | Path,
    batch_size: int,
    condition_frames: int,
    seed: int,
    resume_checkpoint: str | Path | None = None,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> TrainingResult:
    training = config["training"]
    max_steps = int(training["max_steps"])
    accumulation = int(training["gradient_accumulation"])
    precision = str(training["precision"])
    autocast_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    use_autocast = device.type == "cuda" and precision != "fp32"
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.to(device)
    model.train()
    checkpoint_directory = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    batch_generator = torch.Generator().manual_seed(seed + 101)
    completed_steps = 0
    losses: list[float] = []
    accuracies: list[float] = []
    learning_rates: list[float] = []
    resumed_from: str | None = None

    if resume_checkpoint is not None:
        checkpoint = torch.load(
            Path(resume_checkpoint),
            map_location=device,
            weights_only=True,
        )
        if checkpoint["mode"] != mode:
            raise ValueError("checkpoint training mode does not match")
        if checkpoint["config_sha256"] != config_sha256(config):
            raise ValueError("checkpoint resolved configuration does not match")
        if int(checkpoint["batch_size"]) != batch_size:
            raise ValueError("checkpoint batch size does not match")
        if int(checkpoint["condition_frames"]) != condition_frames:
            raise ValueError("checkpoint condition frame count does not match")
        if int(checkpoint["hidden_dim"]) != model.hidden_dim:
            raise ValueError("checkpoint hidden dimension does not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state(optimizer, device)
        scaler.load_state_dict(checkpoint["scaler"])
        completed_steps = int(checkpoint["step"])
        batch_generator.set_state(checkpoint["batch_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        losses = [float(value) for value in checkpoint["losses"]]
        accuracies = [float(value) for value in checkpoint["accuracies"]]
        learning_rates = [
            float(value) for value in checkpoint["learning_rates"]
        ]
        resumed_from = str(Path(resume_checkpoint).resolve())

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    while completed_steps < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_accuracy = 0.0
        for _ in range(accumulation):
            indices = torch.randint(
                0,
                len(dataset),
                (batch_size,),
                generator=batch_generator,
            ).tolist()
            batch = default_collate([dataset[index] for index in indices])
            condition_features = batch["condition_features"][
                :, :condition_frames
            ].to(device, non_blocking=device.type == "cuda")
            condition_quality = batch["condition_quality"][
                :, :condition_frames
            ].to(device, non_blocking=device.type == "cuda")
            target_tokens = batch["target_tokens"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            corrupted_tokens, mask = build_corrupted_tokens(
                target_tokens,
                mask_token_id=model.mask_token_id,
                mode=mode,
                min_mask_ratio=float(
                    training["maskgit_min_mask_ratio"]
                ),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                logits = model(
                    corrupted_tokens,
                    condition_features,
                    condition_quality,
                )
                loss, accuracy = masked_cross_entropy(
                    logits,
                    target_tokens,
                    mask,
                    label_smoothing=float(training["label_smoothing"]),
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach().item())
            accumulated_accuracy += float(accuracy.detach().item())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(training["max_grad_norm"]),
        )
        learning_rate = _learning_rate(
            completed_steps=completed_steps,
            max_steps=max_steps,
            warmup_steps=int(training["warmup_steps"]),
            base_learning_rate=float(training["learning_rate"]),
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        scaler.step(optimizer)
        scaler.update()
        completed_steps += 1

        step_loss = accumulated_loss / accumulation
        step_accuracy = accumulated_accuracy / accumulation
        losses.append(step_loss)
        accuracies.append(step_accuracy)
        learning_rates.append(learning_rate)
        if event_callback is not None:
            event_callback(
                "training_step",
                {
                    "mode": mode,
                    "step": completed_steps,
                    "loss": step_loss,
                    "accuracy": step_accuracy,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "condition_frames": condition_frames,
                },
            )

        if (
            completed_steps % int(training["checkpoint_every_steps"]) == 0
            or completed_steps == max_steps
        ):
            path = checkpoint_path(
                checkpoint_directory,
                mode=mode,
                step=completed_steps,
            )
            _atomic_torch_save(
                path,
                {
                    "schema_version": 1,
                    "mode": mode,
                    "step": completed_steps,
                    "config_sha256": config_sha256(config),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "batch_generator_state": batch_generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "losses": losses,
                    "accuracies": accuracies,
                    "learning_rates": learning_rates,
                    "batch_size": batch_size,
                    "condition_frames": condition_frames,
                    "hidden_dim": model.hidden_dim,
                },
            )

    final_checkpoint = checkpoint_path(
        checkpoint_directory,
        mode=mode,
        step=completed_steps,
    )
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
    return TrainingResult(
        mode=mode,
        model=model,
        steps=completed_steps,
        losses=losses,
        accuracies=accuracies,
        learning_rates=learning_rates,
        checkpoint=final_checkpoint,
        peak_allocated_mib=round(peak_allocated, 3),
        peak_reserved_mib=round(peak_reserved, 3),
        resumed_from=resumed_from,
    )
