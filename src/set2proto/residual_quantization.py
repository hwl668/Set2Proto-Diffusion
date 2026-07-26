"""Train-only Euclidean quantization of anchor-relative prototype residuals.

The absolute prototype codebook in :mod:`set2proto.quantization` is spherical:
it must encode the complete identity prototype.  This module instead keeps a
continuous quality-pooled anchor and quantizes only the correction

``residual = teacher_map - quality_anchor``.

Residual centroids deliberately retain their magnitude.  They are therefore
fit and assigned with Euclidean distance, and are *not* projected to the unit
sphere.  A quantized prototype is reconstructed as
``normalize(anchor + centroid)`` at every spatial position.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Match the existing quantization implementation's Windows-safe joblib setup.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import MiniBatchKMeans


RESIDUAL_QUANTIZATION_SCHEMA_VERSION = 1


def _validate_matching_maps(
    teacher_map: torch.Tensor,
    quality_anchor: torch.Tensor,
) -> None:
    if teacher_map.ndim < 2:
        raise ValueError("prototype maps must have at least 2 dimensions")
    if teacher_map.shape != quality_anchor.shape:
        raise ValueError(
            "teacher_map and quality_anchor must have identical shapes"
        )
    if teacher_map.shape[-1] < 1:
        raise ValueError("prototype feature dimension must be positive")
    if not bool(torch.isfinite(teacher_map).all().item()):
        raise ValueError("teacher_map contains non-finite values")
    if not bool(torch.isfinite(quality_anchor).all().item()):
        raise ValueError("quality_anchor contains non-finite values")


def compute_residuals(
    teacher_map: torch.Tensor,
    quality_anchor: torch.Tensor,
) -> torch.Tensor:
    """Return the unnormalized anchor-relative residual.

    No normalization is applied before subtraction: callers get exactly
    ``teacher_map - quality_anchor`` as required by the representation.
    Computation is promoted to float32 so fitting is independent of an
    on-disk FP16 feature-cache dtype.
    """

    _validate_matching_maps(teacher_map, quality_anchor)
    return teacher_map.float() - quality_anchor.float()


@dataclass(frozen=True)
class ResidualCodebook:
    """A Euclidean residual codebook fitted exclusively on the train split."""

    vectors: torch.Tensor
    fit_tokens: int
    available_train_tokens: int
    seed: int
    fit_split: str = "train"

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise ValueError("residual codebooks must be train-only")
        if self.vectors.ndim != 2:
            raise ValueError("codebook vectors must have shape [K, D]")
        if self.vectors.shape[0] < 1 or self.vectors.shape[1] < 1:
            raise ValueError("codebook dimensions must be positive")
        if not bool(torch.isfinite(self.vectors).all().item()):
            raise ValueError("codebook vectors contain non-finite values")
        if self.fit_tokens < int(self.vectors.shape[0]):
            raise ValueError("fit_tokens cannot be smaller than codebook size")
        if self.available_train_tokens < self.fit_tokens:
            raise ValueError(
                "available_train_tokens cannot be smaller than fit_tokens"
            )

    @property
    def codebook_size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def encode(
        self,
        residuals: torch.Tensor,
        *,
        chunk_size: int = 8192,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return encode_residuals(
            residuals,
            self.vectors,
            chunk_size=chunk_size,
        )

    def reconstruct(
        self,
        quality_anchor: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return reconstruct_from_residual_tokens(
            quality_anchor,
            tokens,
            self.vectors,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RESIDUAL_QUANTIZATION_SCHEMA_VERSION,
            "method": "euclidean_residual_kmeans",
            "fit_split": self.fit_split,
            "fit_tokens": self.fit_tokens,
            "available_train_tokens": self.available_train_tokens,
            "seed": self.seed,
            "codebook_size": self.codebook_size,
            "dimension": self.dimension,
            "vectors": self.vectors.detach().cpu(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ResidualCodebook":
        if payload.get("method") != "euclidean_residual_kmeans":
            raise ValueError("payload is not a residual Euclidean codebook")
        if payload.get("fit_split") != "train":
            raise ValueError("residual codebook payload is not train-only")
        vectors = payload["vectors"].float()
        if int(payload["codebook_size"]) != int(vectors.shape[0]):
            raise ValueError("payload codebook_size does not match vectors")
        if int(payload["dimension"]) != int(vectors.shape[1]):
            raise ValueError("payload dimension does not match vectors")
        return cls(
            vectors=vectors,
            fit_tokens=int(payload["fit_tokens"]),
            available_train_tokens=int(
                payload["available_train_tokens"]
            ),
            seed=int(payload["seed"]),
            fit_split=str(payload["fit_split"]),
        )


def fit_residual_codebook(
    train_teacher_maps: torch.Tensor,
    train_quality_anchors: torch.Tensor,
    *,
    codebook_size: int,
    max_fit_tokens: int,
    batch_size: int,
    iterations: int,
    n_init: int,
    seed: int,
    fit_split: str = "train",
    device: str | torch.device | None = None,
) -> tuple[ResidualCodebook, dict[str, Any]]:
    """Fit deterministic MiniBatch K-means on train residual tokens only.

    The split argument is intentionally guarded instead of being metadata
    supplied after the fact.  This prevents an integration call from silently
    fitting on validation or test samples.
    """

    if fit_split != "train":
        raise ValueError(
            "residual codebook fitting is train-only; fit_split must be 'train'"
        )
    for name, value in (
        ("codebook_size", codebook_size),
        ("max_fit_tokens", max_fit_tokens),
        ("batch_size", batch_size),
        ("iterations", iterations),
        ("n_init", n_init),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")

    residuals = compute_residuals(
        train_teacher_maps,
        train_quality_anchors,
    )
    flattened = residuals.reshape(-1, residuals.shape[-1]).contiguous()
    available = int(flattened.shape[0])
    selected_count = min(available, int(max_fit_tokens))
    if selected_count < int(codebook_size):
        raise ValueError(
            f"codebook K={codebook_size} exceeds fit tokens={selected_count}"
        )

    if selected_count < available:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        indices = torch.randperm(available, generator=generator)[
            :selected_count
        ]
        selected = flattened[indices.to(flattened.device)]
    else:
        selected = flattened

    requested_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None or str(device) == "auto"
        else torch.device(device)
    )
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA residual K-means requested but unavailable")

    if requested_device.type == "cuda":
        vectors, backend_details = _fit_torch_minibatch_kmeans(
            selected,
            codebook_size=int(codebook_size),
            batch_size=min(int(batch_size), selected_count),
            iterations=int(iterations),
            n_init=int(n_init),
            seed=int(seed),
            device=requested_device,
        )
    else:
        estimator = MiniBatchKMeans(
            n_clusters=int(codebook_size),
            init="k-means++",
            n_init=int(n_init),
            max_iter=int(iterations),
            batch_size=min(int(batch_size), selected_count),
            random_state=int(seed),
            reassignment_ratio=0.01,
            verbose=0,
        )
        estimator.fit(selected.detach().cpu().numpy())
        vectors = torch.from_numpy(
            np.asarray(estimator.cluster_centers_)
        ).float()
        backend_details = {
            "backend": "sklearn_minibatch_kmeans",
            "device": "cpu",
            "inertia": float(estimator.inertia_),
            "n_steps": int(getattr(estimator, "n_steps_", 0)),
            "n_iter": int(getattr(estimator, "n_iter_", 0)),
        }
    codebook = ResidualCodebook(
        vectors=vectors,
        fit_tokens=selected_count,
        available_train_tokens=available,
        seed=int(seed),
    )
    details = {
        "method": "euclidean_residual_kmeans",
        "fit_split": "train",
        "fit_tokens": selected_count,
        "available_train_tokens": available,
        **backend_details,
        "mean_centroid_norm": float(vectors.norm(dim=-1).mean().item()),
        "max_centroid_norm": float(vectors.norm(dim=-1).max().item()),
    }
    return codebook, details


def _fit_torch_minibatch_kmeans(
    selected: torch.Tensor,
    *,
    codebook_size: int,
    batch_size: int,
    iterations: int,
    n_init: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """CUDA mini-batch Lloyd updates with chunked final assignment.

    Each update materializes at most ``batch_size × K`` distances.  Random
    data-point initialization avoids the much larger repeated full-dataset
    scans of exact K-means++ at K=1024; multiple initializations retain a
    deterministic best-inertia choice for a fixed CUDA/PyTorch environment.
    """

    samples = selected.detach().to(device=device, dtype=torch.float32)
    sample_count, dimension = samples.shape
    best_centers: torch.Tensor | None = None
    best_inertia = float("inf")
    best_initialization = -1
    final_assignment_chunk = max(1, min(batch_size, 8192))

    for initialization in range(n_init):
        generator = torch.Generator(device=device).manual_seed(
            seed + 104729 * initialization
        )
        initial_indices = torch.randperm(
            sample_count,
            generator=generator,
            device=device,
        )[:codebook_size]
        centers = samples[initial_indices].clone()
        cumulative_counts = torch.zeros(
            codebook_size,
            dtype=torch.float32,
            device=device,
        )

        for _ in range(iterations):
            batch_indices = torch.randint(
                sample_count,
                (batch_size,),
                generator=generator,
                device=device,
            )
            batch = samples[batch_indices]
            assignments, _ = encode_residuals(
                batch,
                centers,
                chunk_size=batch_size,
            )
            batch_counts = torch.bincount(
                assignments,
                minlength=codebook_size,
            ).to(dtype=torch.float32)
            batch_sums = torch.zeros(
                (codebook_size, dimension),
                dtype=torch.float32,
                device=device,
            )
            batch_sums.index_add_(0, assignments, batch)
            updated_counts = cumulative_counts + batch_counts
            observed = batch_counts > 0
            centers[observed] = (
                centers[observed]
                * cumulative_counts[observed].unsqueeze(-1)
                + batch_sums[observed]
            ) / updated_counts[observed].unsqueeze(-1)
            cumulative_counts = updated_counts

        _, squared_distance = encode_residuals(
            samples,
            centers,
            chunk_size=final_assignment_chunk,
        )
        inertia = float(squared_distance.sum().item())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.detach().cpu()
            best_initialization = initialization

    if best_centers is None:
        raise RuntimeError("residual K-means did not produce a codebook")
    return best_centers, {
        "backend": "torch_minibatch_kmeans",
        "device": str(device),
        "inertia": best_inertia,
        "n_steps": int(iterations),
        "n_iter": int(iterations),
        "selected_initialization": best_initialization,
    }


def encode_residuals(
    residuals: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign arbitrary leading residual dimensions by squared L2 distance."""

    if residuals.ndim < 1 or codebook.ndim != 2:
        raise ValueError("residuals must end in D and codebook must be [K, D]")
    if int(residuals.shape[-1]) != int(codebook.shape[-1]):
        raise ValueError("residual and codebook dimensions do not match")
    if int(codebook.shape[0]) < 1:
        raise ValueError("codebook must contain at least one vector")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if not bool(torch.isfinite(residuals).all().item()):
        raise ValueError("residuals contain non-finite values")
    if not bool(torch.isfinite(codebook).all().item()):
        raise ValueError("codebook contains non-finite values")
    if residuals.numel() < 1:
        raise ValueError("residuals cannot be empty")

    original_shape = residuals.shape[:-1]
    flat = residuals.reshape(-1, residuals.shape[-1]).float()
    centers = codebook.to(device=flat.device, dtype=torch.float32)
    center_squared_norm = centers.square().sum(dim=-1).unsqueeze(0)
    token_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    for start in range(0, int(flat.shape[0]), int(chunk_size)):
        chunk = flat[start : start + int(chunk_size)]
        distances = (
            chunk.square().sum(dim=-1, keepdim=True)
            + center_squared_norm
            - 2.0 * torch.matmul(chunk, centers.transpose(0, 1))
        ).clamp_min_(0.0)
        minimum, tokens = torch.min(distances, dim=-1)
        token_chunks.append(tokens)
        distance_chunks.append(minimum)
    tokens = torch.cat(token_chunks).reshape(original_shape).to(torch.int64)
    squared_distance = torch.cat(distance_chunks).reshape(original_shape)
    return tokens, squared_distance


def encode_teacher_residuals(
    teacher_map: torch.Tensor,
    quality_anchor: torch.Tensor,
    codebook: torch.Tensor | ResidualCodebook,
    *,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute and encode ``teacher_map - quality_anchor``."""

    vectors = (
        codebook.vectors
        if isinstance(codebook, ResidualCodebook)
        else codebook
    )
    return encode_residuals(
        compute_residuals(teacher_map, quality_anchor),
        vectors,
        chunk_size=chunk_size,
    )


def reconstruct_from_residual_tokens(
    quality_anchor: torch.Tensor,
    tokens: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct unit prototype vectors as ``normalize(anchor + code)``."""

    if quality_anchor.ndim < 2:
        raise ValueError("quality_anchor must have shape [..., positions, D]")
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [K, D]")
    if tuple(tokens.shape) != tuple(quality_anchor.shape[:-1]):
        raise ValueError("tokens must match quality_anchor leading dimensions")
    if int(codebook.shape[-1]) != int(quality_anchor.shape[-1]):
        raise ValueError("anchor and codebook dimensions do not match")
    if not bool(torch.isfinite(quality_anchor).all().item()):
        raise ValueError("quality_anchor contains non-finite values")
    if not bool(torch.isfinite(codebook).all().item()):
        raise ValueError("codebook contains non-finite values")
    if tokens.numel() < 1:
        raise ValueError("tokens cannot be empty")
    if int(tokens.min().item()) < 0 or int(tokens.max().item()) >= int(
        codebook.shape[0]
    ):
        raise ValueError("tokens fall outside the codebook range")

    vectors = codebook.to(
        device=quality_anchor.device,
        dtype=torch.float32,
    )
    corrections = vectors[tokens.to(device=quality_anchor.device).long()]
    return F.normalize(
        quality_anchor.float() + corrections,
        p=2.0,
        dim=-1,
        eps=1e-6,
    )


def _token_usage(tokens: torch.Tensor, codebook_size: int) -> dict[str, Any]:
    counts = torch.bincount(
        tokens.reshape(-1).cpu(),
        minlength=int(codebook_size),
    ).to(torch.float64)
    used = int((counts > 0).sum().item())
    probabilities = counts[counts > 0] / counts.sum().clamp_min(1.0)
    entropy = float(
        -(probabilities * probabilities.log()).sum().item()
    )
    perplexity = float(math.exp(entropy))
    return {
        "used_codes": used,
        "codebook_utilization": used / float(codebook_size),
        "token_perplexity": perplexity,
        "normalized_perplexity": perplexity / float(codebook_size),
        "token_counts": [int(value) for value in counts.tolist()],
    }


def _metric_summary(
    *,
    teacher_map: torch.Tensor,
    quality_anchor: torch.Tensor,
    tokens: torch.Tensor,
    codebook: torch.Tensor,
) -> dict[str, Any]:
    residual = compute_residuals(teacher_map, quality_anchor)
    local_tokens = tokens.to(device=teacher_map.device)
    codebook_on_device = codebook.to(
        device=teacher_map.device,
        dtype=torch.float32,
    )
    centroids = codebook_on_device[local_tokens.long()]
    reconstruction = reconstruct_from_residual_tokens(
        quality_anchor,
        local_tokens,
        codebook,
    )
    normalized_teacher = F.normalize(
        teacher_map.float(),
        p=2.0,
        dim=-1,
        eps=1e-6,
    )
    token_cosine = F.cosine_similarity(
        normalized_teacher,
        reconstruction,
        dim=-1,
        eps=1e-6,
    )
    teacher_global = F.normalize(
        normalized_teacher.mean(dim=-2),
        p=2.0,
        dim=-1,
        eps=1e-6,
    )
    reconstruction_global = F.normalize(
        reconstruction.mean(dim=-2),
        p=2.0,
        dim=-1,
        eps=1e-6,
    )
    map_cosine = F.cosine_similarity(
        teacher_global,
        reconstruction_global,
        dim=-1,
        eps=1e-6,
    )
    nonzero = residual.norm(dim=-1) > 1e-8
    if bool(nonzero.any().item()):
        residual_cosine = float(
            F.cosine_similarity(
                residual[nonzero],
                centroids[nonzero],
                dim=-1,
                eps=1e-6,
            ).mean().item()
        )
    else:
        residual_cosine = 1.0
    return {
        "residual_mse": float(F.mse_loss(centroids, residual).item()),
        "mean_residual_cosine": residual_cosine,
        "mean_token_cosine": float(token_cosine.mean().item()),
        "median_token_cosine": float(token_cosine.median().item()),
        "p05_token_cosine": float(
            torch.quantile(token_cosine.reshape(-1), 0.05).item()
        ),
        "mean_map_cosine": float(map_cosine.mean().item()),
        "prototype_mse": float(
            F.mse_loss(reconstruction, normalized_teacher).item()
        ),
    }


def compute_residual_quantization_metrics(
    *,
    teacher_map: torch.Tensor,
    quality_anchor: torch.Tensor,
    tokens: torch.Tensor,
    codebook: torch.Tensor | ResidualCodebook,
    reference_tokens: torch.Tensor | None = None,
    scenarios: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report reconstruction and code-usage metrics.

    ``reference_tokens`` is optional because exact token accuracy is meaningful
    for a predicted token map, but trivially one for nearest-centroid targets.
    Scenario summaries are produced without recomputing or refitting anything.
    """

    _validate_matching_maps(teacher_map, quality_anchor)
    vectors = (
        codebook.vectors
        if isinstance(codebook, ResidualCodebook)
        else codebook
    )
    if tuple(tokens.shape) != tuple(teacher_map.shape[:-1]):
        raise ValueError("tokens must match prototype leading dimensions")
    if int(vectors.shape[-1]) != int(teacher_map.shape[-1]):
        raise ValueError("prototype and codebook dimensions do not match")
    if int(tokens.min().item()) < 0 or int(tokens.max().item()) >= int(
        vectors.shape[0]
    ):
        raise ValueError("tokens fall outside the codebook range")

    metrics = _metric_summary(
        teacher_map=teacher_map,
        quality_anchor=quality_anchor,
        tokens=tokens,
        codebook=vectors,
    )
    metrics.update(_token_usage(tokens, int(vectors.shape[0])))
    if reference_tokens is not None:
        if reference_tokens.shape != tokens.shape:
            raise ValueError("reference_tokens must match tokens")
        metrics["exact_token_accuracy"] = float(
            (tokens == reference_tokens.to(tokens.device))
            .float()
            .mean()
            .item()
        )

    if scenarios is not None:
        if teacher_map.ndim != 3:
            raise ValueError(
                "scenario metrics require maps with shape [B, P, D]"
            )
        if len(scenarios) != int(teacher_map.shape[0]):
            raise ValueError("scenarios must contain one label per map")
        labels = np.asarray(list(scenarios), dtype=object)
        local_tokens = tokens.to(device=teacher_map.device)
        by_scenario: dict[str, Any] = {}
        for scenario in sorted(set(str(value) for value in scenarios)):
            mask = torch.from_numpy(labels == scenario).to(
                device=teacher_map.device
            )
            selected_tokens = local_tokens[mask]
            summary = _metric_summary(
                teacher_map=teacher_map[mask],
                quality_anchor=quality_anchor[mask],
                tokens=selected_tokens,
                codebook=vectors,
            )
            summary.update(
                _token_usage(selected_tokens, int(vectors.shape[0]))
            )
            if reference_tokens is not None:
                summary["exact_token_accuracy"] = float(
                    (
                        selected_tokens
                        == reference_tokens.to(teacher_map.device)[mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
            by_scenario[scenario] = summary
        metrics["by_scenario"] = by_scenario
    return metrics
