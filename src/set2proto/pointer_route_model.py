"""Permutation-equivariant dynamic PointerRoute Transformer."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConditionalDecoderBlock
from .pointer_route_oracle import build_route_candidates


def _canonical_sum(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Reduce frame sets in a stable order for permutation checks."""

    return torch.sort(value, dim=dim).values.sum(dim=dim)


class PointerRouteTransformer(nn.Module):
    """Score an anchor and every observed frame with one shared candidate head.

    Frame routes have no learned frame-index embedding. A committed route is
    represented by the selected sample feature itself, making the same module
    suitable for a later masked iterative decoder.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        spatial_tokens: int,
        hidden_dim: int,
        layers: int,
        attention_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or spatial_tokens <= 0 or hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.spatial_tokens = int(spatial_tokens)
        self.hidden_dim = int(hidden_dim)
        self.mask_token_id = 5

        self.condition_projection = nn.Linear(feature_dim, hidden_dim)
        self.condition_quality_projection = nn.Linear(1, hidden_dim)
        self.condition_position = nn.Parameter(
            torch.empty(1, 1, spatial_tokens, hidden_dim)
        )
        self.target_position = nn.Parameter(
            torch.empty(1, spatial_tokens, hidden_dim)
        )
        self.route_feature_projection = nn.Linear(feature_dim, hidden_dim)
        # 0=mask, 1=anchor, 2=frame. Frame identity is carried only by its
        # selected feature, never by an absolute frame-index embedding.
        self.route_type_embedding = nn.Embedding(3, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ConditionalDecoderBlock(
                    hidden_dim=hidden_dim,
                    heads=attention_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.candidate_projection = nn.Linear(feature_dim, hidden_dim)
        self.candidate_scalar_projection = nn.Linear(5, hidden_dim)
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.condition_position, std=0.02)
        nn.init.trunc_normal_(self.target_position, std=0.02)
        nn.init.normal_(self.route_type_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_condition(
        self,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
    ) -> tuple[int, int]:
        if condition_features.ndim != 4:
            raise ValueError(
                "condition_features must have shape [B,F,P,D]"
            )
        batch, frames, positions, dimension = condition_features.shape
        if positions != self.spatial_tokens or dimension != self.feature_dim:
            raise ValueError(
                "condition feature shape does not match model configuration"
            )
        if frames != 4:
            raise ValueError("PointerRoute MVP requires exactly four frames")
        if tuple(condition_quality.shape) != (batch, frames):
            raise ValueError("condition_quality must have shape [B,F]")
        return batch, frames

    def encode_condition(
        self,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames = self._validate_condition(
            condition_features,
            condition_quality,
        )
        quality = condition_quality.float().clamp_min(1e-6)
        quality = torch.log(
            quality
            / quality.mean(dim=1, keepdim=True).clamp_min(1e-6)
        )
        encoded = self.condition_projection(condition_features)
        encoded = encoded + self.condition_position
        encoded = encoded + self.condition_quality_projection(
            quality[:, :, None, None].to(encoded.dtype)
        )
        return encoded.reshape(
            batch,
            frames * self.spatial_tokens,
            self.hidden_dim,
        )

    def _candidate_scalars(
        self,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        features = F.normalize(condition_features.float(), dim=-1)
        local = features.permute(0, 2, 1, 3)
        frames = int(local.shape[2])
        local_mean = F.normalize(
            _canonical_sum(local, dim=2) / float(frames),
            dim=-1,
            eps=1e-6,
        )
        local_consensus = (local * local_mean.unsqueeze(2)).sum(dim=-1)
        frame_global = F.normalize(local.mean(dim=1), dim=-1, eps=1e-6)
        global_mean = F.normalize(
            _canonical_sum(frame_global, dim=1) / float(frames),
            dim=-1,
            eps=1e-6,
        )
        global_consensus = (
            frame_global * global_mean.unsqueeze(1)
        ).sum(dim=-1)
        quality = condition_quality.float().clamp_min(1e-6)
        quality_share = quality / _canonical_sum(
            quality,
            dim=1,
        ).unsqueeze(-1).clamp_min(1e-6)
        anchor = candidates[:, :, 0]
        anchor_local_support = (anchor * local_mean).sum(dim=-1)
        anchor_global = F.normalize(anchor.mean(dim=1), dim=-1, eps=1e-6)
        anchor_global_support = (anchor_global * global_mean).sum(dim=-1)

        batch, positions = anchor.shape[:2]
        is_anchor = torch.zeros(
            (batch, positions, frames + 1),
            dtype=anchor.dtype,
            device=anchor.device,
        )
        is_anchor[..., 0] = 1.0
        candidate_quality = torch.cat(
            [
                torch.ones(
                    (batch, positions, 1),
                    dtype=anchor.dtype,
                    device=anchor.device,
                ),
                quality_share[:, None, :].expand(-1, positions, -1),
            ],
            dim=-1,
        )
        candidate_local_support = torch.cat(
            [
                anchor_local_support.unsqueeze(-1),
                local_consensus,
            ],
            dim=-1,
        )
        candidate_global_support = torch.cat(
            [
                anchor_global_support[:, None, None].expand(
                    -1,
                    positions,
                    1,
                ),
                global_consensus[:, None, :].expand(-1, positions, -1),
            ],
            dim=-1,
        )
        anchor_cosine = (candidates * anchor.unsqueeze(2)).sum(dim=-1)
        return torch.stack(
            (
                is_anchor,
                candidate_quality,
                candidate_local_support,
                candidate_global_support,
                anchor_cosine,
            ),
            dim=-1,
        )

    def _target_state(
        self,
        *,
        input_routes: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if input_routes.ndim != 2 or input_routes.shape[1] != self.spatial_tokens:
            raise ValueError("input_routes must have shape [B,P]")
        if int(input_routes.min().item()) < 0:
            raise ValueError("input_routes cannot be negative")
        if int(input_routes.max().item()) > self.mask_token_id:
            raise ValueError("input route exceeds mask token id")
        masked = input_routes == self.mask_token_id
        gather_routes = input_routes.clamp_max(4)
        selected = torch.gather(
            candidates,
            2,
            gather_routes[..., None, None].expand(
                -1,
                -1,
                1,
                self.feature_dim,
            ),
        ).squeeze(2)
        selected = torch.where(
            masked.unsqueeze(-1),
            torch.zeros_like(selected),
            selected,
        )
        route_type = torch.where(
            masked,
            torch.zeros_like(input_routes),
            torch.where(
                input_routes == 0,
                torch.ones_like(input_routes),
                torch.full_like(input_routes, 2),
            ),
        )
        return (
            self.route_feature_projection(selected)
            + self.route_type_embedding(route_type)
            + self.target_position
        )

    def forward(
        self,
        input_routes: torch.Tensor,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_condition(condition_features, condition_quality)
        candidates = build_route_candidates(
            condition_features,
            condition_quality,
        ).to(condition_features.dtype)
        context = self.encode_condition(
            condition_features,
            condition_quality,
        )
        target = self._target_state(
            input_routes=input_routes,
            candidates=candidates,
        )
        for block in self.blocks:
            target = block(target, context)
        state = self.output_norm(target)
        candidate_state = self.candidate_norm(
            self.candidate_projection(candidates)
            + self.candidate_scalar_projection(
                self._candidate_scalars(
                    condition_features,
                    condition_quality,
                    candidates,
                ).to(candidates.dtype)
            )
        )
        expanded_state = state.unsqueeze(2).expand_as(candidate_state)
        logits = self.score_head(
            torch.cat(
                (
                    expanded_state,
                    candidate_state,
                    expanded_state * candidate_state,
                    expanded_state - candidate_state,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        return logits.float(), candidates.float()

    def all_mask_input(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full(
            (int(batch_size), self.spatial_tokens),
            self.mask_token_id,
            dtype=torch.int64,
            device=device,
        )

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())


def build_pointer_route_model(
    config: Mapping[str, Any],
) -> PointerRouteTransformer:
    grid = config["data"]["feature_grid"]
    model_config = config["model"]
    model = PointerRouteTransformer(
        feature_dim=int(grid["projected_dim"]),
        spatial_tokens=int(grid["height"]) * int(grid["width"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        attention_heads=int(model_config["attention_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
    )
    maximum = int(config["pointer_route"]["one_shot"]["max_parameters"])
    if model.parameter_count() > maximum:
        raise ValueError(
            f"PointerRoute model has {model.parameter_count()} parameters, "
            f"exceeding cap {maximum}"
        )
    return model
