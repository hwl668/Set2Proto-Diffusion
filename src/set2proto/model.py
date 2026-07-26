"""Parameter-matched conditional token Transformer using native PyTorch SDPA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPAMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.reshape(
            batch,
            tokens,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = query if context is None else context
        q = self._reshape(self.query(query))
        k = self._reshape(self.key(context))
        v = self._reshape(self.value(context))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            query.shape[0],
            query.shape[1],
            self.hidden_dim,
        )
        return self.output(attended)


class ConditionalDecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = SDPAMultiheadAttention(
            hidden_dim,
            heads,
            dropout,
        )
        self.cross_target_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = SDPAMultiheadAttention(
            hidden_dim,
            heads,
            dropout,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(round(hidden_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        target = target + self.self_attention(self.target_norm(target))
        target = target + self.cross_attention(
            self.cross_target_norm(target),
            self.context_norm(context),
        )
        target = target + self.mlp(self.mlp_norm(target))
        return target


class ConditionalTokenTransformer(nn.Module):
    """Predict 49 discrete prototype tokens from an unordered frame set."""

    def __init__(
        self,
        *,
        feature_dim: int,
        spatial_tokens: int,
        codebook_size: int,
        hidden_dim: int,
        layers: int,
        attention_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_tokens = spatial_tokens
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        self.mask_token_id = codebook_size

        self.condition_projection = nn.Linear(feature_dim, hidden_dim)
        self.condition_quality_projection = nn.Linear(1, hidden_dim)
        self.condition_position = nn.Parameter(
            torch.empty(1, 1, spatial_tokens, hidden_dim)
        )
        self.token_embedding = nn.Embedding(codebook_size + 1, hidden_dim)
        self.target_position = nn.Parameter(
            torch.empty(1, spatial_tokens, hidden_dim)
        )
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
        self.output_head = nn.Linear(hidden_dim, codebook_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.condition_position, std=0.02)
        nn.init.trunc_normal_(self.target_position, std=0.02)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_condition(
        self,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
    ) -> torch.Tensor:
        if condition_features.ndim != 4:
            raise ValueError(
                "condition_features must have shape [B,M,P,D]"
            )
        batch, frames, positions, feature_dim = condition_features.shape
        if positions != self.spatial_tokens or feature_dim != self.feature_dim:
            raise ValueError(
                "condition feature shape does not match model configuration"
            )
        if tuple(condition_quality.shape) != (batch, frames):
            raise ValueError("condition_quality must have shape [B,M]")

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
        return encoded.reshape(batch, frames * positions, self.hidden_dim)

    def forward(
        self,
        input_tokens: torch.Tensor,
        condition_features: torch.Tensor,
        condition_quality: torch.Tensor,
    ) -> torch.Tensor:
        if input_tokens.ndim != 2:
            raise ValueError("input_tokens must have shape [B,P]")
        if input_tokens.shape[1] != self.spatial_tokens:
            raise ValueError("input token count does not match spatial_tokens")
        if input_tokens.min().item() < 0:
            raise ValueError("input_tokens cannot be negative")
        if input_tokens.max().item() > self.mask_token_id:
            raise ValueError("input token id exceeds mask token id")

        context = self.encode_condition(
            condition_features,
            condition_quality,
        )
        target = self.token_embedding(input_tokens) + self.target_position
        for block in self.blocks:
            target = block(target, context)
        return self.output_head(self.output_norm(target))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model_from_config(
    config: dict,
    *,
    hidden_dim: int | None = None,
) -> ConditionalTokenTransformer:
    grid = config["data"]["feature_grid"]
    model_config = config["model"]
    return ConditionalTokenTransformer(
        feature_dim=int(grid["projected_dim"]),
        spatial_tokens=int(grid["height"]) * int(grid["width"]),
        codebook_size=int(config["quantization"]["codebook_size"]),
        hidden_dim=(
            int(model_config["hidden_dim"])
            if hidden_dim is None
            else int(hidden_dim)
        ),
        layers=int(model_config["layers"]),
        attention_heads=int(model_config["attention_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
    )

