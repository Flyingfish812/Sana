"""Vision Transformer encoder blocks used as propagators."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..base_components.propagator_base import BasePropagator
from ..factory import register


def _apply_layer_norm(module: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """Run ``module`` in its parameter dtype while preserving the input dtype."""

    input_dtype = x.dtype
    norm_dtype = module.weight.dtype if module.weight is not None else input_dtype
    return module(x.to(dtype=norm_dtype)).to(dtype=input_dtype)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        droppath: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
            bias=qkv_bias,
        )
        self.drop_path1 = DropPath(droppath)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(droppath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = _apply_layer_norm(self.norm1, x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path1(attn_out.to(dtype=x.dtype))
        mlp_out = self.mlp(_apply_layer_norm(self.norm2, x))
        x = x + self.drop_path2(mlp_out.to(dtype=x.dtype))
        return x


@register("propagator", "ViTTransformer")
class ViTPropagator(BasePropagator):
    """Stack of ViT blocks operating on patch tokens."""

    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        droppath: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError("Transformer depth must be positive")
        drop_values: List[float] = torch.linspace(0.0, droppath, depth).tolist() if depth > 1 else [droppath]
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    droppath=drop_values[i],
                    qkv_bias=qkv_bias,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x5.shape
        tokens = x5.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = _apply_layer_norm(self.norm, tokens)
        tokens = tokens.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return tokens
