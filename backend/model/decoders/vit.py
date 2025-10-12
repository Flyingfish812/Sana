"""Patch-based decoder to invert the ViT embedding."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..base_components.decoder_base import BaseDecoder
from ..factory import register
from ..utils.norm import ChannelLayerNorm


def _to_2tuple(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


@register("decoder", "ViTPatchDecoder")
class ViTDecoder(BaseDecoder):
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int | Tuple[int, int],
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = _to_2tuple(patch_size)
        self.out_channels = out_channels or embed_dim
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        self.proj = nn.ConvTranspose3d(
            embed_dim,
            self.out_channels,
            kernel_size=(1, self.patch_size[0], self.patch_size[1]),
            stride=(1, self.patch_size[0], self.patch_size[1]),
        )
        self.norm = ChannelLayerNorm(self.out_channels)

    def forward(self, x5: torch.Tensor, skips=None) -> torch.Tensor:
        in_dtype = x5.dtype
        if self.proj.weight.device != x5.device:
            self.proj = self.proj.to(x5.device)

        if torch.is_autocast_enabled():
            with torch.amp.autocast('cuda', enabled=False):
                x = self.proj(x5.to(dtype=self.proj.weight.dtype))
        else:
            x = self.proj(x5.to(dtype=self.proj.weight.dtype))

        x = x.to(in_dtype)
        x = self.dropout(x)
        x = self.norm(x)
        return x
