"""Patch-based decoder to invert the ViT embedding."""
from __future__ import annotations

from typing import Optional, Tuple, List

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
        self.dropout_p = 0.1 if dropout == 0.0 else dropout
        self.neck = self._build_neck(self.embed_dim)
        self.upsample_blocks = nn.ModuleList(self._build_upsample_blocks(self.embed_dim))
        self.final_conv = nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.final_conv.weight, nonlinearity="linear")
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        self.norm = ChannelLayerNorm(self.out_channels)

    def _build_neck(self, channels: int, num_layers: int = 3) -> nn.Sequential:
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            layers.extend([
                conv,
                nn.GELU(),
                nn.Dropout(p=self.dropout_p),
            ])
        return nn.Sequential(*layers)

    def _factorize_scale(self, scale: int) -> List[int]:
        steps: List[int] = []
        remaining = int(scale)
        while remaining > 1:
            if remaining % 2 == 0:
                step = 2
            else:
                step = remaining
            steps.append(step)
            remaining //= step
        return steps

    def _build_upsample_blocks(self, channels: int) -> List[nn.Module]:
        plan: List[Tuple[int, int]] = []
        h_steps = self._factorize_scale(self.patch_size[0])
        w_steps = self._factorize_scale(self.patch_size[1])
        while h_steps or w_steps:
            sh = h_steps.pop(0) if h_steps else 1
            sw = w_steps.pop(0) if w_steps else 1
            plan.append((sh, sw))

        blocks: List[nn.Module] = []
        for sh, sw in plan:
            block = nn.Sequential(
                nn.Upsample(scale_factor=(sh, sw), mode="bilinear", align_corners=False),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(p=self.dropout_p),
            )
            nn.init.kaiming_normal_(block[1].weight, nonlinearity="relu")
            if block[1].bias is not None:
                nn.init.zeros_(block[1].bias)
            blocks.append(block)
        return blocks

    def forward(self, x5: torch.Tensor, skips=None) -> torch.Tensor:
        b, c, t, gh, gw = x5.shape
        x = x5.permute(0, 2, 1, 3, 4).reshape(b * t, c, gh, gw)
        x = self.neck(x)
        for block in self.upsample_blocks:
            x = block(x)
        x = self.final_conv(x)
        H, W = x.shape[-2:]
        x = x.view(b, t, self.out_channels, H, W).permute(0, 2, 1, 3, 4)
        x = self.norm(x)
        return x
