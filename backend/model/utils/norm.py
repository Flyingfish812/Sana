"""Normalization helpers for convolutional backbones."""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    """Apply layer norm over the channel dimension for 4D/5D tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm keeps its parameters in float32; make sure we normalise in
        # that dtype and then restore the original precision so autocast can
        # manage downstream ops.
        input_dtype = x.dtype
        norm_dtype = (
            self.norm.weight.dtype
            if self.norm.weight is not None
            else input_dtype
        )
        if x.ndim == 5:
            # [B,C,T,H,W] -> [B,T,H,W,C]
            x_perm = x.permute(0, 2, 3, 4, 1)
            x_norm = self.norm(x_perm.to(dtype=norm_dtype))
            return x_norm.to(dtype=input_dtype).permute(0, 4, 1, 2, 3)
        if x.ndim == 4:
            # [B,C,H,W] -> [B,H,W,C]
            x_perm = x.permute(0, 2, 3, 1)
            x_norm = self.norm(x_perm.to(dtype=norm_dtype))
            return x_norm.to(dtype=input_dtype).permute(0, 3, 1, 2)
        raise ValueError(f"ChannelLayerNorm expects 4D or 5D input, got shape {tuple(x.shape)}")


__all__ = ["ChannelLayerNorm"]
