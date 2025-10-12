# backend/model/heads/pixel_head.py
from __future__ import annotations
import torch
import torch.nn as nn
from ..base_components.head_base import BaseHead
from ..factory import register

@register("head", "PixelHead")
def PixelHead(out_channels: int = 1):
    return _PixelHead(out_channels=out_channels)

class _PixelHead(BaseHead):
    """
    简单的 1x1 卷积头（只在 H,W 做 1x1，T 维不变）
    输入:  [B,C,T,H,W]  (C ~= base_channels)
    输出:  [B,out,T,H,W]
    """
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=None or 32, out_channels=out_channels, kernel_size=(1,1,1))
        self._lazy_built = False
        self.out_channels = out_channels

    def _lazy_build(self, in_c: int):
        self.conv = nn.Conv3d(in_c, self.out_channels, kernel_size=(1,1,1))
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self._lazy_built = True

    def forward(self, x5: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self._lazy_built:
            self._lazy_build(x5.shape[1])
            # 懒构建后立刻搬到输入设备，避免默认落 CPU
            self.conv = self.conv.to(x5.device)

        in_dtype = x5.dtype
        if self.conv.weight.device != x5.device:
            self.conv = self.conv.to(x5.device)

        if torch.is_autocast_enabled():
            with torch.amp.autocast('cuda', enabled=False):
                y = self.conv(x5.to(dtype=self.conv.weight.dtype))
        else:
            y = self.conv(x5.to(dtype=self.conv.weight.dtype))
        return y.to(in_dtype)
