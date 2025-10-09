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
        self.out_channels = out_channels
        self.conv1: nn.Module = nn.Identity()
        self.conv2: nn.Module = nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self._lazy_built = False

    def _lazy_build(self, in_c: int):
        conv1 = nn.Conv3d(in_c, in_c, kernel_size=(1,1,1))
        conv2 = nn.Conv3d(in_c, self.out_channels, kernel_size=(1,1,1))
        nn.init.kaiming_normal_(conv1.weight, nonlinearity="relu")
        if conv1.bias is not None:
            nn.init.zeros_(conv1.bias)
        nn.init.kaiming_normal_(conv2.weight, nonlinearity="linear")
        if conv2.bias is not None:
            nn.init.zeros_(conv2.bias)
        self.conv1 = conv1
        self.conv2 = conv2
        self._lazy_built = True

    def forward(self, x5: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self._lazy_built:
            self._lazy_build(x5.shape[1])
        x = self.conv1(x5)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x
