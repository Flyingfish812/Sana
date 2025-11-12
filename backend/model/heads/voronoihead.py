# backend/model/heads/voronoihead.py
from __future__ import annotations
import torch
import torch.nn as nn
from ..factory import register
from ..base_components.head_base import BaseHead

@register("head", "VoronoiCNNHead")
class VoronoiCNNHead(BaseHead):
    """
    末端 7×7 输出层：把 encoder 的中间特征映射到 1 通道物理场。
    与论文保持一致：最后一层仍使用 7×7（非 1×1）。
    """
    def __init__(self, in_channels: int = 48, out_channels: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持 4D/5D，统一到 4D 做 2D 卷积再还原
        ref_dtype = next(self.conv.parameters()).dtype
        if x.dtype != ref_dtype:
            x = x.to(ref_dtype)
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x2 = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            y2 = self.conv(x2)
            y5 = y2.reshape(b, t, self.out_channels, h, w).permute(0, 2, 1, 3, 4)
            return y5
        elif x.ndim == 4:
            return self.conv(x).unsqueeze(2)
        raise ValueError(f"Unexpected input shape {tuple(x.shape)}")
