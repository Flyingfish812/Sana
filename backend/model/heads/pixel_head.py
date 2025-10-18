# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ..base_components.head_base import BaseHead
from ..factory import register

@register("head", "PixelHead")
class PixelHead(BaseHead):
    """
    简单 1x1 Conv2d Head (T 维展开到 batch)
    """
    def __init__(self, in_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持 [B,C,H,W] / [B,C,T,H,W]
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
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")
