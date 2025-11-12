# backend/model/encoders/voronoicnn.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..factory import register
from ..base_components.encoder_base import BaseEncoder

def _to_4d(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        x4 = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        return x4, (b, t)
    elif x.ndim == 4:
        return x, (x.shape[0], 1)
    raise ValueError(f"Unexpected shape {tuple(x.shape)}")

def _to_5d(y: torch.Tensor, bt: Tuple[int, int]) -> torch.Tensor:
    b, t = bt
    c, h, w = y.shape[1], y.shape[-2], y.shape[-1]
    if t == 1:
        return y.view(b, c, 1, h, w)
    return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

class _Conv7x7Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=7, padding=3, bias=True)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

@register("encoder", "VoronoiCNN")
class VoronoiCNNEncoder(BaseEncoder):
    """
    Voronoi-CNN encoder:
      - 输入通道：2（Voronoi值 + mask）
      - 结构：连续 7×7 Conv2d 堆叠（论文：lmax=9, 每层通道48）
      - 输出：中间特征 [B, Cmid, T, H, W]（默认 Cmid=48）
    """
    def __init__(
        self,
        in_channels: int = 2,
        mid_channels: int = 48,   # 论文 m=48
        num_layers: int = 8       # 这里做成8层特征堆叠，末层7×7输出在Head；总层数≈9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_layers = num_layers

        layers = []
        layers.append(_Conv7x7Block(in_channels, mid_channels, act=True))
        for _ in range(num_layers - 1):
            layers.append(_Conv7x7Block(mid_channels, mid_channels, act=True))
        self.backbone = nn.Sequential(*layers)
        self.out_channels = mid_channels

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        # 与现有模块保持一致的 dtype 对齐
        ref_dtype = next(self.parameters()).dtype
        if x5.dtype != ref_dtype:
            x5 = x5.to(ref_dtype)

        x4, bt = _to_4d(x5)                 # [B*T, C, H, W]
        feat4 = self.backbone(x4)           # [B*T, mid, H, W]
        feat5 = _to_5d(feat4, bt)           # [B, mid, T, H, W]
        self.skips = None                   # 本网络无多尺度跳连
        return feat5
