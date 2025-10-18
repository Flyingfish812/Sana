# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from ..factory import register
from ..base_components.decoder_base import BaseDecoder

# ======= 基本组件 =======
def _conv_block_2d(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
    )

class _UpBlock2d(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.align = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.fuse = _conv_block_2d(in_c + skip_c, out_c)

    def forward(self, x4: torch.Tensor, skip4: torch.Tensor) -> torch.Tensor:
        ref_dtype = next(self.fuse.parameters()).dtype
        if x4.dtype != ref_dtype:
            x4 = x4.to(ref_dtype)
        if skip4.dtype != ref_dtype:
            skip4 = skip4.to(ref_dtype)

        x4 = F.interpolate(x4, size=(skip4.shape[-2], skip4.shape[-1]),
                       mode="nearest")
        x4 = self.align(x4)
        x4 = torch.cat([x4, skip4], dim=1)
        return self.fuse(x4)

def _to_4d(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, t)
    elif x.ndim == 4:
        return x, (x.shape[0], 1)
    else:
        raise ValueError(f"Unexpected shape {x.shape}")


def _to_5d(y: torch.Tensor, bt: Tuple[int, int]) -> torch.Tensor:
    b, t = bt
    if t == 1:
        return y.reshape(b, -1, 1, y.shape[-2], y.shape[-1])
    return y.reshape(b, t, y.shape[1], y.shape[-2], y.shape[-1]).permute(0, 2, 1, 3, 4)


# ======= 主体结构 =======
@register("decoder", "UNetBase")
class UNetDecoder(BaseDecoder):
    """
    UNet Decoder (Conv2d)
    - 无懒构建，参数初始化于 __init__
    - 自动 dtype 对齐
    """
    def __init__(self, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.base_channels = base_channels
        self.depth = depth

        chans = [base_channels * (2 ** i) for i in range(depth)]
        self.ups = nn.ModuleList()
        in_c = chans[-1]
        for i in range(depth - 2, -1, -1):
            self.ups.append(_UpBlock2d(in_c, chans[i], chans[i]))
            in_c = chans[i]
        self.out_channels = chans[0]

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        ref_dtype = next(self.parameters()).dtype
        if x.dtype != ref_dtype:
            x = x.to(ref_dtype)
        skips = [s.to(ref_dtype) if s.dtype != ref_dtype else s for s in skips]

        x4, bt = _to_4d(x)
        skips4 = [_to_4d(s)[0] for s in skips]

        h = x4
        for i, up in enumerate(self.ups):
            s4 = skips4[-(i + 2)]
            h = up(h, s4)

        y = _to_5d(h, bt)
        return y
