# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple
from ..factory import register
from ..base_components.encoder_base import BaseEncoder

# ========== 基本组件 ==========
def _conv_block_2d(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
    )

def _down_block_2d(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(in_c, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        _conv_block_2d(in_c, out_c),
    )

def _to_4d(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        # permute 后立刻 contiguous，再 view，避免隐式拷贝代价不确定
        x4 = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        return x4, (b, t)
    elif x.ndim == 4:
        return x, (x.shape[0], 1)
    else:
        raise ValueError(f"Unexpected shape {x.shape}")

def _to_5d(y: torch.Tensor, bt: Tuple[int, int]) -> torch.Tensor:
    b, t = bt
    c, h, w = y.shape[1], y.shape[-2], y.shape[-1]
    if t == 1:
        return y.view(b, c, 1, h, w)
    return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

# ========== 主体结构 ==========
@register("encoder", "UNetBase")
class UNetEncoder(BaseEncoder):
    """
    UNet Encoder (Conv2d)
    - 无懒构建（所有层在 __init__ 完成）
    - dtype 与 device 由 Lightning 管理（forward 内仅一次校验）
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth

        # 通道表
        chans = [base_channels * (2 ** i) for i in range(depth)]
        self.stem = _conv_block_2d(in_channels, chans[0])

        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(_down_block_2d(chans[i], chans[i + 1]))

        self.out_channels = chans[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 统一输入 dtype 与权重 dtype
        ref_dtype = next(self.parameters()).dtype
        if x.dtype != ref_dtype:
            x = x.to(ref_dtype)

        x4, bt = _to_4d(x)

        # 下采样路径
        skips4 = []
        h = self.stem(x4)
        skips4.append(h)
        for down in self.downs:
            h = down(h)
            skips4.append(h)

        # 还原形状并缓存
        self.skips = [_to_5d(s, bt) for s in skips4]
        z = self.skips[-1]
        return z
