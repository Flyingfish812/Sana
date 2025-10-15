# backend/model/encoders/unet_base.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List
from ..base_components.encoder_base import BaseEncoder
from ..factory import register
from ..utils.pool import PoolAwareTime
from ..utils.norm import ChannelLayerNorm

def conv_block(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(1,k,k), stride=(1,s,s), padding=(0,p,p), bias=False),
        ChannelLayerNorm(out_c),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=(1,k,k), stride=(1,1,1), padding=(0,p,p), bias=False),
        ChannelLayerNorm(out_c),
        nn.ReLU(inplace=True),
    )

def down_block(in_c, out_c):
    return nn.Sequential(
        PoolAwareTime(kernel_size=(1,2,2), stride=(1,2,2), ceil_mode=False,
                          mode2d="max", mode3d="avg"),
        conv_block(in_c, out_c),
    )

@register("encoder", "UNetBase")
def UNetEncoder(base_channels: int = 32, in_channels: int | None = None, depth: int = 4):
    return _UNetEncoder(base_channels=base_channels, in_channels=in_channels, depth=depth)


class _UNetEncoder(BaseEncoder):
    """
    一个简化版 3D-UNet 编码器（在 T 维上不下采样，只对 H,W 做 2x 下采样）。
    输入: [B,C,T,H,W]
    输出: [B,Cd,T,H',W'] 并在 self.skips 暴露多尺度特征
    """
    def __init__(self, base_channels: int = 32, in_channels: int | None = None, depth: int = 4):
        super().__init__()
        self.base_channels = base_channels
        self.depth = depth
        self.in_channels = in_channels  # 可为 None，懒初始化时从输入推断

        self.stem = None
        self.downs = nn.ModuleList()

    def _lazy_build(self, c_in: int):
        chs: List[int] = [self.base_channels * (2**i) for i in range(self.depth)]
        self.stem = conv_block(c_in, chs[0])
        for i in range(self.depth - 1):
            self.downs.append(down_block(chs[i], chs[i+1]))

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x5.shape
        if self.stem is None:
            self._lazy_build(self.in_channels or C)

        # -- 新增：确保懒构建子模块迁移到输入设备 --
        dev = x5.device
        if next(self.stem.parameters()).device != dev:
            self.stem = self.stem.to(dev)
        # ModuleList 不能整体 .to()，逐个迁移
        for i, m in enumerate(self.downs):
            if next(m.parameters()).device != dev:
                self.downs[i] = m.to(dev)

        # -- 新增：统一精度转换的辅助函数（关闭autocast，按权重dtype计算后再转回输入dtype） --
        def _run_block(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            in_dtype = x.dtype
            # 找一个代表性的参数拿到权重dtype
            try:
                w_dtype = next(p for p in mod.parameters()).dtype
            except StopIteration:
                w_dtype = x.dtype  # 以防极端情况
            if torch.is_autocast_enabled():
                with torch.amp.autocast('cuda', enabled=False):
                    y = mod(x.to(dtype=w_dtype))
            else:
                y = mod(x.to(dtype=w_dtype))
            return y.to(in_dtype)

        skips = []
        x = _run_block(self.stem, x5)      # -> C0
        skips.append(x)
        for down in self.downs:
            x = _run_block(down, x)        # H,W 下采样
            skips.append(x)

        self.skips = skips  # [stem, d1, d2, ...]（保持与输入相同的dtype）
        return x

