from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .probe import ProbeController


class ConvStage(nn.Module):
    """由若干卷积+GELU 组成的基础卷积阶段模块。"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, convs_per_stage: int = 2):
        """构建一个卷积阶段，卷积次数受 convs_per_stage 控制。"""
        super().__init__()
        layers = []
        pad = kernel_size // 2
        c_in = in_ch
        for _ in range(max(1, min(2, int(convs_per_stage)))):
            layers.append(nn.Conv2d(c_in, out_ch, kernel_size=kernel_size, stride=1, padding=pad))
            layers.append(nn.GELU())
            c_in = out_ch
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行当前阶段前向传播。"""
        return self.block(x)


class BaselineUNet(nn.Module):
    """
    Fixed baseline UNet:
    - Encoder stage kernels: 9, 7, 5, 3, 1
    - Decoder stage kernels: 1, 3, 5, 7, 9
    - Downsample: stride=2 conv (consistent across stages)
    - Skip connection: concat (fixed)
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, convs_per_stage: int = 2):
        """初始化固定结构的基线 UNet。"""
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]

        self.enc1 = ConvStage(in_channels, widths[0], kernel_size=9, convs_per_stage=convs_per_stage)
        self.down1 = nn.Conv2d(widths[0], widths[1], kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvStage(widths[1], widths[1], kernel_size=7, convs_per_stage=convs_per_stage)

        self.down2 = nn.Conv2d(widths[1], widths[2], kernel_size=3, stride=2, padding=1)
        self.enc3 = ConvStage(widths[2], widths[2], kernel_size=5, convs_per_stage=convs_per_stage)

        self.down3 = nn.Conv2d(widths[2], widths[3], kernel_size=3, stride=2, padding=1)
        self.enc4 = ConvStage(widths[3], widths[3], kernel_size=3, convs_per_stage=convs_per_stage)

        self.down4 = nn.Conv2d(widths[3], widths[4], kernel_size=3, stride=2, padding=1)
        self.enc5 = ConvStage(widths[4], widths[4], kernel_size=1, convs_per_stage=convs_per_stage)

        self.dec1 = ConvStage(widths[4], widths[4], kernel_size=1, convs_per_stage=convs_per_stage)
        self.up1 = nn.ConvTranspose2d(widths[4], widths[3], kernel_size=2, stride=2)

        self.dec2 = ConvStage(widths[3] * 2, widths[3], kernel_size=3, convs_per_stage=convs_per_stage)
        self.up2 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)

        self.dec3 = ConvStage(widths[2] * 2, widths[2], kernel_size=5, convs_per_stage=convs_per_stage)
        self.up3 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)

        self.dec4 = ConvStage(widths[1] * 2, widths[1], kernel_size=7, convs_per_stage=convs_per_stage)
        self.up4 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)

        self.dec5 = ConvStage(widths[0] * 2, widths[0], kernel_size=9, convs_per_stage=convs_per_stage)
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    @staticmethod
    def _tap(probe: Optional[ProbeController], name: str, x: torch.Tensor) -> torch.Tensor:
        """若启用 probe，则在指定命名节点执行探针逻辑。"""
        return probe.apply(name, x) if probe is not None else x

    @staticmethod
    def _align_hw(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """将 src 的空间尺寸对齐到 ref，避免奇数尺寸下跳连拼接失败。"""
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, probe: Optional[ProbeController] = None) -> torch.Tensor:
        """执行 UNet 前向并在关键层支持 probe 采样。"""
        e1 = self._tap(probe, "enc.stage1.out", self.enc1(x))
        e2 = self._tap(probe, "enc.stage2.out", self.enc2(self.down1(e1)))
        e3 = self._tap(probe, "enc.stage3.out", self.enc3(self.down2(e2)))
        e4 = self._tap(probe, "enc.stage4.out", self.enc4(self.down3(e3)))
        e5 = self._tap(probe, "enc.stage5.out", self.enc5(self.down4(e4)))

        d1 = self._tap(probe, "dec.stage1.out", self.dec1(e5))

        s4 = self._tap(probe, "skip.s4", e4)
        u1 = self._align_hw(self.up1(d1), s4)
        d2_in = torch.cat([u1, s4], dim=1)
        d2 = self._tap(probe, "dec.stage2.out", self.dec2(d2_in))

        s3 = self._tap(probe, "skip.s3", e3)
        u2 = self._align_hw(self.up2(d2), s3)
        d3_in = torch.cat([u2, s3], dim=1)
        d3 = self._tap(probe, "dec.stage3.out", self.dec3(d3_in))

        s2 = self._tap(probe, "skip.s2", e2)
        u3 = self._align_hw(self.up3(d3), s2)
        d4_in = torch.cat([u3, s2], dim=1)
        d4 = self._tap(probe, "dec.stage4.out", self.dec4(d4_in))

        s1 = self._tap(probe, "skip.s1", e1)
        u4 = self._align_hw(self.up4(d4), s1)
        d5_in = torch.cat([u4, s1], dim=1)
        d5 = self._tap(probe, "dec.stage5.out", self.dec5(d5_in))

        out = self._tap(probe, "head.out", self.head(d5))
        return out
