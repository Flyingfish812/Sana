from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .probe import ProbeController


def _conv_block_2d(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
    )


def _down_block_2d(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        _conv_block_2d(in_channels, out_channels),
    )


class _UpBlock2d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.align = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.fuse = _conv_block_2d(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(skip.shape[-2], skip.shape[-1]), mode="nearest")
        x = self.align(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class LegacyUNet(nn.Module):
    """Migrated legacy UNet (encoder -> identity propagator -> decoder -> pixel head)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [int(base_channels) * (2**i) for i in range(int(depth))]
        self.stem = _conv_block_2d(int(in_channels), channels[0])
        self.downs = nn.ModuleList(
            [_down_block_2d(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.ups = nn.ModuleList()
        in_c = channels[-1]
        for i in range(len(channels) - 2, -1, -1):
            self.ups.append(_UpBlock2d(in_c, channels[i], channels[i]))
            in_c = channels[i]

        self.head = nn.Conv2d(channels[0], int(out_channels), kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _tap(probe: Optional[ProbeController], name: str, x: torch.Tensor) -> torch.Tensor:
        return probe.apply(name, x) if probe is not None else x

    def forward(self, x: torch.Tensor, probe: Optional[ProbeController] = None) -> torch.Tensor:
        skips = []
        h = self._tap(probe, "enc.stage1.out", self.stem(x))
        skips.append(h)

        for i, down in enumerate(self.downs, start=2):
            h = self._tap(probe, f"enc.stage{i}.out", down(h))
            skips.append(h)

        h = self._tap(probe, "propagator.out", h)

        for i, up in enumerate(self.ups, start=1):
            skip = self._tap(probe, f"skip.s{i}", skips[-(i + 1)])
            h = self._tap(probe, f"dec.stage{i}.out", up(h, skip))

        return self._tap(probe, "head.out", self.head(h))
