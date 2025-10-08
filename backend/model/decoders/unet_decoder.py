# backend/model/decoders/unet_decoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from ..base_components.decoder_base import BaseDecoder
from ..factory import register

_Number = Union[int, float]
_SF = Union[_Number, Tuple[_Number, _Number], Tuple[_Number, _Number, _Number]]

class UpsampleAwareTime(nn.Module):
    """
    T=1 的 5D 输入: 退回逐帧 2D bilinear（确定性）；
    T>1 的 5D 输入: 用 3D 'nearest'（确定性，避免 trilinear）；
    4D 输入: 标准 2D bilinear。
    """
    def __init__(self, scale_factor: _SF = 2, mode2d: str = "bilinear",
                 mode3d: str = "nearest", align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode2d = mode2d
        self.mode3d = mode3d
        self.align_corners = align_corners

    def _sf2d(self):
        sf = self.scale_factor
        if isinstance(sf, (tuple, list)) and len(sf) == 3:
            # (T,H,W) -> (H,W)
            return (sf[1], sf[2])
        return sf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            # [B,C,T,H,W]
            if x.shape[2] == 1:
                x2d = x.squeeze(2)  # -> [B,C,H,W]
                out2d = F.interpolate(x2d, scale_factor=self._sf2d(),
                                      mode=self.mode2d, align_corners=self.align_corners)
                return out2d.unsqueeze(2)  # -> [B,C,1,H,W]
            # T>1：用确定性的 3D nearest
            return F.interpolate(x, scale_factor=self.scale_factor,
                                 mode=self.mode3d, align_corners=self.align_corners)
        elif x.ndim == 4:
            # [B,C,H,W]
            return F.interpolate(x, scale_factor=self._sf2d(),
                                 mode=self.mode2d, align_corners=self.align_corners)
        raise ValueError(f"UpsampleAwareTime: unsupported ndim={x.ndim}")


def conv_block(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(1,k,k), stride=(1,s,s), padding=(0,p,p), bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=(1,k,k), stride=(1,1,1), padding=(0,p,p), bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = UpsampleAwareTime(scale_factor=(1,2,2), mode2d="bilinear", mode3d="nearest", align_corners=False)
        self.conv = conv_block(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # 对齐空间尺寸（因 rounding 可能差 1）
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = nn.functional.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

@register("decoder", "UNetBase")
def UNetDecoder(base_channels: int = 32, out_channels: int | None = None):
    return _UNetDecoder(base_channels=base_channels, out_channels=out_channels)

class _UNetDecoder(BaseDecoder):
    """
    与上面的 UNetEncoder 对偶：多级 up + concat skip。
    需要 encoder.skips: [stem, d1, d2, ..., d{depth-1}]
    """
    def __init__(self, base_channels: int = 32, out_channels: int | None = None):
        super().__init__()
        self.base_channels = base_channels
        self.out_channels = out_channels

        self.ups = nn.ModuleList()
        self.proj = None  # 输出投影在 Head 里做，这里输出高通道特征

    def _lazy_build(self, skips: List[torch.Tensor]):
        depth = len(skips)
        chs = [self.base_channels * (2**i) for i in range(depth)]
        # bottleneck 通道数 = 最深层通道
        in_c = chs[-1]
        for i in range(depth - 2, -1, -1):
            self.ups.append(UpBlock(in_c, chs[i], chs[i]))
            in_c = chs[i]

    def forward(self, x5: torch.Tensor, skips=None) -> torch.Tensor:
        assert isinstance(skips, (list, tuple)) and len(skips) >= 1, "UNetDecoder requires encoder.skips (list)."
        if len(self.ups) == 0:
            self._lazy_build(skips)
        x = x5
        # 倒序使用 skip：最后一层与倒数第二层 skip 拼接……
        for i, up in enumerate(self.ups):
            skip = skips[-(i+2)]  # 对应 d_{end-i-1}
            x = up(x, skip)
        return x  # 输出特征通道为 base_channels
