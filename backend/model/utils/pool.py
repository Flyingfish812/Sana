from __future__ import annotations
from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

_Number = Union[int, float]
_KS = Union[_Number, Tuple[_Number, _Number], Tuple[_Number, _Number, _Number]]

def _to_hw(v):
    # (T,H,W) or (H,W) or int -> (H,W)
    if isinstance(v, (tuple, list)):
        return (v[-2], v[-1]) if len(v) == 3 else (v[0], v[1])
    return (v, v)

def _to_thw(v):
    # int or (H,W) -> (1,H,W); (T,H,W) -> as is
    if isinstance(v, (tuple, list)):
        return v if len(v) == 3 else (1, v[0], v[1])
    return (1, v, v)

class PoolAwareTime(nn.Module):
    """
    保确定性：
      - 4D [B,C,H,W]         -> MaxPool2d（或 AvgPool2d，见 mode2d）
      - 5D [B,C,1,H,W] (T=1) -> MaxPool2d（或 AvgPool2d）
      - 5D [B,C,T,H,W] T>1   -> AvgPool3d（避免 MaxPool3d 的不确定反向）
    默认：2D 用 max，3D 用 avg（确定）。
    """
    def __init__(self,
                 kernel_size: _KS = 2,
                 stride: _KS | None = None,
                 padding: _KS = 0,
                 ceil_mode: bool = False,
                 mode2d: str = "max",   # "max" 或 "avg"
                 mode3d: str = "avg"):  # 3D 强烈建议 avg
        super().__init__()
        self.ks = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        assert mode2d in ("max", "avg")
        assert mode3d in ("avg", )  # 3D max 不确定，这里只允许 avg
        self.mode2d = mode2d
        self.mode3d = mode3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:  # [B,C,H,W]
            if self.mode2d == "max":
                return F.max_pool2d(x, kernel_size=_to_hw(self.ks), stride=_to_hw(self.stride),
                                    padding=_to_hw(self.padding), ceil_mode=self.ceil_mode)
            else:
                return F.avg_pool2d(x, kernel_size=_to_hw(self.ks), stride=_to_hw(self.stride),
                                    padding=_to_hw(self.padding), ceil_mode=self.ceil_mode)
        if x.ndim == 5:  # [B,C,T,H,W]
            if x.shape[2] == 1:
                x2d = x.squeeze(2)
                if self.mode2d == "max":
                    y2d = F.max_pool2d(x2d, kernel_size=_to_hw(self.ks), stride=_to_hw(self.stride),
                                       padding=_to_hw(self.padding), ceil_mode=self.ceil_mode)
                else:
                    y2d = F.avg_pool2d(x2d, kernel_size=_to_hw(self.ks), stride=_to_hw(self.stride),
                                       padding=_to_hw(self.padding), ceil_mode=self.ceil_mode)
                return y2d.unsqueeze(2)
            # T>1：用 AvgPool3d（确定性）
            return F.avg_pool3d(x, kernel_size=_to_thw(self.ks), stride=_to_thw(self.stride),
                                padding=_to_thw(self.padding), ceil_mode=self.ceil_mode)
        raise ValueError(f"PoolAwareTime: unsupported ndim={x.ndim}")
