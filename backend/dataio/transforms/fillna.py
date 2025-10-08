# backend/dataio/transforms/fillna.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from .base import Transform
from ..schema import ArraySample

@dataclass
class FillNaNTransform(Transform):
    """
    对 sample.frames 进行 NaN 填充：
      - method="value": 用给定 value 填充（如 0 或 -273.15）
      - method="mean":  按通道均值填充（均值只在首次调用时统计一次并缓存）
    适用 frames: [K,H,W,C]，统计均值时会在所有 K,H,W 上聚合到 [C]
    """
    method: str = "value"        # "value" | "mean"
    value: float = 0.0
    _mean: Optional[np.ndarray] = None  # [C]

    def __call__(self, sample: ArraySample) -> ArraySample:
        frames = sample.frames
        is_torch = isinstance(frames, torch.Tensor)
        if is_torch:
            dev = frames.device
            frames = frames.detach().cpu().numpy()

        if self.method == "value":
            frames = np.nan_to_num(frames, nan=self.value, posinf=self.value, neginf=self.value)
        elif self.method == "mean":
            if self._mean is None:
                K,H,W,C = frames.shape
                flat = frames.reshape(-1, C)  # [KHW, C]
                # 仅用有限值参与均值
                mean = np.zeros((C,), dtype=frames.dtype)
                for c in range(C):
                    v = flat[:, c]
                    v = v[np.isfinite(v)]
                    mean[c] = v.mean() if v.size > 0 else 0.0
                self._mean = mean
            # 广播到 [K,H,W,C]
            frames = np.where(np.isfinite(frames), frames, self._mean.reshape(1,1,1,-1))
        else:
            raise ValueError(f"Unknown fillna method: {self.method}")

        if is_torch:
            sample.frames = torch.from_numpy(frames).to(dev).to(dtype=torch.float32)
        else:
            sample.frames = frames
        return sample
