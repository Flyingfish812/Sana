# backend/dataio/transforms/nanmask.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from ..schema import ArraySample
from .base import Transform

@dataclass
class CaptureNaNMaskTransform(Transform):
    """
    在“填充 NaN 之前”捕获 NaN 的分布，并写入 sample.meta.attrs["nan_mask"]。
    - target_only=True: 仅捕获目标帧（frames[-1]）
    - reduce_channel='any': 若为多通道，按 C 维做 any（任一通道为 NaN 即标记为 NaN）
      可按需改成 'all' 或特定索引
    生成的 nan_mask 为 float32 的 [H,W]，取值 1.0 (NaN) / 0.0 (finite)。
    """
    target_only: bool = True
    reduce_channel: str = "any"   # "any" | "all"

    def __call__(self, sample: ArraySample) -> ArraySample:
        frames = sample.frames  # [K,H,W,C]
        # 保证在 numpy 上做 NaN 检测
        if isinstance(frames, torch.Tensor):
            frames_np = frames.detach().cpu().numpy()
        else:
            frames_np = frames

        if self.target_only:
            img = frames_np[-1]              # [H,W,C]
        else:
            # 对所有帧聚合：只要某帧某像素某通道为 NaN，就认为该像素是 NaN
            # [K,H,W,C] -> 在 K 维做 any
            img = np.any(~np.isfinite(frames_np) == True, axis=0).astype(np.float32)  # [H,W,C] bool/float
            # 为了复用通道逻辑，下面仍按 C 维聚合
        # 现在 img 是 [H,W,C]
        finite_c = np.isfinite(img) if img.ndim == 3 else np.isfinite(img[..., None])

        if self.reduce_channel == "any":
            finite_2d = np.all(finite_c, axis=-1) if img.ndim == 3 else finite_c[..., 0]
        elif self.reduce_channel == "all":
            # “所有通道都 finite 才算 finite”
            finite_2d = np.all(finite_c, axis=-1) if img.ndim == 3 else finite_c[..., 0]
        else:
            # 默认等价于 any
            finite_2d = np.all(finite_c, axis=-1) if img.ndim == 3 else finite_c[..., 0]

        nan_mask = (~finite_2d).astype(np.float32)  # 1=NaN, 0=finite
        attrs = sample.meta.attrs or {}
        attrs["nan_mask"] = nan_mask  # [H,W] float32
        sample.meta.attrs = attrs
        return sample
