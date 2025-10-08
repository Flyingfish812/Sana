# backend/dataio/transforms/to_tensor.py
from __future__ import annotations
import numpy as np
import torch
from ..schema import ArraySample
from .base import Transform

class ToTensorTransform(Transform):
    """
    将 ArraySample.frames 从 numpy 转为 float32 的 torch.Tensor（保持 [K,H,W,C]）。
    注：仅用于需要在 Transform 阶段做 torch 运算的场景；若只在 collate 中转 tensor，可不启用。
    """
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, sample: ArraySample) -> ArraySample:
        if isinstance(sample.frames, np.ndarray):
            sample.frames = torch.from_numpy(sample.frames).to(self.dtype)
        return sample
