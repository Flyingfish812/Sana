# backend/dataio/transforms/normalize.py
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from .base import Transform
from ..schema import ArraySample

@dataclass
class Normalizer:
    """
    简化版 Normalizer：目前支持 zscore/minmax。
    注：统计量按通道聚合（对 frames 的最后一维 C 维）
    """
    method: str = "zscore"
    eps: float = 1e-8
    mean: Optional[np.ndarray] = None  # [C]
    std: Optional[np.ndarray] = None   # [C]
    minv: Optional[np.ndarray] = None  # [C]
    maxv: Optional[np.ndarray] = None  # [C]

    def fit(self, frames: np.ndarray):
        # frames: [K,H,W,C] 或 [N,K,H,W,C]（采样前期这里一般是 [K,H,W,C]）
        if frames.ndim == 5:  # 合并 N
            frames = frames.reshape(-1, *frames.shape[-3:])
        C = frames.shape[-1]
        if self.method == "zscore":
            self.mean = frames.reshape(-1, C).mean(axis=0)
            self.std = frames.reshape(-1, C).std(axis=0) + self.eps
        elif self.method == "minmax":
            self.minv = frames.reshape(-1, C).min(axis=0)
            self.maxv = frames.reshape(-1, C).max(axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def transform(self, frames: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            assert self.mean is not None and self.std is not None
            return (frames - self.mean) / self.std
        else:
            assert self.minv is not None and self.maxv is not None
            rng = (self.maxv - self.minv + self.eps)
            return (frames - self.minv) / rng

    def inverse(self, frames: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            assert self.mean is not None and self.std is not None
            return frames * self.std + self.mean
        else:
            assert self.minv is not None and self.maxv is not None
            rng = (self.maxv - self.minv + 1e-8)
            return frames * rng + self.minv

class NormalizeTransform(Transform):
    """
    将 frames 做归一化。默认在第一次调用时“拟合 + 变换”；
    若想离线拟合，可在外部先 normalizer.fit(...) 再构建本 Transform。
    """
    def __init__(self, normalizer: Optional[Normalizer] = None, method: str = "zscore"):
        self.normalizer = normalizer or Normalizer(method=method)

    def __call__(self, sample: ArraySample) -> ArraySample:
        if self.normalizer.mean is None and self.normalizer.std is None \
           and self.normalizer.minv is None and self.normalizer.maxv is None:
            self.normalizer.fit(sample.frames)
        sample.frames = self.normalizer.transform(sample.frames)
        return sample

class InverseNormalizeTransform(Transform):
    """
    将 frames 从“标准化空间”逆变换回原值空间。
    - 若 frames 是 torch.Tensor，将在 GPU/CPU 上就地计算（不改变 dtype）
    - 依赖同一 Normalizer 的统计量
    """
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def __call__(self, sample: ArraySample) -> ArraySample:
        frames = sample.frames
        if isinstance(frames, torch.Tensor):
            # 转到 numpy 做逆变换，再回到 torch（避免重复实现）
            dev = frames.device
            np_frames = frames.detach().cpu().numpy()
            np_inv = self.normalizer.inverse(np_frames)
            sample.frames = torch.from_numpy(np_inv).to(dev).to(frames.dtype)
        else:
            sample.frames = self.normalizer.inverse(frames)
        return sample