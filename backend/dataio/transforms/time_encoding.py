# backend/dataio/transforms/time_encoding.py
from __future__ import annotations
import numpy as np
from ..schema import ArraySample
from .base import Transform

class AddTimeEncodingTransform(Transform):
    """
    给样本加入简单的正弦时间编码（写入 frames 的附加通道）：
      - 对 K 帧分别计算 sin/cos(2π * t / T) 并拼到 C 维（每帧增加 2 通道）
      - 注意：此处只依赖样本内的相对索引（0..K-1），不改动 meta.times
    """
    def __init__(self):
        pass

    def __call__(self, sample: ArraySample) -> ArraySample:
        frames = sample.frames  # [K,H,W,C]
        K, H, W, C = frames.shape
        t_idx = np.arange(K, dtype=np.float32).reshape(K, 1, 1, 1)
        T = max(K - 1, 1)
        pe_sin = np.sin(2 * np.pi * t_idx / T)
        pe_cos = np.cos(2 * np.pi * t_idx / T)
        pe = np.concatenate([np.broadcast_to(pe_sin, (K, H, W, 1)),
                             np.broadcast_to(pe_cos, (K, H, W, 1))], axis=-1)
        sample.frames = np.concatenate([frames, pe], axis=-1)  # [K,H,W,C+2]
        return sample
