from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def _to_frame_indices(idx: int, t_size: int) -> Tuple[int, int]:
    """将展平 frame 索引还原为 (sequence_id, time_id)。"""
    n = idx // t_size
    t = idx % t_size
    return n, t


def compute_train_stats(
    array5d: np.ndarray,
    train_indices: List[int],
    *,
    unit: str,
    method: str = "zscore",
) -> Dict[str, object]:
    """基于训练切分样本计算通道级归一化统计量。"""
    if array5d.ndim != 5:
        raise ValueError(f"expected array5d [N,T,H,W,C], got shape={array5d.shape}")

    n_size, t_size, _, _, c_size = array5d.shape
    method = str(method).lower()

    if unit == "sequence":
        valid_seq = [i for i in train_indices if 0 <= i < n_size]
        if not valid_seq:
            raise ValueError("empty or invalid train_indices for sequence unit")
        train = array5d[valid_seq]  # [Ntrain,T,H,W,C]
    elif unit == "frame":
        picks = [_to_frame_indices(i, t_size) for i in train_indices if 0 <= i < n_size * t_size]
        if not picks:
            raise ValueError("empty or invalid train_indices for frame unit")
        train = np.stack([array5d[n, t] for n, t in picks], axis=0)  # [M,H,W,C]
    else:
        raise ValueError(f"unsupported unit '{unit}'")

    if train.shape[-1] != c_size:
        raise RuntimeError("channel dimension mismatch while computing stats")

    axes = tuple(i for i in range(train.ndim) if i != train.ndim - 1)
    if method == "zscore":
        mean = np.nanmean(train, axis=axes)
        std = np.nanstd(train, axis=axes)
        std = np.where(std < 1e-12, 1.0, std)
        return {
            "method": "zscore",
            "channels": int(c_size),
            "mean": mean.astype(np.float64).tolist(),
            "std": std.astype(np.float64).tolist(),
        }

    if method == "minmax":
        min_v = np.nanmin(train, axis=axes)
        max_v = np.nanmax(train, axis=axes)
        scale = np.where((max_v - min_v) < 1e-12, 1.0, (max_v - min_v))
        return {
            "method": "minmax",
            "channels": int(c_size),
            "min": min_v.astype(np.float64).tolist(),
            "max": max_v.astype(np.float64).tolist(),
            "scale": scale.astype(np.float64).tolist(),
        }

    raise ValueError(f"unsupported normalization method '{method}', expected 'zscore' or 'minmax'")
