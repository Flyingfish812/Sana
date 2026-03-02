from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np


def _to_frame_indices(idx: int, t_size: int) -> Tuple[int, int]:
    """将展平 frame 索引还原为 (sequence_id, time_id)。"""
    n = idx // t_size
    t = idx % t_size
    return n, t


def _iter_train_frames(array5d: np.ndarray, train_indices: List[int], unit: str) -> Iterable[np.ndarray]:
    """按 train 索引流式迭代 frame 视图，避免构造大中间数组。"""
    n_size, t_size = int(array5d.shape[0]), int(array5d.shape[1])
    if unit == "sequence":
        for seq_id in train_indices:
            seq = int(seq_id)
            if seq < 0 or seq >= n_size:
                continue
            for t in range(t_size):
                yield array5d[seq, t]
        return

    if unit == "frame":
        total = n_size * t_size
        for frame_idx in train_indices:
            idx = int(frame_idx)
            if idx < 0 or idx >= total:
                continue
            n, t = _to_frame_indices(idx, t_size)
            yield array5d[n, t]
        return

    raise ValueError(f"unsupported unit '{unit}'")


def compute_train_stats(
    array5d: np.ndarray,
    train_indices: List[int],
    *,
    unit: str,
    method: str = "zscore",
) -> Dict[str, object]:
    """基于训练切分样本流式计算通道级归一化统计量。"""
    if array5d.ndim != 5:
        raise ValueError(f"expected array5d [N,T,H,W,C], got shape={array5d.shape}")

    c_size = int(array5d.shape[-1])
    method = str(method).lower()
    frame_iter = _iter_train_frames(array5d, train_indices, unit)

    if method == "zscore":
        sum_v = np.zeros((c_size,), dtype=np.float64)
        sum_sq_v = np.zeros((c_size,), dtype=np.float64)
        count_v = np.zeros((c_size,), dtype=np.int64)
        seen = 0

        for frame in frame_iter:
            if frame.shape[-1] != c_size:
                raise RuntimeError("channel dimension mismatch while computing stats")
            seen += 1
            finite = np.isfinite(frame)
            count_v += np.sum(finite, axis=(0, 1), dtype=np.int64)

            safe = np.where(finite, frame, 0.0)
            safe64 = safe.astype(np.float64, copy=False)
            sum_v += np.sum(safe64, axis=(0, 1), dtype=np.float64)
            sum_sq_v += np.sum(np.square(safe64, dtype=np.float64), axis=(0, 1), dtype=np.float64)

        if seen == 0:
            raise ValueError(f"empty or invalid train_indices for {unit} unit")

        mean = np.divide(sum_v, count_v, out=np.zeros_like(sum_v), where=count_v > 0)
        ex2 = np.divide(sum_sq_v, count_v, out=np.zeros_like(sum_sq_v), where=count_v > 0)
        var = np.maximum(ex2 - np.square(mean), 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-12, 1.0, std)
        return {
            "method": "zscore",
            "channels": int(c_size),
            "mean": mean.astype(np.float64).tolist(),
            "std": std.astype(np.float64).tolist(),
        }

    if method == "minmax":
        min_v = np.full((c_size,), np.inf, dtype=np.float64)
        max_v = np.full((c_size,), -np.inf, dtype=np.float64)
        seen = 0

        for frame in frame_iter:
            if frame.shape[-1] != c_size:
                raise RuntimeError("channel dimension mismatch while computing stats")
            seen += 1
            finite = np.isfinite(frame)
            frame_min = np.min(np.where(finite, frame, np.inf), axis=(0, 1))
            frame_max = np.max(np.where(finite, frame, -np.inf), axis=(0, 1))
            min_v = np.minimum(min_v, frame_min)
            max_v = np.maximum(max_v, frame_max)

        if seen == 0:
            raise ValueError(f"empty or invalid train_indices for {unit} unit")

        no_valid = ~np.isfinite(min_v) | ~np.isfinite(max_v)
        min_v = np.where(no_valid, 0.0, min_v)
        max_v = np.where(no_valid, 0.0, max_v)
        scale = np.where((max_v - min_v) < 1e-12, 1.0, (max_v - min_v))
        return {
            "method": "minmax",
            "channels": int(c_size),
            "min": min_v.astype(np.float64).tolist(),
            "max": max_v.astype(np.float64).tolist(),
            "scale": scale.astype(np.float64).tolist(),
        }

    raise ValueError(f"unsupported normalization method '{method}', expected 'zscore' or 'minmax'")


def normalize_block(block: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    """对一个 NTHWC 子块执行按通道归一化并返回 float32 结果。"""
    if block.ndim != 5:
        raise ValueError(f"expected block [N,T,H,W,C], got shape={block.shape}")

    method = str(stats.get("method", "zscore")).lower()
    channels = int(block.shape[-1])

    if method == "zscore":
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, 1, 1, -1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, 1, 1, -1)
        if mean.shape[-1] != channels or std.shape[-1] != channels:
            raise ValueError("stats channel count mismatch for zscore")
        out = (block.astype(np.float32, copy=False) - mean) / std
    elif method == "minmax":
        min_v = np.asarray(stats["min"], dtype=np.float32).reshape(1, 1, 1, 1, -1)
        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(1, 1, 1, 1, -1)
        if min_v.shape[-1] != channels or scale.shape[-1] != channels:
            raise ValueError("stats channel count mismatch for minmax")
        out = (block.astype(np.float32, copy=False) - min_v) / scale
    else:
        raise ValueError(f"unsupported normalization method '{method}', expected 'zscore' or 'minmax'")

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32, copy=False)


def write_normalized_array5d_memmap(
    array5d: np.ndarray,
    stats: Dict[str, object],
    out_path: Path,
    *,
    chunk_n: int = 1,
) -> None:
    """分块归一化并写入标准 .npy memmap，避免全量中间数组峰值内存。"""
    if array5d.ndim != 5:
        raise ValueError(f"expected array5d [N,T,H,W,C], got shape={array5d.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_size = int(array5d.shape[0])
    chunk_n = max(int(chunk_n), 1)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=array5d.shape)

    for n0 in range(0, n_size, chunk_n):
        n1 = min(n0 + chunk_n, n_size)
        block = array5d[n0:n1]
        out[n0:n1] = normalize_block(block, stats)
        out.flush()

    out.flush()
