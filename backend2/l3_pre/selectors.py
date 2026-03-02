from __future__ import annotations

from typing import List

import numpy as np


def compute_residual_ch0(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.shape[1] < 1:
        raise ValueError(f"channel dim invalid: {gt.shape}")
    return pred[:, 0] - gt[:, 0]


def rmse_per_sample(residual_hw: np.ndarray) -> np.ndarray:
    if residual_hw.ndim != 3:
        raise ValueError(f"expected residual [N,H,W], got {residual_hw.shape}")
    return np.sqrt(np.mean(np.square(residual_hw), axis=(1, 2)))


def select_samples(strategy: str, k: int, rmse: np.ndarray, seed: int) -> List[int]:
    n = int(rmse.shape[0])
    if n <= 0:
        return []
    k = max(0, min(k, n))
    if k == 0:
        return []

    order_worst = np.argsort(-rmse)
    rng = np.random.default_rng(seed)

    if strategy == "worst_k":
        return [int(v) for v in order_worst[:k]]
    if strategy == "random_k":
        return [int(v) for v in rng.choice(n, size=k, replace=False)]
    if strategy == "mixed":
        k_worst = k // 2
        chosen = [int(v) for v in order_worst[:k_worst]]
        remaining_pool = [int(v) for v in range(n) if v not in set(chosen)]
        k_random = k - len(chosen)
        if k_random > 0 and remaining_pool:
            picks = rng.choice(len(remaining_pool), size=min(k_random, len(remaining_pool)), replace=False)
            chosen.extend([remaining_pool[int(i)] for i in picks])
        return chosen[:k]

    raise ValueError(f"unsupported sample_strategy: {strategy}")


def channel_energy(feat_chw: np.ndarray) -> np.ndarray:
    return np.mean(np.square(feat_chw), axis=(1, 2))


def select_channels(feat_chw: np.ndarray, channel_k: int, channel_select: str) -> List[int]:
    c_total = int(feat_chw.shape[0])
    k = max(0, min(channel_k, c_total))
    if k == 0:
        return []
    if channel_select == "first_k":
        return list(range(k))
    if channel_select == "energy_topk":
        e = channel_energy(feat_chw)
        order = np.argsort(-e)
        return [int(v) for v in order[:k]]
    raise ValueError(f"unsupported channel_select: {channel_select}")
