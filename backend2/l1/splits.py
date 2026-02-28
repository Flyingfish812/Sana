from __future__ import annotations

from typing import Dict, List, Tuple
import math
import random


def _normalize_ratios(ratios: Dict[str, float]) -> Tuple[float, float, float]:
    """将 train/val/test 比例归一化，保证三者和为 1。"""
    train = float(ratios.get("train", 0.8))
    val = float(ratios.get("val", 0.1))
    test = float(ratios.get("test", 0.1))
    s = train + val + test
    if s <= 0:
        raise ValueError("split ratios sum must be > 0")
    return train / s, val / s, test / s


def split_indices(
    *,
    shape5d: Tuple[int, int, int, int, int],
    strategy: str = "temporal",
    unit: str = "frame",
    ratios: Dict[str, float] | None = None,
    seed: int = 123,
) -> Dict[str, List[int]]:
    """按 strategy 与 unit 生成 train/val/test 的样本索引。"""
    n_size, t_size, _, _, _ = shape5d
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    r_train, r_val, _ = _normalize_ratios(ratios)
    rng = random.Random(int(seed))

    if unit == "sequence":
        ids = list(range(n_size))
        if strategy == "random":
            rng.shuffle(ids)
        k1 = math.floor(len(ids) * r_train)
        k2 = math.floor(len(ids) * (r_train + r_val))
        return {"train": ids[:k1], "val": ids[k1:k2], "test": ids[k2:]}

    if unit != "frame":
        raise ValueError(f"Unsupported split unit '{unit}', expected 'frame' or 'sequence'")

    if strategy == "temporal":
        train_ids: List[int] = []
        val_ids: List[int] = []
        test_ids: List[int] = []
        for n in range(n_size):
            t_ids = list(range(t_size))
            k1 = math.floor(t_size * r_train)
            k2 = math.floor(t_size * (r_train + r_val))
            train_ids.extend([n * t_size + t for t in t_ids[:k1]])
            val_ids.extend([n * t_size + t for t in t_ids[k1:k2]])
            test_ids.extend([n * t_size + t for t in t_ids[k2:]])
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    if strategy == "random":
        ids = list(range(n_size * t_size))
        rng.shuffle(ids)
        k1 = math.floor(len(ids) * r_train)
        k2 = math.floor(len(ids) * (r_train + r_val))
        return {"train": ids[:k1], "val": ids[k1:k2], "test": ids[k2:]}

    raise ValueError(f"Unsupported split strategy '{strategy}', expected 'random' or 'temporal'")
