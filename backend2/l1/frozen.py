from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


FramePair = Tuple[int, int]


def _read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_l1_array_and_splits(l1_dir: str | Path) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """从 L1 冻结目录加载 mmap 数组、split 索引与元信息。"""
    base = Path(l1_dir)
    array_path = base / "array5d_norm.npy"
    manifest_path = base / "manifest.json"
    stats_path = base / "stats_train.json"

    if not array_path.exists():
        raise FileNotFoundError(f"missing L1 array: {array_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing L1 manifest: {manifest_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"missing L1 stats: {stats_path}")

    array_mmap = np.load(array_path, mmap_mode="r")
    splits = {
        "train": np.load(base / "splits" / "train.npy", mmap_mode="r"),
        "val": np.load(base / "splits" / "val.npy", mmap_mode="r"),
        "test": np.load(base / "splits" / "test.npy", mmap_mode="r"),
    }
    meta = {
        "manifest": _read_json(manifest_path),
        "stats": _read_json(stats_path),
        "l1_dir": str(base),
    }
    return array_mmap, splits, meta


def _to_pairs(shape5d: Sequence[int], split_indices: Sequence[int], unit: str, target_offset: int) -> List[FramePair]:
    n_size, t_size = int(shape5d[0]), int(shape5d[1])
    pairs: List[FramePair] = []
    if unit == "sequence":
        for n in split_indices:
            seq = int(n)
            if seq < 0 or seq >= n_size:
                continue
            for t in range(0, t_size - target_offset):
                pairs.append((seq, t))
        return pairs

    for idx in split_indices:
        frame_idx = int(idx)
        if frame_idx < 0 or frame_idx >= n_size * t_size:
            continue
        n = frame_idx // t_size
        t = frame_idx % t_size
        if t + target_offset < t_size:
            pairs.append((n, t))
    return pairs


class L1PairDataset(Dataset):
    """基于 L1 冻结数组的监督样本数据集。"""

    def __init__(self, array_mmap: np.ndarray, pairs: Sequence[FramePair], target_offset: int = 1):
        self.array_mmap = array_mmap
        self.pairs = list(pairs)
        self.target_offset = int(target_offset)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        n, t = self.pairs[idx]
        x_hwc = self.array_mmap[n, t]
        y_hwc = self.array_mmap[n, t + self.target_offset]

        mask_hwc = np.isfinite(x_hwc).astype(np.float32)
        x_hwc = np.nan_to_num(x_hwc, nan=0.0, posinf=0.0, neginf=0.0)
        y_hwc = np.nan_to_num(y_hwc, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(np.transpose(x_hwc, (2, 0, 1)).astype(np.float32))
        y = torch.from_numpy(np.transpose(y_hwc, (2, 0, 1)).astype(np.float32))
        mask = torch.from_numpy(np.transpose(mask_hwc, (2, 0, 1)).astype(np.float32))

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "n": torch.tensor(n, dtype=torch.int64),
            "t": torch.tensor(t, dtype=torch.int64),
        }


def build_dataloaders_from_l1(
    l1_dir: str | Path,
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    target_offset: int = 1,
    shuffle_train: bool = True,
) -> Dict[str, object]:
    """从 L1 冻结产物直接构建 train/val/test DataLoader。"""
    array_mmap, splits, meta = load_l1_array_and_splits(l1_dir)
    manifest = dict(meta["manifest"])
    shape5d = manifest["shape5d"]
    unit = str(dict(manifest.get("split", {})).get("unit", "frame"))

    train_pairs = _to_pairs(shape5d, splits["train"], unit, target_offset)
    val_pairs = _to_pairs(shape5d, splits["val"], unit, target_offset)
    test_pairs = _to_pairs(shape5d, splits["test"], unit, target_offset)

    train_ds = L1PairDataset(array_mmap, train_pairs, target_offset=target_offset)
    val_ds = L1PairDataset(array_mmap, val_pairs, target_offset=target_offset)
    test_ds = L1PairDataset(array_mmap, test_pairs, target_offset=target_offset)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return {
        "array_mmap": array_mmap,
        "splits": splits,
        "meta": meta,
        "pairs": {"train": train_pairs, "val": val_pairs, "test": test_pairs},
        "datasets": {"train": train_ds, "val": val_ds, "test": test_ds},
        "loaders": loaders,
    }
