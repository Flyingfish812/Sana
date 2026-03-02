from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from backend2.l1 import load_l1_array_and_splits

from .artifact_io import ArtifactManager
from .sparse_input import SparseInputConfig, apply_sparse_sampling_1nn, build_fixed_points_mask, sample_noise_seed


FramePair = Tuple[int, int]


class IndexedDataset(Dataset):
    """最小索引数据集：仅做索引映射，不复制底层数组。"""

    def __init__(self, array_mmap: np.ndarray, indices: Sequence[int], unit: str, shape5d: Sequence[int]):
        """使用 mmap 数组与切分索引构造数据集。"""
        self.array_mmap = array_mmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.unit = str(unit)
        self.shape5d = tuple(int(x) for x in shape5d)
        if len(self.shape5d) != 5:
            raise ValueError(f"expected shape5d length=5, got {self.shape5d}")
        if self.unit not in ("frame", "sequence"):
            raise ValueError(f"unsupported unit '{self.unit}'")

    def __len__(self) -> int:
        """返回索引数量。"""
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        """按 unit 将 split 索引映射到底层数组视图。"""
        raw = int(self.indices[idx])
        n_size, t_size, _, _, _ = self.shape5d
        if self.unit == "frame":
            if raw < 0 or raw >= n_size * t_size:
                raise IndexError(f"frame index out of range: {raw}")
            n = raw // t_size
            t = raw % t_size
            return {"n": n, "t": t, "frame": self.array_mmap[n, t]}

        if raw < 0 or raw >= n_size:
            raise IndexError(f"sequence index out of range: {raw}")
        n = raw
        return {"n": n, "sequence": self.array_mmap[n]}


def load_l1_array_mmap(manager: ArtifactManager) -> tuple[np.ndarray, Dict[str, object]]:
    """以 mmap 方式加载 L1 归一化数组与 manifest。"""
    array5d, _, meta = load_l1_array_and_splits(manager.l1_dir)
    manifest = dict(meta["manifest"])
    if array5d.ndim != 5:
        raise ValueError(f"expected 5D array from L1, got shape={array5d.shape}")
    return array5d, manifest


def frame_indices_to_pairs(shape5d: Sequence[int], split_indices: Sequence[int], target_offset: int) -> List[FramePair]:
    """将 frame 索引转换为可监督学习的 (n,t) 样本对。"""
    n_size, t_size = int(shape5d[0]), int(shape5d[1])
    pairs: List[FramePair] = []
    for idx in split_indices:
        if idx < 0 or idx >= n_size * t_size:
            continue
        n = idx // t_size
        t = idx % t_size
        if t + target_offset < t_size:
            pairs.append((n, t))
    return pairs


def sequence_indices_to_pairs(shape5d: Sequence[int], split_indices: Sequence[int], target_offset: int) -> List[FramePair]:
    """将 sequence 索引展开为所有可预测时间步的 (n,t) 样本对。"""
    n_size, t_size = int(shape5d[0]), int(shape5d[1])
    valid_seq = [n for n in split_indices if 0 <= int(n) < n_size]
    pairs: List[FramePair] = []
    for n in valid_seq:
        for t in range(0, t_size - target_offset):
            pairs.append((int(n), t))
    return pairs


def load_split_pairs(
    manager: ArtifactManager,
    array5d: np.ndarray,
    manifest: Dict[str, object],
    split_name: str,
    target_offset: int,
) -> List[FramePair]:
    """读取 L1 切分索引并映射为 L2 训练/验证/测试样本对。"""
    split_path = manager.l1_split_path(split_name)
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    raw_idx = np.load(split_path, mmap_mode="r")

    shape5d = manifest["shape5d"]
    unit = str(dict(manifest.get("split", {})).get("unit", "frame"))
    if unit == "sequence":
        seq_ds = IndexedDataset(array_mmap=array5d, indices=raw_idx, unit="sequence", shape5d=shape5d)
        seq_indices = [int(seq_ds[i]["n"]) for i in range(len(seq_ds))]
        return sequence_indices_to_pairs(shape5d, seq_indices, target_offset)
    frame_ds = IndexedDataset(array_mmap=array5d, indices=raw_idx, unit="frame", shape5d=shape5d)
    n_size, t_size = int(shape5d[0]), int(shape5d[1])
    frame_indices = [int(frame_ds[i]["n"]) * t_size + int(frame_ds[i]["t"]) for i in range(len(frame_ds))]
    if n_size <= 0:
        return []
    return frame_indices_to_pairs(shape5d, frame_indices, target_offset)


class PairDataset(Dataset):
    """将 (n,t) 样本对包装为可喂给 PyTorch 的监督数据集。"""
    def __init__(
        self,
        array5d: np.ndarray,
        pairs: Sequence[FramePair],
        target_offset: int = 1,
        sparse_input: Dict[str, object] | None = None,
        dataset_id: str = "",
    ):
        """保存数组、样本对和归一化配置。"""
        self.array5d = array5d
        self.pairs = list(pairs)
        self.target_offset = int(target_offset)
        self.sparse_cfg = SparseInputConfig.from_dict(sparse_input)
        self.sparse_cfg.validate()
        self.dataset_id = str(dataset_id)

        self.sample_points_mask_hw: np.ndarray | None = None
        self.sample_points_xy = np.zeros((0, 2), dtype=np.int64)

        if self.sparse_cfg.enabled:
            h, w, _ = tuple(int(v) for v in self.array5d.shape[2:])
            finite_ref = np.isfinite(self.array5d[0, 0])
            valid_mask_hw = np.all(finite_ref, axis=-1)
            if not np.any(valid_mask_hw):
                valid_mask_hw = None
            mask_hw, points_xy = build_fixed_points_mask(
                h=h,
                w=w,
                sample_p=float(self.sparse_cfg.sample_p),
                seed=int(self.sparse_cfg.sample_seed),
                dataset_id=self.dataset_id,
                valid_mask_hw=valid_mask_hw,
            )
            self.sample_points_mask_hw = mask_hw.astype(np.bool_, copy=False)
            self.sample_points_xy = points_xy.astype(np.int64, copy=False)

    def __len__(self) -> int:
        """返回样本对数量。"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本字典：输入、目标、掩码以及索引坐标。"""
        n, t = self.pairs[idx]
        x_hwc = self.array5d[n, t]
        y_hwc = self.array5d[n, t + self.target_offset]

        if self.sparse_cfg.enabled and self.sample_points_mask_hw is not None:
            x_hwc = apply_sparse_sampling_1nn(
                x_hwc=x_hwc,
                points_mask_hw=self.sample_points_mask_hw,
                sample_sigma=float(self.sparse_cfg.sample_sigma),
                noise_seed=sample_noise_seed(int(self.sparse_cfg.sample_seed), int(n), int(t)),
            )
            c = int(x_hwc.shape[-1])
            mask_hwc = np.repeat(self.sample_points_mask_hw[:, :, None].astype(np.float32), c, axis=2)

            if bool(self.sparse_cfg.append_mask_channel):
                marker = self.sample_points_mask_hw[:, :, None].astype(np.float32)
                x_hwc = np.concatenate([x_hwc, marker], axis=2)
        else:
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
