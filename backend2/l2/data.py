from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from backend2.l1 import build_reader

from .artifact_io import ArtifactManager
from .utils import read_json


FramePair = Tuple[int, int]


@dataclass
class NormSpec:
    """归一化规格，封装 zscore/minmax 参数与应用逻辑。"""
    method: str
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    min_v: np.ndarray | None = None
    scale: np.ndarray | None = None

    @staticmethod
    def from_stats(stats: Dict[str, object]) -> "NormSpec":
        """从 L1 统计文件构建归一化规格对象。"""
        method = str(stats.get("method", "zscore")).lower()
        if method == "zscore":
            return NormSpec(
                method=method,
                mean=np.asarray(stats["mean"], dtype=np.float32),
                std=np.asarray(stats["std"], dtype=np.float32),
            )
        if method == "minmax":
            return NormSpec(
                method=method,
                min_v=np.asarray(stats["min"], dtype=np.float32),
                scale=np.asarray(stats["scale"], dtype=np.float32),
            )
        raise ValueError(f"unsupported norm method: {method}")

    def normalize(self, x_hwc: np.ndarray) -> np.ndarray:
        """对单帧 HWC 数据做按通道归一化。"""
        if self.method == "zscore":
            return (x_hwc - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)
        return (x_hwc - self.min_v.reshape(1, 1, -1)) / self.scale.reshape(1, 1, -1)


def infer_split_tag(manifest: Dict[str, object]) -> str:
    """从 manifest 推断默认 split_tag。"""
    split = dict(manifest.get("split", {}))
    strategy = str(split.get("strategy", "temporal"))
    unit = str(split.get("unit", "frame"))
    seed = int(split.get("seed", 123))
    return f"{strategy}_{unit}_seed{seed}"


def load_l1_array(manager: ArtifactManager) -> tuple[np.ndarray, Dict[str, object], NormSpec]:
    """加载 L1 原始数组、manifest 与归一化规格。"""
    manifest = read_json(manager.l1_manifest_path())
    reader_cfg = dict(manifest["reader"])
    kind = str(reader_cfg.pop("kind"))
    reader = build_reader(kind=kind, **reader_cfg)
    array5d = reader.read_array5d()
    stats = read_json(manager.l1_stats_path())
    norm = NormSpec.from_stats(stats)
    return array5d.astype(np.float32, copy=False), manifest, norm


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
    manifest: Dict[str, object],
    split_name: str,
    split_tag: str | None,
    target_offset: int,
) -> List[FramePair]:
    """读取 L1 切分索引并映射为 L2 训练/验证/测试样本对。"""
    tag = split_tag or infer_split_tag(manifest)
    split_path = manager.l1_split_path(tag, split_name)
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    raw_idx = np.load(split_path).astype(np.int64).tolist()

    shape5d = manifest["shape5d"]
    unit = str(dict(manifest.get("split", {})).get("unit", "frame"))
    if unit == "sequence":
        return sequence_indices_to_pairs(shape5d, raw_idx, target_offset)
    return frame_indices_to_pairs(shape5d, raw_idx, target_offset)


class PairDataset(Dataset):
    """将 (n,t) 样本对包装为可喂给 PyTorch 的监督数据集。"""
    def __init__(self, array5d: np.ndarray, pairs: Sequence[FramePair], norm: NormSpec, target_offset: int = 1):
        """保存数组、样本对和归一化配置。"""
        self.array5d = array5d
        self.pairs = list(pairs)
        self.norm = norm
        self.target_offset = int(target_offset)

    def __len__(self) -> int:
        """返回样本对数量。"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本字典：输入、目标、掩码以及索引坐标。"""
        n, t = self.pairs[idx]
        x_hwc = self.array5d[n, t]
        y_hwc = self.array5d[n, t + self.target_offset]

        x_hwc = np.nan_to_num(x_hwc, nan=0.0, posinf=0.0, neginf=0.0)
        y_hwc = np.nan_to_num(y_hwc, nan=0.0, posinf=0.0, neginf=0.0)
        mask_hwc = np.isfinite(self.array5d[n, t]).astype(np.float32)

        x_norm = self.norm.normalize(x_hwc)
        y_norm = self.norm.normalize(y_hwc)

        x = torch.from_numpy(np.transpose(x_norm, (2, 0, 1)).astype(np.float32))
        y = torch.from_numpy(np.transpose(y_norm, (2, 0, 1)).astype(np.float32))
        mask = torch.from_numpy(np.transpose(mask_hwc, (2, 0, 1)).astype(np.float32))

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "n": torch.tensor(n, dtype=torch.int64),
            "t": torch.tensor(t, dtype=torch.int64),
        }
