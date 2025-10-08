# backend/dataio/dataset/subset.py
from __future__ import annotations
from typing import Sequence, Any
from torch.utils.data import Dataset

class SubsetDataset(Dataset):
    """轻量封装：从一个 base dataset 中选取 indices 作为子集"""
    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

        # 尽可能把元信息透传给检查/可视化
        self.array5d = getattr(base, "array5d", None)
        self.meta = getattr(base, "meta", None)
        self.transforms = getattr(base, "transforms", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Any:
        return self.base[self.indices[i]]
