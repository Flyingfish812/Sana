# backend/dataio/dataset/unified.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from torch.utils.data import Dataset
from ..schema import DataMeta, ArraySample
from ..utils.typing import Array5D
from ..transforms.compose import Compose
from ..sampling.base import SampleSpec

class UnifiedDataset(Dataset):
    """
    统一数据集封装：
      - 持有一个 Reader（外部先读 full 或 lazy 子集亦可）
      - 持有 SampleSpec 列表（由采样器创建）
      - 可选 Transform 管线（对 ArraySample 逐个处理）
    """

    def __init__(
        self,
        array5d: Array5D,
        meta: DataMeta,
        specs: List[SampleSpec],
        transforms: Optional[Compose] = None,
    ):
        assert array5d.ndim == 5, f"expect [N,T,H,W,C], got {array5d.shape}"
        self.array5d = array5d
        self.meta = meta
        self.specs = specs
        self.transforms = transforms

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx: int) -> ArraySample:
        spec = self.specs[idx]
        n = spec.n
        t_idx = spec.t_indices
        # 切片 [K,H,W,C]
        frames = self.array5d[n, t_idx, ...]  # [K,H,W,C]
        target_index = len(t_idx) - 1  # 缺省把最后一帧当监督目标（可由 Adapter 解释）
        sample = ArraySample(frames=frames, target_index=target_index, meta=self.meta, spec={"n": n, "t": t_idx})
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
