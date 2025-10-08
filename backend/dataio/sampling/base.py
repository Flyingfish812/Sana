# backend/dataio/sampling/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable

@dataclass
class SampleSpec:
    """
    采样规范：如何从全量 [N,T,H,W,C] 中提取一个样本片段。
    这里给出最小定义，后续可以按需扩展：
      - n: 第 n 个样本（或场景）索引
      - t_indices: 该样本使用的时间步索引列表（如 [t_hist1, t_hist2, ..., t_target]）
    """
    n: int
    t_indices: List[int]

class ListSampler:
    """最简单的采样器：持有一串 SampleSpec，__iter__ 逐个吐出。"""
    def __init__(self, specs: Iterable[SampleSpec]):
        self.specs: List[SampleSpec] = list(specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self):
        yield from self.specs
