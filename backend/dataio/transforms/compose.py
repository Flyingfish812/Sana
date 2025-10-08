# backend/dataio/transforms/compose.py
from __future__ import annotations
from typing import List, Iterable
from .base import Transform
from ..schema import ArraySample

class Compose:
    """将若干 Transform 串成管线。"""
    def __init__(self, transforms: Iterable[Transform]):
        self.transforms: List[Transform] = list(transforms)

    def __call__(self, sample: ArraySample) -> ArraySample:
        for t in self.transforms:
            sample = t(sample)
        return sample
