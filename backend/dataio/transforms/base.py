# backend/dataio/transforms/base.py
from __future__ import annotations
from typing import Protocol, Dict, Any
from ..schema import ArraySample

class Transform(Protocol):
    """最小 Transform 协议：输入/输出都是 ArraySample。"""
    def __call__(self, sample: ArraySample) -> ArraySample: ...
