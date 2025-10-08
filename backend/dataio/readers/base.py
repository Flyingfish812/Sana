# backend/dataio/readers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from ..utils.typing import Array5D, Shape5D
from ..schema import DataMeta

class BaseReader(ABC):
    """
    Reader 统一契约：输出 MUST 为 [N, T, H, W, C]，且 dtype/填充值由 Reader 自行规范。
    - probe(): 返回数据规模（shape5d）与 DataMeta（坐标/时间/attrs）
    - read_array5d(subset): 支持按需子集读取，避免一次性载入超大数组
      subset 约定：dict，可包含 keys: n, t, h, w, c 的切片或索引（可选）
    """

    @abstractmethod
    def probe(self) -> Tuple[Shape5D, DataMeta]:
        ...

    @abstractmethod
    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> Array5D:
        ...

    # 可选：一些 Reader 通用工具
    @staticmethod
    def _ensure_5d(x: np.ndarray) -> Array5D:
        if x.ndim != 5:
            raise ValueError(f"expect 5D [N,T,H,W,C], got shape={x.shape}")
        return x
