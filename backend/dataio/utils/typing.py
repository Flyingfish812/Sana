# backend/dataio/utils/typing.py
from __future__ import annotations
import numpy as np
from typing import TypedDict, Dict, Any, Optional, Tuple, List

# 统一 5D 数组类型别名：形状严格为 [N, T, H, W, C]
Array5D = np.ndarray

# 一些通用别名
Shape5D = Tuple[int, int, int, int, int]
IndexLike = int
PathLike = str
JSONDict = Dict[str, Any]