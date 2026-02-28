from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

Shape5D = Tuple[int, int, int, int, int]


@dataclass
class DataMeta:
    """L1 数据元信息：时间轴与附加属性。"""
    times: Optional[np.ndarray] = None
    attrs: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        """将元信息转换为可序列化的 JSON 字典。"""
        return {
            "times": self.times.tolist() if isinstance(self.times, np.ndarray) else None,
            "attrs": self.attrs or {},
        }


@dataclass
class L1Summary:
    """L1 流水线执行结果摘要。"""
    dataset_id: str
    shape5d: Shape5D
    split_sizes: Dict[str, int]
    stats_method: str
    artifacts_dir: str
