# backend/dataio/schema.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from .utils.typing import Array5D, Shape5D, JSONDict

@dataclass
class DataMeta:
    """
    数据元信息（与具体 Reader 强绑定，但结构保持通用）：
      - coords_xy: (H,W,2) 的网格坐标，或 None
      - times: (T,) 的时间轴，或 None（可为整数步索引）
      - mask_static: (H,W,1) 的全局静态掩码（如海陆掩码），或 None
      - attrs: 任意额外属性（如变量名、单位、spacing 等）
    """
    coords_xy: Optional[np.ndarray] = None      # [H, W, 2]
    times: Optional[np.ndarray] = None          # [T]
    mask_static: Optional[np.ndarray] = None    # [H, W, 1]
    attrs: Dict[str, Any] = None

    def to_json(self) -> JSONDict:
        # 只序列化轻量信息，重数组部分建议另存（或存形状+dtype）
        return {
            "times": self.times.tolist() if isinstance(self.times, np.ndarray) else None,
            "coords_xy_shape": None if self.coords_xy is None else list(self.coords_xy.shape),
            "mask_static_shape": None if self.mask_static is None else list(self.mask_static.shape),
            "attrs": self.attrs or {},
        }

@dataclass
class ArraySample:
    """
    统一的中间样本表示：Reader→Transforms→Sampler→Adapter 的桥梁。
    frames: [K, H, W, C] 片段（K 可以是 1 或多帧历史/条件帧）
    target_index: int     指明其中哪一帧/索引是监督目标（可选）
    meta: DataMeta        传递必要的上下文（坐标、时间、静态掩码等）
    spec: Dict            原始样本规范（如起止时间步等），便于追溯与调试
    """
    frames: np.ndarray
    target_index: Optional[int]
    meta: DataMeta
    spec: Dict[str, Any]

@dataclass
class AdapterOutput:
    """
    适配器产物：可直接喂给模型（通常为张量；此处留 np.ndarray，DataLoader 中再转 torch）
    x: 模型输入
    y: 监督目标（可选任务可为 None）
    cond: 条件/先验/编码等附加通道（可为 None）
    meta: 透传 DataMeta，供可视化/逆归一化/评估时使用
    """
    x: Any
    y: Optional[Any]
    cond: Optional[Any]
    meta: DataMeta
