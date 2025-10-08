# backend/model/base_components/encoder_base.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Optional, Dict


class BaseEncoder(nn.Module):
    """
    输入:  [B,C,T,H,W]
    输出:  任意 5D 张量（通常 [B,C',T',H',W']）
    可选属性:
      - self.skips: 用于 UNet 解码的多尺度跳连（list[Tensor] 或 dict[str, Tensor]）
    """
    def __init__(self):
        super().__init__()
        self.skips: Optional[Any] = None

    def initialize(self) -> None:
        # 默认权重初始化策略（子类可覆盖）
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def regularizer(self) -> torch.Tensor:
        # 默认无正则
        return x5.new_tensor(0.0) if isinstance(x5 := next(self.parameters(), None), torch.Tensor) else torch.tensor(0.0)

    def save_weight_visualizations(self, out_dir: str, file_format: str = "jpg") -> None:
        # 可在子类实现卷积核/注意力图等可视化
        pass

    def __repr__(self) -> str:
        return super().__repr__()
