# backend/model/base_components/propagator_base.py
from __future__ import annotations
import torch
import torch.nn as nn


class BasePropagator(nn.Module):
    """
    专注于 T 维的传播/递推/注意力等。
    输入/输出通常都是 5D: [B,C,T,H,W]
    """
    def __init__(self):
        super().__init__()

    def initialize(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def regularizer(self) -> torch.Tensor:
        return x5.new_tensor(0.0) if isinstance(x5 := next(self.parameters(), None), torch.Tensor) else torch.tensor(0.0)

    def save_weight_visualizations(self, out_dir: str, file_format: str = "jpg") -> None:
        pass
