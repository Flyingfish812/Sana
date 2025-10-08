# backend/model/base_components/decoder_base.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Optional


class BaseDecoder(nn.Module):
    """
    解码器可选使用 encoder.skips
    """
    def __init__(self):
        super().__init__()

    def initialize(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x5: torch.Tensor, skips: Optional[Any] = None) -> torch.Tensor:
        raise NotImplementedError

    def regularizer(self) -> torch.Tensor:
        return x5.new_tensor(0.0) if isinstance(x5 := next(self.parameters(), None), torch.Tensor) else torch.tensor(0.0)

    def save_weight_visualizations(self, out_dir: str, file_format: str = "jpg") -> None:
        pass
