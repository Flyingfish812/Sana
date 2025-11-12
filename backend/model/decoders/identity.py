# backend/model/decoders/identity.py
from __future__ import annotations
import torch
import torch.nn as nn
from ..factory import register
from ..base_components.decoder_base import BaseDecoder

@register("decoder", "Identity")
class IdentityDecoder(BaseDecoder):
    """解码器直接透传 encoder 特征。"""
    def __init__(self):
        super().__init__()
    def forward(self, x5: torch.Tensor, skips=None) -> torch.Tensor:
        return x5
