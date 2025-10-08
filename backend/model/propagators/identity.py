# backend/model/propagators/identity.py
from __future__ import annotations
import torch
import torch.nn as nn
from ..base_components.propagator_base import BasePropagator
from ..factory import register

@register("propagator", "Identity")
def IdentityPropagator():
    return _IdentityProp()

class _IdentityProp(BasePropagator):
    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        return x5  # 不改变时序与空间形状
