# backend/model/losses.py
from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn.functional as F

def build_loss(cfg: Dict[str, Any]):
    name = (cfg.get("name") or "l1").lower()
    args = cfg.get("args", {}) or {}
    if name == "l1":
        def l1(pred, target):
            return F.l1_loss(pred, target, **args)
        return l1
    elif name == "mse":
        def mse(pred, target):
            return F.mse_loss(pred, target, **args)
        return mse
    else:
        raise ValueError(f"Unsupported loss: {name}")
