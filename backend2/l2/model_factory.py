from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .model_unet import BaselineUNet
from .model_unet_legacy import LegacyUNet
from .model_vit import MinimalViTRegressor


def build_l2_model(config: Dict[str, Any], *, in_channels: int, out_channels: int) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_type = str(config.get("model_type", model_cfg.get("type", "unet"))).lower()

    if model_type == "unet":
        return BaselineUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(model_cfg.get("base_channels", 32)),
            convs_per_stage=int(model_cfg.get("convs_per_stage", 2)),
        )

    if model_type == "unet_legacy":
        return LegacyUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(model_cfg.get("base_channels", 32)),
            depth=int(model_cfg.get("depth", 4)),
        )

    if model_type == "vit":
        return MinimalViTRegressor(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=int(model_cfg.get("patch_size", 16)),
            embed_dim=int(model_cfg.get("embed_dim", 64)),
            depth=int(model_cfg.get("depth", 10)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            attention_dropout=float(model_cfg.get("attention_dropout", 0.15)),
            droppath=float(model_cfg.get("droppath", 0.2)),
        )

    raise ValueError(f"unsupported model_type: {model_type}")
