# backend/model/epd_system.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .factory import build_component, REGISTRY_TYPES
from .losses import build_loss


def _guess_xy_from_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    统一从多种 batch 结构中抽取 (x, y, meta)。
    支持：
      - tuple/list: (x, y) or (x, y, meta)
      - dict: 常见键名别名 {'x'|'input'|'inputs'|'image': x, 'y'|'target'|'targets'|'label': y, 'meta': {...}}
    """
    meta = {}
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) >= 3:
            x, y, meta = batch[0], batch[1], batch[2] if isinstance(batch[2], dict) else {}
        else:
            raise ValueError("Batch tuple/list length should be 2 or 3.")
        return x, y, meta

    if isinstance(batch, dict):
        # x candidates
        x = batch.get("x", None)
        if x is None: x = batch.get("input", None)
        if x is None: x = batch.get("inputs", None)
        if x is None: x = batch.get("image", None)

        # y candidates
        y = batch.get("y", None)
        if y is None: y = batch.get("target", None)
        if y is None: y = batch.get("targets", None)
        if y is None: y = batch.get("label", None)

        meta = batch.get("meta", {})
        if x is None or y is None:
            raise KeyError("Cannot find (x,y) in batch dict. Expected keys like x/input/image and y/target/label.")
        return x, y, meta

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
    """
    接受 [B,C,H,W] 或 [B,C,T,H,W]，统一转为 5D。
    静态图片自动在 T 维加 1。
    """
    if x.ndim == 4:
        B, C, H, W = x.shape
        return x.view(B, C, 1, H, W)
    elif x.ndim == 5:
        return x
    else:
        raise ValueError(f"Input must be 4D or 5D tensor, got shape {tuple(x.shape)}")


class EPDSystem(pl.LightningModule):
    """
    顶层 EPD 拼装系统：
       Encoder -> Propagator -> Decoder -> Head
    - 懒初始化：首次看到 batch 时，跑一遍 dummy forward 检查形状连通；
    - 正则聚合：encoder/propagator/decoder/head 的 regularizer() 统一纳入损失；
    - 统一 I/O：兼容多种 batch 组织结构；输入自动升为 [B,C,T,H,W]。
    """

    def __init__(
        self,
        encoder: Dict[str, Any],
        propagator: Dict[str, Any],
        decoder: Dict[str, Any],
        head: Dict[str, Any],
        loss: Dict[str, Any] | None = None,
        optimizer: Dict[str, Any] | None = None,
        scheduler: Dict[str, Any] | None = None,
        reg_weights: Dict[str, float] | None = None,
        # 允许从 meta.json 之类补充信息（可选）
        data_meta: Optional[Dict[str, Any]] = None,
        log_images_every_n_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_meta"])  # 避免把大 meta 存 ckpt
        self.data_meta = data_meta or {}
        self.log_images_every_n_steps = log_images_every_n_steps

        # 构建组件（注册表基于 name）
        self.encoder = build_component("encoder", encoder)
        self.propagator = build_component("propagator", propagator)
        self.decoder = build_component("decoder", decoder)
        self.head = build_component("head", head)

        # 损失/优化
        self.crit = build_loss(loss or {"name": "l1", "args": {}})
        self.optimizer_cfg = optimizer or {"name": "adamw", "args": {"lr": 1e-3}}
        self.scheduler_cfg = scheduler  # 可为 None

        # 正则项权重
        self.reg_w = {"encoder": 0.0, "propagator": 0.0, "decoder": 0.0, "head": 0.0}
        if reg_weights:
            self.reg_w.update(reg_weights)

        # 懒初始化标志
        self._shape_checked = False

    @torch.no_grad()
    def _lazy_shape_check(self, x5: torch.Tensor) -> None:
        if self._shape_checked:
            return
        self.encoder.initialize()
        self.propagator.initialize()
        self.decoder.initialize()
        self.head.initialize()

        # 走一遍前向，确保维度连通
        enc_out = self.encoder(x5)                              # [B,Ce,Te,He,We] or with skips
        prop_out = self.propagator(enc_out)                     # [B,Cp,Tp,Hp,Wp]
        dec_out = self.decoder(prop_out, skips=getattr(self.encoder, "skips", None))
        _ = self.head(dec_out)

        self._shape_checked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5 = _ensure_5d(x)
        self._lazy_shape_check(x5)
        enc_out = self.encoder(x5)
        prop_out = self.propagator(enc_out)
        dec_out = self.decoder(prop_out, skips=getattr(self.encoder, "skips", None))
        y = self.head(dec_out)
        return y

    # ---- Lightning hooks ----
    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y, meta = _guess_xy_from_batch(batch)
        x5 = _ensure_5d(x)
        y_hat = self(x5).squeeze(2)  # 模型内部是 5D，输出通常 [B,C,T,H,W]；对齐 y 可能是 4D，先 squeeze T

        # 若 y 是 5D 则不 squeeze；若是 4D 则与 y_hat 对齐
        if y.ndim == 5 and y_hat.ndim == 4:
            y_hat = y_hat.unsqueeze(2)

        main_loss = self.crit(y_hat, y)

        reg_loss = (
            self.reg_w["encoder"] * getattr(self.encoder, "regularizer", lambda: 0.0)()
            + self.reg_w["propagator"] * getattr(self.propagator, "regularizer", lambda: 0.0)()
            + self.reg_w["decoder"] * getattr(self.decoder, "regularizer", lambda: 0.0)()
            + self.reg_w["head"] * getattr(self.head, "regularizer", lambda: 0.0)()
        )

        total = main_loss + reg_loss

        # 记录
        self.log(f"{stage}_loss", main_loss, on_step=True, on_epoch=True, prog_bar=True)
        if reg_loss != 0:
            self.log(f"{stage}_reg", reg_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_total", total, on_step=True, on_epoch=True, prog_bar=True)

        return total

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        name = (self.optimizer_cfg.get("name") or "adamw").lower()
        args = self.optimizer_cfg.get("args", {})
        if name in ["adam", "adamw"]:
            opt = torch.optim.AdamW(self.parameters(), **args) if name == "adamw" else torch.optim.Adam(self.parameters(), **args)
        elif name == "sgd":
            opt = torch.optim.SGD(self.parameters(), **args)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

        if not self.scheduler_cfg:
            return opt

        sch_name = (self.scheduler_cfg.get("name") or "").lower()
        sch_args = self.scheduler_cfg.get("args", {})
        if sch_name == "reducelronplateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **sch_args)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": self.scheduler_cfg.get("monitor", "val_total")}}
        elif sch_name == "cosineannealinglr":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, **sch_args)
            return {"optimizer": opt, "lr_scheduler": sch}
        else:
            raise ValueError(f"Unsupported scheduler: {sch_name}")
