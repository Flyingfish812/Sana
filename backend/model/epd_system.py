# backend/model/epd_system.py
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import pytorch_lightning as pl

from backend.common import ensure_5d, extract_xy

from .factory import build_component
from .losses import build_loss


class EPDSystem(pl.LightningModule):
    """Top-level encoder–propagator–decoder system wired together."""

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
        data_meta: Optional[Dict[str, Any]] = None,
        log_images_every_n_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_meta"])
        self.data_meta = data_meta or {}
        self.log_images_every_n_steps = log_images_every_n_steps

        self.encoder = build_component("encoder", encoder)
        self.propagator = build_component("propagator", propagator)
        self.decoder = build_component("decoder", decoder)
        self.head = build_component("head", head)

        self.crit = build_loss(loss or {"name": "l1", "args": {}})
        self.optimizer_cfg = optimizer or {"name": "adamw", "args": {"lr": 1e-3}}
        self.scheduler_cfg = scheduler

        self.reg_w = {"encoder": 0.0, "propagator": 0.0, "decoder": 0.0, "head": 0.0}
        if reg_weights:
            self.reg_w.update(reg_weights)

        self._shape_checked = False

    @torch.no_grad()
    def _lazy_shape_check(self, x5: torch.Tensor) -> None:
        if self._shape_checked:
            return
        self.encoder.initialize()
        self.propagator.initialize()
        self.decoder.initialize()
        self.head.initialize()

        enc_out = self.encoder(x5)
        prop_out = self.propagator(enc_out)
        dec_out = self.decoder(prop_out, skips=getattr(self.encoder, "skips", None))
        _ = self.head(dec_out)

        self._shape_checked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5 = ensure_5d(x)
        self._lazy_shape_check(x5)
        enc_out = self.encoder(x5)
        prop_out = self.propagator(enc_out)
        dec_out = self.decoder(prop_out, skips=getattr(self.encoder, "skips", None))
        y = self.head(dec_out)
        return y

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y, _ = extract_xy(batch)
        x5 = ensure_5d(x)
        y_hat = self(x5).squeeze(2)

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

        self.log(f"{stage}_loss", main_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if reg_loss != 0:
            self.log(f"{stage}_reg", reg_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_total", total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

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
