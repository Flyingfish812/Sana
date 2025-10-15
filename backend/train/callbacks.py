# backend/train/callbacks.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
from .utils import pick_first_key
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

class TripletVizCallback(pl.Callback):
    """可选：把输入-输出-目标三连图写到 TensorBoard（轻量版）"""
    def __init__(self, every_n_steps: int = 200, num_triplets: int = 4):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_triplets = num_triplets

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int):
        if self.every_n_steps <= 0: return
        gstep = trainer.global_step or 0
        if gstep % self.every_n_steps != 0: return
        try:
            # 取一小片做推理
            x, y = None, None
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = pick_first_key(batch, ("x", "input", "inputs", "image"))
                y = pick_first_key(batch, ("y", "target", "targets", "label"))
            else:
                return
            if x is None or y is None:
                return
            if x.ndim == 4: x5 = x[:1].unsqueeze(2)
            else: x5 = x[:1]
            pl_module.eval()
            with torch.no_grad():
                yhat = pl_module(x5).squeeze(2)  # [1,C,H,W]
            import torchvision.utils as vutils
            import torch
            # 拼一行三图（normalize 以便写板）
            grid = vutils.make_grid([
                (x[:1] if x.ndim==4 else x[:1, :, 0]).float(),      # Input
                yhat[:1].float(),                                   # Output
                (y[:1] if y.ndim==4 else y[:1, :, 0]).float(),      # Target
            ], nrow=3, normalize=True)
            logger = trainer.logger
            if hasattr(logger, "experiment"):
                logger.experiment.add_image("triplet/sample", grid, global_step=gstep)
        except Exception:
            pass  # 忽略可视化失败，不影响训练


def build_callbacks(cb_cfg: Dict[str, Any], root_cfg: Dict, run_dir: Path) -> List[pl.Callback]:
    cbs: List[pl.Callback] = []

    es = cb_cfg.get("early_stopping", {})
    if es.get("enable", True):
        cbs.append(EarlyStopping(
            monitor=es.get("monitor", "val_total"),
            mode=es.get("mode", "min"),
            patience=es.get("patience", 10),
            min_delta=es.get("min_delta", 0.0),
            verbose=False,
        ))

    ck = cb_cfg.get("checkpoint", {})
    dirpath = ck.get("dirpath") or str(run_dir / "checkpoints")
    cbs.append(ModelCheckpoint(
        monitor=ck.get("monitor", "val_total"),
        mode=ck.get("mode", "min"),
        save_top_k=ck.get("save_top_k", 1),
        save_last=ck.get("save_last", False),
        dirpath=dirpath,
        filename=ck.get("filename", "{epoch:03d}-{val_total:.4f}"),
        auto_insert_metric_name=False,
    ))

    lrmon = cb_cfg.get("lr_monitor", {})
    if lrmon.get("enable", True):
        cbs.append(LearningRateMonitor(logging_interval=lrmon.get("logging_interval", "epoch")))

    viz = cb_cfg.get("viz_triplets", {})
    if viz.get("enable", False):
        cbs.append(TripletVizCallback(
            every_n_steps=viz.get("every_n_steps", 200),
            num_triplets=viz.get("num_triplets", 4),
        ))

    g = cb_cfg.get("grad_norm", {"enable": False})
    if g.get("enable", False):
        cbs.append(GradNormLogger(every_n_steps=g.get("every_n_steps", 50)))
    return cbs

class GradNormLogger(pl.Callback):
    """记录梯度范数，便于稳定性排查"""
    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.every_n_steps <= 0: return
        gstep = trainer.global_step or 0
        if gstep % self.every_n_steps != 0: return
        total = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total += float(p.grad.data.norm(2).item())
        logger = trainer.logger
        if hasattr(logger, "log_metrics"):
            logger.log_metrics({"grad_norm/l2": total}, step=gstep)