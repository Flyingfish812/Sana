# backend/train/callbacks.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
from .utils import pick_first_key
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

class VizCallback(pl.Callback):
    """把输入-输出-目标三连图写到 TensorBoard（轻量版）"""
    def __init__(self, every_n_steps: int = 200, num_triplets: int = 4):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_triplets = num_triplets

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        # 关闭开关：不画
        if self.every_n_steps <= 0:
            return

        # 多卡时：只在 global rank 0 上画图，避免多进程重复写 TB / 文件
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        gstep = trainer.global_step or 0
        if gstep % self.every_n_steps != 0:
            return

        try:
            import torchvision.utils as vutils

            # 取 batch 中的 x,y
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = pick_first_key(batch, ("x", "input", "inputs", "image"))
                y = pick_first_key(batch, ("y", "target", "targets", "label"))
            else:
                return
            if x is None or y is None:
                return

            # 统一到 [N,C,H,W]；并单样本
            if x.ndim == 5:  # [N,C,T,H,W]
                x_vis = x[:1, :, 0]  # 取 T=0
            else:
                x_vis = x[:1]
            if y.ndim == 5:
                y_vis = y[:1, :, 0]
            else:
                y_vis = y[:1]

            # 推理：保持模型状态
            was_training = pl_module.training
            pl_module.eval()
            with torch.no_grad():
                x5 = x[:1] if x.ndim == 5 else x[:1].unsqueeze(2)  # 保证 [1,C,T,H,W]
                yhat = pl_module(x5).squeeze(2)  # 期望 [1,C,H,W]
            if was_training:
                pl_module.train()

            # 通道名（若上游写入了）
            in_names: List[str] = []
            out_names: List[str] = []
            try:
                # 更稳：从当前 loader 的 dataset 读
                cur_loader = trainer.fit_loop._combined_loader._loader
                ds = getattr(cur_loader, "dataset", None)
                meta = getattr(ds, "meta", {}) if ds is not None else {}
                ch = meta.get("channel_names", {})
                in_names = list(ch.get("in_names", []))
                out_names = list(ch.get("out_names", []))
            except Exception:
                pass

            C_in = int(x_vis.shape[1])
            C_out = int(y_vis.shape[1])
            C_pred = int(yhat.shape[1]) if yhat.ndim == 4 else C_out

            # 关键修复点：三块图在通道维度上必须一致，否则 torch.cat 会报维度不匹配
            K = max(1, min(3, C_in, C_out, C_pred))

            x_vis = x_vis[:, :K]
            yhat = yhat[:, :K]
            y_vis = y_vis[:, :K]

            tiles = [
                x_vis,  # 输入
                yhat,   # 预测
                y_vis,  # 目标
            ]

            grid = vutils.make_grid(
                torch.cat(tiles, dim=0),   # [3, K, H, W]
                nrow=K,                    # 保持原先布局习惯
                normalize=True,
            )

            tag = "triplet/sample"
            if out_names and len(out_names) >= K:
                tag = f"triplet/{'+'.join(out_names[:K])}"

            logger = trainer.logger
            if hasattr(logger, "experiment"):
                logger.experiment.add_image(tag, grid, global_step=gstep)
        except Exception as e:
            print("TripletVizCallback 出图失败：", e)
            return

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
        cbs.append(VizCallback(
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
        if self.every_n_steps <= 0:
            return

        # 多卡：只在 global rank 0 上统计一次
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        gstep = trainer.global_step or 0
        if gstep % self.every_n_steps != 0:
            return

        total = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total += float(p.grad.data.norm(2).item())

        logger = trainer.logger
        if hasattr(logger, "log_metrics"):
            logger.log_metrics({"grad_norm/l2": total}, step=gstep)