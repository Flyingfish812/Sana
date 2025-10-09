# backend/train/runner.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
from pathlib import Path
from torch.utils.data import DataLoader
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import load_config
from .data_adapter import build_dataloaders, maybe_save_dataloaders
from .logging import build_loggers, prepare_run_dir
from .callbacks import build_callbacks
from .inspect import save_model_summary, dump_arch_spec
from .eval import evaluate, render_eval_triplets
from .utils import seed_everything
from backend.model.epd_system import EPDSystem

def build_model_from_cfg(model_cfg: Dict) -> EPDSystem:
    return EPDSystem(
        encoder=model_cfg["encoder"],
        propagator=model_cfg["propagator"],
        decoder=model_cfg["decoder"],
        head=model_cfg["head"],
        loss=model_cfg.get("loss"),
        optimizer=model_cfg.get("optimizer"),
        scheduler=model_cfg.get("scheduler"),
        reg_weights=model_cfg.get("reg_weights"),
        data_meta=model_cfg.get("data_meta"),
    )

def _trainer_from_cfg(cfg: Dict, loggers, callbacks):
    tcfg = cfg["trainer"]
    safe_keys = {
        "max_epochs","precision","accelerator","devices","strategy",
        "log_every_n_steps","val_check_interval","gradient_clip_val",
        "accumulate_grad_batches","deterministic","benchmark","num_sanity_val_steps",
        "enable_checkpointing","enable_model_summary","limit_train_batches",
        "limit_val_batches","limit_test_batches",
    }
    kw = {k: v for k, v in tcfg.items() if k in safe_keys and v is not None}
    return pl.Trainer(logger=loggers, callbacks=callbacks, **kw)

def run_training(
    cfg: Dict,
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
) -> Tuple[EPDSystem, Dict[str, str]]:
    """支持：显式注入 dataloaders 或按 cfg.data 自动读取"""
    cfg = load_config(cfg)
    seed_everything(cfg["train"]["seed"], deterministic=cfg["trainer"].get("deterministic", True))

    run_dir = prepare_run_dir(cfg)
    loggers = build_loggers(cfg["logging"], run_dir)
    callbacks = build_callbacks(cfg["callbacks"], cfg, run_dir)

    # 统一数据入口
    train_dl, val_dl, test_dl = build_dataloaders(
        cfg["data"],
        injected=(train_dl, val_dl, test_dl)
    )

    # 写 data_ref.json，便于下次 from_run_dir 复用
    data_ref = {
        "from_run_dir": cfg["data"].get("from_run_dir"),
        "snapshot_dir": cfg["data"].get("snapshot_dir") or (cfg["data"].get("builder_args", {}) or {}).get("snapshot_dir"),
        "builder": cfg["data"].get("builder"),
        "builder_args": cfg["data"].get("builder_args", {}),
    }
    (run_dir / "data_ref.json").write_text(json.dumps(data_ref, indent=2), encoding="utf-8")

    # 可选：保存 dataloaders 快照
    maybe_save_dataloaders(train_dl, val_dl, test_dl, cfg["data"], run_dir)

    model = build_model_from_cfg(cfg["model"])

    # 训练前检查
    first_batch = next(iter(train_dl)) if train_dl is not None else next(iter(test_dl))
    save_model_summary(model, first_batch, run_dir)
    dump_arch_spec(cfg["model"], cfg["train"]["seed"], run_dir)

    trainer = _trainer_from_cfg(cfg, loggers, callbacks)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    eval_cfg = cfg.get("eval", {})
    eval_enabled = eval_cfg.get("enable", True) and test_dl is not None

    if eval_enabled:
        trainer.test(model, dataloaders=test_dl)

    # Collect checkpoints and evaluation artefacts for consistent outputs.
    best_ckpt: Optional[Path] = None
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            if cb.best_model_path:
                best_ckpt = Path(cb.best_model_path)
            break

    if best_ckpt is None:
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            candidates = sorted(ckpt_dir.glob("*.ckpt"))
            if candidates:
                best_ckpt = candidates[0]

    if eval_enabled:
        model.eval()
        eval_vis = render_eval_triplets(model, test_dl, run_dir, eval_cfg)
        evaluate(model, test_dl, run_dir, eval_cfg)
    else:
        eval_vis = run_dir / "eval_vis"

    artefacts = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt) if best_ckpt else "",
        "config": str(run_dir / "config.dump.yaml"),
        "eval_log": str(run_dir / "eval_log.jsonl"),
        "eval_vis": str(eval_vis),
    }

    return model, artefacts
