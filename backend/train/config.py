# backend/train/config.py
from __future__ import annotations
from typing import Dict, Any
import copy
import datetime as dt

DEFAULT_CFG: Dict[str, Any] = {
    "exp_name": "epd_experiment",
    "model": {
        "encoder": {"name": "UNetBase", "args": {"base_channels": 32, "depth": 4}},
        "propagator": {"name": "Identity", "args": {}},
        "decoder": {"name": "UNetBase", "args": {"base_channels": 32}},
        "head": {"name": "PixelHead", "args": {"out_channels": 1}},
        "loss": {"name": "l1"},
        "optimizer": {"name": "adamw", "args": {"lr": 1e-3, "weight_decay": 1e-4}},
        "scheduler": {"name": "ReduceLROnPlateau", "monitor": "val_total", "args": {"factor": 0.5, "patience": 5}},
        "reg_weights": {"encoder": 0.0, "propagator": 0.0, "decoder": 0.0, "head": 0.0},
    },
    "data": {
        "snapshot_dir": None,     # 直接指向 dataio 快照目录（含 meta.json 或 *.pt）
        "from_run_dir": None,     # 指向上一次训练的 run 目录；会自动解析 dataloader 快照或引用
        "builder": None,          # 例如 "backend.dataio.api:build_all"
        "builder_args": {},       # 传给 builder 的参数（比如 snapshot_dir ...）
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "drop_last": False,
        "save_dataloaders": False # 开启后会把本次 dataloaders 快照保存到本 run 目录
    },
    "trainer": {
        "max_epochs": None,
        "max_steps": 10000,
        "precision": "32",
        "accelerator": "auto",
        "devices": "auto",
        "strategy": "auto",
        "log_every_n_steps": 10,
        "val_check_interval": 0.25,
        "gradient_clip_val": 0.0,
        "accumulate_grad_batches": 1,
        "deterministic": True,
        "benchmark": False,
        "num_sanity_val_steps": 2,
        "enable_checkpointing": True,
        "enable_model_summary": True,
    },
    "logging": {
        "logger": "tensorboard",
        "save_dir": "runs",
        "name": None,     # 默认为 exp_name
        "version": None,  # 默认为时间戳
    },
    "callbacks": {
        "early_stopping": {
            "enable": True, "monitor": "val_total", "mode": "min", "patience": 10, "min_delta": 0.0
        },
        "checkpoint": {
            "monitor": "val_total", "mode": "min", "save_top_k": 1, "save_last": True,
            "dirpath": None, "filename": "{epoch:03d}-{val_total:.4f}", "load_from": None
        },
        "lr_monitor": {"enable": True, "logging_interval": "epoch"},
        "viz_triplets": {"enable": False, "every_n_steps": 200, "num_triplets": 4},
    },
    "eval": {"enable": True, "num_eval_batches": 3, "num_plot_triplets": 6},
    "train": {"seed": 2025},
}

def _deep_update(base: Dict, extra: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict):
            if v.get("__replace__"):
                cleaned = {kk: vv for kk, vv in v.items() if kk != "__replace__"}
                out[k] = _deep_update({}, cleaned)
            else:
                base_section = out.get(k, {}) if isinstance(out.get(k), dict) else {}
                out[k] = _deep_update(base_section, v)
        else:
            out[k] = v
    return out

def load_config(cfg: Dict) -> Dict:
    """接受 dict（来自 YAML 或已合并）→ 叠加默认值 → 基础校验 → 返回新 dict"""
    assert isinstance(cfg, dict), "cfg must be a dict (load YAML first if needed)"
    merged = _deep_update(DEFAULT_CFG, cfg)

    # 填充 logging.name/version
    if merged["logging"]["name"] is None:
        merged["logging"]["name"] = merged["exp_name"]
    if merged["logging"]["version"] is None:
        merged["logging"]["version"] = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 基本字段校验
    need = ["model", "data", "trainer", "logging", "callbacks", "train"]
    for k in need:
        if k not in merged:
            raise KeyError(f"Missing config section: {k}")

    # 最少可用性断言
    for sec in ["encoder", "propagator", "decoder", "head"]:
        if "name" not in merged["model"][sec]:
            raise KeyError(f"model.{sec}.name required")

    return merged
