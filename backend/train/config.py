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
    "eval": {
        "enable": True,
        "num_eval_batches": 3,
        "num_plot_triplets": 6,
        "metrics": ["psnr", "ssim", "corrcoef", "l1", "mse", "grad_mse", "lap_mse"],  # 要计算的指标清单（留空或缺省则退回到只算PSNR）

        # 多尺度（高斯金字塔）评估
        "scales": {
            "enable": False,
            "levels": 3,      # 生成 L0..L{levels-1}，L0为原分辨率
            "sigma": 1.6      # 金字塔平滑的基础sigma（实现里用blur+下采样近似）
        },

        # 频域评估（径向功率谱/相对误差），需要 spectral.py
        "spectral": {
            "enable": False,
            "kbins": 32,
            "fft_pad": True
        },

        # 从 batch 或 dataset.meta 提取采样布局标签的键
        "layout_tag_key": "layout_tag",

        # 是否把逐样本度量也写入log（默认只写 batch 聚合）
        "write_per_item": False
    },
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
    # --- 新增默认字段：data.factors 与 data.sweep ---
    DEFAULT_WITH_FACTORS = copy.deepcopy(DEFAULT_CFG)
    # 单轮注入用的默认 p / σ
    DEFAULT_WITH_FACTORS["data"].setdefault("factors", {
        "sample_density": None,   # None 表示不改写（沿用干净快照或数据层默认）
        "noise_sigma": None,      # None 表示不加噪
        "rng_seed_offset": 0,     # 用于不同组合的可复现派生
    })
    # 多轮 sweep（批次C会用到；此处先加默认，启用与否不影响单轮）
    DEFAULT_WITH_FACTORS["data"].setdefault("sweep", {
        "enable": False,
        "p_list": [],
        "sigma_list": [],
        "mode": "grid",
        "reuse_snapshot": True,   # 在已有 run 的 from_run_dir 时，优先复用数据来源而非 pickled dataloader
    })

    merged = _deep_update(DEFAULT_WITH_FACTORS, cfg)

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

    # 简要校验 factors（允许 None）
    fac = merged["data"].get("factors", {})
    p = fac.get("sample_density", None)
    if p is not None:
        if not (0.0 < float(p) <= 1.0):
            raise ValueError(f"data.factors.sample_density must be in (0,1], got {p}")
    sigma = fac.get("noise_sigma", None)
    if sigma is not None:
        if float(sigma) < 0.0:
            raise ValueError(f"data.factors.noise_sigma must be >= 0, got {sigma}")

    return merged
