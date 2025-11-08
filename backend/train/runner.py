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
        "max_epochs","max_steps","precision","accelerator","devices","strategy",
        "log_every_n_steps","val_check_interval","gradient_clip_val",
        "accumulate_grad_batches","deterministic","benchmark","num_sanity_val_steps",
        "enable_checkpointing","enable_model_summary","limit_train_batches",
        "limit_val_batches","limit_test_batches",
    }
    kw = {k: v for k, v in tcfg.items() if k in safe_keys and v is not None}
    return pl.Trainer(logger=loggers, callbacks=callbacks, **kw)

# 用于把 (p, sigma) 转成稳定、可排序的 run 目录后缀
def _format_run_suffix(p: Optional[float], sigma: Optional[float]) -> str:
    """
    将 p（采样密度）与 sigma（观测噪声）转为后缀，例如：
    p=0.05, sigma=0.01 -> "p05_s010"
    若为 None 则使用 "p--" 或 "s---" 占位，避免冲突。
    """
    def _p(v):
        return f"p{int(round(v * 100)):02d}" if v is not None else "p--"
    def _s(v):
        return f"s{int(round(v * 1000)):03d}" if v is not None else "s---"
    return f"{_p(p)}_{_s(sigma)}"

def _run_single(
    cfg: Dict,
    *,
    run_suffix: Optional[str] = None,
    injected_dls: Optional[Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]] = None,
) -> Tuple[EPDSystem, Dict[str, str]]:
    """
    执行“一组因子”的完整训练-评估-出图。
    - run_suffix: 用于把 logs/version 追加子目录，如 ".../p05_s010/"
    - injected_dls: (train_dl, val_dl, test_dl) 显式注入（通常留空，走 cfg.data 构建）
    """
    import torch
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    cfg = load_config(cfg)

    if run_suffix:
        base_ver = cfg["logging"].get("version")
        if base_ver is None:
            import datetime as dt
            base_ver = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg["logging"]["version"] = base_ver
        cfg["logging"]["version"] = f"{cfg['logging']['version']}/{run_suffix}"

    strategy = cfg["trainer"].get("strategy")
    strategy_name = strategy.lower() if isinstance(strategy, str) else ""
    spawn_like = strategy_name in {"ddp_notebook", "ddp_spawn"}
    if spawn_like:
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    seed_everything(
        cfg["train"]["seed"],
        deterministic=cfg["trainer"].get("deterministic", True),
        skip_cuda_seed=spawn_like,
    )

    run_dir = prepare_run_dir(cfg)
    loggers = build_loggers(cfg["logging"], run_dir)
    callbacks = build_callbacks(cfg["callbacks"], cfg, run_dir)

    if injected_dls is not None:
        train_dl, val_dl, test_dl = injected_dls
    else:
        train_dl, val_dl, test_dl = build_dataloaders(cfg["data"], injected=(None, None, None))

    # 写 data_ref.json
    data_ref = {
        "from_run_dir": cfg["data"].get("from_run_dir"),
        "snapshot_dir": cfg["data"].get("snapshot_dir") or (cfg["data"].get("builder_args", {}) or {}).get("snapshot_dir"),
        "builder": cfg["data"].get("builder"),
        "builder_args": cfg["data"].get("builder_args", {}),
        "factors": cfg["data"].get("factors", {}),
    }
    (run_dir / "data_ref.json").write_text(json.dumps(data_ref, indent=2), encoding="utf-8")

    maybe_save_dataloaders(train_dl, val_dl, test_dl, cfg["data"], run_dir)

    model = build_model_from_cfg(cfg["model"])

    # 训练前检查与结构导出
    first_batch = next(iter(train_dl)) if train_dl is not None else next(iter(test_dl))
    save_model_summary(model, first_batch, run_dir)
    dump_arch_spec(cfg["model"], cfg["train"]["seed"], run_dir)

    trainer = _trainer_from_cfg(cfg, loggers, callbacks)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    eval_cfg = cfg.get("eval", {})
    eval_enabled = eval_cfg.get("enable", True) and (test_dl is not None)
    if eval_enabled:
        trainer.test(model, dataloaders=test_dl)

    # checkpoint & 评估出图
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
        eval_cfg = dict(eval_cfg)
        eval_cfg["factors"] = cfg.get("data", {}).get("factors", {})
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

def run_training(
    cfg: Dict,
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
    run_suffix: Optional[str] = None,
) -> Tuple[EPDSystem, Dict[str, str]]:
    """
    公有入口：
    - 若 data.sweep.enable = true：在函数内部遍历 p × σ，分别训练评估并在 run 目录追加后缀
      （如 ".../<version>/p05_s010/"），同时把每轮 artefacts 收集后写入 sweep_summary.jsonl。
    - 否则：退化为单次训练（保持与 v1 一致）。
    """
    cfg = load_config(cfg)
    data_cfg = cfg.get("data", {}) or {}
    sweep = (data_cfg.get("sweep") or {})
    enable = bool(sweep.get("enable", False))

    # 若显式注入了 dataloaders，则强制单次（常用于 debug）
    if any(x is not None for x in (train_dl, val_dl, test_dl)) or not enable:
        return _run_single(cfg, run_suffix=run_suffix, injected_dls=(train_dl, val_dl, test_dl))

    # ---- 内置 sweep：遍历 p_list × sigma_list ----
    p_list = list(sweep.get("p_list", []))
    s_list = list(sweep.get("sigma_list", []))
    mode = (sweep.get("mode") or "grid").lower().strip()

    if not p_list:
        p_list = [data_cfg.get("factors", {}).get("sample_density")]
    if not s_list:
        s_list = [data_cfg.get("factors", {}).get("noise_sigma")]

    combos = []
    if mode == "grid":
        for p in p_list:
            for s in s_list:
                combos.append((p, s))
    else:
        # 兜底：同长度 zip
        combos = list(zip(p_list, s_list))

    # 运行版本基名（先解析一次，供子轮复用）
    base_version = cfg["logging"].get("version")
    if base_version is None:
        import datetime as dt
        base_version = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["logging"]["version"] = base_version

    summary_lines = []
    last_model, last_art = None, {}

    for (p, s) in combos:
        # 注入当次因子（训练-数据侧 builder 会消费它们）
        cfg_one = json.loads(json.dumps(cfg))  # 深拷贝
        cfg_one.setdefault("data", {}).setdefault("factors", {})
        cfg_one["data"]["factors"]["sample_density"] = p
        cfg_one["data"]["factors"]["noise_sigma"] = s
        cfg_one["data"]["factors"]["rng_seed_offset"] = int(cfg["train"].get("seed", 0))

        suffix = _format_run_suffix(p, s)
        model, artefacts = _run_single(cfg_one, run_suffix=suffix, injected_dls=None)
        last_model, last_art = model, artefacts

        # 记录 sweep 汇总行（轻量）
        summary_lines.append({
            "suffix": suffix,
            "p": p,
            "sigma": s,
            "run_dir": artefacts.get("run_dir", ""),
            "best_checkpoint": artefacts.get("best_checkpoint", ""),
            "eval_log": artefacts.get("eval_log", ""),
        })

    # 写 sweep 汇总
    # 汇总文件放在 runs/<exp>/<base_version>/sweep_summary.jsonl
    run_root = Path(last_art.get("run_dir", ".")).parents[0] if last_art else Path(".")
    out_jsonl = run_root / "sweep_summary.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in summary_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 返回最后一轮的 Model 与 Artefacts（保持返回签名）
    return last_model, last_art