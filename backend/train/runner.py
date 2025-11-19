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
from .utils import seed_everything
from backend.eval import evaluate, render_eval_triplets, ensure_eval_multiscale_vis, append_multiscale_section
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
    p=0.05, sigma=0.01 -> "p050_s010"
    若为 None 则使用 "p---" 或 "s---" 占位，避免冲突。
    """
    def _p(v):
        return f"p{int(round(v * 1000)):03d}" if v is not None else "p---"
    def _s(v):
        return f"s{int(round(v * 1000)):03d}" if v is not None else "s---"
    return f"{_p(p)}_{_s(sigma)}"

def _infer_data_meta_from_loader(dl: DataLoader) -> Dict:
    """
    从 DataLoader 抽一小批，推断数据通道信息，并尽量读取命名通道。
    约定支持：
      - x: [N,C,H,W] 或 [N,C,T,H,W]（T 可为 1）
      - y: 同上，且 y 的 C = 预测通道数
    返回:
      {
        "num_in": C_in,
        "num_out": C_out,
        "channel_names": { "in_names": [...], "out_names": [...] },
        "io_spec": { "predict_channels": [...], "input_channels": [...], "target_channels": [...] }  # 若可获取
      }
    """
    import torch

    # 先试着从 dataset.meta 读取（优先高质量信息）
    meta = {}
    try:
        ds = dl.dataset
        meta = getattr(ds, "meta", {}) or {}
    except Exception:
        pass

    # 最小化地取一个 batch
    it = iter(dl)
    batch = next(it)

    def _pick(tdict, *keys):
        if isinstance(tdict, dict):
            for k in keys:
                if k in tdict:
                    return tdict[k]
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            # 兼容 tuple/list
            return batch[0] if keys[0] in ("x", "input", "inputs", "image") else batch[1]
        raise KeyError("Cannot locate x/y in batch.")

    x = _pick(batch, "x", "input", "inputs", "image")
    y = _pick(batch, "y", "target", "targets", "label")

    def _as_5d(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4:
            return t.unsqueeze(2)
        return t

    x5 = _as_5d(x)
    y5 = _as_5d(y)
    C_in = int(x5.shape[1])
    C_out = int(y5.shape[1])

    # 通道名：尽量从 meta 读取；没有则给默认名
    ch = {"in_names": [f"in{c}" for c in range(C_in)],
          "out_names": [f"out{c}" for c in range(C_out)]}

    try:
        ch_meta = meta.get("channel_names") or {}
        if isinstance(ch_meta.get("in_names"), (list, tuple)) and len(ch_meta["in_names"]) == C_in:
            ch["in_names"] = list(ch_meta["in_names"])
        if isinstance(ch_meta.get("out_names"), (list, tuple)) and len(ch_meta["out_names"]) == C_out:
            ch["out_names"] = list(ch_meta["out_names"])
    except Exception:
        pass

    # io_spec（若上游已写入）
    io_spec = {}
    try:
        io_spec = dict(meta.get("io_spec") or {})
        # 兜底：用 out_names 作为 predict_channels
        if not io_spec.get("predict_channels"):
            io_spec["predict_channels"] = list(ch["out_names"])
        if not io_spec.get("input_channels"):
            io_spec["input_channels"] = list(ch["in_names"])
        if not io_spec.get("target_channels"):
            io_spec["target_channels"] = list(ch["out_names"])
    except Exception:
        io_spec = {
            "predict_channels": list(ch["out_names"]),
            "input_channels": list(ch["in_names"]),
            "target_channels": list(ch["out_names"]),
        }

    return {
        "num_in": C_in,
        "num_out": C_out,
        "channel_names": ch,
        "io_spec": io_spec,
    }

def _run_single(
    cfg: Dict,
    *,
    run_suffix: Optional[str] = None,
    injected_dls: Optional[Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]] = None,
) -> Tuple[EPDSystem, Dict[str, str]]:
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

    # 推断数据通道并写回 model.data_meta，同时对齐 head.out_channels
    data_meta = _infer_data_meta_from_loader(train_dl if train_dl is not None else test_dl)
    cfg.setdefault("model", {}).setdefault("data_meta", {})
    cfg["model"]["data_meta"].update(data_meta)

    # 若 head.out_channels 未设置或与数据侧不一致，则以数据为准覆盖
    try:
        head_args = cfg["model"].setdefault("head", {}).setdefault("args", {})
        o_user = head_args.get("out_channels")
        if (o_user is None) or (int(o_user) != int(data_meta["num_out"])):
            head_args["out_channels"] = int(data_meta["num_out"])
    except Exception:
        pass

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
        ensure_eval_multiscale_vis(model, test_dl, run_dir, eval_cfg)
        ms_vis = ((eval_cfg.get("multiscale_ref") or {}).get("vis") or {}) if isinstance(eval_cfg.get("multiscale_ref"), dict) else {}
        ms_max = int(ms_vis.get("max_samples", 6))
        append_multiscale_section(run_dir, max_samples=ms_max)
    else:
        eval_vis = run_dir / "eval_vis"

    artefacts = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt) if best_ckpt else "",
        "config": str(run_dir / "config.dump.yaml"),
        "eval_log": str(run_dir / "eval_log.jsonl"),
        "eval_vis": str(eval_vis),
        "eval_vis_ms": str(run_dir / "eval_vis_ms"),
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
      （如 ".../<version>/p050_s010/"），同时把每轮 artefacts 收集后写入 sweep_summary.jsonl。
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