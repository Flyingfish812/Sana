from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .artifact_io import ArtifactManager
from .data import PairDataset, load_l1_array_mmap, load_split_pairs
from .model_factory import build_l2_model
from .utils import build_code_version, dump_json, dump_jsonl_line, log_progress, now_iso


def _seed_all(seed: int) -> None:
    """设置 Python/NumPy/PyTorch 随机种子以提升可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_of(cfg: Dict[str, Any]) -> torch.device:
    """根据配置解析训练设备，支持 auto/cuda/cpu。"""
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _run_validation(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    val_loss_sum = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss_sum += float(loss.item())
            val_steps += 1
    return val_loss_sum / max(val_steps, 1)


def run_l2_train(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行 L2 训练：构建数据集、训练模型、保存日志与最佳模型。"""
    t0_all = time.perf_counter()
    dataset_id = str(config["dataset_id"])
    artifacts_dir = str(config.get("artifacts_dir", "artifacts"))
    exp_name = str(config.get("exp_name", "baseline_unet"))
    run_name = config.get("run_name")
    log_enabled = bool(config.get("log_progress", True))

    manager = ArtifactManager(artifacts_dir=artifacts_dir, dataset_id=dataset_id, exp_name=exp_name, run_name=run_name)
    manager.ensure_run_dirs()
    log_progress(log_enabled, "L2-TRAIN", f"start dataset={dataset_id}, run={manager.run_name}")

    dump_json(manager.train_config_json, config)
    dump_json(manager.code_version_json, build_code_version(Path.cwd()))

    seed = int(config.get("seed", 123))
    _seed_all(seed)
    device = _device_of(config)
    log_progress(log_enabled, "L2-TRAIN", f"device={device}, seed={seed}")

    t_data = time.perf_counter()
    target_offset = int(config.get("target_offset", 1))
    array5d, manifest = load_l1_array_mmap(manager)
    train_pairs = load_split_pairs(manager, array5d, manifest, "train", target_offset)
    val_pairs = load_split_pairs(manager, array5d, manifest, "val", target_offset)
    log_progress(
        log_enabled,
        "L2-TRAIN",
        f"data ready: shape={tuple(array5d.shape)}, train_pairs={len(train_pairs)}, val_pairs={len(val_pairs)}, dt={time.perf_counter()-t_data:.2f}s",
    )

    if len(train_pairs) == 0:
        raise ValueError("empty train pairs from L1 split")

    sparse_input_cfg = dict(config.get("sparse_input") or {})
    train_ds = PairDataset(
        array5d=array5d,
        pairs=train_pairs,
        target_offset=target_offset,
        sparse_input=sparse_input_cfg,
        dataset_id=dataset_id,
    )
    val_ds = PairDataset(
        array5d=array5d,
        pairs=val_pairs,
        target_offset=target_offset,
        sparse_input=sparse_input_cfg,
        dataset_id=dataset_id,
    )

    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    log_progress(
        log_enabled,
        "L2-TRAIN",
        f"dataloader ready: batch_size={batch_size}, num_workers={num_workers}, train_steps={len(train_loader)}, val_steps={len(val_loader)}",
    )

    sample0 = train_ds[0]
    in_channels = int(sample0["x"].shape[0])
    out_channels = int(sample0["y"].shape[0])
    model = build_l2_model(config, in_channels=in_channels, out_channels=out_channels).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
    )
    scheduler_cfg = dict(config.get("scheduler") or {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_cfg.get("mode", "min")),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 3)),
        threshold=float(scheduler_cfg.get("threshold", 1e-4)),
        min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
    )
    criterion = nn.MSELoss()
    model_type = str(config.get("model_type", dict(config.get("model", {})).get("type", "unet"))).lower()
    log_progress(
        log_enabled,
        "L2-TRAIN",
        f"model ready: type={model_type}, in_channels={in_channels}, params={sum(p.numel() for p in model.parameters())}",
    )

    best_val = float("inf")
    if "train_steps" in config:
        train_steps = int(config.get("train_steps", 1000))
    elif "max_steps" in config:
        train_steps = int(config.get("max_steps", 1000))
    elif "epochs" in config:
        train_steps = int(config.get("epochs", 1)) * max(1, len(train_loader))
    else:
        train_steps = 1000
    if train_steps <= 0:
        raise ValueError(f"train_steps must be > 0, got {train_steps}")
    val_interval_steps = int(config.get("val_interval_steps", 50))
    if val_interval_steps <= 0:
        raise ValueError(f"val_interval_steps must be > 0, got {val_interval_steps}")
    log_interval_steps = int(config.get("log_interval_steps", max(100, val_interval_steps)))
    if log_interval_steps <= 0:
        raise ValueError(f"log_interval_steps must be > 0, got {log_interval_steps}")

    log_path = manager.logs_dir / "train_log.jsonl"
    log_progress(
        log_enabled,
        "L2-TRAIN",
        f"train_steps={train_steps}, val_interval_steps={val_interval_steps}, log_interval_steps={log_interval_steps}, train_log={log_path}",
    )

    train_iter = iter(train_loader)
    window_losses: list[float] = []
    validated_once = False

    for step in range(1, train_steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        window_losses.append(float(loss.item()))

        should_validate = (step % val_interval_steps == 0) or (step == train_steps)
        if should_validate:
            train_loss = float(np.mean(window_losses)) if window_losses else float("nan")
            val_loss = _run_validation(model=model, val_loader=val_loader, criterion=criterion, device=device)
            scheduler.step(val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_equiv = step / max(1, len(train_loader))

            rec = {
                "time": now_iso(),
                "step": step,
                "epoch_equiv": round(epoch_equiv, 6),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "device": str(device),
                "lr": current_lr,
                "window_size": len(window_losses),
            }
            dump_jsonl_line(log_path, rec)

            state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "epoch_equiv": epoch_equiv,
                "config": config,
                "dataset_id": dataset_id,
                "exp_name": exp_name,
                "run_name": manager.run_name,
            }
            torch.save(state, manager.ckpt_path("model_last.pt"))
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(state, manager.ckpt_path("model_best.pt"))
                log_progress(log_enabled, "L2-TRAIN", f"step {step}/{train_steps} new best val_loss={val_loss:.6f}")

            log_progress(
                log_enabled,
                "L2-TRAIN",
                f"step {step}/{train_steps}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.6e}",
            )
            window_losses = []
            validated_once = True
        elif step % log_interval_steps == 0:
            train_loss = float(np.mean(window_losses)) if window_losses else float("nan")
            current_lr = float(optimizer.param_groups[0]["lr"])
            log_progress(
                log_enabled,
                "L2-TRAIN",
                f"step {step}/{train_steps}: train_loss={train_loss:.6f}, lr={current_lr:.6e}",
            )

    if not validated_once:
        val_loss = _run_validation(model=model, val_loader=val_loader, criterion=criterion, device=device)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        rec = {
            "time": now_iso(),
            "step": train_steps,
            "epoch_equiv": round(train_steps / max(1, len(train_loader)), 6),
            "train_loss": float(np.mean(window_losses)) if window_losses else float("nan"),
            "val_loss": val_loss,
            "device": str(device),
            "lr": current_lr,
            "window_size": len(window_losses),
        }
        dump_jsonl_line(log_path, rec)
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": train_steps,
            "epoch_equiv": train_steps / max(1, len(train_loader)),
            "config": config,
            "dataset_id": dataset_id,
            "exp_name": exp_name,
            "run_name": manager.run_name,
        }
        torch.save(state, manager.ckpt_path("model_last.pt"))
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(state, manager.ckpt_path("model_best.pt"))

    log_progress(log_enabled, "L2-TRAIN", f"finished dataset={dataset_id}, run={manager.run_name}, total_dt={time.perf_counter()-t0_all:.2f}s")

    return {
        **manager.summary(),
        "ckpt_last": str(manager.ckpt_path("model_last.pt")),
        "ckpt_best": str(manager.ckpt_path("model_best.pt")),
        "train_log": str(log_path),
        "best_val_loss": best_val,
        "train_steps": train_steps,
        "val_interval_steps": val_interval_steps,
    }
