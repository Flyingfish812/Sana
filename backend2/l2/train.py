from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .artifact_io import ArtifactManager
from .data import PairDataset, load_l1_array, load_split_pairs
from .model_unet import BaselineUNet
from .utils import build_code_version, dump_json, dump_jsonl_line, now_iso


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


def run_l2_train(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行 L2 训练：构建数据集、训练 UNet、保存日志与最佳模型。"""
    dataset_id = str(config["dataset_id"])
    artifacts_dir = str(config.get("artifacts_dir", "artifacts"))
    exp_name = str(config.get("exp_name", "baseline_unet"))
    run_name = config.get("run_name")

    manager = ArtifactManager(artifacts_dir=artifacts_dir, dataset_id=dataset_id, exp_name=exp_name, run_name=run_name)
    manager.ensure_run_dirs()

    dump_json(manager.train_config_json, config)
    dump_json(manager.code_version_json, build_code_version(Path.cwd()))

    seed = int(config.get("seed", 123))
    _seed_all(seed)
    device = _device_of(config)

    target_offset = int(config.get("target_offset", 1))
    split_tag = config.get("split_tag")
    array5d, manifest, norm = load_l1_array(manager)
    train_pairs = load_split_pairs(manager, manifest, "train", split_tag, target_offset)
    val_pairs = load_split_pairs(manager, manifest, "val", split_tag, target_offset)

    if len(train_pairs) == 0:
        raise ValueError("empty train pairs from L1 split")

    train_ds = PairDataset(array5d=array5d, pairs=train_pairs, norm=norm, target_offset=target_offset)
    val_ds = PairDataset(array5d=array5d, pairs=val_pairs, norm=norm, target_offset=target_offset)

    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    in_channels = int(array5d.shape[-1])
    model_cfg = dict(config.get("model", {}))
    model = BaselineUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
        convs_per_stage=int(model_cfg.get("convs_per_stage", 2)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs = int(config.get("epochs", 20))
    log_path = manager.logs_dir / "train_log.jsonl"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_steps += 1

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

        train_loss = train_loss_sum / max(train_steps, 1)
        val_loss = val_loss_sum / max(val_steps, 1)
        rec = {
            "time": now_iso(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "device": str(device),
        }
        dump_jsonl_line(log_path, rec)

        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "dataset_id": dataset_id,
            "exp_name": exp_name,
            "run_name": manager.run_name,
        }
        torch.save(state, manager.ckpt_path("model_last.pt"))
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(state, manager.ckpt_path("model_best.pt"))

    return {
        **manager.summary(),
        "ckpt_last": str(manager.ckpt_path("model_last.pt")),
        "ckpt_best": str(manager.ckpt_path("model_best.pt")),
        "train_log": str(log_path),
        "best_val_loss": best_val,
    }
