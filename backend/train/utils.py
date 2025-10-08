# backend/train/utils.py
from __future__ import annotations
import os, random
from typing import Optional
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl

def seed_everything(seed: int = 42, deterministic: bool = True):
    pl.seed_everything(seed, workers=True)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def select_device(accelerator: str = "auto", devices: str | int | list | None = "auto") -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch

def pick_first_key(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None
