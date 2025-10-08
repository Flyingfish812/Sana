"""Training utilities and shared helpers."""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from backend.common import ensure_dir, move_batch_to_device
from backend.common.batch import extract_xy, ensure_5d, pick_first_key


__all__ = [
    "seed_everything",
    "select_device",
    "ensure_dir",
    "move_batch_to_device",
    "extract_xy",
    "ensure_5d",
    "pick_first_key",
]


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Lightning, PyTorch and CUDA helpers."""

    pl.seed_everything(seed, workers=True)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(
    accelerator: str = "auto", devices: str | int | list | None = "auto"
) -> torch.device:
    """Select a compute device following Lightning-style accelerator arguments."""

    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
