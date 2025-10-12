"""Training utilities and shared helpers."""
from __future__ import annotations

import os
import random

import numpy as np
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


def _maybe_seed_cuda(seed: int, skip_cuda_seed: bool) -> None:
    if skip_cuda_seed:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            # Defensive: manual_seed_all may raise if CUDA context has not been
            # properly initialised yet on some platforms. Skip silently.
            pass


def seed_everything(
    seed: int = 42,
    deterministic: bool = True,
    skip_cuda_seed: bool = False,
) -> None:
    """Seed Python, NumPy and PyTorch generators without eager CUDA init."""

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _maybe_seed_cuda(seed, skip_cuda_seed=skip_cuda_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def select_device(
    accelerator: str = "auto", devices: str | int | list | None = "auto"
) -> torch.device:
    """Select a compute device following Lightning-style accelerator arguments."""

    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
