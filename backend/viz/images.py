"""Shared image visualisation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence
from matplotlib.colors import TwoSlopeNorm
import numpy as np

import torch

try:  # pragma: no cover - optional dependency in CI
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may be missing
    plt = None  # type: ignore


def _squeeze_to_hw(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().float().cpu()
    while t.ndim > 2:
        t = t[0]
    if t.ndim != 2:
        raise ValueError(f"Unsupported tensor shape for plotting: {tuple(t.shape)}")
    return t


def tensor_to_hw_image(t: torch.Tensor, quantiles: Sequence[float] = (0.01, 0.99)) -> torch.Tensor:
    """Convert arbitrary layout tensors to a normalised ``[H,W]`` image tensor."""

    img = _squeeze_to_hw(t)
    q_low, q_high = quantiles
    flat = img.flatten()
    vmin = torch.quantile(flat, q_low)
    vmax = torch.quantile(flat, q_high)
    if torch.isclose(vmax, vmin):
        vmax = vmin + 1.0
    img = torch.clamp((img - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    return img

def _to_hw(arr: torch.Tensor | np.ndarray, ch: int | None = None) -> np.ndarray:
    """把 BCHW/CHW/HW 转成 HW；ch 为多通道时要显示的通道索引。"""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    # 去 batch
    if arr.ndim == 4:        # [B,C,H,W]
        arr = arr[0]
    if arr.ndim == 3:        # [C,H,W]
        if ch is None:
            ch = 0 if arr.shape[0] > 1 else 0
        arr = arr[ch]
    elif arr.ndim != 2:      # 期望 HW
        raise ValueError(f"Unsupported image shape {arr.shape}")
    return arr

def save_triplet_grid(x, y_hat, y, save_path, *, title=None,
                      labels=("Input", "Output", "Target")) -> bool:
    if plt is None:
        return False

    x_img    = _to_hw(x, ch=0)        # 只画第0通道（recon）
    yhat_img = _to_hw(y_hat, ch=None) # 单通道自动去 [C] 维
    y_img    = _to_hw(y, ch=None)

    imgs = [x_img, yhat_img, y_img]

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_flat = imgs[2][np.isfinite(imgs[2])]
    if y_flat.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.quantile(y_flat, 0.01))
        vmax = float(np.quantile(y_flat, 0.99))
        if vmin == vmax: vmax = vmin + 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = "RdBu_r"

    fig = plt.figure(figsize=(12, 5), constrained_layout=False)
    gs  = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[20, 1], hspace=0.04, wspace=0.06)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax  = fig.add_subplot(gs[1, :])

    im_first = None
    for ax, img, lab in zip(axes, imgs, labels):
        im = ax.imshow(img, cmap=cmap, norm=norm)
        if im_first is None: im_first = im
        ax.set_title(lab); ax.axis("off")

    if title:
        fig.suptitle(title, y=0.98)

    # 专用 cax 放底部，绝不挤压图像
    cbar = fig.colorbar(im_first, cax=cax, orientation="horizontal")
    cbar.set_label("ω (vorticity)", fontsize=10)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
