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

def save_quadruple_grid(
    x, y_hat, y, save_path, *,
    title: Optional[str] = None,
    labels = ("Input + Samples", "Output", "Target", "Diff (Target - Output)"),
    sample_ch: int = 1,           # x 的采样点/掩码通道索引
    input_ch: int = 0,            # x 的可视化底图通道索引（插值图）
    diff_center_zero: bool = True # True 时差分图与其他图共用同一 colorbar（零居中）
) -> bool:
    """
    绘制 2×2 网格：
      [0,0]：带采样点标注的输入插值图（x[input_ch] 为底图，x[sample_ch]==1 处画点）
      [0,1]：模型输出 y_hat
      [1,1]：真实目标 y
      [1,0]：差分图 y - y_hat
    四图共用同一底部 colorbar。
    """
    if plt is None:
        return False

    # —— 数据整理成 HW —— #
    x_img    = _to_hw(x, ch=input_ch)     # 作为输入底图
    yhat_img = _to_hw(y_hat, ch=None)
    y_img    = _to_hw(y, ch=None)
    diff_img = y_img - yhat_img

    # 采样点（mask==1）只用于标注，不参与色标范围
    mask_hw  = _to_hw(x, ch=sample_ch)
    samp_r, samp_c = np.where(mask_hw == 1)

    # —— 统一色标范围（共用 colorbar）——
    # 以 target 的分布给色标（与你之前保持一致），零作为中点
    y_flat = y_img[np.isfinite(y_img)]
    if y_flat.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.quantile(y_flat, 0.01))
        vmax = float(np.quantile(y_flat, 0.99))
        if vmin == vmax:
            vmax = vmin + 1.0

    # 差分图通常希望以 0 居中显示；若与其他共用同一 colorbar，则同样使用该 TwoSlopeNorm
    from matplotlib.colors import TwoSlopeNorm
    cmap = "RdBu_r"
    if diff_center_zero:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        # 若不要求共用，可单独为 diff 设定对称范围；但本需求是共用，默认不走这支
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # —— 布局：2×2 图 + 最下方 1 行 colorbar —— #
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 10), constrained_layout=False)
    gs  = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[20, 20, 1], hspace=0.06, wspace=0.08)

    ax00 = fig.add_subplot(gs[0, 0])  # Input + Samples
    ax01 = fig.add_subplot(gs[0, 1])  # Output
    ax10 = fig.add_subplot(gs[1, 0])  # Diff
    ax11 = fig.add_subplot(gs[1, 1])  # Target
    cax  = fig.add_subplot(gs[2, :])  # Colorbar

    # —— 逐图绘制 —— #
    im00 = ax00.imshow(x_img,   cmap=cmap, norm=norm, origin="upper")
    ax00.set_title(labels[0]); ax00.axis("off")
    # 叠加采样点（空心小圆点，尽量不遮挡底图）
    if samp_r.size > 0:
        ax00.scatter(samp_c, samp_r, s=6, marker='o', facecolors='none', edgecolors='k', linewidths=0.5)

    im01 = ax01.imshow(yhat_img, cmap=cmap, norm=norm, origin="upper")
    ax01.set_title(labels[1]); ax01.axis("off")

    im11 = ax11.imshow(y_img,    cmap=cmap, norm=norm, origin="upper")
    ax11.set_title(labels[2]); ax11.axis("off")

    im10 = ax10.imshow(diff_img, cmap=cmap, norm=norm, origin="upper")
    ax10.set_title(labels[3]); ax10.axis("off")

    if title:
        fig.suptitle(title, y=0.995)

    # —— 共用 colorbar（绑定到第一幅图像的 mappable 即可）—— #
    im_first = im00
    cbar = fig.colorbar(im_first, cax=cax, orientation="horizontal")
    # 你可以换成通用字段或从 cfg 传入
    cbar.set_label("Field value (shared scale)", fontsize=10)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
