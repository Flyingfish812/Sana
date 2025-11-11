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

def save_multiscale_strip(
    *,
    x: torch.Tensor | np.ndarray,
    y_hat: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    ref_by_k: dict[int, torch.Tensor | np.ndarray],
    save_path: str | Path,
    title: str | None = None,
    orientation: str = "row",  # 兼容旧签名；本版固定每行最多2张，忽略该参数
    cmap: str = "RdBu_r",
) -> bool:
    """
    多尺度联图：按“每行最多2张”的网格排布，而非单行铺满。
    面板顺序：Input | Output | Target | Target@k1 | Target@k2 | ...
    - 所有面板共享同一色标（TwoSlopeNorm，center=0）
    - 底部单独放置横向 colorbar，不挤压图像区域
    """
    if plt is None:
        return False

    # —— 收集面板与标签 —— #
    panels: list[np.ndarray] = []
    labels: list[str] = []

    x_img    = _to_hw(x, ch=0)          # Input 画第0通道
    yhat_img = _to_hw(y_hat, ch=None)   # 单通道自动去[C]
    y_img    = _to_hw(y, ch=None)

    panels.extend([x_img, yhat_img, y_img])
    labels.extend(["Input", "Output", "Target"])

    ks_sorted = sorted(list(ref_by_k.keys()))
    for k in ks_sorted:
        panels.append(_to_hw(ref_by_k[k], ch=None))
        labels.append(f"Target@k={k}")

    n = len(panels)
    if n == 0:
        return False

    # —— 共享色标（零居中） —— #
    arr_all = np.stack(panels, axis=0)
    vmin = float(np.nanmin(arr_all))
    vmax = float(np.nanmax(arr_all))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    # —— 布局：每行最多2张，多余换行；最下一行给 colorbar —— #
    import math
    ncols = 2
    nrows = math.ceil(n / ncols)

    # 估算画布尺寸：行列越多，尺寸相应放大一些
    fig_w = max(8.0, 5.0 * ncols)           # 每列 ~5英寸
    fig_h = max(3.0, 3.8 * nrows) + 0.8     # 预留0.8给colorbar
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        nrows=nrows + 1, ncols=ncols,
        height_ratios=[*(1 for _ in range(nrows)), 0.10],
        hspace=0.12, wspace=0.08
    )

    axes = []
    for i in range(n):
        r = i // ncols
        c = i % ncols
        axes.append(fig.add_subplot(gs[r, c]))

    # —— 逐格绘制 —— #
    im_first = None
    for ax, img, lab in zip(axes, panels, labels):
        im = ax.imshow(img, cmap=cmap, norm=norm)
        if im_first is None:
            im_first = im
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis("off")

    if title:
        fig.suptitle(title, y=0.995, fontsize=11)

    # —— colorbar 独占最后一行，跨越所有列 —— #
    cax = fig.add_subplot(gs[-1, :])
    cbar = fig.colorbar(im_first, cax=cax, orientation="horizontal")
    cbar.set_label("Field value (shared, centered at 0)", fontsize=9)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True

def save_metric_curves(
    *,
    curves: dict[str, dict[int, float]],
    best_k_per_metric: dict[str, int] | None,
    save_path: str | Path,
    main_metric: str | None = None,
    max_extras: int = 2,
) -> bool:
    """
    生成“多尺度指标曲线图”：
    - curves: {metric_name: {k: value}}
    - best_k_per_metric: {metric_name: k*}，用于高亮竖带与标注
    - main_metric: 主展示指标（大图）；其余指标作为 small multiples（最多 max_extras 个）
    """
    if plt is None:
        return False

    if not curves:
        return False

    # —— 选择主指标与附加指标 —— #
    metric_names = list(curves.keys())
    if main_metric is None or main_metric not in metric_names:
        main_metric = metric_names[0]
    extras = [m for m in metric_names if m != main_metric][:max_extras]

    # —— ks 轴（统一为并集的升序）——
    ks_all = sorted({k for m in metric_names for k in curves[m].keys()})

    # —— 版式：1 行主图 + 若干行小图 —— #
    nrows = 1 + (len(extras) if extras else 0)
    fig_h = 2.6 * nrows
    fig_w = 7.5
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[:, 0].tolist()

    def _plot_one(ax, metric: str):
        # y 序列：按 ks_all 对齐
        ys = [curves[metric].get(k, np.nan) for k in ks_all]
        ax.plot(ks_all, ys, marker="o", linewidth=1.8)
        ax.set_xlabel("kernel size k")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.6)

        # 高亮 best_k
        k_star = None if best_k_per_metric is None else best_k_per_metric.get(metric)
        if k_star is not None:
            # 竖直带 + 标注
            ax.axvspan(k_star - 0.5, k_star + 0.5, alpha=0.15)
            # 在该点附近加一个更醒目的 marker/文本
            try:
                y_star = curves[metric][k_star]
                ax.plot([k_star], [y_star], marker="o", markersize=8)
                ax.text(k_star, y_star, f"  best k={k_star}\n  {metric}={y_star:.4g}",
                        fontsize=9, va="bottom", ha="left")
            except Exception:
                pass

        # x 轴刻度标注成整数 k
        ax.set_xticks(ks_all)

    # —— 绘制主图 —— #
    _plot_one(axes[0], main_metric)
    axes[0].set_title(f"Multi-scale metric curves — main: {main_metric}", fontsize=11)

    # —— 小 multiples —— #
    for i, m in enumerate(extras, start=1):
        _plot_one(axes[i], m)
        axes[i].set_title(f"{m}", fontsize=10)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True

def save_curves_csv(
    *,
    curves: dict[str, dict[int, float]],
    save_path: str | Path,
) -> bool:
    """
    将样本级的 {metric: {k: value}} 写成 tidy CSV：
        metric,k,value
        psnr,3,XX
        psnr,5,XX
        ...
    """
    try:
        import csv
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "k", "value"])
            for m, kv in curves.items():
                for k in sorted(kv.keys()):
                    writer.writerow([m, int(k), float(kv[k])])
        return True
    except Exception:
        return False