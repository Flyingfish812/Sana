from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .plot_common import robust_vrange


def plot_sample_map_atlas(
    out_path: Path,
    maps_hw: np.ndarray,
    pair_nt: np.ndarray,
    sample_idx: Sequence[int],
    title: str,
    cmap: str = "RdBu_r",
    symmetric: bool = False,
    sample_points_xy: np.ndarray | None = None,
) -> None:
    k = len(sample_idx)
    n_total = int(maps_hw.shape[0])
    cols = 8
    rows = max(1, math.ceil(max(1, k) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), squeeze=False)

    selected_maps = maps_hw[np.asarray(sample_idx, dtype=np.int64)] if k > 0 else maps_hw[:1]
    if symmetric:
        vmin, vmax = robust_vrange(selected_maps, q=0.995)
    else:
        flat = selected_maps[np.isfinite(selected_maps)]
        if flat.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin = float(np.quantile(flat, 0.005))
            vmax = float(np.quantile(flat, 0.995))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0

    im = None
    for p in range(rows * cols):
        r = p // cols
        c = p % cols
        ax = axes[r][c]
        ax.axis("off")
        if p >= k:
            continue
        idx = int(sample_idx[p])
        im = ax.imshow(maps_hw[idx], cmap=cmap, vmin=vmin, vmax=vmax)
        if sample_points_xy is not None and np.asarray(sample_points_xy).size > 0:
            pts = np.asarray(sample_points_xy)
            ax.scatter(pts[:, 0], pts[:, 1], facecolors="none", edgecolors="white", s=18, linewidths=0.7)
        n_i, t_i = int(pair_nt[idx, 0]), int(pair_nt[idx, 1])
        ax.text(
            0.02,
            0.98,
            f"idx={idx} ({n_i},{t_i})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.5),
        )

    fig.suptitle(f"{title} ({k}/{n_total})", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.04,
            pad=0.04,
        )
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_residual_atlas(
    out_path: Path,
    residual: np.ndarray,
    pair_nt: np.ndarray,
    sample_idx: Sequence[int],
    strategy: str,
    n_total: int,
) -> None:
    k = len(sample_idx)
    cols = 8
    rows = max(1, math.ceil(max(1, k) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), squeeze=False)

    selected_maps = residual[np.asarray(sample_idx, dtype=np.int64)] if k > 0 else residual[:1]
    vmin, vmax = robust_vrange(selected_maps, q=0.995)

    im = None
    for p in range(rows * cols):
        r = p // cols
        c = p % cols
        ax = axes[r][c]
        ax.axis("off")
        if p >= k:
            continue
        idx = int(sample_idx[p])
        im = ax.imshow(residual[idx], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        n_i, t_i = int(pair_nt[idx, 0]), int(pair_nt[idx, 1])
        ax.text(
            0.02,
            0.98,
            f"idx={idx} ({n_i},{t_i})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.5),
        )

    fig.suptitle(f"Samples ({k}/{n_total}) | strategy={strategy}", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.04,
            pad=0.04,
        )
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
