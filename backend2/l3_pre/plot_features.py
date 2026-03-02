from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .plot_common import robust_vrange


def plot_layer_channels_atlas(
    out_path: Path,
    layer_name: str,
    feat_chw: np.ndarray,
    selected_channels: Sequence[int],
    sample_index: int,
    shared_c_total: int,
) -> None:
    c_total = int(feat_chw.shape[0])
    k = len(selected_channels)
    cols = 8
    rows = max(1, math.ceil(max(1, k) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), squeeze=False)
    if k > 0:
        channel_stack = feat_chw[np.asarray(selected_channels, dtype=np.int64)]
        vmin, vmax = robust_vrange(channel_stack, q=0.995)
    else:
        vmin, vmax = -1.0, 1.0

    im = None
    for p in range(rows * cols):
        r = p // cols
        c = p % cols
        ax = axes[r][c]
        ax.axis("off")
        if p >= k:
            continue
        ch = int(selected_channels[p])
        im = ax.imshow(feat_chw[ch], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.text(
            0.02,
            0.98,
            f"ch={ch}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.5),
        )

    fig.suptitle(
        f"Layer={layer_name} | Channels ({k}/{shared_c_total}) | Sample idx={sample_index}",
        fontsize=12,
    )
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


def plot_layer_samples_atlas(
    out_path: Path,
    feature_by_sample: Dict[int, np.ndarray],
    sample_idx: Sequence[int],
    fixed_channels: Sequence[int],
    n_total_samples: int,
) -> None:
    valid_samples = [int(i) for i in sample_idx if int(i) in feature_by_sample]
    chs = [int(c) for c in fixed_channels]

    rows = max(1, len(chs))
    cols = max(1, len(valid_samples))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8), squeeze=False)

    all_maps = [feature_by_sample[s][ch] for ch in chs for s in valid_samples]
    if all_maps:
        stack = np.stack(all_maps, axis=0)
        vmin, vmax = robust_vrange(stack, q=0.995)
    else:
        vmin, vmax = -1.0, 1.0

    im = None
    for r, ch in enumerate(chs):
        for c, sidx in enumerate(valid_samples):
            ax = axes[r][c]
            im = ax.imshow(feature_by_sample[sidx][ch], cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"idx={sidx}", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"ch={ch}", fontsize=7)

    c_total = int(next(iter(feature_by_sample.values())).shape[0]) if feature_by_sample else 0
    fig.suptitle(
        f"Samples ({len(valid_samples)}/{n_total_samples}) | Channels ({len(chs)}/{c_total})",
        fontsize=12,
    )
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
