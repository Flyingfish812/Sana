# backend/eval/visuals.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib import ticker

def plot_heatmap(
    P: np.ndarray,
    S: np.ndarray,
    M: np.ndarray,
    metric: str,
    out_png: Path,
    *,
    cmap: str = "RdBu_r",
    dpi: int = 220,
    direction: str | None = None,
    metric_specs_lookup: dict | None = None,
) -> None:
    """
    自动分发：best_k_* → 离散图；否则连续图（方向从 specs 推断）。
    """
    if metric.startswith("best_k_"):
        plot_heatmap_discrete_int(P, S, M, metric, out_png, dpi=dpi, cmap="viridis")
        return

    base_metric = metric[len("at_best_"):] if metric.startswith("at_best_") else metric
    direction = direction or (metric_specs_lookup or {}).get(base_metric, {}).get("direction", "higher")
    title = f"{metric}"
    fullname = (metric_specs_lookup or {}).get(base_metric, {}).get("fullname", None)
    if fullname:
        title = f"{metric} — {fullname} (at best scale)" if metric.startswith("at_best_") else f"{metric} — {fullname}"

    plot_heatmap_annotated(P, S, M, title, out_png,
                           cmap=cmap, dpi=dpi, direction=direction,
                           annotate=True, topk=5)

def plot_heatmap_discrete_int(
    P: np.ndarray,
    S: np.ndarray,
    M: np.ndarray,
    metric: str,
    out_png: Path,
    *,
    dpi: int = 220,
    cmap: str = "viridis",
) -> None:
    valid = np.isfinite(M)
    if not np.any(valid):
        raise ValueError("无有效数据可绘图。")

    uniq_vals = sorted({int(round(v)) for v in M[valid].ravel()})
    if not uniq_vals:
        uniq_vals = [0]
    if len(uniq_vals) == 1:
        boundaries = np.array([uniq_vals[0] - 0.5, uniq_vals[0] + 0.5], dtype=float)
        norm = BoundaryNorm(boundaries, ncolors=256)
    else:
        edges = [uniq_vals[0] - 0.5] + [v + 0.5 for v in uniq_vals]
        boundaries = np.array(edges, dtype=float)
        norm = BoundaryNorm(boundaries, ncolors=256)

    fig = plt.figure(figsize=(8.5, 6.8), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, origin="lower", aspect="auto",
                   extent=[S.min(), S.max(), P.min(), P.max()],
                   cmap=cmap, norm=norm)

    for i, p in enumerate(P):
        for j, s in enumerate(S):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(s, p, f"{int(round(v))}", ha="center", va="center", fontsize=7, color="white", alpha=0.9)

    ax.set_xlabel("sigma")
    ax.set_ylabel("p (sample density)")
    ax.set_title(f"{metric} — Equivalent Kernel Size (discrete)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("kernel size (k)", rotation=90)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

def plot_heatmap_annotated(
    P: np.ndarray,
    S: np.ndarray,
    M: np.ndarray,
    title: str,
    out_png: Path,
    *,
    cmap: str = "RdBu_r",
    dpi: int = 220,
    direction: str = "higher",
    annotate: bool = True,
    topk: int = 5,
) -> None:
    valid = np.isfinite(M)
    if not np.any(valid):
        raise ValueError("无有效数据可绘图。")
    vals = M[valid]
    q5, q95 = np.nanpercentile(vals, [5, 95])
    vmin, vmax = (q5, q95) if q5 < q95 else (np.nanmin(vals), np.nanmax(vals))
    norm = Normalize(vmin=vmin, vmax=vmax)

    lower_is_better = (direction or "higher").lower().strip() == "lower"
    def _score_for_rank(A):
        return -A if lower_is_better else A

    best_j_per_i = []
    for i in range(len(P)):
        row = M[i, :]
        if np.all(~np.isfinite(row)):
            best_j_per_i.append(None)
            continue
        j = int(np.nanargmin(row) if lower_is_better else np.nanargmax(row))
        best_j_per_i.append(j)

    flat = M.copy()
    flat[~valid] = np.nan
    order = np.argsort(_score_for_rank(flat).ravel())[::-1]
    order = [idx for idx in order if np.isfinite(flat.ravel()[idx])]
    top_idx = order[:max(0, int(topk))]
    top_pairs = [(int(k // M.shape[1]), int(k % M.shape[1])) for k in top_idx]

    fig = plt.figure(figsize=(8.5, 6.8), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[8, 2.2], height_ratios=[8, 2.2], wspace=0.15, hspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax)

    im = ax.imshow(M, origin="lower", aspect="auto",
                   extent=[S.min(), S.max(), P.min(), P.max()], cmap=cmap, norm=norm)

    if np.any(~valid):
        ax.imshow(~valid, origin="lower", aspect="auto",
                  extent=[S.min(), S.max(), P.min(), P.max()],
                  cmap="gray", alpha=0.25)

    if annotate:
        rng = vmax - vmin
        if rng >= 10: fmt = lambda x: f"{x:.1f}"
        elif rng >= 1: fmt = lambda x: f"{x:.2f}"
        elif rng >= 0.1: fmt = lambda x: f"{x:.3f}"
        else: fmt = lambda x: f"{x:.4f}"
        for i, p in enumerate(P):
            for j, s in enumerate(S):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(s, p, fmt(v),
                        ha="center", va="center",
                        fontsize=7,
                        color="black")

    path_x, path_y = [], []
    for i, j in enumerate(best_j_per_i):
        if j is None:
            continue
        path_x.append(S[j]); path_y.append(P[i])
    if len(path_x) >= 2:
        ax.plot(path_x, path_y, lw=2, ls="-", marker="o", ms=3, mec="white", mfc="white", alpha=0.9)

    for (i, j) in top_pairs:
        ax.plot(S[j], P[i], marker="o", ms=10, mfc="none", mec="yellow", mew=1.5, alpha=0.9)

    ax.set_xlabel("sigma")
    ax.set_ylabel("p (sample density)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("metric value", rotation=90)

    row_best = np.array([np.nanmin(M[i, :]) if lower_is_better else np.nanmax(M[i, :]) for i in range(len(P))])
    ax_r.plot(row_best, P, lw=1.8)
    ax_r.grid(alpha=0.3)
    ax_r.set_xlabel("best over sigma")
    ax_r.tick_params(labelleft=False)

    col_best = np.array([np.nanmin(M[:, j]) if lower_is_better else np.nanmax(M[:, j]) for j in range(len(S))])
    ax_b.plot(S, col_best, lw=1.8)
    ax_b.grid(alpha=0.3)
    ax_b.set_ylabel("best over p")
    ax_b.yaxis.set_label_position("right")
    ax_b.yaxis.tick_right()

    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
