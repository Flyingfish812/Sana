from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


def _safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(".", "_")


def _kernel_map(weight: np.ndarray, out_idx: int) -> np.ndarray:
    # weight: [C_out, C_in, K, K] -> 聚合输入通道得到 [K, K]
    w = np.asarray(weight[out_idx], dtype=np.float32)
    if w.ndim != 3:
        raise ValueError(f"unexpected kernel rank: {w.shape}")
    return np.mean(w, axis=0)


def plot_model_kernel_atlas(model: nn.Module, out_dir: Path) -> List[Tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Tuple[str, Path]] = []

    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    for layer_name, layer in conv_layers:
        weight = layer.weight.detach().cpu().numpy()  # [C_out, C_in, K, K]
        if weight.ndim != 4:
            continue

        c_out = int(weight.shape[0])
        c_in = int(weight.shape[1])
        k_h = int(weight.shape[2])
        k_w = int(weight.shape[3])

        cols = 8
        rows = max(1, math.ceil(max(1, c_out) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8), squeeze=False)

        maps = np.stack([_kernel_map(weight, i) for i in range(c_out)], axis=0)
        finite = maps[np.isfinite(maps)]
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            q = float(np.quantile(np.abs(finite), 0.995))
            q = max(q, 1e-6)
            vmin, vmax = -q, q

        im = None
        for i in range(rows * cols):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.axis("off")
            if i >= c_out:
                continue
            im = ax.imshow(maps[i], cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.text(
                0.02,
                0.98,
                f"oc={i}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.5),
            )

        fig.suptitle(
            f"Model Kernels | {layer_name} | out={c_out}, in={c_in}, k={k_h}x{k_w}",
            fontsize=11,
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

        out_path = out_dir / f"atlas_model__{_safe_name(layer_name)}__kernels.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        outputs.append((layer_name, out_path))

    return outputs
