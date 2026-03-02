from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_layer_ranking(out_path: Path, rows: Sequence[Dict[str, Any]], selected_layers: Sequence[str]) -> None:
    layer_names = [str(v["name"]) for v in rows]
    energy = np.asarray([float(v["energy"]) for v in rows], dtype=np.float64)
    spec = np.asarray([float(v["spec_mean_amp"]) for v in rows], dtype=np.float64)
    l_total = len(rows)
    k = len(selected_layers)

    h = max(4.0, 0.35 * max(1, l_total))
    fig, axes = plt.subplots(1, 2, figsize=(12, h), squeeze=False)
    ax_e = axes[0][0]
    ax_s = axes[0][1]

    y = np.arange(l_total)
    selected_set = set(selected_layers)
    colors = ["tab:orange" if name in selected_set else "tab:blue" for name in layer_names]

    ax_e.barh(y, energy, color=colors, alpha=0.9)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(layer_names, fontsize=8)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("energy")

    ax_s.barh(y, spec, color=colors, alpha=0.9)
    ax_s.set_yticks(y)
    ax_s.set_yticklabels(layer_names, fontsize=8)
    ax_s.invert_yaxis()
    ax_s.set_xlabel("spec_mean_amp")

    for name in selected_layers:
        if name in layer_names:
            idx = layer_names.index(name)
            ax_e.text(float(energy[idx]), idx, "  selected", va="center", fontsize=8)

    fig.suptitle(f"Layers ({l_total}) | selected ({k}/{l_total})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
