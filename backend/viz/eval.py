"""Evaluation-time visualisation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from backend.common import ensure_dir, ensure_5d, extract_xy, move_batch_to_device

from .images import save_triplet_grid


@torch.no_grad()
def plot_triplets(model, test_dl, run_dir: Path, cfg_eval: Dict[str, Any]):
    """Render input/output/target triplets for a handful of evaluation batches."""

    limit_batches = int(cfg_eval.get("num_eval_batches", 3))
    limit_triplets = int(cfg_eval.get("num_plot_triplets", 6))
    out_dir = ensure_dir(run_dir / "eval_vis")

    device = next(model.parameters()).device
    model.eval()

    plotted = 0
    batches = 0
    for batch in test_dl:
        if batches >= limit_batches or plotted >= limit_triplets:
            break

        batch = move_batch_to_device(batch, device)
        x, y, _ = extract_xy(batch)
        preds = model(ensure_5d(x))
        if x.ndim == 4:
            preds = preds.squeeze(2)

        take = min(x.shape[0], limit_triplets - plotted)
        for idx in range(take):
            ok = save_triplet_grid(
                x[idx] if x.ndim == 4 else x[idx, :, 0],
                preds[idx] if preds.ndim == 4 else preds[idx, :, 0],
                y[idx] if y.ndim == 4 else y[idx, :, 0],
                out_dir / f"triplet_b{batches}_i{idx}.png",
            )
            if not ok:
                return
            plotted += 1

        batches += 1
