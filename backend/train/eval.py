"""Evaluation utilities for offline metrics and artefacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from backend.common import ensure_5d, ensure_dir, extract_xy, move_batch_to_device
from backend.viz.images import save_triplet_grid, save_quadruple_grid

from .metrics import psnr


@torch.no_grad()
def evaluate(model, test_dl, run_dir: Path, cfg_eval: Dict[str, Any]):
    """Evaluate ``model`` on ``test_dl`` and append metrics to ``eval_log.jsonl``."""

    if test_dl is None:
        return

    log_path = run_dir / "eval_log.jsonl"
    device = next(model.parameters()).device
    limit = int(cfg_eval.get("num_eval_batches", 3))

    with log_path.open("a", encoding="utf-8") as fp:
        for batch_idx, batch in enumerate(test_dl):
            if batch_idx >= limit:
                break

            batch = move_batch_to_device(batch, device)
            x, y, _ = extract_xy(batch)
            x5 = ensure_5d(x)
            y_hat = model(x5)

            if y_hat.ndim == 5 and y.ndim == 4:
                y_for_metric = y_hat.squeeze(2)
            else:
                y_for_metric = y_hat

            value = float(psnr(y_for_metric, y).detach().cpu().item())
            fp.write(json.dumps({"psnr": value}) + "\n")
            fp.flush()


@torch.no_grad()
def render_eval_triplets(
    model,
    test_dl,
    run_dir: Path,
    cfg_eval: Dict[str, Any],
) -> Path:
    """Generate qualitative triplet plots from the test set.

    The resulting images are written under ``run_dir / "eval_vis"`` and the path
    is returned for convenience. Missing dataloaders are ignored gracefully.
    """

    if test_dl is None:
        return run_dir / "eval_vis"

    img_dir = ensure_dir(run_dir / "eval_vis")
    device = next(model.parameters()).device

    model.eval()
    model.to(device)

    max_batches = int(cfg_eval.get("num_eval_batches", 3))
    max_triplets = int(cfg_eval.get("num_plot_triplets", 4))

    plotted, batches = 0, 0
    for batch in test_dl:
        if batches >= max_batches or plotted >= max_triplets:
            break

        batch = move_batch_to_device(batch, device)
        x, y, _ = extract_xy(batch)
        y_hat = model(ensure_5d(x))
        if x.ndim == 4:
            y_hat = y_hat.squeeze(2)

        take = min(x.shape[0], max_triplets - plotted)
        for idx in range(take):
            save_quadruple_grid(
                x[idx] if x.ndim == 4 else x[idx, :, 0],
                y_hat[idx] if y_hat.ndim == 4 else y_hat[idx, :, 0],
                y[idx] if y.ndim == 4 else y[idx, :, 0],
                img_dir / f"triplet_b{batches}_i{idx}.png",
            )
            plotted += 1

        batches += 1

    return img_dir
