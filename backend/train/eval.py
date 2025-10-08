"""Evaluation utilities for offline metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from backend.common import extract_xy, ensure_5d, move_batch_to_device

from .metrics import psnr


@torch.no_grad()
def evaluate(model, test_dl, run_dir: Path, cfg_eval: Dict[str, Any]):
    """Evaluate ``model`` on ``test_dl`` and append metrics to ``eval_log.jsonl``."""

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


# Re-export visualisation helper from the dedicated viz module.
try:  # pragma: no cover - optional dependency
    from backend.viz.eval import plot_triplets  # type: ignore
except Exception:  # pragma: no cover - matplotlib might be missing
    plot_triplets = None  # type: ignore
