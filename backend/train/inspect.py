# backend/train/inspect.py
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from .utils import pick_first_key
import json

import torch

# 可选：torchinfo
try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

def _ensure_5d_for_summary(sample_batch):
    if isinstance(sample_batch, (tuple, list)):
        x = sample_batch[0]
    elif isinstance(sample_batch, dict):
        x = pick_first_key(sample_batch, ("x", "input", "inputs", "image"))
    else:
        raise ValueError("Unsupported batch format for summary preview.")
    if x is None:
        raise KeyError("Cannot find input tensor in batch. Tried keys: x/input/inputs/image.")
    if x.ndim == 4:
        return x[:1].unsqueeze(2)
    if x.ndim == 5:
        return x[:1]
    raise ValueError(f"Expect 4D/5D input, got {tuple(x.shape)}")

def save_model_summary(model, sample_batch, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    x5 = _ensure_5d_for_summary(sample_batch)
    text = [repr(model), "\n" + "="*80 + "\n"]
    if torchinfo_summary is not None:
        try:
            info = torchinfo_summary(model, input_data=(x5,), verbose=0, depth=4,
                                     col_names=("input_size","output_size","num_params","kernel_size"))
            text.append(str(info))
        except Exception as e:
            text.append(f"[torchinfo failed] {e}")
    else:
        num_params = sum(p.numel() for p in model.parameters())
        text.append(f"Total parameters: {num_params:,}")
    (run_dir / "model_summary.txt").write_text("\n".join(text), encoding="utf-8")


def _sorted(d):
    if isinstance(d, dict):
        return {k: _sorted(d[k]) for k in sorted(d)}
    if isinstance(d, list):
        return [_sorted(x) for x in d]
    return d


def dump_arch_spec(model_cfg: Dict[str, Any], seed: int, run_dir: Path) -> None:
    spec = {k: model_cfg.get(k, {}) for k in ("encoder","propagator","decoder","head")}
    spec = _sorted(spec)
    spec["_init_seed"] = int(seed)
    (run_dir / "arch_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
