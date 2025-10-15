from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch

def _numel(x):
    if x is None: return 0
    if isinstance(x, torch.Tensor): return x.numel()
    x = np.asarray(x); return int(x.size)

def estimate_batch_bytes(batch: Dict[str, Any], dtype_bytes=4):
    # 估算一个 batch 的字节数（仅 x/y/cond）
    n = 0
    for k in ("x","y","cond"):
        v = batch.get(k)
        if v is None: continue
        n += _numel(v)
    return n * dtype_bytes

def sanity_check_dataset(dataset, dataloader, *, print_first_batch=True):
    arr = dataset.array5d
    N,T,H,W,C = arr.shape
    print(f"[array5d] shape=[{N},{T},{H},{W},{C}], dtype={arr.dtype}")
    print(f"[sampler] #samples in dataset: {len(dataset)}  (expected N*T={N*T} if per_frame)")

    it = iter(dataloader)
    b = next(it)
    bs = estimate_batch_bytes(b, dtype_bytes=4)
    B = b["x"].shape[0] if b.get("x") is not None else (b["y"].shape[0] if b.get("y") is not None else 0)
    print(f"[batch0] batch_size={B}, approx_bytes={bs/1e6:.2f} MB "
          f"(x:{_numel(b.get('x'))}, y:{_numel(b.get('y'))}, cond:{_numel(b.get('cond'))})")
    if print_first_batch:
        return b
