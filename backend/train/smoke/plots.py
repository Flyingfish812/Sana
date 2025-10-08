# backend/train/smoke/plots.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import matplotlib.pyplot as plt

def _to_hw_img(t: torch.Tensor):
    t = t.detach().float().cpu()
    if t.ndim == 2:
        img = t
    elif t.ndim == 3:
        img = t[0]
    else:
        raise ValueError(f"Unsupported tensor shape for image: {t.shape}")
    vmin = torch.quantile(img.flatten(), 0.01)
    vmax = torch.quantile(img.flatten(), 0.99)
    img = torch.clamp((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return img.numpy()

def save_triplet_grid(x, y_hat, y, save_path: str | Path, title: Optional[str] = None):
    def pick_first(t):
        if t.ndim == 5: t = t[:,0]
        elif t.ndim == 4: t = t[:,0]
        elif t.ndim in (2,3): pass
        else: raise ValueError(f"Unexpected ndim={t.ndim} for plotting")
        return t
    x1 = pick_first(x); y1 = pick_first(y_hat); t1 = pick_first(y)
    x_img = _to_hw_img(x1); y_img = _to_hw_img(y1); t_img = _to_hw_img(t1)
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12,4))
    for i, (img, name) in enumerate([(x_img,"Input"),(y_img,"Output"),(t_img,"Target")], 1):
        plt.subplot(1,3,i); plt.imshow(img); plt.title(name); plt.axis("off")
    if title: plt.suptitle(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
