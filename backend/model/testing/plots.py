from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import matplotlib.pyplot as plt

def _to_hw_img(t: torch.Tensor):
    """
    接受:
      - 4D: [C,H,W]    （默认取 C=第一个通道）
      - 3D: [T,H,W]    （默认取 T=0）
      - 2D: [H,W]
      - 5D/4D batch: 上层会切片到单张
    返回: [H,W] numpy
    """
    t = t.detach().float().cpu()
    if t.ndim == 2:
        img = t
    elif t.ndim == 3:
        # 可能是 [C,H,W] 或 [T,H,W]，一律取第 0 维的切片
        img = t[0]
    else:
        raise ValueError(f"Unsupported tensor shape for image: {t.shape}")
    # 归一化到 0-1 便于显示
    vmin = torch.quantile(img.flatten(), 0.01)
    vmax = torch.quantile(img.flatten(), 0.99)
    img = torch.clamp((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return img.numpy()

def save_triplet_grid(x, y_hat, y, save_path: str | Path, title: Optional[str] = None):
    """
    保存一行三图: Input | Output | Target
    - x: [C,H,W] or [T,H,W] or [C,T,H,W]（上层通常已对齐到 4D）
    - y_hat/y: [C,H,W]
    """
    # 统一取第一个时间/通道切片
    def pick_first(t):
        if t.ndim == 5:   # [C,T,H,W]?（非常规；保底分支）
            t = t[:,0]
        elif t.ndim == 4: # [C,T,H,W] or [T,C,H,W]
            # 假设 [C,T,H,W]
            t = t[:,0]
        elif t.ndim == 3: # [C,H,W] or [T,H,W] -> 取第 0 片
            pass
        elif t.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected ndim={t.ndim} for plotting")
        return t

    x1 = pick_first(x)
    y1 = pick_first(y_hat)
    t1 = pick_first(y)

    x_img = _to_hw_img(x1)
    y_img = _to_hw_img(y1)
    t_img = _to_hw_img(t1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    for i, (img, name) in enumerate([(x_img, "Input"), (y_img, "Output"), (t_img, "Target")], 1):
        plt.subplot(1, 3, i)
        plt.imshow(img)
        plt.title(name)
        plt.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
