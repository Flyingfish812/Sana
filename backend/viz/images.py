"""Shared image visualisation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

try:  # pragma: no cover - optional dependency in CI
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may be missing
    plt = None  # type: ignore


def _squeeze_to_hw(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().float().cpu()
    while t.ndim > 2:
        t = t[0]
    if t.ndim != 2:
        raise ValueError(f"Unsupported tensor shape for plotting: {tuple(t.shape)}")
    return t


def tensor_to_hw_image(t: torch.Tensor, quantiles: Sequence[float] = (0.01, 0.99)) -> torch.Tensor:
    """Convert arbitrary layout tensors to a normalised ``[H,W]`` image tensor."""

    img = _squeeze_to_hw(t)
    q_low, q_high = quantiles
    flat = img.flatten()
    vmin = torch.quantile(flat, q_low)
    vmax = torch.quantile(flat, q_high)
    if torch.isclose(vmax, vmin):
        vmax = vmin + 1.0
    img = torch.clamp((img - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    return img


def save_triplet_grid(
    x: torch.Tensor,
    y_hat: torch.Tensor,
    y: torch.Tensor,
    save_path: str | Path,
    *,
    title: Optional[str] = None,
    labels: Iterable[str] = ("Input", "Output", "Target"),
) -> bool:
    """Save a horizontal triplet grid to ``save_path``.

    Returns ``True`` if the figure was written. When Matplotlib is not available the
    function is a no-op and returns ``False``.
    """

    if plt is None:
        return False

    images = [tensor_to_hw_image(t).numpy() for t in (x, y_hat, y)]
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4))
    for idx, (img, label) in enumerate(zip(images, labels), start=1):
        ax = fig.add_subplot(1, 3, idx)
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
