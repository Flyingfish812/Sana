# backend/dataio/transforms/coords.py
from __future__ import annotations
import numpy as np
from ..schema import ArraySample
from .base import Transform

class AddCoordsTransform(Transform):
    """
    为 sample.meta 注入 [H,W,2] 的归一化坐标网格（-1..1）。
    如果已有 coords_xy 则不覆盖（除非 force=True）。
    """
    def __init__(self, force: bool = False):
        self.force = force

    def __call__(self, sample: ArraySample) -> ArraySample:
        if sample.meta.coords_xy is not None and not self.force:
            return sample
        _, H, W, _ = sample.frames.shape  # frames: [K,H,W,C]
        ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        coords = np.stack([grid_x, grid_y], axis=-1)  # [H,W,2] (x,y)
        sample.meta.coords_xy = coords
        return sample
