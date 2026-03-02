from __future__ import annotations

from typing import Tuple

import numpy as np


def robust_vrange(a: np.ndarray, q: float = 0.99) -> Tuple[float, float]:
    flat = a[np.isfinite(a)]
    if flat.size == 0:
        return -1.0, 1.0
    v = float(np.quantile(np.abs(flat), q))
    v = max(v, 1e-6)
    return -v, v
