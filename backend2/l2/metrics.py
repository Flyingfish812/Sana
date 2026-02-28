from __future__ import annotations

from typing import Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算回归任务的 MSE/MAE/RMSE。"""
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(max(mse, 0.0)))
    return {"mse": mse, "mae": mae, "rmse": rmse}
