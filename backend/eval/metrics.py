# backend/eval/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

def _as4d(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 5:
        return t.squeeze(2)
    return t

def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.ndim == 5: pred = pred.squeeze(2)
    if target.ndim == 5: target = target.squeeze(2)
    mse_val = F.mse_loss(pred, target, reduction="none")
    dims = list(range(1, mse_val.ndim))
    mse_b = mse_val.mean(dim=dims)
    psnr_b = 10.0 * torch.log10(1.0 / (mse_b + eps))
    return psnr_b.mean()

@torch.no_grad()
def corrcoef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.ndim == 5: pred = pred.squeeze(2)
    if target.ndim == 5: target = target.squeeze(2)
    B = pred.shape[0]
    p = pred.view(B, -1); t = target.view(B, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = (p.norm(dim=1) * t.norm(dim=1)) + eps
    r = num / den
    return r.mean()

@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, K1=0.01, K2=0.03, win_size: int = 11, eps: float = 1e-8) -> torch.Tensor:
    if pred.ndim == 5: pred = pred.squeeze(2)
    if target.ndim == 5: target = target.squeeze(2)
    pred = pred[:, :1]; target = target[:, :1]
    C1 = (K1 ** 2); C2 = (K2 ** 2)
    pad = win_size // 2
    kernel = torch.ones((1, 1, win_size, win_size), device=pred.device) / (win_size * win_size)
    def filt(x): return torch.conv2d(x, kernel, padding=pad)
    mu_x = filt(pred); mu_y = filt(target)
    sigma_x = filt(pred * pred) - mu_x * mu_x
    sigma_y = filt(target * target) - mu_y * mu_y
    sigma_xy = filt(pred * target) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + eps)
    return ssim_map.mean()

@torch.no_grad()
def grad_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as4d(pred); target = _as4d(target)
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    mse_x = F.mse_loss(dx_p[..., 1:, :], dx_t[..., 1:, :])
    mse_y = F.mse_loss(dy_p[..., :, 1:], dy_t[..., :, 1:])
    return 0.5 * (mse_x + mse_y)

@torch.no_grad()
def lap_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as4d(pred); target = _as4d(target)
    lap_kernel = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=pred.device, dtype=pred.dtype).view(1,1,3,3)
    C = pred.shape[1]
    kernel = lap_kernel.repeat(C, 1, 1, 1)
    lap_p = torch.conv2d(pred, kernel, padding=1, groups=C)
    lap_t = torch.conv2d(target, kernel, padding=1, groups=C)
    return F.mse_loss(lap_p, lap_t)

@torch.no_grad()
def tgrad_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 5 and target.ndim == 5:
        dp = pred[:, 1:, ...] - pred[:, :-1, ...]
        dt = target[:, 1:, ...] - target[:, :-1, ...]
        return F.mse_loss(dp, dt)
    else:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

@torch.no_grad()
def _vorticity_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if u.ndim == 3:  u = u.unsqueeze(1); v = v.unsqueeze(1)
    dVdx = v[..., :, :, 1:] - v[..., :, :, :-1]
    dVdx = torch.nn.functional.pad(dVdx, (1, 0, 0, 0))
    dUdy = u[..., :, 1:, :] - u[..., :, :-1, :]
    dUdy = torch.nn.functional.pad(dUdy, (0, 0, 1, 0))
    return dVdx - dUdy

@torch.no_grad()
def vort_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 5:
        B,T,C,H,W = pred.shape
        pred = pred.view(B*T, C, H, W); target = target.view(B*T, C, H, W)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    u_p, v_p = pred[:,0:1], pred[:,1:2]
    u_t, v_t = target[:,0:1], target[:,1:2]
    w_p = _vorticity_2d(u_p, v_p); w_t = _vorticity_2d(u_t, v_t)
    return F.mse_loss(w_p, w_t)

@torch.no_grad()
def vort_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 5:
        B,T,C,H,W = pred.shape
        pred = pred.view(B*T, C, H, W); target = target.view(B*T, C, H, W)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    u_p, v_p = pred[:,0:1], pred[:,1:2]
    u_t, v_t = target[:,0:1], target[:,1:2]
    w_p = _vorticity_2d(u_p, v_p); w_t = _vorticity_2d(u_t, v_t)
    return F.l1_loss(w_p, w_t)

@torch.no_grad()
def vort_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.ndim == 5:
        B,T,C,H,W = pred.shape
        pred = pred.view(B*T, C, H, W); target = target.view(B*T, C, H, W)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    u_p, v_p = pred[:,0:1], pred[:,1:2]
    u_t, v_t = target[:,0:1], target[:,1:2]
    w_p = _vorticity_2d(u_p, v_p); w_t = _vorticity_2d(u_t, v_t)
    B2 = w_p.shape[0]
    p = w_p.view(B2, -1); t = w_t.view(B2, -1)
    p = p - p.mean(dim=1, keepdim=True); t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1); den = (p.norm(dim=1) * t.norm(dim=1)) + eps
    r = num / den
    return r.mean()

@torch.no_grad()
def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = _as4d(pred); target = _as4d(target)
    num = torch.sum((pred - target) ** 2, dim=list(range(1, pred.ndim)))
    den = torch.sum(target ** 2, dim=list(range(1, target.ndim))) + eps
    return (num / den).mean()

@torch.no_grad()
def nmae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = _as4d(pred); target = _as4d(target)
    num = torch.sum((pred - target).abs(), dim=list(range(1, target.ndim)))
    den = torch.sum(target.abs(), dim=list(range(1, target.ndim))) + eps
    return (num / den).mean()

def metric_specs() -> dict:
    """
    指标词典：简称→{fullname, formula, direction, meaning, typical}
    """
    # 直接移植你现有词典（含 nmse/nmae 两项）
    return {
        "psnr": {
            "fullname": "Peak Signal-to-Noise Ratio",
            "formula": "PSNR = 10*log10(MAX^2 / MSE)，MAX=1（已归一化）",
            "direction": "higher",
            "meaning": "整体像素级重建质量，越高误差越小。",
            "typical": "25–35 dB 一般，35–45 dB 良好，>45 dB 极好（视任务不同）。",
        },
        "ssim": {
            "fullname": "Structural Similarity Index",
            "formula": "比较亮度/对比度/结构的相似度，范围[0,1]。",
            "direction": "higher",
            "meaning": "结构与对比一致性，越高越接近真值。",
            "typical": "0.8–0.9 一般，>0.95 很好。",
        },
        "corrcoef": {
            "fullname": "Pearson Correlation Coefficient",
            "formula": "对pred/gt展平后计算皮尔逊相关系数r∈[-1,1]。",
            "direction": "higher",
            "meaning": "线性相关程度，1表示完全线性一致。",
            "typical": "0.9以上普遍较好。",
        },
        "l1": {
            "fullname": "Mean Absolute Error",
            "formula": "MAE = |pred-gt| 的逐像素均值。",
            "direction": "lower",
            "meaning": "平均绝对偏差，越小越好。",
            "typical": "",
        },
        "mse": {
            "fullname": "Mean Squared Error",
            "formula": "MSE = (pred-gt)^2 的均值。",
            "direction": "lower",
            "meaning": "平方误差，受大误差更敏感。",
            "typical": "",
        },
        "grad_mse": {
            "fullname": "Gradient MSE",
            "formula": "在x/y梯度域比较：MSE(∇pred, ∇gt)。",
            "direction": "lower",
            "meaning": "边缘/纹理锐利度的一致性，小表示细节更准。",
            "typical": "",
        },
        "lap_mse": {
            "fullname": "Laplacian MSE",
            "formula": "在拉普拉斯域比较：MSE(Δpred, Δgt)。",
            "direction": "lower",
            "meaning": "二阶结构/纹理残差的一致性。",
            "typical": "",
        },
        "tgrad_mse": {
            "fullname": "Temporal Gradient MSE",
            "formula": "时间差分一致性：MSE(Δ_t pred, Δ_t gt)。",
            "direction": "lower",
            "meaning": "时序平滑与动态一致性，小表示时间一致性更好。",
            "typical": "",
        },
        "vort_mse": {
            "fullname": "Vorticity MSE",
            "formula": "以(u,v)计算涡量ω，比较 MSE(ω_pred, ω_gt)。",
            "direction": "lower",
            "meaning": "流场旋度重建准确性。",
            "typical": "",
        },
        "vort_mae": {
            "fullname": "Vorticity MAE",
            "formula": "MAE(ω_pred, ω_gt)。",
            "direction": "lower",
            "meaning": "与vort_mse同义，但对异常值更稳健。",
            "typical": "",
        },
        "vort_corr": {
            "fullname": "Vorticity Correlation",
            "formula": "corr(ω_pred, ω_gt)。",
            "direction": "higher",
            "meaning": "旋度的整体相似度。",
            "typical": "",
        },
        "spectral_rrmse": {
            "fullname": "Spectral Relative RMSE (overall)",
            "formula": "对径向频带功率做相对误差：||P−T||^2 / (||T||^2+ε)。",
            "direction": "lower",
            "meaning": "频域整体能量匹配度，小表示频谱更贴近真值。",
            "typical": "",
        },
        "spectral_rrmse_low": {
            "fullname": "Spectral Relative RMSE (low band)",
            "formula": "同上，低频段平均。",
            "direction": "lower",
            "meaning": "低频（整体轮廓/光滑区）重建质量。",
            "typical": "",
        },
        "spectral_rrmse_mid": {
            "fullname": "Spectral Relative RMSE (mid band)",
            "formula": "同上，中频段平均。",
            "direction": "lower",
            "meaning": "中频（纹理主体）重建质量。",
            "typical": "",
        },
        "spectral_rrmse_high": {
            "fullname": "Spectral Relative RMSE (high band)",
            "formula": "同上，高频段平均。",
            "direction": "lower",
            "meaning": "高频（边缘/细节/噪点）重建质量。",
            "typical": "",
        },
        "nmae": {
            "fullname": "Normalized Mean Absolute Error",
            "formula": "NMAE = mean(|pred - gt|) / mean(|gt| + ε)",
            "direction": "lower",
            "meaning": "归一化绝对误差，衡量平均偏差占真实值平均幅度的比例。",
            "typical": "0.0 理想，<0.05 很好，>0.1 偏差明显。",
        },
        "nmse": {
            "fullname": "Normalized Mean Squared Error",
            "formula": "NMSE = mean((pred - gt)^2) / mean(gt^2 + ε)",
            "direction": "lower",
            "meaning": "归一化平方误差，反映整体能量比例偏差。",
            "typical": "0.0 理想，<0.01 很好，>0.05 偏差较大。",
        },
    }

def is_error_like(direction: str) -> bool:
    return (direction or "higher").lower().strip() == "lower"

def summarize_metric_grid(P: np.ndarray, S: np.ndarray, M: np.ndarray, direction: str) -> dict:
    """
    概览：最佳值/位置、均值/标准差、覆盖率、0–100相对评分。
    """
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(M)
        coverage = float(valid.mean())
        if valid.sum() == 0:
            return {"best_val": None, "best_p": None, "best_s": None, "mean": None, "std": None,
                    "coverage": 0.0, "score": 0.0}

        vals = M[valid]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))

        if is_error_like(direction):
            best_val = float(np.nanmin(vals))
            where = np.where(M == best_val)
        else:
            best_val = float(np.nanmax(vals))
            where = np.where(M == best_val)
        i = int(where[0][0]); j = int(where[1][0])
        best_p = float(P[i]); best_s = float(S[j])

        q_lo, q_hi = np.nanpercentile(vals, [5, 95])
        denom = (q_hi - q_lo) if (q_hi > q_lo) else (np.nanmax(vals) - np.nanmin(vals) + 1e-9)
        if is_error_like(direction):
            norm = np.clip((q_hi - vals) / (denom + 1e-9), 0, 1)
        else:
            norm = np.clip((vals - q_lo) / (denom + 1e-9), 0, 1)
        score = float(100.0 * np.nanmean(norm))

        return {
            "best_val": best_val, "best_p": best_p, "best_s": best_s,
            "mean": mean, "std": std, "coverage": coverage, "score": score
        }
