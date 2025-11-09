# backend/train/metrics.py
from __future__ import annotations
import torch
import torch.nn.functional as F

def _as4d(t: torch.Tensor) -> torch.Tensor:
    # 统一到 [B,C,H,W]
    if t.ndim == 5:
        return t.squeeze(2)
    return t

def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 统一到 4D [B,C,H,W]
    def as4d(t):
        if t.ndim == 5: return t.squeeze(2)
        return t
    pred, target = as4d(pred), as4d(target)
    mse_val = F.mse_loss(pred, target, reduction="none")
    # 按样本聚合
    dims = list(range(1, mse_val.ndim))
    mse_b = mse_val.mean(dim=dims)
    psnr_b = 10.0 * torch.log10(1.0 / (mse_b + eps))
    return psnr_b.mean()

@torch.no_grad()
def corrcoef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 统一 4D
    if pred.ndim == 5: pred = pred.squeeze(2)
    if target.ndim == 5: target = target.squeeze(2)
    B = pred.shape[0]
    p = pred.view(B, -1)
    t = target.view(B, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = (p.norm(dim=1) * t.norm(dim=1)) + eps
    r = num / den
    return r.mean()

@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, K1=0.01, K2=0.03, win_size: int = 11, eps: float = 1e-8) -> torch.Tensor:
    """
    简化版 SSIM（Y 通道/单通道），窗口为均值卷积；只用于相对比较。
    """
    if pred.ndim == 5: pred = pred.squeeze(2)
    if target.ndim == 5: target = target.squeeze(2)
    # 仅取首通道计算（若多通道）
    pred = pred[:, :1]
    target = target[:, :1]
    C1 = (K1 ** 2)
    C2 = (K2 ** 2)

    pad = win_size // 2
    kernel = torch.ones((1, 1, win_size, win_size), device=pred.device) / (win_size * win_size)

    def filt(x):
        return torch.conv2d(x, kernel, padding=pad)

    mu_x = filt(pred); mu_y = filt(target)
    sigma_x = filt(pred * pred) - mu_x * mu_x
    sigma_y = filt(target * target) - mu_y * mu_y
    sigma_xy = filt(pred * target) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + eps)
    return ssim_map.mean()

@torch.no_grad()
def grad_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    一阶梯度域MSE：对x/y方向做简单差分，计算 (∇pred - ∇gt)^2 的均值。
    """
    pred = _as4d(pred); target = _as4d(target)
    # 简单前向差分（H-1, W-1 对齐）
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]

    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]

    mse_x = F.mse_loss(dx_p[..., 1:, :], dx_t[..., 1:, :])  # 对齐裁切，避免边界差异
    mse_y = F.mse_loss(dy_p[..., :, 1:], dy_t[..., :, 1:])
    return 0.5 * (mse_x + mse_y)

@torch.no_grad()
def lap_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    拉普拉斯域MSE：3x3离散拉普拉斯核卷积后计算 MSE。
    """
    pred = _as4d(pred); target = _as4d(target)
    lap_kernel = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]], device=pred.device, dtype=pred.dtype
    ).view(1, 1, 3, 3)
    # 针对多通道逐通道应用同一核
    C = pred.shape[1]
    kernel = lap_kernel.repeat(C, 1, 1, 1)
    pad = 1
    lap_p = torch.conv2d(pred, kernel, padding=pad, groups=C)
    lap_t = torch.conv2d(target, kernel, padding=pad, groups=C)
    return F.mse_loss(lap_p, lap_t)

@torch.no_grad()
def tgrad_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    时间梯度一致性 MSE：
    - 若是 5D [B,T,C,H,W]，计算 Δt 一阶差分的一致性误差；
    - 若是 4D [B,C,H,W]（无时间维），返回 0（或可选择返回 NaN，本文实现返回 0 保持评估不中断）。
    """
    if pred.ndim == 5 and target.ndim == 5:
        dp = pred[:, 1:, ...] - pred[:, :-1, ...]
        dt = target[:, 1:, ...] - target[:, :-1, ...]
        return F.mse_loss(dp, dt)
    else:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

@torch.no_grad()
def _vorticity_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    2D 标量涡量：omega = dV/dx - dU/dy
    输入: u,v 为 [B,C=1,H,W] 或 [B,H,W]；返回 [B,1,H,W]
    使用简单中心差分（边界用前向差分近似）。
    """
    if u.ndim == 3:  # [B,H,W] -> [B,1,H,W]
        u = u.unsqueeze(1); v = v.unsqueeze(1)
    # dx
    dVdx = v[..., :, :, 1:] - v[..., :, :, :-1]
    dVdx = torch.nn.functional.pad(dVdx, (1, 0, 0, 0))  # 左边界补齐
    # dy
    dUdy = u[..., :, 1:, :] - u[..., :, :-1, :]
    dUdy = torch.nn.functional.pad(dUdy, (0, 0, 1, 0))  # 上边界补齐
    omega = dVdx - dUdy
    return omega

@torch.no_grad()
def vort_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    涡量 MSE：当通道数>=2（假设 [u,v,...]）时计算 pred/gt 的标量涡量并比较。
    对 5D 时间序列将按全维度一起比较；对 4D 也可用。
    若通道不足，返回 0 保持流程稳定。
    """
    # 兼容 5D：拉平时间维，统一到 [B',C,H,W]
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.view(B * T, C, H, W)
        target = target.view(B * T, C, H, W)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    u_p, v_p = pred[:, 0:1], pred[:, 1:2]
    u_t, v_t = target[:, 0:1], target[:, 1:2]
    w_p = _vorticity_2d(u_p, v_p)
    w_t = _vorticity_2d(u_t, v_t)
    return F.mse_loss(w_p, w_t)

@torch.no_grad()
def vort_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.view(B * T, C, H, W)
        target = target.view(B * T, C, H, W)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    u_p, v_p = pred[:, 0:1], pred[:, 1:2]
    u_t, v_t = target[:, 0:1], target[:, 1:2]
    w_p = _vorticity_2d(u_p, v_p)
    w_t = _vorticity_2d(u_t, v_t)
    return F.l1_loss(w_p, w_t)

@torch.no_grad()
def vort_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.view(B * T, C, H, W)
        target = target.view(B * T, C, H, W)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    u_p, v_p = pred[:, 0:1], pred[:, 1:2]
    u_t, v_t = target[:, 0:1], target[:, 1:2]
    w_p = _vorticity_2d(u_p, v_p)  # [B,1,H,W]
    w_t = _vorticity_2d(u_t, v_t)
    # 与 corrcoef 相同的逐样本皮尔逊写法
    B = w_p.shape[0]
    p = w_p.view(B, -1)
    t = w_t.view(B, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = (p.norm(dim=1) * t.norm(dim=1)) + eps
    r = num / den
    return r.mean()

@torch.no_grad()
def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    归一化 MSE: ||ŷ-y||₂² / (||y||₂² + eps)
    返回 batch 平均标量。
    """
    pred = _as4d(pred); target = _as4d(target)
    num = torch.sum((pred - target) ** 2, dim=list(range(1, pred.ndim)))
    den = torch.sum(target ** 2, dim=list(range(1, target.ndim))) + eps
    return (num / den).mean()

@torch.no_grad()
def nmae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    归一化 MAE: ||ŷ-y||₁ / (||y||₁ + eps)
    返回 batch 平均标量。
    """
    pred = _as4d(pred); target = _as4d(target)
    num = torch.sum((pred - target).abs(), dim=list(range(1, pred.ndim)))
    den = torch.sum(target.abs(), dim=list(range(1, target.ndim))) + eps
    return (num / den).mean()
