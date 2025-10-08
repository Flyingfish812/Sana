# backend/train/metrics.py
from __future__ import annotations
import torch
import torch.nn.functional as F

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
