# backend/train/spectral.py
from __future__ import annotations
from typing import Dict
import torch
import torch.fft as TFF

@torch.no_grad()
def fft2_mag(x: torch.Tensor, pad: bool = True) -> torch.Tensor:
    """
    x: [B,C,H,W] -> 幅度谱 [B,C,H,W]（中心化）
    """
    if pad:
        H, W = x.shape[-2:]
        ph = 1 << (H - 1).bit_length()
        pw = 1 << (W - 1).bit_length()
        x = torch.nn.functional.pad(x, (0, pw - W, 0, ph - H))
    X = TFF.fftshift(TFF.fft2(x, norm="ortho"), dim=(-2, -1))
    mag = torch.abs(X)
    return mag

def _radial_bins(h: int, w: int, kbins: int, device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij"
    )
    rr = torch.sqrt(xx * xx + yy * yy)  # [H,W] in [0,~1.414]
    rr = rr / rr.max().clamp(min=1e-6)  # 归一化到 [0,1]
    bins = torch.clamp((rr * kbins).long(), 0, kbins - 1)
    return bins  # [H,W] int64

@torch.no_grad()
def spectral_rrmse(pred: torch.Tensor, target: torch.Tensor, kbins: int = 32, fft_pad: bool = True) -> Dict[str, torch.Tensor]:
    """
    计算径向谱相对均方误差：
    RRMSE = ||P_k - T_k||_2^2 / (||T_k||_2^2 + eps) ，按 k-bin 统计并给出 overall。
    返回:
      { "per_bin": [kbins], "overall": scalar }
    """
    pred = pred.squeeze(2) if pred.ndim == 5 else pred
    target = target.squeeze(2) if target.ndim == 5 else target
    mag_p = fft2_mag(pred, pad=fft_pad)  # [B,C,H,W]
    mag_t = fft2_mag(target, pad=fft_pad)

    H, W = mag_p.shape[-2:]
    bins = _radial_bins(H, W, kbins, device=mag_p.device)  # [H,W]

    # 逐 bin 聚合功率
    per_bin_p = torch.zeros(kbins, device=mag_p.device)
    per_bin_t = torch.zeros(kbins, device=mag_p.device)
    for k in range(kbins):
        mask = (bins == k).float()[None, None, ...]  # [1,1,H,W]
        num = (mag_p * mag_p * mask).sum()
        den = (mag_t * mag_t * mask).sum()
        per_bin_p[k] = num
        per_bin_t[k] = den

    eps = 1e-8
    rrmse_bins = torch.pow(per_bin_p - per_bin_t, 2) / (per_bin_t + eps)
    overall = rrmse_bins.mean()
    return {"per_bin": rrmse_bins, "overall": overall}

@torch.no_grad()
def slice_rrmse_thirds(rrmse_bins: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    把 kbins 切为低/中/高三段，返回各段均值，便于快速汇报。
    """
    K = rrmse_bins.numel()
    a = K // 3
    b = 2 * (K // 3)
    low = rrmse_bins[:a].mean()
    mid = rrmse_bins[a:b].mean()
    high = rrmse_bins[b:].mean()
    return {"low": low, "mid": mid, "high": high}
