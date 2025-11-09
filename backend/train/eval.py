"""Evaluation utilities for offline metrics and artefacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from backend.common import ensure_5d, ensure_dir, extract_xy, move_batch_to_device
from backend.viz.images import save_triplet_grid, save_quadruple_grid

from . import metrics as M

try:
    from . import spectral as S
except Exception:
    S = None  # 没有就跳过频域指标

def _as4d(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(2) if t.ndim == 5 else t

def _build_pyramid(x: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """
    简易高斯金字塔（blur+downsample 近似），返回 [L0(原), L1, ...]
    """
    out = [_as4d(x)]
    cur = out[0]
    for _ in range(1, max(1, int(levels))):
        # 轻量做法：avgpool 代替 blur+下采样；足以用于相对比较
        cur = F.avg_pool2d(cur, kernel_size=2, stride=2, ceil_mode=False)
        out.append(cur)
    return out

def _resolve_layout_tag(batch, layout_tag_key: str):
    # 试从 batch 的第三项(meta-like)或 dataset.meta 里拿
    tag = None
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        meta = batch[2]
        if isinstance(meta, dict):
            tag = meta.get(layout_tag_key, None)
    if tag is None:
        ds = getattr(batch, "dataset", None)
        if ds is not None and hasattr(ds, "meta"):
            if isinstance(ds.meta, dict):
                tag = ds.meta.get(layout_tag_key, None)
    return tag

@torch.no_grad()
def evaluate(model, test_dl, run_dir: Path, cfg_eval: Dict[str, Any]):
    """
    评估模型，写入 run_dir/eval_log.jsonl
    新增：
      - 相对误差标量：nmse/nmae；
      - 区域误差统计：region_nmse_max/mean 或 region_nmae_max/mean；
      - 逐像素相对误差分位数：rel_err_max/rel_err_pXX；
    频域与多尺度逻辑保持兼容。
    """
    if test_dl is None:
        return
    log_path = run_dir / "eval_log.jsonl"
    device = next(model.parameters()).device
    limit = int(cfg_eval.get("num_eval_batches", 3))

    factors = cfg_eval.get("factors", {})
    p = factors.get("sample_density", None) if isinstance(factors, dict) else None
    sigma = factors.get("noise_sigma", None) if isinstance(factors, dict) else None

    metric_names: List[str] = list(cfg_eval.get("metrics", []))
    use_scales = bool(cfg_eval.get("scales", {}).get("enable", False))
    n_levels   = int(cfg_eval.get("scales", {}).get("levels", 3))
    use_spectral = bool(cfg_eval.get("spectral", {}).get("enable", False))
    kbins      = int(cfg_eval.get("spectral", {}).get("kbins", 32))
    fft_pad    = bool(cfg_eval.get("spectral", {}).get("fft_pad", True))
    layout_key = cfg_eval.get("layout_tag_key", "layout_tag")
    write_per_item = bool(cfg_eval.get("write_per_item", False))

    # 新增配置：相对尺度与区域/分位
    rel_cfg   = cfg_eval.get("rel_error", {})
    use_rel   = bool(rel_cfg.get("enable", True))               # 默认启用相对误差标量
    pct_list  = tuple(rel_cfg.get("percentiles", [95, 99]))

    region_cfg = cfg_eval.get("region_error", {})
    use_region = bool(region_cfg.get("enable", False))
    tiles      = tuple(region_cfg.get("tiles", [3, 3]))
    region_use_nmse = bool(region_cfg.get("use_nmse", True))

    if not metric_names:
        metric_names = ["psnr"]

    METRIC_FNS = {
        "l1": M.l1, "mse": M.mse, "psnr": M.psnr,
        "corrcoef": M.corrcoef, "ssim": M.ssim,
        "grad_mse": getattr(M, "grad_mse", None),
        "lap_mse": getattr(M, "lap_mse", None),
        "tgrad_mse": getattr(M, "tgrad_mse", None),
        "vort_mse": getattr(M, "vort_mse", None),
        "vort_mae": getattr(M, "vort_mae", None),
        "vort_corr": getattr(M, "vort_corr", None),
        # 新增相对尺度标量（整体）
        "nmse": getattr(M, "nmse", None),
        "nmae": getattr(M, "nmae", None),
    }

    with log_path.open("a", encoding="utf-8") as fp:
        for bidx, batch in enumerate(test_dl):
            if bidx >= limit:
                break
            batch = move_batch_to_device(batch, device)
            x, y, _ = extract_xy(batch)
            y_hat = model(ensure_5d(x))

            # 统一到4D
            y4  = y.squeeze(2) if y.ndim == 5 else y
            yh4 = y_hat.squeeze(2) if y_hat.ndim == 5 else y_hat

            record: Dict[str, Any] = {}
            for name in metric_names:
                fn = METRIC_FNS.get(name, None)
                if fn is None:
                    continue
                try:
                    if name == "tgrad_mse":
                        val = float(fn(y_hat, y).detach().cpu().item())
                    else:
                        val = float(fn(yh4, y4).detach().cpu().item())
                    record[name] = val
                except Exception:
                    pass

            # 多尺度
            if use_scales and n_levels > 1:
                pyr_pred = _build_pyramid(yh4, n_levels)
                pyr_tgt  = _build_pyramid(y4,  n_levels)
                for li, (pp, tt) in enumerate(zip(pyr_pred, pyr_tgt)):
                    for name in metric_names:
                        fn = METRIC_FNS.get(name, None)
                        if fn is None:
                            continue
                        try:
                            record[f"{name}@L{li}"] = float(fn(pp, tt).detach().cpu().item())
                        except Exception:
                            pass

            # 频域指标（保持原貌）
            if use_spectral and S is not None:
                try:
                    spec = S.spectral_rrmse(yh4, y4, kbins=kbins, fft_pad=fft_pad)
                    record["spectral_rrmse"] = float(spec["overall"])
                    thirds = S.slice_rrmse_thirds(spec["per_bin"])
                    record["spectral_rrmse_low"]  = float(thirds["low"])
                    record["spectral_rrmse_mid"]  = float(thirds["mid"])
                    record["spectral_rrmse_high"] = float(thirds["high"])
                except Exception:
                    pass

            # 新增：整体相对误差（nmse/nmae 已在上面 metrics；若未列入 metrics 仍可强制补充）
            if use_rel:
                if "nmse" not in record and METRIC_FNS.get("nmse"):
                    try: record["nmse"] = float(METRIC_FNS["nmse"](yh4, y4).detach().cpu().item())
                    except Exception: pass
                if "nmae" not in record and METRIC_FNS.get("nmae"):
                    try: record["nmae"] = float(METRIC_FNS["nmae"](yh4, y4).detach().cpu().item())
                    except Exception: pass
                # 逐像素相对误差的极值/分位数
                try:
                    record.update(_relative_error_stats(yh4, y4, percentiles=pct_list))
                except Exception:
                    pass

            # 新增：区域误差统计
            if use_region:
                try:
                    record.update(_region_error_tiles(yh4, y4, tiles=tiles, use_nmse=region_use_nmse))
                except Exception:
                    pass

            layout_tag = _resolve_layout_tag_from_batch_or_dataset(batch, test_dl, layout_key)
            rec = {"p": p, "sigma": sigma, "layout_tag": layout_tag, **record}

            # 逐样本选写（维持原样）
            if write_per_item:
                # 仅为示例，保留原实现
                per_item = {}
                try:
                    # l1/mse/psnr 的逐样本写法（原逻辑）
                    diff_l1 = (yh4 - y4).abs().mean(dim=list(range(1, yh4.ndim)))
                    diff_m2 = ((yh4 - y4) ** 2).mean(dim=list(range(1, yh4.ndim)))
                    per_item["l1"]  = [float(v) for v in diff_l1.detach().cpu().flatten()]
                    per_item["mse"] = [float(v) for v in diff_m2.detach().cpu().flatten()]
                    mse_b = ((yh4 - y4) ** 2).mean(dim=list(range(1, yh4.ndim)))
                    psnr_b = 10.0 * torch.log10(1.0 / (mse_b + 1e-8))
                    per_item["psnr"] = [float(v) for v in psnr_b.detach().cpu().flatten()]
                except Exception:
                    pass
                if per_item:
                    rec["per_item"] = per_item

            fp.write(json.dumps(rec) + "\n")
            fp.flush()

# 解析批次里的 layout_tag（优先从 batch 元信息，再从 dataset.meta） ===
def _resolve_layout_tag_from_batch_or_dataset(batch, test_dl, layout_key: str = "layout_tag"):
    tag = None
    # 尝试 batch 的第3项（很多管线把 meta dict 放在 [2]）
    if isinstance(batch, (list, tuple)) and len(batch) >= 3 and isinstance(batch[2], dict):
        tag = batch[2].get(layout_key, None)
    # 兜底：看 DataLoader 的 dataset 是否暴露了 .meta
    if tag is None and getattr(test_dl, "dataset", None) is not None:
        meta = getattr(test_dl.dataset, "meta", None)
        if isinstance(meta, dict):
            tag = meta.get(layout_key, None)
    return tag

# 频谱的可视化（保存两张图：error heatmap / radial power spectrum） ===
def _save_error_and_spectrum(
    img_dir: Path,
    prefix: str,
    pred_4d: torch.Tensor,
    gt_4d: torch.Tensor,
    *,
    save_error_heatmap: bool = False,
    spectrum_log_scale: bool = True,
    kbins: int = 64,
    cmap: str = "RdBu_r",
) -> None:
    """
    生成频谱图，并可选保存误差热力图。
    频谱解释（写在注释内，报告也可引用）：
      - 功率谱定义：对 2D 图像 X 的傅里叶变换 F=FFT(X)，功率为 |F|^2；
      - 径向平均：以频率半径 r 分桶，对 |F|^2 在每个半径环带做平均，得到 P(r)；
      - 归一化频率：横轴 r∈[0,1]，0 为直流分量，1 约为 Nyquist；
      - 若 spectrum_log_scale=True，则绘制 10*log10(P+eps)（dB 标度），用于拉伸高低频动态范围；
      - 纵轴标签相应使用 'Radial power (dB)' 或 'Radial power'.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    with torch.no_grad():
        pred = pred_4d.detach().cpu().float().numpy()  # [C,H,W]
        gt   = gt_4d.detach().cpu().float().numpy()
        err  = np.abs(pred - gt)

        # 可选：误差热力图（通道均值）
        if save_error_heatmap:
            err_map = err.mean(axis=0)
            plt.figure()
            plt.imshow(err_map, interpolation="nearest", cmap=cmap)
            plt.title("Error heatmap (|pred - gt|)")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(img_dir / f"{prefix}_error.png", dpi=200)
            plt.close()

        # 径向功率谱（通道先均值，再做 FFT）
        def _radial_power(img2d: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
            F = np.fft.fftshift(np.fft.fft2(img2d, norm="ortho"))
            mag2 = np.abs(F) ** 2
            H, W = mag2.shape
            yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
            rr = np.sqrt(xx * xx + yy * yy)
            rr /= (rr.max() + 1e-12)
            bin_idx = np.clip((rr * bins).astype(np.int64), 0, bins - 1)
            ps = np.zeros(bins, dtype=np.float64)
            cnt = np.zeros(bins, dtype=np.float64)
            for k in range(bins):
                m = (bin_idx == k)
                c = m.sum()
                cnt[k] = c
                if c > 0:
                    ps[k] = mag2[m].mean()
            x = np.arange(bins) / (bins - 1 + 1e-12)
            return x, ps

        x, p_pred = _radial_power(pred.mean(axis=0), kbins)
        _, p_gt   = _radial_power(gt.mean(axis=0),   kbins)

        # 误差谱：用功率谱差的绝对值，仅作直观显示
        p_err = np.abs(p_pred - p_gt)

        eps = 1e-12
        if spectrum_log_scale:
            p_pred = 10.0 * np.log10(p_pred + eps)
            p_gt   = 10.0 * np.log10(p_gt   + eps)
            p_err  = 10.0 * np.log10(p_err  + eps)
            ylab   = "Radial power (dB)"
        else:
            ylab   = "Radial power"

        plt.figure()
        plt.plot(x, p_gt,   label="GT")
        plt.plot(x, p_pred, label="Pred")
        plt.plot(x, p_err,  label="|Pred−GT|")
        plt.xlabel("Normalized spatial frequency")
        plt.ylabel(ylab)
        plt.legend()
        plt.title("Radial Power Spectrum")
        plt.tight_layout()
        plt.savefig(img_dir / f"{prefix}_rps.png", dpi=200)
        plt.close()

# eval.py
@torch.no_grad()
def _relative_error_stats(
    pred4: torch.Tensor,
    tgt4: torch.Tensor,
    *,
    percentiles: tuple[float, ...] = (95.0, 99.0),
    eps: float = 1e-8,
) -> dict:
    """
    逐像素相对误差图 re = |ŷ−y| / (|y|+eps)，聚合出 max 与若干分位点。
    返回键：{"rel_err_max": x, "rel_err_p95": x, ...}
    """
    import numpy as np
    re = (pred4 - tgt4).abs() / (tgt4.abs() + eps)     # [B,C,H,W]
    # 按样本通道空间聚合为单向量
    vec = re.detach().cpu().reshape(-1).numpy()
    out = {"rel_err_max": float(np.max(vec)) if vec.size else 0.0}
    for q in percentiles:
        out[f"rel_err_p{int(q)}"] = float(np.percentile(vec, q)) if vec.size else 0.0
    return out

@torch.no_grad()
def _region_error_tiles(
    pred4: torch.Tensor,
    tgt4: torch.Tensor,
    *,
    tiles: tuple[int, int] = (3, 3),
    use_nmse: bool = True,
    eps: float = 1e-8,
) -> dict:
    """
    将图像分割为 R×C 小块，对每块计算 NMAE 或 NMSE，再对所有块统计 max/mean。
    返回键如：{"region_nmse_max": x, "region_nmse_mean": y}
    """
    B, C, H, W = pred4.shape
    R, Cc = int(tiles[0]), int(tiles[1])
    rh, rw = H // R, W // Cc

    vals = []
    for i in range(R):
        for j in range(Cc):
            y0, y1 = i * rh, (i + 1) * rh if i < R - 1 else H
            x0, x1 = j * rw, (j + 1) * rw if j < Cc - 1 else W
            pp = pred4[..., y0:y1, x0:x1]
            tt = tgt4[...,  y0:y1, x0:x1]
            if use_nmse:
                num = torch.sum((pp - tt) ** 2)
                den = torch.sum(tt ** 2) + eps
                v = (num / den).item()
            else:
                num = torch.sum((pp - tt).abs())
                den = torch.sum(tt.abs()) + eps
                v = (num / den).item()
            vals.append(v)

    import numpy as np
    arr = np.array(vals, dtype=float) if vals else np.array([0.0])
    tag = "nmse" if use_nmse else "nmae"
    return {
        f"region_{tag}_max":  float(np.max(arr)),
        f"region_{tag}_mean": float(np.mean(arr)),
    }

@torch.no_grad()
def render_eval_triplets(
    model,
    test_dl,
    run_dir: Path,
    cfg_eval: Dict[str, Any],
) -> Path:
    """Generate qualitative plots from the test set (quadruple + spectrum).
    - 四联图依旧包含误差子图；
    - 避免重复：此处默认不再单独保存误差热图，只保存频谱图；
    - 频谱默认使用 dB（spectrum_log_scale=True），kbins 可配置。
    """
    if test_dl is None:
        return run_dir / "eval_vis"

    img_dir = ensure_dir(run_dir / "eval_vis")
    device = next(model.parameters()).device
    model.eval(); model.to(device)

    max_batches = int(cfg_eval.get("num_eval_batches", 3))
    max_triplets = int(cfg_eval.get("num_plot_triplets", 4))

    # 可配置
    vis_cfg   = cfg_eval.get("eval_vis", {})
    save_err  = bool(vis_cfg.get("save_error_heatmap", False))      # 默认 False，避免重复
    logscale  = bool(vis_cfg.get("spectrum_log_scale", True))       # 默认 dB 轴
    kbins     = int(vis_cfg.get("spectrum_kbins", 64))
    cmap      = str(vis_cfg.get("cmap", "RdBu_r"))

    plotted, batches = 0, 0
    for batch in test_dl:
        if batches >= max_batches or plotted >= max_triplets:
            break

        batch = move_batch_to_device(batch, device)
        x, y, _ = extract_xy(batch)
        y_hat = model(ensure_5d(x))
        if x.ndim == 4:
            y_hat = y_hat.squeeze(2)

        take = min(x.shape[0], max_triplets - plotted)
        for idx in range(take):
            save_quadruple_grid(
                x[idx] if x.ndim == 4 else x[idx, :, 0],
                y_hat[idx] if y_hat.ndim == 4 else y_hat[idx, :, 0],
                y[idx] if y.ndim == 4 else y[idx, :, 0],
                img_dir / f"triplet_b{batches}_i{idx}.png",
            )
            pred_4d = y_hat[idx] if y_hat.ndim == 4 else y_hat[idx, :, 0]
            gt_4d   = y[idx]    if y.ndim == 4    else y[idx, :, 0]
            prefix  = f"triplet_b{batches}_i{idx}"
            _save_error_and_spectrum(
                img_dir, prefix, pred_4d, gt_4d,
                save_error_heatmap=save_err,
                spectrum_log_scale=logscale,
                kbins=kbins,
                cmap=cmap,
            )
            plotted += 1
        batches += 1
    return img_dir