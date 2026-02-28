# backend/eval/runtime.py
"""Evaluation utilities for offline metrics and artefacts (moved from backend/train/eval)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only

from backend.common import ensure_5d, ensure_dir, extract_xy, move_batch_to_device
from backend.viz.images import save_quadruple_grid

# 指标评估
from . import metrics as M
try:
    from . import spectral as S  # type: ignore
except Exception:
    S = None

def _as4d(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(2) if t.ndim == 5 else t


def _build_pyramid(x: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """
    简易高斯金字塔（avgpool 近似），返回 [L0(原), L1, ...]
    """
    out = [_as4d(x)]
    cur = out[0]
    for _ in range(1, max(1, int(levels))):
        cur = F.avg_pool2d(cur, kernel_size=2, stride=2, ceil_mode=False)
        out.append(cur)
    return out

def _resolve_layout_tag(batch, layout_tag_key: str):
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

@rank_zero_only
@torch.no_grad()
def evaluate(model, test_dl, run_dir: Path, cfg_eval: Dict[str, Any]):
    """
    评估模型，写入 run_dir/eval_log.jsonl

    新增：
    - 逐通道指标：对 y 的每个通道分别计算 metrics，键名格式："{metric}/{ch_name}"
      * ch_name 来源：test_dl.dataset.meta["channel_names"]["out_names"]；否则回退为 "c{i}"
    - 总体指标仍保留为 "{metric}"（对所有通道整体/按C维聚合）
    - 与原版一致：相对/区域误差、多尺度参考匹配、频域指标、样本分布输出等
    """
    if test_dl is None:
        return

    # --------- 配置与公共项 ---------
    log_path = run_dir / "eval_log.jsonl"
    device = next(model.parameters()).device
    limit = int(cfg_eval.get("num_eval_batches", 3))

    factors = cfg_eval.get("factors", {})
    p = factors.get("sample_density", None) if isinstance(factors, dict) else None
    sigma = factors.get("noise_sigma", None) if isinstance(factors, dict) else None

    metric_names: List[str] = list(cfg_eval.get("metrics", []))
    use_scales   = bool(cfg_eval.get("scales", {}).get("enable", False))
    n_levels     = int(cfg_eval.get("scales", {}).get("levels", 3))
    use_spectral = bool(cfg_eval.get("spectral", {}).get("enable", False))
    kbins        = int(cfg_eval.get("spectral", {}).get("kbins", 32))
    fft_pad      = bool(cfg_eval.get("spectral", {}).get("fft_pad", True))
    layout_key   = cfg_eval.get("layout_tag_key", "layout_tag")
    write_per_item = bool(cfg_eval.get("write_per_item", False))

    # 逐通道输出控制
    per_ch_cfg  = cfg_eval.get("per_channel", {}) or {}
    per_ch_enable = bool(per_ch_cfg.get("enable", True))   # 默认开启
    per_ch_metrics: Sequence[str] = list(per_ch_cfg.get("metrics", [])) or metric_names  # 可单独指定要逐通道统计的 metrics

    # 相对/区域误差
    rel_cfg   = cfg_eval.get("rel_error", {})
    use_rel   = bool(rel_cfg.get("enable", True))
    pct_list  = tuple(rel_cfg.get("percentiles", [95, 99]))

    region_cfg = cfg_eval.get("region_error", {})
    use_region = bool(region_cfg.get("enable", False))
    tiles      = tuple(region_cfg.get("tiles", [3, 3]))
    region_use_nmse = bool(region_cfg.get("use_nmse", True))

    # 多尺度参考匹配
    ms_cfg       = cfg_eval.get("multiscale_ref", {}) or {}
    ms_enable    = bool(ms_cfg.get("enable", False))
    ms_kernels   = list(ms_cfg.get("kernel_sizes", [3, 5, 7, 9, 11]))
    ms_ref_mode  = str(ms_cfg.get("ref_mode", "gauss_down_up"))
    ms_upsample  = str(ms_cfg.get("upsample", "bicubic"))
    ms_metrics   = list(ms_cfg.get("metrics", [])) or []
    ms_dump_curv = bool(ms_cfg.get("dump_curves", False))

    if not metric_names:
        metric_names = ["psnr"]
    if not ms_metrics:
        ms_metrics = list(metric_names)

    METRIC_FNS = {
        "l1": M.l1, "mse": M.mse, "psnr": M.psnr,
        "corrcoef": M.corrcoef, "ssim": M.ssim,
        "grad_mse": getattr(M, "grad_mse", None),
        "lap_mse": getattr(M, "lap_mse", None),
        "tgrad_mse": getattr(M, "tgrad_mse", None),
        "vort_mse": getattr(M, "vort_mse", None),
        "vort_mae": getattr(M, "vort_mae", None),
        "vort_corr": getattr(M, "vort_corr", None),
        "nmse": getattr(M, "nmse", None),
        "nmae": getattr(M, "nmae", None),
    }

    # --------- 工具：读取通道名（输出侧） ---------
    def _get_out_channel_names(dl) -> List[str]:
        names: List[str] = []
        try:
            ds = getattr(dl, "dataset", None)
            meta = getattr(ds, "meta", None) if ds is not None else None
            if isinstance(meta, dict):
                ch = meta.get("channel_names") or {}
                outs = ch.get("out_names")
                if isinstance(outs, (list, tuple)):
                    names = list(outs)
        except Exception:
            names = []
        return names

    out_names = _get_out_channel_names(test_dl)

    with log_path.open("a", encoding="utf-8") as fp:
        for bidx, batch in enumerate(test_dl):
            if bidx >= limit:
                break

            batch = move_batch_to_device(batch, device)
            x, y, _ = extract_xy(batch)
            y_hat = model(ensure_5d(x))

            # 统一成 4D [B,C,H,W]
            y4  = y.squeeze(2) if y.ndim == 5 else y
            yh4 = y_hat.squeeze(2) if y_hat.ndim == 5 else y_hat

            B, C, H, W = yh4.shape

            # 准备记录对象
            record: Dict[str, Any] = {}

            # —— 整体指标（沿用原逻辑）—— #
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

            # —— 逐通道指标（新增） —— #
            if per_ch_enable and C >= 1:
                # 通道名回退
                if not out_names or len(out_names) != C:
                    ch_names = [f"c{i}" for i in range(C)]
                else:
                    ch_names = list(out_names)

                for ci in range(C):
                    yi  = y4[:, ci:ci+1]   # 保持 [B,1,H,W]
                    yhi = yh4[:, ci:ci+1]
                    tag = ch_names[ci]
                    for name in per_ch_metrics:
                        fn = METRIC_FNS.get(name, None)
                        if fn is None:
                            continue
                        try:
                            if name == "tgrad_mse":
                                # tgrad 需要 5D；逐通道退化到 4D 不合适，跳过或使用整体的 tgrad
                                continue
                            valc = float(fn(yhi, yi).detach().cpu().item())
                            record[f"{name}/{tag}"] = valc
                        except Exception:
                            pass

            # —— 金字塔（整体）—— #
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

            # —— 频域（整体）—— #
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

            # —— 相对/区域误差（整体）—— #
            if use_rel:
                if "nmse" not in record and METRIC_FNS.get("nmse"):
                    try: record["nmse"] = float(METRIC_FNS["nmse"](yh4, y4).detach().cpu().item())
                    except Exception: pass
                if "nmae" not in record and METRIC_FNS.get("nmae"):
                    try: record["nmae"] = float(METRIC_FNS["nmae"](yh4, y4).detach().cpu().item())
                    except Exception: pass
                try:
                    record.update(_relative_error_stats(yh4, y4, percentiles=pct_list))
                except Exception:
                    pass

            if use_region:
                try:
                    record.update(_region_error_tiles(yh4, y4, tiles=tiles, use_nmse=region_use_nmse))
                except Exception:
                    pass

            # —— 多尺度参考匹配（整体）—— #
            if ms_enable and ms_kernels:
                try:
                    ms_out = _score_vs_scales(
                        yh4, y4,
                        kernel_sizes=ms_kernels,
                        metrics=ms_metrics,
                        metric_fns=METRIC_FNS,
                        ref_mode=ms_ref_mode,
                        upsample=ms_upsample,
                        dump_curves=ms_dump_curv,
                    )
                    record.update(ms_out)
                except Exception:
                    pass

            layout_tag = _resolve_layout_tag_from_batch_or_dataset(batch, test_dl, layout_key)
            rec = {"p": p, "sigma": sigma, "layout_tag": layout_tag, **record}

            # —— 每样本分布（整体） —— #
            if write_per_item:
                per_item = {}
                try:
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

            with log_path.open("a", encoding="utf-8") as _fp:
                _fp.write(json.dumps(rec) + "\n")
                _fp.flush()

    # 多尺度可视化依然在末尾触发（与原版一致）
    ensure_eval_multiscale_vis(model, test_dl, run_dir, cfg_eval)

def _resolve_layout_tag_from_batch_or_dataset(batch, test_dl, layout_key: str = "layout_tag"):
    tag = None
    if isinstance(batch, (list, tuple)) and len(batch) >= 3 and isinstance(batch[2], dict):
        tag = batch[2].get(layout_key, None)
    if tag is None and getattr(test_dl, "dataset", None) is not None:
        meta = getattr(test_dl.dataset, "meta", None)
        if isinstance(meta, dict):
            tag = meta.get(layout_key, None)
    return tag

def _save_error_and_spectrum(
    img_dir: Path,
    prefix: str,
    pred_4d: torch.Tensor,   # [B,C,H,W]
    gt_4d: torch.Tensor,     # [B,C,H,W]
    *,
    save_error_heatmap: bool = False,
    spectrum_log_scale: bool = True,
    kbins: int = 64,
    cmap: str = "RdBu_r",
    channel_names: Optional[Sequence[str]] = None,  # 新增：可传 ["u","v"]；不传则用 c0,c1...
) -> None:
    """
    频谱评估（单/多通道自适应）：
    - 单通道：与旧版一致：GT、Pred、|Pred−GT|
    - 多通道（C>1）：
        1) 逐通道谱：GT / Pred / |Pred−GT|，文件名：..._{name}.png
        2) 总动能谱：Σ_c P_c(k)（逐通道径向功率之和），文件名：..._total.png
        3) 互谱相干度：γ^2_uv(k)（目前仅对 C==2 计算），文件名：..._coh_u_v.png
    """
    import numpy as np
    import matplotlib.pyplot as plt

    with torch.no_grad():
        pred = pred_4d.detach().cpu().float().numpy()  # [B,C,H,W]
        gt   = gt_4d.detach().cpu().float().numpy()    # [B,C,H,W]
        err  = np.abs(pred - gt)

        B, C = pred.shape[:2]
        names = list(channel_names) if (channel_names and len(channel_names) == C) else [f"c{i}" for i in range(C)]

        # ---------- 工具：把 [B,C,H,W] 压到单幅 [H,W] ----------
        def to_hw(a):  # a: [...,H,W]
            a = np.asarray(a)
            if a.ndim == 2:
                return a
            h, w = a.shape[-2], a.shape[-1]
            return a.reshape(-1, h, w).mean(axis=0)  # 默认先对 batch/通道均值；上层会控制取哪个维度

        # ---------- 工具：径向功率（自适应输入，保证 2D） ----------
        def radial_power(img, bins: int) -> tuple[np.ndarray, np.ndarray]:
            a = np.asarray(img)
            if a.ndim > 2:
                a = to_hw(a)
            F = np.fft.fftshift(np.fft.fft2(a, norm="ortho"))
            mag2 = np.abs(F) ** 2
            H, W = mag2.shape
            # 以像素半径分箱（1..bins），跳过 DC
            cy, cx = H//2, W//2
            yy, xx = np.ogrid[:H, :W]
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            r_max = int(r.max())
            nb = min(bins, max(r_max, 1))
            edges = np.linspace(1, r_max, nb + 1)
            ps = np.zeros(nb, dtype=np.float64)
            for i in range(nb):
                m = (r >= edges[i]) & (r < edges[i+1])
                if m.any():
                    ps[i] = mag2[m].mean()
            radii = 0.5 * (edges[:-1] + edges[1:])
            # 归一到 [0,1] 的“归一化波数”坐标，便于横轴对齐
            x = radii / (radii.max() + 1e-12)
            return x, ps

        # ---------- 可选：误差热力图（通道均值） ----------
        if save_error_heatmap:
            err_map = err.mean(axis=(0, 1))  # [H,W]
            plt.figure()
            plt.imshow(err_map, interpolation="nearest", cmap=cmap)
            plt.title("Error heatmap (|pred - gt|), mean over B,C")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(img_dir / f"{prefix}_error.png", dpi=200)
            plt.close()

        eps = 1e-12
        def _maybe_db(y):
            return 10.0 * np.log10(y + eps) if spectrum_log_scale else y
        ylab = "Radial power (dB)" if spectrum_log_scale else "Radial power"

        # =============== 情况 A：单通道 =================
        if C == 1:
            x, p_pred = radial_power(pred.mean(axis=0), kbins)  # [C,B,H,W]→均值到 [H,W]
            _, p_gt   = radial_power(gt.mean(axis=0),   kbins)
            p_err = np.abs(p_pred - p_gt)

            plt.figure()
            plt.plot(x, _maybe_db(p_gt),   label="GT")
            plt.plot(x, _maybe_db(p_pred), label="Pred")
            plt.plot(x, _maybe_db(p_err),  label="|Pred−GT|")
            plt.xlabel("Normalized spatial frequency")
            plt.ylabel(ylab)
            plt.legend()
            plt.title("Radial Power Spectrum")
            plt.tight_layout()
            plt.savefig(img_dir / f"{prefix}_rps.png", dpi=200)
            plt.close()
            return

        # =============== 情况 B：多通道 =================
        # 1) 逐通道谱
        per_ch = {}   # name -> (x, p_gt, p_pred, p_err)
        for ci, nm in enumerate(names):
            x, p_pred = radial_power(pred[:, ci], kbins)  # 对 batch 维均值
            _, p_gt   = radial_power(gt[:, ci],   kbins)
            p_err = np.abs(p_pred - p_gt)
            per_ch[nm] = (x, p_gt, p_pred, p_err)

            plt.figure()
            plt.plot(x, _maybe_db(p_gt),   label=f"GT[{nm}]")
            plt.plot(x, _maybe_db(p_pred), label=f"Pred[{nm}]")
            plt.plot(x, _maybe_db(p_err),  label=f"|Pred−GT|[{nm}]")
            plt.xlabel("Normalized spatial frequency")
            plt.ylabel(ylab)
            plt.legend()
            plt.title(f"Radial Power Spectrum — {nm}")
            plt.tight_layout()
            plt.savefig(img_dir / f"{prefix}_rps_{nm}.png", dpi=200)
            plt.close()

        # 2) 总“动能”谱（逐通道功率之和）
        #    注意：能量应当是功率谱的和，而不是先把场相加再做谱
        #    这里直接把各通道的径向功率序列相加（GT 与 Pred 各自相加）
        #    （如果你未来做 3D/T 频谱，可换成在环上对 |F|^2 先求和再分箱）
        x_ref = next(iter(per_ch.values()))[0]
        p_gt_sum   = np.zeros_like(x_ref)
        p_pred_sum = np.zeros_like(x_ref)
        for (x, p_gt_c, p_pred_c, _) in per_ch.values():
            p_gt_sum   = p_gt_sum   + p_gt_c
            p_pred_sum = p_pred_sum + p_pred_c
        p_err_sum = np.abs(p_pred_sum - p_gt_sum)

        plt.figure()
        plt.plot(x_ref, _maybe_db(p_gt_sum),   label="GT (Σ power)")
        plt.plot(x_ref, _maybe_db(p_pred_sum), label="Pred (Σ power)")
        plt.plot(x_ref, _maybe_db(p_err_sum),  label="|Pred−GT| (Σ)")
        plt.xlabel("Normalized spatial frequency")
        plt.ylabel(ylab)
        plt.legend()
        plt.title("Radial Power Spectrum — Total (Σ over channels)")
        plt.tight_layout()
        plt.savefig(img_dir / f"{prefix}_rps_total.png", dpi=200)
        plt.close()

        # 3) 互谱相干度（目前仅在 C==2 时计算）
        if C == 2:
            # 先把 batch 均值到单幅，再做互谱
            def fft2(img):
                F = np.fft.fftshift(np.fft.fft2(img, norm="ortho"))
                return F

            u_pred = to_hw(pred[:, 0])  # [H,W]
            v_pred = to_hw(pred[:, 1])
            u_gt   = to_hw(gt[:, 0])
            v_gt   = to_hw(gt[:, 1])

            Fu_pred, Fv_pred = fft2(u_pred), fft2(v_pred)
            Fu_gt,   Fv_gt   = fft2(u_gt),   fft2(v_gt)

            # 功率 & 互谱
            Puu_pred = np.abs(Fu_pred) ** 2
            Pvv_pred = np.abs(Fv_pred) ** 2
            Puv_pred = Fu_pred * np.conj(Fv_pred)

            Puu_gt = np.abs(Fu_gt) ** 2
            Pvv_gt = np.abs(Fv_gt) ** 2
            Puv_gt = Fu_gt * np.conj(Fv_gt)

            # 环平均到径向：返回 (x, bin_mean)
            def ringbin(arr):
                H, W = arr.shape
                cy, cx = H//2, W//2
                yy, xx = np.ogrid[:H, :W]
                r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                r_max = int(r.max())
                nb = min(kbins, max(r_max, 1))
                edges = np.linspace(1, r_max, nb + 1)
                out = np.zeros(nb, dtype=np.complex128)
                for i in range(nb):
                    m = (r >= edges[i]) & (r < edges[i+1])
                    if m.any():
                        out[i] = arr[m].mean()
                radii = 0.5 * (edges[:-1] + edges[1:])
                x = radii / (radii.max() + 1e-12)
                return x, out

            x_pred, Puu_pred_r = ringbin(Puu_pred)
            _,     Pvv_pred_r  = ringbin(Pvv_pred)
            _,     Puv_pred_r  = ringbin(Puv_pred)
            x_gt,   Puu_gt_r   = ringbin(Puu_gt)
            _,      Pvv_gt_r   = ringbin(Pvv_gt)
            _,      Puv_gt_r   = ringbin(Puv_gt)

            coh_pred = (np.abs(Puv_pred_r) ** 2) / (Puu_pred_r.real * Pvv_pred_r.real + eps)
            coh_gt   = (np.abs(Puv_gt_r) ** 2) / (Puu_gt_r.real * Pvv_gt_r.real + eps)

            plt.figure()
            plt.plot(x_pred, np.clip(coh_gt,   0, 1), label=f"GT coherence({names[0]},{names[1]})")
            plt.plot(x_pred, np.clip(coh_pred, 0, 1), label=f"Pred coherence({names[0]},{names[1]})")
            plt.xlabel("Normalized spatial frequency")
            plt.ylabel("Magnitude-squared coherence")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.title("Inter-channel Coherence Spectrum")
            plt.tight_layout()
            plt.savefig(img_dir / f"{prefix}_coh_{names[0]}_{names[1]}.png", dpi=200)
            plt.close()

@torch.no_grad()
def _relative_error_stats(
    pred4: torch.Tensor,
    tgt4: torch.Tensor,
    *,
    percentiles: tuple[float, ...] = (95.0, 99.0),
    eps: float = 1e-8,
) -> dict:
    """
    逐像素相对误差 re = |ŷ−y| / (|y|+eps)，聚合出 max 与若干分位点。
    返回键：{"rel_err_max": x, "rel_err_p95": x, ...}
    """
    import numpy as np
    re = (pred4 - tgt4).abs() / (tgt4.abs() + eps)
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
    将图像分割为 R×C 小块，对每块计算 NMAE 或 NMSE，再统计 max/mean。
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

def _is_higher_better(metric_name: str) -> bool:
    """
    指标方向约定：
    - 越大越好：psnr / ssim / corrcoef / vort_corr
    - 越小越好：l1 / mse / nmse / nmae / *mse / *mae / spectral_rrmse / ...
    """
    name = (metric_name or "").lower()
    higher_good = {"psnr", "ssim", "corrcoef", "vort_corr"}
    if name in higher_good:
        return True
    lower_good_prefix = ("l1", "mse", "nmse", "nmae", "grad_", "lap_", "tgrad_", "vort_m", "spectral_rrmse")
    for p in lower_good_prefix:
        if name.startswith(p):
            return False
    return False

def _gaussian_kernel1d(ks: int, sigma: float = None, device=None, dtype=None) -> torch.Tensor:
    """
    生成 1D 高斯核（长度 ks，奇数）。sigma 若为空，用经验值 ks/6。
    """
    assert ks % 2 == 1 and ks >= 1
    if sigma is None or sigma <= 0:
        sigma = ks / 6.0
    half = ks // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    w = torch.exp(-0.5 * (x / sigma) ** 2)
    w = w / (w.sum() + 1e-12)
    return w

def _depthwise_gauss_blur2d(x4: torch.Tensor, ks: int, sigma: float = None) -> torch.Tensor:
    """
    对 x4=[B,C,H,W] 做深度可分离高斯模糊（不改变尺寸）
    """
    B, C, H, W = x4.shape
    k1 = _gaussian_kernel1d(ks, sigma, device=x4.device, dtype=x4.dtype)
    kx = k1.view(1, 1, 1, ks)
    ky = k1.view(1, 1, ks, 1)
    pad = ks // 2
    w_x = kx.repeat(C, 1, 1, 1)
    w_y = ky.repeat(C, 1, 1, 1)
    out = F.conv2d(x4, w_x, padding=(0, pad), groups=C)
    out = F.conv2d(out, w_y, padding=(pad, 0), groups=C)
    return out

def _apply_ref_view(y4: torch.Tensor, k: int, *, ref_mode: str = "gauss_down_up", upsample: str = "bicubic") -> torch.Tensor:
    """
    对 GT y4 施加指定尺度的“低通+下采样+上采样”（或仅低通），返回与原始同分辨率的参考视图。
    """
    assert y4.ndim == 4, "expect [B,C,H,W]"
    B, C, H, W = y4.shape
    ref_mode = str(ref_mode or "gauss_down_up").lower()
    up_mode = str(upsample or "bicubic").lower()
    if k <= 1:
        return y4

    if ref_mode == "avgpool_up":
        y_dn = F.avg_pool2d(y4, kernel_size=k, stride=k, ceil_mode=False)
        y_up = F.interpolate(y_dn, size=(H, W), mode=up_mode, align_corners=False if "linear" in up_mode else None)
        return y_up

    if ref_mode == "gauss_down_up":
        y_blur = _depthwise_gauss_blur2d(y4, ks=k)
        y_dn = y_blur[:, :, ::k, ::k]
        y_up = F.interpolate(y_dn, size=(H, W), mode=up_mode, align_corners=False if "linear" in up_mode else None)
        return y_up

    if ref_mode == "blur_only":
        return _depthwise_gauss_blur2d(y4, ks=k)

    y_dn = F.avg_pool2d(y4, kernel_size=k, stride=k, ceil_mode=False)
    y_up = F.interpolate(y_dn, size=(H, W), mode=up_mode, align_corners=False if "linear" in up_mode else None)
    return y_up

@torch.no_grad()
def _score_vs_scales(
    yhat4: torch.Tensor,
    y4: torch.Tensor,
    *,
    kernel_sizes: List[int],
    metrics: List[str],
    metric_fns: Dict[str, Any],
    ref_mode: str = "gauss_down_up",
    upsample: str = "bicubic",
    dump_curves: bool = False,
) -> Dict[str, Any]:
    """
    针对若干核尺度 k，计算各 metric(ŷ, y^(k))，返回：
      best_k_<m> / at_best_<m> / （可选）curves_<m>
    """
    assert yhat4.shape == y4.shape, "shape mismatch"
    ks_list = [int(k) for k in kernel_sizes if int(k) >= 1 and int(k) % 2 == 1]
    if not ks_list:
        return {}
    out: Dict[str, Any] = {}

    ref_by_k = {k: _apply_ref_view(y4, k, ref_mode=ref_mode, upsample=upsample) for k in ks_list}

    for m in metrics:
        fn = metric_fns.get(m, None)
        if fn is None:
            continue
        vals = []
        for k in ks_list:
            try:
                v = float(fn(yhat4, ref_by_k[k]).detach().cpu().item())
            except Exception:
                v = float("nan")
            vals.append(v)

        import math
        higher_better = _is_higher_better(m)
        safe_vals = []
        for v in vals:
            if math.isnan(v):
                safe_vals.append((-1e30 if higher_better else 1e30))
            else:
                safe_vals.append(v)
        if higher_better:
            idx = int(max(range(len(safe_vals)), key=lambda i: safe_vals[i]))
        else:
            idx = int(min(range(len(safe_vals)), key=lambda i: safe_vals[i]))

        best_k = ks_list[idx]
        best_v = vals[idx]
        out[f"best_k_{m}"] = int(best_k)
        out[f"at_best_{m}"] = float(best_v)

        if dump_curves:
            out[f"curves_{m}"] = {int(k): float(v) for k, v in zip(ks_list, vals)}
    return out

def _try_plot_multiscale(
    *,
    out_dir: Path,
    basename: str,
    x: torch.Tensor,
    y: torch.Tensor,
    yhat: torch.Tensor,
    y_ref_by_k: Dict[int, torch.Tensor],
    curves: Dict[str, Dict[int, float]] | None,
    best_k_per_metric: Dict[str, int] | None,
    strip_orientation: str = "row",
    strip_path: Path | None = None,
    curve_path: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    """
    探测式调用绘图 + 可选 CSV 导出。
    - strip_orientation: 传给 strip 联图的 row/col
    - strip_path / curve_path：若为 None 则不保存
    - csv_path：若非 None，则把 curves 写出 CSV（样本级）
    """
    try:
        from backend.viz.images import save_multiscale_strip, save_metric_curves, save_curves_csv  # type: ignore
    except Exception:
        save_multiscale_strip = None
        save_metric_curves = None
        save_curves_csv = None

    if strip_path is not None and save_multiscale_strip is not None:
        try:
            save_multiscale_strip(
                x=x, y_hat=yhat, y=y, ref_by_k=y_ref_by_k,
                save_path=strip_path, orientation=strip_orientation,
            )
        except Exception:
            pass

    if curves and curve_path is not None and save_metric_curves is not None:
        try:
            main_metric = next(iter(curves.keys()))
        except Exception:
            main_metric = None
        try:
            save_metric_curves(
                curves=curves,
                best_k_per_metric=best_k_per_metric or {},
                save_path=curve_path,
                main_metric=main_metric,
            )
        except Exception:
            pass

    if curves and csv_path is not None and save_curves_csv is not None:
        try:
            save_curves_csv(curves=curves, save_path=csv_path)
        except Exception:
            pass

def _select_plot_curves(
    curves: Dict[str, Dict[int, float]] | None,
    plot_metrics_cfg: Optional[Sequence[str]],
) -> Dict[str, Dict[int, float]] | None:
    """
    根据 vis.plot_metrics 选择要画在曲线图里的指标；
    若未配置则全量返回；若 curves 为空返回 None。
    """
    if not curves:
        return None
    if not plot_metrics_cfg:
        return curves
    keep = {}
    allow = set(plot_metrics_cfg)
    for m, d in curves.items():
        if m in allow:
            keep[m] = d
    if not keep:
        # 防止用户给了错误的 metric 名称导致整图空白，回退到全量
        return curves
    return keep

def ensure_eval_multiscale_vis(
    model,
    test_dl,
    run_dir: Path,
    cfg_eval: Dict[str, Any],
) -> None:
    """
    安全触发多尺度可视化：若 multiscale_ref.enable 且 vis.enable，则执行；否则直接返回。
    这样你只需在评估末尾加一行调用，不必改 evaluate(...) 的主体。
    """
    ms_cfg = cfg_eval.get("multiscale_ref", {}) or {}
    if not bool(ms_cfg.get("enable", False)):
        return
    vis_cfg = (ms_cfg.get("vis") or {}) if isinstance(ms_cfg.get("vis"), dict) else {}
    if not bool(vis_cfg.get("enable", True)):
        return
    try:
        render_multiscale_panels(model, test_dl, run_dir, cfg_eval)
    except Exception as e:
        print(f"[multiscale-vis] skipped due to error: {e}")

@rank_zero_only
@torch.no_grad()
def render_multiscale_panels(
    model,
    test_dl,
    run_dir: Path,
    cfg_eval: Dict[str, Any],
) -> Path:
    """
    多尺度可视化的数据导出 + 可选即时绘图（strip/curve）。
    本版（批次C）加入：
      - vis 开关与数量控制（max_samples）、主曲线指标选择（plot_metrics）
      - 条带方向（strip_orientation）
      - 可选 CSV 导出（save_csv）
      - 统一命名规范：strip_b{b}_i{n}.png / curve_b{b}_i{n}.png / curve_b{b}_i{n}.csv
    """
    if test_dl is None:
        return run_dir / "eval_vis_ms"

    # —— 读取配置 —— #
    device = next(model.parameters()).device
    model.eval(); model.to(device)

    max_batches = int(cfg_eval.get("num_eval_batches", 3))
    # 旧可视化上限（与三/四联图一致）；vis.max_samples 会进一步收紧
    legacy_cap  = int(cfg_eval.get("num_plot_triplets", 4))

    ms_cfg      = cfg_eval.get("multiscale_ref", {}) or {}
    ms_enable   = bool(ms_cfg.get("enable", False))
    if not ms_enable:
        return ensure_dir(run_dir / "eval_vis_ms")

    ks_list     = [int(k) for k in ms_cfg.get("kernel_sizes", [3,5,7,9,11]) if int(k) >= 1 and int(k) % 2 == 1]
    if not ks_list:
        return ensure_dir(run_dir / "eval_vis_ms")

    ref_mode    = str(ms_cfg.get("ref_mode", "gauss_down_up"))
    upsample    = str(ms_cfg.get("upsample", "bicubic"))
    metric_names: List[str] = list(ms_cfg.get("metrics", [])) or list(cfg_eval.get("metrics", [])) or ["psnr"]
    dump_curves = bool(ms_cfg.get("dump_curves", True))

    # —— vis 子配置（本批次新增） —— #
    vis_cfg          = (ms_cfg.get("vis") or {}) if isinstance(ms_cfg.get("vis"), dict) else {}
    vis_enable       = bool(vis_cfg.get("enable", True))
    vis_max_samples  = int(vis_cfg.get("max_samples", 6))
    strip_orientation= str(vis_cfg.get("strip_orientation", "row"))
    plot_metrics_cfg = vis_cfg.get("plot_metrics", None)
    save_csv         = bool(vis_cfg.get("save_csv", True))

    if not vis_enable:
        return ensure_dir(run_dir / "eval_vis_ms")

    # 最终数量上限：受 legacy_cap 与 vis_max_samples 共同约束
    hard_cap = max(1, min(legacy_cap, vis_max_samples))

    out_dir = ensure_dir(run_dir / "eval_vis_ms")

    # —— 指标函数映射（与 evaluate 保持一致）—— #
    METRIC_FNS = {
        "l1": M.l1, "mse": M.mse, "psnr": M.psnr,
        "corrcoef": M.corrcoef, "ssim": M.ssim,
        "grad_mse": getattr(M, "grad_mse", None),
        "lap_mse": getattr(M, "lap_mse", None),
        "tgrad_mse": getattr(M, "tgrad_mse", None),
        "vort_mse": getattr(M, "vort_mse", None),
        "vort_mae": getattr(M, "vort_mae", None),
        "vort_corr": getattr(M, "vort_corr", None),
        "nmse": getattr(M, "nmse", None),
        "nmae": getattr(M, "nmae", None),
    }

    def _metric_dir(name: str) -> bool:
        # True=↑好；False=↓好
        return _is_higher_better(name)

    # —— 遍历采样 —— #
    plotted, batches = 0, 0
    for batch in test_dl:
        if batches >= max_batches or plotted >= hard_cap:
            break

        batch = move_batch_to_device(batch, device)
        x, y, _ = extract_xy(batch)
        y_hat = model(ensure_5d(x))

        # 统一 4D
        y4  = y.squeeze(2)    if y.ndim == 5     else y
        yh4 = y_hat.squeeze(2) if y_hat.ndim == 5 else y_hat
        x4  = x if x.ndim == 4 else x[:, :, 0]

        # 当批次样本数超过剩余额度时裁剪
        take = min(x4.shape[0], hard_cap - plotted)
        for idx in range(take):
            # —— 构造多尺度参考 —— #
            y_ref_by_k = {k: _apply_ref_view(y4[idx:idx+1], k, ref_mode=ref_mode, upsample=upsample) for k in ks_list}

            # —— 逐尺度曲线 —— #
            curves: Dict[str, Dict[int, float]] = {}
            best_k_per_metric: Dict[str, int] = {}
            best_v_per_metric: Dict[str, float] = {}

            for m in metric_names:
                fn = METRIC_FNS.get(m, None)
                if fn is None:
                    continue
                vals = []
                for k in ks_list:
                    try:
                        v = float(fn(yh4[idx:idx+1], y_ref_by_k[k]).detach().cpu().item())
                    except Exception:
                        v = float("nan")
                    vals.append(v)

                # best k
                import math
                higher = _metric_dir(m)
                safe_vals = []
                for v in vals:
                    if math.isnan(v):
                        safe_vals.append((-1e30 if higher else 1e30))
                    else:
                        safe_vals.append(v)
                ii = int(max(range(len(safe_vals)), key=lambda i: safe_vals[i])) if higher \
                     else int(min(range(len(safe_vals)), key=lambda i: safe_vals[i]))

                best_k_per_metric[m] = ks_list[ii]
                best_v_per_metric[m] = vals[ii]

                if dump_curves:
                    curves[m] = {int(k): float(v) for k, v in zip(ks_list, vals)}

            # —— 输出命名（统一规范） —— #
            bname = f"b{batches}_i{idx}"
            npz_path   = out_dir / f"msdata_{bname}.npz"
            json_path  = out_dir / f"mscurves_{bname}.json"
            strip_path = out_dir / f"strip_{bname}.png"
            curve_path = out_dir / f"curve_{bname}.png"
            csv_path   = out_dir / f"curve_{bname}.csv"

            # 保存张量数据（单样本）
            save_dict = {
                "x":    x4[idx:idx+1].detach().cpu().numpy(),
                "y":    y4[idx:idx+1].detach().cpu().numpy(),
                "yhat": yh4[idx:idx+1].detach().cpu().numpy(),
            }
            for k, ref in y_ref_by_k.items():
                save_dict[f"y_ref_k{k}"] = ref.detach().cpu().numpy()
            try:
                import numpy as _np
                _np.savez_compressed(npz_path, **save_dict)
            except Exception:
                import numpy as _np
                _np.savez(npz_path, **save_dict)

            # 保存元信息/曲线
            meta = {
                "kernel_sizes": ks_list,
                "ref_mode": ref_mode,
                "upsample": upsample,
                "metrics": metric_names,
                "best_k": best_k_per_metric,
                "at_best": best_v_per_metric,
            }
            if dump_curves and curves:
                meta["curves"] = curves
            json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            # —— 即时绘图（若可用） —— #
            _try_plot_multiscale(
                out_dir=out_dir,
                basename=bname,
                x=x4[idx:idx+1], y=y4[idx:idx+1], yhat=yh4[idx:idx+1],
                y_ref_by_k=y_ref_by_k,
                curves=_select_plot_curves(curves, plot_metrics_cfg),
                best_k_per_metric={k: v for k, v in best_k_per_metric.items()
                                   if (plot_metrics_cfg is None or k in set(plot_metrics_cfg))},
                strip_orientation=strip_orientation,
                strip_path=strip_path,
                curve_path=curve_path,
                csv_path=csv_path if save_csv else None,
            )

            plotted += 1
        batches += 1

    return out_dir

@rank_zero_only
@torch.no_grad()
def render_eval_triplets(
    model,
    test_dl,
    run_dir: Path,
    cfg_eval: Dict[str, Any],
) -> Path:
    """
    从测试集抽若干样本生成四联图与频谱图。
    扩展：若输出为多通道，文件名追加通道标识（来自 out_names，回退 c{i}）。
    """
    if test_dl is None:
        return run_dir / "eval_vis"

    # 读取通道名（输出侧）
    def _get_out_channel_names(dl) -> List[str]:
        names: List[str] = []
        try:
            ds = getattr(dl, "dataset", None)
            meta = getattr(ds, "meta", None) if ds is not None else None
            if isinstance(meta, dict):
                ch = meta.get("channel_names") or {}
                outs = ch.get("out_names")
                if isinstance(outs, (list, tuple)):
                    names = list(outs)
        except Exception:
            names = []
        return names

    out_names = _get_out_channel_names(test_dl)

    img_dir = ensure_dir(run_dir / "eval_vis")
    device = next(model.parameters()).device
    model.eval(); model.to(device)

    max_batches = int(cfg_eval.get("num_eval_batches", 3))
    max_triplets = int(cfg_eval.get("num_plot_triplets", 4))

    vis_cfg   = cfg_eval.get("eval_vis", {})
    save_err  = bool(vis_cfg.get("save_error_heatmap", False))
    logscale  = bool(vis_cfg.get("spectrum_log_scale", True))
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

        # 统一 4D
        x4  = x if x.ndim == 4 else x[:, :, 0]    # [B,C,H,W]
        y4  = y if y.ndim == 4 else y[:, :, 0]
        yh4 = y_hat if y_hat.ndim == 4 else y_hat[:, :, 0]

        B, C, H, W = y4.shape
        # 单通道与多通道两种命名
        if not out_names or len(out_names) != C:
            ch_names = [f"c{i}" for i in range(C)]
        else:
            ch_names = list(out_names)

        for idx in range(take):
            if C == 1:
                # 与旧版一致的单文件输出
                save_quadruple_grid(
                    x4[idx], yh4[idx], y4[idx],
                    img_dir / f"triplet_b{batches}_i{idx}.png",
                )
                prefix  = f"triplet_b{batches}_i{idx}"
                _save_error_and_spectrum(
                    img_dir, prefix, yh4[idx:idx+1], y4[idx:idx+1],
                    save_error_heatmap=save_err,
                    spectrum_log_scale=logscale,
                    kbins=kbins,
                    cmap=cmap,
                    channel_names=out_names,
                )
            else:
                # 多通道：为每个通道单独出图，便于对照
                for ci, tag in enumerate(ch_names):
                    save_quadruple_grid(
                        x4[idx], yh4[idx, ci:ci+1], y4[idx, ci:ci+1],
                        img_dir / f"triplet_b{batches}_i{idx}_{tag}.png",
                    )
                    prefix  = f"triplet_b{batches}_i{idx}_{tag}"
                    _save_error_and_spectrum(
                        img_dir, prefix, yh4[idx:idx+1, ci:ci+1], y4[idx:idx+1, ci:ci+1],
                        save_error_heatmap=save_err,
                        spectrum_log_scale=logscale,
                        kbins=kbins,
                        cmap=cmap,
                        channel_names=out_names,
                    )
            plotted += 1
        batches += 1
    return img_dir