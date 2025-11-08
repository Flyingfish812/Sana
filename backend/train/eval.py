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
    兼容模式：当 cfg_eval 未声明 metrics/scales/spectral 时，仅写 PSNR/p/sigma。
    """
    if test_dl is None:
        return
    log_path = run_dir / "eval_log.jsonl"
    device = next(model.parameters()).device
    limit = int(cfg_eval.get("num_eval_batches", 3))

    # factors（p/σ）沿用旧逻辑
    factors = cfg_eval.get("factors", {})
    p = factors.get("sample_density", None) if isinstance(factors, dict) else None
    sigma = factors.get("noise_sigma", None) if isinstance(factors, dict) else None

    # 新配置入口
    metric_names: List[str] = list(cfg_eval.get("metrics", []))
    use_scales = bool(cfg_eval.get("scales", {}).get("enable", False))
    n_levels = int(cfg_eval.get("scales", {}).get("levels", 3))
    use_spectral = bool(cfg_eval.get("spectral", {}).get("enable", False))
    kbins = int(cfg_eval.get("spectral", {}).get("kbins", 32))
    fft_pad = bool(cfg_eval.get("spectral", {}).get("fft_pad", True))
    layout_key = cfg_eval.get("layout_tag_key", "layout_tag")
    write_per_item = bool(cfg_eval.get("write_per_item", False))

    # 兼容：若未声明 metrics，把 psnr 作为唯一指标
    if not metric_names:
        metric_names = ["psnr"]

    # 注册指标函数
    METRIC_FNS = {
        "l1": M.l1, "mse": M.mse, "psnr": M.psnr,
        "corrcoef": M.corrcoef, "ssim": M.ssim,
        "grad_mse": getattr(M, "grad_mse", None),
        "lap_mse": getattr(M, "lap_mse", None),
        "tgrad_mse": getattr(M, "tgrad_mse", None),
        "vort_mse": getattr(M, "vort_mse", None),
        "vort_mae": getattr(M, "vort_mae", None),
        "vort_corr": getattr(M, "vort_corr", None),
    }


    with log_path.open("a", encoding="utf-8") as fp:
        for bidx, batch in enumerate(test_dl):
            if bidx >= limit:
                break
            batch = move_batch_to_device(batch, device)
            x, y, _ = extract_xy(batch)
            y_hat = model(ensure_5d(x))

            # 统一到4D
            y4 = _as4d(y)
            yh4 = _as4d(y_hat)

            # 基础指标
            record: Dict[str, Any] = {}
            for name in metric_names:
                fn = METRIC_FNS.get(name, None)
                if fn is None:
                    continue
                try:
                    if name == "tgrad_mse":
                        # 时间一致性指标需要 5D
                        val = float(fn(y_hat, y).detach().cpu().item())
                    else:
                        val = float(fn(yh4, y4).detach().cpu().item())
                    record[name] = val
                except Exception:
                    # 指标失败时略过，避免断评估
                    pass

            # 多尺度指标
            if use_scales and n_levels > 1:
                pyr_pred = _build_pyramid(yh4, n_levels)
                pyr_tgt = _build_pyramid(y4, n_levels)
                for li, (pp, tt) in enumerate(zip(pyr_pred, pyr_tgt)):
                    for name in metric_names:
                        fn = METRIC_FNS.get(name, None)
                        if fn is None:
                            continue
                        try:
                            record[f"{name}@L{li}"] = float(fn(pp, tt).detach().cpu().item())
                        except Exception:
                            pass

            # 频域指标
            if use_spectral and S is not None:
                try:
                    # 返回整体谱相对误差与逐bin误差
                    spec = S.spectral_rrmse(yh4, y4, kbins=kbins, fft_pad=fft_pad)
                    # 记录总体
                    record["spectral_rrmse"] = float(spec["overall"])
                    # 记录关键频段（可选：前/中/后三段）
                    thirds = S.slice_rrmse_thirds(spec["per_bin"])
                    record["spectral_rrmse_low"]  = float(thirds["low"])
                    record["spectral_rrmse_mid"]  = float(thirds["mid"])
                    record["spectral_rrmse_high"] = float(thirds["high"])
                except Exception:
                    pass  # 频域失败不阻断

            # 逐样本写出（可选，通常关）
            per_item = None
            if write_per_item:
                per_item = {}
                for name in metric_names:
                    fn = METRIC_FNS.get(name, None)
                    if fn is None:
                        continue
                    try:
                        # 逐样本：把 reduction='none' 的计算封装一下
                        if name in ("l1", "mse"):
                            # 用元素级再聚合到每样本
                            diff = (yh4 - y4).abs() if name == "l1" else (yh4 - y4) ** 2
                            dims = list(range(1, diff.ndim))
                            vals = diff.mean(dim=dims)
                        elif name == "psnr":
                            # 参考现有 psnr 的逐样本写法
                            mse_val = F.mse_loss(yh4, y4, reduction="none")
                            dims = list(range(1, mse_val.ndim))
                            mse_b = mse_val.mean(dim=dims)
                            vals = 10.0 * torch.log10(1.0 / (mse_b + 1e-8))
                        else:
                            # 其他简单退化为标量复制（占位）
                            vals = None
                        if vals is not None:
                            per_item[name] = [float(v) for v in vals.detach().cpu().flatten()]
                        # 复杂指标（ssim/corr/grad/lap）逐样本实现可在后续批次补齐
                    except Exception:
                        pass

            layout_tag = _resolve_layout_tag_from_batch_or_dataset(batch, test_dl, layout_key)

            rec = {"p": p, "sigma": sigma, "layout_tag": layout_tag, **record}
            if per_item is not None:
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

# 误差与频谱的可视化（保存两张图：error heatmap / radial power spectrum） ===
def _save_error_and_spectrum(img_dir: Path, prefix: str, pred_4d: torch.Tensor, gt_4d: torch.Tensor):
    """
    pred_4d, gt_4d: [C,H,W] on CPU
    生成:
      - {prefix}_error.png           ：|pred - gt| 热力图（按通道求均后显示）
      - {prefix}_rps.png             ：径向功率谱曲线（pred/gt/error 三条）
    """
    import numpy as np
    import matplotlib.pyplot as plt

    with torch.no_grad():
        # 对齐到 [C,H,W]，转 numpy
        pred = pred_4d.detach().cpu().float().numpy()
        gt   = gt_4d.detach().cpu().float().numpy()
        err  = np.abs(pred - gt)

        # 误差热力：各通道取均值，避免多通道堆图
        err_map = err.mean(axis=0)
        plt.figure()
        plt.imshow(err_map, interpolation="nearest")
        plt.title("Error heatmap (|pred - gt|)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(img_dir / f"{prefix}_error.png", dpi=200)
        plt.close()

        # 频谱：幅度谱 -> 径向平均（简单桶平均）
        def _radial_power(img2d):
            # 幅度谱
            F = np.fft.fftshift(np.fft.fft2(img2d, norm="ortho"))
            mag = np.abs(F)
            H, W = mag.shape
            yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
            rr = np.sqrt(xx*xx + yy*yy)
            rr /= rr.max() + 1e-12
            kbins = 64
            bins = np.clip((rr * kbins).astype(np.int64), 0, kbins-1)
            ps = np.zeros(kbins, dtype=np.float64)
            cnt = np.zeros(kbins, dtype=np.float64)
            for k in range(kbins):
                m = (bins == k)
                cnt[k] = m.sum()
                if cnt[k] > 0:
                    ps[k] = (mag[m]**2).mean()
            x = np.arange(kbins) / (kbins - 1 + 1e-12)
            return x, ps

        # 取各通道功率谱的均值
        x, p_pred = _radial_power(pred.mean(axis=0))
        _, p_gt   = _radial_power(gt.mean(axis=0))
        p_err     = np.abs(p_pred - p_gt)

        plt.figure()
        plt.plot(x, p_gt,  label="GT")
        plt.plot(x, p_pred, label="Pred")
        plt.plot(x, p_err, label="|Pred-GT|")
        plt.xlabel("Normalized spatial frequency")
        plt.ylabel("Radial power")
        plt.legend()
        plt.title("Radial Power Spectrum")
        plt.tight_layout()
        plt.savefig(img_dir / f"{prefix}_rps.png", dpi=200)
        plt.close()

@torch.no_grad()
def render_eval_triplets(
    model,
    test_dl,
    run_dir: Path,
    cfg_eval: Dict[str, Any],
) -> Path:
    """Generate qualitative plots from the test set (triplets + error + spectrum)."""
    if test_dl is None:
        return run_dir / "eval_vis"

    img_dir = ensure_dir(run_dir / "eval_vis")
    device = next(model.parameters()).device

    model.eval()
    model.to(device)

    max_batches = int(cfg_eval.get("num_eval_batches", 3))
    max_triplets = int(cfg_eval.get("num_plot_triplets", 4))

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
            # 仍输出原有四联图（你之前的函数名叫 quadruple，其内容安排保持不变）
            save_quadruple_grid(
                x[idx] if x.ndim == 4 else x[idx, :, 0],
                y_hat[idx] if y_hat.ndim == 4 else y_hat[idx, :, 0],
                y[idx] if y.ndim == 4 else y[idx, :, 0],
                img_dir / f"triplet_b{batches}_i{idx}.png",
            )

            # 新增：误差热力图与频谱图
            pred_4d = y_hat[idx] if y_hat.ndim == 4 else y_hat[idx, :, 0]
            gt_4d   = y[idx] if y.ndim == 4 else y[idx, :, 0]
            prefix  = f"triplet_b{batches}_i{idx}"

            _save_error_and_spectrum(img_dir, prefix, pred_4d, gt_4d)

            plotted += 1

        batches += 1

    return img_dir
