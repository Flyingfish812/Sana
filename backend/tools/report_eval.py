# tools/report_eval.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import matplotlib.pyplot as plt
import os

def _metric_specs() -> dict:
    """
    指标词典：每个条目包含
      - fullname: 全称
      - formula: 文字版计算要点（非排版公式，便于md查看）
      - direction: 'higher' 或 'lower'（代表“值越大越好/越小越好”）
      - meaning: 这个指标在直觉上反映了什么
      - typical: 常见区间或经验解读（可为空）
    可按需增删条目；未收录的指标将用通用模板。
    """
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
            "meaning": "归一化绝对误差，衡量平均偏差占真实值平均幅度的比例，消除尺度影响。",
            "typical": "0.0 理想，<0.05 很好，>0.1 说明整体偏差明显。",
        },
        "nmse": {
            "fullname": "Normalized Mean Squared Error",
            "formula": "NMSE = mean((pred - gt)^2) / mean(gt^2 + ε)",
            "direction": "lower",
            "meaning": "归一化平方误差，反映整体能量比例偏差，越小表示与真值能量分布越接近。",
            "typical": "0.0 理想，<0.01 很好，>0.05 说明能量重建偏差较大。",
        },

    }

def _is_error_like(direction: str) -> bool:
    return direction.lower().strip() == "lower"

def _summarize_metric_grid(P, S, M, direction: str) -> dict:
    """
    返回该指标的摘要：最佳值/位置、均值/标准差、有效覆盖率与简单“相对评分”(0–100)。
    评分做法：先对可比元素做分位归一（大=好或小=好），再映射到0–100。
    """
    import numpy as np
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(M)
        coverage = float(valid.mean())  # 有效占比
        if valid.sum() == 0:
            return {"best_val": None, "best_p": None, "best_s": None, "mean": None, "std": None,
                    "coverage": 0.0, "score": 0.0}

        vals = M[valid]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))

        # 最佳值与位置
        if _is_error_like(direction):
            best_val = float(np.nanmin(vals))
            where = np.where(M == best_val)
        else:
            best_val = float(np.nanmax(vals))
            where = np.where(M == best_val)

        # 取第一个最优点
        i = int(where[0][0]); j = int(where[1][0])
        best_p = float(P[i]); best_s = float(S[j])

        # 相对评分：按方向将值归一成 [0,1]，再 *100
        q_lo, q_hi = np.nanpercentile(vals, [5, 95])
        if _is_error_like(direction):
            # 小=好：把小的映射到高分
            denom = (q_hi - q_lo) if (q_hi > q_lo) else (np.nanmax(vals) - np.nanmin(vals) + 1e-9)
            norm = np.clip((q_hi - vals) / (denom + 1e-9), 0, 1)
        else:
            # 大=好
            denom = (q_hi - q_lo) if (q_hi > q_lo) else (np.nanmax(vals) - np.nanmin(vals) + 1e-9)
            norm = np.clip((vals - q_lo) / (denom + 1e-9), 0, 1)
        score = float(100.0 * np.nanmean(norm))

        return {
            "best_val": best_val, "best_p": best_p, "best_s": best_s,
            "mean": mean, "std": std, "coverage": coverage, "score": score
        }

def _read_jsonl(path: Path) -> List[dict]:
    """读取 JSON Lines（每行一个 JSON 对象）。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def _load_sweep_records(run_root: Path) -> List[dict]:
    """
    在 runs/<exp>/<version>/ 下寻找 sweep_summary.jsonl 或 sweep_summary.json，
    返回记录列表，每条至少含 {p, sigma, run_dir, eval_log} 等键。
    """
    cand_jsonl = run_root / "sweep_summary.jsonl"
    cand_json  = run_root / "sweep_summary.json"
    if cand_jsonl.exists():
        return _read_jsonl(cand_jsonl)
    if cand_json.exists():
        obj = json.loads(cand_json.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        # 允许 {"records":[...]}
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            return obj["records"]
    # 兜底：直接在子目录搜 p??_s??? 结构并尝试拼接 eval_log 路径
    recs = []
    for sub in sorted(run_root.glob("p*_s*/")):
        ev = sub / "eval_log.jsonl"
        if ev.exists():
            # 无 p/s 信息时，尝试从后缀解析（尽量不强依赖）
            recs.append({"p": None, "sigma": None, "run_dir": str(sub), "eval_log": str(ev)})
    return recs

def _read_eval_metric_mean(eval_log: Path, metric: str) -> Optional[float]:
    """从 eval_log.jsonl 中读取某个指标的均值。"""
    if not eval_log.exists():
        return None
    vals = []
    with eval_log.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except Exception:
                continue
            if metric in obj:
                try:
                    vals.append(float(obj[metric]))
                except Exception:
                    pass
    if not vals:
        return None
    return float(sum(vals) / len(vals))

def _read_eval_multi_means(eval_log: Path, metric_names: List[str]) -> Dict[str, Optional[float]]:
    """
    读取一组指标均值；同时兼容频域摘要键（若存在）。
    与训练侧 sweep 的读取逻辑保持一致。 
    """
    out: Dict[str, Optional[float]] = {}
    for m in metric_names:
        out[m] = _read_eval_metric_mean(eval_log, m)
    # 频域摘要键（若评估端写入）
    for k in ["spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high"]:
        v = _read_eval_metric_mean(eval_log, k)
        if v is not None:
            out[k] = v
    return out

def _read_eval_available_keys(eval_log: Path) -> List[str]:
    """
    扫描 eval_log.jsonl 中出现过的所有顶层键，返回去重后的键名列表。
    用于动态发现 'best_k_*' / 'at_best_*' 等新指标键。
    """
    if not eval_log.exists():
        return []
    keys = set()
    with eval_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                for k in obj.keys():
                    keys.add(k)
    return sorted(keys)

def _build_metric_grids_from_records(
    records: List[dict],
    metric_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    输入：sweep_summary.jsonl 解析出的记录列表，以及“显式要求统计”的 metric 名单
    输出：
      P: 唯一升序的 p 列表 (np.ndarray)
      S: 唯一升序的 sigma 列表 (np.ndarray)
      grids: {metric -> 形状 [len(P), len(S)] 的矩阵}
    扩展：自动发现并并入 'best_k_*' 与 'at_best_*' 两类新键（若评估端写入）。
    """
    import re
    combos: List[Tuple[float, float]] = []
    values_by_metric: Dict[str, Dict[Tuple[float, float], float]] = {}

    # 显式 + 频域默认键（兼容原行为）
    base_interest = set(metric_names) | {
        "spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high"
    }

    # 先遍历一遍，收集所有 (p,s) 以及该 run 可用的“多尺度键”
    discovered_ms_keys: set[str] = set()
    cache_keys_per_log: Dict[str, List[str]] = {}
    for row in records:
        p = row.get("p", None)
        s = row.get("sigma", None)
        ev = Path(row.get("eval_log", ""))
        if (p is None) or (s is None):
            base = Path(row.get("run_dir", "")).name
            try:
                m = re.match(r"p(\d{2})_s(\d{3})", base)
                if m:
                    p = int(m.group(1)) / 100.0
                    s = int(m.group(2)) / 1000.0
            except Exception:
                pass
        if (p is None) or (s is None) or (not ev.exists()):
            continue
        combos.append((float(p), float(s)))

        # 动态发现：best_k_* / at_best_*
        ks = _read_eval_available_keys(ev)
        cache_keys_per_log[str(ev)] = ks
        for k in ks:
            if k.startswith("best_k_") or k.startswith("at_best_"):
                discovered_ms_keys.add(k)

    if not combos:
        raise RuntimeError("没有可用的 (p, sigma) 组合或 eval_log。")

    # 统一的关注键集合
    interest = sorted(base_interest | discovered_ms_keys)

    # 逐 run 读取均值
    for row in records:
        p = row.get("p", None)
        s = row.get("sigma", None)
        ev = Path(row.get("eval_log", ""))
        if (p is None) or (s is None):
            base = Path(row.get("run_dir", "")).name
            try:
                m = re.match(r"p(\d{2})_s(\d{3})", base)
                if m:
                    p = int(m.group(1)) / 100.0
                    s = int(m.group(2)) / 1000.0
            except Exception:
                pass
        if (p is None) or (s is None) or (not ev.exists()):
            continue

        for k in interest:
            v = _read_eval_metric_mean(ev, k)
            if v is None:
                continue
            values_by_metric.setdefault(k, {})[(float(p), float(s))] = float(v)

    P = np.array(sorted({p for p, _ in combos}), dtype=float)
    S = np.array(sorted({s for _, s in combos}), dtype=float)

    # 按关注键构造网格
    grids: Dict[str, np.ndarray] = {}
    for metric in interest:
        if metric not in values_by_metric:
            continue
        M = np.full((len(P), len(S)), np.nan, dtype=float)
        mp = values_by_metric[metric]
        for i, pv in enumerate(P):
            for j, sv in enumerate(S):
                val = mp.get((float(pv), float(sv)), None)
                if val is not None:
                    M[i, j] = float(val)
        grids[metric] = M

    return P, S, grids

def _write_markdown_report_ex(report_md: Path,
                              figures: Dict[str, Path],
                              P: np.ndarray,
                              S: np.ndarray,
                              grids: Dict[str, np.ndarray],
                              *,
                              title: str = "Evaluation Report",
                              extra_notes: str = "") -> None:
    """
    写出包含“指标说明 + 摘要表 + 热图插图”的 report.md。
    扩展：
      - 对 'best_k_*' 与 'at_best_*' 生成临时词典项：
          * best_k_* : 等效核尺度（方向视作 ↓ 好，代表“可复原到的细尺度越细越好”）
          * at_best_*: 继承基础指标（psnr/ssim/...）的方向与全称，标题标注“at best scale”
    """
    specs = _metric_specs().copy()

    # —— 动态扩展 specs ——
    def _extend_specs_for_multiscale(metric_keys: Iterable[str]) -> None:
        for m in metric_keys:
            if m.startswith("best_k_"):
                base = m[len("best_k_"):]
                base_sp = specs.get(base, {"fullname": base})
                specs[m] = {
                    "fullname": f"Equivalent Kernel Size (best_k of {base})",
                    "formula": "在多尺度参考视图 y^(k) 中，令指定指标在 k 上最优（极小/极大），取其核大小 k。",
                    "direction": "lower",  # 倾向把更小核视为“更细=更好”
                    "meaning": "等效复原尺度；数值越小表示重建能接近更细尺度的参考。",
                    "typical": "",
                }
            elif m.startswith("at_best_"):
                base = m[len("at_best_"):]
                base_sp = specs.get(base, {"fullname": base, "direction": "higher"})
                specs[m] = {
                    "fullname": f"{base_sp.get('fullname', base)} (at best scale)",
                    "formula": f"在 {base} 指标的最优核尺度 k* 下，{base} 的分数。",
                    "direction": base_sp.get("direction", "higher"),
                    "meaning": f"在多尺度参考下，当尺度选到最优 k* 后的 {base} 分数。",
                    "typical": base_sp.get("typical", ""),
                }

    metric_order = sorted(figures.keys())
    _extend_specs_for_multiscale(metric_order)

    lines: List[str] = []
    lines.append(f"# {title}\n")
    if extra_notes:
        lines.append(f"> {extra_notes}\n")

    # 1) 速览表：指标方向、最佳点与相对评分
    lines.append("## 概览（一眼看懂）\n")
    lines.append("| 指标 | 全称 | 方向 | 最佳值 | 位置(p, σ) | 覆盖率 | 评分 |\n")
    lines.append("|---|---|:--:|---:|:--:|:--:|--:|\n")

    for m in metric_order:
        M = grids.get(m, None)
        if M is None:
            continue
        sp = specs.get(m, {"fullname": m, "direction": "higher"})
        summ = _summarize_metric_grid(P, S, M, sp["direction"])
        arrow = "↑" if sp.get("direction", "higher").lower() == "higher" else "↓"
        cov = f"{summ['coverage']*100:.1f}%"
        score = f"{summ['score']:.1f}"
        best_val = "—" if summ["best_val"] is None else f"{summ['best_val']:.6g}"
        pos = "—" if summ["best_p"] is None else f"({summ['best_p']:.3f}, {summ['best_s']:.3f})"
        lines.append(f"| `{m}` | {sp['fullname']} | {arrow} | {best_val} | {pos} | {cov} | {score} |\n")

    lines.append("\n> 说明：评分基于该指标在本次 sweep 的分布做分位归一（5%–95%），按“方向”将数值映射到0–100的相对量。")
    lines.append(" 对于 `best_k_*`，我们将“更小的核＝更细的可复原尺度”视作更优，因此在概览中按“↓好”处理。\n")

    # 2) 指标词典与计算要点
    lines.append("\n## 指标说明（简称 / 全称 / 计算 / 含义）\n")
    for m in metric_order:
        sp = specs.get(m, None)
        if sp is None:
            lines.append(f"### `{m}`\n- **全称**：{m}\n- **方向**：默认 ↑（值越大越好）\n- **含义**：未录入词典，请在实现中补充。\n")
            continue
        arrow = "↑（越大越好）" if sp.get("direction", "higher").lower() == "higher" else "↓（越小越好）"
        typical = f"\n- **经验区间**：{sp['typical']}" if sp.get("typical") else ""
        formula = sp.get("formula", "")
        meaning = sp.get("meaning", "")
        lines.append(
            f"### `{m}` — {sp['fullname']}\n"
            f"- **方向**：{arrow}\n"
            f"- **计算要点**：{formula}\n"
            f"- **含义**：{meaning}{typical}\n"
        )

    # 3) 图像清单（针对多尺度键给出额外解读）
    lines.append("\n## 可视化热力图\n")
    lines.append("横轴为观测噪声 σ，纵轴为采样密度 p。对于“↑好”的指标颜色越暖越好；对于“↓好”的指标颜色越冷越好。\n")
    for m in metric_order:
        fig = figures[m]
        sp = specs.get(m, {"direction": "higher"})
        if m.startswith("best_k_"):
            hint = "色条即核大小（k）。数字越小表示可复原的细尺度越细（更好）。"
        elif m.startswith("at_best_"):
            hint = "该图为在最优尺度 k* 下的分数，方向同基础指标。"
        else:
            hint = "暖色=好（↑）或冷色=好（↓），见概览表方向。"
        lines.append(f"### `{m}`\n![]({fig.name})\n\n*解读提示*：{hint}\n")

    if extra_notes:
        lines.append("\n---\n")
        lines.append(f"_Notes_: {extra_notes}\n")

    report_md.write_text("\n".join(lines), encoding="utf-8")

def _plot_heatmap(P: np.ndarray,
                  S: np.ndarray,
                  M: np.ndarray,
                  metric: str,
                  out_png: Path,
                  *,
                  cmap: str = "RdBu_r",
                  dpi: int = 220) -> None:
    """
    根据 metric 名推断绘图模式：
      - 以 'best_k_' 开头 → 离散整值热图（核大小）
      - 其它 → 连续热图（方向由 _metric_specs 或继承逻辑决定）
    """
    if metric.startswith("best_k_"):
        _plot_heatmap_discrete_int(P, S, M, metric, out_png, dpi=dpi, cmap="viridis")
        return

    # 对 at_best_*：标题/方向参考其基础指标（去掉前缀）
    base_metric = metric
    if metric.startswith("at_best_"):
        base_metric = metric[len("at_best_"):]
    specs = _metric_specs()
    sp = specs.get(base_metric, {"direction": "higher"})
    direction = sp.get("direction", "higher")
    title = f"{metric}"
    if "fullname" in sp:
        title = f"{metric} — {sp['fullname']} (at best scale)" if metric.startswith("at_best_") else f"{metric} — {sp['fullname']}"

    _plot_heatmap_annotated(P, S, M, title, out_png,
                            cmap=cmap, dpi=dpi, direction=direction,
                            annotate=True, topk=5)

def _plot_heatmap_discrete_int(
    P: np.ndarray,
    S: np.ndarray,
    M: np.ndarray,
    metric: str,
    out_png: Path,
    *,
    dpi: int = 220,
    cmap: str = "viridis",
) -> None:
    """
    用于绘制 'best_k_*' 的离散整值热图：
      - 色条刻度=出现的核大小（整型）
      - 默认把“小=细=好”的直觉体现在摘要时的方向（在 _write_markdown_report_ex 里处理）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm

    valid = np.isfinite(M)
    if not np.any(valid):
        raise ValueError("无有效数据可绘图。")

    uniq_vals = sorted({int(round(v)) for v in M[valid].ravel()})
    if not uniq_vals:
        uniq_vals = [0]
    # 构造边界，使得每个离散值有自己的颜色块
    boundaries = np.array(uniq_vals, dtype=float)
    # 若只有一个值，扩一圈边界以便显示色条
    if len(boundaries) == 1:
        boundaries = np.array([boundaries[0] - 0.5, boundaries[0] + 0.5])
        norm = BoundaryNorm(boundaries, ncolors=256)
    else:
        # 把整数点扩成半间隔边界
        edges = [uniq_vals[0] - 0.5] + [v + 0.5 for v in uniq_vals]
        boundaries = np.array(edges, dtype=float)
        norm = BoundaryNorm(boundaries, ncolors=256)

    fig = plt.figure(figsize=(8.5, 6.8), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, origin="lower", aspect="auto",
                   extent=[S.min(), S.max(), P.min(), P.max()],
                   cmap=cmap, norm=norm)

    # 在格子中心标出整数
    for i, p in enumerate(P):
        for j, s in enumerate(S):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(s, p, f"{int(round(v))}", ha="center", va="center", fontsize=7, color="white", alpha=0.9)

    ax.set_xlabel("sigma")
    ax.set_ylabel("p (sample density)")
    ax.set_title(f"{metric} — Equivalent Kernel Size (discrete)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("kernel size (k)", rotation=90)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

def _plot_heatmap_annotated(P: np.ndarray,
                            S: np.ndarray,
                            M: np.ndarray,
                            title: str,
                            out_png: Path,
                            *,
                            cmap: str = "RdBu_r",
                            dpi: int = 220,
                            direction: str = "higher",
                            annotate: bool = True,
                            topk: int = 5) -> None:
    """
    带注释与最优路径的热力图：
      - P: 纵轴（采样密度 p），升序
      - S: 横轴（噪声 sigma），升序
      - M: [len(P), len(S)] 指标矩阵
      - direction: "higher" 或 "lower"
      - annotate: 是否在每个格子写上数值
      - topk: 标注前 K 个最优点（空心圆）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib import ticker

    # 有效/分位剪裁，增强对比
    valid = np.isfinite(M)
    if not np.any(valid):
        raise ValueError("无有效数据可绘图。")
    vals = M[valid]
    q5, q95 = np.nanpercentile(vals, [5, 95])
    vmin, vmax = (q5, q95) if q5 < q95 else (np.nanmin(vals), np.nanmax(vals))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 准备“方向”
    lower_is_better = direction.lower().strip() == "lower"
    def _score_for_rank(A):
        return -A if lower_is_better else A

    # 每行（固定 p）找最优 σ，组成“最优路径”
    best_j_per_i = []
    for i in range(len(P)):
        row = M[i, :]
        if np.all(~np.isfinite(row)):
            best_j_per_i.append(None)
            continue
        if lower_is_better:
            j = np.nanargmin(row)
        else:
            j = np.nanargmax(row)
        best_j_per_i.append(int(j))

    # 全局 Top-K 最优点
    flat = M.copy()
    flat[~valid] = np.nan
    order = np.argsort(_score_for_rank(flat).ravel())[::-1]  # 大到小
    # 过滤 NaN
    order = [idx for idx in order if np.isfinite(flat.ravel()[idx])]
    top_idx = order[:max(0, int(topk))]
    top_pairs = [(int(k // M.shape[1]), int(k % M.shape[1])) for k in top_idx]

    # 画图：主热力 + 旁侧/底部最优曲线两个小轴
    fig = plt.figure(figsize=(8.5, 6.8), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[8, 2.2], height_ratios=[8, 2.2], wspace=0.15, hspace=0.15)
    ax = fig.add_subplot(gs[0, 0])       # 主热力
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax)  # 右侧行最优
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax)  # 底部列最优

    im = ax.imshow(M, origin="lower", aspect="auto",
                   extent=[S.min(), S.max(), P.min(), P.max()], cmap=cmap, norm=norm)

    # NaN 掩膜（灰色斜线）
    if np.any(~valid):
        ax.imshow(~valid, origin="lower", aspect="auto",
                  extent=[S.min(), S.max(), P.min(), P.max()],
                  cmap="gray", alpha=0.25)

    # 标注数值（智能小数位）
    if annotate:
        # 根据范围自动决定小数位
        rng = vmax - vmin
        if rng >= 10: fmt = lambda x: f"{x:.1f}"
        elif rng >= 1: fmt = lambda x: f"{x:.2f}"
        elif rng >= 0.1: fmt = lambda x: f"{x:.3f}"
        else: fmt = lambda x: f"{x:.4f}"

        # 把坐标网格映射到像素内中心
        for i, p in enumerate(P):
            for j, s in enumerate(S):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(s, p, fmt(v),
                        ha="center", va="center",
                        fontsize=7,
                        color="black" if abs(norm(v) - 0.5) > 0.25 else "white")

    # 最优路径（每行的最佳 σ）
    path_x, path_y = [], []
    for i, j in enumerate(best_j_per_i):
        if j is None: 
            continue
        path_x.append(S[j]); path_y.append(P[i])
    if len(path_x) >= 2:
        ax.plot(path_x, path_y, lw=2, ls="-", marker="o", ms=3, mec="white", mfc="white", alpha=0.9)

    # 标出 Top-K 最优点
    for (i, j) in top_pairs:
        ax.plot(S[j], P[i], marker="o", ms=10, mfc="none", mec="yellow", mew=1.5, alpha=0.9)

    ax.set_xlabel("sigma")
    ax.set_ylabel("p (sample density)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("metric value", rotation=90)

    # 右侧：行最优（随 p 变化的最佳值）
    row_best = np.array([np.nanmin(M[i, :]) if lower_is_better else np.nanmax(M[i, :]) for i in range(len(P))])
    ax_r.plot(row_best, P, lw=1.8)
    ax_r.grid(alpha=0.3)
    ax_r.set_xlabel("best over sigma")
    ax_r.tick_params(labelleft=False)

    # 底部：列最优（随 sigma 变化的最佳值）
    col_best = np.array([np.nanmin(M[:, j]) if lower_is_better else np.nanmax(M[:, j]) for j in range(len(S))])
    ax_b.plot(S, col_best, lw=1.8)
    ax_b.grid(alpha=0.3)
    ax_b.set_ylabel("best over p")
    ax_b.yaxis.set_label_position("right")
    ax_b.yaxis.tick_right()

    # 适度格式化刻度
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

def run_report(
    summary_dir: str | Path,
    out_dir: Optional[str | Path] = None,
    *,
    metric_whitelist: Optional[List[str]] = None,
    overwrite: bool = False,
    dpi: int = 220,
    cmap: str = "RdBu_r",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    新行为（无 CSV）：
      - 读取 runs/<exp>/<version>/ 下的 sweep_summary.jsonl/json（或回退到 pXX_sYYY 子目录推断）
      - 逐 run 读取 eval_log.jsonl，计算多指标均值
      - 直接把这些均值构造成每个指标的 p×σ 矩阵，画热力图
      - 输出 report.md（嵌入各指标热图）
    可从 CLI 调用，也可在 ipynb 里一行调用：
        from tools.report_eval import run_report
        run_report("runs/exp_xxx/version_yyy")
    """
    sd = Path(summary_dir).expanduser().resolve()
    records = _load_sweep_records(sd)  # 读取 sweep 索引（或回退扫描）:contentReference[oaicite:5]{index=5}
    if not records:
        raise FileNotFoundError(f"未找到 sweep_summary.jsonl/json，也无法在 {sd} 推断有效子实验。")

    # 选取指标集：用户未指定则给一套通用“多尺度实验”常用指标
    default_metrics = [
        "psnr", "ssim", "corrcoef",
        "l1", "mse", "grad_mse", "lap_mse",
        "tgrad_mse", "vort_mse", "vort_mae", "vort_corr",
        "spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high",
    ]
    metric_names = metric_whitelist or default_metrics

    # 构造 p×σ 网格
    P, S, grids = _build_metric_grids_from_records(records, metric_names)

    # 输出目录
    if out_dir is None:
        out_dir = sd / "report"
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    # 画热力图
    figures: Dict[str, Path] = {}
    for metric, M in sorted(grids.items()):
        fig_path = od / f"{metric}.png"
        if (not fig_path.exists()) or overwrite:
            _plot_heatmap(P, S, M, metric, fig_path, cmap=cmap, dpi=dpi)
        figures[metric] = fig_path
        if verbose:
            print(f"[run_report] {metric} → {fig_path.name}")

    # 写 markdown 报告
    report_md = od / "report.md"
    if (not report_md.exists()) or overwrite:
        _write_markdown_report_ex(
            report_md,
            figures=figures,
            P=P, S=S, grids=grids,
            title="Evaluation Report",
            extra_notes="此报告由 sweep_summary.jsonl 与各组 eval_log.jsonl 自动生成；评分为相对量纲，仅用于横向感知。"
        )
        if verbose:
            print(f"[run_report] 写入 {report_md.name}")

    return {
        "run_root": sd,
        "out_dir": od,
        "figures": figures,
        "report_md": report_md,
        "metrics": sorted(grids.keys()),
        "P": P.tolist(),
        "S": S.tolist(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", type=str, required=True,
                        help="指向 runs/<exp>/<version>/，目录下应含 sweep_summary.jsonl 或 pXX_sYYY 子目录")
    parser.add_argument("--out_dir", type=str, default=None, help="报告输出目录；默认 <version>/report")
    parser.add_argument("--metrics", type=str, nargs="*", default=None, help="metric 白名单，例如 psnr ssim spectral_rrmse")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--cmap", type=str, default="RdBu_r")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_report(
        summary_dir=args.summary_dir,
        out_dir=args.out_dir,
        metric_whitelist=args.metrics,
        overwrite=args.overwrite,
        dpi=args.dpi,
        cmap=args.cmap,
        verbose=not args.quiet,
    )
