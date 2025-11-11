# backend/eval/report.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable, Any
from pathlib import Path
import numpy as np

from .io import load_sweep_records, read_eval_metric_mean, read_eval_available_keys
from .metrics import metric_specs, summarize_metric_grid
from .visuals import plot_heatmap

def _build_metric_grids_from_records(
    records: List[dict],
    metric_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    import numpy as np
    import re
    combos: List[Tuple[float, float]] = []
    values_by_metric: Dict[str, Dict[Tuple[float, float], float]] = {}

    base_interest = set(metric_names) | {
        "spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high"
    }

    discovered_ms_keys: set[str] = set()
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

        ks = read_eval_available_keys(ev)
        for k in ks:
            if k.startswith("best_k_") or k.startswith("at_best_"):
                discovered_ms_keys.add(k)

    if not combos:
        raise RuntimeError("没有可用的 (p, sigma) 组合或 eval_log。")

    interest = sorted(base_interest | discovered_ms_keys)

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
            v = read_eval_metric_mean(ev, k)
            if v is None:
                continue
            values_by_metric.setdefault(k, {})[(float(p), float(s))] = float(v)

    P = np.array(sorted({p for p, _ in combos}), dtype=float)
    S = np.array(sorted({s for _, s in combos}), dtype=float)

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

def _extend_specs_for_multiscale(specs: dict, metric_keys: Iterable[str]) -> dict:
    specs = dict(specs)  # copy
    for m in metric_keys:
        if m.startswith("best_k_"):
            base = m[len("best_k_"):]
            base_sp = specs.get(base, {"fullname": base})
            specs[m] = {
                "fullname": f"Equivalent Kernel Size (best_k of {base})",
                "formula": "在多尺度参考视图 y^(k) 中，令指定指标在 k 上最优（极小/极大），取其核大小 k。",
                "direction": "lower",
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
    return specs

def _write_markdown_report_ex(report_md: Path,
                              figures: Dict[str, Path],
                              P: np.ndarray,
                              S: np.ndarray,
                              grids: Dict[str, np.ndarray],
                              *,
                              title: str = "Evaluation Report",
                              extra_notes: str = "") -> None:
    specs = metric_specs()
    specs = _extend_specs_for_multiscale(specs, figures.keys())

    lines: List[str] = []
    lines.append(f"# {title}\n")
    if extra_notes:
        lines.append(f"> {extra_notes}\n")

    lines.append("## 概览（一眼看懂）\n")
    lines.append("| 指标 | 全称 | 方向 | 最佳值 | 位置(p, σ) | 覆盖率 | 评分 |\n")
    lines.append("|---|---|:--:|---:|:--:|:--:|--:|\n")

    metric_order = sorted(figures.keys())
    for m in metric_order:
        M = grids.get(m, None)
        if M is None:
            continue
        sp = specs.get(m, {"fullname": m, "direction": "higher"})
        summ = summarize_metric_grid(P, S, M, sp["direction"])
        arrow = "↑" if sp.get("direction", "higher").lower() == "higher" else "↓"
        cov = f"{summ['coverage']*100:.1f}%"
        score = f"{summ['score']:.1f}"
        best_val = "—" if summ["best_val"] is None else f"{summ['best_val']:.6g}"
        pos = "—" if summ["best_p"] is None else f"({summ['best_p']:.3f}, {summ['best_s']:.3f})"
        lines.append(f"| `{m}` | {sp['fullname']} | {arrow} | {best_val} | {pos} | {cov} | {score} |\n")

    lines.append("\n> 说明：评分基于该指标在本次 sweep 的分布做分位归一（5%–95%），按“方向”将数值映射到0–100的相对量。")
    lines.append(" 对于 `best_k_*`，我们将“更小的核＝更细的可复原尺度”视作更优，因此按“↓好”处理。\n")

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

    lines.append("\n## 可视化热力图\n")
    lines.append("横轴为观测噪声 σ，纵轴为采样密度 p。对于“↑好”的指标颜色越暖越好；对于“↓好”的指标颜色越冷越好。\n")
    for m in metric_order:
        fig = figures[m]
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
    一行式/CLI 共享入口。签名与旧版一致。
    """
    sd = Path(summary_dir).expanduser().resolve()
    records = load_sweep_records(sd)
    if not records:
        raise FileNotFoundError(f"未找到 sweep_summary.jsonl/json，也无法在 {sd} 推断有效子实验。")

    default_metrics = [
        "psnr", "ssim", "corrcoef",
        "l1", "mse", "grad_mse", "lap_mse",
        "tgrad_mse", "vort_mse", "vort_mae", "vort_corr",
        "spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high",
    ]
    metric_names = metric_whitelist or default_metrics

    P, S, grids = _build_metric_grids_from_records(records, metric_names)

    if out_dir is None:
        out_dir = sd / "report"
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    specs = metric_specs()
    figures: Dict[str, Path] = {}
    for metric, M in sorted(grids.items()):
        fig_path = od / f"{metric}.png"
        if (not fig_path.exists()) or overwrite:
            plot_heatmap(P, S, M, metric, fig_path, cmap=cmap, dpi=dpi,
                         metric_specs_lookup=specs)
        figures[metric] = fig_path
        if verbose:
            print(f"[run_report] {metric} → {fig_path.name}")

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
