# backend/eval/report.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable, Any
from pathlib import Path
import numpy as np
from pytorch_lightning.utilities import rank_zero_only

from .io import load_sweep_records, read_eval_metric_mean, read_eval_available_keys
from .metrics import metric_specs, summarize_metric_grid
from .visuals import plot_heatmap
from .pdfprint import export_report_pdf

def _build_metric_grids_from_records(
    records: List[dict],
    metric_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], List[str], set]:
    """
    读取 sweep 的 eval_log，构建 (p, sigma) 网格：
    - grids_overall: { metric -> M[p_i, s_j] }
    - grids_by_ch  : { channel -> { metric -> M[p_i, s_j] } }
      其中 metric 为不带通道的基础名（如 "psnr"），通道名来自 "psnr/u" 的 "/u"
    另外返回：
    - channels: 出现过的通道名有序列表（按字母序）
    - discovered_ms_keys: 发现的多尺度键（best_k_*, at_best_*）
    """
    import re
    combos: List[Tuple[float, float]] = []
    values_overall: Dict[str, Dict[Tuple[float, float], float]] = {}
    values_by_ch_raw: Dict[Tuple[str, str], Dict[Tuple[float, float], float]] = {}  # (metric, ch) -> {(p,s): v}

    base_interest = set(metric_names) | {
        "spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high"
    }
    discovered_ms_keys: set[str] = set()

    # 首轮扫描：收集 (p, sigma) 与可用 metric 键
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
            # 多尺度自动扩展
            if k.startswith("best_k_") or k.startswith("at_best_"):
                discovered_ms_keys.add(k)

    if not combos:
        raise RuntimeError("没有可用的 (p, sigma) 组合或 eval_log。")

    # 第二轮：读取值
    interest = sorted(base_interest | discovered_ms_keys)
    channel_set: set[str] = set()

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
        key_pos = (float(p), float(s))

        ks = read_eval_available_keys(ev)
        # 1) 整体指标：直接使用基础名
        for m in interest:
            v = read_eval_metric_mean(ev, m)
            if v is None:
                continue
            values_overall.setdefault(m, {})[key_pos] = float(v)

        # 2) 逐通道：扫描形如 "metric/ch" 的键
        for k in ks:
            if "/" not in k:
                continue
            m, ch = k.split("/", 1)
            if (m not in interest) and (m not in base_interest):
                # 对于评估阶段才出现的新指标（但不在 white list），也接受
                pass
            v = read_eval_metric_mean(ev, k)
            if v is None:
                continue
            channel_set.add(ch)
            values_by_ch_raw.setdefault((m, ch), {})[key_pos] = float(v)

    # 组网格
    P = np.array(sorted({p for p, _ in combos}), dtype=float)
    S = np.array(sorted({s for _, s in combos}), dtype=float)

    grids_overall: Dict[str, np.ndarray] = {}
    for metric in sorted(values_overall.keys()):
        M = np.full((len(P), len(S)), np.nan, dtype=float)
        mp = values_overall[metric]
        for i, pv in enumerate(P):
            for j, sv in enumerate(S):
                val = mp.get((float(pv), float(sv)), None)
                if val is not None:
                    M[i, j] = float(val)
        grids_overall[metric] = M

    # 逐通道
    grids_by_ch: Dict[str, Dict[str, np.ndarray]] = {}
    channels = sorted(channel_set)
    for (m, ch), mp in values_by_ch_raw.items():
        M = np.full((len(P), len(S)), np.nan, dtype=float)
        for i, pv in enumerate(P):
            for j, sv in enumerate(S):
                val = mp.get((float(pv), float(sv)), None)
                if val is not None:
                    M[i, j] = float(val)
        grids_by_ch.setdefault(ch, {})[m] = M

    return P, S, grids_overall, grids_by_ch, channels, discovered_ms_keys

def _extend_specs_for_per_channel(specs: dict, grids_by_ch: Dict[str, Dict[str, np.ndarray]]) -> dict:
    """
    为每个 (metric, channel) 组合复制一份指标说明词条：
    例如把 "psnr" 复制成 "psnr/u"，全称追加 "[u]" 后缀，方向/含义/公式沿用基础指标。
    """
    specs = dict(specs)
    for ch, mdict in grids_by_ch.items():
        for m in mdict.keys():
            base = specs.get(m, {"fullname": m, "direction": "higher"})
            specs[f"{m}/{ch}"] = {
                "fullname": f"{base.get('fullname', m)} [{ch}]",
                "formula": base.get("formula", ""),
                "direction": base.get("direction", "higher"),
                "meaning": base.get("meaning", ""),
                "typical": base.get("typical", ""),
            }
    return specs

def _sort_metric_keys_for_display(keys: Iterable[str]) -> List[str]:
    """
    显示排序规则：
    1) 先基础指标（不带 /）
    2) 再多尺度 best_k_*/at_best_*（按字典序）
    3) 逐通道（带 /），按 metric 再按 channel
    """
    base = []
    ms   = []
    per  = []
    for k in keys:
        if "/" in k:
            per.append(k)
        elif k.startswith("best_k_") or k.startswith("at_best_"):
            ms.append(k)
        else:
            base.append(k)
    base.sort()
    ms.sort()
    per.sort(key=lambda s: (s.split("/")[0], s.split("/")[1]))
    return base + ms + per

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
    lines.append("| 指标 | 全称 | 方向 | 最佳值 | 位置(p, σ) | 覆盖率 | 评分 |")
    lines.append("|---|---|:--:|---:|:--:|:--:|--:|")

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
        lines.append(f"| {m} | {sp['fullname']} | {arrow} | {best_val} | {pos} | {cov} | {score} |")

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
        lines.append(f"### `{m}`\n![](./{fig.name})\n\n*解读提示*：{hint}\n")

    if extra_notes:
        lines.append("\n---\n")
        lines.append(f"_Notes_: {extra_notes}\n")

    report_md.write_text("\n".join(lines), encoding="utf-8")

def _write_markdown_report_v2(
    report_md: Path,
    figures_overall: Dict[str, Path],
    figures_per_ch: Dict[str, Dict[str, Path]],
    P: np.ndarray,
    S: np.ndarray,
    grids_overall: Dict[str, np.ndarray],
    grids_by_ch: Dict[str, Dict[str, np.ndarray]],
    *,
    title: str = "Evaluation Report",
    extra_notes: str = "",
) -> None:
    """
    生成报告（多通道友好）：
    - 概览表：先基础指标与多尺度派生，再逐通道指标
    - 指标说明：复用 metric_specs 并为每个 metric/channel 复制词条
    - 可视化：逐通道图放在独立小节，文件名包含通道名
    """
    specs = metric_specs()
    # 为多尺度键补充词条
    all_metric_keys = set(figures_overall.keys())
    for ch, mfigs in figures_per_ch.items():
        all_metric_keys.update(f"{m}/{ch}" for m in mfigs.keys())
    specs = _extend_specs_for_multiscale(specs, all_metric_keys)
    specs = _extend_specs_for_per_channel(specs, grids_by_ch)

    lines: List[str] = []
    lines.append(f"# {title}\n")
    if extra_notes:
        lines.append(f"> {extra_notes}\n")

    # —— 概览表 —— #
    lines.append("## 概览\n")
    lines.append("| 指标 | 全称 | 方向 | 最佳值 | 位置(p, σ) | 覆盖率 | 评分 |")
    lines.append("|---|---|:--:|---:|:--:|:--:|--:|")

    def _append_summary_row(m: str, M: np.ndarray):
        sp = specs.get(m, {"fullname": m, "direction": "higher"})
        summ = summarize_metric_grid(P, S, M, sp["direction"])
        arrow = "↑" if sp.get("direction", "higher").lower() == "higher" else "↓"
        cov = f"{summ['coverage']*100:.1f}%"
        score = f"{summ['score']:.1f}"
        best_val = "—" if summ["best_val"] is None else f"{summ['best_val']:.6g}"
        pos = "—" if summ["best_p"] is None else f"({summ['best_p']:.3f}, {summ['best_s']:.3f})"
        lines.append(f"| {m} | {sp['fullname']} | {arrow} | {best_val} | {pos} | {cov} | {score} |")

    # 1) 基础与多尺度
    for m in _sort_metric_keys_for_display(figures_overall.keys()):
        M = grids_overall.get(m, None)
        if M is not None:
            _append_summary_row(m, M)

    # 2) 逐通道
    if any(figures_per_ch.values()):
        lines.append("\n> **逐通道**：以下各行对应 `metric/channel` 形式的键。\n")
        lines.append("| 指标 | 全称 | 方向 | 最佳值 | 位置(p, σ) | 覆盖率 | 评分 |")
        lines.append("|---|---|:--:|---:|:--:|:--:|--:|")
        for ch in sorted(figures_per_ch.keys()):
            for m in _sort_metric_keys_for_display(figures_per_ch[ch].keys()):
                key = f"{m}/{ch}"
                M = grids_by_ch.get(ch, {}).get(m, None)
                if M is not None:
                    _append_summary_row(key, M)

    # —— 指标说明 —— #
    lines.append("\n## 指标说明（简称 / 全称 / 计算 / 含义）\n")
    shown_keys = list(figures_overall.keys())
    for ch, mfigs in figures_per_ch.items():
        shown_keys += [f"{m}/{ch}" for m in mfigs.keys()]
    for m in _sort_metric_keys_for_display(shown_keys):
        sp = specs.get(m, None)
        if sp is None:
            lines.append(f"### `{m}`\n- **全称**：{m}\n- **方向**：默认 ↑（值越大越好）\n- **含义**：未录入词典。\n")
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

    # —— 热力图 —— #
    lines.append("\n## 可视化热力图（整体）\n")
    lines.append("横轴为观测噪声 σ，纵轴为采样密度 p。对于“↑好”的指标颜色越暖越好；对于“↓好”的指标颜色越冷越好。\n")
    for m in _sort_metric_keys_for_display(figures_overall.keys()):
        fig = figures_overall[m]
        if m.startswith("best_k_"):
            hint = "色条即核大小（k）。数字越小表示可复原的细尺度越细（更好）。"
        elif m.startswith("at_best_"):
            hint = "该图为在最优尺度 k* 下的分数，方向同基础指标。"
        else:
            hint = "暖色=好（↑）或冷色=好（↓），见概览表方向。"
        lines.append(f"### `{m}`\n![](./{fig.name})\n\n*解读提示*：{hint}\n")

    if any(figures_per_ch.values()):
        lines.append("\n## 可视化热力图（逐通道）\n")
        for ch in sorted(figures_per_ch.keys()):
            if not figures_per_ch[ch]:
                continue
            lines.append(f"### 通道 `{ch}`\n")
            for m in _sort_metric_keys_for_display(figures_per_ch[ch].keys()):
                fig = figures_per_ch[ch][m]
                key = f"{m}/{ch}"
                lines.append(f"#### `{key}`\n![](./per_channel/{fig.name})\n")

    if extra_notes:
        lines.append("\n---\n")
        lines.append(f"_Notes_: {extra_notes}\n")

    report_md.write_text("\n".join(lines), encoding="utf-8")

@rank_zero_only
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
    报告流水线主入口（批次D版，多通道友好）：
      1) 读取 sweep_summary 与各组 eval_log，汇总出“整体/逐通道”网格并绘制热力图
      2) 生成 out_dir/report.md（包含概览表、指标说明、整体与逐通道热力图）
      3) 若 out_dir 或 summary_dir 下存在 eval_vis_ms/*，则附加“多尺度样例”章
      4) 可选导出 report.pdf
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

    P, S, grids_overall, grids_by_ch, channels, discovered_ms = _build_metric_grids_from_records(records, metric_names)

    if out_dir is None:
        out_dir = sd / "report"
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)
    od_pc = od / "per_channel"
    od_pc.mkdir(parents=True, exist_ok=True)

    specs = metric_specs()
    # 为多尺度与逐通道键扩充词典（便于绘图函数说明）
    specs_all = _extend_specs_for_multiscale(specs, set(list(grids_overall.keys())) | discovered_ms)
    specs_all = _extend_specs_for_per_channel(specs_all, grids_by_ch)

    # 1) 绘制整体指标热力图
    figures_overall: Dict[str, Path] = {}
    for metric, M in sorted(grids_overall.items()):
        fig_path = od / f"{metric}.png"
        if (not fig_path.exists()) or overwrite:
            plot_heatmap(P, S, M, metric, fig_path, cmap=cmap, dpi=dpi,
                         metric_specs_lookup=specs_all)
        figures_overall[metric] = fig_path
        if verbose:
            print(f"[run_report] overall {metric} → {fig_path.name}")

    # 2) 绘制逐通道热力图（使用 base metric 名 + '__' + channel 作为文件名）
    figures_per_ch: Dict[str, Dict[str, Path]] = {}
    for ch, mdict in sorted(grids_by_ch.items()):
        for metric, M in sorted(mdict.items()):
            fn = f"{metric}__{ch}.png"
            fig_path = od_pc / fn
            if (not fig_path.exists()) or overwrite:
                # 传给 plot_heatmap 的 metric 名就用 "metric/ch"，以便 specs 查到复写后的全称
                plot_heatmap(P, S, M, f"{metric}/{ch}", fig_path, cmap=cmap, dpi=dpi,
                             metric_specs_lookup=specs_all)
            figures_per_ch.setdefault(ch, {})[metric] = Path("per_channel") / fn
            if verbose:
                print(f"[run_report] per-channel {metric}/{ch} → per_channel/{fn}")

    # 3) 写 report.md（v2 版本，包含逐通道）
    report_md = od / "report.md"
    if (not report_md.exists()) or overwrite:
        _write_markdown_report_v2(
            report_md,
            figures_overall=figures_overall,
            figures_per_ch=figures_per_ch,
            P=P, S=S,
            grids_overall=grids_overall,
            grids_by_ch=grids_by_ch,
            title="Evaluation Report",
            extra_notes="本报告由 sweep_summary.jsonl 与 eval_log.jsonl 自动生成；评分为相对量纲，用于横向比较。"
        )
        if verbose:
            print(f"[run_report] 写入 {report_md.name}")

    # 4) 追加“多尺度样例”章（若存在）
    ms_dir_candidates = [od / "eval_vis_ms", sd / "eval_vis_ms"]
    ms_dir = next((p for p in ms_dir_candidates if p.exists() and any(p.glob("strip_*.png"))), None)
    if ms_dir is not None:
        ms_max = 6
        append_multiscale_section(od, max_samples=ms_max)
        if verbose:
            print(f"[run_report] 附加多尺度样例：{ms_dir}")

    return {
        "run_root": sd,
        "out_dir": od,
        "figures_overall": figures_overall,
        "figures_per_channel": figures_per_ch,
        "report_md": report_md,
        "metrics_overall": sorted(grids_overall.keys()),
        "channels": channels,
        "P": P.tolist(),
        "S": S.tolist(),
    }

def _iter_ms_pairs(ms_dir: Path) -> List[Tuple[Optional[Path], Optional[Path]]]:
    """
    容错版：允许只有 strip 或只有 curve。
    返回 (strip_png或None, curve_png或None) 的配对，按基名排序。
    """
    if not ms_dir.exists():
        return []
    strips = {p.stem.replace("strip_", ""): p for p in sorted(ms_dir.glob("strip_*.png"))}
    curves = {p.stem.replace("curve_", ""): p for p in sorted(ms_dir.glob("curve_*.png"))}
    keys = sorted(set(strips.keys()) | set(curves.keys()))
    pairs: List[Tuple[Optional[Path], Optional[Path]]] = []
    for k in keys:
        pairs.append((strips.get(k), curves.get(k)))
    return pairs

@rank_zero_only
def append_multiscale_section(
    run_dir: Path,
    *,
    section_title: str = "多尺度样例（内部判定可视化）",
    max_samples: int = 6,
    rel_base: Optional[Path] = None,
) -> Path:
    """
    在 run_dir/report.md 的末尾追加“多尺度样例”一节，将 eval_vis_ms/ 下
    的 strip_*.png 与 curve_*.png 成对插入，便于审阅 best_k 的判定过程。

    - max_samples: 最多放入多少对样例（默认 6）
    - rel_base: 渲染到 md 的相对基准路径；缺省则以 report.md 所在目录为基准
    返回最终写入的 report.md 路径。
    """
    run_dir = Path(run_dir)
    report_md = run_dir / "report.md"
    ms_dir = run_dir / "eval_vis_ms"

    pairs = _iter_ms_pairs(ms_dir)
    if not pairs:
        # 没有可用的多尺度可视化，直接返回
        return report_md

    # 组装 Markdown 片段
    lines = []
    lines.append("\n---\n")
    lines.append(f"## {section_title}\n")
    lines.append("> 本节展示模型输出与不同尺度参考(target@k)的**联图**与**指标曲线**，便于核对 `best_k` 的来源。\n")
    lines.append("> 左图为条带联图（共享色标，零居中），右图为主指标曲线并高亮最优尺度。\n\n")

    # 相对路径计算（便于 md 在不同位置渲染）
    if rel_base is None:
        rel_base = report_md.parent

    used = 0
    for strip_png, curve_png in pairs:
        if used >= max_samples:
            break
        strip_rel = strip_png.relative_to(rel_base)
        curve_rel = curve_png.relative_to(rel_base)

        # 并排布局：用一个两列表格简单并排
        lines.append("| 多尺度联图 | 指标曲线 |\n")
        lines.append("|:--:|:--:|\n")
        lines.append(f"| ![]({strip_rel.as_posix()}) | ![]({curve_rel.as_posix()}) |\n\n")

        used += 1

    # 若 report.md 不存在则创建；存在则在末尾追加
    report_md.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if report_md.exists() else "w"
    with report_md.open(mode=mode, encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_md