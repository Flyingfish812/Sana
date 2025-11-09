# tools/report_eval.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import matplotlib.pyplot as plt
import os

def run_report(
    summary_dir: str | Path,
    out_dir: Optional[str | Path] = None,
    *,
    metric_whitelist: Optional[Iterable[str]] = None,
    patterns: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    dpi: int = 220,
    cmap: str = "RdBu_r",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    在 ipynb / 脚本中一行运行全因子实验汇总报告，等价于命令行入口。
    参数：
      - summary_dir: 扫描 metrics_grid_*.csv 的目录（或包含子目录的根目录）
      - out_dir: 报告输出目录；默认自动设为 summary_dir / 'report'
      - metric_whitelist: 只处理这些指标名（如 ['psnr','nmse']）；默认 None = 全部发现到的指标
      - patterns: 仅匹配含有这些子串的 csv 文件名；默认 None = 不限
      - overwrite: True 时覆盖已有图/报告
      - dpi, cmap: 绘图参数；cmap 统一默认 'RdBu_r'
      - verbose: 打印进度

    返回：
      {
        'summary_dir': <Path>,
        'out_dir': <Path>,
        'figures': { '<metric>': Path, ... },
        'report_md': Path | None,
        'scanned': [Path, ...]
      }
    """
    sd = Path(summary_dir).expanduser().resolve()
    if out_dir is None:
        out_dir = sd / "report"
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    # 扫描所有 metrics_grid_*.csv
    all_csv: List[Path] = sorted(sd.rglob("metrics_grid_*.csv"))
    if patterns:
        pats = [str(p) for p in patterns]
        all_csv = [p for p in all_csv if any(s in p.name for s in pats)]
    if not all_csv and verbose:
        print(f"[run_report] 未在 {sd} 下发现 metrics_grid_*.csv")

    # 从文件名中推断 metric 名（约定：metrics_grid_<metric>.csv）
    def _metric_from_name(p: Path) -> str:
        name = p.stem  # metrics_grid_<metric>
        return name.replace("metrics_grid_", "", 1)

    # 白名单过滤
    if metric_whitelist:
        allow = set(metric_whitelist)
        all_csv = [p for p in all_csv if _metric_from_name(p) in allow]

    # 读取并绘图（调用你现有的绘图/报告函数）
    figures: Dict[str, Path] = {}
    scanned: List[Path] = []
    for csv_path in all_csv:
        metric = _metric_from_name(csv_path)
        try:
            P, S, M = _load_grid_csv(csv_path)  # 复用你现有的网格加载函数
            title = f"{metric}"
            fig_path = od / f"{metric}.png"

            if (not fig_path.exists()) or overwrite:
                _plot_heatmap(P, S, M, title, fig_path, cmap=cmap, dpi=dpi)  # 修改版见下节
            figures[metric] = fig_path
            scanned.append(csv_path)
            if verbose:
                print(f"[run_report] {metric} → {fig_path.name}")
        except Exception as e:
            if verbose:
                print(f"[run_report] 跳过 {csv_path.name}: {e}")

    # 生成 markdown 报告（若你原本就有 generate_report，可在其中复用）
    report_md = od / "report.md"
    if (not report_md.exists()) or overwrite:
        try:
            _write_markdown_report(report_md, figures)  # 见下节小工具
            if verbose:
                print(f"[run_report] 写入 {report_md.name}")
        except Exception as e:
            if verbose:
                print(f"[run_report] 写入 report.md 失败: {e}")
            report_md = None  # 容错

    return {
        "summary_dir": sd,
        "out_dir": od,
        "figures": figures,
        "report_md": report_md,
        "scanned": scanned,
    }

def _load_grid_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 metrics_grid_*.csv 为 (P_list, S_list, M[p,s]) 三元组。
    CSV 结构：
      header: ["p\\sigma", s1, s2, ...]
      rows  : [p_i, m(i,1), m(i,2), ...]
    """
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if len(rows) < 2:
        raise ValueError(f"Empty grid: {path}")
    s_list = [float(x) for x in rows[0][1:]]
    p_list = []
    mat = []
    for r in rows[1:]:
        p = float(r[0]); p_list.append(p)
        vals = [float(x) if x != "" else np.nan for x in r[1:]]
        mat.append(vals)
    P = np.array(p_list, dtype=float)
    S = np.array(s_list, dtype=float)
    M = np.array(mat, dtype=float)  # shape [len(P), len(S)]
    return P, S, M

def _plot_heatmap(P: np.ndarray, S: np.ndarray, M: np.ndarray, title: str, out_png: Path, cmap: str = "RdBu_r", dpi: int = 220):
    plt.figure()
    plt.imshow(
        M,
        origin="lower",
        aspect="auto",
        extent=[S.min(), S.max(), P.min(), P.max()],
        cmap=cmap,
    )
    plt.xlabel("sigma")
    plt.ylabel("p (sample density)")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

def _write_markdown_report(md_path: Path, figures: Dict[str, Path]) -> None:
    lines = ["# Sweep Report\n"]
    for metric, fig in sorted(figures.items()):
        lines.append(f"## {metric}\n")
        rel = fig.name
        lines.append(f"![{metric}]({rel})\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")

def generate_report(summary_dir: str, top_metrics: Optional[List[str]] = None) -> Dict[str, str]:
    """
    扫描 summary_dir 下的 metrics_grid_*.csv，生成：
      - 每个指标的一张热图 PNG
      - 汇总的 report.md（插图+极值摘要）
    返回: {"report": ".../report.md", "<metric>": "...png", ...}
    """
    sdir = Path(summary_dir)
    csv_files = sorted(sdir.glob("metrics_grid_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metrics_grid_*.csv under {sdir}")

    images_index: Dict[str, str] = {}
    lines = ["# Evaluation Report\n"]
    lines.append(f"_Folder_: `{sdir}`\n")

    for csv_path in csv_files:
        metric = csv_path.stem.replace("metrics_grid_", "")
        if top_metrics and metric not in top_metrics:
            continue
        P, S, M = _load_grid_csv(csv_path)
        out_png = sdir / f"heatmap_{metric}.png"
        _plot_heatmap(P, S, M, f"{metric} (p x sigma)", out_png)
        images_index[metric] = str(out_png)

        # 统计摘要
        with np.errstate(invalid="ignore"):
            vmax = np.nanmax(M); vmin = np.nanmin(M)
            # 对于误差类指标（如 spectral_rrmse* / mse / grad_mse / lap_mse / vort_mse/mae），最佳是更小
            err_like = any(
                metric.startswith(k) for k in [
                    "spectral_rrmse", "mse", "grad_mse", "lap_mse",
                    "vort_mse", "vort_mae",
                    "nmse", "nmae", "rel_err", "region_"
                ]
            )
            best_val = vmin if err_like else vmax
            where = np.where(M == best_val)
        best_p = P[where[0][0]] if where[0].size else None
        best_s = S[where[1][0]] if where[1].size else None

        lines.append(f"## {metric}\n")
        lines.append(f"![]({out_png.name})\n")
        if best_p is not None and best_s is not None:
            lines.append(f"- **Best** `{metric}` = `{best_val:.6f}` at `p={best_p:.3f}`, `sigma={best_s:.3f}`\n")
        lines.append("\n")

    report_md = sdir / "report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    images_index["report"] = str(report_md)
    return images_index

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", required=True, help="Path to sweep summary directory")
    ap.add_argument("--metrics", nargs="*", default=None, help="Subset of metrics to include")
    args = ap.parse_args()
    out = generate_report(args.summary_dir, args.metrics)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--metrics", type=str, nargs="*", default=None, help="metric 白名单，例如 psnr nmse")
    parser.add_argument("--patterns", type=str, nargs="*", default=None, help="文件名包含这些子串才会被处理")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--cmap", type=str, default="RdBu_r")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_report(
        summary_dir=args.summary_dir,
        out_dir=args.out_dir,
        metric_whitelist=args.metrics,
        patterns=args.patterns,
        overwrite=args.overwrite,
        dpi=args.dpi,
        cmap=args.cmap,
        verbose=not args.quiet,
    )
