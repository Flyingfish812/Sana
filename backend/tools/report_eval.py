# tools/report_eval.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

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

def _plot_heatmap(P: np.ndarray, S: np.ndarray, M: np.ndarray, title: str, out_png: Path):
    plt.figure()
    # 注意：imshow 的 x->列、y->行；我们希望横轴 sigma，纵轴 p
    plt.imshow(M, origin="lower", aspect="auto",
               extent=[S.min(), S.max(), P.min(), P.max()])
    plt.xlabel("sigma")
    plt.ylabel("p (sample density)")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

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
            err_like = any(metric.startswith(k) for k in ["spectral_rrmse", "mse", "grad_mse", "lap_mse", "vort_mse", "vort_mae"])
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
    _cli()
