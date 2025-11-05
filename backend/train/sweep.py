# backend/train/sweep.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import csv
import copy

from .config import load_config
from .runner import run_training, _format_run_suffix  # 使用 runner 中已有的格式化逻辑

def _iter_grid(p_list: List[float], sigma_list: List[float]) -> List[Tuple[Optional[float], Optional[float]]]:
    combos = []
    for p in p_list:
        for s in sigma_list:
            combos.append((float(p), float(s)))
    return combos

def _read_eval_psnr(eval_log_path: Path) -> Optional[float]:
    """
    读取 eval_log.jsonl，返回一个稳定代表值：
    - 首选平均 PSNR；如文件为空则返回 None。
    """
    if not eval_log_path.exists():
        return None
    ps = []
    with eval_log_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                obj = json.loads(line.strip())
                if "psnr" in obj:
                    ps.append(float(obj["psnr"]))
            except Exception:
                continue
    if not ps:
        return None
    return sum(ps) / len(ps)

def _ensure_summary_dir(root_run_dir: Path) -> Path:
    out = root_run_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _write_grid_csv(out_csv: Path, combos: List[Tuple[float, float]], metrics: Dict[Tuple[float, float], Optional[float]]):
    """
    将 (p, sigma) -> psnr 的表写成 CSV，行 = p，列 = sigma。
    若某项缺失则写空字符串。
    """
    # 排序，保证稳定输出
    p_sorted = sorted(set(p for p, _ in combos))
    s_sorted = sorted(set(s for _, s in combos))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["p\\sigma"] + [f"{s:.3f}" for s in s_sorted]
        w.writerow(header)
        for p in p_sorted:
            row = [f"{p:.3f}"]
            for s in s_sorted:
                v = metrics.get((p, s))
                row.append("" if v is None else f"{v:.4f}")
            w.writerow(row)

def _read_eval_metric(eval_log_path: Path, metric_name: str) -> Optional[float]:
    """
    从 eval_log.jsonl 读取指定 metric 的均值。
    若该 metric 不存在或文件缺失，返回 None。
    """
    if not eval_log_path.exists():
        return None
    vals = []
    with eval_log_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                obj = json.loads(line.strip())
                if metric_name in obj:
                    vals.append(float(obj[metric_name]))
            except Exception:
                continue
    if not vals:
        return None
    return sum(vals) / len(vals)

def _read_eval_multi(eval_log_path: Path, metric_names: List[str]) -> Dict[str, Optional[float]]:
    """
    读取一组 metric 的均值，返回 {metric: value_or_None}
    额外约定：若存在 spectral_rrmse_low/mid/high/overall 也会被读取（如果用户在评估中启用了频域）。
    """
    results = {}
    for m in metric_names:
        results[m] = _read_eval_metric(eval_log_path, m)

    # 频域摘要（若评估端已写入这些键）
    for k in ["spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high"]:
        v = _read_eval_metric(eval_log_path, k)
        if v is not None:
            results[k] = v
    return results

def _write_multi_grid_csv(summary_dir: Path,
                          combos: List[Tuple[float, float]],
                          multi_metrics: Dict[str, Dict[Tuple[float, float], Optional[float]]],
                          main_metric: str = "psnr") -> Dict[str, str]:
    """
    对每个 metric 写出一张 p×σ 网格 CSV。
    返回 {metric: csv_path_str}，并额外写一个 metrics_grid.csv 作为 main_metric 的别名。
    """
    p_sorted = sorted(set(p for p, _ in combos))
    s_sorted = sorted(set(s for _, s in combos))
    out_index: Dict[str, str] = {}

    # 为每个 metric 写一份
    for metric, grid in multi_metrics.items():
        out_csv = summary_dir / f"metrics_grid_{metric}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["p\\sigma"] + [f"{s:.3f}" for s in s_sorted]
            w.writerow(header)
            for p in p_sorted:
                row = [f"{p:.3f}"]
                for s in s_sorted:
                    v = grid.get((p, s))
                    row.append("" if v is None else f"{v:.6f}")
                w.writerow(row)
        out_index[metric] = str(out_csv)

    # 兼容旧产物：以 main_metric 作为默认 metrics_grid.csv
    if main_metric in out_index:
        alias = summary_dir / "metrics_grid.csv"
        # 直接再写一份（避免创建符号链接在部分平台权限问题）
        with open(out_index[main_metric], "r", encoding="utf-8") as src, alias.open("w", encoding="utf-8") as dst:
            dst.write(src.read())
        out_index["default"] = str(alias)

    return out_index

def run_sweep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    在同一份“干净快照/数据入口”上做 p-σ 全因子 sweep，并汇总**多指标**的 p×σ 网格。
    需要配合 eval.metrics（批次A已加入），未配置则默认只统计 psnr。
    """
    base_cfg = load_config(cfg)
    sweep = base_cfg["data"].get("sweep", {})
    if not sweep.get("enable", False):
        raise ValueError("Sweep disabled. 请在 cfg.data.sweep.enable = true，并设置 p_list / sigma_list。")

    p_list = sweep.get("p_list") or []
    s_list = sweep.get("sigma_list") or []
    if not p_list or not s_list:
        raise ValueError("Empty p_list or sigma_list. 请提供非空列表。")

    # 从评估配置里读取要汇总的指标；为空则退化为 ['psnr']
    eval_cfg = base_cfg.get("eval", {})
    metric_names: List[str] = list(eval_cfg.get("metrics", []) or [])
    if not metric_names:
        metric_names = ["psnr"]

    combos = _iter_grid(p_list, s_list)

    records: List[Dict[str, Any]] = []
    # 多指标网格：{metric_name: {(p,s): value}}
    multi_metrics: Dict[str, Dict[Tuple[float, float], Optional[float]]] = {}

    for i, (p, s) in enumerate(combos):
        cfg_i = copy.deepcopy(base_cfg)
        # 注入 factors
        fac = cfg_i["data"].setdefault("factors", {})
        fac["sample_density"] = float(p)
        fac["noise_sigma"] = float(s)
        fac["rng_seed_offset"] = int(i)

        suffix = _format_run_suffix(p, s)
        _, arte = run_training(cfg_i, run_suffix=suffix)

        eval_log = Path(arte["eval_log"])
        # 读取多指标
        metric_values = _read_eval_multi(eval_log, metric_names)

        # 逐条塞进 multi_metrics
        for m, v in metric_values.items():
            grid = multi_metrics.setdefault(m, {})
            grid[(p, s)] = v

        # 兼容旧字段：把 psnr 单独挂在 records 里（如果有）
        psnr_val = metric_values.get("psnr", None)

        records.append({
            "p": p, "sigma": s,
            "run_dir": arte["run_dir"],
            "eval_log": str(eval_log),
            "best_checkpoint": arte.get("best_checkpoint", ""),
            "psnr": psnr_val,
            **{m: metric_values.get(m, None) for m in metric_values.keys()}
        })

    if not records:
        raise RuntimeError("No sweep records produced.")

    # 根目录与 summary
    first_run_dir = Path(records[0]["run_dir"])
    root_run_dir = first_run_dir.parent
    summary_dir = _ensure_summary_dir(root_run_dir)

    # 写 records.jsonl（记录所有已读指标）
    rec_path = summary_dir / "records.jsonl"
    with rec_path.open("w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r) + "\n")

    # 输出多指标网格：每个 metric 一张 CSV
    grid_index = _write_multi_grid_csv(summary_dir, combos, multi_metrics, main_metric="psnr")

    # 返回值里带上各个网格路径，方便上层脚本使用
    return {
        "root_run_dir": str(root_run_dir),
        "records": records,
        "grid_csv_by_metric": grid_index,   # e.g. {"psnr": ".../metrics_grid_psnr.csv", "ssim": "...", "default": ".../metrics_grid.csv"}
    }