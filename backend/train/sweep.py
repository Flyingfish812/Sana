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

def run_sweep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    在同一份“干净快照/数据入口”上做 p-σ 全因子 sweep。
    - 要求在 cfg["data"]["sweep"] 中设置 enable=True、p_list、sigma_list；
    - 每个组合将：
        * 覆盖 cfg["data"]["factors"].sample_density / noise_sigma；
        * 设置 rng_seed_offset = index；
        * 在 logging.version 下追加子目录后缀 pXX_sYYY；
        * 调用 run_training 一次，并在末尾汇总评估 PSNR。
    返回：
        {
          "root_run_dir": "<主 run 目录>",
          "records": [ { "p":..., "sigma":..., "run_dir":..., "eval_log":..., "psnr":... }, ... ],
          "summary_csv": "<summary/metrics_grid.csv>"
        }
    """
    base_cfg = load_config(cfg)
    sweep = base_cfg["data"].get("sweep", {})
    if not sweep.get("enable", False):
        raise ValueError("Sweep disabled. 请在 cfg.data.sweep.enable = true，并设置 p_list / sigma_list。")

    p_list = sweep.get("p_list") or []
    s_list = sweep.get("sigma_list") or []
    if not p_list or not s_list:
        raise ValueError("Empty p_list or sigma_list. 请提供非空列表。")

    combos = _iter_grid(p_list, s_list)

    # 确定总体 run 根目录（先让一次普通 run 准备好根目录）
    # 我们先复制一个 cfg0，仅用于准备根目录；不真正训练（快速/安全起见，仍通过 run_training 但不附后缀）
    # 这里直接用原始 base_cfg，让第一次 run_sweep 的第一组来创建根目录即可，
    # 因为 runner 会在 version 下拼子目录，所以根目录即 base_cfg.logging.version。
    # => 我们以第一组为主，其他组都会拼接后缀。

    records: List[Dict[str, Any]] = []
    metrics: Dict[Tuple[float, float], Optional[float]] = {}

    for i, (p, s) in enumerate(combos):
        cfg_i = copy.deepcopy(base_cfg)
        # 填充 factors
        fac = cfg_i["data"].setdefault("factors", {})
        fac["sample_density"] = float(p)
        fac["noise_sigma"] = float(s)
        fac["rng_seed_offset"] = int(i)  # 每组不同偏移，保证复现同时互不干扰

        # 组装后缀
        suffix = _format_run_suffix(p, s)

        # 单次训练
        _, arte = run_training(cfg_i, run_suffix=suffix)

        # 读取评估值
        eval_log = Path(arte["eval_log"])
        psnr = _read_eval_psnr(eval_log)

        records.append({
            "p": p, "sigma": s,
            "run_dir": arte["run_dir"],
            "eval_log": str(eval_log),
            "best_checkpoint": arte.get("best_checkpoint", ""),
            "psnr": psnr,
        })
        metrics[(p, s)] = psnr

    # 汇总输出
    # 根 run 目录 = 第一个记录的上级（去掉 pXX_sYYY 子目录）
    if not records:
        raise RuntimeError("No sweep records produced.")
    first_run_dir = Path(records[0]["run_dir"])
    # 去掉最后一段后缀，得到根目录
    root_run_dir = first_run_dir.parent
    summary_dir = _ensure_summary_dir(root_run_dir)

    # 写 records.jsonl
    rec_path = summary_dir / "records.jsonl"
    with rec_path.open("w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r) + "\n")

    # 写 grid CSV
    grid_csv = summary_dir / "metrics_grid.csv"
    _write_grid_csv(grid_csv, combos, metrics)

    return {
        "root_run_dir": str(root_run_dir),
        "records": records,
        "summary_csv": str(grid_csv),
    }
