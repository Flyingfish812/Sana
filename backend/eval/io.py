# backend/eval/io.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import re

def read_jsonl(path: Path) -> List[dict]:
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

def load_sweep_records(run_root: Path) -> List[dict]:
    """
    优先读取 sweep_summary.jsonl/json；兜底扫描 pXX_sYYY 子目录。
    """
    cand_jsonl = run_root / "sweep_summary.jsonl"
    cand_json  = run_root / "sweep_summary.json"
    if cand_jsonl.exists():
        return read_jsonl(cand_jsonl)
    if cand_json.exists():
        obj = json.loads(cand_json.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            return obj["records"]
    recs = []
    for sub in sorted(run_root.glob("p*_s*/")):
        ev = sub / "eval_log.jsonl"
        if ev.exists():
            recs.append({"p": None, "sigma": None, "run_dir": str(sub), "eval_log": str(ev)})
    return recs

def read_eval_metric_mean(eval_log: Path, metric: str) -> Optional[float]:
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

def read_eval_multi_means(eval_log: Path, metric_names: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for m in metric_names:
        out[m] = read_eval_metric_mean(eval_log, m)
    for k in ["spectral_rrmse", "spectral_rrmse_low", "spectral_rrmse_mid", "spectral_rrmse_high"]:
        v = read_eval_metric_mean(eval_log, k)
        if v is not None:
            out[k] = v
    return out

def read_eval_available_keys(eval_log: Path) -> List[str]:
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
