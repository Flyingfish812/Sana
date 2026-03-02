from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json
import time

import numpy as np

from .readers import build_reader
from .splits import split_indices
from .stats import compute_train_stats, write_normalized_array5d_memmap
from .types import L1Summary


def _now_iso() -> str:
    """返回 UTC ISO8601 时间戳，用于产物元数据记录。"""
    return datetime.now(timezone.utc).isoformat()


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    """将字典写入 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _log(enabled: bool, dataset_id: str, message: str) -> None:
    """按开关输出带数据集前缀的 L1 日志。"""
    if enabled:
        print(f"[L1][{dataset_id}] {message}", flush=True)


def run_l1_pipeline(config: Dict[str, Any]) -> L1Summary:
    """执行 L1 全流程：读取数据、训练统计、全量归一化并冻结产物。"""
    t0 = time.perf_counter()
    dataset_id = str(config["dataset_id"])
    log_enabled = bool(config.get("log_progress", True))
    reader_cfg = dict(config["reader"])
    split_cfg = dict(config.get("split", {}))
    norm_cfg = dict(config.get("normalization", {}))

    _log(log_enabled, dataset_id, "start")

    artifacts_root = Path(config.get("artifacts_dir", "artifacts"))
    l1_dir = artifacts_root / dataset_id / "L1"
    l1_dir.mkdir(parents=True, exist_ok=True)
    _log(log_enabled, dataset_id, f"artifacts dir ready: {l1_dir}")

    kind = reader_cfg.pop("kind")
    _log(log_enabled, dataset_id, f"build reader: {kind}")
    reader = build_reader(kind=kind, **reader_cfg)

    t_probe = time.perf_counter()
    shape5d, meta = reader.probe()
    _log(log_enabled, dataset_id, f"probe done: shape={shape5d}, dt={time.perf_counter()-t_probe:.2f}s")

    strategy = str(split_cfg.get("strategy", "temporal"))
    unit = str(split_cfg.get("unit", "frame"))
    ratios = split_cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = int(split_cfg.get("seed", 123))

    t_split = time.perf_counter()
    splits = split_indices(shape5d=shape5d, strategy=strategy, unit=unit, ratios=ratios, seed=seed)
    _log(
        log_enabled,
        dataset_id,
        f"split done: sizes={{train:{len(splits['train'])}, val:{len(splits['val'])}, test:{len(splits['test'])}}}, dt={time.perf_counter()-t_split:.2f}s",
    )

    _log(log_enabled, dataset_id, "read array5d (this may take long)")
    t_read = time.perf_counter()
    array5d = reader.read_array5d()
    _log(log_enabled, dataset_id, f"read done: dtype={array5d.dtype}, dt={time.perf_counter()-t_read:.2f}s")

    method = str(norm_cfg.get("method", "zscore"))
    _log(log_enabled, dataset_id, f"compute train stats: method={method}")
    t_stats = time.perf_counter()
    stats = compute_train_stats(array5d, splits["train"], unit=unit, method=method)
    _log(log_enabled, dataset_id, f"stats done: dt={time.perf_counter()-t_stats:.2f}s")

    split_dir = l1_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    _log(log_enabled, dataset_id, f"write split indices: {split_dir}")
    for name in ("train", "val", "test"):
        np.save(split_dir / f"{name}.npy", np.asarray(splits[name], dtype=np.int64))

    _log(log_enabled, dataset_id, "normalize full array5d with train stats")
    t_norm = time.perf_counter()
    chunk_n = int(config.get("norm_chunk_n", 1))
    out_path = l1_dir / "array5d_norm.npy"
    write_normalized_array5d_memmap(array5d, stats, out_path, chunk_n=chunk_n)
    del array5d
    array5d_norm = np.load(out_path, mmap_mode="r")
    _log(log_enabled, dataset_id, f"write normalized array done: dt={time.perf_counter()-t_norm:.2f}s")

    manifest = {
        "dataset_id": dataset_id,
        "created_at": _now_iso(),
        "layout": "NTHWC",
        "shape5d": list(shape5d),
        "dtype": str(array5d_norm.dtype),
        "array5d_norm": "array5d_norm.npy",
        "reader": {"kind": kind, **reader_cfg},
        "meta": meta.to_json(),
        "split": {
            "strategy": strategy,
            "unit": unit,
            "ratios": ratios,
            "seed": seed,
            "sizes": {k: len(v) for k, v in splits.items()},
            "files": {
                "train": "splits/train.npy",
                "val": "splits/val.npy",
                "test": "splits/test.npy",
            },
        },
    }
    _dump_json(l1_dir / "manifest.json", manifest)
    _dump_json(l1_dir / "stats_train.json", stats)
    _log(log_enabled, dataset_id, f"write manifest/stats done, total dt={time.perf_counter()-t0:.2f}s")

    return L1Summary(
        dataset_id=dataset_id,
        shape5d=shape5d,
        split_sizes={k: len(v) for k, v in splits.items()},
        stats_method=method,
        artifacts_dir=str(l1_dir),
    )
