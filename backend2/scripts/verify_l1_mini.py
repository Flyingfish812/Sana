from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import DataLoader

from backend2.l1 import load_l1_array_and_splits
from backend2.l1.pipeline import run_l1_pipeline
from backend2.l1.readers import build_reader
from backend2.l2.artifact_io import ArtifactManager
from backend2.l2.data import IndexedDataset, PairDataset, load_l1_array_mmap, load_split_pairs


def _l1_configs() -> List[Dict[str, Any]]:
    return [
        {
            "dataset_id": "h5_mini",
            "reader": {
                "kind": "h5",
                "path": "datasets/2D_rdb_NA_NA_mini.h5",
                "dataset": "data",
                "fill_value": 0.0,
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": "artifacts",
            "log_progress": True,
        },
        {
            "dataset_id": "nc_mini",
            "reader": {
                "kind": "nc",
                "path": "datasets/cylinder2d_mini.nc",
                "var_keys": ["u", "v"],
                "time_key": "tdim",
                "y_key": "ydim",
                "x_key": "xdim",
                "fill_value": 0.0,
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": "artifacts",
            "log_progress": True,
        },
        {
            "dataset_id": "sst_mini",
            "reader": {
                "kind": "mat",
                "path": "datasets/sst_weekly_mini.mat",
                "var": "sst",
                "lon_key": "lon",
                "lat_key": "lat",
                "time_key": "time",
                "fill_value": "global_mean",
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": "artifacts",
            "log_progress": True,
        },
    ]


def _compute_train_stats_reference(
    array5d: np.ndarray,
    train_indices: List[int],
    *,
    unit: str,
    method: str,
) -> Dict[str, Any]:
    """旧实现等价参考：仅用于 mini 一致性验证。"""
    n_size, t_size, _, _, c_size = array5d.shape
    if unit == "sequence":
        valid_seq = [i for i in train_indices if 0 <= i < n_size]
        train = array5d[valid_seq]
    else:
        picks = [(idx // t_size, idx % t_size) for idx in train_indices if 0 <= idx < n_size * t_size]
        train = np.stack([array5d[n, t] for n, t in picks], axis=0)

    axes = tuple(i for i in range(train.ndim) if i != train.ndim - 1)
    if method == "zscore":
        mean = np.nanmean(train, axis=axes)
        std = np.nanstd(train, axis=axes)
        std = np.where(std < 1e-12, 1.0, std)
        return {
            "method": "zscore",
            "channels": int(c_size),
            "mean": mean.astype(np.float64).tolist(),
            "std": std.astype(np.float64).tolist(),
        }

    min_v = np.nanmin(train, axis=axes)
    max_v = np.nanmax(train, axis=axes)
    scale = np.where((max_v - min_v) < 1e-12, 1.0, (max_v - min_v))
    return {
        "method": "minmax",
        "channels": int(c_size),
        "min": min_v.astype(np.float64).tolist(),
        "max": max_v.astype(np.float64).tolist(),
        "scale": scale.astype(np.float64).tolist(),
    }


def _max_abs_diff(a: List[float], b: List[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def _validate_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tol = 2e-6
    summary = run_l1_pipeline(cfg)
    l1_dir = Path(summary.artifacts_dir)

    array_path = l1_dir / "array5d_norm.npy"
    train_path = l1_dir / "splits" / "train.npy"
    val_path = l1_dir / "splits" / "val.npy"
    test_path = l1_dir / "splits" / "test.npy"
    stats_path = l1_dir / "stats_train.json"
    manifest_path = l1_dir / "manifest.json"

    for path in [array_path, train_path, val_path, test_path, stats_path, manifest_path]:
        if not path.exists():
            raise FileNotFoundError(f"missing expected L1 artifact: {path}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    manager = ArtifactManager(
        artifacts_dir=str(Path(cfg.get("artifacts_dir", "artifacts"))),
        dataset_id=str(cfg["dataset_id"]),
        exp_name="mini_verify",
        run_name="mini_verify",
    )
    array_mmap, manifest = load_l1_array_mmap(manager)
    if array_mmap.dtype != np.float32:
        raise TypeError(f"array dtype must be float32, got {array_mmap.dtype}")

    array_mmap2, splits2, meta2 = load_l1_array_and_splits(l1_dir)
    if array_mmap2.shape != array_mmap.shape:
        raise RuntimeError("mmap shape mismatch between loading paths")

    unit = str(dict(manifest.get("split", {})).get("unit", "frame"))
    train_indices = np.load(train_path, mmap_mode="r")
    indexed_ds = IndexedDataset(array_mmap=array_mmap, indices=train_indices, unit=unit, shape5d=manifest["shape5d"])

    train_pairs = load_split_pairs(manager, array_mmap, manifest, "train", target_offset=1)
    pair_ds = PairDataset(array5d=array_mmap, pairs=train_pairs, target_offset=1)
    loader = DataLoader(pair_ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    reader_cfg = dict(cfg["reader"])
    kind = str(reader_cfg.pop("kind"))
    raw_array = build_reader(kind=kind, **reader_cfg).read_array5d()
    unit = str(cfg["split"]["unit"])
    method = str(cfg["normalization"]["method"])
    ref_stats = _compute_train_stats_reference(
        raw_array,
        np.load(train_path).astype(np.int64).tolist(),
        unit=unit,
        method=method,
    )

    if method == "zscore":
        stats_diff = {
            "mean_max_abs_diff": _max_abs_diff(stats["mean"], ref_stats["mean"]),
            "std_max_abs_diff": _max_abs_diff(stats["std"], ref_stats["std"]),
        }
        stats_ok = stats_diff["mean_max_abs_diff"] <= tol and stats_diff["std_max_abs_diff"] <= tol
    else:
        stats_diff = {
            "min_max_abs_diff": _max_abs_diff(stats["min"], ref_stats["min"]),
            "max_max_abs_diff": _max_abs_diff(stats["max"], ref_stats["max"]),
            "scale_max_abs_diff": _max_abs_diff(stats["scale"], ref_stats["scale"]),
        }
        stats_ok = (
            stats_diff["min_max_abs_diff"] <= tol
            and stats_diff["max_max_abs_diff"] <= tol
            and stats_diff["scale_max_abs_diff"] <= tol
        )

    return {
        "dataset_id": summary.dataset_id,
        "shape5d": list(summary.shape5d),
        "split_sizes": summary.split_sizes,
        "array5d_norm": str(array_path),
        "array_dtype": str(array_mmap.dtype),
        "array_shape": list(array_mmap.shape),
        "split_sizes_from_loader": {k: int(v.shape[0]) for k, v in splits2.items()},
        "official_loader_manifest_dataset": str(dict(meta2["manifest"]).get("dataset_id", "")),
        "indexed_len": len(indexed_ds),
        "train_pairs": len(train_pairs),
        "stats_check": {
            "ok": bool(stats_ok),
            "tolerance": tol,
            "diff": stats_diff,
        },
        "batch": {
            "x": list(batch["x"].shape),
            "y": list(batch["y"].shape),
            "mask": list(batch["mask"].shape),
        },
    }


def main() -> None:
    logs: List[Dict[str, Any]] = []
    for cfg in _l1_configs():
        logs.append(_validate_one(cfg))

    out_path = Path("artifacts") / "mini_l1_verification_log.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"log": str(out_path), "results": logs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
