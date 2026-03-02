from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from backend2.l1.pipeline import run_l1_pipeline


def full_l1_configs(artifacts_dir: str = "artifacts", h5_full_construction: bool = False) -> List[Dict[str, Any]]:
    return [
        {
            "dataset_id": "h5_full",
            "reader": {
                "kind": "h5",
                "path": "datasets/2D_rdb_NA_NA.h5",
                "dataset": "data",
                "fill_value": 0.0,
                "sample_ratio": 0.1,
                "sample_mode": "interval",
                "full_construction": bool(h5_full_construction),
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": artifacts_dir,
            "log_progress": True,
            "norm_chunk_n": 1,
        },
        {
            "dataset_id": "nc_full",
            "reader": {
                "kind": "nc",
                "path": "datasets/cylinder2d.nc",
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
            "artifacts_dir": artifacts_dir,
            "log_progress": True,
            "norm_chunk_n": 1,
        },
        {
            "dataset_id": "sst_full",
            "reader": {
                "kind": "mat",
                "path": "datasets/sst_weekly.mat",
                "var": "sst",
                "lon_key": "lon",
                "lat_key": "lat",
                "time_key": "time",
                "fill_value": 0.0,
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": artifacts_dir,
            "log_progress": True,
            "norm_chunk_n": 1,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L1 full datasets and freeze artifacts")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "h5_full", "nc_full", "sst_full"])
    parser.add_argument("--h5-full-construction", action="store_true", help="Enable 100% h5_full construction")
    args = parser.parse_args()

    configs = full_l1_configs(args.artifacts_dir, h5_full_construction=args.h5_full_construction)
    if args.dataset != "all":
        configs = [cfg for cfg in configs if cfg["dataset_id"] == args.dataset]

    summaries = []
    overall_t0 = time.perf_counter()
    total = len(configs)
    for i, cfg in enumerate(configs, start=1):
        print(f"\n[L1-FULL] ({i}/{total}) start: {cfg['dataset_id']}", flush=True)
        t0 = time.perf_counter()
        summary = run_l1_pipeline(cfg)
        dt = time.perf_counter() - t0

        l1_dir = Path(summary.artifacts_dir)
        summaries.append(
            {
                "dataset_id": summary.dataset_id,
                "shape5d": list(summary.shape5d),
                "split_sizes": summary.split_sizes,
                "stats_method": summary.stats_method,
                "artifacts_dir": summary.artifacts_dir,
                "elapsed_sec": round(dt, 2),
                "array5d_norm": str(l1_dir / "array5d_norm.npy"),
                "splits": {
                    "train": str(l1_dir / "splits" / "train.npy"),
                    "val": str(l1_dir / "splits" / "val.npy"),
                    "test": str(l1_dir / "splits" / "test.npy"),
                },
                "stats_train": str(l1_dir / "stats_train.json"),
                "manifest": str(l1_dir / "manifest.json"),
            }
        )
        print(f"[L1-FULL] ({i}/{total}) done: {summary.dataset_id} in {dt:.2f}s", flush=True)

    print(f"\n[L1-FULL] all done in {time.perf_counter() - overall_t0:.2f}s", flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
