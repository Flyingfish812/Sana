from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from backend2.l1 import run_l1_pipeline


def _load_config(path: str) -> Dict[str, Any]:
    """从 YAML/JSON 文件加载 L1 配置。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        cfg = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        raise ValueError("config file must be .yaml/.yml/.json")
    if not isinstance(cfg, dict):
        raise ValueError("config root must be an object")
    return cfg


def _inline_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数组装为 L1 配置字典。"""
    reader: Dict[str, Any] = {"kind": args.reader_kind, "path": args.path}
    if args.reader_kind == "h5" and args.dataset:
        reader["dataset"] = args.dataset
    if args.reader_kind == "nc" and args.var_keys:
        reader["var_keys"] = [x.strip() for x in args.var_keys.split(",") if x.strip()]
    if args.reader_kind in ("mat", "sst") and args.var:
        reader["var"] = args.var
    if args.time_key:
        reader["time_key"] = args.time_key
    if args.lon_key:
        reader["lon_key"] = args.lon_key
    if args.lat_key:
        reader["lat_key"] = args.lat_key
    return {
        "dataset_id": args.dataset_id,
        "reader": reader,
        "split": {
            "strategy": args.strategy,
            "unit": args.unit,
            "seed": args.seed,
            "ratios": {"train": args.train, "val": args.val, "test": args.test},
        },
        "normalization": {"method": args.norm},
        "artifacts_dir": args.artifacts_dir,
    }


def main() -> None:
    """CLI 入口：解析参数并执行 L1 流水线。"""
    parser = argparse.ArgumentParser(description="Run backend2 L1 pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")

    parser.add_argument("--dataset-id", type=str, default="dataset")
    parser.add_argument("--reader-kind", type=str, choices=["h5", "nc", "mat", "sst"], default="h5")
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--var-keys", type=str, default="")
    parser.add_argument("--var", type=str, default="sst")
    parser.add_argument("--time-key", type=str, default="")
    parser.add_argument("--lon-key", type=str, default="")
    parser.add_argument("--lat-key", type=str, default="")

    parser.add_argument("--strategy", type=str, choices=["random", "temporal"], default="temporal")
    parser.add_argument("--unit", type=str, choices=["frame", "sequence"], default="frame")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--norm", type=str, choices=["zscore", "minmax"], default="zscore")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    args = parser.parse_args()

    if args.config:
        config = _load_config(args.config)
    else:
        if not args.path:
            raise ValueError("--path is required when --config is not provided")
        config = _inline_config(args)

    summary = run_l1_pipeline(config)
    print(
        json.dumps(
            {
                "dataset_id": summary.dataset_id,
                "shape5d": list(summary.shape5d),
                "split_sizes": summary.split_sizes,
                "stats_method": summary.stats_method,
                "artifacts_dir": summary.artifacts_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
