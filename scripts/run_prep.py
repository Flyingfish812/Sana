# scripts/run_prep.py
"""
用法：
  python scripts/run_prep.py --config examples/configs/h5_static.yaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import importlib

# 确保注册副作用生效（adapters/samplers/readers）
import backend.dataio as _  # noqa: F401

from backend.dataio.api import run
from backend.dataio.io.exporters import dump_prep_outputs

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

def load_config(path: str):
    p = Path(path)
    if p.suffix.lower() in [".yml", ".yaml"]:
        if yaml is None:
            raise RuntimeError("pyyaml not installed. pip install pyyaml")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # 跑流程
    dataset, dataloader, summary = run(cfg)

    # 可选导出
    output_cfg = cfg.get("output", None)
    if output_cfg:
        dump_prep_outputs(dataset=dataset, dataloader=dataloader, summary=summary, output_cfg=output_cfg)

    # 控制台打印摘要
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
