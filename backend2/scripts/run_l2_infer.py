from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from backend2.l2 import run_l2_infer


def _load_config(path: str) -> Dict[str, Any]:
    """从 YAML/JSON 文件加载 L2 推理配置。"""
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


def main() -> None:
    """CLI 入口：读取配置并执行 L2 推理。"""
    parser = argparse.ArgumentParser(description="Run backend2 L2 inference pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML/JSON config path")
    args = parser.parse_args()

    summary = run_l2_infer(_load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
