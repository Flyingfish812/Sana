# backend/train/logging.py
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml
import json

def prepare_run_dir(cfg: Dict, only_prepare: bool = False) -> Path:
    save_dir = Path(cfg["logging"]["save_dir"])
    name = cfg["logging"]["name"]
    version = cfg["logging"]["version"]
    run_dir = save_dir / name / str(version)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 冻结一份配置到 run_dir
    if not only_prepare:
        (run_dir / "config.dump.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return run_dir

class JSONLLogger(pl.loggers.Logger):
    """极简 JSONL Logger：把 logger.experiment.write(rec) 映射到 jsonl 文件"""
    def __init__(self, save_dir: str, name: str, version: str):
        super().__init__()
        self._save_dir = Path(save_dir); self._name = name; self._version = version
        self._path = self._save_dir / name / version / "events" / "metrics.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self._path.open("a", encoding="utf-8")

    @property
    def name(self): return "jsonl"
    @property
    def version(self): return self._version
    @property
    def experiment(self): return self  # 兼容 write

    def log_metrics(self, metrics, step):
        import json
        rec = {"step": int(step)} | {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self._f.write(json.dumps(rec) + "\n"); self._f.flush()

    def log_hyperparams(self, params): pass
    def save(self): self._f.flush()
    def finalize(self, status): self._f.close()

def build_loggers(log_cfg: Dict[str, Any], run_dir: Path):
    typ = (log_cfg.get("logger") or "tensorboard").lower()
    base_save = str(run_dir.parents[1])  # runs/
    name = run_dir.parents[0].name
    version = run_dir.name

    if typ == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        return TensorBoardLogger(save_dir=base_save, name=name, version=version, default_hp_metric=False)
    elif typ == "csv":
        return CSVLogger(save_dir=base_save, name=name, version=version)
    elif typ == "jsonl":
        return JSONLLogger(save_dir=base_save, name=name, version=version)
    else:
        raise ValueError(f"Unsupported logger: {typ}")
