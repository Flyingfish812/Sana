"""Run ``backend.train.runner.run_training`` and emit JSONL events."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - import guarded for environments without Lightning
    import pytorch_lightning as pl  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    pl = None  # type: ignore
    _PL_IMPORT_ERROR = exc
else:  # pragma: no cover - import success is covered via integration test
    _PL_IMPORT_ERROR = None

from backend.train.runner import run_training


def _load_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config_json:
        cfg = json.loads(args.config_json)
    elif args.config:
        import yaml  # type: ignore

        with open(args.config, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
    else:
        raise SystemExit("Either --config or --config-json must be provided.")

    if args.overrides_json:
        overrides = json.loads(args.overrides_json)
        cfg = _deep_update(cfg, overrides)
    elif args.overrides_b64:
        decoded = base64.b64decode(args.overrides_b64.encode("utf-8"))
        overrides = json.loads(decoded.decode("utf-8"))
        cfg = _deep_update(cfg, overrides)

    return cfg


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _serialise(event: Dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)


def _stdout_writer(event: Dict[str, Any]) -> None:
    sys.stdout.write(_serialise(event) + "\n")
    sys.stdout.flush()


class JsonlLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        payload = {
            "type": "log",
            "logger": record.name,
            "level": record.levelname,
            "message": message,
        }
        _stdout_writer(payload)


if pl is not None:

    class JsonlProgressCallback(pl.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._total_epochs: Optional[int] = None
            self._batches_per_epoch: Optional[int] = None

        def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str | None = None) -> None:  # pragma: no cover - lightning hook
            total_epochs = trainer.max_epochs or 0
            if total_epochs <= 0 and trainer.max_steps is not None:
                total_epochs = 1
            self._total_epochs = max(total_epochs, 1)
            num_batches = trainer.num_training_batches
            if isinstance(num_batches, dict):
                total_batches = sum(int(v) for v in num_batches.values())
            elif isinstance(num_batches, Iterable):
                total_batches = sum(int(v) for v in num_batches)
            else:
                total_batches = int(num_batches)
            self._batches_per_epoch = max(total_batches, 1)
            _stdout_writer(
                {
                    "type": "progress",
                    "event": "initialised",
                    "total_epochs": self._total_epochs,
                    "batches_per_epoch": self._batches_per_epoch,
                }
            )

        def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # pragma: no cover - lightning hook
            _stdout_writer(
                {
                    "type": "progress",
                    "event": "epoch_start",
                    "epoch": trainer.current_epoch,
                    "total_epochs": self._total_epochs,
                }
            )

        def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int) -> None:  # pragma: no cover - lightning hook
            if self._total_epochs is None or self._batches_per_epoch is None:
                return
            completed_epochs = trainer.current_epoch
            step_in_epoch = batch_idx + 1
            total_steps = self._total_epochs * self._batches_per_epoch
            completed_steps = completed_epochs * self._batches_per_epoch + step_in_epoch
            progress = min(completed_steps / total_steps, 1.0)
            _stdout_writer(
                {
                    "type": "progress",
                    "event": "batch_end",
                    "epoch": completed_epochs,
                    "batch": batch_idx,
                    "step": trainer.global_step,
                    "progress": progress,
                }
            )

        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # pragma: no cover - lightning hook
            if self._total_epochs is None:
                return
            _stdout_writer(
                {
                    "type": "progress",
                    "event": "epoch_end",
                    "epoch": trainer.current_epoch,
                    "total_epochs": self._total_epochs,
                }
            )

        def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # pragma: no cover - lightning hook
            _stdout_writer({"type": "progress", "event": "completed", "progress": 1.0})

else:  # pragma: no cover - exercised when dependency missing

    class JsonlProgressCallback:  # type: ignore[override]
        def __init__(self) -> None:
            raise RuntimeError("pytorch_lightning is required to run training")


@contextmanager
def _patch_trainer(progress_callback: Any):
    from backend.train import runner as runner_module

    original = runner_module._trainer_from_cfg

    def patched(cfg: Dict[str, Any], loggers, callbacks):
        trainer = original(cfg, loggers, callbacks)
        trainer.callbacks.append(progress_callback)
        return trainer

    runner_module._trainer_from_cfg = patched
    try:
        yield
    finally:
        runner_module._trainer_from_cfg = original


def _configure_logging() -> JsonlLogHandler:
    handler = JsonlLogHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]
    return handler


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a YAML config file.", default=None)
    parser.add_argument("--config-json", help="Config JSON string.", default=None)
    parser.add_argument("--overrides-json", help="JSON string merged into the config.", default=None)
    parser.add_argument("--overrides-b64", help="Base64 encoded JSON overrides.", default=None)
    parser.add_argument("--work-dir", help="Working directory to chdir into before running.", default=None)
    args = parser.parse_args(argv)

    if args.work_dir:
        os.chdir(args.work_dir)

    cfg = _load_config_from_args(args)
    cfg.setdefault("trainer", {})
    cfg["trainer"]["enable_progress_bar"] = False
    handler = _configure_logging()
    _stdout_writer({"type": "status", "event": "starting"})

    if _PL_IMPORT_ERROR is not None:
        _stdout_writer({"type": "error", "message": str(_PL_IMPORT_ERROR)})
        return 1

    progress_callback = JsonlProgressCallback()

    try:
        with _patch_trainer(progress_callback):
            model, artefacts = run_training(cfg)
    except Exception as exc:  # pragma: no cover - exercised via integration tests
        tb = traceback.format_exc()
        _stdout_writer({"type": "error", "message": str(exc), "traceback": tb})
        return 1
    finally:
        root = logging.getLogger()
        if handler in root.handlers:
            root.removeHandler(handler)

    _stdout_writer({"type": "status", "event": "completed", "artefacts": artefacts})
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
