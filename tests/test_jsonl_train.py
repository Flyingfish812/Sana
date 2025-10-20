from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_jsonl_training_script_emits_events(tmp_path: Path) -> None:
    pytest.importorskip("pytorch_lightning")
    config = {
        "exp_name": "jsonl_test",
        "model": {
            "encoder": {"name": "UNetBase", "args": {"in_channels": 1, "base_channels": 8, "depth": 2}},
            "propagator": {"name": "Identity", "args": {}},
            "decoder": {"name": "UNetBase", "args": {"base_channels": 8}},
            "head": {"name": "PixelHead", "args": {"out_channels": 1}},
            "loss": {"name": "l1"},
            "optimizer": {"name": "adamw", "args": {"lr": 1e-3}},
            "scheduler": None,
            "reg_weights": {"encoder": 0.0, "propagator": 0.0, "decoder": 0.0, "head": 0.0},
        },
        "data": {
            "builder": "scripts.dummy_data:build_tiny_dataloaders",
            "builder_args": {"num_samples": 4, "batch_size": 2},
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "drop_last": False,
        },
        "trainer": {
            "max_epochs": 1,
            "max_steps": 2,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
            "log_every_n_steps": 1,
            "accelerator": "cpu",
            "devices": 1,
            "enable_checkpointing": False,
            "enable_model_summary": False,
        },
        "callbacks": {
            "early_stopping": {"enable": False},
            "checkpoint": {"monitor": "val_total", "mode": "min", "save_top_k": 0, "save_last": False},
            "lr_monitor": {"enable": False},
            "viz_triplets": {"enable": False},
            "grad_norm": {"enable": False},
        },
        "eval": {"enable": False},
        "logging": {
            "logger": "csv",
            "save_dir": str(tmp_path / "runs"),
            "name": "jsonl-test",
            "version": "dev",
        },
        "train": {"seed": 123},
    }

    cmd = [
        sys.executable,
        "-m",
        "scripts.jsonl_train",
        "--config-json",
        json.dumps(config),
        "--work-dir",
        str(Path.cwd()),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    lines = [json.loads(line) for line in proc.stdout]
    retcode = proc.wait(timeout=120)
    stderr_text = proc.stderr.read() if proc.stderr else ""
    assert retcode == 0, f"jsonl_train failed: {stderr_text}"

    types = [line.get("type") for line in lines]
    assert "status" in types
    assert "progress" in types
    assert any(line.get("event") == "completed" for line in lines if line.get("type") == "status")
