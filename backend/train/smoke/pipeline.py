"""Lightweight smoke-test training loop used during prototyping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
import yaml

from backend.common import ensure_dir, move_batch_to_device

from backend.model.epd_system import EPDSystem
from ..data_adapter import build_dataloaders
from ..inspect import save_model_summary
from ..eval import evaluate, render_eval_triplets


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def basic_fit(
    model: EPDSystem,
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    device: torch.device,
    out_dir: Path,
    epochs: int = 2,
    max_steps: int | None = None,
) -> None:
    model.to(device)
    opt_cfg = model.configure_optimizers()
    if isinstance(opt_cfg, dict):
        optimizer = opt_cfg["optimizer"]
        scheduler = opt_cfg.get("lr_scheduler")
    else:
        optimizer = opt_cfg
        scheduler = None

    log_fp = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
    global_step = 0

    best_metric = float("inf")
    best_path = out_dir / "model_best.pt"
    last_loss_value = None

    for ep in range(epochs):
        model.train()
        train_losses = []
        for i, batch in enumerate(train_dl):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model._step(batch, stage="train")
            loss.backward()
            optimizer.step()
            global_step += 1

            loss_value = float(loss.detach().cpu().item())
            train_losses.append(loss_value)
            last_loss_value = loss_value

            if scheduler and not isinstance(scheduler, dict):
                scheduler.step()

            if i % 10 == 0:
                rec = {
                    "phase": "train",
                    "epoch": ep,
                    "step": global_step,
                    "loss": loss_value,
                }
                log_fp.write(json.dumps(rec) + "\n")
                log_fp.flush()
                print(rec)

            if max_steps is not None and global_step >= max_steps:
                break

        if max_steps is not None and global_step >= max_steps:
            break

        train_mean = float(sum(train_losses) / len(train_losses)) if train_losses else None
        val_mean = None
        if val_dl is not None:
            model.eval()
            val_losses = []
            for vb in val_dl:
                vb = move_batch_to_device(vb, device)
                with torch.no_grad():
                    vloss = model._step(vb, stage="val")
                val_losses.append(float(vloss.detach().cpu().item()))
            if val_losses:
                val_mean = float(sum(val_losses) / len(val_losses))
                rec = {
                    "phase": "val",
                    "epoch": ep,
                    "step": global_step,
                    "loss": val_mean,
                }
                log_fp.write(json.dumps(rec) + "\n")
                log_fp.flush()
                print(rec)

        if scheduler and isinstance(scheduler, dict):
            sched = scheduler["scheduler"]
            monitor_val = val_mean if val_mean is not None else (train_mean if train_mean is not None else last_loss_value)
            sched.step(monitor_val)

        metric = val_mean if val_mean is not None else (train_mean if train_mean is not None else last_loss_value)
        if metric is not None and metric < best_metric:
            best_metric = metric
            torch.save({"state_dict": model.state_dict()}, best_path)

    log_fp.close()


def run_smoke(
    cfg: Dict[str, Any],
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
):
    """Execute a short training run for rapid validation of configs."""

    out_dir = ensure_dir(Path(cfg["runner"]["out_dir"]).resolve())
    (out_dir / "config.dump.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    train_dl, val_dl, test_dl = build_dataloaders(cfg.get("data", {}), injected=(train_dl, val_dl, test_dl))
    if train_dl is None or test_dl is None:
        raise ValueError("Smoke pipeline requires both train and test dataloaders.")

    model_cfg = cfg["model"]
    model = EPDSystem(
        encoder=model_cfg["encoder"],
        propagator=model_cfg["propagator"],
        decoder=model_cfg["decoder"],
        head=model_cfg["head"],
        loss=model_cfg.get("loss"),
        optimizer=model_cfg.get("optimizer"),
        scheduler=model_cfg.get("scheduler"),
        reg_weights=model_cfg.get("reg_weights"),
    )

    first_batch = next(iter(train_dl))
    save_model_summary(model, first_batch, out_dir)

    device = _select_device(cfg["runner"].get("device", "auto"))
    runner_cfg = cfg.get("runner", {})
    epochs = runner_cfg.get("max_epochs", 2)
    steps = runner_cfg.get("max_steps")
    basic_fit(model, train_dl, val_dl, device, out_dir, epochs=epochs, max_steps=steps)

    last_ckpt = out_dir / "model_last.pt"
    torch.save({"state_dict": model.state_dict()}, last_ckpt)
    try:
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except Exception:
        pass

    eval_cfg = {
        "num_eval_batches": cfg["runner"].get("num_eval_batches", 3),
        "num_plot_triplets": cfg["runner"].get("num_plot_triplets", 4),
    }
    model.to(device)
    model.eval()
    eval_vis = render_eval_triplets(model, test_dl, out_dir, eval_cfg)
    evaluate(model, test_dl, out_dir, eval_cfg)

    best_ckpt = out_dir / "model_best.pt"
    if not best_ckpt.exists():
        torch.save({"state_dict": model.state_dict()}, best_ckpt)

    artefacts = {
        "run_dir": str(out_dir),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "train_log": str(out_dir / "train_log.jsonl"),
        "eval_log": str(out_dir / "eval_log.jsonl"),
        "eval_vis": str(eval_vis),
        "config": str(out_dir / "config.dump.yaml"),
    }

    return model, artefacts
