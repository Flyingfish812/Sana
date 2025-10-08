"""Lightweight smoke-test training loop used during prototyping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from backend.common import ensure_dir, ensure_5d, extract_xy, move_batch_to_device
from backend.viz.images import save_triplet_grid

from backend.model.epd_system import EPDSystem
from ..data_adapter import build_dataloaders
from ..inspect import save_model_summary


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

    for ep in range(epochs):
        model.train()
        for i, batch in enumerate(train_dl):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model._step(batch, stage="train")
            loss.backward()
            optimizer.step()
            global_step += 1

            if scheduler and not isinstance(scheduler, dict):
                scheduler.step()

            if i % 10 == 0:
                rec = {
                    "phase": "train",
                    "epoch": ep,
                    "step": global_step,
                    "loss": float(loss.detach().cpu().item()),
                }
                log_fp.write(json.dumps(rec) + "\n")
                log_fp.flush()
                print(rec)

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
            monitor_val = val_mean if val_mean is not None else float(loss.detach().cpu().item())
            sched.step(monitor_val)

        torch.save({"state_dict": model.state_dict()}, out_dir / f"model_epoch{ep}.pt")

    log_fp.close()


@torch.no_grad()
def eval_and_plot(
    model: EPDSystem,
    test_dl: DataLoader,
    device: torch.device,
    out_dir: Path,
    num_eval_batches: int = 3,
    num_plot_triplets: int = 4,
) -> None:
    img_dir = ensure_dir(out_dir / "eval_vis")
    model.eval()
    model.to(device)

    plotted, batches = 0, 0
    for batch in test_dl:
        if batches >= num_eval_batches or plotted >= num_plot_triplets:
            break

        batch = move_batch_to_device(batch, device)
        x, y, _ = extract_xy(batch)
        y_hat = model(ensure_5d(x))
        if x.ndim == 4:
            y_hat = y_hat.squeeze(2)

        take = min(x.shape[0], num_plot_triplets - plotted)
        for idx in range(take):
            save_triplet_grid(
                x[idx] if x.ndim == 4 else x[idx, :, 0],
                y_hat[idx] if y_hat.ndim == 4 else y_hat[idx, :, 0],
                y[idx] if y.ndim == 4 else y[idx, :, 0],
                img_dir / f"triplet_b{batches}_i{idx}.png",
            )
            plotted += 1

        batches += 1


def run_smoke(
    cfg: Dict[str, Any],
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
):
    """Execute a short training run for rapid validation of configs."""

    out_dir = ensure_dir(Path(cfg["runner"]["out_dir"]).resolve())

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
    epochs = cfg["runner"].get("max_epochs", 2)
    basic_fit(model, train_dl, val_dl, device, out_dir, epochs=epochs)

    last_ckpt = out_dir / "model_last.pt"
    torch.save({"state_dict": model.state_dict()}, last_ckpt)
    try:
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except Exception:
        pass

    eval_and_plot(
        model,
        test_dl,
        device,
        out_dir,
        num_eval_batches=cfg["runner"].get("num_eval_batches", 3),
        num_plot_triplets=cfg["runner"].get("num_plot_triplets", 4),
    )

    return model, {"out_dir": str(out_dir), "ckpt": str(last_ckpt)}
