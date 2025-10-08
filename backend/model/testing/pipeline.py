"""Reusable smoke-test pipeline for validating EPDSystem setups."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader

from backend.common import ensure_dir, ensure_5d, extract_xy, move_batch_to_device
from backend.viz.images import save_triplet_grid
from backend.train.inspect import save_model_summary

from ..epd_system import EPDSystem


@dataclass
class RunConfig:
    max_epochs: int = 2
    log_every_n_steps: int = 10
    device: str = "auto"
    num_eval_batches: int = 3
    num_plot_triplets: int = 4
    out_dir: str = "runs/epd_smoketest_unet"


def build_epd_from_yaml(cfg: Dict[str, Any]) -> EPDSystem:
    mcfg = cfg["model"]
    return EPDSystem(
        encoder=mcfg["encoder"],
        propagator=mcfg["propagator"],
        decoder=mcfg["decoder"],
        head=mcfg["head"],
        loss=mcfg.get("loss"),
        optimizer=mcfg.get("optimizer"),
        scheduler=mcfg.get("scheduler"),
        reg_weights=mcfg.get("reg_weights"),
    )


def _jsonable(obj: Any) -> Any:
    try:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
    except Exception:
        pass
    try:
        return dict(obj)
    except Exception:
        pass
    try:
        return obj.__dict__
    except Exception:
        return str(obj)


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_state(model: EPDSystem, out_dir: Path, tag: str = "last") -> Path:
    out_dir = ensure_dir(out_dir)
    hparams = getattr(model, "hparams", None)
    meta = {
        "class": model.__class__.__name__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hparams_json": _jsonable(hparams),
    }
    ckpt_path = out_dir / f"model_{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "_meta": meta}, ckpt_path)
    (out_dir / f"model_{tag}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ckpt_path


def load_state(model: EPDSystem, path: str | Path) -> None:
    path = str(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            import torch.serialization as ts
            from lightning_fabric.utilities.data import AttributeDict

            ts.add_safe_globals([AttributeDict])
        except Exception:
            pass
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)


def train_one_batch_step(
    model: EPDSystem,
    batch: Any,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model._step(batch, stage="train")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def val_one_batch_step(model: EPDSystem, batch: Any) -> float:
    model.eval()
    with torch.no_grad():
        loss = model._step(batch, stage="val")
    return float(loss.detach().cpu().item())


def basic_fit(
    model: EPDSystem,
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    rcfg: RunConfig,
    out_dir: Path,
) -> None:
    device = _select_device(rcfg.device)
    model.to(device)

    opt_cfg = model.configure_optimizers()
    if isinstance(opt_cfg, dict):
        optimizer = opt_cfg["optimizer"]
        scheduler = opt_cfg.get("lr_scheduler")
    else:
        optimizer = opt_cfg
        scheduler = None

    log_path = out_dir / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fp:
        global_step = 0
        for epoch in range(rcfg.max_epochs):
            for ib, batch in enumerate(train_dl):
                batch = move_batch_to_device(batch, device)
                loss = train_one_batch_step(model, batch, optimizer)
                global_step += 1

                if scheduler and not isinstance(scheduler, dict):
                    scheduler.step()

                if ib % rcfg.log_every_n_steps == 0:
                    rec = {"phase": "train", "epoch": epoch, "step": global_step, "loss": loss}
                    fp.write(json.dumps(rec) + "\n")
                    fp.flush()
                    print(rec)

            if val_dl is not None:
                v_losses = []
                for vb in val_dl:
                    vb = move_batch_to_device(vb, device)
                    v_losses.append(val_one_batch_step(model, vb))
                v_mean = float(sum(v_losses) / max(1, len(v_losses)))
                rec = {"phase": "val", "epoch": epoch, "step": global_step, "loss": v_mean}
                fp.write(json.dumps(rec) + "\n")
                fp.flush()
                print(rec)

                if scheduler and isinstance(scheduler, dict):
                    scheduler["scheduler"].step(v_mean)

            save_state(model, out_dir, tag=f"epoch{epoch}")


@torch.no_grad()
def evaluate_and_plot(
    model: EPDSystem,
    test_dl: DataLoader,
    rcfg: RunConfig,
    out_dir: Path,
) -> None:
    device = next(model.parameters()).device
    img_dir = ensure_dir(out_dir / "eval_vis")

    plotted = 0
    batches = 0
    for batch in test_dl:
        if batches >= rcfg.num_eval_batches or plotted >= rcfg.num_plot_triplets:
            break

        batch = move_batch_to_device(batch, device)
        x, y, _ = extract_xy(batch)
        y_hat = model(ensure_5d(x))
        if x.ndim == 4:
            y_hat = y_hat.squeeze(2)

        take = min(x.shape[0], rcfg.num_plot_triplets - plotted)
        for i in range(take):
            save_triplet_grid(
                x[i] if x.ndim == 4 else x[i, :, 0],
                y_hat[i] if y_hat.ndim == 4 else y_hat[i, :, 0],
                y[i] if y.ndim == 4 else y[i, :, 0],
                img_dir / f"triplet_b{batches}_i{i}.png",
            )
            plotted += 1

        batches += 1


def run_model_smoketest(
    cfg: Dict[str, Any],
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    test_dl: DataLoader,
) -> Tuple[EPDSystem, Dict[str, str]]:
    rcfg = RunConfig(**cfg.get("runner", {}))
    out_dir = ensure_dir(rcfg.out_dir)

    model = build_epd_from_yaml(cfg)

    try:
        first_batch = next(iter(train_dl))
    except StopIteration:
        first_batch = next(iter(test_dl))
    save_model_summary(model, first_batch, out_dir)

    basic_fit(model, train_dl, val_dl, rcfg, out_dir)

    last_ckpt = save_state(model, out_dir, tag="last")
    load_state(model, last_ckpt)

    evaluate_and_plot(model, test_dl, rcfg, out_dir)

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(_select_device(rcfg.device)),
        "train_len": len(train_dl) if hasattr(train_dl, "__len__") else None,
        "val_len": len(val_dl) if val_dl is not None and hasattr(val_dl, "__len__") else None,
        "test_len": len(test_dl) if hasattr(test_dl, "__len__") else None,
        "cfg": cfg,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[SMOKETEST DONE] outputs saved to: {out_dir}")

    return model, {"out_dir": str(out_dir), "ckpt": str(last_ckpt)}
