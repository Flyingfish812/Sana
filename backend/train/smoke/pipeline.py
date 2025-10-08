# backend/train/smoke/pipeline.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import json, time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from backend.model.epd_system import EPDSystem
from backend.model.losses import build_loss
from .plots import save_triplet_grid
from ..data_adapter import build_dataloaders

def _pick_first_key(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(2) if x.ndim == 4 else x

def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def save_model_summary_text(model, sample_batch, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    text = [repr(model)]
    try:
        from torchinfo import summary
        x = sample_batch[0] if isinstance(sample_batch,(tuple,list)) else _pick_first_key(sample_batch, ("x","input","inputs","image"))
        x5 = _ensure_5d(x[:1])
        info = summary(model, input_data=(x5,), verbose=0, depth=4,
                       col_names=("input_size","output_size","num_params","kernel_size"))
        text.append("\n"+"="*80+"\n")
        text.append(str(info))
    except Exception as e:
        text.append(f"\n[torchinfo failed] {e}")
    (out_dir/"model_summary.txt").write_text("\n".join(text), encoding="utf-8")

def basic_fit(model: EPDSystem, train_dl: DataLoader, val_dl: Optional[DataLoader],
              device: torch.device, out_dir: Path, epochs: int = 2):
    model.to(device)
    opt_cfg = model.configure_optimizers()
    if isinstance(opt_cfg, dict):
        optimizer = opt_cfg["optimizer"]
        scheduler = opt_cfg.get("lr_scheduler", None)  # 可能是 {"scheduler": ReduceLROnPlateau, ...}
    else:
        optimizer = opt_cfg
        scheduler = None

    log = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
    global_step = 0

    for ep in range(epochs):
        # ---- train ----
        model.train()
        for i, batch in enumerate(train_dl):
            batch = _to_device(batch, device)

            # 顺序修正：先清梯度 → 前向计算 loss → 反传 → 更新参数
            optimizer.zero_grad(set_to_none=True)
            loss = model._step(batch, stage="train")  # 统一取 (x,y) 并计算已配置的 loss
            loss.backward()
            optimizer.step()
            global_step += 1

            # 对“非监控型”调度器（如 step/epoch 型）按 step 更新
            if scheduler and not isinstance(scheduler, dict):
                scheduler.step()

            if i % 10 == 0:
                rec = {"phase": "train", "epoch": ep, "step": global_step,
                       "loss": float(loss.detach().cpu().item())}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec)

        # ---- valid ----
        val_mean = None
        if val_dl is not None:
            model.eval()
            v_losses = []
            for vb in val_dl:
                vb = _to_device(vb, device)
                with torch.no_grad():
                    vloss = model._step(vb, stage="val")
                v_losses.append(float(vloss.detach().cpu().item()))
            val_mean = float(sum(v_losses) / max(1, len(v_losses)))
            rec = {"phase": "val", "epoch": ep, "step": global_step, "loss": val_mean}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec)

        # 对“监控型”调度器（如 ReduceLROnPlateau）在验证后用监控指标 step
        if scheduler and isinstance(scheduler, dict):
            sch = scheduler["scheduler"]
            monitor_val = val_mean if val_mean is not None else float(loss.detach().cpu().item())
            sch.step(monitor_val)

        # 每个 epoch 都存一份（便于对齐旧版行为）
        torch.save({"state_dict": model.state_dict()}, out_dir / f"model_epoch{ep}.pt")

    log.close()

def _to_device(batch, device):
    if isinstance(batch, torch.Tensor): return batch.to(device, non_blocking=True)
    if isinstance(batch, (tuple,list)): return type(batch)(_to_device(x, device) for x in batch)
    if isinstance(batch, dict): return {k:_to_device(v, device) for k,v in batch.items()}
    return batch

@torch.no_grad()
def eval_and_plot(model: EPDSystem, test_dl: DataLoader, device: torch.device, out_dir: Path, num_eval_batches: int = 3, num_plot_triplets: int = 4):
    out_img = out_dir/"eval_vis"; out_img.mkdir(parents=True, exist_ok=True)
    model.eval(); model.to(device)
    plotted, batches = 0, 0
    for b in test_dl:
        b = _to_device(b, device)
        if isinstance(b, (tuple,list)):
            x, y = b[0], b[1]
        else:
            x = _pick_first_key(b, ("x","input","inputs","image"))
            y = _pick_first_key(b, ("y","target","targets","label"))
        yhat = model(_ensure_5d(x)).squeeze(2) if x.ndim==4 else model(_ensure_5d(x))
        take = min(x.shape[0], num_plot_triplets - plotted)
        for i in range(take):
            save_triplet_grid(x[i] if x.ndim==4 else x[i,:,0], yhat[i] if yhat.ndim==4 else yhat[i,:,0], y[i] if y.ndim==4 else y[i,:,0],
                              out_img/f"triplet_b{batches}_i{i}.png")
            plotted += 1
        batches += 1
        if batches >= num_eval_batches or plotted >= num_plot_triplets: break

def run_smoke(
    cfg: Dict,
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
):
    """
    最短路径冒烟：与完整流程同一路线的数据入口：
      - 若显式传入 train_dl/test_dl → 直接用；
      - 否则按 cfg["data"] 的 from_run_dir / builder / snapshot_dir 自动读取（支持 prep_out/...）。
    """
    out_dir = Path(cfg["runner"]["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一数据入口
    train_dl, val_dl, test_dl = build_dataloaders(cfg.get("data", {}), injected=(train_dl, val_dl, test_dl))
    assert train_dl is not None and test_dl is not None, "smoke 模式需要 train/test 数据。"

    model_cfg = cfg["model"]
    model = EPDSystem(
        encoder=model_cfg["encoder"], propagator=model_cfg["propagator"],
        decoder=model_cfg["decoder"], head=model_cfg["head"],
        loss=model_cfg.get("loss"), optimizer=model_cfg.get("optimizer"),
        scheduler=model_cfg.get("scheduler"), reg_weights=model_cfg.get("reg_weights")
    )

    # 结构摘要
    first_batch = next(iter(train_dl)) if train_dl is not None else next(iter(test_dl))
    save_model_summary_text(model, first_batch, out_dir)

    # 训练
    device = _device(cfg["runner"].get("device", "auto"))
    epochs = cfg["runner"].get("max_epochs", 2)
    basic_fit(model, train_dl, val_dl, device, out_dir, epochs=epochs)

    # 保存/加载 ckpt（验证可恢复）
    last_ckpt = out_dir / "model_last.pt"
    torch.save({"state_dict": model.state_dict()}, last_ckpt)
    try:
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except Exception:
        pass

    # 推理与出图
    eval_and_plot(
        model, test_dl, device, out_dir,
        num_eval_batches=cfg["runner"].get("num_eval_batches", 3),
        num_plot_triplets=cfg["runner"].get("num_plot_triplets", 4),
    )

    return model, {"out_dir": str(out_dir), "ckpt": str(last_ckpt)}