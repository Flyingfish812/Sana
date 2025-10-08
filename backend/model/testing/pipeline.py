# backend/model/testing/pipeline.py
from __future__ import annotations
import os, json, time, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

# 依赖我们第一阶段写好的骨架
from ..epd_system import EPDSystem
from ..factory import build_component
from ..losses import build_loss

# 可选：torchinfo 用于结构摘要
try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

from .plots import save_triplet_grid


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _device_select(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# pipeline.py 顶部（import 之后任意位置）
def _pick_first_key(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

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


def save_model_summary(model: nn.Module, sample_batch: Any, out_dir: Path) -> None:
    out_dir = _ensure_dir(out_dir)
    # 取一个样本，统一成 [B,C,T,H,W]
    if isinstance(sample_batch, (tuple, list)):
        x = sample_batch[0]
    elif isinstance(sample_batch, dict):
        x = _pick_first_key(sample_batch, ("x", "input", "inputs", "image"))
    else:
        raise ValueError("Unsupported batch format for summary preview.")

    if x is None:
        raise KeyError("Cannot find input tensor in batch. Tried keys: x/input/inputs/image.")

    if x.ndim == 4:
        x5 = x[:1].unsqueeze(2)  # [1,C,H,W] -> [1,C,1,H,W]
    elif x.ndim == 5:
        x5 = x[:1]
    else:
        raise ValueError(f"Expect 4D/5D input, got {x.shape}")

    # torchinfo 结构摘要
    text = []
    text.append(repr(model))
    text.append("\n" + "="*80 + "\n")
    if torchinfo_summary is not None:
        try:
            info = torchinfo_summary(model, input_data=(x5,), verbose=0, depth=4, col_names=("input_size","output_size","num_params","kernel_size"))
            text.append(str(info))
        except Exception as e:
            text.append(f"[torchinfo failed] {e}")
    else:
        # 退化：仅统计参数量
        num_params = sum(p.numel() for p in model.parameters())
        text.append(f"Total parameters: {num_params:,}")

    (out_dir / "model_summary.txt").write_text("\n".join(text), encoding="utf-8")

def _jsonable(obj):
    # 尝试把 Lightning 的 AttributeDict / dataclass 等转成普通 dict；不行就转字符串
    try:
        if hasattr(obj, "to_dict"):  # AttributeDict
            return obj.to_dict()
    except Exception:
        pass
    try:
        return dict(obj)
    except Exception:
        try:
            return obj.__dict__
        except Exception:
            return str(obj)

def save_state(model, out_dir: Path, tag: str = "last") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 仅保存 state_dict；其它信息用 JSON 存，避免非安全反序列化
    hparams = getattr(model, "hparams", None)
    meta = {
        "class": model.__class__.__name__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hparams_json": _jsonable(hparams),
    }
    ckpt_path = out_dir / f"model_{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "_meta": meta}, ckpt_path)
    # 同步保存 meta json（便于人工查看）
    (out_dir / f"model_{tag}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ckpt_path

def load_state(model, path: str | Path) -> None:
    path = str(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # 兼容老档：允许 Lightning AttributeDict 进入安全名单
        try:
            import torch.serialization as ts
            from lightning_fabric.utilities.data import AttributeDict
            ts.add_safe_globals([AttributeDict])
        except Exception:
            pass  # 如果没有 lightning_fabric 也没关系
        # 老档需要 weights_only=False 才能完整解出 state_dict
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)  # 兼容非常老的只保存了 state_dict 的情况
    model.load_state_dict(state_dict, strict=True)

def train_one_batch_step(
    model: EPDSystem,
    batch: Any,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # 复用 EPDSystem 的 _step 逻辑（它已统一抽取 (x,y) 并做 loss）
    loss = model._step(batch, stage="train")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def val_one_batch_step(model: EPDSystem, batch: Any) -> float:
    model.eval()
    with torch.no_grad():
        loss = model._step(batch, stage="val")
    return float(loss.detach().cpu().item())


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch


def basic_fit(
    model: EPDSystem,
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    rcfg: RunConfig,
    out_dir: Path,
) -> None:
    device = _device_select(rcfg.device)
    model.to(device)
    # 用 EPDSystem.configure_optimizers() 生成优化器/调度
    opt_cfg = model.configure_optimizers()
    if isinstance(opt_cfg, dict):
        optimizer = opt_cfg["optimizer"]
        scheduler = opt_cfg.get("lr_scheduler", None)
    else:
        optimizer = opt_cfg
        scheduler = None

    log_path = out_dir / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fp:
        global_step = 0
        for epoch in range(rcfg.max_epochs):
            # ---- train ----
            for ib, batch in enumerate(train_dl):
                batch = move_batch_to_device(batch, device)
                loss = train_one_batch_step(model, batch, optimizer)
                global_step += 1

                if scheduler and not isinstance(scheduler, dict):
                    # CosineAnnealingLR 等 step 型
                    scheduler.step()

                if ib % rcfg.log_every_n_steps == 0:
                    rec = {"phase":"train","epoch":epoch,"step":global_step,"loss":loss}
                    fp.write(json.dumps(rec) + "\n"); fp.flush()
                    print(rec)

            # ---- valid ----
            if val_dl is not None:
                v_losses = []
                for j, vb in enumerate(val_dl):
                    vb = move_batch_to_device(vb, device)
                    vloss = val_one_batch_step(model, vb)
                    v_losses.append(vloss)
                v_mean = float(sum(v_losses)/max(1,len(v_losses)))
                rec = {"phase":"val","epoch":epoch,"step":global_step,"loss":v_mean}
                fp.write(json.dumps(rec) + "\n"); fp.flush()
                print(rec)

                # ReduceLROnPlateau 等需要 monitor 的
                if scheduler and isinstance(scheduler, dict):
                    sch = scheduler["scheduler"]
                    sch.step(v_mean)

            save_state(model, out_dir, tag=f"epoch{epoch}")


@torch.no_grad()
def evaluate_and_plot(
    model: EPDSystem,
    test_dl: DataLoader,
    rcfg: RunConfig,
    out_dir: Path,
) -> None:
    device = next(model.parameters()).device
    out_img_dir = _ensure_dir(out_dir / "eval_vis")
    plotted = 0
    batches = 0

    for batch in test_dl:
        batch = move_batch_to_device(batch, device)
        # 取 (x,y,meta)
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x = _pick_first_key(batch, ("x", "input", "inputs", "image"))
            y = _pick_first_key(batch, ("y", "target", "targets", "label"))
        else:
            raise ValueError("Unsupported batch format in evaluate_and_plot.")

        if x is None or y is None:
            raise KeyError("Cannot find (x, y) in batch dict. "
                        "Tried x/input/inputs/image for x and y/target/targets/label for y.")

        # 前向
        model.eval()
        if x.ndim == 4:
            x5 = x.unsqueeze(2)
        else:
            x5 = x
        y_hat = model(x5)               # [B,C,T,H,W]
        if y_hat.ndim == 5 and y.ndim == 4:
            y_hat_show = y_hat.squeeze(2)  # 与 y 对齐
        else:
            y_hat_show = y_hat

        # 存图：输入-输出-目标 三连图
        b = x.shape[0]
        take = min(b, rcfg.num_plot_triplets - plotted)
        for i in range(take):
            save_triplet_grid(
                x[i], y_hat_show[i], y[i],
                out_img_dir / f"triplet_b{batches}_i{i}.png"
            )
            plotted += 1

        batches += 1
        if batches >= rcfg.num_eval_batches or plotted >= rcfg.num_plot_triplets:
            break


def run_model_smoketest(
    cfg: Dict[str, Any],
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    test_dl: DataLoader,
) -> Tuple[EPDSystem, Dict[str, str]]:
    """一键冒烟：组装 -> 保存结构 -> 训练 -> 保存/加载 -> 推理 -> 存图/日志"""
    rcfg = RunConfig(**cfg.get("runner", {}))
    out_dir = _ensure_dir(rcfg.out_dir)

    # 1) 组装模型
    model = build_epd_from_yaml(cfg)

    # 2) 保存训练前的模型结构摘要
    try:
        first_batch = next(iter(train_dl))
    except StopIteration:
        first_batch = next(iter(test_dl))
    save_model_summary(model, first_batch, out_dir)

    # 3) 训练（固定少量 epoch）
    basic_fit(model, train_dl, val_dl, rcfg, out_dir)

    # 4) 保存 & 加载（验证可恢复）
    last_ckpt = save_state(model, out_dir, tag="last")
    load_state(model, last_ckpt)  # 立即测试一次严格加载

    # 5) 推理 + 三连图
    evaluate_and_plot(model, test_dl, rcfg, out_dir)

    # 6) 记录 run 元信息
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(_device_select(rcfg.device)),
        "train_len": len(train_dl) if hasattr(train_dl, "__len__") else None,
        "val_len": len(val_dl) if val_dl is not None and hasattr(val_dl, "__len__") else None,
        "test_len": len(test_dl) if hasattr(test_dl, "__len__") else None,
        "cfg": cfg,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[SMOKETEST DONE] outputs saved to: {out_dir}")

    return model, {"out_dir": str(out_dir), "ckpt": str(last_ckpt)}
