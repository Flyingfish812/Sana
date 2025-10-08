# backend/train/eval.py
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch

from .utils import ensure_dir, move_batch_to_device, select_device
from .metrics import psnr

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def _pick_first_key(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

@torch.no_grad()
def evaluate(model, test_dl, run_dir: Path, cfg_eval: Dict[str, Any]):
    """
    计算基础指标（loss 已由 model 内部记录；这里增加 PSNR 等可观测指标）
    结果写入 runs/<exp>/<ver>/eval_log.jsonl
    """
    log_path = run_dir / "eval_log.jsonl"
    device = next(model.parameters()).device
    with log_path.open("a", encoding="utf-8") as fp:
        n_batches = 0
        for batch in test_dl:
            batch = move_batch_to_device(batch, device)
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = _pick_first_key(batch, ("x","input","inputs","image"))
                y = _pick_first_key(batch, ("y","target","targets","label"))
            else:
                raise ValueError("Unsupported batch format in evaluate.")
            if x.ndim == 4: x5 = x.unsqueeze(2)
            else: x5 = x
            y_hat = model(x5)
            # 对齐维度
            if y_hat.ndim == 5 and y.ndim == 4: y_hat_show = y_hat.squeeze(2)
            else: y_hat_show = y_hat
            psnr_val = float(psnr(y_hat_show, y).cpu().item())
            rec = {"psnr": psnr_val}
            fp.write(json.dumps(rec) + "\n"); fp.flush()
            n_batches += 1
            if n_batches >= cfg_eval.get("num_eval_batches", 3):
                break

def _normalize_img(t: torch.Tensor):
    t = t.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]  # [C,H,W] 取第一个通道
    elif t.ndim == 3:
        pass
    elif t.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected ndim {t.ndim}")
    # [C,H,W] 或 [H,W] → [H,W]
    if t.ndim == 3:
        t = t[0]
    vmin = torch.quantile(t.flatten(), 0.01)
    vmax = torch.quantile(t.flatten(), 0.99)
    t = torch.clamp((t - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return t.numpy()

@torch.no_grad()
def plot_triplets(model, test_dl, run_dir: Path, cfg_eval: Dict[str, Any]):
    if plt is None:
        return
    out_dir = ensure_dir(run_dir / "eval_vis")
    device = next(model.parameters()).device
    plotted, batches = 0, 0
    limit_b = cfg_eval.get("num_eval_batches", 3)
    limit_n = cfg_eval.get("num_plot_triplets", 6)

    for batch in test_dl:
        batch = move_batch_to_device(batch, device)
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x = _pick_first_key(batch, ("x","input","inputs","image"))
            y = _pick_first_key(batch, ("y","target","targets","label"))
        else:
            raise ValueError("Unsupported batch format for plotting.")

        if x.ndim == 4: x5 = x.unsqueeze(2)
        else: x5 = x
        model.eval()
        y_hat = model(x5).squeeze(2) if x.ndim == 4 else model(x5)
        # 对齐以 4D 展示
        if y_hat.ndim == 5: y_hat_show = y_hat.squeeze(2)
        else: y_hat_show = y_hat

        b = x.shape[0]
        take = min(b, limit_n - plotted)
        for i in range(take):
            x_img = _normalize_img(x[i] if x.ndim==4 else x[i,:,0])
            y_img = _normalize_img(y_hat_show[i])
            t_img = _normalize_img(y[i] if y.ndim==4 else y[i,:,0])

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12,4))
            for j, (img, title) in enumerate([(x_img,"Input"),(y_img,"Output"),(t_img,"Target")], 1):
                ax = fig.add_subplot(1,3,j)
                ax.imshow(img); ax.set_title(title); ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"triplet_b{batches}_i{i}.png", dpi=150)
            plt.close(fig)
            plotted += 1

        batches += 1
        if batches >= limit_b or plotted >= limit_n:
            break
