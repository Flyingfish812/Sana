from __future__ import annotations
from typing import Callable, List, Dict, Any
import numpy as np
import torch
from ..schema import AdapterOutput

def make_collate(adapter_fn: Callable):
    """
    将 Dataset 的 ArraySample 先过 adapter_fn，再把 numpy 转成 torch 并拼 batch。
    支持三种 cond 形态：
      1) None
      2) 张量 (np/torch)：将被 stack 成 [B,...]
      3) dict {"mask": np/torch[1,H,W], "points": np[int M,2]}：
         - "mask" 会被 stack 成 [B,1,H,W] 放入 batch["cond"]
         - "points" 会作为 list[torch.LongTensor[M_i,2]] 挂在 batch["sampling_points"]
    """
    def collate(batch_samples):
        outs: List[AdapterOutput] = [adapter_fn(s) for s in batch_samples]

        def to_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x
            return torch.from_numpy(x).float()

        xs = [to_tensor(o.x) for o in outs]
        ys = [to_tensor(o.y) for o in outs]

        # 处理 cond
        cond_kind = None
        cond_masks: List[torch.Tensor] = []
        cond_points: List[torch.Tensor] = []
        for o in outs:
            c = o.cond
            if c is None:
                continue
            if isinstance(c, dict):
                cond_kind = "dict"
                # mask
                m = c.get("mask")
                if m is not None:
                    cond_masks.append(to_tensor(m))
                else:
                    cond_masks.append(None)  # 对齐 batch 长度
                # points（保持 int64）
                p = c.get("points")
                if p is not None:
                    if isinstance(p, torch.Tensor):
                        cond_points.append(p.to(dtype=torch.long))
                    else:
                        cond_points.append(torch.as_tensor(p, dtype=torch.long))
                else:
                    cond_points.append(torch.zeros((0,2), dtype=torch.long))
            else:
                cond_kind = "tensor"

        x = torch.stack(xs, dim=0) if xs[0] is not None else None
        y = torch.stack(ys, dim=0) if ys[0] is not None else None

        batch: Dict[str, Any] = {"x": x, "y": y, "meta": outs[0].meta}

        if cond_kind is None:
            batch["cond"] = None
        elif cond_kind == "tensor":
            cts = [to_tensor(o.cond) for o in outs]
            batch["cond"] = torch.stack(cts, dim=0)
        else:  # dict
            # stack masks（允许某些样本无 mask：用 zeros_like 对齐）
            if any(m is not None for m in cond_masks):
                tpl = next(m for m in cond_masks if m is not None)
                masks = [m if m is not None else torch.zeros_like(tpl) for m in cond_masks]
                batch["cond"] = torch.stack(masks, dim=0)  # [B,1,H,W]
            else:
                batch["cond"] = None
            # points 作为 list[tensor]，不参与 snapshot 的数值堆叠
            batch["sampling_points"] = cond_points

        return batch
    return collate
