# backend/dataio/sampling/__init__.py
from __future__ import annotations
from typing import Dict, Any, List
from .base import SampleSpec, ListSampler

def build_sampler_from_config(shape5d, cfg: Dict[str, Any]):
    """
    最小化的采样器工厂：支持两种模式
      - "static": 每个样本取单帧 t（如 t in [t0..tK)）
      - "multi_frame": 每个样本取若干历史帧 + 1 目标帧
    cfg:
      kind: "static"|"multi_frame"
      history: int, target_offset: int（仅 multi_frame 用）
      t_stride: int 取样步长，默认 1
      n: 固定为 0（单 N），或遍历所有 N（可扩展）
    """
    N, T, H, W, C = shape5d
    kind = cfg.get("kind", "static")
    t_stride = int(cfg.get("t_stride", 1))
    n_choice = int(cfg.get("n", 0))
    specs: List[SampleSpec] = []

    if kind == "static":
        for t in range(0, T, t_stride):
            specs.append(SampleSpec(n=n_choice, t_indices=[t]))
    elif kind == "multi_frame":
        hist = int(cfg.get("history", 3))
        tgt_off = int(cfg.get("target_offset", 0))  # 0 表示最后一帧为目标
        win = hist + 1
        for t in range(hist, T, t_stride):
            t_hist = list(range(t - hist, t + 1))    # [t-hist, ..., t]
            # 允许将目标定义为最后一帧或再偏移
            if tgt_off != 0:
                t_hist = [ti + tgt_off for ti in t_hist]
                if t_hist[-1] >= T or t_hist[0] < 0:
                    continue
            specs.append(SampleSpec(n=n_choice, t_indices=t_hist))
    else:
        raise ValueError(f"Unknown sampler kind: {kind}")

    return ListSampler(specs)
