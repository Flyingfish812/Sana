# backend/dataio/sampling/registry_hooks.py
from __future__ import annotations
from typing import Dict, Any
from . import build_sampler_from_config
from .base import ListSampler, SampleSpec
from ..registry import register_sampler

@register_sampler("static")
def _build_static_sampler(*, shape5d, **cfg):
    """
    用法： s_cfg = {"kind":"static", "t_stride":1, "n":0}
    """
    cfg2: Dict[str, Any] = {"kind": "static"}
    cfg2.update(cfg)
    return build_sampler_from_config(shape5d, cfg2)

@register_sampler("multi_frame")
def _build_multiframe_sampler(*, shape5d, **cfg):
    """
    用法： s_cfg = {"kind":"multi_frame", "history":3, "target_offset":0, "t_stride":1, "n":0}
    """
    cfg2: Dict[str, Any] = {"kind": "multi_frame"}
    cfg2.update(cfg)
    return build_sampler_from_config(shape5d, cfg2)

from .base import ListSampler, SampleSpec
from ..registry import register_sampler
import inspect

@register_sampler("per_frame")
def build_per_frame_sampler(shape5d, **kwargs):
    """
    逐帧采样：为每个 (n, t) 生成一个样本。
    自适配 SampleSpec 的真实签名，兼容以下常见形态：
      - SampleSpec(n, t)
      - SampleSpec(n, t, history)
      - SampleSpec(n=..., t=[...])
      - SampleSpec(n=..., t=[...], history=1)
      - 少数实现用 index/idx 代替 n，用 time/times 代替 t
    """
    N, T, H, W, C = shape5d

    # 读取 SampleSpec 的参数名，以便自适配
    sig = inspect.signature(SampleSpec)
    names = [p.name for p in sig.parameters.values()
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]

    # 可能的别名
    n_keys = ("n", "index", "idx", "i", "sequence", "sample_index")
    t_keys = ("t", "time", "times", "time_idx", "frames")
    h_keys = ("history", "hist", "K", "k", "history_len")

    # 选出可用的关键字名
    n_key = next((k for k in n_keys if k in names), None)
    t_key = next((k for k in t_keys if k in names), None)
    h_key = next((k for k in h_keys if k in names), None)

    def make_spec(n: int, t: int):
        # 1) 尝试完全命名（含 history）
        if n_key and t_key and h_key:
            return SampleSpec(**{n_key: n, t_key: [t], h_key: 1})
        # 2) 仅用 n/t 命名
        if n_key and t_key:
            return SampleSpec(**{n_key: n, t_key: [t]})
        # 3) 两个位置参数 (n, [t])
        try:
            return SampleSpec(n, [t])
        except TypeError:
            # 4) 三个位置参数 (n, [t], 1) —— 只有在确实需要第三参时才会成功
            return SampleSpec(n, [t], 1)

    specs = [make_spec(n, t) for n in range(N) for t in range(T)]
    return ListSampler(specs)

