# backend/dataio/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any
import functools

# 轻量静态注册表：避免“插件系统”复杂度，但仍保留“可扩展入口”
_READERS: Dict[str, Callable[..., Any]] = {}
_TRANSFORMS: Dict[str, Callable[..., Any]] = {}
_SAMPLERS: Dict[str, Callable[..., Any]] = {}
_ADAPTERS: Dict[str, Callable[..., Any]] = {}

def register_reader(name: str):
    def deco(cls_or_fn):
        _READERS[name] = cls_or_fn
        return cls_or_fn
    return deco

def register_transform(name: str):
    def deco(cls_or_fn):
        _TRANSFORMS[name] = cls_or_fn
        return cls_or_fn
    return deco

def register_sampler(name: str):
    def deco(fn):
        _SAMPLERS[name] = fn
        return fn
    return deco

def register_adapter(name: str):
    def deco(fn):
        _ADAPTERS[name] = fn
        return fn
    return deco

def build_reader(kind: str, **kwargs):
    if kind not in _READERS:
        raise KeyError(f"Reader '{kind}' not registered. Known: {list(_READERS)}")
    return _READERS[kind](**kwargs)

def build_transform(kind: str, **kwargs):
    if kind not in _TRANSFORMS:
        raise KeyError(f"Transform '{kind}' not registered. Known: {list(_TRANSFORMS)}")
    return _TRANSFORMS[kind](**kwargs)

def build_sampler(kind: str, **kwargs):
    if kind not in _SAMPLERS:
        raise KeyError(f"Sampler '{kind}' not registered. Known: {list(_SAMPLERS)}")
    return _SAMPLERS[kind](**kwargs)

def build_adapter(kind: str, **kwargs):
    """
    返回一个可调用的 adapter_fn: (sample: ArraySample) -> AdapterOutput
    对于函数式适配器（如 Sparse2DAdapter），这里用 partial 绑定除 sample 外的参数；
    对于类/可调用对象也同样适用（若已是无参可调用，则直接返回）。
    """
    if kind not in _ADAPTERS:
        raise KeyError(f"Adapter '{kind}' not registered. Known: {list(_ADAPTERS)}")
    fn = _ADAPTERS[kind]
    if kwargs:
        return functools.partial(fn, **kwargs)
    return fn
