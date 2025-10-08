# backend/model/factory.py
from __future__ import annotations
from typing import Dict, Any, Callable
from importlib import import_module

ENCODER_REGISTRY: Dict[str, Callable[..., Any]] = {}
PROPAGATOR_REGISTRY: Dict[str, Callable[..., Any]] = {}
DECODER_REGISTRY: Dict[str, Callable[..., Any]] = {}
HEAD_REGISTRY: Dict[str, Callable[..., Any]] = {}

REGISTRY_TYPES = {
    "encoder": ENCODER_REGISTRY,
    "propagator": PROPAGATOR_REGISTRY,
    "decoder": DECODER_REGISTRY,
    "head": HEAD_REGISTRY,
}

# >>> 新增：内置组件模块清单（至少把最小可跑的那几个放进来）
_BUILTIN_MODULES = [
    "backend.model.encoders.unet_base",
    "backend.model.propagators.identity",
    "backend.model.decoders.unet_decoder",
    "backend.model.heads.pixel_head",
]
_BUILTINS_LOADED = False

def _load_builtins_once():
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for mod in _BUILTIN_MODULES:
        import_module(mod)
    _BUILTINS_LOADED = True

def register(kind: str, name: str):
    def deco(fn):
        REGISTRY_TYPES[kind][name.lower()] = fn
        return fn
    return deco

def build_component(kind: str, cfg: Dict[str, Any]):
    # 确保内置组件已注册
    _load_builtins_once()

    name = cfg.get("name")
    args = cfg.get("args", {}) or {}
    if not name:
        raise ValueError(f"{kind} config missing 'name'")

    key = name.lower()
    if key in REGISTRY_TYPES[kind]:
        return REGISTRY_TYPES[kind][key](**args)

    # 备用：支持 "pkg.module:ClassName" 直指类名的加载
    if ":" in name:
        module_path, cls_name = name.split(":")
        mod = import_module(module_path)
        cls = getattr(mod, cls_name)
        return cls(**args)

    raise KeyError(f"Unknown {kind} name: {name}. Registered: {list(REGISTRY_TYPES[kind].keys())}")
