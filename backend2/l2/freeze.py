from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifact_io import ArtifactManager
from .data import PairDataset, load_l1_array_mmap, load_split_pairs
from .model_factory import build_l2_model
from .probe import ProbeCallback, ProbeController
from .utils import dump_json, iter_progress, log_progress, now_iso, read_json


DEFAULT_HOOK_LAYERS: List[str] = [
    "enc.stage1.out",
    "enc.stage2.out",
    "enc.stage3.out",
    "enc.stage4.out",
    "enc.stage5.out",
    "dec.stage1.out",
    "dec.stage2.out",
    "dec.stage3.out",
    "dec.stage4.out",
    "dec.stage5.out",
    "skip.s4",
    "skip.s3",
    "skip.s2",
    "skip.s1",
    "head.out",
]


class FeatureFreezeCollector:
    """Probe 回调：按层收集中间特征并保持 batch 顺序。"""

    def __init__(self, layer_patterns: Sequence[str]):
        self.layer_patterns = [str(v) for v in layer_patterns if str(v).strip()]
        self._batches: Dict[str, List[np.ndarray]] = {}

    def _match(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in self.layer_patterns)

    def __call__(self, name: str, tensor: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        if self._match(name):
            arr = tensor.detach().to(dtype=torch.float32).cpu().numpy()
            self._batches.setdefault(name, []).append(arr)
        return tensor

    def as_arrays(self) -> Dict[str, np.ndarray]:
        outputs: Dict[str, np.ndarray] = {}
        for name, chunks in self._batches.items():
            if not chunks:
                continue
            outputs[name] = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        return outputs


def _device_of(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in items:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def resolve_freeze_layers(config: Dict[str, Any]) -> List[str]:
    """解析待冻结层：优先 freeze_layers，其次 probe.hook_layers，再回退到默认全层。"""
    freeze_layers = config.get("freeze_layers")
    if freeze_layers:
        return _dedupe_keep_order(str(v) for v in freeze_layers if str(v).strip())

    probe_cfg = dict(config.get("probe") or {})
    hook_layers = probe_cfg.get("hook_layers")
    if hook_layers:
        return _dedupe_keep_order(str(v) for v in hook_layers if str(v).strip())

    return list(DEFAULT_HOOK_LAYERS)


def build_probe_config_for_freeze(config: Dict[str, Any], freeze_layers: Sequence[str]) -> Dict[str, Any]:
    """构造冻结场景专用 probe 配置：默认开启并保证可采集目标层。"""
    probe_cfg = dict(config.get("probe") or {})
    hook_layers = list(probe_cfg.get("hook_layers") or [])
    hook_layers.extend(freeze_layers)

    probe_cfg["enabled"] = True
    probe_cfg["record_level"] = 1
    probe_cfg["allow_full_dump"] = True
    probe_cfg["hook_layers"] = _dedupe_keep_order(str(v) for v in hook_layers if str(v).strip())
    return probe_cfg


def save_frozen_features(
    manager: ArtifactManager,
    dataset_id: str,
    exp_name: str,
    run_name: str,
    layer_features: Dict[str, np.ndarray],
    pair_nt: np.ndarray,
) -> Dict[str, Any]:
    """将按层特征写入 L2.5 目录并生成 manifest。"""
    log_enabled = True
    use_tqdm = True
    if pair_nt.ndim != 2 or pair_nt.shape[1] != 2:
        raise ValueError(f"pair_nt must have shape [N,2], got {pair_nt.shape}")

    manager.freeze_layers_dir.mkdir(parents=True, exist_ok=True)
    log_progress(log_enabled, "L2.5-FREEZE", f"start writing layers={len(layer_features)}, samples={pair_nt.shape[0]}")

    shape_per_layer: Dict[str, List[int]] = {}
    layer_items = list(layer_features.items())
    layer_iter = iter_progress(
        layer_items,
        enabled=log_enabled,
        use_tqdm=use_tqdm,
        desc="[L2.5-FREEZE] save layers",
        total=len(layer_items),
        leave=False,
    )
    for layer_name, feats in layer_iter:
        if feats.ndim != 4:
            raise ValueError(f"layer '{layer_name}' must be [N,C,H,W], got {feats.shape}")
        if int(feats.shape[0]) != int(pair_nt.shape[0]):
            raise ValueError(
                f"sample size mismatch for layer '{layer_name}': features={feats.shape[0]} vs pair_nt={pair_nt.shape[0]}"
            )
        out_path = manager.freeze_layers_dir / f"{layer_name}.npz"
        np.savez(
            out_path,
            features=feats.astype(np.float32, copy=False),
            pair_nt=pair_nt.astype(np.int64, copy=False),
        )
        shape_per_layer[layer_name] = [int(v) for v in feats.shape]

    manifest = {
        "dataset_id": str(dataset_id),
        "exp_name": str(exp_name),
        "run_name": str(run_name),
        "layers": sorted(layer_features.keys()),
        "num_samples": int(pair_nt.shape[0]),
        "shape_per_layer": shape_per_layer,
        "created_at": now_iso(),
    }
    dump_json(manager.freeze_manifest_json, manifest)
    dump_json(manager.freeze_dir / "freeze_manifest.json", manifest)
    log_progress(log_enabled, "L2.5-FREEZE", f"manifest saved: {manager.freeze_manifest_json}")
    return {
        "freeze_dir": str(manager.freeze_dir),
        "freeze_manifest": str(manager.freeze_manifest_json),
        "freeze_manifest_legacy": str(manager.freeze_dir / "freeze_manifest.json"),
        "num_layers": int(len(layer_features)),
        "num_samples": int(pair_nt.shape[0]),
    }


def load_frozen_features(manager: ArtifactManager, split: str = "test") -> Dict[str, Any]:
    """读取 freeze 目录中的按层特征。"""
    log_progress(True, "L2.5-FREEZE", f"load frozen features: run={manager.run_name}, split={split}")
    if str(split) != "test":
        raise ValueError("freeze_mode currently supports split='test' only")

    if not manager.freeze_manifest_json.exists():
        raise FileNotFoundError(f"freeze manifest not found: {manager.freeze_manifest_json}")

    manifest = read_json(manager.freeze_manifest_json)
    layers = list(manifest.get("layers") or [])
    if not layers:
        raise ValueError("freeze manifest contains empty layers")

    layer_arrays: Dict[str, np.ndarray] = {}
    pair_nt: Optional[np.ndarray] = None
    layer_iter = iter_progress(
        layers,
        enabled=True,
        use_tqdm=True,
        desc="[L2.5-FREEZE] load layers",
        total=len(layers),
        leave=False,
    )
    for layer_name in layer_iter:
        layer_path = manager.freeze_layers_dir / f"{layer_name}.npz"
        if not layer_path.exists():
            raise FileNotFoundError(f"freeze layer file not found: {layer_path}")
        with np.load(layer_path) as data:
            feats = data["features"].astype(np.float32, copy=False)
            nt = data["pair_nt"].astype(np.int64, copy=False)
        layer_arrays[str(layer_name)] = feats
        pair_nt = nt if pair_nt is None else pair_nt

    if pair_nt is None:
        raise ValueError("no pair_nt found in freeze files")

    return {
        "mode": "freeze",
        "layers": layer_arrays,
        "pair_nt": pair_nt,
    }


def _build_model_for_online(
    in_channels: int,
    out_channels: int,
    config: Dict[str, Any],
    manager: ArtifactManager,
) -> torch.nn.Module:
    device = _device_of(config)
    model = build_l2_model(
        config,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
    ).to(device)

    ckpt_name = str(config.get("ckpt_name", "model_best.pt"))
    ckpt_path = Path(config.get("ckpt_path", manager.ckpt_path(ckpt_name)))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def extract_features_online(
    manager: ArtifactManager,
    config: Dict[str, Any],
    split: str = "test",
) -> Dict[str, Any]:
    """在线前向提取特征，不保存 preds/metrics。"""
    t0_all = time.perf_counter()
    log_enabled = bool(config.get("log_progress", True))
    use_tqdm = bool(config.get("tqdm", True))
    log_progress(log_enabled, "L2.5-ONLINE", f"start online extraction: dataset={manager.dataset_id}, run={manager.run_name}, split={split}")
    if str(split) != "test":
        raise ValueError("freeze_mode currently supports split='test' only")

    target_offset = int(config.get("target_offset", 1))
    array5d, manifest = load_l1_array_mmap(manager)
    pairs = load_split_pairs(manager, array5d, manifest, split, target_offset)
    if len(pairs) == 0:
        raise ValueError(f"empty {split} pairs from L1 split")

    dataset = PairDataset(
        array5d=array5d,
        pairs=pairs,
        target_offset=target_offset,
        sparse_input=dict(config.get("sparse_input") or {}),
        dataset_id=str(config.get("dataset_id", manager.dataset_id)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    log_progress(log_enabled, "L2.5-ONLINE", f"pairs={len(pairs)}, steps={len(loader)}, batch_size={int(config.get('batch_size', 8))}")

    freeze_layers = resolve_freeze_layers(config)
    if not freeze_layers:
        raise ValueError("freeze_layers resolved to empty list")

    probe_cfg = build_probe_config_for_freeze(config, freeze_layers)
    collector = FeatureFreezeCollector(layer_patterns=freeze_layers)
    probe = ProbeController(probe_cfg, callbacks=[collector])

    sample0 = dataset[0]
    model = _build_model_for_online(
        in_channels=int(sample0["x"].shape[0]),
        out_channels=int(sample0["y"].shape[0]),
        config=config,
        manager=manager,
    )
    device = _device_of(config)
    log_progress(log_enabled, "L2.5-ONLINE", f"model loaded on {device}")

    nts = []
    with torch.no_grad():
        forward_iter = iter_progress(
            loader,
            enabled=log_enabled,
            use_tqdm=use_tqdm,
            desc=f"[L2.5-ONLINE][{manager.dataset_id}] forward",
            total=len(loader),
            leave=False,
        )
        for batch in forward_iter:
            x = batch["x"].to(device)
            _ = model(x, probe=probe)
            nts.append(torch.stack([batch["n"], batch["t"]], dim=1).cpu().numpy())

    pair_nt = np.concatenate(nts, axis=0).astype(np.int64, copy=False)
    layers = collector.as_arrays()
    if not layers:
        raise ValueError("no frozen features captured online; check freeze_layers/probe hooks")

    log_progress(
        log_enabled,
        "L2.5-ONLINE",
        f"finished online extraction: layers={list(layers.keys())}, samples={pair_nt.shape[0]}, dt={time.perf_counter()-t0_all:.2f}s",
    )

    return {
        "mode": "online",
        "layers": layers,
        "pair_nt": pair_nt,
    }


def _fallback_infer_config(manager: ArtifactManager) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "dataset_id": manager.dataset_id,
        "artifacts_dir": str(manager.artifacts_root),
        "exp_name": manager.exp_name,
        "run_name": manager.run_name,
        "device": "auto",
        "target_offset": 1,
        "batch_size": 8,
        "num_workers": 0,
        "ckpt_name": "model_best.pt",
        "freeze_features": True,
        "freeze_mode": "test",
    }

    if manager.infer_config_json.exists():
        cfg.update(read_json(manager.infer_config_json))

    ckpt_default = manager.ckpt_path("model_best.pt")
    if ckpt_default.exists():
        device = _device_of(cfg)
        try:
            state = torch.load(ckpt_default, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(ckpt_default, map_location=device)
        if isinstance(state, dict):
            train_cfg = dict(state.get("config") or {})
            if "model" in train_cfg and "model" not in cfg:
                cfg["model"] = train_cfg["model"]
    return cfg


def load_l2_features_or_fallback(manager: ArtifactManager, split: str = "test") -> Dict[str, Any]:
    """L3 入口：优先加载 L2.5 冻结特征，不存在则在线回退提取。"""
    if manager.freeze_manifest_json.exists():
        log_progress(True, "L2.5-ENTRY", f"mode=freeze, run={manager.run_name}")
        return load_frozen_features(manager=manager, split=split)

    log_progress(True, "L2.5-ENTRY", f"mode=online fallback, run={manager.run_name}")
    cfg = _fallback_infer_config(manager)
    cfg["freeze_mode"] = str(split)
    return extract_features_online(manager=manager, config=cfg, split=split)
