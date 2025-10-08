# backend/train/data_adapter.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
from pathlib import Path
from torch.utils.data import DataLoader
from importlib import import_module
import json
import torch
import warnings
import inspect
from torch.utils.data import TensorDataset

from .data_adapter_snapshot import build_from_snapshot  # 支持 prep_out/... 结构

def _with_loader_opts(dl: DataLoader, cfg: Dict) -> DataLoader:
    if dl is None: 
        return None
    opts = dict(
        batch_size=cfg.get("batch_size", dl.batch_size),
        num_workers=cfg.get("num_workers", dl.num_workers),
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", False),
        shuffle=getattr(dl, "shuffle", False),
    )
    return DataLoader(dl.dataset, shuffle=opts.pop("shuffle"), collate_fn=dl.collate_fn, **opts)

def _build_via_builder(builder: str, builder_args: Dict) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    module, fn = builder.split(":")
    func = getattr(import_module(module), fn)
    out = func(**(builder_args or {}))
    if isinstance(out, dict):
        return out.get("train"), out.get("val"), out.get("test")
    return out

def _resolve_from_run_dir(run_dir: str | Path) -> Tuple[Optional[str], Optional[Tuple[DataLoader, Optional[DataLoader], DataLoader]]]:
    """
    解析 runs/<exp>/<ver>/ 下的数据来源：
      1) dataloaders/*.pt 直接反序列化
      2) data_ref.json 里的 snapshot_dir
      3) data_snapshot/ 目录作为 snapshot_dir
    """
    run_dir = Path(run_dir)
    # 1) 直接 pickled dataloaders
    dl_dir = run_dir / "dataloaders"
    t_p, v_p, te_p = dl_dir / "train_dl.pt", dl_dir / "val_dl.pt", dl_dir / "test_dl.pt"
    if t_p.exists() and te_p.exists():
        train_dl = torch.load(t_p, map_location="cpu")
        val_dl = torch.load(v_p, map_location="cpu") if v_p.exists() else None
        test_dl = torch.load(te_p, map_location="cpu")
        return None, (train_dl, val_dl, test_dl)

    # 2) data_ref.json 声明 snapshot_dir
    ref_p = run_dir / "data_ref.json"
    if ref_p.exists():
        ref = json.loads(ref_p.read_text(encoding="utf-8"))
        snap = ref.get("snapshot_dir")
        if snap:
            return snap, None

    # 3) 备选：data_snapshot/
    snap_dir = run_dir / "data_snapshot"
    if snap_dir.exists():
        return str(snap_dir), None

    return None, None

def _save_dataset_safe(ds, path: Path) -> bool:
    """尽量把 Dataset 存成一个 pt；若不可序列化则返回 False。"""
    try:
        # 常见：TensorDataset、简单自定义（含 tensors/numpy）
        torch.save(ds, path)
        return True
    except Exception:
        # 再尝试只保存 tensors 字段（常见于 TensorDataset）
        try:
            tensors = getattr(ds, "tensors", None)
            if tensors is not None:
                torch.save({"__type__": "TensorDataset", "tensors": tensors}, path)
                return True
        except Exception:
            pass
    return False

def _save_loader_meta(dl: DataLoader, path: Path, tag: str):
    meta = {
        "tag": tag,
        "batch_size": getattr(dl, "batch_size", None),
        "shuffle": getattr(dl, "shuffle", None),
        "num_workers": getattr(dl, "num_workers", None),
        "pin_memory": getattr(dl, "pin_memory", None),
        "persistent_workers": getattr(dl, "persistent_workers", None),
        "sampler": dl.sampler.__class__.__name__ if hasattr(dl, "sampler") else None,
        "collate_fn": None,
    }
    cf = getattr(dl, "collate_fn", None)
    if cf is not None:
        # 只记录可读信息，不去保存函数本体
        try:
            meta["collate_fn"] = getattr(cf, "__name__", repr(cf))
        except Exception:
            meta["collate_fn"] = str(cf)
    (path / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def maybe_save_dataloaders(train_dl: DataLoader, val_dl: Optional[DataLoader], test_dl: DataLoader,
                           data_cfg: Dict, run_dir: Path):
    """不再直接 torch.save(DataLoader)；改为保存 dataset.pt + 元信息。"""
    if not data_cfg.get("save_dataloaders", False):
        return
    out = run_dir / "dataloaders"
    out.mkdir(parents=True, exist_ok=True)

    for tag, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl)):
        if dl is None:
            continue
        ok = _save_dataset_safe(dl.dataset, out / f"{tag}_dataset.pt")
        _save_loader_meta(dl, out, tag)
        if not ok:
            warnings.warn(
                f"[dataloader snapshot] {tag}: dataset not picklable; "
                f"saved only meta. Consider using snapshot_dir or builder to reproduce."
            )
    # 再写一个 README 提示如何恢复
    (out / "README.txt").write_text(
        "This folder stores dataset snapshots (if picklable) and loader meta.\n"
        "Dataloaders are NOT pickled to avoid unpicklable local collate functions.\n",
        encoding="utf-8"
    )

def build_dataloaders(
    data_cfg: Dict,
    injected: Optional[Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]] = None
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    统一数据入口：
      - 优先使用 injected（三个 DataLoader 任意一个非 None 即启用注入模式）；
      - 否则按 config.data：from_run_dir → builder → snapshot_dir。
    """
    # A) 显式注入（Notebook 里传入）
    if injected and any(d is not None for d in injected):
        train_dl, val_dl, test_dl = injected
        assert train_dl is not None and test_dl is not None, "Injected mode requires train_dl and test_dl."
        return train_dl, val_dl, test_dl

    # B) from_run_dir
    if data_cfg.get("from_run_dir"):
        snap, dls = _resolve_from_run_dir(data_cfg["from_run_dir"])
        if dls is not None:
            return dls
        if snap:
            train_dl, val_dl, test_dl = build_from_snapshot(snap, data_cfg)
            train_dl = _with_loader_opts(train_dl, data_cfg)
            val_dl = _with_loader_opts(val_dl, data_cfg) if val_dl is not None else None
            test_dl = _with_loader_opts(test_dl, data_cfg)
            return train_dl, val_dl, test_dl
        raise FileNotFoundError(f"from_run_dir={data_cfg['from_run_dir']} 未找到 dataloaders 或 snapshot 线索。")

    # C) builder（推荐）
    if data_cfg.get("builder"):
        train_dl, val_dl, test_dl = _build_via_builder(data_cfg["builder"], data_cfg.get("builder_args", {}))
        train_dl = _with_loader_opts(train_dl, data_cfg)
        val_dl = _with_loader_opts(val_dl, data_cfg) if val_dl is not None else None
        test_dl = _with_loader_opts(test_dl, data_cfg)
        return train_dl, val_dl, test_dl

    # D) snapshot_dir（支持 prep_out/… 的 train|val|test 子目录结构）
    snap = data_cfg.get("snapshot_dir")
    if snap:
        train_dl, val_dl, test_dl = build_from_snapshot(snap, data_cfg)
        train_dl = _with_loader_opts(train_dl, data_cfg)
        val_dl = _with_loader_opts(val_dl, data_cfg) if val_dl is not None else None
        test_dl = _with_loader_opts(test_dl, data_cfg)
        return train_dl, val_dl, test_dl

    raise ValueError("请在 data 中提供 from_run_dir 或 builder（推荐）或 snapshot_dir，或在调用处显式注入 dataloaders。")
