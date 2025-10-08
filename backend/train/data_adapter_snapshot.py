# backend/train/data_adapter_snapshot.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
from pathlib import Path
import glob
import json
import torch
from torch.utils.data import DataLoader

def _load_split_loader(split_dir: Path, loader_cfg: Dict) -> Optional[DataLoader]:
    # 1) 你的 dataio 原生导出的 DataLoader 片段
    pt_list = sorted(glob.glob(str(split_dir / "dl.part*.pt")))
    if pt_list:
        obj = torch.load(pt_list[0], map_location="cpu")
        if isinstance(obj, DataLoader):
            return obj
        try:
            from torch.utils.data import Dataset
            if isinstance(obj, Dataset):
                return DataLoader(
                    obj,
                    batch_size=loader_cfg.get("batch_size", 8),
                    num_workers=loader_cfg.get("num_workers", 4),
                    pin_memory=loader_cfg.get("pin_memory", True),
                    persistent_workers=loader_cfg.get("persistent_workers", False),
                    shuffle=("train" in split_dir.name.lower()),
                )
        except Exception:
            pass

    # 2) 我们训练侧保存的 dataset.pt
    ds_pt = split_dir / f"{split_dir.name}_dataset.pt"
    if ds_pt.exists():
        obj = torch.load(ds_pt, map_location="cpu")
        # 兼容保存的两种形态
        if isinstance(obj, dict) and obj.get("__type__") == "TensorDataset":
            from torch.utils.data import TensorDataset
            obj = TensorDataset(*obj["tensors"])
        return DataLoader(
            obj,
            batch_size=loader_cfg.get("batch_size", 8),
            num_workers=loader_cfg.get("num_workers", 4),
            pin_memory=loader_cfg.get("pin_memory", True),
            persistent_workers=loader_cfg.get("persistent_workers", False),
            shuffle=("train" in split_dir.name.lower()),
        )
    return None

def _detect_root(snapshot_dir: Path) -> Path:
    """
    支持传入：prep_out/h5_sparse 或者 直接 train/val/test 之上的根目录。
    只要该目录下有 train 和 test 子目录即可。
    """
    if (snapshot_dir / "train").exists() and (snapshot_dir / "test").exists():
        return snapshot_dir
    # 允许传到更高层：如果下级只有一个子目录包含 train/test，则下探一层
    subs = [p for p in snapshot_dir.iterdir() if p.is_dir()]
    for sub in subs:
        if (sub / "train").exists() and (sub / "test").exists():
            return sub
    raise FileNotFoundError(f"未在 {snapshot_dir} 下找到 train/test 子目录。")

def build_from_snapshot(snapshot_dir: str | Path, loader_cfg: Dict) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    读取形如：
      prep_out/
        h5_sparse/
          train/ dl.part00001.pt ...
          val/   dl.part00001.pt ...
          test/  dl.part00001.pt ...
    """
    root = _detect_root(Path(snapshot_dir))
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    train_dl = _load_split_loader(train_dir, loader_cfg)
    val_dl = _load_split_loader(val_dir, loader_cfg) if val_dir.exists() else None
    test_dl = _load_split_loader(test_dir, loader_cfg)

    if train_dl is None or test_dl is None:
        raise FileNotFoundError(
            f"在 {root} 未找到可用的 dl.part*.pt（train/test 必须存在）。"
        )
    return train_dl, val_dl, test_dl
