# backend/train/data_adapter_snapshot.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import glob
import json
import torch
import gzip
import io
from torch.utils.data import DataLoader, TensorDataset

def _load_split_loader(split_dir: Path, loader_cfg: Dict) -> Optional[DataLoader]:
    """
    加载单个 split (train/val/test) 的 DataLoader。
    优先顺序：
      (1) snapshot_index.json  → 调用数据层读取器（标准恢复）；
      (2) 兜底：目录下存在 dl.part*.pt[.gz] 但没有 snapshot_index.json → 手工拼接成 TensorDataset；
      (3) 训练侧 fallback：{split}_dataset.pt → 直接还原成 DataLoader。
    """
    # -------- (1) 标准：snapshot_index.json --------
    snap_idx = split_dir / "snapshot_index.json"
    if snap_idx.exists():
        # 直接复用数据层快照读取器（这是 DataloaderSnapshotWriter 的正向配套读取）
        from backend.dataio.cache.snapshot import load_snapshot_as_dataloader
        dl, _ = load_snapshot_as_dataloader(
            str(split_dir),
            batch_size=loader_cfg.get("batch_size", 8),
            num_workers=loader_cfg.get("num_workers", 4),
            shuffle=("train" in split_dir.name.lower()),
            pin_memory=loader_cfg.get("pin_memory", True),
            persistent_workers=loader_cfg.get("persistent_workers", False),
        )
        return dl

    # -------- (2) 兜底：没有 snapshot_index.json，但有分片文件 → 手工拼接 --------
    # 同时匹配 .pt 和 .pt.gz；按文件名排序确保顺序一致
    part_files: List[str] = sorted(
        glob.glob(str(split_dir / "dl.part*.pt")) +
        glob.glob(str(split_dir / "dl.part*.pt.gz"))
    )
    if part_files:
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        cs: List[torch.Tensor] = []
        for fp in part_files:
            if fp.endswith(".gz"):
                with gzip.open(fp, "rb") as gz:
                    payload = torch.load(io.BytesIO(gz.read()), map_location="cpu")
            else:
                payload = torch.load(fp, map_location="cpu")
            # 分片约定是 dict，键来自 {"x","y","cond"}（与写入器约定一致）
            x, y, c = payload.get("x"), payload.get("y"), payload.get("cond")
            if isinstance(x, torch.Tensor): xs.append(x)
            if isinstance(y, torch.Tensor): ys.append(y)
            if isinstance(c, torch.Tensor): cs.append(c)

        tensors: List[torch.Tensor] = []
        if xs: tensors.append(torch.cat(xs, dim=0))
        if ys: tensors.append(torch.cat(ys, dim=0))
        if cs: tensors.append(torch.cat(cs, dim=0))
        if not tensors:
            return None  # 没有任何张量就返回空，交由上层报错

        ds = TensorDataset(*tensors)
        return DataLoader(
            ds,
            batch_size=loader_cfg.get("batch_size", 8),
            num_workers=loader_cfg.get("num_workers", 4),
            pin_memory=loader_cfg.get("pin_memory", True),
            persistent_workers=loader_cfg.get("persistent_workers", False),
            shuffle=("train" in split_dir.name.lower()),
        )

    # -------- (3) 训练侧 fallback：{split}_dataset.pt --------
    ds_pt = split_dir / f"{split_dir.name}_dataset.pt"
    if ds_pt.exists():
        obj = torch.load(ds_pt, map_location="cpu")
        if isinstance(obj, dict) and obj.get("__type__") == "TensorDataset":
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
