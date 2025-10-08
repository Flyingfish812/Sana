# backend/dataio/io/importers.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader
from ..cache.snapshot import load_snapshot_as_dataloader

def load_dataloader_snapshot(
    dir_path: str,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    max_ram_gb: Optional[float] = None,
    force_streaming: bool = False,
    shuffle: bool = False,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    一行加载：
      from backend.dataio.io.importers import load_dataloader_snapshot
      dl, info = load_dataloader_snapshot("./prep_out/xxx", batch_size=64, max_ram_gb=16)
    若给定 max_ram_gb 且足够，则整载入内存（训练时几乎无额外 IO）；否则自动流式读取。
    """
    return load_snapshot_as_dataloader(
        dir_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_ram_gb=max_ram_gb,
        force_streaming=force_streaming,
        shuffle=shuffle,
    )
