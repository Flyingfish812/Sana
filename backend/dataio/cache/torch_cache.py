# backend/dataio/cache/torch_cache.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
import torch
from ..schema import DataMeta

class CacheManager:
    """
    轻量落盘器：
      - save_array5d(): 保存规范化后的全量 Array5D 为 .npy
      - save_batches(): 遍历 DataLoader，按批保存为 batch_000001.pt
      - save_meta_summary(): 保存 meta.json 与 summary.json
    """
    def __init__(self, out_dir: str, overwrite: bool = True):
        self.out_dir = Path(out_dir)
        self.overwrite = overwrite
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _check_overwrite(self, path: Path):
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {path}")

    def save_array5d(self, array5d: np.ndarray, filename: str = "array5d.npy"):
        path = self.out_dir / filename
        self._check_overwrite(path)
        np.save(path, array5d)

    def save_batches(self, dataloader, max_batches: Optional[int] = None, prefix: str = "batch_"):
        """
        将 dataloader 的 batch 逐个保存为 .pt，包含 x/y/cond 的 torch.Tensor。
        """
        for i, batch in enumerate(dataloader, start=1):
            if max_batches is not None and i > max_batches:
                break
            path = self.out_dir / f"{prefix}{i:06d}.pt"
            self._check_overwrite(path)
            torch.save(batch, path)

    def save_meta_summary(self, meta: DataMeta, summary: Dict[str, Any]):
        # meta.json（轻量：仅保存 to_json；必要时可自行扩展保存 coords/mask 的二进制文件）
        meta_json = meta.to_json()
        with open(self.out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_json, f, ensure_ascii=False, indent=2)
        # summary.json
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
