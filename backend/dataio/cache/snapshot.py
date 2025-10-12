# backend/dataio/cache/snapshot.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Tuple
import json
import math
import io
import gzip
import warnings
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

def _safe_torch_load(obj, *, map_location="cpu"):
    try:
        # 新版 PyTorch：优先走安全模式
        return torch.load(obj, map_location=map_location, weights_only=True)
    except TypeError:
        # 旧版 PyTorch：没有 weights_only 参数，回退
        return torch.load(obj, map_location=map_location)

# --------------------
# 工具：张量体积估算
# --------------------
def _tensor_nbytes(x: Optional[torch.Tensor]) -> int:
    if x is None:
        return 0
    return x.numel() * x.element_size()

def _batch_nbytes(b: Dict[str, Any]) -> int:
    # 仅统计 x/y/cond；meta 不入分片（写入 index 里）
    n = 0
    for k in ("x", "y", "cond"):
        v = b.get(k)
        if isinstance(v, torch.Tensor):
            n += _tensor_nbytes(v)
    return n

# --------------------
# 索引结构
# --------------------
@dataclass
class SnapshotIndex:
    version: int
    parts: List[Dict[str, Any]]   # [{"file":"dl.part00001.pt[.gz]","num_samples":M}, ...]
    total_samples: int
    total_nbytes_uncompressed: int
    keys: List[str]               # present keys, subset of ["x","y","cond"]
    batch_first_shapes: Dict[str, List[int]]  # 每样本 shape（不含 batch 维），用于校验/构造内存数据集
    meta: Dict[str, Any]          # 轻量 meta（来自 DataMeta.to_json()）
    compressed: bool              # 是否使用 gzip 压缩

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "parts": self.parts,
            "total_samples": self.total_samples,
            "total_nbytes_uncompressed": self.total_nbytes_uncompressed,
            "keys": self.keys,
            "batch_first_shapes": self.batch_first_shapes,
            "meta": self.meta,
            "compressed": self.compressed,
        }

# --------------------
# 写入器：把 DataLoader 存成若干 ~1GB 的分片
# --------------------
class DataloaderSnapshotWriter:
    """
    将一个 collate 后的 DataLoader 写为快照：
      - 以近似 target_part_bytes 分片（默认 1GB）
      - 索引文件 snapshot_index.json 记录分片与统计信息
      - 可选 gzip 压缩（.pt.gz），牺牲少量 CPU 换更小体积
    约定：batch 是 dict{"x","y","cond","meta"} 且 x/y/cond 为张量（或 None）；同批次 shape 一致。
    """
    def __init__(
        self,
        out_dir: str,
        target_part_bytes: int = 1 << 30,  # ~1GB
        overwrite: bool = True,
        gzip_compress: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        self.target_part_bytes = max(64 << 20, target_part_bytes)  # 不小于 64MB
        self.gzip = gzip_compress

        self._parts: List[Dict[str, Any]] = []
        self._buffers: Dict[str, List[torch.Tensor]] = {"x": [], "y": [], "cond": []}
        self._buf_bytes: int = 0
        self._total_bytes: int = 0
        self._total_samples: int = 0
        self._part_idx: int = 1
        self._keys_present: List[str] = []
        self._batch_first_shapes: Dict[str, List[int]] = {}
        self._meta_cache: Optional[Dict[str, Any]] = None

    def _part_path(self) -> Path:
        ext = ".pt.gz" if self.gzip else ".pt"
        return self.out_dir / f"dl.part{self._part_idx:05d}{ext}"

    def _ensure_keys_shapes(self, batch: Dict[str, Any]):
        keys = []
        for k in ("x", "y", "cond"):
            v = batch.get(k)
            if isinstance(v, torch.Tensor):
                keys.append(k)
                # 样本级 shape
                shp = list(v.shape[1:])  # drop batch dim
                if k not in self._batch_first_shapes:
                    self._batch_first_shapes[k] = shp
        if not self._keys_present:
            self._keys_present = keys

    def _flush(self):
        if self._buf_bytes == 0:
            return
        path = self._part_path()
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"{path} exists and overwrite=False")

        def cat_or_none(lst):
            if not lst or lst[0] is None:
                return None
            return torch.cat(lst, dim=0)  # [sumB, ...]

        # 先拼接
        x_cat = cat_or_none(self._buffers["x"])
        y_cat = cat_or_none(self._buffers["y"])
        cond_cat = cat_or_none(self._buffers["cond"])

        # 计算 num_samples（以首个存在的主键为准：x -> y -> cond）
        if x_cat is not None:
            num_samples = int(x_cat.shape[0])
        elif y_cat is not None:
            num_samples = int(y_cat.shape[0])
        elif cond_cat is not None:
            num_samples = int(cond_cat.shape[0])
        else:
            num_samples = 0

        payload = {
            "x": x_cat,
            "y": y_cat,
            "cond": cond_cat,
            "num_samples": num_samples,
        }

        # 保存（可选 gzip）
        if self.gzip:
            buf = io.BytesIO()
            torch.save(payload, buf)
            with gzip.open(path, "wb") as gz:
                gz.write(buf.getvalue())
        else:
            torch.save(payload, path)

        self._parts.append({"file": path.name, "num_samples": num_samples})
        self._total_samples += num_samples
        # 清空缓冲并推进分片计数
        self._buffers = {"x": [], "y": [], "cond": []}
        self._buf_bytes = 0
        self._part_idx += 1

    def add_batch(self, batch: Dict[str, Any], *, meta_json: Optional[Dict[str, Any]] = None):
        self._ensure_keys_shapes(batch)
        if meta_json is not None and self._meta_cache is None:
            self._meta_cache = meta_json

        # 累加到 buffer
        for k in ("x", "y", "cond"):
            v = batch.get(k)
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu()
                self._buffers[k].append(t)
                nb = _tensor_nbytes(t)
                self._buf_bytes += nb
                self._total_bytes += nb
            else:
                # None：照样占位，保持 keys 一致
                pass

        # 超过目标大小 → flush
        if self._buf_bytes >= self.target_part_bytes:
            self._flush()

    def close(self) -> SnapshotIndex:
        self._flush()
        index = SnapshotIndex(
            version=1,
            parts=self._parts,
            total_samples=self._total_samples,
            total_nbytes_uncompressed=self._total_bytes,
            keys=self._keys_present,
            batch_first_shapes=self._batch_first_shapes,
            meta=self._meta_cache or {},
            compressed=self.gzip,
        )
        with open(self.out_dir / "snapshot_index.json", "w", encoding="utf-8") as f:
            json.dump(index.to_json(), f, ensure_ascii=False, indent=2)
        return index


# --------------------
# 读取端：内存优先 / 不够内存则流式
# --------------------
class _ShardIterator(IterableDataset):
    """按分片逐样本输出（用于低内存流式训练）。"""
    def __init__(self, dir_path: str, parts: List[Dict[str, Any]], compressed: bool):
        super().__init__()
        self.dir = Path(dir_path)
        self.parts = parts
        self.compressed = compressed

    def _iter_range(self, start: int, step: int):
        for i in range(start, len(self.parts), step):
            fn = self.parts[i]["file"]
            p = self.dir / fn
            if self.compressed:
                with gzip.open(p, "rb") as gz:
                    payload = torch.load(io.BytesIO(gz.read()), map_location="cpu")
            else:
                payload = torch.load(p, map_location="cpu")
            x, y, cond = payload.get("x"), payload.get("y"), payload.get("cond")
            m = payload.get("num_samples", (x.shape[0] if x is not None else (y.shape[0] if y is not None else 0)))
            for idx in range(m):
                item = {}
                if x is not None: item["x"] = x[idx]
                if y is not None: item["y"] = y[idx]
                if cond is not None: item["cond"] = cond[idx]
                yield item

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            yield from self._iter_range(0, 1)
        else:
            yield from self._iter_range(info.id, info.num_workers)

# --------------------
# 恢复入口
# --------------------
def load_snapshot_as_dataloader(
    dir_path: str,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    max_ram_gb: Optional[float] = None,
    force_streaming: bool = False,
    shuffle: bool = False,
    **loader_kwargs,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    从快照目录恢复 DataLoader。
      - 若 max_ram_gb 足够（或 force_streaming=False 且满足），则一次性整载入内存 → TensorDataset → DataLoader
      - 否则返回流式 IterableDataset → DataLoader（分片顺序读取）
    返回 (dataloader, info)；info 中含 "mode": "in_memory"|"streaming"
    """
    for k in ("batch_size", "shuffle", "num_workers"):
        if k in loader_kwargs:
            loader_kwargs.pop(k)
    # 低开销保护：当 num_workers==0 时，避免 DataLoader 因不合法组合报错
    if num_workers == 0:
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)
    
    p = Path(dir_path)
    idx_path = p / "snapshot_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"snapshot_index.json not found under {dir_path}")
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    total_bytes = idx["total_nbytes_uncompressed"]
    parts = idx["parts"]
    keys = idx["keys"]
    compressed = bool(idx.get("compressed", False))

    # 判断是否整载入内存
    def _bytes_to_gb(b: int) -> float:
        return b / (1 << 30)

    can_in_memory = (not force_streaming) and (
        (max_ram_gb is None) or (_bytes_to_gb(total_bytes) <= max_ram_gb)
    )

    if can_in_memory and len(parts) > 0:
        # 一次性读入所有分片，拼成大张量
        buffers: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
        for it in parts:
            fn = it["file"]
            fp = p / fn
            if compressed:
                with gzip.open(fp, "rb") as gz:
                    payload = _safe_torch_load(io.BytesIO(gz.read()), map_location="cpu")
            else:
                payload = _safe_torch_load(fp, map_location="cpu")
            for k in keys:
                buffers[k].append(payload[k])
        # cat -> big tensors
        stacked: Dict[str, torch.Tensor] = {k: torch.cat(vs, dim=0) for k, vs in buffers.items() if len(vs) > 0}
        # 构建 in-memory dataset
        ds = _InMemoryTensorDataset(**stacked)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **loader_kwargs)
        return dl, {"mode": "in_memory", "num_samples": len(ds), "parts": len(parts)}
    else:
        # ---- streaming：IterableDataset 不支持 shuffle，忽略任何 shuffle 请求 ----
        if shuffle:
            warnings.warn("IterableDataset 不支持 shuffle；已忽略 shuffle=True。", RuntimeWarning)
        ds = _ShardIterator(str(p), parts, compressed)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, **loader_kwargs)  # 不传 shuffle
        return dl, {"mode": "streaming", "parts": len(parts)}

# --------------------
# 内存数据集
# --------------------
class _InMemoryTensorDataset(torch.utils.data.Dataset):
    """
    用拼接后的大张量作为样本源：__getitem__ 按索引切出 {"x","y","cond"}。
    """
    def __init__(self, x: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 cond: Optional[torch.Tensor] = None):
        self.x = x
        self.y = y
        self.cond = cond
        # 任意一个存在即可确定长度
        if x is not None: self._n = x.shape[0]
        elif y is not None: self._n = y.shape[0]
        elif cond is not None: self._n = cond.shape[0]
        else: self._n = 0

    def __len__(self):
        return self._n

    def __getitem__(self, idx: int):
        item = {}
        if self.x is not None: item["x"] = self.x[idx]
        if self.y is not None: item["y"] = self.y[idx]
        if self.cond is not None: item["cond"] = self.cond[idx]
        return item
