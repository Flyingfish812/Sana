# backend/dataio/readers/h5.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List, Union
import numpy as np
import h5py
from .base import BaseReader
from ..schema import DataMeta
from ..utils.typing import Array5D, Shape5D
from ..registry import register_reader

class H5Reader(BaseReader):
    """
    通用 HDF5 读取器（统一到 [N,T,H,W,C]）——同时适配：
      1) 根路径直挂的 dataset（如 /data, /var1, /var2）
      2) 分组结构：/0000/data, /0001/data ... 且每组有 /grid/{t,x,y}
    使用方式：
      - dataset: "data"        → 在整个文件递归收集 basename==data 的路径，按 N 维堆叠
      - dataset: ["u","v"]     → 分别递归收集后在 C 维拼接
      - dataset: "group/data"  → **精确路径**，只读这一条（N=1）
      - group: "root_group"    → 读取该组下所有叶子 dataset 在 C 维拼接（N=1）
    times_key：
      - 显式给出时优先（可以是绝对路径 "grid/t" 或 "group/grid/t"）
      - 未给出时：若是分组收集，自动在首个命中组内尝试 "grid/t"、"t"、"time"
    """

    def __init__(self, path: str,
                 dataset: Optional[Union[str, List[str]]] = None,
                 group: Optional[str] = None,
                 times_key: Optional[str] = None,
                 fill_value: Optional[float] = None):
        self.path = path
        self.dataset = dataset
        self.group = group
        self.times_key = times_key
        self.fill_value = fill_value
        self._channel_names: Optional[List[str]] = None
        self._mask_static: Optional[np.ndarray] = None
        self._shape5d, self._meta = self._probe_file()

    # ---------- helpers ----------
    @staticmethod
    def _is_ds(x): return isinstance(x, h5py.Dataset)

    def _read_ds(self, ds: h5py.Dataset) -> np.ndarray:
        arr = ds[...]
        if self.fill_value is not None:
            arr = np.nan_to_num(arr, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        return np.asarray(arr)

    def _to_THWC(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 3:  # [T,H,W]
            return a[..., None]
        if a.ndim == 4:  # [T,H,W,C]
            return a
        if a.ndim == 5:  # [N,T,H,W,C]
            return a
        raise ValueError(f"Unsupported array ndim={a.ndim}, shape={a.shape}")

    def _recursive_hits(self, f: h5py.File, basename: str) -> List[str]:
        hits = []
        def walk(g: h5py.Group, prefix=""):
            for k, v in g.items():
                p = f"{prefix}/{k}" if prefix else k
                if self._is_ds(v) and k == basename:
                    hits.append(p)
                elif isinstance(v, h5py.Group):
                    walk(v, p)
        walk(f)
        return hits

    def _stack_N_or_concat_C(self, arrays: List[np.ndarray], expect_stack_N: bool) -> np.ndarray:
        if not arrays:
            raise ValueError("empty arrays")
        norm = [self._to_THWC(a) for a in arrays]
        if any(a.ndim == 5 for a in norm):
            base = norm[0]
            N,T,H,W,_ = base.shape
            for a in norm:
                if a.ndim != 5 or a.shape[:4] != (N,T,H,W):
                    raise ValueError("shape mismatch for 5D concat")
            return np.concatenate(norm, axis=-1)
        T,H,W,_ = norm[0].shape
        for a in norm:
            if a.shape[:3] != (T,H,W):
                raise ValueError("shape mismatch for 4D combine")
        if expect_stack_N:
            return np.stack(norm, axis=0)         # [N,T,H,W,C]
        else:
            return norm[0][None, ...] if len(norm)==1 else np.concatenate(norm, axis=-1)[None, ...]

    def _collect(self, f: h5py.File) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        times = None

        # 优先：按 dataset 配置
        if self.dataset is not None:
            keys = [self.dataset] if isinstance(self.dataset, str) else list(self.dataset)
            per_key: List[np.ndarray] = []
            nN = None
            first_parent = None
            channel_names: List[str] = []

            for k in keys:
                # 情况 a) 精确路径（含斜杠）
                if "/" in k and k in f and self._is_ds(f[k]):
                    arr = self._read_ds(f[k])
                    arr_thwc = self._to_THWC(arr)[None, ...][0]
                    per_key.append(arr_thwc)  # 临时存回 [T,H,W,C]
                    channel_names.extend([k] * arr_thwc.shape[-1])
                    # times：若 times_key 为空，尝试 k 的父组
                    if times is None and not self.times_key:
                        parent = "/".join(k.split("/")[:-1])
                        first_parent = first_parent or parent
                # 情况 b) 根下同名
                elif k in f and self._is_ds(f[k]):
                    arr = self._read_ds(f[k])
                    arr_thwc = self._to_THWC(arr)[None, ...][0]
                    per_key.append(arr_thwc)
                    channel_names.extend([k] * arr_thwc.shape[-1])
                    if times is None and not self.times_key:
                        first_parent = ""
                else:
                    # 情况 c) 递归收集同名 basename
                    hits = self._recursive_hits(f, k.split("/")[-1])
                    if not hits:
                        raise KeyError(f"object '{k}' doesn't exist")
                    arrays = [self._read_ds(f[p]) for p in hits]
                    arrN = self._stack_N_or_concat_C(arrays, expect_stack_N=True)  # [N,T,H,W,C]
                    if nN is None: nN = arrN.shape[0]
                    elif nN != arrN.shape[0]: raise ValueError("Different N across variables")
                    per_key.append(arrN)  # [N,T,H,W,C]
                    channel_names.extend([k] * arrN.shape[-1])
                    if times is None and not self.times_key:
                        first_parent = "/".join(hits[0].split("/")[:-1])

            # 统一到 5D 然后在 C 维拼
            norm5 = []
            Nref = None
            for a in per_key:
                aa = np.asarray(a)
                if aa.ndim == 4: aa = aa[None, ...]
                if Nref is None: Nref = aa.shape[0]
                elif Nref != aa.shape[0]: raise ValueError("N mismatch when joining variables")
                norm5.append(aa)
            out = np.concatenate(norm5, axis=-1)  # [N,T,H,W,Csum]
            if channel_names:
                self._channel_names = channel_names

            # 自动发现 times
            if self.times_key:
                if self.times_key in f and self._is_ds(f[self.times_key]):
                    times = np.asarray(f[self.times_key][...]).reshape(-1)
            elif first_parent is not None:
                for cand in ("grid/t","t","time"):
                    p = f"{first_parent}/{cand}" if first_parent else cand
                    if p in f and self._is_ds(f[p]):
                        times = np.asarray(f[p][...]).reshape(-1); break

            return out, times

        # 次选：按 group 读取
        if self.group is not None:
            if self.group not in f or not isinstance(f[self.group], h5py.Group):
                raise KeyError(f"group '{self.group}' not found")
            def walk(g):
                for k,v in g.items():
                    if self._is_ds(v): yield v
                    elif isinstance(v,h5py.Group): yield from walk(v)
            arrays = [self._read_ds(ds) for ds in walk(f[self.group])]
            out = self._stack_N_or_concat_C(arrays, expect_stack_N=False)  # [1,T,H,W,C]
            if self.group is not None:
                self._channel_names = ["group"] * out.shape[-1]
            # times
            if self.times_key:
                if self.times_key in f and self._is_ds(f[self.times_key]):
                    times = np.asarray(f[self.times_key][...]).reshape(-1)
            else:
                for cand in ("grid/t","t","time"):
                    p = f"{self.group}/{cand}"
                    if p in f and self._is_ds(f[p]):
                        times = np.asarray(f[p][...]).reshape(-1); break
            return out, times

        # 兜底：根下第一个 dataset
        for k,v in f.items():
            if self._is_ds(v):
                out = self._stack_N_or_concat_C([self._read_ds(v)], expect_stack_N=False)
                self._channel_names = [k] * out.shape[-1]
                return out, None
        raise KeyError("No dataset found at root.")

    def _probe_file(self) -> Tuple[Shape5D, DataMeta]:
        with h5py.File(self.path, "r") as f:
            arr, times = self._collect(f)
        N,T,H,W,C = arr.shape
        mask = np.all(np.isfinite(arr), axis=(0,1,4)).astype(np.float32)
        self._mask_static = mask[..., None]
        attrs = {"source":"h5","path":self.path}
        if self._channel_names:
            attrs["channels"] = self._channel_names
        meta = DataMeta(times=times, mask_static=self._mask_static, attrs=attrs)
        return (N,T,H,W,C), meta

    def probe(self) -> Tuple[Shape5D, DataMeta]:
        return self._shape5d, self._meta

    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> Array5D:
        with h5py.File(self.path, "r") as f:
            arr, _ = self._collect(f)
        if self._mask_static is None:
            mask = np.all(np.isfinite(arr), axis=(0,1,4)).astype(np.float32)
            self._mask_static = mask[..., None]
        attrs = {"source":"h5","path":self.path}
        if self._channel_names:
            attrs["channels"] = self._channel_names
        self._meta = DataMeta(times=self._meta.times, mask_static=self._mask_static, attrs=attrs)
        return self._ensure_5d(arr)

@register_reader("h5")
def _build_h5_reader(**kwargs): return H5Reader(**kwargs)
