from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import shutil
import sys
import tempfile

import h5py
import netCDF4 as nc
import numpy as np
import scipy.io as sio

from .types import DataMeta, Shape5D


class BaseReader(ABC):
    """统一数据读取接口，负责输出标准 NTHWC 结构。"""
    @abstractmethod
    def probe(self) -> Tuple[Shape5D, DataMeta]:
        """探测源数据并返回 shape 与元信息，不做完整加载。"""
        ...

    @abstractmethod
    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """读取源数据并返回 5D 数组 [N,T,H,W,C]。"""
        ...

    @staticmethod
    def _ensure_5d(x: np.ndarray) -> np.ndarray:
        """校验数组维度必须为 5。"""
        if x.ndim != 5:
            raise ValueError(f"expect 5D [N,T,H,W,C], got shape={x.shape}")
        return x


class H5Reader(BaseReader):
    """读取 HDF5 数据并转换到统一 NTHWC 格式。"""
    def __init__(
        self,
        path: str,
        dataset: Optional[Union[str, List[str]]] = None,
        group: Optional[str] = None,
        times_key: Optional[str] = None,
        fill_value: Optional[float] = None,
    ):
        """初始化 H5Reader 并缓存探测结果。"""
        self.path = path
        self.dataset = dataset
        self.group = group
        self.times_key = times_key
        self.fill_value = fill_value
        self._shape5d, self._meta = self._probe_file()

    @staticmethod
    def _is_ds(x):
        """判断对象是否为 HDF5 数据集节点。"""
        return isinstance(x, h5py.Dataset)

    def _read_ds(self, ds: h5py.Dataset) -> np.ndarray:
        """读取单个数据集并按需替换 NaN/Inf。"""
        arr = ds[...]
        if self.fill_value is not None:
            arr = np.nan_to_num(arr, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        return np.asarray(arr)

    def _to_thwc(self, a: np.ndarray) -> np.ndarray:
        """将数组规整到 [T,H,W,C] 或保持 [N,T,H,W,C]。"""
        a = np.asarray(a)
        if a.ndim == 3:
            return a[..., None]
        if a.ndim == 4:
            return a
        if a.ndim == 5:
            return a
        raise ValueError(f"Unsupported array ndim={a.ndim}, shape={a.shape}")

    def _recursive_hits(self, f: h5py.File, basename: str) -> List[str]:
        """递归查找同名 dataset 的完整路径列表。"""
        hits: List[str] = []

        def walk(g: h5py.Group, prefix: str = ""):
            for key, val in g.items():
                p = f"{prefix}/{key}" if prefix else key
                if self._is_ds(val) and key == basename:
                    hits.append(p)
                elif isinstance(val, h5py.Group):
                    walk(val, p)

        walk(f)
        return hits

    def _stack_or_concat(self, arrays: List[np.ndarray], expect_stack_n: bool) -> np.ndarray:
        """按数据形态执行按 N 堆叠或按 C 拼接。"""
        if not arrays:
            raise ValueError("empty arrays")
        norm = [self._to_thwc(a) for a in arrays]
        if any(a.ndim == 5 for a in norm):
            base = norm[0]
            n0, t0, h0, w0, _ = base.shape
            for a in norm:
                if a.ndim != 5 or a.shape[:4] != (n0, t0, h0, w0):
                    raise ValueError("shape mismatch for 5D concat")
            return np.concatenate(norm, axis=-1)

        t0, h0, w0, _ = norm[0].shape
        for a in norm:
            if a.shape[:3] != (t0, h0, w0):
                raise ValueError("shape mismatch for 4D combine")
        if expect_stack_n:
            return np.stack(norm, axis=0)
        if len(norm) == 1:
            return norm[0][None, ...]
        return np.concatenate(norm, axis=-1)[None, ...]

    def _collect(self, f: h5py.File) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """根据配置从 HDF5 文件收集主数组与可选时间轴。"""
        times = None
        if self.dataset is not None:
            keys = [self.dataset] if isinstance(self.dataset, str) else list(self.dataset)
            per_key: List[np.ndarray] = []
            n_ref = None
            first_parent = None

            for key in keys:
                if "/" in key and key in f and self._is_ds(f[key]):
                    per_key.append(self._to_thwc(self._read_ds(f[key]))[None, ...][0])
                    if times is None and not self.times_key:
                        parent = "/".join(key.split("/")[:-1])
                        first_parent = first_parent or parent
                elif key in f and self._is_ds(f[key]):
                    per_key.append(self._to_thwc(self._read_ds(f[key]))[None, ...][0])
                    if times is None and not self.times_key:
                        first_parent = ""
                else:
                    hits = self._recursive_hits(f, key.split("/")[-1])
                    if not hits:
                        raise KeyError(f"object '{key}' doesn't exist")
                    arr_n = self._stack_or_concat([self._read_ds(f[p]) for p in hits], expect_stack_n=True)
                    if n_ref is None:
                        n_ref = arr_n.shape[0]
                    elif n_ref != arr_n.shape[0]:
                        raise ValueError("Different N across variables")
                    per_key.append(arr_n)
                    if times is None and not self.times_key:
                        first_parent = "/".join(hits[0].split("/")[:-1])

            norm5 = []
            n_ref = None
            for a in per_key:
                aa = np.asarray(a)
                if aa.ndim == 4:
                    aa = aa[None, ...]
                if n_ref is None:
                    n_ref = aa.shape[0]
                elif n_ref != aa.shape[0]:
                    raise ValueError("N mismatch when joining variables")
                norm5.append(aa)
            out = np.concatenate(norm5, axis=-1)

            if self.times_key and self.times_key in f and self._is_ds(f[self.times_key]):
                times = np.asarray(f[self.times_key][...]).reshape(-1)
            elif first_parent is not None:
                for cand in ("grid/t", "t", "time"):
                    p = f"{first_parent}/{cand}" if first_parent else cand
                    if p in f and self._is_ds(f[p]):
                        times = np.asarray(f[p][...]).reshape(-1)
                        break
            return out, times

        if self.group is not None:
            if self.group not in f or not isinstance(f[self.group], h5py.Group):
                raise KeyError(f"group '{self.group}' not found")

            def walk(g):
                for _, val in g.items():
                    if self._is_ds(val):
                        yield val
                    elif isinstance(val, h5py.Group):
                        yield from walk(val)

            out = self._stack_or_concat([self._read_ds(ds) for ds in walk(f[self.group])], expect_stack_n=False)
            if self.times_key and self.times_key in f and self._is_ds(f[self.times_key]):
                times = np.asarray(f[self.times_key][...]).reshape(-1)
            else:
                for cand in ("grid/t", "t", "time"):
                    p = f"{self.group}/{cand}"
                    if p in f and self._is_ds(f[p]):
                        times = np.asarray(f[p][...]).reshape(-1)
                        break
            return out, times

        for _, val in f.items():
            if self._is_ds(val):
                return self._stack_or_concat([self._read_ds(val)], expect_stack_n=False), None
        raise KeyError("No dataset found at root.")

    def _probe_file(self) -> Tuple[Shape5D, DataMeta]:
        """快速探测 H5 数据形状与元信息。"""
        with h5py.File(self.path, "r") as f:
            arr, times = self._collect(f)
        n, t, h, w, c = arr.shape
        meta = DataMeta(times=times, attrs={"source": "h5", "path": self.path})
        return (n, t, h, w, c), meta

    def probe(self) -> Tuple[Shape5D, DataMeta]:
        """返回初始化时缓存的探测结果。"""
        return self._shape5d, self._meta

    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """完整读取 H5 数据并返回标准 5D 数组。"""
        with h5py.File(self.path, "r") as f:
            arr, _ = self._collect(f)
        return self._ensure_5d(arr)


def _norm_path(p: str) -> str:
    """展开环境变量并标准化为绝对路径。"""
    q = os.path.expanduser(os.path.expandvars(str(p)))
    return str(Path(q).resolve())


def _contains_non_ascii(s: str) -> bool:
    """判断路径是否包含非 ASCII 字符。"""
    try:
        s.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _open_nc_with_compat(path: str, enable_compat: bool = True):
    """打开 NetCDF 文件；在兼容模式下可回退到临时副本。"""
    p = _norm_path(path)
    file_path = Path(p)
    if not file_path.exists():
        raise FileNotFoundError(f"[NCReader] File not found: {p}")
    try:
        return nc.Dataset(p, "r"), None
    except FileNotFoundError:
        if not (enable_compat and sys.platform.startswith("win") and _contains_non_ascii(p)):
            raise
        tmp_dir = Path(tempfile.gettempdir()) / "nc_compat_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / ("nc_" + str(abs(hash(p))) + ".nc")
        if not tmp_path.exists():
            shutil.copyfile(p, tmp_path)
        return nc.Dataset(str(tmp_path), "r"), str(tmp_path)


class NCReader(BaseReader):
    """读取 NetCDF 多变量场并输出 NTHWC。"""
    def __init__(
        self,
        path: str,
        var_keys: Union[str, List[str]],
        u_key: Optional[str] = None,
        v_key: Optional[str] = None,
        want_omega: bool = False,
        dx: float = 1.0,
        dy: float = 1.0,
        time_key: Optional[str] = None,
        y_key: Optional[str] = None,
        x_key: Optional[str] = None,
        fill_value: Optional[float] = None,
        path_compat_workaround: bool = True,
    ):
        """初始化 NCReader 并缓存探测结果。"""
        self.path = _norm_path(path)
        self.var_keys = [var_keys] if isinstance(var_keys, str) else list(var_keys)
        self.u_key = u_key
        self.v_key = v_key
        self.want_omega = bool(want_omega)
        self.dx = float(dx)
        self.dy = float(dy)
        self.time_key = time_key
        self.y_key = y_key
        self.x_key = x_key
        self.fill_value = fill_value
        self.path_compat_workaround = bool(path_compat_workaround)
        self._shape5d, self._meta = self._probe_file()

    @staticmethod
    def _reorder_to_thw(
        ds,
        var_name: str,
        arr: np.ndarray,
        time_key: Optional[str] = None,
        y_key: Optional[str] = None,
        x_key: Optional[str] = None,
    ) -> np.ndarray:
        """依据维度名或尺寸启发式将变量重排为 [T,H,W]。"""
        v = ds.variables[var_name]
        dims = list(v.dimensions)
        shape = list(arr.shape)
        t_names = [time_key, "tdim", "time", "t"]
        y_names = [y_key, "ydim", "lat", "y"]
        x_names = [x_key, "xdim", "lon", "x"]

        def pick_axis_by_name(names):
            for name in names:
                if name and name in dims:
                    return dims.index(name)
            return None

        it = pick_axis_by_name(t_names)
        iy = pick_axis_by_name(y_names)
        ix = pick_axis_by_name(x_names)
        used = set(ax for ax in (it, iy, ix) if ax is not None)

        def coord_len(names):
            for name in names:
                if name and (name in ds.variables):
                    try:
                        return int(ds.variables[name].shape[0])
                    except Exception:
                        pass
            return None

        lt = coord_len(t_names)
        ly = coord_len(y_names)
        lx = coord_len(x_names)

        def pick_by_length(want):
            if want is None:
                return None
            cands = [i for i, s in enumerate(shape) if (s == want and i not in used)]
            return cands[0] if cands else None

        if it is None:
            it = pick_by_length(lt)
            if it is not None:
                used.add(it)
        if iy is None:
            iy = pick_by_length(ly)
            if iy is not None:
                used.add(iy)
        if ix is None:
            ix = pick_by_length(lx)

        if None in (it, iy, ix):
            order = np.argsort(shape)
            iy = order[0] if iy is None else iy
            ix = order[-1] if ix is None else ix
            if it is None:
                remaining = [i for i in range(len(shape)) if i not in (iy, ix)]
                it = remaining[0] if remaining else (order[1] if len(order) > 2 else order[0])

        return np.moveaxis(arr, (it, iy, ix), (0, 1, 2)).astype(np.float32, copy=False)

    def _probe_file(self) -> Tuple[Shape5D, DataMeta]:
        """探测 NetCDF 主变量布局并生成元信息。"""
        ds, tmp_used = _open_nc_with_compat(self.path, enable_compat=self.path_compat_workaround)
        try:
            t_key = self.time_key or next((k for k in ("time", "t") if k in ds.variables), None)
            main = ds.variables[self.var_keys[0]]
            dims = list(main.dimensions)
            shape = list(main.shape)

            def idx_of(names):
                for cand in names:
                    if cand in dims:
                        return dims.index(cand)
                return None

            it = idx_of([t_key] if t_key else ["tdim", "time", "t"])
            iy = idx_of([self.y_key] if self.y_key else ["ydim", "lat", "y"])
            ix = idx_of([self.x_key] if self.x_key else ["xdim", "lon", "x"])
            if (it is None) or (iy is None) or (ix is None):
                order = np.argsort(shape)
                iy = int(order[0]) if iy is None else iy
                ix = int(order[-1]) if ix is None else ix
                if it is None:
                    rest = [i for i in range(len(shape)) if i not in (iy, ix)]
                    it = int(rest[0]) if rest else int(order[1] if len(order) > 2 else order[0])

            t_len = int(shape[it])
            h_len = int(shape[iy])
            w_len = int(shape[ix])
            c_len = len(self.var_keys) + (1 if self.want_omega else 0)

            times = None
            if t_key and t_key in ds.variables:
                try:
                    times = np.asarray(ds.variables[t_key][:]).reshape(-1)
                except Exception:
                    times = None

            channels = list(self.var_keys)
            if self.want_omega:
                channels.append("omega")
            meta = DataMeta(
                times=times,
                attrs={"source": "nc", "path": self.path, "tmp_used": bool(tmp_used), "channels": channels},
            )
            return (1, t_len, h_len, w_len, c_len), meta
        finally:
            ds.close()

    def probe(self) -> Tuple[Shape5D, DataMeta]:
        """返回初始化时缓存的探测结果。"""
        return self._shape5d, self._meta

    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """读取 NetCDF 变量并可选构造涡量通道。"""
        ds, tmp_used = _open_nc_with_compat(self.path, enable_compat=self.path_compat_workaround)
        try:
            thw_list: List[np.ndarray] = []
            for key in self.var_keys:
                raw = np.asarray(ds.variables[key][:])
                thw = self._reorder_to_thw(ds, key, raw, self.time_key, self.y_key, self.x_key)
                if self.fill_value is not None:
                    thw = np.nan_to_num(thw, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
                thw_list.append(thw.astype(np.float32, copy=False))

            if self.want_omega:
                if not (self.u_key and self.v_key):
                    raise ValueError("want_omega=True requires both u_key and v_key")
                u = self._reorder_to_thw(ds, self.u_key, np.asarray(ds.variables[self.u_key][:]), self.time_key, self.y_key, self.x_key)
                v = self._reorder_to_thw(ds, self.v_key, np.asarray(ds.variables[self.v_key][:]), self.time_key, self.y_key, self.x_key)
                dv_dx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2.0 * float(self.dx))
                du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * float(self.dy))
                thw_list.append((dv_dx - du_dy).astype(np.float32))

            thwc = np.stack(thw_list, axis=-1)
            arr5d = thwc[None, ...]
            self._meta = DataMeta(
                times=self._meta.times,
                attrs={"source": "nc", "path": self.path, "tmp_used": bool(tmp_used), "channels": self._meta.attrs.get("channels", [])},
            )
            return arr5d
        finally:
            ds.close()


class MatReader(BaseReader):
    """读取 MAT(v7/v7.3) 文件并构造 NTHWC 数据。"""
    def __init__(
        self,
        path: str,
        var: str = "sst",
        lon_key: Optional[str] = None,
        lat_key: Optional[str] = None,
        time_key: Optional[str] = None,
        fill_value: Optional[float] = None,
    ):
        """初始化 MatReader 并缓存探测结果。"""
        self.path = path
        self.var = var
        self.lon_key = lon_key
        self.lat_key = lat_key
        self.time_key = time_key
        self.fill_value = fill_value
        self._shape5d, self._meta = self._probe_file()

    @staticmethod
    def _is_v73(path: str) -> bool:
        """判断 MAT 文件是否为 v7.3(HDF5) 格式。"""
        try:
            with h5py.File(path, "r") as f:
                _ = list(f.keys())
            return True
        except Exception:
            return False

    def _read_core(self):
        """读取 MAT 主变量及可选坐标/时间数组。"""
        if self._is_v73(self.path):
            f = h5py.File(self.path, "r")
            try:
                var = np.array(f[self.var][...])
                lon = np.array(f[self.lon_key][...]) if self.lon_key and self.lon_key in f else None
                lat = np.array(f[self.lat_key][...]) if self.lat_key and self.lat_key in f else None
                times = np.array(f[self.time_key][...]).reshape(-1) if self.time_key and self.time_key in f else None
            finally:
                f.close()
        else:
            d = sio.loadmat(self.path)
            var = np.asarray(d[self.var]).squeeze()
            lon = np.asarray(d[self.lon_key]).squeeze() if self.lon_key and self.lon_key in d else None
            lat = np.asarray(d[self.lat_key]).squeeze() if self.lat_key and self.lat_key in d else None
            times = np.asarray(d[self.time_key]).squeeze() if self.time_key and self.time_key in d else None
        return var, lon, lat, times

    @staticmethod
    def _reshape_var(var: np.ndarray, lon: Optional[np.ndarray], lat: Optional[np.ndarray]) -> np.ndarray:
        """将原始变量重排为 [T,H,W]，并统一纬向翻转方向。"""
        var = np.asarray(var)
        if var.ndim == 3:
            thw = var
        elif var.ndim == 2 and lon is not None and lat is not None:
            lon_arr = np.asarray(lon).squeeze()
            lat_arr = np.asarray(lat).squeeze()
            linear = var.shape[1]
            lon_len, lat_len = int(lon_arr.size), int(lat_arr.size)
            if lon_len * lat_len != linear:
                raise ValueError(
                    f"Cannot infer H,W from lon/lat for linear size {linear}, "
                    f"lon.shape={np.shape(lon)}, lat.shape={np.shape(lat)}"
                )
            h_len, w_len = lat_len, lon_len
            t_len = int(var.shape[0])
            try:
                thw = np.reshape(var, (t_len, h_len, w_len), order="F")
            except Exception:
                thw = np.reshape(var, (t_len, h_len, w_len))
        else:
            raise ValueError(f"Unsupported var shape {var.shape}. Expect [T,H,W] or [T,H*W] with lon/lat")
        return thw[:, ::-1, :]

    def _build_array5d(self) -> np.ndarray:
        """将 MAT 内容构建为标准 [N,T,H,W,C] 数组。"""
        var, lon, lat, times = self._read_core()
        thw = self._reshape_var(var, lon, lat)
        if self.fill_value is not None:
            thw = np.nan_to_num(thw, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        out = thw[..., None][None, ...]
        self._times = times
        return out

    def _probe_file(self) -> Tuple[Shape5D, DataMeta]:
        """探测 MAT 数据形状与元信息。"""
        arr = self._build_array5d()
        n, t, h, w, c = arr.shape
        meta = DataMeta(times=getattr(self, "_times", None), attrs={"source": "mat", "path": self.path, "var": self.var})
        return (n, t, h, w, c), meta

    def probe(self) -> Tuple[Shape5D, DataMeta]:
        """返回初始化时缓存的探测结果。"""
        return self._shape5d, self._meta

    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """读取 MAT 数据并保证输出为 5D。"""
        return self._ensure_5d(self._build_array5d())


def build_reader(kind: str, **kwargs) -> BaseReader:
    """按 kind 创建对应 Reader 实例。"""
    key = str(kind).lower()
    if key == "h5":
        return H5Reader(**kwargs)
    if key == "nc":
        return NCReader(**kwargs)
    if key in ("mat", "sst"):
        return MatReader(**kwargs)
    raise ValueError(f"Unsupported reader kind '{kind}'. Expected one of: h5, nc, mat/sst")
