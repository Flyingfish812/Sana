# backend/dataio/readers/mat.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import scipy.io as sio   # pip install scipy
import h5py              # for v7.3 mat
from .base import BaseReader
from ..schema import DataMeta
from ..utils.typing import Array5D, Shape5D
from ..registry import register_reader


class MatReader(BaseReader):
    """
    通用 .mat 读取器（兼容 v7 与 v7.3）：
      - 典型 NOAA 风格：主变量为 [T, H*W] 或 [T,H,W]；配套 lon/lat 用于重塑网格
      - 自动规范到 [N=1, T, H, W, C]
    关键参数：
      path: .mat 文件路径
      var: 主变量键名（如 "sst"）
      lon_key, lat_key: 若主变量为 [T,H*W]，通过 lon/lat 推断 H,W（需可整形）
      time_key: 可选，时间轴（1D）
      fill_value: NaN/Inf 替换
    """
    def __init__(
        self,
        path: str,
        var: str = "sst",
        lon_key: Optional[str] = None,
        lat_key: Optional[str] = None,
        time_key: Optional[str] = None,
        fill_value: Optional[float] = None,
    ):
        self.path = path
        self.var = var
        self.lon_key = lon_key
        self.lat_key = lat_key
        self.time_key = time_key
        self.fill_value = fill_value

        self._shape5d, self._meta = self._probe_file()

    # --- helpers ---
    def _load_v7(self):
        return sio.loadmat(self.path)

    def _load_v73(self):
        return h5py.File(self.path, "r")

    @staticmethod
    def _is_v73(path: str) -> bool:
        try:
            with h5py.File(path, "r") as f:
                _ = list(f.keys())
            return True
        except Exception:
            return False

    def _read_core(self):
        if self._is_v73(self.path):
            f = self._load_v73()
            try:
                var = np.array(f[self.var][...])
                lon = np.array(f[self.lon_key][...]) if self.lon_key and self.lon_key in f else None
                lat = np.array(f[self.lat_key][...]) if self.lat_key and self.lat_key in f else None
                times = np.array(f[self.time_key][...]).reshape(-1) if self.time_key and self.time_key in f else None
            finally:
                f.close()
        else:
            d = self._load_v7()
            # MATLAB 变量可能带多余维度，使用 np.squeeze 去掉
            var = np.asarray(d[self.var]).squeeze()
            lon = np.asarray(d[self.lon_key]).squeeze() if self.lon_key and self.lon_key in d else None
            lat = np.asarray(d[self.lat_key]).squeeze() if self.lat_key and self.lat_key in d else None
            times = np.asarray(d[self.time_key]).squeeze() if self.time_key and self.time_key in d else None
        return var, lon, lat, times

    @staticmethod
    def _reshape_var(var: np.ndarray, lon: Optional[np.ndarray], lat: Optional[np.ndarray]) -> np.ndarray:
        """
        将 var 变成 [T,H,W]：
        - 若已是 [T,H,W]：直接返回
        - 若是 [T, H*W] 且提供 lon/lat，则从 lon/lat 推断 H,W 并 reshape
        兼容以下 lon/lat 形态：
        向量：(W,) / (1,W)/(W,1)；(H,) / (1,H)/(H,1)
        网格：(H,W)
        注意：优先使用 MATLAB 导出的列主序（order='F'）重建。
        额外修复：重建后对纬向做翻转（[:, ::-1, :]），保证地图“高纬在上”。
        """
        var = np.asarray(var)
        if var.ndim == 3:
            thw = var
        elif var.ndim == 2 and lon is not None and lat is not None:
            lon_arr = np.asarray(lon).squeeze()
            lat_arr = np.asarray(lat).squeeze()
            L = var.shape[1]

            def try_infer_hw_from_lonlat(lon_a, lat_a, L):
                lon_len, lat_len = int(lon_a.size), int(lat_a.size)
                if lon_len * lat_len == L:
                    return lat_len, lon_len  # H, W
                if lon_a.ndim == 2 and lon_a.size == L:
                    return lon_a.shape
                if lat_a.ndim == 2 and lat_a.size == L:
                    return lat_a.shape
                if lon_a.ndim == 2 and 1 in lon_a.shape:
                    w = lon_a.size
                    if (lat_a.size * w) == L:
                        return int(lat_a.size), int(w)
                if lat_a.ndim == 2 and 1 in lat_a.shape:
                    h = lat_a.size
                    if (h * lon_a.size) == L:
                        return int(h), int(lon_a.size)
                return None

            hw = try_infer_hw_from_lonlat(lon_arr, lat_arr, L)
            if hw is None:
                raise ValueError(
                    f"Cannot infer H,W from lon/lat for linear size {L}. "
                    f"Got lon.shape={np.shape(lon)}, lat.shape={np.shape(lat)}"
                )
            H, W = hw
            T = int(var.shape[0])
            try:
                thw = np.reshape(var, (T, H, W), order="F")
            except Exception:
                thw = np.reshape(var, (T, H, W))
        else:
            raise ValueError(f"Unsupported var shape {var.shape}. Expect [T,H,W] or [T,H*W] with lon/lat.")

        # 关键修复：纬向翻转，保证 (0,0) 对应地图左上角（高纬在上）
        thw = thw[:, ::-1, :]
        return thw

    def _build_array5d(self) -> np.ndarray:
        var, lon, lat, times = self._read_core()
        thw = self._reshape_var(var, lon, lat)          # [T,H,W]
        valid_mask = np.isfinite(thw).all(axis=0, keepdims=False).astype(np.float32)
        self._mask_static = valid_mask[..., None]
        if self.fill_value is not None:
            thw = np.nan_to_num(thw, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        tchw = thw[..., None]                            # [T,H,W,1]
        out = tchw[None, ...]                            # [1,T,H,W,1]
        self._times = times
        return out

    def _probe_file(self) -> Tuple[Shape5D, DataMeta]:
        arr = self._build_array5d()
        mask_static = getattr(self, "_mask_static", None)
        if mask_static is not None:
            mask_static = mask_static.astype(np.float32)
        meta = DataMeta(
            times=getattr(self, "_times", None),
            mask_static=mask_static,
            attrs={
                "source": "mat",
                "path": self.path,
                "var": self.var,
                "channels": [self.var],
            },
        )
        N,T,H,W,C = arr.shape
        return (N,T,H,W,C), meta

    # --- BaseReader API ---
    def probe(self) -> Tuple[Shape5D, DataMeta]:
        return self._shape5d, self._meta

    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> Array5D:
        return self._ensure_5d(self._build_array5d())


@register_reader("mat")
def _build_mat_reader(**kwargs) -> MatReader:
    """
    用法示例：
      r_cfg = {"kind":"mat", "path":"./datasets/noaa.mat", "var":"sst", "lon_key":"lon", "lat_key":"lat", "time_key":"time"}
    """
    return MatReader(**kwargs)
