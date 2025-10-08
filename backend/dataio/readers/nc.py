# backend/dataio/readers/nc.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List, Union
import os, sys, shutil, tempfile, math
from pathlib import Path

import numpy as np
import netCDF4 as nc

from .base import BaseReader
from ..schema import DataMeta
from ..utils.typing import Array5D, Shape5D
from ..registry import register_reader


# ------------------ 路径 & 兼容层 ------------------

def _norm_path(p: str) -> str:
    """展开 ~ 和环境变量，统一为绝对路径字符串。"""
    q = os.path.expanduser(os.path.expandvars(str(p)))
    return str(Path(q).resolve())

def _contains_non_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True

def _open_nc_with_compat(path: str, *, enable_compat: bool = True):
    """
    尝试打开 NC；若 Windows+非ASCII 路径导致 netCDF4 打不开，
    复制到临时 ASCII 目录再打开。返回 (ds, used_tmp_path)
    """
    p = _norm_path(path)

    # 1) 先做一次存在性检查，给出更友好的报错
    P = Path(p)
    if not P.exists():
        siblings = "\n".join(f"  - {q.name}" for q in sorted(P.parent.glob("*.nc"))) if P.parent.exists() else ""
        raise FileNotFoundError(
            f"[NCReader] File not found:\n  {p}\n"
            f"Parent: {P.parent}\n"
            f"*.nc in parent:\n{siblings or '  (none)'}"
        )

    # 2) 直接尝试原路径（大多数环境可用，含中文也OK）
    try:
        return nc.Dataset(p, "r"), None
    except FileNotFoundError as e:
        # 3) Windows + 路径含非ASCII → 启用兼容兜底
        if not (enable_compat and sys.platform.startswith("win") and _contains_non_ascii(p)):
            raise

        tmp_dir = Path(tempfile.gettempdir()) / "nc_compat_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 用长路径哈希生成安全的 ASCII 文件名，避免重名
        safe_name = "nc_" + str(abs(hash(p))) + ".nc"
        tmp_path = tmp_dir / safe_name

        if not tmp_path.exists():
            shutil.copyfile(p, tmp_path)  # 仅首次复制

        ds = nc.Dataset(str(tmp_path), "r")
        return ds, str(tmp_path)


# ------------------ Reader 实现 ------------------

class NCReader(BaseReader):
    """
    NetCDF 通用读取器：统一输出 [N=1, T, H, W, C]
    YAML 支持字段：
      path:            .nc 文件路径（可含中文）
      var_keys:        变量列表（如 ["u","v"]）
      u_key, v_key:    若 want_omega=True，用于计算涡量
      want_omega:      是否附加涡量通道
      dx, dy:          计算涡量的网格间距（标量）
      time_key:        时间坐标名（默认自动在 ["time","t"] 中探测）
      y_key, x_key:    空间坐标名（默认自动在 ["lat","y"] 与 ["lon","x"] 中探测）
      fill_value:      若提供，读入后对 NaN/Inf 做 nan_to_num
      path_compat_workaround: Windows+中文路径兜底开关（默认 True）
    """
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
    def _reorder_to_THW(ds,
                        var_name: str,
                        arr: np.ndarray,
                        time_key: Optional[str] = None,
                        y_key: Optional[str] = None,
                        x_key: Optional[str] = None) -> np.ndarray:
        """
        将变量 arr 重排为 [T,H,W]：
        1) 优先按维度名精确匹配 time/y/x（含常见别名）
        2) 若仍有空缺，用坐标变量长度与轴长度匹配推断
        3) 仍不确定则兜底：X=最大维，Y=最小维，T=剩余的中间维
        """
        import numpy as np

        v = ds.variables[var_name]
        dims = list(v.dimensions)
        shape = list(arr.shape)

        # 候选名
        t_names = [time_key, "time", "t"]
        y_names = [y_key, "lat", "y"]
        x_names = [x_key, "lon", "x"]

        def pick_axis_by_name(names):
            for n in names:
                if n and n in dims:
                    return dims.index(n)
            return None

        it = pick_axis_by_name(t_names)
        iy = pick_axis_by_name(y_names)
        ix = pick_axis_by_name(x_names)

        # 用坐标变量长度匹配未识别的轴
        def coord_len(names):
            for n in names:
                if n and (n in ds.variables):
                    try:
                        return int(ds.variables[n].shape[0])
                    except Exception:
                        pass
            return None

        Lt = coord_len(t_names)
        Ly = coord_len(y_names)
        Lx = coord_len(x_names)

        used = set(ax for ax in (it, iy, ix) if ax is not None)

        def pick_by_length(Lwant):
            if Lwant is None:
                return None
            cands = [i for i, s in enumerate(shape) if (s == Lwant and i not in used)]
            return cands[0] if cands else None

        if it is None:
            it = pick_by_length(Lt);  used.add(it) if it is not None else None
        if iy is None:
            iy = pick_by_length(Ly);  used.add(iy) if iy is not None else None
        if ix is None:
            ix = pick_by_length(Lx);  used.add(ix) if ix is not None else None

        # ---- 修正的兜底：X=最大维，Y=最小维，T=中间维 ----
        if None in (it, iy, ix):
            order = np.argsort(shape)  # 小→大
            iy = order[0] if iy is None else iy     # 最小 → Y
            ix = order[-1] if ix is None else ix    # 最大 → X
            # 余下的未定轴 → T
            if it is None:
                remaining = [i for i in range(len(shape)) if i not in (iy, ix)]
                it = remaining[0] if remaining else (order[1] if len(order) > 2 else order[0])

        # 重排到 [T,H,W]
        arr_thw = np.moveaxis(arr, (it, iy, ix), (0, 1, 2)).astype(np.float32, copy=False)
        return arr_thw

    # ---- 探测形状/坐标名 ----
    def _probe_file(self) -> Tuple[Shape5D, DataMeta]:
        ds, tmp_used = _open_nc_with_compat(self.path, enable_compat=self.path_compat_workaround)
        try:
            # 猜测坐标名
            t_key = self.time_key or next((k for k in ("time", "t") if k in ds.variables), None)
            y_key = self.y_key or next((k for k in ("lat", "y") if k in ds.variables), None)
            x_key = self.x_key or next((k for k in ("lon", "x") if k in ds.variables), None)

            # 取一个主变量探测维度次序
            main = ds.variables[self.var_keys[0]]
            dims = main.dimensions
            shape = main.shape

            dim_names = list(dims)
            def idx_of(name_cands):
                for cand in name_cands:
                    if cand in dim_names:
                        return dim_names.index(cand)
                return None

            it = idx_of([t_key] if t_key else ["time","t"])
            iy = idx_of([y_key] if y_key else ["lat","y"])
            ix = idx_of([x_key] if x_key else ["lon","x"])

            if (it is None) or (iy is None) or (ix is None):
                order = np.argsort(shape)  # 小→大
                # 兜底规则（已与你的 read_array5d 对齐）：X=最大，Y=最小，T=中间
                iy = int(order[0]) if iy is None else iy
                ix = int(order[-1]) if ix is None else ix
                if it is None:
                    rest = [i for i in range(len(shape)) if i not in (iy, ix)]
                    it = int(rest[0]) if rest else int(order[1] if len(order) > 2 else order[0])

            T = int(shape[it]); H = int(shape[iy]); W = int(shape[ix])

            # times（如有）
            times = None
            if t_key and t_key in ds.variables:
                try:
                    times = np.asarray(ds.variables[t_key][:]).reshape(-1)
                except Exception:
                    times = None

            C = len(self.var_keys) + (1 if self.want_omega else 0)

            # ★ 通道名写进 meta，便于下游可视化按名选通道
            channel_names = list(self.var_keys)
            if self.want_omega:
                channel_names.append("omega")

            shape5d: Shape5D = (1, T, H, W, C)
            meta = DataMeta(times=times, attrs={
                "source": "nc",
                "path": self.path,
                "tmp_used": bool(tmp_used),
                "channels": channel_names,   # ← 新增
            })
            return shape5d, meta
        finally:
            ds.close()

    def probe(self) -> Tuple[Shape5D, DataMeta]:
        return self._shape5d, self._meta

    # ---- 读全量数据并规范到 [1,T,H,W,C] ----
    def read_array5d(self, subset: Optional[Dict[str, Any]] = None) -> Array5D:
        """
        读取所选变量，并统一输出 [1,T,H,W,C]。
        修复点：严格按坐标名/坐标长度对齐轴；不再把 T 维错当成 H 维。
        """
        ds, tmp_used = _open_nc_with_compat(self.path, enable_compat=self.path_compat_workaround)
        try:
            thw_list: List[np.ndarray] = []
            for k in self.var_keys:
                raw = np.asarray(ds.variables[k][:])
                thw = self._reorder_to_THW(ds, k, raw, self.time_key, self.y_key, self.x_key)  # [T,H,W]
                if self.fill_value is not None:
                    thw = np.nan_to_num(thw, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
                thw_list.append(thw.astype(np.float32, copy=False))

            # 可选：附加涡量通道（中心差分）
            if self.want_omega:
                if not (self.u_key and self.v_key):
                    raise ValueError("want_omega=True 需要提供 u_key 与 v_key。")
                U = self._reorder_to_THW(ds, self.u_key, np.asarray(ds.variables[self.u_key][:]),
                                        self.time_key, self.y_key, self.x_key)
                V = self._reorder_to_THW(ds, self.v_key, np.asarray(ds.variables[self.v_key][:]),
                                        self.time_key, self.y_key, self.x_key)
                dv_dx = (np.roll(V, -1, axis=2) - np.roll(V, 1, axis=2)) / (2.0 * float(self.dx))
                du_dy = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2.0 * float(self.dy))
                omega = (dv_dx - du_dy).astype(np.float32)
                thw_list.append(omega)

            # 拼通道 → [T,H,W,C]
            thwc = np.stack(thw_list, axis=-1)  # float32
            arr5d = thwc[None, ...]             # [1,T,H,W,C]

            # times
            times = self._meta.times
            if (times is None) and (self.time_key and self.time_key in ds.variables):
                try:
                    times = np.asarray(ds.variables[self.time_key][:]).reshape(-1)
                except Exception:
                    times = None

            self._meta = DataMeta(times=times, attrs={
                "source": "nc",
                "path": self.path,
                "tmp_used": bool(tmp_used),
            })
            return arr5d
        finally:
            ds.close()

@register_reader("nc")
def _build_nc_reader(**kwargs) -> NCReader:
    return NCReader(**kwargs)