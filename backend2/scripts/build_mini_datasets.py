from __future__ import annotations

from pathlib import Path
import argparse

import h5py
import netCDF4 as nc
import numpy as np


MAX_BYTES_DEFAULT = 10 * 1024 * 1024


def _select_even_indices(total: int, keep: int) -> np.ndarray:
    if total <= 0:
        return np.zeros((0,), dtype=np.int64)
    keep = int(max(1, min(keep, total)))
    if keep == total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=keep, dtype=np.int64)


def _cap_keep_by_max_bytes(estimated_bytes_per_item: int, keep: int, max_bytes: int) -> int:
    if estimated_bytes_per_item <= 0:
        return keep
    max_keep = max(1, int(max_bytes // estimated_bytes_per_item))
    return max(1, min(int(keep), max_keep))


def build_h5_mini(src: Path, dst: Path, group_ratio: float = 0.001, max_bytes: int = MAX_BYTES_DEFAULT) -> None:
    """从 H5 原始样本按组抽样生成 mini 版本（不做下采样）。"""
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        groups = sorted([k for k in fin.keys() if isinstance(fin[k], h5py.Group)])
        if not groups:
            raise ValueError(f"No groups found in {src}")
        one = fin[f"{groups[0]}/data"]
        bytes_per_group = int(np.prod(one.shape) * np.dtype(one.dtype).itemsize)

        keep = int(round(float(group_ratio) * len(groups)))
        keep = max(1, min(keep, len(groups)))
        keep = _cap_keep_by_max_bytes(bytes_per_group, keep, max_bytes)
        pick = _select_even_indices(len(groups), keep)

        for out_i, idx in enumerate(pick):
            g_name = groups[int(idx)]
            data = fin[f"{g_name}/data"][:]
            t = fin[f"{g_name}/grid/t"][:]
            x = fin[f"{g_name}/grid/x"][:]
            y = fin[f"{g_name}/grid/y"][:]

            out_group = fout.create_group(f"{out_i:04d}")
            out_group.create_dataset("data", data=data.astype(np.float32), compression="gzip", compression_opts=4)
            grid = out_group.create_group("grid")
            grid.create_dataset("t", data=t.astype(np.float32))
            grid.create_dataset("x", data=x.astype(np.float32))
            grid.create_dataset("y", data=y.astype(np.float32))


def build_nc_mini(src: Path, dst: Path, t_ratio: float = 0.01, max_bytes: int = MAX_BYTES_DEFAULT) -> None:
    """从 NetCDF 数据按时间轴抽样生成 mini 文件（不做空间下采样）。"""
    with nc.Dataset(src, "r") as fin, nc.Dataset(dst, "w", format="NETCDF4") as fout:
        t_total = len(fin.dimensions["tdim"])
        tdim = int(round(float(t_ratio) * t_total))
        tdim = max(2, min(tdim, t_total))

        ydim = len(fin.dimensions["ydim"])
        xdim = len(fin.dimensions["xdim"])

        bytes_per_t = int(ydim * xdim * 2 * np.dtype(np.float32).itemsize)
        tdim = _cap_keep_by_max_bytes(bytes_per_t, tdim, max_bytes)
        t_indices = _select_even_indices(t_total, tdim)

        fout.createDimension("tdim", tdim)
        fout.createDimension("ydim", ydim)
        fout.createDimension("xdim", xdim)
        for dname, dim in fin.dimensions.items():
            if dname not in ("tdim", "ydim", "xdim"):
                fout.createDimension(dname, len(dim))

        tv = fout.createVariable("tdim", "f4", ("tdim",))
        yv = fout.createVariable("ydim", "f4", ("ydim",))
        xv = fout.createVariable("xdim", "f4", ("xdim",))
        tv[:] = np.asarray(fin.variables["tdim"][:])[t_indices]
        yv[:] = fin.variables["ydim"][:]
        xv[:] = fin.variables["xdim"][:]

        uv = fout.createVariable("u", "f4", ("tdim", "ydim", "xdim"), zlib=True, complevel=4)
        vv = fout.createVariable("v", "f4", ("tdim", "ydim", "xdim"), zlib=True, complevel=4)
        u_all = np.asarray(fin.variables["u"][:])
        v_all = np.asarray(fin.variables["v"][:])
        uv[:] = u_all[t_indices, :, :]
        vv[:] = v_all[t_indices, :, :]

        for key in ("nu", "radius", "Re"):
            if key in fin.variables:
                dims = fin.variables[key].dimensions
                out = fout.createVariable(key, "f4", dims)
                out[:] = fin.variables[key][:]


def build_mat_mini(src: Path, dst: Path, t_ratio: float = 0.01, max_bytes: int = MAX_BYTES_DEFAULT) -> None:
    """从 MAT 数据按时间轴抽样生成 mini 文件（不做空间下采样）。"""
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        lon = np.asarray(fin["lon"])
        lat = np.asarray(fin["lat"])
        sst = np.asarray(fin["sst"])
        time = np.asarray(fin["time"])

        t_total = int(sst.shape[0])
        hw = int(sst.shape[1])
        t = int(round(float(t_ratio) * t_total))
        t = max(2, min(t, t_total))
        bytes_per_t = int(hw * np.dtype(sst.dtype).itemsize)
        t = _cap_keep_by_max_bytes(bytes_per_t, t, max_bytes)
        t_idx = _select_even_indices(t_total, t)

        sst_out = sst[t_idx, :]
        lon_out = lon
        lat_out = lat
        time_out = time[:, t_idx]

        fout.create_dataset("lon", data=lon_out.astype(np.float32), compression="gzip", compression_opts=4)
        fout.create_dataset("lat", data=lat_out.astype(np.float32), compression="gzip", compression_opts=4)
        fout.create_dataset("time", data=time_out.astype(np.float64), compression="gzip", compression_opts=4)
        fout.create_dataset("sst", data=sst_out.astype(np.float32), compression="gzip", compression_opts=4)


def build_h5_oneshot_clone(src: Path, dst: Path, t_repeat: int = 1000) -> None:
    """从 full H5 中取单帧并复制为长度 t_repeat 的序列，构造单样本过拟合集。"""
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        groups = sorted([k for k in fin.keys() if isinstance(fin[k], h5py.Group)])
        if not groups:
            raise ValueError(f"No groups found in {src}")

        g0 = groups[0]
        data0 = np.asarray(fin[f"{g0}/data"])  # [T,H,W,C]
        if data0.ndim != 4 or data0.shape[0] < 1:
            raise ValueError(f"unexpected shape for {g0}/data: {data0.shape}")

        t_mid = int(data0.shape[0] // 2)
        frame = data0[t_mid : t_mid + 1].astype(np.float32, copy=False)  # [1,H,W,C]
        data_clone = np.repeat(frame, repeats=int(max(2, t_repeat)), axis=0)

        x = np.asarray(fin[f"{g0}/grid/x"][:], dtype=np.float32)
        y = np.asarray(fin[f"{g0}/grid/y"][:], dtype=np.float32)
        t = np.arange(data_clone.shape[0], dtype=np.float32)

        out_group = fout.create_group("0000")
        out_group.create_dataset("data", data=data_clone, compression="gzip", compression_opts=4)
        grid = out_group.create_group("grid")
        grid.create_dataset("t", data=t)
        grid.create_dataset("x", data=x)
        grid.create_dataset("y", data=y)


def main() -> None:
    """CLI 入口：在目标目录构建三种 mini 数据集 + 单样本复制过拟合集。"""
    parser = argparse.ArgumentParser(description="Build tiny mini datasets by sampling full datasets (no downsampling)")
    parser.add_argument("--src-dir", type=str, default="datasets")
    parser.add_argument("--dst-dir", type=str, default="datasets")
    parser.add_argument("--max-mb", type=float, default=10.0)
    parser.add_argument("--h5-group-ratio", type=float, default=0.001)
    parser.add_argument("--nc-time-ratio", type=float, default=0.01)
    parser.add_argument("--mat-time-ratio", type=float, default=0.01)
    parser.add_argument("--oneshot-repeat", type=int, default=1000)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(float(args.max_mb) * 1024 * 1024)

    if not (0.001 <= float(args.h5_group_ratio) <= 0.01):
        raise ValueError("--h5-group-ratio must be in [0.001, 0.01]")
    if not (0.001 <= float(args.nc_time_ratio) <= 0.01):
        raise ValueError("--nc-time-ratio must be in [0.001, 0.01]")
    if not (0.001 <= float(args.mat_time_ratio) <= 0.01):
        raise ValueError("--mat-time-ratio must be in [0.001, 0.01]")

    build_h5_mini(
        src_dir / "2D_rdb_NA_NA.h5",
        dst_dir / "2D_rdb_NA_NA_mini.h5",
        group_ratio=float(args.h5_group_ratio),
        max_bytes=max_bytes,
    )
    build_nc_mini(
        src_dir / "cylinder2d.nc",
        dst_dir / "cylinder2d_mini.nc",
        t_ratio=float(args.nc_time_ratio),
        max_bytes=max_bytes,
    )
    build_mat_mini(
        src_dir / "sst_weekly.mat",
        dst_dir / "sst_weekly_mini.mat",
        t_ratio=float(args.mat_time_ratio),
        max_bytes=max_bytes,
    )
    build_h5_oneshot_clone(
        src_dir / "2D_rdb_NA_NA.h5",
        dst_dir / "2D_rdb_NA_NA_oneshot_clone.h5",
        t_repeat=int(args.oneshot_repeat),
    )

    for p in [
        dst_dir / "2D_rdb_NA_NA_mini.h5",
        dst_dir / "cylinder2d_mini.nc",
        dst_dir / "sst_weekly_mini.mat",
        dst_dir / "2D_rdb_NA_NA_oneshot_clone.h5",
    ]:
        print(f"{p}: {p.stat().st_size} bytes ({p.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
