from __future__ import annotations

from pathlib import Path
import argparse

import h5py
import netCDF4 as nc
import numpy as np


def build_h5_mini(src: Path, dst: Path, t_keep: int = 16, spatial_stride: int = 2) -> None:
    """从 H5 原始样本裁剪时间与空间维，生成 mini 版本。"""
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        groups = sorted([k for k in fin.keys() if isinstance(fin[k], h5py.Group)])
        if not groups:
            raise ValueError(f"No groups found in {src}")
        g0 = groups[0]
        data = fin[f"{g0}/data"][:t_keep, ::spatial_stride, ::spatial_stride, :]
        t = fin[f"{g0}/grid/t"][:t_keep]
        x = fin[f"{g0}/grid/x"][::spatial_stride]
        y = fin[f"{g0}/grid/y"][::spatial_stride]

        out_group = fout.create_group("0000")
        out_group.create_dataset("data", data=data.astype(np.float32), compression="gzip", compression_opts=4)
        grid = out_group.create_group("grid")
        grid.create_dataset("t", data=t.astype(np.float32))
        grid.create_dataset("x", data=x.astype(np.float32))
        grid.create_dataset("y", data=y.astype(np.float32))


def build_nc_mini(src: Path, dst: Path, t_keep: int = 12, x_stride: int = 2) -> None:
    """从 NetCDF 数据裁剪时间轴并下采样 x 方向，生成 mini 文件。"""
    with nc.Dataset(src, "r") as fin, nc.Dataset(dst, "w", format="NETCDF4") as fout:
        tdim = min(t_keep, len(fin.dimensions["tdim"]))
        ydim = len(fin.dimensions["ydim"])
        xdim = (len(fin.dimensions["xdim"]) + x_stride - 1) // x_stride

        fout.createDimension("tdim", tdim)
        fout.createDimension("ydim", ydim)
        fout.createDimension("xdim", xdim)
        for dname, dim in fin.dimensions.items():
            if dname not in ("tdim", "ydim", "xdim"):
                fout.createDimension(dname, len(dim))

        tv = fout.createVariable("tdim", "f4", ("tdim",))
        yv = fout.createVariable("ydim", "f4", ("ydim",))
        xv = fout.createVariable("xdim", "f4", ("xdim",))
        tv[:] = fin.variables["tdim"][:tdim]
        yv[:] = fin.variables["ydim"][:]
        xv[:] = fin.variables["xdim"][::x_stride]

        uv = fout.createVariable("u", "f4", ("tdim", "ydim", "xdim"), zlib=True, complevel=4)
        vv = fout.createVariable("v", "f4", ("tdim", "ydim", "xdim"), zlib=True, complevel=4)
        uv[:] = fin.variables["u"][:tdim, :, ::x_stride]
        vv[:] = fin.variables["v"][:tdim, :, ::x_stride]

        for key in ("nu", "radius", "Re"):
            if key in fin.variables:
                dims = fin.variables[key].dimensions
                out = fout.createVariable(key, "f4", dims)
                out[:] = fin.variables[key][:]


def build_mat_mini(src: Path, dst: Path, t_keep: int = 12, h_stride: int = 2, w_stride: int = 2) -> None:
    """从 MAT 数据裁剪时间与网格分辨率，生成 mini 数据文件。"""
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        lon = np.asarray(fin["lon"])
        lat = np.asarray(fin["lat"])
        sst = np.asarray(fin["sst"])
        time = np.asarray(fin["time"])

        w = lon.shape[1]
        h = lat.shape[1]
        t = min(t_keep, sst.shape[0])

        sst_thw = sst[:t].reshape(t, h, w)
        sst_thw = sst_thw[:, ::h_stride, ::w_stride]
        h2, w2 = sst_thw.shape[1], sst_thw.shape[2]
        sst_out = sst_thw.reshape(t, h2 * w2)

        lon_out = lon[:, ::w_stride]
        lat_out = lat[:, ::h_stride]
        time_out = time[:, :t]

        fout.create_dataset("lon", data=lon_out.astype(np.float32), compression="gzip", compression_opts=4)
        fout.create_dataset("lat", data=lat_out.astype(np.float32), compression="gzip", compression_opts=4)
        fout.create_dataset("time", data=time_out.astype(np.float64), compression="gzip", compression_opts=4)
        fout.create_dataset("sst", data=sst_out.astype(np.float32), compression="gzip", compression_opts=4)


def main() -> None:
    """CLI 入口：在目标目录构建三种 mini 数据集。"""
    parser = argparse.ArgumentParser(description="Build tiny mini datasets under datasets/")
    parser.add_argument("--src-dir", type=str, default="testdata")
    parser.add_argument("--dst-dir", type=str, default="datasets")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    build_h5_mini(src_dir / "2D_rdb_NA_NA_1pct.h5", dst_dir / "2D_rdb_NA_NA_mini.h5")
    build_nc_mini(src_dir / "cylinder2d_subset_10pct.nc", dst_dir / "cylinder2d_mini.nc")
    build_mat_mini(src_dir / "sst_weekly_10pct.mat", dst_dir / "sst_weekly_mini.mat")

    for p in [
        dst_dir / "2D_rdb_NA_NA_mini.h5",
        dst_dir / "cylinder2d_mini.nc",
        dst_dir / "sst_weekly_mini.mat",
    ]:
        print(f"{p}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
