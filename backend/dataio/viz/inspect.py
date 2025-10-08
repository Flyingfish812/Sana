# backend/dataio/viz/inspect.py
from __future__ import annotations
from typing import Optional, Dict, Any, Union, Sequence, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from ..schema import ArraySample, AdapterOutput
from ..dataset.unified import UnifiedDataset

# 放在文件顶部 import 之后
def _as_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

def _infer_tensor_format(arr: np.ndarray) -> str:
    """
    仅用于打印提示的简单格式推断：
      2 -> HW
      3 -> CHW
      4 -> BCHW
      5 -> BCTHW（若第二维较小，通常是 C 或 T）
    """
    nd = arr.ndim
    if nd == 2: return "HW"
    if nd == 3: return "CHW"
    if nd == 4: return "BCHW"
    if nd == 5: return "BCTHW"
    return f"{nd}D"

def print_array5d_info(arr: np.ndarray, title: str = "Array5D"):
    assert arr.ndim == 5
    N,T,H,W,C = arr.shape
    print(f"{title}: shape=[{N},{T},{H},{W},{C}], dtype={arr.dtype}")

def print_dataset_info(dataset: UnifiedDataset):
    arr = dataset.array5d
    print_array5d_info(arr, "dataset.array5d")
    chans = None
    if hasattr(dataset, "meta") and dataset.meta and isinstance(getattr(dataset, "meta"), object):
        # 兼容不同版本：meta 可能在 dataset 或 dataset.reader/meta
        try:
            chans = dataset.meta.attrs.get("channels")
        except Exception:
            chans = None
    print(f"#samples: {len(dataset)}; has transforms: {dataset.transforms is not None}")
    if chans:
        print(f"channels: {chans}")

def _to_numpy_img(x):
    # 支持 torch/np；[C,H,W] -> HxW
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 3:  # [C,H,W]
        # 默认展示第0通道
        return x[0]
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"Unsupported ndim: {x.ndim}")
    
def _resolve_channel(channel: Optional[Union[int, str]],
                     channels_meta: Optional[Sequence[str]]) -> Tuple[int, str]:
    """
    将 channel 参数解析成 (index, name)：
      - channel 为 int：直接返回该索引与相应名字（若有）
      - channel 为 str：在 channels_meta 中查同名，返回其索引
      - channel 为空：若 channels_meta 含 'omega' 则优先取 'omega'，否则取 0
    """
    # 默认
    if channels_meta is None or len(channels_meta) == 0:
        if isinstance(channel, int):
            return channel, f"ch{channel}"
        if isinstance(channel, str):
            return 0, channel
        return 0, "ch0"

    names = list(channels_meta)
    if isinstance(channel, int):
        idx = int(channel)
        name = names[idx] if 0 <= idx < len(names) else f"ch{idx}"
        return idx, name

    if isinstance(channel, str):
        if channel in names:
            return names.index(channel), channel
        # 支持特殊别名
        if channel.lower() in ("last", "omega") and "omega" in names:
            return names.index("omega"), "omega"
        if channel.lower() == "last":
            return len(names)-1, names[-1]
        # 找不到就回落到 0
        return 0, names[0]

    # 未指定：优先 omega
    if "omega" in names:
        return names.index("omega"), "omega"
    return 0, names[0]

def _extract_points_from_cond(cond: Any, idx: int = 0) -> Optional[np.ndarray]:
    """
    从 batch['cond'] 中尽力提取采样点坐标 (M,2) (y,x)：
      - 若 cond 是 dict 且含 'points'：取 [idx] 或自身（(M,2)）
      - 若 cond 是 dict 且含 'mask'：从 [B,1,H,W]/[B,H,W] 中索引 idx 并 where>0
      - 若 cond 是张量/数组：同上做 mask 推断
      - 否则返回 None
    """
    # dict: points
    if isinstance(cond, dict) and "points" in cond:
        pts = cond["points"]
        if isinstance(pts, (list, tuple)):
            pts = pts[idx]
        pts = _as_numpy(pts)
        return pts if (pts is not None and pts.ndim == 2 and pts.shape[1] == 2) else None

    # dict: mask
    mask = None
    if isinstance(cond, dict) and "mask" in cond:
        mask = _as_numpy(cond["mask"])
    elif cond is not None:
        mask = _as_numpy(cond)

    if mask is not None:
        if mask.ndim == 4:      # [B,1,H,W] 或 [B,H,W,C]
            m = mask[idx]
            if m.shape[0] == 1: m = m[0]
        elif mask.ndim == 3:    # [B,H,W]
            m = mask[idx]
        elif mask.ndim == 2:    # [H,W]
            m = mask
        else:
            return None
        yy, xx = np.where(m > 0.5)
        return np.stack([yy, xx], axis=1) if yy.size > 0 else None

    return None

def show_adapter_pair(adapter_fn, sample: ArraySample, channel: Optional[Union[int,str]] = None, title: str = ""):
    out: AdapterOutput = adapter_fn(sample)
    x = out.x; y = out.y
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()

    # 解析通道（从 sample.meta 或 adapter 输出里拿 channels）
    channels_meta = None
    try:
        channels_meta = sample.meta.attrs.get("channels")
    except Exception:
        channels_meta = None
    ch_idx, ch_name = _resolve_channel(channel, channels_meta)

    x_img = x[ch_idx] if x.ndim == 3 else x[0]
    y_img = y[ch_idx] if y.ndim == 3 else y[0]
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].imshow(x_img); axes[0].set_title(f"input (adapter) [{ch_name}]")
    axes[1].imshow(y_img); axes[1].set_title(f"target [{ch_name}]")
    fig.suptitle(title or "adapter input vs target")
    plt.show()

def show_batch_pair(batch: Dict[str, Any], channel: Optional[Union[int,str]] = None, title: str = ""):
    x = batch["x"][0]; y = batch["y"][0]  # 取第一个样本
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()

    # 从 batch 里尽力找 channels_meta
    channels_meta = None
    if "meta" in batch and isinstance(batch["meta"], dict):
        channels_meta = batch["meta"].get("channels")
    ch_idx, ch_name = _resolve_channel(channel, channels_meta)

    x_img = x[ch_idx] if x.ndim == 3 else x[0]
    y_img = y[ch_idx] if y.ndim == 3 else y[0]

    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].imshow(x_img); axes[0].set_title(f"batch[0].input [{ch_name}]")
    axes[1].imshow(y_img); axes[1].set_title(f"batch[0].target [{ch_name}]")
    fig.suptitle(title or "batch input vs target")
    plt.show()

def quick_check_dataloader(dataloader, n: int = 2, channel: Optional[Union[int,str]] = None):
    """
    在 ipynb 里快速抽 n 个 batch 可视化 input/target 是否匹配。
    默认优先展示 'omega' 通道（若存在），否则 ch0。
    """
    it = iter(dataloader)
    for i in range(n):
        try:
            b = next(it)
        except StopIteration:
            break
        show_batch_pair(b, channel=channel, title=f"batch {i}")

def show_sampling_from_adapter(adapter_fn, sample: ArraySample,
                               channel: Optional[Union[int,str]] = None,
                               title: str = "sampling (adapter)"):
    out: AdapterOutput = adapter_fn(sample)

    channels_meta = None
    try:
        channels_meta = sample.meta.attrs.get("channels")
    except Exception:
        channels_meta = None
    ch_idx, ch_name = _resolve_channel(channel, channels_meta)

    def pick(img):
        if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
        if img.ndim == 3:   # [C,H,W]
            return img[ch_idx]
        if img.ndim == 2:   # [H,W]
            return img
        return img

    x = pick(out.x); y = pick(out.y)

    # 采样点
    pts = None
    if isinstance(out.cond, dict) and ("points" in out.cond):
        pts = _as_numpy(out.cond["points"])
    elif isinstance(out.cond, dict) and ("mask" in out.cond):
        pts = _extract_points_from_cond(out.cond, idx=0)

    _plot_triplet(x, y, pts, title=title, ch_name=ch_name)

def show_sampling_from_batch(batch: Dict[str, Any],
                             channel: Optional[Union[int,str]] = None,
                             idx: int = 0,
                             title: str = "sampling (batch)"):
    x = batch["x"][idx]; y = batch["y"][idx]; cond = batch.get("cond")
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()

    channels_meta = None
    if "meta" in batch and isinstance(batch["meta"], dict):
        channels_meta = batch["meta"].get("channels")
    ch_idx, ch_name = _resolve_channel(channel, channels_meta)

    x_img = x[ch_idx] if x.ndim == 3 else (x if x.ndim == 2 else None)
    y_img = y[ch_idx] if y.ndim == 3 else (y if y.ndim == 2 else None)

    # 采样点（优先 batch['sampling_points']）
    pts = None
    if "sampling_points" in batch:
        ptsi = batch["sampling_points"][idx]
        pts = _as_numpy(ptsi) if ptsi is not None else None
    if pts is None:
        pts = _extract_points_from_cond(cond, idx=idx)

    _plot_triplet(x_img, y_img, pts, title=title, ch_name=ch_name)

def summarize_dataloader(dl) -> Tuple[int, int]:
    """
    打印 dataloader 的 batch 数与首个 batch 的 batch_size，返回 (num_batches, batch_size_first)
    """
    try:
        num_batches = len(dl)
    except TypeError:
        # 某些自定义 DataLoader 不支持 len；遍历一次（小数据可接受）
        num_batches = sum(1 for _ in dl)

    it = iter(dl)
    try:
        b0 = next(it)
    except StopIteration:
        print("[dataloader] 空！")
        return 0, 0

    x0 = _as_numpy(b0.get("x"))
    bs = int(x0.shape[0]) if x0 is not None else 0
    print(f"[dataloader] num_batches={num_batches}, batch_size(first)={bs}")
    return num_batches, bs

def _plot_triplet(x_img: np.ndarray,
                  y_img: np.ndarray,
                  pts: Optional[np.ndarray],
                  title: str,
                  ch_name: str):
    """
    1) input 单通道
    2) output 单通道
    3) output + 采样点（红色）
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(x_img); axes[0].set_title(f"input [{ch_name}]")
    axes[1].imshow(y_img); axes[1].set_title(f"output [{ch_name}]")
    axes[2].imshow(y_img); axes[2].set_title("output + samples")
    if pts is not None and pts.size > 0:
        axes[2].scatter(pts[:, 1], pts[:, 0], s=6, c="red", marker="o")  # 红色点
    fig.suptitle(title)
    plt.show()

def check_dataloader(dl, n_batches: int = 2, channel: Optional[Union[int,str]] = None):
    it = iter(dl)
    chans_meta = None
    for i in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            if i == 0:
                print("[dataloader] 空！")
            break

        x = _as_numpy(batch.get("x"))
        y = _as_numpy(batch.get("y"))
        cond = batch.get("cond", None)

        # 形状与格式打印
        if x is not None:
            print(f"[batch {i}] x.shape={list(x.shape)} -> {_infer_tensor_format(x)}")
        if y is not None:
            print(f"[batch {i}] y.shape={list(y.shape)} -> {_infer_tensor_format(y)}")
        if isinstance(cond, dict) and "mask" in cond:
            m = _as_numpy(cond["mask"])
            print(f"[batch {i}] cond['mask'].shape={list(m.shape)} -> {_infer_tensor_format(m)}")
        elif cond is not None:
            m = _as_numpy(cond)
            print(f"[batch {i}] cond.shape={list(m.shape)} -> {_infer_tensor_format(m)}")

        # 通道名（默认优先 omega）
        if chans_meta is None and isinstance(batch.get("meta"), dict):
            chans_meta = batch["meta"].get("channels")
        ch_idx, ch_name = _resolve_channel(channel, chans_meta)

        # 取第 0 个样本的单通道图像
        def pick_img(t4):
            if t4 is None: return None
            if t4.ndim == 4:  # [B,C,H,W]
                C = t4.shape[1]; idx = min(max(0, ch_idx), C-1)
                return t4[0, idx]
            if t4.ndim == 3:  # [C,H,W]
                C = t4.shape[0]; idx = min(max(0, ch_idx), C-1)
                return t4[idx]
            if t4.ndim == 2:  # [H,W]
                return t4
            return None

        x_img = pick_img(x)
        y_img = pick_img(y)

        # 采样点（优先 batch['sampling_points']，其次 cond 推断）
        pts = None
        if "sampling_points" in batch:
            pts0 = batch["sampling_points"][0]
            pts = _as_numpy(pts0) if pts0 is not None else None
        if pts is None:
            pts = _extract_points_from_cond(cond, idx=0)

        _plot_triplet(x_img, y_img, pts, title=f"batch {i}", ch_name=ch_name)

def print_split_summary(loaders_or_splits_summary, *, indent: str = ""):
    """
    接受两类输入：
      - loaders dict: {"train": DataLoader, "val":..., "test":...}
      - summary dict: {"sizes": {...}, "ratios": {...}, "total": ...}
    打印每个 split 的样本数与占比。
    """
    sizes = None; ratios = None; total = None
    if isinstance(loaders_or_splits_summary, dict) and "sizes" in loaders_or_splits_summary:
        sizes  = loaders_or_splits_summary.get("sizes", {})
        ratios = loaders_or_splits_summary.get("ratios", {})
        total  = loaders_or_splits_summary.get("total", None)
    elif isinstance(loaders_or_splits_summary, dict):
        # 认为是 loaders dict
        sizes = {k: len(v.dataset) if hasattr(v, "dataset") else None for k, v in loaders_or_splits_summary.items()}
        total = sum(s for s in sizes.values() if s is not None)
        ratios = {k: (sizes[k]/total if (total and sizes[k] is not None) else None) for k in sizes}
    else:
        print(f"{indent}[split] 不支持的类型：{type(loaders_or_splits_summary)}")
        return

    print(f"{indent}[split] total samples: {total}")
    for name in ("train","val","test","all"):
        if name in sizes:
            r = ratios.get(name, None)
            rtxt = f"{r*100:.1f}%" if isinstance(r, (int,float)) else "n/a"
            print(f"{indent}  - {name:<5}: {sizes[name]:>6}  ({rtxt})")

def one_click_check(dataset, dl, channel: Optional[Union[int,str]] = None, n_batches: int = 2, with_sizecheck: bool = True):
    """
    支持：
      - dl 为单个 DataLoader（老行为）
      - dl 为 dict：{"train":..., "val":..., "test":...}
    """
    # 1) array5d
    print_dataset_info(dataset)

    # 2) split 概况
    if isinstance(dl, dict):
        print_split_summary(dl, indent="")
        # 默认从 train 抽样可视化；不存在则挑第一个
        pick_key = "train" if "train" in dl else next(iter(dl.keys()))
        dl_vis = dl[pick_key]
        print(f"\n[visualize] 使用 split='{pick_key}' 进行可视化")
        # 2.5 size 概况（取该 split 的 loader）
        if with_sizecheck:
            try:
                from .sizecheck import sanity_check_dataset
                print("\n[sizecheck] —— 估算首个 batch 体积 ——")
                _ = sanity_check_dataset(dataset, dl_vis, print_first_batch=False)
            except Exception as e:
                print(f"[sizecheck] 跳过：{e}")
        # 3) 可视化
        check_dataloader(dl_vis, n_batches=n_batches, channel=channel)
    else:
        # 单一 DataLoader
        summarize_dataloader(dl)
        if with_sizecheck:
            try:
                from .sizecheck import sanity_check_dataset
                print("\n[sizecheck] —— 估算首个 batch 体积 ——")
                _ = sanity_check_dataset(dataset, dl, print_first_batch=False)
            except Exception as e:
                print(f"[sizecheck] 跳过：{e}")
        print("\n[visualize] —— 输入/目标/采样点 ——")
        check_dataloader(dl, n_batches=n_batches, channel=channel)
