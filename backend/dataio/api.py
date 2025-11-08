# backend/dataio/api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import os
from torch.utils.data import DataLoader
from .registry import build_reader, build_sampler, build_adapter
from .transforms import (
    Compose,
    NormalizeTransform,
    AddCoordsTransform,
    AddTimeEncodingTransform,
    ToTensorTransform,
    FillNaNTransform,
    CaptureNaNMaskTransform,
)
from .dataset.unified import UnifiedDataset
from .dataset.collate import make_collate
from .schema import DataMeta
from .dataset.subset import SubsetDataset
from .sampling.splits import split_indices
from backend.train.data_adapter_snapshot import build_from_snapshot

@dataclass
class DatasetBuildResult:
    array5d_shape: Tuple[int, int, int, int, int]
    meta: DataMeta
    normalizer_method: Optional[str]
    num_samples: int

def _build_transforms(cfg: Dict[str, Any]) -> Compose:
    """
    从 config 构造 transforms 管线。
    支持字段：
      normalize: { method: "zscore"|"minmax" }
      add_coords: true|false
      add_time_encoding: true|false
      to_tensor: true|false   # 若希望在 transforms 阶段即转为 torch.Tensor
    顺序：Normalize -> AddCoords -> AddTimeEncoding -> ToTensor
    """
    ts = []
    cap_cfg = cfg.get("capture_nan_mask")
    if cap_cfg:
        ts.append(CaptureNaNMaskTransform(
            target_only=bool(cap_cfg.get("target_only", True)),
            reduce_channel=str(cap_cfg.get("reduce_channel", "any"))
        ))
        
    fill_cfg = cfg.get("fillna")
    if fill_cfg:
        method = fill_cfg.get("method", "value")
        value = float(fill_cfg.get("value", 0.0))
        ts.append(FillNaNTransform(method=method, value=value))
    norm_cfg = cfg.get("normalize")
    if norm_cfg:
        ts.append(NormalizeTransform(method=norm_cfg.get("method", "zscore")))
    if cfg.get("add_coords"):
        ts.append(AddCoordsTransform())
    if cfg.get("add_time_encoding"):
        ts.append(AddTimeEncodingTransform())
    if cfg.get("to_tensor"):
        ts.append(ToTensorTransform())
    return Compose(ts)

# 放在 api.py 里（build_dataloader 同一文件顶部或紧邻处）
def _make_dataloader_kwargs(config, *, shuffle: bool, collate_fn):
    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", True))
    persistent_workers_cfg = bool(config.get("persistent_workers", True))
    prefetch_factor_cfg = config.get("prefetch_factor", None)

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 仅当 num_workers > 0 时才启用 persistent_workers / prefetch_factor
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers_cfg
        if prefetch_factor_cfg is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor_cfg)
    # 否则不要传 persistent_workers（避免报错）

    return kwargs

def build_dataset(config: Dict[str, Any]) -> Tuple[UnifiedDataset, DatasetBuildResult]:
    """
    1) Reader 装载 + probe
    2) Sampler 产生 SampleSpec 列表
    3) Transforms 组装
    4) UnifiedDataset 构造
    """
    r_cfg = config["reader"]            # e.g., {"kind":"h5", "path":"...", ...}
    s_cfg = config.get("sampler", {"kind": "static"})
    t_cfg = config.get("transforms", {})

    reader = build_reader(**r_cfg)      # 注意：register_reader 采用 kwargs: kind=..., 自行在注册时封装
    shape5d, meta = reader.probe()
    array5d = reader.read_array5d()     # 也可以做 lazy；当前示例直接读取全量

    sampler = build_sampler(**s_cfg, shape5d=shape5d) if "shape5d" not in s_cfg else build_sampler(**s_cfg)
    transforms = _build_transforms(t_cfg)

    dataset = UnifiedDataset(array5d=array5d, meta=meta, specs=list(sampler), transforms=transforms)
    result = DatasetBuildResult(
        array5d_shape=array5d.shape, meta=meta,
        normalizer_method=t_cfg.get("normalize", {}).get("method") if t_cfg.get("normalize") else None,
        num_samples=len(dataset),
    )
    return dataset, result

def build_dataloader(config: Dict[str, Any], dataset: UnifiedDataset):
    """
    1) 选择 Adapter（默认 static2d）
    2) 构造 collate_fn
    3) 若开启 split：返回 {"train": dl, "val": dl, "test": dl}
       否则：返回单个 DataLoader
    """
    a_cfg = config.get("adapter", {"kind": "static2d"})
    default_shuffle = bool(config.get("shuffle", True))

    adapter_fn = build_adapter(**a_cfg)   # register_adapter 已装饰函数，返回可调用
    collate_fn = make_collate(adapter_fn)

    # 如果启用切分，则委托到 build_dataloaders_with_splits
    split_cfg = (config.get("split") or {})
    if bool(split_cfg.get("enable", False)):
        loaders, _summary = build_dataloaders_with_splits(config, dataset, collate_fn)
        return loaders  # dict: {"train":..., "val":..., "test":...}

    # 否则返回单一 DataLoader（向后兼容原行为）
    kwargs = _make_dataloader_kwargs(config, shuffle=default_shuffle, collate_fn=collate_fn)
    return DataLoader(dataset, **kwargs)

def build_dataloaders_with_splits(cfg: dict, dataset, collate_fn):
    """
    读取 cfg['split']，基于 dataset 切分索引，构造 train/val/test 三个 DataLoader。
    返回：dict(loaders), split_summary
    """
    split_cfg = cfg.get("split", {}) or {}
    enable = bool(split_cfg.get("enable", True))
    if not enable:
        # 退回单一 DataLoader（与原行为一致）
        kwargs = _make_dataloader_kwargs(cfg, shuffle=bool(cfg.get("shuffle", False)), collate_fn=collate_fn)
        dl = DataLoader(dataset, **kwargs)
        return {"all": dl}, {"mode": "all", "sizes": {"all": len(dataset)}, "ratios": {"all": 1.0}}

    # 1) 生成切分索引
    splits = split_indices(
        dataset,
        strategy = split_cfg.get("strategy", "temporal"),  # "temporal" | "random"
        ratios   = split_cfg.get("ratios", {"train":0.8,"val":0.1,"test":0.1}),
        unit     = split_cfg.get("unit", "frame"),         # "frame" | "sequence"
        seed     = int(split_cfg.get("seed", 123)),
    )

    # 2) 包装成子集 dataset
    subsets = {name: SubsetDataset(dataset, idxs) for name, idxs in splits.items()}

    # 3) 为不同 split 设置 shuffle（train=True, 其他=False），并用安全 kwargs
    loaders = {}
    for name, ds_sub in subsets.items():
        kwargs = _make_dataloader_kwargs(cfg, shuffle=(name == "train"), collate_fn=collate_fn)
        loaders[name] = DataLoader(ds_sub, **kwargs)

    # 4) 汇总信息
    total = sum(len(v) for v in splits.values())
    sizes = {k: len(v) for k, v in splits.items()}
    ratios = {k: (sizes[k]/total if total>0 else 0.0) for k in sizes}

    summary = {
        "mode": f"{split_cfg.get('strategy','temporal')}/{split_cfg.get('unit','frame')}",
        "sizes": sizes,
        "ratios": ratios,
        "total": total
    }
    return loaders, summary

def summarise_pipeline(dataset_result: DatasetBuildResult, dataloader):
    """
    提供一个简易摘要（可扩展为落盘 summary.json / meta.json 等）
    """
    summary = {
        "array5d_shape": list(dataset_result.array5d_shape),
        "normalizer": dataset_result.normalizer_method,
        "num_samples": dataset_result.num_samples,
        "num_batches": len(dataloader),
        "batch_size": dataloader.batch_size,
    }
    return summary

def get_base_dataset_from_snapshot(
    snapshot_dir: str,
    *,
    max_ram_gb: Optional[float] = None,
    force_streaming: bool = False,
):
    from .cache.snapshot import load_snapshot_as_base_dataset
    dataset, info = load_snapshot_as_base_dataset(
        snapshot_dir,
        max_ram_gb=max_ram_gb,
        force_streaming=force_streaming,
    )
    return dataset, info

def run(config: Dict[str, Any]):
    """
    轻量化一键入口：只需准备一个 config，即可：
      Reader → Sampler → Transforms → Dataset → Adapter/Collate → DataLoader(s) → Summary
    返回 (dataset, dataloader_or_dict, summary_dict)
    - 若未开启 split：dataloader 为单个 DataLoader，summary 为单份摘要
    - 若开启 split：dataloader 为 {train/val/test} 字典，summary 包含每个 split 的摘要与全局切分统计
    """
    dataset, info = build_dataset(config)
    dataloader = build_dataloader(config, dataset)  # 允许内部按 config['split'] 返回 dict

    # --- 汇总 summary ---
    def _summarise_one(dl):
        return summarise_pipeline(info, dl)

    summary: Dict[str, Any] = {}
    split_cfg = (config.get("split") or {})
    split_enabled = bool(split_cfg.get("enable", False))

    if not split_enabled or not isinstance(dataloader, dict):
        # 单一 DataLoader
        summary = _summarise_one(dataloader)
    else:
        # 多 DataLoader（train/val/test）
        split_summaries = {}
        sizes, total = {}, 0
        for name, dl in dataloader.items():
            split_summaries[name] = _summarise_one(dl)
            # 统计样本数（尽量稳健）
            try:
                sizes[name] = len(dl.dataset)  # SubsetDataset.dataset / 普通 Dataset
            except Exception:
                # 兜底：按 batch 数 × batch_size 粗估
                try:
                    sizes[name] = len(dl) * next(iter(dl))["x"].shape[0]
                except Exception:
                    sizes[name] = None
            if isinstance(sizes[name], int):
                total += sizes[name]

        ratios = {
            k: (sizes[k] / total if (isinstance(sizes.get(k), int) and total > 0) else None)
            for k in sizes
        }

        summary = {
            "splits": split_summaries,
            "split_overview": {
                "mode": f"{split_cfg.get('strategy','temporal')}/{split_cfg.get('unit','frame')}",
                "sizes": sizes,
                "ratios": ratios,
                "total": total
            }
        }

    # --- 导出（支持单一或多 split）---
    output_cfg = config.get("output")
    if output_cfg:
        from copy import deepcopy
        from .io.exporters import dump_prep_outputs

        if not (split_enabled and isinstance(dataloader, dict)):
            # 单一 DataLoader：原样导出
            dump_prep_outputs(dataset=dataset, dataloader=dataloader, summary=summary, output_cfg=output_cfg)
        else:
            # 多 split：分别落到 out_dir/<split>/
            base = deepcopy(output_cfg)
            base_out = base.get("out_dir", "./prep_out")
            for name, dl in dataloader.items():
                oc = deepcopy(base)
                oc["out_dir"] = os.path.join(base_out, name)
                # 为每个 split 单独带上它的 summary（若存在）
                s_one = summary["splits"].get(name, {})
                dump_prep_outputs(dataset=dataset, dataloader=dl, summary=s_one, output_cfg=oc)

            # 另外把全局切分概览也保存到 base_out 根目录（可选）
            try:
                os.makedirs(base_out, exist_ok=True)
                with open(os.path.join(base_out, "split_overview.json"), "w", encoding="utf-8") as f:
                    import json
                    json.dump(summary.get("split_overview", {}), f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    return dataset, dataloader, summary

def build_all(
    *,
    snapshot_dir: str,
    # 因子化注入（训练侧会把 data.factors 合并注入到这里）
    sample_density: Optional[float] = None,
    noise_sigma: Optional[float] = None,
    rng_seed_offset: int = 0,
    # DataLoader 相关参数（训练侧 data.* 会透传到这里；若未传则用默认）
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 4,
    drop_last: bool = False,
    shuffle: bool = True,
    split: Optional[Dict[str, Any]] = None,
    # 允许冗余参数（从 data_cfg 透传进来，不影响逻辑）
    **data_cfg,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    v2 统一数据入口（builder 模式）：
      1) 先用 v1 的 build_from_snapshot 恢复基础 train/val/test DataLoader；
      2) 在不改变样本划分的前提下，用本地 adapter collate 构造“三输入通道”（recon / mask / obs*mask），
         目标保持单通道（与 v1 兼容：target_channels=["u"]）；
      3) 返回新的 train/val/test 三个 DataLoader。
    """
    import random
    import torch

    # 容错：目录别名（用户若传了 nc_full，尝试 nc_full_v2）
    if not os.path.isdir(snapshot_dir):
        alt = snapshot_dir.rstrip("/\\") + "_v2"
        if os.path.isdir(alt):
            snapshot_dir = alt

    # ---------- 第一步：用 v1 的逻辑从快照恢复基础 dataloaders ----------
    # 注意：此处先不给任何自定义 collate，让它按快照恢复最原始的样本结构（dict：含 'u','v' 等）
    #       随后我们用自己的 collate 替换，以构造三输入通道。
    base_train, base_val, base_test = build_from_snapshot(snapshot_dir, {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "drop_last": drop_last,
        "shuffle": shuffle,
        "split": split or data_cfg.get("split") or {"enable": True, "strategy": "temporal",
                                                    "unit": "frame",
                                                    "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                                                    "seed": 123}
    })

    # ---------- 第二步：p-σ 因子注入 + 三输入通道（样本级映射，保持原 collate 不变） ----------
    import torch
    from torch.utils.data import Dataset

    base_channel = data_cfg.get("base_channel", "u")          # 与 v1 约定保持一致
    target_channels = data_cfg.get("target_channels", ["u"])  # 目前保持单通道预测

    rng = np.random.RandomState(1234 + int(rng_seed_offset))
    torch.manual_seed(1234 + int(rng_seed_offset))

    def _make_mask_like(arr_hw, p: Optional[float]):
        if p is None or p <= 0.0:
            return np.zeros(arr_hw, dtype=np.uint8)
        h, w = arr_hw
        num = int(round(p * h * w))
        idx = rng.choice(h * w, size=num, replace=False)
        mask = np.zeros(h * w, dtype=np.uint8)
        mask[idx] = 1
        return mask.reshape(h, w)

    def _naive_inpaint(observed_map, mask):
        if mask.sum() == 0:
            return np.zeros_like(observed_map, dtype=observed_map.dtype)
        mean_val = observed_map[mask > 0].mean()
        out = observed_map.copy()
        out[mask == 0] = mean_val
        return out

    def _to_numpy(arr):
        import numpy as _np
        import torch as _torch
        if isinstance(arr, _torch.Tensor):
            return arr.detach().cpu().numpy()
        return _np.asarray(arr)

    class _MappedDataset(Dataset):
        """对 base_dataset 做样本级映射：生成三输入通道 x 和单通道目标 y。
        不改 collate，让下游保持 batch.x / batch.y 的属性访问行为。"""
        def __init__(self, base_ds):
            self.base = base_ds
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            sample = self.base[idx]

            # 取“干净底图”u_map：兼容两种快照规范
            # A) {'u': HxW, 'v': HxW, ...}
            # B) {'x': CxHxW, 'y': C'xHxW}（干净底稿：用 y 的第 0 通道作为底图）
            if base_channel in sample:
                u_map = _to_numpy(sample[base_channel]).astype(np.float32)
                if u_map.ndim == 3 and u_map.shape[0] == 1:
                    u_map = u_map[0]
            elif "y" in sample:
                y0 = _to_numpy(sample["y"]).astype(np.float32)
                if y0.ndim == 2:
                    u_map = y0
                elif y0.ndim == 3:
                    u_map = y0[0]
                else:
                    raise ValueError(f"Unsupported 'y' shape {y0.shape} in snapshot sample.")
            else:
                raise KeyError(f"Snapshot sample missing base channel '{base_channel}' and key 'y'. Keys={list(sample.keys())}")

            H, W = u_map.shape[-2], u_map.shape[-1]
            mask = _make_mask_like((H, W), sample_density)

            obs = u_map.astype(np.float32).copy()
            if noise_sigma is not None and noise_sigma > 0.0:
                obs = obs + rng.normal(0.0, float(noise_sigma), size=obs.shape).astype(np.float32)

            obs_times_mask = obs * mask.astype(np.float32)
            recon = _naive_inpaint(obs_times_mask, mask)

            x = np.stack([recon, mask.astype(np.float32), obs_times_mask], axis=0).astype(np.float32)
            y = u_map.astype(np.float32)[None, ...]  # 单通道目标

            # IMPORTANT: 返回 dict（与 v1 快照样本契约一致：per-sample 是 dict）
            # 原有 collate 会把 batch 封装成支持 batch.x / batch.y 的对象；不要在这里改成别的类型。
            return {"x": x, "y": y}

    # ---------- 第三步：复用“原始 dataloader 的 collate_fn”，仅替换 dataset 与常用 loader 选项 ----------
    def _rebuild_like(base_dl: Optional[DataLoader]) -> Optional[DataLoader]:
        if base_dl is None:
            return None
        ds = _MappedDataset(base_dl.dataset)
        # 沿用 v1 的 collate_fn（它会把 batch 封装为支持属性访问的对象）
        collate_fn = base_dl.collate_fn

        # 覆盖常用选项；其余沿用基础 loader 的 worker 等特性
        opts = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers if (num_workers and num_workers > 0) else False),
            shuffle=getattr(base_dl, "shuffle", shuffle),
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        if prefetch_factor is not None and num_workers and num_workers > 0:
            opts["prefetch_factor"] = prefetch_factor
        return DataLoader(ds, **opts)

    train_dl = _rebuild_like(base_train)
    val_dl   = _rebuild_like(base_val)
    test_dl  = _rebuild_like(base_test or base_val)  # 若无 test，用 val 兜底

    return train_dl, val_dl, test_dl