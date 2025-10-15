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
