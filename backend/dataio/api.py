# backend/dataio/api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import rank_zero_only
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

@rank_zero_only
def _export_data_outputs(dataset, dataloader, summary, output_cfg):
    """
    单独抽出 export 部分，保证只有 rank0 会调用 dump_prep_outputs。
    """
    from .io.exporters import dump_prep_outputs
    dump_prep_outputs(
        dataset=dataset,
        dataloader=dataloader,
        summary=summary,
        output_cfg=output_cfg,
    )

def run(config: Dict[str, Any]):
    """
    轻量化一键入口：只需准备一个 config，即可：
      Reader → Sampler → Transforms → Dataset → Adapter/Collate → DataLoader(s) → Summary
    返回 (dataset, dataloader_or_dict, summary_dict)

    多进程适配点：
      - dataset / dataloader 构造在所有 rank 上执行（必要）
      - 但 "output/export" 部分仅由 global rank0 执行
    """
    dataset, info = build_dataset(config)
    dataloader = build_dataloader(config, dataset)

    # --- 汇总 summary --- 
    def _summarise_one(dl):
        return summarise_pipeline(info, dl)

    summary: Dict[str, Any] = {}
    split_cfg = (config.get("split") or {})
    split_enabled = bool(split_cfg.get("enable", False))

    if not split_enabled or not isinstance(dataloader, dict):
        summary = _summarise_one(dataloader)
    else:
        split_summaries = {}
        sizes, total = {}, 0
        for name, dl in dataloader.items():
            split_summaries[name] = _summarise_one(dl)
            try:
                sizes[name] = len(dl.dataset)
            except Exception:
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

    # --- 导出（仅 rank0 执行）---
    output_cfg = config.get("output")
    if output_cfg:
        if not (split_enabled and isinstance(dataloader, dict)):
            # 单一 DataLoader
            _export_data_outputs(dataset, dataloader, summary, output_cfg)
        else:
            # 多 split：分别落盘
            from copy import deepcopy
            base = deepcopy(output_cfg)
            base_out = base.get("out_dir", "./prep_out")

            for name, dl in dataloader.items():
                oc = deepcopy(base)
                oc["out_dir"] = os.path.join(base_out, name)
                s_one = summary["splits"].get(name, {})
                _export_data_outputs(dataset, dl, s_one, oc)

            # 写 split_overview.json（rank0）
            try:
                os.makedirs(base_out, exist_ok=True)
                with open(os.path.join(base_out, "split_overview.json"), "w", encoding="utf-8") as f:
                    import json
                    json.dump(summary.get("split_overview", {}), f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    return dataset, dataloader, summary

# 通道规划
def _build_io_spec(data_cfg: Dict[str, Any], *, available_channels: Optional[list] = None) -> Dict[str, Any]:
    """
    根据 YAML 的 data.io 字段生成 io_spec：
      input_channels = [predict_channels...] + [extras...]
      target_channels = [predict_channels...]
    extras 支持：
      - P:      采样点位置通道（point_mask）
      - P_val:  采样点×采样值（point_value）
    """
    io_cfg = (data_cfg.get("io") or {})
    task = str(io_cfg.get("task", "reconstruct")).lower()
    if task not in ("reconstruct", "separate"):
        task = "reconstruct"  # 占位：仅支持复原任务

    # 预测通道
    predict_channels = list(io_cfg.get("predict_channels") or ["u"])
    if available_channels:
        for ch in predict_channels:
            if ch not in available_channels:
                raise KeyError(f"predict channel '{ch}' not in dataset channels {available_channels}")

    # extras
    extras_cfg = io_cfg.get("extras") or {}
    use_P = bool(extras_cfg.get("point_mask", False))
    use_Pv = bool(extras_cfg.get("point_value", False))

    input_channels = list(predict_channels)
    channel_alias = {}

    if use_P:
        input_channels.append("P")
        channel_alias["P"] = "point_mask"
    if use_Pv:
        input_channels.append("P_val")
        channel_alias["P_val"] = "point_value"

    io_spec = {
        "task": task,
        "input_channels": input_channels,
        "target_channels": list(predict_channels),
        "num_in": len(input_channels),
        "num_out": len(predict_channels),
        "channel_alias": channel_alias,
        "predict_channels": list(predict_channels),
    }
    return io_spec

class _MappedDatasetV2:
    """
    包装基础数据集（从快照恢复的 dataset 或任何 __getitem__ -> dict 样本的 dataset），
    依据 io_spec 组装 x/y：
      x = [predict_channels...] + [P?] + [P_val?]
      y = [predict_channels...]
    其中：
      - 原始样本若为 {'u','v',...}，直接按键取；
      - 若为 {'x': CxHxW, 'y': C'xHxW} 且没有命名通道，则尝试从 y 推断（y 的通道顺序作为 predict_channels 的来源）。
    """
    def __init__(self, base_ds, io_spec: Dict[str, Any],
                 *, sample_density: Optional[float], noise_sigma: Optional[float],
                 rng_seed: int = 1234, base_fallback: Optional[str] = "u"):
        from torch.utils.data import Dataset as _TorchDataset  # 为了 isinstance 判断
        self.base = base_ds
        self.io_spec = io_spec
        self.sample_density = sample_density
        self.noise_sigma = noise_sigma
        self.rng = np.random.RandomState(int(rng_seed))
        self.base_fallback = base_fallback

        # 试探 dataset 可见通道名（仅用于报错/提示）
        self.available = None
        try:
            s0 = base_ds[0]
            self.available = list(s0.keys())
        except Exception:
            self.available = None

    def __len__(self):
        return len(self.base)

    def _to_numpy(self, arr):
        import torch as _torch, numpy as _np
        if isinstance(arr, _torch.Tensor):
            return arr.detach().cpu().numpy()
        return _np.asarray(arr)

    def _make_mask_like(self, arr_hw, p: Optional[float]):
        if p is None or p <= 0.0:
            return np.zeros(arr_hw, dtype=np.uint8)
        h, w = arr_hw
        num = int(round(p * h * w))
        idx = self.rng.choice(h * w, size=num, replace=False)
        mask = np.zeros(h * w, dtype=np.uint8)
        mask[idx] = 1
        return mask.reshape(h, w)

    def _extract_named_channel(self, sample: Dict[str, Any], name: str) -> np.ndarray:
        """
        优先从命名键（'u','v',...）取；若不存在，尝试从 'y'（或 'x'）按通道顺序映射。
        """
        if name in sample:
            a = self._to_numpy(sample[name]).astype(np.float32)
            # 兼容 [1,H,W]
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            return a
        # 回退：若 y 存在且 predict_channels 的顺序与 y 的通道一致，则用对应索引
        if "y" in sample:
            Y = self._to_numpy(sample["y"]).astype(np.float32)
            # [C,H,W] or [H,W]
            if Y.ndim == 2:
                if len(self.io_spec["target_channels"]) == 1:
                    return Y
                raise ValueError("target is 2D but predict_channels>1; cannot map.")
            if Y.ndim == 3:
                # 假设 y 的顺序就是 predict_channels 的顺序
                try:
                    idx = self.io_spec["target_channels"].index(name)
                except ValueError:
                    raise KeyError(f"Channel '{name}' not found in target_channels {self.io_spec['target_channels']}.")
                return Y[idx]
        # 再回退：若 x 存在（但无语义），无法可靠映射 -> 抛错
        raise KeyError(f"Cannot extract channel '{name}' from sample keys {list(sample.keys())}.")

    def __getitem__(self, idx):
        sample = self.base[idx]
        io = self.io_spec

        in_chs      = io["input_channels"]       # e.g. ["u","P","P_val"]
        tgt_chs     = io["target_channels"]      # e.g. ["u"] or ["u","v"]
        pred_chs    = io.get("predict_channels", tgt_chs)

        # --- 1) 拿到监督目标通道（原始 y，不加噪，不插值） ---
        tgt_maps = [self._extract_named_channel(sample, ch) for ch in tgt_chs]
        if len(tgt_maps) == 0:
            raise RuntimeError("io_spec.target_channels 不能为空")
        # 全部转成 float32 + [H,W]
        tgt_maps = [m.astype(np.float32) for m in tgt_maps]

        # --- 2) 选一个基准通道，用来确定 H,W 和生成 mask ---
        #    默认用第一个 predict_channel；若没有则 fallback
        if len(pred_chs) > 0:
            base_name = pred_chs[0]
        else:
            base_name = self.base_fallback or tgt_chs[0]

        base_map = self._extract_named_channel(sample, base_name).astype(np.float32)
        H, W = base_map.shape[-2], base_map.shape[-1]

        # --- 3) 生成公共采样 mask ---
        mask = self._make_mask_like((H, W), self.sample_density).astype(np.float32)  # [H,W]

        # --- 4) 对每个预测通道做：加噪观测 → 1NN 插值成 recon_c ---
        recon_maps = []        # 按 predict_channels 的顺序
        obs_first  = None      # 用于 P_val（第一通道的观测值）
        for ch in pred_chs:
            orig = self._extract_named_channel(sample, ch).astype(np.float32)  # 原始场 [H,W]
            # 加噪观测（只在 mask 上有意义）
            obs = orig.copy()
            if self.noise_sigma is not None and float(self.noise_sigma) > 0.0:
                obs = obs + self.rng.normal(0.0, float(self.noise_sigma), size=obs.shape).astype(np.float32)

            if obs_first is None:
                obs_first = obs

            # 1NN 插值：mask>0 的位置作为样本点
            recon = _interp_1nn(mask, obs)   # [H,W]
            recon_maps.append(recon.astype(np.float32))

        if obs_first is None:
            # 理论上不会发生（pred_chs 至少有一个），兜底一下
            obs_first = base_map

        # --- 5) 拼装 x：先是每个预测通道的 recon，再根据 in_chs 附加 extras ---
        x_parts = []
        pred_iter = iter(recon_maps)
        for ch in in_chs:
            if ch == "P":
                x_parts.append(mask.astype(np.float32))
            elif ch == "P_val":
                # 只用第一预测通道的观测值 × mask（与旧三通道语义兼容）
                x_parts.append((obs_first * mask).astype(np.float32))
            else:
                # 这是一个真正的预测通道名
                try:
                    # recon_maps 的顺序与 pred_chs 一一对应
                    x_parts.append(next(pred_iter))
                except StopIteration:
                    # 万一 input_channels 比 predict_channels 多，退回原始场
                    x_parts.append(self._extract_named_channel(sample, ch).astype(np.float32))

        x = np.stack(x_parts, axis=0).astype(np.float32)     # [C_in,H,W]
        y = np.stack(tgt_maps, axis=0).astype(np.float32)    # [C_out,H,W]
        return {"x": x, "y": y}

# --- 1NN 插值工具函数：mask>0 的点作为采样点，对整张 HxW 网格做最近邻扩散 ---
def _interp_1nn(mask_hw: np.ndarray, values_hw: np.ndarray) -> np.ndarray:
    """
    mask_hw: [H,W], >0 表示有观测点
    values_hw: [H,W], 在观测点处给出观测值，其余位置可任意
    返回：对每个像素找到最近观测点后的插值结果 [H,W]
    """
    mask = (mask_hw > 0).astype(bool)
    H, W = mask.shape
    if mask.sum() == 0:
        # 没有采样点就直接给 0；保持“无观测”的语义
        return np.zeros((H, W), dtype=values_hw.dtype)

    try:
        # 优先使用 scipy 的 EDT（性能最好）
        from scipy.ndimage import distance_transform_edt
        # 反过来：把“没有点”的地方当作需要距离变换的对象
        inv = ~mask
        # return_indices=True 会给出每个像素最近的“False→True 边界”索引
        _, indices = distance_transform_edt(inv, return_indices=True)
        yy, xx = indices  # [2,H,W]
        return values_hw[yy, xx]
    except Exception:
        # 没有 scipy 的兜底实现：退化为“均值填充”，至少不报错
        observed = values_hw[mask]
        if observed.size == 0:
            return np.zeros((H, W), dtype=values_hw.dtype)
        mean_val = observed.mean().astype(values_hw.dtype)
        out = np.full((H, W), mean_val, dtype=values_hw.dtype)
        out[mask] = values_hw[mask]
        return out

def build_all(
    *,
    snapshot_dir: str,
    # 因子化注入
    sample_density: Optional[float] = None,
    noise_sigma: Optional[float] = None,
    rng_seed_offset: int = 0,
    # DataLoader 常用参数
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 4,
    drop_last: bool = False,
    shuffle: bool = True,
    split: Optional[Dict[str, Any]] = None,
    # 透传 data 配置（含 data.io、legacy_three_channel_input 等）
    **data_cfg,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    v2 统一数据入口：
      - 若 data.legacy_three_channel_input=true → 兼容旧“三通道硬编码”；
      - 否则根据 data.io 组装：
          x = [predict_channels...] + [P?] + [P_val?]
          y = [predict_channels...]
    """
    import os
    from torch.utils.data import DataLoader

    # 目录别名容错
    if not os.path.isdir(snapshot_dir):
        alt = snapshot_dir.rstrip("/\\") + "_v2"
        if os.path.isdir(alt):
            snapshot_dir = alt

    # 先从快照恢复基础 dataloaders（保持最原始样本字典结构）
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

    rng_base = 1234 + int(rng_seed_offset)

    # 按 io_spec 组装
    # 1) 探测可用通道名（从样本键）
    try:
        probe_sample = base_train.dataset[0] if base_train is not None else (base_val.dataset[0] if base_val is not None else base_test.dataset[0])
        available = [k for k in probe_sample.keys() if k not in ("x", "y", "cond")]
    except Exception:
        available = None

    io_spec = _build_io_spec(data_cfg, available_channels=available)

    def _rebuild_v2(base_dl: Optional[DataLoader]) -> Optional[DataLoader]:
        if base_dl is None:
            return None
        ds = _MappedDatasetV2(
            base_dl.dataset,
            io_spec,
            sample_density=sample_density,
            noise_sigma=noise_sigma,
            rng_seed=rng_base,
            base_fallback=(data_cfg.get("base_channel") or "u"),
        )
        collate_fn = base_dl.collate_fn
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

    return _rebuild_v2(base_train), _rebuild_v2(base_val), _rebuild_v2(base_test or base_val)