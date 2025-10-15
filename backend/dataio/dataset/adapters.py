# backend/dataio/dataset/adapters.py
from __future__ import annotations
import numpy as np
import math
from typing import Dict, Any, Optional, Sequence
from ..schema import ArraySample, AdapterOutput
from ..registry import register_adapter

from functools import lru_cache
import hashlib

def _hash_points(pts: np.ndarray, H: int, W: int) -> str:
    """给 (H,W,pts) 生成稳定哈希键；pts 为 int32/64 的 [M,2] (y,x)。"""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.int64(H).tobytes()); h.update(np.int64(W).tobytes())
    h.update(np.asarray(pts, dtype=np.int64).tobytes())
    return h.hexdigest()

def _compute_idx_map_1nn(H: int, W: int, pts: np.ndarray, block_size: int = 16384) -> np.ndarray:
    """
    计算像素→最近采样点的索引图：返回 idx_map[H,W]，值域 [0, M-1]。
    仅计算一次，后续样本直接用 idx_map 做 values[idx_map] 的索引映射。
    """
    HW = H * W
    grid_y = np.repeat(np.arange(H), W)     # [HW]
    grid_x = np.tile(np.arange(W), H)       # [HW]
    out = np.empty((HW,), dtype=np.int32)
    pts = np.asarray(pts, dtype=np.float32)  # [M,2]

    for start in range(0, HW, block_size):
        end = min(start + block_size, HW)
        gy = grid_y[start:end][:, None]   # [B,1]
        gx = grid_x[start:end][:, None]   # [B,1]
        # [B,M] 的平方距离
        d2 = (gy - pts[None, :, 0])**2 + (gx - pts[None, :, 1])**2
        out[start:end] = np.argmin(d2, axis=1)
    return out.reshape(H, W)  # [H,W]

# 简单 LRU 缓存（键：hash(pts,H,W)）；注意 numpy 数组不能直接做 lru_cache 参数
_IDX_MAP_CACHE: dict[str, np.ndarray] = {}
_IDX_MAP_CAP = 16  # 最多缓存 16 个不同的采样布局；按需调整

def _to_chw(frames: np.ndarray) -> np.ndarray:
    """[K,H,W,C] → [K,C,H,W]"""
    return np.transpose(frames, (0, 3, 1, 2))

@register_adapter("static2d")
def Static2DAdapter(sample: ArraySample, *, merge_time: bool = False) -> AdapterOutput:
    """
    将单帧样本适配为 2D 输入：
      - 默认 x=[C,H,W]，y=[C,H,W]（与目标同维），如果 merge_time=True 则将 K 维合并到 C
    """
    frames = sample.frames  # [K,H,W,C]
    K = frames.shape[0]
    chw = _to_chw(frames)   # [K,C,H,W]
    if K == 1:
        x = chw[0]
        y = chw[0]
        cond = None
    else:
        if merge_time:
            x = chw.reshape(-1, *chw.shape[-2:])  # [K*C,H,W]
            y = chw[-1]                            # 以最后一帧为目标
        else:
            x = chw[:-1]                           # [K-1,C,H,W]
            y = chw[-1]                            # [C,H,W]
        cond = None
    return AdapterOutput(x=x, y=y, cond=cond, meta=sample.meta)

@register_adapter("timecond2d")
def TimeCond2DAdapter(sample: ArraySample) -> AdapterOutput:
    """
    时间条件输入：x 为所有历史帧堆叠，y 为最后一帧；与 Static2DAdapter(merge_time=True) 类似，但保留 cond 占位。
    """
    frames = sample.frames
    chw = _to_chw(frames)         # [K,C,H,W]
    if chw.shape[0] == 1:
        x = chw[0]
        y = chw[0]
        cond = None
    else:
        x = chw[:-1]              # 历史帧
        y = chw[-1]
        cond = None               # 未来可注入时间编码/物理先验等
    return AdapterOutput(x=x, y=y, cond=cond, meta=sample.meta)

def _finite_mask(img: np.ndarray) -> np.ndarray:
    # img: [H,W] or [H,W,C] -> [H,W] bool
    if img.ndim == 3:
        return np.all(np.isfinite(img), axis=-1)
    return np.isfinite(img)

def _interp_1nn(H: int, W: int,
                coords: np.ndarray,   # [M,2] (y,x)
                values: np.ndarray,   # [M,C]
                *,
                block_size: int = 8192) -> np.ndarray:
    """
    1-NN 最近邻插值（分块版）：
      - 每个网格点只使用最近的一个采样点
      - 复杂度 O(H*W*M)，但按 block 计算，内存峰值 O(block_size * M)
      - 纯 numpy 实现
    返回: [H,W,C]
    """
    HW = H * W
    grid_y = np.repeat(np.arange(H), W)     # [HW]
    grid_x = np.tile(np.arange(W), H)       # [HW]
    grid = np.stack([grid_y, grid_x], axis=1)  # [HW,2]

    out = np.empty((HW, values.shape[1]), dtype=np.float32)
    M = coords.shape[0]

    # 分块计算到所有采样点的距离，取 argmin
    for start in range(0, HW, block_size):
        end = min(start + block_size, HW)
        G = grid[start:end]                                      # [B,2]
        # [B,M] 的平方距离（避免 sqrt）
        d2 = (G[:, None, 0] - coords[None, :, 0])**2 + (G[:, None, 1] - coords[None, :, 1])**2
        nn_idx = np.argmin(d2, axis=1)                           # [B]
        out[start:end, :] = values[nn_idx]                       # 拿最近点的通道值

    return out.reshape(H, W, values.shape[1]).astype(np.float32)

@register_adapter("sparse2d")
def Sparse2DAdapter(
    sample: ArraySample, *,
    mode: str = "random",                 # "random" | "mask"
    num_points: int = 1024,
    seed: int = 42,
    mask_path: Optional[str] = None,
    nn_block_size: int = 16384,
    include_mask_in_cond: bool = True,
    include_points_in_cond: bool = True,
    avoid_nan: bool = True,
    reuse_points: str = "per_dataset",    # "none" | "per_dataset" | "per_mask"
    seed_per_time: bool = False,
    # —— 新增参数（用于通道选择/兼容 NC 的多通道）——
    # 基准通道：用于构造 recon / masked 的“基底”通道；可为 int 索引或通道名（如 "omega"）
    base_channel: Optional[object] = None,
    # 监督目标通道选择：None=保留所有（旧行为）；可为索引或通道名列表（如 ["omega"]）
    target_channels: Optional[Sequence[object]] = None,
    # 追加到输入 x 的额外原始通道（例如把 ["u","v"] 也拼入输入）
    extra_input_channels: Optional[Sequence[object]] = None,
) -> AdapterOutput:
    """
    稀疏采样 + 1-NN 插值（带可配置通道选择）：
      - 依旧输出“三件套”输入：recon(masked by base_channel)、mask、masked(base_channel)
      - 可选把若干原始通道（如 u,v）拼到输入 x
      - y 默认保留全部通道（与旧行为一致）；在 NC 任务下可只保留 ["omega"]
    """
    # ---------- 基础读取 ----------
    frames = sample.frames              # [K,H,W,C]
    target = frames[-1]                 # [H,W,C] 监督基准帧
    H, W, C = target.shape
    meta = sample.meta
    chan_names = list((meta.attrs or {}).get("channels", []))  # 可能是 ["u","v","omega"]，也可能为空

    # ---------- 工具：把“索引或名字”解析成索引列表 ----------
    def _indices_from_spec(spec: Optional[Sequence[object] | object]) -> Optional[list[int]]:
        if spec is None:
            return None
        if isinstance(spec, (int, np.integer, str)):
            spec = [spec]
        out: list[int] = []
        name2idx = {n: i for i, n in enumerate(chan_names)}
        for w in spec:
            if isinstance(w, (int, np.integer)):
                idx = int(w)
                if not (0 <= idx < C):
                    raise IndexError(f"channel index {idx} out of range 0..{C-1}")
                out.append(idx)
            elif isinstance(w, str):
                if w not in name2idx:
                    raise KeyError(f"channel name '{w}' not in {chan_names or '[unnamed channels]'}")
                out.append(int(name2idx[w]))
            else:
                raise TypeError(f"unsupported channel spec: {type(w)} -> {w}")
        return out

    base_idx = 0 if base_channel is None else _indices_from_spec(base_channel)[0]
    tgt_inds = _indices_from_spec(target_channels)             # e.g. ["omega"] -> [2]
    extra_inds = _indices_from_spec(extra_input_channels)      # e.g. ["u","v"] -> [0,1]

    # ---------- 1) 可采样区域 ----------
    nan_mask_from_meta = None
    if (meta.attrs is not None) and ("nan_mask" in meta.attrs):
        nan_mask_from_meta = np.asarray(meta.attrs["nan_mask"]).astype(np.float32)  # [H,W], 1=NaN
        if nan_mask_from_meta.shape != (H, W):
            raise ValueError(f"nan_mask shape mismatch, expect {(H,W)}, got {nan_mask_from_meta.shape}")

    if avoid_nan:
        if nan_mask_from_meta is not None:
            finite = (nan_mask_from_meta < 0.5)  # True = 可采样
        else:
            # 兼容旧行为：没有预捕获时，退化为基于目标帧的有限性判断
            finite = _finite_mask(target)
    else:
        finite = np.ones((H, W), dtype=bool)

    # ---------- 2) 采样点 ----------
    if mode == "random":
        ys, xs = np.where(finite)
        if ys.size == 0:
            ys, xs = np.indices((H, W)).reshape(2, -1)
        time_bump = (sample.spec.get("t", [-1])[-1] if (isinstance(sample.spec, dict) and seed_per_time) else 0)
        rng = np.random.RandomState(seed + int(time_bump))
        M = min(num_points, ys.size)
        idx = rng.choice(ys.size, size=M, replace=False)
        pts = np.stack([ys[idx], xs[idx]], axis=1).astype(np.int64)  # [M,2] (y,x)
    elif mode == "mask":
        assert mask_path is not None, "mode=mask 需要提供 mask_path"
        mask = _load_click_mask(mask_path, H, W)
        pts = np.stack(np.where(mask > 0.5), axis=1).astype(np.int64)
        if pts.shape[0] == 0:
            raise ValueError("指定采样点掩码为空。")
        M = pts.shape[0]
    else:
        raise ValueError(f"Unknown sparse mode: {mode}")

    # ---------- 3) 最近邻索引图（像素→最近采样点）+ 缓存 ----------
    cache_key = None
    if reuse_points == "per_dataset":
        cache_key = _hash_points(pts, H, W)
    elif reuse_points == "per_mask":
        mh = hashlib.blake2b(digest_size=16)
        mh.update(finite.astype(np.uint8).tobytes())
        mh.update(np.int64(H).tobytes()); mh.update(np.int64(W).tobytes())
        cache_key = mh.hexdigest()

    if cache_key is not None and cache_key in _IDX_MAP_CACHE:
        idx_map = _IDX_MAP_CACHE[cache_key]
    else:
        idx_map = _compute_idx_map_1nn(H, W, pts, block_size=nn_block_size)
        if cache_key is not None:
            if len(_IDX_MAP_CACHE) >= _IDX_MAP_CAP:
                _IDX_MAP_CACHE.pop(next(iter(_IDX_MAP_CACHE)))
            _IDX_MAP_CACHE[cache_key] = idx_map

    # ---------- 4) 基于采样点的 1NN 重建 ----------
    vals = target[pts[:, 0], pts[:, 1], :].astype(np.float32)  # [M,C]
    x_est = vals[idx_map]                                      # [H,W,C]

    # ---------- 5) 组装 x（三件套 + 额外通道） ----------
    # 三件套基于 base_idx：recon(base)、mask、masked(base)
    recon  = x_est[..., base_idx].astype(np.float32)           # [H,W]
    mask_img = np.zeros((H, W), dtype=np.float32)
    mask_img[pts[:, 0], pts[:, 1]] = 1.0
    if nan_mask_from_meta is not None:
        mark_map = (-1.0) * (nan_mask_from_meta > 0.5).astype(np.float32)  # 先把 NaN 置为 -1
    else:
        mark_map = np.zeros((H, W), dtype=np.float32)
        # 如果没有 nan_mask，则不写 -1；保持 0/1 行为

    # 采样点位置覆盖为 +1（优先级高于 NaN 标记）
    mark_map[pts[:, 0], pts[:, 1]] = 1.0

    # masked 基于 base 通道原值 * 采样点（保持兼容）
    masked = target[..., base_idx].astype(np.float32) * mask_img

    # 将第二通道改为 mark_map（满足 -1/1/0 需求）
    x_list = [recon[None, ...], mark_map[None, ...], masked[None, ...]]  # [3,H,W]
    if extra_inds:
        x_list.append(np.transpose(target[..., extra_inds], (2, 0, 1)).astype(np.float32))
    x = np.concatenate(x_list, axis=0)

    # ---------- 6) 组装 y（可裁通道；默认旧行为=保留全部） ----------
    if tgt_inds:
        y = np.transpose(target[..., tgt_inds], (2, 0, 1)).astype(np.float32)  # 只保留选定通道
    else:
        y = np.transpose(target, (2, 0, 1)).astype(np.float32)                 # 旧行为：全通道

    # ---------- 7) cond 打包（可选） ----------
    cond = None
    if include_mask_in_cond or include_points_in_cond:
        cond = {}
        if include_mask_in_cond:
            cond["mask"] = mask_img[None, ...]   # [1,H,W]
        if include_points_in_cond:
            cond["points"] = pts                 # [M,2] (y,x)

    return AdapterOutput(x=x, y=y, cond=cond, meta=meta)

# --- helper: 读取点击掩码 ---
def _load_click_mask(path: str, H: int, W: int) -> np.ndarray:
    import json, os
    p = path
    if p.lower().endswith(".npy"):
        m = np.load(p)
    elif p.lower().endswith(".json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if "mask" in d:
            m = np.array(d["mask"], dtype=np.float32)
        elif "points" in d:
            m = np.zeros((H,W), dtype=np.float32)
            for y, x in d["points"]:
                if 0 <= y < H and 0 <= x < W:
                    m[y,x] = 1.0
        else:
            raise ValueError("点击文件不包含 'mask' 或 'points' 字段。")
    else:
        raise ValueError("mask_path 必须是 .npy 或 .json 文件")
    if m.shape != (H,W):
        raise ValueError(f"mask 形状不匹配，期望 {(H,W)} 实际 {m.shape}")
    return m