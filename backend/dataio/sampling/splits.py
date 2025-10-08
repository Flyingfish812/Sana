# backend/dataio/sampling/splits.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import math
import random

def _get_specs(dataset) -> List[Any]:
    """
    尽力从 dataset 中取出样本规格列表（用于切分）。
    兼容多种命名：dataset.sampler.specs / dataset.specs / dataset._specs
    """
    for attr in ("specs", "_specs"):
        if hasattr(dataset, attr) and getattr(dataset, attr) is not None:
            return list(getattr(dataset, attr))
    # 常见：dataset.sampler.specs
    if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "specs"):
        return list(dataset.sampler.specs)
    # 退化：按纯索引切分
    return list(range(len(dataset)))

def _peek_nt_from_spec(sp) -> Tuple[Optional[int], Optional[int]]:
    """
    解析单个 SampleSpec 的 (n,t)；兼容不同命名与位置参数。
    返回 (n, t)；无法解析则 (None, None)
    """
    # 常见：属性 n、t（t 通常是 [t]）
    n = getattr(sp, "n", None)
    t = getattr(sp, "t", None)
    if isinstance(t, (list, tuple)) and len(t) > 0:
        t = t[-1]
    # 备选命名
    if n is None:
        for k in ("index", "idx", "i", "sequence", "sample_index"):
            if hasattr(sp, k):
                n = getattr(sp, k); break
    if t is None:
        for k in ("time", "times", "time_idx", "frame"):
            if hasattr(sp, k):
                v = getattr(sp, k)
                if isinstance(v, (list, tuple)) and len(v) > 0: v = v[-1]
                t = v; break
    # 位置参数（很少见）：sp.args
    if (n is None or t is None) and hasattr(sp, "__dict__") and "args" in sp.__dict__:
        args = sp.__dict__["args"]
        if isinstance(args, (list, tuple)):
            if len(args) >= 1 and n is None: n = args[0]
            if len(args) >= 2 and t is None:
                v = args[1]
                if isinstance(v, (list, tuple)) and len(v) > 0: v = v[-1]
                t = v
    try:
        n = int(n) if n is not None else None
    except Exception:
        n = None
    try:
        t = int(t) if t is not None else None
    except Exception:
        t = None
    return n, t

def split_indices(dataset,
                  *,
                  strategy: str = "random",
                  ratios: Dict[str, float] = None,
                  unit: str = "frame",    # "frame" | "sequence"
                  seed: int = 123) -> Dict[str, List[int]]:
    """
    生成 train/val/test 的索引切分（按 dataset 的样本索引）。
    - strategy="random": 对所有样本随机洗牌后按比例划分
    - strategy="temporal": 按时间 t 做时间留出；对每个 n 平行切分，保持时间连续性
    - unit="frame": 以单帧样本为单位切分；"sequence": 以 n（序列）为单位切分
    """
    ratios = ratios or {"train":0.8, "val":0.1, "test":0.1}
    keys = ["train","val","test"]
    r = np.array([ratios.get(k,0.0) for k in keys], dtype=float)
    if r.sum() <= 0: raise ValueError("split ratios sum must be > 0")
    r = r / r.sum()

    specs = _get_specs(dataset)
    Ntotal = len(specs)
    idx_all = np.arange(Ntotal, dtype=int)

    rng = random.Random(seed)

    # 单位为 sequence：先把样本按 n 分组
    if unit == "sequence":
        # 建 n → 样本索引 列表
        groups: Dict[int, List[int]] = {}
        for i, sp in enumerate(specs):
            n, t = _peek_nt_from_spec(sp)
            if n is None:
                # 无法解析 n，则退化为随机样本切分
                groups.clear(); break
            groups.setdefault(n, []).append(i)
        if groups:
            ns = sorted(groups.keys())
            rng.shuffle(ns)
            k1 = math.floor(len(ns)*r[0]); k2 = math.floor(len(ns)*(r[0]+r[1]))
            split_ns = {
                "train": set(ns[:k1]),
                "val":   set(ns[k1:k2]),
                "test":  set(ns[k2:])
            }
            return {
                name: [i for n in ns_set for i in groups[n]]
                for name, ns_set in split_ns.items()
            }
        # 否则走 random 样本切分

    if strategy == "temporal":
        # 按时间切分：对每个 n 构造它的时间索引，按比例做前/中/后分段
        # 需要能解析 t，否则退化为随机
        per_n: Dict[int, List[Tuple[int,int]]] = {}
        for i, sp in enumerate(specs):
            n, t = _peek_nt_from_spec(sp)
            if n is None or t is None:
                per_n = {}; break
            per_n.setdefault(n, []).append((t, i))
        if per_n:
            train_idx: List[int] = []; val_idx: List[int] = []; test_idx: List[int] = []
            for n, tv_list in per_n.items():
                tv_list.sort(key=lambda x: x[0]) # 按 t 升序
                m = len(tv_list)
                k1 = math.floor(m * r[0]); k2 = math.floor(m * (r[0]+r[1]))
                train_idx += [tv_list[j][1] for j in range(0, k1)]
                val_idx   += [tv_list[j][1] for j in range(k1, k2)]
                test_idx  += [tv_list[j][1] for j in range(k2, m)]
            return {"train": train_idx, "val": val_idx, "test": test_idx}
        # 退化为随机

    # 默认 / 退化：随机样本划分（frame 级）
    idx = list(idx_all)
    rng.shuffle(idx)
    k1 = math.floor(Ntotal * r[0]); k2 = math.floor(Ntotal * (r[0]+r[1]))
    return {"train": idx[:k1], "val": idx[k1:k2], "test": idx[k2:]}
