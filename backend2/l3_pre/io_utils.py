from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_preds(pred_npz: Path) -> Dict[str, np.ndarray]:
    if not pred_npz.exists():
        raise FileNotFoundError(f"pred npz not found: {pred_npz}")
    with np.load(pred_npz, allow_pickle=False) as z:
        keys = set(z.files)
        required = {"gt", "pred", "pair_nt"}
        missing = sorted(list(required - keys))
        if missing:
            raise KeyError(f"pred npz missing keys: {missing}, available={sorted(list(keys))}")
        gt = np.asarray(z["gt"])
        pred = np.asarray(z["pred"])
        pair_nt = np.asarray(z["pair_nt"])
        input_arr = np.asarray(z["input"]) if "input" in keys else None
        obs_arr = np.asarray(z["obs"]) if "obs" in keys else None
        mask_arr = np.asarray(z["mask"]) if "mask" in keys else None
        sample_points_xy = np.asarray(z["sample_points_xy"]) if "sample_points_xy" in keys else None
        sample_points_mask = np.asarray(z["sample_points_mask"]) if "sample_points_mask" in keys else None

    if gt.shape != pred.shape:
        raise ValueError(f"gt/pred shape mismatch: {gt.shape} vs {pred.shape}")
    if gt.ndim != 4:
        raise ValueError(f"expected gt/pred [N,C,H,W], got {gt.shape}")
    if pair_nt.ndim != 2 or pair_nt.shape[1] != 2:
        raise ValueError(f"expected pair_nt [N,2], got {pair_nt.shape}")
    if pair_nt.shape[0] != gt.shape[0]:
        raise ValueError(f"pair_nt N mismatch: {pair_nt.shape[0]} vs {gt.shape[0]}")

    out: Dict[str, np.ndarray] = {"gt": gt, "pred": pred, "pair_nt": pair_nt}
    if input_arr is not None:
        out["input"] = input_arr
    if obs_arr is not None:
        out["obs"] = obs_arr
    if mask_arr is not None:
        out["mask"] = mask_arr
    if sample_points_xy is not None:
        out["sample_points_xy"] = sample_points_xy
    if sample_points_mask is not None:
        out["sample_points_mask"] = sample_points_mask
    return out


def try_load_probe(probe_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, List[np.ndarray]]], List[str]]:
    warnings: List[str] = []
    summary_path = probe_dir / "probe_summary.json"
    snaps_path = probe_dir / "probe_snapshots_level1.npz"

    summary_data: Optional[Dict[str, Any]] = None
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"failed reading probe_summary.json: {exc}")
    else:
        warnings.append(f"missing probe summary: {summary_path}")

    snapshots: Optional[Dict[str, List[np.ndarray]]] = None
    if snaps_path.exists():
        grouped: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        try:
            with np.load(snaps_path, allow_pickle=False) as z:
                for key in z.files:
                    arr = np.asarray(z[key])
                    m = re.match(r"^(.*)__([0-9]+)$", key)
                    if m is None:
                        layer_name = key
                        idx = 0
                    else:
                        layer_name = m.group(1)
                        idx = int(m.group(2))
                    grouped.setdefault(layer_name, []).append((idx, arr))
            snapshots = {}
            for layer_name, items in grouped.items():
                items_sorted = sorted(items, key=lambda x: x[0])
                snapshots[layer_name] = [arr for _, arr in items_sorted]
        except Exception as exc:
            warnings.append(f"failed reading probe snapshots: {exc}")
    else:
        warnings.append(f"missing probe snapshots: {snaps_path}")

    return summary_data, snapshots, warnings


def infer_run_dir_from_pred(pred_npz: Path) -> Optional[Path]:
    if pred_npz.name != "preds_test.npz":
        return None
    if pred_npz.parent.name != "infer":
        return None
    return pred_npz.parent.parent


def freeze_layer_path_from_pred(pred_npz: Path, layer_name: str) -> Optional[Path]:
    run_dir = infer_run_dir_from_pred(pred_npz)
    if run_dir is None:
        return None
    return run_dir / "freeze" / "layers" / f"{layer_name}.npz"


def coerce_to_bchw(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    while x.ndim > 4:
        x = x[0]
    if x.ndim == 2:
        x = x[None, None, :, :]
    elif x.ndim == 3:
        x = x[None, :, :, :]
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"unsupported rank for feature tensor: {arr.shape}")

    if x.ndim != 4:
        raise ValueError(f"failed to coerce to BCHW: {arr.shape} -> {x.shape}")
    return x


def probe_layer_covered_samples(layer_arrs: Sequence[np.ndarray]) -> int:
    total = 0
    for arr in layer_arrs:
        try:
            bchw = coerce_to_bchw(arr)
        except Exception:
            continue
        total += int(bchw.shape[0])
    return total


def get_probe_feature_at(layer_arrs: Sequence[np.ndarray], sample_idx: int) -> Optional[np.ndarray]:
    if sample_idx < 0:
        return None
    offset = 0
    for arr in layer_arrs:
        try:
            bchw = coerce_to_bchw(arr)
        except Exception:
            continue
        b = int(bchw.shape[0])
        if sample_idx < offset + b:
            local_i = sample_idx - offset
            return np.asarray(bchw[local_i], dtype=np.float32)
        offset += b
    return None


def load_freeze_layer_feats(pred_npz: Path, layer_name: str) -> Optional[np.ndarray]:
    p = freeze_layer_path_from_pred(pred_npz, layer_name)
    if p is None or not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        if "features" not in z.files:
            return None
        feats = np.asarray(z["features"])
    if feats.ndim != 4:
        return None
    return feats.astype(np.float32, copy=False)


def safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace("/", "_")
