from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - fallback for environments without scipy
    distance_transform_edt = None  # type: ignore[assignment]


@dataclass
class SparseInputConfig:
    enabled: bool = False
    sample_p: float = 1e-3
    sample_sigma: float = 0.0
    sample_seed: int = 123
    append_mask_channel: bool = False

    @staticmethod
    def from_dict(cfg: Optional[dict]) -> "SparseInputConfig":
        cfg = cfg or {}
        return SparseInputConfig(
            enabled=bool(cfg.get("enabled", False)),
            sample_p=float(cfg.get("sample_p", 1e-3)),
            sample_sigma=float(cfg.get("sample_sigma", 0.0)),
            sample_seed=int(cfg.get("sample_seed", 123)),
            append_mask_channel=bool(cfg.get("append_mask_channel", False)),
        )

    def validate(self) -> None:
        if not (0.0 <= float(self.sample_p) <= 1.0):
            raise ValueError(f"sample_p must be in [0,1], got {self.sample_p}")
        if float(self.sample_sigma) < 0.0:
            raise ValueError(f"sample_sigma must be >= 0, got {self.sample_sigma}")


def _stable_seed(dataset_id: str, h: int, w: int, base_seed: int, sample_p: float) -> int:
    payload = f"{dataset_id}|{h}x{w}|{base_seed}|{sample_p:.12g}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return int(digest, 16)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _points_xy_from_mask(mask_hw: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask_hw)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([xs, ys], axis=1).astype(np.int64, copy=False)


def build_fixed_points_mask(
    h: int,
    w: int,
    sample_p: float,
    seed: int,
    dataset_id: str,
    valid_mask_hw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid spatial size h={h}, w={w}")

    total = int(h * w)
    if valid_mask_hw is None:
        valid_mask_hw = np.ones((h, w), dtype=bool)
    else:
        valid_mask_hw = np.asarray(valid_mask_hw, dtype=bool)
        if valid_mask_hw.shape != (h, w):
            raise ValueError(f"valid_mask_hw shape mismatch: expected {(h, w)}, got {valid_mask_hw.shape}")

    valid_flat_idx = np.flatnonzero(valid_mask_hw.reshape(-1))
    if valid_flat_idx.size == 0:
        mask_hw = np.zeros((h, w), dtype=bool)
        return mask_hw, _points_xy_from_mask(mask_hw)

    target_k = int(round(float(sample_p) * float(total)))
    target_k = max(1, target_k)
    k = min(target_k, int(valid_flat_idx.size))

    seed_final = _stable_seed(dataset_id=dataset_id, h=h, w=w, base_seed=seed, sample_p=sample_p)
    choice = _rng(seed_final).choice(valid_flat_idx, size=k, replace=False)

    mask_flat = np.zeros(total, dtype=bool)
    mask_flat[choice] = True
    mask_hw = mask_flat.reshape(h, w)
    return mask_hw, _points_xy_from_mask(mask_hw)


def apply_sparse_sampling_1nn(
    x_hwc: np.ndarray,
    points_mask_hw: np.ndarray,
    sample_sigma: float,
    noise_seed: int,
) -> np.ndarray:
    x = np.asarray(x_hwc, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x_hwc must be rank-3 [H,W,C], got {x.shape}")
    h, w, c = x.shape

    points_mask_hw = np.asarray(points_mask_hw, dtype=bool)
    if points_mask_hw.shape != (h, w):
        raise ValueError(f"points_mask_hw shape mismatch: expected {(h, w)}, got {points_mask_hw.shape}")

    if not points_mask_hw.any():
        return np.zeros_like(x, dtype=np.float32)

    x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    obs = np.zeros_like(x_clean, dtype=np.float32)
    obs[points_mask_hw, :] = x_clean[points_mask_hw, :]

    sigma = float(sample_sigma)
    if sigma > 0.0:
        vals = obs[points_mask_hw, :]
        vals = vals + _rng(noise_seed).normal(loc=0.0, scale=sigma, size=vals.shape).astype(np.float32)
        obs[points_mask_hw, :] = vals

    if distance_transform_edt is None:
        fill = np.mean(obs[points_mask_hw, :], axis=0, keepdims=True)
        recon = np.broadcast_to(fill, (h, w, c)).copy()
        recon[points_mask_hw, :] = obs[points_mask_hw, :]
        return recon.astype(np.float32, copy=False)

    _, nn_indices = distance_transform_edt(~points_mask_hw, return_indices=True)
    iy = nn_indices[0]
    ix = nn_indices[1]
    recon = obs[iy, ix, :]
    return recon.astype(np.float32, copy=False)


def sample_noise_seed(base_seed: int, n: int, t: int) -> int:
    token = f"{base_seed}|{n}|{t}".encode("utf-8")
    return int(hashlib.md5(token).hexdigest()[:8], 16)
