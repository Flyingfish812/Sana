"""Utilities for dealing with training batches and tensor layouts."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch

INPUT_KEYS: Tuple[str, ...] = ("x", "input", "inputs", "image")
TARGET_KEYS: Tuple[str, ...] = ("y", "target", "targets", "label")
META_KEYS: Tuple[str, ...] = ("meta",)


def pick_first_key(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-``None`` value matched in ``data`` for ``keys``."""

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def extract_xy(
    batch: Any,
    *,
    input_keys: Sequence[str] = INPUT_KEYS,
    target_keys: Sequence[str] = TARGET_KEYS,
    meta_keys: Sequence[str] = META_KEYS,
) -> Tuple[torch.Tensor, torch.Tensor, MutableMapping[str, Any]]:
    """Normalise different batch layouts to ``(x, y, meta)`` tuples."""

    meta: MutableMapping[str, Any] = {}

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list must contain at least two elements (x, y).")
        x = batch[0]
        y = batch[1]
        if len(batch) >= 3 and isinstance(batch[2], MutableMapping):
            meta = batch[2]
        return x, y, meta

    if isinstance(batch, Mapping):
        x = pick_first_key(batch, input_keys)
        y = pick_first_key(batch, target_keys)
        if x is None or y is None:
            raise KeyError(
                "Cannot find input/target tensors in batch. "
                f"Expected keys {tuple(input_keys)} for input and {tuple(target_keys)} for target."
            )
        for key in meta_keys:
            value = batch.get(key)
            if isinstance(value, MutableMapping):
                meta = value
                break
        return x, y, meta

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def ensure_5d(x: torch.Tensor) -> torch.Tensor:
    """Lift ``[B,C,H,W]`` tensors to ``[B,C,1,H,W]`` while leaving 5D untouched."""

    if x.ndim == 4:
        b, c, h, w = x.shape
        return x.view(b, c, 1, h, w)
    if x.ndim == 5:
        return x
    raise ValueError(f"Input must be a 4D or 5D tensor. Received shape {tuple(x.shape)}")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move nested tensors onto ``device``."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(item, device) for item in batch)
    if isinstance(batch, Mapping):
        return type(batch)((key, move_batch_to_device(value, device)) for key, value in batch.items())
    return batch
