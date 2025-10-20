"""Synthetic dataloaders for tests and smoke runs."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def _make_dataset(num_samples: int, in_channels: int, sequence_length: int, height: int, width: int) -> TensorDataset:
    x = torch.randn(num_samples, in_channels, sequence_length, height, width, dtype=torch.float32)
    y = torch.randn(num_samples, 1, height, width, dtype=torch.float32)
    return TensorDataset(x, y)


def build_tiny_dataloaders(
    num_samples: int = 8,
    *,
    in_channels: int = 1,
    sequence_length: int = 4,
    height: int = 16,
    width: int = 16,
    batch_size: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return small dataloaders backed by random tensors.

    The shapes are compatible with the default EPD configuration.
    """

    dataset = _make_dataset(num_samples, in_channels, sequence_length, height, width)
    # Reuse the same dataset for train/val/test to keep the fixture lightweight.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
