"""Vision Transformer encoder for the EPD architecture."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..base_components.encoder_base import BaseEncoder
from ..factory import register


def _to_2tuple(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class PatchEmbed2D(nn.Module):
    """Simple linear embedding using non-overlapping patches."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int | Tuple[int, int]):
        super().__init__()
        patch = _to_2tuple(patch_size)
        self.patch_size = patch
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self.patch_size[0] != 0 or w % self.patch_size[1] != 0:
            raise ValueError(
                "Input spatial dimensions must be divisible by patch size. "
                f"Got H={h}, W={w}, patch_size={self.patch_size}."
            )
        in_dtype = x.dtype

        # ① 确保权重和输入在同一设备（Lightning 有时因“懒构建/子模块晚注册”漏搬）
        if self.proj.weight.device != x.device:
            self.proj = self.proj.to(x.device)

        # ② 关闭 autocast，③ dtype 与权重一致
        if torch.is_autocast_enabled():
            with torch.amp.autocast('cuda', enabled=False):
                out = self.proj(x.to(dtype=self.proj.weight.dtype))
        else:
            out = self.proj(x.to(dtype=self.proj.weight.dtype))

        return out.to(dtype=in_dtype)  # 还原回原 dtype

@register("encoder", "ViTEncoder")
class ViTEncoder(BaseEncoder):
    """Patchify inputs and add learnable positional embeddings."""

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        patch_size: int | Tuple[int, int] = 16,
        in_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = _to_2tuple(patch_size)
        self.in_channels = in_channels
        self.dropout = nn.Dropout(dropout)

        self.patch_embed: Optional[PatchEmbed2D] = None
        self.pos_embed: Optional[nn.Parameter] = None

    def _lazy_build(self, in_channels: int) -> None:
        self.patch_embed = PatchEmbed2D(in_channels, self.embed_dim, self.patch_size)

    def _ensure_pos_embed(self, grid_hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> None:
        h, w = grid_hw
        expected_shape = (1, self.embed_dim, 1, h, w)
        if self.pos_embed is None or tuple(self.pos_embed.shape) != expected_shape:
            pos = torch.zeros(expected_shape, device=device, dtype=dtype)
            nn.init.trunc_normal_(pos, std=0.02)
            self.pos_embed = nn.Parameter(pos)

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x5.shape
        self.skips = None
        if self.patch_embed is None:
            self._lazy_build(self.in_channels or c)

        x_flat = x5.reshape(b * t, c, h, w)
        patches = self.patch_embed(x_flat)  # type: ignore[arg-type]
        _, _, gh, gw = patches.shape
        patches = patches.view(b, t, self.embed_dim, gh, gw).permute(0, 2, 1, 3, 4)

        self._ensure_pos_embed((gh, gw), patches.device, patches.dtype)
        patches = patches + self.pos_embed  # type: ignore[operator]
        patches = self.dropout(patches)
        return patches
