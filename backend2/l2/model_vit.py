from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .probe import ProbeController


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(int(embed_dim) * float(mlp_ratio))
        self.fc1 = nn.Linear(int(embed_dim), hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, int(embed_dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        droppath: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.attn = nn.MultiheadAttention(
            int(embed_dim),
            int(num_heads),
            dropout=float(attention_dropout),
            batch_first=True,
            bias=True,
        )
        self.drop_path1 = DropPath(float(droppath))
        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.mlp = MLP(int(embed_dim), float(mlp_ratio), float(dropout))
        self.drop_path2 = DropPath(float(droppath))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ViTTransformer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        droppath: float,
    ):
        super().__init__()
        depth = int(depth)
        if depth <= 0:
            raise ValueError("depth must be positive")
        drop_values = torch.linspace(0.0, float(droppath), depth).tolist() if depth > 1 else [float(droppath)]
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    int(embed_dim),
                    int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    attention_dropout=float(attention_dropout),
                    droppath=float(drop_values[i]),
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor, probe: Optional[ProbeController] = None) -> torch.Tensor:
        for idx, block in enumerate(self.blocks, start=1):
            x = block(x)
            if probe is not None:
                x = probe.apply(f"vit.block{idx}.out", x)
        return self.norm(x)


class ViTPatchDecoder(nn.Module):
    def __init__(self, *, embed_dim: int, patch_size: int, dropout: float = 0.0):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.ConvTranspose2d(
            int(embed_dim),
            int(embed_dim),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.dropout(x)


class PixelHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MinimalViTRegressor(nn.Module):
    """迁移自旧版结构的 ViT：encoder -> transformer -> patch decoder -> pixel head。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        embed_dim: int = 64,
        depth: int = 10,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.15,
        droppath: float = 0.2,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        self.encoder = PatchEmbed2D(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
        )
        self.propagator = ViTTransformer(
            embed_dim=self.embed_dim,
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            attention_dropout=float(attention_dropout),
            droppath=float(droppath),
        )
        self.decoder = ViTPatchDecoder(embed_dim=self.embed_dim, patch_size=self.patch_size, dropout=float(dropout))
        self.head = PixelHead(in_channels=self.embed_dim, out_channels=self.out_channels)

    @staticmethod
    def _tap(probe: Optional[ProbeController], name: str, x: torch.Tensor) -> torch.Tensor:
        return probe.apply(name, x) if probe is not None else x

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x_pad, h, w

    def forward(self, x: torch.Tensor, probe: Optional[ProbeController] = None) -> torch.Tensor:
        x_pad, h0, w0 = self._pad_to_patch(x)
        tokens_2d = self._tap(probe, "enc.stage1.out", self.encoder(x_pad))
        b, c, hp, wp = tokens_2d.shape

        tokens = tokens_2d.flatten(2).transpose(1, 2).contiguous()
        tokens = self._tap(probe, "propagator.tokens_in", tokens)
        tokens = self._tap(probe, "propagator.out", self.propagator(tokens, probe=probe))

        tokens_2d_out = tokens.transpose(1, 2).contiguous().view(b, c, hp, wp)
        dec = self._tap(probe, "dec.stage1.out", self.decoder(tokens_2d_out))
        out = self._tap(probe, "head.out", self.head(dec))
        out = out[:, :, :h0, :w0]
        return out
