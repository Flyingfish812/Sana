"""Vision Transformer encoder blocks used as propagators."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..base_components.propagator_base import BasePropagator
from ..factory import register


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

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
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        droppath: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
            bias=qkv_bias,
        )
        self.drop_path1 = DropPath(droppath)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(droppath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        # 保证子模块和输入在同一设备（极小改动，但能治“CPU 权重 vs CUDA 输入”）
        if self.attn.in_proj_weight.device != x.device:
            self.attn = self.attn.to(x.device)
        if self.mlp.fc1.weight.device != x.device:
            self.mlp = self.mlp.to(x.device)

        # Attention
        x_norm = self.norm1(x)  # 你的 LN 内部已有精度保护
        attn_w_dtype = self.attn.in_proj_weight.dtype
        if torch.is_autocast_enabled():
            with torch.amp.autocast('cuda', enabled=False):
                q = x_norm.to(dtype=attn_w_dtype)
                attn_out, _ = self.attn(q, q, q, need_weights=False)
        else:
            q = x_norm.to(dtype=attn_w_dtype)
            attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = x + self.drop_path1(attn_out.to(in_dtype))

        # MLP
        mlp_in = self.norm2(x)
        mlp_w_dtype = self.mlp.fc1.weight.dtype
        if torch.is_autocast_enabled():
            with torch.amp.autocast('cuda', enabled=False):
                mlp_out = self.mlp(mlp_in.to(dtype=mlp_w_dtype))
        else:
            mlp_out = self.mlp(mlp_in.to(dtype=mlp_w_dtype))
        x = x + self.drop_path2(mlp_out.to(in_dtype))
        return x

@register("propagator", "ViTTransformer")
class ViTPropagator(BasePropagator):
    """Stack of ViT blocks operating on patch tokens."""

    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        droppath: float = 0.0,
        qkv_bias: bool = True,
        use_post_smooth: bool = True,     # ← 新增：是否在末尾做空间平滑
        post_kernel: int = 3,             # ← 新增：空间平滑核
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError("Transformer depth must be positive")
        drop_values: List[float] = torch.linspace(0.0, droppath, depth).tolist() if depth > 1 else [droppath]

        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    droppath=drop_values[i],
                    qkv_bias=qkv_bias,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ← 新增：对空间维度做轻量深度可分离 3D 卷积（时间核为1，不改变 T）
        self.use_post_smooth = use_post_smooth
        if use_post_smooth:
            pad = post_kernel // 2
            self.post_smooth = nn.Conv3d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(1, post_kernel, post_kernel),  # 仅平滑 H,W
                padding=(0, pad, pad),
                groups=embed_dim,                            # 深度卷积：每通道独立
                bias=False,
            )
            # 初始化为“近似均值滤波”：中心略大，其余均匀
            with torch.no_grad():
                w = self.post_smooth.weight
                w.zero_()
                center = (post_kernel // 2)
                val = 1.0 / (post_kernel * post_kernel)
                w[:, 0, 0, :, :] += val
                # （可选）中心稍大一点，有助于保边，可不加：
                # w[:, 0, 0, center, center] += 0.0
        else:
            self.post_smooth = None

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x5.shape
        tokens = x5.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)  # [B*T, H*W, C]

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)  # [B*T, H*W, C]

        # reshape 回 5D
        x = tokens.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)  # [B,C,T,H,W]

        # 末尾“抹缝”平滑（只在空间维度上做 3×3 深度卷积）
        if self.use_post_smooth and self.post_smooth is not None:
            # 保证模块设备与精度一致（防止出现 CPU 权重/CUDA 输入 的不一致）
            if self.post_smooth.weight.device != x.device:
                self.post_smooth = self.post_smooth.to(x.device)
            x = self.post_smooth(x)

        return x
