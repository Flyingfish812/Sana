from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import dump_json


class ProbeCallback(Protocol):
    """Probe 回调协议：接收层名、张量与上下文并返回处理后的张量。"""
    def __call__(self, name: str, tensor: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """定义探针回调签名。"""
        ...


@dataclass
class ProbeConfig:
    """Probe 开关、记录级别与层级过滤配置。"""
    enabled: bool = False
    record_level: int = 0
    hook_layers: Optional[List[str]] = None
    transforms: Optional[List[Dict[str, Any]]] = None
    snapshot_max_layers: int = 8
    snapshot_max_samples: int = 4
    allow_full_dump: bool = False

    @staticmethod
    def from_dict(cfg: Dict[str, Any] | None) -> "ProbeConfig":
        """从普通字典构建 ProbeConfig。"""
        cfg = cfg or {}
        return ProbeConfig(
            enabled=bool(cfg.get("enabled", False)),
            record_level=int(cfg.get("record_level", 0)),
            hook_layers=list(cfg.get("hook_layers", [])) or None,
            transforms=list(cfg.get("transforms", [])) or None,
            snapshot_max_layers=int(cfg.get("snapshot_max_layers", 8)),
            snapshot_max_samples=int(cfg.get("snapshot_max_samples", 4)),
            allow_full_dump=bool(cfg.get("allow_full_dump", False)),
        )


class ProbeController:
    """在模型前向过程中执行变换、统计与可选快照落盘。"""
    def __init__(self, cfg: Dict[str, Any] | ProbeConfig | None = None, callbacks: Optional[List[ProbeCallback]] = None):
        """初始化 probe 控制器并构建目标层变换映射。"""
        self.cfg = cfg if isinstance(cfg, ProbeConfig) else ProbeConfig.from_dict(cfg)
        self.callbacks = callbacks or []
        self._summaries: List[Dict[str, Any]] = []
        self._snapshots: Dict[str, List[np.ndarray]] = {}
        self._full_dump: Dict[str, List[np.ndarray]] = {}
        self._transform_map: Dict[str, List[Dict[str, Any]]] = {}
        for spec in self.cfg.transforms or []:
            target = str(spec.get("target", ""))
            if not target:
                continue
            self._transform_map.setdefault(target, []).append(spec)

    def _match(self, name: str) -> bool:
        """判断当前层名是否命中 hook_layers 过滤规则。"""
        hooks = self.cfg.hook_layers
        if not hooks:
            return True
        return any(fnmatch.fnmatch(name, pattern) for pattern in hooks)

    @staticmethod
    def _summary(name: str, x: torch.Tensor) -> Dict[str, Any]:
        """提取张量统计摘要，用于后续分析。"""
        d = x.detach()
        xf = d.float()
        energy = float((xf * xf).mean().item())
        spec = torch.fft.fft2(xf, dim=(-2, -1))
        amp = torch.abs(spec)
        return {
            "name": name,
            "shape": list(d.shape),
            "dtype": str(d.dtype),
            "device": str(d.device),
            "mean": float(xf.mean().item()),
            "std": float(xf.std(unbiased=False).item()),
            "energy": energy,
            "spec_mean_amp": float(amp.mean().item()),
        }

    @staticmethod
    def _fft_mask(h: int, w: int, ratio: float, lowpass: bool, device: torch.device) -> torch.Tensor:
        """构造频域低通/高通掩码。"""
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r0 = max(1.0, min(h, w) * float(ratio))
        return (rr <= r0) if lowpass else (rr >= r0)

    def _apply_one_transform(self, x: torch.Tensor, spec: Dict[str, Any]) -> torch.Tensor:
        """对张量应用单个探针变换规则。"""
        kind = str(spec.get("kind", "")).lower()
        if kind in ("zero", "zeros"):
            return torch.zeros_like(x)
        if kind == "truncate":
            min_v = float(spec.get("min", -1.0))
            max_v = float(spec.get("max", 1.0))
            return torch.clamp(x, min=min_v, max=max_v)
        if kind == "channel_select":
            keep = set(int(v) for v in spec.get("channels", []))
            y = torch.zeros_like(x)
            c = int(x.shape[1]) if x.ndim >= 2 else 0
            for i in range(c):
                if i in keep:
                    y[:, i] = x[:, i]
            return y
        if kind in ("fft_lowpass", "fft_highpass"):
            if x.ndim < 4:
                return x
            ratio = float(spec.get("ratio", 0.25))
            xf = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
            mask = self._fft_mask(int(x.shape[-2]), int(x.shape[-1]), ratio, lowpass=(kind == "fft_lowpass"), device=x.device)
            xf = xf * mask.to(xf.dtype).view(1, 1, x.shape[-2], x.shape[-1])
            return torch.fft.ifft2(torch.fft.ifftshift(xf, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype=x.dtype)
        return x

    def apply(self, name: str, tensor: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """在指定层上执行 probe 逻辑并返回处理后的张量。"""
        if not self.cfg.enabled or not self._match(name):
            return tensor

        ctx = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            **(context or {}),
        }
        x = tensor

        for spec in self._transform_map.get(name, []):
            x = self._apply_one_transform(x, spec)

        for callback in self.callbacks:
            x = callback(name, x, ctx)

        level = int(self.cfg.record_level)
        self._summaries.append(self._summary(name, x))

        if level >= 1:
            if len(self._snapshots) < self.cfg.snapshot_max_layers or name in self._snapshots:
                arr = x.detach().to(dtype=torch.float16).cpu().numpy()
                bucket = self._snapshots.setdefault(name, [])
                if len(bucket) < self.cfg.snapshot_max_samples:
                    bucket.append(arr)

        if level >= 2:
            if not self.cfg.allow_full_dump:
                raise ValueError("record_level=2 requires allow_full_dump=True")
            self._full_dump.setdefault(name, []).append(x.detach().cpu().numpy())

        return x

    def save(self, out_dir: Path) -> Dict[str, str]:
        """将 probe 摘要/快照/全量 dump 保存到输出目录。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "probe_summary.json"
        dump_json(summary_path, {"records": self._summaries})
        outputs = {"summary": str(summary_path)}

        if self._snapshots:
            snap_path = out_dir / "probe_snapshots_level1.npz"
            np.savez_compressed(snap_path, **{f"{k}__{i}": v for k, arrs in self._snapshots.items() for i, v in enumerate(arrs)})
            outputs["snapshots"] = str(snap_path)

        if self._full_dump:
            dump_path = out_dir / "probe_dump_level2.npz"
            np.savez_compressed(dump_path, **{f"{k}__{i}": v for k, arrs in self._full_dump.items() for i, v in enumerate(arrs)})
            outputs["full_dump"] = str(dump_path)

        return outputs


class ProbeDebugVisualizer:
    """用于保存 input/pred/target 三联图的调试可视化工具。"""
    @staticmethod
    def save_triplet(
        input_hw: np.ndarray,
        pred_hw: np.ndarray,
        target_hw: np.ndarray,
        out_path: Path,
        sample_points_xy: Optional[np.ndarray] = None,
        cmap: str = "RdBu_r",
    ) -> None:
        """将单样本输入、预测、真值保存为并排图片。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        titles = ["input", "pred", "target"]
        arrays = [input_hw, pred_hw, target_hw]
        for ax, title, arr in zip(axes, titles, arrays):
            im = ax.imshow(arr, cmap=cmap)
            ax.set_title(title)
            ax.set_axis_off()
            if sample_points_xy is not None and len(sample_points_xy) > 0:
                pts = np.asarray(sample_points_xy)
                ax.scatter(pts[:, 0], pts[:, 1], facecolors="none", edgecolors="k", s=24, linewidths=0.8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    @staticmethod
    def save_quadruplet(
        input_hw: np.ndarray,
        pred_hw: np.ndarray,
        target_hw: np.ndarray,
        residual_hw: np.ndarray,
        out_path: Path,
        sample_points_xy: Optional[np.ndarray] = None,
        cmap: str = "RdBu_r",
        residual_cmap: str = "bwr",
    ) -> None:
        """将单样本输入、预测、真值、残差保存为 2x2 四联图。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

        residual_arr = np.asarray(residual_hw, dtype=np.float32)
        finite = residual_arr[np.isfinite(residual_arr)]
        if finite.size == 0:
            residual_vmax = 1.0
        else:
            residual_vmax = float(np.max(np.abs(finite)))
            residual_vmax = max(residual_vmax, 1e-8)

        tiles = [
            ("input", input_hw, cmap),
            ("pred", pred_hw, cmap),
            ("target", target_hw, cmap),
            ("residual", residual_hw, residual_cmap),
        ]

        for ax, (title, arr, cm) in zip(axes.ravel(), tiles):
            if title == "residual":
                im = ax.imshow(arr, cmap=cm, vmin=-residual_vmax, vmax=residual_vmax)
            else:
                im = ax.imshow(arr, cmap=cm)
            ax.set_title(title)
            ax.set_axis_off()
            if sample_points_xy is not None and len(sample_points_xy) > 0 and title == "input":
                pts = np.asarray(sample_points_xy)
                ax.scatter(pts[:, 0], pts[:, 1], facecolors="none", edgecolors="k", s=22, linewidths=0.8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        fig.savefig(out_path, dpi=170)
        plt.close(fig)
