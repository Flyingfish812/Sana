from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class L3PreConfig:
    pred_npz: str
    probe_dir: str
    out_dir: str
    sample_k: int = 64
    channel_k: int = 64
    layer_topk: int = 6
    sample_strategy: str = "mixed"  # worst_k | random_k | mixed
    channel_select: str = "energy_topk"  # first_k | energy_topk
    seed: int = 123
    sample_idx: Optional[int] = None
    sample_channels_k: int = 16
    online_fallback: bool = True
    device: str = "auto"
    ckpt_name: str = "model_best.pt"
    hook_layers: Optional[List[str]] = None
    plot_io_residual: bool = True
    plot_feature_flow: bool = True
    plot_model_kernels: bool = True

    def validate(self) -> None:
        if self.sample_strategy not in {"worst_k", "random_k", "mixed"}:
            raise ValueError(f"unsupported sample_strategy: {self.sample_strategy}")
        if self.channel_select not in {"first_k", "energy_topk"}:
            raise ValueError(f"unsupported channel_select: {self.channel_select}")
        if int(self.sample_k) < 0:
            raise ValueError("sample_k must be >= 0")
        if int(self.channel_k) < 0:
            raise ValueError("channel_k must be >= 0")
        if int(self.layer_topk) < 0:
            raise ValueError("layer_topk must be >= 0")
        if int(self.sample_channels_k) < 0:
            raise ValueError("sample_channels_k must be >= 0")
        if str(self.device).strip() == "":
            raise ValueError("device must not be empty")
        if str(self.ckpt_name).strip() == "":
            raise ValueError("ckpt_name must not be empty")

    @property
    def pred_npz_path(self) -> Path:
        return Path(self.pred_npz)

    @property
    def probe_dir_path(self) -> Path:
        return Path(self.probe_dir)

    @property
    def out_dir_path(self) -> Path:
        return Path(self.out_dir)

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "L3PreConfig":
        return L3PreConfig(
            pred_npz=str(cfg["pred_npz"]),
            probe_dir=str(cfg["probe_dir"]),
            out_dir=str(cfg["out_dir"]),
            sample_k=int(cfg.get("sample_k", 64)),
            channel_k=int(cfg.get("channel_k", 64)),
            layer_topk=int(cfg.get("layer_topk", 6)),
            sample_strategy=str(cfg.get("sample_strategy", "mixed")),
            channel_select=str(cfg.get("channel_select", "energy_topk")),
            seed=int(cfg.get("seed", 123)),
            sample_idx=(None if cfg.get("sample_idx", None) is None else int(cfg["sample_idx"])),
            sample_channels_k=int(cfg.get("sample_channels_k", 16)),
            online_fallback=bool(cfg.get("online_fallback", True)),
            device=str(cfg.get("device", "auto")),
            ckpt_name=str(cfg.get("ckpt_name", "model_best.pt")),
            hook_layers=(list(cfg.get("hook_layers", [])) or None),
            plot_io_residual=bool(cfg.get("plot_io_residual", True)),
            plot_feature_flow=bool(cfg.get("plot_feature_flow", True)),
            plot_model_kernels=bool(cfg.get("plot_model_kernels", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pred_npz": self.pred_npz,
            "probe_dir": self.probe_dir,
            "out_dir": self.out_dir,
            "sample_k": int(self.sample_k),
            "channel_k": int(self.channel_k),
            "layer_topk": int(self.layer_topk),
            "sample_strategy": self.sample_strategy,
            "channel_select": self.channel_select,
            "seed": int(self.seed),
            "sample_idx": self.sample_idx,
            "sample_channels_k": int(self.sample_channels_k),
            "online_fallback": bool(self.online_fallback),
            "device": self.device,
            "ckpt_name": self.ckpt_name,
            "hook_layers": list(self.hook_layers) if self.hook_layers else None,
            "plot_io_residual": bool(self.plot_io_residual),
            "plot_feature_flow": bool(self.plot_feature_flow),
            "plot_model_kernels": bool(self.plot_model_kernels),
        }
