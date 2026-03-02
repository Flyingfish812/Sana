from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import math

from backend2.l1.pipeline import run_l1_pipeline
from backend2.l2.infer import run_l2_infer
from backend2.l2.train import run_l2_train
from backend2.l2.utils import now_tag


MINI_DATASETS = ("h5_mini", "nc_mini", "sst_mini", "h5_oneshot_clone")


def mini_l1_configs(artifacts_dir: str = "artifacts", log_progress: bool = True) -> List[Dict[str, Any]]:
    """返回三组 mini 数据的 L1 构造配置。"""
    return [
        {
            "dataset_id": "h5_mini",
            "reader": {
                "kind": "h5",
                "path": "datasets/2D_rdb_NA_NA_mini.h5",
                "dataset": "data",
                "fill_value": 0.0,
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": artifacts_dir,
            "log_progress": bool(log_progress),
        },
        {
            "dataset_id": "nc_mini",
            "reader": {
                "kind": "nc",
                "path": "datasets/cylinder2d_mini.nc",
                "var_keys": ["u", "v"],
                "time_key": "tdim",
                "y_key": "ydim",
                "x_key": "xdim",
                "fill_value": 0.0,
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": artifacts_dir,
            "log_progress": bool(log_progress),
        },
        {
            "dataset_id": "sst_mini",
            "reader": {
                "kind": "mat",
                "path": "datasets/sst_weekly_mini.mat",
                "var": "sst",
                "lon_key": "lon",
                "lat_key": "lat",
                "time_key": "time",
                "fill_value": "global_mean",
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": artifacts_dir,
            "log_progress": bool(log_progress),
        },
        {
            "dataset_id": "h5_oneshot_clone",
            "reader": {
                "kind": "h5",
                "path": "datasets/2D_rdb_NA_NA_oneshot_clone.h5",
                "dataset": "data",
                "fill_value": 0.0,
            },
            "split": {
                "strategy": "temporal",
                "unit": "frame",
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 123,
            },
            "normalization": {"method": "zscore"},
            "artifacts_dir": artifacts_dir,
            "log_progress": bool(log_progress),
        },
    ]


def _l1_is_ready(artifacts_dir: str, dataset_id: str) -> bool:
    base = Path(artifacts_dir) / dataset_id / "L1"
    required = [
        base / "array5d_norm.npy",
        base / "manifest.json",
        base / "stats_train.json",
        base / "splits" / "train.npy",
        base / "splits" / "val.npy",
        base / "splits" / "test.npy",
    ]
    return all(path.exists() for path in required)


def _read_shape5d(artifacts_dir: str, dataset_id: str) -> List[int]:
    manifest = Path(artifacts_dir) / dataset_id / "L1" / "manifest.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    shape5d = payload.get("shape5d", [])
    return [int(v) for v in shape5d] if isinstance(shape5d, list) else []


def ensure_mini_l1(
    *,
    artifacts_dir: str = "artifacts",
    force_rebuild: bool = False,
    log_progress: bool = True,
) -> List[Dict[str, Any]]:
    """确保 mini 三数据集的 L1 冻结产物存在。"""
    outputs: List[Dict[str, Any]] = []
    for cfg in mini_l1_configs(artifacts_dir=artifacts_dir, log_progress=log_progress):
        dataset_id = str(cfg["dataset_id"])
        if force_rebuild or not _l1_is_ready(artifacts_dir=artifacts_dir, dataset_id=dataset_id):
            summary = run_l1_pipeline(cfg)
            outputs.append(
                {
                    "dataset_id": dataset_id,
                    "status": "rebuilt",
                    "shape5d": [int(v) for v in summary.shape5d],
                    "artifacts_dir": summary.artifacts_dir,
                }
            )
        else:
            outputs.append(
                {
                    "dataset_id": dataset_id,
                    "status": "reused",
                    "shape5d": _read_shape5d(artifacts_dir=artifacts_dir, dataset_id=dataset_id),
                    "artifacts_dir": str((Path(artifacts_dir) / dataset_id / "L1").resolve()),
                }
            )
    return outputs


def run_sideway_mini(
    *,
    sample_p: float,
    model_type: str = "unet",
    sample_sigma: float = 0.0,
    sample_seed: int = 123,
    train_steps: int | None = None,
    val_interval_steps: int = 50,
    log_interval_steps: int = 100,
    epochs: int | None = None,
    lr: float = 1e-3,
    batch_size: int = 8,
    artifacts_dir: str = "artifacts",
    exp_name: str = "sideway_unet",
    datasets: Iterable[str] = MINI_DATASETS,
    base_channels: int = 32,
    convs_per_stage: int = 2,
    force_rebuild_l1: bool = False,
    log_progress: bool = True,
    debug_plot_sample_index: int = 0,
    debug_plot_channel: int = 0,
) -> Dict[str, Any]:
    """运行 sideway 旁路：mini L1 -> L2 train -> L2 infer(四联图)。"""
    l1_bootstrap = ensure_mini_l1(
        artifacts_dir=artifacts_dir,
        force_rebuild=force_rebuild_l1,
        log_progress=log_progress,
    )

    sparse_input_cfg = {
        "enabled": True,
        "sample_p": float(sample_p),
        "sample_sigma": float(sample_sigma),
        "sample_seed": int(sample_seed),
        "append_mask_channel": True,
    }

    runs: List[Dict[str, Any]] = []
    selected_model_type = str(model_type).lower()
    if selected_model_type not in {"unet", "unet_legacy", "vit"}:
        raise ValueError(f"model_type must be 'unet', 'unet_legacy' or 'vit', got: {model_type}")

    for dataset_id in [str(v) for v in datasets]:
        run_name = f"sideway_p{float(sample_p):.3e}_{dataset_id}_{now_tag()}"

        if train_steps is None:
            if epochs is None:
                effective_train_steps = 1000
            else:
                l1_manifest = Path(artifacts_dir) / dataset_id / "L1" / "manifest.json"
                if l1_manifest.exists():
                    payload = json.loads(l1_manifest.read_text(encoding="utf-8"))
                    shape5d = payload.get("shape5d", [])
                    n_size = int(shape5d[0]) if len(shape5d) >= 2 else 1
                    t_size = int(shape5d[1]) if len(shape5d) >= 2 else 2
                    train_pairs_est = max(1, int(0.8 * n_size * max(1, t_size - 1)))
                    steps_per_epoch_est = max(1, math.ceil(train_pairs_est / max(1, int(batch_size))))
                    effective_train_steps = max(1, int(epochs)) * steps_per_epoch_est
                else:
                    effective_train_steps = max(1, int(epochs)) * 100
        else:
            effective_train_steps = int(train_steps)

        if selected_model_type in {"unet", "unet_legacy"}:
            model_cfg = {
                "base_channels": int(base_channels),
                "convs_per_stage": int(convs_per_stage),
                "depth": 4,
            }
        else:
            model_cfg = {
                "patch_size": 16,
                "embed_dim": 64,
                "depth": 10,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
                "attention_dropout": 0.15,
                "droppath": 0.2,
            }

        train_cfg = {
            "dataset_id": dataset_id,
            "artifacts_dir": artifacts_dir,
            "exp_name": exp_name,
            "run_name": run_name,
            "model_type": selected_model_type,
            "device": "auto",
            "seed": int(sample_seed),
            "target_offset": 1,
            "batch_size": int(batch_size),
            "num_workers": 0,
            "train_steps": int(effective_train_steps),
            "val_interval_steps": int(val_interval_steps),
            "log_interval_steps": int(log_interval_steps),
            "lr": float(lr),
            "model": model_cfg,
            "sparse_input": sparse_input_cfg,
            "log_progress": bool(log_progress),
            "tqdm": False,
        }
        infer_cfg = {
            "dataset_id": dataset_id,
            "artifacts_dir": artifacts_dir,
            "exp_name": exp_name,
            "run_name": run_name,
            "model_type": selected_model_type,
            "device": "auto",
            "target_offset": 1,
            "batch_size": int(batch_size),
            "num_workers": 0,
            "ckpt_name": "model_best.pt",
            "freeze_features": False,
            "model": model_cfg,
            "sparse_input": sparse_input_cfg,
            "probe": {"enabled": False},
            "debug_plot": {
                "enabled": True,
                "sample_index": int(debug_plot_sample_index),
                "channel": int(debug_plot_channel),
                "name": f"quad_sample{int(debug_plot_sample_index):04d}_ch{int(debug_plot_channel):02d}.png",
            },
            "log_progress": bool(log_progress),
            "tqdm": False,
        }

        train_summary = run_l2_train(train_cfg)
        infer_summary = run_l2_infer(infer_cfg)
        runs.append(
            {
                "dataset_id": dataset_id,
                "run_name": run_name,
                "model_type": selected_model_type,
                "sample_p": float(sample_p),
                "sample_sigma": float(sample_sigma),
                "train": train_summary,
                "infer": infer_summary,
                "metrics": infer_summary.get("metrics", {}),
                "quad_plot": infer_summary.get("debug_plot", ""),
            }
        )

    return {
        "artifacts_dir": str(Path(artifacts_dir).resolve()),
        "exp_name": exp_name,
        "model_type": selected_model_type,
        "sample_p": float(sample_p),
        "sample_sigma": float(sample_sigma),
        "sample_seed": int(sample_seed),
        "train_steps": (None if train_steps is None else int(train_steps)),
        "val_interval_steps": int(val_interval_steps),
        "epochs": (None if epochs is None else int(epochs)),
        "batch_size": int(batch_size),
        "l1_bootstrap": l1_bootstrap,
        "runs": runs,
    }


def run_sideway_sample_sweep(
    sample_ps: Iterable[float],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """按给定采样率序列执行 sideway 实验，用于排查均值坍缩。"""
    outputs: List[Dict[str, Any]] = []
    for sample_p in sample_ps:
        outputs.append(run_sideway_mini(sample_p=float(sample_p), **kwargs))
    return outputs
