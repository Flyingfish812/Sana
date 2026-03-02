from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifact_io import ArtifactManager
from .data import PairDataset, load_l1_array_mmap, load_split_pairs
from .freeze import FeatureFreezeCollector, build_probe_config_for_freeze, resolve_freeze_layers, save_frozen_features
from .metrics import regression_metrics
from .model_unet import BaselineUNet
from .probe import ProbeController
from .utils import dump_json, iter_progress, log_progress


def _device_of(cfg: Dict[str, Any]) -> torch.device:
    """根据配置解析推理设备，支持 auto/cuda/cpu。"""
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_l2_infer(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行 L2 推理：加载测试对、推理预测、计算指标并写出产物。"""
    t0_all = time.perf_counter()
    dataset_id = str(config["dataset_id"])
    artifacts_dir = str(config.get("artifacts_dir", "artifacts"))
    exp_name = str(config.get("exp_name", "baseline_unet"))
    run_name = str(config["run_name"])
    log_enabled = bool(config.get("log_progress", True))
    use_tqdm = bool(config.get("tqdm", True))

    manager = ArtifactManager(artifacts_dir=artifacts_dir, dataset_id=dataset_id, exp_name=exp_name, run_name=run_name)
    manager.ensure_run_dirs()
    dump_json(manager.infer_config_json, config)
    log_progress(log_enabled, "L2-INFER", f"start dataset={dataset_id}, run={manager.run_name}")

    freeze_features = bool(config.get("freeze_features", True))
    freeze_mode = str(config.get("freeze_mode", "test"))
    if freeze_mode != "test":
        raise ValueError("freeze_mode currently supports 'test' only")
    log_progress(log_enabled, "L2-INFER", f"freeze_features={freeze_features}, freeze_mode={freeze_mode}")

    t_data = time.perf_counter()
    target_offset = int(config.get("target_offset", 1))
    array5d, manifest = load_l1_array_mmap(manager)
    test_pairs = load_split_pairs(manager, array5d, manifest, "test", target_offset)
    if len(test_pairs) == 0:
        raise ValueError("empty test pairs from L1 split")

    test_ds = PairDataset(array5d=array5d, pairs=test_pairs, target_offset=target_offset)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    log_progress(
        log_enabled,
        "L2-INFER",
        f"data ready: shape={tuple(array5d.shape)}, test_pairs={len(test_pairs)}, test_steps={len(test_loader)}, dt={time.perf_counter()-t_data:.2f}s",
    )

    ckpt_name = str(config.get("ckpt_name", "model_best.pt"))
    ckpt_path = Path(config.get("ckpt_path", manager.ckpt_path(ckpt_name)))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    log_progress(log_enabled, "L2-INFER", f"checkpoint={ckpt_path}")

    device = _device_of(config)
    in_channels = int(array5d.shape[-1])
    model_cfg = dict(config.get("model", {}))
    model = BaselineUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
        convs_per_stage=int(model_cfg.get("convs_per_stage", 2)),
    ).to(device)
    log_progress(log_enabled, "L2-INFER", f"model ready: in_channels={in_channels}, device={device}")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    freeze_collector = None
    if freeze_features:
        freeze_layers = resolve_freeze_layers(config)
        if not freeze_layers:
            raise ValueError("freeze_layers resolved to empty list")
        probe_cfg = build_probe_config_for_freeze(config, freeze_layers)
        freeze_collector = FeatureFreezeCollector(layer_patterns=freeze_layers)
        probe = ProbeController(probe_cfg, callbacks=[freeze_collector])
        log_progress(log_enabled, "L2-INFER", f"freeze layers={freeze_layers}")
    else:
        probe = ProbeController(config.get("probe"))

    xs, masks, ys, preds, nts = [], [], [], [], []
    t_forward = time.perf_counter()
    with torch.no_grad():
        infer_iter = iter_progress(
            test_loader,
            enabled=log_enabled,
            use_tqdm=use_tqdm,
            desc=f"[L2-INFER][{dataset_id}] forward",
            total=len(test_loader),
            leave=False,
        )
        for batch in infer_iter:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            pred = model(x, probe=probe)

            xs.append(x.cpu().numpy())
            masks.append(mask.cpu().numpy())
            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
            nts.append(torch.stack([batch["n"], batch["t"]], dim=1).cpu().numpy())
    log_progress(log_enabled, "L2-INFER", f"forward done: dt={time.perf_counter()-t_forward:.2f}s")

    x_all = np.concatenate(xs, axis=0)
    m_all = np.concatenate(masks, axis=0)
    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(preds, axis=0)
    nt_all = np.concatenate(nts, axis=0)

    metrics = regression_metrics(y_all, p_all)
    metrics["num_test_pairs"] = int(y_all.shape[0])
    log_progress(log_enabled, "L2-INFER", f"metrics: mse={metrics.get('mse')}, mae={metrics.get('mae')}, rmse={metrics.get('rmse')}")

    preds_path = manager.infer_dir / "preds_test.npz"
    np.savez_compressed(
        preds_path,
        input=x_all,
        obs=x_all * m_all,
        mask=m_all,
        gt=y_all,
        pred=p_all,
        pair_nt=nt_all,
    )
    metrics_path = manager.infer_dir / "metrics_test.json"
    dump_json(metrics_path, metrics)
    log_progress(log_enabled, "L2-INFER", f"saved infer artifacts: preds={preds_path}, metrics={metrics_path}")

    probe_outputs = {}
    if probe.cfg.enabled:
        probe_outputs = probe.save(manager.probe_dir)
        log_progress(log_enabled, "L2-INFER", f"probe outputs saved: {probe_outputs}")

    freeze_outputs = {}
    if freeze_features:
        layer_features = freeze_collector.as_arrays() if freeze_collector is not None else {}
        if not layer_features:
            raise ValueError("freeze_features=True but no layer outputs were captured")
        freeze_outputs = save_frozen_features(
            manager=manager,
            dataset_id=dataset_id,
            exp_name=exp_name,
            run_name=run_name,
            layer_features=layer_features,
            pair_nt=nt_all,
        )
        log_progress(log_enabled, "L2-INFER", f"freeze outputs saved: {freeze_outputs}")

    log_progress(log_enabled, "L2-INFER", f"finished dataset={dataset_id}, run={manager.run_name}, total_dt={time.perf_counter()-t0_all:.2f}s")

    return {
        **manager.summary(),
        "preds_test": str(preds_path),
        "metrics_test": str(metrics_path),
        "ckpt_used": str(ckpt_path),
        "probe_outputs": probe_outputs,
        "freeze_outputs": freeze_outputs,
        "metrics": metrics,
    }
