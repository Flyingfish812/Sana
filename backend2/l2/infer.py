from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifact_io import ArtifactManager
from .data import PairDataset, load_l1_array, load_split_pairs
from .metrics import regression_metrics
from .model_unet import BaselineUNet
from .probe import ProbeController
from .utils import dump_json


def _device_of(cfg: Dict[str, Any]) -> torch.device:
    """根据配置解析推理设备，支持 auto/cuda/cpu。"""
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_l2_infer(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行 L2 推理：加载测试对、推理预测、计算指标并写出产物。"""
    dataset_id = str(config["dataset_id"])
    artifacts_dir = str(config.get("artifacts_dir", "artifacts"))
    exp_name = str(config.get("exp_name", "baseline_unet"))
    run_name = str(config["run_name"])

    manager = ArtifactManager(artifacts_dir=artifacts_dir, dataset_id=dataset_id, exp_name=exp_name, run_name=run_name)
    manager.ensure_run_dirs()

    target_offset = int(config.get("target_offset", 1))
    split_tag = config.get("split_tag")
    array5d, manifest, norm = load_l1_array(manager)
    test_pairs = load_split_pairs(manager, manifest, "test", split_tag, target_offset)
    if len(test_pairs) == 0:
        raise ValueError("empty test pairs from L1 split")

    test_ds = PairDataset(array5d=array5d, pairs=test_pairs, norm=norm, target_offset=target_offset)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    ckpt_name = str(config.get("ckpt_name", "model_best.pt"))
    ckpt_path = Path(config.get("ckpt_path", manager.ckpt_path(ckpt_name)))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = _device_of(config)
    in_channels = int(array5d.shape[-1])
    model_cfg = dict(config.get("model", {}))
    model = BaselineUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
        convs_per_stage=int(model_cfg.get("convs_per_stage", 2)),
    ).to(device)

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    probe = ProbeController(config.get("probe"))

    xs, masks, ys, preds, nts = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            pred = model(x, probe=probe)

            xs.append(x.cpu().numpy())
            masks.append(mask.cpu().numpy())
            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
            nts.append(torch.stack([batch["n"], batch["t"]], dim=1).cpu().numpy())

    x_all = np.concatenate(xs, axis=0)
    m_all = np.concatenate(masks, axis=0)
    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(preds, axis=0)
    nt_all = np.concatenate(nts, axis=0)

    metrics = regression_metrics(y_all, p_all)
    metrics["num_test_pairs"] = int(y_all.shape[0])

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

    probe_outputs = {}
    if probe.cfg.enabled:
        probe_outputs = probe.save(manager.probe_dir)

    return {
        **manager.summary(),
        "preds_test": str(preds_path),
        "metrics_test": str(metrics_path),
        "ckpt_used": str(ckpt_path),
        "probe_outputs": probe_outputs,
        "metrics": metrics,
    }
