from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from backend2.l2.freeze import DEFAULT_HOOK_LAYERS, FeatureFreezeCollector
from backend2.l2.model_factory import build_l2_model
from backend2.l2.probe import ProbeController

from .config import L3PreConfig
from .io_utils import (
    ensure_dir,
    get_probe_feature_at,
    infer_run_dir_from_pred,
    load_freeze_layer_feats,
    load_preds,
    probe_layer_covered_samples,
    safe_layer_filename,
    try_load_probe,
)
from .layer_ranking import summarize_layers
from .plot_features import plot_layer_channels_atlas
from .plot_model import plot_model_kernel_atlas
from .plot_ranking import plot_layer_ranking
from .plot_residual import plot_sample_map_atlas
from .selectors import compute_residual_ch0, rmse_per_sample, select_channels, select_samples


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _device_of(raw: str) -> torch.device:
    requested = str(raw or "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _infer_model_type(infer_cfg: Dict[str, Any]) -> str:
    model_cfg = dict(infer_cfg.get("model") or {})
    return str(infer_cfg.get("model_type", model_cfg.get("type", "unet"))).lower()


def _infer_channels_from_state_dict(model_state: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None

    for key in ("enc1.block.0.weight", "stem.0.weight", "encoder.proj.weight"):
        w = model_state.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 4:
            in_channels = int(w.shape[1])
            break

    for key in ("head.weight", "head.proj.weight"):
        w = model_state.get(key)
        if isinstance(w, torch.Tensor) and w.ndim >= 1:
            out_channels = int(w.shape[0])
            break

    return in_channels, out_channels


def _load_ckpt_state(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _category_dir(out_dir: Path, category: str) -> Path:
    parts = list(out_dir.parts)
    if "l3_preview" in parts:
        idx = parts.index("l3_preview")
        base = Path(*parts[: idx + 1])
        tail = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        path = base / category / tail
    else:
        path = out_dir / category
    ensure_dir(path)
    return path


def _load_model_for_preview(pred_npz: Path, in_channels: int, out_channels: int, cfg: L3PreConfig) -> Optional[torch.nn.Module]:
    run_dir = infer_run_dir_from_pred(pred_npz)
    if run_dir is None:
        _warn("model atlas unavailable: cannot infer run_dir from pred_npz")
        return None

    infer_cfg = _read_json_if_exists(run_dir / "infer_config.json")
    model_type = _infer_model_type(infer_cfg)

    device = _device_of(cfg.device)
    ckpt_path = run_dir / "ckpt" / str(cfg.ckpt_name)
    if not ckpt_path.exists():
        _warn(f"model atlas unavailable: checkpoint not found: {ckpt_path}")
        return None

    state = _load_ckpt_state(ckpt_path, device)
    model_state = state.get("model_state")
    if not isinstance(model_state, dict):
        _warn(f"model atlas unavailable: invalid checkpoint format: {ckpt_path}")
        return None

    ckpt_in, ckpt_out = _infer_channels_from_state_dict(model_state)
    resolved_in = int(ckpt_in) if ckpt_in is not None else int(in_channels)
    resolved_out = int(ckpt_out) if ckpt_out is not None else int(out_channels)

    model = build_l2_model(
        infer_cfg,
        in_channels=resolved_in,
        out_channels=resolved_out,
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()
    _info(
        f"model atlas loader: model_type={model_type}, ckpt={ckpt_path}, "
        f"in_channels={resolved_in}, out_channels={resolved_out}"
    )
    return model


def _collect_online_rep_features(
    pred_npz: Path,
    representative_idx: int,
    input_arr: Optional[np.ndarray],
    pred_arr: np.ndarray,
    cfg: L3PreConfig,
) -> Dict[str, np.ndarray]:
    if input_arr is None:
        _warn("online fallback unavailable: preds_test.npz missing 'input'")
        return {}

    run_dir = infer_run_dir_from_pred(pred_npz)
    if run_dir is None:
        _warn("online fallback unavailable: cannot infer run_dir from pred_npz")
        return {}

    if representative_idx < 0 or representative_idx >= int(input_arr.shape[0]):
        _warn(f"online fallback unavailable: representative_idx out of range ({representative_idx})")
        return {}

    infer_cfg = _read_json_if_exists(run_dir / "infer_config.json")
    model_type = _infer_model_type(infer_cfg)
    in_channels = int(input_arr.shape[1])
    out_channels = int(pred_arr.shape[1])

    device = _device_of(cfg.device)
    ckpt_path = run_dir / "ckpt" / str(cfg.ckpt_name)
    if not ckpt_path.exists():
        _warn(f"online fallback unavailable: checkpoint not found: {ckpt_path}")
        return {}

    state = _load_ckpt_state(ckpt_path, device)
    model_state = state.get("model_state")
    if not isinstance(model_state, dict):
        _warn(f"online fallback unavailable: invalid checkpoint format: {ckpt_path}")
        return {}

    ckpt_in, ckpt_out = _infer_channels_from_state_dict(model_state)
    resolved_in = int(ckpt_in) if ckpt_in is not None else int(in_channels)
    resolved_out = int(ckpt_out) if ckpt_out is not None else int(out_channels)

    model = build_l2_model(
        infer_cfg,
        in_channels=resolved_in,
        out_channels=resolved_out,
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    hook_layers = list(cfg.hook_layers) if cfg.hook_layers else list(DEFAULT_HOOK_LAYERS)
    collector = FeatureFreezeCollector(layer_patterns=hook_layers)
    probe = ProbeController(
        {
            "enabled": True,
            "record_level": 0,
            "hook_layers": hook_layers,
        },
        callbacks=[collector],
    )

    x = np.asarray(input_arr[representative_idx : representative_idx + 1], dtype=np.float32)
    x_t = torch.from_numpy(x).to(device)
    with torch.no_grad():
        _ = model(x_t, probe=probe)

    layer_out = {}
    for name, arr in collector.as_arrays().items():
        if arr.ndim == 4 and int(arr.shape[0]) >= 1:
            layer_out[str(name)] = np.asarray(arr[0], dtype=np.float32)

    if layer_out:
        dump_path = cfg.out_dir_path / "representative_online_features.npz"
        np.savez_compressed(dump_path, **{k: v for k, v in layer_out.items()})
        _info(
            f"online fallback captured representative layer features: {dump_path} "
            f"(model_type={model_type}, in_channels={resolved_in}, out_channels={resolved_out})"
        )
    else:
        _warn("online fallback captured no layer outputs")
    return layer_out


def run_l3_pre_preview(config: L3PreConfig | Dict[str, Any]) -> Dict[str, Any]:
    cfg = config if isinstance(config, L3PreConfig) else L3PreConfig.from_dict(config)
    cfg.validate()

    pred_npz = cfg.pred_npz_path
    probe_dir = cfg.probe_dir_path
    out_dir = cfg.out_dir_path
    ensure_dir(out_dir)
    sample_atlas_dir = _category_dir(out_dir, "atlas_io_residual")
    feature_atlas_dir = _category_dir(out_dir, "atlas_feature_flow")
    model_atlas_dir = _category_dir(out_dir, "atlas_model_kernels")

    preds = load_preds(pred_npz)
    gt = preds["gt"]
    pred = preds["pred"]
    pair_nt = preds["pair_nt"]
    input_arr = preds.get("input")
    sample_points_xy = preds.get("sample_points_xy")
    n_total = int(gt.shape[0])

    residual = compute_residual_ch0(gt, pred)
    rmse = rmse_per_sample(residual)
    atlas_sample_k = min(64, n_total)
    selected_samples = select_samples(cfg.sample_strategy, atlas_sample_k, rmse, cfg.seed)

    if input_arr is None:
        _warn("preds_test.npz missing 'input'; fallback to pred for input atlas")
        input_arr = pred

    atlas_input = sample_atlas_dir / "atlas_input__samples.png"
    atlas_output = sample_atlas_dir / "atlas_output__samples.png"
    atlas_target = sample_atlas_dir / "atlas_target__samples.png"
    atlas_residual = sample_atlas_dir / "atlas_residual__samples.png"

    if bool(cfg.plot_io_residual):
        plot_sample_map_atlas(
            out_path=atlas_input,
            maps_hw=np.asarray(input_arr)[:, 0],
            pair_nt=pair_nt,
            sample_idx=selected_samples,
            title=f"Input Samples | strategy={cfg.sample_strategy}",
            cmap="RdBu_r",
            symmetric=False,
            sample_points_xy=sample_points_xy,
        )

        plot_sample_map_atlas(
            out_path=atlas_output,
            maps_hw=pred[:, 0],
            pair_nt=pair_nt,
            sample_idx=selected_samples,
            title=f"Output Samples | strategy={cfg.sample_strategy}",
            cmap="RdBu_r",
            symmetric=False,
        )

        plot_sample_map_atlas(
            out_path=atlas_target,
            maps_hw=gt[:, 0],
            pair_nt=pair_nt,
            sample_idx=selected_samples,
            title=f"Target Samples | strategy={cfg.sample_strategy}",
            cmap="RdBu_r",
            symmetric=False,
        )

        plot_sample_map_atlas(
            out_path=atlas_residual,
            maps_hw=residual,
            pair_nt=pair_nt,
            sample_idx=selected_samples,
            title=f"Residual Samples | strategy={cfg.sample_strategy}",
            cmap="RdBu_r",
            symmetric=True,
        )
    else:
        _info("skip atlas_io_residual: plot_io_residual=False")

    summary_data, snapshots, probe_warnings = try_load_probe(probe_dir)
    for w in probe_warnings:
        _warn(w)

    run_config: Dict[str, Any] = {
        "params": cfg.to_dict(),
        "totals": {
            "num_samples": n_total,
            "pred_shape": list(gt.shape),
        },
        "selected_samples": [int(v) for v in selected_samples],
        "atlas_sample_k": int(atlas_sample_k),
        "selected_sample_rmse": {str(int(i)): float(rmse[int(i)]) for i in selected_samples},
        "selected_layers": [],
        "layers": {},
        "warnings": probe_warnings,
        "skipped_layers": {},
        "online_fallback": {
            "enabled": bool(cfg.online_fallback),
            "used": False,
            "captured_layers": 0,
        },
        "plot_flags": {
            "plot_io_residual": bool(cfg.plot_io_residual),
            "plot_feature_flow": bool(cfg.plot_feature_flow),
            "plot_model_kernels": bool(cfg.plot_model_kernels),
        },
        "outputs": {
        },
    }

    if bool(cfg.plot_io_residual):
        run_config["outputs"].update(
            {
                "atlas_input__samples": str(atlas_input),
                "atlas_output__samples": str(atlas_output),
                "atlas_target__samples": str(atlas_target),
                "atlas_residual__samples": str(atlas_residual),
            }
        )

    if sample_points_xy is not None:
        run_config["sample_points_count"] = int(np.asarray(sample_points_xy).shape[0])

    _info(f"N={n_total}")
    _info(f"selected samples ({len(selected_samples)}): {selected_samples}")

    layer_rows = summarize_layers(summary_data) if summary_data is not None else []
    if summary_data is None:
        _warn("probe summary unavailable; ranking disabled, will rely on fallback layer order")
    elif len(layer_rows) == 0:
        _warn("probe summary has no usable layer records; ranking disabled, will rely on fallback layer order")

    if bool(cfg.plot_feature_flow) and len(layer_rows) > 0:
        layer_topk = max(0, min(int(cfg.layer_topk), len(layer_rows)))
        selected_layers = [str(v["name"]) for v in layer_rows[:layer_topk]]
        run_config["selected_layers"] = selected_layers

        ranking_png = feature_atlas_dir / "probe_layer_ranking.png"
        plot_layer_ranking(ranking_png, layer_rows, selected_layers)
        run_config["outputs"]["probe_layer_ranking"] = str(ranking_png)

        _info(f"selected layers ({len(selected_layers)}): {selected_layers}")
    elif not bool(cfg.plot_feature_flow):
        _info("skip atlas_feature_flow ranking: plot_feature_flow=False")

    representative_idx = int(cfg.sample_idx) if cfg.sample_idx is not None else (int(selected_samples[0]) if selected_samples else 0)
    representative_idx = max(0, min(representative_idx, n_total - 1)) if n_total > 0 else 0

    snapshots = snapshots or {}

    # 代表样本的层流可视化使用“全层顺序”（按 probe records 首次出现顺序），不截断 layer_topk。
    records = summary_data.get("records", []) if isinstance(summary_data, dict) else []
    layer_flow_order = []
    seen = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        layer_flow_order.append(name)
    if not layer_flow_order:
        layer_flow_order = [str(v["name"]) for v in layer_rows]
    if not layer_flow_order:
        layer_flow_order = list(cfg.hook_layers) if cfg.hook_layers else list(DEFAULT_HOOK_LAYERS)

    # 第一步：为代表样本抓取所有层的特征，后面在每层内按 shared index 裁剪。
    layer_rep_feature: Dict[str, np.ndarray] = {}
    layer_rep_source: Dict[str, str] = {}

    online_rep_features: Dict[str, np.ndarray] = {}
    online_attempted = False

    for layer_name in layer_flow_order:
        layer_arrs = snapshots.get(layer_name, [])
        probe_cover = probe_layer_covered_samples(layer_arrs)
        use_probe = probe_cover > representative_idx
        feat_rep: Optional[np.ndarray] = None
        source = "none"

        if use_probe:
            feat_rep = get_probe_feature_at(layer_arrs, representative_idx)
            source = "probe"

        if feat_rep is None:
            feats_freeze = load_freeze_layer_feats(pred_npz, layer_name)
            if feats_freeze is not None and 0 <= representative_idx < int(feats_freeze.shape[0]):
                feat_rep = np.asarray(feats_freeze[representative_idx], dtype=np.float32)
                source = "freeze"

        if feat_rep is None and bool(cfg.online_fallback):
            if not online_attempted:
                online_attempted = True
                online_rep_features = _collect_online_rep_features(
                    pred_npz=pred_npz,
                    representative_idx=representative_idx,
                    input_arr=input_arr,
                    pred_arr=pred,
                    cfg=cfg,
                )
                run_config["online_fallback"]["used"] = bool(len(online_rep_features) > 0)
                run_config["online_fallback"]["captured_layers"] = int(len(online_rep_features))

            if layer_name in online_rep_features:
                feat_rep = np.asarray(online_rep_features[layer_name], dtype=np.float32)
                source = "online"

        if feat_rep is None or feat_rep.ndim != 3:
            run_config["skipped_layers"][layer_name] = {
                "reason": "cannot fetch representative sample feature",
                "details": f"sample_idx={representative_idx}",
            }
            _warn(f"skip layer {layer_name}: representative sample unavailable")
            continue

        layer_rep_feature[layer_name] = feat_rep.astype(np.float32, copy=False)
        layer_rep_source[layer_name] = source

    if not bool(cfg.plot_feature_flow):
        _info("skip atlas_feature_flow: plot_feature_flow=False")
    elif not layer_rep_feature:
        run_cfg_path = out_dir / "run_config.json"
        run_cfg_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        _warn("no valid layer feature found for representative sample; stop at sample atlases + ranking")
        _info(f"saved run config: {run_cfg_path}")
        return run_config

    if bool(cfg.plot_feature_flow):
        layer_names_available = list(layer_rep_feature.keys())
        target_k = max(0, int(cfg.channel_k))
        enough_layers = [
            name for name in layer_names_available if int(layer_rep_feature[name].shape[0]) >= target_k
        ]
        if enough_layers:
            anchor_layer = enough_layers[0]
        else:
            anchor_layer = max(layer_names_available, key=lambda name: int(layer_rep_feature[name].shape[0]))
        anchor_feat = layer_rep_feature[anchor_layer]
        anchor_k = max(0, min(target_k, int(anchor_feat.shape[0])))
        shared_channels = select_channels(anchor_feat, anchor_k, str(cfg.channel_select))
        run_config["shared_channel_indices"] = [int(v) for v in shared_channels]
        run_config["shared_channel_anchor_layer"] = anchor_layer

        for layer_name in layer_flow_order:
            if layer_name not in layer_rep_feature:
                continue
            c_total_layer = int(layer_rep_feature[layer_name].shape[0])
            layer_channels = [int(v) for v in shared_channels if int(v) < c_total_layer]
            if not layer_channels:
                local_k = max(0, min(int(cfg.channel_k), c_total_layer))
                layer_channels = select_channels(layer_rep_feature[layer_name], local_k, str(cfg.channel_select))

            layer_rec: Dict[str, Any] = {
                "source": layer_rep_source.get(layer_name, "none"),
                "c_total": c_total_layer,
                "channels_for_channels_atlas": [int(v) for v in layer_channels],
                "representative_idx": representative_idx,
            }

            layer_file = safe_layer_filename(layer_name)
            channels_png = feature_atlas_dir / f"atlas_feat__{layer_file}__channels.png"
            plot_layer_channels_atlas(
                out_path=channels_png,
                layer_name=layer_name,
                feat_chw=layer_rep_feature[layer_name],
                selected_channels=layer_channels,
                sample_index=representative_idx,
                shared_c_total=c_total_layer,
            )

            run_config["outputs"][f"atlas_feat__{layer_name}__channels"] = str(channels_png)
            run_config["layers"][layer_name] = layer_rec

            _info(
                f"layer={layer_name} source={layer_rec['source']} C={layer_rec['c_total']} "
                f"plot_channels={len(layer_channels)}"
            )

    model_atlas_outputs: Dict[str, str] = {}
    if bool(cfg.plot_model_kernels):
        model = _load_model_for_preview(
            pred_npz=pred_npz,
            in_channels=int(pred.shape[1]),
            out_channels=int(gt.shape[1]),
            cfg=cfg,
        )
        if model is not None:
            for layer_name, layer_info in plot_model_kernel_atlas(model=model, out_dir=model_atlas_dir):
                model_atlas_outputs[f"atlas_model__{layer_name}__kernels"] = str(layer_info)
            _info(f"model atlas generated: {len(model_atlas_outputs)} layer(s)")
        else:
            _warn("skip model kernel atlas: model unavailable")
    else:
        _info("skip atlas_model_kernels: plot_model_kernels=False")
    run_config["outputs"].update(model_atlas_outputs)

    run_cfg_path = out_dir / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    _info(f"saved run config: {run_cfg_path}")
    return run_config
