from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from backend2.l2.infer import run_l2_infer
from backend2.l2.train import run_l2_train
from backend2.l2.utils import now_tag


def _parse_datasets(raw: str) -> List[str]:
	"""解析逗号分隔的数据集列表。"""
	items = [x.strip() for x in raw.split(",") if x.strip()]
	if not items:
		raise ValueError("--datasets 不能为空")
	return items


def _parse_layers(raw: str) -> List[str]:
	"""解析逗号分隔的冻结层列表。"""
	items = [x.strip() for x in raw.split(",") if x.strip()]
	if not items:
		raise ValueError("--freeze-layers 不能为空")
	return items


def _parse_model_types(raw: str) -> List[str]:
	"""解析逗号分隔的模型类型列表。"""
	items = [x.strip().lower() for x in raw.split(",") if x.strip()]
	if not items:
		raise ValueError("--model-types 不能为空")
	allowed = {"unet", "unet_legacy", "vit"}
	invalid = [x for x in items if x not in allowed]
	if invalid:
		raise ValueError(f"--model-types 包含不支持的模型: {invalid}，仅支持 {sorted(allowed)}")
	return items


def _build_model_cfg(model_type: str, args: argparse.Namespace) -> Dict[str, Any]:
	"""按模型类型构建 model 配置。"""
	if model_type in {"unet", "unet_legacy"}:
		return {
			"base_channels": args.base_channels,
			"convs_per_stage": args.convs_per_stage,
			"depth": args.unet_legacy_depth,
		}
	if model_type == "vit":
		return {
			"patch_size": args.vit_patch_size,
			"embed_dim": args.vit_embed_dim,
			"depth": args.vit_depth,
			"num_heads": args.vit_num_heads,
			"mlp_ratio": args.vit_mlp_ratio,
			"dropout": args.vit_dropout,
			"attention_dropout": args.vit_attention_dropout,
			"droppath": args.vit_droppath,
		}
	raise ValueError(f"unsupported model_type: {model_type}")


def _build_exp_name(base_exp_name: str, model_type: str, split_exp_by_model: bool) -> str:
	"""按需将实验名拆分到模型粒度。"""
	if split_exp_by_model:
		return f"{base_exp_name}_{model_type}"
	return base_exp_name


def _build_train_cfg(
	args: argparse.Namespace,
	dataset_id: str,
	run_name: str,
	model_type: str,
	exp_name: str,
) -> Dict[str, Any]:
	"""构建单数据集 L2 训练配置。"""
	model_cfg = _build_model_cfg(model_type=model_type, args=args)
	return {
		"dataset_id": dataset_id,
		"artifacts_dir": args.artifacts_dir,
		"exp_name": exp_name,
		"run_name": run_name,
		"model_type": model_type,
		"device": args.device,
		"seed": args.seed,
		"target_offset": args.target_offset,
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"train_steps": args.train_steps,
		"val_interval_steps": args.val_interval_steps,
		"log_interval_steps": args.log_interval_steps,
		"lr": args.lr,
		"model": model_cfg,
		"sparse_input": {
			"enabled": bool(args.sparse_input),
			"sample_p": float(args.sample_p),
			"sample_sigma": float(args.sample_sigma),
			"sample_seed": int(args.sample_seed),
			"append_mask_channel": True,
		},
	}


def _build_infer_cfg(
	args: argparse.Namespace,
	dataset_id: str,
	run_name: str,
	model_type: str,
	exp_name: str,
) -> Dict[str, Any]:
	"""构建单数据集 L2 推理配置。"""
	freeze_layers = _parse_layers(args.freeze_layers)
	model_cfg = _build_model_cfg(model_type=model_type, args=args)
	return {
		"dataset_id": dataset_id,
		"artifacts_dir": args.artifacts_dir,
		"exp_name": exp_name,
		"run_name": run_name,
		"model_type": model_type,
		"device": args.device,
		"target_offset": args.target_offset,
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"ckpt_name": "model_best.pt",
		"freeze_features": bool(args.freeze_features),
		"freeze_layers": freeze_layers,
		"freeze_mode": args.freeze_mode,
		"model": model_cfg,
		"sparse_input": {
			"enabled": bool(args.sparse_input),
			"sample_p": float(args.sample_p),
			"sample_sigma": float(args.sample_sigma),
			"sample_seed": int(args.sample_seed),
			"append_mask_channel": True,
		},
		"probe": {
			"enabled": bool(args.with_probe),
			"record_level": int(args.probe_record_level),
			"hook_layers": ["enc.stage*.out", "dec.stage*.out", "skip.*", "head.out"],
		},
	}


def main() -> None:
	"""批量执行 full 数据集的 L2 训练+推理：三数据集 × 模型列表（默认开启 L2.5 特征冻结）。"""
	parser = argparse.ArgumentParser(description="Offline L2 runner for full datasets")
	parser.add_argument("--datasets", type=str, default="h5_full,nc_full,sst_full")
	parser.add_argument("--model-types", type=str, default="unet,unet_legacy,vit", help="逗号分隔模型列表")
	parser.add_argument("--artifacts-dir", type=str, default="artifacts")
	parser.add_argument("--exp-name", type=str, default="baseline_unet")
	parser.add_argument(
		"--split-exp-by-model",
		dest="split_exp_by_model",
		action="store_true",
		help="按模型拆分实验目录：exp_name_model_type（默认开启）",
	)
	parser.add_argument(
		"--no-split-exp-by-model",
		dest="split_exp_by_model",
		action="store_false",
		help="不同模型复用同一个 exp_name",
	)
	parser.set_defaults(split_exp_by_model=True)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--target-offset", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--train-steps", type=int, default=1000)
	parser.add_argument("--val-interval-steps", type=int, default=50)
	parser.add_argument("--log-interval-steps", type=int, default=100)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--base-channels", type=int, default=32)
	parser.add_argument("--convs-per-stage", type=int, default=2)
	parser.add_argument("--unet-legacy-depth", type=int, default=4)
	parser.add_argument("--vit-patch-size", type=int, default=16)
	parser.add_argument("--vit-embed-dim", type=int, default=64)
	parser.add_argument("--vit-depth", type=int, default=10)
	parser.add_argument("--vit-num-heads", type=int, default=8)
	parser.add_argument("--vit-mlp-ratio", type=float, default=4.0)
	parser.add_argument("--vit-dropout", type=float, default=0.1)
	parser.add_argument("--vit-attention-dropout", type=float, default=0.15)
	parser.add_argument("--vit-droppath", type=float, default=0.2)
	parser.add_argument("--with-infer", dest="with_infer", action="store_true", help="执行训练后推理（默认开启）")
	parser.add_argument("--skip-infer", dest="with_infer", action="store_false", help="仅训练，不执行推理")
	parser.set_defaults(with_infer=True)

	parser.add_argument("--freeze-features", dest="freeze_features", action="store_true", help="推理时冻结特征（默认开启）")
	parser.add_argument("--no-freeze-features", dest="freeze_features", action="store_false", help="关闭 L2.5 特征冻结")
	parser.set_defaults(freeze_features=True)
	parser.add_argument(
		"--freeze-layers",
		type=str,
		default="enc.stage1.out,enc.stage2.out,enc.stage3.out",
		help="逗号分隔的冻结层列表",
	)
	parser.add_argument("--freeze-mode", type=str, default="test", choices=["test"])

	parser.add_argument("--with-probe", dest="with_probe", action="store_true", help="额外启用 probe 产物")
	parser.add_argument("--no-probe", dest="with_probe", action="store_false", help="关闭额外 probe 产物")
	parser.set_defaults(with_probe=True)
	parser.add_argument("--probe-record-level", type=int, default=0)

	parser.add_argument("--sparse-input", dest="sparse_input", action="store_true", help="启用固定点位稀疏采样 + 1-NN 输入")
	parser.add_argument("--no-sparse-input", dest="sparse_input", action="store_false", help="关闭稀疏采样输入")
	parser.set_defaults(sparse_input=True)
	parser.add_argument("--sample-p", type=float, default=5e-3, help="采样率 p，例如 5e-3 表示 0.5%")
	parser.add_argument("--sample-sigma", type=float, default=0.0, help="采样点观测噪声标准差 σ")
	parser.add_argument("--sample-seed", type=int, default=123, help="固定点位采样随机种子")
	args = parser.parse_args()

	datasets = _parse_datasets(args.datasets)
	model_types = _parse_model_types(args.model_types)
	all_summaries: List[Dict[str, Any]] = []

	for model_type in model_types:
		exp_name = _build_exp_name(args.exp_name, model_type, bool(args.split_exp_by_model))
		for dataset_id in datasets:
			run_name = f"run_{dataset_id}_{model_type}_{now_tag()}"
			print(f"[L2] train start: dataset={dataset_id}, model={model_type}, exp={exp_name}, run={run_name}", flush=True)
			train_summary = run_l2_train(_build_train_cfg(args, dataset_id, run_name, model_type, exp_name))

			infer_summary: Dict[str, Any] | None = None
			if args.with_infer:
				print(f"[L2] infer start: dataset={dataset_id}, model={model_type}, exp={exp_name}, run={run_name}", flush=True)
				infer_summary = run_l2_infer(_build_infer_cfg(args, dataset_id, run_name, model_type, exp_name))

			all_summaries.append(
				{
					"dataset_id": dataset_id,
					"model_type": model_type,
					"exp_name": exp_name,
					"run_name": run_name,
					"train": train_summary,
					"infer": infer_summary,
					"freeze": (infer_summary or {}).get("freeze_outputs", {}),
				}
			)
			print(f"[L2] done: dataset={dataset_id}, model={model_type}, run={run_name}", flush=True)

	print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
