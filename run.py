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


def _build_train_cfg(args: argparse.Namespace, dataset_id: str, run_name: str) -> Dict[str, Any]:
	"""构建单数据集 L2 训练配置。"""
	return {
		"dataset_id": dataset_id,
		"artifacts_dir": args.artifacts_dir,
		"exp_name": args.exp_name,
		"run_name": run_name,
		"device": args.device,
		"seed": args.seed,
		"split_tag": args.split_tag,
		"target_offset": args.target_offset,
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"epochs": args.epochs,
		"lr": args.lr,
		"model": {
			"base_channels": args.base_channels,
			"convs_per_stage": args.convs_per_stage,
		},
	}


def _build_infer_cfg(args: argparse.Namespace, dataset_id: str, run_name: str) -> Dict[str, Any]:
	"""构建单数据集 L2 推理配置。"""
	return {
		"dataset_id": dataset_id,
		"artifacts_dir": args.artifacts_dir,
		"exp_name": args.exp_name,
		"run_name": run_name,
		"device": args.device,
		"split_tag": args.split_tag,
		"target_offset": args.target_offset,
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"ckpt_name": "model_best.pt",
		"model": {
			"base_channels": args.base_channels,
			"convs_per_stage": args.convs_per_stage,
		},
		"probe": {
			"enabled": bool(args.with_probe),
			"record_level": int(args.probe_record_level),
			"hook_layers": ["enc.stage*.out", "dec.stage*.out", "skip.*", "head.out"],
		},
	}


def main() -> None:
	"""批量执行三个 full 数据集的 L2 训练（可选推理）。"""
	parser = argparse.ArgumentParser(description="Offline L2 runner for full datasets")
	parser.add_argument("--datasets", type=str, default="h5_full,nc_full,sst_full")
	parser.add_argument("--artifacts-dir", type=str, default="artifacts")
	parser.add_argument("--exp-name", type=str, default="baseline_unet")
	parser.add_argument("--split-tag", type=str, default="temporal_frame_seed123")
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--target-offset", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--base-channels", type=int, default=32)
	parser.add_argument("--convs-per-stage", type=int, default=2)
	parser.add_argument("--with-infer", action="store_true")
	parser.add_argument("--with-probe", action="store_true")
	parser.add_argument("--probe-record-level", type=int, default=0)
	args = parser.parse_args()

	datasets = _parse_datasets(args.datasets)
	all_summaries: List[Dict[str, Any]] = []

	for dataset_id in datasets:
		run_name = f"run_{dataset_id}_{now_tag()}"
		print(f"[L2] train start: dataset={dataset_id}, run={run_name}", flush=True)
		train_summary = run_l2_train(_build_train_cfg(args, dataset_id, run_name))

		infer_summary: Dict[str, Any] | None = None
		if args.with_infer:
			print(f"[L2] infer start: dataset={dataset_id}, run={run_name}", flush=True)
			infer_summary = run_l2_infer(_build_infer_cfg(args, dataset_id, run_name))

		all_summaries.append(
			{
				"dataset_id": dataset_id,
				"run_name": run_name,
				"train": train_summary,
				"infer": infer_summary,
			}
		)
		print(f"[L2] done: dataset={dataset_id}, run={run_name}", flush=True)

	print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
