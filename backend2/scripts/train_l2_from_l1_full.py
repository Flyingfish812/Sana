from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend2.l1 import build_dataloaders_from_l1
from backend2.l2.train import run_l2_train
from backend2.l2.utils import now_tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Train L2 directly from frozen L1 artifacts")
    parser.add_argument("--dataset-id", type=str, required=True, choices=["h5_full", "nc_full", "sst_full"])
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--exp-name", type=str, default="baseline_unet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    l1_dir = Path(args.artifacts_dir) / args.dataset_id / "L1"
    pack = build_dataloaders_from_l1(
        l1_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_offset=1,
        shuffle_train=True,
    )
    train_loader = pack["loaders"]["train"]
    first_batch = next(iter(train_loader))

    run_name = f"run_{args.dataset_id}_{now_tag()}"
    train_cfg = {
        "dataset_id": args.dataset_id,
        "artifacts_dir": args.artifacts_dir,
        "exp_name": args.exp_name,
        "run_name": run_name,
        "device": "auto",
        "seed": 123,
        "target_offset": 1,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": 1e-3,
        "model": {
            "base_channels": 32,
            "convs_per_stage": 2,
        },
    }
    summary = run_l2_train(train_cfg)

    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "l1_dir": str(l1_dir),
                "loader_check": {
                    "train_pairs": len(pack["pairs"]["train"]),
                    "x": list(first_batch["x"].shape),
                    "y": list(first_batch["y"].shape),
                },
                "train_summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
