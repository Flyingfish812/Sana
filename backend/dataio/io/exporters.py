# backend/dataio/io/exporters.py
from __future__ import annotations
from typing import Dict, Any
from ..cache.torch_cache import CacheManager
from ..cache.snapshot import DataloaderSnapshotWriter
from pathlib import Path

def dump_prep_outputs(
    *,
    dataset,
    dataloader,
    summary: Dict[str, Any],
    output_cfg: Dict[str, Any],
):
    """
    统一导出：
      - array5d.npy：保存 Reader 统一后的 5D 原始数组（可选）
      - DataLoader 快照：按近似 1GB 分片 + 索引，一行代码可恢复
    output_cfg 字段：
      out_dir: str
      save_array5d: bool
      snapshot:            # 快照写入配置
        enable: true
        part_size_gb: 1.0
        gzip: false        # 若为 true，则文件为 .pt.gz（更小体积，CPU 稍增）
      overwrite: bool
    """
    out_dir = output_cfg.get("out_dir", "./prep_out")
    overwrite = bool(output_cfg.get("overwrite", True))

    # array5d.npy（原始标准 5D）
    if output_cfg.get("save_array5d", False):
        cm = CacheManager(out_dir, overwrite=overwrite)
        cm.save_array5d(dataset.array5d, filename="array5d.npy")
        cm.save_meta_summary(dataset.meta, summary)

    # DataLoader 快照
    snap_cfg = output_cfg.get("snapshot", {"enable": True})
    if snap_cfg.get("enable", True):
        part_gb = float(snap_cfg.get("part_size_gb", 1.0))
        part_bytes = max(64 << 20, int(part_gb * (1 << 30)))
        writer = DataloaderSnapshotWriter(out_dir, target_part_bytes=part_bytes,
                                          overwrite=overwrite, gzip_compress=bool(snap_cfg.get("gzip", False)))
        meta_json = dataset.meta.to_json()
        for batch in dataloader:
            writer.add_batch(batch, meta_json=meta_json)
        idx = writer.close()
        # 写一份 summary 供人眼查看
        human = {
            "mode": "snapshot",
            "parts": len(idx.parts),
            "total_samples": idx.total_samples,
            "approx_total_gb": round(idx.total_nbytes_uncompressed / (1<<30), 3),
            "keys": idx.keys,
            "compressed": idx.compressed,
        }
        import json, os
        with open((Path(out_dir) / "summary.json"), "w", encoding="utf-8") as f:
            json.dump({**summary, **human}, f, ensure_ascii=False, indent=2)
