# schemas/train_schema.py
from __future__ import annotations

from typing import Any, Dict, List

# 每个 block 描述一个大区；fields 内每个项描述一个字段
SCHEMA: Dict[str, Any] = {
    "blocks": [
        {
            "title": "数据 Data",
            "key": "data",                         # 顶层 key
            "fields": [
                {"name": "dataset", "type": "str", "required": True, "widget": "select",
                 "choices": ["nc_full", "mat_sparse", "rdb_static"], "help": "选择数据集标识"},
                {"name": "root", "type": "path", "required": True, "widget": "path",
                 "help": "远端数据根目录（绝对或相对项目根）"},
                {"name": "batch_size", "type": "int", "required": True, "widget": "number",
                 "min": 1, "max": 4096, "help": "训练 batch 大小"},
                {"name": "num_workers", "type": "int", "required": False, "widget": "number",
                 "min": 0, "max": 64, "default": 4, "help": "DataLoader 线程数"},
            ],
        },
        {
            "title": "模型 Model",
            "key": "model",
            "fields": [
                {"name": "arch", "type": "str", "required": True, "widget": "select",
                 "choices": ["unet", "vit", "swin"], "help": "模型架构"},
                {"name": "in_channels", "type": "int", "required": True, "widget": "number",
                 "min": 1, "max": 32},
                {"name": "out_channels", "type": "int", "required": True, "widget": "number",
                 "min": 1, "max": 32},
            ],
        },
        {
            "title": "训练 Train",
            "key": "train",
            "fields": [
                {"name": "epochs", "type": "int", "required": True, "widget": "number",
                 "min": 1, "max": 10000},
                {"name": "precision", "type": "str", "required": False, "widget": "select",
                 "choices": ["32", "16-mixed", "bf16-mixed"], "default": "32"},
                {"name": "optimizer", "type": "str", "required": True, "widget": "select",
                 "choices": ["adam", "adamw", "sgd"], "default": "adamw"},
                {"name": "lr", "type": "float", "required": True, "widget": "number",
                 "min": 1e-6, "max": 1.0, "step": 1e-6, "default": 3e-4},
            ],
        },
    ],
    # 互斥/依赖示例（选做）
    "rules": [
        # {"if": {"model.arch": "vit"}, "then_required": ["model.patch_size"]},
        # {"mutex": ["train.precision", "train.amp_level"]},
    ],
}
