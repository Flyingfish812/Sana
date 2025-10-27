# schemas/train_schema.py  —— 替换 SCHEMA 常量
from __future__ import annotations
from typing import Any, Dict

SCHEMA: Dict[str, Any] = {
    "blocks": [
        {
            "title": "数据 Data",
            "key": "data",
            "collapsible": True,
            "fields": [
                # ✅ 用 snapshot_dir 作为主字段（问题5）
                {"name": "snapshot_dir", "type": "str", "required": True, "widget": "path",
                 "help": "已保存的 dataloader 快照目录（实验阶段推荐）"},
                # 兼容可选项：如从 run 目录复用 或 从零构造
                {"name": "from_run_dir", "type": "str", "required": False, "widget": "path",
                 "help": "（可选）从已完成的 run 目录重建 dataloader"},
                {"name": "builder", "type": "str", "required": False, "widget": "select",
                 "choices": ["none", "basic", "custom"], "default": "none",
                 "help": "（可选）从零构造 dataloader 的 builder 名"},
                {"name": "builder_args", "type": "kv", "value_type": "str", "required": False, "widget": "kv",
                 "help": "（可选）builder 的参数（key->scalar）"},

                # 常用 Loader 参数（保留）
                {"name": "batch_size", "type": "int", "required": True, "widget": "number",
                 "min": 1, "max": 65536, "default": 32},
                {"name": "num_workers", "type": "int", "required": False, "widget": "number",
                 "min": 0, "max": 128, "default": 4},
                {"name": "pin_memory", "type": "bool", "required": False, "widget": "checkbox", "default": True},
                {"name": "drop_last", "type": "bool", "required": False, "widget": "checkbox", "default": False},
            ],
        },

        {
            "title": "模型 Model（四模块拼装）",
            "key": "model",
            "collapsible": True,
            "fields": [
                # 四个模块，使用自定义 widget=module，kind 指出模块类型
                {"name": "encoder", "type": "dict", "required": True, "widget": "module", "kind": "encoder",
                 "help": "编码器：从 /model/registry 拉取可选项与参数"},
                {"name": "propagator", "type": "dict", "required": True, "widget": "module", "kind": "propagator"},
                {"name": "decoder", "type": "dict", "required": True, "widget": "module", "kind": "decoder"},
                {"name": "head", "type": "dict", "required": True, "widget": "module", "kind": "head"},
            ],
        },

        {
            "title": "训练 Train",
            "key": "train",
            "collapsible": True,
            "fields": [
                {"name": "epochs", "type": "int", "required": True, "widget": "number", "min": 1, "max": 200000, "default": 100},
                {"name": "precision", "type": "str", "required": False, "widget": "select",
                 "choices": ["32", "16-mixed", "bf16-mixed"], "default": "32"},
                {"name": "grad_clip_val", "type": "float", "required": False, "widget": "number",
                 "min": 0.0, "max": 1000.0, "default": 0.0},
                {"name": "accumulate_grad_batches", "type": "int", "required": False, "widget": "number",
                 "min": 1, "max": 1024, "default": 1},
                {"name": "seed", "type": "int", "required": False, "widget": "number",
                 "min": 0, "max": 2**31 - 1, "default": 42},

                {"name": "optimizer", "type": "str", "required": True, "widget": "select",
                 "choices": ["adam", "adamw", "sgd"], "default": "adamw"},
                {"name": "lr", "type": "float", "required": True, "widget": "number",
                 "min": 1e-7, "max": 1.0, "default": 3e-4},
                {"name": "weight_decay", "type": "float", "required": False, "widget": "number",
                 "min": 0.0, "max": 1.0, "default": 0.01, "show_if": {"train.optimizer": "adamw"}},
                {"name": "betas", "type": "list", "item_type": "float", "required": False, "widget": "list",
                 "default": [0.9, 0.999], "show_if": {"train.optimizer": "adam"}},
                {"name": "momentum", "type": "float", "required": False, "widget": "number",
                 "min": 0.0, "max": 0.999, "default": 0.9, "show_if": {"train.optimizer": "sgd"}},

                {"name": "scheduler", "type": "str", "required": False, "widget": "select",
                 "choices": ["none", "cosine", "plateau", "step"], "default": "none"},
                {"name": "cosine_tmax", "type": "int", "required": False, "widget": "number",
                 "min": 1, "max": 100000, "default": 100, "show_if": {"train.scheduler": "cosine"}},
                {"name": "plateau_patience", "type": "int", "required": False, "widget": "number",
                 "min": 1, "max": 1000, "default": 10, "show_if": {"train.scheduler": "plateau"}},
                {"name": "plateau_factor", "type": "float", "required": False, "widget": "number",
                 "min": 0.1, "max": 1.0, "default": 0.5, "show_if": {"train.scheduler": "plateau"}},
                {"name": "step_size", "type": "int", "required": False, "widget": "number",
                 "min": 1, "max": 100000, "default": 30, "show_if": {"train.scheduler": "step"}},
                {"name": "gamma", "type": "float", "required": False, "widget": "number",
                 "min": 0.1, "max": 1.0, "default": 0.1, "show_if": {"train.scheduler": "step"}},
            ],
        },
    ],
    "rules": [
        # 如果选择从零 builder，则需要提供 builder_args
        {"if": {"data.builder": "basic"}, "then_required": ["data.builder_args"]},
        {"if": {"data.builder": "custom"}, "then_required": ["data.builder_args"]},
    ],
}
