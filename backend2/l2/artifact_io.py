from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .utils import now_tag


class ArtifactManager:
    """集中管理 L2 训练/推理产物路径。"""
    def __init__(self, artifacts_dir: str, dataset_id: str, exp_name: str, run_name: Optional[str] = None):
        """初始化数据集、实验与运行目录上下文。"""
        self.artifacts_root = Path(artifacts_dir)
        self.dataset_id = str(dataset_id)
        self.exp_name = str(exp_name)
        self.run_name = str(run_name) if run_name else f"run_{now_tag()}"

    @property
    def l1_dir(self) -> Path:
        """返回对应数据集的 L1 根目录。"""
        return self.artifacts_root / self.dataset_id / "L1"

    @property
    def l2_exp_dir(self) -> Path:
        """返回当前实验在 L2 下的目录。"""
        return self.artifacts_root / self.dataset_id / "L2" / self.exp_name

    @property
    def run_dir(self) -> Path:
        """返回当前运行目录。"""
        return self.l2_exp_dir / self.run_name

    @property
    def ckpt_dir(self) -> Path:
        """返回模型检查点目录。"""
        return self.run_dir / "ckpt"

    @property
    def logs_dir(self) -> Path:
        """返回训练日志目录。"""
        return self.run_dir / "logs"

    @property
    def infer_dir(self) -> Path:
        """返回推理产物目录。"""
        return self.run_dir / "infer"

    @property
    def probe_dir(self) -> Path:
        """返回 probe 调试产物目录。"""
        return self.run_dir / "probe"

    @property
    def freeze_dir(self) -> Path:
        """返回 L2.5 冻结特征目录。"""
        return self.run_dir / "freeze"

    @property
    def freeze_layers_dir(self) -> Path:
        """返回 L2.5 分层特征目录。"""
        return self.freeze_dir / "layers"

    @property
    def freeze_manifest_json(self) -> Path:
        """返回 L2.5 冻结清单路径。"""
        return self.freeze_dir / "manifest.json"

    @property
    def train_config_json(self) -> Path:
        """返回训练配置文件路径。"""
        return self.run_dir / "train_config.json"

    @property
    def infer_config_json(self) -> Path:
        """返回推理配置文件路径。"""
        return self.run_dir / "infer_config.json"

    @property
    def code_version_json(self) -> Path:
        """返回代码版本记录文件路径。"""
        return self.run_dir / "code_version.json"

    def ensure_run_dirs(self) -> None:
        """创建当前运行所需的全部目录。"""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.infer_dir.mkdir(parents=True, exist_ok=True)
        self.probe_dir.mkdir(parents=True, exist_ok=True)
        self.freeze_layers_dir.mkdir(parents=True, exist_ok=True)

    def l1_manifest_path(self) -> Path:
        """返回 L1 manifest 路径。"""
        return self.l1_dir / "manifest.json"

    def l1_stats_path(self) -> Path:
        """返回 L1 训练统计文件路径。"""
        return self.l1_dir / "stats_train.json"

    def l1_array_path(self) -> Path:
        """返回 L1 归一化 5D 数组路径。"""
        return self.l1_dir / "array5d_norm.npy"

    def l1_split_path(self, split_name: str) -> Path:
        """返回 L1 指定切分索引文件路径。"""
        return self.l1_dir / "splits" / f"{split_name}.npy"

    def ckpt_path(self, name: str = "model_last.pt") -> Path:
        """返回指定名称检查点的完整路径。"""
        return self.ckpt_dir / name

    def summary(self) -> Dict[str, str]:
        """返回运行身份信息摘要。"""
        return {
            "dataset_id": self.dataset_id,
            "exp_name": self.exp_name,
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
        }
