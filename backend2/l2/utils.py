from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import hashlib
import json
import subprocess


def now_tag() -> str:
    """返回适合目录/文件命名的 UTC 时间标签。"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    """返回 UTC ISO8601 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    """将字典写入 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    """读取 JSON 对象并校验根节点为 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected json object at {path}")
    return data


def dump_jsonl_line(path: Path, payload: Dict[str, Any]) -> None:
    """向 JSONL 文件追加一条记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _try_git_commit(cwd: Path) -> str:
    """尝试读取当前仓库 HEAD commit，失败返回空字符串。"""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _tree_fingerprint(cwd: Path) -> str:
    """生成轻量代码指纹（非严格内容哈希）。"""
    h = hashlib.sha256()
    h.update(str(cwd).encode("utf-8"))
    h.update(now_iso().encode("utf-8"))
    return h.hexdigest()


def build_code_version(cwd: Path) -> Dict[str, Any]:
    """构建代码版本信息：优先 git commit，回退到临时指纹。"""
    commit = _try_git_commit(cwd)
    return {
        "created_at": now_iso(),
        "git_commit": commit or None,
        "code_hash": _tree_fingerprint(cwd) if not commit else None,
    }
