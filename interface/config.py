"""Utilities for loading configuration payloads provided by the frontend."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, root_validator, validator


class ConfigPayload(BaseModel):
    """Common configuration payload used by multiple endpoints."""

    config: Optional[Dict[str, Any]] = None
    config_path: Optional[str] = None
    config_yaml: Optional[str] = None

    @root_validator(pre=False)
    def _ensure_source(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        provided = [
            values.get("config") is not None,
            bool(values.get("config_path")),
            bool(values.get("config_yaml")),
        ]
        if not any(provided):
            raise ValueError("one of 'config', 'config_path' or 'config_yaml' must be provided")
        return values

    @validator("config_path")
    def _expand_config_path(cls, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        return str(Path(path).expanduser().resolve())

    def load(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""

        if self.config is not None:
            return self.config

        if self.config_yaml:
            return yaml.safe_load(self.config_yaml)

        if self.config_path:
            path = Path(self.config_path)
            text = path.read_text(encoding="utf-8")
            return yaml.safe_load(text)

        raise RuntimeError("no configuration source resolved")

    def describe_source(self) -> str:
        if self.config_path:
            return self.config_path
        if self.config_yaml:
            return "<yaml-inline>"
        if self.config is not None:
            return "<dict>"
        return "<unknown>"
