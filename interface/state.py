"""Shared backend state for the interface layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from torch.utils.data import DataLoader


@dataclass
class DataArtifacts:
    """Cached artefacts produced by the data preparation pipeline."""

    config: Optional[Dict[str, Any]] = None
    dataset: Any = None
    dataloaders: Optional[Union[DataLoader, Dict[str, DataLoader]]] = None
    summary: Optional[Dict[str, Any]] = None


@dataclass
class TrainingArtifacts:
    """Cached artefacts produced by the training pipeline."""

    config: Optional[Dict[str, Any]] = None
    model: Any = None
    artefacts: Optional[Dict[str, Any]] = None


@dataclass
class BackendState:
    """Container for long-lived backend artefacts shared across requests."""

    data: DataArtifacts = field(default_factory=DataArtifacts)
    training: TrainingArtifacts = field(default_factory=TrainingArtifacts)


STATE = BackendState()
