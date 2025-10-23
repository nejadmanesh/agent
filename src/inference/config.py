"""Configuration objects for inference models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a model that can be loaded by the inference engine."""

    name: str
    path: Path | None = None
    memory_bytes: int = 128 * 1024 * 1024
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resolve_path(self) -> Path | None:
        """Return the absolute path to the model artifact, if one was configured."""

        if self.path is None:
            return None
        return self.path.expanduser().resolve()
