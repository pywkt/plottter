"""Layer dataclass representing a single drawing layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from plottter.models.path import Polyline


@dataclass
class Layer:
    name: str
    color: str = "#000000"
    paths: list[Polyline] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    generator_info: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def path_count(self) -> int:
        """Return the number of paths in this layer."""
        return len(self.paths)

    def total_point_count(self) -> int:
        """Return the total number of points across all paths."""
        return sum(len(p) for p in self.paths)

    def clear_paths(self) -> None:
        """Remove all paths from this layer."""
        self.paths.clear()

    def add_paths(self, paths: list[Polyline]) -> None:
        """Append paths to this layer."""
        self.paths.extend(paths)
