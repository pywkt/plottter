"""Project dataclass — top-level container for canvas, layers, and settings."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.path import Polyline


@dataclass
class Project:
    name: str
    canvas: Canvas
    layers: list[Layer] = field(default_factory=list)
    registration_marks: bool = True
    reg_mark_style: str = "corners"
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_layer(self, layer: Layer) -> None:
        """Append a layer to the project."""
        self.layers.append(layer)

    def remove_layer(self, layer_id: str) -> None:
        """Remove a layer by id. Silently ignores unknown ids."""
        self.layers = [l for l in self.layers if l.id != layer_id]

    def reorder_layer(self, layer_id: str, new_index: int) -> None:
        """Move a layer to a new position in the stack."""
        idx = self._index_of(layer_id)
        if idx is None:
            return
        layer = self.layers.pop(idx)
        clamped = max(0, min(new_index, len(self.layers)))
        self.layers.insert(clamped, layer)

    def get_layer(self, layer_id: str) -> Layer | None:
        """Return the layer with the given id, or None."""
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def duplicate_layer(self, layer_id: str) -> Layer:
        """Deep-copy a layer with a new UUID and return it (not yet added to project)."""
        source = self.get_layer(layer_id)
        if source is None:
            raise ValueError(f"Layer {layer_id!r} not found")
        new_layer = copy.deepcopy(source)
        new_layer.id = str(uuid.uuid4())
        new_layer.name = f"{source.name} copy"
        return new_layer

    def merge_layers(self, layer_ids: list[str]) -> Layer:
        """Combine paths from multiple layers into a new layer (not added to project)."""
        combined_paths: list[Polyline] = []
        first_color = "#000000"
        found_first = False
        for layer in self.layers:
            if layer.id in layer_ids:
                combined_paths.extend(copy.deepcopy(layer.paths))
                if not found_first:
                    first_color = layer.color
                    found_first = True
        merged = Layer(name="Merged Layer", color=first_color, paths=combined_paths)
        return merged

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_layer(self) -> Layer | None:
        """Return the first unlocked visible layer, else first layer, else None."""
        for layer in self.layers:
            if layer.visible and not layer.locked:
                return layer
        if self.layers:
            return self.layers[0]
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_of(self, layer_id: str) -> int | None:
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return i
        return None
