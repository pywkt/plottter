"""Per-layer cached ``QPainterPath`` rendering geometry (canvas-performance §6).

The canvas previously redrew every layer by iterating its polylines point-by-point
in Python on each paint event. ``LayerPathCache`` instead builds one
``QPainterPath`` per layer (in **mm** coordinates) and reuses it across frames,
letting Qt do the per-point work in C++ and clip in the viewport transform.

See ``specs/canvas-performance.md`` §6 for the full design. This module covers
§6.1 (data model) only — GUI wiring lives in the canvas widget.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QPainterPath

from plottter.models.layer import Layer


@dataclass
class _Entry:
    """A cached path plus the layer state it was built from (for staleness)."""

    path: QPainterPath
    point_count: int


def build_layer_path(layer: Layer) -> QPainterPath:
    """Build one ``QPainterPath`` (mm coords) from a layer's polylines.

    Each polyline with at least two points becomes a disconnected subpath:
    ``moveTo`` the first point, then ``lineTo`` each subsequent point.
    Polylines with fewer than two points are skipped (nothing to stroke).
    """
    path = QPainterPath()
    for polyline in layer.paths:
        if len(polyline) < 2:
            continue
        x0, y0 = polyline[0]
        path.moveTo(x0, y0)
        for x, y in polyline[1:]:
            path.lineTo(x, y)
    return path


class LayerPathCache:
    """Cached ``QPainterPath`` per layer, in mm coordinates.

    ``get`` lazily builds an entry the first time a layer is drawn (or after it
    has been invalidated / its point count has changed) and returns the same
    cached object on subsequent calls. ``invalidate`` / ``invalidate_all`` drop
    entries so the next ``get`` rebuilds.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def get(self, layer: Layer) -> QPainterPath:
        """Return the cached path for ``layer``, building it lazily if stale."""
        entry = self._entries.get(layer.id)
        point_count = layer.total_point_count()
        if entry is None or entry.point_count != point_count:
            entry = _Entry(path=build_layer_path(layer), point_count=point_count)
            self._entries[layer.id] = entry
        return entry.path

    def invalidate(self, layer_id: str) -> None:
        """Drop the cached entry for ``layer_id`` (no-op if absent)."""
        self._entries.pop(layer_id, None)

    def invalidate_all(self) -> None:
        """Drop all cached entries."""
        self._entries.clear()
