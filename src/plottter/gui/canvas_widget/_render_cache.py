"""Per-layer cached ``QPainterPath`` rendering geometry (canvas-performance §6).

The canvas previously redrew every layer by iterating its polylines point-by-point
in Python on each paint event. ``LayerPathCache`` instead builds one
``QPainterPath`` per layer (in **mm** coordinates) and reuses it across frames,
letting Qt do the per-point work in C++ and clip in the viewport transform.

See ``specs/canvas-performance.md`` §6 for the full design. This module covers
§6.1 (the per-layer data model) and §6.5 (the travel-line path cache); GUI
wiring lives in the canvas widget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtGui import QPainterPath

from plottter.models.layer import Layer

if TYPE_CHECKING:
    from plottter.models.project import Project


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


def build_travel_path(project: "Project") -> QPainterPath:
    """Build one ``QPainterPath`` (mm coords) of pen-up travel moves.

    A travel move is the straight hop between the end of one drawn polyline and
    the start of the next, across all **visible** layers in project order — the
    same traversal the legacy per-frame loop performed. Each hop is its own
    disconnected subpath (``moveTo`` the previous end, ``lineTo`` the next
    start). Empty polylines contribute no endpoint and are skipped.
    """
    path = QPainterPath()
    last_end: tuple[float, float] | None = None
    for layer in project.layers:
        if not layer.visible:
            continue
        for polyline in layer.paths:
            if not polyline:
                continue
            if last_end is not None:
                path.moveTo(last_end[0], last_end[1])
                path.lineTo(polyline[0][0], polyline[0][1])
            last_end = polyline[-1]
    return path


class TravelPathCache:
    """Single cached travel-line ``QPainterPath`` in mm coordinates (§6.5).

    Lazily (re)built on first ``get`` after construction or invalidation. The
    travel path spans every visible layer, so any layer mutation, removal,
    addition, reorder, or visibility toggle invalidates it.
    """

    def __init__(self) -> None:
        self._path: QPainterPath | None = None

    def get(self, project: "Project") -> QPainterPath:
        """Return the cached travel path, building it lazily if invalidated."""
        if self._path is None:
            self._path = build_travel_path(project)
        return self._path

    def invalidate(self) -> None:
        """Drop the cached travel path so the next ``get`` rebuilds."""
        self._path = None
