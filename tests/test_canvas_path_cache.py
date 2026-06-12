"""Tests for ``LayerPathCache`` (canvas-performance §6.1).

The cache builds one mm-coordinate ``QPainterPath`` per layer and reuses it
across frames. These tests cover the data model only (no GUI wiring):

- element count matches the input point count
- disconnected polylines become separate subpaths
- polylines with fewer than two points are skipped
- ``get`` twice returns the same cached object
- ``invalidate`` / ``invalidate_all`` force a rebuild
"""

from __future__ import annotations

from PyQt6.QtGui import QPainterPath

from plottter.gui.canvas_widget._render_cache import LayerPathCache, build_layer_path
from plottter.models.layer import Layer


def _subpath_count(path: QPainterPath) -> int:
    """Number of ``MoveToElement`` markers — i.e. disconnected subpaths."""
    move = QPainterPath.ElementType.MoveToElement
    return sum(1 for i in range(path.elementCount()) if path.elementAt(i).type == move)


class TestBuildLayerPath:
    def test_element_count_matches_point_count(self, qapp):
        layer = Layer(name="a", paths=[[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]])
        path = build_layer_path(layer)
        # 1 moveTo + 2 lineTo == 3 points.
        assert path.elementCount() == 3

    def test_disconnected_polylines_are_subpaths(self, qapp):
        layer = Layer(
            name="a",
            paths=[[(0.0, 0.0), (1.0, 0.0)], [(5.0, 5.0), (6.0, 5.0), (7.0, 5.0)]],
        )
        path = build_layer_path(layer)
        assert _subpath_count(path) == 2
        # 2 + 3 points across the two subpaths.
        assert path.elementCount() == 5

    def test_short_polylines_skipped(self, qapp):
        layer = Layer(
            name="a",
            paths=[
                [(0.0, 0.0)],            # single point — skipped
                [],                       # empty — skipped
                [(1.0, 1.0), (2.0, 2.0)],  # valid
            ],
        )
        path = build_layer_path(layer)
        assert _subpath_count(path) == 1
        assert path.elementCount() == 2

    def test_coordinates_stay_in_mm(self, qapp):
        layer = Layer(name="a", paths=[[(3.5, 7.25), (9.0, 1.0)]])
        path = build_layer_path(layer)
        first = path.elementAt(0)
        assert (first.x, first.y) == (3.5, 7.25)

    def test_empty_layer_yields_empty_path(self, qapp):
        layer = Layer(name="a", paths=[])
        path = build_layer_path(layer)
        assert path.elementCount() == 0


class TestLayerPathCache:
    def test_get_twice_returns_same_object(self, qapp):
        cache = LayerPathCache()
        layer = Layer(name="a", paths=[[(0.0, 0.0), (1.0, 1.0)]])
        first = cache.get(layer)
        second = cache.get(layer)
        assert first is second

    def test_invalidate_forces_rebuild(self, qapp):
        cache = LayerPathCache()
        layer = Layer(name="a", paths=[[(0.0, 0.0), (1.0, 1.0)]])
        first = cache.get(layer)
        cache.invalidate(layer.id)
        second = cache.get(layer)
        assert first is not second

    def test_invalidate_unknown_id_is_noop(self, qapp):
        cache = LayerPathCache()
        layer = Layer(name="a", paths=[[(0.0, 0.0), (1.0, 1.0)]])
        cache.get(layer)
        cache.invalidate("does-not-exist")
        # Existing entry untouched.
        assert cache.get(layer) is cache.get(layer)

    def test_invalidate_all_forces_rebuild(self, qapp):
        cache = LayerPathCache()
        layer_a = Layer(name="a", paths=[[(0.0, 0.0), (1.0, 1.0)]])
        layer_b = Layer(name="b", paths=[[(2.0, 2.0), (3.0, 3.0)]])
        a1, b1 = cache.get(layer_a), cache.get(layer_b)
        cache.invalidate_all()
        assert cache.get(layer_a) is not a1
        assert cache.get(layer_b) is not b1

    def test_point_count_change_rebuilds(self, qapp):
        cache = LayerPathCache()
        layer = Layer(name="a", paths=[[(0.0, 0.0), (1.0, 1.0)]])
        first = cache.get(layer)
        layer.paths[0].append((2.0, 2.0))
        second = cache.get(layer)
        assert first is not second
        assert second.elementCount() == 3
