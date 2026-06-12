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

import pytest

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QImage, QPainterPath

from plottter.gui.canvas_widget._render_cache import LayerPathCache, build_layer_path
from plottter.models.layer import Layer
from tests.canvas_render_ref import (
    make_fixture_project,
    pixel_diff_ratio,
    render_reference,
)

# Render target / view states for the equivalence tests (spec §11.1: two
# zoom/pan states — a fit-like view and a zoomed-in view pushing part of the
# scene off-screen so Qt's C++ clip replaces the deleted Python cull).
SIZE = (400, 400)
VIEW_STATES = [
    pytest.param(3.0, (50.0, 50.0), id="fit-view"),
    pytest.param(5.0, (-80.0, -60.0), id="zoomed-in"),
]


def _make_widget(project, zoom, pan, size, *, cache_enabled: bool = True):
    """Build a CanvasWidget over *project*, pinned to a fixed view state.

    Returns ``(controller, widget)`` — the controller is returned so callers
    that mutate paths through it (invalidation tests) keep it alive.
    """
    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.gui.project_controller import ProjectController

    controller = ProjectController(project)
    widget = CanvasWidget(controller)
    widget._render_cache_enabled = cache_enabled
    widget.resize(*size)
    widget._fitted = True  # block the fit-on-show refit
    widget._zoom = zoom
    widget._pan_offset = QPointF(*pan)
    return controller, widget


def _render(widget, size=SIZE) -> QImage:
    """Render *widget* to a fresh ARGB32 image (paintEvent overwrites all px)."""
    img = QImage(*size, QImage.Format.Format_ARGB32)
    img.fill(0)
    widget.render(img)
    return img


def _shift_paths(paths, dx: float, dy: float):
    """Return *paths* translated by (dx, dy) — for the drag-move oracle."""
    return [[(x + dx, y + dy) for x, y in pl] for pl in paths]


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


class TestPathCacheEquivalence:
    """The cached drawPath render must match the legacy oracle within §5.3
    tolerance (≤ 2% pixels) — covers the ordinary, dots, and opacity layers in
    the fixture at two zoom/pan states (spec §11.1)."""

    @pytest.mark.parametrize("zoom,pan", VIEW_STATES)
    def test_cached_render_matches_reference(self, qapp, zoom, pan):
        project = make_fixture_project()
        ref = render_reference(project, zoom, pan, SIZE)
        _controller, widget = _make_widget(project, zoom, pan, SIZE)
        real = _render(widget)
        ratio = pixel_diff_ratio(real, ref)
        assert ratio <= 0.02, f"cached diff ratio {ratio:.4f} exceeds 0.02"

    @pytest.mark.parametrize("zoom,pan", VIEW_STATES)
    def test_uncached_render_matches_reference(self, qapp, zoom, pan):
        # _render_cache_enabled=False builds the path fresh each frame (spec §9)
        # via the same drawing code, so it must match the oracle too.
        project = make_fixture_project()
        ref = render_reference(project, zoom, pan, SIZE)
        _controller, widget = _make_widget(
            project, zoom, pan, SIZE, cache_enabled=False
        )
        real = _render(widget)
        ratio = pixel_diff_ratio(real, ref)
        assert ratio <= 0.02, f"uncached diff ratio {ratio:.4f} exceeds 0.02"

    def test_cached_and_uncached_are_identical(self, qapp):
        # Same drawing code both sides → pixel-identical output (spec §11.1).
        project = make_fixture_project()
        _c1, cached = _make_widget(project, 3.0, (50.0, 50.0), SIZE)
        _c2, uncached = _make_widget(
            make_fixture_project(), 3.0, (50.0, 50.0), SIZE, cache_enabled=False
        )
        assert pixel_diff_ratio(_render(cached), _render(uncached)) == 0.0


class TestPathCacheInvalidation:
    """``paths_changed`` drops the affected entry and the re-render reflects the
    new geometry (spec §11.2)."""

    def test_paths_changed_invalidates_and_rerenders(self, qapp):
        project = make_fixture_project()
        controller, widget = _make_widget(project, 3.0, (50.0, 50.0), SIZE)
        layer = project.layers[0]

        before = _render(widget)
        # First render populated the cache for every visible layer.
        assert layer.id in widget._path_cache._entries

        # Mutate the layer's geometry through the controller → paths_changed.
        new_paths = [[(10.0, 10.0), (90.0, 90.0)], [(90.0, 10.0), (10.0, 90.0)]]
        controller._raw_set_layer_paths(layer.id, new_paths)
        # The handler dropped the stale entry...
        assert layer.id not in widget._path_cache._entries

        after = _render(widget)
        # ...and the re-render both differs from the old frame and matches a
        # fresh oracle render of the now-updated project.
        assert pixel_diff_ratio(before, after) > 0.0
        ref = render_reference(controller.current_project, 3.0, (50.0, 50.0), SIZE)
        assert pixel_diff_ratio(after, ref) <= 0.02

    def test_paths_changed_invalidates_travel_cache(self, qapp):
        project = make_fixture_project()
        controller, widget = _make_widget(project, 3.0, (50.0, 50.0), SIZE)
        widget._show_travel = True
        _render(widget)
        assert widget._travel_cache._path is not None

        controller._raw_set_layer_paths(project.layers[0].id, [[(5.0, 5.0), (9.0, 9.0)]])
        assert widget._travel_cache._path is None


class TestDragMoveEquivalence:
    """Drag-to-move live preview offsets the active layer via
    ``painter.translate`` — the output must match the oracle rendered with that
    layer's paths shifted by the same offset (spec §11.1)."""

    def test_drag_move_offset_matches_shifted_reference(self, qapp):
        dx, dy = 12.0, -7.0
        project = make_fixture_project()
        active = project.layers[0]

        controller, widget = _make_widget(project, 3.0, (50.0, 50.0), SIZE)
        controller.set_active_layer(active.id)
        widget._drag_move_active = True
        widget._drag_move_start_mm = (0.0, 0.0)
        widget._drag_move_offset_mm = (dx, dy)
        real = _render(widget)

        # Oracle: the same scene with only the active layer's paths translated.
        shifted = make_fixture_project()
        shifted.layers[0].paths = _shift_paths(shifted.layers[0].paths, dx, dy)
        ref = render_reference(shifted, 3.0, (50.0, 50.0), SIZE)

        ratio = pixel_diff_ratio(real, ref)
        assert ratio <= 0.02, f"drag-move diff ratio {ratio:.4f} exceeds 0.02"
