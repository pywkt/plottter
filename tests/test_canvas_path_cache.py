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

from plottter.gui.canvas_widget._render_cache import (
    LayerPathCache,
    build_jittered_layer_chunks,
    build_jittered_layer_path,
    build_layer_chunks,
    build_layer_path,
)
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


def _path_coords(path: QPainterPath) -> list[tuple[float, float]]:
    """All element (x, y) coordinates of *path*, in order."""
    return [
        (path.elementAt(i).x, path.elementAt(i).y)
        for i in range(path.elementCount())
    ]


def _chunks_coords(chunks) -> list[tuple[float, float]]:
    """All coordinates across a chunk list, in deterministic chunk order."""
    return [pt for c in chunks for pt in _path_coords(c.path)]


def _chunks_element_count(chunks) -> int:
    return sum(c.path.elementCount() for c in chunks)


def _chunks_subpath_count(chunks) -> int:
    return sum(_subpath_count(c.path) for c in chunks)


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
        assert _chunks_element_count(second) == 3


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


class TestChunking:
    """Spatial chunking of cached geometry (wide-pen stroker cliff fix).

    Qt's stroker for pens wider than 1 device px is superlinear in per-call
    path size, so zoomed-in views (default pen > ~3.3×, Pointillist dot layers
    > ~1.7×) must draw many small culled chunks instead of one huge path.
    """

    def test_chunks_preserve_geometry(self, qapp):
        # Chunked coords are a permutation of the single-path coords with the
        # same totals — no point is lost, duplicated, or split across chunks.
        project = make_fixture_project()
        for layer in project.layers:
            chunks = build_layer_chunks(layer)
            single = build_layer_path(layer)
            assert _chunks_element_count(chunks) == single.elementCount()
            assert _chunks_subpath_count(chunks) == _subpath_count(single)
            assert sorted(_chunks_coords(chunks)) == sorted(_path_coords(single))

    def test_jittered_chunks_match_jittered_path_geometry(self, qapp):
        # The chunked jitter variant uses the same seed/order as the unchunked
        # builder, so the displaced coordinates are identical.
        layer = Layer(name="a", id="layer-fixed", paths=_JITTER_PATHS)
        chunks = build_jittered_layer_chunks(layer, 0.15)
        single = build_jittered_layer_path(layer, 0.15)
        assert sorted(_chunks_coords(chunks)) == sorted(_path_coords(single))

    def test_chunk_bboxes_cover_their_points(self, qapp):
        project = make_fixture_project()
        for layer in project.layers:
            for chunk in build_layer_chunks(layer):
                x0, y0, x1, y1 = chunk.bbox
                for x, y in _path_coords(chunk.path):
                    assert x0 - 1e-9 <= x <= x1 + 1e-9
                    assert y0 - 1e-9 <= y <= y1 + 1e-9

    def test_dot_layer_zero_height_bbox_not_culled(self, qapp):
        # Pointillist dots are 0.01 mm horizontal segments → zero-height bboxes.
        # QRectF.intersects would cull them (empty-rect semantics); the manual
        # interval test must keep them. Render a dots-only project and assert
        # the dots actually appear.
        from plottter.models import Canvas, Project

        dots = [[(x, 50.0), (x + 0.01, 50.0)] for x in range(10, 90, 5)]
        layer = Layer(
            name="dots",
            color="#000000",
            paths=dots,
            generator_info={"dot_diameter_mm": 2.0},
        )
        project = Project(
            name="p",
            canvas=Canvas(width_mm=100.0, height_mm=100.0),
            registration_marks=False,
        )
        project.add_layer(layer)
        _c, widget = _make_widget(project, 3.0, (50.0, 50.0), SIZE)
        img = _render(widget)
        ref = render_reference(project, 3.0, (50.0, 50.0), SIZE)
        assert pixel_diff_ratio(img, ref) <= 0.02

    def test_wide_pen_zoomed_render_matches_reference(self, qapp):
        # The regression scenario: zoomed in far enough that the default pen
        # exceeds 1 device px (zoom 9 → 2.7 px) — chunked culled drawing must
        # still match the legacy oracle.
        project = make_fixture_project()
        zoom, pan = 9.0, (-300.0, -250.0)
        ref = render_reference(project, zoom, pan, SIZE)
        _c, widget = _make_widget(project, zoom, pan, SIZE)
        real = _render(widget)
        ratio = pixel_diff_ratio(real, ref)
        assert ratio <= 0.02, f"wide-pen zoomed diff {ratio:.4f} exceeds 0.02"


# Fixed-id layers so the crc32(layer.id) seed is reproducible across tests.
_JITTER_PATHS = [
    [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0), (20.0, 20.0)],
    [(20.0, 20.0), (80.0, 80.0)],
]


class TestBakedJitter:
    """Baked per-layer jitter (spec §6.4): deterministic, mm-space (zoom-
    independent), distinct across layers, and stable across frames."""

    def test_build_is_deterministic(self, qapp):
        # Two builds of the same layer at the same intensity are identical —
        # the displacement is seeded only by the layer id, never re-randomised.
        layer = Layer(name="a", id="layer-fixed", paths=_JITTER_PATHS)
        first = build_jittered_layer_path(layer, 0.15)
        second = build_jittered_layer_path(layer, 0.15)
        assert _path_coords(first) == _path_coords(second)

    def test_displacement_actually_applied(self, qapp):
        # The jittered path differs from the un-jittered one (jitter is real)
        # but keeps the same element/subpath structure.
        layer = Layer(name="a", id="layer-fixed", paths=_JITTER_PATHS)
        plain = build_layer_path(layer)
        jittered = build_jittered_layer_path(layer, 0.15)
        assert jittered.elementCount() == plain.elementCount()
        assert _subpath_count(jittered) == _subpath_count(plain)
        assert _path_coords(jittered) != _path_coords(plain)

    def test_differs_between_layers(self, qapp):
        # Same geometry, different ids → different displacement (crc32 seed).
        a = Layer(name="a", id="layer-a", paths=_JITTER_PATHS)
        b = Layer(name="b", id="layer-b", paths=_JITTER_PATHS)
        coords_a = _path_coords(build_jittered_layer_path(a, 0.15))
        coords_b = _path_coords(build_jittered_layer_path(b, 0.15))
        assert coords_a != coords_b

    def test_displacement_is_mm_space_zoom_independent(self, qapp):
        # The baked path lives in mm and is built before the world transform,
        # so changing the widget's zoom must not change the jittered mm coords.
        project = make_fixture_project()
        layer = project.layers[0]
        _controller, widget = _make_widget(project, 2.0, (50.0, 50.0), SIZE)
        widget._jitter_enabled = True

        coords_low = _chunks_coords(widget._layer_chunks(layer))
        widget._zoom = 9.0
        coords_high = _chunks_coords(widget._layer_chunks(layer))
        assert coords_low == coords_high
        # And the displacement is genuinely applied (differs from un-jittered).
        assert coords_low != _chunks_coords(build_layer_chunks(layer))

    def test_renders_are_stable_across_frames(self, qapp):
        # No shimmer-on-pan: two consecutive renders with jitter on are pixel-
        # identical because the displacement is baked, not re-rolled per frame.
        project = make_fixture_project()
        _controller, widget = _make_widget(project, 3.0, (50.0, 50.0), SIZE)
        widget._jitter_enabled = True
        first = _render(widget)
        second = _render(widget)
        assert pixel_diff_ratio(first, second) == 0.0

    def test_intensity_change_invalidates_variant(self, qapp):
        # set_jitter_intensity drops the baked variant so it rebuilds at the new
        # sigma; the un-jittered path is retained (§6.4).
        project = make_fixture_project()
        _controller, widget = _make_widget(project, 3.0, (50.0, 50.0), SIZE)
        widget._jitter_enabled = True
        layer = project.layers[0]

        low = _chunks_coords(widget._layer_chunks(layer))
        widget.set_jitter_intensity(4.0)
        high = _chunks_coords(widget._layer_chunks(layer))
        assert low != high


class TestRenderCachePerfSmoke:
    """CI-robust guard that the layer-path cache makes repaints faster.

    Spec 165.4 asks for a 5000×12 cache-on frame comfortably under the cache-off
    frame. The spec's aspirational "≤ ½" target assumes the later scene-pixmap
    *blit* cache; the layer-path cache alone reliably delivers ~1.5–1.75× (cached
    ≈ 0.6× uncached) because Qt's C++ path stroke — which caching never touches —
    dominates the frame (see ``docs/performance.md`` "Canvas rendering"). So we
    assert a robust ≥1.25× speedup (cached ≤ 0.8× uncached): wide enough that
    scheduler noise on CI can't flip it, tight enough that bypassing the cache
    (cache off ≈ cache on, ratio ≈ 1.0) fails the test outright.
    """

    def test_cached_frame_beats_uncached(self, qapp, monkeypatch):
        from tools.bench_canvas import run_bench

        def warm_floor(no_cache: bool) -> float:
            # The bypass flag is read at CanvasWidget construction, so set the
            # env *before* run_bench builds its widget.
            if no_cache:
                monkeypatch.setenv("PLOTTTER_NO_CANVAS_CACHE", "1")
            else:
                monkeypatch.delenv("PLOTTTER_NO_CANVAS_CACHE", raising=False)
            stats = run_bench(5000, 12, 8)
            assert stats["cache_enabled"] is (not no_cache)
            # Drop the first two frames (cache-build / allocator warmup) and take
            # the steady-state floor — the most noise-resistant statistic, so
            # best-cached vs best-uncached is an apples-to-apples comparison.
            return min(stats["times_ms"][2:])

        cached = warm_floor(no_cache=False)
        uncached = warm_floor(no_cache=True)

        assert cached <= 0.8 * uncached, (
            f"layer-path cache gave no robust speedup: cached={cached:.1f} ms "
            f"vs uncached={uncached:.1f} ms (ratio {cached / uncached:.2f}, "
            f"need ≤ 0.80)"
        )
