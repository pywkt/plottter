"""Tests for canvas-performance §8.1 — mask overlay QImage cache with dirty-bbox updates.

The overlay is a persistent premultiplied-ARGB ``QImage`` rebuilt from
``_mask_array``. A brush stamp rewrites **only** its bbox region of the cached
image; ``set_mask`` / ``clear_mask`` / ``invert_mask`` / shape fills invalidate
the whole cache so the next paint rebuilds from scratch.

Covered here:

- **Dirty-bbox stamp** — a single ``_paint_at`` writes no more overlay pixels
  than the stamp bbox (tracked via the ``_mask_overlay_pixels_written`` hook).
- **Incremental == rebuild** — the cached image after an incremental stroke is
  pixel-identical to a from-scratch rebuild of the same mask.
- **set / clear / invert** — each refreshes (or drops) the cache so the overlay
  matches the mask state.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from PyQt6.QtCore import QPointF

from plottter.gui.canvas_widget.enums import _MASK_PX_PER_MM
from tests.canvas_render_ref import make_fixture_project


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _make_widget(size=(400, 300), zoom=3.0, pan=(40.0, 30.0)):
    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.gui.project_controller import ProjectController

    controller = ProjectController(make_fixture_project())
    widget = CanvasWidget(controller)
    widget._fitted = True
    widget.resize(*size)
    widget._zoom = zoom
    widget._pan_offset = QPointF(*pan)
    return controller, widget


def _img_to_array(img):
    """Copy a QImage's pixel buffer into an (h, w, 4) uint8 ndarray (BGRA)."""
    h, w = img.height(), img.width()
    ptr = img.bits()
    ptr.setsize(h * w * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()


def _stamp_bbox(widget, x_mm, y_mm):
    """Replicate the bbox math from ``_paint_at`` for a stamp at (x_mm, y_mm)."""
    assert widget._mask_array is not None
    h, w = widget._mask_array.shape
    cx = x_mm * _MASK_PX_PER_MM
    cy = y_mm * _MASK_PX_PER_MM
    radius_px = max(0.5, widget._mask_brush_size_mm * _MASK_PX_PER_MM / 2.0)
    x1 = max(0, int(cx - radius_px - 1))
    y1 = max(0, int(cy - radius_px - 1))
    x2 = min(w, int(cx + radius_px + 2))
    y2 = min(h, int(cy + radius_px + 2))
    return x1, y1, x2, y2


# --------------------------------------------------------------------------
# Dirty-bbox stamp — only the stamp footprint is rewritten
# --------------------------------------------------------------------------


def test_stamp_writes_only_bbox(qapp):
    _controller, widget = _make_widget()
    widget.set_mask_paint_active(True)
    widget._mask_brush_size_mm = 5.0

    # Warm the cache with a first stamp + ensure, so the overlay exists and the
    # next stamp takes the incremental path (no full rebuild).
    widget._paint_at(50.0, 50.0)
    widget._ensure_mask_overlay()
    assert widget._mask_overlay_qimage is not None
    full_pixels = widget._mask_array.size

    widget._mask_overlay_pixels_written = 0
    widget._paint_at(30.0, 30.0)

    x1, y1, x2, y2 = _stamp_bbox(widget, 30.0, 30.0)
    bbox_area = (x2 - x1) * (y2 - y1)
    assert widget._mask_overlay_pixels_written <= bbox_area
    # Sanity: a stamp must touch far fewer pixels than a full rebuild.
    assert widget._mask_overlay_pixels_written < full_pixels


# --------------------------------------------------------------------------
# Incremental stroke == from-scratch rebuild
# --------------------------------------------------------------------------


def test_incremental_stroke_matches_rebuild(qapp):
    _controller, widget = _make_widget()
    widget.set_mask_paint_active(True)
    widget._mask_brush_size_mm = 6.0
    widget._mask_brush_hardness = 0.5  # soft falloff exercises sub-1.0 alpha

    # Paint a multi-stamp stroke incrementally (each _paint_at updates its bbox).
    widget._last_brush_pos = (20.0, 20.0)
    widget._interpolate_stroke((20.0, 20.0), (70.0, 60.0))
    widget._ensure_mask_overlay()
    incremental = _img_to_array(widget._mask_overlay_qimage)

    # Force a full rebuild from the identical mask and compare pixel-wise.
    widget._invalidate_mask_overlay()
    widget._ensure_mask_overlay()
    rebuilt = _img_to_array(widget._mask_overlay_qimage)

    assert np.array_equal(incremental, rebuilt)


# --------------------------------------------------------------------------
# set / clear / invert
# --------------------------------------------------------------------------


def test_set_mask_refreshes_overlay(qapp):
    _controller, widget = _make_widget()
    canvas = widget._controller.current_project.canvas
    h = int(canvas.height_mm * _MASK_PX_PER_MM)
    w = int(canvas.width_mm * _MASK_PX_PER_MM)

    # Build an overlay from an empty mask first.
    widget.set_mask(np.zeros((h, w), dtype=np.float32))
    widget._ensure_mask_overlay()
    empty_alpha = _img_to_array(widget._mask_overlay_qimage)[:, :, 3]
    assert empty_alpha.max() == 0

    # Replace with a filled mask — overlay must be invalidated and rebuilt.
    filled = np.ones((h, w), dtype=np.float32)
    widget.set_mask(filled)
    assert widget._mask_overlay_qimage is None  # invalidated
    widget._ensure_mask_overlay()
    arr = _img_to_array(widget._mask_overlay_qimage)
    # Full mask → alpha capped at 150 everywhere; red channel premultiplied.
    assert arr[:, :, 3].min() == 150
    assert arr[:, :, 2].max() == int(180 * 150 // 255)  # premultiplied red


def test_clear_mask_drops_overlay(qapp):
    _controller, widget = _make_widget()
    canvas = widget._controller.current_project.canvas
    h = int(canvas.height_mm * _MASK_PX_PER_MM)
    w = int(canvas.width_mm * _MASK_PX_PER_MM)

    widget.set_mask(np.ones((h, w), dtype=np.float32))
    widget._ensure_mask_overlay()
    assert widget._mask_overlay_qimage is not None

    widget.clear_mask()
    assert widget._mask_overlay_qimage is None
    # With no mask, ensuring leaves the overlay absent.
    widget._ensure_mask_overlay()
    assert widget._mask_overlay_qimage is None


def test_invert_mask_refreshes_overlay(qapp):
    _controller, widget = _make_widget()
    canvas = widget._controller.current_project.canvas
    h = int(canvas.height_mm * _MASK_PX_PER_MM)
    w = int(canvas.width_mm * _MASK_PX_PER_MM)

    # Start from a half-filled mask so inversion is visible.
    mask = np.zeros((h, w), dtype=np.float32)
    mask[: h // 2, :] = 1.0
    widget.set_mask(mask)
    widget._ensure_mask_overlay()

    widget.invert_mask()
    assert widget._mask_overlay_qimage is None  # invalidated by invert
    widget._ensure_mask_overlay()
    after = _img_to_array(widget._mask_overlay_qimage)[:, :, 3]
    # Top half now 0 alpha, bottom half full (150).
    assert after[: h // 2, :].max() == 0
    assert after[h // 2 :, :].min() == 150


# --------------------------------------------------------------------------
# §8.2 Map preview — Mercator path cache + composed transform
# --------------------------------------------------------------------------


def _make_synthetic_map_data(n=50):
    """Build a MapData with ``n`` short open-line features near (40, -74)."""
    from plottter.osm.types import MapData, MapFeature

    features = []
    base_lat, base_lon = 40.0, -74.0
    for i in range(n):
        # Deterministic spread; each polyline is a short 4-point zig-zag.
        a = (i * 0.013) % 0.2
        b = (i * 0.017) % 0.2
        coords = [
            (base_lat + a, base_lon + b),
            (base_lat + a + 0.01, base_lon + b + 0.008),
            (base_lat + a + 0.004, base_lon + b + 0.015),
            (base_lat + a + 0.018, base_lon + b + 0.02),
        ]
        features.append(MapFeature(tags={}, coords=coords, is_area=False))
    data_bounds = (base_lat, base_lon, base_lat + 0.22, base_lon + 0.22)
    return MapData(
        location="synthetic",
        center=(base_lat + 0.1, base_lon + 0.1),
        bbox=(base_lat, base_lon, base_lat + 0.22, base_lon + 0.22),
        features={"roads": features},
    ), data_bounds


def _render_map_preview_new(widget, canvas, size):
    """Render only the preview lines via the production composed-transform path."""
    from PyQt6.QtGui import QImage, QPainter

    img = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    # Bounds rectangle disabled so we compare only the line render.
    widget._map_data_bounds = None
    widget._draw_map_preview(painter, canvas)
    painter.end()
    return _img_to_array(img)


def _render_map_preview_ref(widget, canvas, size):
    """Reference: the legacy per-point projection loop (the oracle)."""
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen

    from plottter.osm.geometry import view_transform

    transform = view_transform(
        widget._map_view["center_lat"],
        widget._map_view["center_lon"],
        widget._map_view["scale"],
        canvas,
    )
    img = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    pen = QPen(QColor(120, 120, 120, 180), max(0.5, widget._zoom * 0.2))
    painter.setPen(pen)
    for polyline in widget._map_preview_polylines:
        if len(polyline) < 2:
            continue
        pts_mm = [
            (
                transform.x_origin + mx * transform.scale,
                transform.y_origin - my * transform.scale,
            )
            for mx, my in polyline
        ]
        pts_px = [widget.mm_to_pixel(p) for p in pts_mm]
        for i in range(len(pts_px) - 1):
            painter.drawLine(pts_px[i], pts_px[i + 1])
    painter.end()
    return _img_to_array(img)


@pytest.mark.parametrize(
    "view",
    [
        {"center_lat": 40.1, "center_lon": -73.9, "scale": 400.0},
        {"center_lat": 40.05, "center_lon": -73.95, "scale": 900.0},
    ],
)
def test_map_preview_composed_transform_matches_reference(qapp, view):
    size = (400, 300)
    _controller, widget = _make_widget(size=size)
    canvas = widget._controller.current_project.canvas

    map_data, data_bounds = _make_synthetic_map_data(50)
    widget.set_map_preview_data(map_data, data_bounds)
    assert widget._map_merc_path.elementCount() > 0

    widget._map_view = dict(view)

    got = _render_map_preview_new(widget, canvas, size)
    ref = _render_map_preview_ref(widget, canvas, size)

    # Mean per-channel absolute difference, normalised to [0, 1].
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16)).mean() / 255.0
    assert diff <= 0.02, f"composed-transform render diverged: mean diff {diff:.4f}"


def test_map_preview_path_not_rebuilt_on_view_change(qapp):
    _controller, widget = _make_widget()
    map_data, data_bounds = _make_synthetic_map_data(50)
    widget.set_map_preview_data(map_data, data_bounds)

    path_before = widget._map_merc_path
    widget.update_map_view({"center_lat": 40.2, "center_lon": -73.8, "scale": 1200.0})
    # Panning/zooming the view must not rebuild the Mercator path.
    assert widget._map_merc_path is path_before


# --------------------------------------------------------------------------
# §8.3 3D wireframe — mm-coordinate path cache + world transform
# --------------------------------------------------------------------------


def _make_wireframe_polylines(n=40):
    """Deterministic mm-coordinate zig-zag polylines inside the 100mm canvas."""
    polylines = []
    for i in range(n):
        a = 10.0 + (i * 1.9) % 70.0
        b = 12.0 + (i * 2.3) % 70.0
        polylines.append(
            [
                (a, b),
                (a + 6.0, b + 4.0),
                (a + 2.0, b + 9.0),
                (a + 11.0, b + 12.0),
            ]
        )
    return polylines


def _render_3d_wire_new(widget, size):
    """Render only the wireframe via the production stored-path + world transform."""
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QTransform

    img = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    mm_to_px = QTransform(
        widget._zoom, 0.0, 0.0, widget._zoom,
        widget._pan_offset.x(), widget._pan_offset.y(),
    )
    pen = QPen(QColor("#00E5FF"), max(0.5 / widget._zoom, 0.25))
    pen.setCosmetic(False)
    painter.setTransform(mm_to_px, combine=True)
    painter.setPen(pen)
    painter.drawPath(widget._3d_wire_path)
    painter.end()
    return _img_to_array(img)


def _render_3d_wire_ref(widget, size):
    """Reference: the legacy per-point cull/convert loop (the oracle)."""
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen

    img = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    pen = QPen(QColor("#00E5FF"), max(0.5, widget._zoom * 0.25))
    painter.setPen(pen)

    vp_left, vp_top = widget.pixel_to_mm(QPointF(0.0, 0.0))
    vp_right, vp_bottom = widget.pixel_to_mm(
        QPointF(float(size[0]), float(size[1]))
    )
    for polyline in widget._3d_wireframe_polylines:
        if len(polyline) < 2:
            continue
        xs = [p[0] for p in polyline]
        ys = [p[1] for p in polyline]
        if (
            max(xs) < vp_left
            or min(xs) > vp_right
            or max(ys) < vp_top
            or min(ys) > vp_bottom
        ):
            continue
        pts = [widget.mm_to_pixel(pt) for pt in polyline]
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])
    painter.end()
    return _img_to_array(img)


def test_3d_wireframe_path_matches_reference(qapp):
    size = (400, 300)
    _controller, widget = _make_widget(size=size)

    polylines = _make_wireframe_polylines(40)
    widget.set_3d_wireframe_polylines(polylines)
    assert widget._3d_wire_path.elementCount() > 0

    got = _render_3d_wire_new(widget, size)
    ref = _render_3d_wire_ref(widget, size)

    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16)).mean() / 255.0
    assert diff <= 0.02, f"wireframe path render diverged: mean diff {diff:.4f}"
    # Sanity: the wireframe actually drew cyan pixels (not two blank images).
    assert got[:, :, 3].max() > 0


def test_3d_wireframe_empty_shows_loading_without_error(qapp):
    """An empty polyline set leaves the path empty and still paints the loading text."""
    from PyQt6.QtGui import QImage, QPainter

    _controller, widget = _make_widget()
    canvas = widget._controller.current_project.canvas

    widget.set_3d_wireframe_polylines([])
    assert widget._3d_wire_path.elementCount() == 0
    assert not widget._3d_wireframe_polylines

    img = QImage(400, 300, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    # Must not raise even with no wireframe — loading text branch is exercised.
    widget._draw_3d_preview(painter, canvas)
    painter.end()
    # The dark viewport + loading label drew some pixels.
    assert _img_to_array(img)[:, :, 3].max() > 0


def test_overlay_matches_reference_build(qapp):
    """The cached overlay equals an independent premultiplied-ARGB computation."""
    _controller, widget = _make_widget()
    canvas = widget._controller.current_project.canvas
    h = int(canvas.height_mm * _MASK_PX_PER_MM)
    w = int(canvas.width_mm * _MASK_PX_PER_MM)

    rng = np.random.default_rng(0)
    mask = rng.random((h, w)).astype(np.float32)
    widget.set_mask(mask)
    widget._ensure_mask_overlay()
    got = _img_to_array(widget._mask_overlay_qimage)

    alpha = (mask * 150.0).astype(np.uint16)
    exp_r = (180 * alpha // 255).astype(np.uint8)
    exp_g = (30 * alpha // 255).astype(np.uint8)
    exp_a = alpha.astype(np.uint8)
    assert np.array_equal(got[:, :, 0], np.zeros((h, w), dtype=np.uint8))  # B
    assert np.array_equal(got[:, :, 1], exp_g)
    assert np.array_equal(got[:, :, 2], exp_r)
    assert np.array_equal(got[:, :, 3], exp_a)


# --------------------------------------------------------------------------
# §8.4 Animation — incremental completed-paths cache
# --------------------------------------------------------------------------


def _render_anim(widget):
    """Render only the animation overlay (done paths + partial + crosshair)."""
    from PyQt6.QtGui import QImage, QPainter

    img = QImage(
        widget.width(), widget.height(), QImage.Format.Format_ARGB32_Premultiplied
    )
    img.fill(0)
    painter = QPainter(img)
    widget._draw_animated_paths(painter)
    painter.end()
    return _img_to_array(img)


def _enter_anim(widget):
    """Put the widget into animation mode with a freshly collected path set."""
    widget._rebuild_anim_paths()
    widget._anim_mode = True


def test_anim_incremental_done_paths_match_rebuild(qapp):
    """At tick N, the incrementally appended cache renders like a fresh rebuild."""
    _controller, widget = _make_widget()
    _enter_anim(widget)
    assert len(widget._anim_all_paths) > 6

    # Advance several paths — each step bakes the completed path into the cache.
    for _ in range(6):
        widget.step_anim_forward()
    # Land partway into the current stroke so a live partial is drawn too.
    widget._anim_current_point = 1

    incremental = _render_anim(widget)
    assert incremental[:, :, 3].max() > 0  # actually drew pixels

    # A from-scratch rebuild at the identical play head must be pixel-identical.
    widget._rebuild_anim_done_paths()
    rebuilt = _render_anim(widget)
    assert np.array_equal(incremental, rebuilt)


def test_anim_seek_matches_incremental(qapp):
    """seek_animation rebuilds a cache that renders like the incremental one."""
    _controller, widget = _make_widget()
    _enter_anim(widget)

    for _ in range(5):
        widget.step_anim_forward()
    incremental = _render_anim(widget)

    widget.seek_animation(5)  # same head, but rebuilt from scratch
    seeked = _render_anim(widget)
    assert np.array_equal(incremental, seeked)


def test_anim_step_backward_render_matches_fresh_seek(qapp):
    """After a backward step the render matches a fresh widget seeked there."""
    _controller, widget = _make_widget()
    _enter_anim(widget)
    for _ in range(8):
        widget.step_anim_forward()
    widget.step_anim_backward()  # play head 8 -> 7, cache rebuilt
    assert widget._anim_current_path == 7
    got = _render_anim(widget)

    _controller2, fresh = _make_widget()
    fresh._anim_mode = True
    fresh._rebuild_anim_paths()
    fresh.seek_animation(7)
    ref = _render_anim(fresh)
    assert np.array_equal(got, ref)


def test_anim_jitter_baked_incremental_matches_rebuild(qapp):
    """Baked (crc-seeded) jitter is order-independent: append == rebuild."""
    _controller, widget = _make_widget()
    widget._jitter_enabled = True
    widget._jitter_intensity = 2.0
    _enter_anim(widget)

    for _ in range(6):
        widget.step_anim_forward()
    # current_point == 0 ⇒ the live partial has <2 points (crosshair only), so
    # the frame is fully determined by the baked done-paths cache.
    incremental = _render_anim(widget)
    assert incremental[:, :, 3].max() > 0

    widget._rebuild_anim_done_paths()
    rebuilt = _render_anim(widget)
    assert np.array_equal(incremental, rebuilt)


def test_anim_done_paths_preserve_colors(qapp):
    """Every visible (color, opacity) combination gets its own cached path."""
    _controller, widget = _make_widget()
    _enter_anim(widget)

    # Run to the end so every path is completed and baked.
    widget.seek_animation(len(widget._anim_all_paths))

    keys = set(widget._anim_done_paths.keys())
    expected = {
        (layer.color, layer.opacity)
        for layer in widget._controller.current_project.layers
        if layer.visible
    }
    assert keys == expected
    assert len(keys) == 3  # fixture: black, red dots, 50%-opacity blue
    for path in widget._anim_done_paths.values():
        assert path.elementCount() > 0
