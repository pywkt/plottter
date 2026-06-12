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
