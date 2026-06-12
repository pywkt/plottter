"""Tests for ``ScenePixmapCache`` + ``scene_revision`` plumbing (canvas-performance §7).

Two concerns are covered here (no paintEvent integration yet — that lands in a
later phase):

- **§7.4 invalidation** — every enumerated setter / controller signal / event
  bumps ``CanvasWidget.scene_revision`` by exactly one, so a stale baked pixmap
  is never blitted.
- **§7.1 / §7.2 rebuild** — ``ScenePixmapCache.rebuild`` produces a non-empty
  pixmap covering the viewport plus 0.5-viewport slop per side, tagged with the
  widget's ``devicePixelRatio`` and anchored at the correct ``origin_mm``, and
  leaves the widget's pan offset untouched.
"""

from __future__ import annotations

import sys

import pytest

from PyQt6.QtCore import QPointF, QSize
from PyQt6.QtGui import QResizeEvent

from tests.canvas_render_ref import make_fixture_project


def _resize(widget, w, h):
    """Deliver a real ``QResizeEvent`` to *widget*.

    A bare ``widget.resize()`` on a hidden widget queues the resize but does
    not synchronously run ``resizeEvent`` under the offscreen platform, so the
    §7.4 bump would not fire deterministically. Dispatch the event directly.
    """
    old = widget.size()
    widget.resize(w, h)
    widget.resizeEvent(QResizeEvent(QSize(w, h), old))


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _make_widget(size=(400, 300), zoom=3.0, pan=(40.0, 30.0)):
    """Build a CanvasWidget over the fixture project, pinned to a fixed view.

    Returns ``(controller, widget)`` so callers keep the controller alive (it
    owns the signals the §7.4 tests emit through).
    """
    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.gui.project_controller import ProjectController

    controller = ProjectController(make_fixture_project())
    widget = CanvasWidget(controller)
    widget._fitted = True  # block the fit-on-show refit
    widget.resize(*size)
    widget._zoom = zoom
    widget._pan_offset = QPointF(*pan)
    return controller, widget


# --------------------------------------------------------------------------
# §7.4 — every trigger bumps scene_revision by one
# --------------------------------------------------------------------------

# Each entry is (id, action) where action(widget, controller) performs one §7.4
# trigger. The harness asserts ``scene_revision`` advances by exactly one.
def _emit(controller, signal_name, *args):
    getattr(controller, signal_name).emit(*args)


SCENE_REVISION_TRIGGERS = [
    pytest.param(lambda w, c: w.set_show_grid(True), id="set_show_grid"),
    pytest.param(lambda w, c: w.set_show_reg_marks(False), id="set_show_reg_marks"),
    pytest.param(lambda w, c: w.set_show_travel(True), id="set_show_travel"),
    pytest.param(lambda w, c: w.set_paper_texture(True), id="set_paper_texture"),
    pytest.param(lambda w, c: w.set_preview_pen_width_mm(0.8), id="set_preview_pen_width_mm"),
    pytest.param(lambda w, c: w.set_jitter_enabled(True), id="set_jitter_enabled"),
    pytest.param(lambda w, c: w.set_jitter_intensity(2.5), id="set_jitter_intensity"),
    pytest.param(lambda w, c: w.set_ink_preview(True), id="set_ink_preview"),
    pytest.param(lambda w, c: w.set_drag_move_active(True), id="set_drag_move_active"),
    pytest.param(lambda w, c: _resize(w, 640, 480), id="resizeEvent"),
    pytest.param(lambda w, c: _emit(c, "project_loaded"), id="project_loaded"),
    pytest.param(
        lambda w, c: _emit(c, "paths_changed", c.current_project.layers[0].id),
        id="paths_changed",
    ),
    pytest.param(
        lambda w, c: _emit(c, "layer_changed", c.current_project.layers[0].id),
        id="layer_changed",
    ),
    pytest.param(lambda w, c: _emit(c, "layer_added", "new-id"), id="layer_added"),
    pytest.param(
        lambda w, c: _emit(c, "layer_removed", c.current_project.layers[0].id),
        id="layer_removed",
    ),
    pytest.param(lambda w, c: _emit(c, "layers_reordered"), id="layers_reordered"),
    pytest.param(lambda w, c: _emit(c, "canvas_changed"), id="canvas_changed"),
]


@pytest.mark.parametrize("trigger", SCENE_REVISION_TRIGGERS)
def test_trigger_bumps_scene_revision(qapp, trigger):
    controller, widget = _make_widget()
    before = widget.scene_revision
    trigger(widget, controller)
    assert widget.scene_revision == before + 1


def test_starts_at_zero(qapp):
    _controller, widget = _make_widget()
    assert widget.scene_revision == 0


# --------------------------------------------------------------------------
# §7.1 / §7.2 — rebuild geometry, DPR, origin, non-emptiness
# --------------------------------------------------------------------------


def _alpha_max(pixmap):
    """Largest alpha value over the whole pixmap (0 → fully transparent)."""
    image = pixmap.toImage()
    max_a = 0
    for y in range(image.height()):
        for x in range(image.width()):
            a = image.pixelColor(x, y).alpha()
            if a > max_a:
                max_a = a
                if max_a == 255:
                    return max_a
    return max_a


def test_rebuild_covers_viewport_plus_slop(qapp):
    _controller, widget = _make_widget(size=(400, 300))
    dpr = widget.devicePixelRatioF()
    entry = widget._scene_cache.rebuild(widget)
    assert entry is not None
    # 2× viewport per dimension (viewport + 0.5 slop each side), at DPR.
    assert entry.pixmap.width() == round(2 * 400 * dpr)
    assert entry.pixmap.height() == round(2 * 300 * dpr)


def test_rebuild_tags_devicepixelratio(qapp):
    _controller, widget = _make_widget()
    entry = widget._scene_cache.rebuild(widget)
    assert entry.pixmap.devicePixelRatio() == widget.devicePixelRatioF()


def test_rebuild_origin_mm_is_top_left_of_covered_region(qapp):
    _controller, widget = _make_widget(size=(400, 300))
    entry = widget._scene_cache.rebuild(widget)
    expected = widget.pixel_to_mm(QPointF(-0.5 * 400, -0.5 * 300))
    assert entry.origin_mm[0] == pytest.approx(expected[0])
    assert entry.origin_mm[1] == pytest.approx(expected[1])


def test_rebuild_records_current_scene_revision(qapp):
    _controller, widget = _make_widget()
    widget.set_show_grid(True)  # bump once
    entry = widget._scene_cache.rebuild(widget)
    assert entry.scene_revision == widget.scene_revision


def test_rebuild_pixmap_is_non_empty(qapp):
    # Fixture has visible black line work → the baked pixmap must have ink.
    _controller, widget = _make_widget()
    entry = widget._scene_cache.rebuild(widget)
    assert not entry.pixmap.isNull()
    assert _alpha_max(entry.pixmap) > 0


def test_rebuild_restores_pan_offset(qapp):
    _controller, widget = _make_widget(pan=(40.0, 30.0))
    saved = QPointF(widget._pan_offset)
    widget._scene_cache.rebuild(widget)
    assert widget._pan_offset == saved


def test_rebuild_excludes_active_layer_during_drag(qapp):
    controller, widget = _make_widget()
    active = controller.current_project.layers[0].id
    controller.set_active_layer(active)
    widget.set_drag_move_active(True)
    entry = widget._scene_cache.rebuild(widget)
    assert entry.excluded_layer_id == active


def test_rebuild_excludes_nothing_outside_drag(qapp):
    _controller, widget = _make_widget()
    entry = widget._scene_cache.rebuild(widget)
    assert entry.excluded_layer_id is None


def test_zero_size_widget_yields_no_entry(qapp):
    _controller, widget = _make_widget()
    widget.resize(0, 0)
    entry = widget._scene_cache.rebuild(widget)
    assert entry is None
    assert widget._scene_cache.entry is None


# --------------------------------------------------------------------------
# §7.2 — validity
# --------------------------------------------------------------------------


def test_is_valid_true_after_rebuild(qapp):
    _controller, widget = _make_widget()
    widget._scene_cache.rebuild(widget)
    assert widget._scene_cache.is_valid(widget)


def test_is_valid_false_after_revision_bump(qapp):
    _controller, widget = _make_widget()
    widget._scene_cache.rebuild(widget)
    widget.set_show_grid(True)
    assert not widget._scene_cache.is_valid(widget)


def test_is_valid_false_after_zoom_change(qapp):
    _controller, widget = _make_widget()
    widget._scene_cache.rebuild(widget)
    widget._zoom *= 1.5
    assert not widget._scene_cache.is_valid(widget)


def test_is_valid_false_when_pan_beyond_slop(qapp):
    _controller, widget = _make_widget(size=(400, 300), zoom=3.0)
    widget._scene_cache.rebuild(widget)
    # Pan the viewport far past the slop region (covered = 2× viewport).
    widget._pan_offset = QPointF(
        widget._pan_offset.x() + 400 * 3.0, widget._pan_offset.y()
    )
    assert not widget._scene_cache.is_valid(widget)


def test_invalidate_drops_entry(qapp):
    _controller, widget = _make_widget()
    widget._scene_cache.rebuild(widget)
    widget._scene_cache.invalidate()
    assert widget._scene_cache.entry is None
    assert not widget._scene_cache.is_valid(widget)
