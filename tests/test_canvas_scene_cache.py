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


# --------------------------------------------------------------------------
# §7.1 / §7.3 / §7.6 — paintEvent integration: blit vs bypass, pan slop
# --------------------------------------------------------------------------


def _render_widget(widget):
    """Run a real paint into a fresh ARGB image and return it.

    ``QWidget.render`` invokes ``paintEvent`` synchronously, so this exercises
    the production blit/bypass branch exactly as an on-screen repaint would.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage

    img = QImage(widget.width(), widget.height(), QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    widget.render(img)
    return img


@pytest.mark.parametrize("pan", [(40.0, 30.0), (120.0, 90.0)])
def test_cached_blit_matches_bypass(qapp, pan):
    """A blitted frame is pixel-identical to a direct §6 draw (same code, §9).

    The cache renders the static scene into a pixmap anchored so that
    ``mm_to_pixel(origin_mm)`` lands on the viewport pixel grid exactly, then
    blits it; the bypass path draws the same ``_draw_*`` helpers live. Both
    must agree at every pan state.
    """
    from tests.canvas_render_ref import pixel_diff_ratio

    _c1, cached = _make_widget(pan=pan)
    cached._render_cache_enabled = True
    img_cached = _render_widget(cached)

    _c2, bypass = _make_widget(pan=pan)
    bypass._render_cache_enabled = False
    img_bypass = _render_widget(bypass)

    assert pixel_diff_ratio(img_cached, img_bypass) <= 0.001


def test_pan_within_slop_reuses_pixmap(qapp):
    """A small pan stays inside the 0.5-viewport slop → no rebuild (§7.2/§7.3)."""
    _c, widget = _make_widget(size=(400, 300), zoom=3.0, pan=(40.0, 30.0))
    widget._render_cache_enabled = True
    _render_widget(widget)  # first paint builds the pixmap
    assert widget._scene_cache.rebuild_count == 1

    # Shift well inside the slop (200 px horizontal / 150 px vertical each side).
    widget._pan_offset = QPointF(
        widget._pan_offset.x() + 20.0, widget._pan_offset.y() + 15.0
    )
    _render_widget(widget)
    assert widget._scene_cache.rebuild_count == 1  # reused, not rebuilt


def test_pan_beyond_slop_triggers_rebuild(qapp):
    """A pan past the slop region forces a synchronous rebuild then blit (§7.3)."""
    _c, widget = _make_widget(size=(400, 300), zoom=3.0, pan=(40.0, 30.0))
    widget._render_cache_enabled = True
    _render_widget(widget)
    assert widget._scene_cache.rebuild_count == 1

    # Shift the viewport far past the 200 px horizontal slop.
    widget._pan_offset = QPointF(
        widget._pan_offset.x() - 400.0, widget._pan_offset.y()
    )
    _render_widget(widget)
    assert widget._scene_cache.rebuild_count == 2


def test_ink_preview_bypasses_cache(qapp):
    """Ink preview is a §7.6 bypass: it never blits, and the live render
    matches an explicitly cache-disabled render of the same scene."""
    from tests.canvas_render_ref import pixel_diff_ratio

    _c1, inked = _make_widget()
    inked._render_cache_enabled = True
    inked.set_ink_preview(True)
    assert inked._scene_cache_active() is False  # bypassed despite cache on
    img_inked = _render_widget(inked)
    # No pixmap was baked while bypassed.
    assert inked._scene_cache.rebuild_count == 0

    _c2, bypass = _make_widget()
    bypass._render_cache_enabled = False
    bypass.set_ink_preview(True)
    img_bypass = _render_widget(bypass)

    assert pixel_diff_ratio(img_inked, img_bypass) <= 0.001


# --------------------------------------------------------------------------
# §7.3 / §2.2 — soft zoom blits + 120 ms idle crisp re-render
# --------------------------------------------------------------------------


def test_soft_zoom_blits_scaled_without_rebuild(qapp):
    """A small wheel-zoom mid-gesture reuses the crisp pixmap scaled (§7.3).

    With ``current/cached ∈ [0.5, 2.0]`` the frame is a scaled blit of the last
    crisp pixmap — no rebuild fires and the cached entry stays at the old zoom.
    """
    _c, widget = _make_widget(size=(400, 300), zoom=3.0, pan=(40.0, 30.0))
    widget._render_cache_enabled = True
    _render_widget(widget)
    assert widget._scene_cache.rebuild_count == 1
    cached_zoom = widget._scene_cache.entry.zoom

    widget._apply_zoom(1.25, widget.rect().center())
    assert widget._zoom != cached_zoom
    assert 0.5 <= widget._zoom / cached_zoom <= 2.0
    assert widget._zoom_idle_timer.isActive()  # idle timer armed by the gesture

    _render_widget(widget)
    # Scaled blit reused the crisp pixmap: no rebuild, entry still at old zoom.
    assert widget._scene_cache.rebuild_count == 1
    assert widget._scene_cache.entry.zoom == cached_zoom


def test_zoom_idle_timeout_rebuilds_crisp(qapp):
    """The 120 ms idle timeout rebuilds crisp and matches a fresh render (§7.3).

    After the soft frame, firing the idle timer's timeout rebuilds the pixmap
    at the now-stable zoom (rebuild count advances, cache zoom matches) and the
    resulting crisp blit equals a direct cache-disabled render of the same view.
    """
    from tests.canvas_render_ref import pixel_diff_ratio

    _c, widget = _make_widget(size=(400, 300), zoom=3.0, pan=(40.0, 30.0))
    widget._render_cache_enabled = True
    _render_widget(widget)
    widget._apply_zoom(1.25, widget.rect().center())
    _render_widget(widget)  # soft scaled frame
    assert widget._scene_cache.rebuild_count == 1

    # Invoke the timer's timeout (no event loop wait): rebuild crisp.
    widget._zoom_idle_timer.timeout.emit()
    assert widget._scene_cache.rebuild_count == 2
    assert widget._scene_cache.entry.zoom == widget._zoom

    img_crisp = _render_widget(widget)

    # A fresh bypass render at the post-zoom view is the oracle.
    _c2, bypass = _make_widget(size=(400, 300))
    bypass._render_cache_enabled = False
    bypass._zoom = widget._zoom
    bypass._pan_offset = QPointF(widget._pan_offset)
    img_bypass = _render_widget(bypass)

    assert pixel_diff_ratio(img_crisp, img_bypass) <= 0.001


def test_large_zoom_step_rebuilds_synchronously(qapp):
    """A zoom ratio outside [0.5, 2.0] rebuilds now rather than blitting garbage."""
    _c, widget = _make_widget(size=(400, 300), zoom=3.0, pan=(40.0, 30.0))
    widget._render_cache_enabled = True
    _render_widget(widget)
    assert widget._scene_cache.rebuild_count == 1

    widget._apply_zoom(0.4, widget.rect().center())  # ratio 0.4 < 0.5
    assert widget._zoom / 3.0 < 0.5
    _render_widget(widget)
    # Out of soft range → synchronous crisp rebuild at the new zoom.
    assert widget._scene_cache.rebuild_count == 2
    assert widget._scene_cache.entry.zoom == widget._zoom
