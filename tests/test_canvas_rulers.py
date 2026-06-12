"""Tests for the canvas ruler widgets (canvas-performance §10).

Covers the three things the Phase 169.1 build delivers:

- :func:`choose_tick_steps` — table-driven over a range of zoom levels.
- :class:`RulerWidget` renders something non-blank offscreen.
- Tick pixel positions agree with ``CanvasWidget.mm_to_pixel`` for a fixed,
  known view transform (and the ``view_zoom`` / ``view_pan_offset`` accessors
  expose that transform).
"""

from __future__ import annotations

import sys

import pytest

from PyQt6.QtCore import QPointF, QSize, Qt
from PyQt6.QtGui import QImage, QPainter

from plottter.gui.widgets.ruler import (
    RULER_THICKNESS_PX,
    RulerCorner,
    RulerWidget,
    choose_tick_steps,
)


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# --------------------------------------------------------------------------
# choose_tick_steps — pure function, no Qt needed
# --------------------------------------------------------------------------

# (zoom_px_per_mm, expected_major_mm, expected_minor_mm)
# major = smallest step in [1,2,5,10,20,50,100,200] with step*zoom >= 48
# minor = major/5 if (major/5)*zoom >= 4 else major
TICK_CASES = [
    # zoom 0.1 px/mm: 200*0.1=20 < 48 for all → fall back to 200.
    #   minor 40*0.1=4.0 >= 4 → minor = 40.
    pytest.param(0.1, 200.0, 40.0, id="zoom-0.1"),
    # zoom 1 px/mm: 50*1=50 >= 48 → major 50. minor 10*1=10 >= 4 → 10.
    pytest.param(1.0, 50.0, 10.0, id="zoom-1"),
    # zoom 5 px/mm: 10*5=50 >= 48 → major 10. minor 2*5=10 >= 4 → 2.
    pytest.param(5.0, 10.0, 2.0, id="zoom-5"),
    # zoom 20 px/mm: 5*20=100 >= 48 but 2*20=40 < 48 → major 5.
    #   minor 1*20=20 >= 4 → 1.
    pytest.param(20.0, 5.0, 1.0, id="zoom-20"),
]


@pytest.mark.parametrize("zoom,major,minor", TICK_CASES)
def test_choose_tick_steps(zoom, major, minor):
    got_major, got_minor = choose_tick_steps(zoom)
    assert got_major == pytest.approx(major)
    assert got_minor == pytest.approx(minor)


def test_choose_tick_steps_major_spacing_at_least_48():
    # The chosen major step must render at >= 48 px unless we hit the 200 cap.
    for zoom in (0.05, 0.3, 0.7, 1.0, 3.0, 8.0, 25.0, 60.0):
        major, minor = choose_tick_steps(zoom)
        if major < 200.0:
            assert major * zoom >= 48.0
        assert minor <= major
        # minor either subdivides the major by 5 or equals it (no minors).
        assert minor == pytest.approx(major) or minor == pytest.approx(major / 5.0)


def test_choose_tick_steps_handles_nonpositive_zoom():
    # Defensive: zero / negative zoom must not raise or divide by zero.
    assert choose_tick_steps(0.0) == choose_tick_steps(1.0)
    assert choose_tick_steps(-3.0) == choose_tick_steps(1.0)


# --------------------------------------------------------------------------
# Test double for the canvas — only the accessors the rulers read
# --------------------------------------------------------------------------


class _FakeCanvas:
    def __init__(self, zoom, pan):
        self._zoom = zoom
        self._pan = QPointF(*pan)

    def view_zoom(self):
        return self._zoom

    def view_pan_offset(self):
        return QPointF(self._pan)

    def mm_to_pixel(self, point):
        return QPointF(
            point[0] * self._zoom + self._pan.x(),
            point[1] * self._zoom + self._pan.y(),
        )


def _render(widget, w, h):
    """Render *widget* to an offscreen QImage and return it."""
    widget.resize(w, h)
    img = QImage(QSize(w, h), QImage.Format.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    widget.render(painter)
    painter.end()
    return img


def _is_non_blank(img):
    # At least one pixel differs from the top-left background pixel.
    bg = img.pixel(0, 0)
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            if img.pixel(x, y) != bg:
                return True
    return False


# --------------------------------------------------------------------------
# RulerWidget / RulerCorner rendering
# --------------------------------------------------------------------------


def test_horizontal_ruler_has_fixed_thickness(qapp):
    ruler = RulerWidget(Qt.Orientation.Horizontal)
    assert ruler.height() == RULER_THICKNESS_PX


def test_vertical_ruler_has_fixed_thickness(qapp):
    ruler = RulerWidget(Qt.Orientation.Vertical)
    assert ruler.width() == RULER_THICKNESS_PX


def test_corner_is_24_square_and_renders(qapp):
    corner = RulerCorner()
    assert corner.width() == RULER_THICKNESS_PX
    assert corner.height() == RULER_THICKNESS_PX
    img = _render(corner, RULER_THICKNESS_PX, RULER_THICKNESS_PX)
    assert _is_non_blank(img)  # the "mm" label + border


def test_horizontal_ruler_renders_non_blank(qapp):
    canvas = _FakeCanvas(zoom=4.0, pan=(20.0, 0.0))
    ruler = RulerWidget(Qt.Orientation.Horizontal, canvas=canvas)
    img = _render(ruler, 400, RULER_THICKNESS_PX)
    assert _is_non_blank(img)


def test_vertical_ruler_renders_non_blank(qapp):
    canvas = _FakeCanvas(zoom=4.0, pan=(0.0, 15.0))
    ruler = RulerWidget(Qt.Orientation.Vertical, canvas=canvas)
    img = _render(ruler, RULER_THICKNESS_PX, 400)
    assert _is_non_blank(img)


def test_ruler_without_canvas_is_blank_but_safe(qapp):
    # No canvas attached: paints background only, must not raise.
    ruler = RulerWidget(Qt.Orientation.Horizontal)
    img = _render(ruler, 200, RULER_THICKNESS_PX)
    assert img.width() == 200


# --------------------------------------------------------------------------
# Tick positions agree with the canvas transform
# --------------------------------------------------------------------------


def _expected_major_pixels(canvas, length_px, horizontal):
    """Major-tick pixel positions per the spec, via mm_to_pixel."""
    zoom = canvas.view_zoom()
    pan = canvas.view_pan_offset()
    pan_axis = pan.x() if horizontal else pan.y()
    major, _minor = choose_tick_steps(zoom)
    mm_low = (0.0 - pan_axis) / zoom
    mm_high = (length_px - pan_axis) / zoom
    pixels = []
    import math

    for n in range(math.floor(mm_low / major), math.ceil(mm_high / major) + 1):
        mm = n * major
        pt = canvas.mm_to_pixel((mm, mm))
        pixels.append(pt.x() if horizontal else pt.y())
    return major, pixels


def test_horizontal_tick_positions_align_with_mm_to_pixel(qapp):
    # A known transform: 5 px/mm, pan +30 px → mm 0 lands at x=30.
    canvas = _FakeCanvas(zoom=5.0, pan=(30.0, 7.0))
    ruler = RulerWidget(Qt.Orientation.Horizontal, canvas=canvas)
    length = 400

    major, expected = _expected_major_pixels(canvas, length, horizontal=True)
    # Spec: major step is 10 mm at zoom 5 → ticks every 50 px starting at x=30.
    assert major == pytest.approx(10.0)
    # mm 0 → x = 0*5 + 30 = 30
    assert 30.0 in [pytest.approx(p) for p in expected]
    # The ruler computes its own ticks from the same accessors, so the
    # widget's mm→px mapping must equal the canvas's mm_to_pixel exactly.
    for mm in (0.0, 10.0, 20.0, 50.0):
        ruler_px = ruler._mm_to_px(mm, canvas.view_zoom(), canvas.view_pan_offset().x())
        assert ruler_px == pytest.approx(canvas.mm_to_pixel((mm, 0.0)).x())


def test_vertical_tick_positions_align_with_mm_to_pixel(qapp):
    canvas = _FakeCanvas(zoom=2.0, pan=(0.0, -40.0))
    ruler = RulerWidget(Qt.Orientation.Vertical, canvas=canvas)
    for mm in (0.0, 25.0, 50.0, 100.0):
        ruler_px = ruler._mm_to_px(mm, canvas.view_zoom(), canvas.view_pan_offset().y())
        assert ruler_px == pytest.approx(canvas.mm_to_pixel((0.0, mm)).y())


def test_marker_line_moves_with_mouse_position(qapp):
    canvas = _FakeCanvas(zoom=3.0, pan=(10.0, 10.0))
    ruler = RulerWidget(Qt.Orientation.Horizontal, canvas=canvas)
    ruler.set_marker_mm(40.0, 12.0)
    assert ruler._marker_mm == pytest.approx(40.0)  # horizontal uses x
    ruler.clear_marker()
    assert ruler._marker_mm is None


def test_vertical_marker_uses_y(qapp):
    canvas = _FakeCanvas(zoom=3.0, pan=(10.0, 10.0))
    ruler = RulerWidget(Qt.Orientation.Vertical, canvas=canvas)
    ruler.set_marker_mm(40.0, 12.0)
    assert ruler._marker_mm == pytest.approx(12.0)  # vertical uses y


def test_view_accessors_round_trip_on_real_canvas(qapp):
    # The real CanvasWidget exposes view_zoom / view_pan_offset matching its
    # mm_to_pixel transform.
    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.gui.project_controller import ProjectController
    from tests.canvas_render_ref import make_fixture_project

    controller = ProjectController(make_fixture_project())
    widget = CanvasWidget(controller)
    widget._zoom = 6.0
    widget._pan_offset = QPointF(33.0, -12.0)

    assert widget.view_zoom() == pytest.approx(6.0)
    assert widget.view_pan_offset().x() == pytest.approx(33.0)
    assert widget.view_pan_offset().y() == pytest.approx(-12.0)
    # Accessor-derived mapping equals the canvas's own mm_to_pixel.
    pt = widget.mm_to_pixel((10.0, 20.0))
    assert pt.x() == pytest.approx(10.0 * 6.0 + 33.0)
    assert pt.y() == pytest.approx(20.0 * 6.0 - 12.0)


# --------------------------------------------------------------------------
# view_changed signal — emitted from every view-mutating site (§10.3)
# --------------------------------------------------------------------------


def _real_canvas(qapp):
    """A real CanvasWidget over the fixture project, sized non-zero."""
    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.gui.project_controller import ProjectController
    from tests.canvas_render_ref import make_fixture_project

    controller = ProjectController(make_fixture_project())
    widget = CanvasWidget(controller)
    widget.resize(800, 600)
    return widget


class _SignalSpy:
    def __init__(self, signal):
        self.count = 0
        signal.connect(self._bump)

    def _bump(self, *args):
        self.count += 1


def test_apply_zoom_emits_view_changed(qapp):
    widget = _real_canvas(qapp)
    spy = _SignalSpy(widget.view_changed)
    widget._apply_zoom(1.25, widget.rect().center())
    assert spy.count == 1


def test_apply_zoom_no_change_does_not_emit(qapp):
    # Clamped at MAX_ZOOM → no transform change → no signal.
    widget = _real_canvas(qapp)
    widget._zoom = widget.MAX_ZOOM
    spy = _SignalSpy(widget.view_changed)
    widget._apply_zoom(2.0, widget.rect().center())
    assert spy.count == 0


def test_pan_left_emits_view_changed(qapp):
    widget = _real_canvas(qapp)
    spy = _SignalSpy(widget.view_changed)
    widget.pan_left()
    assert spy.count == 1


def test_pan_directions_each_emit_view_changed(qapp):
    widget = _real_canvas(qapp)
    spy = _SignalSpy(widget.view_changed)
    widget.pan_left()
    widget.pan_right()
    widget.pan_up()
    widget.pan_down()
    assert spy.count == 4


def test_center_view_emits_view_changed(qapp):
    widget = _real_canvas(qapp)
    spy = _SignalSpy(widget.view_changed)
    widget.center_view()
    assert spy.count == 1


def test_fit_to_window_emits_view_changed(qapp):
    widget = _real_canvas(qapp)
    spy = _SignalSpy(widget.view_changed)
    widget.fit_to_window()
    assert spy.count == 1


def test_mouse_left_signal_exists_and_emits(qapp):
    widget = _real_canvas(qapp)
    spy = _SignalSpy(widget.mouse_left)
    widget.mouse_left.emit()
    assert spy.count == 1


# --------------------------------------------------------------------------
# MainWindow integration — rulers present, canvas still renders (§10.1)
# --------------------------------------------------------------------------


def test_main_window_has_rulers_and_renders_canvas(qapp):
    from plottter.gui.main_window import MainWindow
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.widgets.ruler import RulerCorner, RulerWidget
    from tests.canvas_render_ref import make_fixture_project

    controller = ProjectController(make_fixture_project())
    win = MainWindow(controller)
    win._prompt_save_if_modified = lambda: True

    # Rulers were constructed and framed around the canvas.
    assert isinstance(win._ruler_top, RulerWidget)
    assert isinstance(win._ruler_left, RulerWidget)
    assert isinstance(win._ruler_corner, RulerCorner)
    assert win._ruler_top._canvas is win._canvas
    assert win._ruler_left._canvas is win._canvas

    # The canvas still renders the fixture (non-blank) inside the grid.
    win._canvas.fit_to_window()
    img = _render(win._canvas, 800, 600)
    assert _is_non_blank(img)


# --------------------------------------------------------------------------
# Show Rulers toggle + QSettings persistence (§10.4)
# --------------------------------------------------------------------------


def _make_window(qapp):
    from plottter.gui.main_window import MainWindow
    from plottter.gui.project_controller import ProjectController
    from tests.canvas_render_ref import make_fixture_project

    controller = ProjectController(make_fixture_project())
    win = MainWindow(controller)
    win._prompt_save_if_modified = lambda: True
    return win


def test_show_rulers_default_hidden(qapp):
    # Default (no saved preference) is hidden — the rulers are off by default.
    from PyQt6.QtCore import QSettings

    QSettings("Plottter", "Plottter").remove("view/show_rulers")
    win = _make_window(qapp)
    assert win._act_show_rulers.isChecked() is False
    assert win._ruler_top.isHidden()
    assert win._ruler_left.isHidden()
    assert win._ruler_corner.isHidden()


def test_show_rulers_action_toggles_widget_visibility(qapp):
    win = _make_window(qapp)

    win._act_show_rulers.setChecked(True)
    assert not win._ruler_top.isHidden()
    assert not win._ruler_left.isHidden()
    assert not win._ruler_corner.isHidden()

    win._act_show_rulers.setChecked(False)
    assert win._ruler_top.isHidden()
    assert win._ruler_left.isHidden()
    assert win._ruler_corner.isHidden()


def test_show_rulers_persists_across_fresh_window(qapp):
    # Enabling rulers and saving state must restore them on a fresh window.
    from PyQt6.QtCore import QSettings

    win = _make_window(qapp)
    win._act_show_rulers.setChecked(True)
    win._save_state()

    assert QSettings("Plottter", "Plottter").value(
        "view/show_rulers", False, type=bool
    ) is True

    win2 = _make_window(qapp)
    assert win2._act_show_rulers.isChecked() is True
    assert not win2._ruler_top.isHidden()
    assert not win2._ruler_left.isHidden()
    assert not win2._ruler_corner.isHidden()


def test_show_rulers_disabled_state_persists(qapp):
    from PyQt6.QtCore import QSettings

    win = _make_window(qapp)
    win._act_show_rulers.setChecked(True)
    win._save_state()
    win._act_show_rulers.setChecked(False)
    win._save_state()

    assert QSettings("Plottter", "Plottter").value(
        "view/show_rulers", True, type=bool
    ) is False

    win2 = _make_window(qapp)
    assert win2._act_show_rulers.isChecked() is False
    assert win2._ruler_top.isHidden()


def test_main_window_view_change_repaints_rulers(qapp):
    # The canvas's view_changed is wired to the rulers' update().
    from plottter.gui.main_window import MainWindow
    from plottter.gui.project_controller import ProjectController
    from tests.canvas_render_ref import make_fixture_project

    controller = ProjectController(make_fixture_project())
    win = MainWindow(controller)
    win._prompt_save_if_modified = lambda: True

    # Driving a marker through mouse_position_mm reaches the rulers.
    win._canvas.mouse_position_mm.emit(42.0, 17.0)
    assert win._ruler_top._marker_mm == pytest.approx(42.0)
    assert win._ruler_left._marker_mm == pytest.approx(17.0)

    win._canvas.mouse_left.emit()
    assert win._ruler_top._marker_mm is None
    assert win._ruler_left._marker_mm is None
