"""Tests for direct-manipulation image positioning on the canvas.

Covers:
- Canvas: set_image_position_active() is mutually exclusive with map/mask/3D modes.
- Canvas: drag math translates _image_overlay_rect_mm by the mm-equivalent of the
  pixel delta and emits image_view_changed.
- Canvas: wheel zoom about the cursor leaves the mm point under the cursor fixed.
- Panel: toggling Position Image enables the canvas mode and switches Fit Mode
  to "Custom Size", capturing the current rect into the spinboxes.
- Panel: Reset Position restores Fit (Keep Aspect) with zero offset.
- Panel: _on_canvas_image_view_changed inverts the rect to (w, h, ox, oy) and
  persists image_view to project.metadata.
- Panel: project_loaded restores image_view from project.metadata.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from plottter.models import Canvas, Layer, Project


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="ImgPosTest", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project())


@pytest.fixture
def canvas_widget(qapp, controller):
    from plottter.gui.canvas_widget import CanvasWidget
    w = CanvasWidget(controller)
    w.resize(800, 600)
    yield w
    w.close()


@pytest.fixture
def panel(qapp, controller):
    from plottter.gui.settings_panel import SettingsPanel
    p = SettingsPanel(controller)
    yield p
    p.close()


# ---------------------------------------------------------------------------
# Canvas: mode flag + mutex
# ---------------------------------------------------------------------------


def test_image_position_active_clears_other_modes(canvas_widget):
    canvas_widget._mask_paint_active = True
    canvas_widget._shape_draw_active = True
    canvas_widget._map_position_active = True
    canvas_widget.set_image_position_active(True)
    assert canvas_widget._image_position_active is True
    assert canvas_widget._mask_paint_active is False
    assert canvas_widget._shape_draw_active is False
    assert canvas_widget._map_position_active is False


def test_map_position_active_clears_image_position(canvas_widget):
    canvas_widget.set_image_position_active(True)
    canvas_widget.set_map_position_active(True)
    assert canvas_widget._image_position_active is False
    assert canvas_widget._map_position_active is True


def test_image_position_inactive_clears_drag_state(canvas_widget):
    from PyQt6.QtCore import QPoint
    canvas_widget.set_image_position_active(True)
    canvas_widget._image_pan_drag_start = QPoint(100, 100)
    canvas_widget._image_pan_start_rect = (0, 0, 50, 50)
    canvas_widget.set_image_position_active(False)
    assert canvas_widget._image_pan_drag_start is None
    assert canvas_widget._image_pan_start_rect is None


# ---------------------------------------------------------------------------
# Canvas: drag math
# ---------------------------------------------------------------------------


def test_drag_translates_overlay_rect_by_mm_delta(canvas_widget, qtbot=None):
    """A 30-pixel rightward drag should translate the rect by 30 / zoom mm."""
    from PyQt6.QtCore import QPoint, QPointF, QEvent
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import Qt

    canvas_widget._image_overlay_rect_mm = (50.0, 60.0, 150.0, 160.0)
    canvas_widget.set_image_position_active(True)

    start_pos = QPointF(400.0, 300.0)
    moved_pos = QPointF(430.0, 305.0)

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, start_pos, start_pos,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas_widget.mousePressEvent(press)
    assert canvas_widget._image_pan_start_rect == (50.0, 60.0, 150.0, 160.0)

    move = QMouseEvent(
        QEvent.Type.MouseMove, moved_pos, moved_pos,
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    captured: list[tuple] = []
    canvas_widget.image_view_changed.connect(
        lambda x1, y1, x2, y2: captured.append((x1, y1, x2, y2))
    )
    canvas_widget.mouseMoveEvent(move)

    zoom = canvas_widget._zoom
    expected_dx = 30.0 / zoom
    expected_dy = 5.0 / zoom
    assert captured, "image_view_changed should fire during drag"
    x1, y1, x2, y2 = captured[-1]
    assert x1 == pytest.approx(50.0 + expected_dx)
    assert y1 == pytest.approx(60.0 + expected_dy)
    assert x2 == pytest.approx(150.0 + expected_dx)
    assert y2 == pytest.approx(160.0 + expected_dy)


# ---------------------------------------------------------------------------
# Canvas: wheel zoom about cursor
# ---------------------------------------------------------------------------


def test_wheel_zoom_keeps_cursor_mm_point_fixed(canvas_widget):
    """Zooming about the cursor must leave the mm point under it fixed."""
    from PyQt6.QtCore import QPoint, QPointF, QEvent
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtCore import Qt

    canvas_widget._image_overlay_rect_mm = (0.0, 0.0, 100.0, 100.0)
    canvas_widget.set_image_position_active(True)

    cursor = QPointF(450.0, 320.0)
    cursor_mm = canvas_widget.pixel_to_mm(cursor)

    event = QWheelEvent(
        cursor, cursor, QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    canvas_widget.wheelEvent(event)
    new_rect = canvas_widget._image_overlay_rect_mm
    assert new_rect is not None
    nx1, ny1, nx2, ny2 = new_rect
    scale = (nx2 - nx1) / 100.0
    # Wheel up = zoom in by factor 1.1 (matches the implementation).
    assert scale == pytest.approx(1.1, rel=1e-6)
    # Fixed-point identity: the cursor mm point is the centre of the scaling,
    # so it maps to itself regardless of whether it sits inside the rect.
    # For each old corner X: new_X = cursor + (X - cursor) * scale.
    assert nx1 == pytest.approx(cursor_mm[0] + (0.0 - cursor_mm[0]) * scale, abs=1e-6)
    assert ny1 == pytest.approx(cursor_mm[1] + (0.0 - cursor_mm[1]) * scale, abs=1e-6)
    assert nx2 == pytest.approx(cursor_mm[0] + (100.0 - cursor_mm[0]) * scale, abs=1e-6)
    assert ny2 == pytest.approx(cursor_mm[1] + (100.0 - cursor_mm[1]) * scale, abs=1e-6)


# ---------------------------------------------------------------------------
# Panel: toggle button behaviour
# ---------------------------------------------------------------------------


def _load_dummy_image(panel) -> None:
    """Drop a small numpy array straight into the panel so the rect emits."""
    panel._raw_image = np.zeros((100, 200, 3), dtype=np.uint8)
    panel._update_image_preview()


def test_position_button_disabled_without_image(panel):
    assert not panel._position_image_btn.isEnabled()


def test_position_button_enables_after_image_load(panel):
    _load_dummy_image(panel)
    assert panel._position_image_btn.isEnabled()


def test_position_toggle_requires_canvas_ref(panel):
    _load_dummy_image(panel)
    # No canvas wired yet → toggle should self-uncheck.
    panel._position_image_btn.setChecked(True)
    assert panel._position_image_btn.isChecked() is False


def _wire_panel_to_canvas(panel, canvas_widget) -> None:
    """Replicate the main_window glue that lets the panel's rect feed the canvas."""
    panel.set_canvas(canvas_widget)
    panel.image_rect_changed.connect(canvas_widget.set_image_overlay_rect)


def test_position_toggle_switches_to_custom_mode(panel, canvas_widget):
    _wire_panel_to_canvas(panel, canvas_widget)
    _load_dummy_image(panel)
    # Start in Fit mode.
    panel._image_fit_combo.setCurrentText("Fit (Keep Aspect)")
    assert panel._image_fit_mode() == "fit"
    # Toggle on.
    panel._position_image_btn.setChecked(True)
    assert panel._image_fit_combo.currentText() == "Custom Size"
    assert canvas_widget._image_position_active is True


def test_reset_position_restores_fit_mode(panel, canvas_widget):
    _wire_panel_to_canvas(panel, canvas_widget)
    _load_dummy_image(panel)
    panel._position_image_btn.setChecked(True)
    panel._image_offset_x_spin.setValue(15.0)
    panel._image_offset_y_spin.setValue(-22.0)

    panel._on_reset_image_position()
    assert panel._position_image_btn.isChecked() is False
    assert panel._image_fit_combo.currentText() == "Fit (Keep Aspect)"
    assert panel._image_offset_x_spin.value() == pytest.approx(0.0)
    assert panel._image_offset_y_spin.value() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Panel: rect → spinboxes + persistence
# ---------------------------------------------------------------------------


def test_canvas_view_changed_writes_spinboxes(panel, controller):
    _load_dummy_image(panel)
    # A4 portrait: width 210, height 297, margin 10 → drawing centre (105, 148.5).
    panel._on_canvas_image_view_changed(50.0, 60.0, 150.0, 200.0)
    # rect 50..150 by 60..200 → w=100, h=140, centre (100, 130)
    # offset = centre - draw_centre = (100 - 105, 130 - 148.5) = (-5, -18.5)
    assert panel._image_width_spin.value() == pytest.approx(100.0)
    assert panel._image_height_spin.value() == pytest.approx(140.0)
    assert panel._image_offset_x_spin.value() == pytest.approx(-5.0)
    assert panel._image_offset_y_spin.value() == pytest.approx(-18.5)


def test_canvas_view_changed_persists_metadata(panel, controller):
    _load_dummy_image(panel)
    panel._on_canvas_image_view_changed(50.0, 60.0, 150.0, 200.0)
    view = controller.current_project.metadata["image_view"]
    assert view["fit_mode"] == "custom"
    assert view["custom_w_mm"] == pytest.approx(100.0)
    assert view["custom_h_mm"] == pytest.approx(140.0)
    assert view["offset_x_mm"] == pytest.approx(-5.0)
    assert view["offset_y_mm"] == pytest.approx(-18.5)


def test_restore_image_view_from_metadata(panel, controller):
    controller.current_project.metadata["image_view"] = {
        "fit_mode": "custom",
        "custom_w_mm": 123.0,
        "custom_h_mm": 88.0,
        "offset_x_mm": 7.5,
        "offset_y_mm": -3.0,
    }
    panel._restore_image_view_from_metadata()
    assert panel._image_fit_combo.currentText() == "Custom Size"
    assert panel._image_width_spin.value() == pytest.approx(123.0)
    assert panel._image_height_spin.value() == pytest.approx(88.0)
    assert panel._image_offset_x_spin.value() == pytest.approx(7.5)
    assert panel._image_offset_y_spin.value() == pytest.approx(-3.0)


def test_restore_no_metadata_is_no_op(panel):
    """Restoration on a fresh project (no image_view key) must not raise."""
    panel._restore_image_view_from_metadata()
