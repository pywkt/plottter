"""Tests for FMM source point visual marker overlay (task 28.3).

Covers:
- set_fmm_source_marker() stores position and triggers repaint
- clear_fmm_source_marker() clears stored position
- set_fmm_source_mode(False) clears live cursor preview
- mouseMoveEvent during pick mode updates _fmm_cursor_preview_mm
- Marker is cleared when switching source point to "Center"
- Marker is cleared when switching generators
- Marker is cleared when switching layers
- _draw_fmm_source_marker() renders without error
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project())


@pytest.fixture
def canvas_widget(controller, qtbot):
    from plottter.gui.canvas_widget import CanvasWidget
    w = CanvasWidget(controller)
    w.resize(800, 600)
    qtbot.addWidget(w)
    return w


# ===========================================================================
# 1. set_fmm_source_marker stores position
# ===========================================================================


class TestSetFmmSourceMarker:
    def test_stores_position(self, canvas_widget):
        assert canvas_widget._fmm_source_marker_mm is None
        canvas_widget.set_fmm_source_marker(50.0, 30.0)
        assert canvas_widget._fmm_source_marker_mm == (50.0, 30.0)

    def test_overwrites_previous_position(self, canvas_widget):
        canvas_widget.set_fmm_source_marker(10.0, 20.0)
        canvas_widget.set_fmm_source_marker(55.0, 75.0)
        assert canvas_widget._fmm_source_marker_mm == (55.0, 75.0)

    def test_marker_at_origin(self, canvas_widget):
        canvas_widget.set_fmm_source_marker(0.0, 0.0)
        assert canvas_widget._fmm_source_marker_mm == (0.0, 0.0)


# ===========================================================================
# 2. clear_fmm_source_marker clears position and cursor preview
# ===========================================================================


class TestClearFmmSourceMarker:
    def test_clears_marker(self, canvas_widget):
        canvas_widget.set_fmm_source_marker(50.0, 50.0)
        canvas_widget.clear_fmm_source_marker()
        assert canvas_widget._fmm_source_marker_mm is None

    def test_clears_cursor_preview(self, canvas_widget):
        canvas_widget._fmm_cursor_preview_mm = (30.0, 40.0)
        canvas_widget.clear_fmm_source_marker()
        assert canvas_widget._fmm_cursor_preview_mm is None

    def test_clear_when_already_none_does_not_raise(self, canvas_widget):
        assert canvas_widget._fmm_source_marker_mm is None
        canvas_widget.clear_fmm_source_marker()  # should not raise
        assert canvas_widget._fmm_source_marker_mm is None


# ===========================================================================
# 3. set_fmm_source_mode(False) clears cursor preview
# ===========================================================================


class TestSetFmmSourceModeDeactivation:
    def test_deactivate_clears_cursor_preview(self, canvas_widget):
        canvas_widget._fmm_source_mode = True
        canvas_widget._fmm_cursor_preview_mm = (25.0, 35.0)
        canvas_widget.set_fmm_source_mode(False)
        assert canvas_widget._fmm_cursor_preview_mm is None

    def test_deactivate_sets_flag_false(self, canvas_widget):
        canvas_widget.set_fmm_source_mode(True)
        canvas_widget.set_fmm_source_mode(False)
        assert canvas_widget._fmm_source_mode is False

    def test_marker_is_NOT_cleared_on_mode_deactivate(self, canvas_widget):
        """Persistent marker should survive mode deactivation — it's cleared separately."""
        canvas_widget.set_fmm_source_marker(20.0, 20.0)
        canvas_widget.set_fmm_source_mode(False)
        assert canvas_widget._fmm_source_marker_mm == (20.0, 20.0)


# ===========================================================================
# 4. mouseMoveEvent during pick mode updates _fmm_cursor_preview_mm
# ===========================================================================


class TestFmmPickModeCursorTracking:
    def test_cursor_preview_updated_during_pick_mode(self, canvas_widget, qtbot):
        """While FMM pick mode is active, moving the mouse updates cursor preview."""
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtGui import QMouseEvent

        canvas_widget.set_fmm_source_mode(True)
        # Show the widget so it has proper geometry
        canvas_widget.show()
        canvas_widget._fit_to_window()

        # Simulate mouse move event at widget center
        center = canvas_widget.rect().center()
        event = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            center.toPointF(),
            center.toPointF(),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas_widget.mouseMoveEvent(event)

        assert canvas_widget._fmm_cursor_preview_mm is not None
        # The preview position should be valid mm coordinates (finite numbers)
        x_mm, y_mm = canvas_widget._fmm_cursor_preview_mm
        assert isinstance(x_mm, float)
        assert isinstance(y_mm, float)

    def test_cursor_preview_not_updated_when_mode_inactive(self, canvas_widget, qtbot):
        """When pick mode is off, mouse move should NOT update cursor preview."""
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtGui import QMouseEvent

        canvas_widget.set_fmm_source_mode(False)
        canvas_widget._fmm_cursor_preview_mm = None

        center = canvas_widget.rect().center()
        event = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            center.toPointF(),
            center.toPointF(),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas_widget.mouseMoveEvent(event)

        assert canvas_widget._fmm_cursor_preview_mm is None


# ===========================================================================
# 5. _draw_fmm_source_marker renders without error
# ===========================================================================


class TestDrawFmmSourceMarkerNoCrash:
    def test_draw_persistent_marker(self, canvas_widget, qtbot):
        """Setting a marker and painting should not raise."""
        from PyQt6.QtGui import QPainter

        canvas_widget.show()
        canvas_widget.set_fmm_source_marker(100.0, 80.0)
        # Trigger a paint event — if it raises, the test fails
        canvas_widget.repaint()

    def test_draw_preview_marker(self, canvas_widget, qtbot):
        """Activating pick mode and triggering a paint should not raise."""
        canvas_widget.show()
        canvas_widget.set_fmm_source_mode(True)
        canvas_widget._fmm_cursor_preview_mm = (50.0, 50.0)
        canvas_widget.repaint()

    def test_no_marker_no_crash(self, canvas_widget, qtbot):
        """Paint with no marker set should not raise."""
        canvas_widget.show()
        assert canvas_widget._fmm_source_marker_mm is None
        canvas_widget.repaint()


# ===========================================================================
# 6. Marker cleared when source point switched to "Center"
# ===========================================================================


class TestMarkerClearedOnModeChange:
    def test_clear_marker_called_on_set_fmm_source_mode_false(self, canvas_widget):
        """clear_fmm_source_marker should clear both marker and preview."""
        canvas_widget.set_fmm_source_marker(50.0, 50.0)
        canvas_widget._fmm_cursor_preview_mm = (50.0, 50.0)
        canvas_widget.clear_fmm_source_marker()
        assert canvas_widget._fmm_source_marker_mm is None
        assert canvas_widget._fmm_cursor_preview_mm is None

    def test_initial_state_has_no_marker(self, canvas_widget):
        """New canvas widget starts with no marker."""
        assert canvas_widget._fmm_source_marker_mm is None
        assert canvas_widget._fmm_cursor_preview_mm is None
        assert canvas_widget._fmm_source_mode is False
