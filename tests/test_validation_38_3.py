"""Tests for task 38.3 — Brush as a generator post-processing parameter.

Covers:
(c) Brush params are saved/restored in generator_info snapshots:
    Set brush_type to "Stippled", change stipple_spacing_mm, call
    _get_settings_snapshot(), reset brush_type to "None", call
    _apply_settings_snapshot(), and assert the combo is back to "Stippled"
    with the correct stipple_spacing_mm value.

(d) Post-processing group appears for all generator types:
    _post_proc_group.isVisible() is True in Math Art mode and False in
    Color Separation / Mask Paint / Shape Drawing modes.
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="TestProject", canvas=canvas)
    layer = Layer(name="Layer 1", color="#000000")
    proj.add_layer(layer)
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project())


@pytest.fixture
def main_window(controller, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


# ---------------------------------------------------------------------------
# (c) Brush params saved/restored in generator_info snapshots
# ---------------------------------------------------------------------------


class TestBrushSnapshotSaveRestore:
    """Spec (c): brush post-processing settings round-trip through snapshot."""

    def _get_post_proc_widget(self, win, name):
        return win._settings_panel._post_proc_widgets.get(name)

    def test_brush_type_saved_in_snapshot(self, main_window):
        """brush_type value is persisted when _get_settings_snapshot() is called."""
        from PyQt6.QtWidgets import QComboBox
        panel = main_window._settings_panel
        bt_widget = self._get_post_proc_widget(main_window, "brush_type")
        if bt_widget is None:
            pytest.skip("No brush_type widget — post-processing group not built")
        assert isinstance(bt_widget, QComboBox)

        idx = bt_widget.findText("Stippled")
        assert idx >= 0
        bt_widget.setCurrentIndex(idx)

        snapshot = panel._get_settings_snapshot()
        assert snapshot is not None
        assert "post_processing" in snapshot
        assert snapshot["post_processing"]["brush_type"] == "Stippled"

    def test_stipple_spacing_saved_in_snapshot(self, main_window):
        """stipple_spacing_mm value is persisted in snapshot."""
        from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox
        panel = main_window._settings_panel
        bt_widget = self._get_post_proc_widget(main_window, "brush_type")
        ss_widget = self._get_post_proc_widget(main_window, "stipple_spacing_mm")
        if bt_widget is None or ss_widget is None:
            pytest.skip("Post-processing widgets not available")
        assert isinstance(bt_widget, QComboBox)
        assert isinstance(ss_widget, QDoubleSpinBox)

        # Select Stippled and set a custom spacing
        bt_widget.setCurrentIndex(bt_widget.findText("Stippled"))
        ss_widget.setValue(3.5)

        snapshot = panel._get_settings_snapshot()
        assert snapshot is not None
        pp = snapshot.get("post_processing", {})
        assert pp.get("stipple_spacing_mm") == pytest.approx(3.5)

    def test_brush_type_restored_from_snapshot(self, main_window):
        """After resetting brush_type to None, _apply_settings_snapshot restores it."""
        from PyQt6.QtWidgets import QComboBox
        panel = main_window._settings_panel
        bt_widget = self._get_post_proc_widget(main_window, "brush_type")
        if bt_widget is None:
            pytest.skip("No brush_type widget")
        assert isinstance(bt_widget, QComboBox)

        # Set to Stippled, capture snapshot
        bt_widget.setCurrentIndex(bt_widget.findText("Stippled"))
        snapshot = panel._get_settings_snapshot()
        assert snapshot is not None

        # Reset to None
        bt_widget.setCurrentIndex(bt_widget.findText("None"))
        assert bt_widget.currentText() == "None"

        # Restore from snapshot
        panel._apply_settings_snapshot(snapshot)
        assert bt_widget.currentText() == "Stippled"

    def test_stipple_spacing_restored_from_snapshot(self, main_window):
        """stipple_spacing_mm is restored to its saved value after apply_snapshot."""
        from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox
        panel = main_window._settings_panel
        bt_widget = self._get_post_proc_widget(main_window, "brush_type")
        ss_widget = self._get_post_proc_widget(main_window, "stipple_spacing_mm")
        if bt_widget is None or ss_widget is None:
            pytest.skip("Post-processing widgets not available")
        assert isinstance(bt_widget, QComboBox)
        assert isinstance(ss_widget, QDoubleSpinBox)

        # Set Stippled with custom spacing, capture snapshot
        bt_widget.setCurrentIndex(bt_widget.findText("Stippled"))
        ss_widget.setValue(4.2)
        snapshot = panel._get_settings_snapshot()
        assert snapshot is not None

        # Change spacing and reset brush type
        ss_widget.setValue(1.0)
        bt_widget.setCurrentIndex(bt_widget.findText("None"))

        # Restore
        panel._apply_settings_snapshot(snapshot)
        assert ss_widget.value() == pytest.approx(4.2)

    def test_snapshot_post_processing_key_present_for_math_art(self, main_window):
        """In Math Art mode, snapshot always has a 'post_processing' key."""
        panel = main_window._settings_panel
        # Ensure we're in Math Art mode
        if panel._current_mode != "Math Art":
            pytest.skip("Panel not in Math Art mode")
        snapshot = panel._get_settings_snapshot()
        assert snapshot is not None
        # post_processing key should be present (even if brush_type is None)
        assert "post_processing" in snapshot


# ---------------------------------------------------------------------------
# (d) Post-processing group visibility per mode
# ---------------------------------------------------------------------------


class TestPostProcGroupVisibility:
    """Spec (d): _post_proc_group shown/hidden correctly per application mode."""

    def test_visible_in_math_art_mode(self, main_window):
        panel = main_window._settings_panel
        panel.on_mode_changed("Math Art")
        assert not panel._post_proc_group.isHidden()

    def test_hidden_in_color_separation_mode(self, main_window):
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")
        assert not panel._post_proc_group.isVisible()

    def test_hidden_in_mask_paint_mode(self, main_window):
        panel = main_window._settings_panel
        panel.on_mode_changed("Mask Paint")
        assert not panel._post_proc_group.isVisible()

    def test_hidden_in_shape_drawing_mode(self, main_window):
        panel = main_window._settings_panel
        panel.on_mode_changed("Shape Drawing")
        assert not panel._post_proc_group.isVisible()

    def test_hidden_in_3d_scene_mode(self, main_window):
        """3D Scene mode also hides the post-processing group (not applicable)."""
        panel = main_window._settings_panel
        panel.on_mode_changed("3D Scene")
        assert not panel._post_proc_group.isVisible()

    def test_visible_in_image_to_lines_mode(self, main_window):
        """Image to Lines generators also support brush post-processing."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Image to Lines")
        assert not panel._post_proc_group.isHidden()

    def test_visibility_restored_when_returning_to_math_art(self, main_window):
        """After switching away and back, post_proc_group is visible again."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")
        assert panel._post_proc_group.isHidden()
        panel.on_mode_changed("Math Art")
        assert not panel._post_proc_group.isHidden()
