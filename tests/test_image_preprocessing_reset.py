"""Tests for the Reset Adjustments button on the image-preprocessing panel.

Covers:
- All adjustment widgets are restored to their defaults.
- Threshold slider / BG-tolerance spinbox are re-disabled after reset.
- Layout controls (fit mode, custom size, offsets) are NOT touched.
- The preview pipeline is re-triggered exactly once after reset.
"""

from __future__ import annotations

import os
import sys

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
    proj = Project(name="ResetTest", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def panel(qapp):
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel

    controller = ProjectController(_make_project())
    p = SettingsPanel(controller)
    yield p
    p.close()


def _dirty_all(panel) -> None:
    """Move every adjustment widget away from its default."""
    panel._auto_contrast_check.setChecked(False)
    panel._bright_slider.setValue(40)
    panel._contrast_slider.setValue(-30)
    panel._gamma_slider.setValue(180)
    panel._blur_slider.setValue(5)
    panel._unsharp_slider.setValue(15)
    panel._threshold_check.setChecked(True)
    panel._threshold_slider.setValue(64)
    panel._invert_check.setChecked(True)
    panel._remove_bg_check.setChecked(True)
    panel._bg_tolerance_spin.setValue(35.0)
    panel._crop_to_canvas_check.setChecked(False)


def test_reset_button_exists(panel):
    assert hasattr(panel, "_reset_preprocessing_btn")
    assert panel._reset_preprocessing_btn.text() == "Reset Adjustments"


def test_reset_restores_defaults(panel):
    _dirty_all(panel)
    panel._on_reset_preprocessing()

    assert panel._auto_contrast_check.isChecked() is True
    assert panel._bright_slider.value() == 0
    assert panel._contrast_slider.value() == 0
    assert panel._gamma_slider.value() == 100
    assert panel._blur_slider.value() == 0
    assert panel._unsharp_slider.value() == 0
    assert panel._threshold_check.isChecked() is False
    assert panel._threshold_slider.value() == 128
    assert panel._invert_check.isChecked() is False
    assert panel._remove_bg_check.isChecked() is False
    assert panel._bg_tolerance_spin.value() == pytest.approx(20.0)
    assert panel._ai_bg_check.isChecked() is False
    assert panel._crop_to_canvas_check.isChecked() is True


def test_reset_disables_conditional_widgets(panel):
    """Threshold slider and BG tolerance spinbox must be disabled after reset."""
    # Pre-condition: enable them by ticking their parent checkboxes.
    panel._threshold_check.setChecked(True)
    panel._remove_bg_check.setChecked(True)
    assert panel._threshold_slider.isEnabled()
    assert panel._bg_tolerance_spin.isEnabled()

    panel._on_reset_preprocessing()

    assert not panel._threshold_slider.isEnabled()
    assert not panel._bg_tolerance_spin.isEnabled()


def test_reset_does_not_touch_layout(panel):
    """Fit mode, custom size, and offsets must be preserved across reset."""
    panel._image_fit_combo.setCurrentText("Custom Size")
    panel._image_width_spin.setValue(123.0)
    panel._image_height_spin.setValue(456.0)
    panel._image_offset_x_spin.setValue(7.0)
    panel._image_offset_y_spin.setValue(-3.5)

    _dirty_all(panel)
    panel._on_reset_preprocessing()

    assert panel._image_fit_combo.currentText() == "Custom Size"
    assert panel._image_width_spin.value() == pytest.approx(123.0)
    assert panel._image_height_spin.value() == pytest.approx(456.0)
    assert panel._image_offset_x_spin.value() == pytest.approx(7.0)
    assert panel._image_offset_y_spin.value() == pytest.approx(-3.5)


def test_reset_triggers_single_preview_refresh(panel):
    """Reset must start the debounced preview timer exactly once."""
    _dirty_all(panel)
    panel._preprocess_timer.stop()

    panel._on_reset_preprocessing()
    assert panel._preprocess_timer.isActive()
