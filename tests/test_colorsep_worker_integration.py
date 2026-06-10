"""End-to-end test that color separation runs through its worker thread.

The synchronous separators now run in a _SeparationWorker (QThread) and create
layers on the main thread via the finished signal. This exercises the real
worker + event loop + signal delivery, which the synchronous stub tests cannot.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _make_project():
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4")
    proj = Project(name="Test", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def panel(qapp, qtbot):
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel

    controller = ProjectController(_make_project())
    sp = SettingsPanel(controller)
    qtbot.addWidget(sp)
    sp.on_mode_changed("Color Separation")
    return sp


def _gradient(h=120, w=160):
    """RGB image with a left-to-right brightness ramp (good for Luminance)."""
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    return np.repeat(ramp[None, :], h, axis=0)[:, :, None].repeat(3, axis=2)


def test_luminance_separation_runs_through_worker(panel, qtbot):
    panel._raw_image = _gradient()
    panel._color_sep_method_combo.setCurrentText("Luminance")
    panel._color_sep_num_colors_spin.setValue(3)

    with patch("PyQt6.QtWidgets.QMessageBox.information"):
        panel._on_separate()
        # Worker runs off-thread; wait for the layers to be created.
        qtbot.waitUntil(lambda: len(panel._separated_layer_ids) == 3, timeout=10000)

    assert len(panel._separated_layer_ids) == 3
    # Each created layer has a cached (mask, preprocessed) pair the line
    # generators consume.
    for lid in panel._separated_layer_ids:
        assert lid in panel._layer_masks
        mask, preprocessed = panel._layer_masks[lid]
        assert mask.shape == preprocessed.shape[:2]


def test_downsample_caps_mask_resolution(panel, qtbot):
    # A >2 MP image must come back capped when downsampling is on (default).
    big = _gradient(h=2000, w=2000)  # 4 MP
    panel._raw_image = big
    panel._color_sep_method_combo.setCurrentText("Luminance")
    panel._color_sep_num_colors_spin.setValue(2)
    assert panel._downsample_check.isChecked()  # default on

    with patch("PyQt6.QtWidgets.QMessageBox.information"):
        panel._on_separate()
        qtbot.waitUntil(lambda: len(panel._separated_layer_ids) == 2, timeout=15000)

    mask, _pre = panel._layer_masks[panel._separated_layer_ids[0]]
    assert mask.shape[0] * mask.shape[1] <= 2_000_000
