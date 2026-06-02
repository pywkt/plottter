"""Tests for the post-processing 'Clip to Canvas' toggle.

Covers:
- The checkbox exists and defaults to ON.
- When ON, _maybe_clip_to_canvas clips paths to the canvas drawing area.
- When OFF, _maybe_clip_to_canvas is a no-op.
- The choice persists to QSettings.
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
    proj = Project(name="ClipTest", canvas=canvas)
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


def test_clip_to_canvas_checkbox_exists(panel):
    assert hasattr(panel, "_clip_to_canvas_check")
    assert panel._clip_to_canvas_check.text() == "Clip to Canvas"


def test_clip_to_canvas_default_on(panel):
    """Default ON so first-time users don't get surprise out-of-canvas lines."""
    assert panel._clip_to_canvas_check.isChecked() is True


def test_maybe_clip_drops_out_of_canvas_paths(panel):
    """A path entirely outside the drawing area is dropped when clipping is on."""
    panel._clip_to_canvas_check.setChecked(True)
    # A4 portrait, 10mm margin → drawing area (10, 10, 200, 287).
    out_path = [(-50.0, -50.0), (-40.0, -40.0)]
    in_path = [(50.0, 50.0), (60.0, 60.0)]
    result = panel._maybe_clip_to_canvas([out_path, in_path])
    assert in_path in result
    assert not any(p == out_path for p in result)


def test_maybe_clip_splits_crossing_paths(panel):
    """A path that crosses the boundary is split, keeping only the inside segment."""
    panel._clip_to_canvas_check.setChecked(True)
    # Horizontal line from (-50, 50) to (300, 50) on A4 margin 10 (xmax=200).
    crossing = [(-50.0, 50.0), (300.0, 50.0)]
    result = panel._maybe_clip_to_canvas([crossing])
    assert len(result) == 1
    clipped = result[0]
    assert min(p[0] for p in clipped) >= 10.0 - 1e-6
    assert max(p[0] for p in clipped) <= 200.0 + 1e-6


def test_maybe_clip_off_is_identity(panel):
    """With the toggle off, paths are returned untouched (including out-of-canvas)."""
    panel._clip_to_canvas_check.setChecked(False)
    out_path = [(-50.0, -50.0), (-40.0, -40.0)]
    in_path = [(50.0, 50.0), (60.0, 60.0)]
    paths = [out_path, in_path]
    result = panel._maybe_clip_to_canvas(paths)
    assert result == paths


def test_clip_preference_persists_to_settings(panel):
    from PyQt6.QtCore import QSettings

    panel._clip_to_canvas_check.setChecked(False)
    saved = QSettings("Plottter", "Plottter").value(
        "generate/clip_to_canvas", True, type=bool
    )
    assert saved is False
    # Restore default for the next test in the module.
    panel._clip_to_canvas_check.setChecked(True)
