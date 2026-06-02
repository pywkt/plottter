"""Tests for the auto-crop-to-visible-canvas optimization.

When the source image rect extends past the canvas drawing area, the panel
crops the image and rewrites the rect params so generators only iterate
over the visible portion. This avoids 16× slowdowns when the user zooms
the image to several times the canvas size.
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
    proj = Project(name="CropTest", canvas=canvas)
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


def _image(h: int = 100, w: int = 200) -> np.ndarray:
    """Recognisable gradient so crop bounds are visible if a test inspects pixels."""
    arr = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    return arr


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------


def test_no_crop_when_rect_inside_canvas(panel):
    """Fit (Keep Aspect) on a square image inside A4 → no crop needed."""
    img = _image(200, 200)
    params = {
        "image_fit_mode": "fit",
        "image_offset_x_mm": 0.0,
        "image_offset_y_mm": 0.0,
    }
    assert panel.compute_visible_image_crop(img, params) is None


def test_crop_when_rect_strictly_larger_than_canvas(panel):
    """Custom size 500×500 mm > 190×277 mm drawing area → crop fires."""
    img = _image(500, 500)
    params = {
        "image_fit_mode": "custom",
        "image_width_mm": 500.0,
        "image_height_mm": 500.0,
        "image_offset_x_mm": 0.0,
        "image_offset_y_mm": 0.0,
    }
    result = panel.compute_visible_image_crop(img, params)
    assert result is not None
    cropped, override = result
    # The crop must be strictly smaller than the source.
    assert cropped.shape[0] <= img.shape[0]
    assert cropped.shape[1] <= img.shape[1]
    assert cropped.size > 0
    # Override params switch to custom mode with the visible rect.
    assert override["image_fit_mode"] == "custom"
    # A4 margins: drawing area 190×277. With padding, slightly larger.
    assert override["image_width_mm"] >= 190.0 - 1e-3
    assert override["image_height_mm"] >= 277.0 - 1e-3
    # The original was centred at canvas centre, so the new rect should be
    # centred too (offset ≈ 0 ± padding fraction).
    assert abs(override["image_offset_x_mm"]) < 5.0
    assert abs(override["image_offset_y_mm"]) < 5.0


def test_crop_returns_none_when_rect_entirely_off_canvas(panel):
    """A 50×50 mm rect placed 1000 mm off-centre never touches the canvas."""
    img = _image(50, 50)
    params = {
        "image_fit_mode": "custom",
        "image_width_mm": 50.0,
        "image_height_mm": 50.0,
        "image_offset_x_mm": 1000.0,
        "image_offset_y_mm": 0.0,
    }
    assert panel.compute_visible_image_crop(img, params) is None


def test_crop_applies_padding(panel):
    """The 8-pixel halo means the crop is slightly larger than the strict visible region."""
    img = _image(1000, 1000)
    params = {
        "image_fit_mode": "custom",
        "image_width_mm": 1000.0,
        "image_height_mm": 1000.0,
        "image_offset_x_mm": 0.0,
        "image_offset_y_mm": 0.0,
    }
    result = panel.compute_visible_image_crop(img, params)
    assert result is not None
    cropped, _ = result
    # Visible region in pixels (without padding): (190/1000)*1000 × (277/1000)*1000 = 190×277.
    # With 8-pixel halo on each side, expect roughly 190+16 × 277+16 = 206×293 pixels.
    assert cropped.shape[1] >= 190 + 2 * panel._AUTO_CROP_PADDING_PX - 2
    assert cropped.shape[0] >= 277 + 2 * panel._AUTO_CROP_PADDING_PX - 2
    # And not much bigger than that (well below the full 1000×1000).
    assert cropped.shape[1] < 250
    assert cropped.shape[0] < 350


def test_crop_returns_none_for_empty_image(panel):
    assert panel.compute_visible_image_crop(None, {}) is None
    assert panel.compute_visible_image_crop(np.zeros((0, 0), dtype=np.uint8), {}) is None


def test_crop_compose_with_compute_image_rect(panel):
    """The override params must round-trip through compute_image_rect.

    Calling compute_image_rect on the cropped image + override params should
    yield the same mm rect that the cropped image actually occupies.
    """
    from plottter.generators._helpers import compute_image_rect
    img = _image(500, 500)
    params = {
        "image_fit_mode": "custom",
        "image_width_mm": 500.0,
        "image_height_mm": 500.0,
        "image_offset_x_mm": 0.0,
        "image_offset_y_mm": 0.0,
    }
    result = panel.compute_visible_image_crop(img, params)
    assert result is not None
    cropped, override = result
    canvas = panel._controller.current_project.canvas
    dx1, dy1, dx2, dy2 = canvas.drawing_area()
    rect = compute_image_rect(
        fit_mode=override["image_fit_mode"],
        image_w_px=cropped.shape[1],
        image_h_px=cropped.shape[0],
        draw_x1=dx1,
        draw_y1=dy1,
        draw_x2=dx2,
        draw_y2=dy2,
        custom_w_mm=override["image_width_mm"],
        custom_h_mm=override["image_height_mm"],
        offset_x_mm=override["image_offset_x_mm"],
        offset_y_mm=override["image_offset_y_mm"],
    )
    # The recomputed rect width/height should match the override.
    assert (rect[2] - rect[0]) == pytest.approx(override["image_width_mm"], abs=1e-3)
    assert (rect[3] - rect[1]) == pytest.approx(override["image_height_mm"], abs=1e-3)
