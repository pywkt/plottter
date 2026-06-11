"""Self-tests for the render-equivalence harness (spec §5.3, §11).

Validates the oracle in ``tests/canvas_render_ref.py`` against today's
production paint path: while the legacy algorithm is still in place,
``CanvasWidget.render()`` and ``render_reference()`` implement the same
drawing code, so their output must agree within the equivalence tolerance.
"""
from __future__ import annotations

import pytest

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QImage

from tests.canvas_render_ref import (
    make_fixture_project,
    pixel_diff_ratio,
    render_reference,
)

SIZE = (400, 400)

# Two view states per spec §11: a fit-like view with the full canvas visible,
# and a zoomed-in view that pushes part of the scene off-screen (exercises
# the per-path viewport culling on both sides).
VIEW_STATES = [
    pytest.param(3.0, (50.0, 50.0), id="fit-view"),
    pytest.param(5.0, (-80.0, -60.0), id="zoomed-in"),
]


def _render_widget(qapp, project, zoom, pan, size) -> QImage:
    """Render *project* through the real CanvasWidget at a fixed view state."""
    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.gui.project_controller import ProjectController

    controller = ProjectController(project)
    widget = CanvasWidget(controller)
    widget.resize(*size)
    # Pin the view state directly; _fitted blocks the fit-on-show refit.
    widget._fitted = True
    widget._zoom = zoom
    widget._pan_offset = QPointF(*pan)

    img = QImage(*size, QImage.Format.Format_ARGB32)
    img.fill(0)  # start from defined pixels; paintEvent overwrites the full rect
    widget.render(img)
    return img


class TestReferenceRenderer:
    def test_fixture_render_is_non_blank(self, qapp):
        import numpy as np

        from tests.canvas_render_ref import _image_to_array

        img = render_reference(make_fixture_project(), 3.0, (50.0, 50.0), SIZE)
        arr = _image_to_array(img)
        # BGRA byte order; drop alpha for the color checks.
        rgb = arr[:, :, :3].astype(int)
        # Layer 1 line work: near-black pixels well beyond what the 1px paper
        # border alone would produce.
        dark = (rgb < 80).all(axis=2).sum()
        assert dark > 500, f"expected layer line work, found {dark} dark pixels"
        # Layer 2 dots are red-dominant (#CC2222, BGRA → channel 2 is red).
        red = ((arr[:, :, 2].astype(int) - rgb[:, :, 0]) > 80).sum()
        assert red > 50, f"expected red dot pixels, found {red}"

    def test_self_diff_is_zero(self, qapp):
        img = render_reference(make_fixture_project(), 3.0, (50.0, 50.0), SIZE)
        assert pixel_diff_ratio(img, img) == 0.0

    def test_diff_detects_changed_view(self, qapp):
        project = make_fixture_project()
        a = render_reference(project, 3.0, (50.0, 50.0), SIZE)
        b = render_reference(project, 3.5, (40.0, 40.0), SIZE)
        assert pixel_diff_ratio(a, b) > 0.01

    def test_diff_rejects_size_mismatch(self, qapp):
        project = make_fixture_project()
        a = render_reference(project, 3.0, (50.0, 50.0), SIZE)
        b = render_reference(project, 3.0, (50.0, 50.0), (200, 200))
        with pytest.raises(ValueError):
            pixel_diff_ratio(a, b)

    def test_reference_is_deterministic(self, qapp):
        project = make_fixture_project()
        a = render_reference(project, 3.0, (50.0, 50.0), SIZE)
        b = render_reference(project, 3.0, (50.0, 50.0), SIZE)
        assert pixel_diff_ratio(a, b) == 0.0


class TestWidgetEquivalence:
    """CanvasWidget.render() still runs the legacy algorithm — the oracle
    must match it within the §5.3 tolerance (validates the oracle itself)."""

    @pytest.mark.parametrize("zoom,pan", VIEW_STATES)
    def test_widget_matches_reference(self, qapp, zoom, pan):
        project = make_fixture_project()
        ref = render_reference(project, zoom, pan, SIZE)
        real = _render_widget(qapp, project, zoom, pan, SIZE)
        ratio = pixel_diff_ratio(real, ref)
        assert ratio <= 0.02, f"pixel diff ratio {ratio:.4f} exceeds 0.02"
