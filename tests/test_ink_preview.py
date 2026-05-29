"""Tests for the canvas Ink Preview render mode.

Ink Preview switches the canvas to multiply-blended layer rendering so
stacked colour-separated layers combine subtractively (cyan + yellow →
green, full CMY → near black) — see ``CanvasWidget.set_ink_preview``.
The tests sample a pixel where multiple coloured layers overlap and
assert that the multiply mode darkens the result, since SourceOver
compositing would just show the top-most colour.
"""

from __future__ import annotations

import pytest
from PyQt6.QtGui import QColor, QImage

from plottter.gui.canvas_widget.widget import CanvasWidget
from plottter.gui.project_controller import ProjectController
from plottter.models import Canvas, Layer, Project


def _stacked_cmy_project() -> Project:
    """A project with three overlapping horizontal CMY strokes at y = 50 mm."""
    canvas = Canvas(width_mm=100, height_mm=100, margin_mm=0)
    path = [[(20.0, 50.0), (80.0, 50.0)]]
    return Project(
        name="t",
        canvas=canvas,
        layers=[
            Layer(id="C", name="Cyan",    color="#00FFFF", visible=True, locked=False, paths=path),
            Layer(id="M", name="Magenta", color="#FF00FF", visible=True, locked=False, paths=path),
            Layer(id="Y", name="Yellow",  color="#FFFF00", visible=True, locked=False, paths=path),
        ],
    )


def _render_centre(cw: CanvasWidget, qapp) -> tuple[int, int, int]:
    """Render the canvas off-screen and sample the centre pixel."""
    qapp.processEvents()
    img = QImage(cw.size(), QImage.Format.Format_ARGB32)
    cw.render(img)
    px = img.pixelColor(img.width() // 2, img.height() // 2)
    return px.red(), px.green(), px.blue()


@pytest.fixture
def canvas_widget(qapp):
    proj = _stacked_cmy_project()
    ctrl = ProjectController(proj)
    cw = CanvasWidget(ctrl)
    cw.resize(400, 400)
    cw.show()
    qapp.processEvents()
    yield cw
    cw.close()


def test_ink_preview_defaults_off(canvas_widget):
    assert canvas_widget._ink_preview is False


def test_set_ink_preview_flips_state(canvas_widget):
    canvas_widget.set_ink_preview(True)
    assert canvas_widget._ink_preview is True
    canvas_widget.set_ink_preview(False)
    assert canvas_widget._ink_preview is False


def test_ink_preview_darkens_stacked_cmy(canvas_widget, qapp):
    """With multiply blending, three stacked CMY strokes on white should
    combine to a noticeably darker pixel than SourceOver — which just
    leaves the topmost (yellow) layer visible at the overlap."""
    canvas_widget.set_ink_preview(False)
    r_n, g_n, b_n = _render_centre(canvas_widget, qapp)
    canvas_widget.set_ink_preview(True)
    r_i, g_i, b_i = _render_centre(canvas_widget, qapp)
    # Sum of channels falls when ink mixing happens
    assert (r_i + g_i + b_i) < (r_n + g_n + b_n), (
        f"ink-preview rgb sum {r_i + g_i + b_i} should be < normal sum {r_n + g_n + b_n}"
    )


def test_ink_preview_off_after_on_restores_top_layer_colour(canvas_widget, qapp):
    """Toggling Ink Preview off must return to standard SourceOver
    rendering (i.e. the topmost layer's colour wins on overlap)."""
    canvas_widget.set_ink_preview(True)
    _render_centre(canvas_widget, qapp)
    canvas_widget.set_ink_preview(False)
    r, g, b = _render_centre(canvas_widget, qapp)
    # The yellow layer (#FFFF00) is added last → its red+green channels
    # dominate, blue stays low.
    assert r > 200 and g > 150 and b < 200


# ---------------------------------------------------------------------------
# Preview pen width — display-only setting affecting stroke thickness
# ---------------------------------------------------------------------------


def test_preview_pen_width_defaults_to_fine_pen(canvas_widget):
    """0.3 mm matches the legacy hardcoded value and represents a fine pen."""
    assert canvas_widget.get_preview_pen_width_mm() == pytest.approx(0.3)


def test_set_preview_pen_width_clamps_to_safe_range(canvas_widget):
    """Sub-pixel values fall back to a pixel anyway; very wide values
    drown the path geometry.  Clamp to [0.05, 5.0] mm."""
    canvas_widget.set_preview_pen_width_mm(0.0)
    assert canvas_widget.get_preview_pen_width_mm() == pytest.approx(0.05)
    canvas_widget.set_preview_pen_width_mm(10.0)
    assert canvas_widget.get_preview_pen_width_mm() == pytest.approx(5.0)


def test_wider_pen_covers_more_pixels_around_a_path(canvas_widget, qapp):
    """A 1.2 mm marker preview must cover noticeably more pixels along a
    rendered path than the 0.3 mm default — quick proof the setting is
    actually being honoured by the painter."""
    def count_ink_pixels() -> int:
        qapp.processEvents()
        img = QImage(canvas_widget.size(), QImage.Format.Format_ARGB32)
        canvas_widget.render(img)
        n = 0
        for x in range(img.width()):
            # Sample a vertical column near the centre — should hit our
            # horizontal stroke at y = 50 mm and pick up extra pixels above
            # and below as the pen widens.
            for y in range(img.height()):
                if img.pixelColor(x, y) != QColor("white").rgba():
                    n += 1
        return n

    canvas_widget.set_preview_pen_width_mm(0.3)
    thin = count_ink_pixels()
    canvas_widget.set_preview_pen_width_mm(1.5)
    fat = count_ink_pixels()
    assert fat > thin, f"wide pen should ink more pixels than thin ({fat} vs {thin})"
