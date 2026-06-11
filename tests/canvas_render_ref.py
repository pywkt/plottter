"""Render-equivalence harness for the canvas rewrite (spec §5.3).

Standalone *reference renderer* that preserves the legacy CanvasWidget layer
rendering algorithm (per-point mm→px conversion, per-segment ``drawLine``,
pen width ``max(0.5, zoom * width_mm)``, SquareCap default with a
RoundCap + width override for ``generator_info["dot_diameter_mm"]`` layers,
alpha = layer opacity) as an executable oracle. Once the production paint
path is replaced by cached ``QPainterPath`` rendering, equivalence tests
compare its output against this module.

The reference covers the default static view only: background, paper rect
with black border, dashed margin rect, then visible layer paths. Grid,
registration marks, travel lines, overlays, ink preview, animation, and
jitter are all assumed off — the fixture scene disables them.

Fidelity sources — the exact production code mirrored here (as of Phase
164, before the cache rewrite replaces it):

- ``CanvasWidget.mm_to_pixel`` (gui/canvas_widget/widget.py:812-815):
  ``px = mm * zoom + pan_offset``.
- ``_PaintingMixin._draw_layer`` (gui/canvas_widget/_painting.py:410-464):
  ``color.setAlphaF(layer.opacity)`` (ink preview off);
  ``width_mm = self._preview_pen_width_mm`` (default 0.3, widget.py:176)
  with ``cap = SquareCap``, overridden to
  ``width_mm = float(generator_info["dot_diameter_mm"])`` + ``RoundCap``
  when that hint is a number > 0;
  ``pen = QPen(color, max(0.5, self._zoom * width_mm))``;
  per-path min/max bbox cull against the viewport in mm; then
  ``pts = [mm_to_pixel(p) for p in polyline]`` (jitter off) and one
  ``drawLine(pts[i], pts[i + 1])`` per segment.
- ``paintEvent`` chrome (_painting.py:28-61): fill ``#808080``, white
  paper rect + ``QPen(black, 1.0)`` border, ``QPen("#AAAAAA", 0.5,
  DashLine)`` margin rect, Antialiasing render hint.

Not a pytest test module (no ``test_`` prefix); see
``tests/test_canvas_bench_harness.py`` for the self-tests.
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen

from plottter.models import Canvas, Layer, Project

# Default of CanvasWidget._preview_pen_width_mm (widget.py).
DEFAULT_PREVIEW_PEN_WIDTH_MM = 0.3


def render_reference(
    project: Project,
    zoom: float,
    pan: tuple[float, float],
    size: tuple[int, int],
    preview_pen_width_mm: float = DEFAULT_PREVIEW_PEN_WIDTH_MM,
) -> QImage:
    """Render *project* with the legacy algorithm into a new QImage.

    ``pan`` is the pixel pan offset and ``size`` the viewport size in px,
    mirroring ``CanvasWidget._pan_offset`` / widget size. The mm→px mapping
    is ``px = mm * zoom + pan`` exactly as ``CanvasWidget.mm_to_pixel``.
    """
    width_px, height_px = size
    pan_x, pan_y = pan
    img = QImage(width_px, height_px, QImage.Format.Format_ARGB32)
    img.fill(QColor("#808080"))  # paintEvent background fill

    def mm_to_px(pt: tuple[float, float]) -> QPointF:
        return QPointF(pt[0] * zoom + pan_x, pt[1] * zoom + pan_y)

    painter = QPainter(img)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        canvas = project.canvas

        # Paper boundary (white, 1px black border)
        paper_rect = QRectF(
            mm_to_px((0.0, 0.0)), mm_to_px((canvas.width_mm, canvas.height_mm))
        )
        painter.fillRect(paper_rect, QColor("white"))
        painter.setPen(QPen(QColor("black"), 1.0))
        painter.drawRect(paper_rect)

        # Margin boundary (dashed gray)
        m = canvas.margin_mm
        margin_rect = QRectF(
            mm_to_px((m, m)),
            mm_to_px((canvas.width_mm - m, canvas.height_mm - m)),
        )
        painter.setPen(QPen(QColor("#AAAAAA"), 0.5, Qt.PenStyle.DashLine))
        painter.drawRect(margin_rect)

        # Viewport bounds in mm for bbox culling (legacy per-path cull)
        vp_left = (0.0 - pan_x) / zoom
        vp_top = (0.0 - pan_y) / zoom
        vp_right = (width_px - pan_x) / zoom
        vp_bottom = (height_px - pan_y) / zoom

        for layer in project.layers:
            if not layer.visible:
                continue
            _draw_layer_reference(
                painter,
                layer,
                zoom,
                mm_to_px,
                (vp_left, vp_top, vp_right, vp_bottom),
                preview_pen_width_mm,
            )
    finally:
        painter.end()
    return img


def _draw_layer_reference(
    painter: QPainter,
    layer: Layer,
    zoom: float,
    mm_to_px,
    viewport_mm: tuple[float, float, float, float],
    preview_pen_width_mm: float,
) -> None:
    """Legacy ``_draw_layer``: per-point convert, per-segment drawLine."""
    color = QColor(layer.color)
    color.setAlphaF(layer.opacity)
    width_mm = preview_pen_width_mm
    cap = Qt.PenCapStyle.SquareCap
    gen_info = layer.generator_info
    if isinstance(gen_info, dict):
        dot_dia = gen_info.get("dot_diameter_mm")
        if isinstance(dot_dia, (int, float)) and dot_dia > 0.0:
            width_mm = float(dot_dia)
            cap = Qt.PenCapStyle.RoundCap
    pen = QPen(color, max(0.5, zoom * width_mm))
    pen.setCapStyle(cap)
    painter.setPen(pen)

    vp_left, vp_top, vp_right, vp_bottom = viewport_mm
    for polyline in layer.paths:
        if len(polyline) < 2:
            continue
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for px, py in polyline:
            if px < min_x:
                min_x = px
            if px > max_x:
                max_x = px
            if py < min_y:
                min_y = py
            if py > max_y:
                max_y = py
        if max_x < vp_left or min_x > vp_right or max_y < vp_top or min_y > vp_bottom:
            continue
        pts = [mm_to_px(p) for p in polyline]
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])


def _image_to_array(img: QImage) -> np.ndarray:
    """Return an (h, w, 4) uint8 view of *img* in ARGB32 byte order."""
    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    h, w = img.height(), img.width()
    ptr = img.constBits()
    ptr.setsize(h * img.bytesPerLine())
    flat = np.frombuffer(ptr, dtype=np.uint8).reshape(h, img.bytesPerLine())
    return flat[:, : w * 4].reshape(h, w, 4).copy()


def pixel_diff_ratio(img_a: QImage, img_b: QImage) -> float:
    """Fraction of pixels where any channel differs by more than 16/255."""
    if img_a.size() != img_b.size():
        raise ValueError(
            f"image sizes differ: {img_a.width()}x{img_a.height()} vs "
            f"{img_b.width()}x{img_b.height()}"
        )
    a = _image_to_array(img_a).astype(np.int16)
    b = _image_to_array(img_b).astype(np.int16)
    differs = (np.abs(a - b) > 16).any(axis=2)
    return float(differs.mean())


def make_fixture_project() -> Project:
    """Deterministic 3-layer scene for equivalence tests (spec §5.3).

    100×100 mm canvas, registration marks off (so the chrome drawn by
    paintEvent reduces to paper + margin, both replicated here). Layers:
    ordinary black line work, a Pointillist-style dots layer (0.01 mm
    segments + ``dot_diameter_mm`` render hint), and a 50%-opacity layer
    overlapping the first.
    """
    canvas = Canvas(width_mm=100.0, height_mm=100.0, margin_mm=10.0)
    project = Project(name="RenderRefFixture", canvas=canvas, registration_marks=False)

    lines = Layer(name="lines", color="#000000")
    lines.paths = [
        [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0), (20.0, 20.0)],
        [(20.0, 20.0), (80.0, 80.0)],
        [(20.0, 70.0), (30.0, 50.0), (40.0, 70.0), (50.0, 50.0), (60.0, 70.0), (70.0, 50.0)],
    ]
    project.add_layer(lines)

    dots = Layer(
        name="dots",
        color="#CC2222",
        generator_info={"dot_diameter_mm": 0.6},
    )
    dots.paths = [
        [(float(x), float(y)), (float(x) + 0.01, float(y))]
        for x in range(25, 80, 10)
        for y in range(25, 80, 10)
    ]
    project.add_layer(dots)

    half = Layer(name="half", color="#2244CC", opacity=0.5)
    half.paths = [
        [(15.0, float(y)), (85.0, float(y))] for y in range(30, 75, 8)
    ]
    project.add_layer(half)

    return project
