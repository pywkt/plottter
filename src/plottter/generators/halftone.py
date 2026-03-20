"""HalftoneGenerator — image-driven halftone with configurable grid layouts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    ImageParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


# ---------------------------------------------------------------------------
# Grid point generation
# ---------------------------------------------------------------------------

def _grid_square(
    spacing: float,
    angle: float,
    w: float,
    h: float,
) -> np.ndarray:
    """Regular square grid rotated by *angle* degrees, clipped to (0,0)–(w,h).

    Parameters
    ----------
    spacing:
        Distance between adjacent grid points in mm.
    angle:
        Grid rotation in degrees (counter-clockwise).
    w, h:
        Drawing area width and height in mm.

    Returns
    -------
    Array of shape (N, 2) with (x_mm, y_mm) grid points inside the area.
    """
    diag = math.hypot(w, h)
    n = int(math.ceil(diag / spacing)) + 2

    cols = np.arange(-n, n + 1, dtype=float)
    rows = np.arange(-n, n + 1, dtype=float)
    gx, gy = np.meshgrid(cols * spacing, rows * spacing)
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    # Rotate around origin then translate to canvas centre
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T
    pts[:, 0] += w / 2.0
    pts[:, 1] += h / 2.0

    mask = (pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)
    return pts[mask]


def _grid_hexagonal(
    spacing: float,
    angle: float,
    w: float,
    h: float,
) -> np.ndarray:
    """Offset hexagonal grid rotated by *angle* degrees, clipped to (0,0)–(w,h).

    Odd rows are shifted right by ``spacing / 2``; row spacing is
    ``spacing * √3 / 2``, producing a close-packed hexagonal arrangement.

    Parameters
    ----------
    spacing:
        Distance between adjacent grid points in mm.
    angle:
        Grid rotation in degrees (counter-clockwise).
    w, h:
        Drawing area width and height in mm.

    Returns
    -------
    Array of shape (N, 2) with (x_mm, y_mm) grid points inside the area.
    """
    row_h = spacing * math.sqrt(3.0) / 2.0
    diag = math.hypot(w, h)
    n_cols = int(math.ceil(diag / spacing)) + 2
    n_rows = int(math.ceil(diag / row_h)) + 2

    pts_list: list[tuple[float, float]] = []
    for row in range(-n_rows, n_rows + 1):
        x_off = (spacing / 2.0) if (row % 2 != 0) else 0.0
        for col in range(-n_cols, n_cols + 1):
            pts_list.append((col * spacing + x_off, row * row_h))

    pts = np.array(pts_list, dtype=float)

    # Rotate and translate
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T
    pts[:, 0] += w / 2.0
    pts[:, 1] += h / 2.0

    mask = (pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)
    return pts[mask]


def _grid_diagonal(spacing: float, w: float, h: float) -> np.ndarray:
    """Square grid rotated 45°, clipped to (0,0)–(w,h).

    Equivalent to ``_grid_square(spacing, 45, w, h)``.

    Parameters
    ----------
    spacing:
        Distance between adjacent grid points in mm.
    w, h:
        Drawing area width and height in mm.

    Returns
    -------
    Array of shape (N, 2) with (x_mm, y_mm) grid points inside the area.
    """
    return _grid_square(spacing, 45.0, w, h)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@register_generator
class HalftoneGenerator(Generator):
    """Image-driven halftone with square, hexagonal, or diagonal grid layouts."""

    name = "Dot Grid Halftone"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ImageParam(
                name="_source_image",
                label="Source Image",
                randomizable=False,
                description="Source image used to modulate halftone dot sizes.",
            ),
            FloatParam(
                name="grid_spacing_mm",
                label="Grid Spacing (mm)",
                min=0.5,
                max=20.0,
                step=0.1,
                default=3.0,
                description="Distance between dot centers in mm.",
            ),
            ChoiceParam(
                name="grid_type",
                label="Grid Type",
                choices=["Square", "Hexagonal", "Diagonal"],
                default="Square",
                description="Layout of the halftone grid.",
            ),
            FloatParam(
                name="grid_angle_deg",
                label="Grid Angle (°)",
                min=0.0,
                max=90.0,
                step=0.5,
                default=0.0,
                description="Grid rotation angle in degrees.",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Adjust source image brightness before halftoning.",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Adjust source image contrast before halftoning.",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Gaussian blur radius applied to the source image (px).",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                randomizable=False,
                description="Invert image tones before halftoning.",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the output on the canvas (mm).",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the output on the canvas (mm).",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Classic Halftone",
                params={
                    "grid_spacing_mm": 3.0,
                    "grid_type": "Square",
                    "grid_angle_deg": 45.0,
                },
            ),
            Preset(
                name="Hex Halftone",
                params={
                    "grid_spacing_mm": 3.0,
                    "grid_type": "Hexagonal",
                    "grid_angle_deg": 0.0,
                },
            ),
            Preset(
                name="Fine Diagonal",
                params={
                    "grid_spacing_mm": 1.5,
                    "grid_type": "Diagonal",
                    "grid_angle_deg": 0.0,
                },
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        # Extract parameters
        spacing = float(params.get("grid_spacing_mm", 3.0))
        grid_type = str(params.get("grid_type", "Square"))
        angle = float(params.get("grid_angle_deg", 0.0))
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        # Drawing area dimensions
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        # Load source image (not used yet — scaffold only)
        _source: Any = params.get("_source_image")  # noqa: F841 (used in future steps)

        # Generate grid points in drawing-area-local coordinates
        if grid_type == "Hexagonal":
            pts = _grid_hexagonal(spacing, angle, draw_w, draw_h)
        elif grid_type == "Diagonal":
            pts = _grid_diagonal(spacing, draw_w, draw_h)
        else:  # "Square" (default)
            pts = _grid_square(spacing, angle, draw_w, draw_h)

        # Translate to canvas coordinates (drawing area origin + user offset)
        if len(pts) > 0:
            pts[:, 0] += draw_x1 + x_off
            pts[:, 1] += draw_y1 + y_off

        if progress_callback:
            progress_callback(100)

        # Scaffold: grid points computed, dot polylines not yet generated
        return []
