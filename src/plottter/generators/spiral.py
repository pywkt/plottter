"""SpiralGenerator — Archimedean spiral tracing from image center outward."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _trace_spiral(
    center_x_mm: float,
    center_y_mm: float,
    ring_spacing_mm: float,
    max_radius_mm: float,
    step_size_mm: float,
) -> list[tuple[float, float, float]]:
    """Trace an Archimedean spiral outward from the center.

    The spiral follows ``r = ring_spacing * theta / (2*pi)`` so that the
    radial distance between consecutive rings equals ``ring_spacing_mm``.

    Points are spaced by constant arc length ``step_size_mm`` using the
    approximation ``delta_theta = step_size / r``.

    To avoid division by zero at the origin, the trace starts at the first
    ``theta`` where ``r >= step_size_mm``.

    Parameters
    ----------
    center_x_mm, center_y_mm:
        Spiral center in canvas mm coordinates.
    ring_spacing_mm:
        Radial distance between successive rings (mm).
    max_radius_mm:
        Stop tracing when radius exceeds this value.
    step_size_mm:
        Arc-length distance between consecutive sample points (mm).

    Returns
    -------
    List of ``(x_mm, y_mm, theta)`` tuples.
    """
    two_pi = 2.0 * math.pi

    # Start at theta where r == step_size_mm to avoid division by zero at r=0
    theta_start = step_size_mm * two_pi / ring_spacing_mm

    points: list[tuple[float, float, float]] = []
    theta = theta_start

    while True:
        r = ring_spacing_mm * theta / two_pi
        if r > max_radius_mm:
            break
        x = center_x_mm + r * math.cos(theta)
        y = center_y_mm + r * math.sin(theta)
        points.append((x, y, theta))
        # Constant arc-length step: delta_theta = step_size / r
        delta_theta = step_size_mm / r
        theta += delta_theta

    return points


@register_generator
class SpiralGenerator(Generator):
    """Image-driven Archimedean spiral generator.

    Traces a single continuous spiral from the center outward, covering
    the image area out to the farthest corner.  Future tasks will modulate
    spiral displacement by local image brightness.
    """

    name = "Spiral"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="ring_spacing_mm",
                label="Ring Spacing (mm)",
                min=1.0,
                max=15.0,
                step=0.1,
                default=3.0,
                description="Distance between spiral rings (mm)",
            ),
            FloatParam(
                name="center_x_pct",
                label="Center X (%)",
                min=0.0,
                max=100.0,
                step=0.5,
                default=50.0,
                description="Spiral center X position (% of image width)",
            ),
            FloatParam(
                name="center_y_pct",
                label="Center Y (%)",
                min=0.0,
                max=100.0,
                step=0.5,
                default=50.0,
                description="Spiral center Y position (% of image height)",
            ),
            FloatParam(
                name="step_size_mm",
                label="Step Size (mm)",
                min=0.1,
                max=2.0,
                step=0.05,
                default=0.5,
                description="Distance between sample points along the spiral (mm)",
            ),
            # --- Standard image preprocessing params ---
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image before processing",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before processing (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before processing (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description="Gaussian blur radius applied before processing",
            ),
            # --- Image fit / placement params ---
            ChoiceParam(
                name="image_fit_mode",
                label="Image Fit",
                choices=["fill", "fit", "custom"],
                default="fill",
                description="How to map the source image onto the canvas",
                randomizable=False,
            ),
            FloatParam(
                name="image_width_mm",
                label="Image Width (mm)",
                min=1.0,
                max=2000.0,
                step=1.0,
                default=200.0,
                visible_when={"image_fit_mode": ["custom"]},
                randomizable=False,
                description="Output image width in mm (custom fit mode only)",
            ),
            FloatParam(
                name="image_height_mm",
                label="Image Height (mm)",
                min=1.0,
                max=2000.0,
                step=1.0,
                default=200.0,
                visible_when={"image_fit_mode": ["custom"]},
                randomizable=False,
                description="Output image height in mm (custom fit mode only)",
            ),
            FloatParam(
                name="image_offset_x_mm",
                label="Image Offset X (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                visible_when={"image_fit_mode": ["fit", "custom"]},
                randomizable=False,
                description="Horizontal offset from centered position (fit/custom mode, mm)",
            ),
            FloatParam(
                name="image_offset_y_mm",
                label="Image Offset Y (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                visible_when={"image_fit_mode": ["fit", "custom"]},
                randomizable=False,
                description="Vertical offset from centered position (fit/custom mode, mm)",
            ),
            # --- Output placement params ---
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the generated output on the canvas (mm)",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the generated output on the canvas (mm)",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default",
                params={
                    "ring_spacing_mm": 3.0,
                    "center_x_pct": 50.0,
                    "center_y_pct": 50.0,
                    "step_size_mm": 0.5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "image_fit_mode": "fill",
                    "image_width_mm": 200.0,
                    "image_height_mm": 200.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Fine Detail",
                params={
                    "ring_spacing_mm": 1.5,
                    "center_x_pct": 50.0,
                    "center_y_pct": 50.0,
                    "step_size_mm": 0.2,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.5,
                    "image_fit_mode": "fill",
                    "image_width_mm": 200.0,
                    "image_height_mm": 200.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Loose Coil",
                params={
                    "ring_spacing_mm": 8.0,
                    "center_x_pct": 50.0,
                    "center_y_pct": 50.0,
                    "step_size_mm": 1.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 2.0,
                    "image_fit_mode": "fill",
                    "image_width_mm": 200.0,
                    "image_height_mm": 200.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
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
        source: np.ndarray | None = params.get("_source_image")

        # Determine image dimensions for computing the image rect
        if source is not None:
            img_h, img_w = source.shape[:2]
        else:
            # No image loaded — fall back to canvas drawing area dimensions
            draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
            img_w = int(draw_x2 - draw_x1)
            img_h = int(draw_y2 - draw_y1)

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w,
            img_h,
            draw_x1,
            draw_y1,
            draw_x2,
            draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        # Compute spiral center from percentage parameters
        center_x_pct = float(params.get("center_x_pct", 50.0)) / 100.0
        center_y_pct = float(params.get("center_y_pct", 50.0)) / 100.0
        img_rect_w = img_x2 - img_x1
        img_rect_h = img_y2 - img_y1
        center_x_mm = img_x1 + center_x_pct * img_rect_w
        center_y_mm = img_y1 + center_y_pct * img_rect_h

        # Compute max_radius as the distance to the farthest corner of the image rect
        corners = [
            (img_x1, img_y1),
            (img_x2, img_y1),
            (img_x1, img_y2),
            (img_x2, img_y2),
        ]
        max_radius_mm = max(
            math.sqrt((cx - center_x_mm) ** 2 + (cy - center_y_mm) ** 2)
            for cx, cy in corners
        )

        ring_spacing_mm = float(params.get("ring_spacing_mm", 3.0))
        step_size_mm = float(params.get("step_size_mm", 0.5))

        # Guard against degenerate inputs
        if ring_spacing_mm <= 0.0 or step_size_mm <= 0.0 or max_radius_mm <= 0.0:
            return []

        if progress_callback:
            progress_callback(0)

        spiral_pts = _trace_spiral(
            center_x_mm,
            center_y_mm,
            ring_spacing_mm,
            max_radius_mm,
            step_size_mm,
        )

        if progress_callback:
            progress_callback(90)

        if not spiral_pts:
            return []

        # Build single polyline from spiral points (strip theta)
        polyline: Polyline = [(x, y) for x, y, _ in spiral_pts]

        # Apply output offset
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            polyline = [(x + x_off, y + y_off) for x, y in polyline]

        if progress_callback:
            progress_callback(100)

        return [polyline]
