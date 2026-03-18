"""PolarGenerator — evaluates r(theta) and converts to Cartesian coordinates."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from plottter.generators.base import (
    ExpressionParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.generators.expression_eval import ExpressionError, SafeEvaluator
from plottter.generators import register_generator
from plottter.models import Canvas, Polyline

_TWO_PI = 2.0 * math.pi


@register_generator
class PolarGenerator(Generator):
    """Evaluates user-supplied r(theta) expression and converts to Cartesian."""

    name = "Polar Curves"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ExpressionParam(
                name="r_expr",
                label="r(theta) expression",
                default="sin(4*theta)",
                variables=["theta"],
                description="Radius expression in terms of theta — controls the shape of the polar curve (e.g. sin(4*theta) for a rose)",
            ),
            FloatParam(
                name="theta_start",
                label="Theta start",
                min=-1000.0,
                max=1000.0,
                step=0.01,
                default=0.0,
                description="Starting angle in radians (0 = rightward)",
            ),
            FloatParam(
                name="theta_end",
                label="Theta end",
                min=-1000.0,
                max=1000.0,
                step=0.01,
                default=_TWO_PI,
                description="Ending angle in radians (2π ≈ 6.283 for a full rotation)",
            ),
            IntParam(
                name="num_points",
                label="Number of points",
                min=100,
                max=100000,
                step=100,
                default=5000,
                description="Number of sample points — higher values give smoother curves",
            ),
            FloatParam(
                name="scale",
                label="Scale (0 = auto-fit)",
                min=0.0,
                max=100.0,
                step=0.01,
                default=0.0,
                description="Manual scale factor (0 = auto-fit to canvas drawing area)",
            ),
            FloatParam(
                name="rotation_deg",
                label="Rotation (degrees)",
                min=-360.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate the output by this many degrees around the center",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Shift the output horizontally in millimeters",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Shift the output vertically in millimeters",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Rose (4-petal)",
                params={
                    "r_expr": "cos(4*theta)",
                    "theta_start": 0.0,
                    "theta_end": _TWO_PI,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Rose (3-petal)",
                params={
                    "r_expr": "cos(3*theta)",
                    "theta_start": 0.0,
                    "theta_end": math.pi,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Cardioid",
                params={
                    "r_expr": "1+cos(theta)",
                    "theta_start": 0.0,
                    "theta_end": _TWO_PI,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Archimedean Spiral",
                params={
                    "r_expr": "0.5+0.1*theta",
                    "theta_start": 0.0,
                    "theta_end": 10.0 * math.pi,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Logarithmic Spiral",
                params={
                    "r_expr": "exp(0.1*theta)",
                    "theta_start": 0.0,
                    "theta_end": 4.0 * math.pi,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Limacon",
                params={
                    "r_expr": "1+2*cos(theta)",
                    "theta_start": 0.0,
                    "theta_end": _TWO_PI,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
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
        r_expr: str = str(params.get("r_expr", "sin(4*theta)"))
        theta_start = float(params.get("theta_start", 0.0))
        theta_end = float(params.get("theta_end", _TWO_PI))
        num_points = int(params.get("num_points", 5000))
        scale = float(params.get("scale", 0.0))
        rotation_deg = float(params.get("rotation_deg", 0.0))
        x_offset_mm = float(params.get("x_offset_mm", 0.0))
        y_offset_mm = float(params.get("y_offset_mm", 0.0))

        evaluator = SafeEvaluator()
        try:
            r_fn = evaluator.compile(r_expr, ["theta"])
        except ExpressionError as exc:
            raise ValueError(f"Expression error: {exc}") from exc

        theta_values = np.linspace(theta_start, theta_end, num_points)
        raw_points: list[tuple[float, float]] = []
        start_time = time.monotonic()

        for i, theta_val in enumerate(theta_values):
            if cancelled_callback and cancelled_callback():
                break
            if time.monotonic() - start_time > 5.0:
                raise ValueError("Generation timed out (5 second limit)")
            try:
                r = r_fn(theta=float(theta_val))
            except Exception as exc:
                raise ValueError(f"Error evaluating r at theta={theta_val:.4f}: {exc}") from exc
            if math.isfinite(r):
                x = r * math.cos(float(theta_val))
                y = r * math.sin(float(theta_val))
                if math.isfinite(x) and math.isfinite(y):
                    raw_points.append((x, y))

            if progress_callback and i % 500 == 0:
                progress_callback(int(i / num_points * 80))

        if not raw_points:
            return []

        polyline = _apply_transforms(raw_points, scale, rotation_deg, x_offset_mm, y_offset_mm, canvas)

        if progress_callback:
            progress_callback(100)

        return [polyline]


def _apply_transforms(
    raw_points: list[tuple[float, float]],
    scale: float,
    rotation_deg: float,
    x_offset_mm: float,
    y_offset_mm: float,
    canvas: Canvas,
) -> Polyline:
    """Center, rotate, scale and translate points to fit the canvas."""
    xs = [p[0] for p in raw_points]
    ys = [p[1] for p in raw_points]
    raw_cx = (min(xs) + max(xs)) / 2.0
    raw_cy = (min(ys) + max(ys)) / 2.0

    centered = [(x - raw_cx, y - raw_cy) for x, y in raw_points]

    if rotation_deg != 0.0:
        theta = math.radians(rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        centered = [
            (x * cos_t - y * sin_t, x * sin_t + y * cos_t)
            for x, y in centered
        ]

    cx_vals = [p[0] for p in centered]
    cy_vals = [p[1] for p in centered]
    span_x = (max(cx_vals) - min(cx_vals)) or 1.0
    span_y = (max(cy_vals) - min(cy_vals)) or 1.0

    draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1

    if scale == 0.0:
        scale_factor = min(draw_w / span_x, draw_h / span_y) * 0.9
    else:
        scale_factor = scale

    scaled = [(x * scale_factor, y * scale_factor) for x, y in centered]

    canvas_cx = (draw_x1 + draw_x2) / 2.0 + x_offset_mm
    canvas_cy = (draw_y1 + draw_y2) / 2.0 + y_offset_mm

    return [(x + canvas_cx, y + canvas_cy) for x, y in scaled]
