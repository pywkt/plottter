"""ParametricGenerator — evaluates x(t), y(t) parametric curves."""

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
_LORENZ_SENTINEL = "__lorenz__"


@register_generator
class ParametricGenerator(Generator):
    """Evaluates user-supplied x(t) and y(t) expressions over a t range."""

    name = "Parametric Curves"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ExpressionParam(
                name="x_expr",
                label="x(t) expression",
                default="sin(3*t)",
                variables=["t"],
                description="X coordinate expression in terms of t — use sin, cos, and other math functions",
            ),
            ExpressionParam(
                name="y_expr",
                label="y(t) expression",
                default="sin(4*t)",
                variables=["t"],
                description="Y coordinate expression in terms of t — use sin, cos, and other math functions",
            ),
            FloatParam(
                name="t_start",
                label="t start",
                min=-1000.0,
                max=1000.0,
                step=0.01,
                default=0.0,
                description="Starting value of parameter t — controls where the curve begins",
            ),
            FloatParam(
                name="t_end",
                label="t end",
                min=-1000.0,
                max=1000.0,
                step=0.01,
                default=_TWO_PI,
                description="Ending value of parameter t — controls where the curve ends (2π ≈ 6.283 for one full period)",
            ),
            IntParam(
                name="num_points",
                label="Number of points",
                min=100,
                max=100000,
                step=100,
                default=5000,
                description="Number of sample points along the curve — higher values give smoother curves but slower generation",
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
            FloatParam(
                name="lorenz_sigma",
                label="Lorenz σ (sigma)",
                min=0.1,
                max=50.0,
                step=0.1,
                default=10.0,
                description="Lorenz attractor σ — controls how strongly X influences Y. Classic value: 10",
            ),
            FloatParam(
                name="lorenz_rho",
                label="Lorenz ρ (rho)",
                min=0.1,
                max=100.0,
                step=0.1,
                default=28.0,
                description="Lorenz attractor ρ — related to the Rayleigh number. Classic value: 28 (produces butterfly shape)",
            ),
            FloatParam(
                name="lorenz_beta",
                label="Lorenz β (beta)",
                min=0.01,
                max=10.0,
                step=0.01,
                default=round(8.0 / 3.0, 4),
                description="Lorenz attractor β — geometric factor. Classic value: 8/3 ≈ 2.667",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Lissajous",
                params={
                    "x_expr": "sin(3*t + pi/2)",
                    "y_expr": "sin(2*t)",
                    "t_start": 0.0,
                    "t_end": _TWO_PI,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Butterfly Curve",
                params={
                    "x_expr": "sin(t)*(exp(cos(t))-2*cos(4*t)-pow(sin(t/12),5))",
                    "y_expr": "cos(t)*(exp(cos(t))-2*cos(4*t)-pow(sin(t/12),5))",
                    "t_start": 0.0,
                    "t_end": 24.0 * math.pi,
                    "num_points": 10000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Spirograph (Epitrochoid)",
                params={
                    "x_expr": "6*cos(t)-3*cos(6*t)",
                    "y_expr": "6*sin(t)-3*sin(6*t)",
                    "t_start": 0.0,
                    "t_end": _TWO_PI,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Hypotrochoid",
                params={
                    "x_expr": "3*cos(t)+5*cos(3*t/5)",
                    "y_expr": "3*sin(t)-5*sin(3*t/5)",
                    "t_start": 0.0,
                    "t_end": 10.0 * math.pi,
                    "num_points": 5000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Farris / Mystery Curve",
                params={
                    "x_expr": "cos(t)+cos(7*t)/2+sin(-17*t)/3",
                    "y_expr": "sin(t)+sin(7*t)/2+cos(-17*t)/3",
                    "t_start": 0.0,
                    "t_end": 6.0 * math.pi,
                    "num_points": 10000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Lorenz Attractor",
                params={
                    "x_expr": _LORENZ_SENTINEL,
                    "y_expr": _LORENZ_SENTINEL,
                    "t_start": 0.0,
                    "t_end": 50.0,
                    "num_points": 10000,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "lorenz_sigma": 10.0,
                    "lorenz_rho": 28.0,
                    "lorenz_beta": round(8.0 / 3.0, 4),
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
        x_expr: str = str(params.get("x_expr", "sin(3*t)"))
        y_expr: str = str(params.get("y_expr", "sin(4*t)"))

        if x_expr == _LORENZ_SENTINEL or y_expr == _LORENZ_SENTINEL:
            return self._generate_lorenz(params, canvas, progress_callback, cancelled_callback)

        t_start = float(params.get("t_start", 0.0))
        t_end = float(params.get("t_end", _TWO_PI))
        num_points = int(params.get("num_points", 5000))
        scale = float(params.get("scale", 0.0))
        rotation_deg = float(params.get("rotation_deg", 0.0))
        x_offset_mm = float(params.get("x_offset_mm", 0.0))
        y_offset_mm = float(params.get("y_offset_mm", 0.0))

        evaluator = SafeEvaluator()
        try:
            x_fn = evaluator.compile(x_expr, ["t"])
            y_fn = evaluator.compile(y_expr, ["t"])
        except ExpressionError as exc:
            raise ValueError(f"Expression error: {exc}") from exc

        t_values = np.linspace(t_start, t_end, num_points)
        raw_points: list[tuple[float, float]] = []
        start_time = time.monotonic()

        for i, t_val in enumerate(t_values):
            if cancelled_callback and cancelled_callback():
                break
            if time.monotonic() - start_time > 5.0:
                raise ValueError("Generation timed out (5 second limit on expression evaluation)")
            try:
                x = x_fn(t=float(t_val))
                y = y_fn(t=float(t_val))
            except Exception as exc:
                raise ValueError(f"Error evaluating expression at t={t_val:.4f}: {exc}") from exc
            if math.isfinite(x) and math.isfinite(y):
                raw_points.append((x, y))

            if progress_callback and i % 500 == 0:
                progress_callback(int(i / num_points * 80))

        if not raw_points:
            return []

        polyline = self._apply_transforms(raw_points, scale, rotation_deg, x_offset_mm, y_offset_mm, canvas)

        if progress_callback:
            progress_callback(100)

        return [polyline]

    def _generate_lorenz(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        from scipy.integrate import odeint

        sigma = float(params.get("lorenz_sigma", 10.0))
        rho = float(params.get("lorenz_rho", 28.0))
        beta = float(params.get("lorenz_beta", 8.0 / 3.0))
        t_end = float(params.get("t_end", 50.0))
        num_points = int(params.get("num_points", 10000))
        scale = float(params.get("scale", 0.0))
        rotation_deg = float(params.get("rotation_deg", 0.0))
        x_offset_mm = float(params.get("x_offset_mm", 0.0))
        y_offset_mm = float(params.get("y_offset_mm", 0.0))

        def lorenz_ode(state: list[float], _t: float) -> list[float]:
            x, y, z = state
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

        if progress_callback:
            progress_callback(10)

        # Check for cancellation before the blocking odeint call.
        # odeint is synchronous and cannot be interrupted mid-run.
        if cancelled_callback and cancelled_callback():
            return []

        t_values = np.linspace(0.0, t_end, num_points)
        solution = odeint(lorenz_ode, [0.1, 0.1, 0.1], t_values)

        if progress_callback:
            progress_callback(80)

        raw_points = [
            (float(row[0]), float(row[1]))
            for row in solution
            if math.isfinite(float(row[0])) and math.isfinite(float(row[1]))
        ]
        if not raw_points:
            return []
        polyline = self._apply_transforms(raw_points, scale, rotation_deg, x_offset_mm, y_offset_mm, canvas)

        if progress_callback:
            progress_callback(100)

        return [polyline]

    @staticmethod
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

        # Center at origin
        centered = [(x - raw_cx, y - raw_cy) for x, y in raw_points]

        # Apply rotation
        if rotation_deg != 0.0:
            theta = math.radians(rotation_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            centered = [
                (x * cos_t - y * sin_t, x * sin_t + y * cos_t)
                for x, y in centered
            ]

        # Compute bounding box after rotation
        cx_vals = [p[0] for p in centered]
        cy_vals = [p[1] for p in centered]
        span_x = (max(cx_vals) - min(cx_vals)) or 1.0
        span_y = (max(cy_vals) - min(cy_vals)) or 1.0

        # Compute scale factor
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        if scale == 0.0:
            # Auto-fit: use 90% of drawing area
            scale_factor = min(draw_w / span_x, draw_h / span_y) * 0.9
        else:
            scale_factor = scale

        scaled = [(x * scale_factor, y * scale_factor) for x, y in centered]

        # Translate to canvas center + user offset
        canvas_cx = (draw_x1 + draw_x2) / 2.0 + x_offset_mm
        canvas_cy = (draw_y1 + draw_y2) / 2.0 + y_offset_mm

        return [(x + canvas_cx, y + canvas_cy) for x, y in scaled]
