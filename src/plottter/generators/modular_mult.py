"""ModularMultGenerator — circle patterns via modular multiplication."""

from __future__ import annotations

import math
from typing import Any

from plottter.generators.base import (
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.generators import register_generator
from plottter.models import Canvas, Polyline

_TWO_PI = 2.0 * math.pi


@register_generator
class ModularMultGenerator(Generator):
    """Places N points on a circle; connects each point p to (p*multiplier) % N."""

    name = "Modular Multiplication"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            IntParam(
                name="num_points",
                label="Points on circle",
                min=2,
                max=1000,
                step=1,
                default=200,
                description="Number of evenly-spaced points placed around a circle. Each point p connects to point (p × multiplier) % N",
            ),
            FloatParam(
                name="multiplier",
                label="Multiplier",
                min=0.0,
                max=500.0,
                step=0.1,
                default=2.0,
                description="Multiplication factor — each point connects to the point at (index × multiplier) mod N. Try integer and fractional values for different patterns",
            ),
            FloatParam(
                name="radius_mm",
                label="Radius mm (0 = auto-fit)",
                min=0.0,
                max=500.0,
                step=1.0,
                default=0.0,
                description="Radius of the base circle in millimeters (0 = auto-fit to canvas)",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the generated output on the canvas page (mm)",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the generated output on the canvas page (mm)",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Times 2 (cardioid)",
                params={"num_points": 200, "multiplier": 2.0, "radius_mm": 0.0},
            ),
            Preset(
                name="Times 3",
                params={"num_points": 200, "multiplier": 3.0, "radius_mm": 0.0},
            ),
            Preset(
                name="Times 4",
                params={"num_points": 200, "multiplier": 4.0, "radius_mm": 0.0},
            ),
            Preset(
                name="Times 51",
                params={"num_points": 200, "multiplier": 51.0, "radius_mm": 0.0},
            ),
            Preset(
                name="Star (times 2, 100 pts)",
                params={"num_points": 100, "multiplier": 2.0, "radius_mm": 0.0},
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        num_pts = int(params.get("num_points", 200))
        multiplier = float(params.get("multiplier", 2.0))
        radius = float(params.get("radius_mm", 0.0))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        if radius == 0.0:
            radius = min(draw_x2 - draw_x1, draw_y2 - draw_y1) * 0.45

        # Pre-compute circle points
        angles = [_TWO_PI * i / num_pts for i in range(num_pts)]
        circle_pts = [
            (cx + radius * math.cos(a), cy + radius * math.sin(a))
            for a in angles
        ]

        segments: list[Polyline] = []
        for p in range(num_pts):
            if cancelled_callback and cancelled_callback():
                break
            target = int(round(p * multiplier)) % num_pts
            if target != p:
                segments.append([circle_pts[p], circle_pts[target]])
            if progress_callback and p % 50 == 0:
                progress_callback(int(p / num_pts * 100))

        if progress_callback:
            progress_callback(100)

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            segments = [[(x + x_off, y + y_off) for x, y in path] for path in segments]
        return segments
