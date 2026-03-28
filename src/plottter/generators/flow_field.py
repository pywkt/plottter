"""FlowFieldGenerator — Perlin noise flow field particle trails."""

from __future__ import annotations

import math
import random as _random
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
class FlowFieldGenerator(Generator):
    """Generates particle trails driven by a Perlin noise angle field."""

    name = "Flow Field"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            IntParam(
                name="num_particles",
                label="Number of particles",
                min=100,
                max=10000,
                step=100,
                default=1000,
                description="Number of particle trails to generate — more particles give denser coverage",
            ),
            FloatParam(
                name="step_size_mm",
                label="Step size (mm)",
                min=0.1,
                max=20.0,
                step=0.1,
                default=1.0,
                description="Distance each particle travels per step in millimeters — smaller values give smoother curves",
            ),
            IntParam(
                name="max_steps",
                label="Max steps per particle",
                min=1,
                max=1000,
                step=10,
                default=100,
                description="Maximum number of steps per particle — controls the maximum trail length",
            ),
            FloatParam(
                name="noise_scale",
                label="Noise scale",
                min=0.001,
                max=0.5,
                step=0.001,
                default=0.01,
                description="Spatial scale of the Perlin noise field — smaller values create larger, smoother swirls; larger values create finer, more turbulent patterns",
            ),
            IntParam(
                name="noise_octaves",
                label="Noise octaves",
                min=1,
                max=8,
                step=1,
                default=4,
                description="Number of Perlin noise layers added together — more octaves add fine detail at the cost of performance",
            ),
            IntParam(
                name="seed",
                label="Random seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for reproducible particle starting positions and noise field",
            ),
            FloatParam(
                name="angle_range",
                label="Angle range (radians)",
                min=0.1,
                max=_TWO_PI * 2,
                step=0.1,
                default=_TWO_PI,
                description="Total angular range mapped from noise values (in radians) — 2π means full 360° rotation range",
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
            IntParam(
                name="quantize_directions",
                label="Quantize directions",
                min=0,
                max=36,
                step=1,
                default=0,
                description="Snap flow angles to this many discrete directions. 0 = off (smooth), 4 = cardinal (N/S/E/W), 8 = cardinal + diagonal, higher = more directions. Creates angular, architectural patterns.",
            ),
            FloatParam(
                name="quantize_offset_deg",
                label="Direction offset (degrees)",
                min=0.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate the set of discrete directions by this angle. E.g. offset=45° turns cardinal into diagonal.",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default Flow",
                params={
                    "num_particles": 1000,
                    "step_size_mm": 1.0,
                    "max_steps": 100,
                    "noise_scale": 0.01,
                    "noise_octaves": 4,
                    "seed": 42,
                    "angle_range": _TWO_PI,
                },
            ),
            Preset(
                name="Dense Short Trails",
                params={
                    "num_particles": 5000,
                    "step_size_mm": 0.5,
                    "max_steps": 50,
                    "noise_scale": 0.02,
                    "noise_octaves": 2,
                    "seed": 0,
                    "angle_range": _TWO_PI,
                },
            ),
            Preset(
                name="Long Smooth Trails",
                params={
                    "num_particles": 300,
                    "step_size_mm": 2.0,
                    "max_steps": 300,
                    "noise_scale": 0.005,
                    "noise_octaves": 6,
                    "seed": 123,
                    "angle_range": _TWO_PI,
                },
            ),
            Preset(
                name="Turbulent",
                params={
                    "num_particles": 2000,
                    "step_size_mm": 1.5,
                    "max_steps": 80,
                    "noise_scale": 0.05,
                    "noise_octaves": 8,
                    "seed": 999,
                    "angle_range": _TWO_PI * 2,
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
        try:
            import noise as pnoise
        except ImportError as exc:
            raise RuntimeError(
                "The 'noise' package is required for Flow Field generation. "
                "Install it with: pip install noise"
            ) from exc

        num_particles = int(params.get("num_particles", 1000))
        step_size = float(params.get("step_size_mm", 1.0))
        max_steps = int(params.get("max_steps", 100))
        noise_scale = float(params.get("noise_scale", 0.01))
        octaves = int(params.get("noise_octaves", 4))
        seed = int(params.get("seed", 42))
        angle_range = float(params.get("angle_range", _TWO_PI))
        quantize_directions = int(params.get("quantize_directions", 0))
        quantize_offset_rad = math.radians(float(params.get("quantize_offset_deg", 0.0)))

        rng = _random.Random(seed)
        noise_base = seed % 256

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()

        trails: list[Polyline] = []
        for i in range(num_particles):
            if cancelled_callback and cancelled_callback():
                break

            x = rng.uniform(draw_x1, draw_x2)
            y = rng.uniform(draw_y1, draw_y2)

            trail: Polyline = [(x, y)]
            for _ in range(max_steps):
                n = pnoise.pnoise2(
                    x * noise_scale,
                    y * noise_scale,
                    octaves=octaves,
                    base=noise_base,
                )
                angle = n * angle_range
                if quantize_directions > 0:
                    step_angle = _TWO_PI / quantize_directions
                    angle_shifted = angle - quantize_offset_rad
                    angle = round(angle_shifted / step_angle) * step_angle + quantize_offset_rad
                x += step_size * math.cos(angle)
                y += step_size * math.sin(angle)

                if not (draw_x1 <= x <= draw_x2 and draw_y1 <= y <= draw_y2):
                    break
                trail.append((x, y))

            if len(trail) > 1:
                trails.append(trail)

            if progress_callback and i % 100 == 0:
                progress_callback(int(i / num_particles * 100))

        if progress_callback:
            progress_callback(100)

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            trails = [[(x + x_off, y + y_off) for x, y in path] for path in trails]
        return trails
