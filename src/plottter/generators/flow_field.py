"""FlowFieldGenerator — Perlin noise flow field particle trails."""

from __future__ import annotations

import math
import random as _random
from typing import Any

from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.generators import register_generator
from plottter.models import Canvas, Polyline

_TWO_PI = 2.0 * math.pi


def _superformula(theta: float, m: float, n1: float, n2: float, n3: float) -> float:
    """Gielis superformula: r = (|cos(m*theta/4)|^n2 + |sin(m*theta/4)|^n3)^(-1/n1)."""
    t = m * theta / 4.0
    base = abs(math.cos(t)) ** n2 + abs(math.sin(t)) ** n3
    if base == 0.0:
        return 0.0
    return base ** (-1.0 / n1)


@register_generator
class FlowFieldGenerator(Generator):
    """Generates particle trails driven by a Perlin noise angle field."""

    name = "Flow Field"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="field_mode",
                label="Field mode",
                choices=["Perlin Noise", "Superformula"],
                default="Perlin Noise",
                description="Flow field generation mode — Perlin Noise uses classic noise, Superformula uses Gielis superformula tangents",
            ),
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
            BoolParam(
                name="rectilinear",
                label="Rectilinear mode",
                default=False,
                description="Constrain paths to mostly horizontal/vertical with alternating axes — creates a digital circuit-board aesthetic",
            ),
            FloatParam(
                name="sf_m",
                label="SF: Lobes (m)",
                min=1.0,
                max=20.0,
                step=0.5,
                default=6.0,
                description="Number of petals/lobes in the superformula shape (field_mode=Superformula)",
            ),
            FloatParam(
                name="sf_n1",
                label="SF: Curvature (n1)",
                min=0.1,
                max=40.0,
                step=0.1,
                default=1.0,
                description="Shape curvature — controls how pointy or rounded the lobes are (field_mode=Superformula)",
            ),
            FloatParam(
                name="sf_n2",
                label="SF: Sine factor (n2)",
                min=0.1,
                max=40.0,
                step=0.1,
                default=1.0,
                description="Sine factor — affects lobe symmetry (field_mode=Superformula)",
            ),
            FloatParam(
                name="sf_n3",
                label="SF: Cosine factor (n3)",
                min=0.1,
                max=40.0,
                step=0.1,
                default=1.0,
                description="Cosine factor — affects lobe symmetry (field_mode=Superformula)",
            ),
            FloatParam(
                name="sf_center_x",
                label="SF: Center X (%)",
                min=0.0,
                max=100.0,
                step=1.0,
                default=50.0,
                description="Center X as % of canvas width (field_mode=Superformula)",
            ),
            FloatParam(
                name="sf_center_y",
                label="SF: Center Y (%)",
                min=0.0,
                max=100.0,
                step=1.0,
                default=50.0,
                description="Center Y as % of canvas height (field_mode=Superformula)",
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
            Preset(
                name="Quantized Grid",
                params={
                    "num_particles": 2000,
                    "step_size_mm": 0.8,
                    "max_steps": 150,
                    "noise_scale": 0.008,
                    "noise_octaves": 3,
                    "seed": 42,
                    "angle_range": _TWO_PI,
                    "quantize_directions": 4,
                    "quantize_offset_deg": 0,
                },
            ),
            Preset(
                name="Circuit Board",
                params={
                    "num_particles": 3000,
                    "step_size_mm": 0.5,
                    "max_steps": 100,
                    "noise_scale": 0.015,
                    "noise_octaves": 2,
                    "seed": 42,
                    "angle_range": _TWO_PI,
                    "quantize_directions": 8,
                    "quantize_offset_deg": 0,
                },
            ),
            Preset(
                name="Hex Directions",
                params={
                    "num_particles": 1500,
                    "step_size_mm": 1.0,
                    "max_steps": 120,
                    "noise_scale": 0.01,
                    "noise_octaves": 4,
                    "seed": 42,
                    "angle_range": _TWO_PI,
                    "quantize_directions": 6,
                    "quantize_offset_deg": 0,
                },
            ),
            Preset(
                name="Angular Turbulence",
                params={
                    "num_particles": 2500,
                    "step_size_mm": 1.2,
                    "max_steps": 80,
                    "noise_scale": 0.04,
                    "noise_octaves": 6,
                    "seed": 42,
                    "angle_range": _TWO_PI,
                    "quantize_directions": 12,
                    "quantize_offset_deg": 15,
                },
            ),
            Preset(
                name="Rectilinear",
                params={
                    "num_particles": 2000,
                    "step_size_mm": 1.0,
                    "max_steps": 100,
                    "noise_scale": 0.01,
                    "noise_octaves": 4,
                    "seed": 42,
                    "angle_range": _TWO_PI,
                    "rectilinear": True,
                },
            ),
            Preset(
                name="Flower Flow",
                params={
                    "field_mode": "Superformula",
                    "num_particles": 2000,
                    "step_size_mm": 1.0,
                    "max_steps": 100,
                    "seed": 42,
                    "sf_m": 6.0,
                    "sf_n1": 1.0,
                    "sf_n2": 1.0,
                    "sf_n3": 1.0,
                    "sf_center_x": 50.0,
                    "sf_center_y": 50.0,
                },
            ),
            Preset(
                name="Star Flow",
                params={
                    "field_mode": "Superformula",
                    "num_particles": 1500,
                    "step_size_mm": 1.0,
                    "max_steps": 100,
                    "seed": 42,
                    "sf_m": 5.0,
                    "sf_n1": 0.3,
                    "sf_n2": 0.3,
                    "sf_n3": 0.3,
                    "sf_center_x": 50.0,
                    "sf_center_y": 50.0,
                },
            ),
            Preset(
                name="Organic Swirl",
                params={
                    "field_mode": "Superformula",
                    "num_particles": 2000,
                    "step_size_mm": 1.0,
                    "max_steps": 100,
                    "seed": 42,
                    "sf_m": 3.0,
                    "sf_n1": 2.0,
                    "sf_n2": 2.0,
                    "sf_n3": 2.0,
                    "sf_center_x": 50.0,
                    "sf_center_y": 50.0,
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
        pnoise = None
        if str(params.get("field_mode", "Perlin Noise")) != "Superformula":
            try:
                import noise as pnoise  # type: ignore[assignment]
            except ImportError as exc:
                raise RuntimeError(
                    "The 'noise' package is required for Perlin Noise mode. "
                    "Install it with: pip install noise"
                ) from exc

        field_mode = str(params.get("field_mode", "Perlin Noise"))
        num_particles = int(params.get("num_particles", 1000))
        step_size = float(params.get("step_size_mm", 1.0))
        max_steps = int(params.get("max_steps", 100))
        noise_scale = float(params.get("noise_scale", 0.01))
        octaves = int(params.get("noise_octaves", 4))
        seed = int(params.get("seed", 42))
        angle_range = float(params.get("angle_range", _TWO_PI))
        quantize_directions = int(params.get("quantize_directions", 0))
        quantize_offset_rad = math.radians(float(params.get("quantize_offset_deg", 0.0)))
        rectilinear = bool(params.get("rectilinear", False))

        # Superformula parameters
        sf_m = float(params.get("sf_m", 6.0))
        sf_n1 = float(params.get("sf_n1", 1.0))
        sf_n2 = float(params.get("sf_n2", 1.0))
        sf_n3 = float(params.get("sf_n3", 1.0))
        sf_center_x_pct = float(params.get("sf_center_x", 50.0))
        sf_center_y_pct = float(params.get("sf_center_y", 50.0))

        rng = _random.Random(seed)
        noise_base = seed % 256

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        # Superformula center in mm
        sf_cx = draw_x1 + draw_w * sf_center_x_pct / 100.0
        sf_cy = draw_y1 + draw_h * sf_center_y_pct / 100.0

        # Optional image for brightness-based jitter
        source_image = params.get("_source_image")
        img_gray = None
        if rectilinear and source_image is not None:
            import numpy as np
            img = source_image
            if img.ndim == 3:
                try:
                    import cv2
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                except ImportError:
                    img = img.mean(axis=2).astype(np.uint8)
            img_gray = img

        trails: list[Polyline] = []
        for i in range(num_particles):
            if cancelled_callback and cancelled_callback():
                break

            x = rng.uniform(draw_x1, draw_x2)
            y = rng.uniform(draw_y1, draw_y2)

            trail: Polyline = [(x, y)]
            prev_axis: int | None = None
            for _ in range(max_steps):
                if field_mode == "Superformula":
                    theta = math.atan2(y - sf_cy, x - sf_cx)
                    r_sf = _superformula(theta, sf_m, sf_n1, sf_n2, sf_n3)
                    r_sf2 = _superformula(theta + 0.01, sf_m, sf_n1, sf_n2, sf_n3)
                    dr = r_sf2 - r_sf
                    angle = theta + math.pi / 2.0 + math.atan2(dr, r_sf * 0.01)
                else:
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

                if rectilinear:
                    # Determine preferred axis from noise angle
                    preferred_axis = 0 if abs(math.cos(angle)) >= abs(math.sin(angle)) else 1
                    # Alternate axes: 72% chance to flip from previous axis
                    if prev_axis is None:
                        axis = preferred_axis
                    else:
                        axis = (1 - prev_axis) if rng.random() < 0.72 else preferred_axis
                    prev_axis = axis

                    if axis == 0:
                        sign = 1.0 if math.cos(angle) >= 0.0 else -1.0
                        if rng.random() < 0.25:
                            sign *= -1.0
                        dx = sign * step_size
                        dy = rng.uniform(-0.35, 0.35) * step_size
                    else:
                        sign = 1.0 if math.sin(angle) >= 0.0 else -1.0
                        if rng.random() < 0.25:
                            sign *= -1.0
                        dx = rng.uniform(-0.35, 0.35) * step_size
                        dy = sign * step_size

                    # Compute jitter (increases in bright areas)
                    if img_gray is not None:
                        img_h, img_w = img_gray.shape[:2]
                        px = (x - draw_x1) / draw_w * img_w
                        py = (y - draw_y1) / draw_h * img_h
                        px = max(0.0, min(px, img_w - 1))
                        py = max(0.0, min(py, img_h - 1))
                        brightness = float(img_gray[int(py), int(px)]) / 255.0
                        darkness = 1.0 - brightness
                        jitter = (1.0 - darkness) * 1.3 + 0.15
                    else:
                        jitter = 0.5

                    dx += rng.uniform(-jitter, jitter)
                    dy += rng.uniform(-jitter, jitter)
                    x += dx
                    y += dy
                else:
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
