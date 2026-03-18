"""ConcentricRingsGenerator — concentric shape rings radiating from a centre point."""

from __future__ import annotations

import math
from typing import Any

try:
    import noise as _noise_lib
    _NOISE_AVAILABLE = True
except ImportError:
    _NOISE_AVAILABLE = False

from plottter.generators import register_generator
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline

_TWO_PI = 2.0 * math.pi

# Polygon vertex counts for each shape
_POLYGON_SIDES: dict[str, int] = {
    "Triangle": 3,
    "Square": 4,
    "Pentagon": 5,
    "Hexagon": 6,
    "Octagon": 8,
}


def _circle_ring(
    cx: float,
    cy: float,
    r_base: float,
    n_points: int,
    noise_fn: Any = None,
    noise_scale: float = 0.05,
    noise_amplitude: float = 0.0,
    ring_index: float = 0.0,
    noise_evolution: float = 0.1,
    noise_base: int = 0,
) -> Polyline:
    """Return a closed circle polyline centred at (cx, cy).

    If noise_fn is provided, each point's radius is perturbed by Perlin noise.
    """
    pts = []
    for i in range(n_points + 1):
        angle = _TWO_PI * i / n_points
        r = r_base
        if noise_fn is not None and noise_amplitude > 0.0:
            theta_norm = angle / _TWO_PI  # 0..1
            nv = noise_fn(
                theta_norm * noise_scale * 10.0,
                ring_index * noise_evolution,
                base=noise_base,
            )
            r += nv * noise_amplitude
            if r < 0.0:
                r = 0.0
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def _polygon_ring(
    cx: float,
    cy: float,
    r_base: float,
    sides: int,
    noise_fn: Any = None,
    noise_scale: float = 0.05,
    noise_amplitude: float = 0.0,
    ring_index: float = 0.0,
    noise_evolution: float = 0.1,
    noise_base: int = 0,
) -> Polyline:
    """Return a closed regular polygon polyline centred at (cx, cy).

    Polygon vertices use circumradius r_base but are then optionally
    perturbed by Perlin noise along each vertex's radial direction.
    """
    pts = []
    start_angle = -math.pi / 2
    for i in range(sides + 1):
        angle = start_angle + _TWO_PI * i / sides
        r = r_base
        if noise_fn is not None and noise_amplitude > 0.0:
            theta_norm = (angle - start_angle) / _TWO_PI  # 0..1
            nv = noise_fn(
                theta_norm * noise_scale * 10.0,
                ring_index * noise_evolution,
                base=noise_base,
            )
            r += nv * noise_amplitude
            if r < 0.0:
                r = 0.0
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def _apply_gaps_to_ring(
    pts: Polyline,
    ring_gap_chance: float,
    ring_index: float,
    gap_noise_base: int,
    noise_fn: Any,
) -> list[Polyline]:
    """Split a closed ring polyline into arcs based on noise-driven gaps.

    Returns a list of arc polylines; gaps are segments where the noise value
    exceeds ``(1 - ring_gap_chance * 2)``.
    """
    if ring_gap_chance <= 0.0 or noise_fn is None:
        return [pts]

    n = len(pts) - 1  # last pt == first pt (closed ring)
    arcs: list[Polyline] = []
    current_arc: Polyline = []

    for i in range(n):
        # Normalised angle for this segment's midpoint
        theta_mid = _TWO_PI * (i + 0.5) / n
        gap_nv = noise_fn(theta_mid * 5.0, ring_index * 0.5, base=gap_noise_base)
        is_gap = gap_nv > (1.0 - ring_gap_chance * 2.0)

        if not is_gap:
            if not current_arc:
                current_arc.append(pts[i])
            current_arc.append(pts[i + 1])
        else:
            if len(current_arc) >= 2:
                arcs.append(current_arc)
            current_arc = []

    if len(current_arc) >= 2:
        arcs.append(current_arc)

    return arcs if arcs else []


@register_generator
class ConcentricRingsGenerator(Generator):
    """Generates concentric shape rings (circles or regular polygons) from a centre point."""

    name = "Concentric Rings"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            IntParam(
                name="ring_count",
                label="Ring count",
                min=2,
                max=200,
                step=1,
                default=30,
                description="Number of rings to draw",
            ),
            FloatParam(
                name="ring_spacing_mm",
                label="Ring spacing (mm)",
                min=0.5,
                max=10.0,
                step=0.1,
                default=2.0,
                description="Distance between rings in mm",
            ),
            ChoiceParam(
                name="ring_shape",
                label="Ring shape",
                choices=["Circle", "Square", "Triangle", "Pentagon", "Hexagon", "Octagon"],
                default="Circle",
                description="Shape of each ring",
            ),
            FloatParam(
                name="center_x_mm",
                label="Centre X offset (mm)",
                min=-200.0,
                max=200.0,
                step=0.5,
                default=0.0,
                description="Horizontal offset of ring centre from canvas centre",
            ),
            FloatParam(
                name="center_y_mm",
                label="Centre Y offset (mm)",
                min=-200.0,
                max=200.0,
                step=0.5,
                default=0.0,
                description="Vertical offset of ring centre from canvas centre",
            ),
            IntParam(
                name="points_per_ring",
                label="Points per ring",
                min=16,
                max=512,
                step=8,
                default=64,
                description="Smoothness of each circle ring — more points = smoother curves (circles only)",
            ),
            # --- Noise distortion ---
            FloatParam(
                name="noise_scale",
                label="Noise scale",
                min=0.01,
                max=1.0,
                step=0.01,
                default=0.05,
                description="Frequency of the Perlin noise distortion",
            ),
            FloatParam(
                name="noise_amplitude_mm",
                label="Noise amplitude (mm)",
                min=0.0,
                max=20.0,
                step=0.5,
                default=3.0,
                description="How far noise pushes ring points from their ideal position (0 = no distortion)",
            ),
            IntParam(
                name="noise_seed",
                label="Noise seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for Perlin noise (change for different distortion patterns)",
            ),
            FloatParam(
                name="noise_evolution",
                label="Noise evolution",
                min=0.0,
                max=1.0,
                step=0.01,
                default=0.1,
                description=(
                    "How much the noise pattern changes from inner to outer rings — "
                    "0 = same distortion on all rings, 1 = very different"
                ),
            ),
            # --- Amplitude growth ---
            ChoiceParam(
                name="amplitude_growth",
                label="Amplitude growth",
                choices=["Constant", "Linear", "Exponential"],
                default="Linear",
                description=(
                    "How noise amplitude changes with ring index: "
                    "Constant = uniform, Linear = proportional to radius, "
                    "Exponential = proportional to radius²"
                ),
            ),
            # --- Ring thickness variation ---
            FloatParam(
                name="thickness_noise",
                label="Thickness noise",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.0,
                description=(
                    "Noise-based variation in ring spacing — "
                    "0 = uniform spacing, higher values cause some rings to bunch together"
                ),
            ),
            # --- Gap / partial rings ---
            FloatParam(
                name="ring_gap_chance",
                label="Ring gap chance",
                min=0.0,
                max=0.8,
                step=0.05,
                default=0.0,
                description=(
                    "Probability of gaps in each ring — "
                    "0 = complete rings, higher values create broken/arc segments"
                ),
            ),
            # --- Radial connecting lines ---
            BoolParam(
                name="radial_lines",
                label="Radial lines",
                default=False,
                description="Draw lines from the centre outward through all rings at evenly-spaced angles",
            ),
            IntParam(
                name="radial_line_count",
                label="Radial line count",
                min=4,
                max=64,
                step=1,
                default=8,
                description="Number of radial lines (only used when Radial lines is enabled)",
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
                name="Simple Circles",
                params={
                    "ring_count": 30,
                    "ring_spacing_mm": 2.0,
                    "ring_shape": "Circle",
                    "noise_amplitude_mm": 0.0,
                },
            ),
            Preset(
                name="Tight Circles",
                params={
                    "ring_count": 60,
                    "ring_spacing_mm": 1.0,
                    "ring_shape": "Circle",
                    "points_per_ring": 128,
                    "noise_amplitude_mm": 0.0,
                },
            ),
            Preset(
                name="Concentric Hexagons",
                params={
                    "ring_count": 20,
                    "ring_spacing_mm": 3.0,
                    "ring_shape": "Hexagon",
                    "noise_amplitude_mm": 0.0,
                },
            ),
            Preset(
                name="Square Spiral",
                params={
                    "ring_count": 25,
                    "ring_spacing_mm": 2.5,
                    "ring_shape": "Square",
                    "noise_amplitude_mm": 0.0,
                },
            ),
            Preset(
                name="Triangle Nest",
                params={
                    "ring_count": 20,
                    "ring_spacing_mm": 3.0,
                    "ring_shape": "Triangle",
                    "noise_amplitude_mm": 0.0,
                },
            ),
            Preset(
                name="Ripples",
                params={
                    "ring_count": 40,
                    "ring_spacing_mm": 1.5,
                    "ring_shape": "Circle",
                    "noise_amplitude_mm": 2.0,
                    "noise_scale": 0.08,
                    "amplitude_growth": "Linear",
                    "points_per_ring": 128,
                },
            ),
            Preset(
                name="Organic Rings",
                params={
                    "ring_count": 25,
                    "ring_spacing_mm": 2.5,
                    "ring_shape": "Circle",
                    "noise_amplitude_mm": 5.0,
                    "noise_evolution": 0.3,
                    "ring_gap_chance": 0.2,
                    "thickness_noise": 0.2,
                    "points_per_ring": 128,
                },
            ),
            Preset(
                name="Distorted Squares",
                params={
                    "ring_count": 20,
                    "ring_spacing_mm": 3.0,
                    "ring_shape": "Square",
                    "noise_amplitude_mm": 4.0,
                    "amplitude_growth": "Linear",
                },
            ),
            Preset(
                name="Spider Web",
                params={
                    "ring_count": 30,
                    "ring_spacing_mm": 2.0,
                    "ring_shape": "Circle",
                    "noise_amplitude_mm": 1.0,
                    "amplitude_growth": "Linear",
                    "radial_lines": True,
                    "radial_line_count": 12,
                    "points_per_ring": 128,
                },
            ),
            Preset(
                name="Topographic",
                params={
                    "ring_count": 50,
                    "ring_spacing_mm": 1.0,
                    "ring_shape": "Circle",
                    "noise_amplitude_mm": 8.0,
                    "noise_scale": 0.03,
                    "thickness_noise": 0.3,
                    "amplitude_growth": "Linear",
                    "points_per_ring": 256,
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
        ring_count = int(params.get("ring_count", 30))
        ring_spacing = float(params.get("ring_spacing_mm", 2.0))
        ring_shape = str(params.get("ring_shape", "Circle"))
        center_x_offset = float(params.get("center_x_mm", 0.0))
        center_y_offset = float(params.get("center_y_mm", 0.0))
        points_per_ring = int(params.get("points_per_ring", 64))
        noise_scale = float(params.get("noise_scale", 0.05))
        noise_amplitude = float(params.get("noise_amplitude_mm", 0.0))
        noise_seed = int(params.get("noise_seed", 42))
        noise_evolution = float(params.get("noise_evolution", 0.1))
        amplitude_growth = str(params.get("amplitude_growth", "Linear"))
        thickness_noise = float(params.get("thickness_noise", 0.0))
        ring_gap_chance = float(params.get("ring_gap_chance", 0.0))
        do_radial_lines = bool(params.get("radial_lines", False))
        radial_line_count = int(params.get("radial_line_count", 8))
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        # Centre of the canvas drawing area
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0 + center_x_offset
        cy = (draw_y1 + draw_y2) / 2.0 + center_y_offset

        polygon_sides = _POLYGON_SIDES.get(ring_shape, 0)
        is_circle = ring_shape == "Circle"

        # Resolve noise function — use Perlin if available and amplitude > 0
        noise_fn = None
        if _NOISE_AVAILABLE and noise_amplitude > 0.0:
            noise_fn = _noise_lib.pnoise2
        noise_base = noise_seed % 256
        gap_noise_base = (noise_seed + 2) % 256

        result: list[Polyline] = []

        for i in range(1, ring_count + 1):
            if cancelled_callback and cancelled_callback():
                break

            r = i * ring_spacing

            # Apply thickness noise (bunching) if requested
            if thickness_noise > 0.0 and _NOISE_AVAILABLE:
                thickness_nv = _noise_lib.pnoise2(
                    i * 0.3,
                    0.0,
                    base=(noise_base + 1) % 256,
                )
                r += thickness_nv * ring_spacing * thickness_noise

            if r <= 0.0:
                continue

            # Compute effective noise amplitude for this ring
            if noise_amplitude > 0.0:
                ring_frac = i / ring_count  # 0..1 normalised ring index
                if amplitude_growth == "Constant":
                    effective_amplitude = noise_amplitude
                elif amplitude_growth == "Exponential":
                    effective_amplitude = noise_amplitude * ring_frac * ring_frac
                else:  # "Linear" (default)
                    effective_amplitude = noise_amplitude * ring_frac
            else:
                effective_amplitude = 0.0

            ring_index = float(i)

            if is_circle:
                ring = _circle_ring(
                    cx,
                    cy,
                    r,
                    points_per_ring,
                    noise_fn=noise_fn,
                    noise_scale=noise_scale,
                    noise_amplitude=effective_amplitude,
                    ring_index=ring_index,
                    noise_evolution=noise_evolution,
                    noise_base=noise_base,
                )
            else:
                ring = _polygon_ring(
                    cx,
                    cy,
                    r,
                    polygon_sides,
                    noise_fn=noise_fn,
                    noise_scale=noise_scale,
                    noise_amplitude=effective_amplitude,
                    ring_index=ring_index,
                    noise_evolution=noise_evolution,
                    noise_base=noise_base,
                )

            # Apply gap splitting if requested
            if ring_gap_chance > 0.0 and _NOISE_AVAILABLE:
                arcs = _apply_gaps_to_ring(
                    ring, ring_gap_chance, ring_index, gap_noise_base, _noise_lib.pnoise2
                )
                result.extend(arcs)
            else:
                result.append(ring)

            if progress_callback and i % 10 == 0:
                progress_callback(int(i / ring_count * 100))

        # Generate radial lines from centre through all rings at evenly-spaced angles
        if do_radial_lines and ring_count > 0:
            radial_noise_fn = _noise_lib.pnoise2 if _NOISE_AVAILABLE else None
            for j in range(radial_line_count):
                angle = _TWO_PI * j / radial_line_count
                theta_norm = angle / _TWO_PI
                radial_pts: Polyline = [(cx, cy)]
                for i in range(1, ring_count + 1):
                    r = i * ring_spacing
                    # Apply thickness noise
                    if thickness_noise > 0.0 and _NOISE_AVAILABLE:
                        tnv = _noise_lib.pnoise2(
                            i * 0.3, 0.0, base=(noise_base + 1) % 256
                        )
                        r += tnv * ring_spacing * thickness_noise
                    if r <= 0.0:
                        continue
                    # Compute effective amplitude for this ring
                    if noise_amplitude > 0.0:
                        ring_frac = i / ring_count
                        if amplitude_growth == "Constant":
                            eff_amp = noise_amplitude
                        elif amplitude_growth == "Exponential":
                            eff_amp = noise_amplitude * ring_frac * ring_frac
                        else:
                            eff_amp = noise_amplitude * ring_frac
                    else:
                        eff_amp = 0.0
                    # Apply noise distortion at this angle
                    if radial_noise_fn is not None and eff_amp > 0.0:
                        nv = radial_noise_fn(
                            theta_norm * noise_scale * 10.0,
                            float(i) * noise_evolution,
                            base=noise_base,
                        )
                        r += nv * eff_amp
                        if r < 0.0:
                            r = 0.0
                    radial_pts.append(
                        (cx + r * math.cos(angle), cy + r * math.sin(angle))
                    )
                if len(radial_pts) >= 2:
                    result.append(radial_pts)

        if progress_callback:
            progress_callback(100)

        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result
