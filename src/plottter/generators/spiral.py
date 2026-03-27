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
    IntParam,
    Generator,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _sample_image_at(img: np.ndarray, px: float, py: float) -> float:
    """Bilinear sample a grayscale image at non-integer pixel coordinates."""
    h, w = img.shape[:2]
    x0 = max(0, min(int(px), w - 1))
    y0 = max(0, min(int(py), h - 1))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = px - int(px)
    fy = py - int(py)
    v00 = float(img[y0, x0])
    v10 = float(img[y0, x1])
    v01 = float(img[y1, x0])
    v11 = float(img[y1, x1])
    return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy


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
            # --- Oscillation params ---
            FloatParam(
                name="amplitude",
                label="Amplitude",
                min=0.01,
                max=2.0,
                step=0.01,
                default=0.8,
                description="Oscillation amplitude relative to ring spacing. Higher = more fill in dark areas.",
            ),
            ChoiceParam(
                name="oscillation_mode",
                label="Oscillation Mode",
                choices=["Sawtooth", "Sine", "Square"],
                default="Sawtooth",
                description="Waveform shape for perpendicular oscillation",
            ),
            # --- Variable velocity params ---
            BoolParam(
                name="variable_velocity",
                label="Variable Velocity",
                default=True,
                description="Adjust sampling density by brightness — more detail in dark areas",
            ),
            FloatParam(
                name="min_velocity",
                label="Min Velocity",
                min=0.5,
                max=5.0,
                step=0.1,
                default=0.8,
                visible_when={"variable_velocity": [True]},
                description="Step multiplier in dark areas (smallest step = most detail)",
            ),
            FloatParam(
                name="max_velocity",
                label="Max Velocity",
                min=1.0,
                max=10.0,
                step=0.1,
                default=3.0,
                visible_when={"variable_velocity": [True]},
                description="Step multiplier in bright areas (largest step = coarsest)",
            ),
            # --- White-area skipping params ---
            BoolParam(
                name="skip_white",
                label="Skip White Areas",
                default=True,
                description="Flatten or skip oscillation in bright/white areas",
            ),
            IntParam(
                name="white_threshold",
                label="White Threshold",
                min=0,
                max=255,
                step=1,
                default=240,
                visible_when={"skip_white": [True]},
                description="Brightness above this value is considered white (0–255)",
            ),
            # --- Connected lines ---
            BoolParam(
                name="connected_lines",
                label="Connected Lines",
                default=True,
                description="Output a single continuous polyline. When False, break at white-skipped regions (more pen lifts, cleaner bright areas)",
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
        _defaults = {
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
            "variable_velocity": True,
            "min_velocity": 0.8,
            "max_velocity": 3.0,
            "skip_white": True,
            "white_threshold": 240,
            "connected_lines": True,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        return [
            Preset(
                name="Default",
                params={
                    **_defaults,
                    "ring_spacing_mm": 3.0,
                    "amplitude": 0.8,
                    "oscillation_mode": "Sawtooth",
                },
            ),
            Preset(
                name="Portrait",
                params={
                    **_defaults,
                    "ring_spacing_mm": 2.5,
                    "amplitude": 0.9,
                    "oscillation_mode": "Sawtooth",
                    "variable_velocity": True,
                    "skip_white": True,
                    "white_threshold": 230,
                },
            ),
            Preset(
                name="Bold Spiral",
                params={
                    **_defaults,
                    "ring_spacing_mm": 4.0,
                    "amplitude": 1.2,
                    "oscillation_mode": "Sawtooth",
                    "variable_velocity": True,
                    "min_velocity": 0.5,
                    "max_velocity": 4.0,
                },
            ),
            Preset(
                name="Fine Detail",
                params={
                    **_defaults,
                    "ring_spacing_mm": 1.5,
                    "step_size_mm": 0.3,
                    "amplitude": 0.7,
                    "oscillation_mode": "Sine",
                    "variable_velocity": True,
                },
            ),
            Preset(
                name="Sine Wave",
                params={
                    **_defaults,
                    "ring_spacing_mm": 3.0,
                    "amplitude": 0.8,
                    "oscillation_mode": "Sine",
                    "variable_velocity": False,
                },
            ),
            Preset(
                name="Minimal",
                params={
                    **_defaults,
                    "ring_spacing_mm": 5.0,
                    "amplitude": 0.6,
                    "oscillation_mode": "Sawtooth",
                    "skip_white": True,
                    "white_threshold": 200,
                    "connected_lines": True,
                },
            ),
            Preset(
                name="Loose Coil",
                params={
                    **_defaults,
                    "ring_spacing_mm": 8.0,
                    "step_size_mm": 1.0,
                    "amplitude": 0.8,
                    "oscillation_mode": "Sawtooth",
                    "blur_radius": 2.0,
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

        amplitude = float(params.get("amplitude", 0.8))
        oscillation_mode = str(params.get("oscillation_mode", "Sawtooth"))
        variable_velocity = bool(params.get("variable_velocity", True))
        min_velocity = float(params.get("min_velocity", 0.8))
        max_velocity = float(params.get("max_velocity", 3.0))
        skip_white = bool(params.get("skip_white", True))
        white_threshold = int(params.get("white_threshold", 240))
        connected_lines = bool(params.get("connected_lines", True))
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        # Convert source image to grayscale for brightness sampling
        # (needed for oscillation, variable velocity, and white skipping)
        gray: np.ndarray | None = None
        if source is not None:
            if source.ndim == 3:
                gray = np.mean(source[:, :, :3], axis=2).astype(np.float32)
            else:
                gray = source.astype(np.float32)

        img_rect_w = img_x2 - img_x1
        img_rect_h = img_y2 - img_y1
        can_sample = gray is not None and img_rect_w > 0 and img_rect_h > 0

        if progress_callback:
            progress_callback(0)

        # Integrated spiral loop: trace + variable velocity + oscillation + white skip
        two_pi = 2.0 * math.pi
        theta_start = step_size_mm * two_pi / ring_spacing_mm
        theta = theta_start

        polylines: list[Polyline] = []
        current: Polyline = []
        step_idx = 0

        while True:
            r = ring_spacing_mm * theta / two_pi
            if r > max_radius_mm:
                break

            x = center_x_mm + r * math.cos(theta)
            y = center_y_mm + r * math.sin(theta)

            # Sample brightness
            brightness = 128.0
            if can_sample:
                px = (x - img_x1) / img_rect_w * (img_w - 1)
                py = (y - img_y1) / img_rect_h * (img_h - 1)
                brightness = _sample_image_at(gray, px, py)

            # White-area skipping
            in_white = skip_white and brightness > white_threshold

            if in_white and not connected_lines:
                # Break the polyline at white areas
                if len(current) >= 2:
                    polylines.append(current)
                current = []
            else:
                # Compute oscillation displacement
                if can_sample and amplitude > 0.0 and not in_white:
                    perp_x = math.cos(theta)
                    perp_y = math.sin(theta)
                    offset = amplitude * ring_spacing_mm / 2.0 * (1.0 - brightness / 255.0)
                    if oscillation_mode == "Sine":
                        sign = math.sin(step_idx * math.pi / 2)
                    elif oscillation_mode == "Square":
                        sign = 1.0 if (step_idx // 2) % 2 == 0 else -1.0
                    else:  # Sawtooth
                        sign = 1.0 if step_idx % 2 == 0 else -1.0
                    current.append((x + perp_x * offset * sign + x_off, y + perp_y * offset * sign + y_off))
                else:
                    # Flat point (white area in connected mode, or no image/amplitude)
                    current.append((x + x_off, y + y_off))

            # Variable velocity: adjust angular step by brightness
            if variable_velocity and can_sample:
                t = brightness / 255.0
                ease = t * t  # quadratic ease-in
                effective_step = step_size_mm * (min_velocity + ease * (max_velocity - min_velocity))
            else:
                effective_step = step_size_mm

            theta += effective_step / r
            step_idx += 1

        # Flush the last segment
        if current:
            polylines.append(current)

        if progress_callback:
            progress_callback(100)

        return polylines if polylines else []
