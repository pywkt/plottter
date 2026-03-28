"""HatchingGenerator — variable-density parallel/cross/contour hatching from image brightness."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import _px_to_mm, compute_image_rect
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _brightness_to_spacing(
    brightness: float,
    min_spacing: float,
    max_spacing: float,
    curve: str,
) -> float:
    """Map brightness (0=black, 255=white) to a hatch spacing in mm.

    Dark areas → min_spacing (dense). Light areas → max_spacing (sparse).
    """
    t = max(0.0, min(1.0, brightness / 255.0))
    if curve == "quadratic":
        t = t * t
    elif curve == "logarithmic":
        t = math.log1p(t * (math.e - 1.0))  # log(1 + t*(e-1)) maps 0→0, 1→1

    return min_spacing + t * (max_spacing - min_spacing)


def _sample_image_at(
    img: np.ndarray,
    px: float,
    py: float,
) -> float:
    """Bilinear sample of a grayscale image at non-integer pixel coords."""
    h, w = img.shape[:2]
    x0 = int(px)
    y0 = int(py)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    fx = px - int(px)
    fy = py - int(py)
    v00 = float(img[y0, x0])
    v10 = float(img[y0, x1])
    v01 = float(img[y1, x0])
    v11 = float(img[y1, x1])
    return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy


def _clip_polylines_to_max_length(
    polylines: list[Polyline],
    max_length_mm: float,
) -> list[Polyline]:
    """Split polylines so that no individual segment exceeds max_length_mm."""
    result: list[Polyline] = []
    for poly in polylines:
        current: Polyline = [poly[0]]
        accumulated = 0.0
        for i in range(1, len(poly)):
            x0, y0 = poly[i - 1]
            x1, y1 = poly[i]
            seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            remaining = max_length_mm - accumulated
            if seg_len <= remaining:
                current.append(poly[i])
                accumulated += seg_len
            else:
                # Split at the max_length boundary, then continue
                while seg_len > remaining:
                    if remaining > 0 and seg_len > 0:
                        t = remaining / seg_len
                        mid_x = x0 + t * (x1 - x0)
                        mid_y = y0 + t * (y1 - y0)
                        current.append((mid_x, mid_y))
                        if len(current) >= 2:
                            result.append(current)
                        current = [(mid_x, mid_y)]
                        x0, y0 = mid_x, mid_y
                        seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                    accumulated = 0.0
                    remaining = max_length_mm
                current.append(poly[i])
                accumulated = seg_len
        if len(current) >= 2:
            result.append(current)
    return result


def _apply_oscillation(
    polyline: Polyline,
    cos_a: float,
    sin_a: float,
    osc_amplitude: float,
    osc_wavelength_mm: float,
    osc_mode: str,
    img: np.ndarray,
    img_rect: "tuple[float, float, float, float]",
) -> Polyline:
    """Displace each point in polyline perpendicular to hatch direction using a wave.

    cos_a/sin_a define the hatch direction. Displacement is along the perpendicular
    (-sin_a, cos_a). Amplitude scales per-point with local brightness (darker = more
    displacement). Phase is based on cumulative arc length along the polyline.
    """
    if osc_amplitude == 0.0 or osc_wavelength_mm <= 0.0:
        return polyline

    draw_x1, draw_y1, draw_x2, draw_y2 = img_rect
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1
    img_h, img_w = img.shape[:2]

    result: Polyline = []
    arc_length = 0.0

    for i, (x, y) in enumerate(polyline):
        if i > 0:
            px0, py0 = polyline[i - 1]
            arc_length += math.sqrt((x - px0) ** 2 + (y - py0) ** 2)

        # Sample brightness at (x, y) using bilinear interpolation
        if draw_w > 0 and draw_h > 0:
            px = max(0.0, min(img_w - 1.0, (x - draw_x1) / draw_w * img_w))
            py_img = max(0.0, min(img_h - 1.0, (y - draw_y1) / draw_h * img_h))
            brightness = _sample_image_at(img, px, py_img)
        else:
            brightness = 0.0

        # Scale amplitude by darkness (bright areas → small displacement)
        amp = osc_amplitude * max(0.0, 1.0 - brightness / 255.0)

        t = arc_length / osc_wavelength_mm
        if osc_mode == "Sawtooth":
            wave = 2.0 * (t % 1.0) - 1.0
        else:
            wave = math.sin(2.0 * math.pi * t)

        offset = amp * wave
        result.append((x + offset * (-sin_a), y + offset * cos_a))

    return result


def _generate_parallel_hatch(
    img: np.ndarray,
    angle_deg: float,
    min_spacing_mm: float,
    max_spacing_mm: float,
    density_curve: str,
    canvas: Canvas,
    cancelled_callback: Any,
    progress_callback: Any,
    progress_start: int = 0,
    progress_end: int = 100,
    line_length_mm: float = 0.0,
    img_rect: "tuple[float, float, float, float] | None" = None,
    oscillation: bool = False,
    osc_amplitude: float = 1.0,
    osc_wavelength_mm: float = 2.0,
    osc_mode: str = "Sine",
) -> list[Polyline]:
    """Generate variable-density parallel hatch lines at the given angle."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for hatching. "
            "Install with: pip install opencv-python"
        ) from exc

    if img_rect is not None:
        draw_x1, draw_y1, draw_x2, draw_y2 = img_rect
    else:
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1

    img_h, img_w = img.shape[:2]

    # Rotate image to align hatch with horizontal
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # We'll work in mm space directly.
    # Generate scan lines perpendicular to angle, spaced at min_spacing_mm initially.
    # The scan direction is perpendicular to angle: (-sin_a, cos_a).
    # The hatch direction is (cos_a, sin_a).

    # Determine range of "perpendicular" coordinate
    cx = (draw_x1 + draw_x2) / 2.0
    cy = (draw_y1 + draw_y2) / 2.0

    # Canvas corners in rotated coordinates (perpendicular component)
    corners_mm = [
        (draw_x1, draw_y1), (draw_x2, draw_y1),
        (draw_x1, draw_y2), (draw_x2, draw_y2),
    ]
    perp_coords = [-(x - cx) * sin_a + (y - cy) * cos_a for x, y in corners_mm]
    para_coords = [(x - cx) * cos_a + (y - cy) * sin_a for x, y in corners_mm]

    perp_min = min(perp_coords) - max_spacing_mm
    perp_max = max(perp_coords) + max_spacing_mm
    para_min = min(para_coords) - max_spacing_mm
    para_max = max(para_coords) + max_spacing_mm

    polylines: list[Polyline] = []

    # Iterate across perpendicular axis with adaptive spacing
    perp_pos = perp_min
    total_range = perp_max - perp_min
    iteration = 0
    max_iterations = int(total_range / min_spacing_mm) + 1

    while perp_pos <= perp_max:
        if cancelled_callback and cancelled_callback():
            break
        if iteration > max_iterations:
            break

        if progress_callback and iteration % 20 == 0:
            frac = min(1.0, (perp_pos - perp_min) / max(total_range, 1e-6))
            pct = progress_start + int(frac * (progress_end - progress_start))
            progress_callback(pct)
        iteration += 1

        # Sample brightness at this scan position
        # Convert perpendicular position to canvas mm
        sample_x = cx + perp_pos * (-sin_a)  # point on perpendicular axis
        sample_y = cy + perp_pos * (cos_a)

        # Clamp to canvas
        sample_x = max(draw_x1, min(draw_x2, sample_x))
        sample_y = max(draw_y1, min(draw_y2, sample_y))

        # Sample a few points along the hatch line to get representative brightness
        n_samples = 5
        brightness_sum = 0.0
        brightness_count = 0
        for k in range(n_samples):
            t = para_min + (k + 0.5) / n_samples * (para_max - para_min)
            wx = sample_x + t * cos_a
            wy = sample_y + t * sin_a
            if not (draw_x1 <= wx <= draw_x2 and draw_y1 <= wy <= draw_y2):
                continue
            # Convert to image pixel
            px = (wx - draw_x1) / draw_w * img_w
            py = (wy - draw_y1) / draw_h * img_h
            brightness_sum += _sample_image_at(img, px, py)
            brightness_count += 1

        if brightness_count == 0:
            perp_pos += min_spacing_mm
            continue

        avg_brightness = brightness_sum / brightness_count
        spacing = _brightness_to_spacing(avg_brightness, min_spacing_mm, max_spacing_mm, density_curve)

        # If brightness is very high (near white), skip this line
        if avg_brightness > 240:
            perp_pos += spacing
            continue

        # Build the hatch line: trace from para_min to para_max
        # Only include segments where image is dark enough
        step_mm = max(0.5, min_spacing_mm * 0.5)
        n_steps = max(2, int((para_max - para_min) / step_mm))

        line_segments: list[Polyline] = []
        current_segment: Polyline = []
        for k in range(n_steps + 1):
            t = para_min + k / n_steps * (para_max - para_min)
            wx = sample_x + t * cos_a
            wy = sample_y + t * sin_a

            # Check canvas bounds
            if not (draw_x1 - 0.01 <= wx <= draw_x2 + 0.01 and
                    draw_y1 - 0.01 <= wy <= draw_y2 + 0.01):
                if len(current_segment) >= 2:
                    line_segments.append(current_segment)
                current_segment = []
                continue

            # Sample brightness at this pixel
            px = (wx - draw_x1) / draw_w * img_w
            py = (wy - draw_y1) / draw_h * img_h
            px = max(0.0, min(img_w - 1.0, px))
            py = max(0.0, min(img_h - 1.0, py))
            pixel_brightness = _sample_image_at(img, px, py)

            if pixel_brightness < 240:  # include dark pixels
                current_segment.append((wx, wy))
            else:
                if len(current_segment) >= 2:
                    line_segments.append(current_segment)
                current_segment = []

        if len(current_segment) >= 2:
            line_segments.append(current_segment)

        if oscillation and line_segments:
            line_segments = [
                _apply_oscillation(seg, cos_a, sin_a, osc_amplitude, osc_wavelength_mm, osc_mode, img, (draw_x1, draw_y1, draw_x2, draw_y2))
                for seg in line_segments
            ]

        polylines.extend(line_segments)

        perp_pos += spacing

    if line_length_mm > 0.0:
        polylines = _clip_polylines_to_max_length(polylines, line_length_mm)

    return polylines


def _generate_contour_hatch(
    img: np.ndarray,
    min_spacing_mm: float,
    max_spacing_mm: float,
    canvas: Canvas,
    cancelled_callback: Any,
    progress_callback: Any,
    img_rect: "tuple[float, float, float, float] | None" = None,
) -> list[Polyline]:
    """Generate contour-following hatch lines using image gradient direction."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for contour hatching. "
            "Install with: pip install opencv-python"
        ) from exc

    if img_rect is not None:
        draw_x1, draw_y1, draw_x2, draw_y2 = img_rect
    else:
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1
    img_h, img_w = img.shape[:2]

    # Compute gradient
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    if progress_callback:
        progress_callback(20)

    # Contour hatch: trace streamlines following gradient perpendicular direction
    # (i.e., iso-brightness lines — contours of brightness)
    step_mm = min_spacing_mm * 0.5

    polylines: list[Polyline] = []

    # Sample start positions on a grid.
    # Use max_spacing_mm (not min_spacing_mm) to space starting seeds so that
    # neighbouring seeds do not produce nearly-duplicate streamlines.  Using
    # min_spacing_mm here would create nx*ny = O((W/min_spacing)²) seeds — on an
    # A4 canvas with min_spacing=0.8mm that is ~80 000 starts, producing tens of
    # thousands of overlapping paths completely impractical for plotting.
    grid_spacing_mm = max(min_spacing_mm, max_spacing_mm)
    nx = max(2, int(draw_w / grid_spacing_mm))
    ny = max(2, int(draw_h / grid_spacing_mm))

    total_starts = nx * ny
    for idx in range(total_starts):
        if cancelled_callback and cancelled_callback():
            break

        if progress_callback and idx % 20 == 0:
            progress_callback(20 + int(idx / total_starts * 70))

        col = idx % nx
        row = idx // nx
        start_x = draw_x1 + (col + 0.5) / nx * draw_w
        start_y = draw_y1 + (row + 0.5) / ny * draw_h

        # Convert to pixel
        px = (start_x - draw_x1) / draw_w * img_w
        py = (start_y - draw_y1) / draw_h * img_h

        ipx = int(max(0, min(img_w - 1, px)))
        ipy = int(max(0, min(img_h - 1, py)))
        brightness = float(img[ipy, ipx])

        if brightness > 220:  # skip very bright areas
            continue

        # Trace along iso-brightness direction (perpendicular to gradient)
        trail: Polyline = [(start_x, start_y)]
        cx_mm, cy_mm = start_x, start_y

        max_steps = int((draw_w + draw_h) / step_mm / 4)
        for _ in range(max_steps):
            ipx2 = int(max(0, min(img_w - 1, (cx_mm - draw_x1) / draw_w * img_w)))
            ipy2 = int(max(0, min(img_h - 1, (cy_mm - draw_y1) / draw_h * img_h)))

            gx = float(grad_x[ipy2, ipx2])
            gy = float(grad_y[ipy2, ipx2])
            magnitude = math.sqrt(gx * gx + gy * gy)

            if magnitude < 1.0:
                break

            # Perpendicular to gradient (iso-brightness direction)
            dx = -gy / magnitude
            dy = gx / magnitude

            cx_mm += dx * step_mm
            cy_mm += dy * step_mm

            if not (draw_x1 <= cx_mm <= draw_x2 and draw_y1 <= cy_mm <= draw_y2):
                break

            trail.append((cx_mm, cy_mm))

        if len(trail) >= 2:
            polylines.append(trail)

    if progress_callback:
        progress_callback(100)

    return polylines


@register_generator
class HatchingGenerator(Generator):
    """Variable-density parallel, cross, or contour hatching from image brightness."""

    name = "Hatching"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="mode",
                label="Hatch Mode",
                choices=["parallel", "cross", "contour"],
                default="parallel",
                description="Type of hatching — parallel lines, two-pass cross-hatching, or contour-following lines that trace image edges",
                choice_descriptions={
                    "parallel": "Parallel lines at a fixed angle, density varies inversely with brightness",
                    "cross": "Two overlapping sets of parallel lines at different angles for a cross-hatch texture",
                    "contour": "Lines that follow image edge contours for a woodcut or engraving look",
                },
            ),
            FloatParam(
                name="angle_deg",
                label="Angle (deg)",
                min=-180.0,
                max=180.0,
                step=1.0,
                default=45.0,
                description="Angle of hatch lines in degrees (0 = horizontal, 45 = diagonal, 90 = vertical)",
            ),
            FloatParam(
                name="angle2_deg",
                label="Second Angle (deg, cross mode)",
                min=-180.0,
                max=180.0,
                step=1.0,
                default=135.0,
                description="Angle of the second set of hatch lines for cross-hatch mode",
            ),
            FloatParam(
                name="min_spacing_mm",
                label="Min Spacing (mm)",
                min=0.1,
                max=10.0,
                step=0.1,
                default=0.5,
                description="Minimum spacing between hatch lines in bright areas (mm)",
            ),
            FloatParam(
                name="max_spacing_mm",
                label="Max Spacing (mm)",
                min=0.2,
                max=20.0,
                step=0.2,
                default=5.0,
                description="Maximum spacing between hatch lines in dark areas (mm) — larger gap = sparser hatching in bright regions",
            ),
            ChoiceParam(
                name="density_curve",
                label="Density Curve",
                choices=["linear", "quadratic", "logarithmic"],
                default="linear",
                description="How hatch density varies with image brightness",
                choice_descriptions={
                    "linear": "Density increases linearly with darkness — balanced, natural-looking result",
                    "quadratic": "Density increases faster in dark areas — higher contrast between light and dark regions",
                    "logarithmic": "Density increases gently — preserves detail in both very light and very dark areas",
                },
            ),
            FloatParam(
                name="line_length_mm",
                label="Max Line Length (mm, 0=unlimited)",
                min=0.0,
                max=500.0,
                step=1.0,
                default=0.0,
                description="Maximum length of individual hatch lines (0 = unlimited)",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image (dark areas become sparse, bright areas become dense)",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before hatching (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before hatching (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description="Gaussian blur applied before hatching — reduces noise, produces smoother hatch density transitions",
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
            BoolParam(
                name="oscillation",
                label="Oscillation",
                default=False,
                description="Add perpendicular wave oscillation to hatch lines — amplitude varies by brightness",
            ),
            FloatParam(
                name="osc_amplitude",
                label="Osc Amplitude (mm)",
                min=0.1,
                max=5.0,
                step=0.1,
                default=1.0,
                visible_when={"oscillation": [True]},
                description="Maximum oscillation amplitude in mm",
            ),
            FloatParam(
                name="osc_wavelength_mm",
                label="Osc Wavelength (mm)",
                min=0.5,
                max=10.0,
                step=0.5,
                default=2.0,
                visible_when={"oscillation": [True]},
                description="Wavelength of oscillation in mm",
            ),
            ChoiceParam(
                name="osc_mode",
                label="Osc Waveform",
                choices=["Sine", "Sawtooth"],
                default="Sine",
                visible_when={"oscillation": [True]},
                description="Waveform shape for oscillation",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default Parallel",
                params={
                    "mode": "parallel",
                    "angle_deg": 45.0,
                    "angle2_deg": 135.0,
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "density_curve": "linear",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Cross Hatch",
                params={
                    "mode": "cross",
                    "angle_deg": 45.0,
                    "angle2_deg": 135.0,
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 4.0,
                    "density_curve": "quadratic",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Woodcut",
                params={
                    # Contour mode: streamlines follow iso-brightness curves.
                    # Grid seed spacing = max_spacing_mm so each seed produces an
                    # independent streamline (no near-duplicate paths).
                    "mode": "contour",
                    "angle_deg": 0.0,
                    "angle2_deg": 90.0,
                    "min_spacing_mm": 1.0,
                    "max_spacing_mm": 6.0,
                    "density_curve": "logarithmic",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Fine Detail",
                params={
                    "mode": "parallel",
                    "angle_deg": 30.0,
                    "angle2_deg": 120.0,
                    "min_spacing_mm": 0.3,
                    "max_spacing_mm": 3.0,
                    "density_curve": "quadratic",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Portrait Photo",
                params={
                    # 45° hatching tuned for face/portrait images: logarithmic
                    # density curve emphasises shadow detail in skin tones;
                    # contrast boost separates shadows from highlights.
                    "mode": "parallel",
                    "angle_deg": 45.0,
                    "angle2_deg": 135.0,
                    "min_spacing_mm": 0.4,
                    "max_spacing_mm": 4.0,
                    "density_curve": "logarithmic",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Landscape Photo",
                params={
                    # Horizontal cross-hatch at 0°/90° with wider spacing suits
                    # landscape horizon lines and skies; quadratic curve preserves
                    # bright sky while densely shading foreground detail.
                    "mode": "cross",
                    "angle_deg": 0.0,
                    "angle2_deg": 90.0,
                    "min_spacing_mm": 0.6,
                    "max_spacing_mm": 6.0,
                    "density_curve": "quadratic",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Wavy Hatch",
                params={
                    "mode": "parallel",
                    "angle_deg": 45.0,
                    "angle2_deg": 135.0,
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "density_curve": "linear",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "oscillation": True,
                    "osc_mode": "Sine",
                    "osc_amplitude": 1.0,
                    "osc_wavelength_mm": 2.0,
                },
            ),
            Preset(
                name="Zigzag Fill",
                params={
                    "mode": "parallel",
                    "angle_deg": 0.0,
                    "angle2_deg": 90.0,
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "density_curve": "linear",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "oscillation": True,
                    "osc_mode": "Sawtooth",
                    "osc_amplitude": 1.5,
                    "osc_wavelength_mm": 1.5,
                },
            ),
            Preset(
                name="Oscillating Cross-Hatch",
                params={
                    "mode": "cross",
                    "angle_deg": 45.0,
                    "angle2_deg": 135.0,
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "density_curve": "linear",
                    "line_length_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "oscillation": True,
                    "osc_mode": "Sine",
                    "osc_amplitude": 0.8,
                    "osc_wavelength_mm": 2.5,
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
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        # Apply preprocessing
        img = source.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            img = adjust_brightness(img, brightness)
        if contrast != 0.0:
            img = adjust_contrast(img, contrast)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        if do_invert:
            img = invert_image(img)
        source = img

        # Ensure grayscale
        if source.ndim == 3:
            try:
                import cv2
                source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            except ImportError:
                source = source.mean(axis=2).astype(np.uint8)

        mode = str(params.get("mode", "parallel"))
        angle_deg = float(params.get("angle_deg", 45.0))
        angle2_deg = float(params.get("angle2_deg", 135.0))
        min_spacing = float(params.get("min_spacing_mm", 0.5))
        max_spacing = float(params.get("max_spacing_mm", 5.0))
        density_curve = str(params.get("density_curve", "linear"))
        line_length_mm = float(params.get("line_length_mm", 0.0))
        oscillation = bool(params.get("oscillation", False))
        osc_amplitude = float(params.get("osc_amplitude", 1.0))
        osc_wavelength_mm = float(params.get("osc_wavelength_mm", 2.0))
        osc_mode = str(params.get("osc_mode", "Sine"))

        # Ensure valid range
        if min_spacing > max_spacing:
            min_spacing, max_spacing = max_spacing, min_spacing
        if min_spacing <= 0:
            min_spacing = 0.1

        img_h, img_w = source.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_rect = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        if mode == "contour":
            result = _generate_contour_hatch(
                source, min_spacing, max_spacing, canvas, cancelled_callback, progress_callback,
                img_rect=img_rect,
            )
            if line_length_mm > 0:
                result = _clip_polylines_to_max_length(result, line_length_mm)
        elif mode == "cross":
            if progress_callback:
                progress_callback(0)
            lines1 = _generate_parallel_hatch(
                source, angle_deg, min_spacing, max_spacing, density_curve,
                canvas, cancelled_callback, progress_callback,
                progress_start=0, progress_end=50,
                line_length_mm=line_length_mm,
                img_rect=img_rect,
                oscillation=oscillation,
                osc_amplitude=osc_amplitude,
                osc_wavelength_mm=osc_wavelength_mm,
                osc_mode=osc_mode,
            )
            if cancelled_callback and cancelled_callback():
                result = lines1
            else:
                lines2 = _generate_parallel_hatch(
                    source, angle2_deg, min_spacing, max_spacing, density_curve,
                    canvas, cancelled_callback, progress_callback,
                    progress_start=50, progress_end=100,
                    line_length_mm=line_length_mm,
                    img_rect=img_rect,
                    oscillation=oscillation,
                    osc_amplitude=osc_amplitude,
                    osc_wavelength_mm=osc_wavelength_mm,
                    osc_mode=osc_mode,
                )
                result = lines1 + lines2
        else:  # parallel
            result = _generate_parallel_hatch(
                source, angle_deg, min_spacing, max_spacing, density_curve,
                canvas, cancelled_callback, progress_callback,
                progress_start=0, progress_end=100,
                line_length_mm=line_length_mm,
                img_rect=img_rect,
                oscillation=oscillation,
                osc_amplitude=osc_amplitude,
                osc_wavelength_mm=osc_wavelength_mm,
                osc_mode=osc_mode,
            )

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
        return result
