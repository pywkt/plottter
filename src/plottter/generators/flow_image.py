"""FlowImageGenerator — image-driven flow field streamlines and squiggle scan lines."""

from __future__ import annotations

import math
import random as _random
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
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


def _sample_image_at(img: np.ndarray, px: float, py: float) -> float:
    """Bilinear sample a grayscale image at non-integer coordinates."""
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


def _trace_one_direction(
    field: np.ndarray,
    img: np.ndarray,
    x0_mm: float,
    y0_mm: float,
    direction: float,
    step_size_mm: float,
    max_length_mm: float,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    img_w: int,
    img_h: int,
    draw_w: float,
    draw_h: float,
    skip_background: bool,
    bg_threshold: float,
) -> list[tuple[float, float]]:
    """Trace a streamline in one direction via direct Euler integration.

    direction: +1.0 for forward, -1.0 for backward along the vector field.
    Returns list of points (not including the seed point).
    """
    pts: list[tuple[float, float]] = []
    x_mm, y_mm = x0_mm, y0_mm
    length = 0.0

    while length < max_length_mm:
        px = max(0.0, min(img_w - 1.0, (x_mm - draw_x1) / draw_w * img_w))
        py = max(0.0, min(img_h - 1.0, (y_mm - draw_y1) / draw_h * img_h))

        if skip_background and _sample_image_at(img, px, py) >= bg_threshold:
            break

        # Sample vector field at current position via bilinear interpolation
        vx = _sample_image_at(field[:, :, 0], px, py)
        vy = _sample_image_at(field[:, :, 1], px, py)
        vmag = math.sqrt(vx * vx + vy * vy)

        # Terminate if field magnitude is too weak
        if vmag < 0.01:
            break

        # Normalize and step in the given direction
        vx /= vmag
        vy /= vmag
        x_mm += direction * step_size_mm * vx
        y_mm += direction * step_size_mm * vy
        length += step_size_mm

        # Terminate if streamline exits canvas bounds
        if not (draw_x1 <= x_mm <= draw_x2 and draw_y1 <= y_mm <= draw_y2):
            break

        pts.append((x_mm, y_mm))

    return pts


def _generate_flow_streamlines(
    img: np.ndarray,
    num_lines: int,
    step_size_mm: float,
    max_length_mm: float,
    curvature_strength: float,
    seed: int,
    skip_background: bool,
    bg_threshold: float,
    canvas: Canvas,
    cancelled_callback: Any,
    progress_callback: Any,
    img_rect: "tuple[float, float, float, float] | None" = None,
    vector_field: str = "Edge Flow (ETF)",
    etf_kernel_radius: float = 5.0,
    etf_iterations: int = 3,
) -> list[Polyline]:
    """Generate streamlines guided by the chosen vector field.

    Uses direct Euler integration along the vector field with bidirectional
    tracing: each seed point is traced both forward (+v) and backward (-v),
    producing longer, more natural lines centred on their seed points.

    vector_field choices:
    - "Edge Flow (ETF)": coherent ETF tangent field — streamlines follow edges.
    - "Perpendicular Gradient": Sobel tangent (−gy, gx) — follows edges without
      iterative smoothing.
    - "Gradient": raw Sobel gradient — streamlines cross edges (legacy behavior).
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for Flow Image generation. "
            "Install with: pip install opencv-python"
        ) from exc

    if img_rect is not None:
        draw_x1, draw_y1, draw_x2, draw_y2 = img_rect
    else:
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1
    img_h, img_w = img.shape[:2]

    # Pre-smooth the image before computing gradients so the flow field is
    # coherent rather than responding to per-pixel noise.  Sigma is proportional
    # to image size so the smoothing scales with resolution.
    sigma = max(1.0, min(img_w, img_h) * 0.02)
    img_smooth = cv2.GaussianBlur(img, (0, 0), sigma)

    # Compute raw Sobel gradients
    gx_raw = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy_raw = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx_raw ** 2 + gy_raw ** 2)

    # Build (H, W, 2) unit-direction vector field
    if vector_field == "Edge Flow (ETF)":
        from plottter.generators._helpers import _compute_etf
        img_f = img_smooth.astype(np.float32) / 255.0
        tx, ty = _compute_etf(img_f, float(etf_kernel_radius), int(etf_iterations))
        field = np.stack([tx, ty], axis=-1)
    elif vector_field == "Perpendicular Gradient":
        # Tangent = (−gy, gx), normalised per-pixel
        local_mag = grad_mag + 1e-8
        tx = (-gy_raw / local_mag).astype(np.float32)
        ty = (gx_raw / local_mag).astype(np.float32)
        field = np.stack([tx, ty], axis=-1)
    else:  # "Gradient" — legacy: streamlines cross edges
        local_mag = grad_mag + 1e-8
        tx = (gx_raw / local_mag).astype(np.float32)
        ty = (gy_raw / local_mag).astype(np.float32)
        field = np.stack([tx, ty], axis=-1)

    rng = _random.Random(seed)
    polylines: list[Polyline] = []

    # Half the budget per direction so total length <= max_length_mm
    half_length = max_length_mm / 2.0

    for i in range(num_lines):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback and i % 50 == 0:
            progress_callback(int(i / num_lines * 100))

        # Random seed point within drawing area
        x_seed = rng.uniform(draw_x1, draw_x2)
        y_seed = rng.uniform(draw_y1, draw_y2)

        # Skip streamlines that start in a background area
        if skip_background:
            start_px = max(0.0, min(img_w - 1.0, (x_seed - draw_x1) / draw_w * img_w))
            start_py = max(0.0, min(img_h - 1.0, (y_seed - draw_y1) / draw_h * img_h))
            if _sample_image_at(img, start_px, start_py) >= bg_threshold:
                continue

        common_kwargs = dict(
            field=field,
            img=img,
            x0_mm=x_seed,
            y0_mm=y_seed,
            step_size_mm=step_size_mm,
            max_length_mm=half_length,
            draw_x1=draw_x1,
            draw_y1=draw_y1,
            draw_x2=draw_x2,
            draw_y2=draw_y2,
            img_w=img_w,
            img_h=img_h,
            draw_w=draw_w,
            draw_h=draw_h,
            skip_background=skip_background,
            bg_threshold=bg_threshold,
        )

        # Bidirectional tracing: forward (+1) and backward (-1) from seed
        forward_pts = _trace_one_direction(direction=+1.0, **common_kwargs)
        backward_pts = _trace_one_direction(direction=-1.0, **common_kwargs)

        # Combine: backward (reversed to get correct order) + seed + forward
        trail: Polyline = list(reversed(backward_pts)) + [(x_seed, y_seed)] + forward_pts

        if len(trail) >= 2:
            polylines.append(trail)

    if progress_callback:
        progress_callback(100)

    return polylines


def _compute_squiggle_y_positions(
    img: np.ndarray,
    num_lines: int,
    draw_y1: float,
    draw_y2: float,
    draw_h: float,
    line_spacing: str,
    min_spacing_mm: float,
    max_spacing_mm: float,
    group_size: int,
    group_gap_mm: float,
    group_intra_spacing_mm: float,
) -> list[float]:
    """Compute the list of Y positions for squiggle scan lines based on spacing mode."""
    img_h = img.shape[0]

    if line_spacing == "Uniform":
        return [draw_y1 + (i + 0.5) / num_lines * draw_h for i in range(num_lines)]

    # Vertical brightness profile: mean brightness per row.
    vert_profile: np.ndarray = img.mean(axis=1).astype(np.float32)

    def _brightness_at_y(y_mm: float) -> float:
        """Sample the vertical brightness profile at a given mm Y coordinate."""
        py = (y_mm - draw_y1) / draw_h * img_h
        py = max(0.0, min(img_h - 1.0, py))
        # Linear interpolation between neighbouring rows
        r0 = int(py)
        r1 = min(r0 + 1, img_h - 1)
        frac = py - r0
        return float(vert_profile[r0]) * (1.0 - frac) + float(vert_profile[r1]) * frac

    def _adaptive_spacing(brightness: float) -> float:
        """Map brightness (0=black, 255=white) to a spacing in mm (dark → dense)."""
        t = max(0.0, min(1.0, brightness / 255.0))
        return min_spacing_mm + t * (max_spacing_mm - min_spacing_mm)

    y_positions: list[float] = []

    if line_spacing == "Adaptive":
        y = draw_y1
        while y <= draw_y2:
            y_positions.append(y)
            spacing = _adaptive_spacing(_brightness_at_y(y))
            spacing = max(min_spacing_mm, spacing)
            y += spacing

    elif line_spacing == "Grouped":
        group_start = draw_y1
        while group_start <= draw_y2:
            for j in range(group_size):
                y = group_start + j * group_intra_spacing_mm
                if y <= draw_y2:
                    y_positions.append(y)
            group_start += (group_size - 1) * group_intra_spacing_mm + group_gap_mm

    elif line_spacing == "Adaptive + Grouped":
        group_start = draw_y1
        while group_start <= draw_y2:
            brightness = _brightness_at_y(group_start)
            inter_gap = _adaptive_spacing(brightness)
            inter_gap = max(min_spacing_mm, inter_gap)
            for j in range(group_size):
                y = group_start + j * group_intra_spacing_mm
                if y <= draw_y2:
                    y_positions.append(y)
            group_start += (group_size - 1) * group_intra_spacing_mm + inter_gap

    return y_positions


def _generate_squiggle(
    img: np.ndarray,
    num_lines: int,
    amplitude_mm: float,
    frequency: float,
    wave_spread: int,
    skip_background: bool,
    bg_threshold: float,
    canvas: Canvas,
    cancelled_callback: Any,
    progress_callback: Any,
    img_rect: "tuple[float, float, float, float] | None" = None,
    line_spacing: str = "Uniform",
    min_spacing_mm: float = 0.5,
    max_spacing_mm: float = 5.0,
    group_size: int = 3,
    group_gap_mm: float = 4.0,
    group_intra_spacing_mm: float = 0.5,
    displacement_variation: float = 0.0,
    seed: int = 42,
) -> list[Polyline]:
    """Generate horizontal scan lines with brightness-modulated sinusoidal squiggles.

    wave_spread controls vertical Gaussian blurring of the amplitude map so that
    strong waves influence neighboring scan lines above and below, creating smoother
    transitions across brightness boundaries.  wave_spread=0 preserves the original
    per-pixel amplitude behavior exactly.

    line_spacing controls how scan line Y positions are distributed:
    - "Uniform": evenly spaced (original behavior)
    - "Adaptive": brightness-adaptive density (dark → dense)
    - "Grouped": clusters of lines with gaps between groups
    - "Adaptive + Grouped": grouped with brightness-adaptive inter-group gaps

    displacement_variation adds per-line random amplitude scaling for organic variety.
    """
    if img_rect is not None:
        draw_x1, draw_y1, draw_x2, draw_y2 = img_rect
    else:
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1
    img_h, img_w = img.shape[:2]

    # Pre-compute 2D amplitude map: dark pixels → large amplitude.
    # Values are in [0, amplitude_mm].
    amp_map: np.ndarray = amplitude_mm * (1.0 - img.astype(np.float32) / 255.0)

    if wave_spread > 0:
        try:
            import cv2
            kernel_size = wave_spread * 2 + 1
            sigma = wave_spread * 0.5
            # Blur only vertically (1-pixel wide kernel) so amplitude from dark
            # regions spreads to neighboring scan lines above and below.
            amp_map = cv2.GaussianBlur(amp_map, (1, kernel_size), sigma)
        except ImportError:
            pass  # Fall back to unblurred amplitude map

    # Compute Y positions for all scan lines based on spacing mode.
    y_positions = _compute_squiggle_y_positions(
        img, num_lines, draw_y1, draw_y2, draw_h,
        line_spacing, min_spacing_mm, max_spacing_mm,
        group_size, group_gap_mm, group_intra_spacing_mm,
    )

    # Set up RNG for displacement variation (seeded for reproducibility).
    rng = _random.Random(seed)

    polylines: list[Polyline] = []
    steps_per_line = max(100, img_w * 2)
    total_lines = len(y_positions)

    for line_idx, y_base_mm in enumerate(y_positions):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback and line_idx % 10 == 0:
            progress = int(line_idx / max(1, total_lines) * 100)
            progress_callback(progress)

        # Per-line amplitude scale for displacement variation.
        if displacement_variation > 0.0:
            disp_scale = 1.0 + displacement_variation * rng.uniform(-1.0, 1.0)
            disp_scale = max(0.1, min(2.0, disp_scale))
        else:
            disp_scale = 1.0

        current_segment: Polyline = []
        for k in range(steps_per_line + 1):
            s = k / steps_per_line
            x_mm = draw_x1 + s * draw_w

            # Map mm position to image pixel coordinates
            px = (x_mm - draw_x1) / draw_w * img_w
            py = (y_base_mm - draw_y1) / draw_h * img_h
            px = max(0.0, min(img_w - 1.0, px))
            py = max(0.0, min(img_h - 1.0, py))

            # Use raw brightness for background detection (unblurred)
            brightness = _sample_image_at(img, px, py)

            if skip_background and brightness >= bg_threshold:
                # This pixel is background — finalize the current segment and
                # start a new one when content resumes.
                if len(current_segment) >= 2:
                    polylines.append(current_segment)
                current_segment = []
                continue

            # Sample amplitude from the (optionally blurred) amplitude map,
            # scaled by the per-line displacement variation factor.
            local_amp = _sample_image_at(amp_map, px, py) * disp_scale

            # Sinusoidal offset
            wave = local_amp * math.sin(2.0 * math.pi * frequency * s)
            y_mm = y_base_mm + wave

            # Clamp to drawing area
            y_mm = max(draw_y1, min(draw_y2, y_mm))
            current_segment.append((x_mm, y_mm))

        # Finalize the last segment for this scan line
        if len(current_segment) >= 2:
            polylines.append(current_segment)

    if progress_callback:
        progress_callback(100)

    return polylines


@register_generator
class FlowImageGenerator(Generator):
    """Image-driven flow field streamlines and squiggle scan lines."""

    name = "Flow Image"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="mode",
                label="Mode",
                choices=["flow", "squiggle"],
                default="flow",
                description="Flow mode traces streamlines deflected by image gradients; Squiggle mode draws horizontal waves modulated by brightness",
                choice_descriptions={
                    "flow": "Streamlines follow the direction of image gradients — darker areas deflect lines more strongly, creating organic curves around shapes",
                    "squiggle": "Horizontal scanning lines with amplitude and frequency modulated by per-pixel brightness — darker pixels produce larger waves",
                },
            ),
            IntParam(
                name="num_lines",
                label="Number of Lines",
                min=1,
                max=5000,
                step=10,
                default=200,
                description="Number of flow lines or squiggle rows to generate",
            ),
            FloatParam(
                name="step_size_mm",
                label="Step Size (mm)",
                min=0.1,
                max=10.0,
                step=0.1,
                default=0.5,
                description="Distance between consecutive points along each line in millimeters",
            ),
            FloatParam(
                name="max_length_mm",
                label="Max Length (mm)",
                min=2.0,
                max=100.0,
                step=1.0,
                default=20.0,
                description="Maximum streamline length in mm",
            ),
            FloatParam(
                name="curvature_strength",
                label="Curvature Strength",
                min=0.0,
                max=5.0,
                step=0.1,
                default=1.0,
                description="How strongly image gradients deflect flow lines — higher values create more dramatic curves following edges",
            ),
            ChoiceParam(
                name="vector_field",
                label="Vector Field",
                choices=["Edge Flow (ETF)", "Perpendicular Gradient", "Gradient"],
                default="Edge Flow (ETF)",
                visible_when={"mode": ["flow"]},
                description="Method used to compute the direction field that guides streamlines — Edge Flow (ETF) produces coherent lines that follow edges; Perpendicular Gradient is a fast tangent approximation; Gradient is the raw Sobel output (lines cross edges, legacy behavior)",
                choice_descriptions={
                    "Edge Flow (ETF)": "Iterative edge tangent flow — smooths the gradient tangent field so streamlines coherently follow edges across the image",
                    "Perpendicular Gradient": "Rotate the Sobel gradient 90° to get an edge-tangent direction — faster than ETF but less coherent",
                    "Gradient": "Raw Sobel gradient direction — streamlines cross edges rather than following them (original behavior)",
                },
            ),
            FloatParam(
                name="etf_kernel_radius",
                label="ETF Kernel Radius",
                min=1.0,
                max=10.0,
                step=0.5,
                default=5.0,
                visible_when={"vector_field": ["Edge Flow (ETF)"]},
                description="Spatial scale (pixels) of the ETF smoothing kernel — larger values produce smoother, more global edge flow at the cost of speed",
            ),
            IntParam(
                name="etf_iterations",
                label="ETF Iterations",
                min=1,
                max=10,
                step=1,
                default=3,
                visible_when={"vector_field": ["Edge Flow (ETF)"]},
                description="Number of ETF smoothing passes — more iterations produce a more globally coherent tangent field; 3 is typically sufficient",
            ),
            FloatParam(
                name="amplitude_mm",
                label="Squiggle Amplitude (mm)",
                min=0.0,
                max=20.0,
                step=0.5,
                default=3.0,
                visible_when={"mode": ["squiggle"]},
                description="Maximum amplitude of squiggle waves in millimeters — waves are larger in dark areas",
            ),
            FloatParam(
                name="frequency",
                label="Squiggle Frequency",
                min=0.1,
                max=50.0,
                step=0.5,
                default=5.0,
                visible_when={"mode": ["squiggle"]},
                description="Number of squiggle wave cycles per 100mm of horizontal distance",
            ),
            IntParam(
                name="wave_spread",
                label="Wave Spread",
                min=0,
                max=10,
                step=1,
                default=0,
                visible_when={"mode": ["squiggle"]},
                description="Number of neighboring lines above and below that are influenced by each line's wave amplitude — 0 means no spread (current behavior), higher values create smoother wave transitions between adjacent lines",
            ),
            ChoiceParam(
                name="line_spacing",
                label="Line Spacing Mode",
                choices=["Uniform", "Adaptive", "Grouped", "Adaptive + Grouped"],
                default="Uniform",
                visible_when={"mode": ["squiggle"]},
                description="Controls how scan-line Y positions are distributed — Uniform: evenly spaced (classic behavior); Adaptive: density follows image brightness (dark = dense); Grouped: lines cluster with gaps between groups; Adaptive + Grouped: grouped with brightness-adaptive inter-group gaps",
                choice_descriptions={
                    "Uniform": "Evenly spaced scan lines — classic squiggle behavior",
                    "Adaptive": "Line density follows image brightness — dark areas get more lines, bright areas fewer",
                    "Grouped": "Lines cluster into groups with larger gaps between groups for a sketchy, textured look",
                    "Adaptive + Grouped": "Groups whose spacing adapts to brightness — dark areas have denser groups",
                },
            ),
            FloatParam(
                name="min_spacing_mm",
                label="Min Spacing (mm)",
                min=0.1,
                max=10.0,
                step=0.1,
                default=0.5,
                visible_when={"mode": ["squiggle"], "line_spacing": ["Adaptive", "Adaptive + Grouped"]},
                description="Minimum spacing between scan lines in dark areas (mm) — controls the densest possible line density",
            ),
            FloatParam(
                name="max_spacing_mm",
                label="Max Spacing (mm)",
                min=0.5,
                max=20.0,
                step=0.5,
                default=5.0,
                visible_when={"mode": ["squiggle"], "line_spacing": ["Adaptive", "Adaptive + Grouped"]},
                description="Maximum spacing between scan lines in bright areas (mm) — controls the sparsest possible line density",
            ),
            IntParam(
                name="group_size",
                label="Group Size",
                min=2,
                max=10,
                step=1,
                default=3,
                visible_when={"mode": ["squiggle"], "line_spacing": ["Grouped", "Adaptive + Grouped"]},
                description="Number of lines per group — lines within a group are closely spaced, separated by a larger gap from the next group",
            ),
            FloatParam(
                name="group_gap_mm",
                label="Group Gap (mm)",
                min=1.0,
                max=20.0,
                step=0.5,
                default=4.0,
                visible_when={"mode": ["squiggle"], "line_spacing": ["Grouped"]},
                description="Gap between consecutive groups of scan lines (mm) — larger values create more pronounced clustering",
            ),
            FloatParam(
                name="group_intra_spacing_mm",
                label="Intra-Group Spacing (mm)",
                min=0.1,
                max=5.0,
                step=0.1,
                default=0.5,
                visible_when={"mode": ["squiggle"], "line_spacing": ["Grouped", "Adaptive + Grouped"]},
                description="Spacing between individual lines within a group (mm)",
            ),
            FloatParam(
                name="displacement_variation",
                label="Displacement Variation",
                min=0.0,
                max=1.0,
                step=0.1,
                default=0.0,
                visible_when={"mode": ["squiggle"]},
                description="Per-line random amplitude multiplier — 0 means all lines respond equally, 1 means some lines wave dramatically while neighbours stay flat; uses the random seed for reproducibility",
            ),
            BoolParam(
                name="skip_background",
                label="Skip Background",
                default=True,
                description="Suppress lines in near-white (background) areas — produces cleaner output on images with white backgrounds",
            ),
            FloatParam(
                name="bg_threshold",
                label="Background Threshold",
                min=0.0,
                max=255.0,
                step=1.0,
                default=240.0,
                visible_when={"skip_background": [True]},
                description="Brightness threshold above which pixels are treated as background (0–255)",
            ),
            IntParam(
                name="seed",
                label="Random Seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for reproducible line starting positions",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image (dark and bright areas swap roles)",
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
                description="Gaussian blur applied before processing — smooths the gradient field for more regular flow lines",
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
                name="Default Flow",
                params={
                    "mode": "flow",
                    "num_lines": 200,
                    "step_size_mm": 0.5,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 3.0,
                    "frequency": 5.0,
                    "wave_spread": 0,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Dense Flow",
                params={
                    # More lines with shorter max path to cover the canvas densely.
                    # Higher curvature_strength makes lines follow edges more tightly.
                    "mode": "flow",
                    "num_lines": 800,
                    "step_size_mm": 0.5,
                    "max_length_mm": 20.0,
                    "curvature_strength": 2.0,
                    "amplitude_mm": 3.0,
                    "frequency": 5.0,
                    "wave_spread": 0,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Portrait Flow",
                params={
                    # Moderate line count with high curvature so streamlines
                    # strongly follow facial contours; longer max_length_mm allows
                    # lines to wrap fully around features; contrast boost reveals
                    # soft facial transitions.
                    "mode": "flow",
                    "num_lines": 400,
                    "step_size_mm": 0.5,
                    "max_length_mm": 30.0,
                    "curvature_strength": 2.5,
                    "amplitude_mm": 3.0,
                    "frequency": 5.0,
                    "wave_spread": 0,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 7,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Classic Squiggle",
                params={
                    "mode": "squiggle",
                    "num_lines": 100,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 4.0,
                    "frequency": 8.0,
                    "wave_spread": 0,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Fine Squiggle",
                params={
                    "mode": "squiggle",
                    "num_lines": 200,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 2.0,
                    "frequency": 15.0,
                    "wave_spread": 0,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Dense Squiggle",
                params={
                    # Many tightly spaced scan lines with a large amplitude
                    # produce a heavily-filled look that works well for landscape
                    # photos and high-contrast illustrations.
                    "mode": "squiggle",
                    "num_lines": 400,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 5.0,
                    "frequency": 6.0,
                    "wave_spread": 0,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Squiggle / Portrait",
                params={
                    # Moderate line count with higher frequency and gentle wave
                    # spread for smooth transitions across facial brightness
                    # gradients.  wave_spread=2 ensures a single dark eyebrow
                    # influences its neighbouring scan lines, avoiding jarring
                    # amplitude steps at sharp boundaries.
                    "mode": "squiggle",
                    "num_lines": 150,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 3.5,
                    "frequency": 10.0,
                    "wave_spread": 2,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Squiggle / Landscape",
                params={
                    # More scan lines with a wide amplitude range to capture
                    # the horizon gradient; wave_spread=2 smooths the transition
                    # between sky and ground without blurring detail away.
                    "mode": "squiggle",
                    "num_lines": 250,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 4.0,
                    "frequency": 7.0,
                    "wave_spread": 2,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": False,
                    "bg_threshold": 240.0,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 5.0,
                    "blur_radius": 1.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Adaptive Density Squiggle",
                params={
                    # Line density adapts to brightness: dark areas get denser
                    # lines, light areas get sparse lines.  wave_spread=2 softens
                    # the spacing transitions at brightness boundaries.
                    "mode": "squiggle",
                    "num_lines": 150,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 4.0,
                    "frequency": 8.0,
                    "wave_spread": 2,
                    "line_spacing": "Adaptive",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 0.0,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Sketchy Grouped Strokes",
                params={
                    # Groups of 3 tightly spaced scan lines with wide gaps
                    # between clusters, like rough hatching.  High
                    # displacement_variation makes each line within a group
                    # wave at a different amplitude for a hand-drawn feel.
                    "mode": "squiggle",
                    "num_lines": 100,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 3.0,
                    "frequency": 6.0,
                    "wave_spread": 0,
                    "line_spacing": "Grouped",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 6.0,
                    "group_intra_spacing_mm": 0.8,
                    "displacement_variation": 0.6,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Organic Portrait",
                params={
                    # Adaptive + Grouped spacing for portrait subjects: dark
                    # shadow regions get dense clusters of 4 lines, bright
                    # highlights get widely spaced single clusters.
                    # wave_spread=3 and moderate displacement_variation produce
                    # smooth tonal gradients across facial contours.
                    "mode": "squiggle",
                    "num_lines": 150,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 3.5,
                    "frequency": 10.0,
                    "wave_spread": 3,
                    "line_spacing": "Adaptive + Grouped",
                    "min_spacing_mm": 0.3,
                    "max_spacing_mm": 4.0,
                    "group_size": 4,
                    "group_gap_mm": 3.0,
                    "group_intra_spacing_mm": 0.4,
                    "displacement_variation": 0.3,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 7,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Wild Lines",
                params={
                    # Few lines with maximum displacement_variation: every scan
                    # line has a wildly different amplitude, creating an energetic
                    # chaotic texture.  Low frequency gives broad, sweeping waves.
                    "mode": "squiggle",
                    "num_lines": 80,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 8.0,
                    "frequency": 4.0,
                    "wave_spread": 0,
                    "line_spacing": "Uniform",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 3,
                    "group_gap_mm": 4.0,
                    "group_intra_spacing_mm": 0.5,
                    "displacement_variation": 1.0,
                    "skip_background": False,
                    "bg_threshold": 240.0,
                    "seed": 13,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Loose Sketch",
                params={
                    # Two-line groups with large gaps and high displacement
                    # variation.  skip_background prevents lines in blank areas,
                    # so the subject floats on white paper with gestural marks.
                    "mode": "squiggle",
                    "num_lines": 100,
                    "step_size_mm": 1.0,
                    "max_length_mm": 20.0,
                    "curvature_strength": 1.0,
                    "amplitude_mm": 5.0,
                    "frequency": 5.0,
                    "wave_spread": 0,
                    "line_spacing": "Grouped",
                    "min_spacing_mm": 0.5,
                    "max_spacing_mm": 5.0,
                    "group_size": 2,
                    "group_gap_mm": 8.0,
                    "group_intra_spacing_mm": 1.0,
                    "displacement_variation": 0.8,
                    "skip_background": True,
                    "bg_threshold": 240.0,
                    "seed": 99,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
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
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
            to_grayscale,
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

        # Ensure grayscale
        source = to_grayscale(img)

        mode = str(params.get("mode", "flow"))
        num_lines = int(params.get("num_lines", 200))
        step_size_mm = float(params.get("step_size_mm", 0.5))
        max_length_mm = float(params.get("max_length_mm", 20.0))
        curvature_strength = float(params.get("curvature_strength", 1.0))
        amplitude_mm = float(params.get("amplitude_mm", 3.0))
        frequency = float(params.get("frequency", 5.0))
        wave_spread = int(params.get("wave_spread", 0))
        seed = int(params.get("seed", 42))
        skip_background = bool(params.get("skip_background", True))
        bg_threshold = float(params.get("bg_threshold", 240.0))
        line_spacing = str(params.get("line_spacing", "Uniform"))
        min_spacing_mm = float(params.get("min_spacing_mm", 0.5))
        max_spacing_mm = float(params.get("max_spacing_mm", 5.0))
        group_size = int(params.get("group_size", 3))
        group_gap_mm = float(params.get("group_gap_mm", 4.0))
        group_intra_spacing_mm = float(params.get("group_intra_spacing_mm", 0.5))
        displacement_variation = float(params.get("displacement_variation", 0.0))

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

        if mode == "squiggle":
            result = _generate_squiggle(
                source, num_lines, amplitude_mm, frequency, wave_spread,
                skip_background, bg_threshold,
                canvas, cancelled_callback, progress_callback,
                img_rect=img_rect,
                line_spacing=line_spacing,
                min_spacing_mm=min_spacing_mm,
                max_spacing_mm=max_spacing_mm,
                group_size=group_size,
                group_gap_mm=group_gap_mm,
                group_intra_spacing_mm=group_intra_spacing_mm,
                displacement_variation=displacement_variation,
                seed=seed,
            )
        else:
            vector_field = str(params.get("vector_field", "Edge Flow (ETF)"))
            etf_kernel_radius = float(params.get("etf_kernel_radius", 5.0))
            etf_iterations = int(params.get("etf_iterations", 3))
            result = _generate_flow_streamlines(
                source, num_lines, step_size_mm, max_length_mm, curvature_strength, seed,
                skip_background, bg_threshold,
                canvas, cancelled_callback, progress_callback,
                img_rect=img_rect,
                vector_field=vector_field,
                etf_kernel_radius=etf_kernel_radius,
                etf_iterations=etf_iterations,
            )

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
        return result
