"""ScanlineHalftoneGenerator — parallel scan lines with brightness-based thickness variation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.generators.base import (
    BoolParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _clip_line_to_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rx1: float,
    ry1: float,
    rx2: float,
    ry2: float,
) -> tuple[float, float, float, float] | None:
    """Clip a line segment to an axis-aligned rectangle using Liang-Barsky.

    Returns the clipped (x0, y0, x1, y1) or None if the segment is entirely
    outside the rectangle.
    """
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0

    for p, q in (
        (-dx, x0 - rx1),
        (dx, rx2 - x0),
        (-dy, y0 - ry1),
        (dy, ry2 - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
        elif p < 0.0:
            r = q / p
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            r = q / p
            if r < t0:
                return None
            if r < t1:
                t1 = r

    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)



def _generate_scanlines_with_thickness(
    img_rect: tuple[float, float, float, float],
    angle_deg: float,
    spacing_mm: float,
    gray: np.ndarray,
    max_thickness: int,
    pen_width_mm: float,
    sample_interval_mm: float,
    tone_gamma: float,
    skip_white: bool = False,
    white_threshold: int = 240,
    edge_dist_map: "np.ndarray | None" = None,
    edge_sensitivity: float = 0.0,
    cancelled_callback: Any = None,
    progress_callback: Any = None,
) -> list[Polyline]:
    """Generate scan lines with brightness-based thickness variation.

    The center line is drawn across each scan line (segmented when skip_white is
    True to omit bright regions).  For each offset level k (1..max_thickness),
    parallel offset lines at ±k × pen_width_mm are drawn in segments where the
    local brightness maps to a continuous thickness >= k.

    Thickness is linearly interpolated between brightness-sample points, so
    transitions from lit to un-lit regions taper smoothly rather than stepping.
    All generated polylines (center and offset) are clipped to the drawing area.

    Parameters
    ----------
    img_rect:
        ``(x1, y1, x2, y2)`` — drawing area in mm.
    angle_deg:
        Rotation angle of scan lines. 0 = horizontal, 90 = vertical.
    spacing_mm:
        Distance between adjacent scan lines in mm.
    gray:
        Grayscale uint8 image (H × W). 0 = black, 255 = white.
    max_thickness:
        Maximum number of parallel offset lines per side in darkest areas.
    pen_width_mm:
        Spacing between parallel offset lines in mm.
    sample_interval_mm:
        How often to sample brightness along each scan line in mm.
    tone_gamma:
        Power curve applied to brightness: higher values emphasize dark areas.
    skip_white:
        When True, skip drawing lines (including the center line) in segments
        where the sampled brightness exceeds ``white_threshold``.  This avoids
        faint strokes covering white backgrounds.
    white_threshold:
        Brightness level (0–255) above which lines are skipped when
        ``skip_white`` is True.
    edge_dist_map:
        Optional float32 array (same shape as *gray*) where each value is the
        Euclidean distance (in pixels) to the nearest edge pixel, as produced by
        ``cv2.distanceTransform``.  When provided and *edge_sensitivity* > 0,
        the effective thickness is reduced near edges so thick bands do not
        blur across important image features.
    edge_sensitivity:
        Blend factor in [0, 1] controlling how aggressively thickness is reduced
        near edges.  0 = disabled (thickness unchanged), 1 = fully clamp
        thickness to the edge-distance limit.
    cancelled_callback:
        Optional callable returning True if generation should stop.
    progress_callback:
        Optional callable accepting an int 0–100.
    """
    draw_x1, draw_y1, draw_x2, draw_y2 = img_rect
    img_h, img_w = gray.shape[:2]

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    cx_rect = (draw_x1 + draw_x2) / 2.0
    cy_rect = (draw_y1 + draw_y2) / 2.0

    # Extent of the drawing area in perpendicular / parallel directions
    corners_rel = [
        (draw_x1 - cx_rect, draw_y1 - cy_rect),
        (draw_x2 - cx_rect, draw_y1 - cy_rect),
        (draw_x1 - cx_rect, draw_y2 - cy_rect),
        (draw_x2 - cx_rect, draw_y2 - cy_rect),
    ]
    perp_coords = [-x * sin_a + y * cos_a for x, y in corners_rel]
    para_coords = [x * cos_a + y * sin_a for x, y in corners_rel]

    perp_min = min(perp_coords)
    perp_max = max(perp_coords)
    para_min = min(para_coords) - spacing_mm
    para_max = max(para_coords) + spacing_mm

    perp_pos = math.ceil(perp_min / spacing_mm) * spacing_mm
    total_range = perp_max - perp_min
    max_lines = int(total_range / max(spacing_mm, 0.001)) + 2

    # Perpendicular unit direction (direction of parallel offsets)
    perp_dir_x = -sin_a
    perp_dir_y = cos_a

    # Pixel-space scale factors for world-mm → pixel-index conversion
    rect_w = max(draw_x2 - draw_x1, 1e-6)
    rect_h = max(draw_y2 - draw_y1, 1e-6)
    mm_to_px_x = img_w / rect_w
    mm_to_px_y = img_h / rect_h

    # Pre-compute pen width in pixels for edge-sensitivity normalisation.
    # Use the mean of x and y scale factors as an approximation.
    pen_width_px = pen_width_mm * (mm_to_px_x + mm_to_px_y) / 2.0

    polylines: list[Polyline] = []
    line_count = 0

    while perp_pos <= perp_max:
        if cancelled_callback and cancelled_callback():
            break
        if line_count > max_lines:
            break

        if progress_callback and line_count % 20 == 0:
            frac = min(1.0, (perp_pos - perp_min) / max(total_range, 1e-6))
            progress_callback(int(frac * 100))

        # World-space origin of this scan line (on the perpendicular axis)
        origin_x = cx_rect + perp_pos * perp_dir_x
        origin_y = cy_rect + perp_pos * perp_dir_y

        # Full extent of the line before clipping
        x0 = origin_x + para_min * cos_a
        y0 = origin_y + para_min * sin_a
        x1 = origin_x + para_max * cos_a
        y1 = origin_y + para_max * sin_a

        # Clip center line to drawing area
        clipped = _clip_line_to_rect(x0, y0, x1, y1, draw_x1, draw_y1, draw_x2, draw_y2)
        if clipped is None:
            perp_pos += spacing_mm
            line_count += 1
            continue

        cx0, cy0, cx1, cy1 = clipped
        if math.hypot(cx1 - cx0, cy1 - cy0) < 1e-4:
            perp_pos += spacing_mm
            line_count += 1
            continue

        # Convert clipped endpoints back to parametric along-line coordinates
        para_start = (cx0 - origin_x) * cos_a + (cy0 - origin_y) * sin_a
        para_end = (cx1 - origin_x) * cos_a + (cy1 - origin_y) * sin_a
        line_len = para_end - para_start
        if line_len <= 0.0:
            perp_pos += spacing_mm
            line_count += 1
            continue

        # Sample positions along the clipped scan line
        n_samples = max(2, int(math.ceil(line_len / sample_interval_mm)) + 1)
        sample_paras = np.linspace(para_start, para_end, n_samples)

        # World positions of each sample point (on the center line)
        sample_xs = origin_x + sample_paras * cos_a
        sample_ys = origin_y + sample_paras * sin_a

        # Map to pixel indices (nearest-neighbour, clamped to image bounds)
        px_xs = np.clip(
            ((sample_xs - draw_x1) * mm_to_px_x).astype(int), 0, img_w - 1
        )
        px_ys = np.clip(
            ((sample_ys - draw_y1) * mm_to_px_y).astype(int), 0, img_h - 1
        )

        # Sample brightness and compute per-sample continuous thickness values.
        # Dark pixel (0) → max_thickness; bright pixel (255) → 0.
        brightnesses = gray[px_ys, px_xs].astype(float)
        thicknesses = max_thickness * (1.0 - (brightnesses / 255.0) ** tone_gamma)

        # Edge-aware thickness reduction: near detected edges, reduce thickness
        # so thick bands do not blur across important image boundaries.
        if edge_dist_map is not None and edge_sensitivity > 0.0 and pen_width_px > 0.0:
            for i in range(n_samples):
                edge_dist_px = float(edge_dist_map[px_ys[i], px_xs[i]])
                # Convert edge distance from pixels to "number of pen widths"
                edge_max_thick = edge_dist_px / pen_width_px
                reduced = min(thicknesses[i], edge_max_thick)
                thicknesses[i] = thicknesses[i] + edge_sensitivity * (reduced - thicknesses[i])

        # Center line: drawn across the full clipped extent, but when skip_white
        # is enabled, segments where brightness exceeds white_threshold are omitted.
        if not skip_white:
            # Default behaviour: always draw the full center line
            polylines.append([(cx0, cy0), (cx1, cy1)])
        else:
            # Build a segmented center line, skipping bright sample regions.
            # Walk sample-to-sample; emit a sub-polyline for consecutive
            # non-bright samples, clipped to the drawing area.
            center_poly: list[tuple[float, float]] | None = None
            for i in range(n_samples):
                bright = brightnesses[i]
                if bright > white_threshold:
                    if center_poly is not None and len(center_poly) >= 2:
                        polylines.append(center_poly)
                    center_poly = None
                else:
                    pt_x = float(sample_xs[i])
                    pt_y = float(sample_ys[i])
                    if center_poly is None:
                        center_poly = [(pt_x, pt_y)]
                    else:
                        center_poly.append((pt_x, pt_y))
            if center_poly is not None and len(center_poly) >= 2:
                polylines.append(center_poly)

        # No offset lines needed when max_thickness == 0
        if max_thickness < 1:
            perp_pos += spacing_mm
            line_count += 1
            continue

        # For each offset level k, build sub-polylines where interpolated thickness >= k.
        # For each side (+1 and -1), the offset is k * pen_width_mm in the perpendicular dir.
        for k in range(1, max_thickness + 1):
            for side in (-1, 1):
                offset_dist = k * pen_width_mm * side

                # current_poly accumulates points for the current active sub-polyline
                current_poly: list[tuple[float, float]] | None = None

                for i in range(1, n_samples):
                    thick_prev = thicknesses[i - 1]
                    thick_i = thicknesses[i]
                    active_prev = thick_prev >= k
                    active_i = thick_i >= k

                    # Skip if both samples are inactive
                    if not active_prev and not active_i:
                        if current_poly is not None and len(current_poly) >= 2:
                            polylines.append(current_poly)
                            current_poly = None
                        continue

                    # Compute the parametric range [t_start, t_end] ⊆ [0, 1]
                    # within the segment (sample i-1 → sample i) where offset is active.
                    t_start = 0.0
                    t_end = 1.0

                    if active_i and not active_prev:
                        # Entering active region: find where thickness crosses k
                        dt = thick_i - thick_prev
                        t_start = (k - thick_prev) / dt if abs(dt) > 1e-10 else 0.0
                        t_start = max(0.0, min(1.0, t_start))
                    elif not active_i and active_prev:
                        # Leaving active region: find where thickness drops below k
                        dt = thick_i - thick_prev
                        t_end = (k - thick_prev) / dt if abs(dt) > 1e-10 else 1.0
                        t_end = max(0.0, min(1.0, t_end))

                    # Center-line coordinates for this (possibly partial) segment
                    prev_x = float(sample_xs[i - 1])
                    prev_y = float(sample_ys[i - 1])
                    curr_x = float(sample_xs[i])
                    curr_y = float(sample_ys[i])

                    seg_cx0 = prev_x + t_start * (curr_x - prev_x)
                    seg_cy0 = prev_y + t_start * (curr_y - prev_y)
                    seg_cx1 = prev_x + t_end * (curr_x - prev_x)
                    seg_cy1 = prev_y + t_end * (curr_y - prev_y)

                    # Apply perpendicular offset to get the offset-line coordinates
                    ox0 = seg_cx0 + perp_dir_x * offset_dist
                    oy0 = seg_cy0 + perp_dir_y * offset_dist
                    ox1 = seg_cx1 + perp_dir_x * offset_dist
                    oy1 = seg_cy1 + perp_dir_y * offset_dist

                    # Clip the offset segment to the drawing area
                    seg_clipped = _clip_line_to_rect(
                        ox0, oy0, ox1, oy1, draw_x1, draw_y1, draw_x2, draw_y2
                    )

                    if seg_clipped is None:
                        # Segment is outside the drawing area — commit and reset
                        if current_poly is not None and len(current_poly) >= 2:
                            polylines.append(current_poly)
                        current_poly = None
                    else:
                        sx0, sy0, sx1, sy1 = seg_clipped

                        if current_poly is None:
                            current_poly = [(sx0, sy0), (sx1, sy1)]
                        else:
                            # Check continuity with the previous endpoint
                            px_last, py_last = current_poly[-1]
                            if abs(px_last - sx0) < 1e-4 and abs(py_last - sy0) < 1e-4:
                                current_poly.append((sx1, sy1))
                            else:
                                # Gap due to boundary clipping — commit and start new
                                if len(current_poly) >= 2:
                                    polylines.append(current_poly)
                                current_poly = [(sx0, sy0), (sx1, sy1)]

                    # If leaving active region, commit the polyline
                    if not active_i:
                        if current_poly is not None and len(current_poly) >= 2:
                            polylines.append(current_poly)
                        current_poly = None

                # Commit any remaining open polyline at the end of this scan line
                if current_poly is not None and len(current_poly) >= 2:
                    polylines.append(current_poly)

        perp_pos += spacing_mm
        line_count += 1

    if progress_callback:
        progress_callback(100)

    return polylines


@register_generator
class ScanlineHalftoneGenerator(Generator):
    """Parallel scan lines with brightness-based thickness variation for halftone effects."""

    name = "Scanline Halftone"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="line_spacing_mm",
                label="Line Spacing (mm)",
                min=0.5,
                max=10.0,
                step=0.1,
                default=2.0,
                description="Vertical distance between scan lines",
            ),
            FloatParam(
                name="angle_deg",
                label="Angle (deg)",
                min=-180.0,
                max=180.0,
                step=1.0,
                default=0.0,
                description="Rotation angle of scan lines — 0 is horizontal, 90 is vertical",
            ),
            IntParam(
                name="max_thickness",
                label="Max Thickness",
                min=0,
                max=10,
                step=1,
                default=4,
                description=(
                    "Maximum number of parallel lines per side in darkest areas — "
                    "total visual strokes = 1 + 2 × max_thickness"
                ),
            ),
            FloatParam(
                name="pen_width_mm",
                label="Pen Width (mm)",
                min=0.1,
                max=1.0,
                step=0.05,
                default=0.3,
                description="Pen tip width in mm — controls spacing between parallel offset lines",
            ),
            FloatParam(
                name="sample_interval_mm",
                label="Sample Interval (mm)",
                min=0.5,
                max=5.0,
                step=0.1,
                default=1.0,
                description=(
                    "How often to sample brightness along each line — "
                    "smaller values give more detail but more points"
                ),
            ),
            FloatParam(
                name="tone_gamma",
                label="Tone Gamma",
                min=0.5,
                max=3.0,
                step=0.1,
                default=1.5,
                description="Tone curve — higher values emphasize dark areas",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert image brightness before generating lines",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before generating lines (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before generating lines (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description="Gaussian blur radius applied to the image before generating lines (0 = no blur)",
            ),
            BoolParam(
                name="skip_white",
                label="Skip White Areas",
                default=True,
                description=(
                    "Remove scan line segments in very bright areas for a cleaner look — "
                    "avoids faint lines covering white backgrounds"
                ),
            ),
            IntParam(
                name="white_threshold",
                label="White Threshold",
                min=200,
                max=255,
                step=5,
                default=240,
                description=(
                    "Brightness above which lines are removed when 'Skip White Areas' is enabled "
                    "(0 = black, 255 = white)"
                ),
            ),
            FloatParam(
                name="edge_sensitivity",
                label="Edge Sensitivity",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.0,
                description=(
                    "Reduce line thickness near detected edges to preserve detail — "
                    "0 disables edge-awareness, 1 fully clamps thickness at edge boundaries"
                ),
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
                name="Newspaper",
                params={
                    "line_spacing_mm": 2.0,
                    "angle_deg": 0.0,
                    "max_thickness": 3,
                    "pen_width_mm": 0.3,
                    "sample_interval_mm": 1.0,
                    "tone_gamma": 1.2,
                    "skip_white": True,
                    "white_threshold": 240,
                    "edge_sensitivity": 0.0,
                },
            ),
            Preset(
                name="Engraving",
                params={
                    "line_spacing_mm": 1.2,
                    "angle_deg": 15.0,
                    "max_thickness": 5,
                    "pen_width_mm": 0.25,
                    "sample_interval_mm": 0.5,
                    "tone_gamma": 1.8,
                    "skip_white": True,
                    "white_threshold": 240,
                    "edge_sensitivity": 0.5,
                },
            ),
            Preset(
                name="Bold Poster",
                params={
                    "line_spacing_mm": 3.5,
                    "angle_deg": 0.0,
                    "max_thickness": 6,
                    "pen_width_mm": 0.4,
                    "sample_interval_mm": 1.0,
                    "tone_gamma": 2.0,
                    "skip_white": True,
                    "white_threshold": 220,
                    "edge_sensitivity": 0.0,
                },
            ),
            Preset(
                name="Fine Detail",
                params={
                    "line_spacing_mm": 1.0,
                    "angle_deg": 0.0,
                    "max_thickness": 2,
                    "pen_width_mm": 0.2,
                    "sample_interval_mm": 0.5,
                    "tone_gamma": 1.0,
                    "skip_white": False,
                    "white_threshold": 240,
                    "edge_sensitivity": 0.3,
                },
            ),
            Preset(
                name="Cross Scan",
                params={
                    "line_spacing_mm": 2.5,
                    "angle_deg": 30.0,
                    "max_thickness": 3,
                    "pen_width_mm": 0.3,
                    "sample_interval_mm": 1.0,
                    "tone_gamma": 1.5,
                    "skip_white": True,
                    "white_threshold": 240,
                    "edge_sensitivity": 0.0,
                },
            ),
            Preset(
                name="Vertical Blinds",
                params={
                    "line_spacing_mm": 2.0,
                    "angle_deg": 90.0,
                    "max_thickness": 4,
                    "pen_width_mm": 0.3,
                    "sample_interval_mm": 1.0,
                    "tone_gamma": 1.5,
                    "skip_white": True,
                    "white_threshold": 240,
                    "edge_sensitivity": 0.0,
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

        # Ensure grayscale for brightness sampling
        if source.ndim == 3:
            try:
                import cv2
                gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            except ImportError:
                gray = source.mean(axis=2).astype(np.uint8)
        else:
            gray = source.copy()

        # Apply shared image preprocessing
        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            gray = adjust_brightness(gray, brightness)
        if contrast != 0.0:
            gray = adjust_contrast(gray, contrast)
        if blur_radius > 0.0:
            gray = apply_blur(gray, blur_radius)
        if do_invert:
            gray = invert_image(gray)

        img_h, img_w = gray.shape[:2]

        spacing_mm = float(params.get("line_spacing_mm", 2.0))
        angle_deg = float(params.get("angle_deg", 0.0))
        max_thickness = int(params.get("max_thickness", 4))
        pen_width_mm = float(params.get("pen_width_mm", 0.3))
        sample_interval_mm = float(params.get("sample_interval_mm", 1.0))
        tone_gamma = float(params.get("tone_gamma", 1.5))

        if spacing_mm <= 0:
            spacing_mm = 0.1
        if pen_width_mm <= 0:
            pen_width_mm = 0.05
        if sample_interval_mm <= 0:
            sample_interval_mm = 0.1
        if tone_gamma <= 0:
            tone_gamma = 0.1

        skip_white = bool(params.get("skip_white", True))
        white_threshold = int(params.get("white_threshold", 240))
        edge_sensitivity = float(params.get("edge_sensitivity", 0.0))

        # Precompute edge distance map when edge_sensitivity is requested.
        # cv2.distanceTransform returns a float32 array where each pixel holds
        # the Euclidean distance (in pixels) to the nearest Canny edge pixel.
        edge_dist_map: np.ndarray | None = None
        if edge_sensitivity > 0.0:
            try:
                import cv2 as _cv2_edge
                edges = _cv2_edge.Canny(gray, 50, 150)
                # distanceTransform needs 0 = obstacle (edge), non-zero = free.
                not_edges = _cv2_edge.bitwise_not(edges)
                edge_dist_map = _cv2_edge.distanceTransform(
                    not_edges, _cv2_edge.DIST_L2, 5
                )
            except ImportError:
                edge_dist_map = None

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_rect = compute_image_rect(
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

        result = _generate_scanlines_with_thickness(
            img_rect,
            angle_deg,
            spacing_mm,
            gray,
            max_thickness,
            pen_width_mm,
            sample_interval_mm,
            tone_gamma,
            skip_white=skip_white,
            white_threshold=white_threshold,
            edge_dist_map=edge_dist_map,
            edge_sensitivity=edge_sensitivity,
            cancelled_callback=cancelled_callback,
            progress_callback=progress_callback,
        )

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result
