"""Fast Marching Method helpers for ContourGenerator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators._helpers import _px_to_mm
from plottter.models import Polyline

from ._smoothing import _chaikin_smooth


def _compute_fmm_field(
    gray: np.ndarray,
    source_point: str,
    gamma: float,
    speed_floor: float,
    speed_map_override: np.ndarray | None = None,
    source_x_pct: float = 50.0,
    source_y_pct: float = 50.0,
) -> tuple[np.ndarray, int, int]:
    """Compute the FMM travel-time field from a speed map derived from ``gray``.

    Returns ``(T, sy, sx)`` where ``T`` is the clamped travel-time array
    (finite, shape H×W float64) and ``(sy, sx)`` is the source pixel location.

    Used by ``_trace_fmm_topographic`` and the alternative render modes
    (Displacement, Wave, Radial) so the expensive FMM computation is not
    duplicated across render paths.
    """
    h, w = gray.shape

    # Build speed map
    if speed_map_override is not None:
        normalized = speed_map_override.astype(np.float64)
        max_val = normalized.max()
        if max_val > 1.0:
            normalized = normalized / max_val
    else:
        normalized = gray.astype(np.float64) / 255.0
    if gamma != 1.0:
        speed = np.power(normalized, gamma)
    else:
        speed = normalized.copy()
    speed = np.maximum(speed, speed_floor)

    # Determine source pixel
    if source_point == "Center":
        sy, sx = h // 2, w // 2
    else:
        sx = int(np.clip(source_x_pct / 100.0 * (w - 1), 0, w - 1))
        sy = int(np.clip(source_y_pct / 100.0 * (h - 1), 0, h - 1))

    # Compute T via skfmm or scipy fallback
    T: np.ndarray | None = None
    try:
        import skfmm  # type: ignore[import]
        phi = np.ones((h, w), dtype=np.float64)
        phi[sy, sx] = -1.0
        T = np.asarray(skfmm.travel_time(phi, speed, dx=1.0), dtype=np.float64)
    except ImportError:
        pass

    if T is None:
        from scipy.ndimage import distance_transform_edt
        source_mask = np.zeros((h, w), dtype=bool)
        source_mask[sy, sx] = True
        dist = distance_transform_edt(~source_mask).astype(np.float64)
        T = dist * (1.0 - normalized + speed_floor)

    # Clamp inf/nan to 99.99th percentile
    finite_vals = T[np.isfinite(T)]
    if len(finite_vals) == 0:
        return np.zeros((h, w), dtype=np.float64), sy, sx
    T_min = float(np.min(finite_vals))
    T_max = float(np.percentile(finite_vals, 99.99))
    if T_max <= T_min:
        T_max = T_min + 1.0
    T = np.clip(T, T_min, T_max)

    return T, sy, sx


def _trace_fmm_topographic(
    gray: np.ndarray,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    num_contours: int,
    source_point: str,
    gamma: float,
    speed_floor: float,
    contour_spacing: str,
    min_contour_length_mm: float,
    simplify_tol: float,
    smooth_iterations: int,
    progress_callback: Any = None,
    cancelled_callback: Any = None,
    speed_map_override: np.ndarray | None = None,
    source_x_pct: float = 50.0,
    source_y_pct: float = 50.0,
) -> list[Polyline]:
    """Generate topographic contour lines via the Fast Marching Method.

    Treats the grayscale image as a speed map (dark pixels = slow propagation,
    light pixels = fast propagation).  Runs FMM from a source point to compute
    a travel-time field ``T``, then extracts isocontours at evenly spaced
    (or logarithmic/quadratic) levels.  Lines naturally bunch up where the wave
    moves slowly (dark image regions) and spread apart where it moves fast.

    Uses ``scikit-fmm.travel_time`` when available.  Falls back to
    ``scipy.ndimage.distance_transform_edt`` (plain Euclidean, modulated by the
    inverted image) when ``scikit-fmm`` is not installed.

    Parameters
    ----------
    gray:                  uint8 grayscale image (H×W).
    img_w, img_h:          Source image dimensions in pixels.
    draw_x1..y2:           Canvas drawing area bounds in mm.
    num_contours:          Number of isocontour levels to extract.
    source_point:          "Center" or "Custom" — wave source location.
    gamma:                 Gamma curve applied to the speed map; >1 concentrates
                           lines in dark regions, <1 in light regions.
    speed_floor:           Minimum speed value to prevent T=∞ in black areas.
    contour_spacing:       "Linear", "Logarithmic", or "Quadratic" distribution
                           of contour levels across the T range.
    min_contour_length_mm: Minimum polyline length in mm; shorter contours are
                           discarded as noise.
    simplify_tol:          RDP simplification tolerance in mm (cv2 path only).
    smooth_iterations:     Number of Chaikin smoothing passes.
    progress_callback:     Optional callable(percent: int).
    cancelled_callback:    Optional callable() → bool.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ContourGenerator.")

    h, w = gray.shape

    # --- Steps 1-4: Compute FMM travel-time field (shared with render modes) ---
    T, _sy, _sx = _compute_fmm_field(
        gray, source_point, gamma, speed_floor, speed_map_override, source_x_pct, source_y_pct
    )
    T_min = float(np.min(T))
    T_max = float(np.max(T))
    if T_max <= T_min:
        return []

    # --- Step 5: Compute contour levels ---
    n = max(2, num_contours)
    if contour_spacing == "Logarithmic":
        t_start = max(T_min + (T_max - T_min) * 0.001, 1e-9)
        levels = list(np.logspace(np.log10(t_start), np.log10(T_max), n))
    elif contour_spacing == "Quadratic":
        t_arr = np.linspace(0.0, 1.0, n)
        levels = list(T_min + (T_max - T_min) * t_arr ** 2)
    else:
        # Linear: exclude the very min/max endpoints to avoid degenerate contours
        levels = list(np.linspace(T_min, T_max, n + 2)[1:-1])

    # --- Step 6: Compute scaling factors for length filtering ---
    draw_w = draw_x2 - draw_x1
    draw_h_mm = draw_y2 - draw_y1
    px_per_mm_x = img_w / draw_w if draw_w > 0 else 1.0
    px_per_mm_y = img_h / draw_h_mm if draw_h_mm > 0 else 1.0
    px_per_mm = (px_per_mm_x + px_per_mm_y) / 2.0
    min_len_px = max(3, int(min_contour_length_mm * px_per_mm))

    # --- Step 7: Extract contours at each level ---
    # Normalise T to uint8 for cv2 threshold-based contour extraction.
    # Range [1, 254] reserves 0 and 255 so that threshold levels map cleanly.
    T_norm = np.clip(
        ((T - T_min) / (T_max - T_min) * 253.0 + 1.0),
        1,
        254,
    ).astype(np.uint8)

    all_polylines: list[Polyline] = []
    used_level_norms: set[int] = set()

    # Try skimage for cleaner sub-pixel contour extraction
    _skimage_find_contours = None
    try:
        from skimage.measure import find_contours as _ski_fc  # type: ignore[import]
        _skimage_find_contours = _ski_fc
    except ImportError:
        pass

    total = len(levels)
    for i, level in enumerate(levels):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback and total > 0:
            progress_callback(10 + int(i / total * 85))

        if _skimage_find_contours is not None:
            # skimage path: clean sub-pixel contours directly from float T
            contour_list = _skimage_find_contours(T, float(level))
            for contour in contour_list:
                if len(contour) < min_len_px:
                    continue
                # contour is Nx2 array of (row, col)
                poly: Polyline = [
                    _px_to_mm(float(c[1]), float(c[0]), img_w, img_h,
                              draw_x1, draw_y1, draw_x2, draw_y2)
                    for c in contour
                ]
                poly_len = sum(
                    math.sqrt(
                        (poly[j + 1][0] - poly[j][0]) ** 2
                        + (poly[j + 1][1] - poly[j][1]) ** 2
                    )
                    for j in range(len(poly) - 1)
                )
                if poly_len < min_contour_length_mm:
                    continue
                if smooth_iterations > 0:
                    poly = _chaikin_smooth(poly, smooth_iterations, closed=False)
                if len(poly) >= 2:
                    all_polylines.append(poly)
        else:
            # cv2 fallback: threshold the normalised T field and trace boundaries
            level_norm = int(
                (float(level) - T_min) / (T_max - T_min) * 253.0 + 1.0
            )
            level_norm = max(1, min(253, level_norm))
            # Skip duplicate normalised levels to avoid identical contours
            if level_norm in used_level_norms:
                continue
            used_level_norms.add(level_norm)

            _, binary = cv2.threshold(T_norm, level_norm, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                if len(contour) < min_len_px:
                    continue
                if simplify_tol > 0:
                    tol_px = max(1.0, simplify_tol * px_per_mm)
                    contour = cv2.approxPolyDP(contour, tol_px, closed=True)
                pts = contour.reshape(-1, 2)
                if len(pts) < 2:
                    continue
                poly = [
                    _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                              draw_x1, draw_y1, draw_x2, draw_y2)
                    for p in pts
                ]
                poly_len = sum(
                    math.sqrt(
                        (poly[j + 1][0] - poly[j][0]) ** 2
                        + (poly[j + 1][1] - poly[j][1]) ** 2
                    )
                    for j in range(len(poly) - 1)
                )
                if poly_len < min_contour_length_mm:
                    continue
                if smooth_iterations > 0:
                    poly = _chaikin_smooth(poly, smooth_iterations, closed=True)
                if poly and poly[0] != poly[-1]:
                    poly.append(poly[0])
                if len(poly) >= 2:
                    all_polylines.append(poly)

    return all_polylines


def _fmm_displacement(
    T: np.ndarray,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    num_lines: int,
    displacement_mm: float,
    line_angle_deg: float,
    progress_callback: Any = None,
    cancelled_callback: Any = None,
) -> list[Polyline]:
    """Render the FMM travel-time field as displaced parallel lines.

    Draws ``num_lines`` parallel lines at ``line_angle_deg`` degrees from
    horizontal.  At each sample point, the T value is normalised to [0, 1]
    and used to displace the point perpendicularly to the line direction by
    up to ``displacement_mm`` millimetres.  Lines in dark (slow) image regions
    are strongly displaced while lines in bright regions are nearly straight,
    creating an engraving / relief effect.

    Parameters
    ----------
    T:               FMM travel-time field (H×W float64, finite, clamped).
    img_w, img_h:    Source image dimensions in pixels.
    draw_x1..y2:     Canvas drawing area bounds in mm.
    num_lines:       Number of parallel lines to draw.
    displacement_mm: Maximum perpendicular displacement in mm.
    line_angle_deg:  Angle of the parallel lines in degrees (0 = horizontal).
    """
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1
    if draw_w <= 0 or draw_h <= 0 or num_lines < 1:
        return []

    T_min = float(np.min(T))
    T_max = float(np.max(T))
    T_range = T_max - T_min
    if T_range <= 0:
        return []

    angle_rad = math.radians(line_angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Perpendicular direction (90° CCW from line direction)
    perp_x = -sin_a
    perp_y = cos_a

    # Canvas centre in mm
    cx = (draw_x1 + draw_x2) / 2.0
    cy = (draw_y1 + draw_y2) / 2.0

    # Half-extent in the along-line and perpendicular directions
    half_along = (draw_w * abs(cos_a) + draw_h * abs(sin_a)) / 2.0
    half_perp = (draw_w * abs(sin_a) + draw_h * abs(cos_a)) / 2.0

    # Samples per line — enough to match pixel resolution
    num_samples = max(100, img_w * 2)

    polylines: list[Polyline] = []

    for i in range(num_lines):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback and num_lines > 1:
            progress_callback(10 + int(85 * i / (num_lines - 1)))

        t = i / (num_lines - 1) if num_lines > 1 else 0.5
        # Base position in the perpendicular direction from canvas centre
        perp_offset = -half_perp + t * 2.0 * half_perp
        base_x = cx + perp_offset * perp_x
        base_y = cy + perp_offset * perp_y

        polyline: Polyline = []
        for j in range(num_samples):
            s = -half_along + (j / (num_samples - 1)) * 2.0 * half_along if num_samples > 1 else 0.0
            # Sample point on the un-displaced line
            px_mm = base_x + s * cos_a
            py_mm = base_y + s * sin_a

            # Convert to pixel coordinates
            px_idx = (px_mm - draw_x1) / draw_w * img_w
            py_idx = (py_mm - draw_y1) / draw_h * img_h

            # Skip if outside image
            if px_idx < 0 or px_idx >= img_w or py_idx < 0 or py_idx >= img_h:
                if polyline:
                    polylines.append(polyline)
                    polyline = []
                continue

            # Bilinear sample of T
            px0 = int(px_idx)
            py0 = int(py_idx)
            px1 = min(px0 + 1, img_w - 1)
            py1 = min(py0 + 1, img_h - 1)
            fx = px_idx - px0
            fy = py_idx - py0
            T_val = float(
                T[py0, px0] * (1 - fx) * (1 - fy)
                + T[py0, px1] * fx * (1 - fy)
                + T[py1, px0] * (1 - fx) * fy
                + T[py1, px1] * fx * fy
            )

            T_norm = (T_val - T_min) / T_range
            disp = T_norm * displacement_mm

            # Displaced point
            final_x = px_mm + disp * perp_x
            final_y = py_mm + disp * perp_y
            polyline.append((final_x, final_y))

        if len(polyline) >= 2:
            polylines.append(polyline)

    return polylines


def _compute_fmm_wave_y_positions(
    grad_mag_norm: np.ndarray,
    img_w: int,
    img_h: int,
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
    """Compute Y positions for FMM wave scan lines based on spacing mode.

    Unlike squiggle mode (which uses image brightness), adaptive spacing here
    uses the horizontal mean of the gradient magnitude: high gradient → dense
    lines (waves are large in those regions), low gradient → sparse lines.
    """
    if line_spacing == "Uniform":
        return [draw_y1 + (i + 0.5) / num_lines * draw_h for i in range(num_lines)]

    # Vertical gradient profile: horizontal mean of gradient magnitude per row.
    vert_profile: np.ndarray = grad_mag_norm.mean(axis=1).astype(np.float32)

    def _grad_at_y(y_mm: float) -> float:
        """Sample the vertical gradient profile at a given mm Y coordinate (0–1)."""
        py = (y_mm - draw_y1) / draw_h * img_h
        py = max(0.0, min(img_h - 1.0, py))
        r0 = int(py)
        r1 = min(r0 + 1, img_h - 1)
        frac = py - r0
        return float(vert_profile[r0]) * (1.0 - frac) + float(vert_profile[r1]) * frac

    def _adaptive_spacing(grad: float) -> float:
        """Map gradient magnitude (0=flat, 1=steep) to spacing (high grad → dense)."""
        # Invert: high gradient → small spacing (dense), low gradient → large spacing.
        t = max(0.0, min(1.0, 1.0 - grad))
        return min_spacing_mm + t * (max_spacing_mm - min_spacing_mm)

    y_positions: list[float] = []

    if line_spacing == "Adaptive":
        y = draw_y1
        while y <= draw_y2:
            y_positions.append(y)
            spacing = _adaptive_spacing(_grad_at_y(y))
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
            grad = _grad_at_y(group_start)
            inter_gap = _adaptive_spacing(grad)
            inter_gap = max(min_spacing_mm, inter_gap)
            for j in range(group_size):
                y = group_start + j * group_intra_spacing_mm
                if y <= draw_y2:
                    y_positions.append(y)
            group_start += (group_size - 1) * group_intra_spacing_mm + inter_gap

    return y_positions


def _fmm_wave(
    T: np.ndarray,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    num_lines: int,
    amplitude_mm: float,
    frequency: float,
    line_spacing: str = "Uniform",
    min_spacing_mm: float = 0.5,
    max_spacing_mm: float = 5.0,
    group_size: int = 3,
    group_gap_mm: float = 4.0,
    group_intra_spacing_mm: float = 0.5,
    displacement_variation: float = 0.0,
    seed: int = 0,
    progress_callback: Any = None,
    cancelled_callback: Any = None,
) -> list[Polyline]:
    """Render the FMM travel-time field as gradient-modulated wave scan lines.

    Draws horizontal scan lines where the sinusoidal wave amplitude at each
    point is modulated by the gradient magnitude of T.  Where T changes
    rapidly (bunched contour region = dark image areas), waves are large;
    where T changes slowly, waves are flat.

    Parameters
    ----------
    T:                     FMM travel-time field (H×W float64, finite, clamped).
    img_w, img_h:          Source image dimensions in pixels.
    draw_x1..y2:           Canvas drawing area bounds in mm.
    num_lines:             Number of horizontal scan lines (Uniform mode).
    amplitude_mm:          Maximum wave amplitude in mm.
    frequency:             Wave cycles per canvas width.
    line_spacing:          Spacing mode: "Uniform", "Adaptive", "Grouped", "Adaptive + Grouped".
    min_spacing_mm:        Minimum line spacing (Adaptive modes).
    max_spacing_mm:        Maximum line spacing (Adaptive modes).
    group_size:            Lines per group (Grouped modes).
    group_gap_mm:          Gap between groups (Grouped modes).
    group_intra_spacing_mm: Spacing within a group (Grouped modes).
    displacement_variation: Per-line random amplitude multiplier range [0,1].
    seed:                  RNG seed for displacement_variation reproducibility.
    """
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1
    if draw_w <= 0 or draw_h <= 0 or num_lines < 1:
        return []

    # Compute gradient magnitude of T
    gy, gx = np.gradient(T.astype(np.float64))
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_max = float(grad_mag.max())
    if grad_max > 0:
        grad_mag_norm = (grad_mag / grad_max).astype(np.float32)
    else:
        grad_mag_norm = np.zeros_like(grad_mag, dtype=np.float32)

    num_samples = max(100, img_w * 2)

    # Compute Y positions based on spacing mode
    y_positions = _compute_fmm_wave_y_positions(
        grad_mag_norm,
        img_w,
        img_h,
        num_lines,
        draw_y1,
        draw_y2,
        draw_h,
        line_spacing,
        min_spacing_mm,
        max_spacing_mm,
        group_size,
        group_gap_mm,
        group_intra_spacing_mm,
    )

    total_lines = len(y_positions)
    rng = np.random.default_rng(seed)
    polylines: list[Polyline] = []

    for line_idx, y_base in enumerate(y_positions):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback and total_lines > 1:
            progress_callback(10 + int(85 * line_idx / (total_lines - 1)))

        # Fractional position of this line in the drawing area (for gradient sampling)
        t = (y_base - draw_y1) / draw_h if draw_h > 0 else 0.5
        t = max(0.0, min(1.0, t))

        # Per-line amplitude scale for displacement variation
        if displacement_variation > 0.0:
            disp_scale = 1.0 + displacement_variation * float(rng.uniform(-1.0, 1.0))
            disp_scale = max(0.1, min(2.0, disp_scale))
        else:
            disp_scale = 1.0

        polyline: Polyline = []
        for j in range(num_samples):
            s = j / (num_samples - 1) if num_samples > 1 else 0.0
            x_mm = draw_x1 + s * draw_w

            # Pixel coordinates for gradient sample
            px_idx = s * img_w
            py_idx = t * img_h
            px0 = int(min(max(px_idx, 0), img_w - 1))
            py0 = int(min(max(py_idx, 0), img_h - 1))
            px1 = min(px0 + 1, img_w - 1)
            py1 = min(py0 + 1, img_h - 1)
            fx = px_idx - int(px_idx)
            fy = py_idx - int(py_idx)

            # Bilinear sample of gradient magnitude
            g = float(
                grad_mag_norm[py0, px0] * (1 - fx) * (1 - fy)
                + grad_mag_norm[py0, px1] * fx * (1 - fy)
                + grad_mag_norm[py1, px0] * (1 - fx) * fy
                + grad_mag_norm[py1, px1] * fx * fy
            )

            local_amp = amplitude_mm * g * disp_scale
            wave_phase = 2.0 * math.pi * frequency * s
            y_final = y_base + local_amp * math.sin(wave_phase)
            polyline.append((x_mm, y_final))

        if len(polyline) >= 2:
            polylines.append(polyline)

    return polylines


def _fmm_radial(
    T: np.ndarray,
    source_sy: float,
    source_sx: float,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    num_radials: int,
    step_size_px: float,
    progress_callback: Any = None,
    cancelled_callback: Any = None,
) -> list[Polyline]:
    """Render the FMM travel-time field as radial lines following gradient ascent.

    Draws ``num_radials`` lines that start at the source point and step outward
    following the normalised gradient of T (perpendicular to isocontours).
    Lines spread apart in light/fast regions where T increases slowly, and bunch
    together in dark/slow regions where T increases steeply, producing a
    starburst / sunburst effect that warps naturally around image features.

    The source may lie outside the image (negative or beyond the dimensions);
    rays that point toward the frame then travel straight in until they enter
    it, after which they follow the field — a fan emanating from off-frame.

    Parameters
    ----------
    T:              FMM travel-time field (H×W float64, finite, clamped).
    source_sy, sx:  Source point row and column in pixels (may be off-image).
    img_w, img_h:   Source image dimensions in pixels.
    draw_x1..y2:    Canvas drawing area bounds in mm.
    num_radials:    Number of radial lines evenly distributed around 360°.
    step_size_px:   Step size in pixels per iteration.
    """
    if num_radials < 1 or step_size_px <= 0:
        return []

    # Gradient of T: gy = row-direction, gx = col-direction
    gy, gx = np.gradient(T.astype(np.float64))

    # Step budget: enough to travel from the (possibly off-image) source to the
    # farthest corner, then meander across the diagonal while following T.
    diag = math.sqrt(img_w ** 2 + img_h ** 2)
    far = max(
        math.hypot(source_sx - cx, source_sy - cy)
        for cx, cy in ((0, 0), (img_w, 0), (0, img_h), (img_w, img_h))
    )
    max_steps = int((far + diag) / max(step_size_px, 0.1)) + 10
    max_steps = min(max_steps, 20000)

    polylines: list[Polyline] = []

    for i in range(num_radials):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback and num_radials > 1:
            progress_callback(10 + int(85 * i / (num_radials - 1)))

        angle = 2.0 * math.pi * i / num_radials
        # Start slightly offset from source so initial gradient is defined
        x = float(source_sx) + 0.5 * math.cos(angle)
        y = float(source_sy) + 0.5 * math.sin(angle)

        polyline: Polyline = []
        # Initial direction is the ray's angle — used to coast straight in from
        # an off-image source until it enters the frame; overridden by the
        # gradient once inside.
        ux, uy = math.cos(angle), math.sin(angle)
        entered = False

        for _ in range(max_steps):
            in_bounds = 0.0 <= x < img_w and 0.0 <= y < img_h
            if not in_bounds:
                if entered:
                    break  # left the image after entering — ray is done
                # Not in the frame yet: keep coasting straight toward it.
                x += ux * step_size_px
                y += uy * step_size_px
                continue
            entered = True

            # Record current position in mm
            mm_x, mm_y = _px_to_mm(x, y, img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2)
            polyline.append((mm_x, mm_y))

            # Bilinear gradient at (y, x)
            y0, x0 = int(y), int(x)
            y1 = min(y0 + 1, img_h - 1)
            x1 = min(x0 + 1, img_w - 1)
            y0 = max(y0, 0)
            x0 = max(x0, 0)
            fy = y - y0
            fx = x - x0

            gx_val = float(
                gx[y0, x0] * (1 - fx) * (1 - fy)
                + gx[y0, x1] * fx * (1 - fy)
                + gx[y1, x0] * (1 - fx) * fy
                + gx[y1, x1] * fx * fy
            )
            gy_val = float(
                gy[y0, x0] * (1 - fx) * (1 - fy)
                + gy[y0, x1] * fx * (1 - fy)
                + gy[y1, x0] * (1 - fx) * fy
                + gy[y1, x1] * fx * fy
            )

            mag = math.sqrt(gx_val ** 2 + gy_val ** 2)
            reversed_dir = (ux != 0.0 or uy != 0.0) and (gx_val * ux + gy_val * uy) < 0.0
            if mag < 1e-6 or reversed_dir:
                # The gradient died (a plateau) or reversed (the radial reached
                # a ridge / local maximum of the travel-time field and would
                # otherwise oscillate in place). Nudge one step *outward* from
                # the source to get past it, then resume following the gradient
                # on the next iteration — so the line keeps picking up the next
                # image feature instead of stopping short or running dead
                # straight to the edge.
                ox = x - float(source_sx)
                oy = y - float(source_sy)
                omag = math.sqrt(ox * ox + oy * oy)
                if omag < 1e-6:
                    break  # at the source with no usable direction
                ux, uy = ox / omag, oy / omag
            else:
                ux = gx_val / mag
                uy = gy_val / mag

            # Step along gradient ascent of T (or straight, once coasting)
            x += ux * step_size_px
            y += uy * step_size_px

        if len(polyline) >= 2:
            polylines.append(polyline)

    return polylines
