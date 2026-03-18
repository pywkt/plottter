"""ContourGenerator — topographic-map-style isoline contours from image brightness.

Traces closed contour lines at evenly spaced (or custom) brightness threshold values,
producing concentric contour rings around image features similar to topographic maps.

Also provides a "Line Art Trace" mode that bypasses multi-level thresholding and
instead directly traces the outlines of black regions in a binary image — optimised
for clean B&W line drawings where strokes should be traced faithfully.

A third "Skeleton" mode applies morphological skeletonization (center-line thinning)
to the binary image before tracing, producing single-stroke center-lines rather than
outline loops — useful for thin line art and handwriting.

A fourth "FMM Topographic" mode uses the Fast Marching Method to compute a
travel-time field from a source point across a speed map derived from the image.
Isocontours of the travel-time field are extracted and produce topographic lines that
naturally bunch up in dark (slow) regions and spread apart in light (fast) regions.
When scikit-fmm is not installed, falls back to a scipy.ndimage.distance_transform_edt
approximation.
"""

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
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _chaikin_smooth(
    poly: Polyline,
    iterations: int,
    closed: bool = False,
) -> Polyline:
    """Apply Chaikin's corner-cutting algorithm for smooth plotter-friendly curves.

    Each iteration doubles the number of points while smoothing corners.
    For open polylines the original endpoints are preserved.  For closed
    polylines (where the first and last point are the same) the algorithm
    wraps around and re-closes the result.

    Parameters
    ----------
    poly:       Input polyline.
    iterations: Number of refinement passes (0 = no change, 1–4 typical).
    closed:     True when the first and last points are the same closed loop.
    """
    if iterations <= 0 or len(poly) < 3:
        return poly

    # Strip the repeated closing point for closed curves so the algorithm
    # can wrap cleanly without creating duplicate entries mid-loop.
    if closed and len(poly) >= 2 and poly[0] == poly[-1]:
        result = list(poly[:-1])
        needs_close = True
    else:
        result = list(poly)
        needs_close = False

    for _ in range(iterations):
        new_pts: list[tuple[float, float]] = []
        n = len(result)
        if needs_close:
            # Closed curve: wrap around from last point back to first
            for i in range(n):
                p0 = result[i]
                p1 = result[(i + 1) % n]
                q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
                r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
                new_pts.extend([q, r])
        else:
            # Open curve: preserve the first and last endpoints exactly
            new_pts.append(result[0])
            for i in range(n - 1):
                p0 = result[i]
                p1 = result[i + 1]
                q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
                r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
                new_pts.extend([q, r])
            new_pts.append(result[-1])
        result = new_pts

    if needs_close and result:
        result.append(result[0])

    return result


def _trace_isolines(
    gray: np.ndarray,
    threshold: int,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    simplify_tol: float,
    min_length: int,
    smooth_iterations: int = 0,
) -> list[Polyline]:
    """Trace contour lines at a single brightness threshold using OpenCV."""
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ContourGenerator.")

    # Threshold the grayscale image to isolate the isoline
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours at this threshold level
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    polylines: list[Polyline] = []
    for contour in contours:
        if len(contour) < min_length:
            continue

        # Simplify using approxPolyDP (RDP-like)
        if simplify_tol > 0:
            # Convert simplify_tol from mm to pixels
            px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0
            tol_px = max(1.0, simplify_tol * px_per_mm)
            contour = cv2.approxPolyDP(contour, tol_px, closed=True)

        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue

        # Convert pixel coords to mm
        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts
        ]

        # Apply Chaikin smoothing before closing
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=True)

        # Close the contour
        if poly[0] != poly[-1]:
            poly.append(poly[0])

        polylines.append(poly)

    return polylines


def _trace_line_art(
    gray: np.ndarray,
    threshold: int,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    simplify_tol: float,
    min_length: int,
    smooth_iterations: int,
    adaptive_threshold: bool = False,
    adaptive_c: float = 5.0,
) -> list[Polyline]:
    """Trace line art by direct binary thresholding + full-resolution contour tracing.

    Unlike the multi-level Contour Levels mode, this function uses a single
    threshold to separate ink from background and then traces the outlines of
    every ink region at full pixel resolution.  This is optimised for clean
    B&W line drawings (logos, icons, sketches) where the strokes should be
    reproduced as faithfully as possible rather than approximated by brightness
    isolines.

    Key differences from Contour Levels:
    - Single threshold (THRESH_BINARY_INV so dark pixels become the region)
    - cv2.RETR_CCOMP to capture both outer and hole contours (inner stroke edges)
    - cv2.CHAIN_APPROX_NONE for maximum contour resolution before RDP
    - Optional Chaikin smoothing to convert pixel-stepped boundaries into
      smooth plotter-friendly curves
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ContourGenerator.")

    from plottter.generators._helpers import _apply_threshold

    # Invert threshold: pixels *darker* than threshold are treated as ink
    binary = _apply_threshold(
        gray, threshold, adaptive_threshold, adaptive_c, cv2.THRESH_BINARY_INV
    )

    # RETR_CCOMP returns outer boundaries + inner hole boundaries
    # CHAIN_APPROX_NONE preserves every pixel on the contour boundary
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    polylines: list[Polyline] = []
    for contour in contours:
        if len(contour) < min_length:
            continue

        # RDP simplification — gentler tolerance in line art mode
        if simplify_tol > 0:
            tol_px = max(0.5, simplify_tol * px_per_mm)
            contour = cv2.approxPolyDP(contour, tol_px, closed=True)

        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue

        # Convert pixel coords to mm
        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts
        ]

        # Chaikin smoothing: converts pixel-stepped outlines into smooth curves
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=True)

        # Close the contour (always needed — Chaikin with an open input stays open)
        if poly and poly[0] != poly[-1]:
            poly.append(poly[0])

        polylines.append(poly)

    return polylines


def _trace_skeleton(
    gray: np.ndarray,
    threshold: int,
    cleanup_kernel: int,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    simplify_tol: float,
    min_length: int,
    smooth_iterations: int,
    merge_gap_mm: float = 0.0,
    adaptive_threshold: bool = False,
    adaptive_c: float = 5.0,
) -> list[Polyline]:
    """Trace centerlines of thick ink strokes via morphological skeletonization.

    Pipeline:
    1. Threshold (THRESH_BINARY_INV): dark ink → white foreground.
    2. Optional morphological cleanup (MORPH_CLOSE then MORPH_OPEN).
    3. Skeletonize: reduce ink regions to single-pixel-wide centerlines.
    4. findContours on the skeleton: each connected skeleton component
       produces one contour that — after RDP simplification — approximates
       the centerline of the original thick stroke.
    5. Optional fragment merging: connect nearby endpoint pairs within
       ``merge_gap_mm`` to reduce disconnected short segments at junctions.
    6. Optional Chaikin smoothing.

    This produces approximately **one polyline per original thick line**,
    unlike Line Art Trace which traces both the inner and outer edge of
    each stroke (two polylines per line).

    Parameters
    ----------
    gray:            Grayscale uint8 image (H×W).
    threshold:       Brightness threshold (0–255); pixels ≤ threshold are ink.
    cleanup_kernel:  Morphological kernel size (0 = disabled, 1–10 = enabled).
    img_w, img_h:    Source image dimensions in pixels.
    draw_x1..y2:     Canvas drawing area bounds in mm.
    simplify_tol:    RDP simplification tolerance in mm.
    min_length:      Minimum contour length in pixels (shorter contours discarded).
    smooth_iterations: Number of Chaikin smoothing passes.
    merge_gap_mm:    Maximum endpoint distance (mm) for fragment merging
                     (0 = disabled).
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ContourGenerator.")

    from plottter.generators._helpers import _apply_threshold, _skeletonize

    # Step 1: threshold — dark ink (≤ threshold) → white foreground
    binary = _apply_threshold(
        gray, threshold, adaptive_threshold, adaptive_c, cv2.THRESH_BINARY_INV
    )

    # Step 2: optional morphological cleanup
    if cleanup_kernel >= 1:
        k = int(cleanup_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # MORPH_CLOSE fills small gaps in the ink; MORPH_OPEN removes noise specks
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 3: skeletonize — reduce white ink regions to 1-pixel-wide centerlines
    skeleton = _skeletonize(binary)

    # Step 4: trace contours of the skeleton.
    # findContours finds the boundary of each connected white component.
    # For 1-pixel-wide centerlines, the boundary of each component closely
    # approximates the skeleton segment.  After RDP simplification, these
    # collapse to the line's endpoints — roughly one polyline per original
    # thick stroke.
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    # Collect raw polylines before merging and smoothing
    raw_polylines: list[Polyline] = []
    for contour in contours:
        if len(contour) < min_length:
            continue

        if simplify_tol > 0:
            tol_px = max(0.5, simplify_tol * px_per_mm)
            contour = cv2.approxPolyDP(contour, tol_px, closed=True)

        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue

        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts
        ]

        if len(poly) >= 2:
            raw_polylines.append(poly)

    # Step 5: optional fragment merging — connect nearby endpoint pairs
    # before smoothing so that merged segments receive a single smooth pass.
    if merge_gap_mm > 0 and raw_polylines:
        from plottter.processing.merge import merge_fragments
        raw_polylines = merge_fragments(raw_polylines, merge_gap_mm)

    # Step 6: optional Chaikin smoothing applied to the (possibly merged) polylines
    polylines: list[Polyline] = []
    for poly in raw_polylines:
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=False)
        if len(poly) >= 2:
            polylines.append(poly)

    return polylines


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
    source_sy: int,
    source_sx: int,
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

    Draws ``num_radials`` lines that start at the FMM source point and step
    outward following the normalised gradient of T (perpendicular to isocontours).
    Lines spread apart in light/fast regions where T increases slowly, and bunch
    together in dark/slow regions where T increases steeply, producing a
    starburst / sunburst effect that warps naturally around image features.

    Parameters
    ----------
    T:              FMM travel-time field (H×W float64, finite, clamped).
    source_sy, sx:  Source point pixel row and column.
    img_w, img_h:   Source image dimensions in pixels.
    draw_x1..y2:    Canvas drawing area bounds in mm.
    num_radials:    Number of radial lines evenly distributed around 360°.
    step_size_px:   Step size in pixels per iteration.
    """
    if num_radials < 1 or step_size_px <= 0:
        return []

    # Gradient of T: gy = row-direction, gx = col-direction
    gy, gx = np.gradient(T.astype(np.float64))

    # Max steps: enough to reach the farthest corner from any interior source
    max_steps = int(math.sqrt(img_w ** 2 + img_h ** 2) / max(step_size_px, 0.1)) + 10
    max_steps = min(max_steps, 10000)

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

        for _ in range(max_steps):
            # Bounds check (with half-pixel margin)
            if x < 0.0 or x >= img_w or y < 0.0 or y >= img_h:
                break

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
            if mag < 1e-6:
                break

            # Step along gradient ascent of T
            x += (gx_val / mag) * step_size_px
            y += (gy_val / mag) * step_size_px

        if len(polyline) >= 2:
            polylines.append(polyline)

    return polylines


def _fill_polygon_hatch(
    polygon_pts: Polyline,
    hole_pts_list: list[Polyline],
    angle_deg: float,
    spacing_mm: float,
) -> list[Polyline]:
    """Fill a polygon with parallel hatch lines clipped to its interior.

    Uses Shapely to clip scan lines to the polygon (minus any holes).
    Returns a list of Polylines (line segments inside the polygon).

    Parameters
    ----------
    polygon_pts:    Outer boundary as a list of (x_mm, y_mm) points (closed).
    hole_pts_list:  List of hole boundaries (each a closed Polyline).
    angle_deg:      Angle of hatch lines in degrees (0 = horizontal).
    spacing_mm:     Distance between hatch lines in mm.
    """
    try:
        from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
        from shapely.ops import unary_union
    except ImportError:  # pragma: no cover
        return []

    if len(polygon_pts) < 3:
        return []

    # Build Shapely polygon with optional holes
    shell = [(p[0], p[1]) for p in polygon_pts]
    holes = [[(p[0], p[1]) for p in h] for h in hole_pts_list if len(h) >= 3]
    try:
        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return []
    except Exception:
        return []

    # Handle MultiPolygon (e.g. from self-intersecting contours normalised by
    # buffer(0)): process each sub-polygon independently and combine results.
    if poly.geom_type == "MultiPolygon":
        combined: list[Polyline] = []
        for sub_poly in poly.geoms:
            if sub_poly.is_empty:
                continue
            sub_pts: Polyline = [(float(c[0]), float(c[1])) for c in sub_poly.exterior.coords]
            sub_holes_pts: list[Polyline] = [
                [(float(c[0]), float(c[1])) for c in ring.coords]
                for ring in sub_poly.interiors
            ]
            combined.extend(_fill_polygon_hatch(sub_pts, sub_holes_pts, angle_deg, spacing_mm))
        return combined

    # Rotate coordinate system so hatch lines are always horizontal,
    # then rotate results back.
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cos_na = math.cos(-angle_rad)
    sin_na = math.sin(-angle_rad)

    def rotate(x: float, y: float, c: float, s: float) -> tuple[float, float]:
        return (c * x - s * y, s * x + c * y)

    # Get bounds of rotated polygon
    coords = list(poly.exterior.coords)
    rot_coords = [rotate(x, y, cos_a, sin_a) for x, y in coords]
    xs = [c[0] for c in rot_coords]
    ys = [c[1] for c in rot_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if spacing_mm <= 0:
        return []

    result: list[Polyline] = []
    y = min_y
    while y <= max_y + spacing_mm:
        # Create a long scan line in the rotated coordinate system
        x0 = min_x - 1.0
        x1 = max_x + 1.0
        # Rotate the scan line back to original coordinates
        p0 = rotate(x0, y, cos_na, sin_na)
        p1 = rotate(x1, y, cos_na, sin_na)
        scan = LineString([p0, p1])

        try:
            clipped = poly.intersection(scan)
        except Exception:
            y += spacing_mm
            continue

        if clipped.is_empty:
            y += spacing_mm
            continue

        # Collect all line segments from the intersection result
        geoms = []
        if clipped.geom_type == "LineString":
            geoms = [clipped]
        elif clipped.geom_type in ("MultiLineString", "GeometryCollection"):
            geoms = [g for g in clipped.geoms if g.geom_type == "LineString"]

        for geom in geoms:
            pts = list(geom.coords)
            if len(pts) >= 2:
                result.append([(float(p[0]), float(p[1])) for p in pts])

        y += spacing_mm

    return result


def _fill_polygon_concentric(
    polygon_pts: Polyline,
    hole_pts_list: list[Polyline],
    spacing_mm: float,
) -> list[Polyline]:
    """Fill a polygon with progressively inward offset contours (concentric).

    Uses Shapely buffer(-offset) to compute inward offsets until the shape
    collapses.  Each offset ring is added as a separate Polyline.

    Parameters
    ----------
    polygon_pts:    Outer boundary as a list of (x_mm, y_mm) points.
    hole_pts_list:  List of hole boundaries (each a closed Polyline).
    spacing_mm:     Inward offset step in mm.
    """
    try:
        from shapely.geometry import MultiPolygon, Polygon
    except ImportError:  # pragma: no cover
        return []

    if len(polygon_pts) < 3 or spacing_mm <= 0:
        return []

    shell = [(p[0], p[1]) for p in polygon_pts]
    holes = [[(p[0], p[1]) for p in h] for h in hole_pts_list if len(h) >= 3]
    try:
        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return []
    except Exception:
        return []

    result: list[Polyline] = []
    offset = spacing_mm

    # Safety cap: max 500 rings to avoid infinite loops on degenerate shapes
    for _ in range(500):
        shrunk = poly.buffer(-offset)
        if shrunk.is_empty:
            break

        # Collect all exterior rings from the shrunk geometry
        polys = []
        if shrunk.geom_type == "Polygon":
            polys = [shrunk]
        elif shrunk.geom_type == "MultiPolygon":
            polys = list(shrunk.geoms)
        elif shrunk.geom_type == "GeometryCollection":
            polys = [g for g in shrunk.geoms if g.geom_type == "Polygon"]

        for p in polys:
            coords = list(p.exterior.coords)
            if len(coords) >= 2:
                poly_line: Polyline = [(float(c[0]), float(c[1])) for c in coords]
                # Close if not already closed
                if poly_line[0] != poly_line[-1]:
                    poly_line.append(poly_line[0])
                result.append(poly_line)

        offset += spacing_mm

    return result


def _extract_contours_with_hierarchy(
    gray: np.ndarray,
    threshold: int,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    simplify_tol: float,
    min_length: int,
    smooth_iterations: int,
    adaptive_threshold: bool = False,
    adaptive_c: float = 5.0,
) -> list[tuple[Polyline, list[Polyline]]]:
    """Trace line art contours and return (outer, holes) pairs.

    Returns a list of (outer_poly, [hole_poly, ...]) tuples using RETR_CCOMP
    hierarchy to associate hole contours with their parent outer contours.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ContourGenerator.")

    from plottter.generators._helpers import _apply_threshold

    binary = _apply_threshold(gray, threshold, adaptive_threshold, adaptive_c, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]  # shape (N, 4): next, prev, child, parent
    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    def contour_to_poly(contour: np.ndarray) -> Polyline | None:
        if len(contour) < min_length:
            return None
        c = contour
        if simplify_tol > 0:
            tol_px = max(0.5, simplify_tol * px_per_mm)
            c = cv2.approxPolyDP(c, tol_px, closed=True)
        pts = c.reshape(-1, 2)
        if len(pts) < 2:
            return None
        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts
        ]
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=True)
        if poly and poly[0] != poly[-1]:
            poly.append(poly[0])
        return poly

    # RETR_CCOMP: hierarchy[i] = [next, prev, first_child, parent]
    # Level 0 = outer contours (parent == -1)
    # Level 1 = holes (parent != -1, their parent is a level-0 contour)
    result: list[tuple[Polyline, list[Polyline]]] = []
    n = len(contours)
    for i in range(n):
        parent = hierarchy[i][3]
        if parent != -1:
            # This is a hole — handled when processing its parent
            continue

        outer_poly = contour_to_poly(contours[i])
        if outer_poly is None:
            continue

        # Collect child (hole) contours
        holes: list[Polyline] = []
        child_idx = hierarchy[i][2]  # first child index
        while child_idx != -1:
            hole_poly = contour_to_poly(contours[child_idx])
            if hole_poly is not None:
                holes.append(hole_poly)
            child_idx = hierarchy[child_idx][0]  # next sibling

        result.append((outer_poly, holes))

    return result


@register_generator
class ContourGenerator(Generator):
    """Topographic contour lines traced from image brightness thresholds.

    Provides four operating modes:

    **Contour Levels** (default) — produces concentric isoline rings at
    evenly spaced (or custom) brightness levels, similar to topographic maps.
    Suitable for photographs and continuous-tone images.

    **Line Art Trace** — directly traces the outlines of black regions in a
    binary image using a single threshold.  Optimised for clean B&W line
    drawings (icons, logos, sketches, technical illustrations) where strokes
    should be reproduced faithfully.  Chaikin corner-cutting smoothing
    converts pixel-stepped boundaries into smooth plotter-friendly curves.

    **Skeleton** — reduces thick ink strokes to single-pixel centerlines via
    morphological thinning (scikit-image ``skeletonize``).  Produces one
    plotter stroke per original drawn line instead of two boundary outlines,
    making it ideal for handwriting, calligraphy, and bold graphic marks.

    **FMM Topographic** — uses the Fast Marching Method to propagate a wave
    from a source point across a speed map derived from the image (dark pixels
    = slow, light pixels = fast).  Isocontours of the resulting travel-time
    field bunch up in dark regions (wave slows down) and spread apart in light
    regions, producing portrait-style topographic line art.  Requires
    ``scikit-fmm`` for full quality; falls back to a
    ``scipy.ndimage.distance_transform_edt`` approximation when unavailable.
    """

    name = "Contour Lines"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="mode",
                label="Mode",
                choices=["Contour Levels", "Line Art Trace", "Skeleton", "FMM Topographic"],
                default="Contour Levels",
                description="Contour Levels traces brightness isolines like a topographic map; Line Art Trace traces outlines of shapes from binary-thresholded images; Skeleton reduces thick strokes to single centerlines; FMM Topographic uses Fast Marching Method to produce wave-based topographic lines",
                choice_descriptions={
                    "Contour Levels": "Traces isolines at multiple brightness thresholds — produces topographic-map-like concentric lines",
                    "Line Art Trace": "Binarizes the image and traces the outlines of dark ink shapes — ideal for line drawings and logos",
                    "Skeleton": "Reduces thick ink strokes to single-pixel centerlines via morphological thinning — produces one stroke per original line instead of two boundary outlines",
                    "FMM Topographic": "Uses Fast Marching Method wave propagation to produce topographic contour lines — lines bunch in dark areas (slow wave) and spread in light areas (fast wave)",
                },
            ),
            # --- Contour Levels parameters ---
            IntParam(
                name="num_levels",
                label="Number of Contour Levels",
                min=2,
                max=64,
                step=1,
                default=8,
                visible_when={"mode": ["Contour Levels"]},
                description="Number of brightness levels at which to draw contour lines",
            ),
            ChoiceParam(
                name="spacing",
                label="Level Spacing",
                choices=["linear", "logarithmic", "quadratic"],
                default="linear",
                visible_when={"mode": ["Contour Levels"]},
                description="How contour levels are distributed across the brightness range",
                choice_descriptions={
                    "linear": "Evenly-spaced levels — uniform density of lines across all brightness values",
                    "logarithmic": "More levels in darker areas — emphasizes shadows and detail in dark regions",
                    "quadratic": "More levels in midtones — good for photographs with gradual shading",
                },
            ),
            # --- Line Art Trace parameters ---
            IntParam(
                name="trace_threshold",
                label="Trace Threshold (0–255)",
                min=0,
                max=255,
                step=1,
                default=128,
                visible_when={"mode": ["Line Art Trace", "Skeleton"]},
                description="Brightness threshold for binarizing the image (0–255) — pixels below this are treated as ink",
            ),
            IntParam(
                name="smooth_iterations",
                label="Smooth Iterations (Chaikin)",
                min=0,
                max=5,
                step=1,
                default=0,
                visible_when={"mode": ["Line Art Trace", "Skeleton", "FMM Topographic"]},
                description="Number of Chaikin smoothing passes — each pass rounds corners for more flowing, organic-looking strokes",
            ),
            IntParam(
                name="cleanup_kernel",
                label="Cleanup Kernel Size",
                min=0,
                max=10,
                step=1,
                default=0,
                visible_when={"mode": ["Skeleton"]},
                description="Morphological cleanup kernel size (0 = disabled). MORPH_CLOSE fills small gaps in ink, MORPH_OPEN removes noise specks. Values 2–5 are typical",
            ),
            FloatParam(
                name="merge_gap_mm",
                label="Fragment Merge Gap (mm)",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.5,
                visible_when={"mode": ["Skeleton"]},
                description="Maximum endpoint distance (mm) for merging nearby skeleton fragments — reduces pen lifts by connecting polylines whose endpoints are close. Set to 0 to disable",
            ),
            # --- Adaptive thresholding (Line Art Trace and Skeleton modes) ---
            BoolParam(
                name="adaptive_threshold",
                label="Adaptive Threshold",
                default=False,
                visible_when={"mode": ["Line Art Trace", "Skeleton"]},
                description="Use local adaptive Gaussian thresholding instead of a global value — handles uneven lighting, scanned pages, and phone photos of drawings where global threshold fails",
            ),
            FloatParam(
                name="adaptive_c",
                label="Adaptive C Constant",
                min=-20.0,
                max=20.0,
                step=0.5,
                default=5.0,
                visible_when={"mode": ["Line Art Trace", "Skeleton"], "adaptive_threshold": [True]},
                description="Constant subtracted from the local Gaussian-weighted mean — higher values are stricter (less ink detected), lower/negative values are more permissive",
            ),
            # --- FMM Topographic parameters ---
            IntParam(
                name="fmm_num_contours",
                label="Number of Contours",
                min=2,
                max=100,
                step=1,
                default=20,
                visible_when={"mode": ["FMM Topographic"]},
                description="Number of isocontour levels to extract from the FMM travel-time field",
            ),
            ChoiceParam(
                name="fmm_source_point",
                label="Source Point",
                choices=["Center", "Custom"],
                default="Center",
                visible_when={"mode": ["FMM Topographic"]},
                description="Origin of the FMM wave propagation — Center starts from the image center; Custom uses a point-click interaction to set a user-defined origin",
                choice_descriptions={
                    "Center": "Wave propagates from the center of the image",
                    "Custom": "Wave propagates from a user-defined source point (set via point-click on the image)",
                },
            ),
            FloatParam(
                name="fmm_source_x_pct",
                label="Source X (%)",
                min=0.0,
                max=100.0,
                step=0.5,
                default=50.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_source_point": ["Custom"]},
                description="Horizontal position of FMM source as percentage of image width",
            ),
            FloatParam(
                name="fmm_source_y_pct",
                label="Source Y (%)",
                min=0.0,
                max=100.0,
                step=0.5,
                default=50.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_source_point": ["Custom"]},
                description="Vertical position of FMM source as percentage of image height",
            ),
            FloatParam(
                name="fmm_gamma",
                label="Speed Gamma",
                min=0.1,
                max=5.0,
                step=0.1,
                default=1.0,
                visible_when={"mode": ["FMM Topographic"]},
                description="Gamma curve applied to the speed map — >1 concentrates lines in dark regions (slower propagation), <1 in light regions",
            ),
            FloatParam(
                name="fmm_speed_floor",
                label="Speed Floor",
                min=0.001,
                max=0.5,
                step=0.005,
                default=0.01,
                visible_when={"mode": ["FMM Topographic"]},
                description="Minimum speed value to prevent infinite travel time in fully black areas — higher values reduce line density in very dark regions",
            ),
            ChoiceParam(
                name="fmm_contour_spacing",
                label="Contour Spacing",
                choices=["Linear", "Logarithmic", "Quadratic"],
                default="Linear",
                visible_when={"mode": ["FMM Topographic"]},
                description="Distribution of contour levels across the travel-time range",
                choice_descriptions={
                    "Linear": "Evenly-spaced levels — uniform density of lines across the travel-time range",
                    "Logarithmic": "More levels near the source — concentrates lines close to the wave origin",
                    "Quadratic": "Denser levels farther from source — emphasizes the outer wave fronts",
                },
            ),
            FloatParam(
                name="fmm_min_contour_length_mm",
                label="Min Contour Length (mm)",
                min=0.0,
                max=20.0,
                step=0.5,
                default=2.0,
                visible_when={"mode": ["FMM Topographic"]},
                description="Minimum polyline length in mm — shorter contours are discarded as noise",
            ),
            # --- FMM Render Mode parameters ---
            ChoiceParam(
                name="fmm_render_mode",
                label="Render Mode",
                choices=["Contours", "Displacement", "Wave", "Radial"],
                default="Contours",
                visible_when={"mode": ["FMM Topographic"]},
                description="How to visualise the FMM travel-time field",
                choice_descriptions={
                    "Contours": "Extract isocontour lines at evenly spaced travel-time levels (topographic map style)",
                    "Displacement": "Draw parallel lines displaced perpendicularly by the T-field value — produces an engraving/relief effect",
                    "Wave": "Draw horizontal scan lines whose sinusoidal amplitude is modulated by the T-field gradient magnitude",
                    "Radial": "Draw lines emanating from the source point, following gradient ascent of T (perpendicular to isocontours)",
                },
            ),
            # Shared for Displacement and Wave
            IntParam(
                name="fmm_num_lines",
                label="Number of Lines",
                min=10,
                max=500,
                step=10,
                default=100,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Displacement", "Wave"]},
                description="Number of parallel lines (Displacement) or scan lines (Wave) to draw",
            ),
            # Displacement-specific
            FloatParam(
                name="fmm_displacement_mm",
                label="Max Displacement (mm)",
                min=0.1,
                max=20.0,
                step=0.1,
                default=5.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Displacement"]},
                description="Maximum perpendicular displacement of each line in mm — 0 = no displacement, higher values exaggerate the T-field relief",
            ),
            FloatParam(
                name="fmm_line_angle",
                label="Line Angle (°)",
                min=0.0,
                max=180.0,
                step=1.0,
                default=0.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Displacement"]},
                description="Angle of the parallel lines in degrees (0 = horizontal, 90 = vertical)",
            ),
            # Wave-specific
            FloatParam(
                name="fmm_amplitude_mm",
                label="Wave Amplitude (mm)",
                min=0.5,
                max=10.0,
                step=0.1,
                default=3.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"]},
                description="Maximum sinusoidal wave amplitude in mm — modulated by the T-field gradient magnitude at each point",
            ),
            FloatParam(
                name="fmm_frequency",
                label="Wave Frequency",
                min=1.0,
                max=50.0,
                step=0.5,
                default=10.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"]},
                description="Number of wave cycles across the canvas width — higher values produce more tightly packed waves",
            ),
            ChoiceParam(
                name="fmm_line_spacing",
                label="Line Spacing Mode",
                choices=["Uniform", "Adaptive", "Grouped", "Adaptive + Grouped"],
                default="Uniform",
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"]},
                description="Controls how scan-line Y positions are distributed across the drawing area",
                choice_descriptions={
                    "Uniform": "Evenly spaced scan lines — classic wave pattern",
                    "Adaptive": "Density follows gradient magnitude: dense where T changes rapidly (image detail), sparse in flat areas",
                    "Grouped": "Lines appear in tight clusters separated by larger gaps — creates a banded effect",
                    "Adaptive + Grouped": "Group gaps vary with gradient magnitude — clusters are denser in detailed areas",
                },
            ),
            FloatParam(
                name="fmm_min_spacing_mm",
                label="Min Spacing (mm)",
                min=0.1,
                max=10.0,
                step=0.1,
                default=0.5,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"], "fmm_line_spacing": ["Adaptive", "Adaptive + Grouped"]},
                description="Minimum spacing between scan lines in high-gradient (detailed) areas",
            ),
            FloatParam(
                name="fmm_max_spacing_mm",
                label="Max Spacing (mm)",
                min=0.5,
                max=20.0,
                step=0.5,
                default=5.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"], "fmm_line_spacing": ["Adaptive", "Adaptive + Grouped"]},
                description="Maximum spacing between scan lines in low-gradient (flat) areas",
            ),
            IntParam(
                name="fmm_group_size",
                label="Group Size",
                min=2,
                max=10,
                step=1,
                default=3,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"], "fmm_line_spacing": ["Grouped", "Adaptive + Grouped"]},
                description="Number of lines per group",
            ),
            FloatParam(
                name="fmm_group_gap_mm",
                label="Group Gap (mm)",
                min=1.0,
                max=20.0,
                step=0.5,
                default=4.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"], "fmm_line_spacing": ["Grouped"]},
                description="Gap between consecutive groups of scan lines",
            ),
            FloatParam(
                name="fmm_group_intra_spacing_mm",
                label="Intra-Group Spacing (mm)",
                min=0.1,
                max=5.0,
                step=0.1,
                default=0.5,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"], "fmm_line_spacing": ["Grouped", "Adaptive + Grouped"]},
                description="Spacing between individual lines within a group",
            ),
            FloatParam(
                name="fmm_displacement_variation",
                label="Displacement Variation",
                min=0.0,
                max=1.0,
                step=0.1,
                default=0.0,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Wave"]},
                description="Per-line random amplitude multiplier: 0 = all lines respond equally, 1 = dramatic variation between neighbouring lines",
            ),
            # Radial-specific
            IntParam(
                name="fmm_num_radials",
                label="Number of Radials",
                min=10,
                max=360,
                step=10,
                default=120,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Radial"]},
                description="Number of radial lines evenly distributed around 360° from the source point",
            ),
            FloatParam(
                name="fmm_step_size_mm",
                label="Step Size (mm)",
                min=0.1,
                max=2.0,
                step=0.05,
                default=0.5,
                visible_when={"mode": ["FMM Topographic"], "fmm_render_mode": ["Radial"]},
                description="Step size per iteration along the gradient — smaller values give smoother curves but take longer to compute",
            ),
            # --- Fill parameters (Line Art Trace mode only) ---
            ChoiceParam(
                name="fill",
                label="Fill Style",
                choices=["None", "Solid", "Hatching", "Cross-hatch", "Concentric"],
                default="None",
                visible_when={"mode": ["Line Art Trace"]},
                description="Fill style for enclosed contour shapes — None traces outlines only, other options fill the interior",
                choice_descriptions={
                    "None": "Outline only — traces the boundary of each shape without filling the interior",
                    "Solid": "Fills the interior with densely-packed parallel lines (spacing ~0.3mm for solid appearance)",
                    "Hatching": "Fills with parallel lines at a configurable angle — produces a visible line texture inside the shape",
                    "Cross-hatch": "Fills with two overlapping passes of hatching at perpendicular angles",
                    "Concentric": "Fills with progressively smaller inward offsets of the contour shape — topographic map effect",
                },
            ),
            FloatParam(
                name="fill_spacing_mm",
                label="Fill Spacing (mm)",
                min=0.1,
                max=5.0,
                step=0.05,
                default=0.3,
                visible_when={"mode": ["Line Art Trace"], "fill": ["Solid", "Hatching", "Cross-hatch", "Concentric"]},
                description="Spacing between fill lines in millimeters (smaller = denser fill)",
            ),
            FloatParam(
                name="fill_angle",
                label="Fill Angle (°)",
                min=0.0,
                max=180.0,
                step=1.0,
                default=45.0,
                visible_when={"mode": ["Line Art Trace"], "fill": ["Hatching", "Cross-hatch"]},
                description="Angle of hatch fill lines in degrees (0 = horizontal, 45 = diagonal, 90 = vertical)",
            ),
            # --- Shared parameters (both modes) ---
            FloatParam(
                name="simplify_mm",
                label="Simplify Tolerance (mm)",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.3,
                description="RDP simplification tolerance in mm — reduces point count while preserving shape detail",
            ),
            BoolParam(
                name="smooth_curves",
                label="Smooth Curves (Bezier Fitting)",
                default=False,
                description="Fit cubic Bezier curves to the output polylines for smooth, organic-looking strokes",
            ),
            FloatParam(
                name="curve_tolerance_mm",
                label="Curve Tolerance (mm)",
                min=0.1,
                max=5.0,
                step=0.05,
                default=0.5,
                visible_when={"smooth_curves": [True]},
                description="How closely the fitted curves must follow the original points — lower = more faithful but more points",
            ),
            IntParam(
                name="min_contour_px",
                label="Min Contour Length (px)",
                min=3,
                max=500,
                step=1,
                default=10,
                description="Minimum contour length in pixels — shorter contours are discarded as noise",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image before processing",
            ),
            # Shared preprocessing parameters
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
                description="Gaussian blur radius applied before processing — reduces noise and softens edges",
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
                name="Fine (16 levels)",
                params={
                    "mode": "Contour Levels",
                    "num_levels": 16,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.2,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Coarse (6 levels)",
                params={
                    "mode": "Contour Levels",
                    "num_levels": 6,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.5,
                    "min_contour_px": 20,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 2.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Logarithmic (shadows detail)",
                params={
                    "mode": "Contour Levels",
                    "num_levels": 10,
                    "spacing": "logarithmic",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Inverted (highlights detail)",
                params={
                    "mode": "Contour Levels",
                    "num_levels": 10,
                    "spacing": "logarithmic",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": True,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="High Contrast Graphic",
                params={
                    # Quadratic level spacing emphasises midtone transitions;
                    # extra contrast boost and heavier blur suit illustrations,
                    # logos, and other graphics with flat colour regions.
                    "mode": "Contour Levels",
                    "num_levels": 12,
                    "spacing": "quadratic",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.4,
                    "min_contour_px": 15,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 30.0,
                    "blur_radius": 2.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Portrait Photo",
                params={
                    # Many closely spaced levels with a gentle blur produce
                    # fine topographic lines that reveal skin texture and
                    # facial contours.  Logarithmic spacing gives more detail
                    # in the shadows where most facial information lives.
                    "mode": "Contour Levels",
                    "num_levels": 20,
                    "spacing": "logarithmic",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.2,
                    "min_contour_px": 8,
                    "invert": False,
                    "brightness": 5.0,
                    "contrast": 15.0,
                    "blur_radius": 1.5,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Line Art / Trace",
                params={
                    # Optimised for clean B&W line drawings (icons, logos,
                    # technical illustrations, hand-drawn sketches that have
                    # been scanned and thresholded).
                    #
                    # Single threshold traces the outline of each ink region
                    # at full pixel resolution; Chaikin smoothing (2 passes)
                    # converts the pixel-stepped boundary into a smooth
                    # plotter-friendly curve.  Small RDP tolerance preserves
                    # detail; short-contour filter (min 5 px) removes noise
                    # specks while keeping fine features.
                    "mode": "Line Art Trace",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 2,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.15,
                    "min_contour_px": 5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Line Art / Solid Fill",
                params={
                    # Dense parallel lines fill each traced shape, producing a
                    # solid appearance with tight 0.3 mm spacing (suitable for
                    # 0.4–0.5 mm pens).  Outline is traced alongside the fill.
                    "mode": "Line Art Trace",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 1,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "Solid",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 0.0,
                    "simplify_mm": 0.2,
                    "min_contour_px": 5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Line Art / Hatched Fill",
                params={
                    # 45° hatching at 0.8 mm spacing gives a hand-drawn
                    # crosshatch appearance inside each traced shape.
                    "mode": "Line Art Trace",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 1,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "Hatching",
                    "fill_spacing_mm": 0.8,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.2,
                    "min_contour_px": 5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Line Art / Concentric Fill",
                params={
                    # Inward concentric rings trace the topology of each shape,
                    # producing a topographic appearance inside traced regions.
                    "mode": "Line Art Trace",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 1,
                    "cleanup_kernel": 0,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "Concentric",
                    "fill_spacing_mm": 1.0,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.2,
                    "min_contour_px": 5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Skeleton / Center-line",
                params={
                    # Morphological skeletonization reduces thick strokes to
                    # single-pixel-wide centerlines before tracing, producing
                    # approximately one polyline per original ink line rather
                    # than two parallel boundary outlines.  Recommended for
                    # simple line art imports (logos, comics, ink drawings)
                    # where a clean single-stroke output is desired.
                    "mode": "Skeleton",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 1,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_render_mode": "Contours",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Scanned Line Art",
                params={
                    # Adaptive thresholding handles scanned pages and phone
                    # photos of drawings where global threshold fails due to
                    # uneven lighting or shadow gradients.  Skeleton mode
                    # produces clean centerlines from thick ink strokes.
                    "mode": "Skeleton",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 1,
                    "cleanup_kernel": 2,
                    "merge_gap_mm": 1.0,
                    "adaptive_threshold": True,
                    "adaptive_c": 5.0,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 0.5,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Portrait",
                params={
                    # Portrait-style FMM topographic lines — many closely
                    # spaced contours with linear spacing reveal facial
                    # contours and skin texture.  Invert is off so dark
                    # regions (shadows) slow the wave and bunch lines there.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 1,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 15.0,
                    "blur_radius": 1.5,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 30,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.5,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 3.0,
                    "fmm_render_mode": "Contours",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Landscape",
                params={
                    # Landscape / terrain style — fewer, well-separated contour
                    # lines with logarithmic spacing emphasise the valleys
                    # (dark regions) where contours bunch.  A gentle blur
                    # smoothes out image noise before FMM propagation.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.4,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 2.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 15,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.02,
                    "fmm_contour_spacing": "Logarithmic",
                    "fmm_min_contour_length_mm": 5.0,
                    "fmm_render_mode": "Contours",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Dense Wave",
                params={
                    # Dense concentric wave pattern with many tightly packed
                    # contours.  High gamma (2.0) strongly slows the wave in
                    # dark areas, producing very dense line clusters there
                    # while keeping bright areas relatively open.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 2,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.2,
                    "min_contour_px": 5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 50,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 2.0,
                    "fmm_speed_floor": 0.005,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 1.0,
                    "fmm_render_mode": "Contours",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Depth Portrait",
                params={
                    # Use this preset with the "AI Depth Map" image source selected
                    # in the image source controls — contour lines wrap around the
                    # 3D structure of the face / subject rather than following
                    # lighting boundaries.  Near objects (bright in depth map) slow
                    # the wave so lines bunch up on the closest foreground subject.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 1,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 30,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.0,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 3.0,
                    "fmm_render_mode": "Contours",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Depth Landscape",
                params={
                    # Use this preset with the "AI Depth Map" image source and
                    # "Invert" checked in the depth map source controls — wave
                    # propagates fastest from near objects; distant terrain slows
                    # the wave so contour lines bunch up on far-off hills and
                    # mountains.  Logarithmic spacing adds extra detail near the
                    # wave source for a terrain-map appearance.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.4,
                    "min_contour_px": 10,
                    "invert": True,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.5,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 0.7,
                    "fmm_speed_floor": 0.02,
                    "fmm_contour_spacing": "Logarithmic",
                    "fmm_min_contour_length_mm": 5.0,
                    "fmm_render_mode": "Contours",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Displacement Lines",
                params={
                    # Parallel lines displaced perpendicularly by the FMM
                    # travel-time field value — produces an engraving / relief
                    # effect where dark regions cause strong lateral deflections.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 15.0,
                    "blur_radius": 1.5,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.5,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_render_mode": "Displacement",
                    "fmm_num_lines": 120,
                    "fmm_displacement_mm": 6.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Wave Lines",
                params={
                    # Horizontal scan lines with sinusoidal amplitude modulated
                    # by the FMM travel-time gradient magnitude — waves grow
                    # large in dark / high-gradient regions and flatten out in
                    # bright / smooth areas.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.5,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_render_mode": "Wave",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 4.0,
                    "fmm_frequency": 12.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Radial Lines",
                params={
                    # Lines radiate from the FMM source following gradient
                    # ascent of the travel-time field — they spread in bright /
                    # fast regions and bunch together in dark / slow areas,
                    # creating a starburst effect that warps around features.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 15.0,
                    "blur_radius": 1.5,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.5,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_render_mode": "Radial",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 3.0,
                    "fmm_frequency": 10.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Uniform",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Adaptive Wave",
                params={
                    # Horizontal scan lines with adaptive spacing: lines cluster
                    # densely in high-gradient / dark regions and spread apart in
                    # smooth / bright areas, giving organic topographic density.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.5,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_render_mode": "Wave",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 5.0,
                    "fmm_frequency": 6.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Adaptive",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 6.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 4.0,
                    "fmm_group_intra_spacing_mm": 0.5,
                    "fmm_displacement_variation": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="FMM Grouped Wave",
                params={
                    # Wave scan lines arranged in tight clusters separated by
                    # larger gaps, with random per-line amplitude variation for
                    # a hand-drawn, sketchy topographic feel.
                    "mode": "FMM Topographic",
                    "num_levels": 8,
                    "spacing": "linear",
                    "trace_threshold": 128,
                    "smooth_iterations": 0,
                    "cleanup_kernel": 0,
                    "merge_gap_mm": 0.5,
                    "adaptive_threshold": False,
                    "adaptive_c": 5.0,
                    "fill": "None",
                    "fill_spacing_mm": 0.3,
                    "fill_angle": 45.0,
                    "simplify_mm": 0.3,
                    "min_contour_px": 10,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "fmm_num_contours": 20,
                    "fmm_source_point": "Center",
                    "fmm_source_x_pct": 50.0,
                    "fmm_source_y_pct": 50.0,
                    "fmm_gamma": 1.5,
                    "fmm_speed_floor": 0.01,
                    "fmm_contour_spacing": "Linear",
                    "fmm_min_contour_length_mm": 2.0,
                    "fmm_render_mode": "Wave",
                    "fmm_num_lines": 100,
                    "fmm_displacement_mm": 5.0,
                    "fmm_line_angle": 0.0,
                    "fmm_amplitude_mm": 4.0,
                    "fmm_frequency": 8.0,
                    "fmm_num_radials": 120,
                    "fmm_step_size_mm": 0.5,
                    "fmm_line_spacing": "Grouped",
                    "fmm_min_spacing_mm": 0.5,
                    "fmm_max_spacing_mm": 5.0,
                    "fmm_group_size": 3,
                    "fmm_group_gap_mm": 5.0,
                    "fmm_group_intra_spacing_mm": 0.6,
                    "fmm_displacement_variation": 0.4,
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
        """Generate contour polylines from ``params["_source_image"]``.

        Dispatches to _trace_isolines (Contour Levels mode), _trace_line_art
        (Line Art Trace mode), or _trace_skeleton (Skeleton mode) depending on
        the ``mode`` parameter.
        """
        image = params.get("_source_image")
        if image is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
            to_grayscale,
        )

        # Preprocessing
        img = image.copy()
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

        gray = to_grayscale(img)  # uint8 H×W

        simplify_mm = float(params.get("simplify_mm", 0.3))
        min_contour_px = int(params.get("min_contour_px", 10))
        mode = str(params.get("mode", "Contour Levels"))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_h, img_w = gray.shape

        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        if mode == "Line Art Trace":
            if progress_callback:
                progress_callback(10)
            if cancelled_callback and cancelled_callback():
                return []

            trace_threshold = int(params.get("trace_threshold", 128))
            smooth_iterations = int(params.get("smooth_iterations", 0))
            fill_style = str(params.get("fill", "None"))
            fill_spacing_mm = float(params.get("fill_spacing_mm", 0.3))
            fill_angle = float(params.get("fill_angle", 45.0))
            adaptive_thresh = bool(params.get("adaptive_threshold", False))
            adaptive_c = float(params.get("adaptive_c", 5.0))

            if fill_style == "None":
                # Original outline-only behaviour
                result = _trace_line_art(
                    gray,
                    trace_threshold,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    simplify_mm,
                    min_contour_px,
                    smooth_iterations,
                    adaptive_thresh,
                    adaptive_c,
                )
            else:
                # Extract contours with hierarchy for fill generation
                contour_pairs = _extract_contours_with_hierarchy(
                    gray,
                    trace_threshold,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    simplify_mm,
                    min_contour_px,
                    smooth_iterations,
                    adaptive_thresh,
                    adaptive_c,
                )

                if progress_callback:
                    progress_callback(40)
                if cancelled_callback and cancelled_callback():
                    return []

                result = []
                # Add outline polylines
                for outer, holes in contour_pairs:
                    result.append(outer)

                if progress_callback:
                    progress_callback(60)

                # Add fill lines for each shape
                total = len(contour_pairs)
                for idx, (outer, holes) in enumerate(contour_pairs):
                    if cancelled_callback and cancelled_callback():
                        break
                    if progress_callback and total > 0:
                        progress_callback(60 + int(idx / total * 35))

                    if fill_style == "Solid":
                        fill_lines = _fill_polygon_hatch(
                            outer, holes, 0.0, fill_spacing_mm
                        )
                        result.extend(fill_lines)
                    elif fill_style == "Hatching":
                        fill_lines = _fill_polygon_hatch(
                            outer, holes, fill_angle, fill_spacing_mm
                        )
                        result.extend(fill_lines)
                    elif fill_style == "Cross-hatch":
                        # Two perpendicular passes
                        fill_lines = _fill_polygon_hatch(
                            outer, holes, fill_angle, fill_spacing_mm
                        )
                        result.extend(fill_lines)
                        fill_lines2 = _fill_polygon_hatch(
                            outer, holes, fill_angle + 90.0, fill_spacing_mm
                        )
                        result.extend(fill_lines2)
                    elif fill_style == "Concentric":
                        fill_lines = _fill_polygon_concentric(
                            outer, holes, fill_spacing_mm
                        )
                        result.extend(fill_lines)

            if bool(params.get("smooth_curves", False)):
                from plottter.processing.curves import fit_curves as _fit_curves
                _tol = float(params.get("curve_tolerance_mm", 0.5))
                result = _fit_curves(result, _tol)

            if progress_callback:
                progress_callback(100)
            x_off = float(params.get("x_offset_mm", 0.0))
            y_off = float(params.get("y_offset_mm", 0.0))
            if x_off != 0.0 or y_off != 0.0:
                result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
            return result

        if mode == "Skeleton":
            if progress_callback:
                progress_callback(10)
            if cancelled_callback and cancelled_callback():
                return []

            trace_threshold = int(params.get("trace_threshold", 128))
            smooth_iterations = int(params.get("smooth_iterations", 0))
            cleanup_kernel = int(params.get("cleanup_kernel", 0))
            merge_gap_mm = float(params.get("merge_gap_mm", 0.5))
            adaptive_thresh = bool(params.get("adaptive_threshold", False))
            adaptive_c = float(params.get("adaptive_c", 5.0))

            result = _trace_skeleton(
                gray,
                trace_threshold,
                cleanup_kernel,
                img_w,
                img_h,
                img_x1,
                img_y1,
                img_x2,
                img_y2,
                simplify_mm,
                min_contour_px,
                smooth_iterations,
                merge_gap_mm,
                adaptive_thresh,
                adaptive_c,
            )

            if bool(params.get("smooth_curves", False)):
                from plottter.processing.curves import fit_curves as _fit_curves
                _tol = float(params.get("curve_tolerance_mm", 0.5))
                result = _fit_curves(result, _tol)

            if progress_callback:
                progress_callback(100)
            x_off = float(params.get("x_offset_mm", 0.0))
            y_off = float(params.get("y_offset_mm", 0.0))
            if x_off != 0.0 or y_off != 0.0:
                result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
            return result

        if mode == "FMM Topographic":
            if progress_callback:
                progress_callback(5)
            if cancelled_callback and cancelled_callback():
                return []

            fmm_num_contours = int(params.get("fmm_num_contours", 20))
            fmm_source_point = str(params.get("fmm_source_point", "Center"))
            fmm_source_x_pct = float(params.get("fmm_source_x_pct", 50.0))
            fmm_source_y_pct = float(params.get("fmm_source_y_pct", 50.0))
            fmm_gamma = float(params.get("fmm_gamma", 1.0))
            fmm_speed_floor = float(params.get("fmm_speed_floor", 0.01))
            fmm_contour_spacing = str(params.get("fmm_contour_spacing", "Linear"))
            fmm_min_contour_length_mm = float(params.get("fmm_min_contour_length_mm", 2.0))
            smooth_iterations = int(params.get("smooth_iterations", 0))
            # speed_map_override is always None — the source image (which may be
            # an AI depth map selected via the image source controls) is used
            # directly as the brightness-based speed map.
            speed_map_override: np.ndarray | None = None

            fmm_render_mode = str(params.get("fmm_render_mode", "Contours"))

            if fmm_render_mode == "Displacement":
                T, _sy, _sx = _compute_fmm_field(
                    gray, fmm_source_point, fmm_gamma, fmm_speed_floor, speed_map_override,
                    fmm_source_x_pct, fmm_source_y_pct,
                )
                if progress_callback:
                    progress_callback(10)
                result = _fmm_displacement(
                    T,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    num_lines=int(params.get("fmm_num_lines", 100)),
                    displacement_mm=float(params.get("fmm_displacement_mm", 5.0)),
                    line_angle_deg=float(params.get("fmm_line_angle", 0.0)),
                    progress_callback=progress_callback,
                    cancelled_callback=cancelled_callback,
                )
            elif fmm_render_mode == "Wave":
                T, _sy, _sx = _compute_fmm_field(
                    gray, fmm_source_point, fmm_gamma, fmm_speed_floor, speed_map_override,
                    fmm_source_x_pct, fmm_source_y_pct,
                )
                if progress_callback:
                    progress_callback(10)
                result = _fmm_wave(
                    T,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    num_lines=int(params.get("fmm_num_lines", 100)),
                    amplitude_mm=float(params.get("fmm_amplitude_mm", 3.0)),
                    frequency=float(params.get("fmm_frequency", 10.0)),
                    line_spacing=str(params.get("fmm_line_spacing", "Uniform")),
                    min_spacing_mm=float(params.get("fmm_min_spacing_mm", 0.5)),
                    max_spacing_mm=float(params.get("fmm_max_spacing_mm", 5.0)),
                    group_size=int(params.get("fmm_group_size", 3)),
                    group_gap_mm=float(params.get("fmm_group_gap_mm", 4.0)),
                    group_intra_spacing_mm=float(params.get("fmm_group_intra_spacing_mm", 0.5)),
                    displacement_variation=float(params.get("fmm_displacement_variation", 0.0)),
                    seed=int(params.get("seed", 0)),
                    progress_callback=progress_callback,
                    cancelled_callback=cancelled_callback,
                )
            elif fmm_render_mode == "Radial":
                T, sy, sx = _compute_fmm_field(
                    gray, fmm_source_point, fmm_gamma, fmm_speed_floor, speed_map_override,
                    fmm_source_x_pct, fmm_source_y_pct,
                )
                if progress_callback:
                    progress_callback(10)
                img_rect_w = img_x2 - img_x1
                step_size_mm = float(params.get("fmm_step_size_mm", 0.5))
                px_per_mm = img_w / img_rect_w if img_rect_w > 0 else 1.0
                step_size_px = max(0.1, step_size_mm * px_per_mm)
                result = _fmm_radial(
                    T,
                    sy,
                    sx,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    num_radials=int(params.get("fmm_num_radials", 120)),
                    step_size_px=step_size_px,
                    progress_callback=progress_callback,
                    cancelled_callback=cancelled_callback,
                )
            else:
                # Default: "Contours" mode — isocontour extraction
                result = _trace_fmm_topographic(
                    gray,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    fmm_num_contours,
                    fmm_source_point,
                    fmm_gamma,
                    fmm_speed_floor,
                    fmm_contour_spacing,
                    fmm_min_contour_length_mm,
                    simplify_mm,
                    smooth_iterations,
                    progress_callback,
                    cancelled_callback,
                    speed_map_override=speed_map_override,
                    source_x_pct=fmm_source_x_pct,
                    source_y_pct=fmm_source_y_pct,
                )

            if bool(params.get("smooth_curves", False)):
                from plottter.processing.curves import fit_curves as _fit_curves
                _tol = float(params.get("curve_tolerance_mm", 0.5))
                result = _fit_curves(result, _tol)

            if progress_callback:
                progress_callback(100)
            x_off = float(params.get("x_offset_mm", 0.0))
            y_off = float(params.get("y_offset_mm", 0.0))
            if x_off != 0.0 or y_off != 0.0:
                result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
            return result

        # --- Contour Levels mode (original behaviour) ---
        num_levels = int(params.get("num_levels", 8))
        spacing = str(params.get("spacing", "linear"))
        smooth_iterations = int(params.get("smooth_iterations", 0))

        # Compute threshold values in [1, 254] range
        thresholds = _compute_thresholds(num_levels, spacing)

        all_polylines: list[Polyline] = []
        for i, thr in enumerate(thresholds):
            if cancelled_callback and cancelled_callback():
                break
            if progress_callback:
                progress_callback(int(i / len(thresholds) * 95))

            polys = _trace_isolines(
                gray,
                int(thr),
                img_w,
                img_h,
                img_x1,
                img_y1,
                img_x2,
                img_y2,
                simplify_mm,
                min_contour_px,
                smooth_iterations,
            )
            all_polylines.extend(polys)

        if bool(params.get("smooth_curves", False)):
            from plottter.processing.curves import fit_curves as _fit_curves
            _tol = float(params.get("curve_tolerance_mm", 0.5))
            all_polylines = _fit_curves(all_polylines, _tol)

        if progress_callback:
            progress_callback(100)

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            all_polylines = [[(x + x_off, y + y_off) for x, y in path] for path in all_polylines]
        return all_polylines


def _compute_thresholds(num_levels: int, spacing: str) -> list[float]:
    """Compute a list of threshold values in the range [1, 254]."""
    n = max(2, num_levels)
    if spacing == "logarithmic":
        # Logarithmically spaced: denser in shadows
        log_vals = np.logspace(0, math.log10(254), n + 2)[1:-1]
        return list(log_vals[:n])
    elif spacing == "quadratic":
        # Quadratic spacing: denser in midtones
        t = np.linspace(0.0, 1.0, n + 2)[1:-1]
        vals = 1 + (253 * t**2)
        return list(vals[:n])
    else:
        # Linear spacing
        return list(np.linspace(1, 254, n + 2)[1:-1])
