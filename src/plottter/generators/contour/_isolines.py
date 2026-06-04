"""Isoline tracing and threshold computation for ContourGenerator."""

from __future__ import annotations

import math

import numpy as np

from plottter.generators._helpers import _px_to_mm, _apply_threshold
from plottter.models import Polyline

from ._smoothing import _chaikin_smooth


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
    supersample: int = 1,
) -> list[Polyline]:
    """Trace contour lines at a single brightness threshold using sub-pixel marching squares.

    Uses ``skimage.measure.find_contours`` via :func:`._subpixel.extract_subpixel_contours`
    instead of ``cv2.findContours``.  Marching squares linearly interpolates where the
    iso-value crosses *between* adjacent pixels, so diagonal edges become smooth diagonal
    lines rather than pixel-grid staircases.

    Contour Levels rings are mostly closed interior loops; border-touching contours are
    treated as open polylines and are **not** force-closed (doing so would add a spurious
    chord from the last point back to the first).
    """
    from ._subpixel import extract_subpixel_contours

    raw_contours = extract_subpixel_contours(gray, float(threshold), min_length, supersample)

    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    polylines: list[Polyline] = []
    for pts_xy, is_closed in raw_contours:
        # pts_xy is (N, 2) float array of (x, y) = (col, row) pixel coordinates

        # RDP simplification in pixel space
        if simplify_tol > 0:
            try:
                import cv2
            except ImportError:  # pragma: no cover
                raise RuntimeError("opencv-python is required for ContourGenerator.")
            tol_px = max(1.0, simplify_tol * px_per_mm)
            pts_arr = pts_xy.reshape(-1, 1, 2).astype(np.float32)
            pts_arr = cv2.approxPolyDP(pts_arr, tol_px, closed=is_closed)
            pts_xy = pts_arr.reshape(-1, 2)

        if len(pts_xy) < 2:
            continue

        # Convert pixel coords to mm
        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts_xy
        ]

        # Chaikin smoothing with closed=is_closed: open strokes stay open
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=is_closed)

        # Close only when it is a closed interior loop; border-touching strokes stay open
        if is_closed and poly and poly[0] != poly[-1]:
            poly.append(poly[0])

        polylines.append(poly)

    return polylines


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
    supersample: int = 1,
) -> list[tuple[Polyline, list[Polyline]]]:
    """Trace line art contours and return (outer, holes) pairs.

    Uses sub-pixel contour extraction (skimage marching squares) and geometric
    containment hierarchy (Shapely) instead of cv2.findContours with RETR_CCOMP.
    Only closed rings (those not touching the image border) are retained for fill
    generation; open border-crossing contours cannot bound a fill region.

    For the adaptive threshold path, the adaptive binary mask is computed first
    (dark ink → 255, light background → 0) and then fed to
    ``extract_subpixel_contours`` at ``level=127`` — marching squares on the
    0/255 mask still yields cleaner half-pixel diagonals than ``findContours``.
    """
    from ._subpixel import extract_subpixel_contours, build_contour_hierarchy

    if adaptive_threshold:
        try:
            import cv2
        except ImportError:  # pragma: no cover
            raise RuntimeError("opencv-python is required for ContourGenerator.")
        binary = _apply_threshold(
            gray, threshold, True, adaptive_c, cv2.THRESH_BINARY_INV
        )
        raw_contours = extract_subpixel_contours(binary, 127.0, min_length, supersample)
    else:
        # Marching squares at iso-level=threshold traces the boundary between
        # dark (< threshold, ink) and light (>= threshold, background) pixels.
        raw_contours = extract_subpixel_contours(gray, float(threshold), min_length, supersample)

    # Keep only closed rings — open contours cannot bound a fill region
    closed_rings = [pts for pts, is_closed in raw_contours if is_closed]

    if not closed_rings:
        return []

    # Build (outer, holes) pairs via geometric containment nesting
    hierarchy_pairs = build_contour_hierarchy(closed_rings)

    if not hierarchy_pairs:
        return []

    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    def ring_to_poly(pts: np.ndarray) -> Polyline | None:
        """Convert a pixel-space ring to a mm Polyline with RDP, Chaikin, close."""
        if len(pts) < min_length:
            return None
        c = pts
        if simplify_tol > 0:
            try:
                import cv2  # noqa: F811
            except ImportError:  # pragma: no cover
                raise RuntimeError("opencv-python is required for ContourGenerator.")
            tol_px = max(0.5, simplify_tol * px_per_mm)
            c = pts.reshape(-1, 1, 2).astype(np.float32)
            c = cv2.approxPolyDP(c, tol_px, closed=True)
            c = c.reshape(-1, 2)
        if len(c) < 2:
            return None
        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in c
        ]
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=True)
        if poly and poly[0] != poly[-1]:
            poly.append(poly[0])
        return poly

    result: list[tuple[Polyline, list[Polyline]]] = []
    for outer_ring, hole_rings in hierarchy_pairs:
        outer_poly = ring_to_poly(outer_ring)
        if outer_poly is None:
            continue
        holes: list[Polyline] = []
        for hole_ring in hole_rings:
            hole_poly = ring_to_poly(hole_ring)
            if hole_poly is not None:
                holes.append(hole_poly)
        result.append((outer_poly, holes))

    return result


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
