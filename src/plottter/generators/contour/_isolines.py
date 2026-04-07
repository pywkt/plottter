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
