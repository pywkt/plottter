"""Ramer-Douglas-Peucker path simplification."""

import math
from plottter.models.path import Polyline, Point


def _perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """Perpendicular distance from point to the line defined by line_start and line_end."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0.0:
        # Degenerate segment — just return distance to line_start
        return math.hypot(x0 - x1, y0 - y1)

    # Parameterize and find closest point on the segment (as line, unbounded)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / length_sq
    px = x1 + t * dx
    py = y1 + t * dy
    return math.hypot(x0 - px, y0 - py)


def _rdp(points: list[Point], tolerance: float, start: int, end: int, mask: list[bool]) -> None:
    """Recursive RDP. Sets mask[i] = True for points to keep."""
    if end <= start + 1:
        return

    max_dist = 0.0
    max_idx = start

    for i in range(start + 1, end):
        dist = _perpendicular_distance(points[i], points[start], points[end])
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > tolerance:
        _rdp(points, tolerance, start, max_idx, mask)
        mask[max_idx] = True
        _rdp(points, tolerance, max_idx, end, mask)


def simplify_polyline(polyline: Polyline, tolerance_mm: float = 0.1) -> Polyline:
    """Simplify a single polyline using Ramer-Douglas-Peucker.

    Preserves the first and last points. Removes intermediate points whose
    perpendicular distance to the segment is within *tolerance_mm*.
    """
    n = len(polyline)
    if n < 3:
        return list(polyline)

    mask = [False] * n
    mask[0] = True
    mask[n - 1] = True

    _rdp(polyline, tolerance_mm, 0, n - 1, mask)

    return [p for p, keep in zip(polyline, mask) if keep]


def simplify_paths(paths: list[Polyline], tolerance_mm: float = 0.1) -> list[Polyline]:
    """Simplify a list of polylines using Ramer-Douglas-Peucker.

    Args:
        paths: Input list of polylines (each polyline is a list of (x, y) points in mm).
        tolerance_mm: Maximum allowed perpendicular deviation (mm). Points within this
            distance of the simplified line are removed. Default 0.1mm.

    Returns:
        New list of simplified polylines. Empty polylines are preserved as-is.
    """
    result: list[Polyline] = []
    for polyline in paths:
        result.append(simplify_polyline(polyline, tolerance_mm))
    return result
