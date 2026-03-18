"""Minimum-length path filter — removes artifact polylines."""

import math
from plottter.models.path import Polyline


def _polyline_length(polyline: Polyline) -> float:
    """Total Euclidean length of a polyline (sum of segment distances)."""
    total = 0.0
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def filter_short_paths(paths: list[Polyline], min_length_mm: float = 0.5) -> list[Polyline]:
    """Remove polylines whose total length is less than *min_length_mm*.

    Args:
        paths: Input list of polylines.
        min_length_mm: Minimum allowed path length (mm). Paths shorter than
            this are removed. Default 0.5mm.

    Returns:
        New list containing only paths with total length >= min_length_mm.
        Paths with fewer than 2 points (zero length) are always removed.
    """
    return [p for p in paths if _polyline_length(p) >= min_length_mm]
