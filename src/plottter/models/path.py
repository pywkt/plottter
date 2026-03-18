"""Type aliases and helper functions for 2D path data."""

import math

# Type aliases
Point = tuple[float, float]
Polyline = list[Point]


def polyline_length(polyline: Polyline) -> float:
    """Return the sum of Euclidean distances between consecutive points."""
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for i in range(len(polyline) - 1):
        x0, y0 = polyline[i]
        x1, y1 = polyline[i + 1]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def polyline_bounds(polyline: Polyline) -> tuple[Point, Point]:
    """Return the bounding box as ((min_x, min_y), (max_x, max_y)).

    Raises ValueError for empty polylines.
    """
    if not polyline:
        raise ValueError("Cannot compute bounds of an empty polyline")
    xs = [p[0] for p in polyline]
    ys = [p[1] for p in polyline]
    return (min(xs), min(ys)), (max(xs), max(ys))
