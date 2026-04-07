"""Chaikin corner-cutting smoothing for plotter polylines."""

from __future__ import annotations

from plottter.models import Polyline


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
