"""Cohen-Sutherland line clipping for polylines."""

from plottter.models.path import Polyline, Point

# Outcode bit flags
_LEFT = 1
_RIGHT = 2
_BOTTOM = 4
_TOP = 8


def _outcode(x: float, y: float, xmin: float, ymin: float, xmax: float, ymax: float) -> int:
    code = 0
    if x < xmin:
        code |= _LEFT
    elif x > xmax:
        code |= _RIGHT
    if y < ymin:
        code |= _BOTTOM
    elif y > ymax:
        code |= _TOP
    return code


def _clip_segment(
    x1: float, y1: float,
    x2: float, y2: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> tuple[float, float, float, float] | None:
    """Cohen-Sutherland segment clip.

    Returns (cx1, cy1, cx2, cy2) of the clipped segment, or None if entirely outside.
    """
    code1 = _outcode(x1, y1, xmin, ymin, xmax, ymax)
    code2 = _outcode(x2, y2, xmin, ymin, xmax, ymax)

    while True:
        if not (code1 | code2):
            # Both inside
            return x1, y1, x2, y2
        if code1 & code2:
            # Trivially outside
            return None

        # Pick an outside point
        code_out = code1 if code1 else code2

        dx = x2 - x1
        dy = y2 - y1

        if code_out & _TOP:
            t = (ymax - y1) / dy if dy != 0 else 0.0
            x = x1 + t * dx
            y = ymax
        elif code_out & _BOTTOM:
            t = (ymin - y1) / dy if dy != 0 else 0.0
            x = x1 + t * dx
            y = ymin
        elif code_out & _RIGHT:
            t = (xmax - x1) / dx if dx != 0 else 0.0
            x = xmax
            y = y1 + t * dy
        else:  # LEFT
            t = (xmin - x1) / dx if dx != 0 else 0.0
            x = xmin
            y = y1 + t * dy

        if code_out == code1:
            x1, y1 = x, y
            code1 = _outcode(x1, y1, xmin, ymin, xmax, ymax)
        else:
            x2, y2 = x, y
            code2 = _outcode(x2, y2, xmin, ymin, xmax, ymax)


def _clip_polyline(
    polyline: Polyline,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> list[Polyline]:
    """Clip a single polyline to bounds, potentially producing multiple output polylines."""
    if len(polyline) < 2:
        return []

    result: list[Polyline] = []
    current: Polyline = []

    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]

        clipped = _clip_segment(x1, y1, x2, y2, xmin, ymin, xmax, ymax)

        if clipped is None:
            # Segment outside — flush current stroke
            if len(current) >= 2:
                result.append(current)
            current = []
        else:
            cx1, cy1, cx2, cy2 = clipped
            # near_clipped: the segment's start was moved, meaning the path
            # re-entered the bounds from outside — must begin a new stroke.
            near_clipped = not (abs(cx1 - x1) < 1e-9 and abs(cy1 - y1) < 1e-9)
            if not current:
                current = [(cx1, cy1), (cx2, cy2)]
            else:
                lx, ly = current[-1]
                if near_clipped or not (abs(cx1 - lx) < 1e-9 and abs(cy1 - ly) < 1e-9):
                    # Re-entry or gap introduced by clipping — flush and start new stroke
                    if len(current) >= 2:
                        result.append(current)
                    current = [(cx1, cy1), (cx2, cy2)]
                else:
                    current.append((cx2, cy2))

    if len(current) >= 2:
        result.append(current)

    return result


def clip_to_bounds(
    paths: list[Polyline],
    bounds: tuple[float, float, float, float],
) -> list[Polyline]:
    """Clip all polylines to a rectangular bounding box.

    Paths that cross the boundary are split at the intersection points. A
    single input polyline may produce multiple output polylines.

    Args:
        paths: Input list of polylines (mm coordinates).
        bounds: ``(xmin, ymin, xmax, ymax)`` in mm.

    Returns:
        New list of clipped polylines. Paths entirely outside the bounds are
        dropped. Paths entirely inside are returned unchanged.
    """
    xmin, ymin, xmax, ymax = bounds
    result: list[Polyline] = []
    for polyline in paths:
        result.extend(_clip_polyline(polyline, xmin, ymin, xmax, ymax))
    return result
