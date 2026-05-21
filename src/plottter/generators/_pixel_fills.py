"""Solid-hatch, cross-hatch, and diagonal fill primitives for pixel-art cells.

Each function fills a square cell (defined by its top-left corner and side
length, all in mm) with parallel lines whose spacing is controlled by a
*density* parameter via::

    spacing = lerp(0.6 mm, 0.15 mm, density)

When the optional *polygon* argument is provided (a :class:`shapely.geometry.Polygon`),
each generated line is clipped against it and fragments shorter than 0.5 mm
are discarded.

Public API
----------
fill_solid_hatch(cell_x_mm, cell_y_mm, cell_size_mm, density, polygon=None)
    Horizontal parallel lines.
fill_cross_hatch(cell_x_mm, cell_y_mm, cell_size_mm, density, polygon=None)
    Horizontal *and* vertical parallel lines.
fill_diagonal(cell_x_mm, cell_y_mm, cell_size_mm, density, polygon=None)
    45-degree diagonal lines (bottom-left → top-right).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from plottter.models import Polyline

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_FRAGMENT_MM: float = 0.5  # discard clipped fragments shorter than this
_SPACING_MAX_MM: float = 0.6   # spacing at density = 0  (sparse)
_SPACING_MIN_MM: float = 0.15  # spacing at density = 1  (dense)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lerp_spacing(density: float) -> float:
    """Return line spacing in mm for *density* ∈ [0, 1].

    ``density=0`` → 0.6 mm (sparse), ``density=1`` → 0.15 mm (dense).
    """
    t = max(0.0, min(1.0, float(density)))
    return _SPACING_MAX_MM + (_SPACING_MIN_MM - _SPACING_MAX_MM) * t


def _expected_hatch_count(cell_size_mm: float, density: float) -> int:
    """Number of parallel lines produced inside a square cell.

    Lines start at ``spacing/2`` from the near edge and are spaced
    ``spacing`` apart.  This is the formula used by both
    :func:`fill_solid_hatch` and each axis of :func:`fill_cross_hatch`.
    """
    spacing = _lerp_spacing(density)
    if spacing <= 0.0 or cell_size_mm <= 0.0:
        return 0
    return int((cell_size_mm - spacing / 2.0) / spacing) + 1


def _expected_diagonal_count(cell_size_mm: float, density: float) -> int:
    """Number of 45-degree diagonal lines produced inside a square cell."""
    spacing = _lerp_spacing(density)
    if spacing <= 0.0 or cell_size_mm <= 0.0:
        return 0
    delta_c = spacing * math.sqrt(2.0)
    total_range = 2.0 * cell_size_mm
    return int((total_range - delta_c / 2.0) / delta_c) + 1


def _clip_to_polygon(
    polylines: list[Polyline],
    polygon: Any,
) -> list[Polyline]:
    """Clip *polylines* to *polygon*, dropping fragments shorter than 0.5 mm.

    Parameters
    ----------
    polylines:
        Input list of ``[(x, y), ...]`` polylines in mm.
    polygon:
        A :class:`shapely.geometry.Polygon` used as the clip boundary.

    Returns
    -------
    Filtered list of polylines whose points all lie inside *polygon*.
    """
    from shapely.geometry import LineString  # lazy import

    result: list[Polyline] = []

    for pts in polylines:
        if len(pts) < 2:
            continue
        line = LineString(pts)
        try:
            clipped = polygon.intersection(line)
        except Exception:  # noqa: BLE001
            continue

        if clipped.is_empty:
            continue

        # Normalise to a flat list of LineString geometries
        geom_type = clipped.geom_type
        if geom_type == "LineString":
            candidates = [clipped]
        elif geom_type in ("MultiLineString", "GeometryCollection"):
            candidates = [
                g for g in clipped.geoms if g.geom_type == "LineString"
            ]
        else:
            # Point or other degenerate result — skip
            continue

        for g in candidates:
            if g.length < _MIN_FRAGMENT_MM:
                continue
            coords = list(g.coords)
            if len(coords) >= 2:
                result.append([(float(x), float(y)) for x, y in coords])

    return result


# ---------------------------------------------------------------------------
# Public fill primitives
# ---------------------------------------------------------------------------

def fill_solid_hatch(
    cell_x_mm: float,
    cell_y_mm: float,
    cell_size_mm: float,
    density: float,
    polygon: Any | None = None,
) -> list[Polyline]:
    """Fill a square cell with horizontal parallel lines.

    Parameters
    ----------
    cell_x_mm, cell_y_mm:
        Top-left corner of the cell in mm.
    cell_size_mm:
        Side length of the square cell in mm.
    density:
        Fill density in [0, 1].  0 → 0.6 mm spacing (sparse);
        1 → 0.15 mm spacing (dense).
    polygon:
        Optional :class:`shapely.geometry.Polygon` clip boundary.  Each
        line is clipped against it; fragments < 0.5 mm are discarded.

    Returns
    -------
    List of two-point polylines ``[(x0, y), (x1, y)]``, one per hatch line.
    """
    spacing = _lerp_spacing(density)
    if spacing <= 0.0 or cell_size_mm <= 0.0:
        return []

    x0 = cell_x_mm
    x1 = cell_x_mm + cell_size_mm
    y_end = cell_y_mm + cell_size_mm

    polylines: list[Polyline] = []
    y = cell_y_mm + spacing / 2.0
    while y < y_end:
        polylines.append([(x0, y), (x1, y)])
        y += spacing

    if polygon is not None:
        polylines = _clip_to_polygon(polylines, polygon)

    return polylines


def fill_cross_hatch(
    cell_x_mm: float,
    cell_y_mm: float,
    cell_size_mm: float,
    density: float,
    polygon: Any | None = None,
) -> list[Polyline]:
    """Fill a square cell with horizontal **and** vertical parallel lines.

    The horizontal set is identical to :func:`fill_solid_hatch`; the
    vertical set uses the same spacing but runs top-to-bottom.  Both sets
    share the same *polygon* clip boundary when provided.

    Parameters
    ----------
    cell_x_mm, cell_y_mm:
        Top-left corner of the cell in mm.
    cell_size_mm:
        Side length of the square cell in mm.
    density:
        Fill density in [0, 1].
    polygon:
        Optional :class:`shapely.geometry.Polygon` clip boundary.

    Returns
    -------
    Combined list of horizontal and vertical hatch polylines.
    """
    spacing = _lerp_spacing(density)
    if spacing <= 0.0 or cell_size_mm <= 0.0:
        return []

    x0 = cell_x_mm
    x1 = cell_x_mm + cell_size_mm
    y0 = cell_y_mm
    y1 = cell_y_mm + cell_size_mm

    polylines: list[Polyline] = []

    # Horizontal lines
    y = y0 + spacing / 2.0
    while y < y1:
        polylines.append([(x0, y), (x1, y)])
        y += spacing

    # Vertical lines
    x = x0 + spacing / 2.0
    while x < x1:
        polylines.append([(x, y0), (x, y1)])
        x += spacing

    if polygon is not None:
        polylines = _clip_to_polygon(polylines, polygon)

    return polylines


def fill_diagonal(
    cell_x_mm: float,
    cell_y_mm: float,
    cell_size_mm: float,
    density: float,
    polygon: Any | None = None,
) -> list[Polyline]:
    """Fill a square cell with 45-degree diagonal lines.

    Lines run in the direction (1, 1) (bottom-left to top-right in standard
    screen coordinates where *y* increases downward).  They are spaced
    *spacing* mm apart measured perpendicularly.  In algebraic terms the
    lines satisfy ``y − x = c`` with ``c`` values spaced ``spacing·√2`` mm
    apart.

    Parameters
    ----------
    cell_x_mm, cell_y_mm:
        Top-left corner of the cell in mm.
    cell_size_mm:
        Side length of the square cell in mm.
    density:
        Fill density in [0, 1].
    polygon:
        Optional :class:`shapely.geometry.Polygon` clip boundary.

    Returns
    -------
    List of two-point polylines, one per diagonal line.
    """
    spacing = _lerp_spacing(density)
    if spacing <= 0.0 or cell_size_mm <= 0.0:
        return []

    x0 = cell_x_mm
    y0 = cell_y_mm
    x1 = cell_x_mm + cell_size_mm
    y1 = cell_y_mm + cell_size_mm

    # Lines satisfy  y - x = c.
    # The cell occupies x ∈ [x0, x1], y ∈ [y0, y1].
    # The range of c-values that produce a non-empty intersection:
    #   c_min = y0 - x1  (line through top-right corner)
    #   c_max = y1 - x0  (line through bottom-left corner)
    c_min = y0 - x1
    c_max = y1 - x0
    delta_c = spacing * math.sqrt(2.0)

    polylines: list[Polyline] = []
    c = c_min + delta_c / 2.0

    while c < c_max:
        # Intersect  y = x + c  with the axis-aligned bounding box.
        # Check each of the 4 cell edges and collect valid intersections.
        pts: list[tuple[float, float]] = []

        # Left edge  x = x0
        y_left = x0 + c
        if y0 <= y_left <= y1:
            pts.append((x0, y_left))

        # Bottom edge  y = y1  (larger y = "bottom" in screen coords)
        x_bottom = y1 - c
        if x0 <= x_bottom <= x1:
            pts.append((x_bottom, y1))

        # Right edge  x = x1
        y_right = x1 + c
        if y0 <= y_right <= y1:
            pts.append((x1, y_right))

        # Top edge  y = y0  (smaller y = "top" in screen coords)
        x_top = y0 - c
        if x0 <= x_top <= x1:
            pts.append((x_top, y0))

        # Remove duplicate corners (can occur when c = c_min or c = c_max)
        seen: set[tuple[float, float]] = set()
        unique_pts: list[tuple[float, float]] = []
        for p in pts:
            key = (round(p[0], 9), round(p[1], 9))
            if key not in seen:
                seen.add(key)
                unique_pts.append(p)

        if len(unique_pts) >= 2:
            # Sort by x so the segment direction is consistent
            unique_pts.sort(key=lambda p: p[0])
            polylines.append([unique_pts[0], unique_pts[-1]])

        c += delta_c

    if polygon is not None:
        polylines = _clip_to_polygon(polylines, polygon)

    return polylines


_DIAMOND_RADIUS_MM: float = 0.15  # half-width (~0.3 mm total) of dithered-dot diamonds


def fill_x_mark(
    cell_x_mm: float,
    cell_y_mm: float,
    cell_size_mm: float,
    density: float,
    polygon: Any | None = None,
) -> list[Polyline]:
    """Fill a square cell with an X mark (the two cell corner diagonals).

    Parameters
    ----------
    cell_x_mm, cell_y_mm:
        Top-left corner of the cell in mm.
    cell_size_mm:
        Side length of the square cell in mm.
    density:
        Ignored — the X mark always consists of exactly 2 polylines.
    polygon:
        Optional :class:`shapely.geometry.Polygon` clip boundary.

    Returns
    -------
    Exactly 2 two-point polylines (the two diagonals), clipped to *polygon*
    when provided.
    """
    if cell_size_mm <= 0.0:
        return []

    x0 = cell_x_mm
    y0 = cell_y_mm
    x1 = cell_x_mm + cell_size_mm
    y1 = cell_y_mm + cell_size_mm

    polylines: list[Polyline] = [
        [(x0, y0), (x1, y1)],
        [(x1, y0), (x0, y1)],
    ]

    if polygon is not None:
        polylines = _clip_to_polygon(polylines, polygon)

    return polylines


def fill_outline(
    cell_x_mm: float,
    cell_y_mm: float,
    cell_size_mm: float,
    density: float,
    polygon: Any | None = None,
) -> list[Polyline]:
    """Fill a square cell with its rectangular outline.

    Parameters
    ----------
    cell_x_mm, cell_y_mm:
        Top-left corner of the cell in mm.
    cell_size_mm:
        Side length of the square cell in mm.
    density:
        Ignored — the outline is always a single closed rectangle.
    polygon:
        Optional :class:`shapely.geometry.Polygon` clip boundary.

    Returns
    -------
    A single closed 5-point polyline tracing the cell boundary, or
    multiple fragments when clipped to *polygon*.
    """
    if cell_size_mm <= 0.0:
        return []

    x0 = cell_x_mm
    y0 = cell_y_mm
    x1 = cell_x_mm + cell_size_mm
    y1 = cell_y_mm + cell_size_mm

    polylines: list[Polyline] = [
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
    ]

    if polygon is not None:
        polylines = _clip_to_polygon(polylines, polygon)

    return polylines


def fill_dithered_dots(
    cell_x_mm: float,
    cell_y_mm: float,
    cell_size_mm: float,
    density: float,
    polygon: Any | None = None,
) -> list[Polyline]:
    """Fill a square cell with tiny diamond-shaped dots on a regular grid.

    Each diamond is drawn as a closed 4-segment polyline approximately
    ``2 * _DIAMOND_RADIUS_MM`` (~0.3 mm) wide.  The grid spacing matches
    the hatch spacing returned by :func:`_lerp_spacing`, so dot count
    scales with *density*.

    Parameters
    ----------
    cell_x_mm, cell_y_mm:
        Top-left corner of the cell in mm.
    cell_size_mm:
        Side length of the square cell in mm.
    density:
        Fill density in [0, 1].  0 -> sparse grid; 1 -> dense grid.
    polygon:
        Optional :class:`shapely.geometry.Polygon` clip boundary.

    Returns
    -------
    One closed 5-point polyline per dot, clipped to *polygon* when provided.
    """
    spacing = _lerp_spacing(density)
    if spacing <= 0.0 or cell_size_mm <= 0.0:
        return []

    r = _DIAMOND_RADIUS_MM
    x_end = cell_x_mm + cell_size_mm
    y_end = cell_y_mm + cell_size_mm

    polylines: list[Polyline] = []
    cx = cell_x_mm + spacing / 2.0
    while cx < x_end:
        cy = cell_y_mm + spacing / 2.0
        while cy < y_end:
            diamond: Polyline = [
                (cx,     cy - r),
                (cx + r, cy    ),
                (cx,     cy + r),
                (cx - r, cy    ),
                (cx,     cy - r),
            ]
            polylines.append(diamond)
            cy += spacing
        cx += spacing

    if polygon is not None:
        polylines = _clip_to_polygon(polylines, polygon)

    return polylines


def fill_leave_empty(
    cell_x_mm: float,
    cell_y_mm: float,
    cell_size_mm: float,
    density: float,
    polygon: Any | None = None,
) -> list[Polyline]:
    """Leave the cell empty — always returns an empty list.

    Parameters
    ----------
    cell_x_mm, cell_y_mm, cell_size_mm, density, polygon:
        All parameters are accepted but ignored.

    Returns
    -------
    ``[]``
    """
    return []
