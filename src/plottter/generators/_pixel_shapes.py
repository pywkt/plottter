"""Cell polygon generators for PixelArtGenerator cell shapes.

``cell_polygon`` returns a vertex list that defines the clip boundary for a
single pixel-art cell.  The caller converts the list to a
:class:`shapely.geometry.Polygon` and passes it to fill routines so that hatch
lines are clipped to the desired shape.

All coordinates are in millimetres.  The cell origin is at
``(cell_x_mm, cell_y_mm)`` (top-left corner, screen y-down).
"""

from __future__ import annotations

import math


def cell_polygon(
    shape: str,
    cell_x_mm: float,
    cell_y_mm: float,
    size_mm: float,
) -> list[tuple[float, float]] | None:
    """Return boundary polygon vertices for one pixel-art cell.

    Parameters
    ----------
    shape:
        Cell shape name.  One of ``square``, ``diamond``, ``octagonal``,
        ``circle``, ``rounded_square``.
    cell_x_mm, cell_y_mm:
        Top-left corner of the cell in mm.
    size_mm:
        Cell side length in mm (the fill area after applying the cell gap).

    Returns
    -------
    ``None`` for *square* — the fill routines already operate on a square
    region so no polygon clipping is needed.

    A ``list[tuple[float, float]]`` of ``(x, y)`` vertices in clockwise order
    (screen coordinates, y-down) for all other shapes:

    * ``diamond``        — 4 vertices
    * ``octagonal``      — 8 vertices
    * ``circle``         — 24 vertices (regular polygon approximation)
    * ``rounded_square`` — 28 vertices (4 corners × 7 arc points each)
    """
    if shape == "square":
        return None
    if shape == "diamond":
        return _diamond(cell_x_mm, cell_y_mm, size_mm)
    if shape == "octagonal":
        return _octagonal(cell_x_mm, cell_y_mm, size_mm)
    if shape == "circle":
        return _circle(cell_x_mm, cell_y_mm, size_mm)
    if shape == "rounded_square":
        return _rounded_square(cell_x_mm, cell_y_mm, size_mm)
    # Unknown shape — treat as square (no clipping).
    return None


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _diamond(
    cell_x_mm: float,
    cell_y_mm: float,
    size_mm: float,
) -> list[tuple[float, float]]:
    """4-vertex diamond (rhombus) centred in the cell, pointing at the edges."""
    cx = cell_x_mm + size_mm / 2.0
    cy = cell_y_mm + size_mm / 2.0
    r = size_mm / 2.0
    return [
        (cx,     cy - r),  # top
        (cx + r, cy    ),  # right
        (cx,     cy + r),  # bottom
        (cx - r, cy    ),  # left
    ]


def _octagonal(
    cell_x_mm: float,
    cell_y_mm: float,
    size_mm: float,
) -> list[tuple[float, float]]:
    """8-vertex octagon inscribed in the square cell.

    Corner cut = 0.2 × side length (spec §7.3).
    """
    x0, y0 = cell_x_mm, cell_y_mm
    x1, y1 = x0 + size_mm, y0 + size_mm
    c = size_mm * 0.2
    return [
        (x0 + c, y0    ),
        (x1 - c, y0    ),
        (x1,     y0 + c),
        (x1,     y1 - c),
        (x1 - c, y1    ),
        (x0 + c, y1    ),
        (x0,     y1 - c),
        (x0,     y0 + c),
    ]


def _circle(
    cell_x_mm: float,
    cell_y_mm: float,
    size_mm: float,
    n_verts: int = 24,
) -> list[tuple[float, float]]:
    """*n_verts*-vertex regular polygon approximating the inscribed circle.

    All vertices lie exactly on the circle of radius ``size_mm / 2`` centred
    in the cell, so the polygon is strictly *inside* the true circle.
    """
    cx = cell_x_mm + size_mm / 2.0
    cy = cell_y_mm + size_mm / 2.0
    r = size_mm / 2.0
    step = 2.0 * math.pi / n_verts
    return [
        (cx + r * math.cos(i * step), cy + r * math.sin(i * step))
        for i in range(n_verts)
    ]


def _rounded_square(
    cell_x_mm: float,
    cell_y_mm: float,
    size_mm: float,
    corner_ratio: float = 0.2,
    pts_per_corner: int = 7,
) -> list[tuple[float, float]]:
    """28-vertex rounded-square polygon.

    Each of the 4 corners is approximated by *pts_per_corner* (default 7)
    points evenly spaced over the 90° arc.  The straight flat edges between
    adjacent corners are implicit polygon edges — no extra vertices are
    inserted.  Total vertex count: ``4 × pts_per_corner = 28``.

    The corner radius is ``size_mm × corner_ratio`` (default 20 % of the cell
    side per spec §7.5, so the flat edges are 60 % of the side length).

    Vertices are emitted clockwise starting from the top-right corner arc.
    """
    x0, y0 = cell_x_mm, cell_y_mm
    x1, y1 = x0 + size_mm, y0 + size_mm
    r = size_mm * corner_ratio

    # Each entry: (arc_center_x, arc_center_y, start_angle_deg, end_angle_deg)
    # Clockwise traversal in screen coordinates (y-down).
    corners = [
        (x1 - r, y0 + r, 270.0, 360.0),  # top-right
        (x1 - r, y1 - r,   0.0,  90.0),  # bottom-right
        (x0 + r, y1 - r,  90.0, 180.0),  # bottom-left
        (x0 + r, y0 + r, 180.0, 270.0),  # top-left
    ]

    verts: list[tuple[float, float]] = []
    for acx, acy, a_start, a_end in corners:
        for i in range(pts_per_corner):
            frac = i / (pts_per_corner - 1)
            angle_rad = math.radians(a_start + frac * (a_end - a_start))
            verts.append((acx + r * math.cos(angle_rad), acy + r * math.sin(angle_rad)))

    return verts


# ---------------------------------------------------------------------------
# Hex shape (public — used directly by the hex grid code path)
# ---------------------------------------------------------------------------


def hex_polygon(
    cx_mm: float,
    cy_mm: float,
    radius_mm: float,
) -> list[tuple[float, float]]:
    """6-vertex flat-topped hexagon centred at *(cx_mm, cy_mm)*.

    Parameters
    ----------
    cx_mm, cy_mm:
        Centre of the hexagon in mm.
    radius_mm:
        Circumradius (centre-to-vertex distance) in mm.

    Returns
    -------
    6 vertices starting from the rightmost vertex (angle 0°), suitable for
    use as a :class:`shapely.geometry.Polygon`.

    Flat-topped means the top and bottom edges are horizontal.  Vertex
    angles are 0°, 60°, 120°, 180°, 240°, 300° (y-down screen coordinates).
    """
    verts: list[tuple[float, float]] = []
    for i in range(6):
        angle = math.radians(i * 60)
        verts.append(
            (cx_mm + radius_mm * math.cos(angle), cy_mm + radius_mm * math.sin(angle))
        )
    return verts
