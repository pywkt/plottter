"""Triangle hatching fill for 3D scene rendering.

Provides functions to fill projected 2D triangles with parallel hatching lines,
with density mapped from shading brightness.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, MultiLineString, GeometryCollection, Polygon

_AREA_EPSILON = 1e-6


def _fill_triangle_with_hatching(
    verts_2d: list[tuple[float, float]],
    density: float,
    angle_deg: float,
    cross_hatch: bool,
) -> list[list[tuple[float, float]]]:
    """Fill a triangle with parallel hatching lines.

    Parameters
    ----------
    verts_2d:    3 projected 2D vertices in mm coordinates.
    density:     Hatching density in lines per mm.  A density of 0 or negative
                 produces no lines.
    angle_deg:   Angle (degrees) of the hatching lines measured from the
                 positive x-axis.
    cross_hatch: If True, a second pass of lines at ``angle_deg + 90°`` is
                 appended to the result.

    Returns
    -------
    List of polylines; each polyline is a list of ``(x, y)`` tuples in mm.
    Degenerate triangles (area < epsilon) return an empty list.
    """
    if len(verts_2d) < 3 or density <= 0.0:
        return []

    poly = Polygon(verts_2d)
    if poly.area < _AREA_EPSILON:
        return []

    spacing = 1.0 / density

    result = _hatch_polygon(poly, spacing, angle_deg)
    if cross_hatch:
        result.extend(_hatch_polygon(poly, spacing, angle_deg + 90.0))

    return result


def _hatch_polygon(
    poly: Polygon,
    spacing: float,
    angle_deg: float,
) -> list[list[tuple[float, float]]]:
    """Generate hatching lines clipped to a Shapely polygon.

    Lines run parallel to ``angle_deg`` and are spaced ``spacing`` mm apart
    in the perpendicular direction.

    Parameters
    ----------
    poly:      Shapely Polygon to hatch.
    spacing:   Distance between adjacent hatch lines in mm.
    angle_deg: Direction of hatch lines in degrees from the x-axis.

    Returns
    -------
    List of polylines clipped to the polygon interior.
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Unit vector along the hatch lines and perpendicular to them.
    line_dir = np.array([cos_a, sin_a])
    perp_dir = np.array([-sin_a, cos_a])

    # Project polygon vertices onto the perpendicular direction to find the
    # sweep range of hatch lines.
    coords = np.array(poly.exterior.coords)  # (N+1, 2)
    perp_proj = coords @ perp_dir
    t_min = perp_proj.min()
    t_max = perp_proj.max()

    # Compute the polygon's extent along the line direction so that each
    # generated segment fully spans the polygon regardless of where it sits
    # in 2D space.  Using a symmetric half-length around t * perp_dir is
    # wrong for polygons far from the origin.
    line_proj = coords @ line_dir
    min_para = line_proj.min() - spacing
    max_para = line_proj.max() + spacing

    total_range = t_max - t_min
    max_iterations = int(total_range / spacing) + 2  # guard against infinite loops

    result: list[list[tuple[float, float]]] = []
    t = t_min
    iteration = 0
    while t <= t_max + spacing * 0.5:
        if iteration > max_iterations:
            break
        iteration += 1

        # Point on the perpendicular axis; the line passes through this point
        # and runs in line_dir from min_para to max_para.
        perp_pt = t * perp_dir
        p1 = tuple(perp_pt + min_para * line_dir)
        p2 = tuple(perp_pt + max_para * line_dir)

        line = LineString([p1, p2])
        clipped = line.intersection(poly)

        if not clipped.is_empty:
            for seg in _extract_linestrings(clipped):
                seg_coords = list(seg.coords)
                if len(seg_coords) >= 2:
                    result.append([(float(c[0]), float(c[1])) for c in seg_coords])

        t += spacing

    return result


def _extract_linestrings(geom) -> list[LineString]:
    """Recursively extract all LineString geometries from a Shapely geometry."""
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        result: list[LineString] = []
        for g in geom.geoms:
            result.extend(_extract_linestrings(g))
        return result
    return []


def brightness_to_density(
    brightness: float,
    min_density: float,
    max_density: float,
) -> float:
    """Map a shading brightness value to a hatching line density.

    Darker faces (brightness near 0) get denser hatching (``max_density``).
    Fully lit faces (brightness = 1) get sparse hatching (``min_density``).

    Parameters
    ----------
    brightness:   Diffuse shading brightness in [0, 1].
    min_density:  Hatching density for fully lit faces (lines per mm).
    max_density:  Hatching density for fully dark faces (lines per mm).

    Returns
    -------
    Density in lines per mm.
    """
    brightness = max(0.0, min(1.0, float(brightness)))
    return min_density + (1.0 - brightness) * (max_density - min_density)
