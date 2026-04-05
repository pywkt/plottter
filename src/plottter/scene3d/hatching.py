"""Triangle hatching fill for 3D scene rendering.

Provides functions to fill projected 2D triangles with parallel hatching lines,
with density mapped from shading brightness.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import LineString, MultiLineString, GeometryCollection, Polygon

if TYPE_CHECKING:
    pass

_AREA_EPSILON = 1e-6


def compute_vanishing_point(
    camera: Any,
    hatch_angle_deg: float,
    canvas_w_mm: float,
    canvas_h_mm: float,
    offset_mm: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float] | None:
    """Compute the 2D canvas vanishing point for a 3D hatch direction.

    The hatch direction in world space is derived from *hatch_angle_deg*:

    * ``angle=0°``  → world X direction (horizontal): ``d = (1, 0, 0)``
    * ``angle=90°`` → world Z direction (into scene):  ``d = (0, 0, 1)``

    The vanishing point is the canvas coordinate (in mm) where parallel lines
    receding in direction *d* would converge.  Returns ``None`` when the
    projection is at infinity (no finite vanishing point — falls back to
    parallel hatching).

    Parameters
    ----------
    camera:          Camera with ``view_proj_matrix()`` method.
    hatch_angle_deg: Hatch direction angle in degrees (0 = horizontal / X axis).
    canvas_w_mm:     Canvas width in millimetres.
    canvas_h_mm:     Canvas height in millimetres.
    offset_mm:       Canvas (x, y) offset applied after projection.
    """
    angle_rad = math.radians(hatch_angle_deg)

    # World-space direction (Y-up world, XZ horizontal plane)
    dx = math.cos(angle_rad)
    dz = math.sin(angle_rad)

    # Project the homogeneous direction [dx, 0, dz, 0] through the VP matrix.
    vp_matrix = camera.view_proj_matrix()
    d_hom = np.array([dx, 0.0, dz, 0.0], dtype=np.float64)
    clip = d_hom @ vp_matrix

    w = float(clip[3])
    if abs(w) < 1e-9:
        # Direction has no finite vanishing point in this projection.
        return None

    ndc_x = float(clip[0]) / w
    ndc_y = float(clip[1]) / w

    x_off, y_off = offset_mm
    vp_x = (ndc_x + 1.0) * 0.5 * canvas_w_mm + x_off
    vp_y = (1.0 - (ndc_y + 1.0) * 0.5) * canvas_h_mm + y_off
    return (float(vp_x), float(vp_y))


def _fill_triangle_with_hatching(
    verts_2d: list[tuple[float, float]],
    density: float,
    angle_deg: float,
    cross_hatch: bool,
    vanishing_point: tuple[float, float] | None = None,
) -> list[list[tuple[float, float]]]:
    """Fill a triangle with hatching lines (parallel or perspective-convergent).

    Parameters
    ----------
    verts_2d:         3 projected 2D vertices in mm coordinates.
    density:          Hatching density in lines per mm.  A density of 0 or
                      negative produces no lines.
    angle_deg:        Angle (degrees) of the hatching lines from the x-axis.
    cross_hatch:      If True, a second pass at ``angle_deg + 90°`` is added.
    vanishing_point:  When not ``None``, use perspective-convergent hatching
                      radiating from this 2D canvas point (mm) instead of
                      parallel hatching.

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

    if vanishing_point is not None:
        result = _hatch_polygon_perspective(poly, spacing, angle_deg, vanishing_point)
        if cross_hatch:
            result.extend(
                _hatch_polygon_perspective(poly, spacing, angle_deg + 90.0, vanishing_point)
            )
    else:
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


def _hatch_polygon_perspective(
    poly: Polygon,
    spacing: float,
    angle_deg: float,
    vanishing_point: tuple[float, float],
) -> list[list[tuple[float, float]]]:
    """Generate convergent hatching lines from a vanishing point, clipped to polygon.

    Lines radiate from *vanishing_point*.  The angular step is chosen so that
    adjacent lines are *spacing* mm apart at the polygon centroid (not at the
    vanishing point, which would make them infinitely dense there).

    Falls back to :func:`_hatch_polygon` when the vanishing point is so close to
    the centroid that the angular spacing is ill-defined.

    Parameters
    ----------
    poly:             Shapely Polygon to hatch.
    spacing:          Target distance between lines at the polygon centroid (mm).
    angle_deg:        Nominal hatch angle in degrees (used only for the parallel
                      fallback).
    vanishing_point:  2D canvas position (mm) from which lines radiate.
    """
    from shapely.geometry import Point as _ShapelyPoint

    vp_x, vp_y = vanishing_point

    # Polygon centroid
    cx = float(poly.centroid.x)
    cy = float(poly.centroid.y)

    # Distance from VP to centroid
    dx = cx - vp_x
    dy = cy - vp_y
    r = math.hypot(dx, dy)

    if r < 1e-6:
        # VP coincides with centroid — fall back to parallel hatching.
        return _hatch_polygon(poly, spacing, angle_deg)

    # Angular spacing so lines are `spacing` mm apart at distance r.
    angular_spacing = spacing / r  # radians

    if angular_spacing < 1e-12:
        return []

    # Base angle: direction from VP toward the polygon centroid.
    base_angle = math.atan2(dy, dx)

    # Compute the angular range that covers all polygon vertices (as seen from VP).
    coords = np.array(poly.exterior.coords[:-1])  # exclude duplicate closing point
    vecs = coords - np.array([vp_x, vp_y])
    vertex_angles = np.arctan2(vecs[:, 1], vecs[:, 0])

    # Normalize vertex angles relative to base_angle into [-π, π].
    rel_angles = vertex_angles - base_angle
    rel_angles = (rel_angles + math.pi) % (2.0 * math.pi) - math.pi

    min_rel = float(rel_angles.min())
    max_rel = float(rel_angles.max())

    # When the VP lies inside the polygon, the angular range must span the full circle.
    if poly.contains(_ShapelyPoint(vp_x, vp_y)):
        min_rel = -math.pi
        max_rel = math.pi - angular_spacing  # avoid duplicating the ±π line

    # Add one step of margin on each side.
    min_rel -= angular_spacing
    max_rel += angular_spacing

    # Prevent accidental double-coverage beyond a full circle.
    if max_rel - min_rel >= 2.0 * math.pi:
        min_rel = -math.pi
        max_rel = math.pi

    # Length of each ray — long enough to span the polygon from any direction.
    minx, miny, maxx, maxy = poly.bounds
    extent = math.hypot(maxx - minx, maxy - miny) + r + 10.0

    result: list[list[tuple[float, float]]] = []
    max_iter = int((max_rel - min_rel) / angular_spacing) + 2
    theta = min_rel
    iteration = 0

    while theta <= max_rel + angular_spacing * 0.5:
        if iteration > max_iter:
            break
        iteration += 1

        abs_angle = base_angle + theta
        cos_t = math.cos(abs_angle)
        sin_t = math.sin(abs_angle)

        # Full line through the vanishing point in direction (cos_t, sin_t).
        p1 = (vp_x - cos_t * extent, vp_y - sin_t * extent)
        p2 = (vp_x + cos_t * extent, vp_y + sin_t * extent)

        line = LineString([p1, p2])
        clipped = line.intersection(poly)

        if not clipped.is_empty:
            for seg in _extract_linestrings(clipped):
                seg_coords = list(seg.coords)
                if len(seg_coords) >= 2:
                    result.append([(float(c[0]), float(c[1])) for c in seg_coords])

        theta += angular_spacing

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
