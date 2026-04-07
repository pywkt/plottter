"""Polygon fill helpers for ContourGenerator."""

from __future__ import annotations

import math

from plottter.models import Polyline


def _fill_polygon_hatch(
    polygon_pts: Polyline,
    hole_pts_list: list[Polyline],
    angle_deg: float,
    spacing_mm: float,
) -> list[Polyline]:
    """Fill a polygon with parallel hatch lines clipped to its interior.

    Uses Shapely to clip scan lines to the polygon (minus any holes).
    Returns a list of Polylines (line segments inside the polygon).

    Parameters
    ----------
    polygon_pts:    Outer boundary as a list of (x_mm, y_mm) points (closed).
    hole_pts_list:  List of hole boundaries (each a closed Polyline).
    angle_deg:      Angle of hatch lines in degrees (0 = horizontal).
    spacing_mm:     Distance between hatch lines in mm.
    """
    try:
        from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
        from shapely.ops import unary_union
    except ImportError:  # pragma: no cover
        return []

    if len(polygon_pts) < 3:
        return []

    # Build Shapely polygon with optional holes
    shell = [(p[0], p[1]) for p in polygon_pts]
    holes = [[(p[0], p[1]) for p in h] for h in hole_pts_list if len(h) >= 3]
    try:
        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return []
    except Exception:
        return []

    # Handle MultiPolygon (e.g. from self-intersecting contours normalised by
    # buffer(0)): process each sub-polygon independently and combine results.
    if poly.geom_type == "MultiPolygon":
        combined: list[Polyline] = []
        for sub_poly in poly.geoms:
            if sub_poly.is_empty:
                continue
            sub_pts: Polyline = [(float(c[0]), float(c[1])) for c in sub_poly.exterior.coords]
            sub_holes_pts: list[Polyline] = [
                [(float(c[0]), float(c[1])) for c in ring.coords]
                for ring in sub_poly.interiors
            ]
            combined.extend(_fill_polygon_hatch(sub_pts, sub_holes_pts, angle_deg, spacing_mm))
        return combined

    # Rotate coordinate system so hatch lines are always horizontal,
    # then rotate results back.
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cos_na = math.cos(-angle_rad)
    sin_na = math.sin(-angle_rad)

    def rotate(x: float, y: float, c: float, s: float) -> tuple[float, float]:
        return (c * x - s * y, s * x + c * y)

    # Get bounds of rotated polygon
    coords = list(poly.exterior.coords)
    rot_coords = [rotate(x, y, cos_a, sin_a) for x, y in coords]
    xs = [c[0] for c in rot_coords]
    ys = [c[1] for c in rot_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if spacing_mm <= 0:
        return []

    result: list[Polyline] = []
    y = min_y
    while y <= max_y + spacing_mm:
        # Create a long scan line in the rotated coordinate system
        x0 = min_x - 1.0
        x1 = max_x + 1.0
        # Rotate the scan line back to original coordinates
        p0 = rotate(x0, y, cos_na, sin_na)
        p1 = rotate(x1, y, cos_na, sin_na)
        scan = LineString([p0, p1])

        try:
            clipped = poly.intersection(scan)
        except Exception:
            y += spacing_mm
            continue

        if clipped.is_empty:
            y += spacing_mm
            continue

        # Collect all line segments from the intersection result
        geoms = []
        if clipped.geom_type == "LineString":
            geoms = [clipped]
        elif clipped.geom_type in ("MultiLineString", "GeometryCollection"):
            geoms = [g for g in clipped.geoms if g.geom_type == "LineString"]

        for geom in geoms:
            pts = list(geom.coords)
            if len(pts) >= 2:
                result.append([(float(p[0]), float(p[1])) for p in pts])

        y += spacing_mm

    return result


def _fill_polygon_concentric(
    polygon_pts: Polyline,
    hole_pts_list: list[Polyline],
    spacing_mm: float,
) -> list[Polyline]:
    """Fill a polygon with progressively inward offset contours (concentric).

    Uses Shapely buffer(-offset) to compute inward offsets until the shape
    collapses.  Each offset ring is added as a separate Polyline.

    Parameters
    ----------
    polygon_pts:    Outer boundary as a list of (x_mm, y_mm) points.
    hole_pts_list:  List of hole boundaries (each a closed Polyline).
    spacing_mm:     Inward offset step in mm.
    """
    try:
        from shapely.geometry import MultiPolygon, Polygon
    except ImportError:  # pragma: no cover
        return []

    if len(polygon_pts) < 3 or spacing_mm <= 0:
        return []

    shell = [(p[0], p[1]) for p in polygon_pts]
    holes = [[(p[0], p[1]) for p in h] for h in hole_pts_list if len(h) >= 3]
    try:
        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return []
    except Exception:
        return []

    result: list[Polyline] = []
    offset = spacing_mm

    # Safety cap: max 500 rings to avoid infinite loops on degenerate shapes
    for _ in range(500):
        shrunk = poly.buffer(-offset)
        if shrunk.is_empty:
            break

        # Collect all exterior rings from the shrunk geometry
        polys = []
        if shrunk.geom_type == "Polygon":
            polys = [shrunk]
        elif shrunk.geom_type == "MultiPolygon":
            polys = list(shrunk.geoms)
        elif shrunk.geom_type == "GeometryCollection":
            polys = [g for g in shrunk.geoms if g.geom_type == "Polygon"]

        for p in polys:
            coords = list(p.exterior.coords)
            if len(coords) >= 2:
                poly_line: Polyline = [(float(c[0]), float(c[1])) for c in coords]
                # Close if not already closed
                if poly_line[0] != poly_line[-1]:
                    poly_line.append(poly_line[0])
                result.append(poly_line)

        offset += spacing_mm

    return result
