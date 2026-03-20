"""Path offset post-processing using Shapely geometry.

Generates parallel offset copies of polylines at specified distances.
Supports single-side (left/right) or both-side offsets, multiple offset
copies, and various join styles for corners.
"""

from __future__ import annotations

from plottter.models.path import Polyline

try:
    from shapely.geometry import LineString, LinearRing, Polygon
    import shapely
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

_MIN_DISTANCE_MM = 0.01  # Skip offsets smaller than this


def _join_style_value(join_style: str) -> int:
    """Convert join style name to Shapely join_style integer."""
    styles = {"round": 1, "mitre": 2, "bevel": 3}
    return styles.get(join_style.lower(), 1)


def _linestring_to_polyline(geom) -> list[Polyline]:
    """Extract polylines from a Shapely geometry (LineString or MultiLineString)."""
    result: list[Polyline] = []
    geom_type = geom.geom_type
    if geom_type == "LineString":
        coords = list(geom.coords)
        if len(coords) >= 2:
            result.append([(x, y) for x, y in coords])
    elif geom_type == "MultiLineString":
        for part in geom.geoms:
            coords = list(part.coords)
            if len(coords) >= 2:
                result.append([(x, y) for x, y in coords])
    elif geom_type == "GeometryCollection":
        for geom_part in geom.geoms:
            result.extend(_linestring_to_polyline(geom_part))
    return result


def _offset_single_path(
    polyline: Polyline,
    distance_mm: float,
    sides: str,
    count: int,
    join_style: str,
    include_original: bool,
) -> list[Polyline]:
    """Compute offset copies for a single polyline.

    Args:
        polyline: Input path as list of (x, y) points in mm.
        distance_mm: Offset distance per step in mm.
        sides: "both", "left", or "right".
        count: Number of offset copies per side.
        join_style: Corner join style: "round", "mitre", or "bevel".
        include_original: Whether to include the original path in output.

    Returns:
        List of offset polylines (plus original if include_original is True).
    """
    if len(polyline) < 2:
        return []
    if distance_mm < _MIN_DISTANCE_MM:
        return [list(polyline)] if include_original else []

    result: list[Polyline] = []
    if include_original:
        result.append(list(polyline))

    js = _join_style_value(join_style)

    # Determine which sides to offset
    if sides == "left":
        offset_signs = [1]
    elif sides == "right":
        offset_signs = [-1]
    else:  # "both"
        offset_signs = [1, -1]

    line = LineString(polyline)

    for sign in offset_signs:
        for i in range(1, count + 1):
            d = distance_mm * i * sign
            try:
                # Use offset_curve (Shapely >= 2.0) for clean single-side offsets
                offset_geom = line.offset_curve(d, join_style=js, mitre_limit=5.0)
                if offset_geom is None or offset_geom.is_empty:
                    continue
                # Repair self-intersecting results
                if not offset_geom.is_valid:
                    offset_geom = shapely.make_valid(offset_geom)
                parts = _linestring_to_polyline(offset_geom)
                result.extend(parts)
            except Exception:
                # Fallback: skip this offset on error
                continue

    return result


def _offset_closed_path(
    polyline: Polyline,
    distance_mm: float,
    sides: str,
    count: int,
    join_style: str,
    include_original: bool,
) -> list[Polyline]:
    """Compute offset copies for a closed polyline (first == last point).

    Uses buffer on a LinearRing to produce concentric rings.

    Args:
        polyline: Closed input path (first point == last point).
        distance_mm: Offset distance per step in mm.
        sides: "both", "left"/"right" (outside/inside for closed paths).
        count: Number of offset copies per side.
        join_style: Corner join style.
        include_original: Whether to include the original path in output.

    Returns:
        List of offset polylines.
    """
    if len(polyline) < 4:  # Need at least 3 unique points + closure
        return [list(polyline)] if include_original else []
    if distance_mm < _MIN_DISTANCE_MM:
        return [list(polyline)] if include_original else []

    result: list[Polyline] = []
    if include_original:
        result.append(list(polyline))

    js = _join_style_value(join_style)

    # Map sides to offset directions for closed paths:
    # positive distance → outside (larger ring), negative → inside (smaller ring)
    if sides == "left":
        offset_signs = [1]   # outside
    elif sides == "right":
        offset_signs = [-1]  # inside
    else:  # "both"
        offset_signs = [1, -1]

    ring_coords = polyline[:-1]  # Drop closing duplicate point
    ring = LinearRing(ring_coords)
    # Use Polygon for inward (negative) buffering — LinearRing.buffer(-d) is always empty
    poly = Polygon(ring)

    for sign in offset_signs:
        for i in range(1, count + 1):
            d = distance_mm * i * sign
            try:
                if d > 0:
                    # Outward expansion: buffer the ring
                    buffered = ring.buffer(d, cap_style=1, join_style=js, quad_segs=16)
                else:
                    # Inward contraction: buffer the polygon (negative shrinks it)
                    buffered = poly.buffer(d, join_style=js, quad_segs=16)
                if buffered is None or buffered.is_empty:
                    continue
                # Repair self-intersecting results
                if not buffered.is_valid:
                    buffered = shapely.make_valid(buffered)
                if buffered.is_empty:
                    continue

                # Extract rings from the buffered polygon
                geom_type = buffered.geom_type
                if geom_type == "Polygon":
                    ext = list(buffered.exterior.coords)
                    if len(ext) >= 2:
                        result.append([(x, y) for x, y in ext])
                    for interior in buffered.interiors:
                        int_coords = list(interior.coords)
                        if len(int_coords) >= 2:
                            result.append([(x, y) for x, y in int_coords])
                elif geom_type == "MultiPolygon":
                    for part in buffered.geoms:
                        ext = list(part.exterior.coords)
                        if len(ext) >= 2:
                            result.append([(x, y) for x, y in ext])
                        for interior in part.interiors:
                            int_coords = list(interior.coords)
                            if len(int_coords) >= 2:
                                result.append([(x, y) for x, y in int_coords])
            except Exception:
                continue

    return result


def offset_paths(
    paths: list[Polyline],
    distance_mm: float = 0.5,
    sides: str = "both",
    count: int = 1,
    join_style: str = "round",
    include_original: bool = True,
) -> list[Polyline]:
    """Generate parallel offset copies of polylines.

    For each input polyline, produces one or more offset copies at multiples
    of *distance_mm* on the specified side(s). Handles both open and closed
    paths. Uses Shapely's ``offset_curve`` for open paths and ``buffer`` on
    ``LinearRing`` for closed paths.

    Args:
        paths: Input list of polylines (each a list of (x, y) mm points).
        distance_mm: Offset distance in mm per copy. Default 0.5.
        sides: Which sides to offset — "both", "left", or "right". For closed
            paths "left" = outside, "right" = inside. Default "both".
        count: Number of offset copies per side, each at ``distance_mm * i``
            for i in 1..count. Default 1.
        join_style: Corner join style — "round", "mitre", or "bevel".
            Default "round".
        include_original: If True, include the original path in output.
            Default True.

    Returns:
        New list of polylines containing offsets (and originals if requested).
        Empty or degenerate inputs are skipped. If Shapely is unavailable,
        returns a copy of the input unchanged.
    """
    if not _HAS_SHAPELY:
        # Graceful degradation when Shapely is not installed
        return [list(p) for p in paths]

    # Clamp parameters
    distance_mm = max(0.0, float(distance_mm))
    count = max(1, int(count))
    sides = sides.lower() if sides else "both"
    if sides not in ("both", "left", "right"):
        sides = "both"

    result: list[Polyline] = []

    for polyline in paths:
        if len(polyline) < 2:
            # Skip degenerate paths entirely — cannot form a valid offset
            continue

        # Detect closed paths: first and last point coincide (within 1e-9 mm)
        p0, p1 = polyline[0], polyline[-1]
        is_closed = (
            len(polyline) >= 4
            and abs(p0[0] - p1[0]) < 1e-9
            and abs(p0[1] - p1[1]) < 1e-9
        )

        if is_closed:
            offsets = _offset_closed_path(
                polyline, distance_mm, sides, count, join_style, include_original
            )
        else:
            offsets = _offset_single_path(
                polyline, distance_mm, sides, count, join_style, include_original
            )

        result.extend(offsets)

    return result
