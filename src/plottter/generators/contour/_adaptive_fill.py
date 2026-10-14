"""Perimeter-plus-zigzag solid fill ("Solid (Adaptive)") for ContourGenerator.

The plain Solid fill clips independent horizontal scanlines to each ink
polygon.  On a drawing made of thin strokes that cuts every stroke into
hundreds of short dashes — one pen lift each — and the dashes' ends make a
ragged edge.  This fill does what CNC pocketing and 3D-print slicers do:

1. **Perimeter rings** — ``perimeter_rings`` inward offsets of the shape at
   ``spacing_mm`` steps follow the shape's own outline, so a stroke only a
   few pen widths wide is consumed entirely by long paths that run *along*
   it, and every edge is crisp.
2. **Zigzag core** — whatever area remains inside the last ring is filled
   with scanlines that are chained into continuous boustrophedon paths.
   Consecutive rows are connected when their spans overlap and the short
   connector stays inside the shape, so a broad area becomes one path
   instead of one path per row.

Only broad regions reach step 2, which is where scanlines are efficient.
"""

from __future__ import annotations

import math

from plottter.models import Polyline

#: Connectors between consecutive zigzag rows longer than this many spacings
#: start a new path instead (keeps bridges short on shallow, jagged edges).
_MAX_CONNECT_SPACINGS = 3.0


def fill_polygon_adaptive(
    polygon_pts: Polyline,
    hole_pts_list: list[Polyline],
    spacing_mm: float,
    perimeter_rings: int = 2,
    angle_deg: float = 0.0,
) -> list[Polyline]:
    """Fill a polygon with perimeter rings plus a zigzag core.

    Parameters
    ----------
    polygon_pts:      Outer boundary as a closed list of (x_mm, y_mm).
    hole_pts_list:    Hole boundaries (each a closed Polyline).
    spacing_mm:       Distance between fill lines / rings in mm.
    perimeter_rings:  Number of inward offset rings before the zigzag core
                      (0 = zigzag only).
    angle_deg:        Direction of the zigzag scanlines (0 = horizontal).
    """
    try:
        from shapely.geometry import Polygon
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
    for sub in _polygons_of(poly):
        result.extend(_fill_one(sub, spacing_mm, max(0, int(perimeter_rings)), angle_deg))
    return result


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _polygons_of(geom) -> list:
    """Flatten a (Multi)Polygon / GeometryCollection into Polygons."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_polygons_of(g))
        return out
    return []


def _rings_of(geom) -> list[Polyline]:
    """Every exterior *and* interior ring of a (Multi)Polygon as closed Polylines."""
    rings: list[Polyline] = []
    for p in _polygons_of(geom):
        for ring in (p.exterior, *p.interiors):
            coords = [(float(c[0]), float(c[1])) for c in ring.coords]
            if len(coords) >= 3:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                rings.append(coords)
    return rings


def _fill_one(poly, spacing: float, n_rings: int, angle_deg: float) -> list[Polyline]:
    rings: list[Polyline] = []
    chains: list[Polyline] = []
    core = poly
    consumed = False
    for i in range(1, n_rings + 1):
        shrunk = poly.buffer(-i * spacing)
        if shrunk.is_empty:
            consumed = True  # thin shape: the rings covered it
            break
        rings.extend(_rings_of(shrunk))
        core = shrunk
    if not consumed:
        if n_rings > 0:
            # Drop core slivers no point of which is further than half a
            # spacing from the last ring: the pen drawing that ring already
            # covers them, and scanlines through a hair-thin sliver would
            # only produce confetti.
            from shapely.geometry import MultiPolygon
            keep = [sub for sub in _polygons_of(core) if not sub.buffer(-0.5 * spacing).is_empty]
            core = MultiPolygon(keep) if keep else None
        if core is not None and not core.is_empty:
            chains = _zigzag_fill(core, poly, spacing, angle_deg)
    return _chain_pieces(rings, chains, poly, spacing)


def _chain_pieces(rings: list[Polyline], chains: list[Polyline], container, spacing: float) -> list[Polyline]:
    """Join a shape's rings and zigzag chains into as few paths as possible.

    Walks the pieces greedily from the current path's end: a closed ring is
    rotated to start at its vertex nearest that end, an open chain is
    reversed if its far end is nearer.  The connector must be short and lie
    inside the shape, so joins only ever cross the ink, never a gap.
    """
    from shapely.geometry import LineString
    from shapely.prepared import prep

    import numpy as np

    pieces: list[tuple[bool, Polyline]] = [(True, r) for r in rings] + [(False, c) for c in chains]
    if len(pieces) <= 1:
        return [pl for _, pl in pieces]
    inside = prep(container.buffer(1e-6))
    max_connect = _MAX_CONNECT_SPACINGS * spacing
    # Vertex arrays for the closed rings (nearest-vertex search is vectorised).
    ring_xy = {
        i: np.asarray(pl[:-1] if pl[0] == pl[-1] else pl, dtype=np.float64)
        for i, (closed, pl) in enumerate(pieces) if closed
    }
    remaining = list(range(len(pieces)))
    out: list[Polyline] = []
    current: Polyline = []
    while remaining:
        if not current:
            closed, pl = pieces[remaining.pop(0)]
            current = list(pl)
            continue
        ex, ey = current[-1]
        best = None
        best_d = math.inf
        for idx in remaining:
            closed, pl = pieces[idx]
            if closed:
                xy = ring_xy[idx]
                d2 = (xy[:, 0] - ex) ** 2 + (xy[:, 1] - ey) ** 2
                k = int(np.argmin(d2))
                d = math.sqrt(float(d2[k]))
                cand = (idx, k)
            else:
                d0 = math.hypot(pl[0][0] - ex, pl[0][1] - ey)
                d1 = math.hypot(pl[-1][0] - ex, pl[-1][1] - ey)
                d, cand = (d0, (idx, 0)) if d0 <= d1 else (d1, (idx, -1))
            if d < best_d:
                best, best_d = cand, d
        joined = False
        if best is not None and best_d <= max_connect:
            idx, k = best
            closed, pl = pieces[idx]
            if closed:
                verts = pl[:-1] if pl[0] == pl[-1] else pl
                nxt = verts[k:] + verts[:k] + [verts[k]]
            else:
                nxt = list(pl) if k == 0 else list(reversed(pl))
            if inside.covers(LineString([(ex, ey), nxt[0]])):
                remaining.remove(idx)
                current.extend(nxt)
                joined = True
        if not joined:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _zigzag_fill(core, container, spacing: float, angle_deg: float) -> list[Polyline]:
    """Scanline-fill *core* and chain consecutive rows into zigzag paths.

    A row's segment continues the path of a previous-row segment when the
    connector from that path's end to the segment's nearer end is short and
    lies inside *container* (the un-offset shape, so a connector may cross
    the perimeter band but never leave the ink).  Spans need not overlap:
    on a thin diagonal stroke consecutive rows shift sideways by more than
    their own width, and the inside test is what stops bridging a gap.
    """
    from shapely import affinity
    from shapely.geometry import LineString
    from shapely.prepared import prep

    if core.is_empty:
        return []

    # Rotate so scanlines are horizontal; rotate results back at the end.
    if angle_deg % 180.0 != 0.0:
        core_r = affinity.rotate(core, -angle_deg, origin=(0.0, 0.0))
        cont_r = affinity.rotate(container, -angle_deg, origin=(0.0, 0.0))
    else:
        core_r, cont_r = core, container
    # Tiny tolerance so endpoints that sit exactly on the boundary count.
    inside = prep(cont_r.buffer(1e-6))

    min_x, min_y, max_x, max_y = core_r.bounds
    max_connect = _MAX_CONNECT_SPACINGS * spacing

    chains: list[Polyline] = []
    # Open chains from the previous row: (end_x, end_y, span_lo, span_hi, index)
    open_prev: list[tuple[float, float, float, float, int]] = []

    y = min_y + 0.5 * spacing
    while y <= max_y:
        scan = LineString([(min_x - 1.0, y), (max_x + 1.0, y)])
        try:
            clipped = core_r.intersection(scan)
        except Exception:
            y += spacing
            continue
        segments: list[tuple[float, float]] = []
        geoms = []
        if clipped.geom_type == "LineString":
            geoms = [clipped]
        elif clipped.geom_type in ("MultiLineString", "GeometryCollection"):
            geoms = [g for g in clipped.geoms if g.geom_type == "LineString"]
        for g in geoms:
            xs = [float(c[0]) for c in g.coords]
            if len(xs) < 2:
                continue  # degenerate/empty piece from the clip
            lo, hi = min(xs), max(xs)
            if hi - lo > 1e-9:
                segments.append((lo, hi))
        segments.sort()

        open_now: list[tuple[float, float, float, float, int]] = []
        used: set[int] = set()
        for lo, hi in segments:
            best = None
            best_d = math.inf
            for k, (ex, ey, s_lo, s_hi, idx) in enumerate(open_prev):
                if k in used:
                    continue  # already extended this row
                # Enter the segment at whichever end is nearer the chain's end.
                d_lo = math.hypot(lo - ex, y - ey)
                d_hi = math.hypot(hi - ex, y - ey)
                d, start = (d_lo, "lo") if d_lo <= d_hi else (d_hi, "hi")
                if d < best_d:
                    best, best_d = (k, start), d
            if best is not None and best_d <= max_connect:
                k, start = best
                ex, ey, _, _, idx = open_prev[k]
                first, last = ((lo, y), (hi, y)) if start == "lo" else ((hi, y), (lo, y))
                if inside.covers(LineString([(ex, ey), first])):
                    used.add(k)
                    chains[idx].extend([first, last])
                    open_now.append((last[0], last[1], lo, hi, idx))
                    continue
            # Start a new path on this segment.
            chains.append([(lo, y), (hi, y)])
            open_now.append((hi, y, lo, hi, len(chains) - 1))
        open_prev = open_now
        y += spacing

    if angle_deg % 180.0 != 0.0:
        c = math.cos(math.radians(angle_deg))
        s = math.sin(math.radians(angle_deg))
        chains = [[(c * x - s * yy, s * x + c * yy) for x, yy in ch] for ch in chains]
    return chains
