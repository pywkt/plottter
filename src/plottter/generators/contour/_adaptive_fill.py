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
   Consecutive rows are connected whenever the straight connector between
   them stays inside the shape — its length does not matter, since a
   connector inside the ink is just more ink — so a broad area becomes one
   path instead of one path per row.

Only broad regions reach step 2, which is where scanlines are efficient.

3. **Chaining** — the rings and zigzag chains of one shape are stitched
   into as few pen-down paths as possible.  A short straight connector
   (a few spacings, inside the ink) is drawn directly.  A longer one would
   be a visible chord cutting across the fill at an arbitrary angle, so
   instead the pen walks along the innermost perimeter ring — every chain
   end and ring vertex sits on it — retracing a line that is drawn anyway.
"""

from __future__ import annotations

import math

from plottter.models import Polyline

#: Containment tolerance for connectors, as a fraction of the spacing.
_CONTAIN_TOL = 0.05

#: Straight connectors between pieces longer than this many spacings are not
#: drawn as chords; the pen walks along the perimeter ring instead (or lifts).
_MAX_CONNECT_SPACINGS = 3.0

#: Longest connector routed along the innermost perimeter ring, in spacings.
#: Retracing a ring further than this reads as a darker edge on paper.
_MAX_WALK_SPACINGS = 10.0


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
    # The innermost ring: zigzag chain ends and the last ring's vertices all
    # lie on it, so long connectors can travel along it instead of across
    # the fill.  With no rings the chain ends sit on the outer boundary,
    # which we must not retrace (the user may have turned the outline off).
    walk_boundary = core if (n_rings > 0 and not consumed) else None
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
    return _chain_pieces(rings, chains, poly, spacing, walk_boundary)


class _RingWalker:
    """Connectors that follow a (Multi)Polygon's boundary rings.

    ``arc(a, b)`` gives the length of the shorter arc between *a* and *b*
    along whichever ring carries both (within *tol*), and ``path(a, b)``
    the vertices of that arc.  The arc is drawn ink-on-ink over a ring that
    is part of the fill anyway, so it never shows as a stray line.
    Projections are cached per point: the same chain ends are re-examined
    on every greedy step.
    """

    def __init__(self, geom, tol: float) -> None:
        from shapely.geometry import LineString

        self._tol = tol
        self._rings: list = []
        self._cache: dict[tuple[float, float], tuple[int, float] | None] = {}
        for p in _polygons_of(geom):
            for ring in (p.exterior, *p.interiors):
                if len(ring.coords) >= 4:
                    self._rings.append(LineString(ring.coords))

    def _locate(self, pt: tuple[float, float]) -> tuple[int, float] | None:
        """(ring index, measure along it) for a point on a ring, else None."""
        if pt in self._cache:
            return self._cache[pt]
        from shapely.geometry import Point

        p = Point(pt)
        found = None
        for i, ring in enumerate(self._rings):
            if ring.distance(p) <= self._tol:
                found = (i, ring.project(p))
                break
        self._cache[pt] = found
        return found

    def arc(self, a: tuple[float, float], b: tuple[float, float]) -> float | None:
        la, lb = self._locate(a), self._locate(b)
        if la is None or lb is None or la[0] != lb[0]:
            return None
        length = self._rings[la[0]].length
        forward = (lb[1] - la[1]) % length
        return min(forward, length - forward)

    def path(self, a: tuple[float, float], b: tuple[float, float]) -> Polyline | None:
        from shapely.ops import substring

        la, lb = self._locate(a), self._locate(b)
        if la is None or lb is None or la[0] != lb[0]:
            return None
        ring = self._rings[la[0]]
        length = ring.length
        ma, mb = la[1], lb[1]
        forward = (mb - ma) % length
        if forward <= length - forward:
            # Walk in the ring's own direction from ma to mb.
            if ma <= mb:
                parts = [substring(ring, ma, mb)]
            else:
                parts = [substring(ring, ma, length), substring(ring, 0.0, mb)]
        else:
            # Walk against the ring's direction from ma back to mb.
            if mb <= ma:
                parts = [substring(ring, ma, mb)]  # start > end: reversed
            else:
                parts = [substring(ring, ma, 0.0), substring(ring, length, mb)]
        return _coords_of(parts)


def _coords_of(parts) -> Polyline:
    """Concatenate LineString parts into one Polyline, dropping repeats."""
    out: Polyline = []
    for part in parts:
        if part.is_empty or part.geom_type != "LineString":
            continue
        for c in part.coords:
            pt = (float(c[0]), float(c[1]))
            if not out or out[-1] != pt:
                out.append(pt)
    return out


def _chain_pieces(
    rings: list[Polyline],
    chains: list[Polyline],
    container,
    spacing: float,
    walk_boundary=None,
) -> list[Polyline]:
    """Join a shape's rings and zigzag chains into as few paths as possible.

    Greedy from the current path's end, by connector *cost*: a straight
    connector of up to ``_MAX_CONNECT_SPACINGS`` spacings that lies inside
    the shape costs its length; otherwise, if both ends sit on
    *walk_boundary* (the innermost ring), the connector follows that ring
    and costs the arc length, allowed up to ``_MAX_WALK_SPACINGS``.  A
    longer straight line would be a visible chord across the fill, and a
    longer walk retraces the ring far enough to read as a darker edge, so
    neither is drawn: the path ends and a new one starts.

    A closed ring is entered at the point nearest the current end (the ring
    is cut there), an open chain at whichever end costs less.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import substring
    from shapely.prepared import prep

    pieces: list[tuple[bool, Polyline]] = [(True, r) for r in rings] + [(False, c) for c in chains]
    if len(pieces) <= 1:
        return [pl for _, pl in pieces]
    tol = _CONTAIN_TOL * spacing
    inside = prep(container.buffer(tol))
    max_connect = _MAX_CONNECT_SPACINGS * spacing
    max_walk = _MAX_WALK_SPACINGS * spacing
    walker = _RingWalker(walk_boundary, tol) if walk_boundary is not None else None
    ring_ls = {i: LineString(pl) for i, (closed, pl) in enumerate(pieces) if closed}

    def connector_cost(end: tuple[float, float], pt: tuple[float, float]) -> tuple[float, bool] | None:
        """(cost, walk?) for joining *end* to *pt*, or None if not allowed."""
        d = math.dist(end, pt)
        if d <= max_connect and (d < 1e-9 or inside.covers(LineString([end, pt]))):
            return d, False
        if walker is not None:
            arc = walker.arc(end, pt)
            if arc is not None and arc <= max_walk:
                return arc, True
        return None

    remaining = list(range(len(pieces)))
    out: list[Polyline] = []
    current: Polyline = []
    while remaining:
        if not current:
            closed, pl = pieces[remaining.pop(0)]
            current = list(pl)
            continue
        end = current[-1]
        best = None
        best_cost = math.inf
        for idx in remaining:
            closed, pl = pieces[idx]
            if closed:
                ls = ring_ls[idx]
                m = ls.project(Point(end))
                e = ls.interpolate(m)
                cands = [((float(e.x), float(e.y)), m)]
            else:
                cands = [(pl[0], None), (pl[-1], None)]
            for pt, measure in cands:
                cc = connector_cost(end, pt)
                if cc is not None and cc[0] < best_cost:
                    best, best_cost = (idx, pt, measure, cc[1]), cc[0]
        if best is None:
            out.append(current)
            current = []
            continue
        idx, pt, measure, walk = best
        closed, pl = pieces[idx]
        if closed:
            ls = ring_ls[idx]
            nxt = _coords_of([substring(ls, measure, ls.length), substring(ls, 0.0, measure)])
        else:
            nxt = list(pl) if pt == pl[0] else list(reversed(pl))
        remaining.remove(idx)
        if walk:
            current.extend(walker.path(end, pt) or [])
        current.extend(nxt)
    if current:
        out.append(current)
    return out


def _zigzag_fill(core, container, spacing: float, angle_deg: float) -> list[Polyline]:
    """Scanline-fill *core* and chain consecutive rows into zigzag paths.

    A row's segment continues the path of a previous-row segment when the
    connector from that path's end to the segment's nearer end lies inside
    *container* (the un-offset shape, so a connector may cross the perimeter
    band but never leave the ink).  Neither span overlap nor connector
    length is required: on a thin diagonal stroke consecutive rows shift
    sideways by more than their own width, and along an edge that runs
    nearly parallel to the rows the next row's end can be far away.  The
    inside test alone is what stops bridging a gap.
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
    # With no rings the row endpoints sit exactly on the boundary and the
    # clip can leave them a hair outside; a tolerance of a few percent of the
    # spacing keeps those connectors (a 25 µm excursion at 0.5 mm is nothing)
    # without letting one cross a real gap.
    inside = prep(cont_r.buffer(_CONTAIN_TOL * spacing))

    min_x, min_y, max_x, max_y = core_r.bounds

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
            if best is not None:
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
