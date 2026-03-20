"""PenroseGenerator — Penrose P2 (kite-and-dart) tiling via Robinson triangles.

Robinson triangles are the fundamental building blocks of Penrose P2 tilings.
Two types:
  - Type 0 ("thin" / "red"):  angles 36°-108°-36°
  - Type 1 ("thick" / "blue"): angles 72°-72°-36°

Subdivision rules (one level doubles/triples the triangle count):
  Type 0 (thin)  (0, A, B, C) → P = A + (B-A)/PHI
                               → [(0, C, P, B), (1, P, C, A)]
  Type 1 (thick) (1, A, B, C) → Q = B + (A-B)/PHI, R = B + (C-B)/PHI
                               → [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from plottter.generators import register_generator
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI = (1 + math.sqrt(5)) / 2  # golden ratio ≈ 1.618033...
PSI = 1 / PHI                  # ≈ 0.618033...

# Triangle type tags
TYPE_THIN = 0   # "red" Robinson triangle: angles 36°-108°-36°
TYPE_THICK = 1  # "blue" Robinson triangle: angles 72°-72°-36°

# Type alias for a Robinson triangle: (type, A, B, C) all complex
Triangle = tuple[int, complex, complex, complex]


# ---------------------------------------------------------------------------
# Core subdivision engine
# ---------------------------------------------------------------------------

def _subdivide(triangles: list[Triangle]) -> list[Triangle]:
    """Apply one level of Robinson triangle subdivision.

    Uses an explicit stack (iterative) to avoid Python recursion limits at
    high subdivision depths.

    Type 0 (thin):   (0, A, B, C) → P = A + (B-A)/PHI
                                   → [(0, C, P, B), (1, P, C, A)]
    Type 1 (thick):  (1, A, B, C) → Q = B + (A-B)/PHI, R = B + (C-B)/PHI
                                   → [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
    """
    result: list[Triangle] = []
    stack: list[Triangle] = list(triangles)
    while stack:
        t, A, B, C = stack.pop()
        if t == TYPE_THIN:
            P = A + (B - A) / PHI
            result.append((TYPE_THIN, C, P, B))
            result.append((TYPE_THICK, P, C, A))
        else:  # TYPE_THICK
            Q = B + (A - B) / PHI
            R = B + (C - B) / PHI
            result.append((TYPE_THICK, R, C, A))
            result.append((TYPE_THICK, Q, R, B))
            result.append((TYPE_THIN, R, Q, A))
    return result


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------

def _initial_config(config_name: str, radius: float) -> list[Triangle]:
    """Return the initial Robinson triangle list for a named configuration.

    Parameters
    ----------
    config_name:
        "Sun"  — 10 thick triangles forming a decagonal wheel (P2 sun seed).
        "Star" — 10 thin  triangles forming a star wheel (P2 star seed).
        "Dart" — 4 thin triangles arranged symmetrically as a dart seed.
    radius:
        Outer radius of the initial configuration in the complex-plane units
        used for vertex coordinates.
    """
    triangles: list[Triangle] = []

    if config_name in ("Sun", "Star"):
        tri_type = TYPE_THICK if config_name == "Sun" else TYPE_THIN
        for i in range(10):
            B = radius * cmath.exp(1j * (2 * i - 1) * math.pi / 10)
            C = radius * cmath.exp(1j * (2 * i + 1) * math.pi / 10)
            if i % 2 == 0:
                triangles.append((tri_type, 0 + 0j, B, C))
            else:
                triangles.append((tri_type, 0 + 0j, C, B))

    elif config_name == "Dart":
        # 4 thin triangles arranged symmetrically to form a dart-like seed.
        # Spaced 90° apart, each subtending 36° (π/5) at the origin.
        for i in range(4):
            angle = i * math.pi / 2
            B = radius * cmath.exp(1j * angle)
            C = radius * cmath.exp(1j * (angle + math.pi / 5))
            if i % 2 == 0:
                triangles.append((TYPE_THIN, 0 + 0j, B, C))
            else:
                triangles.append((TYPE_THIN, 0 + 0j, C, B))

    return triangles


# ---------------------------------------------------------------------------
# Rhomb edge extraction helpers
# ---------------------------------------------------------------------------

def _round_v(v: complex, decimals: int = 6) -> tuple[float, float]:
    """Round a complex vertex to a hashable tuple for set/dict keys."""
    return (round(v.real, decimals), round(v.imag, decimals))


def _edge_key(v1: complex, v2: complex) -> tuple:
    """Sorted vertex-pair key for edge deduplication (order-independent)."""
    r1, r2 = _round_v(v1), _round_v(v2)
    return (min(r1, r2), max(r1, r2))


def _long_edge_keys(t: int, A: complex, B: complex, C: complex) -> list[tuple]:
    """Return edge keys for the long edge(s) of a Robinson triangle.

    Type 0 (thin, 36°-108°-36°): A is the 108° apex; BC is the long edge.
    Type 1 (thick, 72°-72°-36°): A is the 36° apex; AB and AC are the long edges.
    """
    if t == TYPE_THIN:
        return [_edge_key(B, C)]
    else:  # TYPE_THICK
        return [_edge_key(A, B), _edge_key(A, C)]


def _triangles_to_edges(triangles: list[Triangle]) -> list[tuple[complex, complex]]:
    """Group Robinson triangles into rhombs and extract deduplicated outer edges.

    A rhomb is formed by two same-type triangles sharing a long edge:
      - Two thick (Type 1) triangles sharing a long edge → kite rhomb
      - Two thin  (Type 0) triangles sharing a long edge → dart rhomb

    Steps:
      1. Build a map from each long-edge key to the triangles that own it.
      2. Pairs of same-type triangles sharing a key form one rhomb; the shared
         edge is the internal diagonal and is excluded from the output.
      3. All remaining triangle edges are emitted once (deduplicated across
         adjacent rhombs via a seen-edges set).

    Parameters
    ----------
    triangles:
        Robinson triangles in complex-plane coordinates (origin-centred).

    Returns
    -------
    List of (v1, v2) complex vertex pairs for the outer edges of all rhombs,
    with no duplicate edges.
    """
    # Map long-edge key → list of triangle indices that own that edge
    long_edge_map: dict[tuple, list[int]] = {}
    for i, (t, A, B, C) in enumerate(triangles):
        for key in _long_edge_keys(t, A, B, C):
            long_edge_map.setdefault(key, []).append(i)

    # Identify the shared internal diagonals (long edges owned by exactly two
    # same-type triangles) and collect all triangle indices that belong to rhombs.
    shared_long_edges: set[tuple] = set()
    rhomb_tris: set[int] = set()
    for key, indices in long_edge_map.items():
        if len(indices) == 2:
            i, j = indices
            if triangles[i][0] == triangles[j][0]:  # same type → valid rhomb
                shared_long_edges.add(key)
                rhomb_tris.add(i)
                rhomb_tris.add(j)

    # Emit each outer edge once (skip shared internal diagonals; deduplicate
    # edges shared between adjacent rhombs).
    seen_edges: set[tuple] = set()
    result: list[tuple[complex, complex]] = []
    for i in rhomb_tris:
        _, A, B, C = triangles[i]
        for v1, v2 in ((A, B), (B, C), (C, A)):
            ek = _edge_key(v1, v2)
            if ek in shared_long_edges:
                continue  # internal diagonal — skip
            if ek in seen_edges:
                continue  # already emitted from a neighbouring rhomb
            seen_edges.add(ek)
            result.append((v1, v2))

    return result


# ---------------------------------------------------------------------------
# Canvas clipping
# ---------------------------------------------------------------------------

def _liang_barsky(
    x0: float, y0: float, x1: float, y1: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> tuple[float, float, float, float] | None:
    """Liang-Barsky line-segment clip against axis-aligned rectangle.

    Returns the clipped (x0, y0, x1, y1) or None if the segment lies
    entirely outside the rectangle.
    """
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-15:       # segment parallel to this edge
            if qi < 0:
                return None       # outside on this axis
        elif pi < 0:
            t0 = max(t0, qi / pi)
        else:
            t1 = min(t1, qi / pi)

    if t0 > t1 + 1e-10:
        return None

    return x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy


def _clip_to_canvas(
    edges: list[tuple[tuple[float, float], tuple[float, float]]],
    canvas_w: float,
    canvas_h: float,
    margin: float,
) -> list[Polyline]:
    """Clip edges (mm coordinates) to the canvas drawing area.

    The drawing area is the rectangle [margin, margin, canvas_w-margin,
    canvas_h-margin].  Edges that lie entirely outside are discarded;
    edges that cross the boundary are trimmed.  Zero-length clipped
    segments are dropped.

    Parameters
    ----------
    edges:
        Sequence of ((x1, y1), (x2, y2)) pairs in mm.
    canvas_w, canvas_h:
        Full canvas dimensions in mm.
    margin:
        Uniform margin in mm on all four sides.

    Returns
    -------
    List of 2-point Polylines for every edge that intersects the drawing area.
    """
    xmin, ymin = margin, margin
    xmax, ymax = canvas_w - margin, canvas_h - margin

    result: list[Polyline] = []
    for (px, py), (qx, qy) in edges:
        clipped = _liang_barsky(px, py, qx, qy, xmin, ymin, xmax, ymax)
        if clipped is None:
            continue
        cx0, cy0, cx1, cy1 = clipped
        # Discard zero-length segments
        if abs(cx1 - cx0) < 1e-10 and abs(cy1 - cy0) < 1e-10:
            continue
        result.append([(cx0, cy0), (cx1, cy1)])
    return result


def _clip_polyline_to_canvas(
    polyline: Polyline,
    canvas_w: float,
    canvas_h: float,
    margin: float,
) -> list[Polyline]:
    """Clip a multi-point polyline to the canvas drawing area.

    Applies Liang–Barsky clipping to each consecutive segment.  Continuous
    runs of clipped segments are assembled into output polylines; a gap
    in the output (segment entirely outside) ends the current run and starts
    a new one.

    Returns a list of 2-or-more-point polylines.
    """
    xmin, ymin = margin, margin
    xmax, ymax = canvas_w - margin, canvas_h - margin

    result: list[Polyline] = []
    current: Polyline = []

    for k in range(len(polyline) - 1):
        (px, py), (qx, qy) = polyline[k], polyline[k + 1]
        seg = _liang_barsky(px, py, qx, qy, xmin, ymin, xmax, ymax)
        if seg is None:
            # Segment fully outside — flush current run
            if len(current) >= 2:
                result.append(current)
            current = []
            continue
        cx0, cy0, cx1, cy1 = seg
        if not current:
            current = [(cx0, cy0), (cx1, cy1)]
        else:
            # If the clipped start differs from last point, flush and restart
            if abs(current[-1][0] - cx0) > 1e-9 or abs(current[-1][1] - cy0) > 1e-9:
                if len(current) >= 2:
                    result.append(current)
                current = [(cx0, cy0), (cx1, cy1)]
            else:
                current.append((cx1, cy1))

    if len(current) >= 2:
        result.append(current)
    return result


def _generate_rhombs(
    triangles: list[Triangle],
    cx: float,
    cy: float,
    scale: float,
    rotation: float,
    canvas_w: float,
    canvas_h: float,
    margin: float,
) -> list[Polyline]:
    """Render the Penrose tiling as kite-and-dart (rhomb) outlines.

    Coordinate transform: complex-plane → canvas mm
        x_mm = cx + Re(z * rot) * scale
        y_mm = cy − Im(z * rot) * scale   (y-axis flipped for screen/canvas)

    Parameters
    ----------
    triangles:           Robinson triangles in normalised complex coordinates.
    cx, cy:              Canvas centre in mm.
    scale:               Multiply complex magnitude by this to get mm distance.
    rotation:            Rotation angle in radians.
    canvas_w, canvas_h:  Full canvas dimensions in mm.
    margin:              Uniform margin in mm on all four sides.
    """
    rot_factor = cmath.exp(1j * rotation)

    def to_mm(z: complex) -> tuple[float, float]:
        z2 = z * rot_factor
        return (cx + z2.real * scale, cy - z2.imag * scale)

    complex_edges = _triangles_to_edges(triangles)
    mm_edges = [(to_mm(v1), to_mm(v2)) for v1, v2 in complex_edges]
    return _clip_to_canvas(mm_edges, canvas_w, canvas_h, margin)


# ---------------------------------------------------------------------------
# Arc decoration helpers
# ---------------------------------------------------------------------------

def _arc_complex(
    center: complex,
    p1: complex,
    p2: complex,
    inside_ref: complex,
    n_segments: int = 12,
) -> list[complex]:
    """Return a polyline approximating the arc from p1 to p2 centred at center.

    Selects the arc (of the two possible arcs connecting p1 and p2 on the
    circle) whose midpoint is closest to *inside_ref*, so the arc lies
    inside the rhomb rather than outside.  Returns n_segments + 1 points.

    Returns an empty list when the inputs are degenerate.
    """
    r1 = abs(p1 - center)
    r2 = abs(p2 - center)
    if r1 < 1e-10 or r2 < 1e-10:
        return []
    r = (r1 + r2) * 0.5

    a1 = cmath.phase(p1 - center)
    a2 = cmath.phase(p2 - center)

    # Angular spans for the two possible arcs (CCW and CW from p1 to p2)
    span_ccw = (a2 - a1) % (2.0 * math.pi)   # going CCW
    span_cw  = (a1 - a2) % (2.0 * math.pi)   # going CW

    # Midpoints of the two possible arcs
    mid_ccw = center + r * cmath.exp(1j * (a1 + span_ccw * 0.5))
    mid_cw  = center + r * cmath.exp(1j * (a1 - span_cw  * 0.5))

    # Pick the arc whose midpoint is closer to the interior reference point
    if abs(mid_ccw - inside_ref) <= abs(mid_cw - inside_ref):
        angles = [a1 + span_ccw * k / n_segments for k in range(n_segments + 1)]
    else:
        angles = [a1 - span_cw * k / n_segments for k in range(n_segments + 1)]

    return [center + r * cmath.exp(1j * a) for a in angles]


def _generate_rhomb_arcs(
    triangles: list[Triangle],
    n_arc_segments: int = 12,
) -> list[list[complex]]:
    """Compute classic Penrose arc decorations for every rhomb in the tiling.

    For each pair of same-type Robinson triangles sharing a long edge
    (i.e. each complete rhomb), two circular arcs are drawn inside the rhomb:

    TYPE_THICK pair (kite):
        Shared-edge vertices are the kite's apex (36°→72° in kite) and
        base (72°→144° in kite).  Wing vertices are the two 72° tips.
        - Arc 1: centred at apex, connects the two wing vertices.
        - Arc 2: centred at base, connects the two wing vertices.
        These arcs use the long side and the short side as radii respectively,
        revealing the nested golden-ratio structure.

    TYPE_THIN pair (dart):
        Shared-edge vertices are the dart's two acute (36°→72°) vertices.
        Wing vertices are the two obtuse (108°) vertices.
        - Arc 1: centred at acute vertex 1, connects the two obtuse vertices.
        - Arc 2: centred at acute vertex 2, connects the two obtuse vertices.

    Returns a list of complex-coordinate polylines (not yet scaled to mm).
    """
    long_edge_map: dict[tuple, list[int]] = {}
    for i, (t, A, B, C) in enumerate(triangles):
        for key in _long_edge_keys(t, A, B, C):
            long_edge_map.setdefault(key, []).append(i)

    arcs: list[list[complex]] = []

    for key, indices in long_edge_map.items():
        if len(indices) != 2:
            continue
        i, j = indices
        ti, Ai, Bi, Ci = triangles[i]
        tj, Aj, Bj, Cj = triangles[j]
        if ti != tj:
            continue  # mixed-type pair — not a valid rhomb

        if ti == TYPE_THIN:
            # Long edge is BC.  Two thin triangles share this edge.
            # Ai, Aj = wing (108°) vertices; Bi/Ci = shared-edge (36°) vertices.
            wing1, wing2 = Ai, Aj
            sh1, sh2 = Bi, Ci
            inside = (wing1 + sh1 + wing2 + sh2) * 0.25
            for center_v in (sh1, sh2):
                pts = _arc_complex(center_v, wing1, wing2, inside, n_arc_segments)
                if pts:
                    arcs.append(pts)

        else:  # TYPE_THICK
            # Long edges are AB and AC.  Find which long edge of T_i matches key.
            key_abi = _edge_key(Ai, Bi)
            key_aci = _edge_key(Ai, Ci)
            if key == key_abi:
                sh_apex, sh_base, wing_i = Ai, Bi, Ci
            elif key == key_aci:
                sh_apex, sh_base, wing_i = Ai, Ci, Bi
            else:
                continue

            # Wing vertex from T_j
            key_abj = _edge_key(Aj, Bj)
            key_acj = _edge_key(Aj, Cj)
            if key == key_abj:
                wing_j = Cj
            elif key == key_acj:
                wing_j = Bj
            else:
                continue

            inside = (sh_apex + sh_base + wing_i + wing_j) * 0.25
            for center_v in (sh_apex, sh_base):
                pts = _arc_complex(center_v, wing_i, wing_j, inside, n_arc_segments)
                if pts:
                    arcs.append(pts)

    return arcs


# ---------------------------------------------------------------------------
# Triangle → polyline conversion (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _triangles_to_polylines(
    triangles: list[Triangle],
    cx: float,
    cy: float,
    scale: float,
    rotation: float,
    draw_mode: str,
    deduplicate: bool,
) -> list[Polyline]:
    """Convert Robinson triangles to plotter polylines (mm coordinates).

    Parameters
    ----------
    triangles:   List of Robinson triangles in normalised complex coordinates.
    cx, cy:      Canvas centre in mm.
    scale:       Multiply complex coordinates by this to get mm values.
    rotation:    Rotation angle in radians applied before scaling.
    draw_mode:   "All edges", "Thin only", or "Thick only".
    deduplicate: If True, shared edges between adjacent triangles are drawn
                 only once (reduces redundant pen lifts).
    """
    rot_factor = cmath.exp(1j * rotation)

    def to_pt(z: complex) -> tuple[float, float]:
        z2 = z * rot_factor
        return (cx + z2.real * scale, cy - z2.imag * scale)

    def edge_key(p1: tuple[float, float], p2: tuple[float, float]) -> frozenset:
        r1 = (round(p1[0], 4), round(p1[1], 4))
        r2 = (round(p2[0], 4), round(p2[1], 4))
        return frozenset((r1, r2))

    seen: set[frozenset] = set()
    polylines: list[Polyline] = []

    for t, A, B, C in triangles:
        if draw_mode == "Thin only" and t != TYPE_THIN:
            continue
        if draw_mode == "Thick only" and t != TYPE_THICK:
            continue

        pa, pb, pc = to_pt(A), to_pt(B), to_pt(C)
        for p1, p2 in ((pa, pb), (pb, pc), (pc, pa)):
            if deduplicate:
                key = edge_key(p1, p2)
                if key in seen:
                    continue
                seen.add(key)
            polylines.append([p1, p2])

    return polylines


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@register_generator
class PenroseGenerator(Generator):
    """Penrose P2 (kite-and-dart) tiling via Robinson triangle subdivision."""

    name = "Penrose Tiling"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="initial_config",
                label="Initial Configuration",
                choices=["Sun", "Star", "Dart"],
                default="Sun",
                description=(
                    "'Sun' (10 thick triangles, decagonal), "
                    "'Star' (10 thin triangles, star), "
                    "'Dart' (4 thin triangles, smaller seed)."
                ),
            ),
            IntParam(
                name="subdivisions",
                label="Subdivisions",
                min=1,
                max=8,
                step=1,
                default=5,
                description=(
                    "Subdivision depth — higher = smaller tiles, more detail. "
                    "Each level multiplies the tile count by ~PHI² ≈ 2.618."
                ),
            ),
            FloatParam(
                name="rotation_deg",
                label="Rotation (°)",
                min=0.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Overall rotation of the tiling in degrees.",
            ),
            ChoiceParam(
                name="render_mode",
                label="Render Mode",
                choices=["Edges Only", "Edges + Arcs", "Arcs Only"],
                default="Edges Only",
                description=(
                    "'Edges Only': rhomb outlines. "
                    "'Edges + Arcs': outlines plus classic arc matching-rule decorations. "
                    "'Arcs Only': arc decorations only (reveals pentagonal symmetry)."
                ),
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-150.0,
                max=150.0,
                step=1.0,
                default=0.0,
                description="Horizontal offset of the tiling centre from canvas centre.",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-150.0,
                max=150.0,
                step=1.0,
                default=0.0,
                description="Vertical offset of the tiling centre from canvas centre.",
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        config = params.get("initial_config", "Sun")
        subdivisions = max(1, int(params.get("subdivisions", 5)))
        rotation_deg = float(params.get("rotation_deg", 0.0))
        render_mode = params.get("render_mode", "Edges Only")
        x_offset_mm = float(params.get("x_offset_mm", 0.0))
        y_offset_mm = float(params.get("y_offset_mm", 0.0))

        # Build initial triangles in normalised complex space (unit radius)
        triangles = _initial_config(config, 1.0)

        # Iterative subdivision with progress reported at 0–50 %
        for i in range(subdivisions):
            if cancelled_callback and cancelled_callback():
                return []
            triangles = _subdivide(triangles)
            if progress_callback:
                progress_callback(int(50 * (i + 1) / max(subdivisions, 1)))

        if progress_callback:
            progress_callback(50)

        # Canvas layout: auto-fit the unit-radius tiling to the drawing area
        x1, y1, x2, y2 = canvas.drawing_area()
        draw_w, draw_h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0 + x_offset_mm
        cy = (y1 + y2) / 2.0 + y_offset_mm
        scale = min(draw_w, draw_h) / 2.0

        rotation_rad = math.radians(rotation_deg)
        rot_factor = cmath.exp(1j * rotation_rad)

        def to_mm(z: complex) -> tuple[float, float]:
            z2 = z * rot_factor
            return (cx + z2.real * scale, cy - z2.imag * scale)

        result: list[Polyline] = []

        # --- Edge polylines (rhomb outlines) ---
        if render_mode in ("Edges Only", "Edges + Arcs"):
            result.extend(
                _generate_rhombs(
                    triangles,
                    cx=cx,
                    cy=cy,
                    scale=scale,
                    rotation=rotation_rad,
                    canvas_w=canvas.width_mm,
                    canvas_h=canvas.height_mm,
                    margin=canvas.margin_mm,
                )
            )

        if progress_callback:
            progress_callback(75)

        # --- Arc decorations ---
        if render_mode in ("Edges + Arcs", "Arcs Only"):
            for arc_curve in _generate_rhomb_arcs(triangles, n_arc_segments=12):
                mm_pts = [to_mm(z) for z in arc_curve]
                result.extend(
                    _clip_polyline_to_canvas(
                        mm_pts,
                        canvas.width_mm,
                        canvas.height_mm,
                        canvas.margin_mm,
                    )
                )

        if progress_callback:
            progress_callback(100)

        return result

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Classic P3",
                params={
                    "initial_config": "Sun",
                    "subdivisions": 5,
                    "rotation_deg": 0.0,
                    "render_mode": "Edges Only",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Penrose Stars",
                params={
                    "initial_config": "Star",
                    "subdivisions": 4,
                    "rotation_deg": 0.0,
                    "render_mode": "Edges Only",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Arc Pattern",
                params={
                    "initial_config": "Sun",
                    "subdivisions": 5,
                    "rotation_deg": 0.0,
                    "render_mode": "Arcs Only",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Full Decoration",
                params={
                    "initial_config": "Sun",
                    "subdivisions": 6,
                    "rotation_deg": 0.0,
                    "render_mode": "Edges + Arcs",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Dense Tiling",
                params={
                    "initial_config": "Sun",
                    "subdivisions": 7,
                    "rotation_deg": 0.0,
                    "render_mode": "Edges Only",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Dart Origin",
                params={
                    "initial_config": "Dart",
                    "subdivisions": 5,
                    "rotation_deg": 0.0,
                    "render_mode": "Edges Only",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
        ]
