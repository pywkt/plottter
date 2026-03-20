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
# Triangle → polyline conversion
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
                    "Starting arrangement: 'Sun' (10 thick triangles), "
                    "'Star' (10 thin triangles), 'Dart' (4 thin triangles)."
                ),
            ),
            IntParam(
                name="depth",
                label="Subdivision Depth",
                min=0,
                max=8,
                step=1,
                default=4,
                description=(
                    "Number of subdivision iterations. Each level multiplies "
                    "the triangle count by ~PHI² ≈ 2.618."
                ),
            ),
            FloatParam(
                name="radius_mm",
                label="Radius (mm)",
                min=10.0,
                max=300.0,
                step=1.0,
                default=90.0,
                description="Outer radius of the tiling in millimeters.",
            ),
            FloatParam(
                name="rotation_deg",
                label="Rotation (°)",
                min=-180.0,
                max=180.0,
                step=1.0,
                default=0.0,
                description="Overall rotation of the tiling in degrees.",
            ),
            ChoiceParam(
                name="draw_mode",
                label="Draw Mode",
                choices=["All edges", "Thin only", "Thick only", "Rhombs"],
                default="All edges",
                description=(
                    "Which triangle type edges to render. 'Rhombs' draws "
                    "kite-and-dart outlines (internal diagonals omitted)."
                ),
            ),
            BoolParam(
                name="deduplicate",
                label="Deduplicate Edges",
                default=True,
                description=(
                    "Skip shared edges between adjacent triangles to reduce "
                    "redundant pen strokes."
                ),
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
        depth = int(params.get("depth", 4))
        radius_mm = float(params.get("radius_mm", 90.0))
        rotation_deg = float(params.get("rotation_deg", 0.0))
        draw_mode = params.get("draw_mode", "All edges")
        deduplicate = bool(params.get("deduplicate", True))

        # Build initial triangles in normalised complex space (unit radius)
        triangles = _initial_config(config, 1.0)

        # Iterative subdivision
        for i in range(depth):
            if cancelled_callback and cancelled_callback():
                return []
            triangles = _subdivide(triangles)
            if progress_callback:
                progress_callback(int(100 * (i + 1) / max(depth, 1)))

        # Canvas centre and drawing area in mm
        x1, y1, x2, y2 = canvas.drawing_area()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        rotation_rad = math.radians(rotation_deg)

        if draw_mode == "Rhombs":
            return _generate_rhombs(
                triangles,
                cx=cx,
                cy=cy,
                scale=radius_mm,
                rotation=rotation_rad,
                canvas_w=canvas.width_mm,
                canvas_h=canvas.height_mm,
                margin=canvas.margin_mm,
            )

        return _triangles_to_polylines(
            triangles,
            cx=cx,
            cy=cy,
            scale=radius_mm,
            rotation=rotation_rad,
            draw_mode=draw_mode,
            deduplicate=deduplicate,
        )

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Classic Sun",
                params={
                    "initial_config": "Sun",
                    "depth": 5,
                    "radius_mm": 100.0,
                    "rotation_deg": 0.0,
                    "draw_mode": "All edges",
                    "deduplicate": True,
                },
            ),
            Preset(
                name="Star Pattern",
                params={
                    "initial_config": "Star",
                    "depth": 5,
                    "radius_mm": 100.0,
                    "rotation_deg": 0.0,
                    "draw_mode": "All edges",
                    "deduplicate": True,
                },
            ),
            Preset(
                name="Thin Triangles Only",
                params={
                    "initial_config": "Sun",
                    "depth": 5,
                    "radius_mm": 100.0,
                    "rotation_deg": 0.0,
                    "draw_mode": "Thin only",
                    "deduplicate": True,
                },
            ),
        ]
