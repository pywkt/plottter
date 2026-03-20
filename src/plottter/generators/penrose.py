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
                choices=["All edges", "Thin only", "Thick only"],
                default="All edges",
                description="Which triangle type edges to render.",
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

        # Canvas centre in mm
        x1, y1, x2, y2 = canvas.drawing_area()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        rotation_rad = math.radians(rotation_deg)
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
