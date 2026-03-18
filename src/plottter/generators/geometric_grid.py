"""GeometricGridGenerator — tessellated grids with noise-driven cell content variation."""

from __future__ import annotations

import math
import random
from typing import Any

try:
    import noise as _noise_lib
    _NOISE_AVAILABLE = True
except ImportError:
    _NOISE_AVAILABLE = False

from plottter.generators import register_generator
from plottter.generators.base import (
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotate_points(
    points: list[tuple[float, float]],
    cx: float,
    cy: float,
    angle_rad: float,
) -> list[tuple[float, float]]:
    """Rotate *points* around (*cx*, *cy*) by *angle_rad* radians."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    result = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        result.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return result


def _apply_rotation(
    paths: list[Polyline],
    cx: float,
    cy: float,
    angle_rad: float,
) -> list[Polyline]:
    """Rotate all *paths* around (*cx*, *cy*). Returns paths unchanged when angle≈0."""
    if abs(angle_rad) < 1e-9:
        return paths
    return [_rotate_points(p, cx, cy, angle_rad) for p in paths]


# ---------------------------------------------------------------------------
# Cell shape helpers — all drawn centred at (cx, cy) within a half-size `s`
# ---------------------------------------------------------------------------

def _cell_outline(cx: float, cy: float, s: float) -> list[Polyline]:
    """Axis-aligned square cell boundary with half-size *s*."""
    return [[(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s), (cx - s, cy - s)]]


def _cell_diagonal(cx: float, cy: float, s: float) -> list[Polyline]:
    """One diagonal of the cell (top-left to bottom-right)."""
    return [[(cx - s, cy - s), (cx + s, cy + s)]]


def _cell_cross(cx: float, cy: float, s: float) -> list[Polyline]:
    """Both diagonals of the cell."""
    return [
        [(cx - s, cy - s), (cx + s, cy + s)],
        [(cx + s, cy - s), (cx - s, cy + s)],
    ]


def _cell_circle_inscribed(cx: float, cy: float, s: float, sides: int = 16) -> list[Polyline]:
    """Circle inscribed in the cell (radius = half cell size)."""
    pts = []
    for i in range(sides):
        angle = _TWO_PI * i / sides
        pts.append((cx + s * math.cos(angle), cy + s * math.sin(angle)))
    pts.append(pts[0])  # Explicitly close — avoids floating-point drift when i==sides
    return [pts]


def _cell_diamond_inscribed(cx: float, cy: float, s: float) -> list[Polyline]:
    """45°-rotated square (diamond) inscribed in the cell."""
    return [[(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy), (cx, cy - s)]]


def _cell_shape_paths(
    shape: str,
    cx: float,
    cy: float,
    s: float,
    rng: random.Random,
) -> list[Polyline]:
    """Return polylines for *shape* centred at (cx, cy) with half-size *s*."""
    _SHAPES = ["Outline", "Diagonal", "Cross", "Circle Inscribed", "Diamond Inscribed"]
    if shape == "Random Fill":
        chosen = rng.choice(_SHAPES)
        return _cell_shape_paths(chosen, cx, cy, s, rng)
    elif shape == "Diagonal":
        return _cell_diagonal(cx, cy, s)
    elif shape == "Cross":
        return _cell_cross(cx, cy, s)
    elif shape == "Circle Inscribed":
        return _cell_circle_inscribed(cx, cy, s)
    elif shape == "Diamond Inscribed":
        return _cell_diamond_inscribed(cx, cy, s)
    # Default: Outline (square)
    return _cell_outline(cx, cy, s)


# ---------------------------------------------------------------------------
# Hexagonal cell helper — handles "Outline" as a proper hexagon boundary
# ---------------------------------------------------------------------------

def _hex_cell_paths(
    cx: float,
    cy: float,
    cell_size: float,
    cell_shape: str,
    rng: random.Random,
) -> list[Polyline]:
    """Return polylines for a pointy-top hexagonal cell.

    * *cell_size* is the circumradius (vertex-to-centre distance = side length).
    * ``cell_shape == "Outline"`` draws only the hexagon boundary (7 points,
      first == last).
    * Any other shape is drawn using the hex inradius as half-size so it fits
      neatly inside the hexagon without protruding past the edges.
    """
    if cell_shape == "Outline":
        pts = []
        for k in range(7):  # 6 vertices + close
            # Pointy-top: vertices at 30° + 60°*k  (i.e. offset by -π/6 gives same)
            angle = _TWO_PI * k / 6 - math.pi / 6
            pts.append((cx + cell_size * math.cos(angle), cy + cell_size * math.sin(angle)))
        return [pts]
    # Inradius: distance from centre to midpoint of an edge = (√3/2) × circumradius
    inradius = cell_size * math.sqrt(3.0) / 2.0
    return _cell_shape_paths(cell_shape, cx, cy, inradius, rng)


# ---------------------------------------------------------------------------
# Triangular cell helper — handles "Outline" as a proper triangle boundary
# ---------------------------------------------------------------------------

def _tri_cell_paths(
    tri_pts: list[tuple[float, float]],
    cx: float,
    cy: float,
    cell_size: float,
    cell_shape: str,
    rng: random.Random,
) -> list[Polyline]:
    """Return polylines for a triangular cell.

    * *tri_pts* is the closed triangle boundary (4 points, first == last).
    * *cx*, *cy* is the centroid.
    * *cell_size* is the side length.
    * ``cell_shape == "Outline"`` draws the triangle boundary; otherwise an
      inscribed shape is drawn using the inradius as half-size.
    """
    if cell_shape == "Outline":
        return [tri_pts]
    # Equilateral triangle inradius = side / (2 × √3) = side × √3 / 6
    inradius = cell_size * math.sqrt(3.0) / 6.0
    return _cell_shape_paths(cell_shape, cx, cy, inradius, rng)


# ---------------------------------------------------------------------------
# Per-cell rotation helper
# ---------------------------------------------------------------------------

def _cell_rotation_rad(
    col: int,
    row: int,
    base_rotation_rad: float,
    rotation_noise_rad: float,
    noise_scale: float,
    noise_seed: int,
) -> float:
    """Compute per-cell rotation angle in radians.

    Combines a uniform *base_rotation_rad* with a noise-driven variation
    (amplitude *rotation_noise_rad*).  Uses a different noise offset (500) to
    the density-variation noise so the two fields are uncorrelated.
    """
    if rotation_noise_rad < 1e-9:
        return base_rotation_rad
    if _NOISE_AVAILABLE:
        noise_val = _noise_lib.pnoise2(
            col * noise_scale + 500.0,
            row * noise_scale + 500.0,
            base=noise_seed,
        )
    else:
        seed_val = (col * 1234567 + row * 7654321 + noise_seed) & 0xFFFF
        noise_val = (seed_val / 0xFFFF) * 2.0 - 1.0  # [-1, 1]
    return base_rotation_rad + noise_val * rotation_noise_rad


# ---------------------------------------------------------------------------
# Noise helper and position-based rotation for subdivision
# ---------------------------------------------------------------------------

def _noise_2d(x: float, y: float, seed: int) -> float:
    """Sample 2D Perlin noise at (x, y) with the given seed.

    Returns value in approximately [-0.7, 0.7] when Perlin noise is available,
    or a deterministic hash-based value in [-1, 1] as fallback.
    """
    if _NOISE_AVAILABLE:
        return _noise_lib.pnoise2(x, y, base=seed)
    xi = int(math.floor(x * 100.0)) & 0xFFFFFFFF
    yi = int(math.floor(y * 100.0)) & 0xFFFFFFFF
    sv = (xi * 2654435761 ^ yi * 2246822519 ^ seed * 1013904223) & 0xFFFFFFFF
    return (sv / 0x7FFFFFFF) - 1.0


def _pos_rotation_rad(
    cx: float,
    cy: float,
    base_rot_rad: float,
    rotation_noise_rad: float,
    noise_scale: float,
    noise_seed: int,
) -> float:
    """Compute rotation angle in radians based on position (cx, cy).

    Used instead of :func:`_cell_rotation_rad` when cell col/row indices are
    not available (e.g. during recursive subdivision).
    """
    if abs(rotation_noise_rad) < 1e-9:
        return base_rot_rad
    nv = _noise_2d(cx * noise_scale + 500.0, cy * noise_scale + 500.0, noise_seed)
    return base_rot_rad + nv * rotation_noise_rad


# ---------------------------------------------------------------------------
# Recursive subdivision helpers
# ---------------------------------------------------------------------------

# Subdivision threshold: cells/positions whose subdivision-noise exceeds this
# value will be further subdivided.  Roughly 35 % of cells trigger subdivision
# at each level when using Perlin noise (pnoise2 values ≈ [-0.7, 0.7]).
_SUBDIV_THRESHOLD = 0.3


def _square_cell_recursive(
    cx: float,
    cy: float,
    cell_size: float,
    cell_shape: str,
    depth: int,
    max_depth: int,
    noise_scale: float,
    noise_seed: int,
    base_rot_rad: float,
    rot_noise_rad: float,
    rng: random.Random,
) -> list[Polyline]:
    """Generate polylines for a square cell, recursively subdividing into 4 sub-cells.

    At each level, a position-based noise sample (offset by +100 to de-correlate
    from density noise) decides whether to subdivide further.  The recursion
    stops at *max_depth*, or when the noise value is below the threshold.
    """
    half = cell_size / 2.0
    if depth < max_depth:
        nv = _noise_2d(cx * noise_scale + 100.0, cy * noise_scale + 100.0, noise_seed)
        if nv > _SUBDIV_THRESHOLD:
            # Subdivide into 4 equal sub-cells, each half the size
            q = half / 2.0
            result: list[Polyline] = []
            for dcx, dcy in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)):
                result.extend(_square_cell_recursive(
                    cx + dcx * q, cy + dcy * q,
                    half, cell_shape, depth + 1, max_depth,
                    noise_scale, noise_seed, base_rot_rad, rot_noise_rad, rng,
                ))
            return result
    # Draw this cell at its current size
    paths = _cell_shape_paths(cell_shape, cx, cy, half, rng)
    rot_rad = _pos_rotation_rad(cx, cy, base_rot_rad, rot_noise_rad, noise_scale, noise_seed)
    return _apply_rotation(paths, cx, cy, rot_rad)


def _hex_cell_recursive(
    cx: float,
    cy: float,
    cell_size: float,
    cell_shape: str,
    depth: int,
    max_depth: int,
    noise_scale: float,
    noise_seed: int,
    base_rot_rad: float,
    rot_noise_rad: float,
    rng: random.Random,
) -> list[Polyline]:
    """Generate polylines for a hexagonal cell, recursively subdividing into 7 sub-hexagons.

    The 7 sub-hexagons consist of 1 central hexagon plus 6 surrounding hexagons
    arranged in a ring.  Sub-hexagon circumradius is chosen so that all 7 fit
    approximately inside the parent hexagon.
    """
    if depth < max_depth:
        nv = _noise_2d(cx * noise_scale + 100.0, cy * noise_scale + 100.0, noise_seed)
        if nv > _SUBDIV_THRESHOLD:
            # Sub-hex circumradius: f=0.35 ensures 1 + 6 satellites fit inside parent
            sub_size = cell_size * 0.35
            # Distance between adjacent touching hex centres = sqrt(3) * circumradius
            sub_dist = sub_size * math.sqrt(3.0)
            result: list[Polyline] = []
            # Central sub-hexagon
            result.extend(_hex_cell_recursive(
                cx, cy, sub_size, cell_shape, depth + 1, max_depth,
                noise_scale, noise_seed, base_rot_rad, rot_noise_rad, rng,
            ))
            # 6 surrounding sub-hexagons (pointy-top offset angles)
            for k in range(6):
                angle = _TWO_PI * k / 6.0 - math.pi / 6.0
                result.extend(_hex_cell_recursive(
                    cx + sub_dist * math.cos(angle),
                    cy + sub_dist * math.sin(angle),
                    sub_size, cell_shape, depth + 1, max_depth,
                    noise_scale, noise_seed, base_rot_rad, rot_noise_rad, rng,
                ))
            return result
    # Draw this hexagonal cell
    paths = _hex_cell_paths(cx, cy, cell_size, cell_shape, rng)
    rot_rad = _pos_rotation_rad(cx, cy, base_rot_rad, rot_noise_rad, noise_scale, noise_seed)
    return _apply_rotation(paths, cx, cy, rot_rad)


def _tri_cell_recursive(
    tri_pts: list[tuple[float, float]],
    cx: float,
    cy: float,
    cell_size: float,
    cell_shape: str,
    depth: int,
    max_depth: int,
    noise_scale: float,
    noise_seed: int,
    base_rot_rad: float,
    rot_noise_rad: float,
    rng: random.Random,
) -> list[Polyline]:
    """Generate polylines for a triangular cell, recursively subdividing into 4 sub-triangles.

    Uses midpoint subdivision: connect the midpoints of all three sides to
    produce 4 congruent equilateral sub-triangles (3 corner triangles + 1
    central inverted triangle).
    """
    if depth < max_depth:
        nv = _noise_2d(cx * noise_scale + 100.0, cy * noise_scale + 100.0, noise_seed)
        if nv > _SUBDIV_THRESHOLD:
            p0, p1, p2 = tri_pts[0], tri_pts[1], tri_pts[2]
            m01 = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
            m12 = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            m20 = ((p2[0] + p0[0]) / 2.0, (p2[1] + p0[1]) / 2.0)
            sub_size = cell_size / 2.0
            # Four sub-triangles with their centroids
            sub_tris: list[tuple[list[tuple[float, float]], float, float]] = [
                ([p0, m01, m20, p0],
                 (p0[0] + m01[0] + m20[0]) / 3.0,
                 (p0[1] + m01[1] + m20[1]) / 3.0),
                ([m01, p1, m12, m01],
                 (m01[0] + p1[0] + m12[0]) / 3.0,
                 (m01[1] + p1[1] + m12[1]) / 3.0),
                ([m20, m12, p2, m20],
                 (m20[0] + m12[0] + p2[0]) / 3.0,
                 (m20[1] + m12[1] + p2[1]) / 3.0),
                ([m01, m12, m20, m01],
                 (m01[0] + m12[0] + m20[0]) / 3.0,
                 (m01[1] + m12[1] + m20[1]) / 3.0),
            ]
            result: list[Polyline] = []
            for sub_tri, scx, scy in sub_tris:
                result.extend(_tri_cell_recursive(
                    sub_tri, scx, scy, sub_size, cell_shape,
                    depth + 1, max_depth, noise_scale, noise_seed,
                    base_rot_rad, rot_noise_rad, rng,
                ))
            return result
    # Draw this triangular cell
    paths = _tri_cell_paths(tri_pts, cx, cy, cell_size, cell_shape, rng)
    rot_rad = _pos_rotation_rad(cx, cy, base_rot_rad, rot_noise_rad, noise_scale, noise_seed)
    return _apply_rotation(paths, cx, cy, rot_rad)


@register_generator
class GeometricGridGenerator(Generator):
    """Generates tessellated grids (square, hexagonal, triangular) with noise-driven content variation."""

    name = "Geometric Grid"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="grid_type",
                label="Grid type",
                choices=["Square", "Hexagonal", "Triangular"],
                default="Square",
                description="Tessellation type for the grid",
            ),
            FloatParam(
                name="cell_size_mm",
                label="Cell size (mm)",
                min=2.0,
                max=50.0,
                step=0.5,
                default=10.0,
                description="Size of each grid cell in mm",
            ),
            ChoiceParam(
                name="cell_shape",
                label="Cell content",
                choices=["Outline", "Diagonal", "Cross", "Circle Inscribed", "Diamond Inscribed", "Random Fill"],
                default="Outline",
                description="Shape drawn inside each grid cell",
            ),
            IntParam(
                name="subdivisions",
                label="Subdivisions",
                min=0,
                max=3,
                step=1,
                default=0,
                description=(
                    "Recursively subdivide cells in dense noise areas — "
                    "0 = no subdivision, 1–3 = up to N levels of sub-cells "
                    "(square → 4 sub-cells, triangular → 4 sub-triangles, "
                    "hexagonal → 7 approximate sub-hexagons)"
                ),
            ),
            FloatParam(
                name="cell_rotation",
                label="Cell rotation (°)",
                min=0.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate each cell's content by this fixed angle in degrees",
            ),
            FloatParam(
                name="rotation_noise",
                label="Rotation noise (°)",
                min=0.0,
                max=90.0,
                step=1.0,
                default=0.0,
                description="Noise-based rotation variation per cell in degrees — 0 = all cells use the same rotation",
            ),
            FloatParam(
                name="noise_scale",
                label="Noise scale",
                min=0.01,
                max=1.0,
                step=0.01,
                default=0.1,
                description="Scale of Perlin noise field — smaller = larger noise features",
            ),
            IntParam(
                name="noise_seed",
                label="Noise seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for noise generation",
            ),
            FloatParam(
                name="density_variation",
                label="Density variation",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.5,
                description="Noise-driven variation in cell content density — higher values create more gaps",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the generated output on the canvas page (mm)",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the generated output on the canvas page (mm)",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Honeycomb",
                params={
                    "grid_type": "Hexagonal",
                    "cell_size_mm": 8.0,
                    "cell_shape": "Outline",
                    "density_variation": 0.0,
                    "subdivisions": 0,
                },
            ),
            Preset(
                name="Broken Tiles",
                params={
                    "grid_type": "Square",
                    "cell_size_mm": 12.0,
                    "cell_shape": "Random Fill",
                    "density_variation": 0.4,
                    "rotation_noise": 15.0,
                    "noise_scale": 0.1,
                    "subdivisions": 0,
                },
            ),
            Preset(
                name="Triangle Mesh",
                params={
                    "grid_type": "Triangular",
                    "cell_size_mm": 10.0,
                    "cell_shape": "Outline",
                    "density_variation": 0.3,
                    "subdivisions": 1,
                },
            ),
            Preset(
                name="City Grid",
                params={
                    "grid_type": "Square",
                    "cell_size_mm": 6.0,
                    "cell_shape": "Cross",
                    "density_variation": 0.6,
                    "noise_scale": 0.05,
                    "subdivisions": 0,
                },
            ),
            Preset(
                name="Hex Detail",
                params={
                    "grid_type": "Hexagonal",
                    "cell_size_mm": 15.0,
                    "cell_shape": "Circle Inscribed",
                    "density_variation": 0.3,
                    "subdivisions": 2,
                },
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        grid_type = str(params.get("grid_type", "Square"))
        cell_size = float(params.get("cell_size_mm", 10.0))
        cell_shape = str(params.get("cell_shape", "Outline"))
        cell_rotation = float(params.get("cell_rotation", 0.0))
        rotation_noise = float(params.get("rotation_noise", 0.0))
        noise_scale = float(params.get("noise_scale", 0.1))
        noise_seed = int(params.get("noise_seed", 42))
        density_variation = float(params.get("density_variation", 0.5))
        subdivisions = int(params.get("subdivisions", 0))
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()

        # Random number generator seeded for reproducibility (used for Random Fill)
        rng = random.Random(noise_seed)

        # Pre-compute rotation in radians
        base_rot_rad = math.radians(cell_rotation)
        rot_noise_rad = math.radians(rotation_noise)

        if grid_type == "Hexagonal":
            result = self._generate_hex_grid(
                cell_size, cell_shape, base_rot_rad, rot_noise_rad,
                noise_scale, noise_seed, density_variation, subdivisions,
                draw_x1, draw_y1, draw_x2, draw_y2, rng,
                progress_callback, cancelled_callback,
            )
        elif grid_type == "Triangular":
            result = self._generate_tri_grid(
                cell_size, cell_shape, base_rot_rad, rot_noise_rad,
                noise_scale, noise_seed, density_variation, subdivisions,
                draw_x1, draw_y1, draw_x2, draw_y2, rng,
                progress_callback, cancelled_callback,
            )
        else:
            result = self._generate_square_grid(
                cell_size, cell_shape, base_rot_rad, rot_noise_rad,
                noise_scale, noise_seed, density_variation, subdivisions,
                draw_x1, draw_y1, draw_x2, draw_y2, rng,
                progress_callback, cancelled_callback,
            )

        if progress_callback:
            progress_callback(100)

        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result

    # ------------------------------------------------------------------
    # Square tessellation
    # ------------------------------------------------------------------

    def _generate_square_grid(
        self,
        cell_size: float,
        cell_shape: str,
        base_rot_rad: float,
        rot_noise_rad: float,
        noise_scale: float,
        noise_seed: int,
        density_variation: float,
        subdivisions: int,
        draw_x1: float,
        draw_y1: float,
        draw_x2: float,
        draw_y2: float,
        rng: random.Random,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        cols = max(1, int(math.ceil(draw_w / cell_size)))
        rows = max(1, int(math.ceil(draw_h / cell_size)))

        # Centre the grid on the drawing area
        grid_w = cols * cell_size
        grid_h = rows * cell_size
        start_x = draw_x1 + (draw_w - grid_w) / 2.0
        start_y = draw_y1 + (draw_h - grid_h) / 2.0

        half = cell_size / 2.0
        use_noise = _NOISE_AVAILABLE and density_variation > 0.0
        # density_variation in [0,1]: when 0, draw everything; when 1, ~50% skipped
        skip_threshold = density_variation - 1.0  # pnoise2 ranges in ~[-1, 1]

        result: list[Polyline] = []
        total = cols * rows

        for row in range(rows):
            for col in range(cols):
                if cancelled_callback and cancelled_callback():
                    break

                cx = start_x + col * cell_size + half
                cy = start_y + row * cell_size + half

                # Skip based on density variation
                if use_noise:
                    density_noise = _noise_lib.pnoise2(
                        col * noise_scale,
                        row * noise_scale,
                        base=noise_seed,
                    )
                    if density_noise < skip_threshold:
                        continue
                elif density_variation > 0.0:
                    # Fallback simple skip when noise unavailable
                    seed_val = (col * 2654435761 + row * 2246822519 + noise_seed) & 0xFFFFFFFF
                    if (seed_val / 0xFFFFFFFF) < density_variation * 0.5:
                        continue

                if subdivisions > 0:
                    paths = _square_cell_recursive(
                        cx, cy, cell_size, cell_shape, 0, subdivisions,
                        noise_scale, noise_seed, base_rot_rad, rot_noise_rad, rng,
                    )
                else:
                    paths = _cell_shape_paths(cell_shape, cx, cy, half, rng)
                    rot_rad = _cell_rotation_rad(col, row, base_rot_rad, rot_noise_rad, noise_scale, noise_seed)
                    paths = _apply_rotation(paths, cx, cy, rot_rad)

                result.extend(paths)

                if progress_callback:
                    idx = row * cols + col
                    if idx % 50 == 0:
                        progress_callback(int(idx / total * 95))

            if cancelled_callback and cancelled_callback():
                break

        return result

    # ------------------------------------------------------------------
    # Hexagonal tessellation
    # ------------------------------------------------------------------

    def _generate_hex_grid(
        self,
        cell_size: float,
        cell_shape: str,
        base_rot_rad: float,
        rot_noise_rad: float,
        noise_scale: float,
        noise_seed: int,
        density_variation: float,
        subdivisions: int,
        draw_x1: float,
        draw_y1: float,
        draw_x2: float,
        draw_y2: float,
        rng: random.Random,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        """Pointy-top hexagonal tessellation using offset-row layout.

        Geometry (side length = *cell_size*):
        - Circumradius (vertex-to-centre) = cell_size
        - Column spacing (same row)   = sqrt(3) * cell_size
        - Row spacing (centre-to-centre) = 1.5 * cell_size
        - Odd rows are offset right by (sqrt(3)/2) * cell_size
        """
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        # Horizontal spacing between adjacent cell centres in the same row
        col_spacing = cell_size * math.sqrt(3.0)
        # Vertical spacing between row centres
        row_spacing = cell_size * 1.5

        cols = max(1, int(math.ceil(draw_w / col_spacing)) + 1)
        rows = max(1, int(math.ceil(draw_h / row_spacing)) + 1)

        grid_w = cols * col_spacing
        grid_h = rows * row_spacing
        start_x = draw_x1 + (draw_w - grid_w) / 2.0
        start_y = draw_y1 + (draw_h - grid_h) / 2.0

        use_noise = _NOISE_AVAILABLE and density_variation > 0.0
        skip_threshold = density_variation - 1.0

        result: list[Polyline] = []
        total = cols * rows

        for row in range(rows):
            # Odd rows are offset to the right by half a column spacing
            row_offset_x = (col_spacing / 2.0) if row % 2 == 1 else 0.0

            for col in range(cols):
                if cancelled_callback and cancelled_callback():
                    break

                cx = start_x + col * col_spacing + row_offset_x + col_spacing / 2.0
                cy = start_y + row * row_spacing + cell_size

                # Skip based on density variation
                if use_noise:
                    density_noise = _noise_lib.pnoise2(
                        col * noise_scale,
                        row * noise_scale,
                        base=noise_seed,
                    )
                    if density_noise < skip_threshold:
                        continue
                elif density_variation > 0.0:
                    seed_val = (col * 2654435761 + row * 2246822519 + noise_seed) & 0xFFFFFFFF
                    if (seed_val / 0xFFFFFFFF) < density_variation * 0.5:
                        continue

                if subdivisions > 0:
                    paths = _hex_cell_recursive(
                        cx, cy, cell_size, cell_shape, 0, subdivisions,
                        noise_scale, noise_seed, base_rot_rad, rot_noise_rad, rng,
                    )
                else:
                    paths = _hex_cell_paths(cx, cy, cell_size, cell_shape, rng)
                    rot_rad = _cell_rotation_rad(col, row, base_rot_rad, rot_noise_rad, noise_scale, noise_seed)
                    paths = _apply_rotation(paths, cx, cy, rot_rad)

                result.extend(paths)

                if progress_callback:
                    idx = row * cols + col
                    if idx % 50 == 0:
                        progress_callback(int(idx / total * 95))

            if cancelled_callback and cancelled_callback():
                break

        return result

    # ------------------------------------------------------------------
    # Triangular tessellation
    # ------------------------------------------------------------------

    def _generate_tri_grid(
        self,
        cell_size: float,
        cell_shape: str,
        base_rot_rad: float,
        rot_noise_rad: float,
        noise_scale: float,
        noise_seed: int,
        density_variation: float,
        subdivisions: int,
        draw_x1: float,
        draw_y1: float,
        draw_x2: float,
        draw_y2: float,
        rng: random.Random,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        """Equilateral triangular tessellation.

        Each grid cell generates two triangles: one pointing up (apex above
        base) and one pointing down (apex below base).  Triangle height =
        cell_size * sqrt(3) / 2.
        """
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        tri_h = cell_size * math.sqrt(3.0) / 2.0
        cols = max(1, int(math.ceil(draw_w / cell_size)) + 1)
        rows = max(1, int(math.ceil(draw_h / tri_h)) + 1)

        grid_w = cols * cell_size
        grid_h = rows * tri_h
        start_x = draw_x1 + (draw_w - grid_w) / 2.0
        start_y = draw_y1 + (draw_h - grid_h) / 2.0

        use_noise = _NOISE_AVAILABLE and density_variation > 0.0
        skip_threshold = density_variation - 1.0

        result: list[Polyline] = []
        total = cols * rows * 2  # 2 triangles per cell

        tri_idx = 0
        for row in range(rows):
            for col in range(cols):
                if cancelled_callback and cancelled_callback():
                    break

                x0 = start_x + col * cell_size
                y0 = start_y + row * tri_h
                x1 = x0 + cell_size
                y1 = y0 + tri_h

                for up in (True, False):
                    if up:
                        # Base at y1 (lower), apex at y0 (upper)
                        tri_pts = [(x0, y1), (x1, y1), (x0 + cell_size / 2.0, y0), (x0, y1)]
                    else:
                        # Base at y0 (upper), apex at y1 (lower)
                        tri_pts = [(x1, y0), (x0, y0), (x1 - cell_size / 2.0, y1), (x1, y0)]

                    cx = sum(p[0] for p in tri_pts[:3]) / 3.0
                    cy = sum(p[1] for p in tri_pts[:3]) / 3.0

                    # Skip based on density variation
                    if use_noise:
                        density_noise = _noise_lib.pnoise2(
                            col * noise_scale + (0.5 if not up else 0.0),
                            row * noise_scale,
                            base=noise_seed,
                        )
                        if density_noise < skip_threshold:
                            tri_idx += 1
                            continue
                    elif density_variation > 0.0:
                        seed_val = (col * 2654435761 + row * 2246822519 + tri_idx + noise_seed) & 0xFFFFFFFF
                        if (seed_val / 0xFFFFFFFF) < density_variation * 0.5:
                            tri_idx += 1
                            continue

                    if subdivisions > 0:
                        paths = _tri_cell_recursive(
                            tri_pts, cx, cy, cell_size, cell_shape,
                            0, subdivisions, noise_scale, noise_seed,
                            base_rot_rad, rot_noise_rad, rng,
                        )
                    else:
                        paths = _tri_cell_paths(tri_pts, cx, cy, cell_size, cell_shape, rng)
                        rot_rad = _cell_rotation_rad(
                            col * 2 + (0 if up else 1), row,
                            base_rot_rad, rot_noise_rad, noise_scale, noise_seed,
                        )
                        paths = _apply_rotation(paths, cx, cy, rot_rad)

                    result.extend(paths)

                    if progress_callback and tri_idx % 50 == 0:
                        progress_callback(int(tri_idx / total * 95))
                    tri_idx += 1

            if cancelled_callback and cancelled_callback():
                break

        return result
