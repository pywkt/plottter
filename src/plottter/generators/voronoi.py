"""VoronoiGenerator — Voronoi / Delaunay pattern generation.

Supports Voronoi cell edges, Delaunay triangulation edges, combined output,
and Voronoi cells with centroid markers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, box as shapely_box

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


# ---------------------------------------------------------------------------
# Seed point generation strategies
# All functions return np.ndarray of shape (N, 2) with coordinates in mm,
# relative to the drawing area origin (i.e. (0,0) → top-left of drawing area).
# ---------------------------------------------------------------------------


def _seeds_random(n: int, w: float, h: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform random seed points within the drawing area (0..w × 0..h)."""
    xy = rng.random((n, 2))
    xy[:, 0] *= w
    xy[:, 1] *= h
    return xy


def _seeds_poisson_disk(
    spacing: float, w: float, h: float, rng: np.random.Generator
) -> np.ndarray:
    """Blue-noise Poisson disk seed points.

    Uses ``scipy.stats.qmc.PoissonDisk`` (available since SciPy 1.8).
    Falls back to Bridson's rejection-sampling algorithm for older SciPy.
    """
    try:
        from scipy.stats.qmc import PoissonDisk  # type: ignore[import]
    except ImportError:
        return _seeds_poisson_disk_bridson(spacing, w, h, rng)

    # radius normalised to [0, 1]^2 (scipy operates in the unit hypercube)
    radius = spacing / max(w, h)
    seed_int = int(rng.integers(0, 2**31))
    sampler = PoissonDisk(d=2, radius=radius, seed=seed_int)

    # Estimate maximum number of points that can fit (upper bound)
    max_pts = max(10, int(w * h / (math.pi * spacing**2)) * 4)
    samples = sampler.random(max_pts)  # shape (<=max_pts, 2) in [0,1]^2

    # Scale from unit square to drawing area
    xy = samples * np.array([w, h])
    return xy


def _seeds_poisson_disk_bridson(
    spacing: float, w: float, h: float, rng: np.random.Generator
) -> np.ndarray:
    """Bridson's algorithm for Poisson disk sampling (SciPy fallback)."""
    cell_size = spacing / math.sqrt(2.0)
    cols = max(1, int(math.ceil(w / cell_size)))
    rows = max(1, int(math.ceil(h / cell_size)))
    grid: list[tuple[float, float] | None] = [None] * (cols * rows)
    spacing_sq = spacing * spacing

    def _grid_idx(x: float, y: float) -> int:
        gx = min(cols - 1, int(x / cell_size))
        gy = min(rows - 1, int(y / cell_size))
        return gy * cols + gx

    def _too_close(x: float, y: float) -> bool:
        gx = int(x / cell_size)
        gy = int(y / cell_size)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ngx, ngy = gx + dx, gy + dy
                if 0 <= ngx < cols and 0 <= ngy < rows:
                    pt = grid[ngy * cols + ngx]
                    if pt is not None:
                        if (pt[0] - x) ** 2 + (pt[1] - y) ** 2 < spacing_sq:
                            return True
        return False

    x0 = float(rng.uniform(0.0, w))
    y0 = float(rng.uniform(0.0, h))
    grid[_grid_idx(x0, y0)] = (x0, y0)
    active: list[tuple[float, float]] = [(x0, y0)]
    points: list[tuple[float, float]] = [(x0, y0)]
    k = 30  # candidates per active point

    while active:
        idx = int(rng.integers(0, len(active)))
        ax, ay = active[idx]
        found = False
        for _ in range(k):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            r = float(rng.uniform(spacing, 2.0 * spacing))
            nx = ax + r * math.cos(angle)
            ny = ay + r * math.sin(angle)
            if 0.0 <= nx <= w and 0.0 <= ny <= h and not _too_close(nx, ny):
                grid[_grid_idx(nx, ny)] = (nx, ny)
                active.append((nx, ny))
                points.append((nx, ny))
                found = True
                break
        if not found:
            active.pop(idx)

    if not points:
        return np.empty((0, 2), dtype=float)
    return np.array(points, dtype=float)


def _seeds_grid_jitter(
    spacing: float, jitter: float, w: float, h: float, rng: np.random.Generator
) -> np.ndarray:
    """Regular grid with a random offset up to ``jitter * spacing`` per point."""
    xs = np.arange(spacing / 2.0, w, spacing)
    ys = np.arange(spacing / 2.0, h, spacing)
    xx, yy = np.meshgrid(xs, ys)
    n = xx.size
    offset = jitter * spacing
    dx = rng.uniform(-offset, offset, n)
    dy = rng.uniform(-offset, offset, n)
    pts = np.column_stack([xx.ravel() + dx, yy.ravel() + dy])
    # Clip to drawing area bounds
    mask = (pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)
    return pts[mask]


def _seeds_phyllotaxis(n: int, w: float, h: float) -> np.ndarray:
    """Golden-angle spiral (phyllotaxis) — uniform area density, centred on canvas.

    The golden angle ≈ 137.508° ensures successive seeds never share a spoke,
    producing the sunflower-seed packing pattern.
    """
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 2.3999 rad ≈ 137.508°
    # Radius scaled so the outermost ring fits inside the smaller half-dimension
    radius = min(w, h) / 2.0
    cx, cy = w / 2.0, h / 2.0

    pts = np.empty((n, 2), dtype=float)
    for i in range(n):
        r = radius * math.sqrt(i / n)
        theta = i * golden_angle
        pts[i, 0] = cx + r * math.cos(theta)
        pts[i, 1] = cy + r * math.sin(theta)
    return pts


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_voronoi(
    seeds: np.ndarray, bbox: tuple[float, float, float, float]
) -> list[Polyline]:
    """Render Voronoi cell edges clipped to *bbox* using the mirror-point approach.

    The mirror-point technique reflects every seed across each of the four
    boundaries, creating 5× as many points.  This guarantees that all cells
    belonging to the original ``n`` seeds are finite (no infinite rays), so
    we never need to manually project infinite vertices.

    Parameters
    ----------
    seeds:
        Array of shape (N, 2) with seed coordinates in canvas mm.
    bbox:
        ``(x_min, y_min, x_max, y_max)`` clipping rectangle in canvas mm.
    """
    from scipy.spatial import Voronoi  # lazy import – optional heavy dep

    x_min, y_min, x_max, y_max = bbox
    n = len(seeds)
    if n < 4:
        return []

    # Reflect seeds across all four boundaries → 4 extra copies
    reflected = np.vstack(
        [
            seeds,  # original: indices 0 … n-1
            np.column_stack([2.0 * x_min - seeds[:, 0], seeds[:, 1]]),  # left
            np.column_stack([2.0 * x_max - seeds[:, 0], seeds[:, 1]]),  # right
            np.column_stack([seeds[:, 0], 2.0 * y_min - seeds[:, 1]]),  # top
            np.column_stack([seeds[:, 0], 2.0 * y_max - seeds[:, 1]]),  # bottom
        ]
    )

    vor = Voronoi(reflected)
    clip_rect = shapely_box(x_min, y_min, x_max, y_max)

    # Collect all ridge edges that border at least one original seed
    seen: set[tuple[int, int]] = set()
    polylines: list[Polyline] = []

    for (p, q), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        # Skip ridges whose both flanking sites are mirror copies
        if p >= n and q >= n:
            continue
        # Infinite ridges should not exist with the mirror approach, but guard anyway
        if v1 < 0 or v2 < 0:
            continue

        key = (min(v1, v2), max(v1, v2))
        if key in seen:
            continue
        seen.add(key)

        a = vor.vertices[v1]
        b = vor.vertices[v2]

        line = LineString([a, b])
        clipped = line.intersection(clip_rect)

        if clipped.is_empty:
            continue

        if clipped.geom_type == "LineString":
            coords = list(clipped.coords)
            if len(coords) >= 2:
                polylines.append([(float(x), float(y)) for x, y in coords])
        elif clipped.geom_type == "MultiLineString":
            for part in clipped.geoms:
                coords = list(part.coords)
                if len(coords) >= 2:
                    polylines.append([(float(x), float(y)) for x, y in coords])

    return polylines


def _render_delaunay(
    seeds: np.ndarray, bbox: tuple[float, float, float, float]
) -> list[Polyline]:
    """Render Delaunay triangulation edges clipped to *bbox*."""
    from scipy.spatial import Delaunay  # lazy import

    x_min, y_min, x_max, y_max = bbox
    n = len(seeds)
    if n < 3:
        return []

    tri = Delaunay(seeds)
    clip_rect = shapely_box(x_min, y_min, x_max, y_max)

    seen: set[tuple[int, int]] = set()
    polylines: list[Polyline] = []

    for simplex in tri.simplices:
        for i in range(3):
            a_idx = int(simplex[i])
            b_idx = int(simplex[(i + 1) % 3])
            key = (min(a_idx, b_idx), max(a_idx, b_idx))
            if key in seen:
                continue
            seen.add(key)

            a = seeds[a_idx]
            b = seeds[b_idx]

            line = LineString([a, b])
            clipped = line.intersection(clip_rect)

            if clipped.is_empty:
                continue

            if clipped.geom_type == "LineString":
                coords = list(clipped.coords)
                if len(coords) >= 2:
                    polylines.append([(float(x), float(y)) for x, y in coords])
            elif clipped.geom_type == "MultiLineString":
                for part in clipped.geoms:
                    coords = list(part.coords)
                    if len(coords) >= 2:
                        polylines.append([(float(x), float(y)) for x, y in coords])

    return polylines


def _render_centroids(seeds: np.ndarray, radius: float) -> list[Polyline]:
    """Render a small circle at each seed location to mark Voronoi centroids."""
    n_pts = 12
    angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    polylines: list[Polyline] = []
    for cx, cy in seeds:
        circle: Polyline = [
            (float(cx + radius * ca), float(cy + radius * sa))
            for ca, sa in zip(cos_a, sin_a)
        ]
        circle.append(circle[0])  # close the ring
        polylines.append(circle)
    return polylines


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


@register_generator
class VoronoiGenerator(Generator):
    """Voronoi / Delaunay pattern generator.

    Supports multiple seed point strategies and render modes:
    Voronoi cell edges, Delaunay triangulation edges, combined output,
    and Voronoi edges with centroid markers.
    """

    name = "Voronoi / Delaunay"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="render_mode",
                label="Render Mode",
                choices=[
                    "Voronoi Edges",
                    "Delaunay Edges",
                    "Both",
                    "Voronoi + Centroids",
                ],
                default="Voronoi Edges",
                description=(
                    "What to draw: Voronoi cell edges, Delaunay triangulation edges, "
                    "both, or Voronoi edges with centroid markers."
                ),
            ),
            FloatParam(
                name="centroid_radius_mm",
                label="Centroid Radius (mm)",
                min=0.1,
                max=5.0,
                step=0.1,
                default=0.5,
                visible_when={"render_mode": ["Voronoi + Centroids"]},
                description="Radius of the centroid marker circles in mm.",
            ),
            IntParam(
                name="num_points",
                label="Number of Points",
                min=50,
                max=10000,
                step=10,
                default=500,
                description="Number of seed points (used by Random and Phyllotaxis methods).",
            ),
            ChoiceParam(
                name="seed_method",
                label="Seed Method",
                choices=["Random", "Poisson Disk", "Grid Jitter", "Phyllotaxis"],
                default="Random",
                description="Strategy for placing seed points.",
            ),
            FloatParam(
                name="poisson_spacing_mm",
                label="Poisson Spacing (mm)",
                min=0.5,
                max=20.0,
                step=0.1,
                default=3.0,
                visible_when={"seed_method": ["Poisson Disk"]},
                description="Minimum distance between Poisson disk sample points in mm.",
            ),
            FloatParam(
                name="grid_spacing_mm",
                label="Grid Spacing (mm)",
                min=1.0,
                max=50.0,
                step=0.5,
                default=5.0,
                visible_when={"seed_method": ["Grid Jitter"]},
                description="Center-to-center grid spacing in mm.",
            ),
            FloatParam(
                name="grid_jitter",
                label="Grid Jitter",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.5,
                visible_when={"seed_method": ["Grid Jitter"]},
                description=(
                    "Random offset magnitude as a fraction of grid spacing "
                    "(0 = no jitter, 1 = full spacing)."
                ),
            ),
            IntParam(
                name="random_seed",
                label="Random Seed",
                min=0,
                max=99999,
                step=1,
                default=42,
                description="Seed for the random number generator (for reproducibility).",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the output on the canvas (mm).",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the output on the canvas (mm).",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Random Scatter",
                params={"num_points": 500, "seed_method": "Random", "random_seed": 42},
            ),
            Preset(
                name="Blue Noise",
                params={
                    "seed_method": "Poisson Disk",
                    "poisson_spacing_mm": 5.0,
                    "random_seed": 42,
                },
            ),
            Preset(
                name="Jittered Grid",
                params={
                    "seed_method": "Grid Jitter",
                    "grid_spacing_mm": 8.0,
                    "grid_jitter": 0.5,
                    "random_seed": 42,
                },
            ),
            Preset(
                name="Phyllotaxis Spiral",
                params={"num_points": 500, "seed_method": "Phyllotaxis"},
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        num_points = int(params.get("num_points", 500))
        seed_method = str(params.get("seed_method", "Random"))
        poisson_spacing = float(params.get("poisson_spacing_mm", 3.0))
        grid_spacing = float(params.get("grid_spacing_mm", 5.0))
        grid_jitter = float(params.get("grid_jitter", 0.5))
        random_seed = int(params.get("random_seed", 42))
        render_mode = str(params.get("render_mode", "Voronoi Edges"))
        centroid_radius = float(params.get("centroid_radius_mm", 0.5))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        rng = np.random.default_rng(random_seed)

        if seed_method == "Poisson Disk":
            seeds = _seeds_poisson_disk(poisson_spacing, draw_w, draw_h, rng)
        elif seed_method == "Grid Jitter":
            seeds = _seeds_grid_jitter(grid_spacing, grid_jitter, draw_w, draw_h, rng)
        elif seed_method == "Phyllotaxis":
            seeds = _seeds_phyllotaxis(num_points, draw_w, draw_h)
        else:  # "Random" (default)
            seeds = _seeds_random(num_points, draw_w, draw_h, rng)

        # Translate seed coordinates from drawing-area-local to canvas coordinates
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        seeds = seeds + np.array([draw_x1 + x_off, draw_y1 + y_off])

        if progress_callback:
            progress_callback(20)

        # Canvas drawing-area bounds used for Voronoi mirroring and clipping
        bbox = (draw_x1, draw_y1, draw_x2, draw_y2)

        polylines: list[Polyline] = []

        if render_mode in ("Voronoi Edges", "Both", "Voronoi + Centroids"):
            polylines.extend(_render_voronoi(seeds, bbox))

        if progress_callback:
            progress_callback(70)

        if render_mode in ("Delaunay Edges", "Both"):
            polylines.extend(_render_delaunay(seeds, bbox))

        if render_mode == "Voronoi + Centroids":
            polylines.extend(_render_centroids(seeds, centroid_radius))

        if progress_callback:
            progress_callback(100)

        return polylines
