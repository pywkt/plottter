"""MosaicHatchGenerator — Delaunay-triangulated or Voronoi-tessellated hatching from image brightness."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
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
from plottter.scene3d.hatching import _hatch_polygon

_AREA_EPSILON = 1e-6


def _edge_aware_seeds(
    gray: np.ndarray,
    img_rect: tuple[float, float, float, float],
    num_points: int,
    edge_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate seed points with higher density near image edges.

    Parameters
    ----------
    gray:
        Grayscale image array (uint8, shape HxW).
    img_rect:
        Bounding box in mm: (x1, y1, x2, y2).
    num_points:
        Target number of seed points (excluding the 4 corners).
    edge_weight:
        0 = uniform distribution, 1 = all points attracted to edges.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with (x_mm, y_mm) coordinates.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for mosaic hatching. "
            "Install with: pip install opencv-python"
        ) from exc

    img_x1, img_y1, img_x2, img_y2 = img_rect
    img_h, img_w = gray.shape[:2]

    # Step 1: Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Step 2: Blur the edge map to create a smooth density field
    edge_density = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 3)

    # Step 3: Combined weight map: uniform + edge attraction
    # Weights in [1 - edge_weight, 1] — never zero, so rejection sampling always converges.
    weights = (1.0 - edge_weight) + edge_weight * (edge_density / 255.0)

    # Step 4: Rejection sampling — generate candidate points uniformly, accept based on weight
    accepted: list[tuple[float, float]] = []
    batch_size = max(num_points * 4, 1000)

    while len(accepted) < num_points:
        # Generate candidates uniformly in mm space
        xs = rng.uniform(img_x1, img_x2, batch_size)
        ys = rng.uniform(img_y1, img_y2, batch_size)

        # Convert mm coords to pixel indices
        px = ((xs - img_x1) / (img_x2 - img_x1) * (img_w - 1)).astype(int)
        py = ((ys - img_y1) / (img_y2 - img_y1) * (img_h - 1)).astype(int)

        # Clamp to valid pixel range
        px = np.clip(px, 0, img_w - 1)
        py = np.clip(py, 0, img_h - 1)

        # Look up weight for each candidate
        w = weights[py, px]

        # Acceptance probability = w / max(w) — max weight is 1.0
        u = rng.uniform(0.0, 1.0, batch_size)
        mask = u <= w

        for i in np.where(mask)[0]:
            accepted.append((float(xs[i]), float(ys[i])))
            if len(accepted) >= num_points:
                break

    # Trim to exactly num_points
    accepted = accepted[:num_points]

    # Step 5: Add the 4 corners of the image rect so triangulation covers the full area
    corners = [
        (img_x1, img_y1),
        (img_x2, img_y1),
        (img_x1, img_y2),
        (img_x2, img_y2),
    ]
    all_points = corners + accepted

    # Step 6: Return as ndarray of shape (N, 2)
    return np.array(all_points, dtype=np.float64)


def _bilinear_sample(gray: np.ndarray, px: float, py: float) -> float:
    """Bilinear interpolation of grayscale pixel value at sub-pixel coordinates."""
    h, w = gray.shape[:2]
    x0 = int(px)
    y0 = int(py)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    fx = px - x0
    fy = py - y0
    v00 = float(gray[y0, x0])
    v10 = float(gray[y0, x1])
    v01 = float(gray[y1, x0])
    v11 = float(gray[y1, x1])
    return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy


def _triangulate_and_sample(
    seeds: np.ndarray,
    gray: np.ndarray,
    img_rect: tuple[float, float, float, float],
) -> list[tuple[list[tuple[float, float]], float]]:
    """Delaunay-triangulate seed points and sample brightness at each centroid.

    Parameters
    ----------
    seeds:
        Array of shape (N, 2) with (x_mm, y_mm) seed coordinates.
    gray:
        Grayscale image array (uint8, shape HxW).
    img_rect:
        Bounding box in mm: (x1, y1, x2, y2).

    Returns
    -------
    list of (verts_mm, brightness) tuples where verts_mm is
    [(x0,y0),(x1,y1),(x2,y2)] in mm and brightness is 0–255.
    """
    from scipy.spatial import Delaunay

    tri = Delaunay(seeds)
    img_x1, img_y1, img_x2, img_y2 = img_rect
    img_h, img_w = gray.shape[:2]
    mm_w = img_x2 - img_x1
    mm_h = img_y2 - img_y1

    result: list[tuple[list[tuple[float, float]], float]] = []
    for simplex in tri.simplices:
        v0 = (float(seeds[simplex[0], 0]), float(seeds[simplex[0], 1]))
        v1 = (float(seeds[simplex[1], 0]), float(seeds[simplex[1], 1]))
        v2 = (float(seeds[simplex[2], 0]), float(seeds[simplex[2], 1]))
        cx = (v0[0] + v1[0] + v2[0]) / 3.0
        cy = (v0[1] + v1[1] + v2[1]) / 3.0
        # Convert mm centroid to pixel coordinates
        px = (cx - img_x1) / mm_w * (img_w - 1)
        py = (cy - img_y1) / mm_h * (img_h - 1)
        brightness = _bilinear_sample(gray, px, py)
        result.append(([v0, v1, v2], brightness))

    return result


def _discard_outside_triangles(
    triangles: list[tuple[list[tuple[float, float]], float]],
    img_rect: tuple[float, float, float, float],
) -> list[tuple[list[tuple[float, float]], float]]:
    """Remove triangles whose centroid falls outside the image rect.

    The Delaunay convex hull can extend beyond the image boundary;
    those hull-edge triangles produce centroids outside the image.
    """
    img_x1, img_y1, img_x2, img_y2 = img_rect
    kept = []
    for verts_mm, brightness in triangles:
        cx = (verts_mm[0][0] + verts_mm[1][0] + verts_mm[2][0]) / 3.0
        cy = (verts_mm[0][1] + verts_mm[1][1] + verts_mm[2][1]) / 3.0
        if img_x1 <= cx <= img_x2 and img_y1 <= cy <= img_y2:
            kept.append((verts_mm, brightness))
    return kept


def _rectangle_cells(
    gray: np.ndarray,
    img_rect: tuple[float, float, float, float],
    cell_size: float,
) -> list[tuple[list[tuple[float, float]], float]]:
    """Generate axis-aligned rectangular cells covering img_rect.

    Each cell is ``cell_size × cell_size`` mm (edge cells are clipped to the
    image rect boundary so they remain axis-aligned quads).

    Parameters
    ----------
    gray:
        Grayscale image array (uint8, shape HxW).
    img_rect:
        Bounding box in mm: (x1, y1, x2, y2).
    cell_size:
        Side length of each cell in mm.

    Returns
    -------
    list of (verts_mm, brightness) tuples where verts_mm is a 4-vertex list
    and brightness is 0–255.
    """
    img_x1, img_y1, img_x2, img_y2 = img_rect
    img_h, img_w = gray.shape[:2]
    mm_w = img_x2 - img_x1
    mm_h = img_y2 - img_y1

    result: list[tuple[list[tuple[float, float]], float]] = []

    y = img_y1
    while y < img_y2:
        y_end = min(y + cell_size, img_y2)
        x = img_x1
        while x < img_x2:
            x_end = min(x + cell_size, img_x2)

            # Sample brightness at cell center
            cx = (x + x_end) / 2.0
            cy = (y + y_end) / 2.0
            px = (cx - img_x1) / mm_w * (img_w - 1)
            py = (cy - img_y1) / mm_h * (img_h - 1)
            brightness = _bilinear_sample(gray, px, py)

            verts: list[tuple[float, float]] = [
                (x, y), (x_end, y), (x_end, y_end), (x, y_end)
            ]
            result.append((verts, brightness))

            x += cell_size
        y += cell_size

    return result


def _hexagon_cells(
    gray: np.ndarray,
    img_rect: tuple[float, float, float, float],
    cell_size: float,
) -> list[tuple[list[tuple[float, float]], float]]:
    """Generate pointy-top hexagonal cells covering img_rect.

    Hex radius = ``cell_size / 2``. Alternate rows are offset by half the
    horizontal spacing. Hexagons that extend beyond the image rect are clipped
    with Shapely.

    Parameters
    ----------
    gray:
        Grayscale image array (uint8, shape HxW).
    img_rect:
        Bounding box in mm: (x1, y1, x2, y2).
    cell_size:
        Nominal diameter of each hexagon in mm; radius = cell_size / 2.

    Returns
    -------
    list of (verts_mm, brightness) tuples where verts_mm is the (clipped)
    hexagon vertices and brightness is 0–255.
    """
    import math

    img_x1, img_y1, img_x2, img_y2 = img_rect
    img_h, img_w = gray.shape[:2]
    mm_w = img_x2 - img_x1
    mm_h = img_y2 - img_y1

    radius = cell_size / 2.0
    # Pointy-top hexagon layout metrics
    col_spacing = math.sqrt(3.0) * radius   # horizontal distance between adjacent centers
    row_spacing = 1.5 * radius              # vertical distance between row centers

    clip_rect = shapely_box(img_x1, img_y1, img_x2, img_y2)

    # How many rows/cols needed to fully cover the image rect
    n_cols = int(math.ceil((img_x2 - img_x1) / col_spacing)) + 3
    n_rows = int(math.ceil((img_y2 - img_y1) / row_spacing)) + 3

    result: list[tuple[list[tuple[float, float]], float]] = []

    for row_idx in range(-1, n_rows + 1):
        cy = img_y1 + row_idx * row_spacing
        # Odd rows are shifted right by half the column spacing
        x_shift = (col_spacing / 2.0) if (row_idx % 2 != 0) else 0.0

        for col_idx in range(-1, n_cols + 1):
            cx = img_x1 + col_idx * col_spacing + x_shift

            # Skip centres clearly outside the image rect
            if cx < img_x1 - radius * 2 or cx > img_x2 + radius * 2:
                continue
            if cy < img_y1 - radius * 2 or cy > img_y2 + radius * 2:
                continue

            # Pointy-top hex vertices: first vertex at 30°, then every 60°
            verts_raw = [
                (
                    cx + radius * math.cos(math.radians(30.0 + 60.0 * i)),
                    cy + radius * math.sin(math.radians(30.0 + 60.0 * i)),
                )
                for i in range(6)
            ]

            hex_poly = ShapelyPolygon(verts_raw)
            if hex_poly.area < _AREA_EPSILON:
                continue
            clipped = hex_poly.intersection(clip_rect)

            if clipped.is_empty or not isinstance(clipped, ShapelyPolygon):
                continue

            cell_coords = list(clipped.exterior.coords)[:-1]
            if len(cell_coords) < 3:
                continue

            # Sample brightness at (clamped) hex centre
            sample_cx = max(img_x1, min(img_x2, cx))
            sample_cy = max(img_y1, min(img_y2, cy))
            px = (sample_cx - img_x1) / mm_w * (img_w - 1)
            py = (sample_cy - img_y1) / mm_h * (img_h - 1)
            brightness = _bilinear_sample(gray, px, py)

            result.append(([(float(x), float(y)) for x, y in cell_coords], brightness))

    return result


def _voronoi_cells(
    seeds: np.ndarray,
    gray: np.ndarray,
    img_rect: tuple[float, float, float, float],
) -> list[tuple[list[tuple[float, float]], float]]:
    """Compute Voronoi cells clipped to img_rect and sample brightness at each centroid.

    Uses the mirror-point boundary approach: reflects all seeds across each of
    the four image rect edges so that all cells for the original seeds are finite
    (no infinite Voronoi rays). Only regions for the original n seed points are
    processed.

    Parameters
    ----------
    seeds:
        Array of shape (N, 2) with (x_mm, y_mm) seed coordinates (includes corners).
    gray:
        Grayscale image array (uint8, shape HxW).
    img_rect:
        Bounding box in mm: (x1, y1, x2, y2).

    Returns
    -------
    list of (verts_mm, brightness) tuples where verts_mm is the clipped cell
    polygon vertices in mm and brightness is 0–255.
    """
    from scipy.spatial import Voronoi

    img_x1, img_y1, img_x2, img_y2 = img_rect
    img_h, img_w = gray.shape[:2]
    mm_w = img_x2 - img_x1
    mm_h = img_y2 - img_y1

    n = len(seeds)
    if n < 4:
        return []

    # Reflect seeds across all four boundaries → 4 extra copies
    reflected = np.vstack([
        seeds,  # original: indices 0 … n-1
        np.column_stack([2.0 * img_x1 - seeds[:, 0], seeds[:, 1]]),  # left
        np.column_stack([2.0 * img_x2 - seeds[:, 0], seeds[:, 1]]),  # right
        np.column_stack([seeds[:, 0], 2.0 * img_y1 - seeds[:, 1]]),  # top
        np.column_stack([seeds[:, 0], 2.0 * img_y2 - seeds[:, 1]]),  # bottom
    ])

    vor = Voronoi(reflected)
    clip_rect = shapely_box(img_x1, img_y1, img_x2, img_y2)

    result: list[tuple[list[tuple[float, float]], float]] = []
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region or len(region) < 3:
            continue

        verts = vor.vertices[region]
        cell_poly = ShapelyPolygon(verts)
        clipped = cell_poly.intersection(clip_rect)

        if clipped.is_empty or not isinstance(clipped, ShapelyPolygon):
            continue

        # Get vertices from clipped polygon (drop closing duplicate)
        cell_coords = list(clipped.exterior.coords)[:-1]
        if len(cell_coords) < 3:
            continue

        # Compute centroid
        cx = clipped.centroid.x
        cy = clipped.centroid.y

        # Convert mm centroid to pixel coordinates and sample brightness
        px = (cx - img_x1) / mm_w * (img_w - 1)
        py = (cy - img_y1) / mm_h * (img_h - 1)
        brightness = _bilinear_sample(gray, px, py)

        result.append(([(float(x), float(y)) for x, y in cell_coords], brightness))

    return result


def _compute_angle_map(gray: np.ndarray, mode: str) -> np.ndarray:
    """Compute per-pixel hatch angle map (degrees) from image gradient.

    Parameters
    ----------
    gray:
        Grayscale image array (uint8, shape HxW).
    mode:
        "Edge Flow" — angle along edges (perpendicular to gradient).
        "Gradient"  — angle of the steepest ascent (along gradient).

    Returns
    -------
    np.ndarray of shape (H, W) with angle in degrees at each pixel.
    """
    try:
        import cv2

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    except ImportError:
        # Fallback: central difference
        gx = np.zeros_like(gray, dtype=np.float64)
        gy = np.zeros_like(gray, dtype=np.float64)
        gx[:, 1:-1] = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
        gy[1:-1, :] = gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64)

    angle = np.degrees(np.arctan2(gy, gx))  # gradient direction

    if mode == "Edge Flow":
        # Rotate 90° to get edge tangent direction
        angle = angle + 90.0

    return angle


@register_generator
class MosaicHatchGenerator(Generator):
    """Delaunay-triangulated or Voronoi-tessellated hatching driven by image brightness."""

    name = "Mosaic Hatching"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="mesh_type",
                label="Mesh Type",
                choices=["Triangles", "Voronoi", "Rectangles", "Hexagons"],
                default="Triangles",
                description="Tessellation method — Triangles, Voronoi, Rectangles, or Hexagons",
            ),
            IntParam(
                name="num_points",
                label="Seed Points",
                min=100,
                max=10000,
                step=100,
                default=1000,
                visible_when={"mesh_type": ["Triangles", "Voronoi"]},
                description="Number of seed points to place across the image — more points produce finer detail",
            ),
            FloatParam(
                name="edge_weight",
                label="Edge Attraction",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.7,
                visible_when={"mesh_type": ["Triangles", "Voronoi"]},
                description="How strongly edges attract mesh vertices — 0 = uniform, 1 = all on edges",
            ),
            FloatParam(
                name="cell_size_mm",
                label="Cell Size (mm)",
                min=1.0,
                max=30.0,
                step=0.5,
                default=5.0,
                visible_when={"mesh_type": ["Rectangles", "Hexagons"]},
                description="Cell size in mm",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before processing (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before processing (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description="Gaussian blur applied before processing — reduces noise",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image before processing",
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
            FloatParam(
                name="min_density",
                label="Min Density",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.0,
                description="Hatching density for brightest areas — 0 = no lines in highlights",
            ),
            FloatParam(
                name="max_density",
                label="Max Density",
                min=1.0,
                max=20.0,
                step=0.5,
                default=6.0,
                description="Hatching density for darkest areas",
            ),
            ChoiceParam(
                name="angle_mode",
                label="Angle Mode",
                choices=["Edge Flow", "Fixed", "Gradient"],
                default="Edge Flow",
                description="How the hatching angle is determined per cell",
            ),
            FloatParam(
                name="fixed_angle_deg",
                label="Fixed Angle (°)",
                min=0.0,
                max=180.0,
                step=1.0,
                default=45.0,
                visible_when={"angle_mode": ["Fixed"]},
                description="Hatch angle in degrees when Angle Mode is Fixed",
            ),
            BoolParam(
                name="cross_hatch",
                label="Cross Hatch",
                default=False,
                description="Add perpendicular cross-hatching in dark cells",
            ),
            FloatParam(
                name="cross_hatch_threshold",
                label="Cross Hatch Threshold",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.3,
                visible_when={"cross_hatch": [True]},
                description="Brightness below which cross-hatching is applied",
            ),
            BoolParam(
                name="draw_edges",
                label="Draw Edges",
                default=False,
                description="Draw mesh edges in addition to hatching — shows the mesh structure",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        _hatch_defaults = {
            "mesh_type": "Triangles",
            "min_density": 0.0,
            "max_density": 6.0,
            "angle_mode": "Edge Flow",
            "fixed_angle_deg": 45.0,
            "cross_hatch": False,
            "cross_hatch_threshold": 0.3,
            "draw_edges": False,
        }
        return [
            Preset(
                name="Default",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 1000,
                    "edge_weight": 0.7,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    **_hatch_defaults,
                },
            ),
            Preset(
                name="Uniform Grid",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 1000,
                    "edge_weight": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    **_hatch_defaults,
                },
            ),
            Preset(
                name="Edge Emphasis",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 2000,
                    "edge_weight": 1.0,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 0.5,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    **_hatch_defaults,
                },
            ),
            Preset(
                name="Cross Hatch Portrait",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 1500,
                    "edge_weight": 0.8,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 8.0,
                    "angle_mode": "Edge Flow",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": True,
                    "cross_hatch_threshold": 0.35,
                    "draw_edges": False,
                },
            ),
            Preset(
                name="Pen & Ink",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 1000,
                    "edge_weight": 0.7,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 6.0,
                    "angle_mode": "Edge Flow",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": False,
                    "cross_hatch_threshold": 0.3,
                    "draw_edges": False,
                },
            ),
            Preset(
                name="Cross-Hatched Portrait",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 1500,
                    "edge_weight": 0.8,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 8.0,
                    "angle_mode": "Edge Flow",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": True,
                    "cross_hatch_threshold": 0.4,
                    "draw_edges": False,
                },
            ),
            Preset(
                name="Geometric Mesh",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 500,
                    "edge_weight": 0.3,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 4.0,
                    "angle_mode": "Fixed",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": False,
                    "cross_hatch_threshold": 0.3,
                    "draw_edges": True,
                },
            ),
            Preset(
                name="Dense Illustration",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 3000,
                    "edge_weight": 0.9,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 10.0,
                    "angle_mode": "Edge Flow",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": True,
                    "cross_hatch_threshold": 0.3,
                    "draw_edges": False,
                },
            ),
            Preset(
                name="Minimal Sketch",
                params={
                    "mesh_type": "Triangles",
                    "num_points": 300,
                    "edge_weight": 0.5,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 3.0,
                    "angle_mode": "Fixed",
                    "fixed_angle_deg": 30.0,
                    "cross_hatch": False,
                    "cross_hatch_threshold": 0.3,
                    "draw_edges": False,
                },
            ),
            Preset(
                name="Voronoi Portrait",
                params={
                    "mesh_type": "Voronoi",
                    "num_points": 800,
                    "edge_weight": 0.8,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 8.0,
                    "angle_mode": "Edge Flow",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": False,
                    "cross_hatch_threshold": 0.3,
                    "draw_edges": False,
                },
            ),
            Preset(
                name="Geometric Grid",
                params={
                    "mesh_type": "Rectangles",
                    "cell_size_mm": 5.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 4.0,
                    "angle_mode": "Fixed",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": False,
                    "cross_hatch_threshold": 0.3,
                    "draw_edges": True,
                },
            ),
            Preset(
                name="Honeycomb",
                params={
                    "mesh_type": "Hexagons",
                    "cell_size_mm": 8.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "min_density": 0.0,
                    "max_density": 5.0,
                    "angle_mode": "Edge Flow",
                    "fixed_angle_deg": 45.0,
                    "cross_hatch": False,
                    "cross_hatch_threshold": 0.3,
                    "draw_edges": True,
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
        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        # Apply preprocessing
        img = source.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            img = adjust_brightness(img, brightness)
        if contrast != 0.0:
            img = adjust_contrast(img, contrast)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        if do_invert:
            img = invert_image(img)

        # Convert to grayscale
        if img.ndim == 3:
            try:
                import cv2
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except ImportError:
                gray = img.mean(axis=2).astype(np.uint8)
        else:
            gray = img

        if progress_callback:
            progress_callback(10)

        img_h, img_w = gray.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_rect = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        mesh_type = str(params.get("mesh_type", "Triangles"))
        cell_size = float(params.get("cell_size_mm", 5.0))

        if progress_callback:
            progress_callback(20)

        # Build the cell list: list of (verts_mm, brightness)
        if mesh_type == "Rectangles":
            cells = _rectangle_cells(gray, img_rect, cell_size)
            if progress_callback:
                progress_callback(60)
        elif mesh_type == "Hexagons":
            cells = _hexagon_cells(gray, img_rect, cell_size)
            if progress_callback:
                progress_callback(60)
        else:
            num_points = int(params.get("num_points", 1000))
            edge_weight = float(params.get("edge_weight", 0.7))
            seed = int(params.get("random_seed", 42))
            rng = np.random.default_rng(seed)

            _seeds = _edge_aware_seeds(gray, img_rect, num_points, edge_weight, rng)

            if progress_callback:
                progress_callback(50)

            if mesh_type == "Voronoi":
                cells = _voronoi_cells(_seeds, gray, img_rect)
            else:
                # Triangles (default)
                cells = _triangulate_and_sample(_seeds, gray, img_rect)
                cells = _discard_outside_triangles(cells, img_rect)

            if progress_callback:
                progress_callback(60)

        # --- Hatching parameters ---
        min_density = float(params.get("min_density", 0.0))
        max_density = float(params.get("max_density", 6.0))
        angle_mode = str(params.get("angle_mode", "Edge Flow"))
        fixed_angle_deg = float(params.get("fixed_angle_deg", 45.0))
        do_cross_hatch = bool(params.get("cross_hatch", False))
        cross_hatch_threshold = float(params.get("cross_hatch_threshold", 0.3))
        draw_edges = bool(params.get("draw_edges", False))

        # Precompute angle map for Edge Flow / Gradient modes
        angle_map: np.ndarray | None = None
        if angle_mode in ("Edge Flow", "Gradient"):
            angle_map = _compute_angle_map(gray, angle_mode)

        img_x1, img_y1, img_x2, img_y2 = img_rect
        mm_w = img_x2 - img_x1
        mm_h = img_y2 - img_y1

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        all_polylines: list[Polyline] = []
        n_cells = len(cells)

        for i, (verts_mm, brightness) in enumerate(cells):
            if cancelled_callback and cancelled_callback():
                break

            # Compute density: dark → dense, bright → sparse.
            brightness_int = max(0.0, min(255.0, round(brightness)))
            density = min_density + (1.0 - brightness_int / 255.0) * (max_density - min_density)

            if density <= 0.0:
                continue

            # Compute hatch angle
            if angle_mode == "Fixed":
                angle_deg = fixed_angle_deg
            else:
                # Sample angle map at cell centroid
                n_verts = len(verts_mm)
                cx = sum(v[0] for v in verts_mm) / n_verts
                cy = sum(v[1] for v in verts_mm) / n_verts
                # Convert mm centroid to pixel coordinates
                px = (cx - img_x1) / mm_w * (img_w - 1)
                py = (cy - img_y1) / mm_h * (img_h - 1)
                px = max(0.0, min(float(img_w - 1), px))
                py = max(0.0, min(float(img_h - 1), py))
                angle_deg = float(angle_map[int(py), int(px)])  # type: ignore[index]

            # Determine whether to cross-hatch this cell
            apply_cross = do_cross_hatch and (brightness_int / 255.0) < cross_hatch_threshold

            # Build Shapely polygon and hatch it
            poly = ShapelyPolygon(verts_mm)
            if poly.area < _AREA_EPSILON:
                continue

            spacing = 1.0 / density
            lines = _hatch_polygon(poly, spacing, angle_deg)
            if apply_cross:
                lines.extend(_hatch_polygon(poly, spacing, angle_deg + 90.0))

            for line in lines:
                shifted = [(x + x_off, y + y_off) for x, y in line]
                all_polylines.append(shifted)

            # Progress 60–100 over cell loop
            if progress_callback and (i % max(1, n_cells // 20) == 0):
                pct = 60 + int(40 * i / max(1, n_cells))
                progress_callback(pct)

        if draw_edges:
            seen_edges: set[frozenset] = set()
            for verts_mm, _ in cells:
                n_verts = len(verts_mm)
                for j in range(n_verts):
                    p_a = verts_mm[j]
                    p_b = verts_mm[(j + 1) % n_verts]
                    r_a = (round(p_a[0], 4), round(p_a[1], 4))
                    r_b = (round(p_b[0], 4), round(p_b[1], 4))
                    key = frozenset((r_a, r_b))
                    if key not in seen_edges:
                        seen_edges.add(key)
                        all_polylines.append([
                            (p_a[0] + x_off, p_a[1] + y_off),
                            (p_b[0] + x_off, p_b[1] + y_off),
                        ])

        if progress_callback:
            progress_callback(100)

        return all_polylines
