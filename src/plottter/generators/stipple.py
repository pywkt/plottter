"""StippleGenerator — weighted Voronoi stippling with optional TSP path connection.

Supports two stippling algorithms:
- Lloyd: iterative weighted Voronoi relaxation (stable, well-tested)
- LBG: Linde-Buzo-Gray dynamic split/merge (converges faster, better
  blue-noise distribution)

Supports structure-aware halftoning (Pang et al., SIGGRAPH 2008):
- Computes an edge importance map from Sobel gradients once before iteration
- Adds edge-weighted bias to pixel weights so points cluster near edges
- Blends between pure tonal optimization (structure_weight=0) and strong
  edge preservation (structure_weight=1)
"""

from __future__ import annotations

import math
import random as _random
from typing import Any

import numpy as np

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
from plottter.generators._helpers import compute_image_rect
from plottter.models import Canvas, Polyline

# Number of points for tiny-circle approximation
_DOT_SIDES = 8
_DOT_RADIUS_MM = 0.15


def _tiny_circle(cx: float, cy: float, radius_mm: float = _DOT_RADIUS_MM) -> Polyline:
    """Return a small circular polyline to represent a stipple dot."""
    pts: Polyline = []
    for k in range(_DOT_SIDES + 1):
        angle = 2.0 * math.pi * k / _DOT_SIDES
        pts.append((cx + radius_mm * math.cos(angle), cy + radius_mm * math.sin(angle)))
    return pts


def _nearest_neighbor_tsp(points: np.ndarray) -> list[int]:
    """Nearest-neighbor TSP heuristic; KDTree-accelerated when scipy is available.

    Uses a KDTree to query the k=20 nearest candidates at each step, falling
    back to brute-force only when all k candidates are already visited (rare).
    This is O(n k log n) vs O(n²) for the pure brute-force approach.
    """
    n = len(points)
    if n == 0:
        return []

    try:
        from scipy.spatial import cKDTree
        return _kd_nearest_neighbor_tsp(points, cKDTree)
    except ImportError:
        pass

    # Fallback: O(n²) brute-force
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = order[-1]
        best_dist = float("inf")
        best_j = -1
        px, py = points[last]
        for j in range(n):
            if visited[j]:
                continue
            dx = px - points[j, 0]
            dy = py - points[j, 1]
            d = dx * dx + dy * dy
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j == -1:
            break
        order.append(best_j)
        visited[best_j] = True
    return order


def _kd_nearest_neighbor_tsp(points: np.ndarray, cKDTree: Any) -> list[int]:
    """KDTree-based nearest-neighbor TSP; expected O(n k log n)."""
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    tree = cKDTree(points)

    for _ in range(n - 1):
        remaining_count = n - len(order)
        last = order[-1]
        # Query enough candidates to likely find an unvisited one.
        # +1 accounts for the current (visited) point potentially being returned.
        k = min(21, remaining_count + 1)
        _, idxs = tree.query(points[last], k=k)
        if np.ndim(idxs) == 0:
            idxs = [int(idxs)]

        best_j = -1
        for idx in idxs:
            idx_int = int(idx)
            if not visited[idx_int]:
                best_j = idx_int
                break

        if best_j == -1:
            # All k candidates are visited — brute-force for this one step
            px, py = points[last]
            best_dist = float("inf")
            for j in range(n):
                if visited[j]:
                    continue
                dx = px - points[j, 0]
                dy = py - points[j, 1]
                d = dx * dx + dy * dy
                if d < best_dist:
                    best_dist = d
                    best_j = j

        if best_j == -1:
            break
        order.append(best_j)
        visited[best_j] = True

    return order


def _weighted_sample_initial_points(
    img: np.ndarray,
    n: int,
    rng: _random.Random,
) -> np.ndarray:
    """Sample n points from the image weighted by darkness (inverse brightness)."""
    h, w = img.shape[:2]
    # Weight: 1 - brightness/255, so dark pixels get more weight
    weights = (255 - img.astype(np.float64)).flatten()
    total = weights.sum()
    if total <= 0:
        # Uniform sampling fallback
        weights = np.ones(h * w, dtype=np.float64)
        total = float(h * w)

    # Normalize to probabilities
    probs = weights / total

    # Use numpy random for performance
    flat_indices = np.random.default_rng(rng.randint(0, 2**31)).choice(
        h * w, size=n, replace=True, p=probs
    )
    rows = flat_indices // w
    cols = flat_indices % w

    # Add small random jitter to avoid all points snapping to exact pixel centers
    jitter_r = np.random.default_rng(rng.randint(0, 2**31)).uniform(-0.5, 0.5, size=n)
    jitter_c = np.random.default_rng(rng.randint(0, 2**31)).uniform(-0.5, 0.5, size=n)

    points = np.column_stack([
        (cols.astype(np.float64) + jitter_c),
        (rows.astype(np.float64) + jitter_r),
    ])
    # Clamp to image bounds
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    return points


def _compute_edge_weight_map(img: np.ndarray, structure_weight: float) -> np.ndarray:
    """Compute an edge importance map scaled by structure_weight.

    Detects edges using Sobel gradients (cv2) or numpy gradient as fallback.
    Returns an (H, W) array with values in [0, structure_weight] that can be
    added to the base brightness-derived pixel weights.  Pixels near strong
    edges receive higher weight, biasing Lloyd/LBG to place more stipple
    points there.

    Reference: Pang et al., "Structure-Aware Halftoning", SIGGRAPH 2008.
    """
    if structure_weight <= 0.0:
        return np.zeros(img.shape[:2], dtype=np.float64)

    gray = img.astype(np.float64)
    try:
        import cv2
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    except ImportError:
        # Pure numpy fallback
        gy = np.gradient(gray, axis=0)
        gx = np.gradient(gray, axis=1)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

    max_mag = float(magnitude.max())
    if max_mag > 0.0:
        magnitude = magnitude / max_mag  # normalise to [0, 1]

    return (magnitude * structure_weight).astype(np.float64)


def _lloyd_simple(
    points: np.ndarray,
    img: np.ndarray,
    iterations: int,
    min_dot_spacing_px: float,
    cancelled_callback: Any,
    progress_callback: Any,
    working_resolution: int = 400,
    convergence_threshold: float = 0.5,
    structure_weight_map: "np.ndarray | None" = None,
) -> np.ndarray:
    """Optimised Lloyd relaxation using a downsampled working image and vectorised centroids.

    Key optimisations vs the original implementation:

    1. **Downsampled working image** — Lloyd iterations run on an image whose
       largest dimension is capped at *working_resolution* pixels (default 400).
       For a 2000×2000 source this gives a ~25× reduction in the pixel-grid
       size and therefore in KDTree query cost.  Final point coordinates are
       scaled back to the original image resolution after convergence.

    2. **Vectorised centroid computation** — replaces the per-point Python loop
       with three ``np.bincount`` calls (weighted x-sum, weighted y-sum, weight
       sum), yielding a single vectorised pass over all pixels per iteration.

    3. **Early stopping** — after each iteration the mean displacement of all
       points is compared against *convergence_threshold* (in working-resolution
       pixels).  When displacement falls below the threshold, iteration stops
       early.  Set to 0.0 to disable.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for stippling. "
            "Install with: pip install scipy"
        ) from exc

    orig_h, orig_w = img.shape[:2]

    # --- Downsample image for Lloyd iterations ---
    work_img = img
    work_h, work_w = orig_h, orig_w
    scale_x = scale_y = 1.0

    if working_resolution > 0 and max(orig_h, orig_w) > working_resolution:
        try:
            import cv2
            scale = working_resolution / max(orig_h, orig_w)
            work_w = max(1, int(orig_w * scale))
            work_h = max(1, int(orig_h * scale))
            work_img = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            # cv2 unavailable — keep full resolution
            work_h, work_w = orig_h, orig_w
            work_img = img

        scale_x = work_w / orig_w
        scale_y = work_h / orig_h

    # Scale initial points to working-resolution coordinates
    work_points = points.copy()
    work_points[:, 0] = np.clip(work_points[:, 0] * scale_x, 0.0, work_w - 1)
    work_points[:, 1] = np.clip(work_points[:, 1] * scale_y, 0.0, work_h - 1)

    # Build pixel grid (in working resolution)
    ys, xs = np.mgrid[0:work_h, 0:work_w]
    pixel_coords_x = xs.ravel().astype(np.float64)
    pixel_coords_y = ys.ravel().astype(np.float64)
    pixel_weights = (255.0 - work_img.ravel().astype(np.float64)) / 255.0

    # --- Structure-aware edge bias (Pang et al. 2008) ---
    if structure_weight_map is not None and structure_weight_map.size > 0:
        # Downsample edge map to working resolution to match pixel grid
        if structure_weight_map.shape != (work_h, work_w):
            try:
                import cv2 as _cv2
                work_edge = _cv2.resize(
                    structure_weight_map.astype(np.float32),
                    (work_w, work_h),
                    interpolation=_cv2.INTER_AREA,
                ).astype(np.float64)
            except ImportError:
                # Cannot resize without cv2; skip edge bias
                work_edge = None
        else:
            work_edge = structure_weight_map.astype(np.float64)
        if work_edge is not None:
            pixel_weights = pixel_weights + work_edge.ravel()

    n = len(work_points)

    # --- Lloyd iteration loop ---
    for it in range(iterations):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback:
            progress_callback(int(it / iterations * 80))

        prev_points = work_points.copy()

        # Assign each pixel to its nearest stipple point
        tree = cKDTree(work_points)
        _, assignments = tree.query(
            np.column_stack([pixel_coords_x, pixel_coords_y])
        )

        # Vectorised weighted centroid: three bincount calls, one pass over pixels
        wx = np.bincount(
            assignments, weights=pixel_weights * pixel_coords_x, minlength=n
        )
        wy = np.bincount(
            assignments, weights=pixel_weights * pixel_coords_y, minlength=n
        )
        wt = np.bincount(assignments, weights=pixel_weights, minlength=n)

        new_points = work_points.copy()
        valid = wt > 0
        new_points[valid, 0] = wx[valid] / wt[valid]
        new_points[valid, 1] = wy[valid] / wt[valid]

        work_points = new_points
        work_points[:, 0] = np.clip(work_points[:, 0], 0.0, work_w - 1)
        work_points[:, 1] = np.clip(work_points[:, 1], 0.0, work_h - 1)

        # Early-stopping convergence check
        if convergence_threshold > 0.0:
            displacements = np.sqrt(np.sum((work_points - prev_points) ** 2, axis=1))
            if float(displacements.mean()) < convergence_threshold:
                if progress_callback:
                    progress_callback(80)
                break

    # Scale points back to original-image coordinates
    work_points[:, 0] /= scale_x if scale_x > 0.0 else 1.0
    work_points[:, 1] /= scale_y if scale_y > 0.0 else 1.0
    work_points[:, 0] = np.clip(work_points[:, 0], 0.0, orig_w - 1)
    work_points[:, 1] = np.clip(work_points[:, 1], 0.0, orig_h - 1)

    # Filter out points that are too close to each other
    if min_dot_spacing_px > 0 and len(work_points) > 1:
        keep = np.ones(len(work_points), dtype=bool)
        tree = cKDTree(work_points)
        pairs = tree.query_pairs(min_dot_spacing_px)
        for i, j in sorted(pairs):
            if keep[i] and keep[j]:
                keep[j] = False
        work_points = work_points[keep]

    return work_points


def _uniform_grid_initial_points(img: np.ndarray, n: int) -> np.ndarray:
    """Generate n points on a uniform grid across the image dimensions."""
    h, w = img.shape[:2]
    aspect = w / max(h, 1)
    cols = max(1, int(math.sqrt(n * aspect)))
    rows = max(1, math.ceil(n / cols))
    points: list[list[float]] = []
    for r in range(rows):
        for c in range(cols):
            if len(points) >= n:
                break
            x = (c + 0.5) * w / cols
            y = (r + 0.5) * h / rows
            points.append([x, y])
    arr = np.array(points[:n], dtype=np.float64)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, h - 1)
    return arr


def _few_seeds_initial_points(
    img: np.ndarray,
    n: int,
    rng: _random.Random,
    seed_count: int | None = None,
) -> np.ndarray:
    """Sample a small number of seed points to let LBG grow to target count.

    ``seed_count`` defaults to max(10, n // 20).
    """
    if seed_count is None:
        seed_count = max(10, n // 20)
    return _weighted_sample_initial_points(img, seed_count, rng)


def _lbg_stipple(
    points: np.ndarray,
    img: np.ndarray,
    iterations: int,
    min_dot_spacing_px: float,
    cancelled_callback: Any,
    progress_callback: Any,
    working_resolution: int = 400,
    convergence_threshold: float = 0.5,
    split_threshold: float = 1.5,
    merge_threshold: float = 0.5,
    rng: _random.Random | None = None,
    num_points: int | None = None,
    structure_weight_map: "np.ndarray | None" = None,
) -> np.ndarray:
    """LBG (Linde-Buzo-Gray) stippling: dynamic split/merge of Voronoi cells.

    Key difference from Lloyd's relaxation: instead of a fixed number of
    points that are iteratively repositioned, LBG dynamically splits cells
    whose weighted area exceeds ``split_threshold × target_area`` and merges
    cells whose area falls below ``merge_threshold × target_area``.

    The point population fluctuates toward the target ``num_points``, with
    convergence reached when no splits or merges occur in an iteration.
    Typically converges in 10-20 iterations vs 30-50 for Lloyd.

    Applies the same downsampling optimisation as ``_lloyd_simple``.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for stippling. "
            "Install with: pip install scipy"
        ) from exc

    if rng is None:
        rng = _random.Random(42)

    orig_h, orig_w = img.shape[:2]

    # --- Downsample image for LBG iterations (same as lloyd_simple) ---
    work_img = img
    work_h, work_w = orig_h, orig_w
    scale_x = scale_y = 1.0

    if working_resolution > 0 and max(orig_h, orig_w) > working_resolution:
        try:
            import cv2
            scale = working_resolution / max(orig_h, orig_w)
            work_w = max(1, int(orig_w * scale))
            work_h = max(1, int(orig_h * scale))
            work_img = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            work_h, work_w = orig_h, orig_w
            work_img = img

        scale_x = work_w / orig_w
        scale_y = work_h / orig_h

    # Scale initial points to working-resolution coordinates
    work_points = points.copy()
    work_points[:, 0] = np.clip(work_points[:, 0] * scale_x, 0.0, work_w - 1)
    work_points[:, 1] = np.clip(work_points[:, 1] * scale_y, 0.0, work_h - 1)

    # Build pixel grid (working resolution)
    ys, xs = np.mgrid[0:work_h, 0:work_w]
    pixel_coords_x = xs.ravel().astype(np.float64)
    pixel_coords_y = ys.ravel().astype(np.float64)
    pixel_weights = (255.0 - work_img.ravel().astype(np.float64)) / 255.0

    # --- Structure-aware edge bias (Pang et al. 2008) ---
    if structure_weight_map is not None and structure_weight_map.size > 0:
        if structure_weight_map.shape != (work_h, work_w):
            try:
                import cv2 as _cv2
                work_edge = _cv2.resize(
                    structure_weight_map.astype(np.float32),
                    (work_w, work_h),
                    interpolation=_cv2.INTER_AREA,
                ).astype(np.float64)
            except ImportError:
                work_edge = None
        else:
            work_edge = structure_weight_map.astype(np.float64)
        if work_edge is not None:
            pixel_weights = pixel_weights + work_edge.ravel()

    total_weight = float(pixel_weights.sum())
    # Use the requested num_points for target_area so the algorithm grows
    # toward the true target regardless of how many seed points were used.
    target_points = num_points if num_points is not None else len(work_points)
    target_area = total_weight / target_points if target_points > 0 else 1.0
    # Small jitter magnitude for split displacement (in working-resolution px)
    jitter_mag = max(0.5, max(work_w, work_h) * 0.005)

    for it in range(iterations):
        if cancelled_callback and cancelled_callback():
            break
        if progress_callback:
            progress_callback(int(it / iterations * 80))

        n = len(work_points)
        if n == 0:
            break

        # Assign each pixel to its nearest stipple point
        tree = cKDTree(work_points)
        _, assignments = tree.query(
            np.column_stack([pixel_coords_x, pixel_coords_y])
        )

        # Vectorised weighted centroid (same as lloyd_simple)
        wx = np.bincount(assignments, weights=pixel_weights * pixel_coords_x, minlength=n)
        wy = np.bincount(assignments, weights=pixel_weights * pixel_coords_y, minlength=n)
        wt = np.bincount(assignments, weights=pixel_weights, minlength=n)

        new_points = work_points.copy()
        valid = wt > 0
        new_points[valid, 0] = wx[valid] / wt[valid]
        new_points[valid, 1] = wy[valid] / wt[valid]

        # --- Split / merge based on weighted area ---
        n_splits = 0
        n_merges = 0
        keep_indices: list[int] = []
        extra_points: list[list[float]] = []

        for i in range(n):
            area = float(wt[i])
            if area > split_threshold * target_area:
                # Keep this point and add a new sibling near its centroid
                keep_indices.append(i)
                cx = float(new_points[i, 0])
                cy = float(new_points[i, 1])
                nx = float(np.clip(
                    cx + rng.uniform(-jitter_mag, jitter_mag), 0.0, work_w - 1
                ))
                ny = float(np.clip(
                    cy + rng.uniform(-jitter_mag, jitter_mag), 0.0, work_h - 1
                ))
                extra_points.append([nx, ny])
                n_splits += 1
            elif area < merge_threshold * target_area:
                # Discard this point (merge into neighbours implicitly)
                n_merges += 1
            else:
                keep_indices.append(i)

        # Rebuild point array
        if keep_indices:
            kept = new_points[keep_indices]
        else:
            kept = np.empty((0, 2), dtype=np.float64)

        if extra_points:
            extras = np.array(extra_points, dtype=np.float64)
            work_points = (
                np.vstack([kept, extras]) if len(kept) > 0 else extras
            )
        elif len(kept) > 0:
            work_points = kept
        else:
            # All points merged away — reseed with a handful of random points
            reseed_n = max(1, target_points // 10)
            work_points = np.array(
                [
                    [rng.uniform(0, work_w - 1), rng.uniform(0, work_h - 1)]
                    for _ in range(reseed_n)
                ],
                dtype=np.float64,
            )

        work_points[:, 0] = np.clip(work_points[:, 0], 0.0, work_w - 1)
        work_points[:, 1] = np.clip(work_points[:, 1], 0.0, work_h - 1)

        # Early stopping: converged when no structural changes occurred
        if n_splits == 0 and n_merges == 0:
            if progress_callback:
                progress_callback(80)
            break

    # Scale back to original-image coordinates
    if scale_x > 0.0:
        work_points[:, 0] /= scale_x
    if scale_y > 0.0:
        work_points[:, 1] /= scale_y
    work_points[:, 0] = np.clip(work_points[:, 0], 0.0, orig_w - 1)
    work_points[:, 1] = np.clip(work_points[:, 1], 0.0, orig_h - 1)

    # Filter out points that are too close to each other
    if min_dot_spacing_px > 0 and len(work_points) > 1:
        keep = np.ones(len(work_points), dtype=bool)
        tree = cKDTree(work_points)
        pairs = tree.query_pairs(min_dot_spacing_px)
        for i, j in sorted(pairs):
            if keep[i] and keep[j]:
                keep[j] = False
        work_points = work_points[keep]

    return work_points


@register_generator
class StippleGenerator(Generator):
    """Weighted Voronoi stippling with optional TSP path connection."""

    name = "Stipple"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="algorithm",
                label="Algorithm",
                choices=["Lloyd", "LBG"],
                default="Lloyd",
                description=(
                    "Stippling algorithm — Lloyd: iterative weighted Voronoi relaxation "
                    "(stable, predictable point count); LBG: Linde-Buzo-Gray dynamic "
                    "split/merge (converges faster, better blue-noise distribution)"
                ),
            ),
            IntParam(
                name="num_points",
                label="Number of Points",
                min=100,
                max=50000,
                step=100,
                default=5000,
                description=(
                    "Number of stipple dots — more dots give a finer, more detailed result. "
                    "For LBG this is a target; the final count may vary by ±10%."
                ),
            ),
            IntParam(
                name="iterations",
                label="Relaxation Iterations",
                min=1,
                max=100,
                step=1,
                default=30,
                description=(
                    "Maximum iterations — more iterations produce more evenly-distributed "
                    "dots weighted by image brightness. LBG typically converges in 10-20 "
                    "iterations and stops early when no splits/merges occur."
                ),
            ),
            ChoiceParam(
                name="render_mode",
                label="Render Mode",
                choices=["Dots", "TSP Path"],
                default="Dots",
                description=(
                    "Output mode — 'Dots': draw each stipple point as a small circle; "
                    "'TSP Path': connect all stipple points into a single continuous path "
                    "using nearest-neighbor TSP with optional 2-opt improvement."
                ),
            ),
            BoolParam(
                name="tsp_optimize",
                label="2-opt Optimization",
                default=True,
                description="Apply 2-opt optimization to reduce travel distance of the TSP path",
                visible_when={"render_mode": ["TSP Path"]},
            ),
            BoolParam(
                name="connect_tsp",
                label="Connect via TSP Path (legacy)",
                default=False,
                description="Connect all dots into a single continuous path using a nearest-neighbor TSP approximation",
            ),
            FloatParam(
                name="min_dot_spacing_mm",
                label="Min Dot Spacing (mm)",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.5,
                description="Minimum distance between dots in millimeters — prevents dots from overlapping",
            ),
            IntParam(
                name="seed",
                label="Random Seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for reproducible initial dot placement",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image (dots concentrate in bright areas instead of dark)",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before stippling (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before stippling (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description="Gaussian blur applied before stippling — smooths the brightness distribution, producing more gradual dot density transitions",
            ),
            IntParam(
                name="working_resolution",
                label="Working Resolution",
                min=100,
                max=2000,
                step=50,
                default=400,
                description=(
                    "Maximum dimension (px) of the downsampled image used during "
                    "iterations — lower values are much faster but may lose fine detail. "
                    "Set to a value larger than your image to disable downsampling."
                ),
            ),
            FloatParam(
                name="convergence_threshold",
                label="Convergence Threshold",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.5,
                description=(
                    "For Lloyd: stop when average point movement (working-resolution px) "
                    "falls below this value (0 disables). "
                    "For LBG: early stopping always triggers when no splits/merges occur."
                ),
            ),
            FloatParam(
                name="split_threshold",
                label="Split Threshold",
                min=1.1,
                max=3.0,
                step=0.1,
                default=1.5,
                description=(
                    "LBG only — cells whose weighted area exceeds this multiple of the "
                    "target area are split into two (adds points in dark areas). "
                    "Lower values split more aggressively."
                ),
                visible_when={"algorithm": ["LBG"]},
            ),
            FloatParam(
                name="merge_threshold",
                label="Merge Threshold",
                min=0.1,
                max=0.9,
                step=0.05,
                default=0.5,
                description=(
                    "LBG only — cells whose weighted area falls below this fraction of "
                    "the target area are removed (removes points from bright areas). "
                    "Higher values merge more aggressively."
                ),
                visible_when={"algorithm": ["LBG"]},
            ),
            ChoiceParam(
                name="initial_distribution",
                label="Initial Distribution",
                choices=["Weighted Random", "Uniform Grid", "Few Seeds"],
                default="Weighted Random",
                description=(
                    "LBG only — starting point distribution. "
                    "'Weighted Random': sample from dark areas (default). "
                    "'Uniform Grid': evenly spaced grid, LBG redistributes. "
                    "'Few Seeds': start sparse and let LBG grow to target."
                ),
                visible_when={"algorithm": ["LBG"]},
            ),
            BoolParam(
                name="structure_aware",
                label="Structure-Aware Halftoning",
                default=False,
                description=(
                    "Bias stipple placement toward image edges, preserving structural "
                    "features like sharp edges, hair, text, and glasses. "
                    "Implements the edge-weighting approach from Pang et al. (SIGGRAPH 2008). "
                    "When enabled, more dots are placed near strong edges across all iterations."
                ),
            ),
            FloatParam(
                name="structure_weight",
                label="Structure Weight",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.3,
                description=(
                    "Controls the balance between tonal accuracy and edge preservation — "
                    "0.0 = pure tonal (standard stippling), 1.0 = strongly favours edge fidelity. "
                    "Values around 0.2–0.4 give a good balance."
                ),
                visible_when={"structure_aware": [True]},
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
        # Shared defaults for all presets (Lloyd algorithm)
        _shared = {
            "algorithm": "Lloyd",
            "working_resolution": 400,
            "convergence_threshold": 0.5,
            "split_threshold": 1.5,
            "merge_threshold": 0.5,
            "initial_distribution": "Weighted Random",
            "structure_aware": False,
            "structure_weight": 0.3,
        }
        return [
            Preset(
                name="Default Stipple",
                params={
                    "num_points": 5000,
                    "iterations": 30,
                    "render_mode": "Dots",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.5,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    **_shared,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Dense Stipple",
                params={
                    "num_points": 15000,
                    "iterations": 30,
                    "render_mode": "Dots",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.3,
                    "seed": 0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    **_shared,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="TSP Art",
                params={
                    "num_points": 5000,
                    "iterations": 30,
                    "render_mode": "TSP Path",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.5,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    **_shared,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Quick Preview",
                params={
                    # Low point count and few iterations for rapid visual feedback;
                    # useful while tuning preprocessing controls before committing
                    # to a high-quality render.
                    "num_points": 1000,
                    "iterations": 5,
                    "render_mode": "Dots",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.5,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    **_shared,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Portrait Photo",
                params={
                    # High point count with generous min_dot_spacing so dots
                    # cluster naturally in shadow areas of facial features;
                    # contrast boost separates highlights from shadows.
                    "num_points": 8000,
                    "iterations": 40,
                    "render_mode": "Dots",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.4,
                    "seed": 7,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.5,
                    **_shared,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="LBG Default",
                params={
                    # LBG algorithm with default settings — faster convergence
                    # and better blue-noise distribution than Lloyd.
                    "num_points": 5000,
                    "iterations": 30,
                    "render_mode": "Dots",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.5,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "algorithm": "LBG",
                    "working_resolution": 400,
                    "convergence_threshold": 0.5,
                    "split_threshold": 1.5,
                    "merge_threshold": 0.5,
                    "initial_distribution": "Weighted Random",
                    "structure_aware": False,
                    "structure_weight": 0.3,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="LBG Few Seeds",
                params={
                    # Start sparse and let LBG grow to target — good for
                    # images with large bright/dark regions.
                    "num_points": 5000,
                    "iterations": 40,
                    "render_mode": "Dots",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.5,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "algorithm": "LBG",
                    "working_resolution": 400,
                    "convergence_threshold": 0.5,
                    "split_threshold": 1.5,
                    "merge_threshold": 0.5,
                    "initial_distribution": "Few Seeds",
                    "structure_aware": False,
                    "structure_weight": 0.3,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="LBG TSP Art",
                params={
                    # LBG stipple with TSP path connection.
                    "num_points": 5000,
                    "iterations": 30,
                    "render_mode": "TSP Path",
                    "tsp_optimize": True,
                    "connect_tsp": False,
                    "min_dot_spacing_mm": 0.5,
                    "seed": 42,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "algorithm": "LBG",
                    "working_resolution": 400,
                    "convergence_threshold": 0.5,
                    "split_threshold": 1.5,
                    "merge_threshold": 0.5,
                    "initial_distribution": "Weighted Random",
                    "structure_aware": False,
                    "structure_weight": 0.3,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
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
        source = img

        # Ensure grayscale
        if source.ndim == 3:
            try:
                import cv2
                source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            except ImportError:
                source = source.mean(axis=2).astype(np.uint8)

        algorithm = str(params.get("algorithm", "Lloyd"))
        num_points = int(params.get("num_points", 5000))
        iterations = int(params.get("iterations", 30))
        render_mode = str(params.get("render_mode", "Dots"))
        tsp_optimize = bool(params.get("tsp_optimize", True))
        connect_tsp = bool(params.get("connect_tsp", False))
        min_dot_spacing_mm = float(params.get("min_dot_spacing_mm", 0.5))
        seed = int(params.get("seed", 42))
        working_resolution = int(params.get("working_resolution", 400))
        convergence_threshold = float(params.get("convergence_threshold", 0.5))
        split_threshold = float(params.get("split_threshold", 1.5))
        merge_threshold = float(params.get("merge_threshold", 0.5))
        initial_distribution = str(params.get("initial_distribution", "Weighted Random"))
        structure_aware = bool(params.get("structure_aware", False))
        structure_weight = float(params.get("structure_weight", 0.3))

        img_h, img_w = source.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()

        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        img_rect_w = img_x2 - img_x1
        img_rect_h = img_y2 - img_y1

        # Convert mm spacing to pixels (based on image rect, not full canvas)
        px_per_mm = img_w / img_rect_w if img_rect_w > 0 else img_w / (draw_x2 - draw_x1)
        min_dot_spacing_px = min_dot_spacing_mm * px_per_mm

        rng = _random.Random(seed)

        # --- Compute structure-aware edge weight map (once, before iterations) ---
        # Implements Pang et al. (SIGGRAPH 2008) simplified approach: detect edges
        # using Sobel, bias pixel weights permanently so Lloyd/LBG places more
        # stipple points near strong edges throughout all iterations.
        structure_weight_map: np.ndarray | None = None
        if structure_aware and structure_weight > 0.0:
            structure_weight_map = _compute_edge_weight_map(source, structure_weight)

        if progress_callback:
            progress_callback(5)

        # --- Initial point distribution ---
        if algorithm == "LBG":
            if initial_distribution == "Uniform Grid":
                points = _uniform_grid_initial_points(source, num_points)
            elif initial_distribution == "Few Seeds":
                points = _few_seeds_initial_points(source, num_points, rng)
            else:
                # "Weighted Random" (default)
                points = _weighted_sample_initial_points(source, num_points, rng)
        else:
            # Lloyd always uses weighted random sampling
            points = _weighted_sample_initial_points(source, num_points, rng)

        if progress_callback:
            progress_callback(10)

        if cancelled_callback and cancelled_callback():
            return []

        # --- Run the selected algorithm ---
        if algorithm == "LBG":
            points = _lbg_stipple(
                points, source, iterations, min_dot_spacing_px,
                cancelled_callback, progress_callback,
                working_resolution=working_resolution,
                convergence_threshold=convergence_threshold,
                split_threshold=split_threshold,
                merge_threshold=merge_threshold,
                rng=rng,
                num_points=num_points,
                structure_weight_map=structure_weight_map,
            )
        else:
            # Lloyd relaxation (with downsampling + vectorised centroids + early stopping)
            points = _lloyd_simple(
                points, source, iterations, min_dot_spacing_px,
                cancelled_callback, progress_callback,
                working_resolution=working_resolution,
                convergence_threshold=convergence_threshold,
                structure_weight_map=structure_weight_map,
            )

        if progress_callback:
            progress_callback(85)

        # Convert pixel coords to mm (mapped to the image rect, not full canvas)
        def px_to_mm(px: float, py: float) -> tuple[float, float]:
            x = img_x1 + px / img_w * img_rect_w
            y = img_y1 + py / img_h * img_rect_h
            return (x, y)

        mm_points = np.array([px_to_mm(p[0], p[1]) for p in points])

        if progress_callback:
            progress_callback(90)

        if render_mode == "TSP Path":
            # New TSP path mode: use reorder_paths + optimize_2opt
            from plottter.processing.optimize import reorder_paths, optimize_2opt

            # Create zero-length segment for each stipple point
            point_paths: list[Polyline] = [
                [(float(x), float(y)), (float(x), float(y))]
                for x, y in mm_points
            ]

            # NN reorder with multiple starts for a good initial tour
            ordered = reorder_paths(point_paths, num_starts=5)

            # 2-opt improvement to reduce travel distance
            if tsp_optimize:
                ordered = optimize_2opt(ordered)

            # Extract ordered positions as a single connected polyline
            tsp_path: Polyline = [path[0] for path in ordered]
            if progress_callback:
                progress_callback(100)
            result: list[Polyline] = [tsp_path] if len(tsp_path) >= 2 else []
        elif connect_tsp:
            # Legacy TSP mode: KDTree-accelerated nearest-neighbor path
            order = _nearest_neighbor_tsp(mm_points)
            path: Polyline = [tuple(mm_points[i]) for i in order]  # type: ignore[misc]
            if progress_callback:
                progress_callback(100)
            result = [path] if len(path) >= 2 else []
        else:
            # Dots mode: draw each point as a tiny circle
            result = []
            for px, py in mm_points:
                result.append(_tiny_circle(px, py))
            if progress_callback:
                progress_callback(100)

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
        return result
