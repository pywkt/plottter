"""CircularScribbleGenerator — tone-aware circular scribble art from an image.

The algorithm:
1. Sample seed points from the image using tone-aware exclusion radii (dark areas
   get dense points, bright areas get sparse points).
2. Connect seed points into a single continuous tracing path using grid-based
   partitioning and circuit merging (task 25.2).
3. Walk the tracing path and synthesize circular scribble patterns whose size
   and density adapt to local image brightness (tasks 25.3-25.5).

This file implements tasks 25.1, 25.2, 25.3, 25.4, and 25.5.
"""

from __future__ import annotations

import math
import random as _random
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import _px_to_mm, compute_image_rect
from plottter.generators.base import (
    BoolParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


# ---------------------------------------------------------------------------
# Tone-aware Poisson-disk sampling (Task 25.1)
# ---------------------------------------------------------------------------

def _tone_aware_sample(
    gray: np.ndarray,
    min_spacing_px: float,
    max_spacing_px: float,
    rng: _random.Random,
) -> list[tuple[float, float]]:
    """Place seed points using a scanline Poisson-disk approach with per-pixel
    exclusion radii driven by local brightness.

    Dark pixels (brightness ~0) use ``min_spacing_px`` as their exclusion
    radius (dense coverage), while bright pixels (brightness ~255) use
    ``max_spacing_px`` (sparse coverage).

    The algorithm sweeps row-by-row; for each candidate pixel it looks up the
    local exclusion radius from brightness, then checks whether any previously
    placed point is closer than that radius.  Optional ±1-pixel random jitter
    breaks up grid artefacts.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale image.
    min_spacing_px:
        Exclusion radius for the darkest pixels (pixel units).
    max_spacing_px:
        Exclusion radius for the brightest pixels (pixel units).
    rng:
        Seeded :class:`random.Random` instance for reproducibility.

    Returns
    -------
    List of ``(x, y)`` points in pixel coordinates.
    """
    h, w = gray.shape[:2]

    # Build an exclusion-radius map in pixel space (float32).
    # radius(x, y) = min_spacing + (brightness/255) * (max_spacing - min_spacing)
    brightness_norm = gray.astype(np.float32) / 255.0
    radius_map = (
        min_spacing_px + brightness_norm * (max_spacing_px - min_spacing_px)
    )  # shape: (h, w), values in [min_spacing_px, max_spacing_px]

    points: list[tuple[float, float]] = []

    # Step size for the scanline sweep — use min_spacing so we don't miss any
    # potential seed point.
    step = max(1, int(min_spacing_px * 0.5))

    for row in range(0, h, step):
        for col in range(0, w, step):
            # Add optional ±1 pixel jitter
            jx = rng.uniform(-1.0, 1.0)
            jy = rng.uniform(-1.0, 1.0)
            px = col + jx
            py = row + jy

            # Clamp to image bounds
            px = max(0.0, min(w - 1.0, px))
            py = max(0.0, min(h - 1.0, py))

            # Local exclusion radius at this pixel
            ipx = int(px)
            ipy = int(py)
            excl = float(radius_map[ipy, ipx])

            # Check if any existing point is within the exclusion radius.
            # We use an early-exit loop; a scanline sweep already limits
            # candidates to O(1/min_spacing²) per cell.
            too_close = False
            for qx, qy in reversed(points):
                dx = px - qx
                dy = py - qy
                if dx * dx + dy * dy < excl * excl:
                    too_close = True
                    break
            if not too_close:
                points.append((px, py))

    return points


# ---------------------------------------------------------------------------
# Tracing path construction (Task 25.2)
# ---------------------------------------------------------------------------

def _partition_by_grid(
    points: list[tuple[float, float]],
    img_w: int,
    img_h: int,
    grid_size: int,
) -> list[list[list[tuple[float, float]]]]:
    """Partition points into a *grid_size* × *grid_size* grid of cells.

    Returns a 2-D list ``cells[row][col]`` where each entry is the list of
    points that fall inside that cell.  Points are distributed based on their
    pixel coordinates relative to the full image dimensions.

    Parameters
    ----------
    points:
        Seed points in pixel coordinates.
    img_w, img_h:
        Image dimensions in pixels.
    grid_size:
        Number of grid divisions along each axis.
    """
    cells: list[list[list[tuple[float, float]]]] = [
        [[] for _ in range(grid_size)] for _ in range(grid_size)
    ]
    for px, py in points:
        col = min(int(px * grid_size / img_w), grid_size - 1)
        row = min(int(py * grid_size / img_h), grid_size - 1)
        cells[row][col].append((px, py))
    return cells


def _nearest_neighbor_circuit(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Build a nearest-neighbor Hamiltonian circuit through *points*.

    Starting at the first point, repeatedly connects to the closest unvisited
    point.  Uses a KD-tree for cells with more than 50 points, falling back
    to a simple O(n²) scan for smaller sets.

    Parameters
    ----------
    points:
        List of ``(x, y)`` points in any coordinate system.

    Returns
    -------
    Ordered list of the same points forming a circuit (without repeating the
    start point at the end).
    """
    n = len(points)
    if n <= 2:
        return list(points)

    pts = np.array(points, dtype=np.float64)

    if n > 50:
        # KD-tree approach: O(n K log n) with K = sqrt(n)
        try:
            from scipy.spatial import cKDTree

            K = min(int(np.sqrt(n)) + 5, n)
            tree = cKDTree(pts)
            used = np.zeros(n, dtype=bool)
            order = [0]
            used[0] = True
            current = 0

            for _ in range(n - 1):
                # Query K nearest; find first unused
                k_query = min(K, n)
                while True:
                    _, inds = tree.query(pts[current], k=k_query)
                    inds = np.atleast_1d(inds)
                    found = False
                    for idx in inds:
                        if not used[idx]:
                            order.append(int(idx))
                            used[int(idx)] = True
                            current = int(idx)
                            found = True
                            break
                    if found:
                        break
                    k_query = min(k_query * 2, n)
                    if k_query == n:
                        # Fallback: linear scan
                        dists = np.sum((pts - pts[current]) ** 2, axis=1)
                        dists[used] = np.inf
                        best = int(np.argmin(dists))
                        order.append(best)
                        used[best] = True
                        current = best
                        break

            return [(float(pts[i, 0]), float(pts[i, 1])) for i in order]
        except ImportError:
            pass  # Fall through to O(n²) version

    # Simple O(n²) for small sets (or if scipy unavailable)
    used = np.zeros(n, dtype=bool)
    order = [0]
    used[0] = True

    for _ in range(n - 1):
        last = pts[order[-1]]
        dists = np.sum((pts - last) ** 2, axis=1)
        dists[used] = np.inf
        best = int(np.argmin(dists))
        order.append(best)
        used[best] = True

    return [(float(pts[i, 0]), float(pts[i, 1])) for i in order]


def _merge_circuits(
    circuit_a: list[tuple[float, float]],
    circuit_b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge two circuits into one by finding the minimum-cost edge swap.

    Implements the merge_ET strategy from the reference code (main.cpp lines
    20-100): for each pair of edges (one from each circuit), compute the cost
    of removing both edges and reconnecting with cross-edges.  The lowest-cost
    reconnection is applied.

    Two reconnection modes are tried:
    - Mode 0: remove (ax, ay) and (bx, by), add (ax, bx) and (ay, by)
    - Mode 1: remove (ax, ay) and (bx, by), add (ax, by) and (ay, bx)

    For large circuits (``n × m > 500²``), a KD-tree limits the candidate
    search to the K nearest neighbours per B point, preserving quality while
    keeping the operation fast.

    Parameters
    ----------
    circuit_a, circuit_b:
        Ordered lists of ``(x, y)`` points representing closed circuits.

    Returns
    -------
    Merged ordered list of all points from both circuits.
    """
    n = len(circuit_a)
    m = len(circuit_b)

    if n == 0:
        return list(circuit_b)
    if m == 0:
        return list(circuit_a)

    pts_a = np.array(circuit_a, dtype=np.float64)  # (n, 2)
    pts_b = np.array(circuit_b, dtype=np.float64)  # (m, 2)

    # --- Special case: single-point circuit (insert into the other) ---
    if n == 1:
        p = pts_a[0]
        bx = pts_b                             # (m, 2) — edge starts
        by = pts_b[np.arange(1, m + 1) % m]   # (m, 2) — edge ends
        # Insertion cost: dist(bx,p) + dist(by,p) - dist(bx,by)
        costs = np.sum((bx - p) ** 2, axis=1) + np.sum((by - p) ** 2, axis=1)
        j = int(np.argmin(costs))
        return list(circuit_b[: j + 1]) + [circuit_a[0]] + list(circuit_b[j + 1 :])

    if m == 1:
        p = pts_b[0]
        ax = pts_a
        ay = pts_a[np.arange(1, n + 1) % n]
        costs = np.sum((ax - p) ** 2, axis=1) + np.sum((ay - p) ** 2, axis=1)
        i = int(np.argmin(costs))
        return list(circuit_a[: i + 1]) + [circuit_b[0]] + list(circuit_a[i + 1 :])

    # --- General case: find best edge-swap ---
    # For large circuits use KD-tree to limit candidate pairs
    MAX_BRUTE_SQ = 500 * 500
    if n * m <= MAX_BRUTE_SQ:
        # Fully vectorised brute-force: evaluate all n×m edge pairs
        i_vals = np.repeat(np.arange(n), m)
        j_vals = np.tile(np.arange(m), n)
    else:
        # KD-tree: for each B point, find K nearest A points as candidates
        K = 10
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(pts_a)
            k = min(K, n)
            _, idx_in_a = tree.query(pts_b, k=k)  # (m, k)
            if k == 1:
                idx_in_a = idx_in_a.reshape(-1, 1)
            j_vals = np.repeat(np.arange(m), k)
            i_vals = idx_in_a.flatten()
        except ImportError:
            # Fallback: random candidates
            rstate = np.random.RandomState(0)
            k = min(20, n)
            i_vals = rstate.randint(0, n, m * k)
            j_vals = np.repeat(np.arange(m), k)

    # Coordinates of edge endpoints
    ax = pts_a[i_vals]               # (candidates, 2)
    ay = pts_a[(i_vals + 1) % n]
    bx = pts_b[j_vals]
    by = pts_b[(j_vals + 1) % m]

    # Squared-distance costs (no sqrt needed for argmin)
    cost0 = np.sum((ax - bx) ** 2, axis=1) + np.sum((ay - by) ** 2, axis=1)
    cost1 = np.sum((ax - by) ** 2, axis=1) + np.sum((ay - bx) ** 2, axis=1)

    best0 = int(np.argmin(cost0))
    best1 = int(np.argmin(cost1))

    if cost0[best0] <= cost1[best1]:
        i, j, mode = int(i_vals[best0]), int(j_vals[best0]), 0
    else:
        i, j, mode = int(i_vals[best1]), int(j_vals[best1]), 1

    # Rotate each circuit so the chain starts at the point after the cut edge.
    # chain_a: starts at pts_a[(i+1)%n] = ay, ends at pts_a[i] = ax
    split_a = (i + 1) % n
    chain_a = np.concatenate([pts_a[split_a:], pts_a[:split_a]], axis=0)

    # chain_b: starts at pts_b[(j+1)%m] = by, ends at pts_b[j] = bx
    split_b = (j + 1) % m
    chain_b = np.concatenate([pts_b[split_b:], pts_b[:split_b]], axis=0)

    if mode == 0:
        # Connect ax→bx (end of chain_a to end of chain_b reversed)
        merged_arr = np.concatenate([chain_a, chain_b[::-1]], axis=0)
    else:
        # Connect ax→by (end of chain_a to start of chain_b)
        merged_arr = np.concatenate([chain_a, chain_b], axis=0)

    return [(float(p[0]), float(p[1])) for p in merged_arr]


def _hermite_smooth(
    path: list[tuple[float, float]],
    steps: int = 10,
) -> list[tuple[float, float]]:
    """Smooth a path using cubic Hermite (Catmull-Rom) interpolation.

    For each consecutive pair of points, computes Catmull-Rom tangent vectors
    and evaluates the cubic Hermite basis at *steps* uniformly spaced parameter
    values.  The output has approximately ``steps × len(path)`` points.

    Parameters
    ----------
    path:
        Ordered list of ``(x, y)`` points to smooth.
    steps:
        Number of interpolation steps per segment (default 10).  More steps
        produce smoother output at the cost of more points.

    Returns
    -------
    Smoothed list of ``(x, y)`` points.
    """
    n = len(path)
    if n < 2:
        return list(path)

    pts = np.array(path, dtype=np.float64)
    result: list[tuple[float, float]] = []

    for i in range(n - 1):
        P1 = pts[i]
        P2 = pts[i + 1]

        # Catmull-Rom tangents: T = (P_{i+1} − P_{i-1}) / 2
        T1 = (pts[i + 1] - pts[i - 1]) / 2.0 if i > 0 else P2 - P1
        T2 = (pts[i + 2] - pts[i]) / 2.0 if i < n - 2 else P2 - P1

        for t_step in range(steps):
            s = t_step / float(steps)
            s2 = s * s
            s3 = s2 * s
            # Cubic Hermite basis functions
            h1 = 2.0 * s3 - 3.0 * s2 + 1.0
            h2 = -2.0 * s3 + 3.0 * s2
            h3 = s3 - 2.0 * s2 + s
            h4 = s3 - s2
            p = h1 * P1 + h2 * P2 + h3 * T1 + h4 * T2
            result.append((float(p[0]), float(p[1])))

    # Include the final point
    result.append((float(pts[-1, 0]), float(pts[-1, 1])))
    return result


# ---------------------------------------------------------------------------
# Circular scribble synthesis (Task 25.3)
# ---------------------------------------------------------------------------

def _synthesize_scribbles(
    tracing_path_px: list[tuple[float, float]],
    gray: np.ndarray,
    min_radius_px: float,
    max_radius_px: float,
    min_speed_px: float,
    max_speed_px: float,
    angle_step_deg: float,
    tone_gamma: float,
    seed: int = 42,
    edge_dist_map: np.ndarray | None = None,
    edge_sensitivity: float = 0.0,
    orientation_strength: float = 0.3,
    skip_background: bool = False,
    progress_callback: Any = None,
    cancelled_callback: Any = None,
) -> list[tuple[float, float]]:
    """Walk the tracing path and synthesize circular scribble points.

    For each consecutive pair of points in *tracing_path_px*, the algorithm
    steps from ``t=0`` to ``t=1`` along the segment.  At each step:

    1. The center position is interpolated along the segment.
    2. The grayscale image is sampled at the center to get luminance
       ``L ∈ [0, 1]`` (0 = dark, 1 = bright).
    3. Tone-gamma mapping is applied: ``mapped = L^tone_gamma``.
    4. The circle radius is computed:
       ``r = min_radius_px + mapped × (max_radius_px − min_radius_px)``
       — small circles in dark areas, large circles in bright areas.
    5. The step size (advance per scribble point) is computed:
       ``dt = min_speed_px + mapped × (max_speed_px − min_speed_px)``
       — small steps (tight loops) in dark areas, large steps in bright.
    6. A point on an ellipse centred at the current position is emitted:
       ``x = centre_x + a·cos(θ)``, ``y = centre_y + b·sin(θ)``
       where ``a = radius``, ``b = radius·cos(40°)`` (slight perspective tilt
       controlled by a smooth noise map for a 3D shading effect), and the
       whole ellipse is rotated by an angle derived from the noise map.
    7. θ is incremented by *angle_step_deg* and the segment parameter ``t``
       is incremented by ``dt / segment_length``.

    Parameters
    ----------
    tracing_path_px:
        Ordered list of ``(x, y)`` points in pixel coordinates forming the
        tracing path (typically the Hermite-smoothed output of
        :func:`_build_tracing_path`).
    gray:
        Single-channel uint8 grayscale image.
    min_radius_px, max_radius_px:
        Circle radius bounds in pixel units (dark areas → min, bright → max).
    min_speed_px, max_speed_px:
        Step size bounds in pixel units (dark areas → min, bright → max).
        Smaller steps produce tighter, denser loops.
    angle_step_deg:
        Angular increment per scribble point in degrees.  20° produces
        smooth circles (18 points per full revolution).
    tone_gamma:
        Power-curve exponent for brightness → radius/speed mapping.
        Values > 1 emphasise dark areas (more detail in shadows).
    seed:
        Random seed for the noise-map generation (for reproducibility).
    edge_dist_map:
        Optional float32 array (same shape as *gray*) where each value is
        the Euclidean distance (in pixels) to the nearest edge pixel, as
        produced by ``cv2.distanceTransform``.  When provided and
        *edge_sensitivity* > 0, circle radii are reduced near edges so
        scribbles do not cross important image features.
    edge_sensitivity:
        Blend factor in [0, 1] controlling how aggressively radii are
        reduced near edges.  0 = disabled (uses computed radius as-is),
        1 = fully clamp radius to the distance to the nearest edge.
        Intermediate values blend between the two.
    orientation_strength:
        Controls the intensity of the noise-driven ellipse tilt effect.
        0.0 = perfect circles (no tilt, no b-axis compression),
        1.0 = full 3D shading effect (ellipses tilted by noise field).
        Intermediate values blend linearly between the two extremes.
    skip_background:
        When True, very bright pixels (brightness > 0.98) are treated as
        background.  The step size is increased dramatically (×20) so the
        pen advances quickly through white/near-white areas, creating
        extremely sparse, barely-visible loops there.
    progress_callback:
        Optional ``f(percent: int)`` for progress updates.
    cancelled_callback:
        Optional ``f() -> bool`` that returns ``True`` to abort.

    Returns
    -------
    Flat list of ``(x, y)`` scribble points in pixel coordinates.
    """
    n = len(tracing_path_px)
    if n < 2:
        return []

    img_h, img_w = gray.shape[:2]

    # Float32 grayscale for fast array indexing
    gray_f = gray.astype(np.float32) / 255.0

    # --- Smooth noise map for 3D shading / ellipse tilt effect ---
    # Mirrors the reference code: generate half-size random noise, blur heavily,
    # resize to full image dimensions. This gives a smooth orientation field that
    # tilts the ellipses differently across the image for a 3D shading look.
    rng_np = np.random.RandomState(seed ^ 0xDEADBEEF)
    noise_h = max(2, img_h // 2)
    noise_w = max(2, img_w // 2)
    noise_small = rng_np.randint(0, 256, (noise_h, noise_w), dtype=np.uint8).astype(np.float32)

    try:
        import cv2 as _cv2
        ksize = min(51, (noise_h // 4) * 2 + 1)  # must be odd, ≤ image dims
        ksize = max(3, ksize)
        noise_blurred = _cv2.GaussianBlur(noise_small, (ksize, ksize), 0)
        noise_map = _cv2.resize(noise_blurred, (img_w, img_h)).astype(np.float32)
    except ImportError:
        try:
            from scipy.ndimage import gaussian_filter, zoom as _zoom
            sigma = max(1.0, min(25.0, noise_h / 4.0))
            noise_blurred = gaussian_filter(noise_small, sigma=sigma)
            noise_map = _zoom(noise_blurred, (img_h / noise_h, img_w / noise_w), order=1).astype(np.float32)
        except ImportError:
            noise_map = np.zeros((img_h, img_w), dtype=np.float32)

    # Normalise noise map to [0, 1]
    nm_min, nm_max = float(noise_map.min()), float(noise_map.max())
    if nm_max > nm_min:
        noise_map = (noise_map - nm_min) / (nm_max - nm_min)
    else:
        noise_map[:] = 0.0

    # --- Precompute trig look-up tables for the angle step ---
    # theta cycles every period steps; we use integer indexing to avoid
    # redundant sin/cos calls (the dominant inner-loop cost).
    if angle_step_deg > 0.0:
        period = max(1, int(round(360.0 / angle_step_deg)))
    else:
        period = 18  # fallback (20° step)

    angles_rad = np.arange(period, dtype=np.float64) * (angle_step_deg * math.pi / 180.0)
    cos_table = np.cos(angles_rad).tolist()  # list for O(1) indexed access
    sin_table = np.sin(angles_rad).tolist()

    # --- Tilt parameters for 3D shading effect ---
    # The ellipse b-axis is foreshortened by cos(40°) to simulate a 3D tilt.
    # The rotation angle (theta_z) comes from the noise map.
    COS_TILT = math.cos(40.0 * math.pi / 180.0)  # ≈ 0.766

    # --- Main synthesis loop ---
    scribble_pts: list[tuple[float, float]] = []
    theta_idx = 0  # cycles 0 … period-1

    n_segments = n - 1
    report_step = max(1, n_segments // 100)

    for seg_idx in range(n_segments):
        if cancelled_callback and cancelled_callback():
            return scribble_pts
        if progress_callback and seg_idx % report_step == 0:
            pct = int(seg_idx / n_segments * 100)
            progress_callback(pct)

        x0, y0 = tracing_path_px[seg_idx]
        x1, y1 = tracing_path_px[seg_idx + 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-9:
            continue

        t = 0.0
        while t < 1.0:
            cx = x0 + t * dx
            cy = y0 + t * dy

            # Integer pixel coords for image sampling (clamped)
            ix = max(0, min(img_w - 1, int(cx + 0.5)))
            iy = max(0, min(img_h - 1, int(cy + 0.5)))

            # Sample grayscale luminance (0 = dark, 1 = bright)
            gv = float(gray_f[iy, ix])

            # Tone-gamma mapping: emphasises dark areas when gamma > 1
            mapped = (gv ** tone_gamma) if gv > 0.0 else 0.0
            if mapped > 1.0:
                mapped = 1.0

            # Radius: small in dark areas, large in bright areas
            radius = min_radius_px + mapped * (max_radius_px - min_radius_px)
            if radius < 0.05:
                radius = 0.05

            # Edge-aware radius reduction (Task 25.4):
            # Reduce the radius near detected edges so scribble circles
            # don't cross important image features.  Uses a precomputed
            # distance transform (edge_dist_map[y, x] = dist to nearest
            # edge pixel in px), so this is a single O(1) lookup per point.
            if edge_dist_map is not None and edge_sensitivity > 0.0:
                edge_dist_px = float(edge_dist_map[iy, ix])
                edge_reduced = min(radius, edge_dist_px)
                radius = radius + edge_sensitivity * (edge_reduced - radius)
                if radius < 0.05:
                    radius = 0.05

            # Step size: small in dark (tight loops), large in bright (open)
            dt = min_speed_px + mapped * (max_speed_px - min_speed_px)
            if dt < 0.05:
                dt = 0.05

            # Background skip: very bright areas (near-white) are advanced through
            # quickly so the pen doesn't linger on the background.
            if skip_background and gv > 0.98:
                dt = max_speed_px * 20.0

            # Noise-based tilt for 3D shading (reference: theta_z = noise*45+45 deg)
            # orientation_strength=0 → perfect circles (theta_z=0, b=a)
            # orientation_strength=1 → full tilt effect
            nm = float(noise_map[iy, ix])  # [0, 1]
            theta_z = (nm * 45.0 + 45.0) * (math.pi / 180.0) * orientation_strength
            cos_tz = math.cos(theta_z)
            sin_tz = math.sin(theta_z)

            # Ellipse: a = radius (horizontal), b-axis compression scaled by orientation_strength.
            # At orientation_strength=0: b = a (perfect circle).
            # At orientation_strength=1: b = radius * COS_TILT (full 3D shading tilt).
            a = radius
            b = radius * (1.0 - orientation_strength * (1.0 - COS_TILT))

            # Circle position from precomputed tables
            ct = cos_table[theta_idx]
            st = sin_table[theta_idx]
            X = a * ct
            Y = b * st

            # Rotate by theta_z (z-axis rotation for 3D orientation effect)
            X_rot = cos_tz * X - sin_tz * Y
            Y_rot = sin_tz * X + cos_tz * Y

            scribble_pts.append((cx + X_rot, cy + Y_rot))

            theta_idx = (theta_idx + 1) % period
            t += dt / dist

    return scribble_pts


def _build_tracing_path(
    seed_pts_px: list[tuple[float, float]],
    img_w: int,
    img_h: int,
    grid_size: int,
    progress_callback: Any = None,
    cancelled_callback: Any = None,
) -> list[tuple[float, float]]:
    """Connect seed points into a single continuous tracing path.

    The algorithm:

    1. Divide the image into a ``grid_size × grid_size`` grid of cells.
    2. For each cell, connect its points into a local nearest-neighbor circuit.
    3. Iteratively merge all cell circuits into one global circuit using the
       minimum-cost edge-swap strategy from the reference code.

    The result is a single ordered list of points (closed loop) suitable for
    Hermite smoothing and subsequent scribble synthesis.

    Parameters
    ----------
    seed_pts_px:
        Seed points in pixel coordinates (from :func:`_tone_aware_sample`).
    img_w, img_h:
        Image dimensions in pixels.
    grid_size:
        Number of grid divisions along each axis (default 10).
    progress_callback:
        Optional callable ``f(percent: int)`` for progress updates.
    cancelled_callback:
        Optional callable ``f() -> bool`` that returns ``True`` to abort.

    Returns
    -------
    Ordered list of ``(x, y)`` points in pixel coordinates forming the tracing
    path.  Returns an empty list if *seed_pts_px* is empty or generation is
    cancelled.
    """
    if not seed_pts_px:
        return []
    if len(seed_pts_px) == 1:
        return list(seed_pts_px)

    # Step 1: Partition points into grid cells
    cells = _partition_by_grid(seed_pts_px, img_w, img_h, grid_size)

    # Collect non-empty cell circuits in raster order
    circuits: list[list[tuple[float, float]]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            cell_pts = cells[row][col]
            if len(cell_pts) >= 2:
                circuits.append(_nearest_neighbor_circuit(cell_pts))
            elif len(cell_pts) == 1:
                circuits.append(list(cell_pts))

    if not circuits:
        return []

    if len(circuits) == 1:
        if progress_callback:
            progress_callback(100)
        return circuits[0]

    # Step 2: Merge all circuits into one (raster order)
    total = len(circuits)
    main_circuit = circuits[0]

    for idx, circuit in enumerate(circuits[1:], start=1):
        if cancelled_callback and cancelled_callback():
            return []
        main_circuit = _merge_circuits(main_circuit, circuit)
        if progress_callback:
            pct = int(idx / (total - 1) * 100)
            progress_callback(pct)

    return main_circuit


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

@register_generator
class CircularScribbleGenerator(Generator):
    """Tone-aware circular scribble art generator.

    Samples seed points from a source image using brightness-adaptive
    exclusion radii (dense in dark areas, sparse in bright areas), then
    connects them into a single continuous tracing path using grid-based
    partitioning and circuit merging.  The tracing path is smoothed with
    cubic Hermite interpolation and returned as a single polyline.

    Circular scribble patterns are synthesized along the tracing path, with
    ellipse size and eccentricity modulated by local image tone (tasks 25.1-25.3).
    """

    name = "Circular Scribble"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="min_radius_mm",
                label="Min Circle Radius (mm)",
                min=0.1,
                max=10.0,
                step=0.1,
                default=0.5,
                description="Circle radius in dark areas — small circles create dense shading in shadows",
            ),
            FloatParam(
                name="max_radius_mm",
                label="Max Circle Radius (mm)",
                min=0.5,
                max=30.0,
                step=0.5,
                default=5.0,
                description="Circle radius in bright areas — large circles create open, airy highlights",
            ),
            FloatParam(
                name="min_sample_spacing_mm",
                label="Min Sample Spacing (mm)",
                min=0.5,
                max=10.0,
                step=0.1,
                default=1.0,
                description="Minimum spacing between seed points in dark areas — controls maximum detail density",
            ),
            FloatParam(
                name="max_sample_spacing_mm",
                label="Max Sample Spacing (mm)",
                min=2.0,
                max=50.0,
                step=0.5,
                default=8.0,
                description="Maximum spacing between seed points in bright areas — controls sparseness in highlights",
            ),
            IntParam(
                name="path_grid_size",
                label="Path Grid Size",
                min=3,
                max=30,
                step=1,
                default=10,
                description=(
                    "Grid divisions for path construction — higher values preserve more detail "
                    "by building local circuits within smaller regions, but may produce a "
                    "slightly choppier path before smoothing"
                ),
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
                name="min_speed",
                label="Min Drawing Speed (mm)",
                min=0.1,
                max=5.0,
                step=0.1,
                default=0.5,
                description=(
                    "Drawing speed in dark areas — how far the pen centre advances "
                    "per scribble point. Lower values produce tighter, denser loops "
                    "that fill dark regions more heavily."
                ),
            ),
            FloatParam(
                name="max_speed",
                label="Max Drawing Speed (mm)",
                min=1.0,
                max=30.0,
                step=0.5,
                default=8.0,
                description=(
                    "Drawing speed in bright areas — how far the pen centre advances "
                    "per scribble point. Higher values create wider spacing between "
                    "loops so bright areas remain open."
                ),
            ),
            FloatParam(
                name="angle_step_deg",
                label="Angle Step (°)",
                min=5.0,
                max=45.0,
                step=1.0,
                default=20.0,
                description=(
                    "Angular increment per scribble sample in degrees. "
                    "20° produces smooth circles (18 points per revolution). "
                    "Larger values give faceted, polygon-like loops."
                ),
            ),
            FloatParam(
                name="tone_gamma",
                label="Tone Gamma",
                min=0.5,
                max=3.0,
                step=0.1,
                default=1.5,
                description=(
                    "Power-curve exponent for the brightness → radius/speed mapping. "
                    "Values > 1 add more detail to shadows; values < 1 emphasise highlights. "
                    "1.0 = linear mapping."
                ),
            ),
            IntParam(
                name="seed",
                label="Random Seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for reproducible point placement",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image (dense sampling in bright areas instead of dark)",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before sampling (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before sampling (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=0.0,
                description="Gaussian blur applied before sampling — smooths the brightness distribution",
            ),
            # --- Edge-aware radius reduction (Task 25.4) ---
            FloatParam(
                name="edge_sensitivity",
                label="Edge Sensitivity",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.7,
                description=(
                    "Controls how aggressively circle radii shrink near detected edges. "
                    "0 = disabled, 1 = circles are fully clipped to the distance to the "
                    "nearest edge — keeps scribbles from crossing sharp features like eyes "
                    "or object contours."
                ),
            ),
            IntParam(
                name="edge_low",
                label="Edge Low Threshold",
                min=10,
                max=200,
                step=5,
                default=50,
                description=(
                    "Lower threshold for Canny edge detection (only used when "
                    "Edge Sensitivity > 0). Lower values detect more subtle edges."
                ),
            ),
            IntParam(
                name="edge_high",
                label="Edge High Threshold",
                min=50,
                max=300,
                step=10,
                default=150,
                description=(
                    "Upper threshold for Canny edge detection (only used when "
                    "Edge Sensitivity > 0). Higher values detect only the strongest edges."
                ),
            ),
            # --- Orientation variation (Task 25.5) ---
            FloatParam(
                name="orientation_strength",
                label="Orientation Variation",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.3,
                description=(
                    "Amount of ellipse tilt variation driven by a smooth noise field. "
                    "0 = perfect circles (no tilt), 1 = full 3D shading effect where "
                    "circles appear tilted into depth for a sculptural look. "
                    "Intermediate values blend smoothly between the two extremes."
                ),
            ),
            BoolParam(
                name="skip_background",
                label="Skip White Background",
                default=False,
                description=(
                    "When enabled, very bright areas (near-white pixels) are skipped "
                    "quickly — the pen advances much faster through the background, "
                    "producing extremely sparse loops there. Useful for images with a "
                    "clean white background where you want minimal scribbles."
                ),
            ),
        ]

    def get_presets(self) -> list[Preset]:
        _base = {
            "min_sample_spacing_mm": 1.0,
            "max_sample_spacing_mm": 8.0,
            "path_grid_size": 10,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            "seed": 42,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "min_speed": 0.5,
            "max_speed": 8.0,
            "angle_step_deg": 20.0,
            "tone_gamma": 1.5,
            "edge_sensitivity": 0.7,
            "edge_low": 50,
            "edge_high": 150,
            "orientation_strength": 0.3,
            "skip_background": False,
        }
        return [
            Preset(
                name="Default",
                params={
                    **_base,
                    # Wider radius range → small tight circles in darks, large open ones in brights
                    "min_radius_mm": 0.4,
                    "max_radius_mm": 5.0,
                    # Wider spacing/speed ratios amplify dark-vs-bright density contrast
                    "min_sample_spacing_mm": 0.8,
                    "max_sample_spacing_mm": 12.0,
                    "min_speed": 0.4,
                    "max_speed": 12.0,
                    # Higher gamma: mid-tones mapped toward darks → more detail in shadows
                    "tone_gamma": 2.0,
                    # Contrast pre-stretch: makes compressed tonal ranges pop
                    "contrast": 20.0,
                },
            ),
            Preset(
                name="Portrait",
                params={
                    **_base,
                    # Fine detail for skin texture and facial features
                    "min_radius_mm": 0.25,
                    "max_radius_mm": 3.0,
                    "min_sample_spacing_mm": 0.6,
                    "max_sample_spacing_mm": 9.0,
                    "min_speed": 0.25,
                    "max_speed": 7.0,
                    # Smooth circles → natural look on face curves
                    "angle_step_deg": 15.0,
                    # Strong gamma: face shadows stay visibly darker
                    "tone_gamma": 2.5,
                    # High edge sensitivity: respect contours of eyes, nose, lips
                    "edge_sensitivity": 0.9,
                    "edge_low": 40,
                    "edge_high": 120,
                    "orientation_strength": 0.4,
                    # Skip white background so the face is the sole subject
                    "skip_background": True,
                    # Moderate contrast boost to pull out face tonal variation
                    "contrast": 30.0,
                },
            ),
            Preset(
                name="Detailed",
                params={
                    **_base,
                    # Very fine radius for maximum resolution
                    "min_radius_mm": 0.2,
                    "max_radius_mm": 2.5,
                    # Dense point cloud everywhere; still sparser in brights
                    "min_sample_spacing_mm": 0.5,
                    "max_sample_spacing_mm": 5.0,
                    "min_speed": 0.2,
                    "max_speed": 4.0,
                    "angle_step_deg": 15.0,
                    # Very high gamma: mid-tones treated like darks → high coverage with contrast
                    "tone_gamma": 2.8,
                    "edge_sensitivity": 0.8,
                    "orientation_strength": 0.2,
                    # Strong contrast pre-stretch to maximise separation between tone bands
                    "contrast": 40.0,
                },
            ),
            Preset(
                name="Loose Sketch",
                params={
                    **_base,
                    "min_radius_mm": 0.4,
                    "max_radius_mm": 6.0,
                    "min_sample_spacing_mm": 1.5,
                    "max_sample_spacing_mm": 10.0,
                    "min_speed": 0.8,
                    "max_speed": 10.0,
                    "angle_step_deg": 30.0,
                    "tone_gamma": 1.0,
                    "seed": 7,
                    "edge_sensitivity": 0.0,
                    "orientation_strength": 0.0,
                },
            ),
            Preset(
                name="Shaded",
                params={
                    **_base,
                    # Moderate radius range; let density carry the tonal weight
                    "min_radius_mm": 0.35,
                    "max_radius_mm": 5.0,
                    # Very wide spacing ratio (21:1) → extreme density contrast dark/bright
                    "min_sample_spacing_mm": 0.7,
                    "max_sample_spacing_mm": 15.0,
                    # Very wide speed ratio → tight loops in darks, sweeping advance in brights
                    "min_speed": 0.3,
                    "max_speed": 15.0,
                    # Highest gamma of all presets: dramatic shadow emphasis
                    "tone_gamma": 3.0,
                    "edge_sensitivity": 0.5,
                    # Strong orientation effect for sculptural shading feel
                    "orientation_strength": 0.8,
                    "seed": 13,
                    "contrast": 20.0,
                },
            ),
            Preset(
                name="Bold Scribble",
                params={
                    **_base,
                    # Large bold circles; clearly visible dark/bright separation
                    "min_radius_mm": 1.2,
                    "max_radius_mm": 10.0,
                    # Wide spacing range (9:1) so bold darks contrast with open brights
                    "min_sample_spacing_mm": 2.0,
                    "max_sample_spacing_mm": 18.0,
                    "min_speed": 1.2,
                    "max_speed": 18.0,
                    "tone_gamma": 2.2,
                    "orientation_strength": 0.3,
                    # Strong contrast pre-stretch for clear large-scale tonal separation
                    "contrast": 35.0,
                },
            ),
            Preset(
                name="Soft Edge Aware",
                params={
                    **_base,
                    # Moderate parameters for smooth tonal transitions
                    "min_radius_mm": 0.4,
                    "max_radius_mm": 6.0,
                    "min_sample_spacing_mm": 0.8,
                    "max_sample_spacing_mm": 12.0,
                    "min_speed": 0.4,
                    "max_speed": 12.0,
                    # Softer gamma than other presets → gentle gradient
                    "tone_gamma": 1.8,
                    # Edge sensitivity tuned for smooth, slightly soft contours
                    "edge_sensitivity": 0.55,
                    "edge_low": 60,
                    "edge_high": 180,
                    "orientation_strength": 0.3,
                    # Blur input slightly so density varies smoothly across gradients
                    "blur_radius": 1.5,
                    # Light contrast boost — enough to reveal gradients, not harsh
                    "contrast": 15.0,
                },
            ),
        ]

    # Maximum points per output polyline.  Very long paths are split into
    # chunks to avoid canvas-widget performance issues; each chunk is a
    # separate polyline in the returned list.
    _MAX_POLYLINE_PTS = 100_000

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Generate circular scribble art from a source image.

        Phase 1 — tone-aware Poisson-disk sampling:
            Dark areas of the image receive densely packed seed points;
            bright areas receive sparse seed points.

        Phase 2 — tracing path construction:
            Seed points are connected into a single continuous closed loop via
            grid-based circuit construction and iterative circuit merging.
            The merged path is smoothed with cubic Hermite interpolation.

        Phase 3 — circular scribble synthesis (task 25.3):
            The algorithm walks along the smooth tracing path.  At each step,
            local image brightness drives the circle radius and step size:
            dark areas produce small, tight circles; bright areas produce
            large, sparse circles.  An ellipse-tilt effect (driven by a
            smooth noise map) adds a 3D shading appearance.

        Returns a list of polylines in mm canvas coordinates.  Very long
        outputs are split into chunks of ``_MAX_POLYLINE_PTS`` points to
        maintain canvas rendering performance.
        """
        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        # --- Apply preprocessing ---
        img = source.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 0.0))
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
                gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            except ImportError:
                gray = source.mean(axis=2).astype(np.uint8)
        else:
            gray = source.copy()

        img_h, img_w = gray.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()

        # Respect image fit/placement parameters
        img_rect = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        r_x1, r_y1, r_x2, r_y2 = img_rect
        rect_w_mm = r_x2 - r_x1
        rect_h_mm = r_y2 - r_y1

        if rect_w_mm <= 0 or rect_h_mm <= 0:
            return []

        min_spacing_mm = float(params.get("min_sample_spacing_mm", 1.0))
        max_spacing_mm = float(params.get("max_sample_spacing_mm", 8.0))

        if min_spacing_mm > max_spacing_mm:
            min_spacing_mm, max_spacing_mm = max_spacing_mm, min_spacing_mm
        min_spacing_mm = max(0.1, min_spacing_mm)

        # Convert mm spacings → pixel spacings using the image→canvas scale
        px_per_mm_x = img_w / rect_w_mm
        px_per_mm_y = img_h / rect_h_mm
        px_per_mm = (px_per_mm_x + px_per_mm_y) / 2.0

        min_spacing_px = max(1.0, min_spacing_mm * px_per_mm)
        max_spacing_px = max(min_spacing_px + 1.0, max_spacing_mm * px_per_mm)

        if progress_callback:
            progress_callback(5)

        # Seeded RNG for reproducibility
        seed = int(params.get("seed", 42))
        rng = _random.Random(seed)

        # --- Step 1: Tone-aware sampling ---
        seed_pts_px = _tone_aware_sample(gray, min_spacing_px, max_spacing_px, rng)

        if progress_callback:
            progress_callback(20)

        if cancelled_callback and cancelled_callback():
            return []

        if not seed_pts_px:
            return []

        # --- Step 2: Build tracing path ---
        grid_size = max(1, int(params.get("path_grid_size", 10)))

        # Nested progress reporting for the merge step (20→70%)
        def _merge_progress(pct: int) -> None:
            if progress_callback:
                progress_callback(20 + int(pct * 0.5))

        tracing_path_px = _build_tracing_path(
            seed_pts_px,
            img_w,
            img_h,
            grid_size,
            progress_callback=_merge_progress,
            cancelled_callback=cancelled_callback,
        )

        if not tracing_path_px:
            return []

        if cancelled_callback and cancelled_callback():
            return []

        if progress_callback:
            progress_callback(70)

        # --- Step 3: Hermite smoothing (in pixel space) ---
        smoothed_px = _hermite_smooth(tracing_path_px, steps=10)

        if cancelled_callback and cancelled_callback():
            return []

        if progress_callback:
            progress_callback(72)

        # --- Step 4: Circular scribble synthesis (in pixel space) ---
        min_radius_mm = float(params.get("min_radius_mm", 0.5))
        max_radius_mm = float(params.get("max_radius_mm", 5.0))
        min_speed_mm = float(params.get("min_speed", 0.5))
        max_speed_mm = float(params.get("max_speed", 8.0))
        angle_step_deg = float(params.get("angle_step_deg", 20.0))
        tone_gamma = float(params.get("tone_gamma", 1.5))

        if min_radius_mm > max_radius_mm:
            min_radius_mm, max_radius_mm = max_radius_mm, min_radius_mm
        if min_speed_mm > max_speed_mm:
            min_speed_mm, max_speed_mm = max_speed_mm, min_speed_mm

        # Convert mm parameters to pixel units
        min_radius_px = max(0.05, min_radius_mm * px_per_mm)
        max_radius_px = max(min_radius_px + 0.05, max_radius_mm * px_per_mm)
        min_speed_px = max(0.05, min_speed_mm * px_per_mm)
        max_speed_px = max(min_speed_px + 0.05, max_speed_mm * px_per_mm)

        # --- Edge-aware radius reduction (Task 25.4) ---
        # Precompute a Canny edge map and distance transform so the synthesis
        # inner loop can look up the distance to the nearest edge in O(1).
        edge_sensitivity = float(params.get("edge_sensitivity", 0.7))
        edge_dist_map: np.ndarray | None = None

        if edge_sensitivity > 0.0:
            edge_low = int(params.get("edge_low", 50))
            edge_high = int(params.get("edge_high", 150))
            try:
                import cv2 as _cv2_edge
                edges = _cv2_edge.Canny(gray, edge_low, edge_high)
                # distanceTransform requires a binary mask where 0 = obstacle
                # (edge pixel), non-zero = free.  Invert the edge map so edge
                # pixels (255) become 0 and background pixels (0) become 255.
                not_edges = _cv2_edge.bitwise_not(edges)
                edge_dist_map = _cv2_edge.distanceTransform(
                    not_edges, _cv2_edge.DIST_L2, 5
                )
            except ImportError:
                pass  # OpenCV unavailable — edge awareness disabled silently

        # --- Orientation variation and background skip (Task 25.5) ---
        orientation_strength = float(params.get("orientation_strength", 0.3))
        orientation_strength = max(0.0, min(1.0, orientation_strength))
        skip_background = bool(params.get("skip_background", False))

        # Nested progress callback: synthesis covers 72 → 95 %
        def _synth_progress(pct: int) -> None:
            if progress_callback:
                progress_callback(72 + int(pct * 0.23))

        scribble_pts_px = _synthesize_scribbles(
            smoothed_px,
            gray,
            min_radius_px=min_radius_px,
            max_radius_px=max_radius_px,
            min_speed_px=min_speed_px,
            max_speed_px=max_speed_px,
            angle_step_deg=angle_step_deg,
            tone_gamma=tone_gamma,
            seed=seed,
            edge_dist_map=edge_dist_map,
            edge_sensitivity=edge_sensitivity,
            orientation_strength=orientation_strength,
            skip_background=skip_background,
            progress_callback=_synth_progress,
            cancelled_callback=cancelled_callback,
        )

        if not scribble_pts_px:
            return []

        if cancelled_callback and cancelled_callback():
            return []

        if progress_callback:
            progress_callback(95)

        # --- Step 5: Convert pixel → mm canvas coordinates ---
        scribble_pts_mm: Polyline = [
            _px_to_mm(px, py, img_w, img_h, r_x1, r_y1, r_x2, r_y2)
            for px, py in scribble_pts_px
        ]

        # --- Step 6: Apply x/y offset ---
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            scribble_pts_mm = [(x + x_off, y + y_off) for x, y in scribble_pts_mm]

        # --- Step 7: Split into polyline chunks if very long ---
        # Very long single polylines can slow down the canvas renderer; split
        # into chunks while keeping each one large enough to be meaningful.
        max_pts = self._MAX_POLYLINE_PTS
        if len(scribble_pts_mm) <= max_pts:
            result = [scribble_pts_mm]
        else:
            result = []
            for start in range(0, len(scribble_pts_mm), max_pts):
                chunk: Polyline = scribble_pts_mm[start: start + max_pts]
                if len(chunk) >= 2:
                    result.append(chunk)

        if progress_callback:
            progress_callback(100)

        return result
