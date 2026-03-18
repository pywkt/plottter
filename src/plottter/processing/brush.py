"""Brush post-processing: replace plain strokes with stylized variants.

Supported brush types:
  - "None"        — pass through unchanged
  - "Stippled"    — replace each polyline with dots placed along its length
  - "Multi-Stroke"  — draw each polyline multiple times with slight offsets (sketchy look)
  - "Calligraphic"  — variable-width stroke based on nib angle (thick/thin contrast)
"""

from __future__ import annotations

import math
import random

try:
    import noise as _noise_lib
    _HAS_NOISE = True
except ImportError:
    _HAS_NOISE = False

from plottter.models.path import Polyline, Point

_DOT_SIDES = 8  # Number of polygon sides for dot circles


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _circle_polyline(cx: float, cy: float, radius_mm: float) -> Polyline:
    """Return an 8-sided polygon approximating a circle."""
    pts: Polyline = []
    for k in range(_DOT_SIDES):
        angle = 2.0 * math.pi * k / _DOT_SIDES
        pts.append((cx + radius_mm * math.cos(angle), cy + radius_mm * math.sin(angle)))
    pts.append(pts[0])  # Close the polygon
    return pts


def _polyline_length(polyline: Polyline) -> float:
    """Return the total arc length of a polyline in mm."""
    total = 0.0
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _point_at_distance(polyline: Polyline, target: float) -> Point | None:
    """Return the interpolated point at *target* mm along the polyline.

    Returns None if the polyline is empty. If target exceeds the total length,
    returns the last point.
    """
    if not polyline:
        return None
    accumulated = 0.0
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if accumulated + seg_len >= target:
            t = (target - accumulated) / seg_len if seg_len > 0.0 else 0.0
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        accumulated += seg_len
    return polyline[-1]


# ---------------------------------------------------------------------------
# Stippled brush
# ---------------------------------------------------------------------------


def _stipple_polyline(
    polyline: Polyline,
    spacing_mm: float,
    size_mm: float,
    randomness: float,
    rng: random.Random,
) -> list[Polyline]:
    """Replace one polyline with dot circles placed along its length.

    Args:
        polyline: Input path.
        spacing_mm: Distance between consecutive dot centres.
        size_mm: Dot radius in mm.
        randomness: 0–1 amount of random variation in position and size.
        rng: Seeded RNG for reproducibility.

    Returns:
        List of circle polylines (one per dot).
    """
    if len(polyline) < 2:
        return []
    total_len = _polyline_length(polyline)
    if total_len <= 0.0:
        return []

    dots: list[Polyline] = []
    pos = spacing_mm * 0.5  # Start half a spacing in so dots don't cluster at the end

    while pos <= total_len:
        pt = _point_at_distance(polyline, pos)
        if pt is None:
            break
        cx, cy = pt

        if randomness > 0.0:
            offset_scale = randomness * spacing_mm
            cx += rng.uniform(-offset_scale, offset_scale)
            cy += rng.uniform(-offset_scale, offset_scale)
            size_scale = 1.0 + rng.uniform(-randomness, randomness)
            dot_size = max(0.05, size_mm * size_scale)
        else:
            dot_size = size_mm

        dots.append(_circle_polyline(cx, cy, dot_size))
        pos += spacing_mm

    return dots


def _apply_stippled(paths: list[Polyline], params: dict) -> list[Polyline]:
    """Apply the stippled brush to a list of polylines.

    Parameters (read from *params*):
        stipple_spacing_mm (float, default 1.0): Distance between dots.
        stipple_size_mm (float, default 0.3): Dot radius.
        stipple_randomness (float 0–1, default 0.2): Position/size variation.
    """
    spacing = max(0.01, float(params.get("stipple_spacing_mm", 1.0)))
    size = max(0.01, float(params.get("stipple_size_mm", 0.3)))
    randomness = max(0.0, min(1.0, float(params.get("stipple_randomness", 0.2))))

    result: list[Polyline] = []
    for i, poly in enumerate(paths):
        rng = random.Random(i)  # Deterministic per path index
        result.extend(_stipple_polyline(poly, spacing, size, randomness, rng))
    return result


# ---------------------------------------------------------------------------
# Multi-stroke brush
# ---------------------------------------------------------------------------


def _perlin2(x: float, y: float, seed: int) -> float:
    """Return a smooth noise value in [-1, 1] using Perlin noise or fallback."""
    if _HAS_NOISE:
        return _noise_lib.pnoise2(x, y, base=seed % 256)
    # Fallback: deterministic pseudo-noise from hash
    h = hash((round(x * 1000), round(y * 1000), seed)) & 0xFFFFFF
    return (h / 0x7FFFFF) - 1.0


def _multi_stroke_polyline(
    polyline: Polyline,
    stroke_count: int,
    spread_mm: float,
    stroke_noise: float,
    path_seed: int,
) -> list[Polyline]:
    """Return *stroke_count* noisy copies of *polyline*.

    For each copy, every point is offset by a Perlin-noise-derived amount
    perpendicular to the stroke, creating a hand-drawn sketchy look.

    Args:
        polyline: Input path.
        stroke_count: Number of parallel strokes to draw (including center).
        spread_mm: Maximum lateral offset in mm.
        stroke_noise: 0–1 intensity of per-point noise variation.
        path_seed: RNG seed derived from the path index.

    Returns:
        List of offset polylines.
    """
    if len(polyline) < 2 or stroke_count <= 0:
        return []

    result: list[Polyline] = []
    for copy_idx in range(stroke_count):
        if stroke_count == 1:
            lateral_base = 0.0
        else:
            # Distribute copies evenly from -spread to +spread
            lateral_base = spread_mm * (2.0 * copy_idx / (stroke_count - 1) - 1.0) if stroke_count > 1 else 0.0

        new_poly: Polyline = []
        for pt_idx, (px, py) in enumerate(polyline):
            # Per-point noise offset using Perlin noise
            noise_val = _perlin2(pt_idx * 0.1, float(copy_idx), path_seed) * stroke_noise
            lateral_offset = lateral_base + noise_val * spread_mm

            # Compute perpendicular direction at this point
            if pt_idx < len(polyline) - 1:
                nx, ny = polyline[pt_idx + 1]
                dx, dy = nx - px, ny - py
            elif pt_idx > 0:
                nx, ny = polyline[pt_idx - 1]
                dx, dy = px - nx, py - ny
            else:
                dx, dy = 1.0, 0.0

            length = math.hypot(dx, dy)
            if length > 1e-9:
                # Perpendicular direction: (-dy, dx) normalised
                perp_x = -dy / length
                perp_y = dx / length
            else:
                perp_x, perp_y = 0.0, 0.0

            new_poly.append((px + perp_x * lateral_offset, py + perp_y * lateral_offset))

        if len(new_poly) >= 2:
            result.append(new_poly)
    return result


def _apply_multi_stroke(paths: list[Polyline], params: dict) -> list[Polyline]:
    """Apply the multi-stroke brush to a list of polylines.

    Parameters (read from *params*):
        stroke_count (int, default 3): Number of parallel strokes.
        stroke_spread_mm (float, default 0.5): Max lateral offset.
        stroke_noise (float 0–1, default 0.3): Per-point noise intensity.
    """
    stroke_count = max(1, int(params.get("stroke_count", 3)))
    spread_mm = max(0.0, float(params.get("stroke_spread_mm", 0.5)))
    stroke_noise = max(0.0, min(1.0, float(params.get("stroke_noise", 0.3))))

    result: list[Polyline] = []
    for i, poly in enumerate(paths):
        result.extend(_multi_stroke_polyline(poly, stroke_count, spread_mm, stroke_noise, i))
    return result


# ---------------------------------------------------------------------------
# Calligraphic brush
# ---------------------------------------------------------------------------


def _compute_segment_normals(polyline: Polyline) -> list[tuple[float, float]]:
    """Return unit normals (perpendicular to travel direction) for each point."""
    n = len(polyline)
    normals: list[tuple[float, float]] = []
    for i in range(n):
        if i == 0:
            dx = polyline[1][0] - polyline[0][0]
            dy = polyline[1][1] - polyline[0][1]
        elif i == n - 1:
            dx = polyline[-1][0] - polyline[-2][0]
            dy = polyline[-1][1] - polyline[-2][1]
        else:
            dx = polyline[i + 1][0] - polyline[i - 1][0]
            dy = polyline[i + 1][1] - polyline[i - 1][1]
        length = math.hypot(dx, dy)
        if length > 1e-9:
            normals.append((-dy / length, dx / length))
        else:
            normals.append((0.0, 0.0) if not normals else normals[-1])
    return normals


def _segment_angle(polyline: Polyline, pt_idx: int) -> float:
    """Return the angle (radians) of the travel direction at point *pt_idx*."""
    n = len(polyline)
    if n < 2:
        return 0.0
    if pt_idx < n - 1:
        dx = polyline[pt_idx + 1][0] - polyline[pt_idx][0]
        dy = polyline[pt_idx + 1][1] - polyline[pt_idx][1]
    else:
        dx = polyline[-1][0] - polyline[-2][0]
        dy = polyline[-1][1] - polyline[-2][1]
    return math.atan2(dy, dx)


def _calligraphic_polyline(
    polyline: Polyline,
    nib_angle_deg: float,
    max_width_mm: float,
    min_width_mm: float,
) -> list[Polyline]:
    """Return parallel offset polylines simulating a flat nib calligraphic stroke.

    The stroke is thick when the direction is perpendicular to the nib angle,
    and thin when parallel, exactly like a flat-nib calligraphy pen.

    Args:
        polyline: Input path (centre line).
        nib_angle_deg: Pen nib angle in degrees (0 = horizontal nib).
        max_width_mm: Total stroke width when perpendicular to nib (full width).
        min_width_mm: Total stroke width when parallel to nib (full width).

    Returns:
        List containing the centre line plus left and right edge offset polylines.
    """
    if len(polyline) < 2:
        return list([polyline]) if polyline else []

    nib_rad = math.radians(nib_angle_deg)
    normals = _compute_segment_normals(polyline)

    left_edge: Polyline = []
    right_edge: Polyline = []

    for i, (px, py) in enumerate(polyline):
        travel_angle = _segment_angle(polyline, i)
        # Total width based on angle between travel direction and nib angle
        diff = travel_angle - nib_rad
        total_width = min_width_mm + (max_width_mm - min_width_mm) * abs(math.sin(diff))
        half_width = total_width / 2.0

        nx, ny = normals[i]
        left_edge.append((px + nx * half_width, py + ny * half_width))
        right_edge.append((px - nx * half_width, py - ny * half_width))

    result: list[Polyline] = [polyline]  # centre line
    if len(left_edge) >= 2:
        result.append(left_edge)
    if len(right_edge) >= 2:
        result.append(right_edge)
    return result


def _apply_calligraphic(paths: list[Polyline], params: dict) -> list[Polyline]:
    """Apply the calligraphic brush to a list of polylines.

    Parameters (read from *params*):
        nib_angle (float, default 45): Nib angle in degrees.
        nib_width_mm (float, default 1.5): Maximum total stroke width.
        min_width_mm (float, default 0.2): Minimum total stroke width.
    """
    nib_angle = float(params.get("nib_angle", 45.0))
    nib_width_mm = max(0.05, float(params.get("nib_width_mm", 1.5)))
    min_width_mm = max(0.01, float(params.get("min_width_mm", 0.2)))
    # Ensure min <= max
    min_width_mm = min(min_width_mm, nib_width_mm)

    result: list[Polyline] = []
    for poly in paths:
        result.extend(_calligraphic_polyline(poly, nib_angle, nib_width_mm, min_width_mm))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_brush(paths: list[Polyline], brush_type: str, params: dict) -> list[Polyline]:
    """Apply a brush effect to a list of polylines.

    Args:
        paths: Input polylines in mm coordinates.
        brush_type: One of "None", "Stippled", "Multi-Stroke", "Calligraphic".
        params: Brush-specific parameter dict. Keys depend on brush_type.

    Returns:
        New list of polylines with the brush applied. The input list is not modified.
    """
    if not brush_type or brush_type == "None":
        return paths
    if brush_type == "Stippled":
        return _apply_stippled(paths, params)
    if brush_type == "Multi-Stroke":
        return _apply_multi_stroke(paths, params)
    if brush_type == "Calligraphic":
        return _apply_calligraphic(paths, params)
    return paths
