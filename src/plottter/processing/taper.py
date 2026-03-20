"""Taper post-processing: generate variable-width tapered stroke outlines.

Each input polyline is replaced by either:
  - "outline" mode: two edge polylines (left + reversed right) forming a closed outline
  - "filled" mode:  N parallel strokes evenly spaced from edge to edge

The width follows a smooth taper profile that fades in and out over configurable
fractions of the path length, using a smoothstep (3x²-2x³) ease function.
"""

from __future__ import annotations

import math

from plottter.models.path import Polyline, Point


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------


def _smoothstep(x: float) -> float:
    """Clamp x to [0,1] and apply smoothstep: 3x² - 2x³."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _taper_profile(t: float, fade_fraction: float) -> float:
    """Return width multiplier in [0,1] at normalized path position t ∈ [0,1].

    Fades in over [0, fade_fraction] and fades out over [1-fade_fraction, 1].

    Args:
        t: Normalized arc-length position along the path (0 = start, 1 = end).
        fade_fraction: Fraction of path used for each fade (0.0–0.5).
            0.0 → uniform width (profile = 1.0 everywhere).
            0.5 → full taper: zero width at both endpoints, max at midpoint.
    """
    if fade_fraction <= 0.0:
        return 1.0
    return _smoothstep(t / fade_fraction) * _smoothstep((1.0 - t) / fade_fraction)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _compute_normals(polyline: Polyline) -> list[tuple[float, float]]:
    """Return unit normals perpendicular to travel direction at each point.

    Uses forward difference at the start, backward difference at the end,
    and central difference at interior points.
    """
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
            # Degenerate segment: reuse previous normal or use (0, 1)
            normals.append(normals[-1] if normals else (0.0, 1.0))
    return normals


# ---------------------------------------------------------------------------
# Core taper logic
# ---------------------------------------------------------------------------


def _taper_polyline(
    polyline: Polyline,
    max_width_mm: float,
    fade_fraction: float,
    fill_spacing_mm: float,
    fill_mode: str,
) -> list[Polyline]:
    """Apply taper to a single polyline.

    Args:
        polyline: Input path.
        max_width_mm: Maximum total stroke width in mm at the widest point.
        fade_fraction: Fraction of path for fade in/out (0.0–0.5).
        fill_spacing_mm: Spacing between parallel strokes in "filled" mode.
        fill_mode: "outline" or "filled".

    Returns:
        List of output polylines. Paths with <3 points are returned unchanged.
        Zero-length paths return an empty list.
    """
    n = len(polyline)
    # Edge case: fewer than 3 points → return unchanged
    if n < 3:
        return [polyline]

    # Compute cumulative arc length
    cum_lengths: list[float] = [0.0]
    for i in range(1, n):
        dx = polyline[i][0] - polyline[i - 1][0]
        dy = polyline[i][1] - polyline[i - 1][1]
        cum_lengths.append(cum_lengths[-1] + math.hypot(dx, dy))
    total_length = cum_lengths[-1]

    # Edge case: zero-length path → skip
    if total_length < 1e-9:
        return []

    # Normalized arc-length parameter t ∈ [0, 1] at each vertex
    t_values = [s / total_length for s in cum_lengths]

    # Unit normals at each vertex
    normals = _compute_normals(polyline)

    half_max = max_width_mm / 2.0

    if fill_mode == "outline":
        # Return two edge polylines: left edge and reversed right edge
        left_edge: Polyline = []
        right_edge: Polyline = []
        for i, (px, py) in enumerate(polyline):
            nx, ny = normals[i]
            hw = half_max * _taper_profile(t_values[i], fade_fraction)
            left_edge.append((px + nx * hw, py + ny * hw))
            right_edge.append((px - nx * hw, py - ny * hw))

        result: list[Polyline] = []
        if len(left_edge) >= 2:
            result.append(left_edge)
        if len(right_edge) >= 2:
            result.append(list(reversed(right_edge)))
        return result

    else:
        # "filled" mode: N parallel strokes at evenly-spaced offsets
        # Offsets range from -half_max to +half_max in steps of fill_spacing_mm
        offsets: list[float] = []
        o = -half_max
        while o <= half_max + 1e-9:
            offsets.append(o)
            o += fill_spacing_mm
        if not offsets:
            offsets = [0.0]

        result = []
        for offset in offsets:
            stroke: Polyline = []
            for i, (px, py) in enumerate(polyline):
                nx, ny = normals[i]
                # Scale offset by taper profile so strokes converge at ends
                actual_offset = offset * _taper_profile(t_values[i], fade_fraction)
                stroke.append((px + nx * actual_offset, py + ny * actual_offset))
            if len(stroke) >= 2:
                result.append(stroke)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def taper_paths(
    paths: list[Polyline],
    max_width_mm: float = 1.0,
    fade_fraction: float = 0.15,
    fill_spacing_mm: float = 0.3,
    fill_mode: str = "filled",
) -> list[Polyline]:
    """Apply taper effect to a list of polylines.

    Each polyline is replaced by a tapered stroke: the stroke starts narrow,
    widens to *max_width_mm* over the first *fade_fraction* of its length, stays
    wide, then narrows again over the last *fade_fraction*.

    Args:
        paths: Input polylines in mm coordinates. Not modified.
        max_width_mm: Maximum total stroke width in mm at the widest point.
        fade_fraction: Fraction of path length for each fade (0.0–0.5).
            0.0 → uniform width (no taper).
            0.5 → full taper (width = 0 at both endpoints).
        fill_spacing_mm: Spacing between parallel fill strokes in "filled" mode.
        fill_mode:
            "outline" — 2 edge polylines per input (fast, minimal ink).
            "filled"  — N parallel strokes per input (solid fill, more ink).

    Returns:
        New list of polylines with the taper applied.
    """
    max_width_mm = max(0.0, float(max_width_mm))
    fade_fraction = max(0.0, min(0.5, float(fade_fraction)))
    fill_spacing_mm = max(0.01, float(fill_spacing_mm))

    result: list[Polyline] = []
    for poly in paths:
        result.extend(
            _taper_polyline(poly, max_width_mm, fade_fraction, fill_spacing_mm, fill_mode)
        )
    return result
