"""Tests for tam._render_strokes() — 41.3."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.tam import _render_strokes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANVAS_W = 190.0
CANVAS_H = 277.0
STROKE_LEN = 5.0  # mm


def _chord(polyline: list[tuple[float, float]]) -> float:
    """Euclidean distance between first and last point."""
    x0, y0 = polyline[0]
    x1, y1 = polyline[-1]
    return math.hypot(x1 - x0, y1 - y0)


def _arc_len(polyline: list[tuple[float, float]]) -> float:
    """Sum of segment lengths."""
    total = 0.0
    for i in range(len(polyline) - 1):
        x0, y0 = polyline[i]
        x1, y1 = polyline[i + 1]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


# ---------------------------------------------------------------------------
# (a) Each stroke produces a polyline with at least 2 points
# ---------------------------------------------------------------------------


def test_at_least_two_points_straight():
    strokes = [(50.0, 80.0, 0.0), (100.0, 140.0, math.pi / 4)]
    polys = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=0.0
    )
    assert len(polys) == len(strokes)
    for poly in polys:
        assert len(poly) >= 2


def test_at_least_two_points_curved():
    strokes = [(50.0, 80.0, 0.0), (100.0, 140.0, math.pi / 4)]
    polys = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=1.0
    )
    assert len(polys) == len(strokes)
    for poly in polys:
        assert len(poly) >= 2


# ---------------------------------------------------------------------------
# (b) Stroke endpoints approximately stroke_length_mm apart
# ---------------------------------------------------------------------------


def test_endpoint_distance_straight():
    """Straight strokes: chord == stroke_length_mm exactly."""
    strokes = [
        (95.0, 138.5, 0.0),
        (95.0, 138.5, math.pi / 3),
        (10.0, 10.0, math.pi),
    ]
    polys = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=0.0
    )
    for poly in polys:
        assert _chord(poly) == pytest.approx(STROKE_LEN, rel=1e-9)


def test_endpoint_distance_curved_leq_arc():
    """Curved strokes: chord ≤ arc length ≈ stroke_length_mm."""
    strokes = [(95.0, 138.5, angle) for angle in [0.0, 0.5, 1.0, 1.5, 2.0]]
    polys = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=0.5, n_samples=7
    )
    for poly in polys:
        arc = _arc_len(poly)
        chord = _chord(poly)
        assert arc == pytest.approx(STROKE_LEN, rel=1e-9)
        assert chord <= STROKE_LEN + 1e-9


# ---------------------------------------------------------------------------
# (c) curvature=0 produces straight 2-point segments
# ---------------------------------------------------------------------------


def test_curvature_zero_two_point_segments():
    strokes = [(x * 10.0, 50.0, 0.3) for x in range(1, 6)]
    polys = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=0.0
    )
    for poly in polys:
        assert len(poly) == 2, f"Expected 2 points, got {len(poly)}"


def test_curvature_zero_is_straight_line():
    """The 2 points define a segment of the correct length and direction."""
    angle = math.pi / 6
    cx, cy = 100.0, 100.0
    strokes = [(cx, cy, angle)]
    poly = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=0.0
    )[0]
    half = STROKE_LEN / 2.0
    expected_start = (cx - math.cos(angle) * half, cy - math.sin(angle) * half)
    expected_end = (cx + math.cos(angle) * half, cy + math.sin(angle) * half)
    assert poly[0] == pytest.approx(expected_start, abs=1e-10)
    assert poly[1] == pytest.approx(expected_end, abs=1e-10)


# ---------------------------------------------------------------------------
# (d) curvature=1 produces curved multi-point strokes following the field
# ---------------------------------------------------------------------------


def test_curvature_one_multi_point():
    strokes = [(50.0, 80.0, 0.0)]
    polys = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=1.0, n_samples=7
    )
    assert len(polys[0]) == 7


def test_curvature_one_follows_field():
    """With a uniform field at angle φ, curvature=1 should walk along φ."""
    field_angle = math.pi / 3  # 60°
    # Uniform scalar field
    strokes = [(50.0, 50.0, 0.0)]  # initial angle differs from field_angle
    poly = _render_strokes(
        strokes, STROKE_LEN, field_angle, CANVAS_W, CANVAS_H,
        curvature=1.0, n_samples=7
    )[0]
    # With curvature=1, each step uses the field angle (π/3), not the
    # initial stroke angle (0). The segment directions should match the field.
    for i in range(len(poly) - 1):
        dx = poly[i + 1][0] - poly[i][0]
        dy = poly[i + 1][1] - poly[i][1]
        seg_angle = math.atan2(dy, dx)
        assert seg_angle == pytest.approx(field_angle, abs=1e-6)


def test_curvature_one_with_2d_field():
    """curvature=1 with a 2-D field bends away from the initial direction."""
    # A constant field pointing at π/2 (vertical, upward)
    H, W = 64, 64
    field = np.full((H, W), math.pi / 2, dtype=np.float64)
    # Initial stroke angle is 0 (horizontal)
    strokes = [(95.0, 138.5, 0.0)]
    poly = _render_strokes(
        strokes, STROKE_LEN, field, CANVAS_W, CANVAS_H,
        curvature=1.0, n_samples=7
    )[0]
    # After the first step, direction should be ~π/2 (vertical)
    dx = poly[2][0] - poly[1][0]
    dy = poly[2][1] - poly[1][1]
    seg_angle = math.atan2(dy, dx)
    assert seg_angle == pytest.approx(math.pi / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_strokes():
    polys = _render_strokes([], STROKE_LEN, 0.0, CANVAS_W, CANVAS_H)
    assert polys == []


def test_zero_length_strokes():
    strokes = [(50.0, 50.0, 0.5)]
    poly = _render_strokes(
        strokes, 0.0, 0.0, CANVAS_W, CANVAS_H, curvature=0.0
    )[0]
    assert len(poly) == 2
    assert poly[0] == pytest.approx(poly[1])


def test_n_samples_respected():
    strokes = [(50.0, 50.0, 0.0)]
    for n in [2, 5, 10]:
        poly = _render_strokes(
            strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H,
            curvature=0.5, n_samples=n
        )[0]
        assert len(poly) == n


def test_curvature_clamp():
    """Curvature values outside [0,1] are clamped without error."""
    strokes = [(50.0, 50.0, 0.0)]
    poly_neg = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=-5.0
    )[0]
    poly_over = _render_strokes(
        strokes, STROKE_LEN, 0.0, CANVAS_W, CANVAS_H, curvature=99.0, n_samples=7
    )[0]
    assert len(poly_neg) == 2  # clamped to 0
    assert len(poly_over) == 7  # clamped to 1
