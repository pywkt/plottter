"""Tests for brush post-processing (Phase 38.1)."""

import math
import pytest

from plottter.processing.brush import (
    apply_brush,
    _circle_polyline,
    _polyline_length,
    _point_at_distance,
    _stipple_polyline,
)
import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _horizontal_line(length_mm: float = 10.0) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (length_mm, 0.0)]


def _vertical_line(length_mm: float = 10.0) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (0.0, length_mm)]


def _path_length(polyline: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


# ---------------------------------------------------------------------------
# circle_polyline
# ---------------------------------------------------------------------------

class TestCirclePolyline:
    def test_returns_closed_polygon(self):
        circle = _circle_polyline(0.0, 0.0, 1.0)
        assert len(circle) == 9  # 8 sides + closing point
        assert circle[0] == circle[-1]

    def test_radius_correct(self):
        cx, cy, r = 5.0, 3.0, 2.0
        circle = _circle_polyline(cx, cy, r)
        for x, y in circle[:-1]:
            dist = math.hypot(x - cx, y - cy)
            assert abs(dist - r) < 1e-10

    def test_center_offset(self):
        circle_origin = _circle_polyline(0.0, 0.0, 1.0)
        circle_offset = _circle_polyline(10.0, 20.0, 1.0)
        for (x0, y0), (x1, y1) in zip(circle_origin, circle_offset):
            assert abs((x1 - x0) - 10.0) < 1e-10
            assert abs((y1 - y0) - 20.0) < 1e-10


# ---------------------------------------------------------------------------
# _polyline_length
# ---------------------------------------------------------------------------

class TestPolylineLength:
    def test_horizontal_line(self):
        assert abs(_polyline_length(_horizontal_line(10.0)) - 10.0) < 1e-10

    def test_vertical_line(self):
        assert abs(_polyline_length(_vertical_line(5.0)) - 5.0) < 1e-10

    def test_diagonal(self):
        poly = [(0.0, 0.0), (3.0, 4.0)]
        assert abs(_polyline_length(poly) - 5.0) < 1e-10

    def test_empty_and_single_point(self):
        assert _polyline_length([]) == 0.0
        assert _polyline_length([(1.0, 2.0)]) == 0.0

    def test_multi_segment(self):
        # A path: (0,0) → (3,4) → (6,8); total = 5 + 5 = 10
        poly = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
        assert abs(_polyline_length(poly) - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# _point_at_distance
# ---------------------------------------------------------------------------

class TestPointAtDistance:
    def test_start(self):
        poly = _horizontal_line(10.0)
        pt = _point_at_distance(poly, 0.0)
        assert abs(pt[0] - 0.0) < 1e-10
        assert abs(pt[1] - 0.0) < 1e-10

    def test_midpoint(self):
        poly = _horizontal_line(10.0)
        pt = _point_at_distance(poly, 5.0)
        assert abs(pt[0] - 5.0) < 1e-10
        assert abs(pt[1] - 0.0) < 1e-10

    def test_end(self):
        poly = _horizontal_line(10.0)
        pt = _point_at_distance(poly, 10.0)
        assert abs(pt[0] - 10.0) < 1e-10

    def test_beyond_end_returns_last_point(self):
        poly = _horizontal_line(10.0)
        pt = _point_at_distance(poly, 100.0)
        assert pt == poly[-1]

    def test_empty_returns_none(self):
        assert _point_at_distance([], 0.0) is None

    def test_multi_segment_path(self):
        poly = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
        # At distance 5 we should be at (5, 0)
        pt = _point_at_distance(poly, 5.0)
        assert abs(pt[0] - 5.0) < 1e-6
        assert abs(pt[1] - 0.0) < 1e-6
        # At distance 7 we should be at (5, 2)
        pt = _point_at_distance(poly, 7.0)
        assert abs(pt[0] - 5.0) < 1e-6
        assert abs(pt[1] - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# _stipple_polyline
# ---------------------------------------------------------------------------

class TestStipplePolyline:
    def test_produces_dots_along_path(self):
        poly = _horizontal_line(10.0)
        rng = random.Random(0)
        dots = _stipple_polyline(poly, spacing_mm=1.0, size_mm=0.3, randomness=0.0, rng=rng)
        assert len(dots) > 0
        # Each dot should be a closed polygon
        for dot in dots:
            assert dot[0] == dot[-1]

    def test_spacing_controls_density(self):
        poly = _horizontal_line(10.0)
        rng_dense = random.Random(0)
        rng_sparse = random.Random(0)
        dense_dots = _stipple_polyline(poly, spacing_mm=0.5, size_mm=0.3, randomness=0.0, rng=rng_dense)
        sparse_dots = _stipple_polyline(poly, spacing_mm=2.0, size_mm=0.3, randomness=0.0, rng=rng_sparse)
        assert len(dense_dots) > len(sparse_dots)

    def test_no_randomness_produces_equally_spaced_dots(self):
        poly = _horizontal_line(10.0)
        rng = random.Random(0)
        spacing = 2.0
        dots = _stipple_polyline(poly, spacing_mm=spacing, size_mm=0.3, randomness=0.0, rng=rng)
        # Check that dot centers are at expected x positions (0.5*spacing, 1.5*spacing, ...)
        # Each dot is centered at its circle centre
        for i, dot in enumerate(dots):
            expected_x = (i + 0.5) * spacing
            # Average x of dot points (excluding the closing duplicate)
            avg_x = sum(x for x, _ in dot[:-1]) / len(dot[:-1])
            assert abs(avg_x - expected_x) < 1e-10, f"Dot {i}: expected x≈{expected_x}, got {avg_x}"

    def test_randomness_is_deterministic(self):
        poly = _horizontal_line(10.0)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        dots1 = _stipple_polyline(poly, spacing_mm=1.0, size_mm=0.3, randomness=0.5, rng=rng1)
        dots2 = _stipple_polyline(poly, spacing_mm=1.0, size_mm=0.3, randomness=0.5, rng=rng2)
        assert dots1 == dots2

    def test_short_path_no_crash(self):
        # Single point or very short path
        assert _stipple_polyline([], spacing_mm=1.0, size_mm=0.3, randomness=0.0, rng=random.Random(0)) == []
        assert _stipple_polyline([(0.0, 0.0)], spacing_mm=1.0, size_mm=0.3, randomness=0.0, rng=random.Random(0)) == []

    def test_dot_size_controls_radius(self):
        poly = _horizontal_line(10.0)
        rng = random.Random(0)
        dots = _stipple_polyline(poly, spacing_mm=2.0, size_mm=0.5, randomness=0.0, rng=rng)
        assert len(dots) > 0
        for dot in dots:
            cx = sum(x for x, _ in dot[:-1]) / len(dot[:-1])
            cy = sum(y for _, y in dot[:-1]) / len(dot[:-1])
            r = math.hypot(dot[0][0] - cx, dot[0][1] - cy)
            assert abs(r - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# apply_brush — the public API
# ---------------------------------------------------------------------------

class TestApplyBrush:
    def test_none_brush_returns_paths_unchanged(self):
        paths = [_horizontal_line(10.0), _vertical_line(5.0)]
        result = apply_brush(paths, "None", {})
        assert result is paths  # Same object reference (pass-through)

    def test_empty_brush_type_returns_unchanged(self):
        paths = [_horizontal_line()]
        result = apply_brush(paths, "", {})
        assert result is paths

    def test_stippled_brush_produces_dots(self):
        paths = [_horizontal_line(10.0)]
        result = apply_brush(paths, "Stippled", {"stipple_spacing_mm": 1.0, "stipple_size_mm": 0.3, "stipple_randomness": 0.0})
        assert len(result) > 0
        # All outputs should be closed circles (first == last)
        for dot in result:
            assert dot[0] == dot[-1]

    def test_stippled_spacing_controls_count(self):
        paths = [_horizontal_line(10.0)]
        dense = apply_brush(paths, "Stippled", {"stipple_spacing_mm": 0.5})
        sparse = apply_brush(paths, "Stippled", {"stipple_spacing_mm": 3.0})
        assert len(dense) > len(sparse)

    def test_stippled_multiple_paths(self):
        paths = [_horizontal_line(10.0), _vertical_line(10.0)]
        result = apply_brush(paths, "Stippled", {"stipple_spacing_mm": 2.0})
        # Each 10mm path with 2mm spacing should produce ~5 dots, so total ~10
        assert len(result) > 5

    def test_unknown_brush_type_returns_paths_unchanged(self):
        paths = [_horizontal_line()]
        result = apply_brush(paths, "NonExistentBrush", {})
        assert result is paths

    def test_multistroke_produces_multiple_strokes(self):
        # Default stroke_count=3: one input path → 3 output strokes
        paths = [_horizontal_line()]
        result = apply_brush(paths, "Multi-Stroke", {})
        assert len(result) > len(paths)
        assert result is not paths

    def test_calligraphic_produces_edge_lines(self):
        # Calligraphic returns centre + left + right edges per path
        paths = [_horizontal_line()]
        result = apply_brush(paths, "Calligraphic", {})
        assert len(result) > len(paths)
        assert result is not paths

    def test_empty_paths_list(self):
        result = apply_brush([], "Stippled", {})
        assert result == []

    def test_stippled_deterministic_per_path_index(self):
        # Running twice should produce identical results
        paths = [_horizontal_line(10.0), _vertical_line(10.0)]
        r1 = apply_brush(paths, "Stippled", {"stipple_spacing_mm": 1.0, "stipple_randomness": 0.5})
        r2 = apply_brush(paths, "Stippled", {"stipple_spacing_mm": 1.0, "stipple_randomness": 0.5})
        assert r1 == r2

    def test_package_level_import(self):
        from plottter.processing import apply_brush as pkg_apply_brush
        paths = [_horizontal_line()]
        assert pkg_apply_brush(paths, "None", {}) is paths
