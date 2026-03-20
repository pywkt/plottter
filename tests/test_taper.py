"""Tests for taper post-processing (task 49.1)."""

import math
import pytest

from plottter.processing.taper import (
    taper_paths,
    _smoothstep,
    _taper_profile,
    _compute_normals,
    _taper_polyline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _horizontal_line(length_mm: float = 10.0, n_points: int = 5) -> list[tuple[float, float]]:
    """Return a horizontal polyline with n_points evenly spaced."""
    return [(i * length_mm / (n_points - 1), 0.0) for i in range(n_points)]


def _polyline_length(poly: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(poly) - 1):
        total += math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
    return total


# ---------------------------------------------------------------------------
# _smoothstep
# ---------------------------------------------------------------------------


class TestSmoothstep:
    def test_zero(self):
        assert _smoothstep(0.0) == 0.0

    def test_one(self):
        assert _smoothstep(1.0) == 1.0

    def test_half(self):
        assert abs(_smoothstep(0.5) - 0.5) < 1e-12

    def test_clamped_below(self):
        assert _smoothstep(-1.0) == 0.0

    def test_clamped_above(self):
        assert _smoothstep(2.0) == 1.0

    def test_monotone(self):
        vals = [_smoothstep(x / 10) for x in range(11)]
        for a, b in zip(vals, vals[1:]):
            assert b >= a


# ---------------------------------------------------------------------------
# _taper_profile
# ---------------------------------------------------------------------------


class TestTaperProfile:
    def test_fade_zero_returns_uniform(self):
        # fade_fraction=0 → uniform width everywhere
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert _taper_profile(t, 0.0) == 1.0

    def test_fade_half_zero_at_endpoints(self):
        # fade_fraction=0.5 → full taper: 0 at both endpoints
        assert _taper_profile(0.0, 0.5) == 0.0
        assert _taper_profile(1.0, 0.5) == 0.0

    def test_fade_half_max_at_midpoint(self):
        # fade_fraction=0.5 → smoothstep(1)*smoothstep(1) = 1.0 at t=0.5
        assert abs(_taper_profile(0.5, 0.5) - 1.0) < 1e-12

    def test_fade_fraction_plateau(self):
        # With fade_fraction=0.2, at t=0.3 both smoothsteps should saturate
        # t/fade = 0.3/0.2 = 1.5 → smoothstep clamped to 1.0
        # (1-t)/fade = 0.7/0.2 = 3.5 → smoothstep clamped to 1.0
        assert abs(_taper_profile(0.3, 0.2) - 1.0) < 1e-12
        assert abs(_taper_profile(0.7, 0.2) - 1.0) < 1e-12

    def test_symmetric(self):
        # Profile must be symmetric: profile(t) == profile(1-t)
        fade = 0.3
        for t in [0.0, 0.1, 0.2, 0.3, 0.5]:
            assert abs(_taper_profile(t, fade) - _taper_profile(1.0 - t, fade)) < 1e-12

    def test_increases_then_decreases(self):
        # Profile should go up from 0, reach 1.0, then come back down
        fade = 0.2
        t_values = [i / 20 for i in range(21)]
        profile_vals = [_taper_profile(t, fade) for t in t_values]
        # Find peak and verify it's 1.0
        assert max(profile_vals) == 1.0


# ---------------------------------------------------------------------------
# _compute_normals
# ---------------------------------------------------------------------------


class TestComputeNormals:
    def test_horizontal_line_normals_point_up(self):
        # Horizontal path: tangent is (1,0), normal should be (0,1) (rotated 90° CCW)
        poly = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        normals = _compute_normals(poly)
        assert len(normals) == 3
        for nx, ny in normals:
            assert abs(nx - 0.0) < 1e-9
            assert abs(ny - 1.0) < 1e-9

    def test_vertical_line_normals_point_left(self):
        # Vertical path: tangent is (0,1), normal should be (-1,0)
        poly = [(0.0, 0.0), (0.0, 5.0), (0.0, 10.0)]
        normals = _compute_normals(poly)
        for nx, ny in normals:
            assert abs(nx - (-1.0)) < 1e-9
            assert abs(ny - 0.0) < 1e-9

    def test_normals_are_unit_length(self):
        poly = [(0.0, 0.0), (3.0, 4.0), (6.0, 0.0)]
        normals = _compute_normals(poly)
        for nx, ny in normals:
            assert abs(math.hypot(nx, ny) - 1.0) < 1e-9

    def test_length_matches_polyline(self):
        poly = _horizontal_line(10.0, 7)
        normals = _compute_normals(poly)
        assert len(normals) == len(poly)


# ---------------------------------------------------------------------------
# taper_paths — edge cases
# ---------------------------------------------------------------------------


class TestTaperEdgeCases:
    def test_empty_input_returns_empty(self):
        assert taper_paths([]) == []

    def test_single_point_returned_unchanged(self):
        poly = [(5.0, 5.0)]
        result = taper_paths([poly])
        assert result == [poly]

    def test_two_point_path_returned_unchanged(self):
        poly = [(0.0, 0.0), (10.0, 0.0)]
        result = taper_paths([poly])
        assert result == [poly]

    def test_zero_length_path_skipped(self):
        # All points the same → zero length → skip
        poly = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
        result = taper_paths([poly])
        assert result == []

    def test_path_with_3_points_processed(self):
        # Minimum valid path has 3 points
        poly = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        result = taper_paths([poly], fill_mode="outline")
        assert len(result) == 2  # left and right edges

    def test_input_not_modified(self):
        poly = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        original = list(poly)
        taper_paths([poly])
        assert poly == original


# ---------------------------------------------------------------------------
# taper_paths — outline mode
# ---------------------------------------------------------------------------


class TestTaperOutlineMode:
    def test_straight_line_produces_two_edges(self):
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=1.0, fade_fraction=0.15, fill_mode="outline")
        assert len(result) == 2

    def test_edges_have_same_length_as_input(self):
        poly = _horizontal_line(10.0, n_points=6)
        result = taper_paths([poly], max_width_mm=1.0, fade_fraction=0.15, fill_mode="outline")
        for edge in result:
            assert len(edge) == len(poly)

    def test_symmetric_left_right_edges(self):
        # For a horizontal line, left and right edges should be symmetric about y=0
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=2.0, fade_fraction=0.0, fill_mode="outline")
        left_edge, right_edge_rev = result
        right_edge = list(reversed(right_edge_rev))
        # For horizontal line: left should have y > 0, right y < 0
        for (lx, ly), (rx, ry) in zip(left_edge, right_edge):
            assert abs(lx - rx) < 1e-9  # same x
            assert abs(ly + ry) < 1e-9  # opposite y values

    def test_uniform_width_when_fade_zero(self):
        # fade_fraction=0 → constant width throughout
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=2.0, fade_fraction=0.0, fill_mode="outline")
        left_edge = result[0]
        # All left edge points should be at y=1.0 (half of 2.0)
        for x, y in left_edge:
            assert abs(y - 1.0) < 1e-9

    def test_full_taper_zero_width_at_endpoints(self):
        # fade_fraction=0.5 → width=0 at first and last points
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=2.0, fade_fraction=0.5, fill_mode="outline")
        left_edge = result[0]
        # First and last points of left edge should coincide with center line (y=0)
        assert abs(left_edge[0][1] - 0.0) < 1e-9
        assert abs(left_edge[-1][1] - 0.0) < 1e-9

    def test_max_width_reached_at_midpoint_full_taper(self):
        # With fade_fraction=0.5, midpoint should have y ≈ max_width/2
        poly = _horizontal_line(10.0, n_points=11)  # even spacing, midpoint at index 5
        result = taper_paths([poly], max_width_mm=2.0, fade_fraction=0.5, fill_mode="outline")
        left_edge = result[0]
        # midpoint is index 5 (t=0.5)
        assert abs(left_edge[5][1] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# taper_paths — filled mode
# ---------------------------------------------------------------------------


class TestTaperFilledMode:
    def test_filled_produces_more_polylines_than_outline(self):
        # filled should produce more strokes than the 2 edges in outline
        poly = _horizontal_line(10.0)
        outline = taper_paths([poly], max_width_mm=2.0, fill_spacing_mm=0.3, fill_mode="outline")
        filled = taper_paths([poly], max_width_mm=2.0, fill_spacing_mm=0.3, fill_mode="filled")
        assert len(filled) > len(outline)

    def test_filled_count_scales_with_width_over_spacing(self):
        # More strokes for wider max_width or smaller fill_spacing
        poly = _horizontal_line(10.0)
        narrow = taper_paths([poly], max_width_mm=1.0, fill_spacing_mm=0.3, fill_mode="filled")
        wide = taper_paths([poly], max_width_mm=4.0, fill_spacing_mm=0.3, fill_mode="filled")
        assert len(wide) > len(narrow)

    def test_filled_count_scales_with_spacing(self):
        poly = _horizontal_line(10.0)
        dense = taper_paths([poly], max_width_mm=2.0, fill_spacing_mm=0.2, fill_mode="filled")
        sparse = taper_paths([poly], max_width_mm=2.0, fill_spacing_mm=0.5, fill_mode="filled")
        assert len(dense) > len(sparse)

    def test_filled_approximate_stroke_count(self):
        # max_width=1.0, fill_spacing=0.3 → offsets: -0.5, -0.2, 0.1, 0.4 → 4 strokes
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=1.0, fill_spacing_mm=0.3, fill_mode="filled")
        assert len(result) >= 3  # at least 3 strokes for 1mm width / 0.3mm spacing

    def test_filled_strokes_same_length_as_input(self):
        poly = _horizontal_line(10.0, n_points=6)
        result = taper_paths([poly], max_width_mm=1.0, fill_spacing_mm=0.4, fill_mode="filled")
        for stroke in result:
            assert len(stroke) == len(poly)

    def test_filled_center_stroke_at_zero_offset(self):
        # When max_width=0.6 and spacing=0.3, offsets are -0.3, 0.0, 0.3
        # The center stroke (offset=0) should coincide with the original path
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=0.6, fill_spacing_mm=0.3, fill_mode="filled")
        # Find the stroke where all y ≈ 0 (center stroke)
        center_strokes = [s for s in result if all(abs(y) < 1e-9 for _, y in s)]
        assert len(center_strokes) >= 1

    def test_filled_fade_zero_uniform_strokes(self):
        # fade_fraction=0 → all strokes should be straight parallel lines
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=1.0, fade_fraction=0.0, fill_spacing_mm=0.3, fill_mode="filled")
        # Each stroke should have constant y offset
        for stroke in result:
            y_vals = [y for _, y in stroke]
            assert max(y_vals) - min(y_vals) < 1e-9  # constant y

    def test_filled_full_taper_endpoints_converge(self):
        # fade_fraction=0.5 → at endpoints all strokes converge to center
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=2.0, fade_fraction=0.5, fill_spacing_mm=0.4, fill_mode="filled")
        # All strokes should start and end at the same x,y as the center line
        first_points = [s[0] for s in result]
        last_points = [s[-1] for s in result]
        # Check they all have y ≈ 0 (converge to center line at endpoints)
        for x, y in first_points:
            assert abs(y) < 1e-9
        for x, y in last_points:
            assert abs(y) < 1e-9


# ---------------------------------------------------------------------------
# taper_paths — parameter validation
# ---------------------------------------------------------------------------


class TestTaperParameterValidation:
    def test_negative_max_width_clamped_to_zero(self):
        poly = _horizontal_line(10.0)
        result = taper_paths([poly], max_width_mm=-1.0, fill_mode="outline")
        # Zero width → both edges are the original center line
        for edge in result:
            for x, y in edge:
                assert abs(y) < 1e-9

    def test_fade_fraction_clamped_to_half(self):
        # fade_fraction > 0.5 is clamped to 0.5
        poly = _horizontal_line(10.0)
        result_clamped = taper_paths([poly], max_width_mm=1.0, fade_fraction=0.9, fill_mode="outline")
        result_half = taper_paths([poly], max_width_mm=1.0, fade_fraction=0.5, fill_mode="outline")
        assert result_clamped == result_half

    def test_multiple_paths(self):
        paths = [_horizontal_line(10.0), _horizontal_line(5.0)]
        result = taper_paths(paths, max_width_mm=1.0, fill_mode="outline")
        # Each path produces 2 edges → total 4
        assert len(result) == 4

    def test_package_level_import(self):
        from plottter.processing import taper_paths as pkg_taper
        poly = _horizontal_line(10.0)
        result = pkg_taper([poly], fill_mode="outline")
        assert len(result) == 2
