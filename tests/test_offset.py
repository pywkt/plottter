"""Tests for the offset_paths post-processing function."""

from __future__ import annotations

import math
import pytest

from plottter.processing.offset import offset_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _polyline_length(poly: list[tuple[float, float]]) -> float:
    return sum(_dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def _point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Perpendicular distance from point (px, py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _min_dist_to_polyline(point: tuple[float, float], poly: list[tuple[float, float]]) -> float:
    """Minimum distance from a point to any segment of a polyline."""
    px, py = point
    return min(
        _point_to_segment_dist(px, py, poly[i][0], poly[i][1], poly[i + 1][0], poly[i + 1][1])
        for i in range(len(poly) - 1)
    )


def make_square(size: float = 10.0) -> list[tuple[float, float]]:
    """Return a closed square polyline (5 points, first == last)."""
    s = size / 2.0
    return [(-s, -s), (s, -s), (s, s), (-s, s), (-s, -s)]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestOffsetBasic:
    def test_empty_input_returns_empty(self) -> None:
        result = offset_paths([])
        assert result == []

    def test_single_point_path_skipped(self) -> None:
        result = offset_paths([[(5.0, 5.0)]], include_original=False)
        assert result == []

    def test_single_point_path_skipped_with_include_original(self) -> None:
        """Degenerate paths (<2 points) are always skipped, even include_original=True."""
        result = offset_paths([[(5.0, 5.0)]], include_original=True)
        assert result == []

    def test_include_original_true(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=1, include_original=True)
        # Should contain: original + left offset + right offset = 3 paths
        assert len(result) >= 2
        # Original must be among results
        assert any(p == line for p in result)

    def test_include_original_false(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=1, include_original=False)
        # Original must NOT be among results
        assert not any(p == line for p in result)

    def test_very_small_distance_skipped(self) -> None:
        """Distance < 0.01mm is skipped — returns only original if include_original."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=0.005, include_original=True)
        assert len(result) == 1
        assert result[0] == line

    def test_zero_distance_skipped(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=0.0, include_original=True)
        assert len(result) == 1

    def test_package_import(self) -> None:
        from plottter.processing import offset_paths as pkg_offset
        assert pkg_offset is offset_paths


# ---------------------------------------------------------------------------
# (a) Straight line: parallel offsets at correct distance
# ---------------------------------------------------------------------------

class TestStraightLineOffset:
    def test_both_sides_produce_parallel_lines(self) -> None:
        """A horizontal line offset both sides → two parallel lines at distance d."""
        d = 2.0
        line = [(0.0, 0.0), (20.0, 0.0)]
        result = offset_paths([line], distance_mm=d, sides="both",
                               count=1, include_original=False)
        assert len(result) == 2

        # Each offset should be a horizontal line at y ≈ ±d
        for poly in result:
            ys = [p[1] for p in poly]
            avg_y = sum(ys) / len(ys)
            assert abs(abs(avg_y) - d) < 0.2, f"Expected |y| ≈ {d}, got {avg_y}"

    def test_offset_distance_accuracy(self) -> None:
        """Every point on the offset curve must be ~d mm from the original line."""
        d = 1.5
        line = [(0.0, 0.0), (20.0, 0.0)]
        result = offset_paths([line], distance_mm=d, sides="left",
                               count=1, include_original=False)
        assert len(result) >= 1
        offset = result[0]
        for pt in offset:
            dist = _min_dist_to_polyline(pt, line)
            assert abs(dist - d) < 0.2, f"Point {pt} is {dist:.3f}mm from line, expected {d}"

    def test_left_side_only(self) -> None:
        """sides='left' produces exactly one offset."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="left",
                               count=1, include_original=False)
        assert len(result) == 1

    def test_right_side_only(self) -> None:
        """sides='right' produces exactly one offset."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="right",
                               count=1, include_original=False)
        assert len(result) == 1

    def test_left_and_right_opposite_sides(self) -> None:
        """Left and right offsets of a horizontal line should be on opposite sides."""
        d = 2.0
        line = [(0.0, 0.0), (20.0, 0.0)]
        left = offset_paths([line], distance_mm=d, sides="left",
                             count=1, include_original=False)
        right = offset_paths([line], distance_mm=d, sides="right",
                              count=1, include_original=False)
        assert len(left) == 1
        assert len(right) == 1
        left_y = sum(p[1] for p in left[0]) / len(left[0])
        right_y = sum(p[1] for p in right[0]) / len(right[0])
        # They should be on opposite sides
        assert left_y * right_y < 0, "Left and right offsets should be on opposite Y sides"


# ---------------------------------------------------------------------------
# (b) Closed square: concentric rings
# ---------------------------------------------------------------------------

class TestClosedPathOffset:
    def test_square_offset_both_sides(self) -> None:
        """Offset a square both sides → one larger and one smaller square."""
        sq = make_square(10.0)
        d = 2.0
        result = offset_paths([sq], distance_mm=d, sides="both",
                               count=1, include_original=False)
        # Should produce 2 rings
        assert len(result) >= 2

    def test_square_outer_is_larger(self) -> None:
        """Left offset (outside) should produce a ring with larger bounding box."""
        sq = make_square(10.0)
        d = 2.0
        result = offset_paths([sq], distance_mm=d, sides="left",
                               count=1, include_original=False)
        assert len(result) >= 1
        outer = result[0]
        # Outer ring should have points farther from origin than original's corners (5mm)
        max_dist = max(math.hypot(p[0], p[1]) for p in outer)
        assert max_dist > 5.0 + d * 0.5, f"Outer ring not large enough: max_dist={max_dist}"

    def test_square_inner_is_smaller(self) -> None:
        """Right offset (inside) should produce a ring with smaller bounding box."""
        sq = make_square(10.0)
        d = 1.5
        result = offset_paths([sq], distance_mm=d, sides="right",
                               count=1, include_original=False)
        assert len(result) >= 1
        inner = result[0]
        max_dist = max(math.hypot(p[0], p[1]) for p in inner)
        # Inner ring corners should be ~(5 - 1.5) * sqrt(2) from origin
        # Just verify they're inside the original 10x10 square (corners at ~7.07mm)
        assert max_dist < 5.0 * math.sqrt(2), f"Inner ring not smaller enough: max_dist={max_dist}"

    def test_closed_path_detection(self) -> None:
        """A polyline with first==last should be treated as closed."""
        sq = make_square(10.0)
        assert sq[0] == sq[-1]  # sanity
        result = offset_paths([sq], distance_mm=1.0, sides="both",
                               count=1, include_original=True)
        # Should contain original plus some offsets
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# (c) count=3 produces 3 offset copies per side
# ---------------------------------------------------------------------------

class TestOffsetCount:
    def test_count_3_both_sides_6_offsets(self) -> None:
        """count=3, sides='both' → 6 offset copies (3 left + 3 right)."""
        line = [(0.0, 0.0), (20.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=3, include_original=False)
        assert len(result) == 6

    def test_count_3_left_only_3_offsets(self) -> None:
        """count=3, sides='left' → 3 offset copies."""
        line = [(0.0, 0.0), (20.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="left",
                               count=3, include_original=False)
        assert len(result) == 3

    def test_count_3_distances_are_multiples(self) -> None:
        """Offsets at count=3 should be at 1d, 2d, 3d from the original."""
        d = 1.0
        line = [(0.0, 0.0), (20.0, 0.0)]
        result = offset_paths([line], distance_mm=d, sides="left",
                               count=3, include_original=False)
        assert len(result) == 3
        # Get average y (distance from x-axis) for each offset
        avg_ys = sorted(abs(sum(p[1] for p in poly) / len(poly)) for poly in result)
        for i, avg_y in enumerate(avg_ys):
            expected = d * (i + 1)
            assert abs(avg_y - expected) < 0.3, f"Offset {i+1}: expected y≈{expected}, got {avg_y:.3f}"

    def test_count_1_default(self) -> None:
        """Default count=1 produces 2 offset copies (one per side) for sides='both'."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=1, include_original=False)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# (d) sides='left' vs sides='right'
# ---------------------------------------------------------------------------

class TestSideSelection:
    def test_left_only_one_side(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="left",
                               count=1, include_original=False)
        assert len(result) == 1

    def test_right_only_one_side(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="right",
                               count=1, include_original=False)
        assert len(result) == 1

    def test_both_two_sides(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=1, include_original=False)
        assert len(result) == 2

    def test_left_offset_positive_y(self) -> None:
        """For a rightward horizontal line, left offset should be at positive y."""
        d = 2.0
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=d, sides="left",
                               count=1, include_original=False)
        assert len(result) == 1
        avg_y = sum(p[1] for p in result[0]) / len(result[0])
        # Left of a rightward line = positive y in standard coordinates
        assert avg_y > 0, f"Left offset should have positive y, got {avg_y}"

    def test_right_offset_negative_y(self) -> None:
        """For a rightward horizontal line, right offset should be at negative y."""
        d = 2.0
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=d, sides="right",
                               count=1, include_original=False)
        assert len(result) == 1
        avg_y = sum(p[1] for p in result[0]) / len(result[0])
        assert avg_y < 0, f"Right offset should have negative y, got {avg_y}"


# ---------------------------------------------------------------------------
# (e) round join produces curves at corners
# ---------------------------------------------------------------------------

class TestJoinStyle:
    def test_round_join_more_points_at_corner(self) -> None:
        """Round join should produce more points at a sharp corner than bevel/mitre."""
        # L-shaped path with a 90-degree corner
        path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        result_round = offset_paths([path], distance_mm=2.0, sides="left",
                                    count=1, join_style="round", include_original=False)
        result_bevel = offset_paths([path], distance_mm=2.0, sides="left",
                                    count=1, join_style="bevel", include_original=False)

        if result_round and result_bevel:
            # Round should have at least as many points (curved corner vs flat cut)
            assert len(result_round[0]) >= len(result_bevel[0])

    def test_all_join_styles_produce_output(self) -> None:
        """All three join styles should produce valid output."""
        path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        for style in ("round", "mitre", "bevel"):
            result = offset_paths([path], distance_mm=1.0, sides="left",
                                  count=1, join_style=style, include_original=False)
            assert len(result) >= 1, f"join_style='{style}' produced no output"
            assert len(result[0]) >= 2

    def test_round_join_preserves_distance(self) -> None:
        """Round join offset points should be approximately d from the original path."""
        d = 1.5
        path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        result = offset_paths([path], distance_mm=d, sides="left",
                               count=1, join_style="round", include_original=False)
        assert len(result) >= 1
        # Check a few interior points of the offset (skip endpoints — at corners)
        offset = result[0]
        for pt in offset[2:-2]:
            dist = _min_dist_to_polyline(pt, path)
            assert abs(dist - d) < 0.4, f"Point {pt} dist={dist:.3f} expected≈{d}"


# ---------------------------------------------------------------------------
# (f) include_original=False omits the center path
# ---------------------------------------------------------------------------

class TestIncludeOriginal:
    def test_include_original_false_no_original_in_output(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=1, include_original=False)
        assert line not in result

    def test_include_original_true_contains_original(self) -> None:
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=1, include_original=True)
        assert line in result

    def test_include_original_false_count_is_correct(self) -> None:
        """Without original: count=1, both → 2 paths; with original → 3."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        without = offset_paths([line], distance_mm=1.0, sides="both",
                                count=1, include_original=False)
        with_ = offset_paths([line], distance_mm=1.0, sides="both",
                              count=1, include_original=True)
        assert len(with_) == len(without) + 1

    def test_multiple_paths_all_originals_included(self) -> None:
        """With include_original=True, all input paths appear in output."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(10.0, 0.0), (15.0, 0.0)],
        ]
        result = offset_paths(paths, distance_mm=1.0, sides="both",
                               count=1, include_original=True)
        for p in paths:
            assert p in result


# ---------------------------------------------------------------------------
# Edge cases and degenerate inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_degenerate_less_than_2_points_skipped(self) -> None:
        """Paths with 0 or 1 points are always skipped (regardless of include_original)."""
        result = offset_paths([[], [(1.0, 1.0)]], distance_mm=1.0,
                               sides="both", count=1, include_original=False)
        assert result == []
        result2 = offset_paths([[], [(1.0, 1.0)]], distance_mm=1.0,
                                sides="both", count=1, include_original=True)
        assert result2 == []

    def test_multiple_input_paths(self) -> None:
        """offset_paths processes all input paths."""
        paths = [
            [(0.0, 0.0), (10.0, 0.0)],
            [(0.0, 5.0), (10.0, 5.0)],
        ]
        result = offset_paths(paths, distance_mm=1.0, sides="both",
                               count=1, include_original=True)
        assert len(result) >= len(paths)

    def test_collinear_points_two_point_line(self) -> None:
        """Two-point line produces valid offsets."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="both",
                               count=1, include_original=False)
        assert len(result) == 2
        for poly in result:
            assert len(poly) >= 2

    def test_invalid_sides_falls_back_to_both(self) -> None:
        """Unknown sides value falls back to 'both'."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        result = offset_paths([line], distance_mm=1.0, sides="invalid",
                               count=1, include_original=False)
        # Falls back to "both" → 2 offsets
        assert len(result) == 2

    def test_count_clamped_to_minimum_1(self) -> None:
        """count=0 or negative should be treated as count=1."""
        line = [(0.0, 0.0), (10.0, 0.0)]
        result0 = offset_paths([line], distance_mm=1.0, sides="both",
                                count=0, include_original=False)
        result_neg = offset_paths([line], distance_mm=1.0, sides="both",
                                   count=-1, include_original=False)
        assert len(result0) == 2
        assert len(result_neg) == 2
