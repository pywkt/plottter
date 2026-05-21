"""Tests for src/plottter/generators/_pixel_fills.py

For each fill function (fill_solid_hatch, fill_cross_hatch, fill_diagonal),
given a 10 mm square cell at the origin and density=0.5, we assert:

  (i)   All returned polyline endpoints lie inside the cell bounds.
  (ii)  The number of returned polylines matches the formula.
  (iii) With a triangular polygon clip, no returned point lies outside the
        polygon.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from plottter.generators._pixel_fills import (
    _expected_diagonal_count,
    _expected_hatch_count,
    _lerp_spacing,
    fill_cross_hatch,
    fill_diagonal,
    fill_dithered_dots,
    fill_leave_empty,
    fill_outline,
    fill_solid_hatch,
    fill_x_mark,
)

# ---------------------------------------------------------------------------
# Test fixtures / parameters
# ---------------------------------------------------------------------------

CELL_X = 0.0
CELL_Y = 0.0
CELL_SIZE = 10.0
DENSITY = 0.5

# A triangle inscribed inside the 10×10 cell
TRIANGLE = ShapelyPolygon([(1.0, 1.0), (9.0, 1.0), (5.0, 9.0)])

# Tolerance for floating-point boundary checks
EPS = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inside_cell(x: float, y: float) -> bool:
    """Return True if (x, y) is within [CELL_X, CELL_X+CELL_SIZE] × [CELL_Y, CELL_Y+CELL_SIZE]."""
    return (
        CELL_X - EPS <= x <= CELL_X + CELL_SIZE + EPS
        and CELL_Y - EPS <= y <= CELL_Y + CELL_SIZE + EPS
    )


def _inside_polygon(x: float, y: float, poly: ShapelyPolygon) -> bool:
    """Return True if (x, y) is inside or on the boundary of *poly*."""
    from shapely.geometry import Point
    return poly.contains(Point(x, y)) or poly.boundary.distance(Point(x, y)) < EPS


# ---------------------------------------------------------------------------
# fill_solid_hatch
# ---------------------------------------------------------------------------

class TestFillSolidHatch:
    def _call(self, polygon=None):
        return fill_solid_hatch(CELL_X, CELL_Y, CELL_SIZE, DENSITY, polygon=polygon)

    def test_returns_list_of_polylines(self):
        result = self._call()
        assert isinstance(result, list)
        assert all(isinstance(pl, list) for pl in result)

    def test_endpoints_inside_cell(self):
        result = self._call()
        assert result, "Expected at least one hatch line"
        for polyline in result:
            for x, y in polyline:
                assert _inside_cell(x, y), (
                    f"Point ({x}, {y}) is outside the 10×10 cell"
                )

    def test_line_count_matches_formula(self):
        result = self._call()
        expected = _expected_hatch_count(CELL_SIZE, DENSITY)
        assert len(result) == expected, (
            f"Expected {expected} lines, got {len(result)}"
        )

    def test_line_count_formula_value(self):
        """Verify the formula gives the expected value for density=0.5."""
        spacing = _lerp_spacing(DENSITY)
        # spacing = 0.6 + (0.15 - 0.6) * 0.5 = 0.375
        assert abs(spacing - 0.375) < 1e-9
        expected = int((CELL_SIZE - spacing / 2.0) / spacing) + 1
        assert expected == 27

    def test_polygon_clip_no_points_outside(self):
        result = self._call(polygon=TRIANGLE)
        for polyline in result:
            for x, y in polyline:
                assert _inside_polygon(x, y, TRIANGLE), (
                    f"Point ({x:.4f}, {y:.4f}) lies outside the clip polygon"
                )

    def test_polygon_clip_returns_fewer_lines(self):
        full = self._call()
        clipped = self._call(polygon=TRIANGLE)
        # Triangle is strictly smaller than the cell — should get fewer lines
        assert len(clipped) <= len(full)

    def test_empty_for_zero_cell_size(self):
        result = fill_solid_hatch(0.0, 0.0, 0.0, DENSITY)
        assert result == []


# ---------------------------------------------------------------------------
# fill_cross_hatch
# ---------------------------------------------------------------------------

class TestFillCrossHatch:
    def _call(self, polygon=None):
        return fill_cross_hatch(CELL_X, CELL_Y, CELL_SIZE, DENSITY, polygon=polygon)

    def test_returns_list_of_polylines(self):
        result = self._call()
        assert isinstance(result, list)
        assert all(isinstance(pl, list) for pl in result)

    def test_endpoints_inside_cell(self):
        result = self._call()
        assert result, "Expected at least one hatch line"
        for polyline in result:
            for x, y in polyline:
                assert _inside_cell(x, y), (
                    f"Point ({x}, {y}) is outside the 10×10 cell"
                )

    def test_line_count_matches_formula(self):
        result = self._call()
        # Cross hatch = horizontal + vertical, each axis uses the same formula
        expected = 2 * _expected_hatch_count(CELL_SIZE, DENSITY)
        assert len(result) == expected, (
            f"Expected {expected} lines (cross), got {len(result)}"
        )

    def test_line_count_formula_value(self):
        """Verify the formula gives 54 for density=0.5, 10mm cell."""
        per_axis = _expected_hatch_count(CELL_SIZE, DENSITY)
        assert per_axis == 27
        assert 2 * per_axis == 54

    def test_polygon_clip_no_points_outside(self):
        result = self._call(polygon=TRIANGLE)
        for polyline in result:
            for x, y in polyline:
                assert _inside_polygon(x, y, TRIANGLE), (
                    f"Point ({x:.4f}, {y:.4f}) lies outside the clip polygon"
                )

    def test_empty_for_zero_cell_size(self):
        result = fill_cross_hatch(0.0, 0.0, 0.0, DENSITY)
        assert result == []


# ---------------------------------------------------------------------------
# fill_diagonal
# ---------------------------------------------------------------------------

class TestFillDiagonal:
    def _call(self, polygon=None):
        return fill_diagonal(CELL_X, CELL_Y, CELL_SIZE, DENSITY, polygon=polygon)

    def test_returns_list_of_polylines(self):
        result = self._call()
        assert isinstance(result, list)
        assert all(isinstance(pl, list) for pl in result)

    def test_endpoints_inside_cell(self):
        result = self._call()
        assert result, "Expected at least one diagonal line"
        for polyline in result:
            for x, y in polyline:
                assert _inside_cell(x, y), (
                    f"Point ({x:.6f}, {y:.6f}) is outside the 10×10 cell"
                )

    def test_line_count_matches_formula(self):
        result = self._call()
        expected = _expected_diagonal_count(CELL_SIZE, DENSITY)
        assert len(result) == expected, (
            f"Expected {expected} diagonal lines, got {len(result)}"
        )

    def test_line_count_formula_value(self):
        """Verify the diagonal formula for density=0.5, 10mm cell."""
        spacing = _lerp_spacing(DENSITY)
        assert abs(spacing - 0.375) < 1e-9
        delta_c = spacing * math.sqrt(2.0)
        total_range = 2.0 * CELL_SIZE
        expected = int((total_range - delta_c / 2.0) / delta_c) + 1
        assert expected == _expected_diagonal_count(CELL_SIZE, DENSITY)
        # Sanity: should be around 37-38 for these parameters
        assert 30 <= expected <= 45

    def test_polygon_clip_no_points_outside(self):
        result = self._call(polygon=TRIANGLE)
        for polyline in result:
            for x, y in polyline:
                assert _inside_polygon(x, y, TRIANGLE), (
                    f"Point ({x:.4f}, {y:.4f}) lies outside the clip polygon"
                )

    def test_polygon_clip_returns_fewer_lines(self):
        full = self._call()
        clipped = self._call(polygon=TRIANGLE)
        assert len(clipped) <= len(full)

    def test_empty_for_zero_cell_size(self):
        result = fill_diagonal(0.0, 0.0, 0.0, DENSITY)
        assert result == []

    def test_lines_are_diagonal(self):
        """Each returned segment should follow y - x = constant (± tolerance)."""
        result = self._call()
        for polyline in result:
            if len(polyline) >= 2:
                x0, y0 = polyline[0]
                x1, y1 = polyline[-1]
                # For a 45-degree line: y - x should be (approximately) constant
                c0 = y0 - x0
                c1 = y1 - x1
                assert abs(c0 - c1) < 1e-6, (
                    f"Segment is not 45-degree diagonal: c0={c0:.6f}, c1={c1:.6f}"
                )


# ---------------------------------------------------------------------------
# _lerp_spacing
# ---------------------------------------------------------------------------

class TestLerpSpacing:
    def test_density_zero(self):
        assert abs(_lerp_spacing(0.0) - 0.6) < 1e-9

    def test_density_one(self):
        assert abs(_lerp_spacing(1.0) - 0.15) < 1e-9

    def test_density_half(self):
        assert abs(_lerp_spacing(0.5) - 0.375) < 1e-9

    def test_clamps_below_zero(self):
        assert abs(_lerp_spacing(-1.0) - 0.6) < 1e-9

    def test_clamps_above_one(self):
        assert abs(_lerp_spacing(2.0) - 0.15) < 1e-9


# ---------------------------------------------------------------------------
# fill_x_mark
# ---------------------------------------------------------------------------


class TestFillXMark:
    def _call(self, polygon=None):
        return fill_x_mark(CELL_X, CELL_Y, CELL_SIZE, DENSITY, polygon=polygon)

    def test_returns_exactly_two_polylines(self):
        result = self._call()
        assert len(result) == 2, f"Expected 2 polylines, got {len(result)}"

    def test_density_does_not_affect_count(self):
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = fill_x_mark(CELL_X, CELL_Y, CELL_SIZE, d)
            assert len(result) == 2, f"density={d}: expected 2 polylines, got {len(result)}"

    def test_each_polyline_has_two_points(self):
        result = self._call()
        for pl in result:
            assert len(pl) == 2, f"Expected 2-point polyline, got {len(pl)} points"

    def test_diagonals_span_cell_corners(self):
        result = self._call()
        x0, y0 = CELL_X, CELL_Y
        x1, y1 = CELL_X + CELL_SIZE, CELL_Y + CELL_SIZE
        corner_sets = {
            frozenset([(x0, y0), (x1, y1)]),
            frozenset([(x1, y0), (x0, y1)]),
        }
        result_sets = {frozenset([tuple(p) for p in pl]) for pl in result}
        assert result_sets == corner_sets, "Diagonals do not connect the expected corners"

    def test_polygon_clip_no_points_outside(self):
        result = self._call(polygon=TRIANGLE)
        for polyline in result:
            for x, y in polyline:
                assert _inside_polygon(x, y, TRIANGLE), (
                    f"Point ({x:.4f}, {y:.4f}) lies outside the clip polygon"
                )

    def test_empty_for_zero_cell_size(self):
        result = fill_x_mark(0.0, 0.0, 0.0, DENSITY)
        assert result == []


# ---------------------------------------------------------------------------
# fill_outline
# ---------------------------------------------------------------------------


class TestFillOutline:
    def _call(self, polygon=None):
        return fill_outline(CELL_X, CELL_Y, CELL_SIZE, DENSITY, polygon=polygon)

    def test_returns_exactly_one_polyline(self):
        result = self._call()
        assert len(result) == 1, f"Expected 1 polyline, got {len(result)}"

    def test_density_does_not_affect_count(self):
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = fill_outline(CELL_X, CELL_Y, CELL_SIZE, d)
            assert len(result) == 1, f"density={d}: expected 1 polyline, got {len(result)}"

    def test_outline_is_closed(self):
        result = self._call()
        rect = result[0]
        assert rect[0] == rect[-1], "Outline polyline is not closed (first != last point)"

    def test_outline_has_five_points(self):
        result = self._call()
        rect = result[0]
        assert len(rect) == 5, f"Expected 5-point closed polyline, got {len(rect)} points"

    def test_outline_matches_cell_corners(self):
        result = self._call()
        rect = result[0]
        x0, y0 = CELL_X, CELL_Y
        x1, y1 = CELL_X + CELL_SIZE, CELL_Y + CELL_SIZE
        expected_corners = {(x0, y0), (x1, y0), (x1, y1), (x0, y1)}
        actual_corners = {tuple(p) for p in rect}
        assert expected_corners == actual_corners, (
            f"Outline corners {actual_corners} do not match cell corners {expected_corners}"
        )

    def test_all_points_on_cell_boundary(self):
        result = self._call()
        x0, y0 = CELL_X, CELL_Y
        x1, y1 = CELL_X + CELL_SIZE, CELL_Y + CELL_SIZE
        for x, y in result[0]:
            on_boundary = (
                abs(x - x0) < EPS or abs(x - x1) < EPS
                or abs(y - y0) < EPS or abs(y - y1) < EPS
            )
            assert on_boundary, f"Point ({x}, {y}) is not on the cell boundary"

    def test_polygon_clip_no_points_outside(self):
        result = self._call(polygon=TRIANGLE)
        for polyline in result:
            for x, y in polyline:
                assert _inside_polygon(x, y, TRIANGLE), (
                    f"Point ({x:.4f}, {y:.4f}) lies outside the clip polygon"
                )

    def test_empty_for_zero_cell_size(self):
        result = fill_outline(0.0, 0.0, 0.0, DENSITY)
        assert result == []


# ---------------------------------------------------------------------------
# fill_dithered_dots
# ---------------------------------------------------------------------------


class TestFillDitheredDots:
    def _call(self, density=DENSITY, polygon=None):
        return fill_dithered_dots(CELL_X, CELL_Y, CELL_SIZE, density, polygon=polygon)

    def test_returns_list_of_polylines(self):
        result = self._call()
        assert isinstance(result, list)
        assert all(isinstance(pl, list) for pl in result)

    def test_each_diamond_has_five_points(self):
        result = self._call()
        assert result, "Expected at least one dot"
        for pl in result:
            assert len(pl) == 5, f"Expected 5-point diamond, got {len(pl)} points"

    def test_each_diamond_is_closed(self):
        result = self._call()
        for pl in result:
            assert pl[0] == pl[-1], "Diamond polyline is not closed"

    def test_count_scales_with_density(self):
        sparse = self._call(density=0.0)
        dense = self._call(density=1.0)
        assert len(dense) > len(sparse), (
            f"Dense ({len(dense)}) should have more dots than sparse ({len(sparse)})"
        )

    def test_centers_inside_cell(self):
        """Diamond centres must lie inside the cell."""
        result = self._call()
        for pl in result:
            # Centre is the average of the 4 cardinal points (ignoring repeated last)
            xs = [p[0] for p in pl[:4]]
            ys = [p[1] for p in pl[:4]]
            cx = sum(xs) / 4
            cy = sum(ys) / 4
            assert _inside_cell(cx, cy), f"Diamond centre ({cx:.4f}, {cy:.4f}) outside cell"

    def test_polygon_clip_no_points_outside(self):
        result = self._call(polygon=TRIANGLE)
        for polyline in result:
            for x, y in polyline:
                assert _inside_polygon(x, y, TRIANGLE), (
                    f"Point ({x:.4f}, {y:.4f}) lies outside the clip polygon"
                )

    def test_polygon_clip_returns_fewer_dots(self):
        full = self._call()
        clipped = self._call(polygon=TRIANGLE)
        assert len(clipped) <= len(full)

    def test_empty_for_zero_cell_size(self):
        result = fill_dithered_dots(0.0, 0.0, 0.0, DENSITY)
        assert result == []


# ---------------------------------------------------------------------------
# fill_leave_empty
# ---------------------------------------------------------------------------


class TestFillLeaveEmpty:
    def test_always_returns_empty_list(self):
        result = fill_leave_empty(CELL_X, CELL_Y, CELL_SIZE, DENSITY)
        assert result == []

    def test_empty_for_all_densities(self):
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = fill_leave_empty(CELL_X, CELL_Y, CELL_SIZE, d)
            assert result == [], f"density={d}: expected [], got {result}"

    def test_empty_with_polygon(self):
        result = fill_leave_empty(CELL_X, CELL_Y, CELL_SIZE, DENSITY, polygon=TRIANGLE)
        assert result == []

    def test_empty_for_zero_cell_size(self):
        result = fill_leave_empty(0.0, 0.0, 0.0, DENSITY)
        assert result == []
