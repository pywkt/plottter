"""Tests for the calibration module."""

import pytest

from plottter.calibration import generate_line_spacing_test
from plottter.models.path import Polyline


# A4: 210 x 297 mm
A4_W, A4_H = 210.0, 297.0
# A3: 297 x 420 mm
A3_W, A3_H = 297.0, 420.0
MARGIN = 10.0
TOLERANCE = 1.0  # mm


def _all_points(polylines: list[Polyline]):
    """Yield every (x, y) point from a list of polylines."""
    for pl in polylines:
        yield from pl


# (a) Returns a non-empty list of polylines for A4.
def test_returns_nonempty_list():
    result = generate_line_spacing_test(A4_W, A4_H, MARGIN)
    assert isinstance(result, list)
    assert len(result) > 0


# (b) All points are within the drawing area bounds (with 1 mm tolerance).
def test_points_within_bounds_a4():
    result = generate_line_spacing_test(A4_W, A4_H, MARGIN)
    x_min = MARGIN - TOLERANCE
    x_max = A4_W - MARGIN + TOLERANCE
    y_min = MARGIN - TOLERANCE
    y_max = A4_H - MARGIN + TOLERANCE
    for x, y in _all_points(result):
        assert x_min <= x <= x_max, f"x={x} outside [{x_min}, {x_max}]"
        assert y_min <= y <= y_max, f"y={y} outside [{y_min}, {y_max}]"


# (c) Result contains more than 100 polylines.
def test_more_than_100_polylines():
    result = generate_line_spacing_test(A4_W, A4_H, MARGIN)
    assert len(result) > 100


# (d) Both A4 and A3 produce output with points within their respective bounds.
def test_different_canvas_sizes():
    for w, h in [(A4_W, A4_H), (A3_W, A3_H)]:
        result = generate_line_spacing_test(w, h, MARGIN)
        assert len(result) > 0, f"Expected output for {w}x{h}"
        x_min = MARGIN - TOLERANCE
        x_max = w - MARGIN + TOLERANCE
        y_min = MARGIN - TOLERANCE
        y_max = h - MARGIN + TOLERANCE
        for x, y in _all_points(result):
            assert x_min <= x <= x_max, f"{w}x{h}: x={x} outside [{x_min}, {x_max}]"
            assert y_min <= y <= y_max, f"{w}x{h}: y={y} outside [{y_min}, {y_max}]"
