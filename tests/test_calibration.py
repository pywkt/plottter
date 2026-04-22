"""Tests for the calibration module."""

import math

import pytest

from plottter.calibration import (
    _hatch_rect,
    generate_angle_test,
    generate_circle_test,
    generate_fill_density_test,
    generate_line_spacing_test,
)
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


# ---------------------------------------------------------------------------
# generate_circle_test
# ---------------------------------------------------------------------------

# (a) Returns a non-empty list of polylines.
def test_circle_test_returns_nonempty():
    result = generate_circle_test(A4_W, A4_H, MARGIN)
    assert isinstance(result, list)
    assert len(result) > 0


# (b) All points are within the drawing area bounds (with 1 mm tolerance).
def test_circle_test_points_within_bounds():
    result = generate_circle_test(A4_W, A4_H, MARGIN)
    x_min = MARGIN - TOLERANCE
    x_max = A4_W - MARGIN + TOLERANCE
    y_min = MARGIN - TOLERANCE
    y_max = A4_H - MARGIN + TOLERANCE
    for x, y in _all_points(result):
        assert x_min <= x <= x_max, f"x={x} outside [{x_min}, {x_max}]"
        assert y_min <= y <= y_max, f"y={y} outside [{y_min}, {y_max}]"


# (c) The largest concentric circle's radius is approximately
#     min(draw_width, draw_height) / 2.
def test_circle_test_largest_concentric_radius():
    result = generate_circle_test(A4_W, A4_H, MARGIN)
    x0, y0 = MARGIN, MARGIN
    x1, y1 = A4_W - MARGIN, A4_H - MARGIN
    page_cx = (x0 + x1) / 2.0
    page_cy = (y0 + y1) / 2.0
    draw_width = x1 - x0
    draw_height = y1 - y0
    expected = min(draw_width, draw_height) / 2.0

    max_r = 0.0
    for pl in result:
        # Only consider polylines with enough points to be a full circle.
        if len(pl) < 70:
            continue
        # Exclude the closing duplicate point before computing the centroid.
        # Use epsilon comparison: sin(2π) ≈ -2.45e-16 so exact equality fails.
        closing_matches = (
            abs(pl[0][0] - pl[-1][0]) < 1e-9 and abs(pl[0][1] - pl[-1][1]) < 1e-9
        )
        pts = pl[:-1] if closing_matches else pl
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        # Keep only circles centred near the page centre.
        if abs(cx - page_cx) < 2.0 and abs(cy - page_cy) < 2.0:
            r = math.sqrt((pts[0][0] - cx) ** 2 + (pts[0][1] - cy) ** 2)
            max_r = max(max_r, r)

    assert abs(max_r - expected) < 1.0, f"max concentric radius {max_r:.2f} != expected {expected:.2f}"


# (d) Works with both landscape and portrait canvas sizes.
def test_circle_test_canvas_sizes():
    sizes = [
        (A4_W, A4_H),   # portrait A4
        (A4_H, A4_W),   # landscape A4
        (A3_W, A3_H),   # portrait A3
    ]
    for w, h in sizes:
        result = generate_circle_test(w, h, MARGIN)
        assert len(result) > 0, f"No output for {w}x{h}"
        x_min = MARGIN - TOLERANCE
        x_max = w - MARGIN + TOLERANCE
        y_min = MARGIN - TOLERANCE
        y_max = h - MARGIN + TOLERANCE
        for x, y in _all_points(result):
            assert x_min <= x <= x_max, f"{w}x{h}: x={x} outside [{x_min}, {x_max}]"
            assert y_min <= y <= y_max, f"{w}x{h}: y={y} outside [{y_min}, {y_max}]"


# ---------------------------------------------------------------------------
# generate_angle_test
# ---------------------------------------------------------------------------

# (a) Returns a non-empty list of polylines.
def test_angle_test_returns_nonempty():
    result = generate_angle_test(A4_W, A4_H, MARGIN)
    assert isinstance(result, list)
    assert len(result) > 0


# (b) All points are within the drawing area bounds (with 1 mm tolerance).
def test_angle_test_points_within_bounds():
    result = generate_angle_test(A4_W, A4_H, MARGIN)
    x_min = MARGIN - TOLERANCE
    x_max = A4_W - MARGIN + TOLERANCE
    y_min = MARGIN - TOLERANCE
    y_max = A4_H - MARGIN + TOLERANCE
    for x, y in _all_points(result):
        assert x_min <= x <= x_max, f"x={x} outside [{x_min}, {x_max}]"
        assert y_min <= y <= y_max, f"y={y} outside [{y_min}, {y_max}]"


# (c) Produces at least 24 polylines (one radial line per 15° increment).
def test_angle_test_at_least_24_polylines():
    result = generate_angle_test(A4_W, A4_H, MARGIN)
    assert len(result) >= 24


# (d) Works with square, portrait, and landscape canvases.
def test_angle_test_canvas_sizes():
    sizes = [
        (200.0, 200.0),   # square
        (A4_W, A4_H),     # portrait A4
        (A4_H, A4_W),     # landscape A4
    ]
    for w, h in sizes:
        result = generate_angle_test(w, h, MARGIN)
        assert len(result) >= 24, f"Expected >=24 polylines for {w}x{h}"
        x_min = MARGIN - TOLERANCE
        x_max = w - MARGIN + TOLERANCE
        y_min = MARGIN - TOLERANCE
        y_max = h - MARGIN + TOLERANCE
        for x, y in _all_points(result):
            assert x_min <= x <= x_max, f"{w}x{h}: x={x} outside [{x_min}, {x_max}]"
            assert y_min <= y <= y_max, f"{w}x{h}: y={y} outside [{y_min}, {y_max}]"


# ---------------------------------------------------------------------------
# generate_fill_density_test
# ---------------------------------------------------------------------------


# (a) Returns a non-empty list of polylines.
def test_fill_density_returns_nonempty():
    result = generate_fill_density_test(A4_W, A4_H, MARGIN)
    assert isinstance(result, list)
    assert len(result) > 0


# (b) All points are within the drawing area bounds (with 1 mm tolerance).
def test_fill_density_points_within_bounds():
    result = generate_fill_density_test(A4_W, A4_H, MARGIN)
    x_min = MARGIN - TOLERANCE
    x_max = A4_W - MARGIN + TOLERANCE
    y_min = MARGIN - TOLERANCE
    y_max = A4_H - MARGIN + TOLERANCE
    for x, y in _all_points(result):
        assert x_min <= x <= x_max, f"x={x} outside [{x_min}, {x_max}]"
        assert y_min <= y <= y_max, f"y={y} outside [{y_min}, {y_max}]"


# (c) Produces at least 16 closed swatch outline rectangles (4×4 grid).
def test_fill_density_at_least_16_swatches():
    result = generate_fill_density_test(A4_W, A4_H, MARGIN)
    # Swatch outlines are 5-point closed polylines (first point == last point).
    closed_rects = [
        pl for pl in result
        if len(pl) == 5 and abs(pl[0][0] - pl[-1][0]) < 1e-9 and abs(pl[0][1] - pl[-1][1]) < 1e-9
    ]
    assert len(closed_rects) >= 16, f"Found only {len(closed_rects)} swatch outlines"


# (d) Hatching at 0° produces only horizontal lines (both endpoints share the same y).
def test_fill_density_zero_degree_horizontal():
    result = generate_fill_density_test(A4_W, A4_H, MARGIN)
    # All 2-point hatch lines are horizontal when angle=0 (dy = sin(0°) = 0).
    # Collect lines where both y-values are essentially identical.
    horizontal = [pl for pl in result if len(pl) == 2 and abs(pl[0][1] - pl[1][1]) < 1e-6]
    assert len(horizontal) > 0, "Expected horizontal hatch lines from 0° swatches"
    # Verify _hatch_rect at 0° produces only truly flat lines.
    lines_0deg = _hatch_rect(0.0, 0.0, 40.0, 50.0, 0, 1.0)
    assert len(lines_0deg) > 0
    for pl in lines_0deg:
        assert len(pl) == 2
        assert abs(pl[0][1] - pl[1][1]) < 1e-9, f"Non-horizontal line at 0°: {pl}"


# (e) Smaller spacing → more hatch lines per swatch than larger spacing.
def test_fill_density_spacing_variation():
    # Use _hatch_rect directly on a representative swatch size.
    swatch_w, swatch_h = 40.0, 50.0
    lines_coarse = _hatch_rect(0.0, 0.0, swatch_w, swatch_h, 0, 2.0)
    lines_fine = _hatch_rect(0.0, 0.0, swatch_w, swatch_h, 0, 0.25)
    assert len(lines_fine) > len(lines_coarse), (
        f"Expected more lines at 0.25mm spacing ({len(lines_fine)}) "
        f"than at 2.0mm spacing ({len(lines_coarse)})"
    )
