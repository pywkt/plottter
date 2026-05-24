"""Tests for the calibration module."""

import math

import pytest

from plottter.calibration import (
    _dashed_rect,
    _hatch_rect,
    generate_angle_test,
    generate_circle_test,
    generate_fill_density_test,
    generate_line_spacing_test,
    generate_paper_size_sheet,
    generate_registration_test,
)
from plottter.models.path import Polyline


# ---------------------------------------------------------------------------
# Task 100.7 — importability and basic validity checks
# ---------------------------------------------------------------------------

# (a) All 6 calibration functions are importable from plottter.calibration and
#     produce non-empty, list-of-polylines output for a standard A4 canvas.
def test_all_six_functions_importable_and_produce_output():
    import plottter.calibration as cal
    functions = [
        cal.generate_line_spacing_test,
        cal.generate_circle_test,
        cal.generate_angle_test,
        cal.generate_fill_density_test,
        cal.generate_registration_test,
        cal.generate_paper_size_sheet,
    ]
    for fn in functions:
        result = fn(210.0, 297.0, 10.0)
        assert isinstance(result, list), f"{fn.__name__} did not return a list"
        assert len(result) > 0, f"{fn.__name__} returned empty list for A4"
        for pl in result:
            assert isinstance(pl, list), f"{fn.__name__} polyline is not a list"
            for pt in pl:
                assert len(pt) == 2, f"{fn.__name__} point has wrong arity: {pt}"


# (b) The Tools > Calibration Plots submenu contains exactly 6 actions.
def test_calibration_submenu_has_six_actions(qapp, qtbot):
    from plottter.models import Canvas, Layer, Project
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.main_window import MainWindow

    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="Test", canvas=canvas)
    project.add_layer(Layer(name="Layer 1", color="#000000"))
    controller = ProjectController(project)
    win = MainWindow(controller)
    qtbot.addWidget(win)

    # Find the Tools menu
    tools_menu = win._tools_menu
    assert tools_menu is not None

    # Find the Calibration Plots submenu
    calib_menu = None
    for action in tools_menu.actions():
        if action.menu() is not None and action.text() == "Calibration Plots":
            calib_menu = action.menu()
            break

    assert calib_menu is not None, "Calibration Plots submenu not found in Tools menu"
    # Count only real actions (not separators)
    real_actions = [a for a in calib_menu.actions() if not a.isSeparator()]
    assert len(real_actions) == 6, (
        f"Expected 6 calibration actions, got {len(real_actions)}: "
        f"{[a.text() for a in real_actions]}"
    )


# (c) Each calibration function produces non-empty output for multiple paper sizes.
@pytest.mark.parametrize("width_mm,height_mm", [
    (210.0, 297.0),   # A4 portrait
    (297.0, 210.0),   # A4 landscape
    (297.0, 420.0),   # A3 portrait
    (420.0, 594.0),   # A2 portrait
])
def test_all_functions_produce_output_for_multiple_sizes(width_mm, height_mm):
    import plottter.calibration as cal
    functions = [
        cal.generate_line_spacing_test,
        cal.generate_circle_test,
        cal.generate_angle_test,
        cal.generate_fill_density_test,
        cal.generate_registration_test,
        cal.generate_paper_size_sheet,
    ]
    for fn in functions:
        result = fn(width_mm, height_mm, 10.0)
        assert len(result) > 0, (
            f"{fn.__name__} returned empty output for {width_mm}x{height_mm}"
        )


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


# (c) The largest concentric circle fits within the upper 70% region and uses
#     most of that space (not trivially small).
def test_circle_test_largest_concentric_radius():
    result = generate_circle_test(A4_W, A4_H, MARGIN)
    x0, y0 = MARGIN, MARGIN
    x1, y1 = A4_W - MARGIN, A4_H - MARGIN
    page_cx = (x0 + x1) / 2.0
    draw_width = x1 - x0
    draw_height = y1 - y0
    y_divider = y0 + 0.7 * draw_height

    max_r = 0.0
    for pl in result:
        # Only consider polylines with enough points to be a full circle.
        if len(pl) < 70:
            continue
        # Exclude the closing duplicate point before computing the centroid.
        closing_matches = (
            abs(pl[0][0] - pl[-1][0]) < 1e-9 and abs(pl[0][1] - pl[-1][1]) < 1e-9
        )
        pts = pl[:-1] if closing_matches else pl
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        # Keep only circles centred near the horizontal page centre.
        if abs(cx - page_cx) < 2.0:
            r = math.sqrt((pts[0][0] - cx) ** 2 + (pts[0][1] - cy) ** 2)
            max_r = max(max_r, r)
            # Verify this circle stays within the upper region.
            max_y = max(p[1] for p in pts)
            assert max_y <= y_divider + TOLERANCE, (
                f"Concentric circle (r={r:.2f}) extends below divider: "
                f"max_y={max_y:.2f} > y_divider={y_divider:.2f}"
            )

    assert max_r > 0.0, "No concentric circles found"
    # Should use a reasonable fraction of the upper region.
    upper_height = y_divider - y0
    upper_max_r = min(draw_width / 2.0, upper_height / 2.0)
    assert max_r >= upper_max_r * 0.5, (
        f"max concentric radius {max_r:.2f} too small "
        f"(< 50% of upper region max {upper_max_r:.2f})"
    )


# (d-extra) No polyline from the reference section has points in the upper 70% region.
def test_circle_test_reference_items_in_bottom_region():
    result = generate_circle_test(A4_W, A4_H, MARGIN)
    x0, y0 = MARGIN, MARGIN
    x1, y1 = A4_W - MARGIN, A4_H - MARGIN
    draw_height = y1 - y0
    y_divider = y0 + 0.7 * draw_height

    # Any polyline whose centroid y is strictly below the divider belongs to
    # the reference section; none of its points should be above the divider.
    for pl in result:
        if not pl:
            continue
        centroid_y = sum(p[1] for p in pl) / len(pl)
        if centroid_y > y_divider:
            for _x, y in pl:
                assert y >= y_divider - TOLERANCE, (
                    f"Reference-section polyline has point above divider: "
                    f"y={y:.2f} < y_divider={y_divider:.2f}"
                )


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


# ---------------------------------------------------------------------------
# generate_registration_test
# ---------------------------------------------------------------------------

# (a) Returns non-empty polylines.
def test_registration_test_returns_nonempty():
    result = generate_registration_test(A4_W, A4_H, MARGIN)
    assert isinstance(result, list)
    assert len(result) > 0


# (b) All points are within the drawing area bounds (with 1 mm tolerance).
def test_registration_test_points_within_bounds():
    result = generate_registration_test(A4_W, A4_H, MARGIN)
    x_min = MARGIN - TOLERANCE
    x_max = A4_W - MARGIN + TOLERANCE
    y_min = MARGIN - TOLERANCE
    y_max = A4_H - MARGIN + TOLERANCE
    for x, y in _all_points(result):
        assert x_min <= x <= x_max, f"x={x} outside [{x_min}, {x_max}]"
        assert y_min <= y <= y_max, f"y={y} outside [{y_min}, {y_max}]"


# (c) Corner crosshairs are positioned at the drawing area corners.
def test_registration_test_corner_crosshairs():
    result = generate_registration_test(A4_W, A4_H, MARGIN)
    x0, y0 = MARGIN, MARGIN
    x1, y1 = A4_W - MARGIN, A4_H - MARGIN
    expected_corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]

    for corner in expected_corners:
        cx, cy = corner
        # Find a 2-point line whose first point is at this corner.
        found = any(
            len(pl) == 2
            and abs(pl[0][0] - cx) < 1e-6
            and abs(pl[0][1] - cy) < 1e-6
            for pl in result
        )
        assert found, f"No crosshair line starting at corner {corner}"


# (d) Tick marks are present along all 4 edges.
def test_registration_test_tick_marks():
    result = generate_registration_test(A4_W, A4_H, MARGIN)
    x0, y0 = MARGIN, MARGIN
    x1, y1 = A4_W - MARGIN, A4_H - MARGIN
    TICK_LEN = 3.0

    # Top edge: vertical 2-point lines at y=y0 going to y0+TICK_LEN.
    top_ticks = [
        pl for pl in result
        if len(pl) == 2
        and abs(pl[0][1] - y0) < 1e-6
        and abs(pl[1][1] - (y0 + TICK_LEN)) < 1e-6
        and abs(pl[0][0] - pl[1][0]) < 1e-6
    ]
    # Bottom edge: vertical 2-point lines at y=y1 going to y1-TICK_LEN.
    bottom_ticks = [
        pl for pl in result
        if len(pl) == 2
        and abs(pl[0][1] - y1) < 1e-6
        and abs(pl[1][1] - (y1 - TICK_LEN)) < 1e-6
        and abs(pl[0][0] - pl[1][0]) < 1e-6
    ]
    # Left edge: horizontal 2-point lines at x=x0 going to x0+TICK_LEN.
    left_ticks = [
        pl for pl in result
        if len(pl) == 2
        and abs(pl[0][0] - x0) < 1e-6
        and abs(pl[1][0] - (x0 + TICK_LEN)) < 1e-6
        and abs(pl[0][1] - pl[1][1]) < 1e-6
    ]
    # Right edge: horizontal 2-point lines at x=x1 going to x1-TICK_LEN.
    right_ticks = [
        pl for pl in result
        if len(pl) == 2
        and abs(pl[0][0] - x1) < 1e-6
        and abs(pl[1][0] - (x1 - TICK_LEN)) < 1e-6
        and abs(pl[0][1] - pl[1][1]) < 1e-6
    ]

    assert len(top_ticks) > 5, f"Too few top edge ticks: {len(top_ticks)}"
    assert len(bottom_ticks) > 5, f"Too few bottom edge ticks: {len(bottom_ticks)}"
    assert len(left_ticks) > 5, f"Too few left edge ticks: {len(left_ticks)}"
    assert len(right_ticks) > 5, f"Too few right edge ticks: {len(right_ticks)}"


# (e) Works with different paper sizes, all points within respective bounds.
def test_registration_test_different_sizes():
    sizes = [(A4_W, A4_H), (A3_W, A3_H), (A4_H, A4_W)]
    for w, h in sizes:
        result = generate_registration_test(w, h, MARGIN)
        assert len(result) > 0, f"No output for {w}x{h}"
        x_min = MARGIN - TOLERANCE
        x_max = w - MARGIN + TOLERANCE
        y_min = MARGIN - TOLERANCE
        y_max = h - MARGIN + TOLERANCE
        for x, y in _all_points(result):
            assert x_min <= x <= x_max, f"{w}x{h}: x={x} outside [{x_min}, {x_max}]"
            assert y_min <= y <= y_max, f"{w}x{h}: y={y} outside [{y_min}, {y_max}]"


# ---------------------------------------------------------------------------
# generate_paper_size_sheet
# ---------------------------------------------------------------------------

# A2: 420 x 594 mm
A2_W, A2_H = 420.0, 594.0


# (a) A2 canvas returns non-empty polylines (A2 fits A3, A4, Letter, Legal).
def test_paper_size_sheet_a2_nonempty():
    result = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    assert isinstance(result, list)
    assert len(result) > 0


# (b) A4 canvas: only paper sizes that fit within 210×297 are drawn.
# From PAPER_PRESETS only A4 portrait (210×297) satisfies both dims ≤ canvas.
# A2 canvas fits many more sizes, so it produces more polylines.
def test_paper_size_sheet_a4_canvas_only_fitting():
    result_a4 = generate_paper_size_sheet(A4_W, A4_H, MARGIN)
    result_a2 = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    assert len(result_a4) > 0
    # A2 canvas contains more fitting paper sizes → more polylines
    assert len(result_a2) > len(result_a4), (
        f"A2 canvas ({len(result_a2)} polylines) should produce more than "
        f"A4 canvas ({len(result_a4)} polylines)"
    )
    # No crosshair arm should start at negative coordinates on A4 canvas.
    for pl in result_a4:
        if len(pl) == 2:
            x, y = pl[0]
            assert x >= -TOLERANCE, f"Crosshair start x={x:.2f} is off-page"
            assert y >= -TOLERANCE, f"Crosshair start y={y:.2f} is off-page"


# (c) Crosshair lines start at paper boundary corners, anchored at (0, 0).
def test_paper_size_sheet_crosshair_positions():
    result = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    # A4 portrait anchored at the home corner (0, 0) on the A2 canvas.
    a4_px0 = 0.0
    a4_py0 = 0.0
    a4_px1 = A4_W                   # 210.0
    a4_py1 = A4_H                   # 297.0
    corners = [(a4_px0, a4_py0), (a4_px1, a4_py0), (a4_px0, a4_py1), (a4_px1, a4_py1)]
    for cx, cy in corners:
        found = any(
            len(pl) == 2
            and abs(pl[0][0] - cx) < TOLERANCE
            and abs(pl[0][1] - cy) < TOLERANCE
            for pl in result
        )
        assert found, f"No crosshair line starting at A4 corner ({cx:.1f}, {cy:.1f})"


# (d) All points are within canvas dimensions [0, width] × [0, height].
# Crosshairs extend into the margin area but stay on the physical page.
def test_paper_size_sheet_points_within_canvas():
    result = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    for x, y in _all_points(result):
        assert 0.0 <= x <= A2_W, f"x={x:.3f} outside [0, {A2_W}]"
        assert 0.0 <= y <= A2_H, f"y={y:.3f} outside [0, {A2_H}]"


# (e) Labels with paper size names are rendered (verified by polyline count).
# Structural elements alone (border, title, ~9 paper entries × 12 lines)
# give roughly 140 polylines; each label adds ~25+ polylines for a total
# well above 200 when labels are present.
def test_paper_size_sheet_labels_present():
    result = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    assert len(result) > 200, (
        f"Expected >200 polylines (labels included), got {len(result)}"
    )


# (f) Paper sizes are drawn largest-area first.
# A3 portrait (297×420, area=124 740) should appear before A4 portrait
# (210×297, area=62 370) in the output.
def test_paper_size_sheet_sorted_by_area():
    result = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    # Every size is anchored at (0, 0), so the top-left corner no longer
    # distinguishes them. Discriminate by the bottom-right corner, which is
    # unique to each paper size: A3 portrait = (297, 420), A4 = (210, 297).
    a3_br = (297.0, 420.0)
    a4_br = (210.0, 297.0)

    def first_index_near(cx: float, cy: float) -> int | None:
        """Return the index of the first 2-point polyline starting near (cx, cy)."""
        for idx, pl in enumerate(result):
            if (
                len(pl) == 2
                and abs(pl[0][0] - cx) < TOLERANCE
                and abs(pl[0][1] - cy) < TOLERANCE
            ):
                return idx
        return None

    idx_a3 = first_index_near(*a3_br)
    idx_a4 = first_index_near(*a4_br)
    assert idx_a3 is not None, "A3 top-left corner not found in output"
    assert idx_a4 is not None, "A4 top-left corner not found in output"
    assert idx_a3 < idx_a4, (
        f"A3 (index {idx_a3}) should appear before A4 (index {idx_a4}) "
        f"because A3 has larger area"
    )


# (g) Solid rectangle outlines are present for each paper size.
def test_paper_size_sheet_solid_rects_present():
    result = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    # Each paper size gets 4 solid rectangle edges (2-point polylines).
    edge_segs = [pl for pl in result if len(pl) == 2]
    assert len(edge_segs) >= 16, (
        f"Expected >=16 edge segments (4 per paper size), got {len(edge_segs)}"
    )


# (g2) Single paper size filtering works.
def test_paper_size_sheet_single_paper():
    result = generate_paper_size_sheet(A2_W, A2_H, MARGIN, paper_name="A4")
    assert len(result) > 0, "Single paper size produced no output"
    result_all = generate_paper_size_sheet(A2_W, A2_H, MARGIN)
    assert len(result) < len(result_all), "Single paper should have fewer paths than all"


# (h) _dashed_rect helper produces expected segment structure.
def test_dashed_rect_structure():
    # A 20×10 mm rectangle with default 2 mm dash / 2 mm gap.
    # Perimeter = 2*(20+10) = 60 mm; with 4 mm cycle → 15 dashes.
    segs = _dashed_rect(0.0, 0.0, 20.0, 10.0)
    # All segments must be 2-point polylines.
    for seg in segs:
        assert len(seg) == 2, f"Segment has {len(seg)} points, expected 2"
    # Each segment length should be ≤ dash_len (2.0 mm).
    for seg in segs:
        length = math.sqrt(
            (seg[1][0] - seg[0][0]) ** 2 + (seg[1][1] - seg[0][1]) ** 2
        )
        assert length <= 2.0 + 1e-9, f"Dash segment too long: {length:.4f} mm"
    # Should produce a reasonable number of segments (around 15 for this rect).
    assert len(segs) >= 12, f"Too few dash segments: {len(segs)}"
