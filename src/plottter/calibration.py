"""Calibration patterns for pen plotter testing."""

from __future__ import annotations

import math

from plottter.models.path import Polyline
from plottter.generators.text import _render_hershey_text


def _label(text: str, x: float, y: float, size_mm: float = 3.0) -> list[Polyline]:
    """Render a Hershey Simplex text label with its top-left corner at (x, y).

    Wraps ``_render_hershey_text`` and translates the centred output so that
    the top-left of the text block lands at the given coordinates.
    """
    polylines, width, height = _render_hershey_text(
        text,
        "Simplex",
        size_mm,
        letter_spacing_mm=0.0,
        line_spacing=1.2,
        text_align="Left",
        stroke_repeat=1,
    )
    # _render_hershey_text centres the block at the origin.
    # Shift so the top-left corner is at (x, y).
    dx = x + width / 2.0
    dy = y + height / 2.0
    return [[(px + dx, py + dy) for px, py in pl] for pl in polylines]


def generate_line_spacing_test(
    width_mm: float,
    height_mm: float,
    margin_mm: float,
) -> list[Polyline]:
    """Generate a line-spacing calibration test page.

    Produces 6 columns of parallel horizontal lines at decreasing spacings
    (2.0, 1.5, 1.0, 0.75, 0.5, 0.25 mm), each labelled at the top.  A title
    and a border rectangle are also included.

    All output coordinates are in mm with (0, 0) at the top-left of the page.

    Parameters
    ----------
    width_mm:   Page width in mm.
    height_mm:  Page height in mm.
    margin_mm:  Margin from each page edge to the drawing area.
    """
    x0 = margin_mm
    y0 = margin_mm
    x1 = width_mm - margin_mm
    y1 = height_mm - margin_mm

    result: list[Polyline] = []

    # -- Border: 4 line segments connecting the corners ----------------------
    result.append([(x0, y0), (x1, y0)])  # top
    result.append([(x1, y0), (x1, y1)])  # right
    result.append([(x1, y1), (x0, y1)])  # bottom
    result.append([(x0, y1), (x0, y0)])  # left

    # -- Title ---------------------------------------------------------------
    title_text = "LINE SPACING TEST"
    title_size = 3.0
    title_polys, title_w, title_h = _render_hershey_text(
        title_text,
        "Simplex",
        title_size,
        letter_spacing_mm=0.0,
        line_spacing=1.2,
        text_align="Center",
        stroke_repeat=1,
    )
    # Centre the title horizontally; position it just inside the top margin.
    title_cx = (x0 + x1) / 2.0
    title_cy = y0 + title_h / 2.0
    for pl in title_polys:
        result.append([(px + title_cx, py + title_cy) for px, py in pl])

    # -- Columns -------------------------------------------------------------
    spacings = [2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
    draw_w = x1 - x0
    col_w = draw_w / len(spacings)

    # Columns start below the title with a small padding gap.
    col_y0 = y0 + title_h + 3.0  # 3 mm padding after title

    # Reserve space at the top of each column for the spacing label.
    label_height = 5.0  # mm

    for i, spacing in enumerate(spacings):
        col_x0 = x0 + i * col_w
        col_x1 = col_x0 + col_w

        # Label
        label_text = f"{spacing}mm"
        result.extend(_label(label_text, col_x0, col_y0))

        # Parallel horizontal lines fill the remainder of the column.
        lines_y0 = col_y0 + label_height
        lines_y1 = y1

        y = lines_y0
        while y <= lines_y1 + 1e-9:
            result.append([(col_x0, y), (col_x1, y)])
            y += spacing

    return result


def generate_circle_test(
    width_mm: float,
    height_mm: float,
    margin_mm: float,
) -> list[Polyline]:
    """Generate a circle and arc calibration test page.

    Draws concentric circles centred on the page, a row of reference circles
    (diameters 1, 2, 3, 5, 8, 10 mm) in the bottom-left quadrant, matching
    quarter-arcs (0°–90°) in the bottom-right quadrant, a title, and a border.

    All output coordinates are in mm with (0, 0) at the top-left of the page.

    Parameters
    ----------
    width_mm:   Page width in mm.
    height_mm:  Page height in mm.
    margin_mm:  Margin from each page edge to the drawing area.
    """
    x0 = margin_mm
    y0 = margin_mm
    x1 = width_mm - margin_mm
    y1 = height_mm - margin_mm
    draw_width = x1 - x0
    draw_height = y1 - y0

    result: list[Polyline] = []

    # -- Border: 4 line segments connecting the corners ----------------------
    result.append([(x0, y0), (x1, y0)])  # top
    result.append([(x1, y0), (x1, y1)])  # right
    result.append([(x1, y1), (x0, y1)])  # bottom
    result.append([(x0, y1), (x0, y0)])  # left

    # -- Title ---------------------------------------------------------------
    title_text = "CIRCLE & ARC TEST"
    title_size = 3.0
    title_polys, title_w, title_h = _render_hershey_text(
        title_text,
        "Simplex",
        title_size,
        letter_spacing_mm=0.0,
        line_spacing=1.2,
        text_align="Center",
        stroke_repeat=1,
    )
    title_cx = (x0 + x1) / 2.0
    title_cy = y0 + title_h / 2.0
    for pl in title_polys:
        result.append([(px + title_cx, py + title_cy) for px, py in pl])

    # -- Concentric circles --------------------------------------------------
    page_cx = (x0 + x1) / 2.0
    page_cy = (y0 + y1) / 2.0
    num_rings = 18
    max_radius = min(draw_width, draw_height) / 2.0
    ring_spacing = max_radius / num_rings

    # 73 points: every 5° from 0° to 360° inclusive (closes the circle).
    circle_angles = [math.radians(deg) for deg in range(0, 361, 5)]

    for i in range(1, num_rings + 1):
        r = i * ring_spacing
        result.append(
            [(page_cx + r * math.cos(a), page_cy + r * math.sin(a)) for a in circle_angles]
        )

    # -- Reference sizes -----------------------------------------------------
    diameters = [1, 2, 3, 5, 8, 10]
    gap_mm = 5.0

    # Vertical centre for the reference rows: mid-point of the bottom half,
    # shifted slightly upward to leave room for labels below.
    ref_cy = (page_cy + y1) / 2.0 - 5.0

    # -- Bottom-left: full circles at reference diameters -------------------
    x_cursor = x0 + gap_mm
    for d in diameters:
        r = d / 2.0
        circ_cx = x_cursor + r
        result.append(
            [(circ_cx + r * math.cos(a), ref_cy + r * math.sin(a)) for a in circle_angles]
        )
        label_x = circ_cx - r
        label_y = ref_cy + r + 1.5
        result.extend(_label(f"{d}mm", label_x, label_y, size_mm=2.5))
        x_cursor = circ_cx + r + gap_mm

    # -- Bottom-right: quarter-arcs (0°–90°) at reference diameters ---------
    # 19 points: every 5° from 0° to 90° inclusive.
    arc_angles = [math.radians(deg) for deg in range(0, 91, 5)]

    x_cursor = page_cx + gap_mm
    for d in diameters:
        r = d / 2.0
        arc_cx = x_cursor + r
        result.append(
            [(arc_cx + r * math.cos(a), ref_cy + r * math.sin(a)) for a in arc_angles]
        )
        label_x = arc_cx - r
        label_y = ref_cy + r + 1.5
        result.extend(_label(f"{d}mm", label_x, label_y, size_mm=2.5))
        x_cursor = arc_cx + r + gap_mm

    return result
