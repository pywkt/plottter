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


def generate_angle_test(
    width_mm: float,
    height_mm: float,
    margin_mm: float,
) -> list[Polyline]:
    """Generate an angle calibration test page.

    Draws 24 radial lines from the centre at every 15° increment, labelled at
    their outer ends, diagonal corner-to-corner lines, 5 concentric squares,
    a title ("ANGLE TEST"), and a border rectangle.

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
    title_text = "ANGLE TEST"
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

    # -- Centre of drawing area ----------------------------------------------
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    # -- Diagonal lines (corner to corner) -----------------------------------
    result.append([(x0, y0), (x1, y1)])
    result.append([(x1, y0), (x0, y1)])

    # -- Concentric squares --------------------------------------------------
    num_squares = 5
    min_dim = min(draw_width, draw_height)
    max_side = min_dim * 0.4
    min_side = 10.0
    for i in range(num_squares):
        t = i / (num_squares - 1) if num_squares > 1 else 1.0
        side = min_side + (max_side - min_side) * t
        half = side / 2.0
        sx0 = max(cx - half, x0)
        sy0 = max(cy - half, y0)
        sx1 = min(cx + half, x1)
        sy1 = min(cy + half, y1)
        result.append([(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1), (sx0, sy0)])

    # -- Radial lines at 15° increments with labels --------------------------
    # Labels are centered on a point 10 mm inward from the boundary endpoint.
    label_inset = 10.0  # mm

    for deg in range(0, 360, 15):
        angle_rad = math.radians(deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        # Ray-rectangle intersection: find smallest positive t so that
        # (cx + t*dx, cy + t*dy) lies on the boundary of [x0,x1]×[y0,y1].
        t_best = float("inf")
        for t_cand in [
            (x0 - cx) / dx if abs(dx) > 1e-12 else float("inf"),
            (x1 - cx) / dx if abs(dx) > 1e-12 else float("inf"),
            (y0 - cy) / dy if abs(dy) > 1e-12 else float("inf"),
            (y1 - cy) / dy if abs(dy) > 1e-12 else float("inf"),
        ]:
            if t_cand <= 1e-9:
                continue
            px_c = cx + t_cand * dx
            py_c = cy + t_cand * dy
            if (x0 - 1e-9 <= px_c <= x1 + 1e-9) and (y0 - 1e-9 <= py_c <= y1 + 1e-9):
                if t_cand < t_best:
                    t_best = t_cand

        if t_best == float("inf"):
            continue

        ex = max(x0, min(x1, cx + t_best * dx))
        ey = max(y0, min(y1, cy + t_best * dy))

        result.append([(cx, cy), (ex, ey)])

        # Label: rendered centered on the inset point (10 mm inside boundary).
        lx = ex - dx * label_inset
        ly = ey - dy * label_inset

        label_text = f"{deg}d"  # "d" for degrees (Hershey lacks the degree glyph)
        label_polys, _lw, _lh = _render_hershey_text(
            label_text,
            "Simplex",
            2.5,
            letter_spacing_mm=0.0,
            line_spacing=1.2,
            text_align="Center",
            stroke_repeat=1,
        )
        # label_polys are centred at origin; shift to (lx, ly).
        for pl in label_polys:
            result.append([(ppx + lx, ppy + ly) for ppx, ppy in pl])

    return result


def _hatch_rect(
    sx0: float,
    sy0: float,
    sx1: float,
    sy1: float,
    angle_deg: float,
    spacing: float,
) -> list[Polyline]:
    """Generate parallel hatch lines at *angle_deg* and *spacing* clipped to the rectangle.

    *angle_deg* is the angle the hatch lines make with the positive x-axis
    (0° → horizontal, 90° → vertical, 45°/135° → diagonal).
    Returns a list of 2-point polylines; empty if the rectangle is degenerate.
    """
    lines: list[Polyline] = []
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    # Perpendicular direction (normal to the hatch lines)
    nx = -math.sin(angle_rad)
    ny = math.cos(angle_rad)

    cx = (sx0 + sx1) / 2.0
    cy = (sy0 + sy1) / 2.0

    # Project all 4 corners onto the normal to find the extent to cover.
    corners = [(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)]
    projs = [nx * (x - cx) + ny * (y - cy) for x, y in corners]
    t_min = min(projs)
    t_max = max(projs)

    # Start at the grid-aligned position at or before t_min.
    t = math.floor(t_min / spacing) * spacing

    while t <= t_max + 1e-9:
        # Point on this hatch line in world coords.
        p0x = cx + t * nx
        p0y = cy + t * ny

        # Clip the parametric line P(s) = (p0x + s*dx, p0y + s*dy)
        # to the rectangle [sx0, sx1] × [sy0, sy1].
        s_lo = -float("inf")
        s_hi = float("inf")

        if abs(dx) > 1e-9:
            ta = (sx0 - p0x) / dx
            tb = (sx1 - p0x) / dx
            s_lo = max(s_lo, min(ta, tb))
            s_hi = min(s_hi, max(ta, tb))
        elif not (sx0 - 1e-9 <= p0x <= sx1 + 1e-9):
            t += spacing
            continue

        if abs(dy) > 1e-9:
            ta = (sy0 - p0y) / dy
            tb = (sy1 - p0y) / dy
            s_lo = max(s_lo, min(ta, tb))
            s_hi = min(s_hi, max(ta, tb))
        elif not (sy0 - 1e-9 <= p0y <= sy1 + 1e-9):
            t += spacing
            continue

        if s_lo < s_hi - 1e-9:
            lx0 = p0x + s_lo * dx
            ly0 = p0y + s_lo * dy
            lx1 = p0x + s_hi * dx
            ly1 = p0y + s_hi * dy
            lines.append([(lx0, ly0), (lx1, ly1)])

        t += spacing

    return lines


def generate_fill_density_test(
    width_mm: float,
    height_mm: float,
    margin_mm: float,
) -> list[Polyline]:
    """Generate a fill density calibration test page.

    Creates a 4×4 grid of hatched swatches:
    - Columns: line spacing 2.0, 1.0, 0.5, 0.25 mm
    - Rows:    hatch angle 0°, 45°, 90°, 135°

    Each swatch shows parallel hatching at the given angle and spacing, with a
    thin outline rectangle.  Column headers label the spacing value; row headers
    label the angle.  A title and border rectangle are also included.

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

    # -- Border -----------------------------------------------------------------
    result.append([(x0, y0), (x1, y0)])
    result.append([(x1, y0), (x1, y1)])
    result.append([(x1, y1), (x0, y1)])
    result.append([(x0, y1), (x0, y0)])

    # -- Title ------------------------------------------------------------------
    title_text = "FILL DENSITY TEST"
    title_size = 3.0
    title_polys, _tw, title_h = _render_hershey_text(
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

    title_area_h = title_h + 3.0  # 3 mm padding below title

    # -- Grid layout -----------------------------------------------------------
    spacings = [2.0, 1.0, 0.5, 0.25]
    angles = [0, 45, 90, 135]
    n_cols = len(spacings)
    n_rows = len(angles)
    gap = 5.0           # mm gap between swatches
    col_header_h = 8.0  # mm reserved above swatches for column labels
    row_header_w = 15.0  # mm reserved left of swatches for row labels

    grid_x0 = x0 + row_header_w
    grid_y0 = y0 + title_area_h + col_header_h
    grid_x1 = x1
    grid_y1 = y1

    grid_w = grid_x1 - grid_x0
    grid_h = grid_y1 - grid_y0

    swatch_w = (grid_w - gap * (n_cols - 1)) / n_cols
    swatch_h = (grid_h - gap * (n_rows - 1)) / n_rows

    # -- Column headers (spacing labels) ----------------------------------------
    for ci, spacing in enumerate(spacings):
        sw_x0 = grid_x0 + ci * (swatch_w + gap)
        label_text = f"{spacing}mm"
        label_y = y0 + title_area_h  # top of column header area
        result.extend(_label(label_text, sw_x0, label_y, size_mm=3.0))

    # -- Rows: row headers + swatches -------------------------------------------
    for ri, angle_deg in enumerate(angles):
        sw_y0 = grid_y0 + ri * (swatch_h + gap)

        # Row header (angle label, "d" for degrees — Hershey lacks the ° glyph)
        row_label = f"{angle_deg}d"
        result.extend(_label(row_label, x0, sw_y0, size_mm=3.0))

        for ci, spacing in enumerate(spacings):
            sw_x0 = grid_x0 + ci * (swatch_w + gap)
            sw_x1 = sw_x0 + swatch_w
            sw_y1 = sw_y0 + swatch_h

            # Swatch outline rectangle (closed, 5 points)
            result.append(
                [(sw_x0, sw_y0), (sw_x1, sw_y0), (sw_x1, sw_y1), (sw_x0, sw_y1), (sw_x0, sw_y0)]
            )

            # Hatching
            result.extend(_hatch_rect(sw_x0, sw_y0, sw_x1, sw_y1, angle_deg, spacing))

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
