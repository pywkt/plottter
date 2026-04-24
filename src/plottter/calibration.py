"""Calibration patterns for pen plotter testing."""

from __future__ import annotations

import math

from plottter.models.canvas import PAPER_PRESETS
from plottter.models.path import Polyline
from plottter.generators.text import _render_hershey_text


def _title(text: str, center_x: float, top_y: float, size_mm: float = 3.0) -> list[Polyline]:
    """Render a centred Hershey Simplex title at the given position.

    The title is horizontally centred on *center_x* with its visual top edge
    at *top_y*.  Uses the actual glyph bounding box for accurate placement.
    """
    polylines, _width, _height = _render_hershey_text(
        text,
        "Simplex",
        size_mm,
        letter_spacing_mm=size_mm * 0.15,
        line_spacing=1.2,
        text_align="Center",
        stroke_repeat=1,
    )
    if not polylines:
        return []
    all_pts = [pt for pl in polylines for pt in pl]
    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    bbox_cx = (min_x + max_x) / 2.0
    dx = center_x - bbox_cx
    dy = top_y - min_y
    return [[(px + dx, py + dy) for px, py in pl] for pl in polylines]


def _title_height(text: str, size_mm: float = 3.0) -> float:
    """Return the visual height of a rendered Hershey Simplex string."""
    polylines, _w, _h = _render_hershey_text(
        text, "Simplex", size_mm,
        letter_spacing_mm=size_mm * 0.15, line_spacing=1.2,
        text_align="Center", stroke_repeat=1,
    )
    if not polylines:
        return size_mm
    all_pts = [pt for pl in polylines for pt in pl]
    return max(p[1] for p in all_pts) - min(p[1] for p in all_pts)


def _label_centered(text: str, cx: float, cy: float, size_mm: float = 3.0) -> list[Polyline]:
    """Render a Hershey Simplex label centred on the point (cx, cy)."""
    polylines, _w, _h = _render_hershey_text(
        text, "Simplex", size_mm,
        letter_spacing_mm=size_mm * 0.15, line_spacing=1.2,
        text_align="Center", stroke_repeat=1,
    )
    if not polylines:
        return []
    all_pts = [pt for pl in polylines for pt in pl]
    bb_cx = (min(p[0] for p in all_pts) + max(p[0] for p in all_pts)) / 2.0
    bb_cy = (min(p[1] for p in all_pts) + max(p[1] for p in all_pts)) / 2.0
    dx = cx - bb_cx
    dy = cy - bb_cy
    return [[(px + dx, py + dy) for px, py in pl] for pl in polylines]


def _label(text: str, x: float, y: float, size_mm: float = 3.0) -> list[Polyline]:
    """Render a Hershey Simplex text label with its top-left corner at (x, y).

    Wraps ``_render_hershey_text`` and translates so that the actual visual
    top-left of the rendered glyphs lands at the given coordinates.
    """
    polylines, _width, _height = _render_hershey_text(
        text,
        "Simplex",
        size_mm,
        letter_spacing_mm=size_mm * 0.15,
        line_spacing=1.2,
        text_align="Left",
        stroke_repeat=1,
    )
    if not polylines:
        return []
    # Compute the actual bounding box of the rendered glyph points,
    # since Hershey left-bearings cause strokes to extend beyond
    # the reported advance width.
    all_pts = [pt for pl in polylines for pt in pl]
    min_x = min(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    dx = x - min_x
    dy = y - min_y
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
    title_h = _title_height(title_text)
    result.extend(_title(title_text, (x0 + x1) / 2.0, y0))

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


def _dashed_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    dash_len: float = 2.0,
    gap_len: float = 2.0,
) -> list[Polyline]:
    """Return a dashed rectangle outline as 2-point polyline segments.

    Alternates between *dash_len* drawn segments and *gap_len* gaps.
    All four edges are processed in order: top, right, bottom, left.
    """
    def _dash_edge(ax: float, ay: float, bx: float, by: float) -> list[Polyline]:
        segs: list[Polyline] = []
        dx, dy = bx - ax, by - ay
        total = math.sqrt(dx * dx + dy * dy)
        if total < 1e-9:
            return segs
        ux, uy = dx / total, dy / total
        t = 0.0
        drawing = True
        while t < total - 1e-9:
            step = dash_len if drawing else gap_len
            t_end = min(t + step, total)
            if drawing and t_end > t + 1e-9:
                segs.append([
                    (ax + t * ux, ay + t * uy),
                    (ax + t_end * ux, ay + t_end * uy),
                ])
            t = t_end
            drawing = not drawing
        return segs

    lines: list[Polyline] = []
    lines.extend(_dash_edge(x0, y0, x1, y0))  # top
    lines.extend(_dash_edge(x1, y0, x1, y1))  # right
    lines.extend(_dash_edge(x1, y1, x0, y1))  # bottom
    lines.extend(_dash_edge(x0, y1, x0, y0))  # left
    return lines


def generate_paper_size_sheet(
    width_mm: float,
    height_mm: float,
    margin_mm: float,
    paper_name: str | None = None,
) -> list[Polyline]:
    """Generate a paper size alignment underlay sheet.

    Draws crosshair marks at the corners of each paper size from
    ``PAPER_PRESETS`` that fits within the canvas, so the user can visually
    align paper on the plotter bed.  Both portrait and landscape orientations
    are drawn when they fit.

    Parameters
    ----------
    width_mm:   Page width in mm.
    height_mm:  Page height in mm.
    margin_mm:  Margin from each page edge to the drawing area.
    paper_name: If given, only draw this paper size (e.g. ``"A3"``).
                If ``None``, draw all fitting paper sizes.
    """
    x0 = margin_mm
    y0 = margin_mm
    x1 = width_mm - margin_mm
    y1 = height_mm - margin_mm

    result: list[Polyline] = []

    # -- Paper size crosshairs -------------------------------------------------
    ARM = 8.0                               # crosshair arm length in mm
    TICK_LEN = 5.0                          # diagonal tick length in mm
    TICK_C = TICK_LEN / math.sqrt(2.0)     # x and y component of 45° tick

    # Collect (paper_w, paper_h, label) entries, avoiding duplicate sizes.
    seen: set[tuple[float, float]] = set()
    entries: list[tuple[float, float, str]] = []

    presets_to_check = PAPER_PRESETS.items()
    if paper_name is not None:
        # Filter to just the requested paper size
        presets_to_check = [
            (n, dims) for n, dims in PAPER_PRESETS.items() if n == paper_name
        ]

    for name, (pw, ph) in presets_to_check:
        # Portrait orientation
        if pw <= width_mm and ph <= height_mm:
            key = (pw, ph)
            if key not in seen:
                seen.add(key)
                entries.append((pw, ph, f"{name} ({int(round(pw))} x {int(round(ph))})"))

        # Landscape orientation (swap); skip if square or already seen
        lw, lh = ph, pw
        if lw != pw and lw <= width_mm and lh <= height_mm:
            key = (lw, lh)
            if key not in seen:
                seen.add(key)
                entries.append((lw, lh, f"{name} ({int(round(lw))} x {int(round(lh))}) (landscape)"))

    # Sort largest area first so smaller outlines are drawn on top of larger ones.
    entries.sort(key=lambda e: e[0] * e[1], reverse=True)

    # Estimated label bounding box (width × height) for collision avoidance.
    LABEL_W = 50.0
    LABEL_H = 5.0
    label_bboxes: list[tuple[float, float, float, float]] = []  # (lx0, ly0, lx1, ly1)

    for pw, ph, label in entries:
        # Centre each paper size on the canvas
        px0 = (width_mm - pw) / 2.0
        py0 = (height_mm - ph) / 2.0
        px1 = px0 + pw
        py1 = py0 + ph

        # Solid rectangle outline showing the full paper boundary.
        result.append([(px0, py0), (px1, py0)])  # top
        result.append([(px1, py0), (px1, py1)])  # right
        result.append([(px1, py1), (px0, py1)])  # bottom
        result.append([(px0, py1), (px0, py0)])  # left

        # Corners: (corner_x, corner_y, h_dir, v_dir)
        #   h_dir: -1 = left (outward for left corners), +1 = right
        #   v_dir: -1 = up   (outward for top corners),  +1 = down
        corners = [
            (px0, py0, -1.0, -1.0),  # top-left:     go left and up
            (px1, py0, +1.0, -1.0),  # top-right:    go right and up
            (px0, py1, -1.0, +1.0),  # bottom-left:  go left and down
            (px1, py1, +1.0, +1.0),  # bottom-right: go right and down
        ]

        for cx, cy, hdir, vdir in corners:
            # Horizontal arm (clamped to canvas)
            hx = max(0.0, min(width_mm, cx + hdir * ARM))
            if abs(hx - cx) > 1e-9:
                result.append([(cx, cy), (hx, cy)])

            # Vertical arm (clamped to canvas)
            vy = max(0.0, min(height_mm, cy + vdir * ARM))
            if abs(vy - cy) > 1e-9:
                result.append([(cx, cy), (cx, vy)])

            # 45° diagonal tick (clamped to canvas)
            tx = max(0.0, min(width_mm, cx + hdir * TICK_C))
            ty = max(0.0, min(height_mm, cy + vdir * TICK_C))
            if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                result.append([(cx, cy), (tx, ty)])

        # Label at the top-left corner of the paper size, offset upward.
        # Resolve overlaps with previously placed labels by pushing further up.
        lx = max(1.0, px0)
        ly = py0 - LABEL_H - 1.0
        for _ in range(20):  # at most 20 upward shifts to avoid infinite loop
            conflict = False
            for bx0, by0, bx1, by1 in label_bboxes:
                if lx < bx1 and lx + LABEL_W > bx0 and ly < by1 and ly + LABEL_H > by0:
                    ly = by0 - LABEL_H - 1.0
                    conflict = True
                    break
            if not conflict:
                break
        label_bboxes.append((lx, ly, lx + LABEL_W, ly + LABEL_H))
        result.extend(_label(label, max(1.0, lx), max(1.0, ly), size_mm=2.5))

    # Clamp every point to the physical page bounds to guard against
    # sub-millimetre Hershey glyph overhangs or floating-point drift.
    result = [
        [(max(0.0, min(width_mm, px)), max(0.0, min(height_mm, py))) for px, py in pl]
        for pl in result
    ]

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
    result.extend(_title("ANGLE TEST", (x0 + x1) / 2.0, y0))

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
        result.extend(_label_centered(label_text, lx, ly, size_mm=2.5))

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
    title_h = _title_height(title_text)
    result.extend(_title(title_text, (x0 + x1) / 2.0, y0))

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


def generate_registration_test(
    width_mm: float,
    height_mm: float,
    margin_mm: float,
) -> list[Polyline]:
    """Generate a registration calibration test page.

    Draws a full border, corner crosshairs, a centre crosshair with circle,
    edge tick marks at 10 mm intervals, coordinate labels at each corner,
    a "CENTER" label, a page-dimensions label, and a title.

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
    page_cx = (x0 + x1) / 2.0
    page_cy = (y0 + y1) / 2.0

    result: list[Polyline] = []

    # -- Border ------------------------------------------------------------------
    result.append([(x0, y0), (x1, y0)])
    result.append([(x1, y0), (x1, y1)])
    result.append([(x1, y1), (x0, y1)])
    result.append([(x0, y1), (x0, y0)])

    # -- Title "REGISTRATION TEST" at top centre --------------------------------
    result.extend(_title("REGISTRATION TEST", page_cx, y0))

    # -- Corner crosshairs (10 mm arms extending inward) -----------------------
    # (xc, yc, hdir, vdir): hdir/vdir = +1 → arm points right/down (inward)
    CORNER_ARM = 10.0
    corner_defs = [
        (x0, y0, +1, +1),  # top-left
        (x1, y0, -1, +1),  # top-right
        (x0, y1, +1, -1),  # bottom-left
        (x1, y1, -1, -1),  # bottom-right
    ]
    for xc, yc, hdir, vdir in corner_defs:
        result.append([(xc, yc), (xc + hdir * CORNER_ARM, yc)])
        result.append([(xc, yc), (xc, yc + vdir * CORNER_ARM)])

    # -- Corner coordinate labels ----------------------------------------------
    label_size = 2.5
    label_gap = 1.5
    corner_label_defs = [
        (x0, y0, +1, +1, f"{int(round(x0))}, {int(round(y0))}"),
        (x1, y0, -1, +1, f"{int(round(x1))}, {int(round(y0))}"),
        (x0, y1, +1, -1, f"{int(round(x0))}, {int(round(y1))}"),
        (x1, y1, -1, -1, f"{int(round(x1))}, {int(round(y1))}"),
    ]
    for xc, yc, hdir, vdir, text in corner_label_defs:
        # Position the label centred at the end of the horizontal crosshair arm
        # so the text stays well within the drawing area boundary.
        lx = xc + hdir * CORNER_ARM
        if vdir > 0:
            ly = yc + CORNER_ARM + label_gap
        else:
            ly = yc - CORNER_ARM - label_gap
        result.extend(_label_centered(text, lx, ly, size_mm=label_size))

    # -- Centre crosshair (20 mm arms) -----------------------------------------
    CENTER_ARM = 10.0  # half of 20 mm
    result.append([(page_cx - CENTER_ARM, page_cy), (page_cx + CENTER_ARM, page_cy)])
    result.append([(page_cx, page_cy - CENTER_ARM), (page_cx, page_cy + CENTER_ARM)])

    # -- Centre circle (5 mm radius) -------------------------------------------
    circle_angles = [math.radians(deg) for deg in range(0, 361, 5)]
    r = 5.0
    result.append(
        [(page_cx + r * math.cos(a), page_cy + r * math.sin(a)) for a in circle_angles]
    )

    # -- "CENTER" label below the centre crosshair -----------------------------
    ctr_label_cy = page_cy + CENTER_ARM + label_gap + 1.5  # approx half text height
    result.extend(_label_centered("CENTER", page_cx, ctr_label_cy))

    # -- Edge tick marks at every 10 mm ----------------------------------------
    TICK_LEN = 3.0
    TICK_STEP = 10.0

    # Top edge (y = y0): ticks extend downward
    x = x0
    while x <= x1 + 1e-9:
        result.append([(x, y0), (x, y0 + TICK_LEN)])
        x += TICK_STEP

    # Bottom edge (y = y1): ticks extend upward
    x = x0
    while x <= x1 + 1e-9:
        result.append([(x, y1), (x, y1 - TICK_LEN)])
        x += TICK_STEP

    # Left edge (x = x0): ticks extend rightward
    y = y0
    while y <= y1 + 1e-9:
        result.append([(x0, y), (x0 + TICK_LEN, y)])
        y += TICK_STEP

    # Right edge (x = x1): ticks extend leftward
    y = y0
    while y <= y1 + 1e-9:
        result.append([(x1, y), (x1 - TICK_LEN, y)])
        y += TICK_STEP

    # -- Page dimensions label at bottom centre --------------------------------
    dims_text = f"{int(round(width_mm))} x {int(round(height_mm))} mm"
    dims_cy = y1 - label_gap - 1.5  # approx half text height
    result.extend(_label_centered(dims_text, page_cx, dims_cy, size_mm=2.5))

    # Clamp all points to the drawing area.
    result = [
        [(max(x0, min(x1, px)), max(y0, min(y1, py))) for px, py in pl]
        for pl in result
    ]

    return result


def generate_circle_test(
    width_mm: float,
    height_mm: float,
    margin_mm: float,
) -> list[Polyline]:
    """Generate a circle and arc calibration test page.

    The page is split into two regions:
    - Upper 70%: concentric circles centred in this region.
    - Lower 30%: reference circles (bottom-left) and quarter-arcs (bottom-right).

    A horizontal divider line separates the two regions.  Sub-headings label
    each section.  Reference diameters are 1, 2, 3, 5, 8, 10 mm.

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
    title_h = _title_height(title_text)
    result.extend(_title(title_text, (x0 + x1) / 2.0, y0))

    # -- Region split: upper 70% for concentric circles, lower 30% for ref --
    y_divider = y0 + 0.7 * draw_height
    page_cx = (x0 + x1) / 2.0

    # -- Horizontal divider line between regions ----------------------------
    result.append([(x0, y_divider), (x1, y_divider)])

    # -- "Concentric Circles" sub-heading in upper region -------------------
    conc_sub_text = "Concentric Circles"
    conc_sub_h = _title_height(conc_sub_text, size_mm=2.5)
    conc_sub_y = y0 + title_h + 2.0
    result.extend(_label(conc_sub_text, x0 + 3.0, conc_sub_y, size_mm=2.5))

    # -- Concentric circles: fill the upper region below the sub-heading ----
    conc_content_y0 = conc_sub_y + conc_sub_h + 2.0
    conc_content_y1 = y_divider - 2.0
    # Guard against degenerate canvases.
    if conc_content_y1 > conc_content_y0:
        circ_cy = (conc_content_y0 + conc_content_y1) / 2.0
        max_radius = min(draw_width / 2.0, (conc_content_y1 - conc_content_y0) / 2.0)
    else:
        circ_cy = (y0 + y_divider) / 2.0
        max_radius = min(draw_width, draw_height * 0.7) / 4.0

    num_rings = 18
    ring_spacing = max_radius / num_rings

    # 73 points: every 5° from 0° to 360° inclusive (closes the circle).
    circle_angles = [math.radians(deg) for deg in range(0, 361, 5)]

    for i in range(1, num_rings + 1):
        r = i * ring_spacing
        result.append(
            [(page_cx + r * math.cos(a), circ_cy + r * math.sin(a)) for a in circle_angles]
        )

    # -- Bottom region: reference circles (left) and quarter-arcs (right) ---
    diameters = [1, 2, 3, 5, 8, 10]
    gap_mm = 5.0
    arc_angles = [math.radians(deg) for deg in range(0, 91, 5)]

    # Sub-headings for the two bottom sections.
    ref_sub_text = "Reference Circles"
    arc_sub_text = "Quarter Arcs"
    ref_sub_h = _title_height(ref_sub_text, size_mm=2.5)
    ref_sub_y = y_divider + 2.0
    result.extend(_label(ref_sub_text, x0 + gap_mm, ref_sub_y, size_mm=2.5))
    result.extend(_label(arc_sub_text, page_cx + gap_mm, ref_sub_y, size_mm=2.5))

    # Row centre y for reference items: below sub-heading, centred on largest circle.
    max_ref_r = max(d / 2.0 for d in diameters)  # = 5.0 mm
    ref_row_cy = ref_sub_y + ref_sub_h + 3.0 + max_ref_r

    # -- Compute effective gap so reference circles fit in the left half -----
    total_d = sum(diameters)
    n = len(diameters)
    left_half_w = page_cx - x0
    # Total space used = n * gap + total_d; last gap is a right margin.
    max_total_gap = left_half_w - total_d - gap_mm
    effective_gap_circles = min(gap_mm, max_total_gap / n) if n > 0 else gap_mm
    effective_gap_circles = max(1.0, effective_gap_circles)

    # -- Bottom-left: full circles at reference diameters -------------------
    x_cursor = x0 + effective_gap_circles
    for d in diameters:
        r = d / 2.0
        circ_cx = x_cursor + r
        result.append(
            [(circ_cx + r * math.cos(a), ref_row_cy + r * math.sin(a)) for a in circle_angles]
        )
        result.extend(_label(f"{d}mm", circ_cx - r, ref_row_cy + r + 1.5, size_mm=2.5))
        x_cursor = circ_cx + r + effective_gap_circles

    # -- Compute effective gap so quarter-arcs fit in the right half ---------
    right_half_w = x1 - page_cx
    max_total_gap_arc = right_half_w - total_d - gap_mm
    effective_gap_arcs = min(gap_mm, max_total_gap_arc / n) if n > 0 else gap_mm
    effective_gap_arcs = max(1.0, effective_gap_arcs)

    # -- Bottom-right: quarter-arcs (0°–90°) at reference diameters ---------
    x_cursor = page_cx + effective_gap_arcs
    for d in diameters:
        r = d / 2.0
        arc_cx = x_cursor + r
        result.append(
            [(arc_cx + r * math.cos(a), ref_row_cy + r * math.sin(a)) for a in arc_angles]
        )
        result.extend(_label(f"{d}mm", arc_cx - r, ref_row_cy + r + 1.5, size_mm=2.5))
        x_cursor = arc_cx + r + effective_gap_arcs

    return result
