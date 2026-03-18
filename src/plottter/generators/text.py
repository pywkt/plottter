"""Text generator — render vector text as plotter-ready polylines.

Two font backends are supported:

* **Hershey fonts** (built-in, zero dependencies) — single-stroke glyphs
  derived from the public-domain Hershey vector font set. Each character is
  traced in a single pen stroke (or a small number of strokes), making them
  ideal for plotters.  Four variants: Simplex, Duplex, Script, Gothic.

* **TTF/OTF outline fonts** (optional, requires ``fonttools``) — load any
  system or user-supplied TrueType/OpenType font, extract glyph outlines as
  cubic Bezier curves, and sample them to polylines.  Fill modes (hatching,
  cross-hatch, concentric rings) are available for these outline glyphs.

Parameters
----------
text : str
    The text to render.  Use ``\\n`` for multi-line text.
font_type : {"Hershey", "System Font"}
    Which font backend to use.
hershey_font : {"Simplex", "Duplex", "Script", "Gothic"}
    Hershey font variant (only when *font_type* is "Hershey").
system_font_path : str
    Path to a .ttf or .otf font file (only when *font_type* is
    "System Font").
font_size_mm : float
    Height of capital letters in mm.
render_mode : {"Outline", "Filled", "Outline + Filled"}
    For System Font only.  Hershey fonts always render as single-stroke.
fill_type : {"Hatching", "Cross-hatch", "Concentric"}
    Fill pattern for "Filled" / "Outline + Filled" render modes.
fill_spacing_mm : float
    Spacing between fill lines/rings in mm.
fill_angle : float
    Angle for hatching fills in degrees.
letter_spacing_mm : float
    Extra space added between characters (may be negative).
line_spacing : float
    Line height as a multiplier of *font_size_mm*.
text_align : {"Left", "Center", "Right"}
    Horizontal alignment relative to *x_offset_mm*.
x_offset_mm : float
    Horizontal offset from the drawing-area centre (positive = right).
y_offset_mm : float
    Vertical offset from the drawing-area centre (positive = down).
rotation_deg : float
    Rotation angle in degrees (counter-clockwise).
stroke_repeat : int
    Hershey only — number of times each stroke is traced for a bold effect.
curve_tolerance_mm : float
    System Font only — chord tolerance when sampling Bezier curves to
    polylines.  Lower values produce more accurate curves but more points.
"""

from __future__ import annotations

import math
import os
from typing import Any

from plottter.generators import register_generator
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FontParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
    StringParam,
)
from plottter.models import Canvas, Polyline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    tolerance_mm: float,
) -> list[tuple[float, float]]:
    """Sample a cubic Bezier curve as a polyline using adaptive subdivision.

    The curve is subdivided recursively until the chord length between
    subdivided midpoints is within *tolerance_mm*.  Always returns at least
    the two endpoints.
    """
    def bezier_pt(t: float) -> tuple[float, float]:
        u = 1.0 - t
        x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
        y = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
        return (x, y)

    def subdivide(pa: tuple, pb: tuple, t0: float, t1: float, depth: int) -> list:
        if depth > 10:
            return [pa, pb]
        tm = (t0 + t1) / 2.0
        pm = bezier_pt(tm)
        mid_chord = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        if mid_chord < tolerance_mm or depth > 6:
            return [pa, pm, pb]
        left = subdivide(pa, pm, t0, tm, depth + 1)
        right = subdivide(pm, pb, tm, t1, depth + 1)
        return left + right[1:]

    return subdivide(p0, p3, 0.0, 1.0, 0)


def _sample_quadratic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    tolerance_mm: float,
) -> list[tuple[float, float]]:
    """Sample a quadratic Bezier curve as a polyline using adaptive subdivision.

    p0 is the start point, p1 the control point, p2 the end point.
    """
    def bezier_pt(t: float) -> tuple[float, float]:
        u = 1.0 - t
        x = u**2 * p0[0] + 2 * u * t * p1[0] + t**2 * p2[0]
        y = u**2 * p0[1] + 2 * u * t * p1[1] + t**2 * p2[1]
        return (x, y)

    def subdivide(pa: tuple, pb: tuple, t0: float, t1: float, depth: int) -> list:
        if depth > 10:
            return [pa, pb]
        tm = (t0 + t1) / 2.0
        pm = bezier_pt(tm)
        mid_chord = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        if mid_chord < tolerance_mm or depth > 6:
            return [pa, pm, pb]
        left = subdivide(pa, pm, t0, tm, depth + 1)
        right = subdivide(pm, pb, tm, t1, depth + 1)
        return left + right[1:]

    return subdivide(p0, p2, 0.0, 1.0, 0)


def _render_hershey_text(
    text: str,
    font_name: str,
    font_size_mm: float,
    letter_spacing_mm: float,
    line_spacing: float,
    text_align: str,
    stroke_repeat: int,
) -> tuple[list[Polyline], float, float]:
    """Render *text* using Hershey vector fonts.

    Returns ``(polylines, total_width_mm, total_height_mm)`` where the
    polylines are in local canvas coordinates with the text block centred at
    the origin (x right, y down).  Callers apply the final translate /
    rotate / clip.
    """
    from plottter.generators._hershey import CAP_HEIGHT, glyph_strokes

    scale = font_size_mm / CAP_HEIGHT
    line_height_mm = font_size_mm * line_spacing
    lines = text.split("\n") if text else [""]

    # ---- first pass: compute per-line widths for alignment -----------------
    def _line_width(line: str) -> float:
        if not line:
            return 0.0
        w = 0.0
        for ch in line:
            left, right, _ = glyph_strokes(ch, font_name)
            w += (right - left) * scale + letter_spacing_mm
        return w - letter_spacing_mm  # no trailing spacing

    line_widths = [_line_width(ln) for ln in lines]
    max_width = max(line_widths, default=0.0)
    total_height = font_size_mm + (len(lines) - 1) * line_height_mm

    # ---- second pass: generate positioned strokes --------------------------
    result: list[Polyline] = []

    for line_idx, line in enumerate(lines):
        lw = line_widths[line_idx]

        if text_align == "Center":
            line_start_x = -lw / 2.0
        elif text_align == "Right":
            line_start_x = -lw
        else:  # Left — default
            line_start_x = -max_width / 2.0  # align whole block left-edge to same x

        # Baseline y in canvas-style coordinates (y DOWN):
        #   line 0 baseline is at +font_size_mm from top of block.
        #   Top of block is -total_height/2 from origin.
        top_y = -total_height / 2.0
        baseline_y = top_y + font_size_mm + line_idx * line_height_mm

        pen_x = line_start_x

        for ch in line:
            left, right, strokes = glyph_strokes(ch, font_name)

            for stroke in strokes:
                if len(stroke) < 2:
                    continue
                polyline: Polyline = []
                for hx, hy in stroke:
                    x_mm = pen_x + hx * scale
                    y_mm = baseline_y - hy * scale  # flip y (Hershey y is up)
                    polyline.append((x_mm, y_mm))
                for _ in range(stroke_repeat):
                    result.append(list(polyline))

            pen_x += (right - left) * scale + letter_spacing_mm

    return result, max_width, total_height


def _render_ttf_text(
    text: str,
    font_path: str,
    font_size_mm: float,
    letter_spacing_mm: float,
    line_spacing: float,
    text_align: str,
    render_mode: str,
    fill_type: str,
    fill_spacing_mm: float,
    fill_angle: float,
    curve_tolerance_mm: float,
) -> tuple[list[Polyline], float, float]:
    """Render *text* using a TrueType/OpenType font file via fonttools.

    Returns ``(polylines, total_width_mm, total_height_mm)``.
    Raises ``ImportError`` if fonttools is not available.
    Raises ``FileNotFoundError`` if *font_path* does not exist.
    """
    from fontTools.ttLib import TTFont  # type: ignore[import]
    from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import]

    ft_font = TTFont(font_path)
    glyph_set = ft_font.getGlyphSet()
    cmap = ft_font.getBestCmap() or {}
    upem: int = ft_font["head"].unitsPerEm
    scale = font_size_mm / upem

    # Ascent / descent from OS/2 or hhea
    try:
        ascent_u = ft_font["OS/2"].sTypoAscender
        descent_u = ft_font["OS/2"].sTypoDescender  # negative
    except (KeyError, AttributeError):
        ascent_u = int(upem * 0.8)
        descent_u = -int(upem * 0.2)

    ascent_mm = ascent_u * scale
    line_height_mm = font_size_mm * line_spacing

    def _extract_glyph_contours(glyph_name: str) -> list[Polyline]:
        """Sample the glyph outline as closed polylines in mm."""
        pen = RecordingPen()
        try:
            glyph_set[glyph_name].draw(pen)
        except Exception:
            return []

        contours: list[Polyline] = []
        current: Polyline = []

        for op, args in pen.value:
            if op == "moveTo":
                if current:
                    contours.append(current)
                x, y = args[0]
                current = [(x * scale, -y * scale)]
            elif op == "lineTo":
                x, y = args[0]
                current.append((x * scale, -y * scale))
            elif op == "curveTo":
                # Cubic Bezier
                if not current:
                    continue
                p0 = current[-1]
                pts = [(ax * scale, -ay * scale) for ax, ay in args]
                # args: p1, p2, p3 for a cubic
                if len(pts) == 3:
                    sampled = _sample_cubic_bezier(p0, pts[0], pts[1], pts[2],
                                                   curve_tolerance_mm)
                    current.extend(sampled[1:])
                elif len(pts) == 1:
                    current.append(pts[0])
            elif op == "qCurveTo":
                # Quadratic Bezier spline (TrueType).  In fontTools, args are
                # zero or more off-curve control points followed by the final
                # on-curve endpoint.  When multiple off-curves appear in a row,
                # implied on-curve points are inserted at their midpoints.
                if not current:
                    continue
                pts = [(ax * scale, -ay * scale) for ax, ay in args]
                if len(pts) == 1:
                    # No control point — treat as a line segment.
                    current.append(pts[0])
                elif len(pts) == 2:
                    # Simple quadratic: p0 → cp → end.
                    sampled = _sample_quadratic_bezier(
                        current[-1], pts[0], pts[1], curve_tolerance_mm
                    )
                    current.extend(sampled[1:])
                else:
                    # TrueType implied on-curve spline: split into individual
                    # quadratic segments with implied midpoints between
                    # consecutive off-curve points.
                    p_start = current[-1]
                    num_offcurve = len(pts) - 1  # last pt is the on-curve end
                    for i in range(num_offcurve):
                        cp = pts[i]
                        if i == num_offcurve - 1:
                            # Last off-curve: pair with the explicit endpoint.
                            p_end = pts[-1]
                        else:
                            # Implied on-curve midpoint between two off-curves.
                            p_end = ((cp[0] + pts[i + 1][0]) / 2,
                                     (cp[1] + pts[i + 1][1]) / 2)
                        sampled = _sample_quadratic_bezier(
                            p_start, cp, p_end, curve_tolerance_mm
                        )
                        current.extend(sampled[1:])
                        p_start = p_end
            elif op in ("closePath", "endPath"):
                if len(current) >= 2:
                    if current[0] != current[-1]:
                        current.append(current[0])
                    contours.append(current)
                current = []

        if len(current) >= 2:
            contours.append(current)

        return contours

    lines = text.split("\n") if text else [""]

    def _glyph_name(ch: str) -> str | None:
        return cmap.get(ord(ch))

    def _advance(ch: str) -> float:
        gn = _glyph_name(ch)
        if gn is None:
            return font_size_mm * 0.5
        try:
            w = glyph_set[gn].width
        except Exception:
            w = upem * 0.5
        return w * scale + letter_spacing_mm

    # ---- compute line widths ------------------------------------------------
    def _line_width(line: str) -> float:
        if not line:
            return 0.0
        return sum(_advance(ch) for ch in line) - letter_spacing_mm

    line_widths = [_line_width(ln) for ln in lines]
    max_width = max(line_widths, default=0.0)
    total_height = ascent_mm + (len(lines) - 1) * line_height_mm

    # ---- render each line ---------------------------------------------------
    result: list[Polyline] = []
    top_y = -total_height / 2.0

    for line_idx, line in enumerate(lines):
        lw = line_widths[line_idx]

        if text_align == "Center":
            line_start_x = -lw / 2.0
        elif text_align == "Right":
            line_start_x = -lw
        else:
            line_start_x = -max_width / 2.0

        # baseline_y: y position of baseline in canvas-mm (y DOWN)
        baseline_y = top_y + ascent_mm + line_idx * line_height_mm

        pen_x = line_start_x

        for ch in line:
            gn = _glyph_name(ch)
            if gn is None:
                pen_x += font_size_mm * 0.5 + letter_spacing_mm
                continue

            contours = _extract_glyph_contours(gn)

            # Translate contours to current pen position
            shifted: list[Polyline] = []
            for contour in contours:
                shifted.append([(x + pen_x, y + baseline_y) for x, y in contour])

            if render_mode in ("Outline", "Outline + Filled"):
                result.extend(shifted)

            if render_mode in ("Filled", "Outline + Filled") and shifted:
                # Classify contours into outer boundaries and holes so that
                # counter spaces ("o", "p", "B", "8", etc.) are not filled.
                glyph_groups = _classify_glyph_contours(shifted)
                for outer, glyph_holes in glyph_groups:
                    fill_lines = _compute_fill(outer, glyph_holes, fill_type,
                                               fill_spacing_mm, fill_angle)
                    result.extend(fill_lines)

            pen_x += _advance(ch)

    return result, max_width, total_height


def _compute_fill(
    polygon: Polyline,
    holes: list[Polyline],
    fill_type: str,
    spacing_mm: float,
    angle_deg: float,
) -> list[Polyline]:
    """Fill *polygon* (minus *holes*) with the selected fill pattern."""
    # NOTE: depends on private helpers from contour.py. If contour.py is
    # refactored, update these imports accordingly (or extract shared fill
    # utilities to a common module).
    from plottter.generators.contour import (  # lazy import to avoid circular
        _fill_polygon_concentric,
        _fill_polygon_hatch,
    )

    if fill_type == "Hatching":
        return _fill_polygon_hatch(polygon, holes, angle_deg, spacing_mm)
    elif fill_type == "Cross-hatch":
        lines1 = _fill_polygon_hatch(polygon, holes, angle_deg, spacing_mm)
        lines2 = _fill_polygon_hatch(polygon, holes, angle_deg + 90.0, spacing_mm)
        return lines1 + lines2
    elif fill_type == "Concentric":
        return _fill_polygon_concentric(polygon, holes, spacing_mm)
    else:
        return _fill_polygon_hatch(polygon, holes, angle_deg, spacing_mm)


def _signed_area(contour: Polyline) -> float:
    """Compute signed area via the shoelace formula.

    Positive result means counter-clockwise winding in standard math
    coordinates (y-up).  Used internally for debugging; the main
    classification uses geometric containment which is font-format
    agnostic.
    """
    n = len(contour)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = contour[i]
        x1, y1 = contour[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area * 0.5


def _point_in_contour(px: float, py: float, contour: Polyline) -> bool:
    """Ray-casting even-odd point-in-polygon test."""
    inside = False
    j = len(contour) - 1
    for i in range(len(contour)):
        xi, yi = contour[i]
        xj, yj = contour[j]
        if (yi > py) != (yj > py):
            denom = yj - yi
            if denom != 0.0 and px < (xj - xi) * (py - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def _classify_glyph_contours(
    contours: list[Polyline],
) -> list[tuple[Polyline, list[Polyline]]]:
    """Separate glyph contours into outer shapes and their holes.

    Returns a list of ``(outer_contour, [hole_contours])`` pairs.
    Works for both TrueType and CFF/OTF fonts regardless of their
    winding-direction convention, because it uses the geometric
    even-odd containment rule rather than signed area:

    * A contour whose centroid is inside an *even* number of other
      contours is an outer boundary (0, 2, 4 … enclosing rings).
    * A contour whose centroid is inside an *odd* number of other
      contours is a hole.

    Each hole is then assigned to the single outer contour that
    directly contains it.
    """
    if not contours:
        return []
    if len(contours) == 1:
        return [(contours[0], [])]

    n = len(contours)

    def _centroid(c: Polyline) -> tuple[float, float]:
        if not c:
            return (0.0, 0.0)
        return (sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c))

    centroids = [_centroid(c) for c in contours]
    # Use absolute area to avoid false containment: a contour can only be
    # contained inside a strictly larger one.  Without the area guard the
    # centroid of a large outer ring can fall inside a small inner hole that
    # is concentric, causing both to be mis-classified as holes.
    areas = [abs(_signed_area(c)) for c in contours]

    containment = [0] * n
    for i in range(n):
        if not contours[i]:
            continue
        px, py = centroids[i]
        for j in range(n):
            if i == j or not contours[j]:
                continue
            # Skip contours that are the same size or smaller; a contour
            # cannot meaningfully be contained inside a smaller one.
            if areas[j] <= areas[i]:
                continue
            if _point_in_contour(px, py, contours[j]):
                containment[i] += 1

    outers = [i for i in range(n) if contours[i] and containment[i] % 2 == 0]
    hole_idxs = [i for i in range(n) if contours[i] and containment[i] % 2 == 1]

    result: list[tuple[Polyline, list[Polyline]]] = []
    for oi in outers:
        outer = contours[oi]
        associated_holes: list[Polyline] = []
        for hi in hole_idxs:
            phx, phy = centroids[hi]
            if _point_in_contour(phx, phy, outer):
                associated_holes.append(contours[hi])
        result.append((outer, associated_holes))

    return result if result else [(c, []) for c in contours]


def _rotate_polylines(
    polylines: list[Polyline],
    angle_deg: float,
    cx: float,
    cy: float,
) -> list[Polyline]:
    """Rotate polylines counter-clockwise by *angle_deg* around ``(cx, cy)``."""
    if angle_deg == 0.0:
        return polylines
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    result: list[Polyline] = []
    for poly in polylines:
        rotated: Polyline = []
        for x, y in poly:
            dx, dy = x - cx, y - cy
            rotated.append((cx + dx * cos_a - dy * sin_a,
                             cy + dx * sin_a + dy * cos_a))
        result.append(rotated)
    return result


def _translate_polylines(
    polylines: list[Polyline],
    tx: float,
    ty: float,
) -> list[Polyline]:
    """Translate all polylines by ``(tx, ty)``."""
    if tx == 0.0 and ty == 0.0:
        return polylines
    return [[(x + tx, y + ty) for x, y in poly] for poly in polylines]


def _resolve_font_path(system_font_path: str, params: dict) -> str:
    """Resolve a font file path from ``system_font_path`` or ``font_family``/``font_style``.

    Priority
    --------
    1. *system_font_path* if it points to an existing file — used as-is.
    2. Resolve from ``params["font_family"]`` + ``params["font_style"]`` via
       the system/Google-font discovery catalog (includes already-downloaded
       Google Fonts in ``~/.plottter/fonts/google/``).
    3. Auto-download the font from Google Fonts via
       :func:`plottter.fonts.google_fonts.download_google_font`.

    Returns an empty string if resolution fails; callers should fall back to
    Hershey rendering in that case.
    """
    import os

    # 1. Direct file path — fast path, no catalog lookup needed.
    if system_font_path and os.path.isfile(system_font_path):
        return system_font_path

    font_family: str = params.get("font_family", "")
    font_style: str = params.get("font_style", "regular")
    if not font_family:
        return ""

    # 2. System catalog (includes previously downloaded Google Fonts).
    try:
        from plottter.fonts.discovery import get_font_path  # type: ignore[import]

        # Discovery uses title-cased style names ("Regular", "Bold", …).
        resolved = get_font_path(font_family, font_style.title())
        if not resolved:
            # Fall back to any available style for the family.
            resolved = get_font_path(font_family)
        if resolved:
            return resolved
    except ImportError:
        pass

    # 3. Auto-download from Google Fonts (first-time access).
    try:
        from plottter.fonts.google_fonts import download_google_font  # type: ignore[import]

        return download_google_font(font_family, font_style)
    except Exception:
        return ""


def _clip_to_canvas(polylines: list[Polyline], canvas: Canvas) -> list[Polyline]:
    """Remove any polylines that are entirely outside the canvas drawing area.

    Points outside the drawing area are kept (the caller should run the full
    clip processor for strict clipping); here we just drop entirely-invisible
    paths to avoid obvious out-of-bounds artefacts.
    """
    x1, y1, x2, y2 = canvas.drawing_area()
    result: list[Polyline] = []
    pad = 5.0  # tolerance
    for poly in polylines:
        if any(x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad
               for x, y in poly):
            result.append(poly)
    return result


# ---------------------------------------------------------------------------
# TextGenerator
# ---------------------------------------------------------------------------

@register_generator
class TextGenerator(Generator):
    """Render vector text as plotter-ready polylines.

    Supports Hershey single-stroke fonts (built-in, no dependencies) and
    TrueType/OpenType outline fonts (optional, requires ``fonttools``).
    """

    name = "Text"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            # ---- content ---------------------------------------------------
            StringParam(
                name="text",
                label="Text",
                default="Hello\nPlotter",
                multiline=True,
                randomizable=False,
                description="Text to render. Use \\n for line breaks.",
            ),

            # ---- font backend ---------------------------------------------
            ChoiceParam(
                name="font_type",
                label="Font type",
                choices=["Hershey", "System Font"],
                default="Hershey",
                description="Hershey fonts are single-stroke (plotter-native, no deps). "
                            "System Font requires fonttools and a .ttf/.otf file.",
            ),

            # ---- Hershey options ------------------------------------------
            ChoiceParam(
                name="hershey_font",
                label="Hershey font",
                choices=["Simplex", "Duplex", "Script", "Gothic"],
                default="Simplex",
                visible_when={"font_type": ["Hershey"]},
                choice_descriptions={
                    "Simplex": "Single-stroke sans-serif (most pen-efficient)",
                    "Duplex": "Slightly wider double-stroke style",
                    "Script": "Slanted cursive single-stroke",
                    "Gothic": "Angular gothic/blackletter single-stroke",
                },
            ),
            IntParam(
                name="stroke_repeat",
                label="Stroke repeat",
                min=1,
                max=3,
                step=1,
                default=1,
                visible_when={"font_type": ["Hershey"]},
                description="Trace each stroke N times for a bolder appearance.",
            ),

            # ---- System Font options --------------------------------------
            FontParam(
                name="system_font_path",
                label="Font",
                default="",
                randomizable=False,
                visible_when={"font_type": ["System Font"]},
                description="Select a font family and style from the system catalog.",
            ),
            FloatParam(
                name="curve_tolerance_mm",
                label="Curve tolerance (mm)",
                min=0.05,
                max=2.0,
                step=0.05,
                default=0.5,
                visible_when={"font_type": ["System Font"]},
                description="Chord tolerance for Bezier curve sampling. "
                            "Lower = more accurate but more points.",
            ),
            ChoiceParam(
                name="render_mode",
                label="Render mode",
                choices=["Outline", "Filled", "Outline + Filled"],
                default="Outline",
                visible_when={"font_type": ["System Font"]},
                description="Outline traces glyph contours; Filled fills interiors; both combines them.",
            ),
            ChoiceParam(
                name="fill_type",
                label="Fill type",
                choices=["Hatching", "Cross-hatch", "Concentric"],
                default="Hatching",
                visible_when={"font_type": ["System Font"],
                              "render_mode": ["Filled", "Outline + Filled"]},
            ),
            FloatParam(
                name="fill_spacing_mm",
                label="Fill spacing (mm)",
                min=0.1,
                max=5.0,
                step=0.05,
                default=0.3,
                visible_when={"font_type": ["System Font"],
                              "render_mode": ["Filled", "Outline + Filled"]},
            ),
            FloatParam(
                name="fill_angle",
                label="Fill angle (°)",
                min=0.0,
                max=180.0,
                step=1.0,
                default=45.0,
                visible_when={"font_type": ["System Font"],
                              "render_mode": ["Filled", "Outline + Filled"],
                              "fill_type": ["Hatching", "Cross-hatch"]},
            ),

            # ---- layout ---------------------------------------------------
            FloatParam(
                name="font_size_mm",
                label="Font size (mm)",
                min=2.0,
                max=200.0,
                step=0.5,
                default=10.0,
                description="Height of capital letters in mm.",
            ),
            FloatParam(
                name="letter_spacing_mm",
                label="Letter spacing (mm)",
                min=-2.0,
                max=10.0,
                step=0.1,
                default=0.0,
                description="Extra space between characters. Negative = tighter.",
            ),
            FloatParam(
                name="line_spacing",
                label="Line spacing",
                min=0.5,
                max=3.0,
                step=0.1,
                default=1.2,
                description="Line height as a multiplier of font size.",
            ),
            ChoiceParam(
                name="text_align",
                label="Alignment",
                choices=["Left", "Center", "Right"],
                default="Center",
            ),

            # ---- position / transform -------------------------------------
            FloatParam(
                name="x_offset_mm",
                label="X offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                description="Horizontal offset from centre of drawing area.",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                description="Vertical offset from centre of drawing area "
                            "(positive = downward).",
            ),
            FloatParam(
                name="rotation_deg",
                label="Rotation (°)",
                min=0.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate text counter-clockwise by this angle.",
            ),
        ]

    # -------------------------------------------------------------------------

    def get_presets(self) -> list[Preset]:
        _shared = {
            "font_type": "Hershey",
            "hershey_font": "Simplex",
            "stroke_repeat": 1,
            "system_font_path": "",
            "curve_tolerance_mm": 0.5,
            "render_mode": "Outline",
            "fill_type": "Hatching",
            "fill_spacing_mm": 0.3,
            "fill_angle": 45.0,
            "letter_spacing_mm": 0.0,
            "line_spacing": 1.2,
            "text_align": "Center",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            "rotation_deg": 0.0,
        }
        return [
            Preset(
                name="Title / Hershey Sans",
                params={
                    **_shared,
                    "text": "Plottter",
                    "hershey_font": "Simplex",
                    "font_size_mm": 20.0,
                    "text_align": "Center",
                },
            ),
            Preset(
                name="Body / Hershey Serif",
                params={
                    **_shared,
                    "text": "Hello\nPlotter",
                    "hershey_font": "Duplex",
                    "font_size_mm": 8.0,
                    "text_align": "Left",
                },
            ),
            Preset(
                name="Script Signature",
                params={
                    **_shared,
                    "text": "Plotter Art",
                    "hershey_font": "Script",
                    "font_size_mm": 14.0,
                    "letter_spacing_mm": 0.5,
                    "text_align": "Center",
                },
            ),
            Preset(
                name="Gothic Bold",
                params={
                    **_shared,
                    "text": "ART",
                    "hershey_font": "Gothic",
                    "font_size_mm": 30.0,
                    "stroke_repeat": 2,
                    "letter_spacing_mm": 1.0,
                    "text_align": "Center",
                },
            ),
            # ---- Google Fonts presets (auto-downloaded on first use) -------
            Preset(
                name="Elegant Serif",
                params={
                    **_shared,
                    "text": "Plottter",
                    "font_type": "System Font",
                    "system_font_path": "",
                    "font_family": "Playfair Display",
                    "font_style": "regular",
                    "font_size_mm": 30.0,
                    "render_mode": "Outline",
                    "text_align": "Center",
                },
            ),
            Preset(
                name="Clean Sans",
                params={
                    **_shared,
                    "text": "Hello\nPlotter",
                    "font_type": "System Font",
                    "system_font_path": "",
                    "font_family": "Open Sans",
                    "font_style": "regular",
                    "font_size_mm": 25.0,
                    "render_mode": "Outline",
                    "text_align": "Center",
                },
            ),
            Preset(
                name="Handwritten",
                params={
                    **_shared,
                    "text": "Plotter Art",
                    "font_type": "System Font",
                    "system_font_path": "",
                    "font_family": "Caveat",
                    "font_style": "regular",
                    "font_size_mm": 35.0,
                    "render_mode": "Outline",
                    "text_align": "Center",
                },
            ),
            Preset(
                name="Hatched Display",
                params={
                    **_shared,
                    "text": "HELLO",
                    "font_type": "System Font",
                    "system_font_path": "",
                    "font_family": "Lobster",
                    "font_style": "regular",
                    "font_size_mm": 40.0,
                    "render_mode": "Filled",
                    "fill_type": "Hatching",
                    "fill_spacing_mm": 0.4,
                    "fill_angle": 45.0,
                    "text_align": "Center",
                },
            ),
            Preset(
                name="Monospace Code",
                params={
                    **_shared,
                    "text": "Hello\nWorld",
                    "font_type": "System Font",
                    "system_font_path": "",
                    "font_family": "JetBrains Mono",
                    "font_style": "regular",
                    "font_size_mm": 15.0,
                    "render_mode": "Outline",
                    "text_align": "Center",
                },
            ),
            Preset(
                name="Concentric Rings",
                params={
                    **_shared,
                    "text": "ART",
                    "font_type": "System Font",
                    "system_font_path": "",
                    "font_family": "Fredoka",
                    "font_style": "regular",
                    "font_size_mm": 45.0,
                    "render_mode": "Outline + Filled",
                    "fill_type": "Concentric",
                    "fill_spacing_mm": 0.5,
                    "text_align": "Center",
                },
            ),
        ]

    # -------------------------------------------------------------------------

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Render text and return polylines in canvas mm coordinates."""

        text: str = params.get("text", "Hello\nPlotter") or ""
        font_type: str = params.get("font_type", "Hershey")
        hershey_font: str = params.get("hershey_font", "Simplex")
        stroke_repeat: int = max(1, int(params.get("stroke_repeat", 1)))
        system_font_path: str = params.get("system_font_path", "")
        curve_tolerance_mm: float = float(params.get("curve_tolerance_mm", 0.5))
        render_mode: str = params.get("render_mode", "Outline")
        fill_type: str = params.get("fill_type", "Hatching")
        fill_spacing_mm: float = float(params.get("fill_spacing_mm", 0.3))
        fill_angle: float = float(params.get("fill_angle", 45.0))
        font_size_mm: float = max(0.5, float(params.get("font_size_mm", 10.0)))
        letter_spacing_mm: float = float(params.get("letter_spacing_mm", 0.0))
        line_spacing: float = max(0.1, float(params.get("line_spacing", 1.2)))
        text_align: str = params.get("text_align", "Center")
        x_offset_mm: float = float(params.get("x_offset_mm", 0.0))
        y_offset_mm: float = float(params.get("y_offset_mm", 0.0))
        rotation_deg: float = float(params.get("rotation_deg", 0.0))

        if not text.strip():
            return []

        if progress_callback:
            progress_callback(10)

        # ---- render text ---------------------------------------------------
        polylines: list[Polyline] = []

        # Resolve the font path — handles system paths, catalog lookup, and
        # automatic Google Fonts downloads for preset-based rendering.
        resolved_font_path = _resolve_font_path(system_font_path, params)

        if font_type == "System Font" and resolved_font_path:
            try:
                polylines, _w, _h = _render_ttf_text(
                    text=text,
                    font_path=resolved_font_path,
                    font_size_mm=font_size_mm,
                    letter_spacing_mm=letter_spacing_mm,
                    line_spacing=line_spacing,
                    text_align=text_align,
                    render_mode=render_mode,
                    fill_type=fill_type,
                    fill_spacing_mm=fill_spacing_mm,
                    fill_angle=fill_angle,
                    curve_tolerance_mm=curve_tolerance_mm,
                )
            except ImportError:
                # fonttools not available — fall back to Hershey
                polylines, _w, _h = _render_hershey_text(
                    text=text,
                    font_name=hershey_font,
                    font_size_mm=font_size_mm,
                    letter_spacing_mm=letter_spacing_mm,
                    line_spacing=line_spacing,
                    text_align=text_align,
                    stroke_repeat=stroke_repeat,
                )
            except Exception:
                # Any other error (bad path, corrupt font) — fall back
                polylines, _w, _h = _render_hershey_text(
                    text=text,
                    font_name=hershey_font,
                    font_size_mm=font_size_mm,
                    letter_spacing_mm=letter_spacing_mm,
                    line_spacing=line_spacing,
                    text_align=text_align,
                    stroke_repeat=stroke_repeat,
                )
        else:
            polylines, _w, _h = _render_hershey_text(
                text=text,
                font_name=hershey_font,
                font_size_mm=font_size_mm,
                letter_spacing_mm=letter_spacing_mm,
                line_spacing=line_spacing,
                text_align=text_align,
                stroke_repeat=stroke_repeat,
            )

        if not polylines:
            return []

        if progress_callback:
            progress_callback(60)

        # ---- position in canvas drawing area --------------------------------
        x1, y1, x2, y2 = canvas.drawing_area()
        cx = (x1 + x2) / 2.0 + x_offset_mm
        cy = (y1 + y2) / 2.0 + y_offset_mm

        # _render_*_text produces polylines centred at the origin; translate
        # to the desired canvas position.
        polylines = _translate_polylines(polylines, cx, cy)

        if rotation_deg != 0.0:
            polylines = _rotate_polylines(polylines, rotation_deg, cx, cy)

        if progress_callback:
            progress_callback(80)

        # ---- keep only paths with at least some visible portion --------------
        polylines = _clip_to_canvas(polylines, canvas)

        if progress_callback:
            progress_callback(100)

        return polylines
