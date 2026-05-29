"""calligraphy.py — Calligraphy / Variable-Width Strokes plugin for Plottter.

This plugin renders paths with parallel offset strokes, producing the look of a
broad-nib calligraphy pen.  The stroke thickness varies based on the direction
of travel relative to a simulated pen nib angle: maximum width when the stroke
is perpendicular to the nib, minimum width when parallel (Phase B).

Plugin Installation
-------------------
Place this file in one of:

- ``~/.config/plottter/plugins/``   (user-level, always available)
- ``<project-root>/plugins/``         (project-level, available when launched
                                        from that directory)

Plottter discovers and registers the plugin automatically on startup.
No extra dependencies are required beyond those already installed with
Plottter (``numpy``, ``math``).

Plugin Development Guide
------------------------
To create a new Plottter generator plugin:

1. **Import the required pieces**::

       from plottter.generators import register_generator
       from plottter.generators.base import (
           ChoiceParam, FloatParam, Generator, IntParam,
           Parameter, Preset,
       )
       from plottter.models import Canvas, Polyline

2. **Create a class** that inherits from ``Generator`` and decorate it with
   ``@register_generator``.  Set ``name`` (unique, user-visible) and
   ``category`` (``"math"`` or ``"image"``).

3. **Implement the three abstract methods**:

   ``get_parameters() -> list[Parameter]``
       Return the list of UI-visible parameters.  Available parameter types:
       ``FloatParam``, ``IntParam``, ``BoolParam``, ``ChoiceParam``,
       ``StringParam``, ``ExpressionParam``, ``ColorParam``, ``ImageParam``.
       Use ``visible_when={"param_name": ["value1", "value2"]}`` to
       conditionally show parameters based on other param values.

   ``get_presets() -> list[Preset]``
       Return named preset configurations (dict of param name → value).

   ``generate(params, canvas, progress_callback, cancelled_callback)``
       The main generation method.  Read param values from the ``params``
       dict (always use ``.get()`` with a default for safety), compute
       polylines in mm coordinates, and return them as ``list[Polyline]``
       (where ``Polyline = list[tuple[float, float]]``).

4. **Common patterns**::

       draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
       progress_callback(0.5)   # report 50 % progress (float 0.0–1.0)
       if cancelled_callback and cancelled_callback():
           return []            # user cancelled — stop early

5. **Testing**:
   Create ``tests/test_<pluginname>_plugin.py``.
   Use ``Canvas(width_mm=210, height_mm=297, margin_mm=10)`` for A4.
   Import ``plottter.generators`` to trigger registration, then access
   ``GENERATORS["Calligraphy"]`` to verify the class was registered.

Common pitfalls
~~~~~~~~~~~~~~~
- Do **not** mutate the ``params`` dict; always use ``.get()`` with defaults.
- Guard against ``cancelled_callback()`` returning ``True`` in long loops.
- Return ``[]`` (empty list) for degenerate inputs — never raise.
- All coordinates must be in **mm** (not pixels).
- Preset param dicts must only reference params defined in ``get_parameters()``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.fonts.hershey import (
    DEFAULT_FONT_NAME,
    choices_for_param as _hershey_choices,
)
from plottter.generators import register_generator
from plottter.generators.base import (
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
    StringParam,
)
from plottter.models import Canvas, Polyline

# ---------------------------------------------------------------------------
# Section: Offset Curve Engine
#
# Given a centerline path and width values, produce a family of parallel
# offset polylines.  This is the geometric heart of the calligraphy effect.
# ---------------------------------------------------------------------------


def _compute_normals(
    path: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Compute a unit normal vector at each point of a polyline.

    Normals are perpendicular to the local tangent direction and point
    "upward" relative to the direction of travel, using the convention::

        normal = (dy, -dx) / |tangent|

    so that a path travelling to the right (positive X) has normals
    pointing in the negative-Y direction, i.e. ``(0, -1)``.

    Central differences are used for interior points; forward/backward
    differences are used at the endpoints.  When consecutive points are
    identical (zero-length segment), the most recent valid normal is
    reused to avoid division-by-zero.

    Parameters
    ----------
    path:
        List of ``(x, y)`` points in mm.

    Returns
    -------
    list of ``(nx, ny)`` unit-normal tuples, same length as ``path``.
    """
    n = len(path)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, -1.0)]

    last_valid: tuple[float, float] = (0.0, -1.0)
    normals: list[tuple[float, float]] = []

    for i in range(n):
        if i == 0:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
        elif i == n - 1:
            dx = path[-1][0] - path[-2][0]
            dy = path[-1][1] - path[-2][1]
        else:
            # Central difference: span two segments for stability
            dx = path[i + 1][0] - path[i - 1][0]
            dy = path[i + 1][1] - path[i - 1][1]

        length = math.hypot(dx, dy)
        if length < 1e-10:
            # Degenerate segment — reuse the last valid normal
            normals.append(last_valid)
            continue

        # Rotate tangent 90° clockwise: (dy, -dx) / length
        nx = dy / length
        ny = -dx / length
        last_valid = (nx, ny)
        normals.append(last_valid)

    return normals


def _clean_offset_path(
    path: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Split an offset path at large jumps caused by self-intersections.

    When a centerline curves sharply, offsetting it inward can produce
    self-intersections that appear as large discontinuous jumps in the
    point sequence.  This function detects segments whose length exceeds
    3× the median segment length and splits the path at those points,
    returning a list of clean sub-paths.

    Each returned sub-path contains at least 2 points.

    Parameters
    ----------
    path:
        A single offset polyline (list of ``(x, y)`` points in mm).

    Returns
    -------
    List of sub-paths.  Returns an empty list if ``path`` has fewer than
    2 points.
    """
    if len(path) < 2:
        return []

    # Compute consecutive segment lengths
    lengths: list[float] = []
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        lengths.append(math.hypot(dx, dy))

    sorted_len = sorted(lengths)
    median_len = sorted_len[len(sorted_len) // 2]
    # Threshold: segments longer than 3× the median are treated as jumps
    threshold = 3.0 * median_len if median_len > 1e-10 else float("inf")

    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [path[0]]

    for i, seg_len in enumerate(lengths):
        if seg_len > threshold:
            # Large jump — end the current sub-path and start a new one
            if len(current) >= 2:
                segments.append(current)
            current = [path[i + 1]]
        else:
            current.append(path[i + 1])

    if len(current) >= 2:
        segments.append(current)

    return segments if segments else [path]


def _offset_polyline(
    path: list[tuple[float, float]],
    widths: list[float],
    num_lines: int,
) -> list[Polyline]:
    """Generate parallel offset polylines from a centerline path.

    Produces ``num_lines`` polylines evenly distributed from ``−width/2``
    to ``+width/2`` on either side of the centerline.  The local width at
    each point is taken from the ``widths`` list.

    Self-intersecting portions of each offset curve are detected and split
    by :func:`_clean_offset_path`, so the total number of returned
    polylines may exceed ``num_lines`` when the path curves sharply.

    Parameters
    ----------
    path:
        Centerline polyline as list of ``(x, y)`` points in mm.
    widths:
        Per-point width values in mm.  If shorter than ``path``, the last
        value is repeated for the remaining points.
    num_lines:
        Number of parallel offset lines to produce.
        ``1`` → center line only (no offset).
        ``2`` → left and right edges only (no interior lines).
        ``N > 2`` → N evenly spaced lines from edge to edge.

    Returns
    -------
    List of Polylines (each a list of ``(x, y)`` points in mm).
    """
    if len(path) < 2:
        return []

    normals = _compute_normals(path)

    # Fractional offsets in [−0.5, +0.5]
    if num_lines == 1:
        t_values = [0.0]
    else:
        t_values = [i / (num_lines - 1) - 0.5 for i in range(num_lines)]

    result: list[Polyline] = []
    for t in t_values:
        offset_pts: list[tuple[float, float]] = []
        for i, (pt, nrm) in enumerate(zip(path, normals)):
            w = widths[i] if i < len(widths) else widths[-1]
            dist = t * w
            offset_pts.append((pt[0] + nrm[0] * dist, pt[1] + nrm[1] * dist))
        sub_paths = _clean_offset_path(offset_pts)
        result.extend(sub_paths)

    return result


# ---------------------------------------------------------------------------
# Section: Calligraphic Width Model
#
# Computes per-point stroke widths based on the angle between the local
# stroke direction and the simulated pen-nib angle.  A broad nib is widest
# when stroked perpendicular to its angle and thinnest when stroked
# parallel — exactly the formula used by calligraphers with flat-edged pens.
# ---------------------------------------------------------------------------


def _calligraphic_widths(
    path: list[tuple[float, float]],
    pen_angle_deg: float,
    min_width_mm: float,
    max_width_mm: float,
) -> list[float]:
    """Compute per-point stroke widths for a calligraphic effect.

    At each point the local stroke direction is computed from the path tangent.
    The width is then:

    .. code-block::

        width = min_w + (max_w - min_w) * |sin(direction - pen_angle)|

    This is maximum when the stroke is **perpendicular** to the pen nib
    (|sin| = 1) and minimum when **parallel** (|sin| = 0).

    A Gaussian blur (σ = 3 points) is applied to the resulting width array to
    smooth out abrupt changes caused by noisy tangent estimates.

    Parameters
    ----------
    path:
        Centerline polyline as list of ``(x, y)`` points in mm.
    pen_angle_deg:
        Angle of the pen nib in degrees, measured from the positive-X axis.
        0° → horizontal nib (thick on vertical strokes).
        45° → italic nib.
        90° → vertical nib (thick on horizontal strokes).
    min_width_mm:
        Stroke width when the path runs parallel to the pen nib, in mm.
    max_width_mm:
        Stroke width when the path runs perpendicular to the pen nib, in mm.

    Returns
    -------
    List of width values (in mm), one per point in ``path``.
    """
    n = len(path)
    if n == 0:
        return []

    pen_angle_rad = math.radians(pen_angle_deg)
    w_range = max(0.0, max_width_mm - min_width_mm)

    raw_widths: list[float] = []
    for i in range(n):
        # Compute tangent via central/forward/backward differences
        if i == 0:
            dx = path[1][0] - path[0][0] if n > 1 else 1.0
            dy = path[1][1] - path[0][1] if n > 1 else 0.0
        elif i == n - 1:
            dx = path[-1][0] - path[-2][0]
            dy = path[-1][1] - path[-2][1]
        else:
            dx = path[i + 1][0] - path[i - 1][0]
            dy = path[i + 1][1] - path[i - 1][1]

        length = math.hypot(dx, dy)
        if length < 1e-10:
            # Degenerate — use the last computed width or min_width
            raw_widths.append(raw_widths[-1] if raw_widths else min_width_mm)
            continue

        direction = math.atan2(dy, dx)
        width = min_width_mm + w_range * abs(math.sin(direction - pen_angle_rad))
        raw_widths.append(width)

    # Gaussian smoothing (σ = 3 points) to avoid abrupt width transitions.
    # We use a simple kernel approach; numpy is already a project dependency.
    if n < 3:
        return raw_widths

    arr = np.array(raw_widths, dtype=float)
    sigma = 3.0
    # Build a kernel of radius 3σ (capped at n)
    radius = min(int(3 * sigma), n - 1)
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()

    # Reflect-pad the array so edge values are smoothed correctly
    padded = np.pad(arr, radius, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")

    # Clamp to [min_width_mm, max_width_mm] to handle floating-point drift
    smoothed = np.clip(smoothed, min_width_mm, max_width_mm)

    return smoothed.tolist()


def _speed_widths(
    path: list[tuple[float, float]],
    min_width_mm: float,
    max_width_mm: float,
) -> list[float]:
    """Compute per-point widths inversely proportional to stroke speed.

    Slower strokes (shorter segments) produce wider lines; faster strokes
    (longer segments) produce narrower lines, mimicking natural brush pressure.

    Parameters
    ----------
    path:
        Centerline polyline as list of ``(x, y)`` points in mm.
    min_width_mm, max_width_mm:
        Output width range in mm.

    Returns
    -------
    List of width values in mm.
    """
    n = len(path)
    if n == 0:
        return []
    if n == 1:
        return [min_width_mm]

    # Compute segment lengths
    seg_lengths: list[float] = []
    for i in range(n - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        seg_lengths.append(math.hypot(dx, dy))

    # Per-point speed = average of adjacent segment lengths
    speeds: list[float] = []
    for i in range(n):
        if i == 0:
            s = seg_lengths[0]
        elif i == n - 1:
            s = seg_lengths[-1]
        else:
            s = (seg_lengths[i - 1] + seg_lengths[i]) * 0.5
        speeds.append(s)

    max_speed = max(speeds) if speeds else 1.0
    if max_speed < 1e-10:
        return [min_width_mm] * n

    w_range = max(0.0, max_width_mm - min_width_mm)
    # Invert: slow (small s) → wide, fast (large s) → thin
    return [
        min_width_mm + w_range * (1.0 - s / max_speed)
        for s in speeds
    ]


# ---------------------------------------------------------------------------
# Section: Path Sources
#
# Each function generates a centerline polyline for a particular demo shape.
# All coordinates are in mm and sized proportionally to the canvas drawing
# area so the output looks good on any paper size.
# ---------------------------------------------------------------------------


def _make_circle_path(
    cx: float,
    cy: float,
    radius: float,
    n_pts: int = 256,
) -> list[tuple[float, float]]:
    """Generate a closed circular path centered at ``(cx, cy)``.

    Parameters
    ----------
    cx, cy:
        Centre of the circle in mm.
    radius:
        Radius of the circle in mm.
    n_pts:
        Number of sample points (higher = smoother).
    """
    pts: list[tuple[float, float]] = []
    for k in range(n_pts + 1):
        angle = 2.0 * math.pi * k / n_pts
        pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return pts


def _make_spiral_path(
    cx: float,
    cy: float,
    r_min: float,
    r_max: float,
    turns: float = 4.0,
    n_pts: int = 512,
) -> list[tuple[float, float]]:
    """Generate an Archimedean spiral path centered at ``(cx, cy)``.

    Parameters
    ----------
    cx, cy:
        Centre of the spiral in mm.
    r_min, r_max:
        Starting and ending radius in mm.
    turns:
        Number of full revolutions.
    n_pts:
        Number of sample points.
    """
    if n_pts < 2:
        return []
    pts: list[tuple[float, float]] = []
    total_angle = turns * 2.0 * math.pi
    for k in range(n_pts):
        frac = k / (n_pts - 1)
        angle = total_angle * frac
        r = r_min + (r_max - r_min) * frac
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def _make_wave_path(
    x1: float,
    x2: float,
    cy: float,
    amplitude: float,
    freq: float = 3.0,
    n_pts: int = 300,
) -> list[tuple[float, float]]:
    """Generate a sinusoidal wave path spanning ``[x1, x2]``.

    Parameters
    ----------
    x1, x2:
        Horizontal extent of the wave in mm.
    cy:
        Vertical centre of the wave in mm.
    amplitude:
        Peak displacement from centre in mm.
    freq:
        Number of complete cycles across the full width.
    n_pts:
        Number of sample points.
    """
    if n_pts < 2:
        return []
    pts: list[tuple[float, float]] = []
    for k in range(n_pts):
        frac = k / (n_pts - 1)
        x = x1 + (x2 - x1) * frac
        y = cy + amplitude * math.sin(2.0 * math.pi * freq * frac)
        pts.append((x, y))
    return pts


def _make_figure8_path(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    n_pts: int = 256,
) -> list[tuple[float, float]]:
    """Generate a figure-8 path (lemniscate) centered at ``(cx, cy)``.

    Uses the lemniscate of Bernoulli parametrisation scaled to fit within
    ``rx × ry`` in the horizontal and vertical directions respectively.

    Parameters
    ----------
    cx, cy:
        Centre of the figure-8 in mm.
    rx, ry:
        Horizontal and vertical extents in mm.
    n_pts:
        Number of sample points.
    """
    if n_pts < 1:
        return []
    pts: list[tuple[float, float]] = []
    for k in range(n_pts + 1):
        t = 2.0 * math.pi * k / n_pts
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        denom = 1.0 + sin_t * sin_t
        x = cx + rx * cos_t / denom
        y = cy + ry * sin_t * cos_t / denom
        pts.append((x, y))
    return pts


# ---------------------------------------------------------------------------
# Section: Text Rendering
#
# Converts a text string into a list of centerline polylines using the
# built-in Hershey single-stroke fonts.  Hershey fonts provide exact
# centerline paths (not outlines), so every stroke already has the correct
# form for the calligraphy offset engine.
# ---------------------------------------------------------------------------


def _text_to_centerlines(
    text: str,
    font_name: str,
    font_size_mm: float,
    letter_spacing_mm: float,
    line_spacing: float,
    text_align: str = "Center",
) -> tuple[list[list[tuple[float, float]]], float, float]:
    """Convert *text* to calligraphy-ready centerline polylines.

    Each Hershey glyph stroke is returned as a separate polyline so the
    calligraphy engine can apply variable-width offsets stroke-by-stroke.
    The entire text block is centred at the local origin ``(0, 0)`` with Y
    increasing downward.  Callers apply the final canvas translation.

    Parameters
    ----------
    text:
        The text to render.  Use ``\\n`` for multi-line input.
    font_name:
        Canonical Hershey/EMS font name from
        :func:`plottter.fonts.hershey.list_names`.  Legacy aliases
        ``"Simplex"``/``"Duplex"``/``"Script"``/``"Gothic"`` are still
        accepted and resolve to their modern equivalents.
    font_size_mm:
        Height of capital letters in mm.
    letter_spacing_mm:
        Extra space added between characters (may be negative).
    line_spacing:
        Line height as a multiplier of *font_size_mm*.
    text_align:
        ``"Left"``, ``"Center"``, or ``"Right"`` alignment within the
        text block.

    Returns
    -------
    ``(centerlines, total_width_mm, total_height_mm)`` where:

    * ``centerlines`` — list of polylines (each a list of ``(x, y)``
      points in mm), centred at origin with Y pointing down.
    * ``total_width_mm`` — width of the widest text line.
    * ``total_height_mm`` — total height of the text block.
    """
    if not text:
        return [], 0.0, 0.0

    from plottter.generators._hershey import CAP_HEIGHT, glyph_strokes

    scale = font_size_mm / CAP_HEIGHT
    line_height_mm = font_size_mm * line_spacing
    lines = text.split("\n")

    # ---- First pass: compute per-line widths for alignment -----------------
    def _line_width(line: str) -> float:
        if not line:
            return 0.0
        w = 0.0
        for ch in line:
            left, right, _ = glyph_strokes(ch, font_name)
            w += (right - left) * scale + letter_spacing_mm
        return max(0.0, w - letter_spacing_mm)  # no trailing spacing

    line_widths = [_line_width(ln) for ln in lines]
    max_width = max(line_widths, default=0.0)
    total_height = font_size_mm + (len(lines) - 1) * line_height_mm

    # ---- Second pass: generate positioned centerline strokes ---------------
    result: list[list[tuple[float, float]]] = []

    for line_idx, line in enumerate(lines):
        lw = line_widths[line_idx]

        # Horizontal start of each line based on alignment.
        if text_align == "Center":
            line_start_x = -lw / 2.0
        elif text_align == "Right":
            line_start_x = -lw
        else:  # "Left" — align whole block's left edge to -max_width/2
            line_start_x = -max_width / 2.0

        # Baseline Y in canvas-style coordinates (Y DOWN):
        #   The text block spans from -total_height/2 to +total_height/2.
        #   Line 0 baseline is font_size_mm below the top of the block.
        top_y = -total_height / 2.0
        baseline_y = top_y + font_size_mm + line_idx * line_height_mm

        pen_x = line_start_x

        for ch in line:
            left, right, strokes = glyph_strokes(ch, font_name)

            for stroke in strokes:
                if len(stroke) < 2:
                    continue
                polyline: list[tuple[float, float]] = []
                for hx, hy in stroke:
                    x_mm = pen_x + hx * scale
                    y_mm = baseline_y - hy * scale  # flip Y (Hershey Y is up)
                    polyline.append((x_mm, y_mm))
                result.append(polyline)

            pen_x += (right - left) * scale + letter_spacing_mm

    return result, max_width, total_height


# ---------------------------------------------------------------------------
# Section: Generator Class
# ---------------------------------------------------------------------------


@register_generator
class CalligraphyGenerator(Generator):
    """Calligraphy / Variable-Width Strokes generator.

    Renders a demo centerline path (Circle, Spiral, Wave, or Figure 8) with
    a bundle of parallel offset polylines, producing the characteristic look
    of a broad-nib calligraphy pen.

    In **Calligraphic** width mode (the default) the stroke width at each
    point varies based on the angle between the local stroke direction and
    the simulated pen nib angle: maximum width when perpendicular to the nib,
    minimum width when parallel.  An optional speed-influence blend further
    modulates width by stroke speed (slower → wider, faster → thinner).

    In **Constant** width mode all parallel offset lines use a fixed stroke
    width (Phase A behaviour).

    Text rendering using Hershey single-stroke fonts is available via the
    ``"Text"`` path source (Phase C, task 17.3).  Each glyph stroke is
    treated as an independent centerline so variable-width offsets are applied
    stroke-by-stroke, giving each letter authentic calligraphic character.
    """

    name = "Calligraphy"
    category = "math"

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[Parameter]:
        """Return the parameter list for the Calligraphy generator.

        Parameters
        ----------
        path_source : ChoiceParam
            Which demo centerline shape to render.
        width_mode : ChoiceParam
            ``"Calligraphic"`` (default) — width varies with stroke direction;
            ``"Constant"`` — uniform stroke width (Phase A behaviour).
        pen_angle_deg : FloatParam
            Pen nib angle in degrees (visible when Calligraphic).
        min_width_mm : FloatParam
            Minimum stroke width in mm (visible when Calligraphic).
        max_width_mm : FloatParam
            Maximum stroke width in mm (visible when Calligraphic).
        speed_influence : FloatParam
            Blend speed-based width variation into the calligraphic model (0 = off).
        num_parallel_lines : IntParam
            How many parallel offset lines make up the stroke bundle.
        stroke_width_mm : FloatParam
            Constant stroke width in mm (visible when Constant mode).
        """
        return [
            ChoiceParam(
                "path_source",
                "Path Source",
                choices=["Circle", "Spiral", "Wave", "Figure 8", "Text"],
                default="Circle",
                description=(
                    "The centerline shape to render with calligraphic strokes. "
                    "'Text' renders typed text using a Hershey single-stroke font."
                ),
                randomizable=False,
            ),
            # ---- Text-mode parameters (visible only when path_source = "Text") ----
            StringParam(
                "text",
                "Text",
                default="Hello",
                description=(
                    "The text to render. Use \\n for multi-line text."
                ),
                visible_when={"path_source": ["Text"]},
            ),
            ChoiceParam(
                "hershey_font",
                "Font",
                # Sourced from the shared catalog so calligraphy gets new fonts
                # automatically.  EMSAllure / HersheyScript1 are the natural
                # picks for calligraphy due to their cursive forms.
                choices=_hershey_choices()[0],
                default="EMSAllure",
                description=(
                    "Single-stroke font.  EMS Allure and Hershey Script give "
                    "the most calligraphic look; EMS Readability is the "
                    "clearest non-cursive choice."
                ),
                choice_descriptions=_hershey_choices()[1],
                visible_when={"path_source": ["Text"]},
                randomizable=False,
            ),
            FloatParam(
                "font_size_mm",
                "Font Size (mm)",
                min=5.0,
                max=100.0,
                step=1.0,
                default=30.0,
                description="Height of capital letters in mm.",
                visible_when={"path_source": ["Text"]},
            ),
            FloatParam(
                "letter_spacing_mm",
                "Letter Spacing (mm)",
                min=-5.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description=(
                    "Extra space added between characters in mm. "
                    "Negative values tighten the spacing."
                ),
                visible_when={"path_source": ["Text"]},
            ),
            FloatParam(
                "line_spacing",
                "Line Spacing",
                min=0.5,
                max=3.0,
                step=0.1,
                default=1.5,
                description=(
                    "Line height as a multiplier of the font size. "
                    "1.0 = single-spaced, 1.5 = standard, 2.0 = double-spaced."
                ),
                visible_when={"path_source": ["Text"]},
            ),
            ChoiceParam(
                "text_align",
                "Alignment",
                choices=["Left", "Center", "Right"],
                default="Center",
                description=(
                    "Horizontal alignment of multi-line text blocks."
                ),
                visible_when={"path_source": ["Text"]},
                randomizable=False,
            ),
            FloatParam(
                "x_offset_mm",
                "X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                description=(
                    "Horizontal offset from the canvas centre in mm. "
                    "Positive = right."
                ),
                visible_when={"path_source": ["Text"]},
            ),
            FloatParam(
                "y_offset_mm",
                "Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                description=(
                    "Vertical offset from the canvas centre in mm. "
                    "Positive = down."
                ),
                visible_when={"path_source": ["Text"]},
            ),
            ChoiceParam(
                "width_mode",
                "Width Mode",
                choices=["Calligraphic", "Constant"],
                default="Calligraphic",
                description=(
                    "Calligraphic: stroke width varies based on pen nib angle — "
                    "thick when perpendicular, thin when parallel. "
                    "Constant: uniform width throughout."
                ),
                randomizable=False,
            ),
            FloatParam(
                "pen_angle_deg",
                "Pen Angle (°)",
                min=0.0,
                max=180.0,
                step=5.0,
                default=45.0,
                description=(
                    "Angle of the pen nib in degrees. "
                    "0° = horizontal nib (thick on vertical strokes). "
                    "45° = italic nib. "
                    "90° = vertical nib (thick on horizontal strokes)."
                ),
                visible_when={"width_mode": ["Calligraphic"]},
            ),
            FloatParam(
                "min_width_mm",
                "Min Width (mm)",
                min=0.1,
                max=10.0,
                step=0.1,
                default=0.5,
                description=(
                    "Minimum stroke width in mm — used when the stroke runs "
                    "parallel to the pen nib."
                ),
                visible_when={"width_mode": ["Calligraphic"]},
            ),
            FloatParam(
                "max_width_mm",
                "Max Width (mm)",
                min=0.5,
                max=20.0,
                step=0.5,
                default=5.0,
                description=(
                    "Maximum stroke width in mm — used when the stroke runs "
                    "perpendicular to the pen nib."
                ),
                visible_when={"width_mode": ["Calligraphic"]},
            ),
            FloatParam(
                "speed_influence",
                "Speed Influence",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.0,
                description=(
                    "Blend speed-based width variation into the calligraphic model. "
                    "0 = pure calligraphic model; 1 = pure speed-based (slower strokes "
                    "wider, faster strokes thinner). Values in between blend both."
                ),
                visible_when={"width_mode": ["Calligraphic"]},
            ),
            IntParam(
                "num_parallel_lines",
                "Parallel Lines",
                min=1,
                max=20,
                step=1,
                default=6,
                description=(
                    "Number of parallel offset lines drawn across the stroke width. "
                    "1 = center line only. 2 = left and right edges only. "
                    "Higher values fill in the interior."
                ),
            ),
            FloatParam(
                "stroke_width_mm",
                "Stroke Width (mm)",
                min=0.5,
                max=20.0,
                step=0.1,
                default=4.0,
                description=(
                    "Total width of the calligraphic stroke in millimetres "
                    "(used in Constant mode only). "
                    "The offset lines are spread evenly across this width."
                ),
                visible_when={"width_mode": ["Constant"]},
            ),
        ]

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def get_presets(self) -> list[Preset]:
        """Return the named presets for the Calligraphy generator."""
        return [
            Preset(
                "Italic Circle",
                {
                    "path_source": "Circle",
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 45.0,
                    "min_width_mm": 0.5,
                    "max_width_mm": 5.0,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 8,
                },
            ),
            Preset(
                "Brush Pen Spiral",
                {
                    "path_source": "Spiral",
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 60.0,
                    "min_width_mm": 0.2,
                    "max_width_mm": 4.0,
                    "speed_influence": 0.3,
                    "num_parallel_lines": 6,
                },
            ),
            Preset(
                "Wavy Calligraphy",
                {
                    "path_source": "Wave",
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 0.0,
                    "min_width_mm": 0.3,
                    "max_width_mm": 6.0,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 8,
                },
            ),
            Preset(
                "Figure 8 Flourish",
                {
                    "path_source": "Figure 8",
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 30.0,
                    "min_width_mm": 0.5,
                    "max_width_mm": 7.0,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 10,
                },
            ),
            # ----------------------------------------------------------------
            # Phase D presets — added in task 17.4
            # ----------------------------------------------------------------
            Preset(
                "Broad Nib Italic",
                {
                    "path_source": "Text",
                    "text": "Hello",
                    "hershey_font": "EMSAllure",
                    "font_size_mm": 35.0,
                    "letter_spacing_mm": 1.0,
                    "line_spacing": 1.5,
                    "text_align": "Center",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 40.0,
                    "min_width_mm": 0.3,
                    "max_width_mm": 5.0,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 8,
                },
            ),
            Preset(
                "Monoline Script",
                {
                    "path_source": "Text",
                    "text": "Hello",
                    "hershey_font": "EMSAllure",
                    "font_size_mm": 25.0,
                    "letter_spacing_mm": 1.0,
                    "line_spacing": 1.5,
                    "text_align": "Center",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "width_mode": "Constant",
                    "stroke_width_mm": 1.0,
                    "num_parallel_lines": 1,
                },
            ),
            Preset(
                "Gothic Blackletter",
                {
                    "path_source": "Text",
                    "text": "Hello",
                    "hershey_font": "HersheyGothEnglish",
                    "font_size_mm": 40.0,
                    "letter_spacing_mm": 1.0,
                    "line_spacing": 1.5,
                    "text_align": "Center",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 30.0,
                    "min_width_mm": 0.5,
                    "max_width_mm": 6.0,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 10,
                },
            ),
            Preset(
                "Thin Copperplate",
                {
                    "path_source": "Text",
                    "text": "Hello",
                    "hershey_font": "EMSReadability",
                    "font_size_mm": 20.0,
                    "letter_spacing_mm": 1.0,
                    "line_spacing": 1.5,
                    "text_align": "Center",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 55.0,
                    "min_width_mm": 0.1,
                    "max_width_mm": 2.0,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 4,
                },
            ),
            Preset(
                "Decorative Wave",
                {
                    "path_source": "Wave",
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 45.0,
                    "min_width_mm": 0.3,
                    "max_width_mm": 3.5,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 8,
                },
            ),
            # ----------------------------------------------------------------
            # Backward-compatible constant-width presets
            # ----------------------------------------------------------------
            Preset(
                "Wide Brush Circle",
                {
                    "path_source": "Circle",
                    "width_mode": "Constant",
                    "num_parallel_lines": 8,
                    "stroke_width_mm": 8.0,
                },
            ),
            Preset(
                "Fine Spiral",
                {
                    "path_source": "Spiral",
                    "width_mode": "Constant",
                    "num_parallel_lines": 4,
                    "stroke_width_mm": 2.0,
                },
            ),
            # Text presets (Phase C)
            Preset(
                "Italic Script Text",
                {
                    "path_source": "Text",
                    "text": "Hello",
                    "hershey_font": "EMSAllure",
                    "font_size_mm": 35.0,
                    "letter_spacing_mm": 1.0,
                    "line_spacing": 1.5,
                    "text_align": "Center",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "width_mode": "Calligraphic",
                    "pen_angle_deg": 45.0,
                    "min_width_mm": 0.3,
                    "max_width_mm": 5.0,
                    "speed_influence": 0.0,
                    "num_parallel_lines": 8,
                },
            ),
            Preset(
                "Monoline Script Text",
                {
                    "path_source": "Text",
                    "text": "Hello",
                    "hershey_font": "EMSAllure",
                    "font_size_mm": 25.0,
                    "letter_spacing_mm": 1.0,
                    "line_spacing": 1.5,
                    "text_align": "Center",
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                    "width_mode": "Constant",
                    "stroke_width_mm": 1.0,
                    "num_parallel_lines": 1,
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Generate parallel-offset calligraphic strokes.

        Reads parameters from ``params``, builds the chosen centerline path
        scaled to the canvas drawing area, computes per-point widths (either
        calligraphic or constant), and returns the set of parallel offset
        polylines.

        In **Calligraphic** mode the width at each point is proportional to
        ``|sin(stroke_direction − pen_angle)|``, producing thick strokes
        where the direction is perpendicular to the nib and thin strokes
        where it is parallel — exactly the behaviour of a broad-nib pen.
        An optional ``speed_influence`` blends speed-based width variation
        (slow strokes wider, fast strokes thinner) into the model.

        Parameters
        ----------
        params:
            Dict of parameter values (from the GUI or a Preset).
        canvas:
            The current canvas — used to obtain the drawing area bounds.
        progress_callback:
            Optional callable accepting a float in ``[0.0, 1.0]``.
        cancelled_callback:
            Optional callable returning ``True`` when the user has
            cancelled generation.

        Returns
        -------
        List of Polylines in mm coordinates.
        """
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        path_source: str = params.get("path_source", "Circle")
        width_mode: str = params.get("width_mode", "Calligraphic")
        # Clamp num_parallel_lines to the valid range [1, 20].
        num_lines: int = max(1, min(20, int(params.get("num_parallel_lines", 6))))
        stroke_width_mm: float = float(params.get("stroke_width_mm", 4.0))
        pen_angle_deg: float = float(params.get("pen_angle_deg", 45.0))
        min_width_mm: float = float(params.get("min_width_mm", 0.5))
        max_width_mm: float = float(params.get("max_width_mm", 5.0))
        speed_influence: float = float(params.get("speed_influence", 0.0))
        speed_influence = max(0.0, min(1.0, speed_influence))

        if progress_callback:
            progress_callback(0.1)

        # ---- Build the centerline path(s) ----
        if path_source == "Text":
            # Text mode: render each Hershey glyph stroke as a separate
            # centerline, apply calligraphic offsets to each, and collect.
            text: str = params.get("text", "Hello") or ""
            hershey_font: str = params.get("hershey_font", "EMSAllure")
            font_size_mm: float = float(params.get("font_size_mm", 30.0))
            letter_spacing_mm: float = float(params.get("letter_spacing_mm", 1.0))
            line_spacing: float = float(params.get("line_spacing", 1.5))
            text_align: str = params.get("text_align", "Center")
            x_offset_mm: float = float(params.get("x_offset_mm", 0.0))
            y_offset_mm: float = float(params.get("y_offset_mm", 0.0))

            if not text:
                return []

            centerlines, _w, _h = _text_to_centerlines(
                text, hershey_font, font_size_mm,
                letter_spacing_mm, line_spacing, text_align,
            )

            if not centerlines:
                return []

            # Translate from local origin to canvas position.
            tx = cx + x_offset_mm
            ty = cy + y_offset_mm
            translated: list[list[tuple[float, float]]] = [
                [(x + tx, y + ty) for x, y in stroke]
                for stroke in centerlines
            ]

            if progress_callback:
                progress_callback(0.4)

            if cancelled_callback and cancelled_callback():
                return []

            # Apply calligraphic offsets to each glyph stroke individually.
            paths: list[Polyline] = []
            n_strokes = len(translated)
            for stroke_idx, centerline in enumerate(translated):
                if len(centerline) < 2:
                    continue
                if cancelled_callback and cancelled_callback():
                    return []

                if width_mode == "Constant":
                    widths = [stroke_width_mm] * len(centerline)
                else:
                    callig_w = _calligraphic_widths(
                        centerline, pen_angle_deg, min_width_mm, max_width_mm
                    )
                    if speed_influence > 0.0:
                        spd_w = _speed_widths(centerline, min_width_mm, max_width_mm)
                        widths = [
                            cw * (1.0 - speed_influence) + sw * speed_influence
                            for cw, sw in zip(callig_w, spd_w)
                        ]
                    else:
                        widths = callig_w

                paths.extend(_offset_polyline(centerline, widths, num_lines))

                if progress_callback:
                    progress_callback(0.4 + 0.58 * (stroke_idx + 1) / n_strokes)

            if progress_callback:
                progress_callback(1.0)

            return paths

        # ---- Non-text path sources ----
        if cancelled_callback and cancelled_callback():
            return []

        if path_source == "Circle":
            radius = min(draw_w, draw_h) * 0.35
            centerline: list[tuple[float, float]] = _make_circle_path(cx, cy, radius)

        elif path_source == "Spiral":
            r_min = min(draw_w, draw_h) * 0.04
            r_max = min(draw_w, draw_h) * 0.38
            centerline = _make_spiral_path(cx, cy, r_min, r_max, turns=4.0)

        elif path_source == "Wave":
            amplitude = draw_h * 0.15
            centerline = _make_wave_path(draw_x1, draw_x2, cy, amplitude, freq=3.0)

        else:  # "Figure 8"
            rx = draw_w * 0.25
            ry = draw_h * 0.20
            centerline = _make_figure8_path(cx, cy, rx, ry)

        if progress_callback:
            progress_callback(0.4)

        if cancelled_callback and cancelled_callback():
            return []

        # ---- Compute per-point widths ----
        if width_mode == "Constant":
            widths = [stroke_width_mm] * len(centerline)
        else:
            # Calligraphic mode: angle-based width
            callig_widths = _calligraphic_widths(
                centerline, pen_angle_deg, min_width_mm, max_width_mm
            )
            if speed_influence > 0.0:
                spd_widths = _speed_widths(centerline, min_width_mm, max_width_mm)
                widths = [
                    cw * (1.0 - speed_influence) + sw * speed_influence
                    for cw, sw in zip(callig_widths, spd_widths)
                ]
            else:
                widths = callig_widths

        if cancelled_callback and cancelled_callback():
            return []

        paths = _offset_polyline(centerline, widths, num_lines)

        if progress_callback:
            progress_callback(1.0)

        return paths
