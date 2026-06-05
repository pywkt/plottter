"""vectorize_trace.py — Vectorize / Trace Bitmap plugin for Plottter.

Converts a raster source image into clean vector polylines using the potrace
algorithm (via the ``potracer`` Python package). This produces the "Trace
Bitmap" / "Image Trace" aesthetic familiar from Inkscape and Illustrator —
optimal smooth Bezier curves tracing filled silhouette regions.

Licensing context
-----------------
This app is **MIT-licensed**. The ``potracer`` package is **GPLv2**. To avoid
GPL distribution obligations, this plugin is shipped in the un-packaged
``plugins/`` directory (not included in the wheel) and uses the standard
lazy-import optional-dependency pattern. Users opt in to the GPL terms by
installing the package on their own machine::

    pip install potracer

The built-in **Contour Lines** generator (Phase 162) offers a MIT-clean
alternative: sub-pixel marching-squares + optional Bezier curve-fit, which
covers ~90 % of potrace's smoothness for most line-art and outline work.
Use that generator if you want zero GPL-touching code.

Pipeline
--------
1. Load the grayscale source image via ``params["_source_image"]``.
2. For each threshold level (``num_levels ≥ 1``):
   a. Binarize: pixels darker than ``threshold`` become foreground.
   b. Trace with ``potrace.Bitmap.trace()`` (the ``potracer`` package installs
      as the importable module ``potrace``).
   c. Flatten each cubic Bezier segment to polyline points at
      ``curve_tolerance_mm`` tolerance via adaptive De Casteljau subdivision.
   d. Map potrace pixel coordinates to mm using ``compute_image_rect`` (honours
      ``image_fit_mode``).
3. Return one ``LayerSpec`` per level (``generate_layers()`` / multi-layer
   mode) or concatenate all layers into a flat polyline list (``generate()``).

Optional-dependency guard
-------------------------
``_require_potracer()`` is called at the top of ``generate()`` and
``generate_layers()``. It raises ``RuntimeError`` with a helpful install
message when ``potracer`` is absent. The module itself imports cleanly without
``potracer``, so the plugin loads and registers even when the package is not
installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    LayerSpec,
    Preset,
)
from plottter.models.canvas import Canvas

logger = logging.getLogger(__name__)

Polyline = list[tuple[float, float]]

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------


def _require_potracer():
    """Import and return the potracer module, or raise a friendly RuntimeError.

    Called at the top of ``generate()`` and ``generate_layers()``. Never
    called at import time so the plugin registers even when potracer is absent.
    """
    try:
        # The `potracer` PyPI package installs as the importable module
        # `potrace` (it is a drop-in for pypotrace).
        import potrace  # type: ignore[import]

        return potrace
    except ImportError:
        raise RuntimeError(
            "The potracer package is required for the Vectorize / Trace Bitmap "
            "generator.\nInstall it with:\n  pip install potracer"
        )


# ---------------------------------------------------------------------------
# Bezier flattening helpers
# ---------------------------------------------------------------------------


def _flatten_bezier_segment(
    p0: tuple[float, float],
    c0: tuple[float, float],
    c1: tuple[float, float],
    p1: tuple[float, float],
    tolerance_sq: float,
    depth: int = 0,
) -> list[tuple[float, float]]:
    """Adaptively flatten a cubic Bezier to polyline points (excluding p0).

    Uses De Casteljau subdivision until the midpoint deviation from the chord
    is within *sqrt(tolerance_sq)* pixels.  ``depth`` caps recursion at 12
    levels to guard against degenerate curves.
    """
    if depth >= 12:
        return [p1]
    # Midpoint on the bezier curve (de Casteljau at t=0.5)
    mx = (p0[0] + 3 * c0[0] + 3 * c1[0] + p1[0]) / 8.0
    my = (p0[1] + 3 * c0[1] + 3 * c1[1] + p1[1]) / 8.0
    # Midpoint of the chord
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    dx, dy = mx - cx, my - cy
    if dx * dx + dy * dy <= tolerance_sq:
        return [p1]
    # Subdivide via De Casteljau
    p01 = ((p0[0] + c0[0]) / 2.0, (p0[1] + c0[1]) / 2.0)
    p12 = ((c0[0] + c1[0]) / 2.0, (c0[1] + c1[1]) / 2.0)
    p23 = ((c1[0] + p1[0]) / 2.0, (c1[1] + p1[1]) / 2.0)
    p012 = ((p01[0] + p12[0]) / 2.0, (p01[1] + p12[1]) / 2.0)
    p123 = ((p12[0] + p23[0]) / 2.0, (p12[1] + p23[1]) / 2.0)
    pmid = ((p012[0] + p123[0]) / 2.0, (p012[1] + p123[1]) / 2.0)
    return _flatten_bezier_segment(
        p0, p01, p012, pmid, tolerance_sq, depth + 1
    ) + _flatten_bezier_segment(pmid, p123, p23, p1, tolerance_sq, depth + 1)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def _potrace_to_mm(
    px: float,
    py: float,
    img_w: int,
    img_h: int,
    img_x1: float,
    img_y1: float,
    img_x2: float,
    img_y2: float,
) -> tuple[float, float]:
    """Convert potrace pixel coordinates to mm.

    The ``potracer`` package traces a numpy array and returns coordinates in
    **array space**: ``y`` matches the row index, so (x=0, y=0) is the
    *top-left* corner and ``y`` increases *downward* — the same convention as
    our screen-space mm rect, where (img_x1, img_y1) is the top-left and
    (img_x2, img_y2) is the bottom-right.  No vertical flip is applied (an
    earlier version assumed a y-up convention and produced upside-down output).
    """
    mm_x = img_x1 + px * (img_x2 - img_x1) / img_w
    mm_y = img_y1 + py * (img_y2 - img_y1) / img_h
    return (mm_x, mm_y)


# ---------------------------------------------------------------------------
# Single-level trace
# ---------------------------------------------------------------------------


def _trace_level(
    gray: np.ndarray,
    threshold: int,
    img_w: int,
    img_h: int,
    img_x1: float,
    img_y1: float,
    img_x2: float,
    img_y2: float,
    curve_tolerance_mm: float,
    turdsize: int,
    alphamax: float,
    opttolerance: float,
) -> list[Polyline]:
    """Binarize at *threshold* and trace with potrace.

    Returns a list of closed polylines in mm coordinates.
    """
    import potrace  # the `potracer` PyPI package installs as module `potrace`

    # Binarize: dark pixels (< threshold) become foreground (True = black)
    binary = gray < threshold

    # Compute flattening tolerance in pixel space
    img_w_mm = max(img_x2 - img_x1, 1e-6)
    px_per_mm = img_w / img_w_mm
    tol_px = max(0.05, curve_tolerance_mm * px_per_mm)
    tol_sq = tol_px * tol_px

    # Trace
    bm = potrace.Bitmap(binary)
    path = bm.trace(turdsize=turdsize, alphamax=alphamax, opttolerance=opttolerance)

    def _xy(p) -> tuple[float, float]:
        # potrace Point exposes .x / .y attributes (it is NOT subscriptable).
        return (p.x, p.y)

    polylines: list[Polyline] = []
    for curve in path:
        segments = curve.segments
        if not segments:
            continue

        start = _xy(curve.start_point)
        pts: list[tuple[float, float]] = [
            _potrace_to_mm(start[0], start[1], img_w, img_h, img_x1, img_y1, img_x2, img_y2)
        ]
        prev = start  # pixel-space tuple

        for segment in segments:
            end = _xy(segment.end_point)
            if not segment.is_corner:
                # BezierSegment: cubic control points are c1, c2.
                c1 = _xy(segment.c1)
                c2 = _xy(segment.c2)
                flat = _flatten_bezier_segment(prev, c1, c2, end, tol_sq)
                pts.extend(
                    _potrace_to_mm(p[0], p[1], img_w, img_h, img_x1, img_y1, img_x2, img_y2)
                    for p in flat
                )
            else:
                # CornerSegment: line to the corner vertex c, then to end_point.
                c = _xy(segment.c)
                pts.append(
                    _potrace_to_mm(c[0], c[1], img_w, img_h, img_x1, img_y1, img_x2, img_y2)
                )
                pts.append(
                    _potrace_to_mm(end[0], end[1], img_w, img_h, img_x1, img_y1, img_x2, img_y2)
                )
            prev = end

        if len(pts) >= 2:
            # Close the curve
            pts.append(pts[0])
            polylines.append(pts)

    return polylines


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

# Palette of layer colors for multi-level tonal output
_LEVEL_COLORS = ["#000000", "#333333", "#666666", "#888888", "#aaaaaa", "#bbbbbb", "#cccccc", "#dddddd"]


@register_generator
class VectorizeTraceGenerator(Generator):
    """Vectorize / Trace Bitmap — potrace-based raster-to-vector generator.

    Traces the source image with the potrace algorithm (via the optional
    ``potracer`` package) to produce clean filled-shape outlines with smooth
    cubic Bezier curves flattened to plotter-friendly polylines.

    This generator requires ``pip install potracer`` (GPLv2).  The built-in
    **Contour Lines** generator (Phase 162 sub-pixel mode) is the MIT-clean
    alternative that requires no extra install.
    """

    name = "Vectorize / Trace Bitmap"
    category = "image"
    uses_source_image = True
    emits_multiple_layers = True

    def get_parameters(self):
        return [
            # --- Image placement ---
            ChoiceParam(
                name="image_fit_mode",
                label="Image Fit",
                choices=["fill", "fit", "custom"],
                default="fill",
                description=(
                    "fill: image occupies the full drawing area; "
                    "fit: scale to fit preserving aspect ratio; "
                    "custom: explicit width/height."
                ),
            ),
            FloatParam(
                name="image_width_mm",
                label="Width (mm)",
                min=1.0,
                max=2000.0,
                step=1.0,
                default=200.0,
                visible_when={"image_fit_mode": ["custom"]},
                description="Output image width in mm (custom fit mode only).",
            ),
            FloatParam(
                name="image_height_mm",
                label="Height (mm)",
                min=1.0,
                max=2000.0,
                step=1.0,
                default=200.0,
                visible_when={"image_fit_mode": ["custom"]},
                description="Output image height in mm (custom fit mode only).",
            ),
            FloatParam(
                name="image_offset_x_mm",
                label="Offset X (mm)",
                min=-1000.0,
                max=1000.0,
                step=1.0,
                default=0.0,
                visible_when={"image_fit_mode": ["fit", "custom"]},
                description="Horizontal offset from the centered position in mm.",
            ),
            FloatParam(
                name="image_offset_y_mm",
                label="Offset Y (mm)",
                min=-1000.0,
                max=1000.0,
                step=1.0,
                default=0.0,
                visible_when={"image_fit_mode": ["fit", "custom"]},
                description="Vertical offset from the centered position in mm.",
            ),
            # --- Threshold ---
            IntParam(
                name="threshold",
                label="Threshold",
                min=1,
                max=254,
                step=1,
                default=128,
                description=(
                    "Pixels darker than this value (0–255) are treated as "
                    "foreground and traced. Lower = only very dark areas traced."
                ),
            ),
            # --- Curve quality ---
            FloatParam(
                name="curve_tolerance_mm",
                label="Curve Tolerance (mm)",
                min=0.01,
                max=5.0,
                step=0.01,
                default=0.2,
                description=(
                    "Bezier flattening tolerance in mm. Lower values produce "
                    "smoother (more detailed) polylines at the cost of more points."
                ),
            ),
            # --- Potrace knobs ---
            IntParam(
                name="turdsize",
                label="Despeckle",
                min=0,
                max=500,
                step=1,
                default=2,
                description=(
                    "Suppress speckles and regions smaller than this area "
                    "(in pixels). Increase to remove noise."
                ),
            ),
            FloatParam(
                name="alphamax",
                label="Corner Sharpness",
                min=0.0,
                max=1.334,
                step=0.05,
                default=1.0,
                description=(
                    "Corner detection threshold. Lower values → sharper corners; "
                    "higher values → rounder corners / more Bezier curves."
                ),
            ),
            FloatParam(
                name="opttolerance",
                label="Optimize Tolerance",
                min=0.0,
                max=1.0,
                step=0.01,
                default=0.2,
                description=(
                    "potrace's internal Bezier curve optimisation tolerance. "
                    "Higher = fewer, coarser curves."
                ),
            ),
            # --- Multi-level tonal ---
            IntParam(
                name="num_levels",
                label="Tone Levels",
                min=1,
                max=8,
                step=1,
                default=1,
                description=(
                    "Number of threshold levels to trace for layered tonal output. "
                    "1 = single silhouette outline; >1 = one layer per tone band, "
                    "thresholds spaced evenly from 32 to the configured Threshold."
                ),
            ),
        ]

    def get_presets(self):
        return [
            Preset(
                name="Logo / Silhouette",
                description="Clean single-threshold potrace trace of a high-contrast image",
                params={
                    "threshold": 128,
                    "num_levels": 1,
                    "turdsize": 2,
                    "alphamax": 1.0,
                    "opttolerance": 0.2,
                    "curve_tolerance_mm": 0.2,
                    "image_fit_mode": "fill",
                },
            ),
            Preset(
                name="Tonal Layers (4)",
                description="Four threshold levels traced as separate layers for shaded output",
                params={
                    "threshold": 200,
                    "num_levels": 4,
                    "turdsize": 4,
                    "alphamax": 1.0,
                    "opttolerance": 0.2,
                    "curve_tolerance_mm": 0.3,
                    "image_fit_mode": "fill",
                },
            ),
            Preset(
                name="Fine Detail",
                description="High-quality trace with tight curve tolerance for detailed artwork",
                params={
                    "threshold": 140,
                    "num_levels": 1,
                    "turdsize": 0,
                    "alphamax": 0.5,
                    "opttolerance": 0.1,
                    "curve_tolerance_mm": 0.05,
                    "image_fit_mode": "fill",
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_image_rect(
        self, params: dict[str, Any], gray: np.ndarray, canvas: Canvas
    ) -> tuple[int, int, float, float, float, float]:
        """Return (img_w, img_h, img_x1, img_y1, img_x2, img_y2) from params."""
        img_h, img_w = gray.shape
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w,
            img_h,
            draw_x1,
            draw_y1,
            draw_x2,
            draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        return img_w, img_h, img_x1, img_y1, img_x2, img_y2

    def _compute_thresholds(self, num_levels: int, max_threshold: int) -> list[int]:
        """Compute evenly-spaced threshold values for multi-level tracing."""
        if num_levels == 1:
            return [max_threshold]
        step = max(1, max_threshold // num_levels)
        thresholds = []
        for i in range(num_levels):
            t = max(1, min(254, step + i * step))
            thresholds.append(t)
        return sorted(set(thresholds))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Trace the source image and return flat polylines in mm.

        Calls ``generate_layers()`` internally and concatenates all layer
        paths into a single list.  Raises ``RuntimeError`` with a helpful
        ``pip install potracer`` message when ``potracer`` is not installed.
        """
        _require_potracer()
        layers = self.generate_layers(params, canvas, progress_callback, cancelled_callback)
        return [path for layer in layers for path in layer.paths]

    def generate_layers(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        """Trace the source image and return one LayerSpec per tone level.

        Raises ``RuntimeError`` with a helpful ``pip install potracer`` message
        when ``potracer`` is not installed.
        """
        _require_potracer()

        image = params.get("_source_image")
        if image is None:
            return []

        from plottter.io.image_import import to_grayscale

        gray = to_grayscale(image.copy())

        if progress_callback:
            progress_callback(5)
        if cancelled_callback and cancelled_callback():
            return []

        img_w, img_h, img_x1, img_y1, img_x2, img_y2 = self._get_image_rect(
            params, gray, canvas
        )

        threshold = int(params.get("threshold", 128))
        curve_tolerance_mm = float(params.get("curve_tolerance_mm", 0.2))
        turdsize = int(params.get("turdsize", 2))
        alphamax = float(params.get("alphamax", 1.0))
        opttolerance = float(params.get("opttolerance", 0.2))
        num_levels = int(params.get("num_levels", 1))

        thresholds = self._compute_thresholds(num_levels, threshold)
        layer_specs: list[LayerSpec] = []

        for i, thr in enumerate(thresholds):
            if cancelled_callback and cancelled_callback():
                break
            if progress_callback:
                progress_callback(5 + int(i / len(thresholds) * 90))

            polylines = _trace_level(
                gray,
                thr,
                img_w,
                img_h,
                img_x1,
                img_y1,
                img_x2,
                img_y2,
                curve_tolerance_mm,
                turdsize,
                alphamax,
                opttolerance,
            )

            color = _LEVEL_COLORS[i % len(_LEVEL_COLORS)]
            name = f"Level {i + 1} (thr={thr})" if len(thresholds) > 1 else "Trace"
            layer_specs.append(LayerSpec(name=name, color=color, paths=polylines))

        if progress_callback:
            progress_callback(100)

        return layer_specs
