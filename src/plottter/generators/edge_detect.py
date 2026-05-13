"""EdgeDetectGenerator — Canny edge detection + contour tracing."""

from __future__ import annotations

from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import _px_to_mm, compute_image_rect
from plottter.generators.base import (
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
    BoolParam,
)
from plottter.models import Canvas, Polyline


def _close_gaps(
    polylines: list[Polyline],
    threshold_mm: float,
) -> list[Polyline]:
    """Connect endpoints of nearby polylines within threshold_mm distance."""
    if threshold_mm <= 0 or len(polylines) < 2:
        return polylines

    threshold_sq = threshold_mm * threshold_mm
    result = [list(p) for p in polylines]
    merged = [False] * len(result)
    output: list[Polyline] = []

    i = 0
    while i < len(result):
        if merged[i]:
            i += 1
            continue
        current = result[i]
        changed = True
        while changed:
            changed = False
            for j in range(i + 1, len(result)):
                if merged[j]:
                    continue
                other = result[j]
                if not current or not other:
                    continue
                # Check all 4 endpoint combinations
                cs, ce = current[0], current[-1]
                os_, oe = other[0], other[-1]
                combos = [
                    (ce, os_, "ce-os"),
                    (ce, oe, "ce-oe"),
                    (cs, os_, "cs-os"),
                    (cs, oe, "cs-oe"),
                ]
                best = None
                best_dist = threshold_sq
                for c_pt, o_pt, combo_type in combos:
                    dx = c_pt[0] - o_pt[0]
                    dy = c_pt[1] - o_pt[1]
                    d2 = dx * dx + dy * dy
                    if d2 <= best_dist:
                        best_dist = d2
                        best = combo_type

                if best is not None:
                    merged[j] = True
                    changed = True
                    if best == "ce-os":
                        current = current + other
                    elif best == "ce-oe":
                        current = current + list(reversed(other))
                    elif best == "cs-os":
                        current = list(reversed(other)) + current
                    elif best == "cs-oe":
                        current = other + current
                    result[i] = current
                    break
        if not merged[i]:
            output.append(current)
        i += 1

    return output


@register_generator
class EdgeDetectGenerator(Generator):
    """Canny edge detection with contour tracing to produce plotter-ready polylines."""

    name = "Edge Detect"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="low_threshold",
                label="Canny Low Threshold",
                min=0.0,
                max=255.0,
                step=1.0,
                default=50.0,
                description="Lower Canny threshold — edges with gradient below this value are rejected",
            ),
            FloatParam(
                name="high_threshold",
                label="Canny High Threshold",
                min=0.0,
                max=255.0,
                step=1.0,
                default=150.0,
                description="Upper Canny threshold — edges with gradient above this value are always accepted",
            ),
            IntParam(
                name="min_contour_length",
                label="Min Contour Length (pts)",
                min=2,
                max=1000,
                step=1,
                default=10,
                description="Minimum number of points in a contour to keep — removes tiny specks and noise",
            ),
            FloatParam(
                name="simplify_tolerance_mm",
                label="Simplify Tolerance (mm)",
                min=0.0,
                max=10.0,
                step=0.1,
                default=0.5,
                description="RDP simplification tolerance in mm — reduces point count while preserving shape (0 = no simplification)",
            ),
            FloatParam(
                name="close_gaps_mm",
                label="Close Gaps (mm)",
                min=0.0,
                max=20.0,
                step=0.1,
                default=2.0,
                description="Maximum gap between endpoints to bridge — connects nearby open contours into longer paths",
            ),
            BoolParam(
                name="smooth_curves",
                label="Smooth Curves (Bezier Fitting)",
                default=False,
                description="Fit cubic Bezier curves to the output polylines for smooth, organic-looking strokes",
            ),
            FloatParam(
                name="curve_tolerance_mm",
                label="Curve Tolerance (mm)",
                min=0.1,
                max=5.0,
                step=0.05,
                default=0.5,
                visible_when={"smooth_curves": [True]},
                description="How closely the fitted curves must follow the original points — lower = more faithful but more points",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image before processing (useful for white line art on dark backgrounds)",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before processing (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before processing (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=0.0,
                description="Gaussian blur radius applied before edge detection — reduces noise but softens edges (0 = no blur)",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the generated output on the canvas page (mm)",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the generated output on the canvas page (mm)",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default",
                params={
                    "low_threshold": 50.0,
                    "high_threshold": 150.0,
                    "min_contour_length": 10,
                    "simplify_tolerance_mm": 0.5,
                    "close_gaps_mm": 2.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Fine Detail",
                params={
                    "low_threshold": 20.0,
                    "high_threshold": 80.0,
                    "min_contour_length": 5,
                    "simplify_tolerance_mm": 0.2,
                    "close_gaps_mm": 1.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Bold Outlines",
                params={
                    # Moderate Canny thresholds combined with blur + contrast
                    # preprocessing: blur suppresses fine texture so only the
                    # strongest gradient edges (mountain silhouettes, horizon
                    # lines) survive; contrast boost amplifies those main
                    # transitions.  min_contour_length and large simplify
                    # tolerance keep only the prominent, continuous strokes.
                    "low_threshold": 40.0,
                    "high_threshold": 110.0,
                    "min_contour_length": 15,
                    "simplify_tolerance_mm": 1.5,
                    "close_gaps_mm": 5.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 2.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Portrait Photo",
                params={
                    # Low thresholds to capture soft transitions in photographs;
                    # small min_contour_length keeps facial features; gap closing
                    # connects broken curves common in photo edges; contrast boost
                    # and slight blur help reveal soft facial gradients.
                    "low_threshold": 15.0,
                    "high_threshold": 60.0,
                    "min_contour_length": 8,
                    "simplify_tolerance_mm": 0.3,
                    "close_gaps_mm": 3.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="High Contrast Illustration",
                params={
                    # Higher thresholds pick out only strong edges typical of
                    # ink illustrations, cartoons, or line art; generous gap
                    # closing reconnects dashed strokes.
                    "low_threshold": 80.0,
                    "high_threshold": 180.0,
                    "min_contour_length": 15,
                    "simplify_tolerance_mm": 0.8,
                    "close_gaps_mm": 4.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 30.0,
                    "blur_radius": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Landscape Photo",
                params={
                    # Moderate thresholds to capture horizon lines and terrain
                    # edges; slight contrast boost helps distant features; blur
                    # suppresses sky noise common in photographs.
                    "low_threshold": 30.0,
                    "high_threshold": 100.0,
                    "min_contour_length": 10,
                    "simplify_tolerance_mm": 0.5,
                    "close_gaps_mm": 2.0,
                    "smooth_curves": False,
                    "curve_tolerance_mm": 0.5,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 15.0,
                    "blur_radius": 1.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "The 'opencv-python' package is required for Edge Detect generation. "
                "Install it with: pip install opencv-python"
            ) from exc

        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
            to_grayscale,
        )

        # Apply preprocessing
        img = source.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 0.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            img = adjust_brightness(img, brightness)
        if contrast != 0.0:
            img = adjust_contrast(img, contrast)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        if do_invert:
            img = invert_image(img)

        # Ensure grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        source = img

        low_thresh = float(params.get("low_threshold", 50.0))
        high_thresh = float(params.get("high_threshold", 150.0))
        min_len = int(params.get("min_contour_length", 10))
        simplify_tol_mm = float(params.get("simplify_tolerance_mm", 0.5))
        close_gaps_mm = float(params.get("close_gaps_mm", 2.0))

        if progress_callback:
            progress_callback(10)

        if cancelled_callback and cancelled_callback():
            return []

        # Canny edge detection
        edges = cv2.Canny(source, low_thresh, high_thresh)

        if progress_callback:
            progress_callback(30)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        if progress_callback:
            progress_callback(50)

        if not contours:
            return []

        img_h, img_w = source.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        # Convert mm tolerance to pixels for RDP
        mm_per_px = (img_x2 - img_x1) / img_w
        simplify_tol_px = simplify_tol_mm / mm_per_px if mm_per_px > 0 else 1.0

        polylines: list[Polyline] = []
        total = len(contours)
        for idx, contour in enumerate(contours):
            if cancelled_callback and cancelled_callback():
                break

            if len(contour) < min_len:
                continue

            # RDP simplification
            epsilon = simplify_tol_px
            simplified = cv2.approxPolyDP(contour, epsilon, closed=False)

            if len(simplified) < 2:
                continue

            # Convert to mm coordinates
            poly: Polyline = []
            for pt in simplified:
                px_x = float(pt[0][0])
                px_y = float(pt[0][1])
                mm_pt = _px_to_mm(px_x, px_y, img_w, img_h, img_x1, img_y1, img_x2, img_y2)
                poly.append(mm_pt)

            if len(poly) >= 2:
                polylines.append(poly)

            if progress_callback and idx % 50 == 0:
                progress_callback(50 + int(idx / total * 40))

        if progress_callback:
            progress_callback(90)

        # Close gaps between nearby endpoints
        if close_gaps_mm > 0 and polylines:
            polylines = _close_gaps(polylines, close_gaps_mm)

        if bool(params.get("smooth_curves", False)):
            from plottter.processing.curves import fit_curves as _fit_curves
            _tol = float(params.get("curve_tolerance_mm", 0.5))
            polylines = _fit_curves(polylines, _tol)

        if progress_callback:
            progress_callback(100)

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            polylines = [[(x + x_off, y + y_off) for x, y in path] for path in polylines]
        return polylines
