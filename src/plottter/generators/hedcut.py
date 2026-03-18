"""HedcutGenerator — Wall Street Journal-style hedcut portrait.

Combines three techniques into a cohesive single-layer output:

1. **Edge outlines** — XDoG, FDoG, or Canny edge detection traces crisp ink
   contour lines that define facial features, hair, and structural elements.

2. **Midtone stipple dots** — Weighted Voronoi stippling via Lloyd relaxation
   in the midtone brightness zone renders the graduated tonal range as
   characteristic hand-drawn dot clusters.

3. **Shadow hatching** — Parallel (and optionally cross-) hatch lines fill the
   darkest shadow regions with density proportional to shadow depth, producing
   the inky engraving quality of authentic hedcut illustrations.

Reference: WSJ editorial illustration style.
"""

from __future__ import annotations

import math
import random as _random
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import _px_to_mm, compute_image_rect
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline

# Sides used to approximate stipple dots as polygons
_DOT_SIDES = 8


def _tiny_circle(cx: float, cy: float, radius_mm: float) -> Polyline:
    """Return a small circular polyline representing a stipple dot."""
    pts: Polyline = []
    for k in range(_DOT_SIDES):
        angle = 2.0 * math.pi * k / _DOT_SIDES
        pts.append((cx + radius_mm * math.cos(angle), cy + radius_mm * math.sin(angle)))
    pts.append(pts[0])
    return pts


def _filled_circle(
    cx: float,
    cy: float,
    radius_mm: float,
    pen_width_mm: float = 0.3,
) -> list[Polyline]:
    """Return concentric circular polylines that fill a stipple dot.

    Starts at the outer radius and steps inward by ``pen_width_mm`` each ring
    until the ring radius would be smaller than half a pen width.  For very
    small dots (radius < pen_width_mm) only a single outline ring is returned
    since one pen stroke will visually fill it.
    """
    if pen_width_mm <= 0:
        pen_width_mm = 0.3
    rings: list[Polyline] = []
    r = radius_mm
    min_r = pen_width_mm / 2.0
    while r > min_r:
        rings.append(_tiny_circle(cx, cy, r))
        r -= pen_width_mm
    if not rings:
        # Degenerate case: dot smaller than half a pen width — draw one ring
        rings.append(_tiny_circle(cx, cy, radius_mm))
    return rings


def _get_edge_binary(
    gray_f: np.ndarray,
    gray_uint8: np.ndarray,
    edge_method: str,
    edge_sigma: float,
) -> np.ndarray:
    """Compute a binary edge map for contour tracing.

    Returns an image in findContours convention:
    - 255 = background / non-edge regions
    - 0   = detected edge bands (boundaries between 255 regions = the actual lines)

    Parameters
    ----------
    gray_f:      Float32 grayscale image in [0, 1].
    gray_uint8:  uint8 grayscale image in [0, 255].
    edge_method: "XDoG", "FDoG", or "Canny".
    edge_sigma:  Edge detection scale parameter.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for HedcutGenerator. "
            "Install with: pip install opencv-python"
        ) from exc

    if edge_method == "XDoG":
        from plottter.generators.xdog import _xdog
        T = _xdog(gray_f, sigma=edge_sigma, k=1.6, phi=100.0, epsilon=0.01)
        # T >= 0.5 = non-edge background → 255; T < 0.5 = edges → 0
        return (T >= 0.5).astype(np.uint8) * 255

    if edge_method == "FDoG":
        from plottter.generators.fdog import _fdog
        T = _fdog(
            gray_f,
            sigma_c=edge_sigma,
            rho=1.6,
            sigma_m=max(0.5, edge_sigma * 2.0),
            etf_iterations=3,
            fdog_iterations=1,
        )
        # Same convention as XDoG: bright = background, dark = edges
        return (T >= 0.5).astype(np.uint8) * 255

    # Canny
    # Derive thresholds from sigma: smaller sigma → detect finer edges
    canny_t1 = max(10, int(50 / max(0.1, edge_sigma)))
    canny_t2 = max(20, int(150 / max(0.1, edge_sigma)))
    edges = cv2.Canny(gray_uint8, canny_t1, canny_t2)
    # Canny: 255=edge, 0=background → invert to match findContours convention
    return cv2.bitwise_not(edges)


def _trace_edge_binary(
    binary: np.ndarray,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    min_len: int = 5,
    simplify_tol_mm: float = 0.3,
    smooth_iterations: int = 1,
) -> list[Polyline]:
    """Trace contours from an edge binary image to mm polylines.

    Expects the findContours convention (255=background, 0=edge bands) —
    contours trace the boundaries of bright regions, producing edge lines.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for HedcutGenerator.") from exc

    from plottter.generators.contour import _chaikin_smooth

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines: list[Polyline] = []
    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    for contour in contours:
        if simplify_tol_mm > 0:
            tol_px = max(1.0, simplify_tol_mm * px_per_mm)
            contour = cv2.approxPolyDP(contour, tol_px, closed=True)

        pts = contour.reshape(-1, 2)
        if len(pts) < min_len:
            continue

        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts
        ]

        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=True)

        if poly and poly[0] != poly[-1]:
            poly.append(poly[0])

        polylines.append(poly)

    return polylines


@register_generator
class HedcutGenerator(Generator):
    """Wall Street Journal-style hedcut portrait generator.

    Combines three techniques into a cohesive single-layer output:

    1. **Edge outlines** — XDoG, FDoG, or Canny edge detection traces crisp
       ink contour lines that define features, hair, and other structural
       elements of the portrait.

    2. **Midtone stipple dots** — Weighted Voronoi stippling (Lloyd relaxation)
       in the midtone brightness zone renders the graduated tonal range as
       characteristic hand-drawn dot clusters.

    3. **Shadow hatching** — Parallel (and optionally cross-) hatch lines fill
       the darkest shadow regions with density proportional to shadow depth,
       producing the inky engraving quality of authentic hedcut illustrations.

    Reference: WSJ editorial illustration style.
    """

    name = "Hedcut"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            # --- Tonal zone thresholds ---
            IntParam(
                name="highlight_threshold",
                label="Highlight Threshold",
                min=128,
                max=255,
                step=1,
                default=200,
                description=(
                    "Brightness above which pixels are highlights — "
                    "no dots or hatching in these bright areas (0=black, 255=white)"
                ),
            ),
            IntParam(
                name="shadow_threshold",
                label="Shadow Threshold",
                min=0,
                max=128,
                step=1,
                default=80,
                description=(
                    "Brightness below which pixels are shadows — "
                    "directional hatching fills these dark areas (0=black, 255=white)"
                ),
            ),
            # --- Edge outlines ---
            ChoiceParam(
                name="edge_method",
                label="Edge Method",
                choices=["XDoG", "FDoG", "Canny"],
                default="XDoG",
                description="Edge detection algorithm for outline extraction",
                choice_descriptions={
                    "XDoG": "Extended DoG — pencil/woodcut/charcoal styles from a single formula",
                    "FDoG": "Flow-guided DoG — longer, smoother coherent strokes",
                    "Canny": "Canny edge detector — fast, simple, good for clean line art",
                },
            ),
            FloatParam(
                name="edge_sigma",
                label="Edge Scale (σ)",
                min=0.3,
                max=3.0,
                step=0.1,
                default=1.0,
                description=(
                    "Edge detection scale — smaller values detect fine details, "
                    "larger values detect broader features"
                ),
            ),
            IntParam(
                name="edge_min_len",
                label="Min Edge Length (pts)",
                min=2,
                max=200,
                step=1,
                default=10,
                description="Minimum contour length in points — removes tiny specks and noise",
            ),
            FloatParam(
                name="edge_simplify_mm",
                label="Edge Simplify (mm)",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.3,
                description="RDP simplification tolerance for edge polylines in mm",
            ),
            # --- Midtone stipple ---
            IntParam(
                name="stipple_points",
                label="Stipple Points",
                min=500,
                max=30000,
                step=100,
                default=5000,
                description=(
                    "Number of stipple dots in the midtone zone — "
                    "more = finer tonal detail but slower to generate"
                ),
            ),
            IntParam(
                name="stipple_iterations",
                label="Stipple Iterations",
                min=5,
                max=50,
                step=1,
                default=20,
                description=(
                    "Lloyd relaxation iterations — more = more evenly-spaced dots "
                    "weighted by image brightness"
                ),
            ),
            FloatParam(
                name="min_dot_size_mm",
                label="Min Dot Size (mm)",
                min=0.1,
                max=2.0,
                step=0.05,
                default=0.2,
                description="Dot radius in bright (highlight) areas",
            ),
            FloatParam(
                name="max_dot_size_mm",
                label="Max Dot Size (mm)",
                min=0.2,
                max=4.0,
                step=0.05,
                default=0.8,
                description="Dot radius in dark (shadow) areas",
            ),
            FloatParam(
                name="dot_size_gamma",
                label="Dot Size Gamma",
                min=0.5,
                max=3.0,
                step=0.1,
                default=1.0,
                description=(
                    "Tone curve for dot sizing — higher values emphasize dark areas, "
                    "giving larger dots more aggressively in shadows"
                ),
            ),
            ChoiceParam(
                name="dot_style",
                label="Dot Style",
                choices=["Outline", "Filled"],
                default="Outline",
                description=(
                    "Outline draws a single circle ring per dot. "
                    "Filled draws concentric rings inward so the dot appears solid "
                    "(pen spirals inward to fill the dot)."
                ),
            ),
            FloatParam(
                name="pen_width_mm",
                label="Pen Width (mm)",
                min=0.1,
                max=1.0,
                step=0.05,
                default=0.3,
                description=(
                    "Pen tip width — controls spacing between concentric fill rings. "
                    "Only relevant when Dot Style is Filled."
                ),
            ),
            # --- Shadow hatching ---
            FloatParam(
                name="hatch_angle",
                label="Hatch Angle (deg)",
                min=0.0,
                max=180.0,
                step=1.0,
                default=45.0,
                description="Direction of shadow hatch lines in degrees",
            ),
            FloatParam(
                name="hatch_spacing_mm",
                label="Hatch Spacing (mm)",
                min=0.2,
                max=3.0,
                step=0.1,
                default=0.5,
                description="Minimum spacing between shadow hatch lines in mm",
            ),
            BoolParam(
                name="cross_hatch_shadows",
                label="Cross-Hatch Deep Shadows",
                default=False,
                description=(
                    "Add a second hatch pass perpendicular to the first in the "
                    "deepest shadow areas (below half the shadow threshold) — "
                    "produces richer, darker fills for maximum contrast"
                ),
            ),
            # --- Preprocessing ---
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
                default=1.0,
                description="Gaussian blur applied before processing — smooths tonal transitions",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image before processing",
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
        # Shared defaults applied to all presets
        _shared: dict[str, Any] = {
            "highlight_threshold": 200,
            "shadow_threshold": 80,
            "edge_method": "XDoG",
            "edge_sigma": 1.0,
            "edge_min_len": 10,
            "edge_simplify_mm": 0.3,
            "stipple_points": 5000,
            "stipple_iterations": 20,
            "min_dot_size_mm": 0.2,
            "max_dot_size_mm": 0.8,
            "dot_size_gamma": 1.0,
            "hatch_angle": 45.0,
            "hatch_spacing_mm": 0.5,
            "cross_hatch_shadows": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 1.0,
            "invert": False,
            "dot_style": "Outline",
            "pen_width_mm": 0.3,
        }
        return [
            Preset(
                name="Hedcut / Classic WSJ",
                params={
                    **_shared,
                    # Traditional newspaper hedcut: XDoG outlines, moderate stipple,
                    # 45-degree shadow hatching — the signature portrait illustration style.
                    "edge_method": "XDoG",
                    "stipple_points": 5000,
                    "stipple_iterations": 20,
                    "hatch_angle": 45.0,
                    "hatch_spacing_mm": 0.5,
                    "contrast": 10.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Hedcut / Dense Detail",
                params={
                    **_shared,
                    # High dot count with cross-hatching: for complex portraits with
                    # fine hair and shadow detail — FDoG produces longer coherent strokes.
                    "edge_method": "FDoG",
                    "stipple_points": 15000,
                    "stipple_iterations": 25,
                    "min_dot_size_mm": 0.15,
                    "max_dot_size_mm": 0.5,
                    "hatch_angle": 45.0,
                    "hatch_spacing_mm": 0.4,
                    "cross_hatch_shadows": True,
                    "contrast": 15.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Hedcut / Minimal",
                params={
                    **_shared,
                    # Clean modern hedcut with fewer dots and no cross-hatching —
                    # simpler, faster to plot.
                    "edge_method": "XDoG",
                    "stipple_points": 2000,
                    "stipple_iterations": 15,
                    "min_dot_size_mm": 0.2,
                    "max_dot_size_mm": 0.7,
                    "hatch_angle": 45.0,
                    "hatch_spacing_mm": 0.7,
                    "cross_hatch_shadows": False,
                    "contrast": 5.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Hedcut / Bold Hedcut",
                params={
                    **_shared,
                    # Dramatic filled-dot style with a wide size range and deep shadows.
                    # Large filled dots in shadow areas contrast sharply with tiny dots
                    # in highlights, producing bold, high-impact tonal variation.
                    "dot_style": "Filled",
                    "min_dot_size_mm": 0.2,
                    "max_dot_size_mm": 1.8,
                    "dot_size_gamma": 1.8,
                    "pen_width_mm": 0.3,
                    "edge_method": "XDoG",
                    "edge_sigma": 1.2,
                    "stipple_points": 5000,
                    "stipple_iterations": 20,
                    "shadow_threshold": 80,
                    "highlight_threshold": 200,
                    "hatch_angle": 45.0,
                    "hatch_spacing_mm": 0.5,
                    "cross_hatch_shadows": True,
                    "contrast": 20.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Hedcut / Fine Stipple",
                params={
                    **_shared,
                    # Subtle, detailed stipple with outline dots in a narrow size range.
                    # High dot count with fine edge detection captures fine detail
                    # without the bold character of filled-dot styles.
                    "dot_style": "Outline",
                    "min_dot_size_mm": 0.1,
                    "max_dot_size_mm": 0.4,
                    "dot_size_gamma": 1.0,
                    "edge_method": "XDoG",
                    "edge_sigma": 0.7,
                    "edge_simplify_mm": 0.2,
                    "edge_min_len": 8,
                    "stipple_points": 15000,
                    "stipple_iterations": 25,
                    "shadow_threshold": 70,
                    "highlight_threshold": 210,
                    "hatch_angle": 45.0,
                    "hatch_spacing_mm": 0.4,
                    "cross_hatch_shadows": False,
                    "contrast": 10.0,
                    "blur_radius": 0.5,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Hedcut / WSJ Portrait",
                params={
                    **_shared,
                    # Tuned to mimic the classic Wall Street Journal hedcut portrait:
                    # filled dots with moderate size variation, gamma 1.5 to push
                    # shadow dots larger, FDoG for smooth coherent facial contours,
                    # and moderate contrast to preserve skin tone gradation.
                    "dot_style": "Filled",
                    "min_dot_size_mm": 0.2,
                    "max_dot_size_mm": 0.9,
                    "dot_size_gamma": 1.5,
                    "pen_width_mm": 0.3,
                    "edge_method": "FDoG",
                    "edge_sigma": 1.0,
                    "edge_simplify_mm": 0.3,
                    "edge_min_len": 10,
                    "stipple_points": 8000,
                    "stipple_iterations": 25,
                    "shadow_threshold": 80,
                    "highlight_threshold": 200,
                    "hatch_angle": 45.0,
                    "hatch_spacing_mm": 0.5,
                    "cross_hatch_shadows": False,
                    "contrast": 15.0,
                    "blur_radius": 1.0,
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
        """Generate hedcut portrait polylines from ``params["_source_image"]``."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for HedcutGenerator. "
                "Install with: pip install opencv-python"
            ) from exc

        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        # ---------------------------------------------------------------
        # Step 1: Preprocessing
        # ---------------------------------------------------------------
        img = source.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast_val = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            img = adjust_brightness(img, brightness)
        if contrast_val != 0.0:
            img = adjust_contrast(img, contrast_val)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        if do_invert:
            img = invert_image(img)

        # ---------------------------------------------------------------
        # Step 2: Convert to grayscale
        # ---------------------------------------------------------------
        if img.ndim == 3:
            gray_uint8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_uint8 = img.copy().astype(np.uint8)

        gray_f = gray_uint8.astype(np.float32) / 255.0

        img_h, img_w = gray_uint8.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        img_rect_w = img_x2 - img_x1
        img_rect_h = img_y2 - img_y1

        if progress_callback:
            progress_callback(5)
        if cancelled_callback and cancelled_callback():
            return []

        # ---------------------------------------------------------------
        # Step 3: Tonal zone parameters
        # ---------------------------------------------------------------
        shadow_threshold = int(params.get("shadow_threshold", 80))
        highlight_threshold = int(params.get("highlight_threshold", 200))
        # Clamp to valid ordered range
        shadow_threshold = max(0, min(127, shadow_threshold))
        highlight_threshold = max(shadow_threshold + 1, min(255, highlight_threshold))

        # ---------------------------------------------------------------
        # Step 4: Edge outlines
        # ---------------------------------------------------------------
        edge_method = str(params.get("edge_method", "XDoG"))
        edge_sigma = float(params.get("edge_sigma", 1.0))
        edge_min_len = int(params.get("edge_min_len", 10))
        edge_simplify_mm = float(params.get("edge_simplify_mm", 0.3))

        edge_binary = _get_edge_binary(gray_f, gray_uint8, edge_method, edge_sigma)
        edge_polylines = _trace_edge_binary(
            edge_binary, img_w, img_h,
            img_x1, img_y1, img_x2, img_y2,
            min_len=edge_min_len,
            simplify_tol_mm=edge_simplify_mm,
            smooth_iterations=1,
        )

        if progress_callback:
            progress_callback(25)
        if cancelled_callback and cancelled_callback():
            return edge_polylines

        # ---------------------------------------------------------------
        # Step 5: Midtone stippling
        # ---------------------------------------------------------------
        stipple_points = int(params.get("stipple_points", 5000))
        stipple_iterations = int(params.get("stipple_iterations", 20))
        dot_style = str(params.get("dot_style", "Outline"))
        pen_width_mm = float(params.get("pen_width_mm", 0.3))

        # Backwards compatibility: old projects may have a single dot_size_mm saved
        if "dot_size_mm" in params:
            _legacy = float(params["dot_size_mm"])
            min_dot_size_mm = _legacy
            max_dot_size_mm = _legacy
        else:
            min_dot_size_mm = float(params.get("min_dot_size_mm", 0.2))
            max_dot_size_mm = float(params.get("max_dot_size_mm", 0.8))
        dot_size_gamma = float(params.get("dot_size_gamma", 1.0))
        # Ensure min <= max
        min_dot_size_mm = min(min_dot_size_mm, max_dot_size_mm)

        # Masked image for stippling: set highlights and shadows to 255 (zero weight
        # in Lloyd's relaxation) so dots are only placed in the midtone zone.
        midtone_img = gray_uint8.copy()
        midtone_img[gray_uint8 > highlight_threshold] = 255
        midtone_img[gray_uint8 < shadow_threshold] = 255

        stipple_polylines: list[Polyline] = []
        has_midtones = bool(np.any(midtone_img < 255))

        if has_midtones and stipple_points > 0:
            from plottter.generators.stipple import (
                _lloyd_simple,
                _weighted_sample_initial_points,
            )

            rng = _random.Random(42)
            px_per_mm = img_w / img_rect_w if img_rect_w > 0 else 1.0
            # Base minimum dot spacing on the largest possible dot size so dots don't overlap
            min_spacing_px = max_dot_size_mm * 2.0 * px_per_mm

            initial_pts = _weighted_sample_initial_points(midtone_img, stipple_points, rng)
            final_pts = _lloyd_simple(
                initial_pts,
                midtone_img,
                stipple_iterations,
                min_dot_spacing_px=min_spacing_px,
                cancelled_callback=cancelled_callback,
                progress_callback=None,  # managed at top level to avoid backward jumps
                working_resolution=400,
                convergence_threshold=0.5,
            )

            # Convert pixel coords → mm and render each point as a tone-aware dot.
            # Darker pixels (low brightness) → larger dots; brighter → smaller dots.
            img_w_clamp = img_w - 1
            img_h_clamp = img_h - 1
            for px_coord, py_coord in final_pts:
                cx = img_x1 + float(px_coord) / img_w * img_rect_w
                cy = img_y1 + float(py_coord) / img_h * img_rect_h
                # Sample brightness from the (preprocessed) grayscale image
                sx = max(0, min(img_w_clamp, int(px_coord)))
                sy = max(0, min(img_h_clamp, int(py_coord)))
                pixel_lum = float(gray_uint8[sy, sx])  # 0=black, 255=white
                # Apply gamma to the normalised luminance value
                brightness_mapped = (pixel_lum / 255.0) ** dot_size_gamma
                # Large dots in dark areas, small dots in bright areas
                radius = max_dot_size_mm - brightness_mapped * (max_dot_size_mm - min_dot_size_mm)
                if dot_style == "Filled":
                    stipple_polylines.extend(_filled_circle(cx, cy, radius, pen_width_mm))
                else:
                    stipple_polylines.append(_tiny_circle(cx, cy, radius))

        if progress_callback:
            progress_callback(75)
        if cancelled_callback and cancelled_callback():
            return edge_polylines + stipple_polylines

        # ---------------------------------------------------------------
        # Step 6: Shadow hatching
        # ---------------------------------------------------------------
        hatch_angle = float(params.get("hatch_angle", 45.0))
        hatch_spacing_mm = float(params.get("hatch_spacing_mm", 0.5))
        cross_hatch = bool(params.get("cross_hatch_shadows", False))

        # Masked image for hatching: set non-shadow pixels to white so the
        # parallel-hatch function naturally avoids highlight and midtone areas.
        shadow_img = gray_uint8.copy()
        shadow_img[gray_uint8 >= shadow_threshold] = 255

        hatch_polylines: list[Polyline] = []
        has_shadows = bool(np.any(shadow_img < 255))

        if has_shadows:
            from plottter.generators.hatching import _generate_parallel_hatch

            # Primary hatch pass on all shadow areas
            hatch_polylines = _generate_parallel_hatch(
                shadow_img,
                angle_deg=hatch_angle,
                min_spacing_mm=hatch_spacing_mm,
                max_spacing_mm=hatch_spacing_mm * 4.0,
                density_curve="linear",
                canvas=canvas,
                cancelled_callback=cancelled_callback,
                progress_callback=progress_callback,
                progress_start=75,
                progress_end=90,
                img_rect=(img_x1, img_y1, img_x2, img_y2),
            )

            # Optional cross-hatch pass on the deepest shadows only
            if cross_hatch and not (cancelled_callback and cancelled_callback()):
                deep_threshold = max(0, shadow_threshold // 2)
                deep_shadow_img = gray_uint8.copy()
                deep_shadow_img[gray_uint8 >= deep_threshold] = 255

                if np.any(deep_shadow_img < 255):
                    hatch_polylines += _generate_parallel_hatch(
                        deep_shadow_img,
                        angle_deg=(hatch_angle + 45.0) % 180.0,
                        min_spacing_mm=hatch_spacing_mm,
                        max_spacing_mm=hatch_spacing_mm * 3.0,
                        density_curve="linear",
                        canvas=canvas,
                        cancelled_callback=cancelled_callback,
                        progress_callback=progress_callback,
                        progress_start=90,
                        progress_end=100,
                        img_rect=(img_x1, img_y1, img_x2, img_y2),
                    )

        if progress_callback:
            progress_callback(100)

        # ---------------------------------------------------------------
        # Combine: edges + stipple dots + shadow hatching
        # ---------------------------------------------------------------
        result = edge_polylines + stipple_polylines + hatch_polylines
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
        return result
