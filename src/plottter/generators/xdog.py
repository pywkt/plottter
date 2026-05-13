"""XDoGGenerator — Extended Difference-of-Gaussians edge detection.

Produces artistic line styles from photographs by applying a tunable
soft-threshold filter to a Difference-of-Gaussians image.  Different
parameter combinations yield radically different visual styles — from
clean pencil sketches to bold woodcut art to soft charcoal.

Reference: Winnemöller et al., "XDoG: An eXtended difference-of-Gaussians
compendium", Computers & Graphics, 2012.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import _px_to_mm, compute_image_rect
from plottter.generators.base import (
    BoolParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _xdog(
    gray_f: np.ndarray,
    sigma: float,
    k: float,
    phi: float,
    epsilon: float,
) -> np.ndarray:
    """Apply XDoG filter to a float32 grayscale image in [0, 1].

    Formula (Winnemöller et al. 2012):
        D(x) = G(σ) * I(x) − G(k·σ) * I(x)
        T(x) = 1                          if D(x) ≥ ε
               1 + tanh(φ · (D(x) − ε))  otherwise

    Parameters
    ----------
    gray_f:  Float32 grayscale image, values in [0, 1].
    sigma:   Standard deviation of the first (smaller) Gaussian kernel.
    k:       Scale ratio — the second kernel uses sigma * k.
    phi:     Sharpness of the soft threshold; higher = crisper edges.
    epsilon: Black-level offset; positive = fewer, harder edges.

    Returns
    -------
    Float32 image with T values in [0, 1].  Values near 1.0 correspond to
    "bright" areas that include edge transitions; values near 0.0 correspond
    to dark regions.  The boundary between bright and dark regions marks
    the detected edges.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The 'opencv-python' package is required for XDoG generation. "
            "Install it with: pip install opencv-python"
        ) from exc

    # Two Gaussian blurs: ksize=(0,0) lets OpenCV derive kernel size from sigma
    G1 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma)
    G2 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=k * sigma)

    D = G1 - G2

    # Soft threshold: regions where D >= epsilon stay at 1; others are pulled
    # toward 0 with strength controlled by phi.
    T = np.where(D >= epsilon, 1.0, 1.0 + np.tanh(phi * (D - epsilon)))

    return T.astype(np.float32)


@register_generator
class XDoGGenerator(Generator):
    """XDoG (Extended Difference-of-Gaussians) edge detection.

    Produces artistic line styles from photographs — pencil sketches,
    woodcut art, charcoal drawings — all from the same algorithm by
    changing four sliders.

    How it works:
    1. Two Gaussian blurs are applied at scales σ and k·σ.
    2. Their difference D highlights edges.
    3. A soft tanh threshold converts D to a binary-like image whose
       boundaries are the detected edge lines.
    4. OpenCV contour tracing turns those boundaries into polylines.
    """

    name = "XDoG"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="sigma",
                label="Edge Scale (σ)",
                min=0.3,
                max=3.0,
                step=0.1,
                default=1.0,
                description="Edge detection scale — smaller values detect fine details, larger values detect broader features",
            ),
            FloatParam(
                name="k",
                label="DoG Ratio (k)",
                min=1.1,
                max=5.0,
                step=0.1,
                default=1.6,
                description="DoG ratio — controls the width difference between the two Gaussians. Higher values increase frequency separation",
            ),
            FloatParam(
                name="phi",
                label="Sharpness (φ)",
                min=1.0,
                max=200.0,
                step=1.0,
                default=100.0,
                description="Sharpness of the black/white transition — higher values produce crisper, harder edges; lower values produce softer, more gradual transitions",
            ),
            FloatParam(
                name="epsilon",
                label="Black Level (ε)",
                min=-0.5,
                max=0.5,
                step=0.01,
                default=0.01,
                description="Edge threshold — adjusts where edges appear. Negative values reveal more edge detail; positive values suppress subtle edges",
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
                description="RDP simplification tolerance in mm — reduces point count while preserving shape",
            ),
            FloatParam(
                name="close_gaps_mm",
                label="Close Gaps (mm)",
                min=0.0,
                max=20.0,
                step=0.1,
                default=2.0,
                description="Maximum gap between endpoints to bridge — connects nearby open contours",
            ),
            IntParam(
                name="smooth_iterations",
                label="Smooth Iterations (Chaikin)",
                min=0,
                max=5,
                step=1,
                default=0,
                description="Number of Chaikin smoothing passes for more flowing, organic-looking strokes",
            ),
            BoolParam(
                name="centerline",
                label="Center-line Trace",
                default=False,
                description="Thin edge bands to single-pixel centerlines before tracing — reduces hollow outline artifacts on thick lines. Best for line art and sketch inputs",
            ),
            FloatParam(
                name="merge_gap_mm",
                label="Fragment Merge Gap (mm)",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.5,
                visible_when={"centerline": [True]},
                description="Maximum endpoint distance (mm) for merging nearby skeleton fragments — reduces pen lifts by connecting polylines whose endpoints are close. Set to 0 to disable",
            ),
            BoolParam(
                name="adaptive_threshold",
                label="Adaptive Threshold",
                default=False,
                visible_when={"centerline": [True]},
                description="Use local adaptive Gaussian thresholding on the XDoG T-image instead of the fixed T ≥ 0.5 cutoff — handles uneven lighting on scanned or photographed line art",
            ),
            FloatParam(
                name="adaptive_c",
                label="Adaptive C Constant",
                min=-20.0,
                max=20.0,
                step=0.5,
                default=5.0,
                visible_when={"centerline": [True], "adaptive_threshold": [True]},
                description="Constant subtracted from the local Gaussian-weighted mean — higher values are stricter (fewer edges detected), lower/negative values are more permissive",
            ),
            BoolParam(
                name="smooth_curves",
                label="Smooth Curves (Bezier Fitting)",
                default=False,
                description="Fit cubic Bezier curves to output polylines for smooth, organic strokes",
            ),
            FloatParam(
                name="curve_tolerance_mm",
                label="Curve Tolerance (mm)",
                min=0.1,
                max=5.0,
                step=0.05,
                default=0.5,
                visible_when={"smooth_curves": [True]},
                description="How closely fitted curves must follow the original points — lower = more faithful but more points",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image before processing",
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
                description="Gaussian blur applied before XDoG processing — pre-smoothing reduces noise",
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
        _shared = {
            "min_contour_length": 10,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 2.0,
            "smooth_iterations": 0,
            "centerline": False,
            "merge_gap_mm": 0.0,
            "adaptive_threshold": False,
            "adaptive_c": 5.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        return [
            Preset(
                name="Pencil Sketch",
                params={
                    **_shared,
                    "sigma": 0.8,
                    "k": 1.6,
                    "phi": 80.0,
                    "epsilon": -0.02,
                    "blur_radius": 1.0,   # mild pre-smooth reduces noise in photos
                    "centerline": True,   # thin edges to single-pixel centerlines
                    "merge_gap_mm": 0.5,  # merge skeleton fragments to reduce pen lifts
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Woodcut",
                params={
                    **_shared,
                    "sigma": 1.5,
                    "k": 3.0,
                    "phi": 100.0,
                    "epsilon": -0.04,   # was -0.1 which excluded all edges in real images
                    "blur_radius": 1.0,  # mild pre-smooth
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Soft Charcoal",
                params={
                    **_shared,
                    "sigma": 2.0,
                    "k": 2.5,
                    "phi": 15.0,
                    "epsilon": -0.04,
                    "blur_radius": 2.0,    # stronger pre-smooth for organic feel
                    "smooth_iterations": 1, # gentle Chaikin smoothing
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
        """Generate XDoG edge polylines from ``params["_source_image"]``."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "The 'opencv-python' package is required for XDoG generation. "
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
        )

        # --- Preprocessing ---
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

        # Convert to float32 grayscale [0, 1]
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_f = img.astype(np.float32) / 255.0

        sigma = float(params.get("sigma", 1.0))
        k = float(params.get("k", 1.6))
        phi = float(params.get("phi", 100.0))
        epsilon = float(params.get("epsilon", 0.01))
        min_len = int(params.get("min_contour_length", 10))
        simplify_tol_mm = float(params.get("simplify_tolerance_mm", 0.5))
        close_gaps_mm = float(params.get("close_gaps_mm", 2.0))
        smooth_iterations = int(params.get("smooth_iterations", 0))

        if progress_callback:
            progress_callback(10)
        if cancelled_callback and cancelled_callback():
            return []

        # --- XDoG filter ---
        T = _xdog(gray_f, sigma, k, phi, epsilon)

        if progress_callback:
            progress_callback(30)
        if cancelled_callback and cancelled_callback():
            return []

        # Binarize: regions where T >= 0.5 are "bright"; their boundaries are edges.
        # When adaptive_threshold is enabled, use local Gaussian adaptive thresholding
        # on the uint8 T-image instead of the fixed 0.5 cutoff — this handles images
        # where the XDoG response has a global brightness bias (e.g. scanned art).
        adaptive_thresh = bool(params.get("adaptive_threshold", False))
        adaptive_c_val = float(params.get("adaptive_c", 5.0))
        if adaptive_thresh:
            from plottter.generators._helpers import _apply_threshold
            T_uint8 = (T * 255.0).clip(0, 255).astype(np.uint8)
            binary = _apply_threshold(T_uint8, 128, True, adaptive_c_val, cv2.THRESH_BINARY)
        else:
            binary = (T >= 0.5).astype(np.uint8) * 255

        # Optional centerline tracing: thin dark edge bands to single-pixel
        # centerlines before contour tracing.  This eliminates the "hollow
        # outline" artefact produced when a thick source line generates a wide
        # dark band — whose boundary traces as two parallel contours.
        if bool(params.get("centerline", False)):
            from plottter.generators._helpers import _skeletonize
            # Edge bands are dark (0) in binary — invert to make them the
            # foreground, skeletonize, then invert back.
            inverted = cv2.bitwise_not(binary)
            thinned = _skeletonize(inverted)
            binary = cv2.bitwise_not(thinned)

        # Trace boundaries of bright regions to produce edge polylines
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        if progress_callback:
            progress_callback(50)

        if not contours:
            return []

        img_h, img_w = gray_f.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        # Convert mm tolerance to pixels for RDP simplification
        mm_per_px = (img_x2 - img_x1) / img_w if img_w > 0 else 1.0
        simplify_tol_px = simplify_tol_mm / mm_per_px if mm_per_px > 0 else 1.0

        polylines: list[Polyline] = []
        total = len(contours)

        for idx, contour in enumerate(contours):
            if cancelled_callback and cancelled_callback():
                break

            if len(contour) < min_len:
                continue

            # RDP simplification
            simplified = cv2.approxPolyDP(contour, simplify_tol_px, closed=False)
            if len(simplified) < 2:
                continue

            # Convert pixel coordinates to mm
            poly: Polyline = [
                _px_to_mm(
                    float(pt[0][0]),
                    float(pt[0][1]),
                    img_w, img_h,
                    img_x1, img_y1, img_x2, img_y2,
                )
                for pt in simplified
            ]

            if len(poly) >= 2:
                # Optional Chaikin smoothing for softer curves
                if smooth_iterations > 0:
                    from plottter.generators.contour import _chaikin_smooth
                    poly = _chaikin_smooth(poly, smooth_iterations, closed=False)
                polylines.append(poly)

            if progress_callback and idx % 50 == 0:
                progress_callback(50 + int(idx / total * 40))

        if progress_callback:
            progress_callback(90)

        # Merge nearby skeleton fragments when centerline mode is active
        merge_gap_mm = float(params.get("merge_gap_mm", 0.5))
        if bool(params.get("centerline", False)) and merge_gap_mm > 0 and polylines:
            from plottter.processing.merge import merge_fragments
            polylines = merge_fragments(polylines, merge_gap_mm)

        # Connect nearby endpoints to reduce pen lifts
        if close_gaps_mm > 0 and polylines:
            from plottter.generators.edge_detect import _close_gaps
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
