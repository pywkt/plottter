"""FDoGGenerator — Coherent Line Drawing via Flow-guided Difference-of-Gaussians.

Produces long, smooth, hand-drawn-looking edge lines from photographs by:

1. Computing an Edge Tangent Flow (ETF) — a smooth vector field capturing
   edge directions throughout the image.  The ETF is initialised from the
   Sobel gradient and iteratively refined with a magnitude-weighted
   Gaussian to encourage coherent tangent alignment along edges.

2. Smoothing the image *along* the ETF flow lines (reducing noise without
   blurring across edges) to improve edge continuity.

3. Applying an XDoG-style Difference-of-Gaussians filter to the flow-
   smoothed image, then soft-thresholding the result to produce a binary
   edge map that is traced into plotter polylines.

Compared with Canny (jagged pixel-boundary fragments) or XDoG (isotropic
Gaussians), FDoG produces significantly fewer fragmented short lines and
longer, more coherent strokes that follow image contours naturally.

Reference: Kang et al., "Coherent Line Drawing", NPAR 2007.
           https://cg.postech.ac.kr/papers/kang_npar07_hi.pdf
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


def _compute_etf(
    gray_f: np.ndarray,
    sigma_m: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Edge Tangent Flow (ETF) field.

    The ETF is a dense vector field where each pixel holds the unit tangent
    direction of the nearest edge.  It is initialised from the normalised
    Sobel gradient rotated 90° (tangent = perpendicular to gradient) and
    then iteratively smoothed with a magnitude-weighted Gaussian kernel.
    During each pass, tangent vectors at pixels with stronger gradients
    exert more influence, pulling nearby vectors into alignment and
    producing a smooth, coherent flow along edges.

    Parameters
    ----------
    gray_f:     Float32 grayscale image, values in [0, 1].
    sigma_m:    Spatial scale for ETF smoothing (pixels).  Controls how far
                edge tangents influence their neighbourhood.
    iterations: Number of smoothing passes (3–5 typical).

    Returns
    -------
    (tx, ty): Pair of float32 arrays of shape (H, W) holding the x and y
              components of the unit tangent at each pixel.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for FDoG/Coherent Line generation."
        ) from exc

    # Compute image gradient with a 5×5 Sobel kernel for stability
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=5)

    mag = np.sqrt(gx * gx + gy * gy)
    mag_max = float(mag.max())
    mag_norm = mag / (mag_max + 1e-8)

    # Initial tangent: rotate gradient 90° — edges run perpendicular to ∇f
    # gradient (gx, gy) → tangent (-gy, gx)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(mag > 1e-8, -gy / (mag + 1e-8), 0.0).astype(np.float32)
        ty = np.where(mag > 1e-8,  gx / (mag + 1e-8), 0.0).astype(np.float32)

    # Clamp sigma_m to avoid degenerate kernel sizes
    sigma_m = max(float(sigma_m), 0.5)
    r = max(2, int(round(2.5 * sigma_m)))
    ksize = 2 * r + 1

    for _ in range(max(1, iterations)):
        # Magnitude-weighted smoothing: pixels with larger gradients pull
        # nearby tangents into alignment.
        smooth_tx = cv2.GaussianBlur(tx * mag_norm, (ksize, ksize), sigma_m)
        smooth_ty = cv2.GaussianBlur(ty * mag_norm, (ksize, ksize), sigma_m)

        # Re-normalise to keep unit vectors
        new_mag = np.sqrt(smooth_tx * smooth_tx + smooth_ty * smooth_ty)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(
                new_mag > 1e-8, smooth_tx / (new_mag + 1e-8), 0.0
            ).astype(np.float32)
            ty = np.where(
                new_mag > 1e-8, smooth_ty / (new_mag + 1e-8), 0.0
            ).astype(np.float32)

        # Update the magnitude estimate used for the next pass
        smooth_mag = cv2.GaussianBlur(mag_norm, (ksize, ksize), sigma_m)
        smooth_max = float(smooth_mag.max())
        mag_norm = smooth_mag / (smooth_max + 1e-8)

    return tx, ty


def _flow_smooth(
    gray_f: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    sigma_t: float,
) -> np.ndarray:
    """Smooth image along ETF flow lines with a 1-D Gaussian kernel.

    Each pixel p is replaced by the Gaussian-weighted average of image
    values sampled along the ETF tangent flow line passing through p.
    This reduces noise along edges while preserving the sharpness of
    transitions *across* edges — unlike isotropic Gaussian blurring which
    smears both.

    Bilinear interpolation is used to sample sub-pixel positions on each
    flow line.

    Parameters
    ----------
    gray_f: Float32 image, values in [0, 1].  Shape (H, W).
    tx, ty: Unit tangent vectors from _compute_etf.  Shape (H, W).
    sigma_t: Standard deviation of the along-flow Gaussian (pixels).

    Returns
    -------
    Float32 smoothed image, values in [0, 1].  Shape (H, W).
    """
    h, w = gray_f.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    n_steps = max(int(np.ceil(2.5 * sigma_t)), 2)
    dists = np.arange(-n_steps, n_steps + 1, dtype=np.float32)
    weights = np.exp(-0.5 * (dists / sigma_t) ** 2)
    total_w = float(weights.sum())

    result = np.zeros((h, w), dtype=np.float32)

    for d, w_d in zip(dists, weights):
        # Sample positions along the tangent direction
        sx = np.clip(xs + d * tx, 0.0, float(w - 1))
        sy = np.clip(ys + d * ty, 0.0, float(h - 1))

        # Bilinear interpolation
        x0 = sx.astype(np.int32)
        y0 = sy.astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = sx - x0
        fy = sy - y0

        val = (
            gray_f[y0, x0] * (1.0 - fx) * (1.0 - fy)
            + gray_f[y0, x1] * fx * (1.0 - fy)
            + gray_f[y1, x0] * (1.0 - fx) * fy
            + gray_f[y1, x1] * fx * fy
        )
        result += w_d * val

    return result / total_w


def _fdog(
    gray_f: np.ndarray,
    sigma_c: float,
    rho: float,
    sigma_m: float,
    etf_iterations: int,
    fdog_iterations: int,
    phi: float = 200.0,
    epsilon: float = -0.1,
) -> np.ndarray:
    """Apply the full FDoG pipeline to a float32 grayscale image.

    Algorithm:
    1. Compute the Edge Tangent Flow with sigma_m and etf_iterations.
    2. Repeat fdog_iterations times:
       a. Smooth the image along ETF flow lines (Gaussian, width sigma_m).
          This extends edge continuity without blurring across edges.
       b. Apply an XDoG-style soft threshold:
             D  = G_σc * H_smooth − G_(ρ·σc) * H_smooth
             T  = 1                           if D ≥ ε
                  1 + tanh(φ · (D − ε))  otherwise
          Edges appear where T < 0.5 (boundary of bright "non-edge" regions).
    3. Return the final T image.  Values ≥ 0.5 are bright (background /
       non-edges); boundaries of bright regions trace the detected edges.

    Parameters
    ----------
    gray_f:          Float32 image in [0, 1].
    sigma_c:         DoG small-sigma (controls edge width in pixels).
    rho:             DoG ratio — large sigma = rho × sigma_c.
    sigma_m:         ETF smoothing scale and flow-smoothing width (pixels).
    etf_iterations:  Number of ETF refinement passes (3–5 recommended).
    fdog_iterations: Number of FDoG filter + threshold passes (1–3 typical).
    phi:             XDoG soft-threshold sharpness; higher → crisper edges.
    epsilon:         Black-level offset; negative allows softer dark regions.

    Returns
    -------
    Float32 image with values in [0, 1].
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for FDoG/Coherent Line generation."
        ) from exc

    tx, ty = _compute_etf(gray_f, sigma_m, etf_iterations)

    H = gray_f.copy()
    for _ in range(max(1, fdog_iterations)):
        # Flow-smooth along ETF to extend edge continuity
        H_smooth = _flow_smooth(H, tx, ty, sigma_m)

        # XDoG-style DoG + soft threshold applied to the flow-smoothed image
        G1 = cv2.GaussianBlur(H_smooth, (0, 0), sigmaX=sigma_c)
        G2 = cv2.GaussianBlur(H_smooth, (0, 0), sigmaX=rho * sigma_c)
        D = G1 - G2

        T = np.where(
            D >= epsilon,
            1.0,
            1.0 + np.tanh(phi * (D - epsilon)),
        )
        H = T.astype(np.float32)

    return H


@register_generator
class FDoGGenerator(Generator):
    """Coherent Line Drawing via Flow-guided Difference-of-Gaussians (FDoG).

    Produces long, smooth, hand-drawn-looking edge lines from photographs.
    Unlike Canny (which gives jagged pixel-boundary fragments) or XDoG
    (which uses isotropic Gaussians), FDoG:

    1. Computes an Edge Tangent Flow (ETF) — a smooth vector field that
       captures the direction of edges throughout the image.
    2. Uses that flow to smooth the image *along* edges without blurring
       *across* them, extending edge continuity and reducing gaps.
    3. Applies a Difference-of-Gaussians filter to detect edges in the
       flow-smoothed result, thresholded with a smooth ramp.

    The output typically has fewer fragmented short lines and longer,
    more connected strokes than Canny or XDoG.

    Key parameters
    --------------
    sigma_c:         Controls line width (pixels) — larger → thicker edges.
    sigma_m:         Flow scale — how far the ETF smoothing reaches; also
                     the Gaussian width used for along-flow smoothing.
    rho:             DoG ratio — ratio of large/small Gaussian sigma.
    etf_iterations:  Number of ETF refinement passes.
    fdog_iterations: Number of FDoG filter passes applied iteratively.

    Reference: Kang et al., "Coherent Line Drawing", NPAR 2007.
    """

    name = "Coherent Lines (FDoG)"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="sigma_c",
                label="Line Width (σ_c)",
                min=0.5,
                max=3.0,
                step=0.1,
                default=1.0,
                description="Line width scale — controls how wide the detected edges appear before binarization",
            ),
            FloatParam(
                name="sigma_m",
                label="Flow Scale (σ_m)",
                min=0.5,
                max=6.0,
                step=0.1,
                default=3.0,
                description="Flow smoothing scale — controls how far the ETF direction field is smoothed. Larger values produce more coherent, longer strokes",
            ),
            FloatParam(
                name="rho",
                label="DoG Ratio (ρ)",
                min=1.1,
                max=5.0,
                step=0.1,
                default=3.0,
                description="DoG ratio along the flow direction — higher values produce a wider frequency gap and more selective edge detection",
            ),
            IntParam(
                name="etf_iterations",
                label="ETF Iterations",
                min=1,
                max=10,
                step=1,
                default=3,
                description="Number of Edge Tangent Flow smoothing passes — more passes produce smoother, more coherent direction fields",
            ),
            IntParam(
                name="fdog_iterations",
                label="FDoG Iterations",
                min=1,
                max=5,
                step=1,
                default=1,
                description="Number of FDoG filtering passes — additional passes sharpen and clean up the edge response",
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
                default=1,
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
                description="Use local adaptive Gaussian thresholding on the FDoG T-image instead of the fixed T ≥ 0.5 cutoff — handles uneven lighting on scanned or photographed line art",
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
                description="Gaussian blur applied before FDoG processing — pre-smoothing reduces noise",
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
            "smooth_iterations": 1,
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
                name="Coherent Lines",
                params={
                    **_shared,
                    "sigma_c": 1.0,
                    "sigma_m": 3.0,
                    "rho": 3.0,
                    "etf_iterations": 3,
                    "fdog_iterations": 1,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Fine Lines",
                params={
                    **_shared,
                    "sigma_c": 0.5,
                    "sigma_m": 2.0,
                    "rho": 2.5,
                    "etf_iterations": 3,
                    "fdog_iterations": 2,
                    "smooth_iterations": 0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Bold Strokes",
                params={
                    **_shared,
                    "sigma_c": 1.5,
                    "sigma_m": 4.0,
                    "rho": 3.5,
                    "etf_iterations": 5,
                    "fdog_iterations": 2,
                    "brightness": 10.0,
                    "contrast": 20.0,
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
        """Generate coherent edge polylines from ``params["_source_image"]``.

        Returns an empty list when no source image is provided.
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "The 'opencv-python' package is required for Coherent Lines (FDoG) "
                "generation.  Install it with: pip install opencv-python"
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

        sigma_c = float(params.get("sigma_c", 1.0))
        sigma_m = float(params.get("sigma_m", 3.0))
        rho = float(params.get("rho", 3.0))
        etf_iters = int(params.get("etf_iterations", 3))
        fdog_iters = int(params.get("fdog_iterations", 1))
        min_len = int(params.get("min_contour_length", 10))
        simplify_tol_mm = float(params.get("simplify_tolerance_mm", 0.5))
        close_gaps_mm = float(params.get("close_gaps_mm", 2.0))
        smooth_iterations = int(params.get("smooth_iterations", 1))

        if progress_callback:
            progress_callback(5)
        if cancelled_callback and cancelled_callback():
            return []

        # --- FDoG pipeline ---
        T = _fdog(gray_f, sigma_c, rho, sigma_m, etf_iters, fdog_iters)

        if progress_callback:
            progress_callback(40)
        if cancelled_callback and cancelled_callback():
            return []

        # Binarize: bright regions (T >= 0.5) are non-edges; their
        # boundaries are the detected edge lines.
        # When adaptive_threshold is enabled, use local Gaussian adaptive thresholding
        # on the uint8 T-image instead of the fixed 0.5 cutoff — handles images where
        # the FDoG response has a global brightness bias (e.g. scanned art).
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

        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        if progress_callback:
            progress_callback(55)

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

        mm_per_px = (img_x2 - img_x1) / img_w if img_w > 0 else 1.0
        simplify_tol_px = simplify_tol_mm / mm_per_px if mm_per_px > 0 else 1.0

        polylines: list[Polyline] = []
        total = len(contours)

        for idx, contour in enumerate(contours):
            if cancelled_callback and cancelled_callback():
                break

            if len(contour) < min_len:
                continue

            simplified = cv2.approxPolyDP(contour, simplify_tol_px, closed=False)
            if len(simplified) < 2:
                continue

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
                if smooth_iterations > 0:
                    from plottter.generators.contour import _chaikin_smooth
                    poly = _chaikin_smooth(poly, smooth_iterations, closed=False)
                polylines.append(poly)

            if progress_callback and idx % 50 == 0:
                progress_callback(55 + int(idx / total * 35))

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
