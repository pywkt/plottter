"""LIC (Line Integral Convolution) generator — streamline-based flow visualisation.

Provides:
- ``_seed_grid()``            — generates a jittered regular grid of seed points
                                covering the canvas.
- ``_brightness_filter()``    — removes seeds in bright image regions using
                                bilinear brightness sampling.
- ``_trace_streamline()``     — traces a single streamline from a seed point in
                                both directions along a vector field.
- ``_filter_by_separation()`` — removes streamlines whose midpoints are too
                                close to already-accepted streamlines, using a
                                KD-tree for efficient proximity queries.
- ``LICGenerator``            — Generator ABC implementation wiring all helpers
                                together for image-driven streamline art.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from plottter.generators import register_generator
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


# ---------------------------------------------------------------------------
# Seed grid
# ---------------------------------------------------------------------------


def _seed_grid(
    canvas_w: float,
    canvas_h: float,
    seed_spacing_mm: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Generate a jittered regular grid of seed points over the canvas.

    The canvas is divided into cells of side ``seed_spacing_mm``.  One seed
    point is placed inside each cell at a uniformly random position within a
    jitter radius of ``seed_spacing_mm * 0.3`` around the cell centre.

    Parameters
    ----------
    canvas_w, canvas_h:
        Canvas dimensions in millimetres.
    seed_spacing_mm:
        Nominal distance between adjacent seed points (mm).  Controls grid
        resolution.  Smaller values produce denser streamlines.
    rng:
        NumPy random generator instance for reproducible results.

    Returns
    -------
    List of ``(x_mm, y_mm)`` tuples, one per grid cell, with positions
    clamped to ``[0, canvas_w] × [0, canvas_h]``.
    """
    seed_spacing_mm = max(1e-3, float(seed_spacing_mm))
    canvas_w = float(canvas_w)
    canvas_h = float(canvas_h)

    nx = max(1, int(math.ceil(canvas_w / seed_spacing_mm)))
    ny = max(1, int(math.ceil(canvas_h / seed_spacing_mm)))

    jitter_radius = seed_spacing_mm * 0.3

    seeds: list[tuple[float, float]] = []
    for iy in range(ny):
        for ix in range(nx):
            # Cell centre in mm
            cx = (ix + 0.5) * canvas_w / nx
            cy = (iy + 0.5) * canvas_h / ny

            # Uniform jitter within a square of side 2*jitter_radius
            dx = float(rng.uniform(-jitter_radius, jitter_radius))
            dy = float(rng.uniform(-jitter_radius, jitter_radius))

            x = cx + dx
            y = cy + dy

            # Clamp to canvas bounds
            x = max(0.0, min(canvas_w, x))
            y = max(0.0, min(canvas_h, y))

            seeds.append((x, y))

    return seeds


# ---------------------------------------------------------------------------
# Brightness filter
# ---------------------------------------------------------------------------


def _brightness_filter(
    seeds: list[tuple[float, float]],
    canvas_w: float,
    canvas_h: float,
    brightness_img: np.ndarray,
    brightness_threshold: float,
) -> list[tuple[float, float]]:
    """Remove seed points that fall in bright image regions.

    Bilinearly samples ``brightness_img`` at each seed position (mapped from
    mm coordinates to pixel coordinates) and keeps only those seeds whose
    sampled brightness is *below* ``brightness_threshold``.

    Parameters
    ----------
    seeds:
        List of ``(x_mm, y_mm)`` points produced by :func:`_seed_grid`.
    canvas_w, canvas_h:
        Canvas dimensions in millimetres (must match the coordinate space of
        the seed positions).
    brightness_img:
        2-D grayscale array of shape ``(H, W)`` with float values in
        ``[0, 1]`` (0 = black, 1 = white).
    brightness_threshold:
        Seeds with sampled brightness ≥ this value are discarded.  A value of
        ``0.0`` removes all seeds (everything is at least as bright as 0),
        and ``1.0`` removes no seeds (nothing exceeds 1 in normalised images;
        passing ``> 1.0`` guarantees no seeds are removed even with slight
        floating-point drift, but the public interface uses ``255`` for 8-bit
        images — see notes below).

    Notes
    -----
    The caller is responsible for normalising ``brightness_img`` to ``[0, 1]``
    before calling this function.  To emulate a threshold expressed on a
    ``[0, 255]`` scale simply divide: ``brightness_threshold / 255``.

    Returns
    -------
    Filtered list of ``(x_mm, y_mm)`` tuples.
    """
    if len(seeds) == 0:
        return []

    img = np.asarray(brightness_img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(
            f"brightness_img must be 2-D (H, W), got shape {img.shape}"
        )

    h, w = img.shape
    canvas_w = float(canvas_w)
    canvas_h = float(canvas_h)

    kept: list[tuple[float, float]] = []
    for x_mm, y_mm in seeds:
        brightness = _bilinear_sample(img, x_mm, y_mm, canvas_w, canvas_h, w, h)
        if brightness < brightness_threshold:
            kept.append((x_mm, y_mm))

    return kept


# ---------------------------------------------------------------------------
# Streamline tracer
# ---------------------------------------------------------------------------


def _trace_streamline(
    seed: tuple[float, float],
    vector_field: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    kernel_length_mm: float,
    step_size_mm: float,
) -> list[tuple[float, float]]:
    """Trace a single streamline from *seed* along *vector_field*.

    Integrates in both forward (+v) and backward (−v) directions using
    first-order Euler steps of length *step_size_mm*.  Each direction is
    capped at half of *kernel_length_mm* of arc length.

    Tracing stops early when any of the following conditions is met:

    * the current position exits ``[0, canvas_w] × [0, canvas_h]``
    * the bilinearly sampled vector magnitude falls below 0.01

    Parameters
    ----------
    seed:
        Starting point ``(x_mm, y_mm)`` in canvas millimetre coordinates.
    vector_field:
        NumPy array of shape ``(H, W, 2)`` containing unit direction vectors.
        ``field[row, col, 0]`` is the *x* component and
        ``field[row, col, 1]`` is the *y* component.  Matches the format
        returned by ``_compute_etf()`` in ``_helpers.py``.
    canvas_w, canvas_h:
        Canvas dimensions in millimetres.
    kernel_length_mm:
        Maximum total arc length of the streamline.  Each direction is
        traced for at most ``kernel_length_mm / 2`` mm.
    step_size_mm:
        Euler integration step size in millimetres.  Smaller values yield
        smoother curves but are more expensive to compute.

    Returns
    -------
    A single ``Polyline`` — a ``list[tuple[float, float]]`` of
    ``(x_mm, y_mm)`` points.  The seed point is always included (index 0 of
    the returned list after the backward segment is prepended).  If the seed
    is outside canvas bounds or the vector field is zero at the seed, a
    single-point polyline ``[seed]`` is returned.
    """
    step_size_mm = max(1e-4, float(step_size_mm))
    half_len = float(kernel_length_mm) / 2.0
    canvas_w = float(canvas_w)
    canvas_h = float(canvas_h)

    vf = np.asarray(vector_field, dtype=np.float32)
    if vf.ndim != 3 or vf.shape[2] != 2:
        raise ValueError(
            f"vector_field must have shape (H, W, 2), got {vf.shape}"
        )
    vf_h, vf_w = vf.shape[:2]

    def _in_bounds(x: float, y: float) -> bool:
        return 0.0 <= x <= canvas_w and 0.0 <= y <= canvas_h

    def _sample_vf(x: float, y: float) -> tuple[float, float]:
        """Bilinearly sample *vector_field* at mm position (x, y)."""
        # Map mm → fractional pixel coordinates
        px = (x / canvas_w) * vf_w - 0.5
        py = (y / canvas_h) * vf_h - 0.5

        x0 = int(math.floor(px))
        y0 = int(math.floor(py))
        x1 = x0 + 1
        y1 = y0 + 1

        x0c = max(0, min(vf_w - 1, x0))
        x1c = max(0, min(vf_w - 1, x1))
        y0c = max(0, min(vf_h - 1, y0))
        y1c = max(0, min(vf_h - 1, y1))

        fx = px - x0
        fy = py - y0

        vx = (
            float(vf[y0c, x0c, 0]) * (1 - fx) * (1 - fy)
            + float(vf[y0c, x1c, 0]) * fx * (1 - fy)
            + float(vf[y1c, x0c, 0]) * (1 - fx) * fy
            + float(vf[y1c, x1c, 0]) * fx * fy
        )
        vy = (
            float(vf[y0c, x0c, 1]) * (1 - fx) * (1 - fy)
            + float(vf[y0c, x1c, 1]) * fx * (1 - fy)
            + float(vf[y1c, x0c, 1]) * (1 - fx) * fy
            + float(vf[y1c, x1c, 1]) * fx * fy
        )
        return vx, vy

    def _trace_direction(
        start_x: float, start_y: float, sign: float
    ) -> list[tuple[float, float]]:
        """Trace one direction; *sign* is +1 (forward) or −1 (backward)."""
        pts: list[tuple[float, float]] = []
        x, y = start_x, start_y
        arc = 0.0

        while arc < half_len:
            vx, vy = _sample_vf(x, y)
            mag = math.hypot(vx, vy)
            if mag < 0.01:
                break

            # Normalise to unit step
            nx_v = vx / mag
            ny_v = vy / mag

            x += sign * nx_v * step_size_mm
            y += sign * ny_v * step_size_mm
            arc += step_size_mm

            if not _in_bounds(x, y):
                break

            pts.append((x, y))

        return pts

    sx, sy = float(seed[0]), float(seed[1])

    if not _in_bounds(sx, sy):
        return [(sx, sy)]

    forward = _trace_direction(sx, sy, +1.0)
    backward = _trace_direction(sx, sy, -1.0)

    # Combine: reversed-backward + seed + forward
    return list(reversed(backward)) + [(sx, sy)] + forward


# ---------------------------------------------------------------------------
# Separation filter
# ---------------------------------------------------------------------------


def _filter_by_separation(
    streamlines: Sequence[list[tuple[float, float]]],
    brightnesses: Sequence[float],
    separation_distance_mm: float,
) -> list[list[tuple[float, float]]]:
    """Remove streamlines that are too close to already-accepted streamlines.

    Processes streamlines in brightness-priority order (darkest first) so that
    when two streamlines compete for the same spatial region, the one in the
    darker area is preferentially kept.  Proximity is measured between
    streamline *midpoints* using a KD-tree for O(n log n) queries.

    Parameters
    ----------
    streamlines:
        Sequence of polylines, each a ``list[tuple[float, float]]`` of
        ``(x_mm, y_mm)`` points as returned by :func:`_trace_streamline`.
    brightnesses:
        Sequence of float values in ``[0, 1]`` (0 = black, 1 = white), one
        per streamline, representing the local image brightness at the seed
        point.  Used to establish processing order: darker seeds (lower
        brightness) are processed first and therefore win conflicts.
    separation_distance_mm:
        Minimum allowed distance (mm) between accepted streamline midpoints.
        Any candidate whose midpoint is within this distance of an
        already-accepted midpoint is discarded.

    Returns
    -------
    Filtered list of polylines in their original order (not in processing
    order).  Streamlines with fewer than one point are silently dropped.
    """
    if not streamlines:
        return []

    n = len(streamlines)
    if n != len(brightnesses):
        raise ValueError(
            f"streamlines and brightnesses must have equal length, got {n} vs {len(brightnesses)}"
        )

    # Compute midpoint for each streamline
    midpoints: list[tuple[float, float]] = []
    for sl in streamlines:
        if not sl:
            midpoints.append((float("nan"), float("nan")))
            continue
        mid_idx = len(sl) // 2
        midpoints.append(sl[mid_idx])

    # Sort indices by brightness ascending (darkest first)
    order = sorted(range(n), key=lambda i: float(brightnesses[i]))

    accepted_midpoints: list[list[float]] = []  # list of [x, y] for KD-tree
    accepted_indices: set[int] = set()

    for i in order:
        sl = streamlines[i]
        if not sl:
            continue  # skip empty streamlines

        mx, my = midpoints[i]
        if math.isnan(mx) or math.isnan(my):
            continue

        if not accepted_midpoints:
            # First streamline is always accepted
            accepted_midpoints.append([mx, my])
            accepted_indices.add(i)
            continue

        # Query KD-tree for nearest accepted midpoint
        tree = cKDTree(accepted_midpoints)
        dist, _ = tree.query([mx, my], k=1)

        if dist >= separation_distance_mm:
            accepted_midpoints.append([mx, my])
            accepted_indices.add(i)

    # Return accepted streamlines in original order
    return [streamlines[i] for i in range(n) if i in accepted_indices]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bilinear_sample(
    img: np.ndarray,
    x_mm: float,
    y_mm: float,
    canvas_w: float,
    canvas_h: float,
    img_w: int,
    img_h: int,
) -> float:
    """Bilinearly interpolate *img* at a canvas-mm coordinate.

    Maps ``(x_mm, y_mm)`` into pixel coordinates using the canvas dimensions
    and performs standard bilinear interpolation, clamping at image borders.

    Parameters
    ----------
    img:          Float32 array of shape ``(img_h, img_w)``.
    x_mm, y_mm:  Position in mm within ``[0, canvas_w] × [0, canvas_h]``.
    canvas_w/h:  Canvas size in mm.
    img_w/h:     Image dimensions in pixels.

    Returns
    -------
    Interpolated brightness in ``[0, 1]``.
    """
    # Map mm → fractional pixel coordinates (pixel centre = 0.5, 1.5, …)
    px = (x_mm / canvas_w) * img_w - 0.5
    py = (y_mm / canvas_h) * img_h - 0.5

    x0 = int(math.floor(px))
    y0 = int(math.floor(py))
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp indices to valid range
    x0c = max(0, min(img_w - 1, x0))
    x1c = max(0, min(img_w - 1, x1))
    y0c = max(0, min(img_h - 1, y0))
    y1c = max(0, min(img_h - 1, y1))

    # Fractional parts for blending
    fx = px - x0
    fy = py - y0

    v00 = float(img[y0c, x0c])
    v10 = float(img[y0c, x1c])
    v01 = float(img[y1c, x0c])
    v11 = float(img[y1c, x1c])

    return (
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )


# ---------------------------------------------------------------------------
# LICGenerator — Generator ABC implementation
# ---------------------------------------------------------------------------


def _to_float_gray_lic(img: np.ndarray) -> np.ndarray:
    """Convert any image array to float32 grayscale in [0, 1]."""
    arr = np.asarray(img)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = (
                0.2126 * arr[:, :, 0]
                + 0.7152 * arr[:, :, 1]
                + 0.0722 * arr[:, :, 2]
            ).astype(np.float32)
        else:
            arr = arr[:, :, 0]
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


@register_generator
class LICGenerator(Generator):
    """Line Integral Convolution: image-driven streamline art for pen plotters.

    Traces streamlines from a jittered seed grid along a dense vector field
    derived from the source image (Sobel gradient, ETF, or perpendicular
    gradient).  Density modulation and streamline separation keep the output
    visually balanced and artefact-free.
    """

    name = "Line Integral Convolution"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="vector_field",
                label="Vector Field",
                choices=["gradient", "etf", "perpendicular_gradient"],
                default="etf",
                description="How the flow direction is derived from the source image.",
                choice_descriptions={
                    "gradient": (
                        "Streamlines follow the Sobel brightness gradient — "
                        "they cross edges perpendicularly."
                    ),
                    "etf": (
                        "Streamlines follow the Edge Tangent Flow — "
                        "coherent alignment along image edges (painterly look)."
                    ),
                    "perpendicular_gradient": (
                        "Sobel gradient rotated 90° — "
                        "streamlines run parallel to edges (contour-like)."
                    ),
                },
            ),
            FloatParam(
                name="kernel_length_mm",
                label="Kernel Length (mm)",
                min=2.0,
                max=50.0,
                step=0.5,
                default=15.0,
                description=(
                    "Length of each streamline — controls streak appearance. "
                    "Longer values produce bolder, more sweeping strokes."
                ),
            ),
            FloatParam(
                name="seed_spacing_mm",
                label="Seed Spacing (mm)",
                min=0.5,
                max=10.0,
                step=0.1,
                default=2.0,
                description=(
                    "Distance between streamline seeds — lower = denser coverage. "
                    "Very small values produce many candidate streamlines (before "
                    "separation filtering)."
                ),
            ),
            FloatParam(
                name="separation_distance_mm",
                label="Separation Distance (mm)",
                min=0.2,
                max=5.0,
                step=0.1,
                default=0.8,
                description=(
                    "Minimum distance between neighboring streamlines. "
                    "Increase to spread streamlines further apart."
                ),
            ),
            FloatParam(
                name="step_size_mm",
                label="Step Size (mm)",
                min=0.1,
                max=2.0,
                step=0.05,
                default=0.5,
                description=(
                    "Euler integration step in mm — smaller = smoother curves "
                    "but more computation."
                ),
            ),
            BoolParam(
                name="density_modulation",
                label="Density Modulation",
                default=True,
                description=(
                    "Thin streamlines in bright areas based on image brightness. "
                    "Disable for uniform coverage regardless of image tone."
                ),
            ),
            IntParam(
                name="brightness_threshold",
                label="Brightness Threshold",
                min=0,
                max=255,
                step=1,
                default=220,
                description=(
                    "Brightness above which streamlines are removed (0–255). "
                    "Only active when Density Modulation is enabled."
                ),
                visible_when={"density_modulation": [True]},
            ),
            FloatParam(
                name="etf_kernel_radius",
                label="ETF Kernel Radius",
                min=1.0,
                max=10.0,
                step=0.5,
                default=5.0,
                description="Spatial scale for ETF smoothing in pixels.",
                visible_when={"vector_field": ["etf"]},
            ),
            IntParam(
                name="etf_iterations",
                label="ETF Iterations",
                min=1,
                max=10,
                step=1,
                default=3,
                description="Number of ETF smoothing passes (more = smoother flow).",
                visible_when={"vector_field": ["etf"]},
            ),
            # Standard image preprocessing params
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before processing (-100 to +100).",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before processing (-100 to +100).",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description=(
                    "Gaussian blur applied before vector field computation — "
                    "reduces noise and smooths flow direction."
                ),
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description=(
                    "Invert image tones before processing "
                    "(swaps dense/sparse regions)."
                ),
            ),
            # Offset params
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the output on the canvas page (mm).",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the output on the canvas page (mm).",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default",
                params={
                    "vector_field": "etf",
                    "kernel_length_mm": 15.0,
                    "seed_spacing_mm": 2.0,
                    "separation_distance_mm": 0.8,
                    "step_size_mm": 0.5,
                    "density_modulation": True,
                    "brightness_threshold": 220,
                    "etf_kernel_radius": 5.0,
                    "etf_iterations": 3,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Dense ETF Flow",
                params={
                    "vector_field": "etf",
                    "kernel_length_mm": 20.0,
                    "seed_spacing_mm": 1.0,
                    "separation_distance_mm": 0.5,
                    "step_size_mm": 0.3,
                    "density_modulation": True,
                    "brightness_threshold": 200,
                    "etf_kernel_radius": 7.0,
                    "etf_iterations": 5,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.5,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Contour Lines",
                params={
                    "vector_field": "perpendicular_gradient",
                    "kernel_length_mm": 25.0,
                    "seed_spacing_mm": 2.5,
                    "separation_distance_mm": 1.2,
                    "step_size_mm": 0.5,
                    "density_modulation": True,
                    "brightness_threshold": 230,
                    "etf_kernel_radius": 5.0,
                    "etf_iterations": 3,
                    "brightness": 0.0,
                    "contrast": 15.0,
                    "blur_radius": 2.0,
                    "invert": False,
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
        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        # ------------------------------------------------------------------
        # 1. Image preprocessing
        # ------------------------------------------------------------------
        img = source.copy()
        brightness_val = float(params.get("brightness", 0.0))
        contrast_val = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness_val != 0.0:
            img = adjust_brightness(img, brightness_val)
        if contrast_val != 0.0:
            img = adjust_contrast(img, contrast_val)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        if do_invert:
            img = invert_image(img)

        img_gray = _to_float_gray_lic(img)

        if progress_callback:
            progress_callback(10)

        # ------------------------------------------------------------------
        # 2. Extract parameters
        # ------------------------------------------------------------------
        vf_mode = str(params.get("vector_field", "etf"))
        kernel_length_mm = float(params.get("kernel_length_mm", 15.0))
        seed_spacing_mm = float(params.get("seed_spacing_mm", 2.0))
        separation_distance_mm = float(params.get("separation_distance_mm", 0.8))
        step_size_mm = float(params.get("step_size_mm", 0.5))
        density_modulation = bool(params.get("density_modulation", True))
        brightness_threshold = int(params.get("brightness_threshold", 220))
        etf_kernel_radius = float(params.get("etf_kernel_radius", 5.0))
        etf_iterations = int(params.get("etf_iterations", 3))
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        canvas_w = draw_x2 - draw_x1
        canvas_h = draw_y2 - draw_y1

        if canvas_w <= 0 or canvas_h <= 0:
            return []

        # ------------------------------------------------------------------
        # 3. Compute vector field
        # ------------------------------------------------------------------
        try:
            import cv2
        except ImportError:
            return []

        if vf_mode == "etf":
            from plottter.generators._helpers import _compute_etf
            tx, ty = _compute_etf(img_gray, etf_kernel_radius, etf_iterations)
            vf = np.stack([tx, ty], axis=-1)
        elif vf_mode == "gradient":
            gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=5)
            gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)
            mag = np.sqrt(gx * gx + gy * gy) + 1e-8
            vf = np.stack([gx / mag, gy / mag], axis=-1)
        else:  # perpendicular_gradient — tangent to edges
            gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=5)
            gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=5)
            mag = np.sqrt(gx * gx + gy * gy) + 1e-8
            vf = np.stack([-gy / mag, gx / mag], axis=-1)

        if progress_callback:
            progress_callback(25)
        if cancelled_callback and cancelled_callback():
            return []

        # ------------------------------------------------------------------
        # 4. Seed grid
        # ------------------------------------------------------------------
        rng = np.random.default_rng(42)
        seeds = _seed_grid(canvas_w, canvas_h, seed_spacing_mm, rng)

        if progress_callback:
            progress_callback(35)

        # ------------------------------------------------------------------
        # 5. Brightness filter (density modulation)
        # ------------------------------------------------------------------
        if density_modulation:
            seeds = _brightness_filter(
                seeds,
                canvas_w,
                canvas_h,
                img_gray,
                brightness_threshold / 255.0,
            )

        if progress_callback:
            progress_callback(40)
        if cancelled_callback and cancelled_callback():
            return []

        if not seeds:
            return []

        # ------------------------------------------------------------------
        # 6. Trace streamlines
        # ------------------------------------------------------------------
        streamlines: list[list[tuple[float, float]]] = []
        for seed in seeds:
            sl = _trace_streamline(
                seed, vf, canvas_w, canvas_h, kernel_length_mm, step_size_mm
            )
            streamlines.append(sl)

        if progress_callback:
            progress_callback(70)
        if cancelled_callback and cancelled_callback():
            return []

        # ------------------------------------------------------------------
        # 7. Compute seed brightnesses for separation filter priority
        # ------------------------------------------------------------------
        h_px, w_px = img_gray.shape
        brightnesses: list[float] = [
            _bilinear_sample(img_gray, x_mm, y_mm, canvas_w, canvas_h, w_px, h_px)
            for x_mm, y_mm in seeds
        ]

        # ------------------------------------------------------------------
        # 8. Separation filter
        # ------------------------------------------------------------------
        filtered = _filter_by_separation(streamlines, brightnesses, separation_distance_mm)

        if progress_callback:
            progress_callback(90)

        # ------------------------------------------------------------------
        # 9. Convert from local canvas coords → page mm coordinates
        # ------------------------------------------------------------------
        result: list[Polyline] = [
            [(x + draw_x1 + x_off, y + draw_y1 + y_off) for x, y in sl]
            for sl in filtered
            if len(sl) >= 2
        ]

        if progress_callback:
            progress_callback(100)

        return result
