"""PairedWaveShadingGenerator — paired horizontal-line shading.

Implements a single-stroke shading technique well-suited to colour
separation: every scan line is rendered as **two** polylines whose vertical
deviation from the baseline encodes the channel's darkness at each x.  Where
the channel is dark the lines spread apart (simulating a thick stroke); where
it's light they converge to a thin pair near the baseline.

Algorithm
---------
For each scan line at ``y = i · line_spacing_mm`` inside the image rect:

1.  Sample the channel's brightness at uniform ``sample_interval_mm`` along x.
2.  Convert brightness → darkness ``D(x) ∈ [0, 1]`` with the configured gamma.
3.  Optionally box-smooth ``D(x)`` to remove sample-level jitter.
4.  Compute deviation ``d(x) = min_deviation_mm + D(x) · max_deviation_mm``.
5.  Emit two polylines:

    * **top**:    ``y(x) = baseline − d(x) / 2``
    * **bottom**: ``y(x) = baseline + d(x) / 2``

By design, neither line on its own carries shading information — both
share a centred baseline and only the *pair together* encodes the image
via their symmetric ±d/2 split.  Run on each CMYK channel from Color
Separation and the stacked plot reproduces apparent colour.

Pixels brighter than ``skip_white_above`` are dropped entirely on that x
sample so the pair breaks into separate sub-segments across white space
(prevents a flat double-line being drawn over white backgrounds).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.generators.base import (
    BoolParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _sample_brightness_row(
    gray: np.ndarray,
    y_mm: float,
    x_mm_samples: np.ndarray,
    img_rect: tuple[float, float, float, float],
) -> np.ndarray:
    """Bilinearly sample *gray* at the given mm coordinates.

    Returns an ``(N,)`` uint8 array of brightness values; samples outside the
    image rectangle come back as 255 (white = "no ink").
    """
    rx1, ry1, rx2, ry2 = img_rect
    img_h, img_w = gray.shape[:2]
    rw = max(rx2 - rx1, 1e-9)
    rh = max(ry2 - ry1, 1e-9)

    px = (x_mm_samples - rx1) / rw * img_w
    py = (y_mm - ry1) / rh * img_h

    out = np.full_like(x_mm_samples, 255.0, dtype=np.float32)
    # Inclusive of the right/bottom edge — ``x == rx2`` legitimately maps to
    # the last pixel column.  Indices are clamped below so px == img_w
    # samples the rightmost pixel.
    valid = (px >= 0) & (px <= img_w) & (py >= 0) & (py <= img_h)
    if not valid.any():
        return out.astype(np.uint8)

    # Clamp index *after* the int cast — float32 of (img_w - 1e-6) rounds back
    # to img_w on large indices and would index off the end of the array.
    px_v = np.maximum(px[valid], 0.0)
    px0 = np.minimum(px_v.astype(np.int32), img_w - 1)
    px1 = np.minimum(px0 + 1, img_w - 1)
    fx = np.clip(px_v - px0, 0.0, 1.0)

    py_clamped = max(0.0, float(py))
    py0 = min(int(py_clamped), img_h - 1)
    py1 = min(py0 + 1, img_h - 1)
    fy = min(max(py_clamped - py0, 0.0), 1.0)

    g = gray.astype(np.float32)
    sampled = (
        g[py0, px0] * (1 - fx) * (1 - fy)
        + g[py0, px1] * fx * (1 - fy)
        + g[py1, px0] * (1 - fx) * fy
        + g[py1, px1] * fx * fy
    )
    out[valid] = sampled
    return np.clip(out, 0, 255).astype(np.uint8)


def _box_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Cheap centred box filter for 1-D float arrays.

    Used to suppress per-sample jitter in the deviation curve without pulling
    in scipy.  ``window <= 1`` returns the input unchanged.
    """
    if window <= 1 or arr.size == 0:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="same")


def _emit_pairs_for_row(
    baseline_y_mm: float,
    x_samples: np.ndarray,
    deviation_mm: np.ndarray,
    keep_mask: np.ndarray,
) -> list[Polyline]:
    """Produce up to two polylines (top + bottom) for one scan line.

    Drops samples where ``keep_mask`` is False, splitting both polylines at
    those gaps so white regions become true breaks rather than degenerate
    flat segments.  Sub-segments shorter than 2 points are discarded.
    """
    if not keep_mask.any():
        return []

    half = deviation_mm * 0.5
    y_top = baseline_y_mm - half
    y_bot = baseline_y_mm + half

    out: list[Polyline] = []
    cur_top: list[tuple[float, float]] = []
    cur_bot: list[tuple[float, float]] = []

    for i, keep in enumerate(keep_mask):
        if keep:
            cur_top.append((float(x_samples[i]), float(y_top[i])))
            cur_bot.append((float(x_samples[i]), float(y_bot[i])))
        else:
            if len(cur_top) >= 2:
                out.append(cur_top)
                out.append(cur_bot)
            cur_top = []
            cur_bot = []
    if len(cur_top) >= 2:
        out.append(cur_top)
        out.append(cur_bot)
    return out


@register_generator
class PairedWaveShadingGenerator(Generator):
    """Paired horizontal-line shading — two polylines per scan line whose
    vertical separation tracks local darkness.  Stack across CMYK channels
    via Color Separation for the "hidden image" colour-plot technique
    where neither layer on its own shows the image.
    """

    name = "Paired Wave Shading"
    category = "image"

    # ----------------------------------------------------------------- params
    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="line_spacing_mm",
                label="Line Spacing (mm)",
                min=0.3,
                max=10.0,
                step=0.1,
                default=1.5,
                description="Vertical distance between scan-line pairs.",
            ),
            FloatParam(
                name="max_deviation_mm",
                label="Max Deviation (mm)",
                min=0.0,
                max=5.0,
                step=0.05,
                default=0.8,
                description=(
                    "How far the two lines spread apart in the darkest "
                    "regions — the apparent 'ink width' knob."
                ),
            ),
            FloatParam(
                name="min_deviation_mm",
                label="Min Deviation (mm)",
                min=0.0,
                max=2.0,
                step=0.05,
                default=0.0,
                description=(
                    "Baseline gap between the two lines even in pure white. "
                    "Set > 0 to keep the pair always visible."
                ),
            ),
            FloatParam(
                name="sample_interval_mm",
                label="Sample Interval (mm)",
                min=0.05,
                max=2.0,
                step=0.05,
                default=0.3,
                description=(
                    "How finely to sample brightness along each scan line. "
                    "Smaller = more detail + more points."
                ),
            ),
            FloatParam(
                name="tone_gamma",
                label="Tone Gamma",
                min=0.3,
                max=3.0,
                step=0.1,
                default=1.0,
                description=(
                    "Power curve on darkness: > 1 emphasises mid-tones, < 1 "
                    "pushes more energy into shadows."
                ),
            ),
            FloatParam(
                name="smoothing_mm",
                label="Smoothing (mm)",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.5,
                description=(
                    "Box-smooth the deviation curve over this length to kill "
                    "high-frequency jitter.  0 = raw per-sample values."
                ),
            ),
            IntParam(
                name="skip_white_above",
                label="Skip White Above (0–255)",
                min=0,
                max=255,
                step=1,
                default=240,
                description=(
                    "Drop the pair where brightness exceeds this threshold "
                    "so near-white areas leave the paper blank.  255 = never "
                    "skip."
                ),
            ),
            # ---- shared image preprocessing ----------------------------------
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description=(
                    "Invert brightness before shading.  Useful for CMYK "
                    "channel images that already represent ink coverage "
                    "(0 = no ink, 255 = full ink)."
                ),
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Brightness adjustment applied before sampling.",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Contrast adjustment applied before sampling.",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius (px)",
                min=0.0,
                max=20.0,
                step=0.5,
                default=0.0,
                description=(
                    "Gaussian pre-blur in pixels.  Softens speckle so the "
                    "paired lines glide instead of jittering."
                ),
            ),
        ]

    # --------------------------------------------------------------- presets
    def get_presets(self) -> list[Preset]:
        return [
            # Presets leave ``invert`` at its False default — the colour-
            # separation flow pre-inverts CMYK/RGB channel images at the
            # boundary so every line generator sees luminance semantics
            # uniformly.  Set invert=True only when feeding the generator
            # an ink-coverage image directly (not via Color Separation).
            Preset(
                name="Color Split — Channel Pair",
                description=(
                    "Starting point for the per-channel colour-separation "
                    "workflow: 1.5 mm spacing, 0.8 mm max deviation, gentle "
                    "smoothing.  Pick this under Color Separation → "
                    "Generate Lines and apply it to every channel — works "
                    "for both CMYK and RGB splits.  CMYK pens give faithful "
                    "colour reproduction on white paper; RGB pens give a "
                    "stylised channel-art look since RGB inks mix "
                    "subtractively on paper."
                ),
                params={
                    "line_spacing_mm": 1.5,
                    "max_deviation_mm": 0.8,
                    "min_deviation_mm": 0.0,
                    "sample_interval_mm": 0.3,
                    "tone_gamma": 1.0,
                    "smoothing_mm": 0.5,
                    "skip_white_above": 240,
                },
            ),
            Preset(
                name="Fine Detail",
                description=(
                    "Tighter spacing + smaller deviation for fine-tipped "
                    "pens.  Good for portrait-style colour plots."
                ),
                params={
                    "line_spacing_mm": 0.8,
                    "max_deviation_mm": 0.5,
                    "min_deviation_mm": 0.0,
                    "sample_interval_mm": 0.2,
                    "tone_gamma": 0.9,
                    "smoothing_mm": 0.3,
                    "skip_white_above": 240,
                },
            ),
            Preset(
                name="Bold Shading",
                description=(
                    "Wider spacing + larger deviation for chunky markers and "
                    "large-format plots."
                ),
                params={
                    "line_spacing_mm": 3.0,
                    "max_deviation_mm": 1.6,
                    "min_deviation_mm": 0.2,
                    "sample_interval_mm": 0.5,
                    "tone_gamma": 1.2,
                    "smoothing_mm": 1.0,
                    "skip_white_above": 235,
                },
            ),
            Preset(
                name="Marker (~1.2 mm tip)",
                description=(
                    "Tuned for wide-tip markers (~1 mm class).  "
                    "``min_deviation_mm`` is set above the tip width so the "
                    "two lines stay distinguishable even in light areas, "
                    "``max_deviation_mm`` is generous so dark areas clearly "
                    "spread, and ``skip_white_above`` is aggressive so "
                    "near-white areas drop out entirely instead of getting "
                    "a fat redundant double-line.  Open View → Preview Pen "
                    "Width and set it to 1.2 mm to eyeball the result."
                ),
                params={
                    "line_spacing_mm": 3.5,
                    "max_deviation_mm": 4.0,
                    "min_deviation_mm": 1.4,
                    "sample_interval_mm": 0.5,
                    "tone_gamma": 1.1,
                    "smoothing_mm": 1.5,
                    "skip_white_above": 220,
                },
            ),
        ]

    # ------------------------------------------------------------- generate
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

        # ---- 1. Coerce to grayscale --------------------------------------
        if source.ndim == 3:
            try:
                import cv2
                gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            except ImportError:
                gray = source.mean(axis=2).astype(np.uint8)
        else:
            gray = source.copy()

        # ---- 2. Shared preprocessing -------------------------------------
        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 0.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            gray = adjust_brightness(gray, brightness)
        if contrast != 0.0:
            gray = adjust_contrast(gray, contrast)
        if blur_radius > 0.0:
            gray = apply_blur(gray, blur_radius)
        if do_invert:
            gray = invert_image(gray)

        # ---- 3. Resolve canvas / image rectangle -------------------------
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_h, img_w = gray.shape[:2]
        img_rect = compute_image_rect(
            str(params.get("image_fit_mode", "fit")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        rx1, ry1, rx2, ry2 = img_rect

        # ---- 4. Resolve geometric params (with safety floors) ------------
        spacing = max(float(params.get("line_spacing_mm", 1.5)), 0.05)
        max_dev = max(float(params.get("max_deviation_mm", 0.8)), 0.0)
        min_dev = max(float(params.get("min_deviation_mm", 0.0)), 0.0)
        sample = max(float(params.get("sample_interval_mm", 0.3)), 0.01)
        gamma = max(float(params.get("tone_gamma", 1.0)), 0.01)
        smoothing_mm = max(float(params.get("smoothing_mm", 0.5)), 0.0)
        skip_above = int(np.clip(int(params.get("skip_white_above", 240)), 0, 255))

        # ---- 5. Pre-compute x sample grid (shared across all rows) -------
        rect_w = rx2 - rx1
        rect_h = ry2 - ry1
        if rect_w <= 0 or rect_h <= 0:
            return []

        n_x = max(2, int(np.ceil(rect_w / sample)) + 1)
        x_samples = np.linspace(rx1, rx2, n_x, dtype=np.float32)
        smooth_window = max(1, int(round(smoothing_mm / sample)))

        # ---- 6. Walk scan-line baselines top-to-bottom -------------------
        result: list[Polyline] = []
        n_lines = max(1, int(np.floor(rect_h / spacing)))
        for i in range(n_lines + 1):
            if cancelled_callback and cancelled_callback():
                break
            if progress_callback and n_lines > 0:
                progress_callback(int(100 * i / max(1, n_lines)))

            baseline_y = ry1 + i * spacing
            if baseline_y > ry2:
                break

            brightness_row = _sample_brightness_row(gray, baseline_y, x_samples, img_rect)
            # Brightness 0–255 → darkness 0–1 (255 = no ink)
            darkness = (255.0 - brightness_row.astype(np.float32)) / 255.0
            if gamma != 1.0:
                darkness = np.power(darkness, gamma, dtype=np.float32)
            darkness = _box_smooth(darkness, smooth_window)

            deviation = min_dev + darkness * max_dev
            keep = brightness_row <= skip_above

            result.extend(_emit_pairs_for_row(baseline_y, x_samples, deviation, keep))

        if progress_callback:
            progress_callback(100)
        return result
