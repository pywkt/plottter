"""Sketch generator — darkest-area finder scaffold.

Phase 1: implements the core darkest-region / darkest-pixel search helpers and
the generator scaffold.  The generate() method loads + preprocesses the source
image and validates the helpers, but returns empty polylines for now.
"""

from __future__ import annotations

import math as _math
from typing import Any

import numpy as np

# Direction mode angle offsets (in radians)
_STRAIGHT_OFFSETS: list[float] = [_math.radians(a) for a in (-14, -7, 0, 7, 14)]
_CURVE_OFFSETS: list[float] = [_math.radians(a) for a in (-80, -45, -20, 0, 20, 45, 80)]
_AXIAL_OFFSETS: list[float] = [_math.radians(a) for a in (0, 90, 180, 270)]

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
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


@register_generator
class SketchGenerator(Generator):
    name = "Sketch"
    category = "image"

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="line_density",
                label="Line Density",
                min=0.1,
                max=5.0,
                step=0.1,
                default=1.0,
                description="Target density — higher = more lines, darker result",
            ),
            IntParam(
                name="line_max_limit",
                label="Max Line Segments",
                min=100,
                max=100_000,
                step=100,
                default=10_000,
                description="Maximum number of line segments",
            ),
            IntParam(
                name="line_length_px",
                label="Line Length (px)",
                min=5,
                max=100,
                step=1,
                default=20,
                description="Length of each line segment in pixels",
            ),
            IntParam(
                name="angle_tests",
                label="Angle Tests",
                min=4,
                max=72,
                step=1,
                default=36,
                description="Candidate directions tested per step — values <36 use evenly-spaced angles; ≥36 tests all integer points on a Bresenham circle (full pixel-resolution coverage)",
            ),
            # --- squiggle params ---
            IntParam(
                name="squiggle_min_length",
                label="Squiggle Min Length",
                min=1,
                max=100,
                step=1,
                default=3,
                description="Minimum number of segments a squiggle must have to be kept",
            ),
            IntParam(
                name="squiggle_max_length",
                label="Squiggle Max Length",
                min=5,
                max=500,
                step=1,
                default=50,
                description="Maximum number of segments per squiggle",
            ),
            FloatParam(
                name="squiggle_max_deviation",
                label="Squiggle Max Deviation",
                min=0.0,
                max=100.0,
                step=1.0,
                default=25.0,
                description="How far into bright areas a squiggle can go before stopping — higher = longer squiggles",
            ),
            # --- erase params ---
            IntParam(
                name="erase_min",
                label="Erase Min",
                min=1,
                max=100,
                step=1,
                default=1,
                description="Minimum erase amount applied at dark pixels — small = many fine strokes in shadows",
            ),
            IntParam(
                name="erase_max",
                label="Erase Max",
                min=10,
                max=255,
                step=1,
                default=100,
                description="Maximum erase amount applied at bright pixels — large = bright areas quickly used up",
            ),
            IntParam(
                name="erase_radius_min",
                label="Erase Radius Min (px)",
                min=1,
                max=10,
                step=1,
                default=1,
                description="Erase radius at dark pixels — smaller = finer strokes in shadows",
            ),
            IntParam(
                name="erase_radius_max",
                label="Erase Radius Max (px)",
                min=1,
                max=20,
                step=1,
                default=4,
                description="Erase radius at bright pixels — larger = wider brightening in highlights",
            ),
            FloatParam(
                name="tone",
                label="Tone Curve",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.5,
                description="Blends linear (0) and cubic (1) easing — higher = more contrast between dark/bright erase amounts",
            ),
            # --- directionality / edge params ---
            FloatParam(
                name="directionality",
                label="Directionality",
                min=0.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Follow natural contours — higher values push lines along directions of lowest brightness variance",
            ),
            FloatParam(
                name="edge_power",
                label="Edge Power",
                min=0.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Attract lines toward detected edges",
            ),
            IntParam(
                name="max_pixel_coverage",
                label="Max Pixel Coverage",
                min=1,
                max=10,
                step=1,
                default=2,
                description="Maximum times a pixel can be used by accepted paths — prevents over-inking",
            ),
            FloatParam(
                name="max_overlap_ratio",
                label="Max Overlap Ratio",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.55,
                description="Reject paths where this fraction of pixels already have ink",
            ),
            IntParam(
                name="coverage_radius",
                label="Coverage Radius (px)",
                min=0,
                max=5,
                step=1,
                default=1,
                description="Pixel radius to reserve around accepted paths",
            ),
            BoolParam(
                name="continuous",
                label="Continuous",
                default=True,
                description="Chain strokes into continuous paths — fewer pen lifts",
            ),
            IntParam(
                name="chain_max",
                label="Chain Max",
                min=1,
                max=50,
                step=1,
                default=18,
                description="Maximum segments chained without pen lift",
            ),
            FloatParam(
                name="long_line_bias",
                label="Long Line Bias",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.5,
                description="Probability of random long-line bonus in dark areas — 0 = uniform length, 1 = frequent long strokes",
            ),
            FloatParam(
                name="straight_bias",
                label="Straight Bias",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.7,
                description="Tendency to maintain stroke direction — higher = more flowing parallel strokes",
            ),
            BoolParam(
                name="multi_pass",
                label="Multi-Pass",
                default=True,
                description="Use 3-pass generation with varying stroke profiles — long strokes first, fine detail last",
            ),
            # --- standard image preprocessing params ---
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image (dark and bright areas swap roles)",
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
                default=1.0,
                description="Gaussian blur applied before processing — smooths brightness transitions",
            ),
            IntParam(
                name="unsharp_amount",
                label="Unsharp Mask",
                min=0,
                max=5,
                step=1,
                default=2,
                description="Unsharp mask strength applied after blur — 0 = disabled, higher = sharper edges",
            ),
            # --- offset params ---
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
            ChoiceParam(
                name="mark_mode",
                label="Mark Mode",
                choices=["Squiggle Only", "Hybrid"],
                default="Squiggle Only",
                description="Squiggle Only: existing tracing only. Hybrid: pick best of squiggle, straight lines, hatch marks, and circles per seed.",
            ),
        ]

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def get_presets(self) -> list[Preset]:
        _base = {
            "line_density": 1.0,
            "line_max_limit": 10_000,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 1.0,
            "unsharp_amount": 2,
            "squiggle_min_length": 3,
            "squiggle_max_length": 50,
            "squiggle_max_deviation": 25.0,
            "angle_tests": 36,
            "line_length_px": 20,
            "erase_min": 1,
            "erase_max": 100,
            "erase_radius_min": 1,
            "erase_radius_max": 4,
            "tone": 0.5,
            "directionality": 0.0,
            "edge_power": 0.0,
            "max_pixel_coverage": 2,
            "max_overlap_ratio": 0.55,
            "coverage_radius": 1,
            "continuous": True,
            "chain_max": 18,
            "long_line_bias": 0.5,
            "straight_bias": 0.7,
            "multi_pass": True,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            "mark_mode": "Squiggle Only",
        }
        return [
            # --- Quick / light presets (fast, good for preview) ---
            Preset(
                name="Quick Sketch",
                params={
                    **_base,
                    "line_max_limit": 3000,
                    "angle_tests": 12,
                    "line_length_px": 15,
                    "squiggle_max_length": 30,
                    "chain_max": 12,
                    "multi_pass": False,
                },
                description="Fast preview — fewer lines, lower quality, good for testing settings",
            ),
            # --- Portrait / figure presets ---
            Preset(
                name="Portrait",
                params={
                    **_base,
                    "line_density": 1.5,
                    "line_max_limit": 8000,
                    "angle_tests": 16,
                    "line_length_px": 15,
                    "squiggle_max_length": 40,
                    "squiggle_max_deviation": 20.0,
                    "directionality": 30.0,
                    "edge_power": 15.0,
                    "straight_bias": 0.8,
                    "long_line_bias": 0.3,
                    "erase_radius_max": 3,
                    "tone": 0.6,
                    "unsharp_amount": 3,
                    "chain_max": 20,
                },
                description="Optimized for faces and figures — follows contours, fine detail in features",
            ),
            Preset(
                name="Contour Portrait",
                params={
                    **_base,
                    "line_density": 1.2,
                    "line_max_limit": 6000,
                    "angle_tests": 16,
                    "line_length_px": 18,
                    "directionality": 60.0,
                    "edge_power": 30.0,
                    "straight_bias": 0.6,
                    "squiggle_max_length": 60,
                    "squiggle_max_deviation": 15.0,
                    "long_line_bias": 0.2,
                    "chain_max": 25,
                    "unsharp_amount": 3,
                    "tone": 0.7,
                },
                description="Strong contour following — lines trace along edges and features",
            ),
            # --- Dense / detailed presets ---
            Preset(
                name="Dense Ink",
                params={
                    **_base,
                    "line_density": 2.5,
                    "line_max_limit": 15000,
                    "angle_tests": 16,
                    "line_length_px": 12,
                    "squiggle_max_length": 25,
                    "max_pixel_coverage": 4,
                    "max_overlap_ratio": 0.7,
                    "coverage_radius": 0,
                    "erase_min": 1,
                    "erase_max": 60,
                    "erase_radius_min": 1,
                    "erase_radius_max": 2,
                    "tone": 0.3,
                    "straight_bias": 0.5,
                    "chain_max": 10,
                },
                description="Very dense coverage — many short strokes, dark result, good for high contrast images",
            ),
            Preset(
                name="Crosshatch",
                params={
                    **_base,
                    "line_density": 1.8,
                    "line_max_limit": 8000,
                    "angle_tests": 4,
                    "line_length_px": 25,
                    "squiggle_max_length": 15,
                    "squiggle_max_deviation": 40.0,
                    "straight_bias": 0.9,
                    "long_line_bias": 0.6,
                    "max_pixel_coverage": 3,
                    "continuous": False,
                    "erase_radius_max": 3,
                },
                description="Angular crosshatch pattern — 4 directions only, long straight strokes",
            ),
            # --- Loose / artistic presets ---
            Preset(
                name="Loose Sketch",
                params={
                    **_base,
                    "line_density": 0.6,
                    "line_max_limit": 3000,
                    "angle_tests": 12,
                    "line_length_px": 25,
                    "squiggle_max_length": 80,
                    "squiggle_max_deviation": 50.0,
                    "straight_bias": 0.85,
                    "long_line_bias": 0.8,
                    "chain_max": 35,
                    "erase_radius_min": 2,
                    "erase_radius_max": 6,
                    "erase_min": 5,
                    "erase_max": 120,
                    "max_pixel_coverage": 1,
                    "coverage_radius": 2,
                },
                description="Few long flowing strokes — sparse, gestural, hand-drawn feel",
            ),
            Preset(
                name="Scribble",
                params={
                    **_base,
                    "line_density": 1.0,
                    "line_max_limit": 5000,
                    "angle_tests": 8,
                    "line_length_px": 10,
                    "squiggle_max_length": 100,
                    "squiggle_max_deviation": 60.0,
                    "straight_bias": 0.3,
                    "long_line_bias": 0.1,
                    "chain_max": 40,
                    "directionality": 10.0,
                    "continuous": True,
                    "max_pixel_coverage": 2,
                    "erase_radius_max": 5,
                },
                description="Chaotic scribble — short erratic steps, long wandering chains",
            ),
            # --- Edge-focused presets ---
            Preset(
                name="Edge Trace",
                params={
                    **_base,
                    "line_density": 1.0,
                    "line_max_limit": 5000,
                    "angle_tests": 16,
                    "line_length_px": 18,
                    "edge_power": 60.0,
                    "directionality": 25.0,
                    "straight_bias": 0.7,
                    "squiggle_max_length": 60,
                    "squiggle_max_deviation": 12.0,
                    "unsharp_amount": 4,
                    "chain_max": 20,
                    "long_line_bias": 0.3,
                    "max_pixel_coverage": 2,
                },
                description="Lines concentrate on edges — strong unsharp mask, tight squiggle deviation",
            ),
            # --- Hybrid presets ---
            Preset(
                name="Hybrid Portrait",
                params={
                    **_base,
                    "mark_mode": "Hybrid",
                    "line_density": 1.5,
                    "line_max_limit": 6000,
                    "angle_tests": 12,
                    "line_length_px": 15,
                    "directionality": 20.0,
                    "edge_power": 15.0,
                    "straight_bias": 0.7,
                    "unsharp_amount": 3,
                    "chain_max": 15,
                    "max_pixel_coverage": 2,
                    "squiggle_max_length": 30,
                },
                description="Mixed mark types — squiggles on edges, dots in flat areas, lines in gradients",
            ),
            Preset(
                name="Hybrid Dense",
                params={
                    **_base,
                    "mark_mode": "Hybrid",
                    "line_density": 2.0,
                    "line_max_limit": 10000,
                    "angle_tests": 12,
                    "line_length_px": 12,
                    "max_pixel_coverage": 3,
                    "max_overlap_ratio": 0.65,
                    "straight_bias": 0.5,
                    "long_line_bias": 0.2,
                    "chain_max": 12,
                    "squiggle_max_length": 20,
                },
                description="Dense hybrid — heavy coverage with varied mark types",
            ),
        ]

    # ------------------------------------------------------------------
    # Seed sampling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_seed_batch(
        residual_dark: np.ndarray,
        edge_normalized: np.ndarray,
        coverage: np.ndarray,
        img_h: int,
        img_w: int,
        batch_size: int,
        dark_power: float,
        edge_bias: float,
        max_pixel_coverage: int,
        min_darkness: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a batch of seed pixel coordinates weighted by darkness/edge/coverage.

        Builds a probability distribution from the current residual darkness,
        edge, and coverage maps, then samples ``batch_size`` seed candidates.

        Parameters
        ----------
        residual_dark:
            Float32 array [0, 1] of remaining darkness (1=black, 0=white).
        edge_normalized:
            Float32 normalized Sobel magnitude [0, 1].
        coverage:
            uint8 array counting accepted path hits per pixel.
        img_h, img_w:
            Image dimensions in pixels.
        batch_size:
            Number of seed candidates to sample.
        dark_power:
            Exponent applied to darkness — higher concentrates seeds in dark areas.
        edge_bias:
            Weight for the edge term in [0, 1].
        max_pixel_coverage:
            Maximum coverage count used to compute the ink penalty.
        min_darkness:
            Minimum darkness below which pixels are excluded from sampling.
        rng:
            NumPy random Generator instance.

        Returns
        -------
        ``(ys, xs)`` arrays of shape ``(batch_size,)`` containing pixel row/col
        indices sampled proportionally to the computed weight map.
        """
        # Downsample the maps for faster probability computation.
        # Working at 1/4 resolution (16x fewer pixels) is sufficient for
        # seed placement accuracy and dramatically speeds up rng.choice().
        DOWNSAMPLE = 4
        ds_h = max(1, img_h // DOWNSAMPLE)
        ds_w = max(1, img_w // DOWNSAMPLE)

        try:
            import cv2
            ds_dark = cv2.resize(residual_dark, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            ds_edge = cv2.resize(edge_normalized, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            ds_cov = cv2.resize(coverage.astype(np.float32), (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback: simple stride-based downsampling
            ds_dark = residual_dark[::DOWNSAMPLE, ::DOWNSAMPLE]
            ds_edge = edge_normalized[::DOWNSAMPLE, ::DOWNSAMPLE]
            ds_cov = coverage[::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float32)

        darkness = np.clip(ds_dark - min_darkness, 0.0, 1.0)
        edge_term = ds_edge ** 1.2
        ink_penalty = np.clip(
            1.0 - ds_cov / max(1, max_pixel_coverage),
            0.05, 1.0,
        )
        weights = (darkness ** dark_power) * (0.85 + edge_bias * edge_term) * ink_penalty + 1e-9
        weights /= weights.sum()

        flat_weights = weights.ravel()
        indices = rng.choice(flat_weights.size, size=batch_size, replace=True, p=flat_weights)
        ds_ys, ds_xs = np.divmod(indices, ds_w)

        # Scale back to full resolution with small random jitter within each cell
        ys = np.clip(ds_ys * DOWNSAMPLE + rng.integers(0, DOWNSAMPLE, size=batch_size), 0, img_h - 1)
        xs = np.clip(ds_xs * DOWNSAMPLE + rng.integers(0, DOWNSAMPLE, size=batch_size), 0, img_w - 1)
        return ys, xs

    # ------------------------------------------------------------------
    # Bresenham helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bresenham_line(
        x0: int, y0: int, x1: int, y1: int
    ):
        """Yield ``(x, y)`` integer pixel coordinates along the line from
        ``(x0, y0)`` to ``(x1, y1)`` using Bresenham's algorithm.

        Both endpoints are included.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            yield (x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # ------------------------------------------------------------------
    # Darkest-line finding
    # ------------------------------------------------------------------

    @staticmethod
    def _find_darkest_line(
        lightened: np.ndarray,
        cur_x: int,
        cur_y: int,
        angle_tests: int,
        line_length_px: int,
        residual_dark: np.ndarray | None = None,
        edge_normalized: np.ndarray | None = None,
        coverage: np.ndarray | None = None,
        max_pixel_coverage: int = 1,
        base_angle: float | None = None,
        angle_offsets: list[float] | None = None,
    ) -> tuple[int, int, float, float] | None:
        """Find the single best line segment from ``(cur_x, cur_y)``.

        Tests candidate endpoints at distance ``line_length_px`` from the
        current position. Scoring uses a weighted formula combining residual
        darkness, peak darkness, edge strength, and coverage penalty.

        Candidate generation:
        - If ``angle_tests < 36``: test ``angle_tests`` evenly-spaced angles.
        - If ``angle_tests >= 36``: test all integer points on the Bresenham
          circle of radius ``line_length_px`` (full pixel-resolution coverage).

        Scoring formula (higher = better)::

            score = dark * 1.45 + dark_peak * 0.45 + edge_strength * 0.24 - cov * 1.08

        Samples are drawn via ``np.linspace`` along each candidate line.

        Parameters
        ----------
        lightened:
            2-D uint8 grayscale image (0=black, 255=white). Used to derive
            darkness when ``residual_dark`` is not provided, and to compute
            ``avg_brightness`` for the squiggle stopping check.
        cur_x, cur_y:
            Current pixel position (column, row).
        angle_tests:
            Number of candidate angles (< 36) or Bresenham-circle mode (>= 36).
        line_length_px:
            Radius / length of each candidate line segment in pixels.
        residual_dark:
            Float32 array [0, 1] of remaining darkness (1=black, 0=white).
        edge_normalized:
            Float32 normalized Sobel magnitude [0, 1].
        coverage:
            uint8 array counting how many accepted paths touched each pixel.
        max_pixel_coverage:
            Maximum coverage count used to normalize the coverage penalty.

        Returns
        -------
        ``(end_x, end_y, avg_brightness, best_score)`` for the winning endpoint, or
        ``None`` if no valid candidate exists.  ``avg_brightness`` is the
        mean of ``lightened`` values along the sampled points (0–255 range).
        ``best_score`` is the raw weighted score used for candidate ranking.
        """
        h, w = lightened.shape[:2]

        # Generate candidate endpoints
        r = max(1, line_length_px)
        candidates: list[tuple[int, int]] = []

        if base_angle is not None and angle_offsets is not None:
            # Direction-constrained: test specific angular offsets from the base angle
            seen: set[tuple[int, int]] = set()
            for offset in angle_offsets:
                angle = base_angle + offset
                ex = cur_x + int(round(np.cos(angle) * r))
                ey = cur_y + int(round(np.sin(angle) * r))
                pt = (ex, ey)
                if pt not in seen:
                    seen.add(pt)
                    candidates.append(pt)
        elif angle_tests >= 36:
            # Midpoint circle algorithm: all integer points on a circle of radius r
            xc, yc = r, 0
            err = 0
            pts: set[tuple[int, int]] = set()
            while xc >= yc:
                for dpx, dpy in [
                    (xc, yc), (-xc, yc), (xc, -yc), (-xc, -yc),
                    (yc, xc), (-yc, xc), (yc, -xc), (-yc, -xc),
                ]:
                    pts.add((cur_x + dpx, cur_y + dpy))
                if err <= 0:
                    yc += 1
                    err += 2 * yc + 1
                if err > 0:
                    xc -= 1
                    err -= 2 * xc + 1
            candidates = list(pts)
        else:
            for i in range(angle_tests):
                angle = i * 2.0 * np.pi / angle_tests
                ex = cur_x + int(round(np.cos(angle) * r))
                ey = cur_y + int(round(np.sin(angle) * r))
                candidates.append((ex, ey))

        # Filter candidates to those within image bounds
        valid = [(ex, ey) for ex, ey in candidates if 0 <= ey < h and 0 <= ex < w]
        if not valid:
            return None

        # Vectorized scoring: sample a fixed number of points along each
        # candidate line and score all candidates in one batch of numpy ops.
        N_SAMPLES = 8  # fixed sample count for uniform array shapes
        n_cands = len(valid)
        ends_x = np.array([ex for ex, _ in valid], dtype=np.int32)
        ends_y = np.array([ey for _, ey in valid], dtype=np.int32)

        # Build sample coordinates for all candidates at once: (n_cands, N_SAMPLES)
        t = np.linspace(0.0, 1.0, N_SAMPLES).reshape(1, N_SAMPLES)  # (1, N)
        sx = np.clip(np.round(cur_x + (ends_x[:, None] - cur_x) * t).astype(np.int32), 0, w - 1)  # (n_cands, N)
        sy = np.clip(np.round(cur_y + (ends_y[:, None] - cur_y) * t).astype(np.int32), 0, h - 1)  # (n_cands, N)

        # Gather values for all candidates × all samples in one indexing op
        if residual_dark is not None:
            dark_vals = residual_dark[sy, sx]  # (n_cands, N)
        else:
            dark_vals = 1.0 - lightened[sy, sx].astype(np.float32) / 255.0

        dark_means = dark_vals.mean(axis=1)       # (n_cands,)
        dark_peaks = dark_vals.max(axis=1)         # (n_cands,)

        if edge_normalized is not None:
            edge_means = edge_normalized[sy, sx].mean(axis=1)
        else:
            edge_means = np.zeros(n_cands, dtype=np.float32)

        denom_cov = 1.0 / max(1, max_pixel_coverage)
        if coverage is not None:
            cov_means = (coverage[sy, sx].astype(np.float32) * denom_cov).mean(axis=1)
        else:
            cov_means = np.zeros(n_cands, dtype=np.float32)

        scores = dark_means * 1.45 + dark_peaks * 0.45 + edge_means * 0.24 - cov_means * 1.08

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_end = (int(ends_x[best_idx]), int(ends_y[best_idx]))
        best_avg_brightness = float(lightened[sy[best_idx], sx[best_idx]].astype(np.float32).mean())

        if best_end is None:
            return None
        return (best_end[0], best_end[1], best_avg_brightness, best_score)

    # ------------------------------------------------------------------
    # Erase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _erase_along_path(
        lightened: np.ndarray,
        path: list[tuple[float, float]],
        erase_min: int,
        erase_max: int,
        radius_min: int,
        radius_max: int,
        tone: float,
    ) -> float:
        """Brighten a square region around each point in *path*, with the
        erase amount and radius scaled by the local pixel luminance.

        Dark pixels (lum≈0) receive a tiny erase_min amount over a radius_min
        region — barely lightened.  Already-bright pixels (lum≈255) receive a
        large erase_max amount over a radius_max region — aggressively lightened.
        This causes dark areas to accumulate many fine strokes while bright
        areas are quickly "used up".

        Parameters
        ----------
        lightened:
            2-D uint8 grayscale image that is modified in-place.
        path:
            List of ``(px_x, px_y)`` float pixel coordinates.
        erase_min:
            Minimum erase amount applied at fully dark pixels.
        erase_max:
            Maximum erase amount applied at fully bright pixels.
        radius_min:
            Erase radius (in pixels) at fully dark pixels.
        radius_max:
            Erase radius (in pixels) at fully bright pixels.
        tone:
            Blending weight (0–1) between linear easing (0) and cubic (1).

        Returns
        -------
        Total sum of actual pixel increments (accounting for 255 clamping).
        Use this to maintain an incremental running brightness sum.
        """
        if not path:
            return 0.0

        h, w = lightened.shape[:2]
        total_delta = 0.0

        for px_x, px_y in path:
            cx = int(round(px_x))
            cy = int(round(px_y))
            if not (0 <= cy < h and 0 <= cx < w):
                continue

            # Luminance-dependent easing
            lum = float(lightened[cy, cx])
            t = lum / 255.0
            eased = (t ** 3 * tone) + (t * (1.0 - tone))

            amount = int(round(erase_min + eased * (erase_max - erase_min)))
            r = int(round(radius_min + eased * (radius_max - radius_min)))

            y0 = max(0, cy - r)
            y1 = min(h, cy + r + 1)
            x0 = max(0, cx - r)
            x1 = min(w, cx + r + 1)
            if y0 >= y1 or x0 >= x1:
                continue

            region = lightened[y0:y1, x0:x1]
            new_region = np.minimum(
                region.astype(np.uint16) + amount, 255
            ).astype(np.uint8)
            total_delta += float(new_region.sum()) - float(region.sum())
            lightened[y0:y1, x0:x1] = new_region

        return total_delta

    # ------------------------------------------------------------------
    # Coverage-map helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _path_pixel_coords(
        points: list[tuple[float, float]],
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize a polyline to unique pixel coordinates using np.linspace interpolation.

        Returns ``(xs, ys)`` arrays of int32 pixel coordinates within the image bounds.
        """
        if len(points) < 2:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        xs_parts: list[np.ndarray] = []
        ys_parts: list[np.ndarray] = []

        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0))))
            seg_x = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int32)
            seg_y = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int32)
            valid = (seg_x >= 0) & (seg_x < width) & (seg_y >= 0) & (seg_y < height)
            if np.any(valid):
                xs_parts.append(seg_x[valid])
                ys_parts.append(seg_y[valid])

        if not xs_parts:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        xs = np.concatenate(xs_parts)
        ys = np.concatenate(ys_parts)
        flat = ys * width + xs
        unique_flat = np.unique(flat)
        return (unique_flat % width).astype(np.int32), (unique_flat // width).astype(np.int32)

    @staticmethod
    def _expand_pixel_coords(
        xs: np.ndarray,
        ys: np.ndarray,
        width: int,
        height: int,
        radius: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand pixel coordinates outward by *radius* using meshgrid offsets.

        Returns a deduplicated ``(xs, ys)`` pair clamped to image bounds.
        """
        if xs.size == 0 or ys.size == 0 or radius <= 0:
            return xs, ys

        offsets = np.arange(-radius, radius + 1, dtype=np.int32)
        dx, dy = np.meshgrid(offsets, offsets)
        dx = dx.ravel()
        dy = dy.ravel()

        ex = (xs[:, None] + dx[None, :]).ravel()
        ey = (ys[:, None] + dy[None, :]).ravel()
        valid = (ex >= 0) & (ex < width) & (ey >= 0) & (ey < height)
        ex = ex[valid]
        ey = ey[valid]

        flat = ey * width + ex
        unique_flat = np.unique(flat)
        return (unique_flat % width).astype(np.int32), (unique_flat // width).astype(np.int32)

    # ------------------------------------------------------------------
    # Hybrid mark scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_path(
        points: list[tuple[float, float]],
        residual_dark: np.ndarray,
        edge_normalized: np.ndarray,
        coverage: np.ndarray,
        max_pixel_coverage: int,
        img_w: int,
        img_h: int,
    ) -> float:
        """Score a candidate path using dark/peak/edge/coverage weighted formula.

        Higher score = better candidate (more dark coverage, less already-inked).

        Returns
        -------
        Weighted score: ``dark*1.42 + dark_peak*0.40 + edge*0.22 - cov*1.05``.
        Returns ``-1e9`` when the path has no valid pixel coordinates.
        """
        px, py = SketchGenerator._path_pixel_coords(points, img_w, img_h)
        if px.size == 0:
            return -1e9
        dark_vals = residual_dark[py, px]
        dark = float(np.mean(dark_vals))
        dark_peak = float(np.max(dark_vals))
        edge_strength = float(np.mean(edge_normalized[py, px]))
        cov = float(np.mean(coverage[py, px].astype(np.float32) / max(1, max_pixel_coverage)))
        return dark * 1.42 + dark_peak * 0.40 + edge_strength * 0.22 - cov * 1.05

    @staticmethod
    def _circle_mark_pixels(
        cx: float,
        cy: float,
        radius: float,
        steps: int,
        img_w: int,
        img_h: int,
    ) -> list[tuple[float, float]]:
        """Generate a closed circle polyline in pixel space.

        Parameters
        ----------
        cx, cy:
            Circle centre in pixel coordinates.
        radius:
            Circle radius in pixels.
        steps:
            Number of arc segments (result has ``steps + 1`` points).
        img_w, img_h:
            Image dimensions used for clamping.

        Returns
        -------
        List of ``(x, y)`` pixel-space points (closed: first == last).
        """
        pts: list[tuple[float, float]] = []
        for i in range(steps + 1):
            a = (2.0 * _math.pi * i) / steps
            x = max(0.0, min(float(img_w - 1), cx + _math.cos(a) * radius))
            y = max(0.0, min(float(img_h - 1), cy + _math.sin(a) * radius))
            pts.append((x, y))
        return pts

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

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

        # --- preprocessing ---
        img = source.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            img = adjust_brightness(img, brightness)
        if contrast != 0.0:
            img = adjust_contrast(img, contrast)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        unsharp_amount = int(params.get("unsharp_amount", 2))
        if unsharp_amount > 0:
            gray_f = img.astype(np.float32) / 255.0
            blurred_f = apply_blur(img, 2.0).astype(np.float32) / 255.0
            gray_f = np.clip(gray_f + (gray_f - blurred_f) * unsharp_amount, 0.0, 1.0)
            img = (gray_f * 255.0).astype(np.uint8)
        if do_invert:
            img = invert_image(img)

        # Ensure grayscale
        if img.ndim == 3:
            try:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except ImportError:
                img = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(np.uint8)

        img_h, img_w = img.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_rect = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        # Working copy for darkness search
        lightened = img.copy()

        # Coverage map: counts how many accepted paths have touched each pixel.
        # residual_dark: float32 array tracking remaining darkness per pixel (0=white, 1=black).
        coverage = np.zeros((img_h, img_w), dtype=np.uint8)
        residual_dark = np.clip(1.0 - img.astype(np.float32) / 255.0, 0.0, 1.0)

        # Original darkness — used for step-length modulation (constant, not updated during generation).
        orig_dark = residual_dark.copy()

        # Internal constants (not exposed as parameters)
        brightness_ceiling = 250

        # --- generation parameters ---
        line_density = float(params.get("line_density", 1.0))
        line_max_limit = int(params.get("line_max_limit", 10_000))
        squiggle_min_length = int(params.get("squiggle_min_length", 3))
        squiggle_max_length = int(params.get("squiggle_max_length", 50))
        squiggle_max_deviation = float(params.get("squiggle_max_deviation", 25.0))
        angle_tests = int(params.get("angle_tests", 36))
        line_length_px = int(params.get("line_length_px", 20))
        erase_min = int(params.get("erase_min", 1))
        erase_max = int(params.get("erase_max", 100))
        erase_radius_min = int(params.get("erase_radius_min", 1))
        erase_radius_max = int(params.get("erase_radius_max", 4))
        tone = float(params.get("tone", 0.5))
        directionality = float(params.get("directionality", 0.0))
        edge_power = float(params.get("edge_power", 0.0))
        max_pixel_coverage = int(params.get("max_pixel_coverage", 2))
        max_overlap_ratio = float(params.get("max_overlap_ratio", 0.55))
        coverage_radius = int(params.get("coverage_radius", 1))
        # Amount to subtract from residual_dark per accepted coverage hit.
        # After max_pixel_coverage hits, residual reaches 0 for that pixel.
        lighten_amount = 1.0 / max(1, max_pixel_coverage)
        continuous = bool(params.get("continuous", True))
        chain_max = int(params.get("chain_max", 18))
        long_line_bias = float(params.get("long_line_bias", 0.5))
        straight_bias = float(params.get("straight_bias", 0.7))
        mark_mode = str(params.get("mark_mode", "Squiggle Only"))
        _min_dark_threshold = 0.02

        # --- precompute normalized Sobel edge map ---
        try:
            import cv2
            _gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            _gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        except ImportError:
            _gy, _gx = np.gradient(img.astype(np.float32))
        _mag = np.sqrt(_gx ** 2 + _gy ** 2).astype(np.float32)
        edge_normalized = (_mag / (_mag.max() + 1e-8)).astype(np.float32)

        # --- coordinate mapping: pixel → mm ---
        rect_x1, rect_y1, rect_x2, rect_y2 = img_rect
        rect_w = rect_x2 - rect_x1
        rect_h = rect_y2 - rect_y1

        # --- multi-pass profiles ---
        _pass_profiles = [
            {"fraction": 0.52, "len_scale": 1.90, "dark_power": 1.22, "lighten": 0.068, "straight_bias": 0.78},
            {"fraction": 0.33, "len_scale": 1.15, "dark_power": 1.42, "lighten": 0.048, "straight_bias": 0.62},
            {"fraction": 0.15, "len_scale": 0.62, "dark_power": 1.70, "lighten": 0.034, "straight_bias": 0.36},
        ]
        multi_pass = bool(params.get("multi_pass", True))
        if multi_pass:
            profiles_to_run = _pass_profiles
        else:
            profiles_to_run = [{"fraction": 1.0, "len_scale": 1.0, "dark_power": 1.0, "lighten": None, "straight_bias": 1.0}]

        # Segment budget per pass
        pass_targets = [int(line_max_limit * p["fraction"]) for p in profiles_to_run]
        pass_targets[-1] = line_max_limit - sum(pass_targets[:-1])

        result: list[Polyline] = []
        total_segments = 0
        num_passes = len(profiles_to_run)

        # Seed batch state — rebuilt at the start of each pass and when exhausted
        rng = np.random.default_rng()
        _seed_batch_ys: np.ndarray | None = None
        _seed_batch_xs: np.ndarray | None = None
        _seed_batch_idx: int = 0
        # Scratch lightened buffer for hybrid squiggle candidate tracing (lazy alloc)
        _lightened_scratch: np.ndarray | None = None

        for pass_idx, profile in enumerate(profiles_to_run):
            if cancelled_callback and cancelled_callback():
                break

            pass_target = pass_targets[pass_idx]
            if pass_target <= 0:
                continue

            # Per-pass derived parameters
            pass_line_length = max(1, int(line_length_px * profile["len_scale"]))
            pass_lighten = profile["lighten"] if profile["lighten"] is not None else lighten_amount
            pass_ceiling = max(10, int(brightness_ceiling / profile["dark_power"]))
            pass_max_squiggle = max(squiggle_min_length + 1, int(squiggle_max_length * profile["straight_bias"]))

            pass_segments = 0
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = 1000
            # Reset seed batch at the start of each pass so the distribution
            # is rebuilt from the current residual_dark / coverage state.
            _seed_batch_ys = None
            _seed_batch_xs = None
            _seed_batch_idx = 0

            # Continuous-chaining state (reset each pass)
            current_chain: list[tuple[float, float]] = []
            chain_seg_count = 0
            chaining = False
            seed_x = seed_y = 0  # satisfy type-checker; always set before use
            prev_angle: float | None = None  # direction momentum; reset per new stroke

            while pass_segments < pass_target:
                if cancelled_callback and cancelled_callback():
                    break

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break

                # ----------------------------------------------------------------
                # Hybrid mark selection (when mark_mode == "Hybrid")
                # ----------------------------------------------------------------
                if mark_mode == "Hybrid":
                    # Sample seed from weighted distribution
                    if _seed_batch_ys is None or _seed_batch_idx >= len(_seed_batch_ys):
                        _hrem = max(1, pass_target - pass_segments)
                        _hbatch = min(5000, max(2000, _hrem * 2))
                        _seed_batch_ys, _seed_batch_xs = self._sample_seed_batch(
                            residual_dark, edge_normalized, coverage,
                            img_h, img_w, _hbatch,
                            dark_power=profile["dark_power"],
                            edge_bias=edge_power / 100.0,
                            max_pixel_coverage=max_pixel_coverage,
                            min_darkness=_min_dark_threshold,
                            rng=rng,
                        )
                        _seed_batch_idx = 0
                    _hseed_y = int(_seed_batch_ys[_seed_batch_idx])
                    _hseed_x = int(_seed_batch_xs[_seed_batch_idx])
                    _seed_batch_idx += 1

                    if int(lightened[_hseed_y, _hseed_x]) > pass_ceiling:
                        consecutive_failures += 1
                        continue

                    # Local image properties at seed
                    _hld = float(residual_dark[_hseed_y, _hseed_x])
                    _hle = float(edge_normalized[_hseed_y, _hseed_x])
                    _hgx = float(_gx[_hseed_y, _hseed_x])
                    _hgy = float(_gy[_hseed_y, _hseed_x])
                    _hgrad = _math.atan2(-_hgx, _hgy)          # gradient direction
                    _hnorm = _hgrad - _math.pi / 2.0            # gradient normal
                    _hbase = pass_line_length * (1.4 + _hld * 2.2)  # base mark length

                    hybrid_candidates: list[tuple[float, list[tuple[float, float]]]] = []

                    # --- Candidate: squiggle (traced on scratch lightened copy) ---
                    if _lightened_scratch is None:
                        _lightened_scratch = np.empty_like(lightened)
                    np.copyto(_lightened_scratch, lightened)
                    _sq_path: list[tuple[float, float]] = [(float(_hseed_x), float(_hseed_y))]
                    _sq_cx, _sq_cy = _hseed_x, _hseed_y
                    for _sq_i in range(pass_max_squiggle):
                        _sq_ld = float(orig_dark[int(_sq_cy), int(_sq_cx)])
                        _sq_ell = int(round(pass_line_length * (0.85 + _sq_ld * 1.15)))
                        if long_line_bias > 0.0:
                            _sq_lp = long_line_bias * (0.30 + 0.70 * _sq_ld)
                            if np.random.random() < _sq_lp:
                                _sq_ell = int(round(_sq_ell * np.random.uniform(1.8, 3.8)))
                        _sq_ell = max(1, min(_sq_ell, max(1, min(img_w, img_h) // 3 - 1)))
                        _sq_res = self._find_darkest_line(
                            _lightened_scratch, int(_sq_cx), int(_sq_cy),
                            angle_tests, _sq_ell,
                            residual_dark=residual_dark, edge_normalized=edge_normalized,
                            coverage=coverage, max_pixel_coverage=max_pixel_coverage,
                        )
                        if _sq_res is None:
                            break
                        _sq_ex, _sq_ey, _sq_br, _sq_sc = _sq_res
                        if _sq_br * 100.0 / 255.0 > squiggle_max_deviation or _sq_sc < 0.01:
                            break
                        _sq_path.append((float(_sq_ex), float(_sq_ey)))
                        _sq_seg = [(float(_bx), float(_by)) for _bx, _by in self._bresenham_line(
                            int(round(_sq_cx)), int(round(_sq_cy)), _sq_ex, _sq_ey)]
                        self._erase_along_path(_lightened_scratch, _sq_seg, erase_min, erase_max,
                                               erase_radius_min, erase_radius_max, tone)
                        _sq_cx, _sq_cy = _sq_ex, _sq_ey
                    if len(_sq_path) >= 2:
                        _sq_sc2 = self._score_path(_sq_path, residual_dark, edge_normalized,
                                                   coverage, max_pixel_coverage, img_w, img_h)
                        hybrid_candidates.append((_sq_sc2 + 0.03, _sq_path))

                    # --- Candidates: straight lines (4 angles × 2 length scales) ---
                    for _ha in (_hgrad, _hnorm, 0.0, _math.pi / 2.0):
                        for _hs in (2.3, 3.6):
                            _hl = _hbase * _hs
                            _hlex = int(round(max(0.0, min(float(img_w - 1), _hseed_x + _math.cos(_ha) * _hl))))
                            _hley = int(round(max(0.0, min(float(img_h - 1), _hseed_y + _math.sin(_ha) * _hl))))
                            _hlpts: list[tuple[float, float]] = [
                                (float(_hseed_x), float(_hseed_y)), (float(_hlex), float(_hley))
                            ]
                            _hls = self._score_path(_hlpts, residual_dark, edge_normalized,
                                                    coverage, max_pixel_coverage, img_w, img_h)
                            hybrid_candidates.append((_hls + 0.12 * _hle, _hlpts))

                    # --- Candidates: hatch marks (±45° from gradient normal) ---
                    for _ha in (_hnorm + _math.pi / 4.0, _hnorm - _math.pi / 4.0):
                        _hl = _hbase * 1.8
                        _hhx0 = max(0.0, min(float(img_w - 1), _hseed_x - _math.cos(_ha) * _hl * 0.5))
                        _hhy0 = max(0.0, min(float(img_h - 1), _hseed_y - _math.sin(_ha) * _hl * 0.5))
                        _hhx1 = max(0.0, min(float(img_w - 1), _hseed_x + _math.cos(_ha) * _hl * 0.5))
                        _hhy1 = max(0.0, min(float(img_h - 1), _hseed_y + _math.sin(_ha) * _hl * 0.5))
                        _hhpts: list[tuple[float, float]] = [(_hhx0, _hhy0), (_hhx1, _hhy1)]
                        _hhs = self._score_path(_hhpts, residual_dark, edge_normalized,
                                                coverage, max_pixel_coverage, img_w, img_h)
                        hybrid_candidates.append((_hhs + 0.08 * _hle, _hhpts))

                    # --- Candidate: circle dot (only when local_edge < 0.38) ---
                    if _hle < 0.38:
                        _hcr = pass_line_length * (0.7 + _hld * 1.3)
                        _hcpts = self._circle_mark_pixels(
                            float(_hseed_x), float(_hseed_y), _hcr, 10, img_w, img_h)
                        _hcs = self._score_path(_hcpts, residual_dark, edge_normalized,
                                                coverage, max_pixel_coverage, img_w, img_h)
                        hybrid_candidates.append((_hcs + 0.10 * (1.0 - _hle), _hcpts))

                    if not hybrid_candidates:
                        self._erase_along_path(
                            lightened, [(float(_hseed_x), float(_hseed_y))],
                            erase_min, erase_max, erase_radius_min, erase_radius_max, tone)
                        consecutive_failures += 1
                        continue

                    _, _hbest_pts = max(hybrid_candidates, key=lambda _c: _c[0])
                    _hbest_n = len(_hbest_pts) - 1

                    # Coverage overlap check
                    _hpx_arr, _hpy_arr = self._path_pixel_coords(_hbest_pts, img_w, img_h)
                    if (_hpx_arr.size > 0
                            and np.mean(coverage[_hpy_arr, _hpx_arr] >= max_pixel_coverage) > max_overlap_ratio):
                        self._erase_along_path(
                            lightened, [(float(_hseed_x), float(_hseed_y))],
                            erase_min, erase_max, erase_radius_min, erase_radius_max, tone)
                        consecutive_failures += 1
                        continue

                    # Accept: apply erasure and update coverage/residual maps
                    consecutive_failures = 0
                    self._erase_along_path(lightened, list(_hbest_pts), erase_min, erase_max,
                                           erase_radius_min, erase_radius_max, tone)
                    _hex_arr, _hey_arr = self._expand_pixel_coords(
                        _hpx_arr, _hpy_arr, img_w, img_h, coverage_radius)
                    if _hex_arr.size > 0:
                        coverage[_hey_arr, _hex_arr] = np.minimum(
                            coverage[_hey_arr, _hex_arr].astype(np.int32) + 1, 255
                        ).astype(np.uint8)
                        residual_dark[_hey_arr, _hex_arr] = np.maximum(
                            residual_dark[_hey_arr, _hex_arr] - pass_lighten, 0.0)

                    if _hbest_n >= 1:
                        _hmm_path: Polyline = [
                            (rect_x1 + _hpx * rect_w / img_w, rect_y1 + _hpy * rect_h / img_h)
                            for _hpx, _hpy in _hbest_pts
                        ]
                        result.append(_hmm_path)
                        total_segments += _hbest_n
                        pass_segments += _hbest_n

                    if progress_callback:
                        pct = int(total_segments / max(1, line_max_limit) * 100)
                        progress_callback(min(99, max(0, pct)))
                    continue  # skip squiggle-only logic below

                # ----------------------------------------------------------------
                # Squiggle-only mode (mark_mode != "Hybrid")
                # ----------------------------------------------------------------
                was_chaining = chaining

                if not chaining:
                    # (1) Sample seed from weighted probability distribution
                    if _seed_batch_ys is None or _seed_batch_idx >= len(_seed_batch_ys):
                        remaining = max(1, pass_target - pass_segments)
                        batch_size = min(5000, max(2000, remaining * 2))
                        _seed_batch_ys, _seed_batch_xs = self._sample_seed_batch(
                            residual_dark, edge_normalized, coverage,
                            img_h, img_w, batch_size,
                            dark_power=profile["dark_power"],
                            edge_bias=edge_power / 100.0,
                            max_pixel_coverage=max_pixel_coverage,
                            min_darkness=_min_dark_threshold,
                            rng=rng,
                        )
                        _seed_batch_idx = 0
                    seed_y = int(_seed_batch_ys[_seed_batch_idx])
                    seed_x = int(_seed_batch_xs[_seed_batch_idx])
                    _seed_batch_idx += 1

                    # Check if seed is too bright to start from
                    seed_val = int(lightened[seed_y, seed_x])
                    if seed_val > pass_ceiling:
                        consecutive_failures += 1
                        continue

                    cur_x, cur_y = seed_x, seed_y
                    current_chain = [(float(seed_x), float(seed_y))]
                    chain_seg_count = 0
                    prev_angle = None  # no direction memory for a new stroke

                # (2) Inner squiggle / chain-extension loop
                squiggle_segs = 0
                if continuous:
                    max_segs_this = min(pass_max_squiggle, max(1, chain_max - chain_seg_count))
                else:
                    max_segs_this = pass_max_squiggle

                chain_start_len = len(current_chain)

                for _ in range(max_segs_this):
                    # Variable step length by original darkness (not residual, so it's
                    # independent of coverage/lighten settings and stays constant per pixel)
                    local_dark = float(orig_dark[int(cur_y), int(cur_x)])
                    effective_length = int(round(pass_line_length * (0.85 + local_dark * 1.15)))
                    if long_line_bias > 0.0:
                        long_prob = long_line_bias * (0.30 + 0.70 * local_dark)
                        if np.random.random() < long_prob:
                            effective_length = int(round(effective_length * np.random.uniform(1.8, 3.8)))
                    # Cap so candidates stay within image bounds.
                    # Using //3 - 1 ensures 64×64 images cap at 20 (≈ pass_line_length default),
                    # preserving original behaviour while giving room for bonus on larger images.
                    max_effective = max(1, min(img_w, img_h) // 3 - 1)
                    effective_length = max(1, min(effective_length, max_effective))

                    # Direction mode selection based on local edge strength
                    local_edge = float(edge_normalized[int(cur_y), int(cur_x)])
                    # Contour direction: perpendicular to brightness gradient
                    _gx_val = float(_gx[int(cur_y), int(cur_x)])
                    _gy_val = float(_gy[int(cur_y), int(cur_x)])
                    gradient_angle = float(np.arctan2(-_gx_val, _gy_val))

                    _base_angle: float | None = None
                    _offsets: list[float] | None = None

                    if prev_angle is not None:
                        if np.random.random() < straight_bias * (0.35 + 0.65 * local_edge):
                            _base_angle = 0.88 * prev_angle + 0.12 * gradient_angle
                            _offsets = _STRAIGHT_OFFSETS
                        elif local_edge > 0.08 and np.random.random() < (0.30 + 0.50 * local_edge):
                            _base_angle = 0.70 * prev_angle + 0.30 * gradient_angle
                            _offsets = _CURVE_OFFSETS
                        elif local_edge < 0.22 and np.random.random() < (0.20 + 0.35 * (1.0 - local_edge)):
                            _base_angle = 0.64 * prev_angle + 0.36 * gradient_angle
                            _offsets = _AXIAL_OFFSETS
                        else:
                            _base_angle = 0.64 * prev_angle + 0.36 * gradient_angle
                            # _offsets stays None → fall back to full angle_tests

                    line_result = self._find_darkest_line(
                        lightened, cur_x, cur_y,
                        angle_tests, effective_length,
                        residual_dark=residual_dark,
                        edge_normalized=edge_normalized,
                        coverage=coverage,
                        max_pixel_coverage=max_pixel_coverage,
                        base_angle=_base_angle,
                        angle_offsets=_offsets,
                    )
                    # Fallback: if direction-constrained candidates all scored poorly
                    # OR the best one leads into a bright area (would terminate squiggle),
                    # retry with full angle_tests to avoid premature squiggle termination
                    if _offsets is not None and (
                        line_result is None
                        or line_result[3] < 0.01
                        or line_result[2] * 100.0 / 255.0 > squiggle_max_deviation
                    ):
                        line_result = self._find_darkest_line(
                            lightened, cur_x, cur_y,
                            angle_tests, effective_length,
                            residual_dark=residual_dark,
                            edge_normalized=edge_normalized,
                            coverage=coverage,
                            max_pixel_coverage=max_pixel_coverage,
                        )
                    if line_result is None:
                        break

                    end_x, end_y, avg_brightness, best_score = line_result

                    # Stop if segment ventures too far into bright areas
                    if avg_brightness * 100.0 / 255.0 > squiggle_max_deviation:
                        break
                    # Stop if best candidate score is too low (continuous mode only)
                    if continuous and best_score < 0.01:
                        break

                    current_chain.append((float(end_x), float(end_y)))
                    squiggle_segs += 1
                    prev_angle = float(np.arctan2(end_y - cur_y, end_x - cur_x))

                    # Erase along this segment immediately
                    seg_pixels = [
                        (float(bx), float(by))
                        for bx, by in self._bresenham_line(
                            int(round(cur_x)), int(round(cur_y)),
                            end_x, end_y,
                        )
                    ]
                    self._erase_along_path(
                        lightened, seg_pixels,
                        erase_min, erase_max,
                        erase_radius_min, erase_radius_max, tone,
                    )
                    cur_x, cur_y = end_x, end_y

                # (3) After inner loop: accept / reject / chain / finalize
                if squiggle_segs == 0:
                    # No segments traced
                    if not was_chaining:
                        # Erase at seed to avoid re-selection
                        self._erase_along_path(
                            lightened, [(float(seed_x), float(seed_y))],
                            erase_min, erase_max,
                            erase_radius_min, erase_radius_max, tone,
                        )
                    consecutive_failures += 1
                    chaining = False
                    if continuous:
                        # Finalize any accumulated chain
                        n_segs = len(current_chain) - 1
                        if n_segs >= squiggle_min_length:
                            mm_path: Polyline = [
                                (rect_x1 + px_x * rect_w / img_w, rect_y1 + px_y * rect_h / img_h)
                                for px_x, px_y in current_chain
                            ]
                            result.append(mm_path)
                            total_segments += n_segs
                            pass_segments += n_segs
                    current_chain = []
                    chain_seg_count = 0

                elif not continuous:
                    # Original behavior: squiggle = current_chain
                    consecutive_failures = 0
                    squiggle = current_chain

                    if squiggle_segs >= squiggle_min_length:
                        px_arr, py_arr = self._path_pixel_coords(squiggle, img_w, img_h)
                        if px_arr.size > 0 and np.mean(coverage[py_arr, px_arr] >= max_pixel_coverage) > max_overlap_ratio:
                            # Reject: over-inked region
                            consecutive_failures += 1
                        else:
                            # Accept: update coverage map and residual_dark
                            ex_arr, ey_arr = self._expand_pixel_coords(px_arr, py_arr, img_w, img_h, coverage_radius)
                            if ex_arr.size > 0:
                                coverage[ey_arr, ex_arr] = np.minimum(
                                    coverage[ey_arr, ex_arr].astype(np.int32) + 1, 255
                                ).astype(np.uint8)
                                residual_dark[ey_arr, ex_arr] = np.maximum(
                                    residual_dark[ey_arr, ex_arr] - pass_lighten, 0.0
                                )
                            mm_path = [
                                (rect_x1 + px_x * rect_w / img_w, rect_y1 + px_y * rect_h / img_h)
                                for px_x, px_y in squiggle
                            ]
                            result.append(mm_path)
                            total_segments += squiggle_segs
                            pass_segments += squiggle_segs

                    current_chain = []
                    chain_seg_count = 0
                    chaining = False

                else:
                    # continuous mode, squiggle_segs > 0
                    consecutive_failures = 0

                    # Coverage overlap check for newly added segment(s)
                    new_seg_pts = current_chain[chain_start_len - 1:]  # include start of new segs
                    px_arr, py_arr = self._path_pixel_coords(new_seg_pts, img_w, img_h)

                    if px_arr.size > 0 and np.mean(coverage[py_arr, px_arr] >= max_pixel_coverage) > max_overlap_ratio:
                        # Over-inked: reject new segments, finalize accumulated chain
                        current_chain = current_chain[:chain_start_len]
                        consecutive_failures += 1
                        chaining = False
                        n_segs = len(current_chain) - 1
                        if n_segs >= squiggle_min_length:
                            mm_path = [
                                (rect_x1 + px_x * rect_w / img_w, rect_y1 + px_y * rect_h / img_h)
                                for px_x, px_y in current_chain
                            ]
                            result.append(mm_path)
                            total_segments += n_segs
                            pass_segments += n_segs
                        current_chain = []
                        chain_seg_count = 0
                    else:
                        # Accept: update coverage map and residual_dark for new segments
                        ex_arr, ey_arr = self._expand_pixel_coords(px_arr, py_arr, img_w, img_h, coverage_radius)
                        if ex_arr.size > 0:
                            coverage[ey_arr, ex_arr] = np.minimum(
                                coverage[ey_arr, ex_arr].astype(np.int32) + 1, 255
                            ).astype(np.uint8)
                            residual_dark[ey_arr, ex_arr] = np.maximum(
                                residual_dark[ey_arr, ex_arr] - pass_lighten, 0.0
                            )

                        chain_seg_count += squiggle_segs

                        # Check if endpoint is still viable for continuing
                        end_dark = float(residual_dark[int(cur_y), int(cur_x)])
                        end_cov = int(coverage[int(cur_y), int(cur_x)])
                        force_new_seed = chain_seg_count >= chain_max

                        if not force_new_seed and end_dark > _min_dark_threshold and end_cov < max_pixel_coverage:
                            # Continue chaining from endpoint — no pen lift
                            chaining = True
                        else:
                            # Finalize chain (bright area, over-inked endpoint, or chain_max reached)
                            chaining = False
                            n_segs = len(current_chain) - 1
                            if n_segs >= squiggle_min_length:
                                mm_path = [
                                    (rect_x1 + px_x * rect_w / img_w, rect_y1 + px_y * rect_h / img_h)
                                    for px_x, px_y in current_chain
                                ]
                                result.append(mm_path)
                                total_segments += n_segs
                                pass_segments += n_segs
                            current_chain = []
                            chain_seg_count = 0

                # (E) Report progress across all passes
                if progress_callback:
                    pct = int(total_segments / max(1, line_max_limit) * 100)
                    progress_callback(min(99, max(0, pct)))

            # End of pass: finalize any pending chain (cancelled or budget exhausted)
            if continuous and chaining and current_chain:
                n_segs = len(current_chain) - 1
                if n_segs >= squiggle_min_length:
                    mm_path = [
                        (rect_x1 + px_x * rect_w / img_w, rect_y1 + px_y * rect_h / img_h)
                        for px_x, px_y in current_chain
                    ]
                    result.append(mm_path)
                    total_segments += n_segs
            current_chain = []
            chaining = False

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result
