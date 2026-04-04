"""Sketch generator — darkest-area finder scaffold.

Phase 1: implements the core darkest-region / darkest-pixel search helpers and
the generator scaffold.  The generate() method loads + preprocesses the source
image and validates the helpers, but returns empty polylines for now.
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
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        return [
            Preset(name="Default", params=dict(_base)),
            Preset(
                name="Sketch Lines",
                params={
                    **_base,
                    "line_density": 0.8,
                    "angle_tests": 16,
                    "squiggle_max_length": 30,
                },
            ),
            Preset(
                name="Contour Sketch",
                params={
                    **_base,
                    "directionality": 60.0,
                    "edge_power": 20.0,
                    "squiggle_max_length": 80,
                },
            ),
            Preset(
                name="Dense Crosshatch",
                params={
                    **_base,
                    "line_density": 2.0,
                    "angle_tests": 4,
                    "line_length_px": 15,
                    "squiggle_max_length": 20,
                },
            ),
            Preset(
                name="Loose Sketch",
                params={
                    **_base,
                    "line_density": 0.5,
                    "squiggle_max_length": 100,
                    "squiggle_max_deviation": 50.0,
                },
            ),
            Preset(
                name="Edge Trace",
                params={
                    **_base,
                    "directionality": 30.0,
                    "edge_power": 60.0,
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Darkest-area helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_darkest_region(
        lightened: np.ndarray,
        block_size: int,
    ) -> tuple[int, int]:
        """Divide image into blocks and return the (row, col) of the darkest block.

        Parameters
        ----------
        lightened:
            2-D uint8 grayscale image (0=black, 255=white).  The array is
            treated as-is — no copy is made.
        block_size:
            Side length of each square block in pixels.

        Returns
        -------
        (block_row, block_col) — zero-based index of the block with the lowest
        mean brightness.  Both values are guaranteed to be within the valid
        range of blocks for the given image size.
        """
        h, w = lightened.shape[:2]
        block_size = max(1, block_size)

        n_rows = max(1, (h + block_size - 1) // block_size)
        n_cols = max(1, (w + block_size - 1) // block_size)

        # Pad to exact block multiples so cv2 / numpy block boundaries align with
        # _find_darkest_pixel which uses region_y * block_size offsets.
        pad_h = n_rows * block_size - h
        pad_w = n_cols * block_size - w
        padded = (
            np.pad(lightened, ((0, pad_h), (0, pad_w)), mode="edge")
            if (pad_h or pad_w)
            else lightened
        )

        try:
            import cv2
            # INTER_AREA on a padded image whose size == n_rows*block_size × n_cols*block_size
            # gives exactly the mean of each block_size×block_size tile.
            block_means = cv2.resize(
                padded, (n_cols, n_rows), interpolation=cv2.INTER_AREA
            )
        except ImportError:
            block_means = padded.reshape(
                n_rows, block_size, n_cols, block_size
            ).mean(axis=(1, 3))

        flat_idx = int(np.argmin(block_means))
        best_row = flat_idx // n_cols
        best_col = flat_idx % n_cols
        return best_row, best_col

    @staticmethod
    def _find_darkest_pixel(
        lightened: np.ndarray,
        region_y: int,
        region_x: int,
        block_size: int,
    ) -> tuple[int, int]:
        """Within a specific block, find the single darkest pixel.

        Parameters
        ----------
        lightened:
            2-D uint8 grayscale image.
        region_y:
            Block row index (as returned by _find_darkest_region).
        region_x:
            Block column index (as returned by _find_darkest_region).
        block_size:
            Side length of the block in pixels.

        Returns
        -------
        (px_y, px_x) — absolute pixel coordinates of the darkest pixel within
        the block.
        """
        h, w = lightened.shape[:2]
        block_size = max(1, block_size)

        r0 = region_y * block_size
        r1 = min(r0 + block_size, h)
        c0 = region_x * block_size
        c1 = min(c0 + block_size, w)

        block = lightened[r0:r1, c0:c1]
        flat_idx = int(np.argmin(block))
        local_y = flat_idx // block.shape[1]
        local_x = flat_idx % block.shape[1]

        return r0 + local_y, c0 + local_x

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
        sobel_x: np.ndarray | None = None,
        sobel_y: np.ndarray | None = None,
        edge_map: np.ndarray | None = None,
        directionality: float = 0.0,
        edge_power: float = 0.0,
    ) -> tuple[int, int, float] | None:
        """Find the single best line segment from ``(cur_x, cur_y)``.

        Tests candidate endpoints at distance ``line_length_px`` from the
        current position and returns the one whose full Bresenham line has
        the lowest average brightness.

        Candidate generation:
        - If ``angle_tests < 36``: test ``angle_tests`` evenly-spaced angles.
        - If ``angle_tests >= 36``: test all integer points on the Bresenham
          circle of radius ``line_length_px`` (full pixel-resolution coverage).

        Scoring blends three components (same weights as the legacy tracer):
        - **luminance**: average brightness along the Bresenham line (lower = darker).
        - **directionality**: alignment with local iso-brightness contour tangent.
        - **edge_power**: average edge density along the line.

        Parameters
        ----------
        lightened:
            2-D uint8 grayscale image (0=black, 255=white).
        cur_x, cur_y:
            Current pixel position (column, row).
        angle_tests:
            Number of candidate angles (< 36) or Bresenham-circle mode (>= 36).
        line_length_px:
            Radius / length of each candidate line segment in pixels.
        sobel_x, sobel_y:
            Float32 Sobel gradient arrays. Required when *directionality* > 0.
        edge_map:
            uint8 binary edge map (0 or 255). Required when *edge_power* > 0.
        directionality, edge_power:
            Blend weights (0–100) for contour-following / edge-attraction.

        Returns
        -------
        ``(end_x, end_y, avg_brightness)`` for the winning endpoint, or
        ``None`` if no valid candidate exists.
        """
        h, w = lightened.shape[:2]

        # Normalised blending weights
        lum_power = max(0.0, 100.0 - directionality - edge_power)
        total_power = lum_power + directionality + edge_power
        if total_power < 1e-9:
            lum_w, dir_w, edge_w = 1.0, 0.0, 0.0
        else:
            lum_w = lum_power / total_power
            dir_w = directionality / total_power
            edge_w = edge_power / total_power

        use_dir = dir_w > 1e-6 and sobel_x is not None and sobel_y is not None
        use_edge = edge_w > 1e-6 and edge_map is not None

        # Gradient at current position for directionality scoring
        gx_cur = gy_cur = grad_mag = 0.0
        if use_dir:
            iy_cur = max(0, min(h - 1, cur_y))
            ix_cur = max(0, min(w - 1, cur_x))
            gx_cur = float(sobel_x[iy_cur, ix_cur])
            gy_cur = float(sobel_y[iy_cur, ix_cur])
            grad_mag = (gx_cur ** 2 + gy_cur ** 2) ** 0.5

        # Generate candidate endpoints
        r = max(1, line_length_px)
        candidates: list[tuple[int, int]] = []

        if angle_tests >= 36:
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

        best_score = float("inf")
        best_end: tuple[int, int] | None = None
        best_avg_brightness = float("inf")

        for ex, ey in candidates:
            # Skip endpoints outside image bounds
            if not (0 <= ey < h and 0 <= ex < w):
                continue

            # Trace Bresenham line; accumulate brightness (and optional edge) sum
            pixel_sum = 0
            pixel_count = 0
            edge_sum = 0
            valid = True

            for px, py in SketchGenerator._bresenham_line(cur_x, cur_y, ex, ey):
                if not (0 <= py < h and 0 <= px < w):
                    valid = False
                    break
                pixel_sum += int(lightened[py, px])
                pixel_count += 1
                if use_edge:
                    edge_sum += int(edge_map[py, px])

            if not valid or pixel_count == 0:
                continue

            avg_brightness = pixel_sum / pixel_count
            score = lum_w * (avg_brightness / 255.0)

            if use_dir and grad_mag > 1e-6:
                ddx = ex - cur_x
                ddy = ey - cur_y
                seg_mag = (ddx ** 2 + ddy ** 2) ** 0.5
                if seg_mag > 1e-6:
                    # Alignment with contour tangent (-gy, gx)
                    alignment = abs(-gy_cur * (ddx / seg_mag) + gx_cur * (ddy / seg_mag)) / grad_mag
                    score -= dir_w * alignment

            if use_edge:
                score -= edge_w * (edge_sum / pixel_count / 255.0)

            if score < best_score:
                best_score = score
                best_end = (ex, ey)
                best_avg_brightness = avg_brightness

        if best_end is None:
            return None
        return (best_end[0], best_end[1], best_avg_brightness)

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

        # Internal constants (not exposed as parameters)
        block_size = 8
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

        # --- residual darkness tracking for stopping condition ---
        total_pixels = img_h * img_w
        initial_residual_mean = float(residual_dark.mean())
        # stop_denom_r: higher line_density → larger denominator → harder to stop → more lines
        stop_denom_r = initial_residual_mean * max(1e-9, line_density)

        # --- precompute gradient and edge maps ---
        sobel_x_map: np.ndarray | None = None
        sobel_y_map: np.ndarray | None = None
        edge_map: np.ndarray | None = None

        if directionality > 0.0:
            try:
                import cv2
                sobel_x_map = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y_map = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            except ImportError:
                gy_np, gx_np = np.gradient(img.astype(np.float32))
                sobel_x_map = gx_np
                sobel_y_map = gy_np

        if edge_power > 0.0:
            try:
                import cv2
                edge_map = cv2.Canny(img, 50, 150)
            except ImportError:
                pass  # Canny unavailable without OpenCV; edge_power has no effect

        # --- coordinate mapping: pixel → mm ---
        rect_x1, rect_y1, rect_x2, rect_y2 = img_rect
        rect_w = rect_x2 - rect_x1
        rect_h = rect_y2 - rect_y1

        result: list[Polyline] = []
        total_segments = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 1000

        while True:
            if cancelled_callback and cancelled_callback():
                break

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                break

            current_residual_mean = float(residual_dark.mean())
            lum_progress = (initial_residual_mean - current_residual_mean) / max(1e-9, stop_denom_r)
            if lum_progress >= 1.0 or total_segments >= line_max_limit:
                break

            # (1) Find darkest tile → darkest seed pixel
            br, bc = self._find_darkest_region(lightened, block_size)
            seed_y, seed_x = self._find_darkest_pixel(lightened, br, bc, block_size)

            # Check if seed is too bright to start a squiggle from
            seed_val = int(lightened[seed_y, seed_x])
            if seed_val > brightness_ceiling:
                lightened[seed_y, seed_x] = 255
                consecutive_failures += 1
                continue

            # (2) Start a new squiggle from seed
            squiggle: list[tuple[float, float]] = [(float(seed_x), float(seed_y))]
            cur_x, cur_y = seed_x, seed_y
            squiggle_segments = 0

            # (3) Inner squiggle loop: chain segments
            for _ in range(squiggle_max_length):
                line_result = self._find_darkest_line(
                    lightened, cur_x, cur_y,
                    angle_tests, line_length_px,
                    sobel_x=sobel_x_map,
                    sobel_y=sobel_y_map,
                    edge_map=edge_map,
                    directionality=directionality,
                    edge_power=edge_power,
                )
                if line_result is None:
                    break

                end_x, end_y, avg_brightness = line_result

                # Stop squiggle if segment ventures too far into bright areas
                if avg_brightness * 100.0 / 255.0 > squiggle_max_deviation:
                    break

                squiggle.append((float(end_x), float(end_y)))
                squiggle_segments += 1

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

            if squiggle_segments == 0:
                # No segments traced — erase at seed to avoid re-selection
                self._erase_along_path(
                    lightened, [(float(seed_x), float(seed_y))],
                    erase_min, erase_max,
                    erase_radius_min, erase_radius_max, tone,
                )
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                # (4) Add squiggle polyline to output if it meets minimum length
                if squiggle_segments >= squiggle_min_length:
                    # Coverage overlap check — reject paths where too many pixels are already inked
                    px_arr, py_arr = self._path_pixel_coords(squiggle, img_w, img_h)
                    if px_arr.size > 0 and np.mean(coverage[py_arr, px_arr] >= max_pixel_coverage) > max_overlap_ratio:
                        # Reject: over-inked region — count as failure to avoid infinite loops
                        consecutive_failures += 1
                    else:
                        # Accept: update coverage map and residual_dark
                        ex_arr, ey_arr = self._expand_pixel_coords(px_arr, py_arr, img_w, img_h, coverage_radius)
                        if ex_arr.size > 0:
                            coverage[ey_arr, ex_arr] = np.minimum(
                                coverage[ey_arr, ex_arr].astype(np.int32) + 1, 255
                            ).astype(np.uint8)
                            residual_dark[ey_arr, ex_arr] = np.maximum(
                                residual_dark[ey_arr, ex_arr] - lighten_amount, 0.0
                            )
                        mm_path: Polyline = [
                            (
                                rect_x1 + px_x * rect_w / img_w,
                                rect_y1 + px_y * rect_h / img_h,
                            )
                            for px_x, px_y in squiggle
                        ]
                        result.append(mm_path)
                        total_segments += squiggle_segments

            # (E) Report progress as min(line_progress, lum_progress)
            if progress_callback:
                line_progress = total_segments / max(1, line_max_limit)
                pct = int(min(line_progress, lum_progress) * 100)
                progress_callback(min(99, max(0, pct)))

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result
