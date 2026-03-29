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
                max=100.0,
                step=0.1,
                default=50.0,
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
                name="block_size",
                label="Block Size (px)",
                min=4,
                max=64,
                step=1,
                default=8,
                description="Search region size in pixels — smaller = more precise seed placement",
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
            # --- tracing params ---
            IntParam(
                name="line_min_length",
                label="Min Line Length (steps)",
                min=2,
                max=100,
                step=1,
                default=5,
                description="Minimum number of steps a traced path must have to be kept",
            ),
            IntParam(
                name="line_max_length",
                label="Max Line Length (steps)",
                min=10,
                max=500,
                step=1,
                default=150,
                description="Maximum number of steps to trace per path",
            ),
            IntParam(
                name="angle_tests",
                label="Angle Tests",
                min=4,
                max=360,
                step=1,
                default=16,
                description="Candidate directions tested per step — values <36 use evenly-spaced angles; ≥36 tests all integer points on a Bresenham circle (full pixel-resolution coverage)",
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
            IntParam(
                name="brightness_ceiling",
                label="Brightness Ceiling",
                min=200,
                max=255,
                step=1,
                default=250,
                description="Tracing stops when a pixel exceeds this brightness — lower = trace more into bright areas",
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
            "line_density": 50.0,
            "line_max_limit": 10_000,
            "block_size": 8,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 1.0,
            "line_min_length": 5,
            "line_max_length": 150,
            "angle_tests": 16,
            "line_length_px": 20,
            "erase_min": 1,
            "erase_max": 100,
            "erase_radius_min": 1,
            "erase_radius_max": 4,
            "tone": 0.5,
            "directionality": 0.0,
            "edge_power": 0.0,
            "brightness_ceiling": 250,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        return [
            Preset(name="Default", params=dict(_base)),
            Preset(
                name="Sketch Lines",
                params={
                    **_base,
                    "line_max_length": 80,
                },
            ),
            Preset(
                name="Contour Sketch",
                params={
                    **_base,
                    "line_max_length": 120,
                    "erase_max": 120,
                    "directionality": 60.0,
                    "edge_power": 20.0,
                },
            ),
            Preset(
                name="Dense Crosshatch",
                params={
                    **_base,
                    "angle_tests": 4,
                    "line_max_length": 60,
                    "erase_radius_max": 2,
                    "line_density": 80.0,
                },
            ),
            Preset(
                name="Loose Sketch",
                params={
                    **_base,
                    "angle_tests": 12,
                    "line_max_length": 200,
                    "erase_radius_max": 6,
                    "erase_max": 80,
                    "line_density": 30.0,
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
                img = img.mean(axis=2).astype(np.uint8)

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

        block_size = int(params.get("block_size", 8))
        block_size = max(1, block_size)

        # --- generation parameters ---
        line_density = float(params.get("line_density", 50.0))
        line_max_limit = int(params.get("line_max_limit", 10_000))
        line_min_length = int(params.get("line_min_length", 5))
        line_max_length = int(params.get("line_max_length", 150))
        angle_tests = int(params.get("angle_tests", 16))
        line_length_px = int(params.get("line_length_px", 20))
        erase_min = int(params.get("erase_min", 1))
        erase_max = int(params.get("erase_max", 100))
        erase_radius_min = int(params.get("erase_radius_min", 1))
        erase_radius_max = int(params.get("erase_radius_max", 4))
        tone = float(params.get("tone", 0.5))
        directionality = float(params.get("directionality", 0.0))
        edge_power = float(params.get("edge_power", 0.0))
        brightness_ceiling = int(params.get("brightness_ceiling", 250))

        # --- running brightness sum (avoids full-image mean() each iteration) ---
        total_pixels = img_h * img_w
        running_sum = float(lightened.sum())

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

        # --- brightness targets ---
        initial_avg_brightness = running_sum / total_pixels
        # target_brightness: how bright the lightened image must become before
        # we stop.  Higher line_density → brighter target → more lines drawn.
        target_brightness = initial_avg_brightness + (255.0 - initial_avg_brightness) * (line_density / 100.0) ** 0.5

        result: list[Polyline] = []
        total_segments = 0
        _iter_count = 0
        current_avg = initial_avg_brightness

        if target_brightness > initial_avg_brightness:
            while total_segments < line_max_limit:
                if cancelled_callback and cancelled_callback():
                    break

                # Check brightness target every 50 iterations to avoid per-iteration
                # mean computation overhead on large images.
                if _iter_count % 50 == 0:
                    current_avg = running_sum / total_pixels
                _iter_count += 1

                if current_avg >= target_brightness:
                    break

                # Find darkest region → darkest seed pixel
                br, bc = self._find_darkest_region(lightened, block_size)
                seed_y, seed_x = self._find_darkest_pixel(lightened, br, bc, block_size)

                # Build path by repeatedly finding the best next line segment
                path: list[tuple[float, float]] = [(float(seed_x), float(seed_y))]
                cur_x, cur_y = seed_x, seed_y

                for _ in range(line_max_length - 1):
                    # Stop if current position is too bright
                    iy = max(0, min(img_h - 1, cur_y))
                    ix = max(0, min(img_w - 1, cur_x))
                    if lightened[iy, ix] > brightness_ceiling:
                        break

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
                    if avg_brightness > brightness_ceiling:
                        break
                    path.append((float(end_x), float(end_y)))
                    cur_x, cur_y = end_x, end_y

                # Expand path to individual Bresenham pixels for proper erasing
                erase_pixels: list[tuple[float, float]] = []
                for i in range(len(path)):
                    px_x, px_y = path[i]
                    erase_pixels.append((px_x, px_y))
                    if i + 1 < len(path):
                        nx, ny = path[i + 1]
                        for bx, by in self._bresenham_line(
                            int(round(px_x)), int(round(px_y)),
                            int(round(nx)), int(round(ny)),
                        ):
                            erase_pixels.append((float(bx), float(by)))

                # Erase regardless of path length to prevent infinite loops;
                # accumulate the actual brightness delta into the running sum
                if erase_pixels:
                    delta = self._erase_along_path(
                        lightened, erase_pixels,
                        erase_min, erase_max,
                        erase_radius_min, erase_radius_max, tone,
                    )
                    running_sum += delta

                # Only keep paths that meet minimum length
                if len(path) >= line_min_length:
                    mm_path: Polyline = [
                        (
                            rect_x1 + px_x * rect_w / img_w,
                            rect_y1 + px_y * rect_h / img_h,
                        )
                        for px_x, px_y in path
                    ]
                    result.append(mm_path)
                    total_segments += len(path) - 1

                # Update progress
                if progress_callback:
                    denom = target_brightness - initial_avg_brightness
                    if denom > 0:
                        pct = int(100.0 * (current_avg - initial_avg_brightness) / denom)
                    else:
                        pct = 100
                    progress_callback(min(99, max(0, pct)))

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result
