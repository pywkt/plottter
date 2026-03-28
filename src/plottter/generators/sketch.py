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
                default=16,
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
                default=100,
                description="Maximum number of steps to trace per path",
            ),
            IntParam(
                name="angle_tests",
                label="Angle Tests",
                min=4,
                max=36,
                step=1,
                default=8,
                description="Number of directions to test at each step — 4 = square grid, 8 = octagonal, higher = smoother",
            ),
            IntParam(
                name="step_size_px",
                label="Step Size (px)",
                min=1,
                max=10,
                step=1,
                default=2,
                description="Pixel distance to advance per tracing step",
            ),
            # --- erase params ---
            IntParam(
                name="erase_radius",
                label="Erase Radius (px)",
                min=1,
                max=20,
                step=1,
                default=3,
                description="Radius of erased area around each drawn line — larger = sparser result",
            ),
            IntParam(
                name="erase_amount",
                label="Erase Amount",
                min=10,
                max=200,
                step=1,
                default=50,
                description="How much to brighten drawn areas — higher = lines spread out faster",
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
            "block_size": 16,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 1.0,
            "line_min_length": 5,
            "line_max_length": 100,
            "angle_tests": 8,
            "step_size_px": 2,
            "erase_radius": 3,
            "erase_amount": 50,
            "directionality": 0.0,
            "edge_power": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        return [
            Preset(name="Default", params=dict(_base)),
            Preset(
                name="Sketch Lines",
                params={
                    **_base,
                    "angle_tests": 8,
                    "line_max_length": 80,
                    "erase_radius": 3,
                    "erase_amount": 50,
                    "directionality": 0.0,
                    "edge_power": 0.0,
                },
            ),
            Preset(
                name="Contour Sketch",
                params={
                    **_base,
                    "angle_tests": 16,
                    "line_max_length": 120,
                    "erase_radius": 4,
                    "erase_amount": 40,
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
                    "erase_radius": 2,
                    "erase_amount": 30,
                    "line_density": 80.0,
                },
            ),
            Preset(
                name="Loose Sketch",
                params={
                    **_base,
                    "angle_tests": 12,
                    "line_max_length": 200,
                    "erase_radius": 6,
                    "erase_amount": 80,
                    "line_density": 30.0,
                },
            ),
            Preset(
                name="Edge Trace",
                params={
                    **_base,
                    "angle_tests": 16,
                    "line_max_length": 150,
                    "directionality": 30.0,
                    "edge_power": 60.0,
                    "erase_radius": 3,
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
    # Darkest-path tracing
    # ------------------------------------------------------------------

    _BRIGHTNESS_CEILING = 240

    @staticmethod
    def _trace_darkest_path(
        lightened: np.ndarray,
        seed_y: int,
        seed_x: int,
        angle_tests: int,
        max_length: int,
        step_size_px: int,
        sobel_x: np.ndarray | None = None,
        sobel_y: np.ndarray | None = None,
        edge_map: np.ndarray | None = None,
        directionality: float = 0.0,
        edge_power: float = 0.0,
    ) -> list[tuple[float, float]]:
        """Trace a path from (seed_y, seed_x) by greedily choosing the best
        neighbouring direction at each step.

        Direction scoring blends three components:
        - **luminance**: prefers darker pixels (brightness-seeking)
        - **directionality**: prefers directions aligned with the local
          iso-brightness contour tangent (i.e. perpendicular to the image
          gradient), so lines tend to follow edges instead of crossing them.
        - **edge_power**: prefers directions whose target pixel lies on a
          detected Canny edge.

        Parameters
        ----------
        lightened:
            2-D uint8 grayscale image (0=black, 255=white).
        seed_y, seed_x:
            Starting pixel coordinates (row, col).
        angle_tests:
            Number of evenly-spaced directions to evaluate at each step.
        max_length:
            Maximum total number of positions in the returned path (including
            the seed). The tracing loop runs at most ``max_length - 1`` times.
        step_size_px:
            Distance in pixels to advance per step.
        sobel_x, sobel_y:
            Float32 Sobel gradient arrays (same shape as *lightened*).  Required
            when *directionality* > 0; ignored otherwise.
        edge_map:
            uint8 binary edge map (0 or 255, same shape as *lightened*).
            Required when *edge_power* > 0; ignored otherwise.
        directionality:
            Weight (0–100) for contour-following.
        edge_power:
            Weight (0–100) for edge-attraction.

        Returns
        -------
        List of ``(px_x, px_y)`` pixel coordinates.
        """
        h, w = lightened.shape[:2]
        CEIL = SketchGenerator._BRIGHTNESS_CEILING

        # Seed must be inside the image
        if not (0 <= seed_y < h and 0 <= seed_x < w):
            return []

        # Compute normalised blending weights so they sum to 1.
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

        # Pre-compute direction unit vectors (cos, sin) for all test angles
        angles = [i * 2.0 * np.pi / angle_tests for i in range(angle_tests)]
        dirs = [(np.cos(a), np.sin(a)) for a in angles]

        cur_x = float(seed_x)
        cur_y = float(seed_y)
        path: list[tuple[float, float]] = [(cur_x, cur_y)]

        for _ in range(max_length - 1):
            # Stop if current position is too bright
            iy = int(round(cur_y))
            ix = int(round(cur_x))
            if lightened[iy, ix] > CEIL:
                break

            # Sample gradient at current position for directionality scoring.
            # The contour tangent is perpendicular to the gradient: (-gy, gx).
            if use_dir:
                gx_cur = float(sobel_x[iy, ix])
                gy_cur = float(sobel_y[iy, ix])
                grad_mag = (gx_cur * gx_cur + gy_cur * gy_cur) ** 0.5
            else:
                gx_cur = gy_cur = grad_mag = 0.0

            # Test all directions; pick the lowest-score look-ahead.
            # Score = lum_w*(brightness/255) - dir_w*alignment - edge_w*(edge/255)
            best_dx: float = 0.0
            best_dy: float = 0.0
            best_score: float = float("inf")
            found = False

            for dx, dy in dirs:
                nx = cur_x + dx * step_size_px
                ny = cur_y + dy * step_size_px
                iny = int(round(ny))
                inx = int(round(nx))
                if not (0 <= iny < h and 0 <= inx < w):
                    continue

                brightness = float(lightened[iny, inx])
                score = lum_w * (brightness / 255.0)

                if use_dir and grad_mag > 1e-6:
                    # Alignment with contour tangent (-gy, gx):
                    # alignment = |dx*(-gy) + dy*gx| / grad_mag
                    alignment = abs(-gy_cur * dx + gx_cur * dy) / grad_mag
                    score -= dir_w * alignment

                if use_edge:
                    edge_val = float(edge_map[iny, inx]) / 255.0
                    score -= edge_w * edge_val

                if score < best_score:
                    best_score = score
                    best_dx, best_dy = dx, dy
                    found = True

            if not found:
                break  # All look-ahead positions are out of bounds

            # Advance one step
            cur_x += best_dx * step_size_px
            cur_y += best_dy * step_size_px

            # Stop if the new position is outside the image
            iy = int(round(cur_y))
            ix = int(round(cur_x))
            if not (0 <= iy < h and 0 <= ix < w):
                break

            path.append((cur_x, cur_y))

        return path

    # ------------------------------------------------------------------
    # Erase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _erase_along_path(
        lightened: np.ndarray,
        path: list[tuple[float, float]],
        erase_radius: int,
        erase_amount: int,
    ) -> float:
        """Brighten a square region around each point in *path*.

        Modifies *lightened* in-place using numpy slice assignment — faster than
        allocating a scratch buffer and drawing circles for small radii.

        Parameters
        ----------
        lightened:
            2-D uint8 grayscale image that is modified in-place.
        path:
            List of ``(px_x, px_y)`` float pixel coordinates.
        erase_radius:
            Half-side of the square brightening region in pixels.
        erase_amount:
            How much to add to pixel values within the region (0–255).

        Returns
        -------
        Total sum of actual pixel increments (accounting for 255 clamping).
        Use this to maintain an incremental running brightness sum.
        """
        if not path:
            return 0.0

        h, w = lightened.shape[:2]
        r = erase_radius
        total_delta = 0.0

        for px_x, px_y in path:
            cx = int(round(px_x))
            cy = int(round(px_y))
            y0 = max(0, cy - r)
            y1 = min(h, cy + r + 1)
            x0 = max(0, cx - r)
            x1 = min(w, cx + r + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            region = lightened[y0:y1, x0:x1]
            new_region = np.minimum(
                region.astype(np.uint16) + erase_amount, 255
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

        block_size = int(params.get("block_size", 16))
        block_size = max(1, block_size)

        # --- generation parameters ---
        line_density = float(params.get("line_density", 50.0))
        line_max_limit = int(params.get("line_max_limit", 10_000))
        line_min_length = int(params.get("line_min_length", 5))
        line_max_length = int(params.get("line_max_length", 100))
        angle_tests = int(params.get("angle_tests", 8))
        step_size_px = int(params.get("step_size_px", 2))
        erase_radius = int(params.get("erase_radius", 3))
        erase_amount = int(params.get("erase_amount", 50))
        directionality = float(params.get("directionality", 0.0))
        edge_power = float(params.get("edge_power", 0.0))

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
        target_brightness = initial_avg_brightness + (255.0 - initial_avg_brightness) * (line_density / 100.0)

        result: list[Polyline] = []
        total_segments = 0

        if target_brightness > initial_avg_brightness:
            while total_segments < line_max_limit:
                if cancelled_callback and cancelled_callback():
                    break

                current_avg = running_sum / total_pixels
                if current_avg >= target_brightness:
                    break

                # Find darkest region → darkest seed pixel
                br, bc = self._find_darkest_region(lightened, block_size)
                seed_y, seed_x = self._find_darkest_pixel(lightened, br, bc, block_size)

                # Trace path from seed
                path = self._trace_darkest_path(
                    lightened, seed_y, seed_x, angle_tests, line_max_length, step_size_px,
                    sobel_x=sobel_x_map,
                    sobel_y=sobel_y_map,
                    edge_map=edge_map,
                    directionality=directionality,
                    edge_power=edge_power,
                )

                # Erase regardless of path length to prevent infinite loops;
                # accumulate the actual brightness delta into the running sum
                if path:
                    delta = self._erase_along_path(lightened, path, erase_radius, erase_amount)
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
