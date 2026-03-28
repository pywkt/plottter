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
        return [
            Preset(
                name="Default",
                params={
                    "line_density": 50.0,
                    "line_max_limit": 10000,
                    "block_size": 16,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
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

        best_row, best_col = 0, 0
        best_mean = 256.0  # above max possible value

        for br in range(n_rows):
            r0 = br * block_size
            r1 = min(r0 + block_size, h)
            for bc in range(n_cols):
                c0 = bc * block_size
                c1 = min(c0 + block_size, w)
                block = lightened[r0:r1, c0:c1]
                mean_val = float(block.mean())
                if mean_val < best_mean:
                    best_mean = mean_val
                    best_row, best_col = br, bc

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

        # Validate darkest-area finder (raises if image is empty)
        br, bc = self._find_darkest_region(lightened, block_size)
        _px_y, _px_x = self._find_darkest_pixel(lightened, br, bc, block_size)

        # Future phases will use img_rect, br/bc, px_y/px_x to build strokes.
        # For now, return empty polylines.
        result: list[Polyline] = []

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result
