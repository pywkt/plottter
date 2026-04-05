"""ASCIIArtGenerator — Grid-based ASCII art placement from image brightness."""

from __future__ import annotations

from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline

# Characters ordered from lightest (index 0) to heaviest (index -1) visual weight.
# Darkest pixels map to heavy characters; brightest skip entirely.
ASCII_CHARS: str = ". :-=+*#%@"


def compute_cell_characters(
    img: np.ndarray,
    canvas: Canvas,
    params: dict[str, Any],
    img_rect: tuple[float, float, float, float],
) -> list[tuple[float, float, str]]:
    """Compute per-cell character assignments.

    Parameters
    ----------
    img:
        Grayscale image as uint8 numpy array (H x W).
    canvas:
        Canvas defining the drawing area.
    params:
        Generator parameters dict.
    img_rect:
        Tuple (x1, y1, x2, y2) in mm of the image placement rectangle.

    Returns
    -------
    list of (center_x_mm, center_y_mm, char) for each non-skipped cell.
    """
    chars = ASCII_CHARS
    cell_size_mm = float(params.get("cell_size_mm", 6.0))
    min_darkness = float(params.get("min_darkness", 0.1))

    img_x1, img_y1, img_x2, img_y2 = img_rect
    img_w_mm = img_x2 - img_x1
    img_h_mm = img_y2 - img_y1

    if img_w_mm <= 0 or img_h_mm <= 0 or cell_size_mm <= 0:
        return []

    img_h_px, img_w_px = img.shape[:2]
    if img_h_px == 0 or img_w_px == 0:
        return []

    # Number of cells that fit in the image rect
    n_cols = max(1, int(img_w_mm / cell_size_mm))
    n_rows = max(1, int(img_h_mm / cell_size_mm))

    # Pixels per cell
    cell_w_px = img_w_px / n_cols
    cell_h_px = img_h_px / n_rows

    result: list[tuple[float, float, str]] = []

    for row in range(n_rows):
        for col in range(n_cols):
            # Pixel bounds of this cell in the source image
            px0 = int(col * cell_w_px)
            py0 = int(row * cell_h_px)
            px1 = int((col + 1) * cell_w_px)
            py1 = int((row + 1) * cell_h_px)
            px1 = min(px1, img_w_px)
            py1 = min(py1, img_h_px)

            if px1 <= px0 or py1 <= py0:
                continue

            cell_region = img[py0:py1, px0:px1]
            avg_brightness = float(np.mean(cell_region))

            # Skip bright cells (normalized brightness above min_darkness threshold)
            normalized = avg_brightness / 255.0
            if (1.0 - normalized) < min_darkness:
                continue

            # Map brightness to character index: dark → heavy, bright → light
            idx = int((1.0 - normalized) * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            char = chars[idx]

            # Center of this cell in mm (relative to image rect)
            cx_mm = img_x1 + (col + 0.5) * cell_size_mm
            cy_mm = img_y1 + (row + 0.5) * cell_size_mm

            result.append((cx_mm, cy_mm, char))

    return result


@register_generator
class ASCIIArtGenerator(Generator):
    """ASCII art generator — maps image brightness to character density on a grid."""

    name = "ASCII Art"
    category = "image"
    uses_source_image = True

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="cell_size_mm",
                label="Cell Size (mm)",
                min=2.0,
                max=20.0,
                step=0.5,
                default=6.0,
                description="Size of each character cell in mm — smaller = finer detail",
            ),
            FloatParam(
                name="min_darkness",
                label="Min Darkness",
                min=0.0,
                max=1.0,
                step=0.01,
                default=0.1,
                description="Skip cells brighter than this (0 = keep all, 1 = only very dark)",
            ),
            FloatParam(
                name="char_scale",
                label="Char Scale",
                min=0.3,
                max=2.0,
                step=0.05,
                default=0.75,
                description="Character size relative to cell",
            ),
            ChoiceParam(
                name="image_fit_mode",
                label="Image Fit",
                choices=["fill", "fit", "custom"],
                default="fill",
                description="How the source image is mapped onto the canvas drawing area",
            ),
            FloatParam(
                name="image_width_mm",
                label="Image Width (mm)",
                min=1.0,
                max=2000.0,
                step=1.0,
                default=100.0,
                visible_when={"image_fit_mode": ["custom"]},
                randomizable=False,
                description="Custom image width in mm (only used when fit mode is 'custom')",
            ),
            FloatParam(
                name="image_height_mm",
                label="Image Height (mm)",
                min=1.0,
                max=2000.0,
                step=1.0,
                default=100.0,
                visible_when={"image_fit_mode": ["custom"]},
                randomizable=False,
                description="Custom image height in mm (only used when fit mode is 'custom')",
            ),
            FloatParam(
                name="image_offset_x_mm",
                label="Image X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Horizontal offset of the image on the canvas",
            ),
            FloatParam(
                name="image_offset_y_mm",
                label="Image Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Vertical offset of the image on the canvas",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert image before processing",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before processing",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before processing",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=0.0,
                description="Gaussian blur radius applied before processing (0 = none)",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default",
                params={
                    "cell_size_mm": 6.0,
                    "min_darkness": 0.1,
                    "char_scale": 0.75,
                    "image_fit_mode": "fill",
                    "image_width_mm": 100.0,
                    "image_height_mm": 100.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                },
            ),
            Preset(
                name="Fine Detail",
                params={
                    "cell_size_mm": 3.0,
                    "min_darkness": 0.05,
                    "char_scale": 0.8,
                    "image_fit_mode": "fill",
                    "image_width_mm": 100.0,
                    "image_height_mm": 100.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 10.0,
                    "blur_radius": 0.0,
                },
            ),
            Preset(
                name="Bold Blocks",
                params={
                    "cell_size_mm": 10.0,
                    "min_darkness": 0.15,
                    "char_scale": 0.9,
                    "image_fit_mode": "fill",
                    "image_width_mm": 100.0,
                    "image_height_mm": 100.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 1.0,
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

        # Ensure grayscale
        if img.ndim == 3:
            try:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except ImportError:
                img = img.mean(axis=2).astype(np.uint8)

        if progress_callback:
            progress_callback(10)

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

        if progress_callback:
            progress_callback(20)

        # Compute cell character assignments (glyph rendering in next task)
        _cells = compute_cell_characters(img, canvas, params, img_rect)

        if progress_callback:
            progress_callback(100)

        # Glyph rendering not yet implemented — return empty polylines
        # (cell data is computed but strokes will be added in the next task)
        return []
