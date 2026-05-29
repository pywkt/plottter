"""ASCIIArtGenerator — Grid-based ASCII art placement from image brightness."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.fonts.hershey import CAP_HEIGHT, DEFAULT_FONT_NAME, glyph_strokes
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


def _render_glyph(
    char: str,
    x_mm: float,
    y_mm: float,
    size_mm: float,
    angle_deg: float,
    font: str = DEFAULT_FONT_NAME,
) -> list[Polyline]:
    """Render a Hershey glyph as polylines centered at (x_mm, y_mm).

    Parameters
    ----------
    char:
        Character to render.
    x_mm, y_mm:
        Canvas position (cell center) in mm.
    size_mm:
        Target cap height in mm.
    angle_deg:
        Rotation in degrees (0 = upright, positive = counter-clockwise).
    font:
        Hershey font variant.
    """
    left, right, strokes = glyph_strokes(char, font)
    if not strokes:
        return []

    scale = size_mm / CAP_HEIGHT

    # Center of the glyph bounding box in Hershey units
    cx_h = (left + right) / 2.0
    cy_h = CAP_HEIGHT / 2.0  # baseline=0, cap-top=21 → center at 10.5

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    result: list[Polyline] = []
    for stroke in strokes:
        polyline: Polyline = []
        for hx, hy in stroke:
            # Scale and center; flip y because canvas y increases downward
            dx = (hx - cx_h) * scale
            dy = -(hy - cy_h) * scale
            # Rotate around glyph center
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            polyline.append((x_mm + rx, y_mm + ry))
        if len(polyline) >= 2:
            result.append(polyline)

    return result


def _gradient_angles(img: np.ndarray) -> np.ndarray:
    """Return per-pixel gradient direction in degrees using Sobel operators.

    Result shape matches input shape.  The angle is atan2(gy, gx) in degrees,
    where positive x is right and positive y is down.
    """
    img_f = img.astype(np.float32)
    try:
        import cv2
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    except ImportError:
        gx = np.zeros_like(img_f)
        gy = np.zeros_like(img_f)
        gx[:, 1:-1] = img_f[:, 2:] - img_f[:, :-2]
        gy[1:-1, :] = img_f[2:, :] - img_f[:-2, :]
    return np.degrees(np.arctan2(gy, gx))


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
            ChoiceParam(
                name="rotation_mode",
                label="Rotation Mode",
                choices=["Fixed", "Random", "Gradient"],
                default="Fixed",
                description=(
                    "Fixed: all characters at the same angle; "
                    "Random: each character rotated randomly; "
                    "Gradient: characters aligned with image edge direction (Sobel)"
                ),
            ),
            FloatParam(
                name="fixed_angle_deg",
                label="Angle (deg)",
                min=0.0,
                max=360.0,
                step=1.0,
                default=0.0,
                visible_when={"rotation_mode": ["Fixed"]},
                description="Character rotation angle in degrees (used when Rotation Mode is Fixed)",
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
            Preset(
                name="Typewriter",
                params={
                    "cell_size_mm": 5.0,
                    "min_darkness": 0.1,
                    "char_scale": 0.7,
                    "image_fit_mode": "fill",
                    "image_width_mm": 100.0,
                    "image_height_mm": 100.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "rotation_mode": "Fixed",
                    "fixed_angle_deg": 0.0,
                },
            ),
            Preset(
                name="Scattered Type",
                params={
                    "cell_size_mm": 6.0,
                    "min_darkness": 0.1,
                    "char_scale": 0.6,
                    "image_fit_mode": "fill",
                    "image_width_mm": 100.0,
                    "image_height_mm": 100.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "rotation_mode": "Random",
                    "fixed_angle_deg": 0.0,
                },
            ),
            Preset(
                name="Contour Text",
                params={
                    "cell_size_mm": 4.0,
                    "min_darkness": 0.1,
                    "char_scale": 0.65,
                    "image_fit_mode": "fill",
                    "image_width_mm": 100.0,
                    "image_height_mm": 100.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "rotation_mode": "Gradient",
                    "fixed_angle_deg": 0.0,
                },
            ),
            Preset(
                name="Large Print",
                params={
                    "cell_size_mm": 10.0,
                    "min_darkness": 0.1,
                    "char_scale": 0.8,
                    "image_fit_mode": "fill",
                    "image_width_mm": 100.0,
                    "image_height_mm": 100.0,
                    "image_offset_x_mm": 0.0,
                    "image_offset_y_mm": 0.0,
                    "invert": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "rotation_mode": "Fixed",
                    "fixed_angle_deg": 0.0,
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

        # Compute cell character assignments
        cells = compute_cell_characters(img, canvas, params, img_rect)

        if progress_callback:
            progress_callback(40)

        # --- Rotation setup ---
        rotation_mode = str(params.get("rotation_mode", "Fixed"))
        fixed_angle = float(params.get("fixed_angle_deg", 0.0))

        grad_angles: np.ndarray | None = None
        if rotation_mode == "Gradient":
            grad_angles = _gradient_angles(img)

        rng = np.random.default_rng()

        # --- Glyph size ---
        cell_size_mm = float(params.get("cell_size_mm", 6.0))
        char_scale = float(params.get("char_scale", 0.75))
        size_mm = char_scale * cell_size_mm

        # Mapping from mm back to pixel coords (for gradient sampling)
        img_x1, img_y1, img_x2, img_y2 = img_rect
        img_h_px, img_w_px = img.shape[:2]
        img_w_mm = img_x2 - img_x1
        img_h_mm = img_y2 - img_y1

        # --- Render glyphs ---
        polylines: list[Polyline] = []
        total = len(cells)
        for i, (cx_mm, cy_mm, char) in enumerate(cells):
            if rotation_mode == "Fixed":
                angle = fixed_angle
            elif rotation_mode == "Random":
                angle = float(rng.uniform(0.0, 360.0))
            else:  # Gradient — align character with edge direction (perpendicular to gradient)
                px = int((cx_mm - img_x1) / img_w_mm * img_w_px) if img_w_mm > 0 else 0
                py = int((cy_mm - img_y1) / img_h_mm * img_h_px) if img_h_mm > 0 else 0
                px = max(0, min(img_w_px - 1, px))
                py = max(0, min(img_h_px - 1, py))
                angle = float(grad_angles[py, px]) + 90.0  # type: ignore[index]

            glyphs = _render_glyph(char, cx_mm, cy_mm, size_mm, angle)
            polylines.extend(glyphs)

            if progress_callback and total > 0 and i % max(1, total // 20) == 0:
                progress_callback(40 + int(60 * i / total))

        if progress_callback:
            progress_callback(100)

        return polylines
