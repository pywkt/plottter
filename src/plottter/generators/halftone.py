"""HalftoneGenerator — image-driven halftone with configurable grid layouts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    ImageParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


# ---------------------------------------------------------------------------
# Image sampling
# ---------------------------------------------------------------------------

def _sample_image_at(img: np.ndarray, px: float, py: float) -> float:
    """Bilinear sample of a grayscale image at non-integer pixel coords."""
    h, w = img.shape[:2]
    x0 = int(px)
    y0 = int(py)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    fx = px - int(px)
    fy = py - int(py)
    v00 = float(img[y0, x0])
    v10 = float(img[y0, x1])
    v01 = float(img[y1, x0])
    v11 = float(img[y1, x1])
    return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy


def _brightness_to_radius(
    brightness: float,
    max_radius: float,
    min_radius: float,
    curve: str,
    gamma: float,
) -> float:
    """Map brightness (0=black, 255=white) to a dot radius in mm.

    Dark areas → large radius (near max_radius).
    Light areas → small radius (near min_radius).
    Returns -1.0 if the dot should be skipped entirely.
    """
    t = max(0.0, min(1.0, brightness / 255.0))

    if curve == "Area-Proportional":
        # area ∝ (1 - t^gamma) → radius = max_r * sqrt(1 - t^gamma)
        t_g = t ** gamma
        area_frac = max(0.0, 1.0 - t_g)
        r = max_radius * math.sqrt(area_frac)
    elif curve == "Logarithmic":
        # log1p maps 0→0, 1→1 monotonically
        t_g = t ** gamma
        log_val = math.log1p(t_g * (math.e - 1.0))
        r = max_radius * max(0.0, 1.0 - log_val)
    else:  # "Linear"
        t_g = t ** gamma
        r = max_radius * max(0.0, 1.0 - t_g)

    # Clamp to [min_radius, max_radius]; min_radius acts as a floor for visible dots
    r = max(min_radius, min(max_radius, r))

    # Skip sub-micrometer dots: physically unrenderable and may be floating-point
    # artifacts from bilinear sampling (e.g. pure white image with min_radius=0).
    if r < 1e-6:
        return -1.0
    return r


# ---------------------------------------------------------------------------
# Grid point generation
# ---------------------------------------------------------------------------

def _grid_square(
    spacing: float,
    angle: float,
    w: float,
    h: float,
) -> np.ndarray:
    """Regular square grid rotated by *angle* degrees, clipped to (0,0)–(w,h).

    Parameters
    ----------
    spacing:
        Distance between adjacent grid points in mm.
    angle:
        Grid rotation in degrees (counter-clockwise).
    w, h:
        Drawing area width and height in mm.

    Returns
    -------
    Array of shape (N, 2) with (x_mm, y_mm) grid points inside the area.
    """
    diag = math.hypot(w, h)
    n = int(math.ceil(diag / spacing)) + 2

    cols = np.arange(-n, n + 1, dtype=float)
    rows = np.arange(-n, n + 1, dtype=float)
    gx, gy = np.meshgrid(cols * spacing, rows * spacing)
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    # Rotate around origin then translate to canvas centre
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T
    pts[:, 0] += w / 2.0
    pts[:, 1] += h / 2.0

    mask = (pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)
    return pts[mask]


def _grid_hexagonal(
    spacing: float,
    angle: float,
    w: float,
    h: float,
) -> np.ndarray:
    """Offset hexagonal grid rotated by *angle* degrees, clipped to (0,0)–(w,h).

    Odd rows are shifted right by ``spacing / 2``; row spacing is
    ``spacing * √3 / 2``, producing a close-packed hexagonal arrangement.

    Parameters
    ----------
    spacing:
        Distance between adjacent grid points in mm.
    angle:
        Grid rotation in degrees (counter-clockwise).
    w, h:
        Drawing area width and height in mm.

    Returns
    -------
    Array of shape (N, 2) with (x_mm, y_mm) grid points inside the area.
    """
    row_h = spacing * math.sqrt(3.0) / 2.0
    diag = math.hypot(w, h)
    n_cols = int(math.ceil(diag / spacing)) + 2
    n_rows = int(math.ceil(diag / row_h)) + 2

    pts_list: list[tuple[float, float]] = []
    for row in range(-n_rows, n_rows + 1):
        x_off = (spacing / 2.0) if (row % 2 != 0) else 0.0
        for col in range(-n_cols, n_cols + 1):
            pts_list.append((col * spacing + x_off, row * row_h))

    pts = np.array(pts_list, dtype=float)

    # Rotate and translate
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T
    pts[:, 0] += w / 2.0
    pts[:, 1] += h / 2.0

    mask = (pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)
    return pts[mask]


def _grid_diagonal(spacing: float, w: float, h: float) -> np.ndarray:
    """Square grid rotated 45°, clipped to (0,0)–(w,h).

    Equivalent to ``_grid_square(spacing, 45, w, h)``.

    Parameters
    ----------
    spacing:
        Distance between adjacent grid points in mm.
    w, h:
        Drawing area width and height in mm.

    Returns
    -------
    Array of shape (N, 2) with (x_mm, y_mm) grid points inside the area.
    """
    return _grid_square(spacing, 45.0, w, h)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@register_generator
class HalftoneGenerator(Generator):
    """Image-driven halftone with square, hexagonal, or diagonal grid layouts."""

    name = "Dot Grid Halftone"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            ImageParam(
                name="_source_image",
                label="Source Image",
                randomizable=False,
                description="Source image used to modulate halftone dot sizes.",
            ),
            FloatParam(
                name="grid_spacing_mm",
                label="Grid Spacing (mm)",
                min=0.5,
                max=20.0,
                step=0.1,
                default=3.0,
                description="Distance between dot centers in mm.",
            ),
            ChoiceParam(
                name="grid_type",
                label="Grid Type",
                choices=["Square", "Hexagonal", "Diagonal"],
                default="Square",
                description="Layout of the halftone grid.",
            ),
            FloatParam(
                name="grid_angle_deg",
                label="Grid Angle (°)",
                min=0.0,
                max=90.0,
                step=0.5,
                default=0.0,
                description="Grid rotation angle in degrees.",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Adjust source image brightness before halftoning.",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Adjust source image contrast before halftoning.",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Gaussian blur radius applied to the source image (px).",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                randomizable=False,
                description="Invert image tones before halftoning.",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the output on the canvas (mm).",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the output on the canvas (mm).",
            ),
            FloatParam(
                name="max_dot_radius_mm",
                label="Max Dot Radius (mm)",
                min=0.2,
                max=10.0,
                step=0.05,
                default=1.4,
                description="Radius of dots in darkest areas.",
            ),
            FloatParam(
                name="min_dot_radius_mm",
                label="Min Dot Radius (mm)",
                min=0.0,
                max=2.0,
                step=0.05,
                default=0.1,
                description="Radius of dots in lightest areas — set to 0 to skip light areas entirely.",
            ),
            ChoiceParam(
                name="size_curve",
                label="Size Curve",
                choices=["Area-Proportional", "Linear", "Logarithmic"],
                default="Area-Proportional",
                description="Mapping curve from image brightness to dot size.",
            ),
            FloatParam(
                name="size_gamma",
                label="Size Gamma",
                min=0.5,
                max=3.0,
                step=0.05,
                default=1.5,
                description="Gamma — <1 emphasizes highlights, >1 emphasizes shadows.",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Classic Halftone",
                params={
                    "grid_spacing_mm": 3.0,
                    "grid_type": "Square",
                    "grid_angle_deg": 45.0,
                },
            ),
            Preset(
                name="Hex Halftone",
                params={
                    "grid_spacing_mm": 3.0,
                    "grid_type": "Hexagonal",
                    "grid_angle_deg": 0.0,
                },
            ),
            Preset(
                name="Fine Diagonal",
                params={
                    "grid_spacing_mm": 1.5,
                    "grid_type": "Diagonal",
                    "grid_angle_deg": 0.0,
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
        # Extract parameters
        spacing = float(params.get("grid_spacing_mm", 3.0))
        grid_type = str(params.get("grid_type", "Square"))
        angle = float(params.get("grid_angle_deg", 0.0))
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        max_radius = float(params.get("max_dot_radius_mm", 1.4))
        min_radius = float(params.get("min_dot_radius_mm", 0.1))
        size_curve = str(params.get("size_curve", "Area-Proportional"))
        size_gamma = float(params.get("size_gamma", 1.5))

        # Drawing area dimensions
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        # Generate grid points in drawing-area-local coordinates
        if grid_type == "Hexagonal":
            pts = _grid_hexagonal(spacing, angle, draw_w, draw_h)
        elif grid_type == "Diagonal":
            pts = _grid_diagonal(spacing, draw_w, draw_h)
        else:  # "Square" (default)
            pts = _grid_square(spacing, angle, draw_w, draw_h)

        # Translate to canvas coordinates (drawing area origin + user offset)
        if len(pts) > 0:
            pts[:, 0] += draw_x1 + x_off
            pts[:, 1] += draw_y1 + y_off

        # Load and preprocess source image
        source: np.ndarray | None = params.get("_source_image")
        dots: list[tuple[float, float, float]] = []

        if source is not None:
            from plottter.io.image_import import (
                adjust_brightness,
                adjust_contrast,
                apply_blur,
                invert_image,
            )

            img = source.copy()
            brightness_adj = float(params.get("brightness", 0.0))
            contrast_adj = float(params.get("contrast", 0.0))
            blur_radius = float(params.get("blur_radius", 0.0))
            do_invert = bool(params.get("invert", False))

            if brightness_adj != 0.0:
                img = adjust_brightness(img, brightness_adj)
            if contrast_adj != 0.0:
                img = adjust_contrast(img, contrast_adj)
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
            img_rect = compute_image_rect(
                "fill",
                img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            )
            img_x1, img_y1, img_x2, img_y2 = img_rect
            img_rect_w = img_x2 - img_x1
            img_rect_h = img_y2 - img_y1

            for x_mm, y_mm in pts:
                # Map mm coordinate to pixel coordinate in the source image
                if img_rect_w > 0 and img_rect_h > 0:
                    px = (x_mm - img_x1) / img_rect_w * img_w
                    py = (y_mm - img_y1) / img_rect_h * img_h
                else:
                    px = 0.0
                    py = 0.0

                brightness_val = _sample_image_at(img, px, py)
                r = _brightness_to_radius(brightness_val, max_radius, min_radius, size_curve, size_gamma)

                if r < 0.0:
                    continue

                dots.append((x_mm, y_mm, r))
        else:
            # No image: use max radius for all dots
            dots = [(float(x), float(y), max_radius) for x, y in pts]

        if progress_callback:
            progress_callback(100)

        # Store dots as layer attribute for next task (dot rendering)
        # Return empty for now — rendering implemented in next task
        self._computed_dots = dots
        return []
