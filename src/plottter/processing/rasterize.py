"""Rasterize a layer's polylines to a grayscale numpy array.

This module enables generator chaining: a layer's vector output can be
rasterized to a bitmap and used as the source image for any image-based
generator (edge detection, hatching, flow field, stippling, etc.).

Coordinate convention
---------------------
The produced image covers the canvas *drawing area* (canvas minus margins),
matching the convention used by all image-based generators via ``_px_to_mm()``:
pixel (0, 0) maps to ``(draw_x1, draw_y1)`` and pixel ``(img_w, img_h)`` maps
to ``(draw_x2, draw_y2)``.  Paths whose coordinates fall outside the drawing
area are clipped to the image bounds by PIL's rendering.
"""

from __future__ import annotations

import warnings

import numpy as np

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer


def rasterize_layer(
    layer: Layer,
    canvas: Canvas,
    resolution_dpi: int = 300,
    stroke_width_mm: float = 0.3,
    invert: bool = False,
) -> np.ndarray:
    """Render a layer's polylines as a grayscale numpy array.

    Draws black strokes on a white background (or white on black when
    *invert* is True) using PIL's ImageDraw.

    The output image covers the canvas *drawing area* (``canvas.drawing_area()``)
    so that pixel (0, 0) corresponds to ``(draw_x1, draw_y1)`` mm.  This matches
    the coordinate mapping used by ``_px_to_mm()`` in all image-based generators,
    ensuring chained output aligns 1:1 with the source layer.

    Args:
        layer: The layer whose paths will be rasterized.
        canvas: The canvas defining the coordinate space.
        resolution_dpi: Output image resolution in DPI. Higher values
            produce more detailed images but use more memory.
        stroke_width_mm: Line thickness in millimetres.
        invert: When True, return white strokes on a black background
            instead of the default black-on-white. Useful when feeding
            output into generators that interpret bright areas as content.

    Returns:
        A uint8 grayscale numpy array with shape (height_px, width_px)
        covering the drawing area of the canvas.

    Raises:
        ValueError: If *canvas* dimensions are non-positive, or if
            *resolution_dpi* is not positive.
        MemoryError: (from Pillow) if the requested resolution would
            produce an image exceeding available memory.
    """
    if canvas.width_mm <= 0 or canvas.height_mm <= 0:
        raise ValueError("Canvas dimensions must be positive.")
    if resolution_dpi <= 0:
        raise ValueError("resolution_dpi must be positive.")

    # Use the drawing area (canvas minus margins) so that pixel (0, 0)
    # corresponds to the drawing area origin (draw_x1, draw_y1).  This matches
    # the coordinate convention expected by _px_to_mm() in all generators.
    draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    draw_w_mm = draw_x2 - draw_x1
    draw_h_mm = draw_y2 - draw_y1

    if draw_w_mm <= 0 or draw_h_mm <= 0:
        raise ValueError(
            f"Canvas drawing area is degenerate: margin ({canvas.margin_mm}mm) exceeds "
            f"canvas dimensions ({canvas.width_mm}×{canvas.height_mm}mm)."
        )

    mm_per_inch = 25.4
    px_per_mm = resolution_dpi / mm_per_inch

    width_px = max(1, int(round(draw_w_mm * px_per_mm)))
    height_px = max(1, int(round(draw_h_mm * px_per_mm)))

    # Warn when the resulting image is very large (>50 MP)
    megapixels = (width_px * height_px) / 1_000_000
    if megapixels > 50:
        warnings.warn(
            f"Rasterizing layer at {resolution_dpi} DPI produces a {megapixels:.1f} MP image "
            f"({width_px}×{height_px} px). Consider using a lower resolution.",
            ResourceWarning,
            stacklevel=2,
        )

    stroke_width_px = max(1, int(round(stroke_width_mm * px_per_mm)))

    # Background colour depends on invert flag
    bg_color = 0 if invert else 255
    line_color = 255 if invert else 0

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for layer rasterization.") from exc

    img = Image.new("L", (width_px, height_px), color=bg_color)
    draw = ImageDraw.Draw(img)

    for polyline in layer.paths:
        if len(polyline) < 2:
            # Single-point paths: draw a small dot
            if len(polyline) == 1:
                x_px = (polyline[0][0] - draw_x1) * px_per_mm
                y_px = (polyline[0][1] - draw_y1) * px_per_mm
                r = stroke_width_px / 2
                draw.ellipse(
                    [x_px - r, y_px - r, x_px + r, y_px + r],
                    fill=line_color,
                )
            continue

        # Convert mm coordinates to pixel coordinates, offset by drawing area origin
        pixel_coords = [
            ((pt[0] - draw_x1) * px_per_mm, (pt[1] - draw_y1) * px_per_mm)
            for pt in polyline
        ]

        draw.line(pixel_coords, fill=line_color, width=stroke_width_px)

    return np.array(img, dtype=np.uint8)
