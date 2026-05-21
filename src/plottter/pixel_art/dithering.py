"""Dithering algorithms for color approximation in pixel art.

This module provides various dithering algorithms that help approximate
colors when mapping to a limited palette, creating the illusion of
more colors than are actually available.

Key algorithms:
    NONE: No dithering (simple quantization)
    FLOYD_STEINBERG: Error diffusion dithering (most natural)
    ORDERED: Ordered (Bayer) matrix dithering (retro pattern)
    ATKINSON: Bill Atkinson's dithering (Mac-style, preserves detail)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from plottter.pixel_art.exceptions import InvalidImageError, InvalidPaletteError
from plottter.pixel_art.palette import Palette


class DitherMethod(Enum):
    """Available dithering algorithms.

    Attributes:
        NONE: No dithering - simple nearest-color quantization.
        FLOYD_STEINBERG: Error diffusion dithering.
        ORDERED: Ordered (Bayer) dithering.
        ATKINSON: Bill Atkinson's dithering algorithm.
    """

    NONE = "none"
    FLOYD_STEINBERG = "floyd-steinberg"
    ORDERED = "ordered"
    ATKINSON = "atkinson"


@dataclass
class DitherOptions:
    """Options for dithering operations.

    Attributes:
        method: Dithering algorithm to use.
        strength: Dithering strength from 0.0 (none) to 1.0 (full).
        preserve_alpha: Whether to preserve the alpha channel.
        ordered_matrix_size: Size of Bayer matrix for ordered dithering.
        serpentine: Use serpentine (alternating direction) scanning.
    """

    method: DitherMethod = DitherMethod.NONE
    strength: float = 1.0
    preserve_alpha: bool = True
    ordered_matrix_size: int = 4
    serpentine: bool = True


def apply_dithering(
    image: Image.Image,
    palette: Palette,
    options: Optional[DitherOptions] = None,
) -> Image.Image:
    """Apply dithering algorithm to quantize image to palette.

    Args:
        image: PIL Image to dither (RGB or RGBA).
        palette: Target palette to map colors to.
        options: Dithering options. If None, uses defaults (no dithering).

    Returns:
        Dithered PIL Image with colors from palette.

    Raises:
        InvalidPaletteError: If palette is empty.
        InvalidImageError: If image cannot be processed.
    """
    if palette.color_count == 0:
        raise InvalidPaletteError("Cannot dither to empty palette")

    if options is None:
        options = DitherOptions()

    if not 0.0 <= options.strength <= 1.0:
        raise InvalidImageError(f"Dither strength must be 0.0-1.0, got {options.strength}")

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    has_alpha = image.mode == "RGBA"

    pixels = np.array(image, dtype=np.float32)

    alpha_channel: Optional[np.ndarray] = None
    if has_alpha:
        rgb_pixels = pixels[:, :, :3].copy()
        alpha_channel = pixels[:, :, 3].copy()
    else:
        rgb_pixels = pixels.copy()

    palette_colors = np.array(palette.colors, dtype=np.float32)

    if options.method == DitherMethod.NONE or options.strength == 0.0:
        dithered_rgb = _no_dither(rgb_pixels, palette_colors)
    elif options.method == DitherMethod.FLOYD_STEINBERG:
        dithered_rgb = _floyd_steinberg_dither(
            rgb_pixels, palette_colors, options.strength, options.serpentine
        )
    elif options.method == DitherMethod.ORDERED:
        dithered_rgb = _ordered_dither(
            rgb_pixels, palette_colors, options.strength, options.ordered_matrix_size
        )
    elif options.method == DitherMethod.ATKINSON:
        dithered_rgb = _atkinson_dither(
            rgb_pixels, palette_colors, options.strength, options.serpentine
        )
    else:
        dithered_rgb = _no_dither(rgb_pixels, palette_colors)

    dithered_rgb = np.clip(dithered_rgb, 0, 255).astype(np.uint8)

    if has_alpha and alpha_channel is not None and options.preserve_alpha:
        result = np.zeros(pixels.shape, dtype=np.uint8)
        result[:, :, :3] = dithered_rgb
        result[:, :, 3] = alpha_channel.astype(np.uint8)
        return Image.fromarray(result, "RGBA")
    else:
        return Image.fromarray(dithered_rgb, "RGB")


def _no_dither(
    pixels: np.ndarray,
    palette_colors: np.ndarray,
) -> np.ndarray:
    """Simple nearest-neighbor quantization without dithering (vectorized)."""
    height, width = pixels.shape[:2]

    flat_pixels = pixels.reshape(-1, 3)

    diff = flat_pixels[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]
    distances = np.sum(diff**2, axis=2)

    best_indices = np.argmin(distances, axis=1)

    result = palette_colors[best_indices].reshape(height, width, 3)
    return result


def _floyd_steinberg_dither(
    pixels: np.ndarray,
    palette_colors: np.ndarray,
    strength: float = 1.0,
    serpentine: bool = True,
) -> np.ndarray:
    """Floyd-Steinberg error diffusion dithering."""
    height, width = pixels.shape[:2]
    result = pixels.copy()

    for y in range(height):
        if serpentine and y % 2 == 1:
            x_range = range(width - 1, -1, -1)
            direction = -1
        else:
            x_range = range(width)
            direction = 1

        for x in x_range:
            old_pixel = result[y, x].copy()
            new_pixel = _find_nearest_color(old_pixel, palette_colors)
            result[y, x] = new_pixel

            quant_error = (old_pixel - new_pixel) * strength

            if direction == 1:
                if x + 1 < width:
                    result[y, x + 1] += quant_error * (7.0 / 16.0)
                if y + 1 < height:
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += quant_error * (3.0 / 16.0)
                    result[y + 1, x] += quant_error * (5.0 / 16.0)
                    if x + 1 < width:
                        result[y + 1, x + 1] += quant_error * (1.0 / 16.0)
            else:
                if x - 1 >= 0:
                    result[y, x - 1] += quant_error * (7.0 / 16.0)
                if y + 1 < height:
                    if x + 1 < width:
                        result[y + 1, x + 1] += quant_error * (3.0 / 16.0)
                    result[y + 1, x] += quant_error * (5.0 / 16.0)
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += quant_error * (1.0 / 16.0)

    return np.asarray(result)


def _ordered_dither(
    pixels: np.ndarray,
    palette_colors: np.ndarray,
    strength: float = 1.0,
    matrix_size: int = 4,
) -> np.ndarray:
    """Ordered (Bayer) dithering using a threshold matrix (vectorized)."""
    height, width = pixels.shape[:2]

    bayer_matrix = _get_bayer_matrix(matrix_size)
    actual_matrix_size = bayer_matrix.shape[0]

    n_squared = actual_matrix_size * actual_matrix_size
    normalized_matrix = (bayer_matrix / n_squared - 0.5) * strength

    spread = 64.0

    tiles_y = (height + actual_matrix_size - 1) // actual_matrix_size
    tiles_x = (width + actual_matrix_size - 1) // actual_matrix_size

    tiled_matrix = np.tile(normalized_matrix, (tiles_y, tiles_x))[:height, :width]

    threshold_offsets = tiled_matrix[:, :, np.newaxis] * spread
    adjusted_pixels = pixels + threshold_offsets

    flat_adjusted = adjusted_pixels.reshape(-1, 3)

    diff = flat_adjusted[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]
    distances = np.sum(diff**2, axis=2)

    best_indices = np.argmin(distances, axis=1)

    result = palette_colors[best_indices].reshape(height, width, 3)
    return result


def _atkinson_dither(
    pixels: np.ndarray,
    palette_colors: np.ndarray,
    strength: float = 1.0,
    serpentine: bool = True,
) -> np.ndarray:
    """Atkinson dithering algorithm."""
    height, width = pixels.shape[:2]
    result = pixels.copy()

    for y in range(height):
        if serpentine and y % 2 == 1:
            x_range = range(width - 1, -1, -1)
            direction = -1
        else:
            x_range = range(width)
            direction = 1

        for x in x_range:
            old_pixel = result[y, x].copy()
            new_pixel = _find_nearest_color(old_pixel, palette_colors)
            result[y, x] = new_pixel

            quant_error = (old_pixel - new_pixel) * strength / 8.0

            if direction == 1:
                if x + 1 < width:
                    result[y, x + 1] += quant_error
                if x + 2 < width:
                    result[y, x + 2] += quant_error
                if y + 1 < height:
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += quant_error
                    result[y + 1, x] += quant_error
                    if x + 1 < width:
                        result[y + 1, x + 1] += quant_error
                if y + 2 < height:
                    result[y + 2, x] += quant_error
            else:
                if x - 1 >= 0:
                    result[y, x - 1] += quant_error
                if x - 2 >= 0:
                    result[y, x - 2] += quant_error
                if y + 1 < height:
                    if x + 1 < width:
                        result[y + 1, x + 1] += quant_error
                    result[y + 1, x] += quant_error
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += quant_error
                if y + 2 < height:
                    result[y + 2, x] += quant_error

    return np.asarray(result)


def _find_nearest_color(
    pixel: np.ndarray,
    palette_colors: np.ndarray,
) -> np.ndarray:
    """Find the nearest palette color to a pixel."""
    distances = np.sum((palette_colors - pixel) ** 2, axis=1)
    nearest_idx = int(np.argmin(distances))
    return np.asarray(palette_colors[nearest_idx].copy())


def _get_bayer_matrix(size: int) -> np.ndarray:
    """Generate a Bayer (ordered) dither matrix.

    Args:
        size: Matrix size (2, 4, or 8). Other values default to 4.

    Returns:
        Bayer matrix as numpy array.
    """
    if size == 2:
        return np.array(
            [
                [0, 2],
                [3, 1],
            ],
            dtype=np.float32,
        )

    elif size == 8:
        return np.array(
            [
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21],
            ],
            dtype=np.float32,
        )

    else:
        return np.array(
            [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ],
            dtype=np.float32,
        )


def get_dither_methods() -> List[str]:
    """Get list of available dithering method names.

    Returns:
        List of dithering method value strings.
    """
    return [method.value for method in DitherMethod]


def create_dither_options(
    method: str = "none",
    strength: float = 1.0,
    preserve_alpha: bool = True,
    matrix_size: int = 4,
    serpentine: bool = True,
) -> DitherOptions:
    """Create DitherOptions from string parameters.

    Args:
        method: Dithering method name string.
        strength: Dithering strength 0.0-1.0.
        preserve_alpha: Whether to preserve alpha channel.
        matrix_size: Bayer matrix size for ordered dithering.
        serpentine: Use serpentine scanning.

    Returns:
        DitherOptions instance.

    Raises:
        ValueError: If method name is invalid.
    """
    method_map = {method.value: method for method in DitherMethod}

    method_lower = method.lower()
    if method_lower not in method_map:
        valid = ", ".join(method_map.keys())
        raise ValueError(f"Invalid dither method '{method}'. Valid methods: {valid}")

    return DitherOptions(
        method=method_map[method_lower],
        strength=strength,
        preserve_alpha=preserve_alpha,
        ordered_matrix_size=matrix_size,
        serpentine=serpentine,
    )


def dither_preview(
    image: Image.Image,
    palette: Palette,
    methods: Optional[List[DitherMethod]] = None,
    strength: float = 1.0,
) -> List[Tuple[str, Image.Image]]:
    """Generate preview images with different dithering methods.

    Args:
        image: PIL Image to dither.
        palette: Target palette.
        methods: List of methods to preview. If None, uses all methods.
        strength: Dithering strength to use for all previews.

    Returns:
        List of (method_name, dithered_image) tuples.
    """
    if methods is None:
        methods = list(DitherMethod)

    results: List[Tuple[str, Image.Image]] = []

    for method in methods:
        options = DitherOptions(method=method, strength=strength)
        dithered = apply_dithering(image, palette, options)
        results.append((method.value, dithered))

    return results
