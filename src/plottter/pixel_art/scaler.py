"""Image scaling algorithms optimized for pixel art.

This module provides various scaling algorithms suitable for pixel art conversion.
The NEAREST method is preferred for pixel art as it preserves hard edges,
while other methods are available for specific use cases.

Key functions:
    scale_image: Scale image to specific target size
    scale_by_factor: Scale image by a multiplier (0.5 = half, 2.0 = double)
    calculate_target_size: Calculate dimensions with aspect ratio preservation
"""

from enum import Enum
from typing import Optional, Tuple

from PIL import Image

from plottter.pixel_art.exceptions import InvalidImageError
from plottter.pixel_art.validators import validate_dimensions, validate_scale_factor


class ScaleMethod(Enum):
    """Available scaling algorithms.

    Attributes:
        NEAREST: Nearest-neighbor interpolation. Best for pixel art as it
            preserves hard edges and doesn't blur pixels.
        BILINEAR: Bilinear interpolation. Produces smooth scaling but may
            blur pixel art.
        BICUBIC: Bicubic interpolation. Higher quality smooth scaling.
        LANCZOS: Lanczos resampling. Best quality for downscaling photos.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


# Map ScaleMethod to PIL resampling filters
_PIL_RESAMPLING = {
    ScaleMethod.NEAREST: Image.Resampling.NEAREST,
    ScaleMethod.BILINEAR: Image.Resampling.BILINEAR,
    ScaleMethod.BICUBIC: Image.Resampling.BICUBIC,
    ScaleMethod.LANCZOS: Image.Resampling.LANCZOS,
}


def scale_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    method: ScaleMethod = ScaleMethod.NEAREST,
    allow_oversized: bool = False,
) -> Image.Image:
    """Scale image to target size using specified method.

    Args:
        image: PIL Image to scale
        target_size: Target (width, height) tuple
        method: Scaling algorithm to use
        allow_oversized: If True, skip size limit validation.

    Returns:
        Scaled PIL Image

    Raises:
        InvalidImageError: If target_size is invalid
    """
    target_width, target_height = target_size

    validate_dimensions(target_width, target_height, allow_oversized)

    resampling = _PIL_RESAMPLING.get(method, Image.Resampling.NEAREST)

    return image.resize((target_width, target_height), resampling)


def scale_by_factor(
    image: Image.Image,
    factor: float,
    method: ScaleMethod = ScaleMethod.NEAREST,
) -> Image.Image:
    """Scale image by a factor (0.5 = half, 2.0 = double).

    Args:
        image: PIL Image to scale
        factor: Scale factor (must be positive)
        method: Scaling algorithm to use

    Returns:
        Scaled PIL Image

    Raises:
        InvalidImageError: If factor is invalid or result dimensions are invalid
    """
    factor = validate_scale_factor(factor)

    original_width, original_height = image.size
    new_width = max(1, int(round(original_width * factor)))
    new_height = max(1, int(round(original_height * factor)))

    return scale_image(image, (new_width, new_height), method)


def calculate_target_size(
    original_size: Tuple[int, int],
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    maintain_aspect: bool = True,
) -> Tuple[int, int]:
    """Calculate target dimensions with optional aspect ratio preservation.

    Args:
        original_size: Original (width, height) tuple
        target_width: Desired width (optional)
        target_height: Desired height (optional)
        maintain_aspect: Whether to preserve aspect ratio

    Returns:
        Calculated (width, height) tuple

    Raises:
        InvalidImageError: If original_size is invalid or target dimensions
            would result in invalid size
    """
    orig_width, orig_height = original_size

    if orig_width <= 0 or orig_height <= 0:
        raise InvalidImageError(f"Invalid original dimensions: {orig_width}x{orig_height}")

    if target_width is None and target_height is None:
        return original_size

    if target_width is not None and target_width <= 0:
        raise InvalidImageError(f"Invalid target width: {target_width}")
    if target_height is not None and target_height <= 0:
        raise InvalidImageError(f"Invalid target height: {target_height}")

    aspect_ratio = orig_width / orig_height

    if target_width is not None and target_height is None:
        calculated_height = max(1, int(round(target_width / aspect_ratio)))
        return (target_width, calculated_height)

    if target_height is not None and target_width is None:
        calculated_width = max(1, int(round(target_height * aspect_ratio)))
        return (calculated_width, target_height)

    if target_width is not None and target_height is not None:
        if not maintain_aspect:
            return (target_width, target_height)

        target_aspect = target_width / target_height

        if aspect_ratio > target_aspect:
            new_width = target_width
            new_height = max(1, int(round(target_width / aspect_ratio)))
        else:
            new_height = target_height
            new_width = max(1, int(round(target_height * aspect_ratio)))

        return (new_width, new_height)

    return original_size


def scale_to_fit(
    image: Image.Image,
    max_width: int,
    max_height: int,
    method: ScaleMethod = ScaleMethod.NEAREST,
    upscale: bool = False,
) -> Image.Image:
    """Scale image to fit within maximum dimensions while preserving aspect ratio.

    Args:
        image: PIL Image to scale
        max_width: Maximum width
        max_height: Maximum height
        method: Scaling algorithm to use
        upscale: If True, upscale images smaller than max dimensions.

    Returns:
        Scaled PIL Image (or original if no scaling needed)

    Raises:
        InvalidImageError: If max dimensions are invalid
    """
    if max_width <= 0 or max_height <= 0:
        raise InvalidImageError(f"Invalid maximum dimensions: {max_width}x{max_height}")

    orig_width, orig_height = image.size

    if orig_width <= max_width and orig_height <= max_height:
        if not upscale:
            return image.copy()

    target_size = calculate_target_size(
        (orig_width, orig_height),
        target_width=max_width,
        target_height=max_height,
        maintain_aspect=True,
    )

    if not upscale:
        if target_size[0] > orig_width or target_size[1] > orig_height:
            return image.copy()

    return scale_image(image, target_size, method)


def upscale_pixels(
    image: Image.Image,
    pixel_size: int,
    allow_oversized: bool = False,
) -> Image.Image:
    """Upscale image by replicating each pixel into an NxN block.

    This function implements pixel-perfect upscaling for pixel art output.
    Each pixel in the input image becomes a pixel_size x pixel_size block
    in the output image.

    This should be the FINAL step in the conversion pipeline.

    Args:
        image: PIL Image to upscale.
        pixel_size: Size of each output pixel (1-32).
        allow_oversized: If True, skip size limit validation.

    Returns:
        Upscaled PIL Image. If pixel_size is 1, returns a copy of the
        original image.

    Raises:
        InvalidImageError: If pixel_size is invalid or would result in
            an image exceeding maximum dimensions.
    """
    from plottter.pixel_art.validators import validate_pixel_size

    validate_pixel_size(pixel_size)

    if pixel_size == 1:
        return image.copy()

    original_width, original_height = image.size
    new_width = original_width * pixel_size
    new_height = original_height * pixel_size

    validate_dimensions(new_width, new_height, allow_oversized)

    return image.resize((new_width, new_height), Image.Resampling.NEAREST)


def scale_to_power_of_two(
    image: Image.Image,
    method: ScaleMethod = ScaleMethod.NEAREST,
    round_up: bool = True,
) -> Image.Image:
    """Scale image to nearest power-of-two dimensions.

    Args:
        image: PIL Image to scale
        method: Scaling algorithm to use
        round_up: If True, round up to next power of two.

    Returns:
        Scaled PIL Image with power-of-two dimensions
    """
    orig_width, orig_height = image.size

    def nearest_power_of_two(n: int, up: bool) -> int:
        if n <= 0:
            return 1

        if n & (n - 1) == 0:
            return n

        power = 1
        while power < n:
            power *= 2

        if up:
            return power
        else:
            return power // 2 if power // 2 >= 1 else 1

    new_width = nearest_power_of_two(orig_width, round_up)
    new_height = nearest_power_of_two(orig_height, round_up)

    if new_width == orig_width and new_height == orig_height:
        return image.copy()

    return scale_image(image, (new_width, new_height), method)
