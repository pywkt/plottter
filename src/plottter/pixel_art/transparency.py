"""Alpha channel and transparency handling for pixel art conversion.

This module provides various methods for handling transparency and alpha
channels in images, essential for game sprite creation where transparency
control is critical.

Key features:
    PRESERVE: Keep original alpha channel intact
    THRESHOLD: Convert to binary alpha at specified threshold
    REMOVE: Remove alpha and composite on background color
    INDEX: Use palette index for transparency (for indexed PNG)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from plottter.pixel_art.exceptions import InvalidImageError


class AlphaHandling(Enum):
    """Available methods for handling alpha channels.

    Attributes:
        PRESERVE: Keep the original alpha channel without modification.
        THRESHOLD: Convert alpha to binary (fully transparent or opaque)
            using a threshold value.
        REMOVE: Remove alpha channel entirely and composite the image
            onto a solid background color.
        INDEX: Mark fully transparent pixels for indexed color mode.
    """

    PRESERVE = "preserve"
    THRESHOLD = "threshold"
    REMOVE = "remove"
    INDEX = "index"


@dataclass
class TransparencyOptions:
    """Options for transparency handling operations.

    Attributes:
        mode: How to handle the alpha channel.
        threshold: Alpha threshold for THRESHOLD mode (0-255).
        background_color: RGB tuple for background in REMOVE mode.
        transparent_index: Palette index to mark as transparent for INDEX mode.
        preserve_semi_transparent: In THRESHOLD mode, whether to preserve
            semi-transparent pixels or force them to binary.
    """

    mode: AlphaHandling = AlphaHandling.PRESERVE
    threshold: int = 128
    background_color: Tuple[int, int, int] = (255, 255, 255)
    transparent_index: Optional[int] = None
    preserve_semi_transparent: bool = False


def handle_transparency(
    image: Image.Image,
    options: Optional[TransparencyOptions] = None,
) -> Image.Image:
    """Process image transparency according to specified options.

    Args:
        image: PIL Image to process (RGB or RGBA).
        options: Transparency options. If None, uses defaults (preserve alpha).

    Returns:
        Processed PIL Image with transparency handled as specified.

    Raises:
        InvalidImageError: If image cannot be processed or options are invalid.
    """
    if options is None:
        options = TransparencyOptions()

    if not 0 <= options.threshold <= 255:
        raise InvalidImageError(f"Alpha threshold must be 0-255, got {options.threshold}")

    if len(options.background_color) != 3:
        raise InvalidImageError(
            f"Background color must be RGB tuple, got {options.background_color}"
        )
    for component in options.background_color:
        if not 0 <= component <= 255:
            raise InvalidImageError(
                f"Background color components must be 0-255, got {options.background_color}"
            )

    if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "PA"):
        image = image.convert("RGBA")

    if options.mode == AlphaHandling.PRESERVE:
        return _preserve_alpha(image)
    elif options.mode == AlphaHandling.THRESHOLD:
        return _threshold_alpha(image, options.threshold, options.preserve_semi_transparent)
    elif options.mode == AlphaHandling.REMOVE:
        return _remove_alpha(image, options.background_color)
    elif options.mode == AlphaHandling.INDEX:
        return _index_alpha(
            image,
            options.threshold,
            options.transparent_index,
            options.preserve_semi_transparent,
        )
    else:
        return _preserve_alpha(image)


def _preserve_alpha(image: Image.Image) -> Image.Image:
    """Preserve original alpha channel."""
    if image.mode == "RGBA":
        return image.copy()
    elif image.mode == "RGB":
        return image.convert("RGBA")
    elif image.mode in ("L", "LA"):
        return image.convert("RGBA")
    elif image.mode in ("P", "PA"):
        return image.convert("RGBA")
    else:
        return image.convert("RGBA")


def _threshold_alpha(
    image: Image.Image,
    threshold: int = 128,
    preserve_semi_transparent: bool = False,
) -> Image.Image:
    """Convert alpha channel to binary using threshold."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    pixels = np.array(image)
    alpha = pixels[:, :, 3]

    if preserve_semi_transparent:
        pass  # Alpha is already preserved, nothing to modify
    else:
        binary_alpha = np.where(alpha >= threshold, 255, 0).astype(np.uint8)
        pixels[:, :, 3] = binary_alpha

    return Image.fromarray(pixels, "RGBA")


def _remove_alpha(
    image: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Remove alpha by compositing on background color."""
    if image.mode != "RGBA":
        rgba_image = image.convert("RGBA")
    else:
        rgba_image = image

    background = Image.new("RGB", rgba_image.size, background_color)
    background.paste(rgba_image, mask=rgba_image.split()[3])

    return background


def _index_alpha(
    image: Image.Image,
    threshold: int = 128,
    transparent_index: Optional[int] = None,
    preserve_semi_transparent: bool = False,
) -> Image.Image:
    """Handle alpha for indexed color mode."""
    result = _threshold_alpha(image, threshold, preserve_semi_transparent)

    if transparent_index is not None:
        result.info["transparent_index"] = transparent_index

    return result


def extract_alpha_mask(image: Image.Image) -> Optional[np.ndarray]:
    """Extract alpha channel as numpy array.

    Args:
        image: PIL Image (any mode).

    Returns:
        NumPy array of alpha values (H, W) with values 0-255,
        or None if image has no alpha channel.
    """
    if image.mode == "RGBA":
        arr: np.ndarray = np.array(image)[:, :, 3].copy()
        return arr
    elif image.mode == "LA":
        arr = np.array(image)[:, :, 1].copy()
        return arr
    elif image.mode == "PA":
        rgba = image.convert("RGBA")
        arr = np.array(rgba)[:, :, 3].copy()
        return arr
    elif image.mode == "P" and "transparency" in image.info:
        rgba = image.convert("RGBA")
        arr = np.array(rgba)[:, :, 3].copy()
        return arr
    else:
        return None


def apply_alpha_mask(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Apply alpha mask to image.

    Args:
        image: PIL Image (any mode).
        mask: NumPy array of alpha values (H, W) with values 0-255.

    Returns:
        PIL Image in RGBA mode with the specified alpha mask.

    Raises:
        InvalidImageError: If mask dimensions don't match image.
    """
    if mask.shape != (image.height, image.width):
        raise InvalidImageError(
            f"Mask dimensions {mask.shape} don't match image dimensions "
            f"({image.height}, {image.width})"
        )

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    pixels = np.array(image)
    pixels[:, :, 3] = mask.astype(np.uint8)

    return Image.fromarray(pixels, "RGBA")


def composite_on_background(
    image: Image.Image,
    background: Tuple[int, int, int],
) -> Image.Image:
    """Composite transparent image onto solid background.

    Args:
        image: PIL Image with alpha channel.
        background: RGB tuple for background color.

    Returns:
        PIL Image composited on background (RGB mode, no alpha).
    """
    return _remove_alpha(image, background)


def has_alpha(image: Image.Image) -> bool:
    """Check if image has an alpha channel.

    Args:
        image: PIL Image in any mode.

    Returns:
        True if image has alpha channel, False otherwise.
    """
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def has_transparency(image: Image.Image) -> bool:
    """Check if image actually has any transparent or semi-transparent pixels.

    Args:
        image: PIL Image in any mode.

    Returns:
        True if image has any non-opaque pixels, False otherwise.
    """
    mask = extract_alpha_mask(image)
    if mask is None:
        return False

    return bool(np.any(mask < 255))


def get_alpha_handling_modes() -> List[str]:
    """Get list of available alpha handling mode names.

    Returns:
        List of alpha handling mode value strings.
    """
    return [mode.value for mode in AlphaHandling]


def create_transparency_options(
    mode: str = "preserve",
    threshold: int = 128,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    transparent_index: Optional[int] = None,
    preserve_semi_transparent: bool = False,
) -> TransparencyOptions:
    """Create TransparencyOptions from string parameters.

    Args:
        mode: Alpha handling mode name string.
        threshold: Alpha threshold for THRESHOLD/INDEX modes (0-255).
        background_color: RGB tuple for REMOVE mode.
        transparent_index: Palette index for INDEX mode.
        preserve_semi_transparent: If True, preserve semi-transparent pixels.

    Returns:
        TransparencyOptions instance.

    Raises:
        ValueError: If mode name is invalid.
    """
    mode_map = {mode.value: mode for mode in AlphaHandling}

    mode_lower = mode.lower()
    if mode_lower not in mode_map:
        valid = ", ".join(mode_map.keys())
        raise ValueError(f"Invalid alpha handling mode '{mode}'. Valid modes: {valid}")

    return TransparencyOptions(
        mode=mode_map[mode_lower],
        threshold=threshold,
        background_color=background_color,
        transparent_index=transparent_index,
        preserve_semi_transparent=preserve_semi_transparent,
    )


def transparency_preview(
    image: Image.Image,
    modes: Optional[List[AlphaHandling]] = None,
    threshold: int = 128,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> List[Tuple[str, Image.Image]]:
    """Generate preview images with different transparency handling modes.

    Args:
        image: PIL Image to process.
        modes: List of modes to preview. If None, uses all modes.
        threshold: Alpha threshold for THRESHOLD/INDEX modes.
        background_color: Background for REMOVE mode.

    Returns:
        List of (mode_name, processed_image) tuples.
    """
    if modes is None:
        modes = list(AlphaHandling)

    results: List[Tuple[str, Image.Image]] = []

    for mode in modes:
        options = TransparencyOptions(
            mode=mode,
            threshold=threshold,
            background_color=background_color,
        )
        processed = handle_transparency(image, options)
        results.append((mode.value, processed))

    return results


def create_alpha_gradient(
    width: int,
    height: int,
    direction: str = "horizontal",
) -> np.ndarray:
    """Create an alpha gradient mask for testing or effects.

    Args:
        width: Mask width in pixels.
        height: Mask height in pixels.
        direction: Gradient direction - "horizontal", "vertical",
            "diagonal", or "radial".

    Returns:
        NumPy array of alpha values (H, W) with values 0-255.

    Raises:
        InvalidImageError: If dimensions are invalid.
    """
    if width <= 0 or height <= 0:
        raise InvalidImageError(f"Dimensions must be positive, got {width}x{height}")

    if direction == "horizontal":
        gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    elif direction == "vertical":
        gradient = np.tile(np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1), (1, width))
    elif direction == "diagonal":
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height).reshape(-1, 1)
        gradient = ((x + y) / 2 * 255).astype(np.uint8)
    elif direction == "radial":
        center_x, center_y = width / 2, height / 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        gradient = (255 * (1 - distance / max_distance)).clip(0, 255).astype(np.uint8)
    else:
        raise InvalidImageError(
            f"Invalid direction '{direction}'. Valid: horizontal, vertical, diagonal, radial"
        )

    return gradient


def feather_alpha(
    image: Image.Image,
    radius: int = 3,
) -> Image.Image:
    """Apply feathering (blur) to alpha channel edges.

    Args:
        image: PIL Image with alpha channel.
        radius: Blur radius in pixels.

    Returns:
        PIL Image with feathered alpha.

    Raises:
        InvalidImageError: If radius is negative.
    """
    if radius < 0:
        raise InvalidImageError(f"Radius must be non-negative, got {radius}")

    if radius == 0:
        return image.copy() if image.mode == "RGBA" else image.convert("RGBA")

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    r, g, b, a = image.split()

    from PIL import ImageFilter

    a_blurred = a.filter(ImageFilter.GaussianBlur(radius=radius))

    return Image.merge("RGBA", (r, g, b, a_blurred))


def dilate_alpha(
    image: Image.Image,
    pixels: int = 1,
) -> Image.Image:
    """Expand (dilate) the opaque area of the alpha channel.

    Args:
        image: PIL Image with alpha channel.
        pixels: Number of pixels to expand.

    Returns:
        PIL Image with dilated alpha.

    Raises:
        InvalidImageError: If pixels is negative.
    """
    if pixels < 0:
        raise InvalidImageError(f"Pixels must be non-negative, got {pixels}")

    if pixels == 0:
        return image.copy() if image.mode == "RGBA" else image.convert("RGBA")

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    r, g, b, a = image.split()

    from PIL import ImageFilter

    for _ in range(pixels):
        a = a.filter(ImageFilter.MaxFilter(size=3))

    return Image.merge("RGBA", (r, g, b, a))


def erode_alpha(
    image: Image.Image,
    pixels: int = 1,
) -> Image.Image:
    """Shrink (erode) the opaque area of the alpha channel.

    Args:
        image: PIL Image with alpha channel.
        pixels: Number of pixels to shrink.

    Returns:
        PIL Image with eroded alpha.

    Raises:
        InvalidImageError: If pixels is negative.
    """
    if pixels < 0:
        raise InvalidImageError(f"Pixels must be non-negative, got {pixels}")

    if pixels == 0:
        return image.copy() if image.mode == "RGBA" else image.convert("RGBA")

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    r, g, b, a = image.split()

    from PIL import ImageFilter

    for _ in range(pixels):
        a = a.filter(ImageFilter.MinFilter(size=3))

    return Image.merge("RGBA", (r, g, b, a))
