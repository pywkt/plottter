"""Image I/O and manipulation utilities.

This module provides functions for loading, saving, and manipulating images.
It wraps PIL/Pillow functionality with validation and error handling specific
to the pixel art converter's needs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import logging

import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError, UnidentifiedImageError

from plottter.pixel_art.exceptions import ExportError, InputFileNotFoundError, InvalidImageError, ResourceLimitError
from plottter.pixel_art.validators import (
    validate_dimensions,
    validate_input_path,
    validate_not_compression_bomb,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata about a loaded image.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        format: Image file format (e.g., "PNG", "JPEG")
        mode: PIL image mode (e.g., "RGB", "RGBA", "L")
        has_alpha: Whether image has an alpha channel
        color_count: Number of unique colors (None if not computed)
    """

    width: int
    height: int
    format: str
    mode: str
    has_alpha: bool
    color_count: Optional[int] = None

    @property
    def size(self) -> tuple[int, int]:
        """Return (width, height) tuple."""
        return (self.width, self.height)

    @property
    def pixel_count(self) -> int:
        """Return total number of pixels."""
        return self.width * self.height


def load_image(
    path: Path, allow_oversized: bool = False
) -> tuple[Image.Image, ImageMetadata]:
    """Load image and extract metadata.

    Args:
        path: Path to the image file
        allow_oversized: If True, skip size limit validation.

    Returns:
        Tuple of (PIL Image in RGBA mode, ImageMetadata)

    Raises:
        InputFileNotFoundError: If file does not exist
        InvalidImageError: If file is not a valid image or unsupported format
    """
    validated_path = validate_input_path(path)

    try:
        with Image.open(validated_path) as img:
            original_format = img.format or "UNKNOWN"
            original_mode = img.mode
            has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info

            try:
                validate_not_compression_bomb(validated_path, img.width, img.height)
            except ResourceLimitError as e:
                raise InvalidImageError(str(e)) from e

            validate_dimensions(img.width, img.height, allow_oversized)

            if img.mode != "RGBA":
                rgba_img: Image.Image = img.convert("RGBA")
            else:
                rgba_img = img

            result_image = rgba_img.copy()

            metadata = ImageMetadata(
                width=result_image.width,
                height=result_image.height,
                format=original_format,
                mode=original_mode,
                has_alpha=has_alpha,
            )

            return result_image, metadata

    except FileNotFoundError as e:
        raise InputFileNotFoundError(f"Image file not found: {validated_path}") from e
    except UnidentifiedImageError as e:
        raise InvalidImageError(
            f"Cannot read {validated_path} as image. File may be corrupted or unsupported."
        ) from e
    except DecompressionBombError as e:
        raise InvalidImageError(
            f"Image {validated_path} is a decompression bomb (would decompress to unsafe size)."
        ) from e
    except OSError as e:
        raise InvalidImageError(f"Error loading image {validated_path}: {e}") from e
    except (ValueError, TypeError) as e:
        raise InvalidImageError(f"Invalid image data in {validated_path}: {e}") from e
    except Exception as e:
        logger.warning(f"Unexpected error loading {validated_path}: {type(e).__name__}: {e}")
        raise InvalidImageError(f"Unexpected error loading image {validated_path}: {e}") from e


def save_image(
    image: Image.Image,
    path: Path,
    format: Optional[str] = None,
    optimize: bool = True,
    compression: int = 6,
    **kwargs: Any,
) -> None:
    """Save image with format-specific options.

    Args:
        image: PIL Image to save
        path: Output path
        format: Output format (auto-detected from extension if None)
        optimize: Whether to optimize file size (PNG/GIF)
        compression: Compression level 0-9 for PNG (default 6)
        **kwargs: Additional format-specific options passed to PIL save()

    Raises:
        ExportError: If save operation fails
    """
    if isinstance(path, str):
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        ext = path.suffix.lower()
        format_map = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".gif": "GIF",
            ".bmp": "BMP",
            ".webp": "WEBP",
            ".tiff": "TIFF",
            ".tif": "TIFF",
        }
        format = format_map.get(ext, "PNG")

    try:
        save_kwargs: dict[str, Any] = dict(kwargs)

        if format == "PNG":
            save_kwargs.setdefault("optimize", optimize)
            save_kwargs.setdefault("compress_level", compression)
        elif format == "JPEG":
            save_kwargs.setdefault("quality", 95)
            save_kwargs.setdefault("optimize", optimize)
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
        elif format == "GIF":
            save_kwargs.setdefault("optimize", optimize)
        elif format == "WEBP":
            save_kwargs.setdefault("quality", 95)
            save_kwargs.setdefault("lossless", True)

        image.save(path, format=format, **save_kwargs)

    except OSError as e:
        raise ExportError(f"Failed to save image to {path}: {e}") from e
    except (ValueError, TypeError) as e:
        raise ExportError(f"Invalid data for saving image to {path}: {e}") from e
    except Exception as e:
        logger.warning(f"Unexpected error saving to {path}: {type(e).__name__}: {e}")
        raise ExportError(f"Failed to save image to {path}: {e}") from e


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array.

    Args:
        image: PIL Image (any mode)

    Returns:
        NumPy array
    """
    return np.array(image)


def array_to_image(array: np.ndarray, mode: Optional[str] = None) -> Image.Image:
    """Convert numpy array to PIL Image.

    Args:
        array: NumPy array with shape (H, W), (H, W, 3), or (H, W, 4)
        mode: PIL mode to use. If None, auto-detected from array shape.

    Returns:
        PIL Image

    Raises:
        InvalidImageError: If array shape is invalid
    """
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if mode is None:
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3:
            if array.shape[2] == 3:
                mode = "RGB"
            elif array.shape[2] == 4:
                mode = "RGBA"
            else:
                raise InvalidImageError(
                    f"Invalid array shape {array.shape}: expected 3 or 4 channels"
                )
        else:
            raise InvalidImageError(f"Invalid array dimensions {array.ndim}: expected 2 or 3")

    return Image.fromarray(array, mode=mode)


def get_unique_colors(image: Image.Image) -> int:
    """Count unique colors in an image.

    Args:
        image: PIL Image

    Returns:
        Number of unique colors
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    colors = image.getcolors(maxcolors=image.width * image.height)
    if colors is None:
        arr = np.array(image)
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
            unique = np.unique(flat, axis=0)
            return len(unique)
        return 0

    return len(colors)


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode.

    Args:
        image: PIL Image in any mode

    Returns:
        PIL Image in RGBA mode
    """
    if image.mode != "RGBA":
        return image.convert("RGBA")
    return image


def composite_on_background(
    image: Image.Image,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Composite transparent image onto solid background.

    Args:
        image: PIL Image with alpha channel
        background_color: RGB tuple for background (default white)

    Returns:
        PIL Image composited on background (RGB mode, no alpha)
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    background = Image.new("RGB", image.size, background_color)
    background.paste(image, mask=image.split()[3])

    return background
