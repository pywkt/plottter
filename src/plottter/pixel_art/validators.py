"""Input validation functions.

This module provides validation functions for all inputs to the pixel art converter.
These functions follow a fail-fast approach, raising specific exceptions early
to provide clear error messages before processing begins.

Functions follow a common pattern:
- Accept input value to validate
- Return the validated value (possibly transformed)
- Raise specific PixelArtError subclass on validation failure
"""

from pathlib import Path
from typing import List, Optional, Set, Tuple

from plottter.pixel_art.exceptions import (
    InputFileNotFoundError,
    InvalidImageError,
    InvalidPaletteError,
    MemoryLimitError,
    PathSecurityError,
    ResourceLimitError,
)

# Supported input image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Valid output sizes for pixel art (power of 2, suitable for game sprites)
VALID_SIZES = {8, 16, 32, 64, 128, 256}

# Maximum image dimensions to prevent memory issues (16 megapixels)
MAX_PIXELS = 4096 * 4096
MAX_WIDTH = 4096
MAX_HEIGHT = 4096

# Resource limits for batch processing
MAX_MANIFEST_FILE_SIZE = 10 * 1024 * 1024  # 10 MB max manifest file size
MAX_MANIFEST_FILE_COUNT = 10000  # Maximum files allowed in a manifest

# Compression bomb detection thresholds
COMPRESSION_BOMB_THRESHOLD_RATIO = 100  # Max ratio of uncompressed/compressed size
MIN_FILE_SIZE_FOR_BOMB_CHECK = 1024  # Don't check files smaller than 1KB

# Palette constraints
MIN_PALETTE_COLORS = 2
MAX_PALETTE_COLORS = 256

# RGB value range
RGB_MIN = 0
RGB_MAX = 255


def validate_input_path(path: Path) -> Path:
    """Validate input file exists and is readable.

    Args:
        path: Path to the input file

    Returns:
        The validated Path object (resolved to absolute path)

    Raises:
        InputFileNotFoundError: If file does not exist or is not a file
        InvalidImageError: If file extension is not a supported image format
    """
    if isinstance(path, str):
        path = Path(path)

    path = path.resolve()

    if not path.exists():
        raise InputFileNotFoundError(f"Input file not found: {path}")

    if not path.is_file():
        raise InputFileNotFoundError(f"Input path is not a file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise InvalidImageError(f"Unsupported image format '{ext}'. Supported formats: {supported}")

    return path


def validate_output_path(path: Path, overwrite: bool = False) -> Path:
    """Validate output path is writable.

    Args:
        path: Path where output file will be written
        overwrite: If False, raise error if file already exists

    Returns:
        The validated Path object (resolved to absolute path)

    Raises:
        InputFileNotFoundError: If parent directory does not exist
        InvalidImageError: If file exists and overwrite is False
    """
    if isinstance(path, str):
        path = Path(path)

    path = path.resolve()

    parent = path.parent
    if not parent.exists():
        raise InputFileNotFoundError(f"Output directory does not exist: {parent}")

    if not parent.is_dir():
        raise InputFileNotFoundError(f"Output path parent is not a directory: {parent}")

    if path.exists() and not overwrite:
        raise InvalidImageError(
            f"Output file already exists: {path}. Use overwrite=True to replace."
        )

    return path


def validate_dimensions(width: int, height: int, allow_oversized: bool = False) -> Tuple[int, int]:
    """Validate image dimensions are positive and within limits.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        allow_oversized: If True, skip MAX_WIDTH/MAX_HEIGHT/MAX_PIXELS checks.

    Returns:
        Tuple of (width, height) if valid

    Raises:
        InvalidImageError: If dimensions are invalid
        MemoryLimitError: If image would exceed memory limits
    """
    if width <= 0:
        raise InvalidImageError(f"Invalid width {width}: must be positive")

    if height <= 0:
        raise InvalidImageError(f"Invalid height {height}: must be positive")

    if not allow_oversized:
        if width > MAX_WIDTH:
            raise InvalidImageError(f"Width {width} exceeds maximum {MAX_WIDTH}")

        if height > MAX_HEIGHT:
            raise InvalidImageError(f"Height {height} exceeds maximum {MAX_HEIGHT}")

        pixel_count = width * height
        if pixel_count > MAX_PIXELS:
            raise MemoryLimitError(
                f"Image too large ({pixel_count:,} pixels). Maximum supported: {MAX_PIXELS:,} pixels"
            )

    return (width, height)


def validate_output_size(size: int) -> int:
    """Validate output size is a valid pixel art dimension.

    Args:
        size: Target output size in pixels

    Returns:
        The validated size

    Raises:
        InvalidImageError: If size is not in VALID_SIZES
    """
    if size not in VALID_SIZES:
        valid = ", ".join(str(s) for s in sorted(VALID_SIZES))
        raise InvalidImageError(f"Invalid output size {size}. Must be one of: {valid}")

    return size


def validate_palette(colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Validate palette colors are valid RGB tuples.

    Args:
        colors: List of RGB color tuples (r, g, b)

    Returns:
        The validated list of colors

    Raises:
        InvalidPaletteError: If palette is empty, has invalid colors, or exceeds limits
    """
    if not colors:
        raise InvalidPaletteError("Palette cannot be empty")

    if len(colors) < MIN_PALETTE_COLORS:
        raise InvalidPaletteError(
            f"Palette must have at least {MIN_PALETTE_COLORS} colors, got {len(colors)}"
        )

    if len(colors) > MAX_PALETTE_COLORS:
        raise InvalidPaletteError(
            f"Palette cannot have more than {MAX_PALETTE_COLORS} colors, got {len(colors)}"
        )

    validated: List[Tuple[int, int, int]] = []
    for i, color in enumerate(colors):
        try:
            if not isinstance(color, (tuple, list)):
                raise InvalidPaletteError(
                    f"Color at index {i} must be a tuple or list, got {type(color).__name__}"
                )

            if len(color) != 3:
                raise InvalidPaletteError(
                    f"Color at index {i} must have 3 components (RGB), got {len(color)}"
                )

            r, g, b = color
            for component_name, value in [("R", r), ("G", g), ("B", b)]:
                if not isinstance(value, int):
                    raise InvalidPaletteError(
                        f"Color at index {i}: {component_name} value must be int, "
                        f"got {type(value).__name__}"
                    )
                if value < RGB_MIN or value > RGB_MAX:
                    raise InvalidPaletteError(
                        f"Color at index {i}: {component_name} value {value} "
                        f"out of range [{RGB_MIN}, {RGB_MAX}]"
                    )

            validated.append((int(r), int(g), int(b)))

        except (TypeError, ValueError) as e:
            raise InvalidPaletteError(f"Invalid color at index {i}: {e}") from e

    return validated


def validate_scale_factor(factor: float) -> float:
    """Validate scale factor is positive.

    Args:
        factor: Scale factor (e.g., 0.5 for half size, 2.0 for double)

    Returns:
        The validated scale factor

    Raises:
        InvalidImageError: If factor is not positive
    """
    if not isinstance(factor, (int, float)):
        raise InvalidImageError(f"Scale factor must be a number, got {type(factor).__name__}")

    if factor <= 0:
        raise InvalidImageError(f"Scale factor must be positive, got {factor}")

    if factor > 100:
        raise InvalidImageError(f"Scale factor {factor} is unreasonably large (max 100)")

    return float(factor)


def validate_alpha_threshold(threshold: int) -> int:
    """Validate alpha threshold is in valid range.

    Args:
        threshold: Alpha threshold value (0-255)

    Returns:
        The validated threshold

    Raises:
        InvalidImageError: If threshold is out of range
    """
    if not isinstance(threshold, int):
        raise InvalidImageError(
            f"Alpha threshold must be an integer, got {type(threshold).__name__}"
        )

    if threshold < 0 or threshold > 255:
        raise InvalidImageError(f"Alpha threshold must be 0-255, got {threshold}")

    return threshold


def validate_dither_strength(strength: float) -> float:
    """Validate dither strength is in valid range.

    Args:
        strength: Dither strength (0.0 to 1.0)

    Returns:
        The validated strength

    Raises:
        InvalidImageError: If strength is out of range
    """
    if not isinstance(strength, (int, float)):
        raise InvalidImageError(f"Dither strength must be a number, got {type(strength).__name__}")

    if strength < 0.0 or strength > 1.0:
        raise InvalidImageError(f"Dither strength must be 0.0-1.0, got {strength}")

    return float(strength)


def validate_rgb_tuple(rgb: Tuple[int, int, int], context: str = "Color") -> Tuple[int, int, int]:
    """Validate an RGB color tuple.

    Args:
        rgb: A tuple of three integers (R, G, B), each 0-255.
        context: Description for error messages (e.g., "Background color").

    Returns:
        The validated RGB tuple.

    Raises:
        InvalidImageError: If the tuple doesn't have 3 components or
            any component is outside the 0-255 range.
    """
    if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
        raise InvalidImageError(f"{context} must be an RGB tuple of 3 values, got {rgb}")

    for i, (component, name) in enumerate(zip(rgb, ("R", "G", "B"))):
        if not isinstance(component, int):
            raise InvalidImageError(
                f"{context} {name} component must be an integer, got {type(component).__name__}"
            )
        if component < 0 or component > 255:
            raise InvalidImageError(f"{context} {name} component must be 0-255, got {component}")

    return tuple(rgb)  # type: ignore[return-value]


# Pixel size constraints
MIN_PIXEL_SIZE = 1
MAX_PIXEL_SIZE = 32


def validate_pixel_size(pixel_size: int) -> int:
    """Validate pixel size is in valid range for output upscaling.

    Args:
        pixel_size: Size of each output pixel (1-32).

    Returns:
        The validated pixel size.

    Raises:
        InvalidImageError: If pixel_size is not an integer or is
            outside the valid range (1-32).
    """
    if not isinstance(pixel_size, int):
        raise InvalidImageError(f"Pixel size must be an integer, got {type(pixel_size).__name__}")

    if pixel_size < MIN_PIXEL_SIZE or pixel_size > MAX_PIXEL_SIZE:
        raise InvalidImageError(
            f"Pixel size must be {MIN_PIXEL_SIZE}-{MAX_PIXEL_SIZE}, got {pixel_size}"
        )

    return pixel_size


def validate_file_size(path: Path, max_size: int, context: str = "File") -> int:
    """Validate that a file does not exceed the maximum size limit.

    Args:
        path: Path to the file to check.
        max_size: Maximum allowed file size in bytes.
        context: Description for error messages.

    Returns:
        The actual file size in bytes.

    Raises:
        ResourceLimitError: If the file exceeds the maximum size.
        InputFileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise InputFileNotFoundError(f"{context} does not exist: {path}")

    file_size = path.stat().st_size
    if file_size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        file_size_mb = file_size / (1024 * 1024)
        raise ResourceLimitError(
            f"{context} is too large ({file_size_mb:.2f} MB). "
            f"Maximum allowed size is {max_size_mb:.2f} MB."
        )

    return file_size


def validate_file_count(count: int, max_count: int, context: str = "Batch") -> int:
    """Validate that a file count does not exceed the maximum limit.

    Args:
        count: The number of files to validate.
        max_count: Maximum allowed file count.
        context: Description for error messages.

    Returns:
        The validated file count.

    Raises:
        ResourceLimitError: If the count exceeds the maximum.
    """
    if count > max_count:
        raise ResourceLimitError(
            f"{context} contains too many files ({count:,}). "
            f"Maximum allowed is {max_count:,} files."
        )

    return count


def check_compression_bomb(
    file_path: Path,
    claimed_width: int,
    claimed_height: int,
) -> bool:
    """Check if an image file might be a compression bomb.

    Args:
        file_path: Path to the image file.
        claimed_width: Width claimed by the image header.
        claimed_height: Height claimed by the image header.

    Returns:
        True if the file appears to be a compression bomb, False otherwise.
    """
    try:
        file_size = file_path.stat().st_size
    except OSError:
        return False

    if file_size < MIN_FILE_SIZE_FOR_BOMB_CHECK:
        return False

    uncompressed_size = claimed_width * claimed_height * 4

    if file_size > 0:
        ratio = uncompressed_size / file_size
    else:
        return True

    return ratio > COMPRESSION_BOMB_THRESHOLD_RATIO


def validate_not_compression_bomb(
    file_path: Path,
    claimed_width: int,
    claimed_height: int,
) -> None:
    """Validate that an image file is not a compression bomb.

    Args:
        file_path: Path to the image file.
        claimed_width: Width claimed by the image header.
        claimed_height: Height claimed by the image header.

    Raises:
        ResourceLimitError: If the file appears to be a compression bomb.
    """
    if check_compression_bomb(file_path, claimed_width, claimed_height):
        file_size = file_path.stat().st_size
        uncompressed_size = claimed_width * claimed_height * 4
        ratio = uncompressed_size / file_size if file_size > 0 else float("inf")
        raise ResourceLimitError(
            f"Possible compression bomb detected: {file_path.name} "
            f"({file_size:,} bytes on disk) claims dimensions {claimed_width}x{claimed_height} "
            f"(~{uncompressed_size / (1024*1024):.1f} MB uncompressed, ratio: {ratio:.0f}x). "
            f"Maximum allowed ratio is {COMPRESSION_BOMB_THRESHOLD_RATIO}x."
        )


def validate_output_dimensions(
    width: int, height: int, pixel_size: int, allow_oversized: bool = False
) -> Tuple[int, int]:
    """Validate that output dimensions after pixel_size scaling won't exceed limits.

    Args:
        width: Base output width before pixel_size scaling.
        height: Base output height before pixel_size scaling.
        pixel_size: Pixel size multiplier (1-32).
        allow_oversized: If True, skip the check (dangerous).

    Returns:
        Tuple of (final_width, final_height) after pixel_size scaling.

    Raises:
        MemoryLimitError: If final dimensions would exceed limits.
    """
    final_width = width * pixel_size
    final_height = height * pixel_size
    final_pixels = final_width * final_height

    if not allow_oversized and final_pixels > MAX_PIXELS:
        raise MemoryLimitError(
            f"Output dimensions too large: {width}x{height} with pixel_size={pixel_size} "
            f"would produce {final_width}x{final_height} ({final_pixels:,} pixels). "
            f"Maximum supported: {MAX_PIXELS:,} pixels. "
            f"Reduce dimensions or pixel_size, or use --no-size-limit (with caution)."
        )

    return (final_width, final_height)


def is_path_within_directory(path: Path, base_dir: Path) -> bool:
    """Check if a path is safely within a base directory.

    Args:
        path: The path to check.
        base_dir: The directory that should contain the path.

    Returns:
        True if path is within base_dir, False otherwise.
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()

        try:
            resolved_path.relative_to(resolved_base)
            return True
        except ValueError:
            return False
    except (OSError, RuntimeError):
        return False


def validate_path_within_directory(path: Path, base_dir: Path, context: str = "Path") -> Path:
    """Validate that a path is within a base directory, raising on violation.

    Args:
        path: The path to validate.
        base_dir: The directory that should contain the path.
        context: Description of the path for error messages.

    Returns:
        The resolved path if valid.

    Raises:
        PathSecurityError: If the path escapes the base directory.
    """
    if not is_path_within_directory(path, base_dir):
        raise PathSecurityError(
            f"{context} path '{path}' escapes base directory '{base_dir}'. "
            "Path traversal is not allowed for security reasons."
        )
    return path.resolve()


def is_symlink_safe(path: Path, base_dir: Path) -> bool:
    """Check if a symlink target is within the allowed base directory.

    Args:
        path: The path to check (may or may not be a symlink).
        base_dir: The directory that should contain the symlink target.

    Returns:
        True if path is not a symlink or if its target is within base_dir.
        False if path is a symlink pointing outside base_dir.
    """
    try:
        if not path.is_symlink():
            return True

        real_target = path.resolve()
        resolved_base = base_dir.resolve()

        try:
            real_target.relative_to(resolved_base)
            return True
        except ValueError:
            return False
    except (OSError, RuntimeError):
        return False


def sanitize_filename_component(name: str) -> str:
    """Sanitize a filename component to prevent path traversal.

    Args:
        name: The filename component to sanitize.

    Returns:
        The sanitized string with dangerous characters removed.
    """
    if not name:
        return ""

    result = name.replace("\x00", "")
    result = result.replace("/", "").replace("\\", "")

    while ".." in result:
        result = result.replace("..", "")

    result = result.strip().strip(".")

    return result


def validate_manifest_path(
    file_path: str, manifest_dir: Path, allowed_extensions: Optional[Set[str]] = None
) -> Path:
    """Validate and resolve a path from a manifest file.

    Args:
        file_path: The path string from the manifest.
        manifest_dir: The directory containing the manifest file.
        allowed_extensions: Set of allowed extensions.

    Returns:
        The validated, resolved Path object.

    Raises:
        PathSecurityError: If the path is unsafe.
        InvalidImageError: If the extension is not in allowed_extensions.
    """
    path = Path(file_path)

    if not path.is_absolute():
        path = manifest_dir / path

    if not is_path_within_directory(path, manifest_dir):
        raise PathSecurityError(
            f"Manifest path '{file_path}' escapes manifest directory. "
            "Path traversal via '..' or absolute paths is not allowed."
        )

    if not is_symlink_safe(path, manifest_dir):
        raise PathSecurityError(
            f"Manifest path '{file_path}' is a symlink pointing outside "
            f"the allowed directory. Symlink attacks are not allowed."
        )

    if allowed_extensions is not None:
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise InvalidImageError(
                f"File '{file_path}' has unsupported extension '{ext}'. "
                f"Allowed extensions: {allowed}"
            )

    return path.resolve()


def has_path_traversal_sequences(path_str: str) -> bool:
    """Check if a path string contains path traversal sequences.

    Args:
        path_str: The path string to check.

    Returns:
        True if the path contains suspicious traversal sequences.
    """
    if path_str.startswith(".."):
        return True
    if "/.." in path_str or "\\.." in path_str:
        return True
    if "../" in path_str or "..\\" in path_str:
        return True

    return False
