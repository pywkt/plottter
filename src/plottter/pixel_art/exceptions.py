"""Custom exception hierarchy for pixel art converter.

This module defines the exception hierarchy used throughout the pixel art converter.
All exceptions inherit from PixelArtError, making it easy to catch all application
errors while still allowing specific error handling.

Exception Hierarchy:
    PixelArtError (base)
    ├── InputFileNotFoundError - Input file does not exist
    ├── InvalidImageError - Image format invalid or unsupported
    ├── MemoryLimitError - Operation would exceed memory limits
    ├── InvalidPaletteError - Palette is invalid or malformed
    ├── ConfigurationError - Configuration is invalid
    ├── ExportError - Export operation failed
    ├── BatchProcessingError - Batch processing encountered errors
    ├── ResourceLimitError - Resource limit exceeded
    └── PathSecurityError - Path security violation detected

Note: Exception names are chosen to avoid shadowing Python builtins:
    - InputFileNotFoundError (not FileNotFoundError)
    - MemoryLimitError (not MemoryError)
"""


class PixelArtError(Exception):
    """Base exception for all pixel art converter errors.

    All custom exceptions in this application inherit from this class,
    allowing callers to catch all application-specific errors with a
    single except clause.
    """

    pass


class InputFileNotFoundError(PixelArtError):
    """Raised when input file does not exist.

    This is used instead of the builtin FileNotFoundError to maintain
    a consistent exception hierarchy and provide more context about
    the error occurring in the pixel art conversion context.
    """

    pass


class InvalidImageError(PixelArtError):
    """Raised when image format is invalid or unsupported.

    This includes cases where:
    - The file exists but cannot be read as an image
    - The image format is not supported (not PNG, JPEG, GIF, BMP, WebP)
    - The image file is corrupted
    """

    pass


class MemoryLimitError(PixelArtError):
    """Raised when operation would exceed memory limits.

    This is used instead of the builtin MemoryError to provide
    more context about memory limits in the pixel art converter.
    The default limit is 16 megapixels (4096x4096).
    """

    pass


class InvalidPaletteError(PixelArtError):
    """Raised when palette is invalid or malformed.

    This includes cases where:
    - Palette file cannot be parsed (invalid JSON, GPL format)
    - Palette contains invalid color values (not 0-255)
    - Palette is empty or has too few colors
    - Palette preset name is not recognized
    """

    pass


class ConfigurationError(PixelArtError):
    """Raised when configuration is invalid.

    This includes cases where:
    - Config file cannot be parsed
    - Config values are out of valid range
    - Required config options are missing
    - Conflicting config options are specified
    """

    pass


class ExportError(PixelArtError):
    """Raised when export operation fails.

    This includes cases where:
    - Output path is not writable
    - Export format is not supported
    - PNG compression fails
    - Indexed PNG cannot represent all colors
    """

    pass


class BatchProcessingError(PixelArtError):
    """Raised when batch processing encounters errors.

    This is raised when batch processing fails in a way that
    prevents continuation. Individual file failures during batch
    processing are typically logged and collected, not raised.
    """

    pass


class ResourceLimitError(PixelArtError):
    """Raised when a resource limit is exceeded.

    This includes cases where:
    - Manifest file is too large
    - Too many files in a batch operation
    - Compression bomb detected (small file claiming huge dimensions)
    - File size exceeds configured limits

    Security Note:
        These limits help prevent denial of service attacks where
        malicious inputs could cause memory exhaustion or excessive
        processing time.
    """

    pass


class PathSecurityError(PixelArtError):
    """Raised when a path security violation is detected.

    This exception is raised when:
    - A path attempts to escape its intended directory (path traversal)
    - A path is a symlink pointing outside the allowed directory
    - A filename contains unsafe characters for path construction

    Security Note:
        These checks are critical for preventing attackers from reading
        or writing files outside intended directories via malicious
        manifest files or crafted input paths.
    """

    pass
