"""Main pixel art conversion pipeline.

This module provides the PixelArtConverter class which orchestrates the complete
pixel art conversion pipeline, including scaling, quantization, dithering,
and transparency handling.

Pipeline order (per H12 specification):
    1. Load image
    2. Handle transparency (pre-processing)
    3. Scale to target size
    4. Quantize colors to palette
    5. Apply dithering (if enabled)
    6. Handle transparency (post-processing)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from PIL import Image

from plottter.pixel_art.dithering import DitherMethod, DitherOptions, apply_dithering
from plottter.pixel_art.quantizer import ColorSpace, QuantizeMethod, QuantizeOptions, quantize_to_palette
from plottter.pixel_art.scaler import ScaleMethod, calculate_target_size, scale_image, upscale_pixels
from plottter.pixel_art.transparency import (
    AlphaHandling,
    TransparencyOptions,
    handle_transparency,
)
from plottter.pixel_art.exceptions import InvalidImageError, InvalidPaletteError, MemoryLimitError
from plottter.pixel_art.image_utils import ImageMetadata, get_unique_colors
from plottter.pixel_art.validators import (
    validate_alpha_threshold,
    validate_dither_strength,
    validate_output_dimensions,
    validate_pixel_size,
    validate_rgb_tuple,
    validate_scale_factor,
)
from plottter.pixel_art.palette import Palette


@dataclass
class ConversionOptions:
    """All options for pixel art conversion.

    Attributes:
        # Scaling options
        target_width: Target width in pixels.
        target_height: Target height in pixels.
        scale_factor: Scale factor (0.5 = half, 2.0 = double).
        scale_method: Scaling algorithm to use.
        maintain_aspect: Whether to preserve aspect ratio when scaling.

        # Quantization options
        quantize_method: Color quantization algorithm.
        color_space: Color space for distance calculations (rgb or lab).

        # Dithering options
        dither_method: Dithering algorithm. NONE disables dithering.
        dither_strength: Dithering intensity from 0.0 to 1.0.
        bayer_size: Bayer matrix size for ordered dithering (2, 4, or 8).

        # Transparency options
        alpha_handling: How to handle alpha channels.
        alpha_threshold: Threshold for binary alpha (0-255).
        background_color: Background color for REMOVE alpha mode.

        # Output upscaling
        pixel_size: Size of each output pixel (1-32).

        # Size limit override
        allow_oversized: If True, skip MAX_WIDTH/MAX_HEIGHT/MAX_PIXELS
            validation checks.
    """

    # Scaling
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    scale_factor: Optional[float] = None
    scale_method: ScaleMethod = ScaleMethod.NEAREST
    maintain_aspect: bool = True

    # Quantization
    quantize_method: QuantizeMethod = QuantizeMethod.NEAREST
    color_space: ColorSpace = ColorSpace.RGB

    # Dithering
    dither_method: DitherMethod = DitherMethod.NONE
    dither_strength: float = 1.0
    bayer_size: int = 4

    # Transparency
    alpha_handling: AlphaHandling = AlphaHandling.PRESERVE
    alpha_threshold: int = 128
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # Output upscaling (final step)
    pixel_size: int = 1

    # Size limit override
    allow_oversized: bool = False


@dataclass
class ConversionResult:
    """Result of a pixel art conversion operation.

    Attributes:
        image: The converted PIL Image.
        original_metadata: Metadata about the original input image.
        final_size: Final (width, height) of the converted image.
        palette_used: The palette used for quantization (if any).
        colors_in_output: Number of unique colors in the output image.
        original_size: Original (width, height) of the input image.
        scaling_applied: Whether scaling was applied during conversion.
        dithering_applied: Whether dithering was applied during conversion.
    """

    image: Image.Image
    original_metadata: ImageMetadata
    final_size: Tuple[int, int] = field(default=(0, 0))
    palette_used: Optional[Palette] = None
    colors_in_output: int = 0
    original_size: Tuple[int, int] = field(default=(0, 0))
    scaling_applied: bool = False
    dithering_applied: bool = False

    def __post_init__(self) -> None:
        """Initialize computed fields if not set."""
        if self.final_size == (0, 0):
            self.final_size = (self.image.width, self.image.height)
        if self.original_size == (0, 0):
            self.original_size = (self.original_metadata.width, self.original_metadata.height)


class PixelArtConverter:
    """Main converter class orchestrating the pixel art conversion pipeline.

    The PixelArtConverter handles the complete process of converting a regular
    image into pixel art, including:
    - Scaling to target dimensions
    - Quantizing colors to a palette
    - Applying dithering for better color approximation
    - Handling transparency appropriately for game sprites

    Attributes:
        palette: The target palette for color quantization. If None,
            colors are preserved (only scaling/transparency handled).
    """

    def __init__(self, palette: Optional[Palette] = None) -> None:
        """Initialize the converter with an optional palette.

        Args:
            palette: Target palette for color quantization. If None,
                no color quantization is performed.
        """
        self.palette = palette
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate converter configuration.

        Raises:
            InvalidPaletteError: If palette is invalid or empty.
        """
        if self.palette is not None:
            if self.palette.color_count == 0:
                raise InvalidPaletteError("Palette has no colors")

    def convert_image(
        self,
        image: Image.Image,
        options: Optional[ConversionOptions] = None,
        metadata: Optional[ImageMetadata] = None,
    ) -> ConversionResult:
        """Convert a PIL Image to pixel art.

        Processes an in-memory image through the conversion pipeline.

        Args:
            image: PIL Image to convert (RGB or RGBA).
            options: Conversion options. If None, uses default options.
            metadata: Optional pre-computed metadata. If None, metadata
                is generated from the image.

        Returns:
            ConversionResult containing the converted image and metadata.

        Raises:
            InvalidImageError: If image is invalid.
            InvalidPaletteError: If palette is required but invalid.
        """
        if options is None:
            options = ConversionOptions()

        self._validate_options(options)

        if metadata is None:
            metadata = self._generate_metadata(image)

        original_size = (image.width, image.height)

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")

        # === STEP 1: Pre-process transparency (for REMOVE/INDEX modes) ===
        if options.alpha_handling == AlphaHandling.REMOVE:
            trans_options = TransparencyOptions(
                mode=AlphaHandling.REMOVE,
                background_color=options.background_color,
            )
            image = handle_transparency(image, trans_options)

        # === STEP 2: Scale image ===
        image, scaling_applied = self._apply_scaling(image, options)

        # === STEP 3: Quantize colors to palette ===
        if self.palette is not None:
            alpha_backup = None
            if image.mode == "RGBA":
                alpha_backup = image.split()[3]
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, mask=alpha_backup)
            else:
                rgb_image = image

            quant_options = QuantizeOptions(
                method=options.quantize_method,
                color_space=options.color_space,
            )
            quantized = quantize_to_palette(
                rgb_image,
                self.palette,
                method=options.quantize_method,
                color_space=options.color_space,
                options=quant_options,
            )

            # === STEP 4: Apply dithering ===
            dithering_applied = False
            if options.dither_method != DitherMethod.NONE:
                dither_options = DitherOptions(
                    method=options.dither_method,
                    strength=options.dither_strength,
                    preserve_alpha=True,
                    ordered_matrix_size=options.bayer_size,
                )
                quantized = apply_dithering(quantized, self.palette, dither_options)
                dithering_applied = True

            image = quantized

            if alpha_backup is not None and options.alpha_handling != AlphaHandling.REMOVE:
                if alpha_backup.size != image.size:
                    alpha_backup = alpha_backup.resize(image.size, Image.Resampling.NEAREST)

                if image.mode == "RGB":
                    rgba = image.convert("RGBA")
                    r, g, b, _ = rgba.split()
                    image = Image.merge("RGBA", (r, g, b, alpha_backup))
                elif image.mode == "RGBA":
                    r, g, b, _ = image.split()
                    image = Image.merge("RGBA", (r, g, b, alpha_backup))
        else:
            dithering_applied = False

        # === STEP 5: Post-process transparency ===
        if options.alpha_handling != AlphaHandling.REMOVE:
            trans_options = TransparencyOptions(
                mode=options.alpha_handling,
                threshold=options.alpha_threshold,
                background_color=options.background_color,
            )
            image = handle_transparency(image, trans_options)

        # === STEP 6: Apply pixel-size upscaling (final step) ===
        if options.pixel_size > 1:
            try:
                validate_output_dimensions(
                    image.width, image.height, options.pixel_size, options.allow_oversized
                )
            except MemoryLimitError:
                raise

            image = upscale_pixels(image, options.pixel_size, options.allow_oversized)

        colors_in_output = get_unique_colors(image)

        result = ConversionResult(
            image=image,
            original_metadata=metadata,
            final_size=(image.width, image.height),
            palette_used=self.palette,
            colors_in_output=colors_in_output,
            original_size=original_size,
            scaling_applied=scaling_applied,
            dithering_applied=dithering_applied,
        )

        return result

    def _validate_options(self, options: ConversionOptions) -> None:
        """Validate conversion options.

        Args:
            options: Conversion options to validate.

        Raises:
            InvalidImageError: If options contain invalid values.
        """
        if options.scale_factor is not None:
            validate_scale_factor(options.scale_factor)

        if options.target_width is not None and options.target_width <= 0:
            raise InvalidImageError(f"Target width must be positive, got {options.target_width}")
        if options.target_height is not None and options.target_height <= 0:
            raise InvalidImageError(f"Target height must be positive, got {options.target_height}")

        validate_dither_strength(options.dither_strength)
        validate_alpha_threshold(options.alpha_threshold)
        validate_rgb_tuple(options.background_color, "Background color")
        validate_pixel_size(options.pixel_size)

        if options.bayer_size not in (2, 4, 8):
            raise InvalidImageError(
                f"Invalid bayer_size {options.bayer_size}. Must be 2, 4, or 8."
            )

    def _generate_metadata(self, image: Image.Image) -> ImageMetadata:
        """Generate metadata from an image.

        Args:
            image: PIL Image.

        Returns:
            ImageMetadata for the image.
        """
        has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
        color_count = get_unique_colors(image)

        return ImageMetadata(
            width=image.width,
            height=image.height,
            format=image.format or "UNKNOWN",
            mode=image.mode,
            has_alpha=has_alpha,
            color_count=color_count,
        )

    def _apply_scaling(
        self,
        image: Image.Image,
        options: ConversionOptions,
    ) -> Tuple[Image.Image, bool]:
        """Apply scaling based on options.

        Args:
            image: Image to scale.
            options: Conversion options.

        Returns:
            Tuple of (scaled image, whether scaling was applied).
        """
        original_size = (image.width, image.height)

        if options.scale_factor is not None:
            new_width = max(1, int(round(image.width * options.scale_factor)))
            new_height = max(1, int(round(image.height * options.scale_factor)))
            target_size = (new_width, new_height)
        elif options.target_width is not None or options.target_height is not None:
            target_size = calculate_target_size(
                original_size,
                target_width=options.target_width,
                target_height=options.target_height,
                maintain_aspect=options.maintain_aspect,
            )
        else:
            return image, False

        if target_size == original_size:
            return image, False

        scaled = scale_image(image, target_size, options.scale_method, options.allow_oversized)
        return scaled, True


def create_conversion_options(
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    scale_factor: Optional[float] = None,
    scale_method: str = "nearest",
    maintain_aspect: bool = True,
    quantize_method: str = "nearest",
    color_space: str = "rgb",
    dither_method: str = "none",
    dither_strength: float = 1.0,
    bayer_size: int = 4,
    alpha_handling: str = "preserve",
    alpha_threshold: int = 128,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    pixel_size: int = 1,
    allow_oversized: bool = False,
) -> ConversionOptions:
    """Create ConversionOptions from string parameters.

    Convenience function for creating conversion options from configuration
    where enum values are represented as strings.

    Args:
        target_width: Target width in pixels.
        target_height: Target height in pixels.
        scale_factor: Scale factor (e.g., 0.5 for half size).
        scale_method: Scaling method string ("nearest", "bilinear", etc.).
        maintain_aspect: Whether to preserve aspect ratio when scaling.
        quantize_method: Quantization method string ("nearest", "kmeans", etc.).
        color_space: Color space string ("rgb" or "lab").
        dither_method: Dithering method string ("none", "floyd-steinberg", etc.).
        dither_strength: Dithering strength (0.0 to 1.0).
        bayer_size: Bayer matrix size for ordered dithering (2, 4, or 8).
        alpha_handling: Alpha handling mode string.
        alpha_threshold: Alpha threshold for threshold/index modes (0-255).
        background_color: Background RGB tuple for remove mode.
        pixel_size: Size of each output pixel (1-32).
        allow_oversized: If True, skip size limit validation.

    Returns:
        ConversionOptions instance.

    Raises:
        ValueError: If any string parameter is invalid.
    """
    scale_method_map = {method.value: method for method in ScaleMethod}
    quantize_method_map = {method.value: method for method in QuantizeMethod}
    color_space_map = {cs.value: cs for cs in ColorSpace}
    dither_method_map = {method.value: method for method in DitherMethod}
    alpha_handling_map = {mode.value: mode for mode in AlphaHandling}

    scale_method_lower = scale_method.lower()
    if scale_method_lower not in scale_method_map:
        valid = ", ".join(scale_method_map.keys())
        raise ValueError(f"Invalid scale method '{scale_method}'. Valid: {valid}")
    scale_method_enum = scale_method_map[scale_method_lower]

    quantize_method_lower = quantize_method.lower()
    if quantize_method_lower not in quantize_method_map:
        valid = ", ".join(quantize_method_map.keys())
        raise ValueError(f"Invalid quantize method '{quantize_method}'. Valid: {valid}")
    quantize_method_enum = quantize_method_map[quantize_method_lower]

    color_space_lower = color_space.lower()
    if color_space_lower not in color_space_map:
        valid = ", ".join(color_space_map.keys())
        raise ValueError(f"Invalid color space '{color_space}'. Valid: {valid}")
    color_space_enum = color_space_map[color_space_lower]

    dither_method_lower = dither_method.lower()
    if dither_method_lower not in dither_method_map:
        valid = ", ".join(dither_method_map.keys())
        raise ValueError(f"Invalid dither method '{dither_method}'. Valid: {valid}")
    dither_method_enum = dither_method_map[dither_method_lower]

    alpha_handling_lower = alpha_handling.lower()
    if alpha_handling_lower not in alpha_handling_map:
        valid = ", ".join(alpha_handling_map.keys())
        raise ValueError(f"Invalid alpha handling '{alpha_handling}'. Valid: {valid}")
    alpha_handling_enum = alpha_handling_map[alpha_handling_lower]

    return ConversionOptions(
        target_width=target_width,
        target_height=target_height,
        scale_factor=scale_factor,
        scale_method=scale_method_enum,
        maintain_aspect=maintain_aspect,
        quantize_method=quantize_method_enum,
        color_space=color_space_enum,
        dither_method=dither_method_enum,
        dither_strength=dither_strength,
        bayer_size=bayer_size,
        alpha_handling=alpha_handling_enum,
        alpha_threshold=alpha_threshold,
        background_color=background_color,
        pixel_size=pixel_size,
        allow_oversized=allow_oversized,
    )
