"""Unit tests for vendored pixel art conversion modules.

Tests that each module imports correctly and core classes can be instantiated.
"""

import pytest
from PIL import Image


# --- Import tests ---

def test_import_converter():
    from plottter.pixel_art import converter
    assert hasattr(converter, "PixelArtConverter")


def test_import_quantizer():
    from plottter.pixel_art import quantizer
    assert hasattr(quantizer, "QuantizeMethod")
    assert hasattr(quantizer, "quantize_to_palette")


def test_import_dithering():
    from plottter.pixel_art import dithering
    assert hasattr(dithering, "DitherMethod")
    assert hasattr(dithering, "apply_dithering")


def test_import_scaler():
    from plottter.pixel_art import scaler
    assert hasattr(scaler, "ScaleMethod")
    assert hasattr(scaler, "scale_image")


def test_import_transparency():
    from plottter.pixel_art import transparency
    assert hasattr(transparency, "AlphaHandling")
    assert hasattr(transparency, "handle_transparency")


# --- PixelArtConverter instantiation tests ---

def test_pixel_art_converter_no_palette():
    """PixelArtConverter can be instantiated with no palette (default behaviour)."""
    from plottter.pixel_art.converter import PixelArtConverter

    converter = PixelArtConverter()
    assert converter.palette is None


def test_pixel_art_converter_with_palette():
    """PixelArtConverter accepts a Palette."""
    from plottter.pixel_art.converter import PixelArtConverter
    from plottter.pixel_art.palette import FixedPalette, PaletteMetadata

    colors = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)]
    metadata = PaletteMetadata(name="Test", system="Custom")
    palette = FixedPalette(colors, metadata)

    converter = PixelArtConverter(palette=palette)
    assert converter.palette is not None
    assert converter.palette.color_count == 4


def test_pixel_art_converter_convert_image_no_palette():
    """convert_image works without a palette (no quantization)."""
    from plottter.pixel_art.converter import PixelArtConverter, ConversionOptions

    img = Image.new("RGB", (16, 16), (200, 100, 50))
    converter = PixelArtConverter()
    result = converter.convert_image(img)

    assert result.image is not None
    assert result.image.size == (16, 16)


def test_pixel_art_converter_convert_image_with_scaling():
    """convert_image scales image when target size is provided."""
    from plottter.pixel_art.converter import PixelArtConverter, ConversionOptions

    img = Image.new("RGB", (64, 64), (128, 64, 32))
    converter = PixelArtConverter()
    options = ConversionOptions(target_width=32, target_height=32)
    result = converter.convert_image(img, options=options)

    assert result.image.size == (32, 32)
    assert result.scaling_applied is True


def test_pixel_art_converter_convert_image_with_palette():
    """convert_image quantizes colors when palette is set."""
    from plottter.pixel_art.converter import PixelArtConverter, ConversionOptions
    from plottter.pixel_art.palette import FixedPalette, PaletteMetadata

    colors = [(0, 0, 0), (255, 255, 255)]
    metadata = PaletteMetadata(name="BW", system="Custom")
    palette = FixedPalette(colors, metadata)

    img = Image.new("RGB", (8, 8), (200, 200, 200))
    converter = PixelArtConverter(palette=palette)
    result = converter.convert_image(img)

    assert result.image is not None
    assert result.palette_used is not None


# --- Scaler tests ---

def test_scale_image():
    from plottter.pixel_art.scaler import scale_image, ScaleMethod

    img = Image.new("RGB", (64, 64), (100, 100, 100))
    scaled = scale_image(img, (32, 32), ScaleMethod.NEAREST)
    assert scaled.size == (32, 32)


def test_calculate_target_size_aspect_ratio():
    from plottter.pixel_art.scaler import calculate_target_size

    result = calculate_target_size((100, 50), target_width=50)
    assert result == (50, 25)


def test_upscale_pixels():
    from plottter.pixel_art.scaler import upscale_pixels

    img = Image.new("RGB", (8, 8), (255, 0, 0))
    upscaled = upscale_pixels(img, 4)
    assert upscaled.size == (32, 32)


# --- Transparency tests ---

def test_handle_transparency_preserve():
    from plottter.pixel_art.transparency import handle_transparency, TransparencyOptions, AlphaHandling

    img = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    options = TransparencyOptions(mode=AlphaHandling.PRESERVE)
    result = handle_transparency(img, options)
    assert result.mode == "RGBA"


def test_handle_transparency_remove():
    from plottter.pixel_art.transparency import handle_transparency, TransparencyOptions, AlphaHandling

    img = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    options = TransparencyOptions(mode=AlphaHandling.REMOVE, background_color=(255, 255, 255))
    result = handle_transparency(img, options)
    assert result.mode == "RGB"


# --- Dithering tests ---

def test_apply_dithering_none():
    from plottter.pixel_art.dithering import apply_dithering, DitherOptions, DitherMethod
    from plottter.pixel_art.palette import FixedPalette, PaletteMetadata

    colors = [(0, 0, 0), (255, 255, 255)]
    metadata = PaletteMetadata(name="BW", system="Custom")
    palette = FixedPalette(colors, metadata)

    img = Image.new("RGB", (8, 8), (128, 128, 128))
    options = DitherOptions(method=DitherMethod.NONE)
    result = apply_dithering(img, palette, options)

    assert result is not None
    assert result.size == (8, 8)


def test_apply_dithering_floyd_steinberg():
    from plottter.pixel_art.dithering import apply_dithering, DitherOptions, DitherMethod
    from plottter.pixel_art.palette import FixedPalette, PaletteMetadata

    colors = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]
    metadata = PaletteMetadata(name="Test", system="Custom")
    palette = FixedPalette(colors, metadata)

    img = Image.new("RGB", (16, 16), (128, 64, 32))
    options = DitherOptions(method=DitherMethod.FLOYD_STEINBERG, strength=0.8)
    result = apply_dithering(img, palette, options)

    assert result is not None
    assert result.size == (16, 16)


# --- Quantizer tests ---

def test_quantize_to_palette():
    from plottter.pixel_art.quantizer import quantize_to_palette, QuantizeMethod
    from plottter.pixel_art.palette import FixedPalette, PaletteMetadata

    colors = [(0, 0, 0), (255, 255, 255), (255, 0, 0)]
    metadata = PaletteMetadata(name="Test", system="Custom")
    palette = FixedPalette(colors, metadata)

    img = Image.new("RGB", (16, 16), (200, 50, 50))
    result = quantize_to_palette(img, palette, method=QuantizeMethod.NEAREST)

    assert result is not None
    assert result.size == (16, 16)


# --- Palette tests ---

def test_palette_classes():
    from plottter.pixel_art.palette import FixedPalette, GeneratedPalette, PaletteMetadata

    colors = [(0, 0, 0), (255, 255, 255)]
    metadata = PaletteMetadata(name="BW", system="Custom")
    fp = FixedPalette(colors, metadata)
    assert fp.color_count == 2

    gp_meta = PaletteMetadata(name="2-bit", system="Generated")
    gp = GeneratedPalette(2, gp_meta)  # 2 bits = 64 colors
    assert gp.color_count == 64


# --- Exception tests ---

def test_exceptions_importable():
    from plottter.pixel_art.exceptions import (
        PixelArtError,
        InvalidImageError,
        InvalidPaletteError,
        MemoryLimitError,
        ExportError,
    )
    assert issubclass(InvalidImageError, PixelArtError)
    assert issubclass(InvalidPaletteError, PixelArtError)
