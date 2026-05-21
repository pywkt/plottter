"""Vendored pixel art conversion core — converter, quantizer, dithering, scaler, transparency."""

from plottter.pixel_art.converter import ConversionOptions, PixelArtConverter
from plottter.pixel_art.grid import image_to_palette_grid
from plottter.pixel_art.palettes import get_palette, list_palettes

__all__ = [
    "PixelArtConverter",
    "ConversionOptions",
    "get_palette",
    "list_palettes",
    "image_to_palette_grid",
]
