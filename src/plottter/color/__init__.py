"""Color separation utilities."""

from plottter.color.kmeans import kmeans_separate
from plottter.color.luminance import luminance_separate
from plottter.color.channels import rgb_separate, cmyk_separate
from plottter.color.palette import PenPalette, palette_to_dict, palette_from_dict
from plottter.color.palette_separator import palette_separate
from plottter.color.palettes import get_preset, list_presets, PALETTE_PRESETS

__all__ = [
    "kmeans_separate",
    "luminance_separate",
    "rgb_separate",
    "cmyk_separate",
    "PenPalette",
    "palette_separate",
    "palette_to_dict",
    "palette_from_dict",
    "get_preset",
    "list_presets",
    "PALETTE_PRESETS",
]
