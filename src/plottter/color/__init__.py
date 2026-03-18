"""Color separation utilities."""

from plottter.color.kmeans import kmeans_separate
from plottter.color.luminance import luminance_separate
from plottter.color.channels import rgb_separate, cmyk_separate

__all__ = [
    "kmeans_separate",
    "luminance_separate",
    "rgb_separate",
    "cmyk_separate",
]
