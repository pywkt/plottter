"""Font discovery and management for Plottter.

This package provides:

- System font discovery and cataloging (:mod:`~plottter.fonts.discovery`)
- Google Fonts browsing and download (:mod:`~plottter.fonts.google_fonts`)
"""

from plottter.fonts.discovery import (
    FontInfo,
    discover_system_fonts,
    get_font_families,
    get_font_path,
    invalidate_font_cache,
)
from plottter.fonts.google_fonts import (
    GoogleFontInfo,
    download_google_font,
    get_google_fonts_catalog,
    search_google_fonts,
)

__all__ = [
    # System fonts
    "FontInfo",
    "discover_system_fonts",
    "get_font_families",
    "get_font_path",
    "invalidate_font_cache",
    # Google Fonts
    "GoogleFontInfo",
    "get_google_fonts_catalog",
    "search_google_fonts",
    "download_google_font",
]
