"""Hershey vector-font data — compatibility shim.

The real implementation lives in :mod:`plottter.fonts.hershey`, which
parses Inkscape-format SVG single-stroke fonts (Hershey originals + the
modern OFL-licensed EMS family + symbol fonts) lazily on first use.

This module exists for backward compatibility with code that imports
from ``plottter.generators._hershey``.  Everything is re-exported from
the new location; the four legacy font names —
``"Simplex"`` / ``"Duplex"`` / ``"Script"`` / ``"Gothic"`` — are aliased
to their modern equivalents so existing projects keep loading without
modification.

See :mod:`plottter.fonts.hershey` for the rich API (font metrics,
catalog browsing, raw native-unit access).
"""

from __future__ import annotations

from plottter.fonts.hershey import (
    CAP_HEIGHT,
    DEFAULT_FONT_NAME,
    DESCENDER,
    FONTS,
    Font,
    FontEntry,
    FontMetrics,
    Glyph,
    X_HEIGHT,
    entries_by_category,
    get_entry,
    glyph_strokes,
    list_categories,
    list_entries,
    list_names,
    load_font,
    resolve_name,
)

__all__ = [
    "CAP_HEIGHT",
    "DEFAULT_FONT_NAME",
    "DESCENDER",
    "FONTS",
    "Font",
    "FontEntry",
    "FontMetrics",
    "Glyph",
    "X_HEIGHT",
    "entries_by_category",
    "get_entry",
    "glyph_strokes",
    "list_categories",
    "list_entries",
    "list_names",
    "load_font",
    "resolve_name",
]
