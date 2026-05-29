"""Hershey + EMS single-stroke font support.

Public API
----------

* :func:`glyph_strokes` — backward-compatible accessor used by the
  existing text generator, OSM labels, calibration plots, and the
  calligraphy plugin.  Returns Hershey-unit-scaled glyph data so
  legacy callers keep working unchanged.
* :func:`load_font` — return the rich :class:`~loader.Font` object for
  new code that wants direct access to native metrics + raw strokes.
* :data:`CAP_HEIGHT`, :data:`X_HEIGHT`, :data:`DESCENDER` — Hershey-unit
  constants preserved from the old module.  ``CAP_HEIGHT == 21`` defines
  the unit system every legacy caller assumes.
* :data:`FONTS` — names list (was a dict in the old module; we expose a
  proxy so ``"Simplex" in FONTS`` still works).

Coordinate convention
---------------------

The old ``_hershey.py`` used a 21-unit cap height with y pointing up and
descenders down to ``y=-7``.  Every shipped SVG font normalises to those
units inside :func:`glyph_strokes` so the existing scaling code
(``scale = size_mm / 21``) keeps producing correctly-sized text.
"""

from __future__ import annotations

from .catalog import (
    DEFAULT_FONT_NAME,
    FontEntry,
    choices_for_param,
    entries_by_category,
    get_entry,
    list_categories,
    list_entries,
    list_names,
    load_font,
    resolve_name,
)
from .loader import Font, FontMetrics, Glyph, load_svg_font

# Hershey-unit constants — legacy callers import these directly.
CAP_HEIGHT: int = 21
X_HEIGHT: int = 14
DESCENDER: int = 7


class _NamesProxy:
    """Read-only proxy that makes ``name in FONTS`` and ``FONTS[name]`` work.

    The old module exposed ``FONTS`` as a dict of glyph data; nothing in the
    codebase iterated the values, only checked membership and resolved
    names — so a thin proxy covers every real usage without forcing the
    loader to parse all 32 SVGs eagerly.
    """

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return resolve_name(name) in {e.name for e in list_entries()}

    def __iter__(self):
        # Includes legacy aliases so existing code that lists font choices
        # still sees the old names.
        yield from list_names()

    def __getitem__(self, name: str):
        return load_font(name)

    def keys(self):  # noqa: D401 — dict-like API
        return list_names()


FONTS = _NamesProxy()


def glyph_strokes(
    char: str,
    font: str = "EMSReadability",
) -> tuple[float, float, list[list[tuple[float, float]]]]:
    """Return ``(left, right, strokes)`` for *char* in *font*.

    Compatibility shim — coordinates are scaled into the **legacy Hershey
    unit system** (``cap_height = 21``, baseline at ``y = 0``, y pointing
    up) regardless of the source font's native units.  ``left``/``right``
    define the bearing box (advance width = ``right - left``).

    Returned ``left``/``right`` are floats; the old hand-coded module used
    integers but every caller multiplies them by a scale factor, so float
    precision avoids cumulative width drift at small font sizes.  Missing
    glyphs fall back to ``"?"``, then to a zero-width empty glyph.  Strokes
    shorter than 2 points are filtered out.
    """
    parsed = load_font(font)
    g = parsed.glyph(char) or parsed.glyph("?")
    if g is None:
        return (-3.0, 3.0, [])

    scale = CAP_HEIGHT / parsed.metrics.cap_height
    advance = g.advance * scale

    # Legacy convention puts the pen origin at the glyph's horizontal centre,
    # so ``left`` is negative and ``right`` positive.  SVG-font glyphs are
    # drawn from x=0 rightward, so we centre them by shifting half the advance.
    left = -advance / 2.0
    right = advance + left

    strokes: list[list[tuple[float, float]]] = []
    for poly in g.strokes:
        shifted = [((px * scale) + left, py * scale) for px, py in poly]
        if len(shifted) >= 2:
            strokes.append(shifted)
    return left, right, strokes


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
    "choices_for_param",
    "entries_by_category",
    "get_entry",
    "glyph_strokes",
    "list_categories",
    "list_entries",
    "list_names",
    "load_font",
    "load_svg_font",
    "resolve_name",
]
