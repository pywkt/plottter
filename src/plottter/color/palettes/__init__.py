"""Built-in pen-palette presets.

Usage::

    from plottter.color.palettes import PALETTE_PRESETS, list_presets, get_preset

    all_palettes = list_presets()          # ordered list of PenPalette
    p = get_preset("basic 6")             # case-insensitive lookup
    p = get_preset("Copic 12")

Adding a new preset only requires:
1. A new ``<name>.py`` file under this package exporting ``PALETTE: PenPalette``.
2. One import + entry in ``PALETTE_PRESETS`` below.
"""

from __future__ import annotations

from plottter.color.palette import PenPalette

from plottter.color.palettes.basic_6 import PALETTE as _BASIC_6
from plottter.color.palettes.copic_12 import PALETTE as _COPIC_12
from plottter.color.palettes.sakura_metallic_5 import PALETTE as _SAKURA_METALLIC_5
from plottter.color.palettes.grayscale_5 import PALETTE as _GRAYSCALE_5
from plottter.color.palettes.rybk_4 import PALETTE as _RYBK_4
from plottter.color.palettes.cmykog_6 import PALETTE as _CMYKOG_6
from plottter.color.palettes.risograph_6 import PALETTE as _RISOGRAPH_6
from plottter.color.palettes.stabilo_point_88 import PALETTE as _STABILO_POINT_88
from plottter.color.palettes.sakura_micron_8 import PALETTE as _SAKURA_MICRON_8
from plottter.color.palettes.earth_tones_6 import PALETTE as _EARTH_TONES_6
from plottter.color.palettes.pastel_6 import PALETTE as _PASTEL_6
from plottter.color.palettes.neon_5 import PALETTE as _NEON_5
from plottter.color.palettes.skin_tones_6 import PALETTE as _SKIN_TONES_6
from plottter.color.palettes.papermate_inkjoy_30 import PALETTE as _PAPERMATE_INKJOY_30

#: Ordered dict of built-in presets, keyed by their canonical ``PenPalette.name``.
PALETTE_PRESETS: dict[str, PenPalette] = {
    _BASIC_6.name: _BASIC_6,
    _COPIC_12.name: _COPIC_12,
    _SAKURA_METALLIC_5.name: _SAKURA_METALLIC_5,
    _GRAYSCALE_5.name: _GRAYSCALE_5,
    _RYBK_4.name: _RYBK_4,
    _CMYKOG_6.name: _CMYKOG_6,
    _RISOGRAPH_6.name: _RISOGRAPH_6,
    _STABILO_POINT_88.name: _STABILO_POINT_88,
    _SAKURA_MICRON_8.name: _SAKURA_MICRON_8,
    _EARTH_TONES_6.name: _EARTH_TONES_6,
    _PASTEL_6.name: _PASTEL_6,
    _NEON_5.name: _NEON_5,
    _SKIN_TONES_6.name: _SKIN_TONES_6,
    _PAPERMATE_INKJOY_30.name: _PAPERMATE_INKJOY_30,
}


def list_presets() -> list[PenPalette]:
    """Return all built-in presets as an ordered list (insertion order)."""
    return list(PALETTE_PRESETS.values())


def get_preset(name: str) -> PenPalette:
    """Return the built-in preset whose name matches *name* (case-insensitive).

    Raises ``KeyError`` if no preset with that name exists.
    """
    needle = name.strip().lower()
    for key, palette in PALETTE_PRESETS.items():
        if key.lower() == needle:
            return palette
    raise KeyError(f"No built-in palette preset named {name!r}")


__all__ = [
    "PALETTE_PRESETS",
    "list_presets",
    "get_preset",
]
