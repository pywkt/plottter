"""Risograph 6 — the classic six-colour Riso duplicator ink palette.

Risograph printers expose a small library of spot inks rather than process
colours; six of the most-used inks form a recognisable zine / illustration
palette.  Fluorescent Pink in particular is a Riso signature: it has no
clean CMYK equivalent because it relies on a fluorescent pigment.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Risograph 6",
    colors=(
        "#FF48B0",  # Fluorescent Pink
        "#3D5588",  # Federal Blue
        "#FFE800",  # Yellow
        "#FF665E",  # Bright Red
        "#000000",  # Black
        "#00A95C",  # Green
    ),
    description="Six Risograph standard inks: Fluorescent Pink, Federal Blue, "
    "Yellow, Bright Red, Black, and Green. Recognisable zine / illustration palette.",
    source="Riso ink reference (stencil.wiki)",
)
