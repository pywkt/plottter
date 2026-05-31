"""CMYKOG 6 — Hexachrome-style extended-gamut process colours.

Pantone Hexachrome extended the four-colour CMYK process with spot orange
and spot green to cover the vivid hues CMYK can't reproduce.  Discontinued
by Pantone in 2008 but the colour set is still widely used in extended-gamut
printing and is a natural fit for six-pen plotter work.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="CMYKOG 6",
    colors=(
        "#00AEEF",  # Cyan (process cyan)
        "#EC008C",  # Magenta (process magenta)
        "#FFF200",  # Yellow (process yellow)
        "#000000",  # Black (key)
        "#F7941D",  # Orange (Hexachrome spot orange)
        "#00A651",  # Green (Hexachrome spot green)
    ),
    description="CMYK plus spot Orange and Green — Hexachrome-style extended "
    "gamut for vivid oranges and greens that CMYK can't hit cleanly.",
    source="Pantone Hexachrome (discontinued 2008)",
)
