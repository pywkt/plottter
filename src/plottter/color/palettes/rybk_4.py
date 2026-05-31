"""RYBK 4 — traditional artist's primaries (Red, Yellow, Blue) plus Black.

The pre-CMYK colour model used in painting and printmaking since the 18th
century.  Gamut is smaller than CMYK (no vivid magentas or cyans) but the
primaries map directly to off-the-shelf paint and ink that art-supply stores
already sell, which makes it convenient for pen-plotter use.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="RYBK 4",
    colors=(
        "#DA1F26",  # Red (cadmium-red-light reference)
        "#FFD800",  # Yellow (cadmium-yellow reference)
        "#0247FE",  # Blue (ultramarine / Munsell primary blue)
        "#000000",  # Black
    ),
    description="Traditional artist primaries (Red, Yellow, Blue) plus Black. "
    "Smaller gamut than CMYK but matches pens you can buy off the shelf.",
    source="Munsell colour system / traditional painter primaries",
)
