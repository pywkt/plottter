"""Copic 12 — a balanced 12-pen Copic marker set spanning the hue wheel.

Hex values sourced from community-documented Copic ink approximations.
Copic does not publish official hex codes; these are widely-used RGB
approximations accurate enough for color-separation workflow purposes.

Markers selected (one per hue region):
  R29  Lipstick Red      — red
  YR09 Chinese Orange    — red-orange
  Y08  Acid Yellow       — yellow
  YG07 Acid Green        — yellow-green
  G07  Nile Green        — green
  BG18 Teal Blue         — blue-green
  B06  Peacock Blue      — blue
  BV04 Blue Berry        — blue-violet
  V09  Violet            — violet
  RV06 Cerise            — red-violet
  E57  Light Walnut      — earth / brown
  N5   Neutral Gray No.5 — neutral mid-tone
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Copic 12",
    colors=(
        "#ED174B",  # R29  Lipstick Red
        "#F15524",  # YR09 Chinese Orange
        "#FEF200",  # Y08  Acid Yellow
        "#A5CF4F",  # YG07 Acid Green
        "#7BC576",  # G07  Nile Green
        "#37C0B0",  # BG18 Teal Blue
        "#00B3E6",  # B06  Peacock Blue
        "#7C97CE",  # BV04 Blue Berry
        "#8754A1",  # V09  Violet
        "#F386AF",  # RV06 Cerise
        "#CB9B6F",  # E57  Light Walnut
        "#A8A9AD",  # N5   Neutral Gray No.5
    ),
    description="Twelve Copic markers spanning the full hue wheel — ideal for realistic "
    "figurative plots with a broad tonal range.",
    source="http://rgb-codes.blogspot.com/2016/10/hex-codes-copic-markers.html",
)
