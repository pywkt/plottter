"""Sakura Micron 8 — the eight-colour Pigma Micron archival liner set.

Pigma Micron pens are pigment-ink fineliners widely used for plotting because
the ink is lightfast and waterproof. The classic eight-colour set (Black, Blue,
Red, Green, Brown, Purple, Rose, Sepia) is a compact, true-to-pen palette. Hex
values approximate the inks.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Sakura Micron 8",
    colors=(
        "#000000",  # Black
        "#1B4DA1",  # Blue
        "#C8102E",  # Red
        "#00843D",  # Green
        "#6B4423",  # Brown
        "#5B2A86",  # Purple
        "#E0457B",  # Rose
        "#6E5848",  # Sepia
    ),
    description="The eight-colour Sakura Pigma Micron pigment-liner set: Black, "
    "Blue, Red, Green, Brown, Purple, Rose, Sepia. Lightfast, true-to-pen.",
    source="Approximated from the Sakura Pigma Micron 8-pen set",
)
