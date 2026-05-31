"""Basic 6 — a first-time-user starter palette spanning primary + earth + paper."""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Basic 6",
    colors=(
        "#000000",  # Black
        "#E63946",  # Red
        "#F4A261",  # Orange (earth)
        "#2A9D8F",  # Teal
        "#264653",  # Deep blue-green
        "#FFFFFF",  # White (paper / no ink)
    ),
    description="Six versatile colors: black, red, orange, teal, deep blue-green, and white. "
    "A good starting point for first-time users.",
    source="plottter built-in preset",
)
