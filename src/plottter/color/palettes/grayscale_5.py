"""Grayscale 5 — a five-step warm gray ladder for tonal monochrome work."""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Grayscale 5",
    colors=(
        "#1A1A1A",  # Near-black
        "#4D4D4D",  # Dark gray
        "#808080",  # Mid gray
        "#B3B3B3",  # Light gray
        "#FFFFFF",  # White (paper / no ink)
    ),
    description="Five evenly-spaced gray tones from near-black to white. "
    "Ideal for tonal monochrome and value-study plots.",
    source="plottter built-in preset",
)
