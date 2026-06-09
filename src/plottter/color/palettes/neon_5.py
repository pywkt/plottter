"""Neon 5 — vivid fluorescent tones plus a black anchor.

Saturated gel / highlighter-style colours for high-impact posters and pop art.
The fluorescent hues have no clean CMYK equivalent (like the Risograph pink), so
they're worth a dedicated pen. Black anchors line work and shadows.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Neon 5",
    colors=(
        "#FF2D95",  # Fluoro Pink
        "#39FF14",  # Fluoro Green
        "#FF6A00",  # Fluoro Orange
        "#00E5FF",  # Fluoro Blue
        "#000000",  # Black
    ),
    description="Vivid fluorescent tones — pink, green, orange, blue — anchored "
    "with black. High-impact pop / poster work.",
    source="Curated fluorescent palette",
)
