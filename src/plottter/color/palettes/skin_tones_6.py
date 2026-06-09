"""Skin Tones 6 — a warm flesh-tone ramp plus a shadow tone for portraits.

Five flesh tints from light to deep, finished with a cool shadow tone so the
Custom Palette / Pointillist separators can model form without going muddy. A
spread, not every complexion — combine or edit in the palette editor as needed.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Skin Tones 6",
    colors=(
        "#F2D6C2",  # Porcelain
        "#E5B595",  # Beige
        "#C98A5E",  # Tan
        "#A66A40",  # Bronze
        "#6E4422",  # Umber
        "#4A3340",  # Shadow Plum
    ),
    description="Warm flesh-tone ramp (porcelain → umber) with a cool shadow "
    "tone, for figure and portrait work.",
    source="Curated portrait palette",
)
