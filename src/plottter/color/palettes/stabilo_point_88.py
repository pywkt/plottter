"""Stabilo Point 88 (20) — a curated spread of the popular fineliner range.

The STABILO Point 88 is a 0.4 mm fineliner sold in sets up to ~50 colours and
a long-time favourite for multi-pen plotting. This is a 20-colour selection that
spans the hue wheel plus neutrals, so the Custom Palette separator has a broad,
buyable pen set to map to. Hex values are approximations of the inks (the brand
publishes no exact sRGB values), so treat them as a close match, not a spec.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Stabilo Point 88 (20)",
    colors=(
        "#000000",  # Black
        "#8E8E8E",  # Grey
        "#1F3F94",  # Blue
        "#0E2C66",  # Deep Blue
        "#56A8DD",  # Light Blue
        "#00A8A0",  # Turquoise
        "#00785A",  # Emerald
        "#009B47",  # Green
        "#A6CE39",  # Light Green
        "#FFD500",  # Yellow
        "#F7A600",  # Apricot
        "#F39200",  # Orange
        "#E32119",  # Red
        "#93001E",  # Dark Red
        "#C2185B",  # Carmine
        "#E5007E",  # Pink
        "#7C4DA0",  # Lilac
        "#6D2C66",  # Plum
        "#A85A2A",  # Sienna
        "#7A4B16",  # Brown
    ),
    description="20-colour selection from the STABILO Point 88 fineliner range — "
    "a broad, buyable multi-pen set spanning the hue wheel plus neutrals.",
    source="Approximated from the STABILO Point 88 fineliner range",
)
