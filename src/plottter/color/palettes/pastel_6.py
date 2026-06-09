"""Pastel 6 — soft, light tints for gentle multi-pen shading.

Low-saturation, high-value colours that keep optical mixing soft. Best on white
paper with the Custom Palette or Pointillist separators; very light tints may not
show on dark stock.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Pastel 6",
    colors=(
        "#F7C6C7",  # Blush
        "#BFE3CE",  # Mint
        "#F6E7A8",  # Butter
        "#BBC4E8",  # Periwinkle
        "#D8C2E0",  # Lilac
        "#F8CBA6",  # Peach
    ),
    description="Soft pastel tints — blush, mint, butter, periwinkle, lilac, "
    "peach. Gentle optical mixing on white paper.",
    source="Curated pastel palette",
)
