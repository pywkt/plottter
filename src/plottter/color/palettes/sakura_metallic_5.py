"""Sakura Metallic 5 — five metallic gel-pen colors for night-mode aesthetics.

Inspired by the Sakura Gelly Roll Metallic line.  Intended for dark-paper
plots where metallic inks catch the light (gold, silver, rose/lilac, green,
and a white highlight).
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Sakura Metallic 5",
    colors=(
        "#C9A85B",  # Metallic Gold
        "#B8B8C0",  # Metallic Silver
        "#C8A2C8",  # Metallic Lilac / Rose
        "#8FA85B",  # Metallic Green
        "#FFFFFF",  # White (paper / highlight — no ink layer)
    ),
    description="Five metallic gel-pen tones: gold, silver, lilac, green, and white. "
    "Designed for gel-pen night-mode aesthetics on dark paper.",
    source="plottter built-in preset — inspired by Sakura Gelly Roll Metallic line",
)
