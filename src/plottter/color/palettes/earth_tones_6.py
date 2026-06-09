"""Earth Tones 6 — a muted natural palette for landscape / botanical work.

Warm, desaturated pigments that read as soil, foliage and stone. Pairs well with
the Map generator and contour landscapes where vivid primaries would look wrong.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="Earth Tones 6",
    colors=(
        "#C16A4F",  # Terracotta
        "#C99A2E",  # Ochre
        "#6E7B3D",  # Olive
        "#8FA67E",  # Sage
        "#9A5A3B",  # Clay
        "#4A5A63",  # Slate
    ),
    description="Muted natural palette — terracotta, ochre, olive, sage, clay, "
    "slate. For landscapes, maps and botanical art.",
    source="Curated earth-tone palette",
)
