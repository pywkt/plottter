"""BETEM Dual Tip Acrylic Paint Markers — 22 non-metallic colours.

The BETEM dual-tip acrylic marker set ships 24 pens, but two of them (323 Gold
and 322 Silver) are metallics that don't reproduce as flat plotted ink, so this
palette omits them and lists the remaining 22 true colours.

BETEM does not publish official hex codes, so these values were sampled from a
product photo of the flat ink swatches (resources/canvas-pixels/betem-01.png).
Each colour was read from the saturated core of its blob (excluding paper, the
glossy specular highlight, and the number label) to best match the ink rather
than the photo's lighting. Treat them as close approximations — easy to
fine-tune in the Palette Editor.

The comment beside each colour is the pen number printed under its swatch.
Listed in spectrum order (warm → cool → neutral) for a tidy picker.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="BETEM 22 Acrylic",
    colors=(
        # — warm —
        "#D42A22",  # 316  Red
        "#F86734",  # 326  Orange
        "#FFA032",  # 303  Marigold
        "#FFA887",  # 333  Peach
        "#A0615D",  # 318  Brown
        "#FFDF1F",  # 325  Yellow
        # — greens —
        "#DFEF44",  # 313  Lime
        "#02D08F",  # 328  Emerald
        "#18A89A",  # 315  Teal
        "#11D0BC",  # 314  Turquoise
        # — blues —
        "#1F5D87",  # 330  Petrol Blue
        "#1392CD",  # 329  Blue
        "#3AB2F8",  # 309  Sky Blue
        "#2248BF",  # 311  Royal Blue
        # — purples / pinks —
        "#6B4CC6",  # 332  Violet
        "#8F58DF",  # 308  Purple
        "#DF80DF",  # 327  Orchid
        "#D880AF",  # 306  Rose
        "#F89AB2",  # 305  Pink
        # — neutrals —
        "#000000",  # 324  Black
        "#B8B8B8",  # 319  Gray
        "#FFFFFF",  # 320  White
    ),
    description="22 non-metallic colours of the BETEM dual-tip acrylic marker set (approx.).",
    source=(
        "Sampled from product swatch photo "
        "resources/canvas-pixels/betem-01.png (flat ink swatches); approximate. "
        "Metallic gold (323) and silver (322) omitted."
    ),
)
