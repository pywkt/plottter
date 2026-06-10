"""PaperMate InkJoy Gel 36 — the 30 unique colours of the 36-pen set.

The 36-pack ships 30 distinct colours; six are doubled (two pens each of
Black, Blue Mist, Periwinkle, Pure Blue, Seafoam, and Strawberry), so the
palette lists each colour once — the duplicates are simply spares.

PaperMate does not publish official hex codes, and no flat ink swatch was
available, so these values were sampled/estimated from a product photo of the
pen barrels (resources/pen-swatches/inkjoy-pens.avif). Treat them as close
approximations rather than exact ink colours — glossy-photo lighting and white
balance shift the readings. They're easy to fine-tune in the Palette Editor.

Listed in spectrum order (warm → cool → neutral) for a tidy picker.
"""

from plottter.color.palette import PenPalette

PALETTE = PenPalette(
    name="PaperMate InkJoy 30",
    colors=(
        "#CD2418",  # Red
        "#962E23",  # Red Velvet
        "#A0314A",  # Garnet
        "#E64F80",  # Strawberry
        "#F2768D",  # Pink
        "#B75096",  # Berry
        "#DDA0C8",  # Pink Topaz
        "#6B3A6E",  # Plum
        "#6A5EB0",  # Amethyst
        "#6E3F9E",  # Purple
        "#8F92DE",  # Periwinkle
        "#47729A",  # Slate Blue
        "#2A50C0",  # Bright Blue
        "#2F6FD2",  # Pure Blue
        "#74CCF7",  # Blue Mist
        "#2F97AD",  # Teal
        "#36A877",  # Jade
        "#93E0D2",  # Seafoam
        "#6CCBC4",  # Aquamarine
        "#1F5C32",  # Evergreen
        "#2E8B3E",  # Green
        "#A6D957",  # Lime
        "#7D944D",  # Olive
        "#FAD23C",  # Yellow
        "#F9B233",  # Marigold
        "#D6A24E",  # Gold Mine
        "#F37513",  # Orange
        "#8A5A48",  # Cocoa
        "#8C8E91",  # Pewter
        "#1A1A1A",  # Black
    ),
    description="The 30 unique colours of the PaperMate InkJoy Gel 36-pen set "
    "(approximated from a product photo). A broad, vivid range well suited to "
    "full-colour pointillist and palette-separation plots.",
    source="Approximated from a product photo of the PaperMate InkJoy Gel 36 set.",
)
