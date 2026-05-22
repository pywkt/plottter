"""CGA (Color Graphics Adapter) palette."""

from typing import List

from plottter.pixel_art.palettes.palette import RGB, FixedPalette, PaletteMetadata

_CGA_COLORS: List[RGB] = [
    (0x00, 0x00, 0x00),
    (0x00, 0x00, 0xAA),
    (0x00, 0xAA, 0x00),
    (0x00, 0xAA, 0xAA),
    (0xAA, 0x00, 0x00),
    (0xAA, 0x00, 0xAA),
    (0xAA, 0x55, 0x00),
    (0xAA, 0xAA, 0xAA),
    (0x55, 0x55, 0x55),
    (0x55, 0x55, 0xFF),
    (0x55, 0xFF, 0x55),
    (0x55, 0xFF, 0xFF),
    (0xFF, 0x55, 0x55),
    (0xFF, 0x55, 0xFF),
    (0xFF, 0xFF, 0x55),
    (0xFF, 0xFF, 0xFF),
]


class CGAPalette(FixedPalette):
    """IBM CGA 16-color palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="CGA",
            system="IBM CGA",
            year=1981,
            description="IBM Color Graphics Adapter 16-color RGBI palette",
            source="https://en.wikipedia.org/wiki/Color_Graphics_Adapter",
        )
        super().__init__(_CGA_COLORS, metadata)


# CGA Mode 4, Palette 1 (low intensity) + Yellow — the "famous" 4-color CGA look.
# Colors: Black, Cyan, Magenta, Yellow (all authentic CGA RGBI values).
_CGA_MODE4_COLORS: List[RGB] = [
    (0x00, 0x00, 0x00),  # Black
    (0x00, 0xAA, 0xAA),  # Cyan
    (0xAA, 0x00, 0xAA),  # Magenta
    (0xFF, 0xFF, 0x55),  # Yellow
]


class CGAMode4Palette(FixedPalette):
    """CGA Mode 4 iconic 4-color palette: Black, Cyan, Magenta, Yellow."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="CGA Mode 4",
            system="IBM CGA",
            year=1981,
            description="CGA 4-color palette: Black, Cyan, Magenta, Yellow",
            source="https://en.wikipedia.org/wiki/Color_Graphics_Adapter",
        )
        super().__init__(_CGA_MODE4_COLORS, metadata)
