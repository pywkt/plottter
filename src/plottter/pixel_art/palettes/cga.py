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
