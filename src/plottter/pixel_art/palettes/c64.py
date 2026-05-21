"""Commodore 64 palette."""

from typing import List

from plottter.pixel_art.palettes.palette import RGB, FixedPalette, PaletteMetadata

_C64_COLORS: List[RGB] = [
    (0x00, 0x00, 0x00),
    (0xFF, 0xFF, 0xFF),
    (0x88, 0x39, 0x32),
    (0x67, 0xB6, 0xBD),
    (0x8B, 0x3F, 0x96),
    (0x55, 0xA0, 0x49),
    (0x40, 0x31, 0x8D),
    (0xBF, 0xCE, 0x72),
    (0x8B, 0x54, 0x29),
    (0x57, 0x42, 0x00),
    (0xB8, 0x69, 0x62),
    (0x50, 0x50, 0x50),
    (0x78, 0x78, 0x78),
    (0x94, 0xE0, 0x89),
    (0x78, 0x69, 0xC4),
    (0x9F, 0x9F, 0x9F),
]


class C64Palette(FixedPalette):
    """Commodore 64 16-color palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Commodore 64",
            system="Commodore 64",
            year=1982,
            description="Commodore 64 VIC-II 16-color palette",
            source="https://www.c64-wiki.com/wiki/Color",
        )
        super().__init__(_C64_COLORS, metadata)


Commodore64Palette = C64Palette
