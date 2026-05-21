"""PICO-8 palette."""

from typing import List

from plottter.pixel_art.palettes.palette import RGB, FixedPalette, PaletteMetadata

_PICO8_COLORS: List[RGB] = [
    (0, 0, 0),
    (29, 43, 83),
    (126, 37, 83),
    (0, 135, 81),
    (171, 82, 54),
    (95, 87, 79),
    (194, 195, 199),
    (255, 241, 232),
    (255, 0, 77),
    (255, 163, 0),
    (255, 236, 39),
    (0, 228, 54),
    (41, 173, 255),
    (131, 118, 156),
    (255, 119, 168),
    (255, 204, 170),
]

_PICO8_EXTENDED_COLORS: List[RGB] = [
    (41, 24, 20),
    (17, 29, 53),
    (66, 33, 54),
    (18, 83, 89),
    (116, 47, 41),
    (73, 51, 59),
    (162, 136, 121),
    (243, 239, 125),
    (190, 18, 80),
    (255, 108, 36),
    (168, 231, 46),
    (0, 181, 67),
    (6, 90, 181),
    (117, 70, 101),
    (255, 110, 89),
    (255, 157, 129),
]


class PICO8Palette(FixedPalette):
    """PICO-8 16-color palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="PICO-8",
            system="PICO-8 Fantasy Console",
            year=2015,
            description="PICO-8 fantasy console 16-color palette",
            source="https://pico-8.fandom.com/wiki/Palette",
        )
        super().__init__(_PICO8_COLORS, metadata)


class PICO8ExtendedPalette(FixedPalette):
    """PICO-8 extended 32-color palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="PICO-8 Extended",
            system="PICO-8 Fantasy Console",
            year=2015,
            description="PICO-8 extended 32-color palette (standard + secret colors)",
            source="https://pico-8.fandom.com/wiki/Palette",
        )
        super().__init__(_PICO8_COLORS + _PICO8_EXTENDED_COLORS, metadata)
