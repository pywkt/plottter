"""Game Boy palette variants."""

from typing import List

from plottter.pixel_art.palettes.palette import RGB, FixedPalette, PaletteMetadata

_GAMEBOY_DMG_COLORS: List[RGB] = [
    (155, 188, 15),
    (139, 172, 15),
    (48, 98, 48),
    (15, 56, 15),
]

_GAMEBOY_POCKET_COLORS: List[RGB] = [
    (224, 248, 208),
    (136, 192, 112),
    (52, 104, 86),
    (8, 24, 32),
]

_GAMEBOY_SGB_COLORS: List[RGB] = [
    (248, 248, 248),
    (168, 168, 168),
    (88, 88, 88),
    (0, 0, 0),
]

_GAMEBOY_GBC_GRAY: List[RGB] = [
    (255, 255, 255),
    (170, 170, 170),
    (85, 85, 85),
    (0, 0, 0),
]

_GAMEBOY_COLOR_FULL: List[RGB] = [
    (255, 255, 255), (255, 173, 99),  (132, 49, 0),   (0, 0, 0),
    (255, 255, 255), (255, 132, 132), (148, 58, 58),   (0, 0, 0),
    (255, 255, 255), (173, 123, 82),  (74, 33, 0),     (0, 0, 0),
    (255, 255, 255), (99, 165, 255),  (0, 0, 255),     (0, 0, 0),
    (255, 255, 255), (139, 139, 222), (57, 57, 206),   (0, 0, 0),
    (255, 255, 255), (165, 165, 165), (82, 82, 82),    (0, 0, 0),
    (255, 255, 255), (82, 255, 0),    (0, 139, 0),     (0, 0, 0),
    (255, 255, 255), (123, 255, 49),  (0, 99, 0),      (0, 0, 0),
    (255, 255, 255), (132, 255, 156), (0, 139, 66),    (0, 0, 0),
    (0, 0, 0),       (0, 132, 132),   (255, 222, 0),   (255, 255, 255),
    (248, 248, 248), (120, 192, 120), (48, 96, 48),    (8, 24, 8),
    (248, 248, 248), (120, 192, 248), (48, 96, 168),   (8, 16, 48),
    (248, 248, 248), (248, 184, 120), (192, 80, 48),   (48, 16, 8),
    (248, 248, 248), (120, 248, 120), (48, 144, 48),   (8, 48, 8),
]


class GameBoyPalette(FixedPalette):
    """Original Game Boy (DMG) 4-shade green palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Game Boy",
            system="Nintendo Game Boy",
            year=1989,
            description="Original Game Boy 4-shade green palette (DMG)",
            source="https://gbdev.io/pandocs/Palettes.html",
        )
        super().__init__(_GAMEBOY_DMG_COLORS, metadata)


class GameBoyPocketPalette(FixedPalette):
    """Game Boy Pocket/Light 4-shade palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Game Boy Pocket",
            system="Nintendo Game Boy Pocket",
            year=1996,
            description="Game Boy Pocket 4-shade gray-green palette",
        )
        super().__init__(_GAMEBOY_POCKET_COLORS, metadata)


class SuperGameBoyPalette(FixedPalette):
    """Super Game Boy 4-shade grayscale palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Super Game Boy",
            system="Nintendo Super Game Boy",
            year=1994,
            description="Super Game Boy default grayscale palette",
        )
        super().__init__(_GAMEBOY_SGB_COLORS, metadata)


class GameBoyColorGrayscalePalette(FixedPalette):
    """Game Boy Color grayscale palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Game Boy Color (Gray)",
            system="Nintendo Game Boy Color",
            year=1998,
            description="Game Boy Color grayscale palette for DMG games",
        )
        super().__init__(_GAMEBOY_GBC_GRAY, metadata)


class GameBoyColorPalette(FixedPalette):
    """Game Boy Color extended 56-color palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Game Boy Color",
            system="Nintendo Game Boy Color",
            year=1998,
            description="Game Boy Color extended palette (56 colors)",
            source="https://gbdev.io/pandocs/Palettes.html",
        )
        super().__init__(_GAMEBOY_COLOR_FULL, metadata)
