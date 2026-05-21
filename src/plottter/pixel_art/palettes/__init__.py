"""Palette registry for the pixel art converter.

Usage:
    >>> from plottter.pixel_art.palettes import get_palette, list_palettes
    >>> palette = get_palette("nes")
    >>> print(f"NES has {palette.color_count} colors")
    NES has 54 colors
"""

from typing import Dict, List, Type

from plottter.pixel_art.palettes.c64 import C64Palette, Commodore64Palette
from plottter.pixel_art.palettes.cga import CGAPalette
from plottter.pixel_art.palettes.ega import EGAPalette
from plottter.pixel_art.palettes.gameboy import (
    GameBoyColorGrayscalePalette,
    GameBoyColorPalette,
    GameBoyPalette,
    GameBoyPocketPalette,
    SuperGameBoyPalette,
)
from plottter.pixel_art.palettes.genesis import GenesisPalette
from plottter.pixel_art.palettes.grayscale import (
    Grayscale2Palette,
    Grayscale4Palette,
    Grayscale8Palette,
    Grayscale16Palette,
    Grayscale256Palette,
    GrayscalePalette,
)
from plottter.pixel_art.palettes.modern import (
    DB32Palette,
    Endesga32Palette,
    Endesga64Palette,
    Resurrect64Palette,
    Sweetie16Palette,
)
from plottter.pixel_art.palettes.nes import NESPalette
from plottter.pixel_art.palettes.palette import (
    FixedPalette,
    GeneratedPalette,
    Palette,
    PaletteMetadata,
    RGB,
    SubPalette,
)
from plottter.pixel_art.palettes.pico8 import PICO8ExtendedPalette, PICO8Palette
from plottter.pixel_art.palettes.snes import SNESPalette

# Registry maps palette name → class. Primary keys use underscores; hyphens are aliases.
PRESET_REGISTRY: Dict[str, Type[Palette]] = {
    # Nintendo
    "nes": NESPalette,
    "gameboy": GameBoyPalette,
    "gb": GameBoyPalette,
    "gameboy_pocket": GameBoyPocketPalette,
    "gameboy-pocket": GameBoyPocketPalette,
    "gbp": GameBoyPocketPalette,
    "super_gameboy": SuperGameBoyPalette,
    "super-gameboy": SuperGameBoyPalette,
    "sgb": SuperGameBoyPalette,
    "gameboy_color": GameBoyColorPalette,
    "gameboy-color": GameBoyColorPalette,
    "gbc": GameBoyColorPalette,
    "gbc_gray": GameBoyColorGrayscalePalette,
    "gbc-gray": GameBoyColorGrayscalePalette,
    "snes": SNESPalette,
    # Sega
    "genesis": GenesisPalette,
    "megadrive": GenesisPalette,
    # IBM PC
    "cga": CGAPalette,
    "ega": EGAPalette,
    # Commodore
    "c64": C64Palette,
    "commodore64": C64Palette,
    # Modern/Fantasy
    "pico8": PICO8Palette,
    "pico-8": PICO8Palette,
    "pico8_extended": PICO8ExtendedPalette,
    "pico8-extended": PICO8ExtendedPalette,
    # Grayscale (underscore primary, hyphen aliases)
    "grayscale": Grayscale16Palette,
    "grayscale_2": Grayscale2Palette,
    "grayscale-2": Grayscale2Palette,
    "grayscale_4": Grayscale4Palette,
    "grayscale-4": Grayscale4Palette,
    "grayscale_8": Grayscale8Palette,
    "grayscale-8": Grayscale8Palette,
    "grayscale_16": Grayscale16Palette,
    "grayscale-16": Grayscale16Palette,
    "grayscale_256": Grayscale256Palette,
    "grayscale-256": Grayscale256Palette,
    # Modern Limited Palettes
    "endesga32": Endesga32Palette,
    "endesga-32": Endesga32Palette,
    "endesga64": Endesga64Palette,
    "endesga-64": Endesga64Palette,
    "sweetie16": Sweetie16Palette,
    "sweetie-16": Sweetie16Palette,
    "db32": DB32Palette,
    "dawnbringer32": DB32Palette,
    "dawnbringer-32": DB32Palette,
    "resurrect64": Resurrect64Palette,
    "resurrect-64": Resurrect64Palette,
}


def get_palette(name: str) -> Palette:
    """Get a preset palette by name.

    Args:
        name: Palette name (case-insensitive, underscores or hyphens accepted).

    Returns:
        Palette instance.

    Raises:
        ValueError: If palette name is not found.
    """
    name_lower = name.lower()
    if name_lower not in PRESET_REGISTRY:
        available = ", ".join(sorted(set(PRESET_REGISTRY.keys())))
        raise ValueError(f"Unknown palette '{name}'. Available: {available}")
    return PRESET_REGISTRY[name_lower]()


def list_palettes() -> List[str]:
    """List all available preset palette names (primary names only, no aliases).

    Returns:
        Sorted list of primary palette names.
    """
    primary: List[str] = []
    seen: set = set()
    for name in sorted(PRESET_REGISTRY.keys()):
        cls = PRESET_REGISTRY[name]
        if cls not in seen:
            primary.append(name)
            seen.add(cls)
    return primary


__all__ = [
    "Palette",
    "FixedPalette",
    "GeneratedPalette",
    "SubPalette",
    "PaletteMetadata",
    "RGB",
    "NESPalette",
    "GameBoyPalette",
    "GameBoyPocketPalette",
    "SuperGameBoyPalette",
    "GameBoyColorGrayscalePalette",
    "GameBoyColorPalette",
    "SNESPalette",
    "GenesisPalette",
    "CGAPalette",
    "EGAPalette",
    "C64Palette",
    "PICO8Palette",
    "PICO8ExtendedPalette",
    "GrayscalePalette",
    "Grayscale2Palette",
    "Grayscale4Palette",
    "Grayscale8Palette",
    "Grayscale16Palette",
    "Grayscale256Palette",
    "Endesga32Palette",
    "Endesga64Palette",
    "Sweetie16Palette",
    "DB32Palette",
    "Resurrect64Palette",
    "PRESET_REGISTRY",
    "get_palette",
    "list_palettes",
]
