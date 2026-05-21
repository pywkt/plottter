"""Grayscale palettes."""

from typing import List

from plottter.pixel_art.palettes.palette import RGB, FixedPalette, PaletteMetadata


def _generate_grayscale_colors(shades: int) -> List[RGB]:
    shades = max(2, min(256, shades))
    colors: List[RGB] = []
    for i in range(shades):
        value = int(round(i * 255 / (shades - 1))) if shades > 1 else 128
        colors.append((value, value, value))
    return colors


_GRAYSCALE_2_COLORS = _generate_grayscale_colors(2)
_GRAYSCALE_4_COLORS = _generate_grayscale_colors(4)
_GRAYSCALE_8_COLORS = _generate_grayscale_colors(8)
_GRAYSCALE_16_COLORS = _generate_grayscale_colors(16)
_GRAYSCALE_256_COLORS = _generate_grayscale_colors(256)


class GrayscalePalette(FixedPalette):
    """Grayscale palette with configurable shade count."""

    def __init__(self, shades: int = 16) -> None:
        shades = max(2, min(256, shades))
        metadata = PaletteMetadata(
            name=f"Grayscale ({shades} shades)",
            system="Grayscale",
            description=f"{shades}-shade grayscale palette from black to white",
        )
        super().__init__(_generate_grayscale_colors(shades), metadata)


class Grayscale2Palette(FixedPalette):
    """2-shade grayscale palette (black and white)."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Grayscale (2 shades)",
            system="Grayscale",
            description="Black and white only",
        )
        super().__init__(_GRAYSCALE_2_COLORS, metadata)


class Grayscale4Palette(FixedPalette):
    """4-shade grayscale palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Grayscale (4 shades)",
            system="Grayscale",
            description="4-shade grayscale palette",
        )
        super().__init__(_GRAYSCALE_4_COLORS, metadata)


class Grayscale8Palette(FixedPalette):
    """8-shade grayscale palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Grayscale (8 shades)",
            system="Grayscale",
            description="8-shade grayscale palette",
        )
        super().__init__(_GRAYSCALE_8_COLORS, metadata)


class Grayscale16Palette(FixedPalette):
    """16-shade grayscale palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Grayscale (16 shades)",
            system="Grayscale",
            description="16-shade grayscale palette",
        )
        super().__init__(_GRAYSCALE_16_COLORS, metadata)


class Grayscale256Palette(FixedPalette):
    """Full 256-shade grayscale palette."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Grayscale (256 shades)",
            system="Grayscale",
            description="Full 8-bit grayscale palette",
        )
        super().__init__(_GRAYSCALE_256_COLORS, metadata)
