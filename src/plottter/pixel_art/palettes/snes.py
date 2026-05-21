"""SNES (Super Nintendo Entertainment System) palette."""

from plottter.pixel_art.palettes.palette import GeneratedPalette, PaletteMetadata


class SNESPalette(GeneratedPalette):
    """SNES 15-bit color palette (32,768 colors)."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="SNES",
            system="Super Nintendo Entertainment System",
            year=1990,
            description="SNES 15-bit color palette (5 bits per channel, 32768 colors)",
            source="https://en.wikibooks.org/wiki/Super_NES_Programming/Backgrounds",
        )
        super().__init__(5, metadata)
