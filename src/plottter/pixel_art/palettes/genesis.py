"""Sega Genesis/Mega Drive palette."""

from plottter.pixel_art.palettes.palette import GeneratedPalette, PaletteMetadata


class GenesisPalette(GeneratedPalette):
    """Sega Genesis/Mega Drive 9-bit color palette (512 colors)."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Sega Genesis",
            system="Sega Genesis / Mega Drive",
            year=1988,
            description="Sega Genesis 9-bit color palette (3 bits per channel, 512 colors)",
            source="https://segaretro.org/Sega_Mega_Drive/Palettes_and_CRAM",
        )
        super().__init__(3, metadata)


MegaDrivePalette = GenesisPalette
