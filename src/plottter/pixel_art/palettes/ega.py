"""EGA (Enhanced Graphics Adapter) palette."""

from plottter.pixel_art.palettes.palette import GeneratedPalette, PaletteMetadata


class EGAPalette(GeneratedPalette):
    """IBM EGA 64-color palette (2 bits per channel)."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="EGA",
            system="IBM EGA",
            year=1984,
            description="IBM Enhanced Graphics Adapter 64-color palette (2 bits per channel)",
            source="https://en.wikipedia.org/wiki/Enhanced_Graphics_Adapter",
        )
        super().__init__(2, metadata)
