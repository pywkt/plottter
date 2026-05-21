"""Modern limited palettes for pixel art (Endesga32, Sweetie16, DB32, Resurrect64)."""

from typing import List

from plottter.pixel_art.palettes.palette import RGB, FixedPalette, PaletteMetadata

_ENDESGA32_COLORS: List[RGB] = [
    (190, 74, 47), (215, 118, 67), (234, 212, 170), (228, 166, 114),
    (184, 111, 80), (115, 62, 57), (62, 39, 49), (162, 38, 51),
    (228, 59, 68), (247, 118, 34), (254, 174, 52), (254, 231, 97),
    (99, 199, 77), (62, 137, 72), (38, 92, 66), (25, 60, 62),
    (18, 78, 137), (0, 153, 219), (44, 232, 245), (255, 255, 255),
    (192, 203, 220), (139, 155, 180), (90, 105, 136), (58, 68, 102),
    (38, 43, 68), (24, 20, 37), (255, 0, 68), (104, 56, 108),
    (181, 80, 136), (246, 117, 122), (232, 183, 150), (194, 133, 105),
]

_SWEETIE16_COLORS: List[RGB] = [
    (26, 28, 44), (93, 39, 93), (177, 62, 83), (239, 125, 87),
    (255, 205, 117), (167, 240, 112), (56, 183, 100), (37, 113, 121),
    (41, 54, 111), (59, 93, 201), (65, 166, 246), (115, 239, 247),
    (244, 244, 244), (148, 176, 194), (86, 108, 134), (51, 60, 87),
]

_DB32_COLORS: List[RGB] = [
    (0, 0, 0), (34, 32, 52), (69, 40, 60), (102, 57, 49),
    (143, 86, 59), (223, 113, 38), (217, 160, 102), (238, 195, 154),
    (251, 242, 54), (153, 229, 80), (106, 190, 48), (55, 148, 110),
    (75, 105, 47), (82, 75, 36), (50, 60, 57), (63, 63, 116),
    (48, 96, 130), (91, 110, 225), (99, 155, 255), (95, 205, 228),
    (203, 219, 252), (255, 255, 255), (155, 173, 183), (132, 126, 135),
    (105, 106, 106), (89, 86, 82), (118, 66, 138), (172, 50, 50),
    (217, 87, 99), (215, 123, 186), (143, 151, 74), (138, 111, 48),
]

_RESURRECT64_COLORS: List[RGB] = [
    (46, 34, 47), (62, 53, 70), (98, 85, 101), (150, 108, 108),
    (171, 148, 122), (105, 79, 98), (127, 112, 138), (155, 171, 178),
    (199, 220, 208), (255, 255, 255), (110, 39, 39), (179, 56, 49),
    (234, 79, 54), (245, 125, 74), (174, 35, 52), (232, 59, 59),
    (251, 107, 29), (247, 150, 23), (249, 194, 43), (122, 48, 69),
    (158, 69, 57), (205, 104, 61), (230, 144, 78), (251, 185, 84),
    (76, 62, 36), (103, 102, 51), (162, 169, 71), (213, 224, 75),
    (251, 255, 134), (22, 90, 76), (35, 144, 99), (30, 188, 115),
    (145, 219, 105), (205, 223, 108), (49, 54, 56), (55, 78, 74),
    (84, 126, 100), (146, 169, 132), (178, 186, 144), (11, 94, 101),
    (11, 138, 143), (14, 175, 155), (48, 225, 185), (143, 248, 226),
    (50, 51, 83), (72, 74, 119), (77, 101, 180), (77, 155, 230),
    (143, 211, 255), (69, 41, 63), (107, 62, 117), (144, 94, 169),
    (168, 132, 243), (234, 173, 237), (117, 60, 84), (162, 75, 111),
    (207, 101, 127), (237, 128, 153), (131, 28, 93), (195, 36, 84),
    (240, 79, 120), (246, 129, 129), (252, 167, 144), (253, 203, 176),
]

_ENDESGA64_COLORS: List[RGB] = [
    (255, 0, 64), (19, 19, 19), (27, 27, 27), (39, 39, 39),
    (61, 61, 61), (93, 93, 93), (133, 133, 133), (180, 180, 180),
    (255, 255, 255), (199, 207, 221), (146, 161, 185), (101, 115, 146),
    (66, 76, 110), (42, 47, 78), (26, 25, 50), (14, 7, 27),
    (28, 18, 28), (57, 31, 33), (93, 44, 40), (138, 72, 54),
    (191, 111, 74), (230, 156, 105), (246, 202, 159), (249, 230, 207),
    (237, 171, 80), (224, 116, 56), (198, 69, 36), (142, 37, 29),
    (255, 80, 0), (237, 118, 20), (255, 162, 20), (255, 200, 37),
    (255, 235, 87), (211, 252, 126), (153, 230, 95), (90, 197, 79),
    (51, 152, 75), (30, 111, 80), (19, 76, 76), (12, 46, 68),
    (0, 57, 109), (0, 105, 170), (0, 152, 220), (0, 205, 249),
    (12, 241, 255), (148, 253, 255), (253, 210, 237), (243, 137, 245),
    (219, 63, 253), (122, 9, 250), (48, 3, 217), (12, 2, 147),
    (3, 25, 63), (59, 20, 67), (98, 36, 97), (147, 56, 143),
    (202, 82, 201), (200, 80, 134), (246, 129, 135), (245, 85, 93),
    (234, 50, 60), (196, 36, 48), (137, 30, 43), (87, 28, 39),
]


class Endesga32Palette(FixedPalette):
    """Endesga 32 palette — 32 colors."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Endesga 32",
            system="Modern Pixel Art",
            year=2017,
            description="Popular 32-color palette for pixel art games",
            source="https://lospec.com/palette-list/endesga-32",
        )
        super().__init__(_ENDESGA32_COLORS, metadata)


class Sweetie16Palette(FixedPalette):
    """Sweetie 16 palette — 16 colors."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Sweetie 16",
            system="Modern Pixel Art",
            year=2018,
            description="Clean and versatile 16-color palette",
            source="https://lospec.com/palette-list/sweetie-16",
        )
        super().__init__(_SWEETIE16_COLORS, metadata)


class DB32Palette(FixedPalette):
    """DawnBringer's 32 palette — 32 colors."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="DawnBringer 32",
            system="Modern Pixel Art",
            year=2011,
            description="Classic pixel art palette by DawnBringer",
            source="https://lospec.com/palette-list/dawnbringer-32",
        )
        super().__init__(_DB32_COLORS, metadata)


class Resurrect64Palette(FixedPalette):
    """Resurrect 64 palette — 64 colors."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Resurrect 64",
            system="Modern Pixel Art",
            year=2018,
            description="Large modern retro palette with 64 colors",
            source="https://lospec.com/palette-list/resurrect-64",
        )
        super().__init__(_RESURRECT64_COLORS, metadata)


class Endesga64Palette(FixedPalette):
    """Endesga 64 palette — 64 colors."""

    def __init__(self) -> None:
        metadata = PaletteMetadata(
            name="Endesga 64",
            system="Modern Pixel Art",
            year=2020,
            description="Premium 64-color palette with high contrast and saturation",
            source="https://lospec.com/palette-list/endesga-64",
        )
        super().__init__(_ENDESGA64_COLORS, metadata)
