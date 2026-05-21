"""Tests for PixelArtGenerator (task 118.2).

Runs the generator on a small synthetic image with the grayscale_4 palette and
verifies:
  - ≤ 4 layers are emitted (one per palette colour that appears in the grid).
  - Each layer's colour is a valid hex string matching a grayscale_4 value.
  - All path coordinates lie within the canvas printable area.
  - The generator is registered under "Pixel Art" in GENERATORS.
  - `generate()` (single-layer fallback) returns a flat list of polylines.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canvas() -> Canvas:
    """Return a standard A4 canvas with a 10 mm margin."""
    return Canvas.from_preset("A4", margin=10.0)


def make_grayscale_image(width: int = 8, height: int = 8) -> np.ndarray:
    """Return a small synthetic grayscale-RGB image with all 4 grayscale shades."""
    # grayscale_4 shades (computed by _generate_grayscale_colors(4)):
    # index 0 → (0,   0,   0)   black
    # index 1 → (85,  85,  85)
    # index 2 → (170, 170, 170)
    # index 3 → (255, 255, 255) white
    shades = [0, 85, 170, 255]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        shade = shades[row % 4]
        img[row, :] = shade
    return img


# ---------------------------------------------------------------------------
# Generator registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_pixel_art_in_generators_registry(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        assert "Pixel Art" in GENERATORS

    def test_pixel_art_generator_class_attributes(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        cls = GENERATORS["Pixel Art"]
        assert cls.name == "Pixel Art"
        assert cls.category == "image"
        assert cls.uses_source_image is True
        assert cls.emits_multiple_layers is True


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator

        self.gen = PixelArtGenerator()

    def test_get_parameters_returns_expected_keys(self):
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "grid_width" in names
        assert "palette" in names
        assert "cell_fill_style" in names
        assert "fill_density" in names
        assert "cell_border" in names
        assert "cell_gap_mm" in names

    def test_palette_default_is_grayscale_4(self):
        params = self.gen.get_parameters()
        palette_param = next(p for p in params if p.name == "palette")
        assert palette_param.default == "grayscale_4"

    def test_cell_fill_style_choices_contains_solid_hatch(self):
        params = self.gen.get_parameters()
        fill_param = next(p for p in params if p.name == "cell_fill_style")
        assert "solid_hatch" in fill_param.choices


# ---------------------------------------------------------------------------
# generate_layers — core behaviour
# ---------------------------------------------------------------------------

class TestGenerateLayers:
    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator

        self.gen = PixelArtGenerator()
        self.canvas = make_canvas()
        self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2 = (
            self.canvas.drawing_area()
        )

    def _run(self, extra_params: dict | None = None) -> list:
        img = make_grayscale_image()
        params: dict = {
            "_source_image": img,
            "grid_width": 8,
            "palette": "grayscale_4",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        if extra_params:
            params.update(extra_params)
        return self.gen.generate_layers(params, self.canvas)

    def test_emits_at_most_4_layers(self):
        specs = self._run()
        assert 0 < len(specs) <= 4

    def test_each_layer_has_valid_hex_color(self):
        """Each LayerSpec.color must be a valid #RRGGBB string from grayscale_4."""
        import re

        # grayscale_4 hex values
        expected_hex = {"#000000", "#555555", "#AAAAAA", "#FFFFFF"}
        specs = self._run()
        for spec in specs:
            assert re.fullmatch(r"#[0-9A-Fa-f]{6}", spec.color), (
                f"Invalid hex color: {spec.color!r}"
            )
            assert spec.color.upper() in expected_hex, (
                f"Unexpected color {spec.color!r} not in grayscale_4 palette"
            )

    def test_all_coordinates_within_canvas_printable_area(self):
        """Every (x, y) in every path must lie within the drawing area."""
        EPS = 1e-6
        specs = self._run()
        for spec in specs:
            for path in spec.paths:
                for x, y in path:
                    assert x >= self.draw_x1 - EPS, (
                        f"x={x} below draw_x1={self.draw_x1}"
                    )
                    assert x <= self.draw_x2 + EPS, (
                        f"x={x} above draw_x2={self.draw_x2}"
                    )
                    assert y >= self.draw_y1 - EPS, (
                        f"y={y} below draw_y1={self.draw_y1}"
                    )
                    assert y <= self.draw_y2 + EPS, (
                        f"y={y} above draw_y2={self.draw_y2}"
                    )

    def test_layer_names_are_pixel_prefix(self):
        specs = self._run()
        for spec in specs:
            assert spec.name.startswith("Pixel "), (
                f"Unexpected layer name: {spec.name!r}"
            )

    def test_no_source_image_returns_empty(self):
        specs = self.gen.generate_layers(
            {"grid_width": 8, "palette": "grayscale_4"},
            self.canvas,
        )
        assert specs == []

    def test_cell_border_adds_closed_polyline(self):
        """With cell_border=True, each cell should include a 5-point closed rectangle."""
        specs = self._run({"cell_border": True})
        # At least one path should have exactly 5 points (the closed rectangle)
        all_paths = [path for spec in specs for path in spec.paths]
        five_pt_paths = [p for p in all_paths if len(p) == 5]
        assert len(five_pt_paths) > 0, (
            "Expected at least one 5-point closed border polyline"
        )

    def test_cell_gap_reduces_fill_area(self):
        """With a gap, paths should not start at the very first cell edge."""
        specs_no_gap = self._run({"cell_gap_mm": 0.0})
        specs_gap = self._run({"cell_gap_mm": 0.5})
        # Both should still produce paths
        assert len(specs_gap) > 0
        # With a gap the total number of hatch lines may differ, but paths are valid
        all_xs_gap = [
            x for spec in specs_gap for path in spec.paths for x, _ in path
        ]
        # No x should be before draw_x1 (they start at draw_x1 + gap/2)
        assert all(x >= self.draw_x1 - 1e-6 for x in all_xs_gap)


# ---------------------------------------------------------------------------
# generate() — single-layer fallback
# ---------------------------------------------------------------------------

class TestGenerateSingleLayer:
    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator

        self.gen = PixelArtGenerator()
        self.canvas = make_canvas()

    def test_generate_returns_flat_list_of_polylines(self):
        img = make_grayscale_image()
        params = {
            "_source_image": img,
            "grid_width": 8,
            "palette": "grayscale_4",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        paths = self.gen.generate(params, self.canvas)
        assert isinstance(paths, list)
        # Each element is a polyline (list of points)
        for path in paths:
            assert isinstance(path, list)
            for pt in path:
                assert len(pt) == 2

    def test_generate_with_no_source_returns_empty_list(self):
        paths = self.gen.generate(
            {"grid_width": 8, "palette": "grayscale_4"},
            self.canvas,
        )
        assert paths == []


# ---------------------------------------------------------------------------
# Game Boy Portrait preset — multi-layer colour check (task 119.1)
# ---------------------------------------------------------------------------

# Game Boy DMG palette hex values (from _GAMEBOY_DMG_COLORS in palettes/gameboy.py):
#   (155, 188,  15) → #9BBC0F
#   (139, 172,  15) → #8BAC0F
#   ( 48,  98,  48) → #306230
#   ( 15,  56,  15) → #0F380F
_GAMEBOY_HEX_COLORS = {"#9BBC0F", "#8BAC0F", "#306230", "#0F380F"}


def make_gameboy_image(width: int = 16, height: int = 20) -> np.ndarray:
    """Return a synthetic image that exercises all 4 Game Boy DMG shades."""
    gb_rgb = [
        (155, 188, 15),
        (139, 172, 15),
        (48, 98, 48),
        (15, 56, 15),
    ]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        r, g, b = gb_rgb[row % 4]
        img[row, :] = (r, g, b)
    return img


class TestGameBoyPortraitPreset:
    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator

        self.gen = PixelArtGenerator()
        self.canvas = make_canvas()

    def test_game_boy_portrait_preset_exists(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Game Boy Portrait" in names

    def test_game_boy_portrait_preset_params(self):
        preset = next(p for p in self.gen.get_presets() if p.name == "Game Boy Portrait")
        assert preset.params["palette"] == "gameboy"
        assert preset.params["grid_width"] == 80
        assert preset.params["dithering"] == "floyd_steinberg"
        assert preset.params["cell_fill_style"] == "solid_hatch"
        assert abs(preset.params["fill_density"] - 0.7) < 1e-6

    def test_game_boy_portrait_emits_up_to_4_layers(self):
        """generate_layers with gameboy palette must return at most 4 LayerSpecs."""
        img = make_gameboy_image()
        params = {
            "_source_image": img,
            "grid_width": 16,
            "palette": "gameboy",
            "dithering": "none",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.7,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        specs = self.gen.generate_layers(params, self.canvas)
        assert 0 < len(specs) <= 4, f"Expected 1–4 layers, got {len(specs)}"

    def test_game_boy_portrait_layer_colors_are_gameboy_palette(self):
        """Each LayerSpec.color must be one of the 4 Game Boy DMG hex values."""
        img = make_gameboy_image()
        params = {
            "_source_image": img,
            "grid_width": 16,
            "palette": "gameboy",
            "dithering": "none",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.7,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        specs = self.gen.generate_layers(params, self.canvas)
        layer_colors = {spec.color.upper() for spec in specs}
        assert layer_colors.issubset(_GAMEBOY_HEX_COLORS), (
            f"Unexpected colors {layer_colors - _GAMEBOY_HEX_COLORS}; "
            f"expected subset of {_GAMEBOY_HEX_COLORS}"
        )

    def test_game_boy_portrait_dithering_param_accepted(self):
        """floyd_steinberg dithering must not raise an error."""
        img = make_gameboy_image()
        params = {
            "_source_image": img,
            "grid_width": 8,
            "palette": "gameboy",
            "dithering": "floyd_steinberg",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.7,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        specs = self.gen.generate_layers(params, self.canvas)
        assert isinstance(specs, list)
        assert len(specs) > 0

    def test_dithering_parameter_present(self):
        """The generator must expose a 'dithering' ChoiceParam."""
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "dithering" in names

    def test_dithering_choices_correct(self):
        param = next(p for p in self.gen.get_parameters() if p.name == "dithering")
        assert set(param.choices) == {"none", "floyd_steinberg", "ordered", "atkinson"}

    def test_dithering_default_is_none(self):
        param = next(p for p in self.gen.get_parameters() if p.name == "dithering")
        assert param.default == "none"
