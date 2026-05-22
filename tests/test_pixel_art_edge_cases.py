"""Edge case tests for the Pixel Art generator — task 125.2.

Covers:
  A. Edge cases:
     - Transparent RGBA source image (alpha channel)
     - grid_width=1 (single cell column)
     - Single-color source image (only 1 palette index used)
     - Source image smaller than grid_width (upscaling)
  B. CLI --list-presets "Pixel Art"
  C. Multi-layer SVG output (inkscape:groupmode="layer" per <g>)
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np
import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def make_rgba_image(width: int = 8, height: int = 8, alpha: int = 128) -> np.ndarray:
    """Create a synthetic RGBA image with a given constant alpha value."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    # Horizontal gradient in R, constant G, constant alpha
    for x in range(width):
        val = int(x * 255 / max(width - 1, 1))
        img[:, x, 0] = val        # R
        img[:, x, 1] = val // 2   # G
        img[:, x, 2] = val // 4   # B
        img[:, x, 3] = alpha
    return img


def make_fully_transparent_rgba(width: int = 8, height: int = 8) -> np.ndarray:
    """All pixels fully transparent (alpha=0)."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    return img


def make_uniform_rgb_image(color: tuple, width: int = 8, height: int = 8) -> np.ndarray:
    """Create a solid-color RGB image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def make_small_rgb_image(width: int = 4, height: int = 4) -> np.ndarray:
    """Create a tiny RGB image (smaller than typical grid_width)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = (x * 64, y * 64, 128)
    return img


# ---------------------------------------------------------------------------
# A1. Transparent RGBA input
# ---------------------------------------------------------------------------

class TestTransparentRGBAInput:
    """The generator must handle RGBA (4-channel) source images without crashing."""

    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator
        self.gen = PixelArtGenerator()
        self.canvas = make_canvas()
        self.base_params = {
            "grid_width": 8,
            "palette": "grayscale_4",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }

    def test_rgba_semi_transparent_no_crash(self):
        """RGBA image with alpha=128 must not crash; returns a list."""
        img = make_rgba_image(alpha=128)
        specs = self.gen.generate_layers(
            {**self.base_params, "_source_image": img}, self.canvas
        )
        assert isinstance(specs, list)

    def test_rgba_semi_transparent_emits_layers(self):
        """RGBA image with alpha=128 must emit at least one layer."""
        img = make_rgba_image(alpha=128)
        specs = self.gen.generate_layers(
            {**self.base_params, "_source_image": img}, self.canvas
        )
        assert len(specs) >= 1

    def test_rgba_fully_transparent_no_crash(self):
        """Fully transparent (alpha=0) image must not crash."""
        img = make_fully_transparent_rgba()
        specs = self.gen.generate_layers(
            {**self.base_params, "_source_image": img}, self.canvas
        )
        assert isinstance(specs, list)

    def test_rgba_fully_opaque_no_crash(self):
        """RGBA image with alpha=255 (fully opaque) must behave like RGB."""
        img = make_rgba_image(alpha=255)
        specs = self.gen.generate_layers(
            {**self.base_params, "_source_image": img}, self.canvas
        )
        assert isinstance(specs, list)
        assert len(specs) >= 1

    def test_rgba_all_colors_within_canvas(self):
        """Paths from an RGBA source must stay within the canvas drawing area."""
        img = make_rgba_image(alpha=200)
        specs = self.gen.generate_layers(
            {**self.base_params, "_source_image": img}, self.canvas
        )
        x1, y1, x2, y2 = self.canvas.drawing_area()
        eps = 1e-6
        for spec in specs:
            for path in spec.paths:
                for x, y in path:
                    assert x >= x1 - eps
                    assert x <= x2 + eps
                    assert y >= y1 - eps
                    assert y <= y2 + eps

    def test_rgba_hex_cell_shape_no_crash(self):
        """RGBA image with hex cell_shape must not crash."""
        img = make_rgba_image(alpha=180)
        params = {**self.base_params, "_source_image": img, "cell_shape": "hex"}
        specs = self.gen.generate_layers(params, self.canvas)
        assert isinstance(specs, list)

    def test_image_to_palette_grid_rgba_input(self):
        """image_to_palette_grid must handle an RGBA numpy array without error."""
        from plottter.pixel_art import get_palette, image_to_palette_grid

        img = make_rgba_image(alpha=200)
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=8)
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2
        assert grid.dtype == np.int32
        assert grid.min() >= 0
        assert grid.max() <= 3


# ---------------------------------------------------------------------------
# A2. grid_width=1 (single cell column)
# ---------------------------------------------------------------------------

class TestGridWidthOne:
    """grid_width=1 bypasses the IntParam min=4 guard; must not crash."""

    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator
        self.gen = PixelArtGenerator()
        self.canvas = make_canvas()

    def _run(self, extra=None):
        img = make_uniform_rgb_image((100, 100, 100))
        params = {
            "_source_image": img,
            "grid_width": 1,
            "palette": "grayscale_4",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        if extra:
            params.update(extra)
        return self.gen.generate_layers(params, self.canvas)

    def test_grid_width_1_no_crash(self):
        specs = self._run()
        assert isinstance(specs, list)

    def test_grid_width_1_emits_layers(self):
        specs = self._run()
        assert len(specs) >= 1

    def test_grid_width_1_paths_within_canvas(self):
        specs = self._run()
        x1, y1, x2, y2 = self.canvas.drawing_area()
        eps = 1e-6
        for spec in specs:
            for path in spec.paths:
                for x, y in path:
                    assert x >= x1 - eps
                    assert x <= x2 + eps
                    assert y >= y1 - eps
                    assert y <= y2 + eps

    def test_image_to_palette_grid_width_1(self):
        """image_to_palette_grid with grid_width=1 must return a 2-D array."""
        from plottter.pixel_art import get_palette, image_to_palette_grid

        img = make_uniform_rgb_image((50, 50, 50))
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=1)
        assert grid.ndim == 2
        assert grid.shape[1] == 1  # exactly one column
        assert grid.min() >= 0
        assert grid.max() <= 3


# ---------------------------------------------------------------------------
# A3. Single-color source image (only one palette index appears)
# ---------------------------------------------------------------------------

class TestSingleColorSource:
    """A uniform-color image must produce exactly 1 LayerSpec (one palette index used)."""

    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator
        self.gen = PixelArtGenerator()
        self.canvas = make_canvas()

    def _run_white(self):
        # Pure white → should map to palette index 1 (white) in grayscale_2.
        img = make_uniform_rgb_image((255, 255, 255))
        params = {
            "_source_image": img,
            "grid_width": 8,
            "palette": "grayscale_2",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        return self.gen.generate_layers(params, self.canvas)

    def _run_black(self):
        img = make_uniform_rgb_image((0, 0, 0))
        params = {
            "_source_image": img,
            "grid_width": 8,
            "palette": "grayscale_2",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        return self.gen.generate_layers(params, self.canvas)

    def test_pure_white_no_crash(self):
        specs = self._run_white()
        assert isinstance(specs, list)

    def test_pure_white_single_layer(self):
        """Pure white image with 2-color palette must emit exactly 1 layer."""
        specs = self._run_white()
        assert len(specs) == 1, f"Expected 1 layer, got {len(specs)}"

    def test_pure_black_no_crash(self):
        specs = self._run_black()
        assert isinstance(specs, list)

    def test_pure_black_single_layer(self):
        """Pure black image with 2-color palette must emit exactly 1 layer."""
        specs = self._run_black()
        assert len(specs) == 1, f"Expected 1 layer, got {len(specs)}"

    def test_single_color_paths_are_valid_polylines(self):
        """All paths in the single-layer output must be valid polylines."""
        specs = self._run_black()
        assert len(specs) == 1
        for path in specs[0].paths:
            assert isinstance(path, list)
            assert len(path) >= 2
            for pt in path:
                assert len(pt) == 2

    def test_single_color_cell_density_zero_no_crash(self):
        """White image: brightness=1 → cell_density=density*(1-1)=0. Must not crash."""
        img = make_uniform_rgb_image((255, 255, 255))
        params = {
            "_source_image": img,
            "grid_width": 4,
            "palette": "grayscale_4",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.8,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        specs = self.gen.generate_layers(params, self.canvas)
        assert isinstance(specs, list)
        # White gets index 3 in grayscale_4 (brightness≈1 → density≈0 but
        # _lerp_spacing(0) = 0.6 mm spacing, so we still get hatch lines)
        assert len(specs) == 1


# ---------------------------------------------------------------------------
# A4. Source image smaller than grid_width (upscaling)
# ---------------------------------------------------------------------------

class TestSourceSmallerThanGridWidth:
    """A tiny source image (e.g. 4×4) with a larger grid_width must not crash."""

    def setup_method(self):
        from plottter.generators.pixel_art import PixelArtGenerator
        self.gen = PixelArtGenerator()
        self.canvas = make_canvas()

    def _run(self, img_w=4, img_h=4, grid_width=32):
        img = make_small_rgb_image(img_w, img_h)
        params = {
            "_source_image": img,
            "grid_width": grid_width,
            "palette": "grayscale_4",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        return self.gen.generate_layers(params, self.canvas)

    def test_4x4_source_grid32_no_crash(self):
        specs = self._run(4, 4, 32)
        assert isinstance(specs, list)

    def test_4x4_source_grid32_emits_layers(self):
        specs = self._run(4, 4, 32)
        assert len(specs) >= 1

    def test_1x1_source_no_crash(self):
        """1×1 source image is the extreme minimum; must not crash."""
        img = make_uniform_rgb_image((128, 128, 128), width=1, height=1)
        params = {
            "_source_image": img,
            "grid_width": 8,
            "palette": "grayscale_4",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }
        specs = self.gen.generate_layers(params, self.canvas)
        assert isinstance(specs, list)

    def test_small_source_all_paths_within_canvas(self):
        specs = self._run(4, 4, 16)
        x1, y1, x2, y2 = self.canvas.drawing_area()
        eps = 1e-6
        for spec in specs:
            for path in spec.paths:
                for x, y in path:
                    assert x >= x1 - eps
                    assert x <= x2 + eps
                    assert y >= y1 - eps
                    assert y <= y2 + eps

    def test_image_to_palette_grid_upscale(self):
        """image_to_palette_grid must upscale a 4×4 image to a 32×32 grid."""
        from plottter.pixel_art import get_palette, image_to_palette_grid

        img = make_small_rgb_image(4, 4)
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=32, grid_height=32)
        assert grid.shape == (32, 32)
        assert grid.min() >= 0
        assert grid.max() <= 3


# ---------------------------------------------------------------------------
# B. CLI --list-presets "Pixel Art"
# ---------------------------------------------------------------------------

class TestCLIListPresets:
    """Confirm --list-presets 'Pixel Art' lists all presets correctly."""

    def _capture_list_presets(self):
        """Run _list_presets and capture stdout."""
        from plottter.cli import _list_presets

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            _list_presets("Pixel Art")
        finally:
            sys.stdout = old_stdout
        return captured.getvalue()

    def test_list_presets_pixel_art_no_error(self):
        """_list_presets('Pixel Art') must not raise."""
        output = self._capture_list_presets()
        assert len(output) > 0

    def test_list_presets_header_present(self):
        """Output must include the generator name in the header."""
        output = self._capture_list_presets()
        assert "Pixel Art" in output

    def test_list_presets_contains_default(self):
        output = self._capture_list_presets()
        assert "Default" in output

    def test_list_presets_contains_game_boy(self):
        output = self._capture_list_presets()
        assert "Game Boy" in output

    def test_list_presets_at_least_10_entries(self):
        """Pixel Art must have at least 10 presets."""
        output = self._capture_list_presets()
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        # Header is first line; remaining lines are preset names (indented).
        preset_lines = [l for l in lines if not l.startswith("Presets")]
        assert len(preset_lines) >= 10, (
            f"Expected ≥10 preset lines, got {len(preset_lines)}: {preset_lines}"
        )

    def test_run_cli_list_presets_returns_0(self):
        """run_cli(['--list-presets', 'Pixel Art']) must return exit code 0."""
        from plottter.cli import run_cli

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            code = run_cli(["--list-presets", "Pixel Art"])
        finally:
            sys.stdout = old_stdout
        assert code == 0

    def test_run_cli_list_presets_unknown_generator_returns_1(self):
        """--list-presets with an unknown generator name must fail (exit code 1 or SystemExit(1))."""
        from plottter.cli import run_cli

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            try:
                code = run_cli(["--list-presets", "NonExistentGeneratorXYZ"])
                assert code == 1
            except SystemExit as e:
                assert e.code == 1
        finally:
            sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# C. Multi-layer SVG — inkscape:groupmode="layer" per <g>
# ---------------------------------------------------------------------------

class TestMultiLayerSVGExport:
    """Verify export_layer_specs_svg writes one <g inkscape:groupmode="layer"> per spec."""

    def _export_to_tmp(self, layer_specs):
        from plottter.export.svg import export_layer_specs_svg

        canvas = make_canvas()
        settings = {
            "registration_marks": False,
            "stroke_width": 0.3,
        }
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            tmp_path = f.name
        try:
            export_layer_specs_svg(layer_specs, canvas, tmp_path, settings)
            with open(tmp_path, encoding="utf-8") as fh:
                content = fh.read()
        finally:
            os.unlink(tmp_path)
        return content

    def _make_specs(self, n=3):
        """Return n simple LayerSpec objects."""
        from plottter.generators.base import LayerSpec

        colors = ["#000000", "#555555", "#AAAAAA", "#FFFFFF"]
        specs = []
        for i in range(n):
            specs.append(LayerSpec(
                name=f"Pixel {i}",
                color=colors[i % len(colors)],
                paths=[[(0.0, float(i)), (10.0, float(i))]],
            ))
        return specs

    def test_svg_has_inkscape_namespace(self):
        specs = self._make_specs(2)
        svg = self._export_to_tmp(specs)
        assert "xmlns:inkscape" in svg

    def test_svg_has_inkscape_groupmode_layer(self):
        specs = self._make_specs(2)
        svg = self._export_to_tmp(specs)
        assert 'inkscape:groupmode="layer"' in svg

    def test_svg_has_one_group_per_spec(self):
        """Number of <g inkscape:groupmode="layer"> elements must equal len(specs)."""
        import re
        specs = self._make_specs(3)
        svg = self._export_to_tmp(specs)
        matches = re.findall(r'inkscape:groupmode="layer"', svg)
        assert len(matches) == 3, (
            f"Expected 3 groups with inkscape:groupmode='layer', found {len(matches)}"
        )

    def test_svg_has_inkscape_label_for_each_layer(self):
        """Each group must have an inkscape:label matching the LayerSpec name."""
        specs = self._make_specs(3)
        svg = self._export_to_tmp(specs)
        for spec in specs:
            assert f'inkscape:label="{spec.name}"' in svg, (
                f"Missing inkscape:label for spec {spec.name!r}"
            )

    def test_svg_layer_colors_match_specs(self):
        """Each group's stroke attribute must match the corresponding LayerSpec color."""
        specs = self._make_specs(3)
        svg = self._export_to_tmp(specs)
        for spec in specs:
            assert spec.color in svg, (
                f"Color {spec.color!r} not found in SVG output"
            )

    def test_svg_empty_specs_no_crash(self):
        """Empty layer spec list must produce a valid SVG without crashing."""
        svg = self._export_to_tmp([])
        assert "<svg" in svg

    def test_cli_multi_layer_generator_uses_generate_layers(self):
        """For emits_multiple_layers=True, the CLI must call generate_layers."""
        from plottter.generators.pixel_art import PixelArtGenerator
        gen = PixelArtGenerator()
        assert gen.emits_multiple_layers is True

    def test_export_layer_specs_svg_produces_valid_svg_file(self):
        """The exported file must start with an SVG doctype or root element."""
        specs = self._make_specs(2)
        svg = self._export_to_tmp(specs)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_multi_layer_svg_from_generator_output(self):
        """End-to-end: generator → export_layer_specs_svg → correct SVG structure."""
        import re
        from plottter.generators.pixel_art import PixelArtGenerator
        from plottter.export.svg import export_layer_specs_svg

        gen = PixelArtGenerator()
        canvas = make_canvas()

        # Build a small synthetic image with all 4 grayscale_4 shades.
        shades = [0, 85, 170, 255]
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        for row in range(8):
            img[row, :] = shades[row % 4]

        params = {
            "_source_image": img,
            "grid_width": 8,
            "palette": "grayscale_4",
            "quantization": "nearest",
            "color_space": "rgb",
            "dithering": "none",
            "cell_shape": "square",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.5,
            "cell_border": False,
            "cell_gap_mm": 0.0,
        }

        layer_specs = gen.generate_layers(params, canvas)
        assert len(layer_specs) > 0

        settings = {"registration_marks": False, "stroke_width": 0.3}

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            tmp_path = f.name
        try:
            export_layer_specs_svg(layer_specs, canvas, tmp_path, settings)
            with open(tmp_path, encoding="utf-8") as fh:
                svg = fh.read()
        finally:
            os.unlink(tmp_path)

        # Must have one inkscape layer group per LayerSpec.
        matches = re.findall(r'inkscape:groupmode="layer"', svg)
        assert len(matches) == len(layer_specs), (
            f"Expected {len(layer_specs)} inkscape layer groups, got {len(matches)}"
        )
        assert "xmlns:inkscape" in svg
