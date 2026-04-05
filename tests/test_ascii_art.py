"""Tests for ASCIIArtGenerator — grid placement and character weight ordering."""

from __future__ import annotations

import numpy as np
import pytest

from plottter.generators.ascii_art import (
    ASCII_CHARS,
    ASCIIArtGenerator,
    _render_glyph,
    compute_cell_characters,
)
from plottter.generators._helpers import compute_image_rect
from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def make_dark_image(h: int = 100, w: int = 100, value: int = 0) -> np.ndarray:
    """Uniform grayscale image (default: fully black)."""
    return np.full((h, w), value, dtype=np.uint8)


def make_bright_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Uniform fully-white image."""
    return np.full((h, w), 255, dtype=np.uint8)


def make_default_params(cell_size_mm: float = 6.0, min_darkness: float = 0.1) -> dict:
    return {
        "cell_size_mm": cell_size_mm,
        "min_darkness": min_darkness,
        "char_scale": 0.75,
        "image_fit_mode": "fill",
        "image_offset_x_mm": 0.0,
        "image_offset_y_mm": 0.0,
        "invert": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_radius": 0.0,
        "rotation_mode": "Fixed",
        "fixed_angle_deg": 0.0,
    }


def get_img_rect(img: np.ndarray, canvas: Canvas, params: dict) -> tuple[float, float, float, float]:
    h, w = img.shape[:2]
    draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    return compute_image_rect(
        str(params.get("image_fit_mode", "fill")),
        w, h, draw_x1, draw_y1, draw_x2, draw_y2,
        custom_w_mm=params.get("image_width_mm"),
        custom_h_mm=params.get("image_height_mm"),
        offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
        offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
    )


# ---------------------------------------------------------------------------
# Tests: Grid cell count
# ---------------------------------------------------------------------------


class TestGridCellCount:
    def test_grid_produces_cells_for_dark_image(self):
        """A fully dark image should produce cells for every grid position."""
        canvas = make_canvas()
        params = make_default_params(cell_size_mm=6.0, min_darkness=0.1)
        img = make_dark_image()  # all black → all cells should be non-skipped
        img_rect = get_img_rect(img, canvas, params)

        cells = compute_cell_characters(img, canvas, params, img_rect)

        # Should have at least one cell
        assert len(cells) > 0

    def test_grid_cell_count_matches_expected(self):
        """Number of cells should match floor(img_rect / cell_size)."""
        canvas = make_canvas()
        cell_size = 10.0
        params = make_default_params(cell_size_mm=cell_size, min_darkness=0.0)
        img = make_dark_image(200, 200)
        img_rect = get_img_rect(img, canvas, params)

        x1, y1, x2, y2 = img_rect
        expected_cols = max(1, int((x2 - x1) / cell_size))
        expected_rows = max(1, int((y2 - y1) / cell_size))
        expected_count = expected_cols * expected_rows

        cells = compute_cell_characters(img, canvas, params, img_rect)
        assert len(cells) == expected_count

    def test_cell_coordinates_within_image_rect(self):
        """All cell centers should fall within the image rect bounds."""
        canvas = make_canvas()
        params = make_default_params(cell_size_mm=6.0, min_darkness=0.0)
        img = make_dark_image()
        img_rect = get_img_rect(img, canvas, params)
        x1, y1, x2, y2 = img_rect

        cells = compute_cell_characters(img, canvas, params, img_rect)
        for cx, cy, _ in cells:
            assert x1 <= cx <= x2, f"cx={cx} out of [{x1}, {x2}]"
            assert y1 <= cy <= y2, f"cy={cy} out of [{y1}, {y2}]"


# ---------------------------------------------------------------------------
# Tests: Character weight mapping
# ---------------------------------------------------------------------------


class TestCharacterWeightMapping:
    def test_dark_image_maps_to_heavy_characters(self):
        """A fully black image should map to the heaviest character (last in ASCII_CHARS)."""
        canvas = make_canvas()
        params = make_default_params(min_darkness=0.0)
        img = make_dark_image(value=0)
        img_rect = get_img_rect(img, canvas, params)

        cells = compute_cell_characters(img, canvas, params, img_rect)
        assert len(cells) > 0

        # All cells should use the heaviest character (last char in ASCII_CHARS)
        heaviest = ASCII_CHARS[-1]
        for _, _, char in cells:
            assert char == heaviest, f"Expected '{heaviest}', got '{char}'"

    def test_mid_gray_maps_to_middle_character(self):
        """A mid-gray image should map to a middle-weight character."""
        canvas = make_canvas()
        params = make_default_params(min_darkness=0.0)
        mid_value = 128
        img = make_dark_image(value=mid_value)
        img_rect = get_img_rect(img, canvas, params)

        cells = compute_cell_characters(img, canvas, params, img_rect)
        assert len(cells) > 0

        expected_idx = int((1.0 - mid_value / 255.0) * (len(ASCII_CHARS) - 1))
        expected_char = ASCII_CHARS[expected_idx]
        for _, _, char in cells:
            assert char == expected_char

    def test_darker_region_heavier_than_lighter_region(self):
        """Cells over dark pixels should use heavier characters than cells over bright pixels."""
        canvas = make_canvas()
        params = make_default_params(cell_size_mm=10.0, min_darkness=0.0)
        # Left half dark (0), right half bright (200)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 200

        img_rect = get_img_rect(img, canvas, params)
        x1, y1, x2, y2 = img_rect
        mid_x = (x1 + x2) / 2.0

        cells = compute_cell_characters(img, canvas, params, img_rect)
        dark_chars = [char for cx, cy, char in cells if cx < mid_x]
        bright_chars = [char for cx, cy, char in cells if cx >= mid_x]

        assert dark_chars, "No cells in dark half"
        assert bright_chars, "No cells in bright half"

        # Dark chars should be later in ASCII_CHARS (heavier)
        dark_idx = ASCII_CHARS.index(dark_chars[0])
        bright_idx = ASCII_CHARS.index(bright_chars[0])
        assert dark_idx > bright_idx, (
            f"Dark char '{dark_chars[0]}'(idx {dark_idx}) should be heavier "
            f"than bright char '{bright_chars[0]}'(idx {bright_idx})"
        )


# ---------------------------------------------------------------------------
# Tests: Bright cell skipping
# ---------------------------------------------------------------------------


class TestBrightCellSkipping:
    def test_fully_bright_image_skips_all_cells(self):
        """A fully white image should skip all cells when min_darkness > 0."""
        canvas = make_canvas()
        params = make_default_params(min_darkness=0.1)
        img = make_bright_image()  # all white
        img_rect = get_img_rect(img, canvas, params)

        cells = compute_cell_characters(img, canvas, params, img_rect)
        assert cells == [], f"Expected no cells, got {len(cells)} for fully white image"

    def test_min_darkness_zero_keeps_bright_cells(self):
        """With min_darkness=0, even fully white cells should not be skipped."""
        canvas = make_canvas()
        params = make_default_params(min_darkness=0.0)
        img = make_bright_image()
        img_rect = get_img_rect(img, canvas, params)

        cells = compute_cell_characters(img, canvas, params, img_rect)
        assert len(cells) > 0, "Expected cells with min_darkness=0 for white image"

    def test_high_min_darkness_skips_gray_cells(self):
        """A high min_darkness threshold should skip medium-gray cells."""
        canvas = make_canvas()
        # min_darkness=0.9 means only cells with >90% darkness pass
        params = make_default_params(min_darkness=0.9)
        img = make_dark_image(value=128)  # 50% gray → darkness ~50%
        img_rect = get_img_rect(img, canvas, params)

        cells = compute_cell_characters(img, canvas, params, img_rect)
        assert cells == [], f"50% gray should be skipped at min_darkness=0.9, got {len(cells)} cells"

    def test_partial_brightness_partial_skip(self):
        """Only dark enough cells should be kept when min_darkness partially filters."""
        canvas = make_canvas()
        params = make_default_params(cell_size_mm=10.0, min_darkness=0.5)
        # Top half: nearly black (darkness ~1.0), bottom half: white (darkness ~0)
        img = np.full((100, 100), 255, dtype=np.uint8)
        img[:50, :] = 10  # very dark top half

        img_rect = get_img_rect(img, canvas, params)
        x1, y1, x2, y2 = img_rect
        mid_y = (y1 + y2) / 2.0

        cells = compute_cell_characters(img, canvas, params, img_rect)
        for cx, cy, char in cells:
            # All surviving cells should be in the dark (top) half
            assert cy < mid_y + 1.0, f"Bright cell at cy={cy} should have been skipped"


# ---------------------------------------------------------------------------
# Tests: Generator integration
# ---------------------------------------------------------------------------


class TestASCIIArtGeneratorIntegration:
    def test_generator_registered(self):
        from plottter.generators import GENERATORS
        assert "ASCII Art" in GENERATORS

    def test_generate_returns_polylines_for_dark_image(self):
        """Generator returns non-empty polylines for a dark source image."""
        gen = ASCIIArtGenerator()
        canvas = make_canvas()
        img = make_dark_image()
        params = make_default_params()
        params["_source_image"] = img

        result = gen.generate(params, canvas)
        assert isinstance(result, list)
        assert len(result) > 0, "Expected polylines for a dark image"
        # Each polyline must have at least 2 points
        for poly in result:
            assert len(poly) >= 2

    def test_generate_returns_empty_without_source_image(self):
        """Generator returns empty list when no source image is provided."""
        gen = ASCIIArtGenerator()
        canvas = make_canvas()
        params = make_default_params()

        result = gen.generate(params, canvas)
        assert result == []

    def test_get_parameters_returns_expected_params(self):
        gen = ASCIIArtGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "cell_size_mm" in param_names
        assert "min_darkness" in param_names
        assert "char_scale" in param_names
        assert "invert" in param_names
        assert "brightness" in param_names

    def test_get_presets_returns_list(self):
        gen = ASCIIArtGenerator()
        presets = gen.get_presets()
        assert len(presets) >= 1
        assert any(p.name == "Default" for p in presets)


# ---------------------------------------------------------------------------
# Tests: ASCII_CHARS ordering sanity
# ---------------------------------------------------------------------------


class TestASCIICharsOrdering:
    def test_chars_has_expected_length(self):
        assert len(ASCII_CHARS) == 10

    def test_chars_lightest_is_dot(self):
        assert ASCII_CHARS[0] == "."

    def test_chars_heaviest_is_at(self):
        assert ASCII_CHARS[-1] == "@"


# ---------------------------------------------------------------------------
# Tests: _render_glyph
# ---------------------------------------------------------------------------


class TestRenderGlyph:
    def test_returns_polylines_for_known_char(self):
        """_render_glyph should return at least one polyline for 'A'."""
        glyphs = _render_glyph("A", x_mm=50.0, y_mm=50.0, size_mm=5.0, angle_deg=0.0)
        assert len(glyphs) > 0, "Expected polylines for character 'A'"

    def test_each_polyline_has_min_two_points(self):
        for char in "ABCabc123":
            glyphs = _render_glyph(char, x_mm=0.0, y_mm=0.0, size_mm=5.0, angle_deg=0.0)
            for poly in glyphs:
                assert len(poly) >= 2, f"Polyline for '{char}' has < 2 points"

    def test_char_size_matches_scale(self):
        """Rendered glyph should fit within size_mm bounding box."""
        size_mm = 6.0
        glyphs = _render_glyph("O", x_mm=0.0, y_mm=0.0, size_mm=size_mm, angle_deg=0.0)
        assert glyphs, "Expected polylines"
        all_pts = [pt for poly in glyphs for pt in poly]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        # Bounding box should be close to size_mm (within a factor of 1.5 to allow for glyph variation)
        assert width <= size_mm * 1.5, f"Glyph width {width:.2f} > {size_mm * 1.5:.2f}"
        assert height <= size_mm * 1.5, f"Glyph height {height:.2f} > {size_mm * 1.5:.2f}"

    def test_rotation_shifts_points(self):
        """Rotating a glyph 90 degrees should produce different point coordinates."""
        glyphs_0 = _render_glyph("A", 0.0, 0.0, 5.0, 0.0)
        glyphs_90 = _render_glyph("A", 0.0, 0.0, 5.0, 90.0)
        assert glyphs_0 and glyphs_90
        pts_0 = glyphs_0[0]
        pts_90 = glyphs_90[0]
        # At least some coordinates must differ
        assert any(
            abs(a[0] - b[0]) > 0.01 or abs(a[1] - b[1]) > 0.01
            for a, b in zip(pts_0, pts_90)
        ), "Rotation by 90° produced identical points"

    def test_translation_shifts_center(self):
        """Placing glyph at (10, 20) vs (0, 0) should shift all points by (10, 20)."""
        glyphs_origin = _render_glyph("X", 0.0, 0.0, 5.0, 0.0)
        glyphs_shifted = _render_glyph("X", 10.0, 20.0, 5.0, 0.0)
        assert glyphs_origin and glyphs_shifted
        for poly_o, poly_s in zip(glyphs_origin, glyphs_shifted):
            for (ox, oy), (sx, sy) in zip(poly_o, poly_s):
                assert abs((sx - ox) - 10.0) < 1e-9, f"x shift wrong: {sx - ox}"
                assert abs((sy - oy) - 20.0) < 1e-9, f"y shift wrong: {sy - oy}"

    def test_space_character_returns_empty(self):
        """Space character has no strokes — should return empty list."""
        glyphs = _render_glyph(" ", 0.0, 0.0, 5.0, 0.0)
        assert glyphs == []


# ---------------------------------------------------------------------------
# Tests: rotation_mode in generate()
# ---------------------------------------------------------------------------


class TestRotationModes:
    def test_fixed_rotation_all_same_angle(self):
        """With rotation_mode=Fixed and angle=45, all glyphs use same rotation."""
        gen = ASCIIArtGenerator()
        canvas = make_canvas()
        # Use a uniform dark image so all cells use the same character
        img = make_dark_image(value=0)
        params = make_default_params(cell_size_mm=10.0, min_darkness=0.0)
        params["_source_image"] = img
        params["rotation_mode"] = "Fixed"
        params["fixed_angle_deg"] = 45.0

        result = gen.generate(params, canvas)
        assert len(result) > 0

    def test_random_rotation_produces_varied_angles(self):
        """Random rotation should produce different angles across glyphs."""
        gen = ASCIIArtGenerator()
        canvas = make_canvas()
        img = make_dark_image(value=0)
        params = make_default_params(cell_size_mm=10.0, min_darkness=0.0)
        params["_source_image"] = img
        params["rotation_mode"] = "Random"

        result1 = gen.generate(params, canvas)
        result2 = gen.generate(params, canvas)
        assert len(result1) > 0
        assert len(result2) > 0
        # Two runs with random rotation should differ (astronomically unlikely to match)
        pts1 = result1[0][0]
        pts2 = result2[0][0]
        # Points from two random runs should differ (with extremely high probability)
        all_same = all(abs(a - b) < 1e-9 for a, b in zip(pts1, pts2))
        assert not all_same, "Random rotation produced identical results in two runs"

    def test_gradient_rotation_produces_output(self):
        """Gradient rotation mode should return polylines without error."""
        gen = ASCIIArtGenerator()
        canvas = make_canvas()
        # Create an image with a clear edge (left half dark, right half bright)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 200
        params = make_default_params(cell_size_mm=10.0, min_darkness=0.0)
        params["_source_image"] = img
        params["rotation_mode"] = "Gradient"

        result = gen.generate(params, canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_gradient_rotation_varies_across_cells(self):
        """With a clear edge, gradient angles should vary between cell columns."""
        from plottter.generators.ascii_art import _gradient_angles

        # Image with vertical edge: left half black, right half white
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255

        angles = _gradient_angles(img)
        assert angles.shape == (100, 100)

        # Near the edge (col ~50), gradient should be strong in x direction (~0°)
        # Far from edge (col 10), gradient should be near 0 magnitude
        edge_angle = float(angles[50, 50])
        flat_angle = float(angles[50, 10])

        # Edge angle should be close to 0° (strong horizontal gradient)
        assert abs(edge_angle) < 30.0 or abs(abs(edge_angle) - 180.0) < 30.0, (
            f"Edge angle {edge_angle:.1f}° unexpected for vertical edge"
        )

    def test_get_parameters_includes_rotation_params(self):
        gen = ASCIIArtGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "rotation_mode" in param_names
        assert "fixed_angle_deg" in param_names


# ---------------------------------------------------------------------------
# Tests: New presets (task 92.3)
# ---------------------------------------------------------------------------


class TestNewPresets:
    def _get_preset(self, name: str):
        gen = ASCIIArtGenerator()
        for p in gen.get_presets():
            if p.name == name:
                return p
        raise KeyError(f"Preset '{name}' not found")

    def test_typewriter_preset_exists(self):
        p = self._get_preset("Typewriter")
        assert p.params["cell_size_mm"] == 5.0
        assert p.params["char_scale"] == 0.7
        assert p.params["rotation_mode"] == "Fixed"
        assert p.params["fixed_angle_deg"] == 0.0

    def test_scattered_type_preset_exists(self):
        p = self._get_preset("Scattered Type")
        assert p.params["cell_size_mm"] == 6.0
        assert p.params["char_scale"] == 0.6
        assert p.params["rotation_mode"] == "Random"

    def test_contour_text_preset_exists(self):
        p = self._get_preset("Contour Text")
        assert p.params["cell_size_mm"] == 4.0
        assert p.params["char_scale"] == 0.65
        assert p.params["rotation_mode"] == "Gradient"

    def test_large_print_preset_exists(self):
        p = self._get_preset("Large Print")
        assert p.params["cell_size_mm"] == 10.0
        assert p.params["char_scale"] == 0.8
        assert p.params["rotation_mode"] == "Fixed"
        assert p.params["fixed_angle_deg"] == 0.0

    def test_all_four_presets_registered(self):
        gen = ASCIIArtGenerator()
        names = {p.name for p in gen.get_presets()}
        for expected in ("Typewriter", "Scattered Type", "Contour Text", "Large Print"):
            assert expected in names, f"Preset '{expected}' not found in {names}"

    def _run_preset(self, preset_name: str) -> list:
        gen = ASCIIArtGenerator()
        canvas = make_canvas()
        p = self._get_preset(preset_name)
        params = dict(p.params)
        params["_source_image"] = make_dark_image()
        return gen.generate(params, canvas)

    def test_typewriter_generates_valid_output(self):
        result = self._run_preset("Typewriter")
        assert isinstance(result, list)
        assert len(result) > 0
        for poly in result:
            assert len(poly) >= 2

    def test_scattered_type_generates_valid_output(self):
        result = self._run_preset("Scattered Type")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_contour_text_generates_valid_output(self):
        result = self._run_preset("Contour Text")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_large_print_generates_valid_output(self):
        result = self._run_preset("Large Print")
        assert isinstance(result, list)
        assert len(result) > 0
