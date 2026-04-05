"""Tests for ASCIIArtGenerator — grid placement and character weight ordering."""

from __future__ import annotations

import numpy as np
import pytest

from plottter.generators.ascii_art import ASCII_CHARS, ASCIIArtGenerator, compute_cell_characters
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

    def test_generate_returns_empty_polylines_for_now(self):
        """Generator returns empty list (glyph rendering not yet implemented)."""
        gen = ASCIIArtGenerator()
        canvas = make_canvas()
        img = make_dark_image()
        params = make_default_params()
        params["_source_image"] = img

        result = gen.generate(params, canvas)
        assert isinstance(result, list)
        assert result == []

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
