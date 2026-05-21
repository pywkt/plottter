"""Tests for image_to_palette_grid and the palettes registry."""

import numpy as np
import pytest
from PIL import Image

from plottter.pixel_art import get_palette, image_to_palette_grid, list_palettes


def _make_gradient_image(width: int, height: int) -> np.ndarray:
    """Create a synthetic grayscale gradient image (H×W×3, uint8)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Value ranges from 0 (top-left) to 255 (bottom-right)
            val = int(255 * (x + y) / (width + height - 2))
            arr[y, x] = (val, val, val)
    return arr


class TestGrayscale4Grid:
    """Tests for grayscale_4 palette conversion."""

    def test_output_shape(self):
        img = _make_gradient_image(32, 32)
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=32, grid_height=32)
        assert grid.shape == (32, 32)

    def test_dtype_is_int32(self):
        img = _make_gradient_image(32, 32)
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=32, grid_height=32)
        assert grid.dtype == np.int32

    def test_exactly_four_unique_indices(self):
        """A smooth gradient through grayscale_4 must use all 4 indices (0,1,2,3)."""
        img = _make_gradient_image(32, 32)
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=32, grid_height=32)
        unique = set(int(v) for v in grid.flat)
        assert unique == {0, 1, 2, 3}, f"Expected indices {{0,1,2,3}}, got {sorted(unique)}"

    def test_indices_within_range(self):
        img = _make_gradient_image(32, 32)
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=32, grid_height=32)
        assert grid.min() >= 0
        assert grid.max() <= 3


class TestNESPalette:
    """Tests for NES palette conversion."""

    def test_nes_palette_color_count(self):
        palette = get_palette("nes")
        assert palette.color_count == 54

    def test_nes_indices_valid_range(self):
        """NES palette has 54 colors; all output indices must be in [0, 53]."""
        img = _make_gradient_image(16, 16)
        palette = get_palette("nes")
        grid = image_to_palette_grid(img, palette, grid_width=16, grid_height=16)
        assert grid.min() >= 0, f"Min index {grid.min()} is negative"
        assert grid.max() <= 53, f"Max index {grid.max()} exceeds 53"

    def test_nes_output_shape(self):
        img = _make_gradient_image(16, 16)
        palette = get_palette("nes")
        grid = image_to_palette_grid(img, palette, grid_width=8, grid_height=8)
        assert grid.shape == (8, 8)


class TestGridAutoHeight:
    """Tests for auto aspect-ratio grid height."""

    def test_auto_height_preserves_aspect(self):
        img = _make_gradient_image(64, 32)  # 2:1 aspect ratio
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(img, palette, grid_width=32)
        # Expected height: round(32 * 32 / 64) = 16
        assert grid.shape == (16, 32)


class TestPilImageInput:
    """Tests for PIL Image input."""

    def test_pil_rgb_input(self):
        pil_img = Image.fromarray(_make_gradient_image(16, 16), "RGB")
        palette = get_palette("grayscale_4")
        grid = image_to_palette_grid(pil_img, palette, grid_width=16, grid_height=16)
        assert grid.shape == (16, 16)
        assert set(int(v) for v in grid.flat).issubset({0, 1, 2, 3})


class TestGetPalette:
    """Tests for palette registry."""

    def test_get_palette_by_name(self):
        p = get_palette("nes")
        assert p.color_count == 54

    def test_get_palette_hyphen_alias(self):
        p = get_palette("grayscale-4")
        assert p.color_count == 4

    def test_get_palette_underscore_primary(self):
        p = get_palette("grayscale_4")
        assert p.color_count == 4

    def test_get_palette_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown palette"):
            get_palette("nonexistent_palette_xyz")

    def test_list_palettes_returns_list(self):
        names = list_palettes()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "nes" in names
        assert "grayscale_2" in names or "grayscale-2" in names
