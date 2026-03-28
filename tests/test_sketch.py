"""Tests for SketchGenerator scaffold — darkest-area finder."""

from __future__ import annotations

import numpy as np
import pytest

from plottter.generators.sketch import SketchGenerator
from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def make_white_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Pure white grayscale image."""
    return np.full((h, w), 255, dtype=np.uint8)


def make_single_dark_block(
    h: int = 64,
    w: int = 64,
    dark_block_row: int = 1,
    dark_block_col: int = 2,
    block_size: int = 16,
) -> np.ndarray:
    """White image with one block filled black, making that block the darkest."""
    arr = np.full((h, w), 255, dtype=np.uint8)
    r0 = dark_block_row * block_size
    r1 = min(r0 + block_size, h)
    c0 = dark_block_col * block_size
    c1 = min(c0 + block_size, w)
    arr[r0:r1, c0:c1] = 0
    return arr


def make_single_dark_pixel(
    h: int = 32,
    w: int = 32,
    py: int = 5,
    px: int = 7,
) -> np.ndarray:
    """White image with one pixel set to black."""
    arr = np.full((h, w), 255, dtype=np.uint8)
    arr[py, px] = 0
    return arr


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registered_in_generators(self):
        from plottter.generators import GENERATORS
        assert "Sketch" in GENERATORS

    def test_category_is_image(self):
        from plottter.generators import GENERATORS
        assert GENERATORS["Sketch"].category == "image"

    def test_name(self):
        gen = SketchGenerator()
        assert gen.name == "Sketch"


# ---------------------------------------------------------------------------
# _find_darkest_region
# ---------------------------------------------------------------------------


class TestFindDarkestRegion:
    def setup_method(self):
        self.gen = SketchGenerator()

    def test_returns_correct_block_for_dark_region(self):
        """The block containing the single black region should be returned."""
        dark_row, dark_col = 1, 2
        block_size = 16
        img = make_single_dark_block(
            h=64, w=64,
            dark_block_row=dark_row,
            dark_block_col=dark_col,
            block_size=block_size,
        )
        br, bc = self.gen._find_darkest_region(img, block_size)
        assert br == dark_row
        assert bc == dark_col

    def test_pure_white_returns_a_block(self):
        """All-white image: any block is valid — just ensure no exception and valid range."""
        img = make_white_image(64, 64)
        block_size = 16
        br, bc = self.gen._find_darkest_region(img, block_size)
        n_rows = (64 + block_size - 1) // block_size
        n_cols = (64 + block_size - 1) // block_size
        assert 0 <= br < n_rows
        assert 0 <= bc < n_cols

    def test_single_pixel_image(self):
        """1×1 image should not crash and should return (0, 0)."""
        img = np.array([[128]], dtype=np.uint8)
        br, bc = self.gen._find_darkest_region(img, block_size=16)
        assert br == 0
        assert bc == 0

    def test_gradient_horizontal(self):
        """Left columns are darker — darkest block should be on the left."""
        h, w = 32, 64
        img = np.zeros((h, w), dtype=np.uint8)
        for x in range(w):
            img[:, x] = int(x / (w - 1) * 255)
        block_size = 16
        br, bc = self.gen._find_darkest_region(img, block_size)
        # Leftmost column of blocks (bc==0) is darkest
        assert bc == 0

    def test_image_smaller_than_block_size(self):
        """Image smaller than block_size should still return (0, 0)."""
        img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        br, bc = self.gen._find_darkest_region(img, block_size=16)
        assert br == 0
        assert bc == 0


# ---------------------------------------------------------------------------
# _find_darkest_pixel
# ---------------------------------------------------------------------------


class TestFindDarkestPixel:
    def setup_method(self):
        self.gen = SketchGenerator()

    def test_finds_single_dark_pixel(self):
        """Single black pixel should be identified correctly."""
        py, px = 5, 7
        img = make_single_dark_pixel(h=32, w=32, py=py, px=px)
        block_size = 16
        # The black pixel is in block (0, 0) since 5<16, 7<16
        found_y, found_x = self.gen._find_darkest_pixel(img, 0, 0, block_size)
        assert found_y == py
        assert found_x == px

    def test_coordinates_within_block_bounds(self):
        """Returned pixel coordinates must be within the specified block."""
        block_size = 16
        dark_row, dark_col = 1, 2
        img = make_single_dark_block(
            h=64, w=64,
            dark_block_row=dark_row,
            dark_block_col=dark_col,
            block_size=block_size,
        )
        py, px = self.gen._find_darkest_pixel(img, dark_row, dark_col, block_size)
        r0 = dark_row * block_size
        c0 = dark_col * block_size
        assert r0 <= py < r0 + block_size
        assert c0 <= px < c0 + block_size

    def test_white_block_returns_valid_coords(self):
        """All-white block: no crash, coordinates stay in range."""
        img = make_white_image(64, 64)
        block_size = 16
        py, px = self.gen._find_darkest_pixel(img, 0, 0, block_size)
        assert 0 <= py < block_size
        assert 0 <= px < block_size

    def test_last_block_clipped(self):
        """Block that extends beyond image edge should be clipped correctly."""
        # 20×20 image with block_size=16 — the second block is only 4px wide
        img = np.zeros((20, 20), dtype=np.uint8)
        img[17, 17] = 0  # darkest in clipped block
        py, px = self.gen._find_darkest_pixel(img, 1, 1, block_size=16)
        assert 16 <= py < 20
        assert 16 <= px < 20


# ---------------------------------------------------------------------------
# generate() scaffold
# ---------------------------------------------------------------------------


class TestGenerateScaffold:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def test_returns_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_returns_empty_list_with_source_image(self):
        """Current scaffold always returns [] even with a real image."""
        img = make_single_dark_block()
        params = {"_source_image": img}
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert result == []

    def test_pure_white_image_does_not_crash(self):
        img = make_white_image()
        params = {"_source_image": img}
        result = self.gen.generate(params, self.canvas)
        assert result == []

    def test_parameters_defined(self):
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "line_density" in names
        assert "line_max_limit" in names
        assert "block_size" in names
        assert "invert" in names
        assert "brightness" in names
        assert "contrast" in names
        assert "blur_radius" in names
        assert "x_offset_mm" in names
        assert "y_offset_mm" in names

    def test_presets_defined(self):
        presets = self.gen.get_presets()
        assert len(presets) >= 1
        assert presets[0].name == "Default"
