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


def _make_horizontal_gradient(width: int = 256, height: int = 16) -> np.ndarray:
    """Return a (height × width × 3) uint8 image with a pure left-to-right gradient.

    Column x has all pixels set to (x * 255 / (width - 1), same, same), so the
    left edge is black and the right edge is white.  There is no vertical
    variation, which makes the expected band boundaries of a 4-colour palette
    easy to reason about analytically.
    """
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    row = np.stack([xs, xs, xs], axis=-1)          # shape (width, 3)
    return np.tile(row[np.newaxis, :, :], (height, 1, 1))  # shape (height, width, 3)


class TestFloydSteinbergEffect:
    """Gradient fixture confirms F-S dithering changes the palette-index grid."""

    def setup_method(self):
        from plottter.pixel_art import get_palette
        self.palette = get_palette("grayscale_4")
        self.img = _make_horizontal_gradient(256, 16)

    def test_horizontal_gradient_fixture_shape(self):
        """Fixture must be 16 rows × 256 cols × 3 channels, uint8."""
        assert self.img.shape == (16, 256, 3)
        assert self.img.dtype == np.uint8

    def test_horizontal_gradient_left_black_right_white(self):
        """Leftmost column must be near-black, rightmost near-white."""
        assert int(self.img[0, 0, 0]) == 0
        assert int(self.img[0, -1, 0]) == 255

    def test_floyd_steinberg_differs_from_none(self):
        """Floyd-Steinberg dithering must produce a different palette grid than no dithering.

        A smooth 0-255 horizontal gradient quantised to 4 shades produces sharp
        colour bands with dithering='none'.  Floyd-Steinberg error-diffusion
        spreads quantisation error to neighbouring pixels, creating a stippled
        pattern at the band boundaries — so the two grids must not be identical.
        """
        from plottter.pixel_art import image_to_palette_grid

        grid_none = image_to_palette_grid(
            self.img, self.palette, grid_width=256, grid_height=16, dithering="none"
        )
        grid_fs = image_to_palette_grid(
            self.img, self.palette, grid_width=256, grid_height=16,
            dithering="floyd_steinberg",
        )

        assert grid_none.shape == grid_fs.shape, "Both grids must have the same shape"
        assert not np.array_equal(grid_none, grid_fs), (
            "Floyd-Steinberg dithering produced an identical grid to no-dithering on a "
            "smooth gradient — dithering is not being applied."
        )

    def test_floyd_steinberg_uses_more_index_transitions(self):
        """Dithered grid must have more index-change transitions than the hard-banded grid.

        With 'none', each row has exactly 3 band boundaries (0→1, 1→2, 2→3).
        Floyd-Steinberg creates many additional transitions because error is
        diffused laterally, so the transition count must be strictly higher.
        """
        from plottter.pixel_art import image_to_palette_grid

        grid_none = image_to_palette_grid(
            self.img, self.palette, grid_width=256, grid_height=16, dithering="none"
        )
        grid_fs = image_to_palette_grid(
            self.img, self.palette, grid_width=256, grid_height=16,
            dithering="floyd_steinberg",
        )

        def count_transitions(grid: np.ndarray) -> int:
            """Count the total number of adjacent-cell index changes across all rows."""
            return int(np.sum(grid[:, 1:] != grid[:, :-1]))

        transitions_none = count_transitions(grid_none)
        transitions_fs = count_transitions(grid_fs)
        assert transitions_fs > transitions_none, (
            f"Floyd-Steinberg produced {transitions_fs} transitions vs "
            f"{transitions_none} for no-dithering; expected more transitions with dithering."
        )


class TestAllDitheringModes:
    """Programmatic visual check: all 4 dithering modes run and affect output.

    This mirrors the Mona Lisa visual comparison described in task 120.2(C):
    each dithering mode is applied to a representative smooth gradient image and
    the outputs are compared to confirm dithering is wired end-to-end for every
    supported mode.  The 'none' baseline is the reference; every other mode must
    produce a grid that differs from 'none'.
    """

    def setup_method(self):
        from plottter.pixel_art import get_palette
        self.palette = get_palette("grayscale_4")
        self.img = _make_horizontal_gradient(256, 16)

    def _get_grid(self, dithering: str) -> np.ndarray:
        from plottter.pixel_art import image_to_palette_grid
        return image_to_palette_grid(
            self.img, self.palette,
            grid_width=256, grid_height=16,
            dithering=dithering,
        )

    def test_all_modes_run_without_error(self):
        for mode in ("none", "floyd_steinberg", "ordered", "atkinson"):
            grid = self._get_grid(mode)
            assert grid.ndim == 2, f"dithering={mode!r}: expected 2-D grid"
            assert grid.dtype == np.int32, f"dithering={mode!r}: expected int32"
            assert grid.shape == (16, 256), f"dithering={mode!r}: unexpected shape {grid.shape}"

    def test_floyd_steinberg_differs_from_none(self):
        assert not np.array_equal(self._get_grid("none"), self._get_grid("floyd_steinberg")), (
            "floyd_steinberg produced same grid as none on a smooth gradient"
        )

    def test_ordered_differs_from_none(self):
        assert not np.array_equal(self._get_grid("none"), self._get_grid("ordered")), (
            "ordered dithering produced same grid as none on a smooth gradient"
        )

    def test_atkinson_differs_from_none(self):
        assert not np.array_equal(self._get_grid("none"), self._get_grid("atkinson")), (
            "atkinson dithering produced same grid as none on a smooth gradient"
        )

    def test_all_indices_valid(self):
        for mode in ("none", "floyd_steinberg", "ordered", "atkinson"):
            grid = self._get_grid(mode)
            assert grid.min() >= 0, f"dithering={mode!r}: negative index found"
            assert grid.max() <= 3, f"dithering={mode!r}: index > 3 found"
