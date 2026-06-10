"""Tests for palette_separate() correctness."""
import numpy as np
import pytest

from plottter.color.palette import PenPalette
from plottter.color.palette_separator import palette_separate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(r: int, g: int, b: int, h: int = 16, w: int = 16) -> np.ndarray:
    """Create a solid-colour uint8 RGB image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _two_color_image(
    r1: int, g1: int, b1: int,
    r2: int, g2: int, b2: int,
    h: int = 16,
    w: int = 16,
) -> np.ndarray:
    """Top half colour 1, bottom half colour 2."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, :, :] = [r1, g1, b1]
    img[h // 2 :, :, :] = [r2, g2, b2]
    return img


BASIC_PALETTE = PenPalette(
    name="Test",
    colors=("#ff0000", "#00ff00", "#0000ff"),
)

GRAYSCALE_PALETTE = PenPalette(
    name="Gray",
    colors=("#000000", "#808080", "#ffffff"),
)


# ---------------------------------------------------------------------------
# Mask invariants
# ---------------------------------------------------------------------------

class TestMaskExhaustivenessAndExclusivity:
    """sum(masks) == 255 everywhere; masks are mutually exclusive."""

    def _assert_invariants(self, result, image):
        h, w = image.shape[:2]
        assert len(result) > 0
        total = np.zeros((h, w), dtype=np.int32)
        for mask, hex_color in result:
            assert mask.shape == (h, w)
            assert mask.dtype == np.uint8
            # Each mask contains only 0 or 255
            unique = set(np.unique(mask).tolist())
            assert unique <= {0, 255}, f"mask for {hex_color} has values {unique}"
            total += mask.astype(np.int32)
        # Exhaustive: every pixel covered exactly once (sum == 255)
        assert np.all(total == 255), f"sum not 255 everywhere; range [{total.min()}, {total.max()}]"

    def test_no_dither_lab(self):
        img = _solid_image(255, 0, 0)
        result = palette_separate(img, BASIC_PALETTE, dither="none", color_space="lab")
        self._assert_invariants(result, img)

    def test_no_dither_rgb(self):
        img = _solid_image(0, 255, 0)
        result = palette_separate(img, BASIC_PALETTE, dither="none", color_space="rgb")
        self._assert_invariants(result, img)

    def test_floyd_steinberg(self):
        img = _two_color_image(200, 50, 50, 50, 200, 50)
        result = palette_separate(img, BASIC_PALETTE, dither="floyd-steinberg")
        self._assert_invariants(result, img)

    def test_ordered(self):
        img = _two_color_image(100, 100, 100, 200, 200, 200)
        result = palette_separate(img, GRAYSCALE_PALETTE, dither="ordered")
        self._assert_invariants(result, img)

    def test_atkinson(self):
        img = _two_color_image(255, 0, 0, 0, 0, 255)
        result = palette_separate(img, BASIC_PALETTE, dither="atkinson")
        self._assert_invariants(result, img)


class TestMaskCount:
    def test_count_matches_palette(self):
        img = _solid_image(128, 128, 128)
        for palette in [BASIC_PALETTE, GRAYSCALE_PALETTE]:
            result = palette_separate(img, palette)
            assert len(result) == palette.count

    def test_single_color_palette(self):
        one_color = PenPalette(name="One", colors=("#ff0000",))
        img = _solid_image(128, 0, 0)
        result = palette_separate(img, one_color)
        assert len(result) == 1
        # Only one mask; it must be all 255
        mask, _ = result[0]
        assert np.all(mask == 255)


class TestHexAssignment:
    def test_hex_matches_palette_order(self):
        img = _solid_image(0, 0, 0)
        result = palette_separate(img, BASIC_PALETTE)
        assert len(result) == BASIC_PALETTE.count
        for (_, hex_color), expected in zip(result, BASIC_PALETTE.colors):
            assert hex_color == expected


# ---------------------------------------------------------------------------
# One-colour image → single non-zero mask
# ---------------------------------------------------------------------------

class TestSolidColorImage:
    @pytest.mark.parametrize("hex_color,rgb", [
        ("#FF0000", (255, 0, 0)),
        ("#00FF00", (0, 255, 0)),
        ("#0000FF", (0, 0, 255)),
    ])
    def test_exact_palette_color_maps_to_single_mask(self, hex_color, rgb):
        img = _solid_image(*rgb)
        result = palette_separate(img, BASIC_PALETTE, dither="none", color_space="lab")
        nonzero_masks = [(m, h) for m, h in result if np.any(m > 0)]
        assert len(nonzero_masks) == 1
        mask, matched_hex = nonzero_masks[0]
        assert matched_hex == hex_color
        assert np.all(mask == 255)

    def test_solid_black_with_grayscale_palette(self):
        img = _solid_image(0, 0, 0)
        result = palette_separate(img, GRAYSCALE_PALETTE, dither="none", color_space="lab")
        nonzero = [(m, h) for m, h in result if np.any(m > 0)]
        assert len(nonzero) == 1
        _, matched_hex = nonzero[0]
        assert matched_hex == "#000000"

    def test_solid_white_with_grayscale_palette(self):
        img = _solid_image(255, 255, 255)
        result = palette_separate(img, GRAYSCALE_PALETTE, dither="none", color_space="lab")
        nonzero = [(m, h) for m, h in result if np.any(m > 0)]
        assert len(nonzero) == 1
        _, matched_hex = nonzero[0]
        assert matched_hex == "#FFFFFF"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("dither", ["none", "ordered"])
    def test_deterministic_output(self, dither):
        """Same (image, palette, dither, color_space) → same masks both times."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        r1 = palette_separate(img, BASIC_PALETTE, dither=dither, color_space="lab")
        r2 = palette_separate(img, BASIC_PALETTE, dither=dither, color_space="lab")
        for (m1, h1), (m2, h2) in zip(r1, r2):
            assert h1 == h2
            np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# Color space both produce valid masks
# ---------------------------------------------------------------------------

class TestColorSpace:
    def test_lab_produces_valid_masks(self):
        img = _two_color_image(200, 50, 50, 50, 50, 200)
        result = palette_separate(img, BASIC_PALETTE, dither="none", color_space="lab")
        total = sum(m.astype(np.int32) for m, _ in result)
        assert np.all(total == 255)

    def test_rgb_produces_valid_masks(self):
        img = _two_color_image(200, 50, 50, 50, 50, 200)
        result = palette_separate(img, BASIC_PALETTE, dither="none", color_space="rgb")
        total = sum(m.astype(np.int32) for m, _ in result)
        assert np.all(total == 255)

    def test_lab_and_rgb_both_run_without_error(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        for cs in ("lab", "rgb"):
            result = palette_separate(img, GRAYSCALE_PALETTE, dither="none", color_space=cs)
            assert len(result) == GRAYSCALE_PALETTE.count


# ---------------------------------------------------------------------------
# Dither modes change output vs none
# ---------------------------------------------------------------------------

class TestDitherChangesOutput:
    """Dithering should produce different pixel assignments than no-dither on an
    image whose colour is not in the palette (so dithering scatters pixels)."""

    # Black/white-only palette forces maximum dithering on any non-black, non-white image.
    BW_PALETTE = PenPalette(name="BW", colors=("#000000", "#FFFFFF"))

    def _get_mask_arrays(self, img, palette, dither, color_space="lab"):
        result = palette_separate(img, palette, dither=dither, color_space=color_space)
        return [m for m, _ in result]

    def test_floyd_steinberg_differs_from_none(self):
        # Mid-grey is equidistant from black and white; dithering will scatter.
        img = _solid_image(128, 128, 128, h=32, w=32)
        masks_none = self._get_mask_arrays(img, self.BW_PALETTE, "none")
        masks_fs = self._get_mask_arrays(img, self.BW_PALETTE, "floyd-steinberg")
        any_diff = any(
            not np.array_equal(m1, m2) for m1, m2 in zip(masks_none, masks_fs)
        )
        assert any_diff, "Floyd-Steinberg dithering produced identical output to no-dither"

    def test_ordered_differs_from_none(self):
        img = _solid_image(128, 128, 128, h=32, w=32)
        masks_none = self._get_mask_arrays(img, self.BW_PALETTE, "none")
        masks_ord = self._get_mask_arrays(img, self.BW_PALETTE, "ordered")
        any_diff = any(
            not np.array_equal(m1, m2) for m1, m2 in zip(masks_none, masks_ord)
        )
        assert any_diff, "Ordered dithering produced identical output to no-dither"

    def test_atkinson_differs_from_none(self):
        img = _solid_image(128, 128, 128, h=32, w=32)
        masks_none = self._get_mask_arrays(img, self.BW_PALETTE, "none")
        masks_atk = self._get_mask_arrays(img, self.BW_PALETTE, "atkinson")
        any_diff = any(
            not np.array_equal(m1, m2) for m1, m2 in zip(masks_none, masks_atk)
        )
        assert any_diff, "Atkinson dithering produced identical output to no-dither"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_dtype_raises(self):
        img = np.zeros((8, 8, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            palette_separate(img, BASIC_PALETTE)

    def test_wrong_channels_raises(self):
        img = np.zeros((8, 8, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            palette_separate(img, BASIC_PALETTE)

    def test_invalid_color_space_raises(self):
        img = _solid_image(0, 0, 0)
        with pytest.raises(ValueError, match="color_space"):
            palette_separate(img, BASIC_PALETTE, color_space="xyz")

    def test_invalid_dither_raises(self):
        img = _solid_image(0, 0, 0)
        with pytest.raises(ValueError, match="dither"):
            palette_separate(img, BASIC_PALETTE, dither="bad-dither")


class TestVectorizedLabEquivalence:
    """The LAB palette path was sped up ~5-6x by replacing a per-pixel Python
    loop (rgb_to_lab) with a vectorized conversion (rgb_to_lab_array). That is
    only safe if the two produce identical values — identical LAB => identical
    nearest-neighbour argmin => identical masks. Guard that invariant here so
    the slow loop can't quietly creep back."""

    def test_vectorized_lab_matches_scalar(self):
        from plottter.pixel_art.color_utils import rgb_to_lab, rgb_to_lab_array

        rng = np.random.default_rng(7)
        colors = np.vstack([
            rng.integers(0, 256, size=(5000, 3), dtype=np.uint8),
            # boundary / pure colours that exercise both linearization branches
            np.array(
                [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
                 [0, 0, 255], [10, 10, 10], [11, 11, 11], [128, 64, 200]],
                dtype=np.uint8,
            ),
        ])
        vec = rgb_to_lab_array(colors)
        scalar = np.array(
            [rgb_to_lab(tuple(int(c) for c in p)) for p in colors],
            dtype=np.float32,
        )
        assert vec.shape == scalar.shape
        np.testing.assert_array_equal(vec, scalar)

    def test_vectorized_lab_preserves_array_shape(self):
        from plottter.pixel_art.color_utils import rgb_to_lab_array

        img = np.zeros((4, 5, 3), dtype=np.uint8)
        out = rgb_to_lab_array(img)
        assert out.shape == (4, 5, 3)
        assert out.dtype == np.float32
