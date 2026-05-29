"""Tests for color separation methods (Phase 9)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from plottter.color.kmeans import kmeans_separate
from plottter.color.luminance import luminance_separate
from plottter.color.channels import rgb_separate, cmyk_separate
from plottter.color import (
    kmeans_separate as pkg_kmeans,
    luminance_separate as pkg_luminance,
    rgb_separate as pkg_rgb,
    cmyk_separate as pkg_cmyk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(h: int = 20, w: int = 20) -> np.ndarray:
    """Create an RGB test image with three distinct color regions."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Top half: red region
    img[: h // 2, :, 0] = 200
    # Bottom-left quarter: green region
    img[h // 2 :, : w // 2, 1] = 200
    # Bottom-right quarter: blue region
    img[h // 2 :, w // 2 :, 2] = 200
    return img


def make_grayscale_gradient(h: int = 20, w: int = 20) -> np.ndarray:
    """Create a 2-D grayscale gradient (H×W, uint8) spanning 0–255."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


# ---------------------------------------------------------------------------
# Package-level convenience imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_kmeans_import(self):
        assert pkg_kmeans is kmeans_separate

    def test_luminance_import(self):
        assert pkg_luminance is luminance_separate

    def test_rgb_import(self):
        assert pkg_rgb is rgb_separate

    def test_cmyk_import(self):
        assert pkg_cmyk is cmyk_separate


# ---------------------------------------------------------------------------
# K-Means color separation
# ---------------------------------------------------------------------------


class TestKmeansSeparate:
    def test_returns_correct_number_of_clusters(self):
        img = make_rgb_image()
        results = kmeans_separate(img, num_colors=3)
        assert len(results) == 3

    def test_each_result_is_bool_mask_and_hex_color(self):
        img = make_rgb_image()
        results = kmeans_separate(img, num_colors=2)
        for mask, hex_color in results:
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.bool_
            assert mask.shape == img.shape[:2]
            assert _HEX_RE.match(hex_color), f"Invalid hex color: {hex_color}"

    def test_masks_cover_every_pixel(self):
        """All masks OR'd together must be True everywhere (full coverage)."""
        img = make_rgb_image(10, 10)
        results = kmeans_separate(img, num_colors=3)
        combined = np.zeros(img.shape[:2], dtype=np.int32)
        for mask, _ in results:
            combined += mask.astype(np.int32)
        assert np.all(combined == 1), "Some pixels are unassigned or double-assigned"

    def test_masks_are_disjoint(self):
        """No pixel may belong to more than one cluster."""
        img = make_rgb_image(10, 10)
        results = kmeans_separate(img, num_colors=3)
        masks = [m for m, _ in results]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                assert not np.any(masks[i] & masks[j]), (
                    f"Clusters {i} and {j} overlap"
                )

    def test_num_colors_clamped_to_minimum_2(self):
        img = make_rgb_image(10, 10)
        results = kmeans_separate(img, num_colors=1)
        assert len(results) == 2

    def test_num_colors_clamped_to_maximum_8(self):
        img = make_rgb_image(10, 10)
        results = kmeans_separate(img, num_colors=10)
        assert len(results) == 8

    def test_invalid_image_shape_raises(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            kmeans_separate(gray, num_colors=2)

    def test_small_sample_size_still_covers_all_pixels(self):
        """With aggressive subsampling the full image must still be fully assigned."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        results = kmeans_separate(img, num_colors=3, sample_size=50, iterations=5)
        assert len(results) == 3
        combined = np.zeros((30, 30), dtype=np.int32)
        for mask, _ in results:
            combined += mask.astype(np.int32)
        assert np.all(combined == 1)


# ---------------------------------------------------------------------------
# Luminance / Tone Band Splitting
# ---------------------------------------------------------------------------


class TestLuminanceSeparate:
    def test_returns_correct_number_of_bands(self):
        img = make_grayscale_gradient()
        results = luminance_separate(img, num_bands=3)
        assert len(results) == 3

    def test_each_result_is_bool_mask_and_hex_color(self):
        img = make_grayscale_gradient()
        results = luminance_separate(img, num_bands=3)
        for mask, hex_color in results:
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.bool_
            assert mask.shape == img.shape
            assert _HEX_RE.match(hex_color), f"Invalid hex color: {hex_color}"

    def test_bands_cover_all_pixels(self):
        """Every pixel must be assigned to exactly one band."""
        img = make_grayscale_gradient(10, 10)
        results = luminance_separate(img, num_bands=3)
        combined = np.zeros(img.shape, dtype=np.int32)
        for mask, _ in results:
            combined += mask.astype(np.int32)
        assert np.all(combined == 1), "Some pixels are unassigned or double-assigned"

    def test_bands_do_not_overlap(self):
        img = make_grayscale_gradient(10, 10)
        results = luminance_separate(img, num_bands=4)
        masks = [m for m, _ in results]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                assert not np.any(masks[i] & masks[j]), (
                    f"Bands {i} and {j} overlap"
                )

    def test_custom_thresholds_cover_all_pixels(self):
        img = make_grayscale_gradient(10, 10)
        results = luminance_separate(img, num_bands=3, thresholds=[85.0, 170.0])
        assert len(results) == 3
        combined = np.zeros(img.shape, dtype=np.int32)
        for mask, _ in results:
            combined += mask.astype(np.int32)
        assert np.all(combined == 1)

    def test_wrong_threshold_count_raises(self):
        img = make_grayscale_gradient(10, 10)
        with pytest.raises(ValueError):
            # 3 bands need 2 thresholds — only 1 provided
            luminance_separate(img, num_bands=3, thresholds=[128.0])

    def test_rgb_input_auto_converted_to_grayscale(self):
        img = make_rgb_image(10, 10)
        results = luminance_separate(img, num_bands=2)
        assert len(results) == 2
        for mask, _ in results:
            assert mask.shape == img.shape[:2]
        combined = np.zeros(img.shape[:2], dtype=np.int32)
        for mask, _ in results:
            combined += mask.astype(np.int32)
        assert np.all(combined == 1)

    def test_num_bands_clamped_to_minimum_2(self):
        img = make_grayscale_gradient(10, 10)
        results = luminance_separate(img, num_bands=1)
        assert len(results) == 2

    def test_num_bands_clamped_to_maximum_5(self):
        img = make_grayscale_gradient(10, 10)
        results = luminance_separate(img, num_bands=10)
        assert len(results) == 5

    def test_default_colors_dark_to_light_order(self):
        """Default assigned hex colors should progress from darkest to lightest."""
        img = make_grayscale_gradient()
        results = luminance_separate(img, num_bands=3)
        colors = [hex_color for _, hex_color in results]

        def lum(h: str) -> float:
            r = int(h[1:3], 16)
            g = int(h[3:5], 16)
            b = int(h[5:7], 16)
            return 0.299 * r + 0.587 * g + 0.114 * b

        luminances = [lum(c) for c in colors]
        assert luminances == sorted(luminances), (
            f"Colors not ordered dark→light: {colors}"
        )

    def test_full_range_covered_for_2_bands(self):
        """With 2 bands every pixel value 0–255 must be covered."""
        img = np.arange(256, dtype=np.uint8).reshape(1, 256)
        results = luminance_separate(img, num_bands=2)
        combined = np.zeros((1, 256), dtype=np.int32)
        for mask, _ in results:
            combined += mask.astype(np.int32)
        assert np.all(combined == 1)


# ---------------------------------------------------------------------------
# RGB Channel Separation
# ---------------------------------------------------------------------------


class TestRgbSeparate:
    def test_returns_three_channels(self):
        img = make_rgb_image()
        results = rgb_separate(img)
        assert len(results) == 3

    def test_channel_hex_colors(self):
        img = make_rgb_image()
        results = rgb_separate(img)
        _, r_hex = results[0]
        _, g_hex = results[1]
        _, b_hex = results[2]
        assert r_hex == "#FF0000"
        assert g_hex == "#00FF00"
        assert b_hex == "#0000FF"

    def test_each_channel_is_grayscale_uint8_array(self):
        img = make_rgb_image(10, 10)
        results = rgb_separate(img)
        for channel, _ in results:
            assert isinstance(channel, np.ndarray)
            assert channel.ndim == 2
            assert channel.shape == img.shape[:2]
            assert channel.dtype == np.uint8

    def test_red_channel_matches_r_component(self):
        img = make_rgb_image(10, 10)
        r_channel, _ = rgb_separate(img)[0]
        np.testing.assert_array_equal(r_channel, img[:, :, 0])

    def test_green_channel_matches_g_component(self):
        img = make_rgb_image(10, 10)
        g_channel, _ = rgb_separate(img)[1]
        np.testing.assert_array_equal(g_channel, img[:, :, 1])

    def test_blue_channel_matches_b_component(self):
        img = make_rgb_image(10, 10)
        b_channel, _ = rgb_separate(img)[2]
        np.testing.assert_array_equal(b_channel, img[:, :, 2])

    def test_invalid_image_shape_raises(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            rgb_separate(gray)


# ---------------------------------------------------------------------------
# CMYK Channel Separation
# ---------------------------------------------------------------------------


class TestCmykSeparate:
    def test_returns_four_channels(self):
        img = make_rgb_image()
        results = cmyk_separate(img)
        assert len(results) == 4

    def test_channel_hex_colors(self):
        img = make_rgb_image()
        results = cmyk_separate(img)
        _, c_hex = results[0]
        _, m_hex = results[1]
        _, y_hex = results[2]
        _, k_hex = results[3]
        assert c_hex == "#00FFFF"
        assert m_hex == "#FF00FF"
        assert y_hex == "#FFFF00"
        assert k_hex == "#000000"

    def test_each_channel_is_grayscale_uint8_array(self):
        img = make_rgb_image(10, 10)
        results = cmyk_separate(img)
        for channel, _ in results:
            assert isinstance(channel, np.ndarray)
            assert channel.ndim == 2
            assert channel.shape == img.shape[:2]
            assert channel.dtype == np.uint8

    def test_channel_values_in_valid_range(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        results = cmyk_separate(img)
        for channel, _ in results:
            assert int(channel.min()) >= 0
            assert int(channel.max()) <= 255

    def test_pure_red_cmyk_values(self):
        """RGB(255,0,0) → C=0, M=255, Y=255, K=0."""
        img = np.array([[[255, 0, 0]]], dtype=np.uint8)
        results = cmyk_separate(img)
        c = int(results[0][0][0, 0])
        m = int(results[1][0][0, 0])
        y = int(results[2][0][0, 0])
        k = int(results[3][0][0, 0])
        assert c == 0
        assert m == 255
        assert y == 255
        assert k == 0

    def test_pure_black_cmyk_values(self):
        """RGB(0,0,0) → C=0, M=0, Y=0, K=255."""
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        results = cmyk_separate(img)
        c = int(results[0][0][0, 0])
        m = int(results[1][0][0, 0])
        y = int(results[2][0][0, 0])
        k = int(results[3][0][0, 0])
        assert c == 0
        assert m == 0
        assert y == 0
        assert k == 255

    def test_pure_white_cmyk_values(self):
        """RGB(255,255,255) → C=0, M=0, Y=0, K=0."""
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        results = cmyk_separate(img)
        c = int(results[0][0][0, 0])
        m = int(results[1][0][0, 0])
        y = int(results[2][0][0, 0])
        k = int(results[3][0][0, 0])
        assert c == 0
        assert m == 0
        assert y == 0
        assert k == 0

    def test_invalid_image_shape_raises(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            cmyk_separate(gray)

    # ---- k_amount: scales K channel without touching CMY -----------------

    def test_k_amount_zero_zeroes_out_key_channel(self):
        """k_amount=0 must produce K=0 everywhere — CMY only."""
        img = np.full((4, 4, 3), 64, dtype=np.uint8)  # dark gray with K
        results = cmyk_separate(img, k_amount=0.0)
        k_channel = results[3][0]
        assert (k_channel == 0).all()

    def test_k_amount_half_scales_key_by_half(self):
        """k_amount=0.5 must scale K by ~50% while leaving CMY untouched."""
        img = np.full((4, 4, 3), 128, dtype=np.uint8)  # mid gray
        full = cmyk_separate(img, k_amount=1.0)
        half = cmyk_separate(img, k_amount=0.5)
        # CMY unchanged (all zeros for neutral gray anyway)
        for i in range(3):
            assert np.array_equal(full[i][0], half[i][0])
        # K halved (within int-rounding tolerance)
        full_k = int(full[3][0][0, 0])
        half_k = int(half[3][0][0, 0])
        assert abs(half_k - full_k // 2) <= 1, f"expected ~{full_k // 2}, got {half_k}"

    def test_k_amount_default_is_full(self):
        """No-arg call keeps the original full-K behaviour for back-compat."""
        img = np.full((2, 2, 3), 100, dtype=np.uint8)
        no_arg = cmyk_separate(img)
        explicit = cmyk_separate(img, k_amount=1.0)
        for ch_a, ch_b in zip(no_arg, explicit):
            assert np.array_equal(ch_a[0], ch_b[0])

    def test_k_amount_clamps_to_valid_range(self):
        """Out-of-range values clamp to [0, 1] without raising."""
        img = np.full((2, 2, 3), 100, dtype=np.uint8)
        clamped_low  = cmyk_separate(img, k_amount=-5.0)
        clamped_high = cmyk_separate(img, k_amount=10.0)
        equiv_zero = cmyk_separate(img, k_amount=0.0)
        equiv_one  = cmyk_separate(img, k_amount=1.0)
        assert np.array_equal(clamped_low[3][0], equiv_zero[3][0])
        assert np.array_equal(clamped_high[3][0], equiv_one[3][0])
