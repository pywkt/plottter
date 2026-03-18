"""Tests for src/plottter/io/image_import.py — Phase 6.6."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from PIL import Image as PILImage

from plottter.io.image_import import (
    ImageImportError,
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    apply_blur,
    apply_sharpen,
    apply_threshold,
    crop_to_aspect,
    invert_image,
    load_image,
    preprocess,
    remove_background,
    to_grayscale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_test_image(path: str, mode: str = "RGB", size: tuple = (64, 64)) -> None:
    """Save a simple gradient image to *path*."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        val = int(x / (size[0] - 1) * 255)
        arr[:, x, :] = val  # left-to-right gradient
    pil = PILImage.fromarray(arr, mode="RGB")
    pil.save(path)


def _make_rgb_array(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a synthetic RGB gradient array."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        arr[:, x, :] = int(x / (w - 1) * 255)
    return arr


# ---------------------------------------------------------------------------
# 6.1 — load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_load_jpg(self, tmp_path):
        path = str(tmp_path / "test.jpg")
        _save_test_image(path)
        img = load_image(path)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_load_png(self, tmp_path):
        path = str(tmp_path / "test.png")
        _save_test_image(path)
        img = load_image(path)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_load_webp(self, tmp_path):
        path = str(tmp_path / "test.webp")
        _save_test_image(path)
        img = load_image(path)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_output_is_rgb(self, tmp_path):
        """RGBA PNG should be converted to RGB."""
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, 3] = 128  # semi-transparent
        path = str(tmp_path / "rgba.png")
        PILImage.fromarray(arr, mode="RGBA").save(path)
        img = load_image(path)
        assert img.shape[2] == 3

    def test_load_grayscale_converted_to_rgb(self, tmp_path):
        arr = np.arange(0, 256, 4, dtype=np.uint8).reshape(8, 8)
        path = str(tmp_path / "gray.png")
        PILImage.fromarray(arr, mode="L").save(path)
        img = load_image(path)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_file_not_found_raises(self):
        with pytest.raises(ImageImportError, match="not found"):
            load_image("/nonexistent/path/image.png")

    def test_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "test.bmp")
        PILImage.new("RGB", (10, 10)).save(path, format="BMP")
        with pytest.raises(ImageImportError, match="Unsupported"):
            load_image(path)

    def test_output_shape_matches_file(self, tmp_path):
        path = str(tmp_path / "sized.png")
        PILImage.new("RGB", (100, 80)).save(path)
        img = load_image(path)
        assert img.shape == (80, 100, 3)  # H × W × 3


# ---------------------------------------------------------------------------
# 6.3 — to_grayscale
# ---------------------------------------------------------------------------


class TestToGrayscale:
    def test_output_is_2d(self):
        arr = _make_rgb_array()
        gray = to_grayscale(arr)
        assert gray.ndim == 2

    def test_output_dtype_uint8(self):
        arr = _make_rgb_array()
        gray = to_grayscale(arr)
        assert gray.dtype == np.uint8

    def test_output_shape(self):
        arr = _make_rgb_array(48, 80)
        gray = to_grayscale(arr)
        assert gray.shape == (48, 80)

    def test_already_grayscale_passthrough(self):
        gray_in = np.arange(0, 64, dtype=np.uint8).reshape(8, 8)
        out = to_grayscale(gray_in)
        assert out is gray_in  # must return input unchanged

    def test_pure_red_channel(self):
        """Pure red: 0.299*255 ≈ 76."""
        arr = np.zeros((1, 1, 3), dtype=np.uint8)
        arr[0, 0, 0] = 255  # R
        gray = to_grayscale(arr)
        assert 74 <= int(gray[0, 0]) <= 78  # allow ±2 rounding

    def test_pure_green_channel(self):
        """Pure green: 0.587*255 ≈ 150."""
        arr = np.zeros((1, 1, 3), dtype=np.uint8)
        arr[0, 0, 1] = 255  # G
        gray = to_grayscale(arr)
        assert 148 <= int(gray[0, 0]) <= 152

    def test_white_pixel(self):
        arr = np.full((1, 1, 3), 255, dtype=np.uint8)
        gray = to_grayscale(arr)
        assert gray[0, 0] == 255

    def test_black_pixel(self):
        arr = np.zeros((1, 1, 3), dtype=np.uint8)
        gray = to_grayscale(arr)
        assert gray[0, 0] == 0


# ---------------------------------------------------------------------------
# 6.2 — Preprocessing functions
# ---------------------------------------------------------------------------


class TestAdjustBrightness:
    def test_positive_increases_pixel_values(self):
        arr = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = adjust_brightness(arr, 50)
        assert int(result[0, 0, 0]) > 100

    def test_negative_decreases_pixel_values(self):
        arr = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = adjust_brightness(arr, -50)
        assert int(result[0, 0, 0]) < 100

    def test_zero_no_change(self):
        arr = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = adjust_brightness(arr, 0)
        assert np.array_equal(result, arr)

    def test_clamps_at_255(self):
        arr = np.full((2, 2, 3), 250, dtype=np.uint8)
        result = adjust_brightness(arr, 100)
        assert result.max() == 255

    def test_clamps_at_0(self):
        arr = np.full((2, 2, 3), 10, dtype=np.uint8)
        result = adjust_brightness(arr, -100)
        assert result.min() == 0

    def test_output_shape_preserved(self):
        arr = _make_rgb_array(32, 48)
        result = adjust_brightness(arr, 10)
        assert result.shape == arr.shape


class TestAdjustContrast:
    def test_positive_contrast_expands_range(self):
        """After positive contrast, max should be >= original max."""
        arr = np.array([[[100, 100, 100], [150, 150, 150]]], dtype=np.uint8)
        result = adjust_contrast(arr, 50)
        assert int(result.max()) >= int(arr.max())

    def test_negative_contrast_reduces_range(self):
        arr = np.array([[[50, 50, 50], [200, 200, 200]]], dtype=np.uint8)
        result = adjust_contrast(arr, -50)
        assert int(result.max()) - int(result.min()) < int(arr.max()) - int(arr.min())

    def test_zero_no_change(self):
        arr = np.full((2, 2, 3), 128, dtype=np.uint8)
        result = adjust_contrast(arr, 0)
        # 128 midpoint: factor*(128-128)+128 = 128 exactly
        assert np.allclose(result.astype(float), arr.astype(float), atol=1)

    def test_output_dtype_uint8(self):
        arr = _make_rgb_array()
        result = adjust_contrast(arr, 30)
        assert result.dtype == np.uint8


class TestAdjustGamma:
    def test_gamma_1_no_change(self):
        arr = _make_rgb_array()
        result = adjust_gamma(arr, 1.0)
        assert np.array_equal(result, arr)

    def test_gamma_less_than_1_brightens(self):
        """gamma < 1 should brighten mid-tones (increase values)."""
        arr = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = adjust_gamma(arr, 0.5)
        assert int(result[0, 0, 0]) > 128

    def test_gamma_greater_than_1_darkens(self):
        arr = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = adjust_gamma(arr, 2.0)
        assert int(result[0, 0, 0]) < 128

    def test_black_and_white_unchanged(self):
        arr = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        result = adjust_gamma(arr, 2.0)
        assert result[0, 0, 0] == 0
        assert result[0, 1, 0] == 255

    def test_invalid_gamma_raises(self):
        arr = _make_rgb_array()
        with pytest.raises(ValueError):
            adjust_gamma(arr, 0)


class TestApplyBlur:
    def test_zero_radius_returns_copy(self):
        arr = _make_rgb_array()
        result = apply_blur(arr, 0)
        assert result is not arr
        assert np.array_equal(result, arr)

    def test_blur_smooths_edges(self):
        """A sharp edge should have reduced gradient after blurring."""
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        arr[:, 10:, :] = 255
        result = apply_blur(arr, 3.0)
        # Interior of blurred region should not be all-0 or all-255
        mid_col = int(result[:, 10, 0].mean())
        assert 0 < mid_col < 255

    def test_output_shape_preserved(self):
        arr = _make_rgb_array(50, 60)
        result = apply_blur(arr, 2.0)
        assert result.shape == arr.shape


class TestApplySharpen:
    def test_zero_amount_returns_copy(self):
        arr = _make_rgb_array()
        result = apply_sharpen(arr, 0)
        assert result is not arr
        assert np.array_equal(result, arr)

    def test_negative_amount_returns_copy(self):
        arr = _make_rgb_array()
        result = apply_sharpen(arr, -1.0)
        assert result is not arr
        assert np.array_equal(result, arr)

    def test_positive_amount_changes_values(self):
        """Sharpening a gradient image should produce at least some different values."""
        arr = _make_rgb_array()
        result = apply_sharpen(arr, 1.0)
        assert not np.array_equal(result, arr)

    def test_output_shape_preserved(self):
        arr = _make_rgb_array(50, 60)
        result = apply_sharpen(arr, 0.5)
        assert result.shape == arr.shape

    def test_output_dtype_uint8(self):
        arr = _make_rgb_array()
        result = apply_sharpen(arr, 0.5)
        assert result.dtype == np.uint8


class TestApplyThreshold:
    def test_output_is_binary(self):
        arr = _make_rgb_array()
        result = apply_threshold(arr, 128)
        unique_vals = set(result.flatten().tolist())
        assert unique_vals <= {0, 255}

    def test_output_is_2d(self):
        arr = _make_rgb_array()
        result = apply_threshold(arr, 128)
        assert result.ndim == 2

    def test_bright_pixels_become_255(self):
        arr = np.full((1, 1, 3), 200, dtype=np.uint8)
        result = apply_threshold(arr, 128)
        assert result[0, 0] == 255

    def test_dark_pixels_become_0(self):
        arr = np.full((1, 1, 3), 50, dtype=np.uint8)
        result = apply_threshold(arr, 128)
        assert result[0, 0] == 0

    def test_threshold_at_value_inclusive(self):
        """Pixel equal to threshold should map to 255 (>=)."""
        arr = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = apply_threshold(arr, 128)
        assert result[0, 0] == 255

    def test_grayscale_input(self):
        gray = np.array([[50, 150, 200]], dtype=np.uint8)
        result = apply_threshold(gray, 100)
        assert result[0, 0] == 0
        assert result[0, 1] == 255


class TestInvertImage:
    def test_invert_white_gives_black(self):
        arr = np.full((1, 1, 3), 255, dtype=np.uint8)
        result = invert_image(arr)
        assert result[0, 0, 0] == 0

    def test_invert_black_gives_white(self):
        arr = np.zeros((1, 1, 3), dtype=np.uint8)
        result = invert_image(arr)
        assert result[0, 0, 0] == 255

    def test_double_invert_identity(self):
        arr = _make_rgb_array()
        result = invert_image(invert_image(arr))
        assert np.array_equal(result, arr)

    def test_midpoint_preserved(self):
        arr = np.full((1, 1, 3), 127, dtype=np.uint8)
        result = invert_image(arr)
        assert result[0, 0, 0] == 128


class TestRemoveBackground:
    def test_near_white_pixels_set_to_white(self):
        arr = np.full((1, 1, 3), 250, dtype=np.uint8)
        result = remove_background(arr, tolerance=30)
        assert np.all(result == 255)

    def test_non_white_pixels_unchanged(self):
        arr = np.full((1, 1, 3), 100, dtype=np.uint8)
        result = remove_background(arr, tolerance=30)
        assert np.all(result == 100)

    def test_mixed_image(self):
        arr = np.array([[[255, 255, 255], [100, 100, 100]]], dtype=np.uint8)
        result = remove_background(arr, tolerance=30)
        assert np.all(result[0, 0] == 255)
        assert np.all(result[0, 1] == 100)

    def test_boundary_pixel_at_threshold_is_removed(self):
        """Pixel at exactly the threshold (255 - tolerance) should be removed (>=, not >)."""
        # tolerance=20 → threshold=235; pixel at exactly 235 in all channels should be removed
        arr = np.full((1, 1, 3), 235, dtype=np.uint8)
        result = remove_background(arr, tolerance=20)
        assert np.all(result == 255)

    def test_pixel_one_below_threshold_unchanged(self):
        """Pixel just below threshold should NOT be removed."""
        # tolerance=20 → threshold=235; pixel at 234 should remain
        arr = np.full((1, 1, 3), 234, dtype=np.uint8)
        result = remove_background(arr, tolerance=20)
        assert np.all(result == 234)

    def test_off_white_background_removed_with_sufficient_tolerance(self):
        """Off-white background (e.g. 210 brightness) is removed when tolerance is high enough."""
        # tolerance=45 → threshold=210; pixel at (215, 212, 210) should be removed
        arr = np.array([[[215, 212, 210]]], dtype=np.uint8)
        result = remove_background(arr, tolerance=45)
        assert np.all(result == 255)

    def test_off_white_background_not_removed_with_low_tolerance(self):
        """Off-white background is NOT removed when tolerance is too low."""
        # tolerance=20 → threshold=235; pixel at (215, 212, 210) should remain
        arr = np.array([[[215, 212, 210]]], dtype=np.uint8)
        result = remove_background(arr, tolerance=20)
        assert np.all(result[0, 0] == np.array([215, 212, 210], dtype=np.uint8))

    def test_grayscale_input_2d(self):
        """2D grayscale input: pixels at or above threshold become 255."""
        gray = np.array([[240, 220, 180]], dtype=np.uint8)
        # tolerance=20 → threshold=235; 240>=235 → removed, 220<235 → unchanged
        result = remove_background(gray, tolerance=20)
        assert result[0, 0] == 255
        assert result[0, 1] == 220
        assert result[0, 2] == 180

    def test_grayscale_boundary_at_threshold(self):
        """2D grayscale: pixel at exactly the threshold value is removed."""
        gray = np.array([[235]], dtype=np.uint8)
        result = remove_background(gray, tolerance=20)
        assert result[0, 0] == 255

    def test_original_unchanged(self):
        """Input image is not modified in-place."""
        arr = np.full((4, 4, 3), 250, dtype=np.uint8)
        original = arr.copy()
        remove_background(arr, tolerance=30)
        assert np.array_equal(arr, original)


class TestCropToAspect:
    def test_output_shape_correct(self):
        arr = np.zeros((200, 100, 3), dtype=np.uint8)  # taller than wide
        result = crop_to_aspect(arr, 100, 100)  # square target
        assert result.shape == (100, 100, 3)

    def test_maintains_aspect_ratio_when_larger_target(self):
        """crop_to_aspect should resize to (int(width), int(height))."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = crop_to_aspect(arr, 200, 200)
        assert result.shape == (200, 200, 3)

    def test_wider_source_cropped_correctly(self):
        """When source is wider than target ratio, width should be trimmed."""
        arr = np.zeros((100, 300, 3), dtype=np.uint8)  # 3:1 source
        # Target 1:1 → expects square output
        result = crop_to_aspect(arr, 100, 100)
        assert result.shape[0] == result.shape[1]  # square

    def test_taller_source_cropped_correctly(self):
        arr = np.zeros((300, 100, 3), dtype=np.uint8)  # 1:3 source
        result = crop_to_aspect(arr, 100, 100)
        assert result.shape[0] == result.shape[1]

    def test_no_change_when_aspect_matches(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = crop_to_aspect(arr, 100, 100)
        assert result.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# preprocess pipeline
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_empty_params_returns_copy(self):
        arr = _make_rgb_array()
        result = preprocess(arr, {})
        assert np.array_equal(result, arr)
        assert result is not arr

    def test_invert_applied(self):
        arr = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = preprocess(arr, {"invert": True})
        assert np.all(result == 155)

    def test_brightness_applied(self):
        arr = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = preprocess(arr, {"brightness": 50})
        assert np.all(result > 100)

    def test_threshold_applied(self):
        arr = _make_rgb_array()
        result = preprocess(arr, {"threshold": 128.0})
        # Should be 2D binary
        assert result.ndim == 2
        unique = set(result.flatten().tolist())
        assert unique <= {0, 255}

    def test_pipeline_chain(self):
        """All steps combined must not crash and must return ndarray."""
        arr = _make_rgb_array()
        params = {
            "brightness": 10,
            "contrast": 10,
            "gamma": 1.2,
            "blur": 1.0,
            "sharpen": 0.5,
            "remove_background": 20.0,
            "invert": True,
        }
        result = preprocess(arr, params)
        assert isinstance(result, np.ndarray)

    def test_crop_params_applied(self):
        """crop_width/crop_height resize output to the requested dimensions."""
        arr = _make_rgb_array(100, 100)
        result = preprocess(arr, {"crop_width": 50, "crop_height": 50})
        # Output should be (50, 50, 3) — square crop of a square source
        assert result.shape == (50, 50, 3)
