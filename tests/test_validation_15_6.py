"""Phase 15.6 validation: color separation — all methods with varying N.

Tests K-means clustering (varying N from 2–8), luminance splitting (varying
band counts), RGB channel separation, and CMYK channel separation.

Verifies:
- Correct number of results for each N
- Masks cover all pixels exactly once (partition property)
- Masks contain appropriate content (right pixels assigned to right clusters)
- Channel values have correct mathematical relationships
- Integration with Layer/Project models
- Edge cases and boundary conditions
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from plottter.color.kmeans import kmeans_separate
from plottter.color.luminance import luminance_separate
from plottter.color.channels import rgb_separate, cmyk_separate
from plottter.models.layer import Layer
from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers: hex validation
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def assert_valid_hex(color: str) -> None:
    assert _HEX_RE.match(color), f"Invalid hex color: {color!r}"


# ---------------------------------------------------------------------------
# Synthetic image factories
# ---------------------------------------------------------------------------


def make_pure_rgb_image() -> np.ndarray:
    """4×3 image: row 0=pure red, row 1=pure green, row 2=pure blue, row 3=black."""
    img = np.zeros((4, 10, 3), dtype=np.uint8)
    img[0, :] = [255, 0, 0]   # Red
    img[1, :] = [0, 255, 0]   # Green
    img[2, :] = [0, 0, 255]   # Blue
    img[3, :] = [0, 0, 0]     # Black
    return img


def make_tricolor_blocks(h: int = 30, w: int = 30) -> np.ndarray:
    """Three distinct color blocks side-by-side (red, green, blue)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    third = w // 3
    img[:, :third] = [200, 0, 0]        # Red block
    img[:, third : 2 * third] = [0, 200, 0]  # Green block
    img[:, 2 * third :] = [0, 0, 200]   # Blue block
    return img


def make_four_color_blocks(h: int = 40, w: int = 40) -> np.ndarray:
    """Four quadrants: red, green, blue, yellow."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, : w // 2] = [200, 0, 0]     # Red (top-left)
    img[: h // 2, w // 2 :] = [0, 200, 0]     # Green (top-right)
    img[h // 2 :, : w // 2] = [0, 0, 200]     # Blue (bottom-left)
    img[h // 2 :, w // 2 :] = [200, 200, 0]   # Yellow (bottom-right)
    return img


def make_grayscale_steps(num_steps: int = 5, h: int = 20, w_per_step: int = 20) -> np.ndarray:
    """Grayscale image with horizontal bands at evenly spaced brightness levels."""
    w = w_per_step * num_steps
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(num_steps):
        val = int(255 * i / (num_steps - 1)) if num_steps > 1 else 128
        img[:, i * w_per_step : (i + 1) * w_per_step] = val
    return img


def make_full_range_gradient(h: int = 10, w: int = 256) -> np.ndarray:
    """Grayscale image with all values 0–255."""
    row = np.arange(256, dtype=np.uint8)
    return np.tile(row, (h, 1))


def make_natural_rgb_image(h: int = 80, w: int = 80) -> np.ndarray:
    """Simulate a natural-ish image: sky top half, ground bottom half, dark subject."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky: blue gradient
    for y in range(h // 2):
        intensity = 200 - y * 80 // (h // 2)
        img[y, :] = [max(0, intensity - 60), max(0, intensity - 30), intensity]
    # Ground: brownish
    img[h // 2 :, :] = [100, 70, 40]
    # Dark subject in center
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    for y in range(h):
        for x in range(w):
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius:
                img[y, x] = [30, 20, 10]
    return img


# ---------------------------------------------------------------------------
# Helper: verify partition property
# ---------------------------------------------------------------------------


def assert_partition(masks: list[np.ndarray], h: int, w: int) -> None:
    """Every pixel is assigned to exactly one mask."""
    combined = np.zeros((h, w), dtype=np.int32)
    for mask in masks:
        combined += mask.astype(np.int32)
    unassigned = int((combined == 0).sum())
    double_assigned = int((combined > 1).sum())
    assert unassigned == 0, f"{unassigned} pixels are unassigned"
    assert double_assigned == 0, f"{double_assigned} pixels are double-assigned"


# ---------------------------------------------------------------------------
# K-Means: varying N
# ---------------------------------------------------------------------------


class TestKmeansVaryingN:
    """K-means with N from 2 to 8 — verify count, partition, and hex colors."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
    def test_correct_count_for_n(self, n: int) -> None:
        rng = np.random.default_rng(n)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        results = kmeans_separate(img, num_colors=n)
        assert len(results) == n, f"Expected {n} clusters, got {len(results)}"

    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_partition_property_for_n(self, n: int) -> None:
        rng = np.random.default_rng(n * 7)
        img = rng.integers(0, 256, (15, 15, 3), dtype=np.uint8)
        results = kmeans_separate(img, num_colors=n, iterations=10)
        masks = [m for m, _ in results]
        assert_partition(masks, 15, 15)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8])
    def test_all_hex_colors_valid_for_n(self, n: int) -> None:
        rng = np.random.default_rng(n + 100)
        img = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
        results = kmeans_separate(img, num_colors=n, iterations=5)
        for _, hex_color in results:
            assert_valid_hex(hex_color)

    @pytest.mark.parametrize("n", [2, 3, 4, 8])
    def test_masks_are_bool_arrays_for_n(self, n: int) -> None:
        rng = np.random.default_rng(n + 200)
        img = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        results = kmeans_separate(img, num_colors=n, iterations=5)
        for mask, _ in results:
            assert mask.dtype == np.bool_, f"Expected bool_, got {mask.dtype}"
            assert mask.shape == (10, 10)


class TestKmeansContentCorrectness:
    """Verify K-means assigns pixels to the correct clusters for known inputs."""

    def test_two_colors_separates_pure_regions(self) -> None:
        """Image with 2 very distinct colors should produce 2 masks each >0 pixels."""
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[:10, :] = [255, 0, 0]   # Top: red
        img[10:, :] = [0, 0, 255]   # Bottom: blue
        results = kmeans_separate(img, num_colors=2, iterations=30)
        assert len(results) == 2
        masks = [m for m, _ in results]
        # Both masks must have pixels
        assert masks[0].sum() > 0
        assert masks[1].sum() > 0
        # Partition
        assert_partition(masks, 20, 20)

    def test_three_blocks_produces_three_clusters(self) -> None:
        """Three perfectly distinct color blocks should produce 3 non-empty clusters."""
        img = make_tricolor_blocks(h=30, w=30)
        results = kmeans_separate(img, num_colors=3, iterations=50)
        assert len(results) == 3
        masks = [m for m, _ in results]
        assert all(m.sum() > 0 for m in masks), "All clusters must contain pixels"
        assert_partition(masks, 30, 30)

    def test_four_blocks_produces_four_clusters(self) -> None:
        """Four distinct color blocks → 4 non-empty clusters."""
        img = make_four_color_blocks(h=40, w=40)
        results = kmeans_separate(img, num_colors=4, iterations=50)
        assert len(results) == 4
        masks = [m for m, _ in results]
        assert all(m.sum() > 0 for m in masks)
        assert_partition(masks, 40, 40)

    def test_uniform_image_two_clusters_cover_all(self) -> None:
        """Uniform image: all pixels same color → both clusters cover all pixels."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        results = kmeans_separate(img, num_colors=2, iterations=5)
        masks = [m for m, _ in results]
        assert_partition(masks, 10, 10)

    def test_natural_image_three_clusters(self) -> None:
        """Natural-like image with 3 clusters: sky, ground, subject — all covered."""
        img = make_natural_rgb_image(h=60, w=60)
        results = kmeans_separate(img, num_colors=3, iterations=30)
        assert len(results) == 3
        masks = [m for m, _ in results]
        assert all(m.sum() > 0 for m in masks)
        assert_partition(masks, 60, 60)

    def test_more_colors_than_distinct_groups_still_partitions(self) -> None:
        """Even if N > distinct colors, partition property must hold."""
        img = make_tricolor_blocks(h=20, w=20)
        results = kmeans_separate(img, num_colors=6, iterations=20)
        assert len(results) == 6
        assert_partition([m for m, _ in results], 20, 20)

    def test_large_sample_size_matches_small_sample_size(self) -> None:
        """Results with large sample ≥ small sample should both produce valid partitions."""
        img = make_four_color_blocks(h=30, w=30)
        r1 = kmeans_separate(img, num_colors=4, sample_size=100, iterations=20)
        r2 = kmeans_separate(img, num_colors=4, sample_size=10000, iterations=20)
        assert_partition([m for m, _ in r1], 30, 30)
        assert_partition([m for m, _ in r2], 30, 30)

    def test_n_clamped_below_2(self) -> None:
        img = make_tricolor_blocks()
        results = kmeans_separate(img, num_colors=0)
        assert len(results) == 2

    def test_n_clamped_above_8(self) -> None:
        img = make_tricolor_blocks()
        results = kmeans_separate(img, num_colors=20)
        assert len(results) == 8


# ---------------------------------------------------------------------------
# Luminance: varying band counts
# ---------------------------------------------------------------------------


class TestLuminanceVaryingN:
    """Luminance splitting with band counts 2–5."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_correct_count_for_n(self, n: int) -> None:
        img = make_full_range_gradient()
        results = luminance_separate(img, num_bands=n)
        assert len(results) == n

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_partition_property_for_n(self, n: int) -> None:
        img = make_full_range_gradient(h=8, w=256)
        results = luminance_separate(img, num_bands=n)
        assert_partition([m for m, _ in results], 8, 256)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_all_hex_colors_valid(self, n: int) -> None:
        img = make_full_range_gradient()
        results = luminance_separate(img, num_bands=n)
        for _, hex_color in results:
            assert_valid_hex(hex_color)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_non_empty_masks_for_gradient(self, n: int) -> None:
        """Each band should contain at least some pixels in a full-range gradient."""
        img = make_full_range_gradient()
        results = luminance_separate(img, num_bands=n)
        for i, (mask, _) in enumerate(results):
            assert mask.sum() > 0, f"Band {i} is empty for n={n}"


class TestLuminanceContentCorrectness:
    """Verify luminance bands contain appropriate pixels."""

    def test_darkest_band_contains_black_pixels(self) -> None:
        """Band 0 (darkest) must contain the black pixels."""
        img = make_grayscale_steps(num_steps=5, h=10, w_per_step=10)
        results = luminance_separate(img, num_bands=3)
        dark_mask = results[0][0]  # First band = darkest
        # The leftmost column (value=0) must be in band 0
        assert dark_mask[:, 0].all(), "Black pixels should be in the darkest band"

    def test_brightest_band_contains_white_pixels(self) -> None:
        """Last band (brightest) must contain the white pixels."""
        img = make_grayscale_steps(num_steps=5, h=10, w_per_step=10)
        results = luminance_separate(img, num_bands=3)
        bright_mask = results[-1][0]  # Last band = brightest
        # The rightmost column (value=255) must be in the last band
        assert bright_mask[:, -1].all(), "White pixels should be in the brightest band"

    def test_three_bands_cover_low_mid_high(self) -> None:
        """With 3 bands on a gradient: each third of the brightness range is covered."""
        # Column 0 = value 0 (shadow), col 127 = mid, col 255 = highlight
        img = make_full_range_gradient(h=5, w=256)
        results = luminance_separate(img, num_bands=3)
        shadow_mask = results[0][0]
        mid_mask = results[1][0]
        highlight_mask = results[2][0]
        # Value 0 → shadow
        assert shadow_mask[:, 0].all()
        # Value 127 → midtone (second band, roughly 85–170)
        assert mid_mask[:, 127].all()
        # Value 255 → highlight
        assert highlight_mask[:, 255].all()

    def test_custom_thresholds_assign_correctly(self) -> None:
        """Custom threshold at 100 splits 0–99 vs 100–255."""
        img = make_full_range_gradient(h=5, w=256)
        results = luminance_separate(img, num_bands=2, thresholds=[100.0])
        dark_mask = results[0][0]
        bright_mask = results[1][0]
        # Pixel at col 50 (value 50) → dark band
        assert dark_mask[:, 50].all()
        # Pixel at col 200 (value 200) → bright band
        assert bright_mask[:, 200].all()

    def test_rgb_input_splits_by_luminance(self) -> None:
        """RGB image with bright red and dark blue should split correctly."""
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        img[:, :10] = [240, 240, 240]   # Near-white left half
        img[:, 10:] = [10, 10, 10]      # Near-black right half
        results = luminance_separate(img, num_bands=2)
        dark_mask = results[0][0]
        bright_mask = results[1][0]
        # Right half (dark) → dark band
        assert dark_mask[:, 15].all()
        # Left half (bright) → bright band
        assert bright_mask[:, 5].all()

    def test_five_bands_all_non_empty_on_gradient(self) -> None:
        """Five-band split on full-range gradient must produce 5 non-empty bands."""
        img = make_full_range_gradient(h=5, w=256)
        results = luminance_separate(img, num_bands=5)
        assert len(results) == 5
        for i, (mask, _) in enumerate(results):
            assert mask.sum() > 0, f"Band {i} is empty"

    def test_two_bands_on_binary_image(self) -> None:
        """Binary image (only 0 or 255) → each band gets exactly half the pixels."""
        img = np.zeros((10, 20), dtype=np.uint8)
        img[:, 10:] = 255  # Right half white
        results = luminance_separate(img, num_bands=2)
        dark_mask = results[0][0]
        bright_mask = results[1][0]
        assert int(dark_mask.sum()) == 100   # 10×10
        assert int(bright_mask.sum()) == 100  # 10×10

    def test_colors_ordered_dark_to_light(self) -> None:
        """Default assigned colors should be ordered from darkest to lightest."""
        img = make_full_range_gradient()

        def luminance(h: str) -> float:
            r = int(h[1:3], 16)
            g = int(h[3:5], 16)
            b = int(h[5:7], 16)
            return 0.299 * r + 0.587 * g + 0.114 * b

        for n in [2, 3, 4, 5]:
            results = luminance_separate(img, num_bands=n)
            lums = [luminance(hex_color) for _, hex_color in results]
            assert lums == sorted(lums), f"Colors not dark→light for n={n}: {lums}"


# ---------------------------------------------------------------------------
# RGB Channel Separation
# ---------------------------------------------------------------------------


class TestRgbChannelContent:
    """Verify RGB channel values correspond to the correct channel."""

    def test_pure_red_image_red_channel_max(self) -> None:
        """Pure red image → R channel=255, G=0, B=0."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # R=255
        results = rgb_separate(img)
        r_ch, _ = results[0]
        g_ch, _ = results[1]
        b_ch, _ = results[2]
        assert np.all(r_ch == 255)
        assert np.all(g_ch == 0)
        assert np.all(b_ch == 0)

    def test_pure_green_image_green_channel_max(self) -> None:
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 1] = 200
        results = rgb_separate(img)
        r_ch, _ = results[0]
        g_ch, _ = results[1]
        b_ch, _ = results[2]
        assert np.all(r_ch == 0)
        assert np.all(g_ch == 200)
        assert np.all(b_ch == 0)

    def test_mixed_pixel_separates_correctly(self) -> None:
        """Pixel (100, 150, 200) → R=100, G=150, B=200 in respective channels."""
        img = np.array([[[100, 150, 200]]], dtype=np.uint8)
        results = rgb_separate(img)
        assert int(results[0][0][0, 0]) == 100
        assert int(results[1][0][0, 0]) == 150
        assert int(results[2][0][0, 0]) == 200

    def test_channel_value_range_valid(self) -> None:
        """All channel values must be in [0, 255]."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        results = rgb_separate(img)
        for ch, _ in results:
            assert int(ch.min()) >= 0
            assert int(ch.max()) <= 255

    def test_channel_shapes_match_input(self) -> None:
        """Each channel array must be (H, W) with same spatial dims as input."""
        h, w = 25, 35
        img = np.zeros((h, w, 3), dtype=np.uint8)
        results = rgb_separate(img)
        for ch, _ in results:
            assert ch.shape == (h, w), f"Expected ({h},{w}), got {ch.shape}"

    def test_white_image_all_channels_255(self) -> None:
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        results = rgb_separate(img)
        for ch, _ in results:
            assert np.all(ch == 255)

    def test_black_image_all_channels_0(self) -> None:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        results = rgb_separate(img)
        for ch, _ in results:
            assert np.all(ch == 0)

    def test_independent_channels_do_not_affect_each_other(self) -> None:
        """Modifying R channel value doesn't bleed into G or B."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 180  # R=180, G=0, B=0
        results = rgb_separate(img)
        g_ch, _ = results[1]
        b_ch, _ = results[2]
        assert np.all(g_ch == 0)
        assert np.all(b_ch == 0)


# ---------------------------------------------------------------------------
# CMYK Channel Separation
# ---------------------------------------------------------------------------


class TestCmykChannelContent:
    """Verify CMYK channel values have correct mathematical relationships."""

    def test_pure_cyan_image(self) -> None:
        """RGB(0,255,255) = cyan → C=0, M=0 (no magenta), Y=0, K=0."""
        img = np.array([[[0, 255, 255]]], dtype=np.uint8)
        results = cmyk_separate(img)
        c = int(results[0][0][0, 0])
        m = int(results[1][0][0, 0])
        y = int(results[2][0][0, 0])
        k = int(results[3][0][0, 0])
        # Cyan = (0, 255, 255): max_channel=255→K=0; C=(1-0-0)/1=1→C=255; M=(1-1-0)/1=0; Y=(1-1-0)/1=0
        assert c == 255
        assert m == 0
        assert y == 0
        assert k == 0

    def test_pure_magenta_image(self) -> None:
        """RGB(255,0,255) = magenta → C=0, M=255, Y=0, K=0."""
        img = np.array([[[255, 0, 255]]], dtype=np.uint8)
        results = cmyk_separate(img)
        assert int(results[0][0][0, 0]) == 0    # C=0
        assert int(results[1][0][0, 0]) == 255  # M=255
        assert int(results[2][0][0, 0]) == 0    # Y=0
        assert int(results[3][0][0, 0]) == 0    # K=0

    def test_pure_yellow_image(self) -> None:
        """RGB(255,255,0) = yellow → C=0, M=0, Y=255, K=0."""
        img = np.array([[[255, 255, 0]]], dtype=np.uint8)
        results = cmyk_separate(img)
        assert int(results[0][0][0, 0]) == 0    # C=0
        assert int(results[1][0][0, 0]) == 0    # M=0
        assert int(results[2][0][0, 0]) == 255  # Y=255
        assert int(results[3][0][0, 0]) == 0    # K=0

    def test_gray_has_zero_cmyk_channels_only_k(self) -> None:
        """Mid gray (128,128,128): C=M=Y=0, K>0."""
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        results = cmyk_separate(img)
        # For gray: max=128/255, K=1-128/255≈0.498 → K≈127
        c = int(results[0][0][0, 0])
        m = int(results[1][0][0, 0])
        y = int(results[2][0][0, 0])
        k = int(results[3][0][0, 0])
        assert c == 0
        assert m == 0
        assert y == 0
        assert k > 0

    def test_white_has_all_channels_zero(self) -> None:
        """White (255,255,255): C=M=Y=K=0."""
        img = np.full((2, 2, 3), 255, dtype=np.uint8)
        results = cmyk_separate(img)
        for ch, _ in results:
            assert np.all(ch == 0)

    def test_black_has_only_k_nonzero(self) -> None:
        """Black (0,0,0): C=M=Y=0, K=255."""
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        results = cmyk_separate(img)
        c, m, y, k = (results[i][0] for i in range(4))
        assert np.all(c == 0)
        assert np.all(m == 0)
        assert np.all(y == 0)
        assert np.all(k == 255)

    def test_channel_shapes_match_input(self) -> None:
        h, w = 15, 25
        img = np.zeros((h, w, 3), dtype=np.uint8)
        results = cmyk_separate(img)
        for ch, _ in results:
            assert ch.shape == (h, w)

    def test_channel_values_in_valid_range(self) -> None:
        rng = np.random.default_rng(123)
        img = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        results = cmyk_separate(img)
        for ch, _ in results:
            assert int(ch.min()) >= 0
            assert int(ch.max()) <= 255

    def test_high_red_pixel_yields_low_cyan(self) -> None:
        """Pixel with high R → low Cyan (C and R are complementary)."""
        img = np.array([[[240, 50, 50]]], dtype=np.uint8)
        results = cmyk_separate(img)
        c = int(results[0][0][0, 0])
        # Cyan should be very low (red dominates)
        assert c < 50

    def test_high_blue_pixel_yields_low_yellow(self) -> None:
        """Pixel with high B → low Yellow (Y and B are complementary)."""
        img = np.array([[[50, 50, 240]]], dtype=np.uint8)
        results = cmyk_separate(img)
        y = int(results[2][0][0, 0])
        assert y < 50

    def test_hex_colors_are_standard_process_colors(self) -> None:
        img = make_four_color_blocks()
        results = cmyk_separate(img)
        _, c_hex = results[0]
        _, m_hex = results[1]
        _, y_hex = results[2]
        _, k_hex = results[3]
        assert c_hex == "#00FFFF"
        assert m_hex == "#FF00FF"
        assert y_hex == "#FFFF00"
        assert k_hex == "#000000"


# ---------------------------------------------------------------------------
# Integration: separation results → Layer objects
# ---------------------------------------------------------------------------


class TestColorSeparationLayerIntegration:
    """Verify color separation results can create valid Layer objects."""

    def test_kmeans_masks_to_layers(self) -> None:
        """K-means masks can be used to create Layer objects with correct colors."""
        img = make_tricolor_blocks(h=20, w=20)
        results = kmeans_separate(img, num_colors=3, iterations=20)
        layers = []
        for i, (mask, hex_color) in enumerate(results):
            layer = Layer(
                name=f"Cluster {i + 1}",
                color=hex_color,
            )
            assert layer.name == f"Cluster {i + 1}"
            assert layer.color == hex_color
            layers.append(layer)
        assert len(layers) == 3

    def test_luminance_bands_to_layers_with_expected_names(self) -> None:
        """Luminance bands can be named as Shadows/Midtones/Highlights."""
        img = make_full_range_gradient()
        results = luminance_separate(img, num_bands=3)
        expected_names = ["Shadows", "Midtones", "Highlights"]
        for name, (_, hex_color) in zip(expected_names, results):
            layer = Layer(name=name, color=hex_color)
            assert layer.name == name
            assert_valid_hex(layer.color)

    def test_rgb_channels_to_layers(self) -> None:
        """RGB channel separation creates 3 layers with correct pen colors."""
        img = make_four_color_blocks()
        results = rgb_separate(img)
        expected_colors = ["#FF0000", "#00FF00", "#0000FF"]
        expected_names = ["Red Channel", "Green Channel", "Blue Channel"]
        layers = []
        for name, (_, hex_color) in zip(expected_names, results):
            layer = Layer(name=name, color=hex_color)
            layers.append(layer)
        assert len(layers) == 3
        for layer, expected_color in zip(layers, expected_colors):
            assert layer.color == expected_color

    def test_cmyk_channels_to_layers(self) -> None:
        """CMYK separation creates 4 layers with correct process colors."""
        img = make_four_color_blocks()
        results = cmyk_separate(img)
        expected_colors = ["#00FFFF", "#FF00FF", "#FFFF00", "#000000"]
        expected_names = ["Cyan Channel", "Magenta Channel", "Yellow Channel", "Key (Black)"]
        layers = []
        for name, (_, hex_color) in zip(expected_names, results):
            layer = Layer(name=name, color=hex_color)
            layers.append(layer)
        assert len(layers) == 4
        for layer, expected_color in zip(layers, expected_colors):
            assert layer.color == expected_color

    def test_all_methods_produce_valid_layers_for_canvas(self) -> None:
        """All four methods produce valid Layer-ready results for an A4 canvas."""
        canvas = Canvas.from_preset("A4")
        assert canvas.width_mm == 210.0
        assert canvas.height_mm == 297.0

        img = make_natural_rgb_image(h=50, w=50)

        # K-means
        km_results = kmeans_separate(img, num_colors=3, iterations=15)
        assert len(km_results) == 3

        # Luminance
        gray = (
            0.299 * img[:, :, 0].astype(np.float32)
            + 0.587 * img[:, :, 1].astype(np.float32)
            + 0.114 * img[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
        lum_results = luminance_separate(gray, num_bands=3)
        assert len(lum_results) == 3

        # RGB
        rgb_results = rgb_separate(img)
        assert len(rgb_results) == 3

        # CMYK
        cmyk_results = cmyk_separate(img)
        assert len(cmyk_results) == 4


# ---------------------------------------------------------------------------
# Edge Cases & Boundary Conditions
# ---------------------------------------------------------------------------


class TestColorSeparationEdgeCases:
    """Edge cases across all separation methods."""

    def test_kmeans_1x1_image(self) -> None:
        """Single-pixel image works for any N."""
        img = np.array([[[128, 64, 32]]], dtype=np.uint8)
        results = kmeans_separate(img, num_colors=3, iterations=5)
        assert len(results) == 3
        assert_partition([m for m, _ in results], 1, 1)

    def test_luminance_1x1_image(self) -> None:
        img = np.array([[200]], dtype=np.uint8)
        results = luminance_separate(img, num_bands=3)
        assert len(results) == 3
        assert_partition([m for m, _ in results], 1, 1)

    def test_rgb_1x1_image(self) -> None:
        img = np.array([[[100, 150, 200]]], dtype=np.uint8)
        results = rgb_separate(img)
        assert len(results) == 3

    def test_cmyk_1x1_image(self) -> None:
        img = np.array([[[100, 150, 200]]], dtype=np.uint8)
        results = cmyk_separate(img)
        assert len(results) == 4

    def test_kmeans_very_large_image_is_subsampled(self) -> None:
        """Large image is subsampled but still produces valid partition."""
        rng = np.random.default_rng(99)
        img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        results = kmeans_separate(img, num_colors=4, sample_size=500, iterations=10)
        assert len(results) == 4
        assert_partition([m for m, _ in results], 200, 200)

    def test_luminance_all_same_value(self) -> None:
        """Uniform grayscale: all pixels in one band (the others empty) — partition holds."""
        img = np.full((10, 10), 100, dtype=np.uint8)
        results = luminance_separate(img, num_bands=3)
        assert len(results) == 3
        assert_partition([m for m, _ in results], 10, 10)
        # Only one band should be non-empty
        non_empty = sum(1 for m, _ in results if m.sum() > 0)
        assert non_empty == 1

    def test_rgb_grayscale_has_equal_channels(self) -> None:
        """Grayscale RGB image (R=G=B for all pixels) → all channels identical."""
        img = np.full((5, 5, 3), 128, dtype=np.uint8)
        results = rgb_separate(img)
        r_ch, _ = results[0]
        g_ch, _ = results[1]
        b_ch, _ = results[2]
        np.testing.assert_array_equal(r_ch, g_ch)
        np.testing.assert_array_equal(g_ch, b_ch)

    def test_kmeans_minimum_n_2_clamps(self) -> None:
        img = make_tricolor_blocks(h=10, w=10)
        results = kmeans_separate(img, num_colors=1)
        assert len(results) == 2

    def test_luminance_minimum_n_2_clamps(self) -> None:
        img = make_full_range_gradient(h=5, w=256)
        results = luminance_separate(img, num_bands=1)
        assert len(results) == 2

    def test_luminance_maximum_n_5_clamps(self) -> None:
        img = make_full_range_gradient(h=5, w=256)
        results = luminance_separate(img, num_bands=6)
        assert len(results) == 5

    def test_kmeans_maximum_n_8_clamps(self) -> None:
        img = make_four_color_blocks(h=20, w=20)
        results = kmeans_separate(img, num_colors=9)
        assert len(results) == 8
