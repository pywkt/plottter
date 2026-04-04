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

    def test_returns_polylines_with_dark_source_image(self):
        """An image with a dark region should produce at least one polyline."""
        img = make_single_dark_block()
        params = {"_source_image": img, "multi_pass": False}
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_pure_white_image_does_not_crash(self):
        """All-white image: no dark areas to trace, so result is empty."""
        img = make_white_image()
        params = {"_source_image": img}
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        # White image has nothing to draw (all pixels above brightness ceiling)
        assert result == []

    def test_erase_params_defined(self):
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "erase_min" in names
        assert "erase_max" in names
        assert "erase_radius_min" in names
        assert "erase_radius_max" in names
        assert "tone" in names
        assert "erase_radius" not in names
        assert "erase_amount" not in names

    def test_parameters_defined(self):
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "line_density" in names
        assert "line_max_limit" in names
        assert "invert" in names
        assert "brightness" in names
        assert "contrast" in names
        assert "blur_radius" in names
        assert "x_offset_mm" in names
        assert "y_offset_mm" in names
        # Deprecated params must not be present
        assert "block_size" not in names
        assert "brightness_ceiling" not in names
        assert "step_size_px" not in names
        assert "max_steps" not in names

    def test_presets_defined(self):
        presets = self.gen.get_presets()
        assert len(presets) >= 1
        assert presets[0].name == "Default"

    def test_new_parameters_defined(self):
        """squiggle params, angle_tests, line_length_px must exist."""
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "squiggle_min_length" in names
        assert "squiggle_max_length" in names
        assert "squiggle_max_deviation" in names
        assert "angle_tests" in names
        assert "line_length_px" in names
        assert "step_size_px" not in names


# ---------------------------------------------------------------------------
# _bresenham_line
# ---------------------------------------------------------------------------


class TestBresenhamLine:
    def setup_method(self):
        self.gen = SketchGenerator()

    def test_horizontal_line(self):
        """Horizontal line should yield all x values between start and end."""
        pts = list(self.gen._bresenham_line(0, 5, 4, 5))
        assert pts == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]

    def test_vertical_line(self):
        """Vertical line should yield all y values between start and end."""
        pts = list(self.gen._bresenham_line(3, 0, 3, 3))
        assert pts == [(3, 0), (3, 1), (3, 2), (3, 3)]

    def test_single_point(self):
        """Zero-length line should yield just the single point."""
        pts = list(self.gen._bresenham_line(7, 3, 7, 3))
        assert pts == [(7, 3)]

    def test_both_endpoints_included(self):
        """Start and end points must both appear in the result."""
        x0, y0, x1, y1 = 2, 3, 8, 7
        pts = list(self.gen._bresenham_line(x0, y0, x1, y1))
        assert (x0, y0) in pts
        assert (x1, y1) in pts

    def test_diagonal_has_correct_length(self):
        """45-degree diagonal: number of pixels ≈ max(|dx|, |dy|) + 1."""
        pts = list(self.gen._bresenham_line(0, 0, 5, 5))
        assert len(pts) == 6  # 0..5 inclusive

    def test_no_duplicate_pixels(self):
        """Each pixel coordinate should appear at most once."""
        pts = list(self.gen._bresenham_line(0, 0, 10, 7))
        assert len(pts) == len(set(pts))

    def test_reversed_line_same_pixels(self):
        """Reversing start/end should trace the same pixel set."""
        pts_fwd = set(self.gen._bresenham_line(1, 2, 8, 6))
        pts_rev = set(self.gen._bresenham_line(8, 6, 1, 2))
        assert pts_fwd == pts_rev


# ---------------------------------------------------------------------------
# _find_darkest_line
# ---------------------------------------------------------------------------


class TestFindDarkestLine:
    def setup_method(self):
        self.gen = SketchGenerator()

    def test_returns_none_when_all_candidates_out_of_bounds(self):
        """If current position is at the center of a tiny image, most circle
        points are outside — with radius > image_size/2 all are out of bounds."""
        img = np.full((5, 5), 128, dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 2, 2, 8, 50)
        assert result is None

    def test_finds_darker_direction(self):
        """Returned endpoint should be in the darker half of the image."""
        # Left half black, right half white; seed near left edge
        h, w = 64, 64
        img = np.full((h, w), 255, dtype=np.uint8)
        img[:, : w // 2] = 0  # left half black
        # Seed in the right-half, near the boundary; dark half is to the left
        result = self.gen._find_darkest_line(img, w // 2 + 1, h // 2, 8, 10)
        assert result is not None
        end_x, end_y, avg_brightness = result
        # The winning direction should lead into the darker left half
        assert avg_brightness < 255.0

    def test_avg_brightness_is_in_range(self):
        """avg_brightness must be in [0, 255]."""
        img = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 32, 32, 8, 10)
        if result is not None:
            _, _, avg_brightness = result
            assert 0.0 <= avg_brightness <= 255.0

    def test_endpoint_within_image_bounds(self):
        """The returned endpoint must be inside the image."""
        img = np.zeros((64, 64), dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 32, 32, 16, 10)
        assert result is not None
        end_x, end_y, _ = result
        assert 0 <= end_x < 64
        assert 0 <= end_y < 64

    def test_best_direction_lower_brightness_than_worst(self):
        """The chosen direction should have lower (or equal) avg brightness
        than any other valid direction."""
        rng = np.random.default_rng(99)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        angle_tests = 8
        line_length_px = 8
        result = self.gen._find_darkest_line(img, 32, 32, angle_tests, line_length_px)
        assert result is not None
        best_end_x, best_end_y, best_avg = result

        # Manually compute avg brightness for each candidate; best must win
        h, w = img.shape[:2]
        for i in range(angle_tests):
            angle = i * 2.0 * np.pi / angle_tests
            ex = 32 + int(round(np.cos(angle) * line_length_px))
            ey = 32 + int(round(np.sin(angle) * line_length_px))
            if not (0 <= ey < h and 0 <= ex < w):
                continue
            pixels = list(self.gen._bresenham_line(32, 32, ex, ey))
            if not all(0 <= py < h and 0 <= px < w for px, py in pixels):
                continue
            avg = sum(int(img[py, px]) for px, py in pixels) / len(pixels)
            assert best_avg <= avg + 1e-9, (
                f"Best avg {best_avg:.1f} > candidate avg {avg:.1f} for angle {i}"
            )

    def test_bresenham_circle_mode_angle_tests_36(self):
        """angle_tests=36 triggers Bresenham circle mode; result must be valid."""
        img = np.zeros((128, 128), dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 64, 64, 36, 20)
        assert result is not None
        end_x, end_y, avg_brightness = result
        assert 0 <= end_x < 128
        assert 0 <= end_y < 128
        assert avg_brightness == 0.0  # all-black image


# ---------------------------------------------------------------------------
# _erase_along_path
# ---------------------------------------------------------------------------


class TestEraseAlongPath:
    def setup_method(self):
        self.gen = SketchGenerator()

    def test_erase_brightens_along_path(self):
        """After erasing, pixels along the path should be brighter."""
        img = np.zeros((32, 32), dtype=np.uint8)  # all black
        path = [(16.0, 16.0), (17.0, 16.0), (18.0, 16.0)]
        # Dark pixels: eased=0, so amount=erase_min
        self.gen._erase_along_path(img, path, erase_min=50, erase_max=100, radius_min=1, radius_max=4, tone=0.5)
        # Center should be brightened by at least erase_min
        assert img[16, 16] >= 50

    def test_erase_does_not_exceed_255(self):
        """Erasing on a bright image must clamp to 255."""
        img = np.full((32, 32), 230, dtype=np.uint8)
        path = [(16.0, 16.0)]
        # Bright pixel (230): high eased value → amount approaches erase_max=200
        self.gen._erase_along_path(img, path, erase_min=1, erase_max=200, radius_min=1, radius_max=5, tone=0.5)
        assert img[16, 16] == 255

    def test_empty_path_no_change(self):
        """Empty path must not modify the image."""
        img = np.zeros((16, 16), dtype=np.uint8)
        original = img.copy()
        self.gen._erase_along_path(img, [], erase_min=1, erase_max=50, radius_min=1, radius_max=3, tone=0.5)
        np.testing.assert_array_equal(img, original)

    def test_erase_affects_region_based_on_brightness(self):
        """Bright center pixel → larger erase region; pixels beyond radius_max untouched."""
        # All grey (200) image: eased > 0 so radius > radius_min
        img = np.full((32, 32), 200, dtype=np.uint8)
        path = [(16.0, 16.0)]
        radius_min, radius_max = 1, 4
        self.gen._erase_along_path(img, path, erase_min=1, erase_max=80,
                                   radius_min=radius_min, radius_max=radius_max, tone=0.5)
        # Center should be brighter
        assert img[16, 16] > 200
        # Pixel well beyond radius_max should be unchanged
        assert img[16, 16 + radius_max + 2] == 200

    def test_erase_delta_matches_actual_sum_change(self):
        """Returned delta equals the actual change in array sum (within ±1 fp tolerance).

        This verifies criterion (b) from the task: the incremental running-sum
        approach tracks `lightened.mean()` within ±1 of the full computation.
        """
        rng = np.random.default_rng(42)
        # Mix of dark, mid, and near-white pixels so clamping is exercised.
        img = rng.integers(50, 250, (32, 32), dtype=np.uint8)
        path = [(10.0, 10.0), (12.0, 10.0), (14.0, 10.0), (14.0, 12.0)]
        before_sum = float(img.sum())
        delta = self.gen._erase_along_path(img, path, erase_min=1, erase_max=60,
                                           radius_min=1, radius_max=3, tone=0.5)
        after_sum = float(img.sum())
        assert abs(delta - (after_sum - before_sum)) < 1.0

    def test_erase_delta_with_full_clamping(self):
        """Delta is correct even when all affected pixels clamp to 255."""
        img = np.full((32, 32), 230, dtype=np.uint8)
        path = [(16.0, 16.0)]
        before_sum = float(img.sum())
        delta = self.gen._erase_along_path(img, path, erase_min=1, erase_max=200,
                                           radius_min=1, radius_max=5, tone=0.5)
        after_sum = float(img.sum())
        assert abs(delta - (after_sum - before_sum)) < 1.0

    def test_dark_pixels_barely_brightened(self):
        """Dark pixels get eased≈0 → amount=erase_min (barely brightened)."""
        img = np.zeros((32, 32), dtype=np.uint8)
        path = [(16.0, 16.0)]
        erase_min = 3
        self.gen._erase_along_path(img, path, erase_min=erase_min, erase_max=100,
                                   radius_min=1, radius_max=4, tone=0.5)
        # lum=0 → t=0 → eased=0 → amount=erase_min exactly
        assert img[16, 16] == erase_min

    def test_bright_pixels_significantly_more_brightened_than_dark(self):
        """Bright pixels receive much more brightening per erase call than dark pixels."""
        # Compare delta for a single bright path point vs a single dark path point
        img_bright = np.zeros((32, 32), dtype=np.uint8)
        img_bright[16, 16] = 200
        img_dark = np.zeros((32, 32), dtype=np.uint8)
        img_dark[16, 16] = 5

        delta_bright = self.gen._erase_along_path(
            img_bright, [(16.0, 16.0)], erase_min=1, erase_max=100,
            radius_min=1, radius_max=4, tone=0.5)
        delta_dark = self.gen._erase_along_path(
            img_dark, [(16.0, 16.0)], erase_min=1, erase_max=100,
            radius_min=1, radius_max=4, tone=0.5)

        assert delta_bright > delta_dark

    def test_dark_spot_requires_many_iterations_to_reach_200(self):
        """Repeated erasing at a dark spot takes many iterations to reach brightness 200+."""
        img = np.zeros((10, 10), dtype=np.uint8)
        path = [(5.0, 5.0)]
        iterations = 0
        while img[5, 5] < 200 and iterations < 10_000:
            self.gen._erase_along_path(img, path, erase_min=1, erase_max=100,
                                       radius_min=1, radius_max=4, tone=0.5)
            iterations += 1
        assert iterations > 15  # dark areas accumulate many fine strokes


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


class TestGenerateLoop:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def _make_dark_image(self, h: int = 64, w: int = 64, dark_value: int = 0) -> np.ndarray:
        """Solid dark image."""
        return np.full((h, w), dark_value, dtype=np.uint8)

    def test_higher_density_produces_more_paths(self):
        """Higher line_density → more output polylines."""
        img = make_single_dark_block(h=64, w=64, dark_block_row=1, dark_block_col=1)
        params_low = {"_source_image": img.copy(), "line_density": 10.0, "line_max_limit": 500}
        params_high = {"_source_image": img.copy(), "line_density": 80.0, "line_max_limit": 500}
        result_low = self.gen.generate(params_low, self.canvas)
        result_high = self.gen.generate(params_high, self.canvas)
        assert len(result_high) >= len(result_low)

    def test_lines_concentrate_in_dark_areas(self):
        """Paths should originate from dark regions, not bright regions."""
        # Image with left half black, right half white
        h, w = 64, 64
        img = np.full((h, w), 255, dtype=np.uint8)
        img[:, : w // 2] = 0  # left half black
        params = {
            "_source_image": img,
            "line_density": 30.0,
            "line_max_limit": 200,
            "line_min_length": 2,
        }
        canvas = self.canvas
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        result = self.gen.generate(params, canvas)
        assert len(result) > 0
        # Most path points should be in the left (dark) half of the mm range
        draw_mid_x = (draw_x1 + draw_x2) / 2.0
        points_in_dark = sum(
            1 for path in result for x, _y in path if x < draw_mid_x
        )
        total_points = sum(len(p) for p in result)
        assert points_in_dark > total_points * 0.7

    def test_output_in_mm_coordinates(self):
        """All path points must be within the canvas drawing area (mm)."""
        img = make_single_dark_block()
        params = {"_source_image": img, "line_density": 30.0, "multi_pass": False}
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        # Allow a small floating-point margin
        margin = 0.5
        for path in result:
            for x, y in path:
                assert draw_x1 - margin <= x <= draw_x2 + margin, f"x={x} out of range"
                assert draw_y1 - margin <= y <= draw_y2 + margin, f"y={y} out of range"

    def test_line_max_limit_respected(self):
        """Total line segments must stay close to line_max_limit.

        The limit governs when we *stop starting* new paths, so the total can
        overshoot by at most ``line_max_length - 1`` segments (the tail of the
        last path).
        """
        line_max_limit = 50
        line_max_length = 10
        img = self._make_dark_image()
        params = {
            "_source_image": img,
            "line_density": 90.0,
            "line_max_limit": line_max_limit,
            "line_min_length": 2,
            "line_max_length": line_max_length,
            "multi_pass": False,
        }
        result = self.gen.generate(params, self.canvas)
        total_segments = sum(len(p) - 1 for p in result if len(p) > 1)
        # Allow the last path to push slightly over the limit
        assert total_segments <= line_max_limit + line_max_length - 1

    def test_x_y_offset_applied(self):
        """x_offset_mm / y_offset_mm must shift all output coordinates."""
        img = make_single_dark_block()
        params_base = {"_source_image": img.copy(), "line_density": 20.0, "multi_pass": False}
        params_shifted = {
            "_source_image": img.copy(),
            "line_density": 20.0,
            "x_offset_mm": 10.0,
            "y_offset_mm": 5.0,
            "multi_pass": False,
        }
        result_base = self.gen.generate(params_base, self.canvas)
        result_shifted = self.gen.generate(params_shifted, self.canvas)
        assert len(result_base) > 0
        assert len(result_shifted) == len(result_base)
        for path_b, path_s in zip(result_base, result_shifted):
            for (xb, yb), (xs, ys) in zip(path_b, path_s):
                assert abs(xs - xb - 10.0) < 1e-6
                assert abs(ys - yb - 5.0) < 1e-6

    def test_dark_image_produces_more_lines_than_bright(self):
        """A dark image should produce more polylines than a bright image.

        Uses short paths (line_max_length=10) and a high segment limit so the
        density target — not the segment cap — controls when each run stops.
        A very dark image (avg=10) needs much more brightening to reach its
        50%-density target than a bright image (avg=200), so it produces
        significantly more paths.
        """
        dark_img = np.full((64, 64), 10, dtype=np.uint8)
        bright_img = np.full((64, 64), 200, dtype=np.uint8)
        common = {
            "line_density": 50.0,
            "line_max_limit": 10_000,
            "line_max_length": 10,
            "line_min_length": 2,
            "line_length_px": 5,
        }
        result_dark = self.gen.generate({"_source_image": dark_img, **common}, self.canvas)
        result_bright = self.gen.generate({"_source_image": bright_img, **common}, self.canvas)
        assert len(result_dark) > len(result_bright)

    def test_fit_mode_respected(self):
        """'fit' mode should map output to a smaller coordinate rect than 'fill' mode.

        A wide image (4:1 aspect ratio) on a portrait A4 canvas:
        - fill: maps to the full drawing area in y
        - fit:  maps to a height-constrained centered rect (~47.5mm tall)

        The maximum y-coordinate in 'fit' mode must be well below the full
        canvas height seen in 'fill' mode.
        """
        # Wide image (4:1 aspect) — in 'fit' mode this will be width-limited,
        # producing a much shorter y extent than in 'fill' mode.
        img = make_single_dark_block(h=32, w=128)
        common = {
            "line_density": 1.0,
            "line_max_limit": 200,
            "squiggle_min_length": 1,
            "squiggle_max_deviation": 50.0,
            "line_length_px": 8,
        }
        result_fill = self.gen.generate(
            {"_source_image": img.copy(), "image_fit_mode": "fill", **common},
            self.canvas,
        )
        result_fit = self.gen.generate(
            {"_source_image": img.copy(), "image_fit_mode": "fit", **common},
            self.canvas,
        )
        assert len(result_fill) > 0, "fill mode produced no output"
        assert len(result_fit) > 0, "fit mode produced no output"

        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        draw_h = draw_y2 - draw_y1

        max_y_fill = max(y for path in result_fill for _, y in path)
        max_y_fit = max(y for path in result_fit for _, y in path)

        # fit rect for a 4:1 image on portrait canvas ≈ 47.5mm tall (centred).
        # The fill max_y should reach much further down the canvas than fit max_y.
        assert max_y_fit < max_y_fill, (
            f"Expected fit max_y ({max_y_fit:.1f}) < fill max_y ({max_y_fill:.1f})"
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def test_preset_names(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Default" in names
        assert "Sketch Lines" in names
        assert "Contour Sketch" in names
        assert "Dense Crosshatch" in names
        assert "Loose Sketch" in names
        assert "Edge Trace" in names

    def test_all_presets_generate_valid_output(self):
        """Every preset must produce a non-empty list of polylines on a dark image."""
        # Use a fully dark image so squiggles can always trace without hitting
        # the brightness deviation limit.
        img = np.zeros((64, 64), dtype=np.uint8)
        for preset in self.gen.get_presets():
            params = dict(preset.params)
            params["_source_image"] = img.copy()
            # Keep limits small for speed; override squiggle_min_length for permissiveness
            params["line_max_limit"] = 200
            params["squiggle_min_length"] = 1
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list), f"Preset '{preset.name}' did not return a list"
            assert len(result) > 0, f"Preset '{preset.name}' produced no output"
            for path in result:
                assert len(path) >= 2, f"Preset '{preset.name}' produced a degenerate path"

    def test_presets_are_complete(self):
        """Every preset must include all generator parameters (no partial presets)."""
        param_names = {p.name for p in self.gen.get_parameters()}
        for preset in self.gen.get_presets():
            missing = param_names - set(preset.params.keys())
            assert not missing, (
                f"Preset '{preset.name}' is missing params: {missing}"
            )


# ---------------------------------------------------------------------------
# Coverage map
# ---------------------------------------------------------------------------


class TestCoverageMap:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def test_path_pixel_coords_horizontal(self):
        """Horizontal segment (0,5)→(4,5) produces exactly 5 unique pixels."""
        xs, ys = SketchGenerator._path_pixel_coords(
            [(0.0, 5.0), (4.0, 5.0)], width=10, height=10
        )
        assert xs.size == 5
        assert ys.size == 5
        np.testing.assert_array_equal(ys, [5, 5, 5, 5, 5])
        np.testing.assert_array_equal(np.sort(xs), [0, 1, 2, 3, 4])

    def test_path_pixel_coords_unique(self):
        """Rasterized path must have no duplicate pixel positions."""
        pts = [(0.0, 0.0), (10.0, 10.0), (5.0, 5.0)]
        xs, ys = SketchGenerator._path_pixel_coords(pts, width=20, height=20)
        flat = ys * 20 + xs
        assert flat.size == np.unique(flat).size

    def test_path_pixel_coords_empty_for_single_point(self):
        """Single-point path produces empty arrays (no segments)."""
        xs, ys = SketchGenerator._path_pixel_coords([(5.0, 5.0)], width=10, height=10)
        assert xs.size == 0
        assert ys.size == 0

    def test_expand_pixel_coords_radius_0_unchanged(self):
        """Radius 0 returns the original coordinates unchanged."""
        xs = np.array([5, 6, 7], dtype=np.int32)
        ys = np.array([5, 5, 5], dtype=np.int32)
        ex, ey = SketchGenerator._expand_pixel_coords(xs, ys, 20, 20, 0)
        assert ex.size == 3
        np.testing.assert_array_equal(ex, xs)
        np.testing.assert_array_equal(ey, ys)

    def test_expand_pixel_coords_radius_1_single_pixel(self):
        """Radius 1 around a single interior pixel produces 9 pixels."""
        xs = np.array([5], dtype=np.int32)
        ys = np.array([5], dtype=np.int32)
        ex, ey = SketchGenerator._expand_pixel_coords(xs, ys, 20, 20, 1)
        assert ex.size == 9

    def test_expand_pixel_coords_clamps_to_bounds(self):
        """Expansion at image edge must not produce out-of-bounds coordinates."""
        xs = np.array([0], dtype=np.int32)
        ys = np.array([0], dtype=np.int32)
        ex, ey = SketchGenerator._expand_pixel_coords(xs, ys, 10, 10, 2)
        assert np.all(ex >= 0) and np.all(ex < 10)
        assert np.all(ey >= 0) and np.all(ey < 10)

    def test_coverage_increments_on_accepted_path(self):
        """Coverage accumulates — verified indirectly: first path is always accepted."""
        img = np.zeros((32, 32), dtype=np.uint8)  # all black
        params = {
            "_source_image": img,
            "line_density": 5.0,
            "line_max_limit": 50,
            "max_pixel_coverage": 1,
            "max_overlap_ratio": 0.55,
            "coverage_radius": 0,
            "squiggle_min_length": 1,
        }
        result = self.gen.generate(params, self.canvas)
        # With coverage_radius=0 and a fresh image, the first path is always accepted
        assert len(result) > 0

    def test_paths_rejected_when_overlap_exceeds_threshold(self):
        """Paths rejected when overlap ratio is very strict (max_overlap_ratio≈0)."""
        img = np.zeros((64, 64), dtype=np.uint8)
        # With max_overlap_ratio=0.01 and max_pixel_coverage=1, almost every second
        # path will be rejected since the first path covers a region.
        # Permissive settings allow far more paths.
        common = {
            "line_density": 5.0,
            "line_max_limit": 200,
            "squiggle_min_length": 1,
            "coverage_radius": 1,
        }
        result_strict = self.gen.generate(
            {**common, "_source_image": img.copy(), "max_pixel_coverage": 1, "max_overlap_ratio": 0.01},
            self.canvas,
        )
        result_permissive = self.gen.generate(
            {**common, "_source_image": img.copy(), "max_pixel_coverage": 10, "max_overlap_ratio": 0.99},
            self.canvas,
        )
        # Permissive settings allow more or equal accepted paths
        assert len(result_permissive) >= len(result_strict)

    def test_higher_max_pixel_coverage_produces_more_paths(self):
        """max_pixel_coverage=1 produces sparser output than max_pixel_coverage=5."""
        img = np.zeros((64, 64), dtype=np.uint8)
        common = {
            "line_density": 3.0,
            "line_max_limit": 300,
            "squiggle_min_length": 1,
            "coverage_radius": 1,
            "max_overlap_ratio": 0.55,
        }
        result_low = self.gen.generate(
            {**common, "_source_image": img.copy(), "max_pixel_coverage": 1},
            self.canvas,
        )
        result_high = self.gen.generate(
            {**common, "_source_image": img.copy(), "max_pixel_coverage": 5},
            self.canvas,
        )
        assert len(result_high) >= len(result_low)

    def test_gradient_data_stays_accurate_during_generation(self):
        """Original source image is not modified during generation (gradient integrity)."""
        img = make_single_dark_block()
        source_copy = img.copy()
        params = {
            "_source_image": img,
            "line_density": 2.0,
            "line_max_limit": 50,
            "directionality": 30.0,
            "edge_power": 20.0,
        }
        result = self.gen.generate(params, self.canvas)
        # Source image passed to generator must be unchanged
        np.testing.assert_array_equal(img, source_copy)
        assert isinstance(result, list)

    def test_new_params_defined(self):
        """max_pixel_coverage, max_overlap_ratio, coverage_radius must be parameters."""
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "max_pixel_coverage" in names
        assert "max_overlap_ratio" in names
        assert "coverage_radius" in names


# ---------------------------------------------------------------------------
# Multi-pass generation
# ---------------------------------------------------------------------------


class TestMultiPass:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def test_multi_pass_param_defined(self):
        """multi_pass BoolParam must be registered."""
        names = {p.name for p in self.gen.get_parameters()}
        assert "multi_pass" in names

    def test_multi_pass_produces_output(self):
        """multi_pass=True on a dark image should produce paths (layered output)."""
        img = np.zeros((80, 80), dtype=np.uint8)  # all black — enough room for len_scale=1.9
        params = {
            "_source_image": img,
            "multi_pass": True,
            "line_max_limit": 150,
            "squiggle_min_length": 1,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "multi_pass=True should produce paths on a dark image"

    def test_pass1_longer_strokes_than_pass3(self):
        """Pass 1 (len_scale=1.9) produces longer strokes than pass 3 (len_scale=0.62).

        Simulated by running single-pass with each pass's effective line_length_px.
        """
        img = np.zeros((80, 80), dtype=np.uint8)  # all black
        common = {
            "_source_image": img,
            "multi_pass": False,
            "line_max_limit": 200,
            "squiggle_min_length": 1,
            "squiggle_max_length": 10,
            "squiggle_max_deviation": 90.0,
        }
        # Simulate pass 1 and pass 3 profiles via single-pass with their len_scale
        result_p1 = self.gen.generate({**common, "line_length_px": max(1, int(20 * 1.90))}, self.canvas)
        result_p3 = self.gen.generate({**common, "line_length_px": max(1, int(20 * 0.62))}, self.canvas)

        def avg_seg_length(paths):
            lengths = []
            for path in paths:
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i - 1][0]
                    dy = path[i][1] - path[i - 1][1]
                    lengths.append((dx * dx + dy * dy) ** 0.5)
            return sum(lengths) / len(lengths) if lengths else 0.0

        assert result_p1, "pass1 config should produce paths"
        assert result_p3, "pass3 config should produce paths"
        avg1 = avg_seg_length(result_p1)
        avg3 = avg_seg_length(result_p3)
        assert avg1 > avg3, f"pass1 avg seg {avg1:.3f}mm should exceed pass3 avg {avg3:.3f}mm"

    def test_single_pass_still_works(self):
        """multi_pass=False should produce output identically to original behavior."""
        img = make_single_dark_block()
        params = {
            "_source_image": img,
            "multi_pass": False,
            "line_max_limit": 100,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "single-pass mode should still produce paths"
