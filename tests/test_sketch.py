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
        params = {"_source_image": img}
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

    def test_new_parameters_defined(self):
        """line_min_length, line_max_length, angle_tests, step_size_px must exist."""
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "line_min_length" in names
        assert "line_max_length" in names
        assert "angle_tests" in names
        assert "step_size_px" in names


# ---------------------------------------------------------------------------
# _trace_darkest_path
# ---------------------------------------------------------------------------


class TestTraceDarkestPath:
    def setup_method(self):
        self.gen = SketchGenerator()

    def test_path_starts_at_seed(self):
        """First point of the path must be the seed pixel."""
        img = make_white_image(64, 64)
        path = self.gen._trace_darkest_path(img, 10, 20, 8, 50, 2)
        assert len(path) >= 1
        assert path[0] == (20.0, 10.0)  # (px_x, px_y) = (seed_x, seed_y)

    def test_path_moves_toward_dark_area(self):
        """Path should advance toward a dark column on the right of the seed."""
        # Image: left half light-grey (below brightness ceiling), right half black
        h, w = 64, 64
        img = np.full((h, w), 200, dtype=np.uint8)  # 200 < 240 ceiling
        img[:, w // 2 :] = 0  # right half is black — clearly darker
        # Seed just left of the midpoint; 8-direction search will find rightward dark
        seed_y, seed_x = h // 2, w // 2 - 2
        path = self.gen._trace_darkest_path(img, seed_y, seed_x, 8, 30, 2)
        assert len(path) > 1
        # The x-coordinate should generally increase (moving right toward dark)
        xs = [p[0] for p in path]
        assert xs[-1] > xs[0]

    def test_path_stops_at_image_boundary(self):
        """Path coordinates must stay within image bounds even when tracing near an edge."""
        # All-black image: brightness never exceeds the ceiling, so the boundary
        # (not the brightness check) must be what stops the path.
        img = np.zeros((32, 32), dtype=np.uint8)
        path = self.gen._trace_darkest_path(img, 1, 1, 8, 200, 3)
        assert len(path) > 1, "Path should trace before hitting boundary"
        for px_x, px_y in path:
            assert 0 <= round(px_x) < 32, f"px_x={px_x} out of bounds"
            assert 0 <= round(px_y) < 32, f"px_y={px_y} out of bounds"

    def test_path_length_bounded_by_max_length(self):
        """Returned path must have at most max_length positions."""
        # All-black image: brightness never exceeds the ceiling, so max_length
        # (not the brightness check) must be what caps the path.
        img = np.zeros((128, 128), dtype=np.uint8)
        max_length = 20
        path = self.gen._trace_darkest_path(img, 64, 64, 8, max_length, 2)
        assert len(path) == max_length  # should reach the cap, not stop early

    def test_angle_tests_4_axis_aligned(self):
        """With 4 directions (0°, 90°, 180°, 270°), each step must be axis-aligned."""
        # Horizontal dark stripe to give the path a clear dark direction
        h, w = 64, 128
        img = np.full((h, w), 255, dtype=np.uint8)
        img[32, :] = 0  # horizontal dark line at row 32
        path = self.gen._trace_darkest_path(img, 32, 0, 4, 30, 1)
        assert len(path) > 1
        # Each step must move purely horizontally or purely vertically (step_size_px=1)
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            # One of dx, dy must be ~0 and the other ~step_size (1)
            assert (dx < 0.01 or dy < 0.01), f"Non-axis-aligned step: dx={dx}, dy={dy}"

    def test_out_of_bounds_seed_returns_empty(self):
        """Seed outside image bounds should return an empty path."""
        img = make_white_image(32, 32)
        path = self.gen._trace_darkest_path(img, 100, 100, 8, 50, 2)
        assert path == []

    def test_bright_seed_stops_immediately(self):
        """If the seed pixel is above the brightness ceiling, stop after the seed."""
        # All pixels set to 252 (above ceiling of 250)
        img = np.full((32, 32), 252, dtype=np.uint8)
        path = self.gen._trace_darkest_path(img, 16, 16, 8, 50, 2)
        # Should return only the seed (stopped immediately due to brightness)
        assert len(path) == 1
        assert path[0] == (16.0, 16.0)


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
        params = {"_source_image": img, "line_density": 30.0}
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
        }
        result = self.gen.generate(params, self.canvas)
        total_segments = sum(len(p) - 1 for p in result if len(p) > 1)
        # Allow the last path to push slightly over the limit
        assert total_segments <= line_max_limit + line_max_length - 1

    def test_x_y_offset_applied(self):
        """x_offset_mm / y_offset_mm must shift all output coordinates."""
        img = make_single_dark_block()
        params_base = {"_source_image": img.copy(), "line_density": 20.0}
        params_shifted = {
            "_source_image": img.copy(),
            "line_density": 20.0,
            "x_offset_mm": 10.0,
            "y_offset_mm": 5.0,
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
            "step_size_px": 2,
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
            "line_density": 20.0,
            "line_max_limit": 200,
            "line_min_length": 2,
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
        img = make_single_dark_block(h=64, w=64)
        for preset in self.gen.get_presets():
            params = dict(preset.params)
            params["_source_image"] = img.copy()
            # Keep limits small for speed
            params["line_max_limit"] = 200
            params["line_min_length"] = 2
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
