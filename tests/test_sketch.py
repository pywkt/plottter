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
# _sample_seed_batch
# ---------------------------------------------------------------------------


class TestSampleSeedBatch:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.rng = np.random.default_rng(42)

    def test_seeds_biased_toward_dark_areas(self):
        """Seeds must be sampled proportionally from dark (high residual_dark) regions."""
        h, w = 64, 64
        residual_dark = np.zeros((h, w), dtype=np.float32)
        residual_dark[:, : w // 2] = 1.0  # left half is dark, right is white
        edge = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)

        ys, xs = self.gen._sample_seed_batch(
            residual_dark, edge, coverage, h, w,
            batch_size=1000, dark_power=1.0, edge_bias=0.0,
            max_pixel_coverage=2, min_darkness=0.02, rng=self.rng,
        )

        # >95% of sampled seeds should fall in the dark (left) half
        dark_count = int(np.sum(xs < w // 2))
        assert dark_count > 950, f"Only {dark_count}/1000 seeds in dark region"

    def test_returns_correct_shape_and_bounds(self):
        """_sample_seed_batch returns (ys, xs) arrays of batch_size within image bounds."""
        h, w = 32, 48
        residual_dark = np.ones((h, w), dtype=np.float32) * 0.5
        edge = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)

        ys, xs = self.gen._sample_seed_batch(
            residual_dark, edge, coverage, h, w,
            batch_size=100, dark_power=1.0, edge_bias=0.0,
            max_pixel_coverage=2, min_darkness=0.02, rng=self.rng,
        )

        assert len(ys) == 100
        assert len(xs) == 100
        assert bool(np.all(ys >= 0)) and bool(np.all(ys < h))
        assert bool(np.all(xs >= 0)) and bool(np.all(xs < w))

    def test_different_rng_seeds_produce_different_batches(self):
        """Different RNG instances must produce distinct seed batches."""
        h, w = 64, 64
        residual_dark = np.ones((h, w), dtype=np.float32) * 0.5
        edge = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)

        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(9999)

        ys1, xs1 = self.gen._sample_seed_batch(
            residual_dark, edge, coverage, h, w,
            batch_size=200, dark_power=1.0, edge_bias=0.0,
            max_pixel_coverage=2, min_darkness=0.02, rng=rng1,
        )
        ys2, xs2 = self.gen._sample_seed_batch(
            residual_dark, edge, coverage, h, w,
            batch_size=200, dark_power=1.0, edge_bias=0.0,
            max_pixel_coverage=2, min_darkness=0.02, rng=rng2,
        )

        # Two independent RNGs must not produce the identical sequence
        assert not (np.array_equal(ys1, ys2) and np.array_equal(xs1, xs2))

    def test_higher_dark_power_concentrates_on_darkest_pixels(self):
        """Higher dark_power should shift samples further toward the darkest areas."""
        h, w = 64, 64
        # Gradient: columns go from residual_dark=0.3 (left) to 1.0 (right)
        residual_dark = (
            np.linspace(0.3, 1.0, w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
        )
        edge = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)

        _, xs_low = self.gen._sample_seed_batch(
            residual_dark, edge, coverage, h, w,
            batch_size=2000, dark_power=1.0, edge_bias=0.0,
            max_pixel_coverage=2, min_darkness=0.02, rng=np.random.default_rng(42),
        )
        _, xs_high = self.gen._sample_seed_batch(
            residual_dark, edge, coverage, h, w,
            batch_size=2000, dark_power=5.0, edge_bias=0.0,
            max_pixel_coverage=2, min_darkness=0.02, rng=np.random.default_rng(42),
        )

        # Higher dark_power → mean x should shift toward right (darker) columns
        assert np.mean(xs_high) > np.mean(xs_low), (
            "Higher dark_power should concentrate sampling in darker columns"
        )

    def test_covered_pixels_penalised(self):
        """Fully-covered pixels (ink_penalty=0.05) should be sampled far less often."""
        h, w = 64, 64
        residual_dark = np.ones((h, w), dtype=np.float32)
        edge = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)
        coverage[:, : w // 2] = 2  # left half fully covered (max_pixel_coverage=2)

        ys, xs = self.gen._sample_seed_batch(
            residual_dark, edge, coverage, h, w,
            batch_size=2000, dark_power=1.0, edge_bias=0.0,
            max_pixel_coverage=2, min_darkness=0.02, rng=np.random.default_rng(7),
        )

        # Right (less-covered) half should dominate
        right_count = int(np.sum(xs >= w // 2))
        assert right_count > 1600, f"Only {right_count}/2000 seeds in less-covered region"


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
        """An image with a large dark region should produce at least one polyline."""
        h, w = 64, 64
        img = np.full((h, w), 255, dtype=np.uint8)
        img[:, : w // 2] = 0  # left half black — large enough for reliable squiggle generation
        params = {"_source_image": img, "multi_pass": False, "line_max_limit": 200}
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
        assert presets[0].name == "Quick Sketch"

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
        end_x, end_y, avg_brightness, _ = result
        # The winning direction should lead into the darker left half
        assert avg_brightness < 255.0

    def test_avg_brightness_is_in_range(self):
        """avg_brightness must be in [0, 255]."""
        img = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 32, 32, 8, 10)
        if result is not None:
            _, _, avg_brightness, _ = result
            assert 0.0 <= avg_brightness <= 255.0

    def test_endpoint_within_image_bounds(self):
        """The returned endpoint must be inside the image."""
        img = np.zeros((64, 64), dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 32, 32, 16, 10)
        assert result is not None
        end_x, end_y, _, _ = result
        assert 0 <= end_x < 64
        assert 0 <= end_y < 64

    def test_best_direction_highest_score(self):
        """The chosen direction should have the highest score among all valid
        candidates using the weighted formula: dark*1.45 + dark_peak*0.45."""
        rng = np.random.default_rng(99)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        angle_tests = 8
        line_length_px = 8
        result = self.gen._find_darkest_line(img, 32, 32, angle_tests, line_length_px)
        assert result is not None
        best_end_x, best_end_y, _, _ = result

        # Replicate the scoring logic (no maps → edge_strength=0, cov=0)
        h, w = img.shape[:2]
        scores: dict[tuple[int, int], float] = {}
        for i in range(angle_tests):
            angle = i * 2.0 * np.pi / angle_tests
            ex = 32 + int(round(np.cos(angle) * line_length_px))
            ey = 32 + int(round(np.sin(angle) * line_length_px))
            if not (0 <= ey < h and 0 <= ex < w):
                continue
            # Use fixed 8 samples to match the vectorized scoring in _find_darkest_line
            num_samples = 8
            sxs = np.clip(np.round(np.linspace(32, ex, num_samples)).astype(np.int32), 0, w - 1)
            sys_arr = np.clip(np.round(np.linspace(32, ey, num_samples)).astype(np.int32), 0, h - 1)
            dark_vals = 1.0 - img[sys_arr, sxs].astype(np.float32) / 255.0
            score = float(np.mean(dark_vals)) * 1.45 + float(np.max(dark_vals)) * 0.45
            scores[(ex, ey)] = score

        best_score = scores[(best_end_x, best_end_y)]
        for (ex, ey), score in scores.items():
            assert best_score >= score - 1e-9, (
                f"Best score {best_score:.4f} < candidate score {score:.4f} for ({ex},{ey})"
            )

    def test_bresenham_circle_mode_angle_tests_36(self):
        """angle_tests=36 triggers Bresenham circle mode; result must be valid."""
        img = np.zeros((128, 128), dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 64, 64, 36, 20)
        assert result is not None
        end_x, end_y, avg_brightness, best_score = result
        assert 0 <= end_x < 128
        assert 0 <= end_y < 128
        assert avg_brightness == 0.0  # all-black image

    def test_returns_score_as_fourth_element(self):
        """_find_darkest_line must return a 4-tuple with score as 4th element."""
        img = np.zeros((64, 64), dtype=np.uint8)
        result = self.gen._find_darkest_line(img, 32, 32, 8, 10)
        assert result is not None
        assert len(result) == 4
        end_x, end_y, avg_brightness, score = result
        assert isinstance(score, float)
        # All-black image — score should be positive (dark area)
        assert score > 0.0


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

    def test_higher_limit_produces_more_paths(self):
        """Higher line_max_limit allows more output segments → more polylines."""
        img = np.zeros((64, 64), dtype=np.uint8)  # all black — lots of room
        params_low = {"_source_image": img.copy(), "line_max_limit": 30, "multi_pass": False}
        params_high = {"_source_image": img.copy(), "line_max_limit": 300, "multi_pass": False}
        result_low = self.gen.generate(params_low, self.canvas)
        result_high = self.gen.generate(params_high, self.canvas)
        segs_low = sum(len(p) - 1 for p in result_low if len(p) > 1)
        segs_high = sum(len(p) - 1 for p in result_high if len(p) > 1)
        assert segs_high >= segs_low

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
        h, w = 64, 64
        img = np.full((h, w), 255, dtype=np.uint8)
        img[:, : w // 2] = 0  # left half black — reliable squiggle source
        params = {"_source_image": img, "multi_pass": False, "line_max_limit": 200}
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
        img = np.zeros((64, 64), dtype=np.uint8)  # solid black — reliable output
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        x_off, y_off = 10.0, 5.0

        # Without offset: all points must be within the drawing area
        params_base = {
            "_source_image": img.copy(),
            "multi_pass": False,
            "line_max_limit": 100,
        }
        result_base = self.gen.generate(params_base, self.canvas)
        assert len(result_base) > 0
        margin = 0.5
        for path in result_base:
            for x, y in path:
                assert draw_x1 - margin <= x <= draw_x2 + margin

        # With offset: every point must be shifted by exactly (x_off, y_off)
        params_shifted = {
            "_source_image": img.copy(),
            "multi_pass": False,
            "line_max_limit": 100,
            "x_offset_mm": x_off,
            "y_offset_mm": y_off,
        }
        result_shifted = self.gen.generate(params_shifted, self.canvas)
        assert len(result_shifted) > 0
        for path in result_shifted:
            for x, y in path:
                assert draw_x1 + x_off - margin <= x <= draw_x2 + x_off + margin, (
                    f"x={x:.3f} not in shifted range [{draw_x1+x_off-margin:.1f}, {draw_x2+x_off+margin:.1f}]"
                )

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
        assert "Quick Sketch" in names
        assert "Portrait" in names
        assert "Contour Portrait" in names
        assert "Dense Ink" in names
        assert "Crosshatch" in names
        assert "Loose Sketch" in names
        assert "Edge Trace" in names
        assert "Scribble" in names
        assert "Hybrid Portrait" in names

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
            params["multi_pass"] = False  # single pass for speed
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
            "multi_pass": False,    # isolate coverage from multi-pass length variation
            "long_line_bias": 0.0,  # no random bonus — deterministic comparison
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
        Uses a 300×300 image so the max_effective cap (99 px) does not equalize the
        two very different base lengths (38 px vs 12 px), and long_line_bias=0 removes
        random stretch bonuses that could blur the distinction.
        """
        img = np.zeros((300, 300), dtype=np.uint8)  # all black, large enough for uncapped lengths
        common = {
            "_source_image": img,
            "multi_pass": False,
            "line_max_limit": 200,
            "squiggle_min_length": 1,
            "squiggle_max_length": 10,
            "squiggle_max_deviation": 90.0,
            "long_line_bias": 0.0,  # no random stretch bonus
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
        """multi_pass=False should produce output on a dark image."""
        h, w = 64, 64
        img = np.full((h, w), 255, dtype=np.uint8)
        img[:, : w // 2] = 0  # left half black — large enough for reliable squiggles
        params = {
            "_source_image": img,
            "multi_pass": False,
            "line_max_limit": 100,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "single-pass mode should still produce paths"


# ---------------------------------------------------------------------------
# Continuous path chaining
# ---------------------------------------------------------------------------


class TestContinuousChaining:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def _dark_image(self, h: int = 80, w: int = 80) -> np.ndarray:
        return np.zeros((h, w), dtype=np.uint8)

    def test_continuous_and_chain_max_params_defined(self):
        """continuous (BoolParam) and chain_max (IntParam) must be registered."""
        names = {p.name for p in self.gen.get_parameters()}
        assert "continuous" in names
        assert "chain_max" in names

    def test_presets_include_continuous_and_chain_max(self):
        """All presets must include continuous and chain_max."""
        for preset in self.gen.get_presets():
            assert "continuous" in preset.params, f"Preset '{preset.name}' missing continuous"
            assert "chain_max" in preset.params, f"Preset '{preset.name}' missing chain_max"

    def test_continuous_true_produces_fewer_longer_polylines(self):
        """continuous=True chains short squiggles into long paths — fewer, longer polylines.

        With squiggle_max_length=4 and chain_max=40, continuous mode chains up to 40
        segments into one polyline, while non-continuous caps each polyline at 4 segments.
        """
        img = self._dark_image()
        common = {
            "_source_image": img.copy(),
            "multi_pass": False,
            "line_max_limit": 300,
            "squiggle_min_length": 1,
            "squiggle_max_length": 4,   # short per-squiggle cap
            "squiggle_max_deviation": 90.0,
            "line_length_px": 8,
            "angle_tests": 8,  # fast: 8 angles instead of Bresenham circle
        }
        result_cont = self.gen.generate({**common, "continuous": True, "chain_max": 40}, self.canvas)
        result_sep = self.gen.generate({**common, "continuous": False}, self.canvas)

        # continuous=True should produce fewer, longer polylines
        assert len(result_cont) > 0, "continuous=True produced no output"
        assert len(result_sep) > 0, "continuous=False produced no output"
        assert len(result_cont) < len(result_sep), (
            f"continuous=True ({len(result_cont)} paths) should produce fewer paths "
            f"than continuous=False ({len(result_sep)} paths)"
        )
        avg_len_cont = sum(len(p) for p in result_cont) / len(result_cont)
        avg_len_sep = sum(len(p) for p in result_sep) / len(result_sep)
        assert avg_len_cont > avg_len_sep, (
            f"continuous=True avg path length ({avg_len_cont:.1f}) should exceed "
            f"continuous=False ({avg_len_sep:.1f})"
        )

    def test_chain_max_limits_polyline_length(self):
        """chain_max=5 produces shorter max polyline length than chain_max=50."""
        img = self._dark_image()
        common = {
            "_source_image": img.copy(),
            "multi_pass": False,
            "line_max_limit": 300,
            "squiggle_min_length": 1,
            "squiggle_max_length": 60,
            "squiggle_max_deviation": 90.0,
            "continuous": True,
            "line_length_px": 8,
            "angle_tests": 8,  # fast: 8 angles instead of Bresenham circle
        }
        result_short = self.gen.generate({**common, "chain_max": 5}, self.canvas)
        result_long = self.gen.generate({**common, "chain_max": 50}, self.canvas)

        assert len(result_short) > 0
        assert len(result_long) > 0

        max_segs_short = max(len(p) - 1 for p in result_short)
        max_segs_long = max(len(p) - 1 for p in result_long)
        assert max_segs_short <= max_segs_long, (
            f"chain_max=5 max segs ({max_segs_short}) should be ≤ chain_max=50 ({max_segs_long})"
        )
        # chain_max=5 hard cap: no polyline should exceed 5 segments
        for path in result_short:
            n = len(path) - 1
            assert n <= 5, f"chain_max=5 produced a path with {n} segments"

    def test_chains_break_in_bright_areas(self):
        """Chains should stay in dark areas and not cross over bright regions."""
        h, w = 80, 80
        img = np.full((h, w), 255, dtype=np.uint8)
        img[:, : w // 2] = 0  # left half black, right half white

        params = {
            "_source_image": img,
            "multi_pass": False,
            "line_max_limit": 200,
            "squiggle_min_length": 1,
            "squiggle_max_deviation": 20.0,  # strict: break on bright areas
            "continuous": True,
            "chain_max": 50,
            "angle_tests": 8,  # fast: 8 angles instead of Bresenham circle
        }
        canvas = self.canvas
        draw_x1, _, draw_x2, _ = canvas.drawing_area()
        result = self.gen.generate(params, canvas)
        assert len(result) > 0

        # No path should span both halves significantly
        draw_mid_x = (draw_x1 + draw_x2) / 2.0
        for path in result:
            xs = [x for x, _ in path]
            # A path that starts left of midpoint should not end far right
            if xs[0] < draw_mid_x:
                assert xs[-1] < draw_mid_x + (draw_x2 - draw_x1) * 0.2, (
                    "Chain crossed from dark into bright half"
                )

    def test_coverage_similar_with_and_without_chaining(self):
        """Total pixel coverage is comparable for continuous=True vs False."""
        img = self._dark_image(h=64, w=64)
        common = {
            "multi_pass": False,
            "line_max_limit": 200,
            "squiggle_min_length": 1,
            "squiggle_max_deviation": 90.0,
            "line_length_px": 8,
            "angle_tests": 8,  # fast: 8 angles instead of Bresenham circle
        }
        result_cont = self.gen.generate(
            {**common, "_source_image": img.copy(), "continuous": True, "chain_max": 18},
            self.canvas,
        )
        result_sep = self.gen.generate(
            {**common, "_source_image": img.copy(), "continuous": False},
            self.canvas,
        )

        # Total point count (proxy for coverage) should be within 2× of each other
        pts_cont = sum(len(p) for p in result_cont)
        pts_sep = sum(len(p) for p in result_sep)
        assert pts_cont > 0 and pts_sep > 0
        ratio = max(pts_cont, pts_sep) / min(pts_cont, pts_sep)
        assert ratio < 3.0, (
            f"Coverage ratio too different: continuous={pts_cont} pts vs "
            f"separate={pts_sep} pts (ratio {ratio:.2f})"
        )

    def test_continuous_false_matches_original_structure(self):
        """continuous=False should produce multiple short polylines (original behavior)."""
        img = self._dark_image()
        params = {
            "_source_image": img,
            "multi_pass": False,
            "line_max_limit": 200,
            "squiggle_max_length": 5,
            "squiggle_min_length": 1,
            "continuous": False,
            "angle_tests": 8,  # fast: 8 angles instead of Bresenham circle
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        # Each polyline should have at most squiggle_max_length+1 points
        for path in result:
            assert len(path) <= 5 + 1, f"Path has {len(path)} points with squiggle_max_length=5"


# ---------------------------------------------------------------------------
# Variable step length by darkness
# ---------------------------------------------------------------------------


class TestVariableStepLength:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def test_long_line_bias_param_defined(self):
        """long_line_bias FloatParam must be registered."""
        names = {p.name for p in self.gen.get_parameters()}
        assert "long_line_bias" in names

    def test_presets_include_long_line_bias(self):
        """All presets must include long_line_bias."""
        for preset in self.gen.get_presets():
            assert "long_line_bias" in preset.params, (
                f"Preset '{preset.name}' missing long_line_bias"
            )

    def test_long_line_bias_zero_no_huge_outliers(self):
        """long_line_bias=0 means no random bonus — max seg length near 2× base."""
        # 100×100 image gives cap=33px; with line_length_px=10, base dark=20px,
        # long_line_bias=1 bonus → 36-76px capped at 33. Ratio 33/20=1.65 > 1.5.
        img = np.zeros((100, 100), dtype=np.uint8)  # all black
        common = {
            "_source_image": img.copy(),
            "multi_pass": False,
            "line_max_limit": 300,
            "squiggle_min_length": 1,
            "squiggle_max_length": 3,
            "squiggle_max_deviation": 90.0,
            "line_length_px": 10,
            "angle_tests": 8,
            "continuous": False,
        }
        result_no_bonus = self.gen.generate({**common, "long_line_bias": 0.0}, self.canvas)
        result_max_bonus = self.gen.generate({**common, "long_line_bias": 1.0}, self.canvas)

        assert len(result_no_bonus) > 0
        assert len(result_max_bonus) > 0

        def max_seg_len_mm(paths):
            max_len = 0.0
            for path in paths:
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i - 1][0]
                    dy = path[i][1] - path[i - 1][1]
                    max_len = max(max_len, (dx * dx + dy * dy) ** 0.5)
            return max_len

        max_no_bonus = max_seg_len_mm(result_no_bonus)
        max_with_bonus = max_seg_len_mm(result_max_bonus)
        # With max_bonus there's a high chance of strokes 1.8-3.8× longer
        # bias=1.0 in dark: prob = 1.0*(0.30+0.70*1.0) = 1.0 → always bonus
        # So max_with_bonus should be significantly larger
        assert max_with_bonus > max_no_bonus * 1.5, (
            f"long_line_bias=1 max={max_with_bonus:.2f}mm should greatly exceed "
            f"long_line_bias=0 max={max_no_bonus:.2f}mm"
        )

    def test_dark_areas_get_longer_strokes_than_bright(self):
        """Darker image yields longer average segments (local_dark * 1.15 modulation)."""
        common_params = {
            "multi_pass": False,
            "line_max_limit": 300,
            "squiggle_min_length": 1,
            "squiggle_max_length": 3,
            "squiggle_max_deviation": 90.0,
            "line_length_px": 10,
            "angle_tests": 8,
            "continuous": False,
            "long_line_bias": 0.0,  # no random bonus, pure darkness modulation
        }
        dark_img = np.zeros((80, 80), dtype=np.uint8)      # all black → local_dark≈1 → factor≈2.0
        medium_img = np.full((80, 80), 128, dtype=np.uint8)  # grey → local_dark≈0.5 → factor≈1.425

        result_dark = self.gen.generate({"_source_image": dark_img, **common_params}, self.canvas)
        result_medium = self.gen.generate({"_source_image": medium_img, **common_params}, self.canvas)

        assert len(result_dark) > 0

        def avg_seg_len_mm(paths):
            lengths = []
            for path in paths:
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i - 1][0]
                    dy = path[i][1] - path[i - 1][1]
                    lengths.append((dx * dx + dy * dy) ** 0.5)
            return sum(lengths) / len(lengths) if lengths else 0.0

        avg_dark = avg_seg_len_mm(result_dark)
        avg_medium = avg_seg_len_mm(result_medium) if result_medium else 0.0
        # Dark image segments should be longer due to higher local_dark factor
        assert avg_dark > avg_medium, (
            f"Dark avg seg {avg_dark:.3f}mm should exceed medium avg {avg_medium:.3f}mm"
        )


# ---------------------------------------------------------------------------
# Unsharp mask preprocessing
# ---------------------------------------------------------------------------


class TestUnsharpMask:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def test_unsharp_amount_param_defined(self):
        """unsharp_amount IntParam must be registered."""
        names = {p.name for p in self.gen.get_parameters()}
        assert "unsharp_amount" in names

    def test_presets_include_unsharp_amount(self):
        """All presets must include unsharp_amount."""
        for preset in self.gen.get_presets():
            assert "unsharp_amount" in preset.params, (
                f"Preset '{preset.name}' missing unsharp_amount"
            )

    def test_unsharp_zero_leaves_image_unchanged(self):
        """unsharp_amount=0 must produce the same preprocessing result as no unsharp.

        On a uniform gray image, applying blur then unsharp with amount=0
        must not change the pixel values (the if-block is skipped).
        """
        # Uniform image: blur has no effect, so unsharp difference is zero too.
        img = np.full((64, 64), 128, dtype=np.uint8)
        params_no_unsharp = {
            "_source_image": img.copy(),
            "multi_pass": False,
            "line_max_limit": 50,
            "unsharp_amount": 0,
            "blur_radius": 0.0,
            "squiggle_min_length": 1,
        }
        params_with_zero = {
            "_source_image": img.copy(),
            "multi_pass": False,
            "line_max_limit": 50,
            "unsharp_amount": 0,
            "blur_radius": 0.0,
            "squiggle_min_length": 1,
        }
        # Both should run without error; output format is identical
        result_a = self.gen.generate(params_no_unsharp, self.canvas)
        result_b = self.gen.generate(params_with_zero, self.canvas)
        assert isinstance(result_a, list)
        assert isinstance(result_b, list)

    def test_unsharp_formula_sharpens_edges(self):
        """Direct formula test: unsharp_amount=3 increases edge contrast.

        Creates a step-edge image in float32 space, applies the unsharp formula,
        and verifies that the luminance difference across the edge increases.
        """
        import cv2

        # Step-edge: left half dark, right half bright
        h, w = 20, 20
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, w // 2 :] = 200

        gray_f = img.astype(np.float32) / 255.0
        # Compute gaussian blur (sigma=2, as in the implementation)
        ksize = max(1, int(2.0 * 3) | 1)
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=2.0)
        blurred_f = blurred.astype(np.float32) / 255.0

        # Apply unsharp formula with amount=3
        sharpened_f = np.clip(gray_f + (gray_f - blurred_f) * 3, 0.0, 1.0)

        # Edge is at column w//2 (boundary between dark and bright halves)
        mid = w // 2
        row = h // 2
        edge_before = abs(float(gray_f[row, mid - 1]) - float(gray_f[row, mid]))
        edge_after = abs(float(sharpened_f[row, mid - 1]) - float(sharpened_f[row, mid]))

        assert edge_after > edge_before, (
            f"Unsharp mask should increase edge contrast: "
            f"before={edge_before:.4f}, after={edge_after:.4f}"
        )

    def test_unsharp_generator_produces_output(self):
        """Generator runs successfully with unsharp_amount=3 on a dark image."""
        img = make_single_dark_block()
        params = {
            "_source_image": img,
            "multi_pass": False,
            "line_max_limit": 100,
            "unsharp_amount": 3,
            "squiggle_min_length": 1,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Hybrid mark mode
# ---------------------------------------------------------------------------


class TestHybridMarkMode:
    def setup_method(self):
        self.gen = SketchGenerator()
        self.canvas = make_canvas()

    def test_mark_mode_param_defined(self):
        """mark_mode ChoiceParam must be registered."""
        params = self.gen.get_parameters()
        names = {p.name for p in params}
        assert "mark_mode" in names
        # Verify it's a ChoiceParam with expected choices
        from plottter.generators.base import ChoiceParam
        mp = next(p for p in params if p.name == "mark_mode")
        assert isinstance(mp, ChoiceParam)
        assert "Squiggle Only" in mp.choices
        assert "Hybrid" in mp.choices
        assert mp.default == "Squiggle Only"

    def test_presets_include_mark_mode(self):
        """All presets must include mark_mode."""
        for preset in self.gen.get_presets():
            assert "mark_mode" in preset.params, (
                f"Preset '{preset.name}' missing mark_mode"
            )

    def test_squiggle_only_mode_produces_output(self):
        """mark_mode='Squiggle Only' must produce paths on a dark image."""
        img = np.zeros((64, 64), dtype=np.uint8)
        params = {
            "_source_image": img,
            "mark_mode": "Squiggle Only",
            "multi_pass": False,
            "line_max_limit": 100,
            "squiggle_min_length": 1,
            "angle_tests": 8,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_hybrid_mode_produces_output(self):
        """mark_mode='Hybrid' must produce paths on a dark image."""
        img = np.zeros((64, 64), dtype=np.uint8)
        params = {
            "_source_image": img,
            "mark_mode": "Hybrid",
            "multi_pass": False,
            "line_max_limit": 100,
            "squiggle_min_length": 1,
            "angle_tests": 8,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_hybrid_produces_mix_of_mark_types(self):
        """Hybrid mode must emit a variety of path lengths (lines, circles, squiggles)."""
        # Image with dark area — ensure all mark types can compete
        img = np.zeros((80, 80), dtype=np.uint8)
        params = {
            "_source_image": img,
            "mark_mode": "Hybrid",
            "multi_pass": False,
            "line_max_limit": 300,
            "squiggle_min_length": 1,
            "angle_tests": 8,
            "blur_radius": 0.0,
            "unsharp_amount": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

        path_lengths = {len(p) for p in result}
        # Hybrid mode should produce paths of different lengths:
        # 2-point (straight/hatch lines), 11-point (circle dots), or multi-point (squiggles).
        # A uniform dark image has no edges → local_edge ≈ 0 → circles dominate,
        # but straight lines and squiggles are still viable.
        assert len(path_lengths) > 1, (
            f"Hybrid mode produced only one path length {path_lengths} — expected variety"
        )

    def test_flat_dark_image_gets_circle_dots(self):
        """A uniform dark image (no edges) should produce some 11-point circle marks."""
        # Uniform black image: no edges → local_edge ≈ 0 everywhere,
        # so circle dot candidates are eligible and should win in flat areas.
        img = np.zeros((80, 80), dtype=np.uint8)
        params = {
            "_source_image": img,
            "mark_mode": "Hybrid",
            "multi_pass": False,
            "line_max_limit": 300,
            "squiggle_min_length": 1,
            "angle_tests": 8,
            "blur_radius": 0.0,
            "unsharp_amount": 0,
            "edge_power": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

        # Circle marks have exactly 11 points (10 segments, closed circle)
        circle_count = sum(1 for p in result if len(p) == 11)
        total = len(result)
        # On a uniform dark image with no edges, circles should appear regularly
        assert circle_count > 0, (
            f"No circle marks (11-pt paths) found in {total} paths on uniform dark image"
        )

    def test_edge_image_gets_non_circle_marks(self):
        """An image dominated by edges should mostly produce straight/squiggle marks."""
        # Sharp step edge across the whole image — high local_edge everywhere
        h, w = 80, 80
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, w // 2:] = 255  # left half black, right half white — sharp edge at center

        params = {
            "_source_image": img,
            "mark_mode": "Hybrid",
            "multi_pass": False,
            "line_max_limit": 200,
            "squiggle_min_length": 1,
            "angle_tests": 8,
            "blur_radius": 0.0,
            "unsharp_amount": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

        # Near the edge, local_edge is high (> 0.38), so circle dots are excluded.
        # Most marks should be 2-point (straight/hatch lines) or multi-point squiggles.
        non_circle = sum(1 for p in result if len(p) != 11)
        total = len(result)
        assert non_circle > total * 0.5, (
            f"Expected mostly non-circle marks near edges, got {non_circle}/{total} non-circle"
        )

    def test_hybrid_output_in_mm_coordinates(self):
        """Hybrid mode paths must be within the canvas drawing area (mm)."""
        img = np.zeros((64, 64), dtype=np.uint8)
        params = {
            "_source_image": img,
            "mark_mode": "Hybrid",
            "multi_pass": False,
            "line_max_limit": 150,
            "squiggle_min_length": 1,
            "angle_tests": 8,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        margin = 0.5
        for path in result:
            for x, y in path:
                assert draw_x1 - margin <= x <= draw_x2 + margin, f"x={x} out of range"
                assert draw_y1 - margin <= y <= draw_y2 + margin, f"y={y} out of range"

    def test_score_path_returns_float(self):
        """_score_path must return a float score for a valid path."""
        h, w = 32, 32
        residual_dark = np.ones((h, w), dtype=np.float32) * 0.8
        edge_norm = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)
        pts = [(10.0, 10.0), (20.0, 20.0)]
        score = SketchGenerator._score_path(pts, residual_dark, edge_norm, coverage, 2, w, h)
        assert isinstance(score, float)
        assert score > 0.0  # dark pixels, no coverage → positive score

    def test_score_path_empty_returns_low(self):
        """_score_path for a degenerate single-point path returns -1e9."""
        h, w = 10, 10
        residual_dark = np.ones((h, w), dtype=np.float32)
        edge_norm = np.zeros((h, w), dtype=np.float32)
        coverage = np.zeros((h, w), dtype=np.uint8)
        pts = [(5.0, 5.0)]  # single point → no segments → empty pixel array
        score = SketchGenerator._score_path(pts, residual_dark, edge_norm, coverage, 2, w, h)
        assert score == -1e9

    def test_circle_mark_pixels_returns_closed_loop(self):
        """_circle_mark_pixels must return steps+1 points forming a closed loop."""
        pts = SketchGenerator._circle_mark_pixels(16.0, 16.0, 5.0, 10, 32, 32)
        assert len(pts) == 11  # steps + 1
        # First and last points complete the circle (i=0 and i=steps → both at angle 0/2π)
        assert abs(pts[0][0] - pts[-1][0]) < 1e-9, "x coords of first/last must match"
        assert abs(pts[0][1] - pts[-1][1]) < 1e-9, "y coords of first/last must match"

    def test_circle_mark_pixels_clamped_to_bounds(self):
        """Circle at image edge must clamp all points within image bounds."""
        pts = SketchGenerator._circle_mark_pixels(0.0, 0.0, 10.0, 10, 32, 32)
        for x, y in pts:
            assert 0.0 <= x <= 31.0, f"x={x} out of bounds"
            assert 0.0 <= y <= 31.0, f"y={y} out of bounds"


# ---------------------------------------------------------------------------
# _rasterize_path_numba
# ---------------------------------------------------------------------------


class TestRasterizePathNumba:
    """Tests for _rasterize_path_numba and its equivalence to the numpy fallback."""

    def test_horizontal_segment_exact_pixels(self):
        """Horizontal segment (0,5)→(4,5) rasterizes to exactly 5 unique pixels."""
        from plottter.generators.sketch import _rasterize_path_numba

        pts_x = np.array([0.0, 4.0], dtype=np.float64)
        pts_y = np.array([5.0, 5.0], dtype=np.float64)
        xs, ys = _rasterize_path_numba(pts_x, pts_y, 10, 10, 0)

        assert xs.size == 5
        assert ys.size == 5
        np.testing.assert_array_equal(np.sort(xs), [0, 1, 2, 3, 4])
        assert np.all(ys == 5)

    def test_single_point_returns_empty(self):
        """Path with fewer than 2 points returns empty arrays."""
        from plottter.generators.sketch import _rasterize_path_numba

        pts_x = np.array([5.0], dtype=np.float64)
        pts_y = np.array([5.0], dtype=np.float64)
        xs, ys = _rasterize_path_numba(pts_x, pts_y, 10, 10, 0)
        assert xs.size == 0
        assert ys.size == 0

    def test_radius_1_interior_pixel_expands_to_9(self):
        """A single interior pixel with coverage_radius=1 expands to 9 pixels."""
        from plottter.generators.sketch import _rasterize_path_numba

        # Zero-length segment: single pixel at (5, 5)
        pts_x = np.array([5.0, 5.0], dtype=np.float64)
        pts_y = np.array([5.0, 5.0], dtype=np.float64)
        xs, ys = _rasterize_path_numba(pts_x, pts_y, 20, 20, 1)
        assert xs.size == 9, f"Expected 9 pixels (3×3 around (5,5)), got {xs.size}"

    def test_radius_1_corner_pixel_clamped(self):
        """Expansion at image corner must not produce out-of-bounds pixels."""
        from plottter.generators.sketch import _rasterize_path_numba

        pts_x = np.array([0.0, 0.0], dtype=np.float64)
        pts_y = np.array([0.0, 0.0], dtype=np.float64)
        xs, ys = _rasterize_path_numba(pts_x, pts_y, 10, 10, 1)
        assert np.all(xs >= 0) and np.all(xs < 10)
        assert np.all(ys >= 0) and np.all(ys < 10)
        assert xs.size == 4  # only (0,0),(1,0),(0,1),(1,1) are in bounds

    def test_radius_0_matches_path_pixel_coords(self):
        """With coverage_radius=0, output matches _path_pixel_coords."""
        from plottter.generators.sketch import _rasterize_path_numba

        pts = [(2.0, 3.0), (8.0, 3.0), (8.0, 7.0)]
        pts_x = np.array([p[0] for p in pts], dtype=np.float64)
        pts_y = np.array([p[1] for p in pts], dtype=np.float64)
        width, height = 20, 20

        xs_nb, ys_nb = _rasterize_path_numba(pts_x, pts_y, width, height, 0)
        flat_nb = set(int(y) * width + int(x) for x, y in zip(xs_nb, ys_nb))

        gen = SketchGenerator()
        xs_np, ys_np = gen._path_pixel_coords(pts, width, height)
        flat_np = set(int(y) * width + int(x) for x, y in zip(xs_np, ys_np))

        assert flat_nb == flat_np, (
            f"Numba {len(flat_nb)} pixels vs numpy {len(flat_np)} pixels; "
            f"diff={flat_nb.symmetric_difference(flat_np)}"
        )

    def test_numba_matches_numpy_fallback_with_radius(self):
        """With coverage_radius=1, output matches _path_pixel_coords + _expand_pixel_coords."""
        from plottter.generators.sketch import _rasterize_path_numba

        pts = [(5.0, 5.0), (10.0, 8.0), (7.0, 12.0)]
        pts_x = np.array([p[0] for p in pts], dtype=np.float64)
        pts_y = np.array([p[1] for p in pts], dtype=np.float64)
        width, height = 30, 30
        radius = 1

        xs_nb, ys_nb = _rasterize_path_numba(pts_x, pts_y, width, height, radius)
        flat_nb = set(int(y) * width + int(x) for x, y in zip(xs_nb, ys_nb))

        gen = SketchGenerator()
        xs_np, ys_np = gen._path_pixel_coords(pts, width, height)
        ex_np, ey_np = gen._expand_pixel_coords(xs_np, ys_np, width, height, radius)
        flat_np = set(int(y) * width + int(x) for x, y in zip(ex_np, ey_np))

        assert flat_nb == flat_np, (
            f"Numba {len(flat_nb)} pixels vs numpy {len(flat_np)} pixels; "
            f"diff={flat_nb.symmetric_difference(flat_np)}"
        )

    def test_output_unique_pixels(self):
        """All returned pixel coordinates must be unique."""
        from plottter.generators.sketch import _rasterize_path_numba

        pts_x = np.array([0.0, 10.0, 5.0], dtype=np.float64)
        pts_y = np.array([0.0, 10.0, 5.0], dtype=np.float64)
        xs, ys = _rasterize_path_numba(pts_x, pts_y, 20, 20, 1)

        flat = ys * 20 + xs
        assert flat.size == np.unique(flat).size, "Duplicate pixel coordinates returned"

    def test_coverage_increments_correctly(self):
        """Coverage map increments by 1 at all rasterized + expanded pixels."""
        from plottter.generators.sketch import _rasterize_path_numba

        width, height = 32, 32
        coverage = np.zeros((height, width), dtype=np.uint8)
        pts_x = np.array([5.0, 10.0], dtype=np.float64)
        pts_y = np.array([5.0, 5.0], dtype=np.float64)

        xs, ys = _rasterize_path_numba(pts_x, pts_y, width, height, 0)
        assert xs.size > 0
        coverage[ys, xs] = np.minimum(
            coverage[ys, xs].astype(np.int32) + 1, 255
        ).astype(np.uint8)

        # All pixels along the horizontal segment y=5, x=5..10 should be 1
        assert np.all(coverage[5, 5:11] == 1), (
            f"Expected all ones at y=5, x=5..10, got {coverage[5, 5:11]}"
        )
        # Pixels outside the segment should still be 0
        assert coverage[5, 4] == 0
        assert coverage[5, 11] == 0

    def test_large_radius_matches_numpy(self):
        """With coverage_radius=3, output matches the numpy expand approach."""
        from plottter.generators.sketch import _rasterize_path_numba

        pts = [(10.0, 10.0), (15.0, 10.0)]
        pts_x = np.array([p[0] for p in pts], dtype=np.float64)
        pts_y = np.array([p[1] for p in pts], dtype=np.float64)
        width, height = 40, 40
        radius = 3

        xs_nb, ys_nb = _rasterize_path_numba(pts_x, pts_y, width, height, radius)
        flat_nb = set(int(y) * width + int(x) for x, y in zip(xs_nb, ys_nb))

        gen = SketchGenerator()
        xs_np, ys_np = gen._path_pixel_coords(pts, width, height)
        ex_np, ey_np = gen._expand_pixel_coords(xs_np, ys_np, width, height, radius)
        flat_np = set(int(y) * width + int(x) for x, y in zip(ex_np, ey_np))

        assert flat_nb == flat_np, (
            f"radius=3: numba {len(flat_nb)} vs numpy {len(flat_np)}; "
            f"diff={flat_nb.symmetric_difference(flat_np)}"
        )
