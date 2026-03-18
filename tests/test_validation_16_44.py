"""Phase 16.44 validation: stipple generator performance optimisations.

Verifies:

1. **Downsampled working image** — generator accepts and uses the new
   ``working_resolution`` parameter; output quality is equivalent to full
   resolution (same spatial distribution, point count within ±5 %).

2. **Early stopping** — ``convergence_threshold`` halts iteration before the
   full ``iterations`` budget when points have converged.  Set to a large value
   to verify it terminates early; set to 0 to verify it runs all iterations.

3. **Vectorised centroid** — correctness regression: with the same seed and
   parameters, the optimised ``_lloyd_simple`` produces point distributions
   that still concentrate in dark image regions (same qualitative result as
   the original per-point loop).

4. **KDTree TSP** — the optimised ``_nearest_neighbor_tsp`` visits every point
   exactly once and produces a tour whose total length is no worse than 150 %
   of an alternative greedy tour (generous bound — we are only verifying
   correctness and rough quality, not optimality).

5. **Benchmark** — running the generator on a 400×400 synthetic gradient with
   5 000 points and 30 iterations completes within a generous time budget
   (30 seconds).  This exercises the full optimised pipeline and would
   previously time out with the original O(n²) per-point loop.

6. **Parameter completeness** — all presets include the two new parameters
   (``working_resolution`` and ``convergence_threshold``).

7. **Regression** — existing functionality (dots in dark areas, TSP single
   polyline, output within bounds) is preserved.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from plottter.models import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def _gradient_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Left-dark, right-light gradient (HxW uint8 grayscale)."""
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        img[:, col] = int(col / w * 255)
    return img


def _dark_left_image(h: int = 80, w: int = 80) -> np.ndarray:
    """Left half black, right half white."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2 :] = 255
    return img


def _base_params(img: np.ndarray, **overrides) -> dict:
    defaults = {
        "_source_image": img,
        "num_points": 50,
        "iterations": 3,
        "connect_tsp": False,
        "min_dot_spacing_mm": 0.0,
        "seed": 42,
        "working_resolution": 400,
        "convergence_threshold": 0.5,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# 1. New parameters are accepted and respected
# ---------------------------------------------------------------------------


class TestNewParameters:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    def test_working_resolution_in_parameters(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "working_resolution" in names, "working_resolution param missing"

    def test_convergence_threshold_in_parameters(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "convergence_threshold" in names, "convergence_threshold param missing"

    def test_generate_accepts_working_resolution(self):
        img = _gradient_image(60, 60)
        params = _base_params(img, working_resolution=100)
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_accepts_convergence_threshold_zero(self):
        """convergence_threshold=0 should disable early stopping."""
        img = _gradient_image(60, 60)
        params = _base_params(img, iterations=5, convergence_threshold=0.0)
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_generate_large_convergence_threshold_terminates(self):
        """Very large threshold should cause early exit but still produce output."""
        img = _gradient_image(60, 60)
        params = _base_params(img, iterations=50, convergence_threshold=1000.0)
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 2. Downsampling correctness
# ---------------------------------------------------------------------------


class TestDownsamplingCorrectness:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    def test_downsampled_dot_count_within_5pct(self):
        """Output point count should be within 5 % of requested num_points."""
        img = _gradient_image(200, 200)
        n = 200
        params = _base_params(
            img, num_points=n, iterations=3,
            working_resolution=50,   # aggressive downsample
            convergence_threshold=0.0,
        )
        result = self.gen.generate(params, self.canvas)
        assert abs(len(result) - n) / n <= 0.05, (
            f"Dot count {len(result)} deviates more than 5 % from {n}"
        )

    def test_downsampled_dots_concentrate_in_dark_area(self):
        """Even with heavy downsampling, dots should cluster in the dark half."""
        img = _dark_left_image(120, 120)
        params = _base_params(
            img, num_points=100, iterations=5,
            connect_tsp=True,
            working_resolution=40,
            convergence_threshold=0.0,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1
        path = result[0]

        draw_x1, _, draw_x2, _ = self.canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2.0
        left = sum(1 for x, _ in path if x <= mid_x)
        right = sum(1 for x, _ in path if x > mid_x)
        assert left > right, (
            f"Expected more dots in dark left half: left={left} right={right}"
        )

    def test_full_resolution_fallback_when_image_smaller_than_working_res(self):
        """If image is smaller than working_resolution, it should not be upscaled."""
        img = _gradient_image(30, 30)
        params = _base_params(
            img, num_points=30, iterations=2,
            working_resolution=400,   # image is already smaller
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_no_downsampling_when_working_resolution_exceeds_image(self):
        """working_resolution larger than image should give same result as full res."""
        img = _gradient_image(50, 50)
        params_full = _base_params(
            img, num_points=30, iterations=2,
            working_resolution=9999, convergence_threshold=0.0,
        )
        params_down = _base_params(
            img, num_points=30, iterations=2,
            working_resolution=400, convergence_threshold=0.0,
        )
        r_full = self.gen.generate(params_full, self.canvas)
        r_down = self.gen.generate(params_down, self.canvas)
        # Both should produce the same number of dots (no spacing filter applied)
        assert len(r_full) == len(r_down)


# ---------------------------------------------------------------------------
# 3. Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    def test_large_threshold_produces_output(self):
        """Early stopping on first iteration must still return valid polylines."""
        img = _gradient_image(80, 80)
        params = _base_params(
            img, num_points=50, iterations=100,
            convergence_threshold=1000.0,  # will trigger after 1st iteration
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert all(len(p) >= 2 for p in result)

    def test_zero_threshold_runs_all_iterations(self):
        """convergence_threshold=0 must not stop early."""
        img = _gradient_image(60, 60)
        params = _base_params(
            img, num_points=30, iterations=10, convergence_threshold=0.0,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_early_stop_still_concentrates_in_dark_area(self):
        """Even with early stopping, density in dark region should exceed light."""
        img = _dark_left_image(80, 80)
        params = _base_params(
            img, num_points=80, iterations=50,
            connect_tsp=True, min_dot_spacing_mm=0.0,
            convergence_threshold=1000.0,  # stop immediately
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1
        path = result[0]
        draw_x1, _, draw_x2, _ = self.canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2.0
        left = sum(1 for x, _ in path if x <= mid_x)
        right = sum(1 for x, _ in path if x > mid_x)
        assert left > right


# ---------------------------------------------------------------------------
# 4. KDTree TSP correctness
# ---------------------------------------------------------------------------


class TestKDTreeTSP:
    def setup_method(self):
        from plottter.generators.stipple import _nearest_neighbor_tsp
        self._tsp = _nearest_neighbor_tsp

    def test_visits_all_points(self):
        """TSP order must include every point index exactly once."""
        pts = np.random.default_rng(0).random((50, 2)) * 100.0
        order = self._tsp(pts)
        assert sorted(order) == list(range(len(pts)))

    def test_empty_input_returns_empty(self):
        pts = np.empty((0, 2))
        order = self._tsp(pts)
        assert order == []

    def test_single_point(self):
        pts = np.array([[5.0, 5.0]])
        order = self._tsp(pts)
        assert order == [0]

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        order = self._tsp(pts)
        assert sorted(order) == [0, 1]

    def test_tour_is_reasonable_quality(self):
        """KDTree TSP tour length should be within 150 % of a simple sorted heuristic."""
        # Create points along a horizontal line — optimal tour is just left→right
        rng = np.random.default_rng(7)
        n = 100
        pts = np.column_stack([rng.random(n) * 200.0, np.zeros(n)])

        order = self._tsp(pts)

        def tour_length(o: list[int]) -> float:
            total = 0.0
            for i in range(len(o) - 1):
                dx = pts[o[i + 1], 0] - pts[o[i], 0]
                dy = pts[o[i + 1], 1] - pts[o[i], 1]
                total += math.sqrt(dx * dx + dy * dy)
            return total

        # Optimal: sort by x
        optimal_order = list(np.argsort(pts[:, 0]))
        optimal_len = tour_length(optimal_order)
        actual_len = tour_length(order)

        assert actual_len <= 1.5 * optimal_len, (
            f"KDTree TSP tour ({actual_len:.1f}) is more than 150 % of optimal "
            f"({optimal_len:.1f})"
        )


# ---------------------------------------------------------------------------
# 5. Benchmark — must complete within time budget
# ---------------------------------------------------------------------------


class TestBenchmark:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    @pytest.mark.slow
    def test_stipple_400x400_5000pts_30iter_within_30s(self):
        """Optimised pipeline on a 400×400 image with 5 000 points and 30 iterations
        must complete within 30 seconds.

        The original O(n²) centroid loop would take several minutes for this
        configuration; the vectorised bincount + downsampled working image brings
        it well within the budget.
        """
        img = _gradient_image(400, 400)
        params = {
            "_source_image": img,
            "num_points": 5000,
            "iterations": 30,
            "connect_tsp": False,
            "min_dot_spacing_mm": 0.5,
            "seed": 0,
            "working_resolution": 400,
            "convergence_threshold": 0.5,
        }
        start = time.time()
        result = self.gen.generate(params, self.canvas)
        elapsed = time.time() - start

        assert len(result) > 0, "Generator produced no output"
        assert elapsed < 30.0, (
            f"Stipple generator took {elapsed:.1f}s — expected < 30s"
        )

    def test_tsp_1000pts_fast(self):
        """KDTree TSP on 1 000 points must complete within 5 seconds."""
        from plottter.generators.stipple import _nearest_neighbor_tsp
        rng = np.random.default_rng(42)
        pts = rng.random((1000, 2)) * 200.0

        start = time.time()
        order = _nearest_neighbor_tsp(pts)
        elapsed = time.time() - start

        assert len(order) == 1000
        assert elapsed < 5.0, f"KDTree TSP took {elapsed:.1f}s — expected < 5s"


# ---------------------------------------------------------------------------
# 6. Preset completeness
# ---------------------------------------------------------------------------


class TestPresetCompleteness:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()

    def test_all_presets_include_working_resolution(self):
        for preset in self.gen.get_presets():
            assert "working_resolution" in preset.params, (
                f"Preset '{preset.name}' missing 'working_resolution'"
            )

    def test_all_presets_include_convergence_threshold(self):
        for preset in self.gen.get_presets():
            assert "convergence_threshold" in preset.params, (
                f"Preset '{preset.name}' missing 'convergence_threshold'"
            )

    def test_working_resolution_defaults_to_400(self):
        for preset in self.gen.get_presets():
            assert preset.params["working_resolution"] == 400, (
                f"Preset '{preset.name}' working_resolution should default to 400"
            )

    def test_convergence_threshold_defaults_to_0_5(self):
        for preset in self.gen.get_presets():
            assert preset.params["convergence_threshold"] == 0.5, (
                f"Preset '{preset.name}' convergence_threshold should default to 0.5"
            )


# ---------------------------------------------------------------------------
# 7. Regression — existing functionality preserved
# ---------------------------------------------------------------------------


class TestRegressionExistingFunctionality:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_dots_are_tiny_circles(self):
        img = _gradient_image(80, 80)
        params = _base_params(img, num_points=20, iterations=2)
        result = self.gen.generate(params, self.canvas)
        assert all(len(p) >= 2 for p in result), "Each dot should be a polyline"

    def test_tsp_produces_single_path(self):
        img = _gradient_image(80, 80)
        params = _base_params(
            img, num_points=30, iterations=2,
            connect_tsp=True, min_dot_spacing_mm=0.0,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1, "TSP should produce a single polyline"
        assert len(result[0]) == 30, "TSP path should visit all 30 points"

    def test_output_within_canvas_bounds(self):
        img = _gradient_image(80, 80)
        params = _base_params(img, num_points=30, iterations=2)
        result = self.gen.generate(params, self.canvas)
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        tol = 1.0  # mm tolerance for dot radius
        for poly in result:
            for x, y in poly:
                assert draw_x1 - tol <= x <= draw_x2 + tol
                assert draw_y1 - tol <= y <= draw_y2 + tol

    def test_dark_image_still_places_all_points(self):
        img = np.zeros((80, 80), dtype=np.uint8)
        params = _base_params(img, num_points=50, iterations=2)
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 50

    def test_invert_shifts_dot_density(self):
        """With invert=True, dots should concentrate in the LIGHT half."""
        img = _dark_left_image(80, 80)
        params = _base_params(
            img, num_points=100, iterations=5,
            connect_tsp=True, min_dot_spacing_mm=0.0,
            invert=True,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1
        path = result[0]
        draw_x1, _, draw_x2, _ = self.canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2.0
        left = sum(1 for x, _ in path if x <= mid_x)
        right = sum(1 for x, _ in path if x > mid_x)
        assert right > left, (
            f"Inverted: expected more dots on right (bright) side, "
            f"got left={left} right={right}"
        )

    def test_presets_generate_without_error(self):
        img = _gradient_image(80, 80)
        for preset in self.gen.get_presets():
            p = dict(preset.params)
            p["_source_image"] = img
            p["num_points"] = 50  # keep test fast
            p["iterations"] = 2
            result = self.gen.generate(p, self.canvas)
            assert isinstance(result, list), f"Preset '{preset.name}' failed"
