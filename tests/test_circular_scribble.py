"""Tests for the CircularScribble generator (tasks 25.1, 25.2, and 25.3)."""

from __future__ import annotations

import math
import random as _random
import time

import numpy as np
import pytest

from plottter.generators import GENERATORS
from plottter.generators.circular_scribble import (
    CircularScribbleGenerator,
    _build_tracing_path,
    _hermite_smooth,
    _merge_circuits,
    _nearest_neighbor_circuit,
    _partition_by_grid,
    _synthesize_scribbles,
    _tone_aware_sample,
)
from plottter.models import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canvas(w_mm: float = 200.0, h_mm: float = 200.0) -> Canvas:
    return Canvas(width_mm=w_mm, height_mm=h_mm, margin_mm=10.0)


def _make_gray(h: int = 100, w: int = 100, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _make_gradient_gray(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a left-dark → right-bright gradient image."""
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        img[:, col] = int(col * 255 / (w - 1))
    return img


# ---------------------------------------------------------------------------
# Task 25.1 — Tone-aware sampling
# ---------------------------------------------------------------------------

class TestToneAwareSample:
    def test_dark_image_produces_more_points_than_bright(self):
        rng = _random.Random(0)
        dark = _make_gray(100, 100, 20)
        bright = _make_gray(100, 100, 235)
        pts_dark = _tone_aware_sample(dark, min_spacing_px=5.0, max_spacing_px=15.0, rng=rng)
        rng2 = _random.Random(0)
        pts_bright = _tone_aware_sample(bright, min_spacing_px=5.0, max_spacing_px=15.0, rng=rng2)
        assert len(pts_dark) > len(pts_bright)

    def test_no_two_points_closer_than_exclusion_radius(self):
        rng = _random.Random(42)
        gray = _make_gray(80, 80, 100)
        min_sp = 8.0
        pts = _tone_aware_sample(gray, min_spacing_px=min_sp, max_spacing_px=20.0, rng=rng)
        for i, (ax, ay) in enumerate(pts):
            for bx, by in pts[i + 1 :]:
                dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                assert dist >= min_sp - 0.01  # small tolerance for floating point


# ---------------------------------------------------------------------------
# Task 25.2 — Grid partitioning
# ---------------------------------------------------------------------------

class TestPartitionByGrid:
    def test_all_points_assigned(self):
        pts = [(5.0, 5.0), (50.0, 50.0), (95.0, 95.0)]
        cells = _partition_by_grid(pts, img_w=100, img_h=100, grid_size=2)
        all_pts = [p for row in cells for col in row for p in col]
        assert sorted(all_pts) == sorted(pts)

    def test_grid_size_correct(self):
        cells = _partition_by_grid([], img_w=100, img_h=100, grid_size=5)
        assert len(cells) == 5
        assert all(len(row) == 5 for row in cells)

    def test_boundary_points_dont_go_out_of_bounds(self):
        # Points at the exact edges should land in the last cell, not crash
        pts = [(0.0, 0.0), (99.9, 99.9), (50.0, 50.0)]
        cells = _partition_by_grid(pts, img_w=100, img_h=100, grid_size=3)
        all_pts = [p for row in cells for col in row for p in col]
        assert len(all_pts) == 3


# ---------------------------------------------------------------------------
# Task 25.2 — Nearest-neighbor circuit
# ---------------------------------------------------------------------------

class TestNearestNeighborCircuit:
    def test_returns_all_points(self):
        pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)]
        circuit = _nearest_neighbor_circuit(pts)
        assert len(circuit) == len(pts)
        assert set(circuit) == set(pts)

    def test_single_point(self):
        assert _nearest_neighbor_circuit([(3.0, 4.0)]) == [(3.0, 4.0)]

    def test_two_points(self):
        pts = [(0.0, 0.0), (1.0, 1.0)]
        result = _nearest_neighbor_circuit(pts)
        assert len(result) == 2

    def test_result_is_permutation_of_input(self):
        rng = _random.Random(7)
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(50)]
        circuit = _nearest_neighbor_circuit(pts)
        assert len(circuit) == len(pts)
        assert sorted(circuit) == sorted(pts)

    def test_large_set_uses_kdtree(self):
        """Circuit construction should complete quickly for 200+ points."""
        rng = _random.Random(1)
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(200)]
        t0 = time.time()
        circuit = _nearest_neighbor_circuit(pts)
        elapsed = time.time() - t0
        assert len(circuit) == len(pts)
        assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Task 25.2 — Circuit merging
# ---------------------------------------------------------------------------

class TestMergeCircuits:
    def test_merge_empty_a_returns_b(self):
        b = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        assert _merge_circuits([], b) == b

    def test_merge_empty_b_returns_a(self):
        a = [(0.0, 0.0), (1.0, 0.0)]
        assert _merge_circuits(a, []) == a

    def test_merged_length_is_sum_of_inputs(self):
        a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        b = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]
        merged = _merge_circuits(a, b)
        assert len(merged) == len(a) + len(b)

    def test_merged_contains_all_points(self):
        a = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        b = [(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 12.0)]
        merged = _merge_circuits(a, b)
        for p in a + b:
            assert p in merged

    def test_merge_single_point_a(self):
        a = [(5.0, 5.0)]
        b = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        merged = _merge_circuits(a, b)
        assert len(merged) == 5
        assert (5.0, 5.0) in merged

    def test_merge_single_point_b(self):
        a = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        b = [(5.0, 5.0)]
        merged = _merge_circuits(a, b)
        assert len(merged) == 5
        assert (5.0, 5.0) in merged

    def test_merge_is_better_than_naive_for_nearby_circuits(self):
        """Merged path should avoid very long cross-edges between nearby clusters."""
        # Two clusters, 10 units apart
        a = [(float(i), 0.0) for i in range(5)]
        b = [(float(i), 2.0) for i in range(5)]
        merged = _merge_circuits(a, b)
        # Maximum jump should be ≤ a threshold (not jumping 50 units across)
        max_jump_sq = max(
            (merged[i + 1][0] - merged[i][0]) ** 2 + (merged[i + 1][1] - merged[i][1]) ** 2
            for i in range(len(merged) - 1)
        )
        assert max_jump_sq < 30 ** 2  # no jump > 30 units


# ---------------------------------------------------------------------------
# Task 25.2 — Hermite smoothing
# ---------------------------------------------------------------------------

class TestHermiteSmooth:
    def test_output_has_more_points_than_input(self):
        path = [(0.0, 0.0), (10.0, 5.0), (20.0, 0.0), (30.0, 5.0)]
        smoothed = _hermite_smooth(path, steps=10)
        assert len(smoothed) > len(path)

    def test_output_length_approximately_steps_times_input(self):
        path = [(float(i), 0.0) for i in range(5)]
        steps = 10
        smoothed = _hermite_smooth(path, steps=steps)
        # Expected: (n-1)*steps + 1 = 4*10+1 = 41 points
        assert len(smoothed) == (len(path) - 1) * steps + 1

    def test_single_point_returns_single_point(self):
        assert _hermite_smooth([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_two_points_returns_two_or_more(self):
        result = _hermite_smooth([(0.0, 0.0), (1.0, 1.0)], steps=5)
        assert len(result) >= 2

    def test_endpoints_are_preserved(self):
        path = [(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)]
        smoothed = _hermite_smooth(path, steps=10)
        # First point should be at or very near the start
        assert abs(smoothed[0][0] - path[0][0]) < 1e-9
        assert abs(smoothed[0][1] - path[0][1]) < 1e-9
        # Last point should be the final input point
        assert abs(smoothed[-1][0] - path[-1][0]) < 1e-9
        assert abs(smoothed[-1][1] - path[-1][1]) < 1e-9

    def test_smooth_curve_stays_near_original_points(self):
        """Hermite smoothing should not deviate wildly from the input polygon."""
        path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        smoothed = _hermite_smooth(path, steps=20)
        # All smoothed points should be within bounding box + small margin
        margin = 5.0
        for x, y in smoothed:
            assert -margin <= x <= 10.0 + margin
            assert -margin <= y <= 10.0 + margin


# ---------------------------------------------------------------------------
# Task 25.2 — Build tracing path
# ---------------------------------------------------------------------------

class TestBuildTracingPath:
    def test_empty_input_returns_empty(self):
        result = _build_tracing_path([], 100, 100, grid_size=5)
        assert result == []

    def test_single_point_returns_single_point(self):
        result = _build_tracing_path([(50.0, 50.0)], 100, 100, grid_size=5)
        assert result == [(50.0, 50.0)]

    def test_result_contains_all_sample_points(self):
        """Every sampled point should appear in the tracing path."""
        pts = [
            (10.0, 10.0), (50.0, 10.0), (90.0, 10.0),
            (10.0, 50.0), (50.0, 50.0), (90.0, 50.0),
            (10.0, 90.0), (50.0, 90.0), (90.0, 90.0),
        ]
        path = _build_tracing_path(pts, 100, 100, grid_size=3)
        assert len(path) == len(pts)
        for p in pts:
            assert p in path

    def test_path_is_connected(self):
        """No jump in the path should exceed 50% of image size."""
        rng = _random.Random(42)
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(80)]
        path = _build_tracing_path(pts, 100, 100, grid_size=5)
        assert len(path) == len(pts)
        max_jump_sq = max(
            (path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2
            for i in range(len(path) - 1)
        )
        # Max acceptable jump: 50 px (50% of image width)
        assert max_jump_sq < 50.0 ** 2

    def test_performance_for_typical_input(self):
        """Path construction should complete in under 10 seconds for typical density."""
        rng = _random.Random(0)
        # ~500 points — typical for default settings on a small image
        pts = [(rng.uniform(0, 200), rng.uniform(0, 200)) for _ in range(500)]
        t0 = time.time()
        path = _build_tracing_path(pts, 200, 200, grid_size=10)
        elapsed = time.time() - t0
        assert len(path) == len(pts)
        assert elapsed < 10.0

    def test_cancellation_returns_empty(self):
        rng = _random.Random(0)
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(100)]
        result = _build_tracing_path(
            pts, 100, 100, grid_size=5,
            cancelled_callback=lambda: True,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------

class TestCircularScribbleGenerator:
    def test_generator_is_registered(self):
        assert "Circular Scribble" in GENERATORS

    def test_generate_returns_single_polyline(self):
        """generate() should return a list with exactly one polyline."""
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gradient_gray(80, 80)
        params = {
            "_source_image": np.stack([gray, gray, gray], axis=2),
            "min_sample_spacing_mm": 5.0,
            "max_sample_spacing_mm": 15.0,
            "path_grid_size": 5,
            "seed": 0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        result = gen.generate(params, canvas)
        assert isinstance(result, list)
        assert len(result) == 1  # single polyline
        polyline = result[0]
        assert len(polyline) > 0

    def test_generate_returns_empty_for_no_source(self):
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        result = gen.generate({"seed": 0}, canvas)
        assert result == []

    def test_generate_with_grayscale_source(self):
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gray(60, 60, 80)
        params = {
            "_source_image": gray,
            "min_sample_spacing_mm": 5.0,
            "max_sample_spacing_mm": 15.0,
            "path_grid_size": 4,
            "seed": 1,
        }
        result = gen.generate(params, canvas)
        assert len(result) == 1

    def test_x_y_offset_shifts_output(self):
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gradient_gray(60, 60)
        source = np.stack([gray, gray, gray], axis=2)
        base_params = {
            "_source_image": source,
            "min_sample_spacing_mm": 8.0,
            "max_sample_spacing_mm": 20.0,
            "path_grid_size": 3,
            "seed": 5,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        result_base = gen.generate(base_params, canvas)

        shifted_params = dict(base_params)
        shifted_params["x_offset_mm"] = 10.0
        shifted_params["y_offset_mm"] = 5.0
        result_shifted = gen.generate(shifted_params, canvas)

        assert result_base and result_shifted
        # Every point in the shifted result should be offset from the base
        for (bx, by), (sx, sy) in zip(result_base[0], result_shifted[0]):
            assert abs(sx - bx - 10.0) < 1e-9
            assert abs(sy - by - 5.0) < 1e-9

    def test_same_seed_produces_identical_output(self):
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gradient_gray(60, 60)
        source = np.stack([gray, gray, gray], axis=2)
        params = {
            "_source_image": source,
            "min_sample_spacing_mm": 6.0,
            "max_sample_spacing_mm": 18.0,
            "path_grid_size": 5,
            "seed": 99,
        }
        r1 = gen.generate(params, canvas)
        r2 = gen.generate(params, canvas)
        assert r1 == r2

    def test_default_preset_has_path_grid_size(self):
        gen = CircularScribbleGenerator()
        presets = gen.get_presets()
        assert presets
        default = next(p for p in presets if p.name == "Default")
        assert "path_grid_size" in default.params
        assert default.params["path_grid_size"] == 10

    def test_path_grid_size_parameter_exists(self):
        gen = CircularScribbleGenerator()
        params = gen.get_parameters()
        names = [p.name for p in params]
        assert "path_grid_size" in names


# ---------------------------------------------------------------------------
# Task 25.3 — Circular scribble synthesis
# ---------------------------------------------------------------------------

class TestSynthesizeScribbles:
    """Unit tests for _synthesize_scribbles()."""

    def _straight_path(self, n: int = 50) -> list[tuple[float, float]]:
        """Horizontal straight line from (0, 50) to (100, 50) with n points."""
        return [(float(i) * 100.0 / (n - 1), 50.0) for i in range(n)]

    def test_output_is_nonempty_for_valid_input(self):
        gray = _make_gray(100, 100, 128)
        path = self._straight_path(20)
        pts = _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=2.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        assert len(pts) > 0

    def test_empty_path_returns_empty(self):
        gray = _make_gray(100, 100, 128)
        pts = _synthesize_scribbles(
            [], gray,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=2.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        assert pts == []

    def test_single_point_path_returns_empty(self):
        gray = _make_gray(100, 100, 128)
        pts = _synthesize_scribbles(
            [(50.0, 50.0)], gray,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=2.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        assert pts == []

    def test_dark_image_produces_more_points_than_bright(self):
        """Dark areas have smaller dt → more steps → more scribble points."""
        path = self._straight_path(30)
        gray_dark = _make_gray(100, 100, 20)
        gray_bright = _make_gray(100, 100, 235)
        pts_dark = _synthesize_scribbles(
            path, gray_dark,
            min_radius_px=2.0, max_radius_px=10.0,
            min_speed_px=1.0, max_speed_px=8.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        pts_bright = _synthesize_scribbles(
            path, gray_bright,
            min_radius_px=2.0, max_radius_px=10.0,
            min_speed_px=1.0, max_speed_px=8.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        # Darker → smaller step size → more points
        assert len(pts_dark) > len(pts_bright)

    def test_dark_image_has_smaller_radius_deviation(self):
        """Dark areas produce smaller circles — points should stay closer to the path."""
        path = self._straight_path(30)
        gray_dark = _make_gray(100, 100, 10)
        gray_bright = _make_gray(100, 100, 245)
        pts_dark = _synthesize_scribbles(
            path, gray_dark,
            min_radius_px=2.0, max_radius_px=15.0,
            min_speed_px=2.0, max_speed_px=8.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        pts_bright = _synthesize_scribbles(
            path, gray_bright,
            min_radius_px=2.0, max_radius_px=15.0,
            min_speed_px=2.0, max_speed_px=8.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        # Average distance from the path y=50 should be smaller for dark
        def _avg_dev(pts: list) -> float:
            if not pts:
                return 0.0
            return sum(abs(y - 50.0) for _, y in pts) / len(pts)

        avg_dev_dark = _avg_dev(pts_dark)
        avg_dev_bright = _avg_dev(pts_bright)
        assert avg_dev_dark < avg_dev_bright

    def test_cancellation_stops_synthesis(self):
        gray = _make_gray(100, 100, 100)
        path = self._straight_path(200)
        pts = _synthesize_scribbles(
            path, gray,
            min_radius_px=2.0, max_radius_px=8.0,
            min_speed_px=1.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            cancelled_callback=lambda: True,
        )
        # Should return early (may be empty or partial)
        assert isinstance(pts, list)

    def test_same_seed_produces_identical_output(self):
        gray = _make_gradient_gray(60, 60)
        path = self._straight_path(30)
        pts1 = _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=10.0,
            min_speed_px=2.0, max_speed_px=7.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=77,
        )
        pts2 = _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=10.0,
            min_speed_px=2.0, max_speed_px=7.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=77,
        )
        assert pts1 == pts2

    def test_different_seeds_produce_different_output(self):
        gray = _make_gradient_gray(60, 60)
        path = self._straight_path(30)
        pts1 = _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=10.0,
            min_speed_px=2.0, max_speed_px=7.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=1,
        )
        pts2 = _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=10.0,
            min_speed_px=2.0, max_speed_px=7.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=2,
        )
        # Different seeds → different noise maps → different tilt angles
        assert pts1 != pts2

    def test_larger_radius_produces_greater_deviation(self):
        """Larger max_radius_px should produce points further from the path."""
        path = self._straight_path(30)
        gray_mid = _make_gray(100, 100, 128)

        pts_small = _synthesize_scribbles(
            path, gray_mid,
            min_radius_px=1.0, max_radius_px=3.0,
            min_speed_px=2.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )
        pts_large = _synthesize_scribbles(
            path, gray_mid,
            min_radius_px=5.0, max_radius_px=15.0,
            min_speed_px=2.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
        )

        def _max_dev(pts: list) -> float:
            if not pts:
                return 0.0
            return max(abs(y - 50.0) for _, y in pts)

        assert _max_dev(pts_large) > _max_dev(pts_small)

    def test_progress_callback_is_called(self):
        gray = _make_gradient_gray(100, 100)
        path = self._straight_path(100)
        calls: list[int] = []
        _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=2.0, max_speed_px=6.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            progress_callback=calls.append,
        )
        assert len(calls) > 0


class TestCircularScribbleGeneratorTask253:
    """Integration tests for the full generator with scribble synthesis."""

    def test_generate_has_scribble_params(self):
        gen = CircularScribbleGenerator()
        names = [p.name for p in gen.get_parameters()]
        assert "min_speed" in names
        assert "max_speed" in names
        assert "angle_step_deg" in names
        assert "tone_gamma" in names

    def test_generate_returns_nonempty_polylines(self):
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gradient_gray(80, 80)
        params = {
            "_source_image": np.stack([gray, gray, gray], axis=2),
            "min_sample_spacing_mm": 6.0,
            "max_sample_spacing_mm": 18.0,
            "path_grid_size": 5,
            "seed": 0,
            "min_speed": 0.5,
            "max_speed": 8.0,
            "angle_step_deg": 20.0,
            "tone_gamma": 1.5,
        }
        result = gen.generate(params, canvas)
        assert len(result) >= 1
        total_pts = sum(len(pl) for pl in result)
        assert total_pts > 0

    def test_generate_output_is_continuous_path(self):
        """Each polyline must have at least 2 points; no polyline should be empty."""
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gradient_gray(60, 60)
        params = {
            "_source_image": np.stack([gray, gray, gray], axis=2),
            "min_sample_spacing_mm": 8.0,
            "max_sample_spacing_mm": 20.0,
            "path_grid_size": 4,
            "seed": 3,
            "min_speed": 0.5,
            "max_speed": 8.0,
            "angle_step_deg": 20.0,
            "tone_gamma": 1.5,
        }
        result = gen.generate(params, canvas)
        for polyline in result:
            assert len(polyline) >= 2

    def test_all_presets_include_new_params(self):
        gen = CircularScribbleGenerator()
        for preset in gen.get_presets():
            assert "min_speed" in preset.params, f"Preset '{preset.name}' missing min_speed"
            assert "max_speed" in preset.params, f"Preset '{preset.name}' missing max_speed"
            assert "angle_step_deg" in preset.params, f"Preset '{preset.name}' missing angle_step_deg"
            assert "tone_gamma" in preset.params, f"Preset '{preset.name}' missing tone_gamma"

    def test_invert_image_affects_output(self):
        """Inverting the image should change the scribble distribution."""
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gradient_gray(60, 60)
        base_params = {
            "_source_image": np.stack([gray, gray, gray], axis=2),
            "min_sample_spacing_mm": 8.0,
            "max_sample_spacing_mm": 20.0,
            "path_grid_size": 3,
            "seed": 5,
            "invert": False,
            "min_speed": 0.5,
            "max_speed": 8.0,
            "angle_step_deg": 20.0,
            "tone_gamma": 1.5,
        }
        result_normal = gen.generate(base_params, canvas)
        inverted_params = {**base_params, "invert": True}
        result_inverted = gen.generate(inverted_params, canvas)
        # Total points should differ when image is inverted
        pts_normal = sum(len(pl) for pl in result_normal)
        pts_inverted = sum(len(pl) for pl in result_inverted)
        assert pts_normal != pts_inverted


# ---------------------------------------------------------------------------
# Task 25.5 — Orientation variation and presets
# ---------------------------------------------------------------------------

class TestOrientationVariation:
    """Tests for orientation_strength and skip_background (Task 25.5)."""

    def _straight_path(self, n: int = 50) -> list[tuple[float, float]]:
        return [(float(i) * 100.0 / (n - 1), 50.0) for i in range(n)]

    def test_orientation_zero_produces_circles(self):
        """With orientation_strength=0, scribbles should be perfect circles (b == a).

        Verifying indirectly: no tilt means max |y-dev| ≈ radius for all points.
        """
        path = self._straight_path(30)
        gray = _make_gray(100, 100, 128)
        pts = _synthesize_scribbles(
            path, gray,
            min_radius_px=5.0, max_radius_px=5.0,  # fixed radius for predictability
            min_speed_px=3.0, max_speed_px=3.0,
            angle_step_deg=20.0, tone_gamma=1.0,
            seed=0, orientation_strength=0.0,
        )
        assert len(pts) > 0
        # All y-deviations from path y=50 should be ≤ radius (5.0 px)
        # With a perfect circle and radius=5, max deviation is exactly 5.0
        y_devs = [abs(y - 50.0) for _, y in pts]
        assert max(y_devs) <= 5.0 + 1e-6  # tolerance for floating-point

    def test_orientation_zero_vs_one_differ(self):
        """orientation_strength=0 and 1 should produce different outputs."""
        path = self._straight_path(40)
        gray = _make_gradient_gray(100, 100)
        pts_no_tilt = _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=10.0,
            min_speed_px=2.0, max_speed_px=7.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=42, orientation_strength=0.0,
        )
        pts_full_tilt = _synthesize_scribbles(
            path, gray,
            min_radius_px=3.0, max_radius_px=10.0,
            min_speed_px=2.0, max_speed_px=7.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=42, orientation_strength=1.0,
        )
        # Different orientation_strength → different point coordinates
        assert pts_no_tilt != pts_full_tilt

    def test_orientation_half_strength_produces_intermediate_result(self):
        """orientation_strength=0.5 should produce different output from both 0 and 1."""
        path = self._straight_path(40)
        gray = _make_gradient_gray(100, 100)
        common = dict(
            min_radius_px=3.0, max_radius_px=10.0,
            min_speed_px=2.0, max_speed_px=7.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=42,
        )
        pts_0 = _synthesize_scribbles(path, gray, orientation_strength=0.0, **common)
        pts_half = _synthesize_scribbles(path, gray, orientation_strength=0.5, **common)
        pts_1 = _synthesize_scribbles(path, gray, orientation_strength=1.0, **common)
        assert pts_half != pts_0
        assert pts_half != pts_1

    def test_orientation_strength_zero_makes_b_equal_a(self):
        """With orientation_strength=0, b-axis == a-axis (circle, not ellipse).

        For a path at y=50 with radius=8.0, all scribble y-deviations should
        be ≤ 8.0.  With orientation_strength=1 the ellipse is compressed and
        rotated, which can mix axes, but with 0 we get pure sin/cos at radius.
        """
        path = self._straight_path(20)
        gray = _make_gray(100, 100, 128)
        pts = _synthesize_scribbles(
            path, gray,
            min_radius_px=8.0, max_radius_px=8.0,
            min_speed_px=4.0, max_speed_px=4.0,
            angle_step_deg=20.0, tone_gamma=1.0,
            seed=7, orientation_strength=0.0,
        )
        # Max deviation = radius (8.0 px). With orientation=0 there's no tilt,
        # so b = a = 8.0, and no rotation.
        y_devs = [abs(y - 50.0) for _, y in pts]
        assert max(y_devs) <= 8.0 + 1e-6


class TestSkipBackground:
    """Tests for skip_background parameter (Task 25.5)."""

    def _straight_path(self, n: int = 50) -> list[tuple[float, float]]:
        return [(float(i) * 100.0 / (n - 1), 50.0) for i in range(n)]

    def test_skip_background_reduces_points_on_white_image(self):
        """On a near-white image, skip_background should produce fewer points."""
        path = self._straight_path(60)
        gray_white = _make_gray(100, 100, 254)  # near-white (> 0.98 threshold)
        pts_normal = _synthesize_scribbles(
            path, gray_white,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=0.3, max_speed_px=0.5,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=0, skip_background=False,
        )
        pts_skip = _synthesize_scribbles(
            path, gray_white,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=0.3, max_speed_px=0.5,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=0, skip_background=True,
        )
        # Skipping background advances much faster → fewer total points
        assert len(pts_skip) < len(pts_normal)

    def test_skip_background_no_effect_on_dark_image(self):
        """On a fully dark image, skip_background has no effect (gv never > 0.98)."""
        path = self._straight_path(40)
        gray_dark = _make_gray(100, 100, 10)
        pts_normal = _synthesize_scribbles(
            path, gray_dark,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=1.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=0, skip_background=False,
        )
        pts_skip = _synthesize_scribbles(
            path, gray_dark,
            min_radius_px=3.0, max_radius_px=8.0,
            min_speed_px=1.0, max_speed_px=5.0,
            angle_step_deg=20.0, tone_gamma=1.5,
            seed=0, skip_background=True,
        )
        # Dark image: gv ≈ 0.04, nowhere near 0.98 threshold → identical output
        assert pts_normal == pts_skip


class TestTask255Presets:
    """Tests for presets and new parameters from Task 25.5."""

    def test_all_presets_include_orientation_and_skip_params(self):
        gen = CircularScribbleGenerator()
        for preset in gen.get_presets():
            assert "orientation_strength" in preset.params, (
                f"Preset '{preset.name}' missing orientation_strength"
            )
            assert "skip_background" in preset.params, (
                f"Preset '{preset.name}' missing skip_background"
            )

    def test_portrait_preset_exists(self):
        gen = CircularScribbleGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Portrait" in names

    def test_detailed_preset_exists(self):
        gen = CircularScribbleGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Detailed" in names

    def test_loose_sketch_preset_exists(self):
        gen = CircularScribbleGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Loose Sketch" in names

    def test_shaded_preset_exists(self):
        gen = CircularScribbleGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Shaded" in names

    def test_loose_sketch_has_zero_orientation(self):
        """Loose Sketch should have orientation_strength=0 for pure circles."""
        gen = CircularScribbleGenerator()
        loose = next(p for p in gen.get_presets() if p.name == "Loose Sketch")
        assert loose.params["orientation_strength"] == 0.0

    def test_shaded_has_high_orientation(self):
        """Shaded preset should emphasise the 3D shading effect."""
        gen = CircularScribbleGenerator()
        shaded = next(p for p in gen.get_presets() if p.name == "Shaded")
        assert shaded.params["orientation_strength"] >= 0.5

    def test_orientation_strength_parameter_exists(self):
        gen = CircularScribbleGenerator()
        names = [p.name for p in gen.get_parameters()]
        assert "orientation_strength" in names

    def test_skip_background_parameter_exists(self):
        gen = CircularScribbleGenerator()
        names = [p.name for p in gen.get_parameters()]
        assert "skip_background" in names

    def test_generate_with_orientation_zero(self):
        """Generator end-to-end with orientation_strength=0 should succeed."""
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gradient_gray(60, 60)
        params = {
            "_source_image": np.stack([gray, gray, gray], axis=2),
            "min_sample_spacing_mm": 6.0,
            "max_sample_spacing_mm": 18.0,
            "path_grid_size": 4,
            "seed": 0,
            "orientation_strength": 0.0,
            "skip_background": False,
        }
        result = gen.generate(params, canvas)
        assert len(result) >= 1
        assert sum(len(pl) for pl in result) > 0

    def test_generate_with_skip_background(self):
        """Generator end-to-end with skip_background=True on a white image."""
        gen = CircularScribbleGenerator()
        canvas = _make_canvas()
        gray = _make_gray(60, 60, 255)
        params = {
            "_source_image": np.stack([gray, gray, gray], axis=2),
            "min_sample_spacing_mm": 4.0,
            "max_sample_spacing_mm": 12.0,
            "path_grid_size": 4,
            "seed": 0,
            "orientation_strength": 0.3,
            "skip_background": True,
        }
        result = gen.generate(params, canvas)
        # Should produce valid output (may be empty or sparse for all-white)
        assert isinstance(result, list)

    def test_each_preset_produces_visually_distinct_output(self):
        """Different presets with the same image should differ in total point count."""
        gen = CircularScribbleGenerator()
        canvas = _make_canvas(100.0, 100.0)
        gray = _make_gradient_gray(50, 50)
        source = np.stack([gray, gray, gray], axis=2)

        counts: dict[str, int] = {}
        for preset in gen.get_presets():
            params = dict(preset.params)
            params["_source_image"] = source
            result = gen.generate(params, canvas)
            counts[preset.name] = sum(len(pl) for pl in result)

        # At least some presets should produce different point counts
        # (loose sketch vs detailed have very different densities)
        unique_counts = set(counts.values())
        assert len(unique_counts) > 1, f"All presets produced identical counts: {counts}"
