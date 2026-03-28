"""Tests for StippleGenerator render_mode parameter (TSP Path mode).

Tests:
(a) "Dots" mode produces same output as before (tiny circle polylines)
(b) "TSP Path" mode produces exactly 1 polyline
(c) TSP path visits all stipple points
(d) 2-opt optimization reduces path length compared to unoptimized
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def make_canvas():
    from plottter.models.canvas import Canvas
    return Canvas(width_mm=200.0, height_mm=200.0, margin_mm=10.0)


def make_gradient_image(w: int = 80, h: int = 80) -> np.ndarray:
    """Create a simple gradient image (left dark, right bright)."""
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        img[:, col] = int(col / (w - 1) * 255)
    return img


def _path_length(polyline) -> float:
    """Compute total Euclidean length of a polyline."""
    total = 0.0
    for i in range(len(polyline) - 1):
        dx = polyline[i + 1][0] - polyline[i][0]
        dy = polyline[i + 1][1] - polyline[i][1]
        total += math.hypot(dx, dy)
    return total


class TestStippleRenderMode:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = make_canvas()
        self.img = make_gradient_image(80, 80)

    def _base_params(self, **overrides) -> dict:
        base = {
            "_source_image": self.img,
            "num_points": 30,
            "iterations": 2,
            "min_dot_spacing_mm": 0.0,
            "seed": 42,
            "render_mode": "Dots",
            "tsp_optimize": True,
        }
        base.update(overrides)
        return base

    # -------------------------------------------------------------------------
    # (a) "Dots" mode produces same output as before (tiny circle polylines)
    # -------------------------------------------------------------------------

    def test_dots_mode_produces_circle_polylines(self):
        """Dots mode should produce one small-circle polyline per stipple point."""
        from plottter.generators.stipple import _DOT_SIDES
        params = self._base_params(render_mode="Dots", num_points=20)
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 20, f"Expected 20 dot polylines, got {len(result)}"
        # Each dot is a closed circle: _DOT_SIDES+1 points
        expected_pts = _DOT_SIDES + 1
        for poly in result:
            assert len(poly) == expected_pts, (
                f"Dot circle should have {expected_pts} points, got {len(poly)}"
            )

    def test_dots_mode_default_when_no_render_mode(self):
        """Omitting render_mode defaults to Dots behavior."""
        params = {
            "_source_image": self.img,
            "num_points": 20,
            "iterations": 2,
            "min_dot_spacing_mm": 0.0,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        # Should produce individual polylines (dots), not a single TSP path
        assert len(result) == 20

    def test_render_mode_param_exists(self):
        """render_mode and tsp_optimize must be exposed as parameters."""
        names = [p.name for p in self.gen.get_parameters()]
        assert "render_mode" in names, "render_mode param must be defined"
        assert "tsp_optimize" in names, "tsp_optimize param must be defined"

    # -------------------------------------------------------------------------
    # (b) "TSP Path" mode produces exactly 1 polyline
    # -------------------------------------------------------------------------

    def test_tsp_path_mode_produces_single_polyline(self):
        """TSP Path mode must return exactly one polyline."""
        params = self._base_params(render_mode="TSP Path", num_points=30)
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1, (
            f"TSP Path mode should produce 1 polyline, got {len(result)}"
        )

    def test_tsp_path_mode_gradient_image(self):
        """TSP Path mode on gradient image returns a single polyline."""
        params = self._base_params(render_mode="TSP Path", num_points=50)
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1

    # -------------------------------------------------------------------------
    # (c) TSP path visits all stipple points
    # -------------------------------------------------------------------------

    def test_tsp_path_visits_all_points(self):
        """The TSP path must contain exactly num_points vertices."""
        n = 40
        params = self._base_params(render_mode="TSP Path", num_points=n)
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1
        assert len(result[0]) == n, (
            f"TSP path should visit all {n} points, got {len(result[0])}"
        )

    def test_tsp_path_points_are_unique_positions(self):
        """Each vertex in the TSP path should be a distinct stipple position."""
        n = 30
        params = self._base_params(render_mode="TSP Path", num_points=n)
        result = self.gen.generate(params, self.canvas)
        path = result[0]
        # All points should be distinct (no duplicated positions)
        unique = set((round(x, 6), round(y, 6)) for x, y in path)
        assert len(unique) == n, (
            f"Expected {n} unique positions, got {len(unique)}"
        )

    # -------------------------------------------------------------------------
    # (d) 2-opt optimization reduces path length compared to unoptimized
    # -------------------------------------------------------------------------

    def test_2opt_reduces_or_maintains_path_length(self):
        """Path with tsp_optimize=True should be <= length of tsp_optimize=False."""
        n = 100  # larger point count for meaningful optimization
        # Use a dark image so points spread across the canvas
        img = np.full((80, 80), 128, dtype=np.uint8)

        params_no_opt = {
            "_source_image": img,
            "num_points": n,
            "iterations": 2,
            "min_dot_spacing_mm": 0.0,
            "seed": 7,
            "render_mode": "TSP Path",
            "tsp_optimize": False,
        }
        params_opt = dict(params_no_opt)
        params_opt["tsp_optimize"] = True

        result_no_opt = self.gen.generate(params_no_opt, self.canvas)
        result_opt = self.gen.generate(params_opt, self.canvas)

        assert len(result_no_opt) == 1
        assert len(result_opt) == 1

        len_no_opt = _path_length(result_no_opt[0])
        len_opt = _path_length(result_opt[0])

        # 2-opt optimized path should not be longer than unoptimized
        assert len_opt <= len_no_opt + 1e-6, (
            f"Optimized path ({len_opt:.2f} mm) should be <= unoptimized ({len_no_opt:.2f} mm)"
        )

    def test_tsp_optimize_param_controls_optimization(self):
        """tsp_optimize=False should skip 2-opt; True should apply it."""
        # Both should still produce a single polyline
        n = 50
        img = np.full((80, 80), 100, dtype=np.uint8)
        base = {
            "_source_image": img,
            "num_points": n,
            "iterations": 1,
            "min_dot_spacing_mm": 0.0,
            "seed": 99,
            "render_mode": "TSP Path",
        }

        result_false = self.gen.generate({**base, "tsp_optimize": False}, self.canvas)
        result_true = self.gen.generate({**base, "tsp_optimize": True}, self.canvas)

        assert len(result_false) == 1, "tsp_optimize=False should still return 1 polyline"
        assert len(result_true) == 1, "tsp_optimize=True should still return 1 polyline"
        assert len(result_false[0]) == n
        assert len(result_true[0]) == n

    # -------------------------------------------------------------------------
    # Presets include new params
    # -------------------------------------------------------------------------

    def test_all_presets_include_render_mode_and_tsp_optimize(self):
        """Every preset must include render_mode and tsp_optimize."""
        required = {"render_mode", "tsp_optimize"}
        for preset in self.gen.get_presets():
            missing = required - set(preset.params.keys())
            assert not missing, (
                f"Preset '{preset.name}' missing keys: {missing}"
            )

    def test_tsp_art_presets_use_tsp_path_mode(self):
        """TSP Art presets must use render_mode='TSP Path'."""
        tsp_presets = [p for p in self.gen.get_presets() if "TSP" in p.name]
        assert len(tsp_presets) >= 1, "At least one TSP Art preset must exist"
        for preset in tsp_presets:
            assert preset.params.get("render_mode") == "TSP Path", (
                f"Preset '{preset.name}' should have render_mode='TSP Path'"
            )
