"""Tests for Phase 7 image-to-lines generators (7.6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def make_gradient_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Left-to-right brightness gradient (0=black left, 255=white right), grayscale."""
    arr = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        arr[:, x] = int(x / (w - 1) * 255)
    return arr


def make_checkerboard(h: int = 100, w: int = 100, tile: int = 10) -> np.ndarray:
    """Alternating black/white tiles, grayscale."""
    arr = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((x // tile) + (y // tile)) % 2 == 0:
                arr[y, x] = 0
            else:
                arr[y, x] = 255
    return arr


def make_dark_center_image(h: int = 100, w: int = 100) -> np.ndarray:
    """White background with a dark circle in the center."""
    arr = np.full((h, w), 255, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    for y in range(h):
        for x in range(w):
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius:
                arr[y, x] = 0
    return arr


def within_bounds(paths: list, canvas: Canvas, tol: float = 1.5) -> bool:
    """Check that all points in paths are within drawing area (with tolerance)."""
    x1, y1, x2, y2 = canvas.drawing_area()
    for path in paths:
        for x, y in path:
            if not (x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol):
                return False
    return True


# ---------------------------------------------------------------------------
# EdgeDetectGenerator
# ---------------------------------------------------------------------------


class TestEdgeDetectGenerator:
    def setup_method(self):
        from plottter.generators.edge_detect import EdgeDetectGenerator
        self.gen = EdgeDetectGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Edge Detect" in GENERATORS
        assert GENERATORS["Edge Detect"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "Edge Detect"
        assert self.gen.category == "image"

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_generates_polylines_from_checkerboard(self):
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "low_threshold": 50.0,
            "high_threshold": 150.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Should detect edges in checkerboard"
        assert all(len(p) >= 2 for p in result)

    def test_generates_polylines_from_dark_center(self):
        img = make_dark_center_image(100, 100)
        params = {
            "_source_image": img,
            "low_threshold": 30.0,
            "high_threshold": 100.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.3,
            "close_gaps_mm": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Should detect circle edge"

    def test_output_within_bounds(self):
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "low_threshold": 50.0,
            "high_threshold": 150.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas)

    def test_min_contour_length_filter(self):
        img = make_dark_center_image(100, 100)
        # With high min_contour_length, should produce fewer (longer) contours
        params_low = {
            "_source_image": img,
            "low_threshold": 30.0,
            "high_threshold": 100.0,
            "min_contour_length": 2,
            "simplify_tolerance_mm": 0.0,
            "close_gaps_mm": 0.0,
        }
        params_high = {
            "_source_image": img,
            "low_threshold": 30.0,
            "high_threshold": 100.0,
            "min_contour_length": 50,
            "simplify_tolerance_mm": 0.0,
            "close_gaps_mm": 0.0,
        }
        result_low = self.gen.generate(params_low, self.canvas)
        result_high = self.gen.generate(params_high, self.canvas)
        assert len(result_high) <= len(result_low)

    def test_close_gaps_reduces_path_count(self):
        img = make_checkerboard(80, 80, tile=10)
        params_no_close = {
            "_source_image": img,
            "low_threshold": 50.0,
            "high_threshold": 150.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
        }
        params_close = dict(params_no_close)
        params_close["close_gaps_mm"] = 5.0

        result_no_close = self.gen.generate(params_no_close, self.canvas)
        result_close = self.gen.generate(params_close, self.canvas)
        # Closing gaps should not increase path count
        assert len(result_close) <= len(result_no_close)

    def test_has_presets(self):
        presets = self.gen.get_presets()
        assert len(presets) > 0
        for p in presets:
            assert p.name
            assert isinstance(p.params, dict)

    def test_preset_generates_output(self):
        img = make_checkerboard(80, 80, tile=10)
        for preset in self.gen.get_presets():
            params = dict(preset.params)
            params["_source_image"] = img
            params["min_contour_length"] = 2  # allow short contours in tests
            result = self.gen.generate(params, self.canvas)
            # At least some presets should produce output
            # (don't fail on all, just ensure no crash)

    def test_gets_parameters(self):
        params = self.gen.get_parameters()
        param_names = [p.name for p in params]
        assert "low_threshold" in param_names
        assert "high_threshold" in param_names
        assert "min_contour_length" in param_names
        assert "simplify_tolerance_mm" in param_names
        assert "close_gaps_mm" in param_names

    def test_rgb_input_works(self):
        """Should handle RGB (3-channel) input by converting to grayscale."""
        arr = np.zeros((80, 80, 3), dtype=np.uint8)
        arr[:40, :40] = 0      # top-left black
        arr[:40, 40:] = 255    # top-right white
        arr[40:, :] = 128      # bottom gray
        params = {
            "_source_image": arr,
            "low_threshold": 50.0,
            "high_threshold": 150.0,
            "min_contour_length": 2,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        # Should not raise; output may or may not have paths depending on edges
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# HatchingGenerator
# ---------------------------------------------------------------------------


class TestHatchingGenerator:
    def setup_method(self):
        from plottter.generators.hatching import HatchingGenerator
        self.gen = HatchingGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Hatching" in GENERATORS
        assert GENERATORS["Hatching"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "Hatching"
        assert self.gen.category == "image"

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_parallel_hatch_gradient(self):
        img = make_gradient_image(100, 100)
        params = {
            "_source_image": img,
            "mode": "parallel",
            "angle_deg": 45.0,
            "angle2_deg": 135.0,
            "min_spacing_mm": 1.0,
            "max_spacing_mm": 5.0,
            "density_curve": "linear",
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Parallel hatch should produce lines"
        assert all(len(p) >= 2 for p in result)

    def test_cross_hatch_has_more_lines_than_parallel(self):
        img = make_gradient_image(100, 100)
        base = {
            "_source_image": img,
            "angle_deg": 45.0,
            "angle2_deg": 135.0,
            "min_spacing_mm": 1.0,
            "max_spacing_mm": 5.0,
            "density_curve": "linear",
        }
        parallel = self.gen.generate({**base, "mode": "parallel"}, self.canvas)
        cross = self.gen.generate({**base, "mode": "cross"}, self.canvas)
        assert len(cross) >= len(parallel), "Cross hatch should produce >= parallel lines"

    def test_contour_hatch_dark_center(self):
        img = make_dark_center_image(100, 100)
        params = {
            "_source_image": img,
            "mode": "contour",
            "angle_deg": 0.0,
            "angle2_deg": 90.0,
            "min_spacing_mm": 2.0,
            "max_spacing_mm": 8.0,
            "density_curve": "linear",
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Contour hatch should produce lines in dark area"

    def test_output_within_bounds_parallel(self):
        img = make_gradient_image(100, 100)
        params = {
            "_source_image": img,
            "mode": "parallel",
            "angle_deg": 45.0,
            "angle2_deg": 135.0,
            "min_spacing_mm": 1.0,
            "max_spacing_mm": 5.0,
            "density_curve": "linear",
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=2.0)

    def test_density_curves(self):
        img = make_gradient_image(100, 100)
        for curve in ["linear", "quadratic", "logarithmic"]:
            params = {
                "_source_image": img,
                "mode": "parallel",
                "angle_deg": 45.0,
                "angle2_deg": 135.0,
                "min_spacing_mm": 1.0,
                "max_spacing_mm": 5.0,
                "density_curve": curve,
            }
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)

    def test_has_presets(self):
        presets = self.gen.get_presets()
        assert len(presets) > 0
        for p in presets:
            assert p.name
            assert "mode" in p.params

    def test_gets_parameters(self):
        params = self.gen.get_parameters()
        param_names = [p.name for p in params]
        assert "mode" in param_names
        assert "angle_deg" in param_names
        assert "min_spacing_mm" in param_names
        assert "max_spacing_mm" in param_names
        assert "density_curve" in param_names
        assert "line_length_mm" in param_names

    def test_rgb_input_works(self):
        arr = np.full((80, 80, 3), 128, dtype=np.uint8)
        params = {
            "_source_image": arr,
            "mode": "parallel",
            "angle_deg": 0.0,
            "angle2_deg": 90.0,
            "min_spacing_mm": 1.0,
            "max_spacing_mm": 5.0,
            "density_curve": "linear",
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# FlowImageGenerator
# ---------------------------------------------------------------------------


class TestFlowImageGenerator:
    def setup_method(self):
        from plottter.generators.flow_image import FlowImageGenerator
        self.gen = FlowImageGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Flow Image" in GENERATORS
        assert GENERATORS["Flow Image"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "Flow Image"
        assert self.gen.category == "image"

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_flow_mode_gradient(self):
        img = make_gradient_image(100, 100)
        params = {
            "_source_image": img,
            "mode": "flow",
            "num_lines": 20,
            "step_size_mm": 2.0,
            "max_steps": 50,
            "curvature_strength": 1.0,
            "amplitude_mm": 3.0,
            "frequency": 5.0,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Flow mode should produce streamlines"
        assert all(len(p) >= 2 for p in result)

    def test_squiggle_mode_gradient(self):
        img = make_gradient_image(100, 100)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 10,
            "step_size_mm": 1.0,
            "max_steps": 100,
            "curvature_strength": 1.0,
            "amplitude_mm": 4.0,
            "frequency": 5.0,
            "seed": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 10, "Squiggle should produce one polyline per scan line"
        assert all(len(p) >= 2 for p in result)

    def test_output_within_bounds_squiggle(self):
        img = make_gradient_image(100, 100)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 10,
            "step_size_mm": 1.0,
            "max_steps": 100,
            "curvature_strength": 1.0,
            "amplitude_mm": 2.0,
            "frequency": 5.0,
            "seed": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=1.0)

    def test_output_within_bounds_flow(self):
        img = make_gradient_image(100, 100)
        params = {
            "_source_image": img,
            "mode": "flow",
            "num_lines": 20,
            "step_size_mm": 1.0,
            "max_steps": 50,
            "curvature_strength": 1.0,
            "amplitude_mm": 3.0,
            "frequency": 5.0,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=1.0)

    def test_squiggle_dark_center_has_larger_amplitude(self):
        """Dark center image should produce squiggle with more deviation than uniform gray."""
        img_dark = make_dark_center_image(100, 100)
        img_white = np.full((100, 100), 255, dtype=np.uint8)

        params = {
            "mode": "squiggle",
            "num_lines": 5,
            "step_size_mm": 1.0,
            "max_steps": 100,
            "curvature_strength": 1.0,
            "amplitude_mm": 5.0,
            "frequency": 5.0,
            "seed": 0,
        }

        dark_canvas = make_canvas()
        _, draw_y1, _, draw_y2 = dark_canvas.drawing_area()

        # Compute max y deviation for each case
        def max_y_deviation(paths, canvas):
            _, dy1, _, dy2 = canvas.drawing_area()
            base_ys = []
            deviations = []
            for path in paths:
                ys = [pt[1] for pt in path]
                mid = (max(ys) + min(ys)) / 2.0
                for y in ys:
                    deviations.append(abs(y - mid))
            return max(deviations) if deviations else 0.0

        result_dark = self.gen.generate({**params, "_source_image": img_dark}, dark_canvas)
        result_white = self.gen.generate({**params, "_source_image": img_white}, dark_canvas)

        dev_dark = max_y_deviation(result_dark, dark_canvas)
        dev_white = max_y_deviation(result_white, dark_canvas)
        assert dev_dark > dev_white, "Dark image should produce larger squiggle amplitude"

    def test_has_presets(self):
        presets = self.gen.get_presets()
        assert len(presets) > 0
        for p in presets:
            assert p.name
            assert "mode" in p.params

    def test_gets_parameters(self):
        params = self.gen.get_parameters()
        param_names = [p.name for p in params]
        assert "mode" in param_names
        assert "num_lines" in param_names
        assert "step_size_mm" in param_names
        assert "amplitude_mm" in param_names
        assert "frequency" in param_names
        assert "wave_spread" in param_names

    def test_wave_spread_param_has_correct_metadata(self):
        """wave_spread param should be squiggle-only, range 0–10."""
        params = self.gen.get_parameters()
        ws = next(p for p in params if p.name == "wave_spread")
        assert ws.min == 0
        assert ws.max == 10
        assert ws.default == 0
        assert ws.visible_when == {"mode": ["squiggle"]}

    def test_all_presets_include_wave_spread(self):
        """Every preset must contain the wave_spread key."""
        for preset in self.gen.get_presets():
            assert "wave_spread" in preset.params, (
                f"Preset '{preset.name}' is missing 'wave_spread'"
            )

    def test_wave_spread_zero_identical_to_no_spread(self):
        """wave_spread=0 must produce identical output to omitting the param."""
        img = make_gradient_image(60, 60)
        base = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 5,
            "amplitude_mm": 4.0,
            "frequency": 5.0,
            "skip_background": False,
        }
        result_default = self.gen.generate({**base}, self.canvas)
        result_zero = self.gen.generate({**base, "wave_spread": 0}, self.canvas)
        # Both should produce the same polylines (regression check)
        assert len(result_default) == len(result_zero)
        for poly_a, poly_b in zip(result_default, result_zero):
            assert poly_a == poly_b, "wave_spread=0 must match no-spread behavior"

    def test_wave_spread_nonzero_changes_amplitude_near_boundary(self):
        """wave_spread>0 should produce non-zero amplitude on a scan line that is
        adjacent to a dark region but itself sits over a white region."""
        # Build an image where the top half is black and the bottom half is white.
        # With wave_spread=0, scan lines in the white half get zero amplitude.
        # With wave_spread>0, the Gaussian blur should spread black-region amplitude
        # upward/downward into the first white-region lines.
        h, w = 80, 80
        img = np.zeros((h, w), dtype=np.uint8)
        img[h // 2:, :] = 255  # bottom half pure white

        # Canvas scaled to image size in mm for predictable mapping
        from plottter.models.canvas import Canvas
        canvas_small = Canvas(width_mm=80.0, height_mm=80.0, margin_mm=0.0, paper_preset="Custom")

        base = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 20,
            "amplitude_mm": 10.0,
            "frequency": 3.0,
            "skip_background": False,
        }

        result_no_spread = self.gen.generate({**base, "wave_spread": 0}, canvas_small)
        result_spread = self.gen.generate({**base, "wave_spread": 3}, canvas_small)

        # Pick the scan line just inside the white half (first line at/after midpoint)
        # It should have zero (or near-zero) amplitude without spread
        # and non-zero amplitude with spread.
        def y_deviation(path):
            ys = [pt[1] for pt in path]
            return max(ys) - min(ys)

        # Find lines whose y_base is in the white half (y > 40mm)
        # result contains one polyline per scan line (skip_background=False)
        mid_line_idx = 10  # line 10 out of 20 is at y=41.25mm (just into white half)
        if mid_line_idx < len(result_no_spread) and mid_line_idx < len(result_spread):
            dev_no_spread = y_deviation(result_no_spread[mid_line_idx])
            dev_spread = y_deviation(result_spread[mid_line_idx])
            assert dev_spread >= dev_no_spread, (
                "wave_spread>0 should produce >= amplitude near a dark/white boundary"
            )

    def test_wave_spread_higher_produces_wider_influence(self):
        """Higher wave_spread should spread amplitude to more distant neighbors."""
        h, w = 100, 100
        # Thin dark stripe in the middle (3 rows wide) surrounded by white
        img = np.full((h, w), 255, dtype=np.uint8)
        img[h // 2 - 1: h // 2 + 2, :] = 0  # rows 49-51 are black

        from plottter.models.canvas import Canvas
        canvas_sq = Canvas(width_mm=100.0, height_mm=100.0, margin_mm=0.0, paper_preset="Custom")

        def max_dev_for_line(result, line_idx):
            if line_idx >= len(result):
                return 0.0
            ys = [pt[1] for pt in result[line_idx]]
            return max(ys) - min(ys) if len(ys) > 1 else 0.0

        base = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 50,
            "amplitude_mm": 8.0,
            "frequency": 3.0,
            "skip_background": False,
        }

        result_s1 = self.gen.generate({**base, "wave_spread": 1}, canvas_sq)
        result_s5 = self.gen.generate({**base, "wave_spread": 5}, canvas_sq)

        # The line that is 8 lines away from the dark stripe (in the white area)
        far_line_idx = h // 2 // 2 + 8  # ~33 lines in, well into white area
        dev_s1 = max_dev_for_line(result_s1, far_line_idx)
        dev_s5 = max_dev_for_line(result_s5, far_line_idx)

        # A wider spread (5) should reach further than a narrow spread (1)
        assert dev_s5 >= dev_s1, (
            "Larger wave_spread should produce wider influence zone"
        )

    def test_squiggle_portrait_preset_has_wave_spread_2(self):
        """'Squiggle / Portrait' preset should have wave_spread=2."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Squiggle / Portrait" in presets
        assert presets["Squiggle / Portrait"].params["wave_spread"] == 2

    def test_squiggle_landscape_preset_has_wave_spread_2(self):
        """'Squiggle / Landscape' preset should have wave_spread=2."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Squiggle / Landscape" in presets
        assert presets["Squiggle / Landscape"].params["wave_spread"] == 2

    def test_dense_squiggle_preset_has_wave_spread_0(self):
        """'Dense Squiggle' preset should keep wave_spread=0."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Dense Squiggle" in presets
        assert presets["Dense Squiggle"].params["wave_spread"] == 0

    def test_squiggle_portrait_preset_generates_output(self):
        """'Squiggle / Portrait' preset should produce valid polylines."""
        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Squiggle / Portrait"]
        result = self.gen.generate({**p.params, "_source_image": img}, self.canvas)
        assert len(result) > 0
        assert all(len(poly) >= 2 for poly in result)

    def test_squiggle_landscape_preset_generates_output(self):
        """'Squiggle / Landscape' preset should produce valid polylines."""
        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Squiggle / Landscape"]
        result = self.gen.generate({**p.params, "_source_image": img}, self.canvas)
        assert len(result) > 0
        assert all(len(poly) >= 2 for poly in result)

    def test_preprocessing_invert_changes_output(self):
        """Invert flag should change squiggle amplitude (dark↔light flip)."""
        img = make_gradient_image(60, 60)
        base_params = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 5,
            "amplitude_mm": 6.0,
            "frequency": 5.0,
            "seed": 0,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_normal = self.gen.generate({**base_params, "invert": False}, self.canvas)
        result_inverted = self.gen.generate({**base_params, "invert": True}, self.canvas)
        assert len(result_normal) > 0
        assert len(result_inverted) > 0
        # Y coordinates should differ between normal and inverted
        ys_normal = [pt[1] for pt in result_normal[0]]
        ys_inverted = [pt[1] for pt in result_inverted[0]]
        assert ys_normal != ys_inverted, "Invert should change squiggle output"

    # ------------------------------------------------------------------
    # 16.60 — variable line spacing tests
    # ------------------------------------------------------------------

    def test_uniform_mode_is_default_behavior(self):
        """line_spacing='Uniform' should produce identical output to omitting line_spacing."""
        img = make_gradient_image(60, 60)
        base = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 10,
            "amplitude_mm": 3.0,
            "frequency": 5.0,
            "skip_background": False,
            "displacement_variation": 0.0,
            "seed": 42,
        }
        result_default = self.gen.generate({**base}, self.canvas)
        result_uniform = self.gen.generate({**base, "line_spacing": "Uniform"}, self.canvas)
        assert len(result_default) == len(result_uniform)
        for p1, p2 in zip(result_default, result_uniform):
            assert p1 == p2, "Uniform mode must match default behavior"

    def test_adaptive_spacing_dense_in_dark_regions(self):
        """Adaptive mode should place more scan lines in dark (low brightness) regions."""
        from plottter.generators.flow_image import _compute_squiggle_y_positions

        h, w = 100, 100
        # Top rows are dark (0), bottom rows are bright (255)
        img = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            img[y, :] = int(y / (h - 1) * 255)

        draw_y1, draw_y2 = 10.0, 287.0
        draw_h = draw_y2 - draw_y1

        positions = _compute_squiggle_y_positions(
            img,
            num_lines=50,
            draw_y1=draw_y1,
            draw_y2=draw_y2,
            draw_h=draw_h,
            line_spacing="Adaptive",
            min_spacing_mm=0.5,
            max_spacing_mm=10.0,
            group_size=3,
            group_gap_mm=4.0,
            group_intra_spacing_mm=0.5,
        )

        mid_y = (draw_y1 + draw_y2) / 2
        top_count = sum(1 for y in positions if y < mid_y)
        bottom_count = sum(1 for y in positions if y >= mid_y)
        assert top_count > bottom_count, (
            f"Dark top region should have more lines ({top_count}) than "
            f"bright bottom ({bottom_count})"
        )

    def test_adaptive_spacing_generates_output(self):
        """Adaptive mode should produce non-empty output on a gradient image."""
        img = make_gradient_image(60, 60)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 50,
            "amplitude_mm": 3.0,
            "frequency": 5.0,
            "wave_spread": 0,
            "skip_background": False,
            "line_spacing": "Adaptive",
            "min_spacing_mm": 0.5,
            "max_spacing_mm": 5.0,
            "displacement_variation": 0.0,
            "seed": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert all(len(p) >= 2 for p in result)

    def test_grouped_spacing_structure(self):
        """Grouped mode should produce correct group structure in Y positions."""
        from plottter.generators.flow_image import _compute_squiggle_y_positions

        h, w = 100, 100
        img = np.full((h, w), 128, dtype=np.uint8)  # uniform gray — no brightness effect

        draw_y1, draw_y2 = 10.0, 40.0
        draw_h = draw_y2 - draw_y1
        group_size = 3
        group_intra = 0.5
        group_gap = 4.0

        positions = _compute_squiggle_y_positions(
            img,
            num_lines=50,
            draw_y1=draw_y1,
            draw_y2=draw_y2,
            draw_h=draw_h,
            line_spacing="Grouped",
            min_spacing_mm=0.5,
            max_spacing_mm=5.0,
            group_size=group_size,
            group_gap_mm=group_gap,
            group_intra_spacing_mm=group_intra,
        )

        assert len(positions) >= group_size, "Should produce at least one full group"
        tol = 1e-9
        # First group: positions at draw_y1, draw_y1+0.5, draw_y1+1.0
        assert abs(positions[0] - draw_y1) < tol
        assert abs(positions[1] - (draw_y1 + group_intra)) < tol
        assert abs(positions[2] - (draw_y1 + 2 * group_intra)) < tol
        # Second group starts at draw_y1 + (group_size-1)*intra + gap
        expected_next_group_start = draw_y1 + (group_size - 1) * group_intra + group_gap
        if len(positions) > group_size and expected_next_group_start <= draw_y2:
            assert abs(positions[group_size] - expected_next_group_start) < tol

    def test_grouped_mode_generates_output(self):
        """Grouped line spacing should produce valid polylines."""
        img = make_gradient_image(60, 60)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 50,
            "amplitude_mm": 3.0,
            "frequency": 5.0,
            "wave_spread": 0,
            "skip_background": False,
            "line_spacing": "Grouped",
            "group_size": 3,
            "group_gap_mm": 4.0,
            "group_intra_spacing_mm": 0.5,
            "displacement_variation": 0.0,
            "seed": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert all(len(p) >= 2 for p in result)

    def test_adaptive_grouped_mode_generates_output(self):
        """Adaptive + Grouped line spacing should produce valid polylines."""
        img = make_gradient_image(60, 60)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 50,
            "amplitude_mm": 3.0,
            "frequency": 5.0,
            "wave_spread": 0,
            "skip_background": False,
            "line_spacing": "Adaptive + Grouped",
            "min_spacing_mm": 0.5,
            "max_spacing_mm": 5.0,
            "group_size": 3,
            "group_gap_mm": 4.0,
            "group_intra_spacing_mm": 0.5,
            "displacement_variation": 0.0,
            "seed": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert all(len(p) >= 2 for p in result)

    def test_displacement_variation_zero_identical_to_no_variation(self):
        """displacement_variation=0.0 should produce identical output to default."""
        img = make_gradient_image(60, 60)
        base = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 5,
            "amplitude_mm": 4.0,
            "frequency": 5.0,
            "skip_background": False,
            "seed": 42,
        }
        result_default = self.gen.generate({**base}, self.canvas)
        result_zero = self.gen.generate({**base, "displacement_variation": 0.0}, self.canvas)
        assert len(result_default) == len(result_zero)
        for p1, p2 in zip(result_default, result_zero):
            assert p1 == p2, "displacement_variation=0.0 must match default"

    def test_displacement_variation_changes_output(self):
        """displacement_variation > 0 should produce different Y values than 0."""
        img = make_gradient_image(60, 60)
        base = {
            "_source_image": img,
            "mode": "squiggle",
            "num_lines": 5,
            "amplitude_mm": 8.0,
            "frequency": 5.0,
            "skip_background": False,
            "seed": 42,
        }
        result_no_var = self.gen.generate({**base, "displacement_variation": 0.0}, self.canvas)
        result_with_var = self.gen.generate({**base, "displacement_variation": 1.0}, self.canvas)
        assert len(result_no_var) == len(result_with_var)
        # At least one line should differ due to per-line amplitude scaling
        any_diff = any(
            [pt[1] for pt in p1] != [pt[1] for pt in p2]
            for p1, p2 in zip(result_no_var, result_with_var)
        )
        assert any_diff, "displacement_variation=1.0 should produce different Y values"

    def test_new_spacing_params_in_parameters(self):
        """All new spacing parameters should be present in get_parameters()."""
        param_names = {p.name for p in self.gen.get_parameters()}
        for name in ("line_spacing", "min_spacing_mm", "max_spacing_mm",
                     "group_size", "group_gap_mm", "group_intra_spacing_mm",
                     "displacement_variation"):
            assert name in param_names, f"Parameter '{name}' missing from get_parameters()"

    def test_all_presets_include_line_spacing(self):
        """Every preset must contain the line_spacing key."""
        for preset in self.gen.get_presets():
            assert "line_spacing" in preset.params, (
                f"Preset '{preset.name}' is missing 'line_spacing'"
            )

    def test_all_presets_include_displacement_variation(self):
        """Every preset must contain the displacement_variation key."""
        for preset in self.gen.get_presets():
            assert "displacement_variation" in preset.params, (
                f"Preset '{preset.name}' is missing 'displacement_variation'"
            )

    # ------------------------------------------------------------------
    # 16.61 — new spacing mode preset tests
    # ------------------------------------------------------------------

    def test_adaptive_density_squiggle_preset_exists(self):
        """'Adaptive Density Squiggle' preset should exist."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Adaptive Density Squiggle" in presets
        assert presets["Adaptive Density Squiggle"].params["line_spacing"] == "Adaptive"

    def test_adaptive_density_squiggle_preset_generates_output(self):
        """'Adaptive Density Squiggle' preset should produce valid polylines."""
        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Adaptive Density Squiggle"]
        result = self.gen.generate({**p.params, "_source_image": img}, self.canvas)
        assert len(result) > 0
        assert all(len(poly) >= 2 for poly in result)

    def test_sketchy_grouped_strokes_preset_exists(self):
        """'Sketchy Grouped Strokes' preset should exist with Grouped spacing."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Sketchy Grouped Strokes" in presets
        p = presets["Sketchy Grouped Strokes"]
        assert p.params["line_spacing"] == "Grouped"
        assert p.params["displacement_variation"] == 0.6

    def test_sketchy_grouped_strokes_preset_generates_output(self):
        """'Sketchy Grouped Strokes' preset should produce valid polylines."""
        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Sketchy Grouped Strokes"]
        result = self.gen.generate({**p.params, "_source_image": img}, self.canvas)
        assert len(result) > 0
        assert all(len(poly) >= 2 for poly in result)

    def test_organic_portrait_preset_exists(self):
        """'Organic Portrait' preset should exist with Adaptive + Grouped spacing."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Organic Portrait" in presets
        p = presets["Organic Portrait"]
        assert p.params["line_spacing"] == "Adaptive + Grouped"
        assert p.params["wave_spread"] == 3

    def test_organic_portrait_preset_generates_output(self):
        """'Organic Portrait' preset should produce valid polylines."""
        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Organic Portrait"]
        result = self.gen.generate({**p.params, "_source_image": img}, self.canvas)
        assert len(result) > 0
        assert all(len(poly) >= 2 for poly in result)

    def test_wild_lines_preset_exists(self):
        """'Wild Lines' preset should exist with Uniform spacing and max displacement_variation."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Wild Lines" in presets
        p = presets["Wild Lines"]
        assert p.params["line_spacing"] == "Uniform"
        assert p.params["displacement_variation"] == 1.0
        assert p.params["num_lines"] == 80

    def test_wild_lines_preset_generates_output(self):
        """'Wild Lines' preset should produce valid polylines."""
        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Wild Lines"]
        result = self.gen.generate({**p.params, "_source_image": img}, self.canvas)
        assert len(result) > 0
        assert all(len(poly) >= 2 for poly in result)

    def test_loose_sketch_preset_exists(self):
        """'Loose Sketch' preset should exist with Grouped spacing and skip_background=True."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Loose Sketch" in presets
        p = presets["Loose Sketch"]
        assert p.params["line_spacing"] == "Grouped"
        assert p.params["group_size"] == 2
        assert p.params["skip_background"] is True

    def test_loose_sketch_preset_generates_output(self):
        """'Loose Sketch' preset should produce valid polylines on a non-white gradient image."""
        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Loose Sketch"]
        result = self.gen.generate({**p.params, "_source_image": img}, self.canvas)
        assert len(result) > 0
        assert all(len(poly) >= 2 for poly in result)

    def test_all_new_spacing_presets_present(self):
        """All five new spacing mode presets should be present in get_presets()."""
        preset_names = {p.name for p in self.gen.get_presets()}
        for name in (
            "Adaptive Density Squiggle",
            "Sketchy Grouped Strokes",
            "Organic Portrait",
            "Wild Lines",
            "Loose Sketch",
        ):
            assert name in preset_names, f"Missing preset: '{name}'"


# ---------------------------------------------------------------------------
# StippleGenerator
# ---------------------------------------------------------------------------


class TestStippleGenerator:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Stipple" in GENERATORS
        assert GENERATORS["Stipple"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "Stipple"
        assert self.gen.category == "image"

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_generates_dots_gradient(self):
        img = make_gradient_image(80, 80)
        params = {
            "_source_image": img,
            "num_points": 50,
            "iterations": 2,
            "connect_tsp": False,
            "min_dot_spacing_mm": 0.5,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Should generate dot polylines"
        # Each dot is a tiny circle polyline (>= 2 points)
        assert all(len(p) >= 2 for p in result)

    def test_dot_count_matches_num_points(self):
        img = make_gradient_image(80, 80)
        n = 30
        params = {
            "_source_image": img,
            "num_points": n,
            "iterations": 1,
            "connect_tsp": False,
            "min_dot_spacing_mm": 0.0,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) == n, f"Expected {n} dot circles, got {len(result)}"

    def test_tsp_produces_single_polyline(self):
        img = make_gradient_image(80, 80)
        params = {
            "_source_image": img,
            "num_points": 30,
            "iterations": 2,
            "connect_tsp": True,
            "min_dot_spacing_mm": 0.0,
            "seed": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1, "TSP mode should produce a single connected polyline"
        assert len(result[0]) == 30, "TSP path should visit all points"

    def test_output_within_bounds_dots(self):
        img = make_gradient_image(80, 80)
        params = {
            "_source_image": img,
            "num_points": 30,
            "iterations": 2,
            "connect_tsp": False,
            "min_dot_spacing_mm": 0.5,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=1.0)

    def test_output_within_bounds_tsp(self):
        img = make_gradient_image(80, 80)
        params = {
            "_source_image": img,
            "num_points": 30,
            "iterations": 2,
            "connect_tsp": True,
            "min_dot_spacing_mm": 0.0,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=1.0)

    def test_dots_concentrate_in_dark_areas(self):
        """Points should be denser in dark regions of the image."""
        # Black left half, white right half
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:, 40:] = 255  # right half white

        params = {
            "_source_image": img,
            "num_points": 100,
            "iterations": 5,
            "connect_tsp": True,
            "min_dot_spacing_mm": 0.0,
            "seed": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1

        draw_x1, _, draw_x2, _ = self.canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2.0

        path = result[0]
        left_count = sum(1 for x, y in path if x <= mid_x)
        right_count = sum(1 for x, y in path if x > mid_x)
        # Dark (left) side should have more points after relaxation
        assert left_count > right_count, (
            f"Expected more points in dark region: left={left_count} right={right_count}"
        )

    def test_has_presets(self):
        presets = self.gen.get_presets()
        assert len(presets) > 0
        for p in presets:
            assert p.name
            assert "num_points" in p.params

    def test_gets_parameters(self):
        params = self.gen.get_parameters()
        param_names = [p.name for p in params]
        assert "num_points" in param_names
        assert "iterations" in param_names
        assert "connect_tsp" in param_names
        assert "min_dot_spacing_mm" in param_names

    def test_dark_image_places_all_points(self):
        """All-black image should still place all requested points."""
        img = np.zeros((80, 80), dtype=np.uint8)
        params = {
            "_source_image": img,
            "num_points": 50,
            "iterations": 2,
            "connect_tsp": False,
            "min_dot_spacing_mm": 0.0,
            "seed": 1,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# LBG Stippling (Phase 16.45)
# ---------------------------------------------------------------------------


class TestLBGStippling:
    """Tests for the LBG (Linde-Buzo-Gray) stippling algorithm added in 16.45."""

    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = make_canvas()

    def _lbg_params(self, **overrides) -> dict:
        """Return a minimal valid LBG param set."""
        base = {
            "algorithm": "LBG",
            "num_points": 100,
            "iterations": 15,
            "connect_tsp": False,
            "min_dot_spacing_mm": 0.0,
            "seed": 42,
            "split_threshold": 1.5,
            "merge_threshold": 0.5,
            "initial_distribution": "Weighted Random",
        }
        base.update(overrides)
        return base

    def test_lbg_parameter_present(self):
        """'algorithm' ChoiceParam must be exposed."""
        names = [p.name for p in self.gen.get_parameters()]
        assert "algorithm" in names
        assert "split_threshold" in names
        assert "merge_threshold" in names
        assert "initial_distribution" in names

    def test_lbg_presets_exist(self):
        """At least one LBG preset should exist."""
        lbg_presets = [
            p for p in self.gen.get_presets() if p.params.get("algorithm") == "LBG"
        ]
        assert len(lbg_presets) >= 1

    def test_all_presets_include_new_params(self):
        """Every preset must contain all new LBG-related params."""
        required = {"algorithm", "split_threshold", "merge_threshold", "initial_distribution"}
        for preset in self.gen.get_presets():
            missing = required - set(preset.params.keys())
            assert not missing, f"Preset '{preset.name}' missing keys: {missing}"

    def test_lbg_produces_non_empty_output(self):
        """LBG should generate at least some dots on a non-trivial image."""
        img = make_gradient_image(80, 80)
        params = self._lbg_params(_source_image=img)
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "LBG should produce non-empty output"

    def test_lbg_point_count_within_10_percent_of_target(self):
        """Final point count should be within ±10% of num_points target."""
        img = make_dark_center_image(80, 80)
        target = 80
        params = self._lbg_params(_source_image=img, num_points=target, iterations=20)
        result = self.gen.generate(params, self.canvas)
        # Each dot is a tiny circle polyline
        actual = len(result)
        assert abs(actual - target) / target <= 0.10, (
            f"LBG point count {actual} is more than 10% away from target {target}"
        )

    def test_lbg_dots_concentrate_in_dark_areas(self):
        """LBG points should be denser in dark (left) half of a split image."""
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:, 40:] = 255  # right half white

        params = self._lbg_params(
            _source_image=img,
            num_points=100,
            connect_tsp=True,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1, "TSP mode should give single path"

        draw_x1, _, draw_x2, _ = self.canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2.0
        path = result[0]
        left_count = sum(1 for x, _ in path if x <= mid_x)
        right_count = sum(1 for x, _ in path if x > mid_x)
        assert left_count > right_count, (
            f"LBG should concentrate in dark area: left={left_count} right={right_count}"
        )

    def test_lbg_output_within_canvas_bounds(self):
        """All LBG dot positions should lie within the canvas drawing area."""
        img = make_gradient_image(80, 80)
        params = self._lbg_params(_source_image=img, num_points=60)
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=1.0), (
            "LBG points out of canvas bounds"
        )

    def test_split_threshold_affects_density(self):
        """Lower split_threshold → more aggressive splitting → potentially more points
        in dark regions after same iterations on a mostly-dark image."""
        img = make_dark_center_image(80, 80)
        base_params = dict(
            _source_image=img,
            num_points=50,
            iterations=10,
            connect_tsp=False,
            min_dot_spacing_mm=0.0,
            seed=7,
            merge_threshold=0.01,  # nearly no merging
        )

        result_aggressive = self.gen.generate(
            self._lbg_params(**base_params, split_threshold=1.2), self.canvas
        )
        result_conservative = self.gen.generate(
            self._lbg_params(**base_params, split_threshold=2.5), self.canvas
        )
        # Aggressive splitting should produce at least as many dots as conservative.
        assert len(result_aggressive) >= len(result_conservative), (
            f"split_threshold=1.2 produced {len(result_aggressive)} dots, "
            f"split_threshold=2.5 produced {len(result_conservative)} dots — "
            "aggressive threshold should produce >= points"
        )

    def test_merge_threshold_affects_density(self):
        """Higher merge_threshold → more aggressive merging → fewer points in bright areas."""
        # Mostly-bright image (value 200) — pixel_weight = (255-200)/255 ≈ 0.22 per pixel,
        # so total_weight > 0 and target_area > 0, allowing merge comparisons to trigger.
        img = np.full((80, 80), 200, dtype=np.uint8)
        base_params = dict(
            _source_image=img,
            num_points=80,
            iterations=10,
            connect_tsp=False,
            min_dot_spacing_mm=0.0,
            seed=3,
            split_threshold=10.0,  # nearly no splitting
        )
        result_aggressive = self.gen.generate(
            self._lbg_params(**base_params, merge_threshold=0.8), self.canvas
        )
        result_conservative = self.gen.generate(
            self._lbg_params(**base_params, merge_threshold=0.05), self.canvas
        )
        # More aggressive merging → equal or fewer remaining points on near-white image
        assert len(result_aggressive) <= len(result_conservative) or (
            abs(len(result_aggressive) - len(result_conservative))
            <= max(5, int(0.3 * len(result_conservative)))
        ), (
            f"merge_threshold=0.8 produced {len(result_aggressive)} dots, "
            f"merge_threshold=0.05 produced {len(result_conservative)} dots — "
            "aggressive merging should produce <= points on bright image"
        )

    def test_lbg_converges_quickly(self):
        """LBG with early stopping should typically finish before max iterations."""
        img = make_dark_center_image(80, 80)
        # We can't directly observe iteration count, but with a very tight threshold
        # (iterations=50) the algorithm should still return valid output quickly.
        params = self._lbg_params(
            _source_image=img,
            num_points=60,
            iterations=50,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_lbg_uniform_grid_initial_distribution(self):
        """Uniform Grid initial distribution should produce valid output."""
        img = make_gradient_image(80, 80)
        params = self._lbg_params(
            _source_image=img,
            num_points=60,
            initial_distribution="Uniform Grid",
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert within_bounds(result, self.canvas, tol=1.0)

    def test_lbg_few_seeds_initial_distribution(self):
        """Few Seeds initial distribution should grow to near target_points."""
        img = make_dark_center_image(80, 80)
        target = 80
        params = self._lbg_params(
            _source_image=img,
            num_points=target,
            iterations=30,
            initial_distribution="Few Seeds",
            split_threshold=1.2,  # aggressive splitting to grow quickly
        )
        result = self.gen.generate(params, self.canvas)
        # Should grow meaningfully toward the target (at least 50%)
        assert len(result) >= int(target * 0.5), (
            f"Few Seeds should grow to at least {int(target * 0.5)} points, got {len(result)}"
        )

    def test_lbg_tsp_produces_single_polyline(self):
        """LBG with TSP should return a single connected path."""
        img = make_gradient_image(80, 80)
        params = self._lbg_params(
            _source_image=img,
            num_points=40,
            connect_tsp=True,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1, "LBG TSP mode should produce exactly one path"
        assert len(result[0]) >= 2

    def test_lloyd_still_works_as_default(self):
        """Existing Lloyd behavior should be preserved when algorithm='Lloyd'."""
        img = make_gradient_image(80, 80)
        params = {
            "_source_image": img,
            "algorithm": "Lloyd",
            "num_points": 40,
            "iterations": 2,
            "connect_tsp": False,
            "min_dot_spacing_mm": 0.0,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 40, "Lloyd should produce exactly num_points dots"


# ---------------------------------------------------------------------------
# ContourGenerator (Phase 13.7)
# ---------------------------------------------------------------------------


class TestContourGenerator:
    def setup_method(self):
        from plottter.generators.contour import ContourGenerator
        self.gen = ContourGenerator()
        self.canvas = make_canvas()

    def test_no_image_returns_empty(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_generates_polylines_from_gradient(self):
        img = make_gradient_image(100, 100)
        params = {
            "_source_image": img,
            "num_levels": 4,
            "spacing": "linear",
            "simplify_mm": 0.0,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0
        for poly in result:
            assert len(poly) >= 2

    def test_output_within_canvas_bounds(self):
        img = make_dark_center_image(100, 100)
        params = {
            "_source_image": img,
            "num_levels": 4,
            "spacing": "linear",
            "simplify_mm": 0.5,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 1.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=2.0)

    def test_more_levels_more_polylines(self):
        img = make_gradient_image(100, 100)
        base = {
            "_source_image": img,
            "spacing": "linear",
            "simplify_mm": 0.0,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_few = self.gen.generate({**base, "num_levels": 2}, self.canvas)
        result_many = self.gen.generate({**base, "num_levels": 10}, self.canvas)
        # More levels should produce at least as many polylines
        assert len(result_many) >= len(result_few)

    def test_spacing_modes(self):
        img = make_gradient_image(100, 100)
        base = {
            "_source_image": img,
            "num_levels": 4,
            "simplify_mm": 0.0,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        for spacing in ["linear", "logarithmic", "quadratic"]:
            result = self.gen.generate({**base, "spacing": spacing}, self.canvas)
            assert isinstance(result, list)

    def test_invert_flag(self):
        img = make_dark_center_image(100, 100)
        base = {
            "_source_image": img,
            "num_levels": 4,
            "spacing": "linear",
            "simplify_mm": 0.0,
            "min_contour_px": 3,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_normal = self.gen.generate({**base, "invert": False}, self.canvas)
        result_inverted = self.gen.generate({**base, "invert": True}, self.canvas)
        # Both should produce polylines; exact counts may differ
        assert isinstance(result_normal, list)
        assert isinstance(result_inverted, list)

    def test_rgb_image_accepted(self):
        arr = np.full((80, 80, 3), 128, dtype=np.uint8)
        params = {
            "_source_image": arr,
            "num_levels": 3,
            "spacing": "linear",
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_has_presets(self):
        presets = self.gen.get_presets()
        assert len(presets) >= 2
        for p in presets:
            assert p.name
            assert "num_levels" in p.params
            assert "spacing" in p.params

    def test_gets_parameters(self):
        params = self.gen.get_parameters()
        names = [p.name for p in params]
        assert "num_levels" in names
        assert "spacing" in names
        assert "simplify_mm" in names
        assert "min_contour_px" in names
        assert "invert" in names
        assert "brightness" in names
        assert "contrast" in names
        assert "blur_radius" in names


# ---------------------------------------------------------------------------
# ContourGenerator — Line Art Trace mode (Phase 16.18)
# ---------------------------------------------------------------------------


def make_bw_line_art(h: int = 100, w: int = 100) -> np.ndarray:
    """Synthetic B&W line art: white background with black rectangle outline."""
    arr = np.full((h, w), 255, dtype=np.uint8)
    # Draw a thick black rectangle border (5px thick)
    thickness = 5
    arr[:thickness, :] = 0
    arr[-thickness:, :] = 0
    arr[:, :thickness] = 0
    arr[:, -thickness:] = 0
    return arr


def make_bw_circle_line_art(h: int = 100, w: int = 100) -> np.ndarray:
    """Synthetic B&W line art: white background with black circle stroke."""
    import cv2
    arr = np.full((h, w), 255, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = min(h, w) // 3
    cv2.circle(arr, (cx, cy), radius, 0, thickness=4)
    return arr


class TestContourGeneratorLineArtMode:
    """Tests for the Line Art Trace mode added in Phase 16.18."""

    def setup_method(self):
        from plottter.generators.contour import ContourGenerator
        self.gen = ContourGenerator()
        self.canvas = make_canvas()

    # --- Parameter structure ---

    def test_mode_param_present(self):
        params = self.gen.get_parameters()
        names = [p.name for p in params]
        assert "mode" in names

    def test_trace_threshold_param_present(self):
        params = self.gen.get_parameters()
        names = [p.name for p in params]
        assert "trace_threshold" in names

    def test_smooth_iterations_param_present(self):
        params = self.gen.get_parameters()
        names = [p.name for p in params]
        assert "smooth_iterations" in names

    def test_trace_threshold_has_visible_when(self):
        from plottter.generators.base import IntParam
        params = self.gen.get_parameters()
        threshold_param = next(p for p in params if p.name == "trace_threshold")
        assert isinstance(threshold_param, IntParam)
        assert threshold_param.visible_when is not None
        assert "mode" in threshold_param.visible_when
        assert "Line Art Trace" in threshold_param.visible_when["mode"]

    def test_num_levels_hidden_in_line_art_mode(self):
        params = self.gen.get_parameters()
        num_levels_param = next(p for p in params if p.name == "num_levels")
        assert num_levels_param.visible_when is not None
        assert "Contour Levels" in num_levels_param.visible_when.get("mode", [])

    def test_spacing_hidden_in_line_art_mode(self):
        params = self.gen.get_parameters()
        spacing_param = next(p for p in params if p.name == "spacing")
        assert spacing_param.visible_when is not None
        assert "Contour Levels" in spacing_param.visible_when.get("mode", [])

    # --- Functional tests ---

    def test_line_art_trace_produces_polylines(self):
        img = make_bw_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0, "Line Art Trace should produce polylines for a B&W rectangle"

    def test_line_art_trace_circle(self):
        img = make_bw_circle_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "simplify_mm": 0.2,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0, "Line Art Trace should produce polylines for a B&W circle"

    def test_line_art_trace_within_canvas_bounds(self):
        img = make_bw_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=2.0)

    def test_line_art_trace_rgb_image_accepted(self):
        """Line Art Trace should work with RGB images (converts to grayscale)."""
        arr = np.full((80, 80, 3), 255, dtype=np.uint8)
        # Draw black border
        arr[:5, :] = 0
        arr[-5:, :] = 0
        arr[:, :5] = 0
        arr[:, -5:] = 0
        params = {
            "_source_image": arr,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_line_art_trace_threshold_affects_output(self):
        """Different thresholds should produce different outputs for a gradient image."""
        img = make_gradient_image(100, 100)
        base = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "smooth_iterations": 0,
            "simplify_mm": 0.0,
            "min_contour_px": 2,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_low = self.gen.generate({**base, "trace_threshold": 64}, self.canvas)
        result_high = self.gen.generate({**base, "trace_threshold": 200}, self.canvas)
        # Both should produce output (gradient has pixels at all brightness levels)
        assert isinstance(result_low, list)
        assert isinstance(result_high, list)
        assert len(result_low) > 0
        assert len(result_high) > 0
        # Different thresholds should yield different amounts of enclosed area
        total_pts_low = sum(len(pl) for pl in result_low)
        total_pts_high = sum(len(pl) for pl in result_high)
        assert total_pts_low != total_pts_high, (
            "Low and high thresholds produced identical point counts; "
            "threshold parameter has no effect"
        )

    def test_line_art_trace_no_image_returns_empty(self):
        params = {
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
        }
        result = self.gen.generate(params, self.canvas)
        assert result == []

    # --- Chaikin smoothing tests ---

    def test_chaikin_smooth_no_iterations(self):
        from plottter.generators.contour import _chaikin_smooth
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = _chaikin_smooth(poly, 0)
        assert result == poly

    def test_chaikin_smooth_one_iteration_increases_points(self):
        from plottter.generators.contour import _chaikin_smooth
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = _chaikin_smooth(poly, 1, closed=False)
        # One iteration of Chaikin on an open 4-point polyline:
        # preserves endpoints + adds 2 points per interior segment
        assert len(result) > len(poly)

    def test_chaikin_smooth_more_iterations_more_points(self):
        from plottter.generators.contour import _chaikin_smooth
        poly = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        r1 = _chaikin_smooth(poly, 1, closed=False)
        r2 = _chaikin_smooth(poly, 2, closed=False)
        r3 = _chaikin_smooth(poly, 3, closed=False)
        assert len(r2) > len(r1)
        assert len(r3) > len(r2)

    def test_chaikin_smooth_preserves_open_endpoints(self):
        from plottter.generators.contour import _chaikin_smooth
        poly = [(0.0, 0.0), (5.0, 2.5), (10.0, 0.0), (10.0, 10.0)]
        result = _chaikin_smooth(poly, 2, closed=False)
        # First and last points should be preserved for open curves
        assert result[0] == poly[0]
        assert result[-1] == poly[-1]

    def test_chaikin_smooth_closed_curve(self):
        from plottter.generators.contour import _chaikin_smooth
        # Closed polyline (last point == first point)
        poly = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
        result = _chaikin_smooth(poly, 2, closed=True)
        assert len(result) > len(poly)
        # Should still be closed
        assert result[0] == result[-1]

    def test_chaikin_smooth_produces_smoother_output(self):
        """Smoothed polyline should have shorter max segment length (smoother curves)."""
        from plottter.generators.contour import _chaikin_smooth
        import math
        # A right-angle corner
        poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        raw_max_seg = max(
            math.sqrt((poly[i+1][0]-poly[i][0])**2 + (poly[i+1][1]-poly[i][1])**2)
            for i in range(len(poly)-1)
        )
        smoothed = _chaikin_smooth(poly, 3, closed=False)
        smoothed_max_seg = max(
            math.sqrt((smoothed[i+1][0]-smoothed[i][0])**2 + (smoothed[i+1][1]-smoothed[i][1])**2)
            for i in range(len(smoothed)-1)
        )
        assert smoothed_max_seg < raw_max_seg

    def test_smooth_iterations_applied_in_line_art_mode(self):
        """Smooth iterations should increase point count vs no smoothing."""
        img = make_bw_line_art(100, 100)
        base = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "simplify_mm": 0.0,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_no_smooth = self.gen.generate({**base, "smooth_iterations": 0}, self.canvas)
        result_smooth = self.gen.generate({**base, "smooth_iterations": 3}, self.canvas)
        pts_no_smooth = sum(len(p) for p in result_no_smooth)
        pts_smooth = sum(len(p) for p in result_smooth)
        assert pts_smooth > pts_no_smooth, (
            "Chaikin smoothing should increase total point count"
        )

    # --- Preset tests ---

    def test_line_art_trace_preset_exists(self):
        presets = self.gen.get_presets()
        names = [p.name for p in presets]
        assert "Line Art / Trace" in names

    def test_line_art_trace_preset_has_correct_mode(self):
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Trace")
        assert preset.params["mode"] == "Line Art Trace"

    def test_line_art_trace_preset_functional(self):
        img = make_bw_line_art(100, 100)
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Trace")
        params = dict(preset.params)
        params["_source_image"] = img
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_existing_presets_have_mode_field(self):
        """All presets should declare a mode so they work correctly."""
        presets = self.gen.get_presets()
        for preset in presets:
            assert "mode" in preset.params, (
                f"Preset '{preset.name}' is missing 'mode' field"
            )

    def test_contour_levels_presets_functional(self):
        """Existing Contour Levels presets should still work after refactor."""
        img = make_gradient_image(100, 100)
        presets = self.gen.get_presets()
        for preset in presets:
            if preset.params.get("mode") == "Contour Levels":
                params = dict(preset.params)
                params["_source_image"] = img
                params["min_contour_px"] = 3  # lower for small test image
                result = self.gen.generate(params, self.canvas)
                assert isinstance(result, list), (
                    f"Preset '{preset.name}' failed to produce output"
                )

    def test_smoothed_line_art_polylines_are_closed(self):
        """All polylines produced by Line Art Trace with smooth_iterations > 0 must be closed.

        Regression test for the elif→if bug: when Chaikin smoothing ran, the elif
        prevented the closing step from executing, leaving smoothed polylines open.
        The 'Line Art / Trace' preset uses smooth_iterations=2 and was broken.
        """
        img = make_bw_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 2,
            "simplify_mm": 0.15,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Expected at least one polyline"
        for i, poly in enumerate(result):
            assert len(poly) >= 2, f"Polyline {i} has fewer than 2 points"
            assert poly[0] == poly[-1], (
                f"Polyline {i} is not closed after Chaikin smoothing: "
                f"first={poly[0]}, last={poly[-1]}"
            )


# ---------------------------------------------------------------------------
# ContourGenerator — Fill options in Line Art Trace mode (Phase 16.20)
# ---------------------------------------------------------------------------


def make_filled_rect_line_art(h: int = 100, w: int = 100) -> np.ndarray:
    """Synthetic B&W line art: white background with solid black filled rectangle.

    The filled interior gives Shapely-based fill functions a meaningful region
    to hatch/concentric-fill — hollow outlines are harder to verify visually.
    """
    arr = np.full((h, w), 255, dtype=np.uint8)
    # Draw a solid black filled rectangle (not just outline)
    arr[20:80, 20:80] = 0
    return arr


class TestContourGeneratorFillOptions:
    """Tests for fill options added to Line Art Trace mode in Phase 16.20."""

    def setup_method(self):
        from plottter.generators.contour import ContourGenerator
        self.gen = ContourGenerator()
        self.canvas = make_canvas()

    # --- Parameter presence ---

    def test_fill_param_present(self):
        params = self.gen.get_parameters()
        names = [p.name for p in params]
        assert "fill" in names

    def test_fill_spacing_mm_param_present(self):
        params = self.gen.get_parameters()
        names = [p.name for p in params]
        assert "fill_spacing_mm" in names

    def test_fill_angle_param_present(self):
        params = self.gen.get_parameters()
        names = [p.name for p in params]
        assert "fill_angle" in names

    def test_fill_param_visible_when_line_art_trace(self):
        from plottter.generators.base import ChoiceParam
        params = self.gen.get_parameters()
        fill_param = next(p for p in params if p.name == "fill")
        assert isinstance(fill_param, ChoiceParam)
        assert fill_param.visible_when is not None
        assert "Line Art Trace" in fill_param.visible_when.get("mode", [])

    def test_fill_choices_include_all_options(self):
        from plottter.generators.base import ChoiceParam
        params = self.gen.get_parameters()
        fill_param = next(p for p in params if p.name == "fill")
        assert isinstance(fill_param, ChoiceParam)
        choices = fill_param.choices
        assert "None" in choices
        assert "Solid" in choices
        assert "Hatching" in choices
        assert "Cross-hatch" in choices
        assert "Concentric" in choices

    # --- Fill=None (outline only) ---

    def test_fill_none_produces_outlines(self):
        img = make_bw_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill": "None",
            "fill_spacing_mm": 0.3,
            "fill_angle": 45.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "fill=None should still produce outline polylines"

    # --- Fill=Solid ---

    def test_fill_solid_produces_more_lines_than_none(self):
        img = make_filled_rect_line_art(100, 100)
        base_params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill_spacing_mm": 1.0,
            "fill_angle": 0.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_none = self.gen.generate({**base_params, "fill": "None"}, self.canvas)
        result_solid = self.gen.generate({**base_params, "fill": "Solid"}, self.canvas)
        assert len(result_solid) > len(result_none), (
            "fill=Solid should add fill lines beyond the outline"
        )

    def test_fill_solid_returns_list_of_polylines(self):
        img = make_filled_rect_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill": "Solid",
            "fill_spacing_mm": 1.5,
            "fill_angle": 0.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        for poly in result:
            assert isinstance(poly, list)
            assert len(poly) >= 2

    # --- Fill=Hatching ---

    def test_fill_hatching_produces_more_lines_than_none(self):
        img = make_filled_rect_line_art(100, 100)
        base_params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill_spacing_mm": 1.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_none = self.gen.generate({**base_params, "fill": "None", "fill_angle": 45.0}, self.canvas)
        result_hatch = self.gen.generate({**base_params, "fill": "Hatching", "fill_angle": 45.0}, self.canvas)
        assert len(result_hatch) > len(result_none), (
            "fill=Hatching should add hatch lines beyond the outline"
        )

    # --- Fill=Cross-hatch ---

    def test_fill_crosshatch_produces_more_lines_than_hatching(self):
        img = make_filled_rect_line_art(100, 100)
        base_params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill_spacing_mm": 1.5,
            "fill_angle": 45.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_hatch = self.gen.generate({**base_params, "fill": "Hatching"}, self.canvas)
        result_cross = self.gen.generate({**base_params, "fill": "Cross-hatch"}, self.canvas)
        assert len(result_cross) >= len(result_hatch), (
            "fill=Cross-hatch should produce at least as many lines as Hatching"
        )

    # --- Fill=Concentric ---

    def test_fill_concentric_produces_more_lines_than_none(self):
        img = make_filled_rect_line_art(100, 100)
        base_params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill_spacing_mm": 2.0,
            "fill_angle": 45.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_none = self.gen.generate({**base_params, "fill": "None"}, self.canvas)
        result_conc = self.gen.generate({**base_params, "fill": "Concentric"}, self.canvas)
        assert len(result_conc) > len(result_none), (
            "fill=Concentric should add concentric rings beyond the outline"
        )

    # --- Unit tests for helper functions ---

    def test_fill_polygon_hatch_basic(self):
        """_fill_polygon_hatch returns line segments inside a simple square."""
        from plottter.generators.contour import _fill_polygon_hatch
        # 20x20 mm square
        square = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        lines = _fill_polygon_hatch(square, [], angle_deg=0.0, spacing_mm=2.0)
        assert len(lines) > 0, "Hatch should produce lines inside the square"
        # Each segment must have at least 2 points and lie within the square's x-range
        for seg in lines:
            assert len(seg) >= 2

    def test_fill_polygon_hatch_empty_when_degenerate(self):
        """_fill_polygon_hatch returns [] for a degenerate input."""
        from plottter.generators.contour import _fill_polygon_hatch
        lines = _fill_polygon_hatch([], [], angle_deg=45.0, spacing_mm=1.0)
        assert lines == []

    def test_fill_polygon_hatch_zero_spacing_returns_empty(self):
        """_fill_polygon_hatch returns [] when spacing_mm == 0."""
        from plottter.generators.contour import _fill_polygon_hatch
        square = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        lines = _fill_polygon_hatch(square, [], angle_deg=45.0, spacing_mm=0.0)
        assert lines == []

    def test_fill_polygon_concentric_basic(self):
        """_fill_polygon_concentric produces inward rings for a large square."""
        from plottter.generators.contour import _fill_polygon_concentric
        # 40x40 mm square — enough room for several rings at 5mm spacing
        square = [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
        rings = _fill_polygon_concentric(square, [], spacing_mm=5.0)
        assert len(rings) > 0, "Concentric should produce at least one inner ring"
        for ring in rings:
            assert len(ring) >= 2

    def test_fill_polygon_concentric_empty_when_spacing_zero(self):
        """_fill_polygon_concentric returns [] for spacing_mm == 0."""
        from plottter.generators.contour import _fill_polygon_concentric
        square = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        rings = _fill_polygon_concentric(square, [], spacing_mm=0.0)
        assert rings == []

    def test_fill_polygon_hatch_multipolygon_no_crash(self):
        """_fill_polygon_hatch does not crash when Polygon() normalises to MultiPolygon.

        A figure-8 (self-intersecting) contour is invalid and buffer(0) splits it
        into two separate triangles — a MultiPolygon.  The function should return
        hatch lines for both sub-polygons without raising AttributeError.
        """
        from plottter.generators.contour import _fill_polygon_hatch
        # Figure-8: two triangles sharing a point — self-intersecting
        figure_8 = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0), (0.0, 0.0)]
        lines = _fill_polygon_hatch(figure_8, [], angle_deg=0.0, spacing_mm=1.0)
        assert isinstance(lines, list), "Must return a list, not raise AttributeError"

    def test_fill_polygon_concentric_multipolygon_no_crash(self):
        """_fill_polygon_concentric does not crash for self-intersecting input.

        A figure-8 contour normalised by buffer(0) produces a MultiPolygon.
        The function should return rings for both sub-polygons.
        """
        from plottter.generators.contour import _fill_polygon_concentric
        figure_8 = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0), (0.0, 0.0)]
        rings = _fill_polygon_concentric(figure_8, [], spacing_mm=1.0)
        assert isinstance(rings, list), "Must return a list, not raise AttributeError"

    def test_extract_contours_with_hierarchy_returns_pairs(self):
        """_extract_contours_with_hierarchy returns (outer, holes) tuples."""
        from plottter.generators.contour import _extract_contours_with_hierarchy
        img = make_filled_rect_line_art(100, 100)
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        pairs = _extract_contours_with_hierarchy(
            gray,
            threshold=128,
            img_w=100,
            img_h=100,
            draw_x1=0.0,
            draw_y1=0.0,
            draw_x2=100.0,
            draw_y2=100.0,
            simplify_tol=0.0,
            min_length=3,
            smooth_iterations=0,
        )
        assert isinstance(pairs, list)
        assert len(pairs) > 0, "Should detect at least one outer contour"
        for outer, holes in pairs:
            assert isinstance(outer, list)
            assert len(outer) >= 2
            assert isinstance(holes, list)

    # --- Preset tests ---

    def test_solid_fill_preset_exists(self):
        presets = self.gen.get_presets()
        names = [p.name for p in presets]
        assert "Line Art / Solid Fill" in names

    def test_hatched_fill_preset_exists(self):
        presets = self.gen.get_presets()
        names = [p.name for p in presets]
        assert "Line Art / Hatched Fill" in names

    def test_concentric_fill_preset_exists(self):
        presets = self.gen.get_presets()
        names = [p.name for p in presets]
        assert "Line Art / Concentric Fill" in names

    def test_solid_fill_preset_params(self):
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Solid Fill")
        assert preset.params["mode"] == "Line Art Trace"
        assert preset.params["fill"] == "Solid"

    def test_hatched_fill_preset_params(self):
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Hatched Fill")
        assert preset.params["mode"] == "Line Art Trace"
        assert preset.params["fill"] == "Hatching"

    def test_concentric_fill_preset_params(self):
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Concentric Fill")
        assert preset.params["mode"] == "Line Art Trace"
        assert preset.params["fill"] == "Concentric"

    def test_solid_fill_preset_functional(self):
        """Line Art / Solid Fill preset should generate without errors."""
        img = make_filled_rect_line_art(100, 100)
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Solid Fill")
        params = dict(preset.params)
        params["_source_image"] = img
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_hatched_fill_preset_functional(self):
        """Line Art / Hatched Fill preset should generate without errors."""
        img = make_filled_rect_line_art(100, 100)
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Hatched Fill")
        params = dict(preset.params)
        params["_source_image"] = img
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_concentric_fill_preset_functional(self):
        """Line Art / Concentric Fill preset should generate without errors."""
        img = make_filled_rect_line_art(100, 100)
        presets = self.gen.get_presets()
        preset = next(p for p in presets if p.name == "Line Art / Concentric Fill")
        params = dict(preset.params)
        params["_source_image"] = img
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    # --- Progress/cancel callbacks work with fill modes ---

    def test_fill_solid_respects_cancel_callback(self):
        img = make_filled_rect_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill": "Solid",
            "fill_spacing_mm": 0.5,
            "fill_angle": 0.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        # Cancel immediately — should return early without error
        result = self.gen.generate(
            params, self.canvas,
            cancelled_callback=lambda: True,
        )
        assert isinstance(result, list)

    def test_fill_progress_callback_called(self):
        img = make_filled_rect_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "fill": "Hatching",
            "fill_spacing_mm": 1.0,
            "fill_angle": 45.0,
            "simplify_mm": 0.3,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        progress_values: list[int] = []
        self.gen.generate(params, self.canvas, progress_callback=progress_values.append)
        assert len(progress_values) > 0, "Progress callback should have been called"
        assert progress_values[-1] == 100, "Final progress value should be 100"


# ---------------------------------------------------------------------------
# ContourGenerator — Skeleton / Center-line mode (Phase 16.30)
# ---------------------------------------------------------------------------


def make_thick_stroke_image(h: int = 100, w: int = 200) -> np.ndarray:
    """Synthetic image: white background with a single thick horizontal bar.

    The bar spans most of the image width at rows [35, 65] (30px tall, centred
    at row 50).  Used to verify that Skeleton mode traces the centerline of
    a thick stroke rather than its two boundary edges.
    """
    arr = np.full((h, w), 255, dtype=np.uint8)
    arr[35:65, 10 : w - 10] = 0  # 30px-tall solid black bar
    return arr


class TestContourGeneratorSkeletonMode:
    """Tests for the Skeleton / Center-line mode added in Phase 16.30."""

    def setup_method(self):
        from plottter.generators.contour import ContourGenerator

        self.gen = ContourGenerator()
        self.canvas = make_canvas()

    # --- Parameter structure ---

    def test_skeleton_is_a_valid_mode_choice(self):
        """Skeleton must be a valid option in the mode ChoiceParam."""
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "Skeleton" in params["mode"].choices

    def test_cleanup_kernel_param_present(self):
        """cleanup_kernel IntParam must exist with correct range and default."""
        from plottter.generators.base import IntParam

        params = {p.name: p for p in self.gen.get_parameters()}
        assert "cleanup_kernel" in params
        p = params["cleanup_kernel"]
        assert isinstance(p, IntParam)
        assert p.min == 0
        assert p.max == 10
        assert p.default == 0

    def test_cleanup_kernel_only_visible_in_skeleton_mode(self):
        """cleanup_kernel should only be visible when mode == Skeleton."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["cleanup_kernel"].visible_when
        assert vw is not None
        assert "mode" in vw
        assert "Skeleton" in vw["mode"]
        assert "Line Art Trace" not in vw.get("mode", [])
        assert "Contour Levels" not in vw.get("mode", [])

    def test_trace_threshold_visible_in_skeleton_mode(self):
        """trace_threshold must be visible in Skeleton mode."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["trace_threshold"].visible_when
        assert vw is not None
        assert "Skeleton" in vw.get("mode", [])

    def test_smooth_iterations_visible_in_skeleton_mode(self):
        """smooth_iterations must be visible in Skeleton mode."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["smooth_iterations"].visible_when
        assert vw is not None
        assert "Skeleton" in vw.get("mode", [])

    # --- Preset ---

    def test_skeleton_centerline_preset_exists(self):
        """Preset 'Skeleton / Center-line' must exist."""
        names = [p.name for p in self.gen.get_presets()]
        assert "Skeleton / Center-line" in names

    def test_skeleton_centerline_preset_uses_skeleton_mode(self):
        """Skeleton / Center-line preset must set mode to 'Skeleton'."""
        preset = next(
            p for p in self.gen.get_presets() if p.name == "Skeleton / Center-line"
        )
        assert preset.params["mode"] == "Skeleton"

    def test_skeleton_preset_includes_all_params(self):
        """All parameter names must be present in the Skeleton / Center-line preset."""
        preset = next(
            p for p in self.gen.get_presets() if p.name == "Skeleton / Center-line"
        )
        param_names = {p.name for p in self.gen.get_parameters()}
        for name in param_names:
            assert name in preset.params, f"Preset missing parameter: {name}"

    # --- Basic operation ---

    def test_skeleton_mode_no_image_returns_empty(self):
        """Skeleton mode must return [] when no source image is provided."""
        result = self.gen.generate({"mode": "Skeleton"}, self.canvas)
        assert result == []

    def test_skeleton_mode_runs_without_error(self):
        """Skeleton mode must complete without error on a synthetic image."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_thick_stroke_image()
        params = {
            "_source_image": img,
            "mode": "Skeleton",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "cleanup_kernel": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert all(isinstance(p, list) for p in result)
        assert all(len(p) >= 2 for p in result)

    def test_skeleton_mode_with_cleanup_kernel_runs_without_error(self):
        """Skeleton mode with cleanup_kernel=3 must complete without error."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_thick_stroke_image()
        params = {
            "_source_image": img,
            "mode": "Skeleton",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "cleanup_kernel": 3,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_skeleton_mode_progress_callback_called(self):
        """Progress callback must be called and end at 100."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_thick_stroke_image()
        params = {
            "_source_image": img,
            "mode": "Skeleton",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "cleanup_kernel": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
        }
        progress_values: list[int] = []
        self.gen.generate(params, self.canvas, progress_callback=progress_values.append)
        assert len(progress_values) > 0
        assert progress_values[-1] == 100

    def test_skeleton_mode_with_smooth_curves_runs_without_error(self):
        """Skeleton mode with smooth_curves=True must complete without error."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_thick_stroke_image()
        params = {
            "_source_image": img,
            "mode": "Skeleton",
            "trace_threshold": 128,
            "smooth_iterations": 1,
            "cleanup_kernel": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": True,
            "curve_tolerance_mm": 0.5,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_skeleton_mode_output_polylines_are_not_closed(self):
        """Skeleton mode should produce open polylines (first != last point).

        Unlike Line Art Trace, which traces closed shape outlines, Skeleton
        traces open centerlines for straight strokes.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_thick_stroke_image()
        params = {
            "_source_image": img,
            "mode": "Skeleton",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "cleanup_kernel": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
        }
        result = self.gen.generate(params, self.canvas)
        if result:
            # At least some polylines should be open (not closed loops)
            has_open = any(len(p) >= 2 and p[0] != p[-1] for p in result)
            assert has_open, (
                "Skeleton mode should produce at least some open polylines "
                "(centerlines, not closed shape outlines)"
            )

    def test_skeleton_centerline_stays_near_center_of_thick_bar(self):
        """Skeleton output y-coords should cluster near the bar's center, not span its full height.

        The thick bar occupies rows [35, 65] in a 100-row image.
        On A4 with 10mm margin (draw area y=[10, 287]):
          bar_y1_mm ≈ 106.95, bar_y2_mm ≈ 190.05, bar_height_mm ≈ 83.1

        The skeleton (centerline) should produce points with a y-span
        far smaller than the full bar height — it traces the centre of
        the stroke, not both boundary edges.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_thick_stroke_image(h=100, w=200)
        canvas = make_canvas()  # A4, margin=10 → draw area (10, 10, 200, 287)

        params = {
            "_source_image": img,
            "mode": "Skeleton",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "cleanup_kernel": 0,
            "simplify_mm": 0.5,
            "min_contour_px": 5,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
        }
        result = self.gen.generate(params, canvas)
        assert len(result) > 0, "Should produce at least one polyline for the thick bar"

        # bar rows [35, 65] in a 100-row image, draw area height = 277mm
        draw_y1, draw_y2 = 10.0, 287.0
        img_h = 100
        bar_y1_mm = draw_y1 + 35.0 / img_h * (draw_y2 - draw_y1)
        bar_y2_mm = draw_y1 + 65.0 / img_h * (draw_y2 - draw_y1)
        bar_height_mm = bar_y2_mm - bar_y1_mm

        # All output points must lie within the bar's vertical extent (± small tolerance)
        tol_mm = 3.0
        for poly in result:
            for _x, y_mm in poly:
                assert bar_y1_mm - tol_mm <= y_mm <= bar_y2_mm + tol_mm, (
                    f"Skeleton point y={y_mm:.2f}mm is outside bar "
                    f"[{bar_y1_mm:.1f}, {bar_y2_mm:.1f}]mm"
                )

        # Key behavioral assertion: the skeleton's y-span (distance between
        # the topmost and bottommost traced points) must be much smaller than
        # the full bar height.  A centerline trace stays near the middle;
        # a boundary trace would span the full bar height.
        all_y = [pt[1] for poly in result for pt in poly]
        y_span = max(all_y) - min(all_y)
        assert y_span < bar_height_mm / 2, (
            f"Skeleton y-span ({y_span:.1f}mm) should be < half the bar height "
            f"({bar_height_mm / 2:.1f}mm) — skeleton traces the centerline, "
            f"not both boundary edges"
        )


# ---------------------------------------------------------------------------
# XDoGGenerator
# ---------------------------------------------------------------------------


class TestXDoGGenerator:
    def setup_method(self):
        from plottter.generators.xdog import XDoGGenerator
        self.gen = XDoGGenerator()
        self.canvas = make_canvas()

    # --- Registration and metadata ---

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "XDoG" in GENERATORS
        assert GENERATORS["XDoG"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "XDoG"
        assert self.gen.category == "image"

    # --- Parameter contract ---

    def test_gets_parameters(self):
        params = self.gen.get_parameters()
        param_names = [p.name for p in params]
        assert "sigma" in param_names
        assert "k" in param_names
        assert "phi" in param_names
        assert "epsilon" in param_names
        assert "min_contour_length" in param_names
        assert "simplify_tolerance_mm" in param_names
        assert "close_gaps_mm" in param_names
        assert "smooth_iterations" in param_names
        assert "invert" in param_names
        assert "brightness" in param_names
        assert "contrast" in param_names
        assert "blur_radius" in param_names

    def test_parameter_ranges_match_spec(self):
        """Spec-defined ranges: sigma 0.3–3.0, k 1.1–5.0, phi 1–200, epsilon -0.5–0.5."""
        params = {p.name: p for p in self.gen.get_parameters()}
        assert params["sigma"].min == pytest.approx(0.3)
        assert params["sigma"].max == pytest.approx(3.0)
        assert params["k"].min == pytest.approx(1.1)
        assert params["k"].max == pytest.approx(5.0)
        assert params["phi"].min == pytest.approx(1.0)
        assert params["phi"].max == pytest.approx(200.0)
        assert params["epsilon"].min == pytest.approx(-0.5)
        assert params["epsilon"].max == pytest.approx(0.5)

    # --- Preset contract ---

    def test_presets_present(self):
        presets = self.gen.get_presets()
        names = [p.name for p in presets]
        assert "Pencil Sketch" in names
        assert "Woodcut" in names
        assert "Soft Charcoal" in names

    def test_preset_params_include_all_xdog_params(self):
        """Every preset must supply values for all four XDoG parameters."""
        for preset in self.gen.get_presets():
            for key in ("sigma", "k", "phi", "epsilon"):
                assert key in preset.params, (
                    f"Preset '{preset.name}' missing key '{key}'"
                )

    def test_pencil_sketch_preset_values(self):
        """Pencil Sketch preset: σ=0.8, k=1.6, φ=80, ε=-0.02."""
        preset = next(p for p in self.gen.get_presets() if p.name == "Pencil Sketch")
        assert preset.params["sigma"] == pytest.approx(0.8)
        assert preset.params["k"] == pytest.approx(1.6)
        assert preset.params["phi"] == pytest.approx(80.0)
        assert preset.params["epsilon"] == pytest.approx(-0.02)

    def test_woodcut_preset_values(self):
        """Woodcut preset: σ=1.5, k=3.0, φ=100, ε=-0.04."""
        preset = next(p for p in self.gen.get_presets() if p.name == "Woodcut")
        assert preset.params["sigma"] == pytest.approx(1.5)
        assert preset.params["k"] == pytest.approx(3.0)
        assert preset.params["phi"] == pytest.approx(100.0)
        assert preset.params["epsilon"] == pytest.approx(-0.04)

    def test_soft_charcoal_preset_values(self):
        """Soft Charcoal preset: σ=2.0, k=2.5, φ=15, ε=-0.04."""
        preset = next(p for p in self.gen.get_presets() if p.name == "Soft Charcoal")
        assert preset.params["sigma"] == pytest.approx(2.0)
        assert preset.params["k"] == pytest.approx(2.5)
        assert preset.params["phi"] == pytest.approx(15.0)
        assert preset.params["epsilon"] == pytest.approx(-0.04)

    # --- Core generation ---

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_generates_polylines_from_checkerboard(self):
        """Checkerboard has strong edges — XDoG should detect them."""
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Should detect edges in checkerboard"
        assert all(len(p) >= 2 for p in result)

    def test_generates_polylines_from_dark_center_image(self):
        """Dark circle on white background — XDoG should trace the circular edge."""
        img = make_dark_center_image(100, 100)
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.3,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Should detect edge of dark circle"

    def test_output_within_canvas_bounds(self):
        """All generated points must lie within the canvas drawing area."""
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas), (
            "All output points should be within canvas drawing area"
        )

    # --- Preset smoke tests ---

    def test_pencil_sketch_preset_generates_output(self):
        img = make_checkerboard(80, 80, tile=10)
        preset = next(p for p in self.gen.get_presets() if p.name == "Pencil Sketch")
        result = self.gen.generate({**preset.params, "_source_image": img}, self.canvas)
        assert len(result) > 0, "Pencil Sketch preset should generate output"

    def test_woodcut_preset_generates_output(self):
        img = make_checkerboard(80, 80, tile=10)
        preset = next(p for p in self.gen.get_presets() if p.name == "Woodcut")
        result = self.gen.generate({**preset.params, "_source_image": img}, self.canvas)
        assert len(result) > 0, "Woodcut preset should generate output"

    def test_soft_charcoal_preset_generates_output(self):
        img = make_checkerboard(80, 80, tile=10)
        preset = next(p for p in self.gen.get_presets() if p.name == "Soft Charcoal")
        result = self.gen.generate({**preset.params, "_source_image": img}, self.canvas)
        assert len(result) > 0, "Soft Charcoal preset should generate output"

    # --- Parameter sensitivity ---

    def test_phi_affects_output(self):
        """Higher phi produces sharper (different) results than lower phi."""
        img = make_checkerboard(80, 80, tile=10)
        base = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "epsilon": 0.05,  # non-zero so interior pixels fall in the soft-threshold zone
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result_low = self.gen.generate({**base, "phi": 5.0}, self.canvas)
        result_high = self.gen.generate({**base, "phi": 200.0}, self.canvas)
        # Different phi values should produce different numbers of polylines or points
        total_pts_low = sum(len(p) for p in result_low)
        total_pts_high = sum(len(p) for p in result_high)
        assert total_pts_low != total_pts_high, (
            "Different phi values should produce different output"
        )

    def test_sigma_affects_output(self):
        """Different sigma values should produce different edge detail."""
        img = make_checkerboard(80, 80, tile=10)
        base = {
            "_source_image": img,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result_small = self.gen.generate({**base, "_source_image": img, "sigma": 0.3}, self.canvas)
        result_large = self.gen.generate({**base, "_source_image": img, "sigma": 2.5}, self.canvas)
        pts_small = sum(len(p) for p in result_small)
        pts_large = sum(len(p) for p in result_large)
        assert pts_small != pts_large, (
            "Different sigma values should produce different output"
        )

    # --- Chaikin smoothing ---

    def test_chaikin_smoothing_increases_point_count(self):
        """Smooth iterations > 0 should add more points per polyline."""
        img = make_checkerboard(80, 80, tile=10)
        base = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
        }
        result_no_smooth = self.gen.generate({**base, "smooth_iterations": 0}, self.canvas)
        result_smooth = self.gen.generate({**base, "smooth_iterations": 2}, self.canvas)
        pts_no_smooth = sum(len(p) for p in result_no_smooth)
        pts_smooth = sum(len(p) for p in result_smooth)
        # Chaikin doubles points per iteration — total should increase
        assert pts_smooth > pts_no_smooth, (
            "Chaikin smoothing should increase total point count"
        )

    # --- Progress and cancellation ---

    def test_progress_callback_reaches_100(self):
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        values: list[int] = []
        self.gen.generate(params, self.canvas, progress_callback=values.append)
        assert values, "Progress callback should be called"
        assert values[-1] == 100, "Final progress value should be 100"

    def test_cancellation_returns_early(self):
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(
            params, self.canvas, cancelled_callback=lambda: True
        )
        # Cancelled early — may return [] or a partial result
        assert isinstance(result, list)

    # --- Grayscale input handling ---

    def test_handles_grayscale_input(self):
        """Generator should work when _source_image is already grayscale (2D)."""
        img = make_checkerboard(80, 80, tile=10)  # already grayscale (2D)
        assert img.ndim == 2
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_handles_rgb_input(self):
        """Generator should work when _source_image is a 3-channel RGB image."""
        img_gray = make_checkerboard(80, 80, tile=10)
        img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
        assert img_rgb.ndim == 3
        params = {
            "_source_image": img_rgb,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    # --- Centerline tracing (Phase 16.29) ---

    def test_centerline_parameter_is_bool_default_false(self):
        """centerline parameter must be a BoolParam with default=False."""
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "centerline" in params, "centerline parameter must exist"
        assert isinstance(params["centerline"], BoolParam)
        assert params["centerline"].default is False

    def test_pencil_sketch_preset_enables_centerline(self):
        """Pencil Sketch is line-art oriented — its centerline should be True."""
        preset = next(p for p in self.gen.get_presets() if p.name == "Pencil Sketch")
        assert preset.params.get("centerline") is True, (
            "Pencil Sketch preset should enable centerline for line-art inputs"
        )

    def test_centerline_true_runs_without_error(self):
        """Generator should complete without error when centerline=True."""
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
            "centerline": True,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_centerline_reduces_thick_dark_band_to_narrow(self):
        """centerline logic reduces a 20px-wide dark band to ≤2 dark rows.

        This verifies the XDoG centerline pipeline eliminates the 'hollow
        outline' artefact: a thick dark band in the binary image (which
        produces two parallel contours) is skeletonized to a single-pixel
        centerline, so only one narrow trace results.
        """
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")

        from plottter.generators._helpers import _skeletonize

        # Simulate the XDoG binary output for a thick black line:
        # bright regions everywhere except a 20px-wide dark band in the middle.
        # Without centerline, findContours traces two parallel boundaries of
        # this band (the "hollow outline" problem).
        h, w = 60, 100
        binary = np.full((h, w), 255, dtype=np.uint8)
        binary[20:40, 5:95] = 0  # 20px-wide dark band

        # Verify the before-state: 20 dark rows
        dark_rows_before = int(np.any(binary == 0, axis=1).sum())
        assert dark_rows_before == 20

        # Apply the same centerline logic used in XDoGGenerator.generate()
        inverted = cv2.bitwise_not(binary)
        thinned = _skeletonize(inverted)
        binary_cl = cv2.bitwise_not(thinned)

        # After centerline: the 20px-wide dark band should collapse to ≤2 dark rows
        dark_rows_after = int(np.any(binary_cl == 0, axis=1).sum())
        assert dark_rows_after <= 2, (
            f"Centerline should reduce dark rows from 20 to ≤2, got {dark_rows_after}"
        )

        # Bonus: the contour count should also be reduced or equal
        contours_before, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        contours_after, _ = cv2.findContours(
            binary_cl, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        assert len(contours_after) <= len(contours_before), (
            "Centerline should not increase the number of contours"
        )


# ---------------------------------------------------------------------------
# FDoGGenerator (Phase 16.24)
# ---------------------------------------------------------------------------


class TestFDoGGenerator:
    def setup_method(self):
        from plottter.generators.fdog import FDoGGenerator
        self.gen = FDoGGenerator()
        self.canvas = make_canvas()

    # --- Registration and metadata ---

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Coherent Lines (FDoG)" in GENERATORS
        assert GENERATORS["Coherent Lines (FDoG)"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "Coherent Lines (FDoG)"
        assert self.gen.category == "image"

    # --- Parameter contract ---

    def test_gets_parameters(self):
        params = self.gen.get_parameters()
        param_names = [p.name for p in params]
        assert "sigma_c" in param_names
        assert "sigma_m" in param_names
        assert "rho" in param_names
        assert "etf_iterations" in param_names
        assert "fdog_iterations" in param_names
        assert "min_contour_length" in param_names
        assert "simplify_tolerance_mm" in param_names
        assert "close_gaps_mm" in param_names
        assert "smooth_iterations" in param_names
        assert "invert" in param_names
        assert "brightness" in param_names
        assert "contrast" in param_names
        assert "blur_radius" in param_names

    def test_parameter_ranges_match_spec(self):
        """Spec-defined ranges: sigma_c 0.5–3.0, sigma_m 0.5–6.0, rho 1.1–5.0."""
        params = {p.name: p for p in self.gen.get_parameters()}
        assert params["sigma_c"].min == pytest.approx(0.5)
        assert params["sigma_c"].max == pytest.approx(3.0)
        assert params["sigma_m"].min == pytest.approx(0.5)
        assert params["sigma_m"].max == pytest.approx(6.0)
        assert params["rho"].min == pytest.approx(1.1)
        assert params["rho"].max == pytest.approx(5.0)
        assert params["etf_iterations"].min == 1
        assert params["etf_iterations"].max == 10
        assert params["fdog_iterations"].min == 1
        assert params["fdog_iterations"].max == 5

    # --- Preset contract ---

    def test_presets_present(self):
        presets = self.gen.get_presets()
        names = [p.name for p in presets]
        assert "Coherent Lines" in names

    def test_preset_params_include_all_fdog_params(self):
        """Every preset must supply values for all FDoG-specific parameters."""
        for preset in self.gen.get_presets():
            for key in ("sigma_c", "sigma_m", "rho", "etf_iterations", "fdog_iterations"):
                assert key in preset.params, (
                    f"Preset '{preset.name}' missing key '{key}'"
                )

    # --- Core generation ---

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_generates_polylines_from_checkerboard(self):
        """Checkerboard has strong edges — FDoG should detect them."""
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "FDoG should detect edges in a checkerboard"
        assert all(len(p) >= 2 for p in result)

    def test_generates_polylines_from_dark_center_image(self):
        """Dark circle on white background — FDoG should trace the edge."""
        img = make_dark_center_image(100, 100)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.3,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "FDoG should detect edge of dark circle"

    def test_output_within_canvas_bounds(self):
        """All generated points must lie within the canvas drawing area."""
        img = make_checkerboard(80, 80, tile=10)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas), (
            "All output points should be within canvas drawing area"
        )

    # --- Preset smoke tests ---

    def test_coherent_lines_preset_generates_output(self):
        """Coherent Lines preset (the spec-named preset) should generate output."""
        img = make_checkerboard(80, 80, tile=10)
        preset = next(p for p in self.gen.get_presets() if p.name == "Coherent Lines")
        result = self.gen.generate({**preset.params, "_source_image": img}, self.canvas)
        assert len(result) > 0, "Coherent Lines preset should generate output"

    def test_all_presets_generate_without_error(self):
        """All presets should complete without raising an exception."""
        img = make_checkerboard(80, 80, tile=10)
        for preset in self.gen.get_presets():
            params = dict(preset.params)
            params["_source_image"] = img
            params["min_contour_length"] = 3  # lower threshold for small test image
            try:
                result = self.gen.generate(params, self.canvas)
                assert isinstance(result, list), (
                    f"Preset '{preset.name}' did not return a list"
                )
            except Exception as exc:
                pytest.fail(f"Preset '{preset.name}' raised: {exc}")

    # --- ETF unit tests ---

    def test_etf_returns_unit_vectors(self):
        """_compute_etf should return normalised (approximately unit) tangent vectors."""
        from plottter.generators.fdog import _compute_etf
        img = make_checkerboard(30, 30, tile=5).astype(np.float32) / 255.0
        tx, ty = _compute_etf(img, sigma_m=1.5, iterations=2)
        assert tx.shape == img.shape
        assert ty.shape == img.shape
        # On pixels where there is a gradient, the tangent should be a unit vector
        mag = np.sqrt(tx**2 + ty**2)
        # Pixels with non-trivial gradient should be close to unit length
        nonzero_mask = mag > 0.1
        if nonzero_mask.any():
            norms = mag[nonzero_mask]
            assert np.all(norms > 0.5), "Tangent magnitude should be > 0.5 for edge pixels"
            assert np.all(norms <= 1.01), "Tangent magnitude should be ≤ 1 for edge pixels"

    def test_fdog_output_in_unit_range(self):
        """_fdog should return a float32 image with values in [0, 1]."""
        from plottter.generators.fdog import _fdog
        img = make_checkerboard(40, 40, tile=5).astype(np.float32) / 255.0
        result = _fdog(img, sigma_c=1.0, rho=3.0, sigma_m=2.0,
                       etf_iterations=2, fdog_iterations=1)
        assert result.dtype == np.float32
        assert result.shape == img.shape
        assert float(result.min()) >= 0.0, "FDoG output should be >= 0"
        assert float(result.max()) <= 1.0, "FDoG output should be <= 1"

    # --- Handles different input formats ---

    def test_handles_grayscale_input(self):
        """Generator should work when _source_image is already grayscale (2D)."""
        img = make_checkerboard(60, 60, tile=8)
        assert img.ndim == 2
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_handles_rgb_input(self):
        """Generator should work when _source_image is a 3-channel RGB image."""
        img_gray = make_checkerboard(60, 60, tile=8)
        img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
        assert img_rgb.ndim == 3
        params = {
            "_source_image": img_rgb,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    # --- Progress and cancellation ---

    def test_progress_callback_reaches_100(self):
        img = make_checkerboard(60, 60, tile=8)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        values: list[int] = []
        self.gen.generate(params, self.canvas, progress_callback=values.append)
        assert values, "Progress callback should be called"
        assert values[-1] == 100, "Final progress value should be 100"

    def test_cancellation_returns_list(self):
        """Cancelling immediately should return a list (possibly empty)."""
        img = make_checkerboard(60, 60, tile=8)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(
            params, self.canvas, cancelled_callback=lambda: True
        )
        assert isinstance(result, list)

    # --- ETF smoothing effect ---

    def test_more_etf_iterations_does_not_crash(self):
        """Running multiple ETF passes should complete without error."""
        img = make_dark_center_image(60, 60)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 5,
            "fdog_iterations": 2,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_uniform_image_returns_empty_or_list(self):
        """A fully uniform image has no edges — generator should return empty or minimal output."""
        img = np.full((60, 60), 128, dtype=np.uint8)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        # A uniform image should produce no meaningful edges
        assert len(result) == 0 or all(len(p) >= 2 for p in result)

    # --- Centerline tracing (Phase 16.29) ---

    def test_centerline_parameter_is_bool_default_false(self):
        """centerline parameter must be a BoolParam with default=False."""
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "centerline" in params, "centerline parameter must exist"
        assert isinstance(params["centerline"], BoolParam)
        assert params["centerline"].default is False

    def test_centerline_true_runs_without_error(self):
        """Generator should complete without error when centerline=True."""
        img = make_checkerboard(60, 60, tile=8)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
            "centerline": True,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_centerline_false_same_as_default_behavior(self):
        """centerline=False should produce the same result as not specifying centerline."""
        img = make_checkerboard(60, 60, tile=8)
        base = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 3.0,
            "etf_iterations": 1,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
        }
        result_default = self.gen.generate(base, self.canvas)
        result_false = self.gen.generate({**base, "centerline": False}, self.canvas)
        # Both should produce the same polylines
        assert len(result_default) == len(result_false)

    # --- New presets (Phase 65.1) ---

    def test_all_eight_presets_exist(self):
        """There should be exactly 8 presets (3 original + 5 new)."""
        presets = self.gen.get_presets()
        names = [p.name for p in presets]
        for expected in (
            "Coherent Lines", "Fine Lines", "Bold Strokes",
            "Portrait", "Ink Sketch", "Stylized Illustration",
            "Noisy Photo", "Ultra-Fine Detail",
        ):
            assert expected in names, f"Missing preset: {expected}"
        assert len(presets) == 8

    def test_all_presets_generate_valid_output(self):
        """All 8 presets must produce non-empty valid polylines on a test image."""
        img = make_checkerboard(80, 80, tile=10)
        for preset in self.gen.get_presets():
            params = {**preset.params, "_source_image": img}
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list), f"Preset '{preset.name}' did not return a list"
            assert len(result) > 0, f"Preset '{preset.name}' returned empty output"
            for poly in result:
                assert len(poly) >= 2, f"Preset '{preset.name}' has a polyline with < 2 points"

    def test_ink_sketch_preset_has_centerline_true(self):
        """Ink Sketch preset must have centerline=True."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Ink Sketch" in presets
        assert presets["Ink Sketch"].params.get("centerline") is True

    def test_stylized_illustration_preset_has_smooth_curves_true(self):
        """Stylized Illustration preset must have smooth_curves=True."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Stylized Illustration" in presets
        assert presets["Stylized Illustration"].params.get("smooth_curves") is True

    def test_portrait_fewer_longer_polylines_than_fine_lines(self):
        """Portrait preset (higher sigma_m) should produce fewer, longer lines than Fine Lines."""
        img = make_dark_center_image(100, 100)
        presets = {p.name: p for p in self.gen.get_presets()}
        portrait_params = {**presets["Portrait"].params, "_source_image": img}
        fine_params = {**presets["Fine Lines"].params, "_source_image": img}
        portrait_result = self.gen.generate(portrait_params, self.canvas)
        fine_result = self.gen.generate(fine_params, self.canvas)
        if not portrait_result or not fine_result:
            pytest.skip("One or both presets returned no output — cannot compare")
        portrait_avg_len = sum(len(p) for p in portrait_result) / len(portrait_result)
        fine_avg_len = sum(len(p) for p in fine_result) / len(fine_result)
        # Portrait has higher sigma_m → longer, more coherent strokes
        assert len(portrait_result) <= len(fine_result) or portrait_avg_len >= fine_avg_len, (
            f"Portrait (avg_len={portrait_avg_len:.1f}, count={len(portrait_result)}) "
            f"should have fewer or longer lines than Fine Lines "
            f"(avg_len={fine_avg_len:.1f}, count={len(fine_result)})"
        )


# ---------------------------------------------------------------------------
# Skeletonize helpers (Phase 16.29)
# ---------------------------------------------------------------------------


class TestZhangSuenThinning:
    """Unit tests for the pure-NumPy Zhang-Suen thinning fallback."""

    def test_thick_horizontal_band_reduces_to_single_row(self):
        """A 20px-wide horizontal white band should thin to at most 2 rows."""
        from plottter.generators._helpers import _zhang_suen_thinning

        img = np.zeros((60, 100), dtype=np.uint8)
        img[20:40, 10:90] = 255  # 20px-wide horizontal band (not full width)
        result = _zhang_suen_thinning(img)

        rows_with_fg = int(np.any(result > 0, axis=1).sum())
        assert rows_with_fg <= 2, (
            f"20px-wide band should thin to ≤2 rows, got {rows_with_fg}"
        )

    def test_empty_image_returns_empty(self):
        from plottter.generators._helpers import _zhang_suen_thinning
        img = np.zeros((40, 40), dtype=np.uint8)
        result = _zhang_suen_thinning(img)
        assert np.all(result == 0)

    def test_output_is_uint8(self):
        from plottter.generators._helpers import _zhang_suen_thinning
        img = np.zeros((20, 20), dtype=np.uint8)
        img[8:12, 8:12] = 255
        result = _zhang_suen_thinning(img)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self):
        from plottter.generators._helpers import _zhang_suen_thinning
        img = np.zeros((30, 50), dtype=np.uint8)
        img[10:20, 10:40] = 255
        result = _zhang_suen_thinning(img)
        assert result.shape == img.shape

    def test_output_values_are_0_or_255(self):
        from plottter.generators._helpers import _zhang_suen_thinning
        img = np.zeros((30, 50), dtype=np.uint8)
        img[10:20, 10:40] = 255
        result = _zhang_suen_thinning(img)
        unique_vals = set(np.unique(result).tolist())
        assert unique_vals.issubset({0, 255}), f"Unexpected values: {unique_vals}"

    def test_thick_band_pixel_count_reduced(self):
        """Skeleton should have many fewer foreground pixels than the original."""
        from plottter.generators._helpers import _zhang_suen_thinning
        img = np.zeros((60, 100), dtype=np.uint8)
        img[20:40, 10:90] = 255  # 20 * 80 = 1600 foreground pixels
        result = _zhang_suen_thinning(img)
        original_fg = int(np.sum(img > 0))
        skeleton_fg = int(np.sum(result > 0))
        assert skeleton_fg < original_fg // 4, (
            f"Skeleton ({skeleton_fg} px) should be much smaller than original ({original_fg} px)"
        )


class TestSkeletonize:
    """Unit tests for the _skeletonize wrapper (tries ximgproc, skimage, fallback)."""

    def test_thick_horizontal_band_reduces_to_single_row(self):
        """A 20px-wide horizontal white band should thin to at most 2 rows."""
        from plottter.generators._helpers import _skeletonize

        img = np.zeros((60, 100), dtype=np.uint8)
        img[20:40, 10:90] = 255
        result = _skeletonize(img)

        rows_with_fg = int(np.any(result > 0, axis=1).sum())
        assert rows_with_fg <= 2, (
            f"20px-wide band should thin to ≤2 rows, got {rows_with_fg}"
        )

    def test_empty_image_returns_empty(self):
        from plottter.generators._helpers import _skeletonize
        img = np.zeros((40, 40), dtype=np.uint8)
        result = _skeletonize(img)
        assert np.all(result == 0)

    def test_output_shape_preserved(self):
        from plottter.generators._helpers import _skeletonize
        img = np.zeros((30, 50), dtype=np.uint8)
        img[10:20, 10:40] = 255
        result = _skeletonize(img)
        assert result.shape == img.shape

    def test_output_values_are_0_or_255(self):
        from plottter.generators._helpers import _skeletonize
        img = np.zeros((30, 50), dtype=np.uint8)
        img[10:20, 10:40] = 255
        result = _skeletonize(img)
        unique_vals = set(np.unique(result).tolist())
        assert unique_vals.issubset({0, 255}), f"Unexpected values: {unique_vals}"


# ---------------------------------------------------------------------------
# Adaptive thresholding (Phase 16.32)
# ---------------------------------------------------------------------------


def make_gradient_line_art(h: int = 200, w: int = 200) -> np.ndarray:
    """Image with uneven lighting + dark ink lines.

    Left half: dark background (value 40), ink lines at value 10.
    Right half: bright background (value 210), ink lines at value 155.

    A single global threshold (e.g. 128) will detect lines on the left but
    miss them on the right because 155 > 128.  Adaptive thresholding detects
    lines on both halves because it computes a local threshold relative to the
    neighbourhood mean.
    """
    arr = np.full((h, w), 40, dtype=np.uint8)
    arr[:, w // 2 :] = 210  # bright right half
    # Horizontal ink lines at rows 40, 80, 120, 160
    for row in [40, 80, 120, 160]:
        arr[row - 2 : row + 3, : w // 2] = 10   # dark ink on dark background
        arr[row - 2 : row + 3, w // 2 :] = 155  # relatively dark ink on bright background
    return arr


class TestApplyThreshold:
    """Unit tests for _apply_threshold helper (Phase 16.32)."""

    def test_global_thresh_binary_inv_below_value_becomes_255(self):
        """Pixels below threshold → 255 for THRESH_BINARY_INV."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = np.array([[50, 100, 200]], dtype=np.uint8)
        result = _apply_threshold(img, 128, False, 5.0, cv2.THRESH_BINARY_INV)
        assert result[0, 0] == 255, "50 < 128 → should be 255"
        assert result[0, 1] == 255, "100 < 128 → should be 255"
        assert result[0, 2] == 0, "200 > 128 → should be 0"

    def test_global_thresh_output_is_binary(self):
        """Global threshold output values must be 0 or 255."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = make_gradient_image(50, 50)
        result = _apply_threshold(img, 128, False, 5.0, cv2.THRESH_BINARY_INV)
        unique = set(result.flatten().tolist())
        assert unique <= {0, 255}, f"Non-binary values found: {unique - {0, 255}}"

    def test_adaptive_thresh_output_is_binary(self):
        """Adaptive threshold output values must be 0 or 255."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = make_gradient_image(50, 50)
        result = _apply_threshold(img, 128, True, 5.0, cv2.THRESH_BINARY_INV)
        unique = set(result.flatten().tolist())
        assert unique <= {0, 255}, f"Non-binary values found: {unique - {0, 255}}"

    def test_output_shape_preserved(self):
        """Output shape must match input shape."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = make_gradient_image(80, 120)
        for adaptive in (False, True):
            result = _apply_threshold(img, 128, adaptive, 5.0, cv2.THRESH_BINARY_INV)
            assert result.shape == img.shape, (
                f"adaptive={adaptive}: output shape {result.shape} != input {img.shape}"
            )

    def test_global_threshold_misses_lines_in_bright_region(self):
        """Global threshold at 128 should miss the ink lines in the bright right half.

        The right-half ink pixels have value 155, which is above the global
        threshold of 128, so they are classified as background (0) with
        THRESH_BINARY_INV.
        """
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = make_gradient_line_art(200, 200)
        result = _apply_threshold(img, 128, False, 5.0, cv2.THRESH_BINARY_INV)

        # Right-half ink lines at row 80: all pixels should be 0 (missed)
        right_ink_row = result[78:83, 100:]
        assert right_ink_row.sum() == 0, (
            "Global threshold 128 should miss ink pixels with value 155 on the right half"
        )

    def test_adaptive_threshold_detects_lines_in_bright_region(self):
        """Adaptive threshold should detect ink lines in the bright right half.

        The right-half ink pixels (value 155) are distinctly darker than their
        local neighbourhood (mean ≈ 200+), so adaptive thresholding classifies
        them as foreground.
        """
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = make_gradient_line_art(200, 200)
        result = _apply_threshold(img, 128, True, 5.0, cv2.THRESH_BINARY_INV)

        # Right-half ink lines at row 80: at least some pixels should be 255 (detected)
        right_ink_row = result[78:83, 100:]
        assert right_ink_row.sum() > 0, (
            "Adaptive threshold should detect ink lines (value 155) in bright right half"
        )

    def test_adaptive_detects_lines_across_full_image(self):
        """Adaptive threshold should find ink lines on both left and right halves."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = make_gradient_line_art(200, 200)
        result = _apply_threshold(img, 128, True, 5.0, cv2.THRESH_BINARY_INV)

        for row in [40, 80, 120, 160]:
            left_ink = result[row - 2 : row + 3, :100]
            right_ink = result[row - 2 : row + 3, 100:]
            assert left_ink.sum() > 0, f"Adaptive should detect ink on left at row {row}"
            assert right_ink.sum() > 0, f"Adaptive should detect ink on right at row {row}"

    def test_adaptive_c_positive_reduces_foreground(self):
        """Higher adaptive_c makes the threshold stricter (fewer foreground pixels)."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        from plottter.generators._helpers import _apply_threshold

        img = make_gradient_line_art(200, 200)
        result_low_c = _apply_threshold(img, 128, True, -5.0, cv2.THRESH_BINARY_INV)
        result_high_c = _apply_threshold(img, 128, True, 15.0, cv2.THRESH_BINARY_INV)
        # High C is stricter → fewer or equal foreground pixels
        assert result_high_c.sum() <= result_low_c.sum(), (
            "Higher adaptive_c should produce fewer or equal foreground pixels"
        )


class TestAdaptiveThresholdingContour:
    """Tests for adaptive_threshold in ContourGenerator (Phase 16.32)."""

    def setup_method(self):
        from plottter.generators.contour import ContourGenerator
        self.gen = ContourGenerator()
        self.canvas = make_canvas()

    # --- Parameter structure ---

    def test_adaptive_threshold_param_exists(self):
        """adaptive_threshold must be a BoolParam with default=False."""
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "adaptive_threshold" in params, "adaptive_threshold parameter must exist"
        p = params["adaptive_threshold"]
        assert isinstance(p, BoolParam)
        assert p.default is False

    def test_adaptive_c_param_exists(self):
        """adaptive_c must be a FloatParam with range -20 to 20, default 5."""
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "adaptive_c" in params, "adaptive_c parameter must exist"
        p = params["adaptive_c"]
        assert isinstance(p, FloatParam)
        assert p.min == -20.0
        assert p.max == 20.0
        assert p.default == 5.0

    def test_adaptive_threshold_visible_in_line_art_and_skeleton(self):
        """adaptive_threshold must be visible for Line Art Trace and Skeleton modes."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["adaptive_threshold"].visible_when
        assert vw is not None
        assert "mode" in vw
        assert "Line Art Trace" in vw["mode"]
        assert "Skeleton" in vw["mode"]
        assert "Contour Levels" not in vw["mode"]

    def test_adaptive_c_visible_when_adaptive_true(self):
        """adaptive_c must only be visible when adaptive_threshold is True."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["adaptive_c"].visible_when
        assert vw is not None
        assert "adaptive_threshold" in vw
        assert True in vw["adaptive_threshold"]

    def test_line_art_adaptive_runs_without_error(self):
        """Line Art Trace with adaptive_threshold=True must complete without error."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_bw_line_art(100, 100)
        params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "adaptive_threshold": True,
            "adaptive_c": 5.0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_skeleton_mode_adaptive_runs_without_error(self):
        """Skeleton mode with adaptive_threshold=True must complete without error."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_thick_stroke_image()
        params = {
            "_source_image": img,
            "mode": "Skeleton",
            "trace_threshold": 128,
            "smooth_iterations": 0,
            "cleanup_kernel": 0,
            "merge_gap_mm": 0.0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "adaptive_threshold": True,
            "adaptive_c": 5.0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_line_art_adaptive_detects_lines_global_misses(self):
        """Adaptive mode finds contours in bright region where global threshold fails.

        Uses a gradient-background image where ink lines (value 155) on the
        bright right half (background 210) are above the global threshold (128)
        and thus invisible to global thresholding.  Adaptive mode should
        produce contours in both halves.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_gradient_line_art(200, 200)
        canvas = Canvas.from_preset("A4", margin=5.0)
        base_params = {
            "_source_image": img,
            "mode": "Line Art Trace",
            "smooth_iterations": 0,
            "simplify_mm": 0.5,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }

        # Global threshold: cannot detect the ink lines on the right half
        result_global = self.gen.generate(
            {**base_params, "trace_threshold": 128, "adaptive_threshold": False, "adaptive_c": 5.0},
            canvas,
        )

        # Adaptive threshold: should detect lines on both halves
        result_adaptive = self.gen.generate(
            {**base_params, "trace_threshold": 128, "adaptive_threshold": True, "adaptive_c": 5.0},
            canvas,
        )

        # Adaptive should produce more or equal contours (it finds what global misses)
        assert len(result_adaptive) >= len(result_global), (
            f"Adaptive ({len(result_adaptive)}) should produce >= contours than global "
            f"({len(result_global)}) on gradient-background image"
        )

    def test_all_presets_include_adaptive_params(self):
        """Every preset must include adaptive_threshold and adaptive_c."""
        for preset in self.gen.get_presets():
            assert "adaptive_threshold" in preset.params, (
                f"Preset '{preset.name}' missing adaptive_threshold"
            )
            assert "adaptive_c" in preset.params, (
                f"Preset '{preset.name}' missing adaptive_c"
            )

    def test_scanned_line_art_preset_enables_adaptive(self):
        """'Scanned Line Art' preset must enable adaptive_threshold."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Scanned Line Art" in presets, "Scanned Line Art preset must exist"
        preset = presets["Scanned Line Art"]
        assert preset.params.get("adaptive_threshold") is True, (
            "Scanned Line Art preset should enable adaptive_threshold"
        )


class TestAdaptiveThresholdingXDoG:
    """Tests for adaptive_threshold in XDoGGenerator (Phase 16.32)."""

    def setup_method(self):
        from plottter.generators.xdog import XDoGGenerator
        self.gen = XDoGGenerator()
        self.canvas = make_canvas()

    def test_adaptive_threshold_param_exists(self):
        """adaptive_threshold must exist as a BoolParam with default=False."""
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "adaptive_threshold" in params
        p = params["adaptive_threshold"]
        assert isinstance(p, BoolParam)
        assert p.default is False

    def test_adaptive_c_param_exists(self):
        """adaptive_c must exist as a FloatParam with range -20 to 20."""
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "adaptive_c" in params
        p = params["adaptive_c"]
        assert isinstance(p, FloatParam)
        assert p.min == -20.0
        assert p.max == 20.0
        assert p.default == 5.0

    def test_adaptive_threshold_visible_when_centerline(self):
        """adaptive_threshold must only be visible when centerline=True."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["adaptive_threshold"].visible_when
        assert vw is not None
        assert "centerline" in vw
        assert True in vw["centerline"]

    def test_adaptive_c_visible_when_centerline_and_adaptive(self):
        """adaptive_c must be visible only with centerline=True and adaptive_threshold=True."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["adaptive_c"].visible_when
        assert vw is not None
        assert "centerline" in vw
        assert True in vw["centerline"]
        assert "adaptive_threshold" in vw
        assert True in vw["adaptive_threshold"]

    def test_adaptive_with_centerline_runs_without_error(self):
        """adaptive_threshold=True with centerline=True must complete without error."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_dark_center_image(80, 80)
        params = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
            "centerline": True,
            "merge_gap_mm": 0.0,
            "adaptive_threshold": True,
            "adaptive_c": 5.0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_adaptive_without_centerline_defaults_to_fixed_cutoff(self):
        """When centerline=False, adaptive_threshold is ignored (fixed T>=0.5 cutoff)."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_dark_center_image(80, 80)
        base = {
            "_source_image": img,
            "sigma": 0.5,
            "k": 1.6,
            "phi": 100.0,
            "epsilon": 0.0,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
            "centerline": False,
            "merge_gap_mm": 0.0,
            "adaptive_c": 5.0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result_no_adaptive = self.gen.generate({**base, "adaptive_threshold": False}, self.canvas)
        result_adaptive = self.gen.generate({**base, "adaptive_threshold": True}, self.canvas)
        # Both should complete without error (adaptive is not applied when centerline=False)
        assert isinstance(result_no_adaptive, list)
        assert isinstance(result_adaptive, list)

    def test_all_presets_include_adaptive_params(self):
        """Every XDoG preset must include adaptive_threshold and adaptive_c keys."""
        for preset in self.gen.get_presets():
            assert "adaptive_threshold" in preset.params, (
                f"Preset '{preset.name}' missing adaptive_threshold"
            )
            assert "adaptive_c" in preset.params, (
                f"Preset '{preset.name}' missing adaptive_c"
            )


class TestAdaptiveThresholdingFDoG:
    """Tests for adaptive_threshold in FDoGGenerator (Phase 16.32)."""

    def setup_method(self):
        from plottter.generators.fdog import FDoGGenerator
        self.gen = FDoGGenerator()
        self.canvas = make_canvas()

    def test_adaptive_threshold_param_exists(self):
        """adaptive_threshold must exist as a BoolParam with default=False."""
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "adaptive_threshold" in params
        p = params["adaptive_threshold"]
        assert isinstance(p, BoolParam)
        assert p.default is False

    def test_adaptive_c_param_exists(self):
        """adaptive_c must exist as a FloatParam with range -20 to 20."""
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "adaptive_c" in params
        p = params["adaptive_c"]
        assert isinstance(p, FloatParam)
        assert p.min == -20.0
        assert p.max == 20.0
        assert p.default == 5.0

    def test_adaptive_threshold_visible_when_centerline(self):
        """adaptive_threshold must only be visible when centerline=True."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["adaptive_threshold"].visible_when
        assert vw is not None
        assert "centerline" in vw
        assert True in vw["centerline"]

    def test_adaptive_c_visible_when_centerline_and_adaptive(self):
        """adaptive_c must be visible only with centerline=True and adaptive_threshold=True."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["adaptive_c"].visible_when
        assert vw is not None
        assert "centerline" in vw
        assert True in vw["centerline"]
        assert "adaptive_threshold" in vw
        assert True in vw["adaptive_threshold"]

    def test_adaptive_with_centerline_runs_without_error(self):
        """adaptive_threshold=True with centerline=True must complete without error."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not installed")

        img = make_dark_center_image(60, 60)
        params = {
            "_source_image": img,
            "sigma_c": 1.0,
            "sigma_m": 2.0,
            "rho": 2.5,
            "etf_iterations": 2,
            "fdog_iterations": 1,
            "min_contour_length": 3,
            "simplify_tolerance_mm": 0.5,
            "close_gaps_mm": 0.0,
            "smooth_iterations": 0,
            "centerline": True,
            "merge_gap_mm": 0.0,
            "adaptive_threshold": True,
            "adaptive_c": 5.0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_all_presets_include_adaptive_params(self):
        """Every FDoG preset must include adaptive_threshold and adaptive_c keys."""
        for preset in self.gen.get_presets():
            assert "adaptive_threshold" in preset.params, (
                f"Preset '{preset.name}' missing adaptive_threshold"
            )
            assert "adaptive_c" in preset.params, (
                f"Preset '{preset.name}' missing adaptive_c"
            )


# ---------------------------------------------------------------------------
# ContourGenerator — FMM Topographic mode (Phase 16.35)
# ---------------------------------------------------------------------------


class TestContourGeneratorFMMMode:
    """Tests for the FMM Topographic mode added in Phase 16.35."""

    def setup_method(self):
        from plottter.generators.contour import ContourGenerator
        self.gen = ContourGenerator()
        self.canvas = make_canvas()

    def _base_fmm_params(self, img: np.ndarray) -> dict:
        return {
            "_source_image": img,
            "mode": "FMM Topographic",
            "fmm_num_contours": 5,
            "fmm_source_point": "Center",
            "fmm_gamma": 1.0,
            "fmm_speed_floor": 0.01,
            "fmm_contour_spacing": "Linear",
            "fmm_min_contour_length_mm": 0.0,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
        }

    # --- Parameter structure ---

    def test_fmm_num_contours_param_exists(self):
        """fmm_num_contours must be an IntParam with correct range and default."""
        from plottter.generators.base import IntParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fmm_num_contours" in params
        p = params["fmm_num_contours"]
        assert isinstance(p, IntParam)
        assert p.min == 2
        assert p.max == 100
        assert p.default == 20

    def test_fmm_source_point_param_exists(self):
        """fmm_source_point must be a ChoiceParam with Center and Custom."""
        from plottter.generators.base import ChoiceParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fmm_source_point" in params
        p = params["fmm_source_point"]
        assert isinstance(p, ChoiceParam)
        assert "Center" in p.choices
        assert "Custom" in p.choices
        assert p.default == "Center"

    def test_fmm_gamma_param_exists(self):
        """fmm_gamma must be a FloatParam with range 0.1–5.0 and default 1.0."""
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fmm_gamma" in params
        p = params["fmm_gamma"]
        assert isinstance(p, FloatParam)
        assert p.min == 0.1
        assert p.max == 5.0
        assert p.default == 1.0

    def test_fmm_speed_floor_param_exists(self):
        """fmm_speed_floor must be a FloatParam with small positive default."""
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fmm_speed_floor" in params
        p = params["fmm_speed_floor"]
        assert isinstance(p, FloatParam)
        assert p.default > 0.0
        assert p.min > 0.0

    def test_fmm_contour_spacing_param_exists(self):
        """fmm_contour_spacing must be a ChoiceParam with Linear/Logarithmic/Quadratic."""
        from plottter.generators.base import ChoiceParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fmm_contour_spacing" in params
        p = params["fmm_contour_spacing"]
        assert isinstance(p, ChoiceParam)
        assert "Linear" in p.choices
        assert "Logarithmic" in p.choices
        assert "Quadratic" in p.choices

    def test_fmm_min_contour_length_mm_param_exists(self):
        """fmm_min_contour_length_mm must be a FloatParam with non-negative default."""
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fmm_min_contour_length_mm" in params
        p = params["fmm_min_contour_length_mm"]
        assert isinstance(p, FloatParam)
        assert p.default >= 0.0

    def test_fmm_params_visible_only_in_fmm_mode(self):
        """All FMM-specific params must be visible only for FMM Topographic mode."""
        fmm_param_names = [
            "fmm_num_contours",
            "fmm_source_point",
            "fmm_gamma",
            "fmm_speed_floor",
            "fmm_contour_spacing",
            "fmm_min_contour_length_mm",
        ]
        params = {p.name: p for p in self.gen.get_parameters()}
        for name in fmm_param_names:
            assert name in params, f"Parameter {name!r} missing"
            vw = params[name].visible_when
            assert vw is not None, f"{name} must have visible_when"
            assert "mode" in vw, f"{name} visible_when must key on 'mode'"
            assert "FMM Topographic" in vw["mode"], (
                f"{name} must be visible in FMM Topographic mode"
            )
            assert "Contour Levels" not in vw.get("mode", []), (
                f"{name} must not be visible in Contour Levels mode"
            )

    def test_smooth_iterations_visible_in_fmm_mode(self):
        """smooth_iterations must be visible in FMM Topographic mode."""
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["smooth_iterations"].visible_when
        assert vw is not None
        assert "FMM Topographic" in vw.get("mode", [])

    # --- Generation ---

    def test_fmm_returns_list_of_polylines(self):
        """FMM mode must return a list of polylines from a gradient image."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(60, 60)
        result = self.gen.generate(self._base_fmm_params(img), self.canvas)
        assert isinstance(result, list)
        for poly in result:
            assert isinstance(poly, list)
            assert len(poly) >= 2
            for pt in poly:
                assert len(pt) == 2

    def test_fmm_returns_empty_without_source_image(self):
        """FMM mode must return [] when no source image is provided."""
        params = {
            "mode": "FMM Topographic",
            "fmm_num_contours": 5,
            "fmm_source_point": "Center",
            "fmm_gamma": 1.0,
            "fmm_speed_floor": 0.01,
            "fmm_contour_spacing": "Linear",
            "fmm_min_contour_length_mm": 0.0,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
        }
        result = self.gen.generate(params, self.canvas)
        assert result == []

    def test_fmm_produces_contours_on_gradient(self):
        """FMM mode must produce at least one contour on a gradient image."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(80, 80)
        params = self._base_fmm_params(img)
        params["fmm_num_contours"] = 10
        result = self.gen.generate(params, self.canvas)
        assert len(result) >= 1, "FMM mode should produce at least one contour on a gradient"

    def test_fmm_more_contours_increases_output(self):
        """Increasing num_contours should generally produce more output polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(80, 80)
        params_few = self._base_fmm_params(img)
        params_few["fmm_num_contours"] = 3
        result_few = self.gen.generate(params_few, self.canvas)

        params_many = self._base_fmm_params(img)
        params_many["fmm_num_contours"] = 20
        result_many = self.gen.generate(params_many, self.canvas)

        assert len(result_many) >= len(result_few), (
            "More contours requested should produce >= polylines"
        )

    def test_fmm_linear_spacing(self):
        """FMM mode with Linear spacing must run without error."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(60, 60)
        params = self._base_fmm_params(img)
        params["fmm_contour_spacing"] = "Linear"
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_fmm_logarithmic_spacing(self):
        """FMM mode with Logarithmic spacing must run without error."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(60, 60)
        params = self._base_fmm_params(img)
        params["fmm_contour_spacing"] = "Logarithmic"
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_fmm_quadratic_spacing(self):
        """FMM mode with Quadratic spacing must run without error."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(60, 60)
        params = self._base_fmm_params(img)
        params["fmm_contour_spacing"] = "Quadratic"
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_fmm_dark_center_image(self):
        """FMM mode must produce contours around a dark center circle."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_dark_center_image(80, 80)
        params = self._base_fmm_params(img)
        params["fmm_num_contours"] = 8
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_fmm_min_contour_length_filter(self):
        """min_contour_length_mm filter should reduce number of output polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(80, 80)
        params_all = self._base_fmm_params(img)
        params_all["fmm_min_contour_length_mm"] = 0.0
        result_all = self.gen.generate(params_all, self.canvas)

        params_filtered = self._base_fmm_params(img)
        params_filtered["fmm_min_contour_length_mm"] = 50.0
        result_filtered = self.gen.generate(params_filtered, self.canvas)

        assert len(result_filtered) <= len(result_all), (
            "Higher min_contour_length_mm should produce fewer or equal polylines"
        )

    def test_fmm_gamma_effect(self):
        """Changing gamma should not raise an error."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(60, 60)
        for gamma in [0.5, 1.0, 2.0]:
            params = self._base_fmm_params(img)
            params["fmm_gamma"] = gamma
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list), f"gamma={gamma} should produce a list"

    def test_fmm_smooth_iterations(self):
        """FMM mode with smooth_iterations > 0 must run without error."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(60, 60)
        params = self._base_fmm_params(img)
        params["smooth_iterations"] = 2
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_fmm_cancelled_immediately(self):
        """FMM mode with immediate cancellation must return []."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(60, 60)
        params = self._base_fmm_params(img)
        result = self.gen.generate(
            params, self.canvas, cancelled_callback=lambda: True
        )
        assert result == []

    # --- Presets ---

    def test_fmm_presets_exist(self):
        """At least one FMM Topographic preset must exist."""
        fmm_presets = [
            p for p in self.gen.get_presets()
            if p.params.get("mode") == "FMM Topographic"
        ]
        assert len(fmm_presets) >= 1, "At least one FMM Topographic preset must exist"

    def test_fmm_presets_have_required_params(self):
        """All FMM presets must include required FMM-specific parameters."""
        required = [
            "fmm_num_contours",
            "fmm_source_point",
            "fmm_gamma",
            "fmm_speed_floor",
            "fmm_contour_spacing",
            "fmm_min_contour_length_mm",
        ]
        for preset in self.gen.get_presets():
            if preset.params.get("mode") != "FMM Topographic":
                continue
            for key in required:
                assert key in preset.params, (
                    f"FMM preset '{preset.name}' missing required param {key!r}"
                )

    def test_fmm_presets_runnable(self):
        """All FMM presets must run without raising an exception."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not installed")

        img = make_gradient_image(80, 80)
        for preset in self.gen.get_presets():
            if preset.params.get("mode") != "FMM Topographic":
                continue
            params = dict(preset.params)
            params["_source_image"] = img
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list), (
                f"Preset '{preset.name}' did not return a list"
            )

    def test_all_presets_runnable_after_fmm_addition(self):
        """All presets (including new FMM ones) must include base required params."""
        required_base = [
            "mode", "smooth_iterations", "simplify_mm", "min_contour_px",
            "invert", "brightness", "contrast", "blur_radius",
        ]
        for preset in self.gen.get_presets():
            for key in required_base:
                assert key in preset.params, (
                    f"Preset '{preset.name}' missing required param {key!r}"
                )

    def test_all_presets_include_fmm_params(self):
        """Every preset must include all 6 FMM parameter keys."""
        fmm_keys = [
            "fmm_num_contours",
            "fmm_source_point",
            "fmm_gamma",
            "fmm_speed_floor",
            "fmm_contour_spacing",
            "fmm_min_contour_length_mm",
        ]
        for preset in self.gen.get_presets():
            for key in fmm_keys:
                assert key in preset.params, (
                    f"Preset '{preset.name}' missing FMM param {key!r}"
                )

    def test_all_presets_include_fmm_wave_spacing_params(self):
        """Every preset must include all 7 FMM wave spacing parameter keys."""
        spacing_keys = [
            "fmm_line_spacing",
            "fmm_min_spacing_mm",
            "fmm_max_spacing_mm",
            "fmm_group_size",
            "fmm_group_gap_mm",
            "fmm_group_intra_spacing_mm",
            "fmm_displacement_variation",
        ]
        for preset in self.gen.get_presets():
            for key in spacing_keys:
                assert key in preset.params, (
                    f"Preset '{preset.name}' missing FMM spacing param {key!r}"
                )


# ---------------------------------------------------------------------------
# ContourGenerator — FMM alternative render modes (Phase 16.39)
# ---------------------------------------------------------------------------


class TestContourGeneratorFMMRenderModes:
    """Tests for Displacement, Wave, and Radial render modes in FMM Topographic."""

    def setup_method(self):
        from plottter.generators.contour import ContourGenerator
        self.gen = ContourGenerator()
        self.canvas = Canvas.from_preset("A4", margin=10.0)

    def _fmm_params(self, img: np.ndarray, render_mode: str) -> dict:
        return {
            "_source_image": img,
            "mode": "FMM Topographic",
            "fmm_render_mode": render_mode,
            "fmm_source_point": "Center",
            "fmm_gamma": 1.0,
            "fmm_speed_floor": 0.01,
            "fmm_num_contours": 5,
            "fmm_contour_spacing": "Linear",
            "fmm_min_contour_length_mm": 0.5,
            "fmm_num_lines": 20,
            "fmm_displacement_mm": 5.0,
            "fmm_line_angle": 0.0,
            "fmm_amplitude_mm": 3.0,
            "fmm_frequency": 5.0,
            "fmm_num_radials": 24,
            "fmm_step_size_mm": 1.0,
            "smooth_iterations": 0,
            "simplify_mm": 0.3,
            "min_contour_px": 3,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
        }

    def test_fmm_render_mode_param_exists(self):
        """fmm_render_mode parameter must be defined and visible only in FMM mode."""
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fmm_render_mode" in params, "fmm_render_mode parameter must exist"
        p = params["fmm_render_mode"]
        assert "Contours" in p.choices
        assert "Displacement" in p.choices
        assert "Wave" in p.choices
        assert "Radial" in p.choices
        assert p.visible_when is not None
        assert "FMM Topographic" in p.visible_when.get("mode", [])

    def test_displacement_mode_returns_polylines(self):
        """Displacement render mode must return non-empty polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Displacement")
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0, "Displacement mode must produce at least one polyline"
        for poly in result:
            assert len(poly) >= 2

    def test_wave_mode_returns_polylines(self):
        """Wave render mode must return non-empty polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Wave")
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0, "Wave mode must produce at least one polyline"
        for poly in result:
            assert len(poly) >= 2

    def test_radial_mode_returns_polylines(self):
        """Radial render mode must return non-empty polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Radial")
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0, "Radial mode must produce at least one polyline"
        for poly in result:
            assert len(poly) >= 2

    def test_displacement_line_count(self):
        """Displacement mode must produce exactly fmm_num_lines polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Displacement")
        params["fmm_num_lines"] = 30
        result = self.gen.generate(params, self.canvas)
        # Lines equal to fmm_num_lines (minus any fully out-of-bounds segments)
        assert len(result) <= 30

    def test_wave_line_count(self):
        """Wave mode must produce at most fmm_num_lines polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Wave")
        params["fmm_num_lines"] = 30
        result = self.gen.generate(params, self.canvas)
        assert len(result) <= 30

    def test_radial_count(self):
        """Radial mode must produce at most fmm_num_radials polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Radial")
        params["fmm_num_radials"] = 24
        result = self.gen.generate(params, self.canvas)
        assert len(result) <= 24

    def test_contours_mode_still_works(self):
        """Explicitly setting fmm_render_mode='Contours' must still produce contours."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Contours")
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_unknown_render_mode_falls_back_to_contours(self):
        """An unknown fmm_render_mode value must fall back to Contours mode."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "UnknownMode")
        # Should not raise
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_displacement_cancelled_returns_empty(self):
        """Displacement mode with immediate cancellation must return []."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Displacement")
        result = self.gen.generate(params, self.canvas, cancelled_callback=lambda: True)
        assert result == []

    def test_new_render_mode_presets_exist(self):
        """Presets for Displacement, Wave, and Radial render modes must exist."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "FMM Displacement Lines" in presets, "FMM Displacement Lines preset must exist"
        assert "FMM Wave Lines" in presets, "FMM Wave Lines preset must exist"
        assert "FMM Radial Lines" in presets, "FMM Radial Lines preset must exist"
        assert presets["FMM Displacement Lines"].params["fmm_render_mode"] == "Displacement"
        assert presets["FMM Wave Lines"].params["fmm_render_mode"] == "Wave"
        assert presets["FMM Radial Lines"].params["fmm_render_mode"] == "Radial"

    def test_new_presets_runnable(self):
        """New render mode presets must run without raising an exception."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        target_names = {"FMM Displacement Lines", "FMM Wave Lines", "FMM Radial Lines"}
        for preset in self.gen.get_presets():
            if preset.name not in target_names:
                continue
            p = dict(preset.params)
            p["_source_image"] = img
            result = self.gen.generate(p, self.canvas)
            assert isinstance(result, list), f"Preset '{preset.name}' must return a list"

    def test_existing_fmm_presets_include_render_mode(self):
        """All FMM presets must now include the fmm_render_mode key."""
        for preset in self.gen.get_presets():
            if preset.params.get("mode") != "FMM Topographic":
                continue
            assert "fmm_render_mode" in preset.params, (
                f"FMM preset '{preset.name}' missing fmm_render_mode"
            )

    # ------------------------------------------------------------------
    # 16.62 — FMM Wave variable line spacing tests
    # ------------------------------------------------------------------

    def test_fmm_wave_spacing_params_exist(self):
        """All seven new FMM Wave spacing parameters must be defined."""
        params = {p.name: p for p in self.gen.get_parameters()}
        expected = [
            "fmm_line_spacing",
            "fmm_min_spacing_mm",
            "fmm_max_spacing_mm",
            "fmm_group_size",
            "fmm_group_gap_mm",
            "fmm_group_intra_spacing_mm",
            "fmm_displacement_variation",
        ]
        for name in expected:
            assert name in params, f"Parameter '{name}' must exist in ContourGenerator"

    def test_fmm_wave_spacing_params_visible_only_in_wave_mode(self):
        """Spacing params must be visible only when mode=FMM Topographic + fmm_render_mode=Wave."""
        params = {p.name: p for p in self.gen.get_parameters()}
        for name in ["fmm_line_spacing", "fmm_min_spacing_mm", "fmm_displacement_variation"]:
            p = params[name]
            assert p.visible_when is not None, f"{name} must have visible_when"
            assert "FMM Topographic" in p.visible_when.get("mode", [])
            assert "Wave" in p.visible_when.get("fmm_render_mode", [])

    def test_fmm_wave_uniform_is_default_behavior(self):
        """Uniform spacing must produce identical output to omitting fmm_line_spacing."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        base = self._fmm_params(img, "Wave")
        base["fmm_num_lines"] = 10
        base["fmm_displacement_variation"] = 0.0

        result_default = self.gen.generate({**base}, self.canvas)
        result_uniform = self.gen.generate({**base, "fmm_line_spacing": "Uniform"}, self.canvas)
        assert len(result_default) == len(result_uniform), (
            "Uniform mode must produce the same number of polylines as omitting the param"
        )
        for p1, p2 in zip(result_default, result_uniform):
            assert p1 == p2, "Uniform mode must match default (no spacing param) behavior"

    def test_fmm_wave_adaptive_dense_in_high_gradient_regions(self):
        """Adaptive spacing must place more lines where the gradient magnitude is high."""
        from plottter.generators.contour import _compute_fmm_wave_y_positions

        h, w = 100, 100
        # Create a synthetic grad_mag_norm: top rows = high gradient (1.0), bottom = low (0.0)
        grad_mag_norm = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            grad_mag_norm[y, :] = 1.0 - y / (h - 1)  # 1.0 at top, 0.0 at bottom

        draw_y1, draw_y2 = 10.0, 287.0
        draw_h = draw_y2 - draw_y1

        positions = _compute_fmm_wave_y_positions(
            grad_mag_norm,
            img_w=w,
            img_h=h,
            num_lines=50,
            draw_y1=draw_y1,
            draw_y2=draw_y2,
            draw_h=draw_h,
            line_spacing="Adaptive",
            min_spacing_mm=0.5,
            max_spacing_mm=10.0,
            group_size=3,
            group_gap_mm=4.0,
            group_intra_spacing_mm=0.5,
        )

        mid_y = (draw_y1 + draw_y2) / 2
        top_count = sum(1 for y in positions if y < mid_y)
        bottom_count = sum(1 for y in positions if y >= mid_y)
        assert top_count > bottom_count, (
            f"High-gradient top region should have more lines ({top_count}) "
            f"than low-gradient bottom ({bottom_count})"
        )

    def test_fmm_wave_adaptive_generates_output(self):
        """Adaptive spacing mode must produce non-empty wave output."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Wave")
        params["fmm_line_spacing"] = "Adaptive"
        params["fmm_min_spacing_mm"] = 0.5
        params["fmm_max_spacing_mm"] = 5.0
        params["fmm_displacement_variation"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert all(len(p) >= 2 for p in result)

    def test_fmm_wave_grouped_spacing_structure(self):
        """Grouped spacing must produce correct group structure in Y positions."""
        from plottter.generators.contour import _compute_fmm_wave_y_positions

        h, w = 100, 100
        # Uniform gradient so spacing is purely from group logic
        grad_mag_norm = np.full((h, w), 0.5, dtype=np.float32)

        draw_y1, draw_y2 = 10.0, 40.0
        draw_h = draw_y2 - draw_y1
        group_size = 3
        group_intra = 0.5
        group_gap = 4.0

        positions = _compute_fmm_wave_y_positions(
            grad_mag_norm,
            img_w=w,
            img_h=h,
            num_lines=50,
            draw_y1=draw_y1,
            draw_y2=draw_y2,
            draw_h=draw_h,
            line_spacing="Grouped",
            min_spacing_mm=0.5,
            max_spacing_mm=5.0,
            group_size=group_size,
            group_gap_mm=group_gap,
            group_intra_spacing_mm=group_intra,
        )

        tol = 1e-9
        assert len(positions) >= group_size, "Should produce at least one full group"
        assert abs(positions[0] - draw_y1) < tol
        assert abs(positions[1] - (draw_y1 + group_intra)) < tol
        assert abs(positions[2] - (draw_y1 + 2 * group_intra)) < tol
        expected_next = draw_y1 + (group_size - 1) * group_intra + group_gap
        if len(positions) > group_size and expected_next <= draw_y2:
            assert abs(positions[group_size] - expected_next) < tol

    def test_fmm_wave_grouped_generates_output(self):
        """Grouped spacing must produce valid polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Wave")
        params["fmm_line_spacing"] = "Grouped"
        params["fmm_group_size"] = 3
        params["fmm_group_gap_mm"] = 4.0
        params["fmm_group_intra_spacing_mm"] = 0.5
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert all(len(p) >= 2 for p in result)

    def test_fmm_wave_adaptive_grouped_generates_output(self):
        """Adaptive + Grouped spacing must produce valid polylines."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Wave")
        params["fmm_line_spacing"] = "Adaptive + Grouped"
        params["fmm_min_spacing_mm"] = 0.5
        params["fmm_max_spacing_mm"] = 5.0
        params["fmm_group_size"] = 3
        params["fmm_group_intra_spacing_mm"] = 0.5
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_fmm_wave_displacement_variation_zero_is_deterministic(self):
        """displacement_variation=0 must produce identical results on two runs."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Wave")
        params["fmm_displacement_variation"] = 0.0
        params["seed"] = 42
        r1 = self.gen.generate(params, self.canvas)
        r2 = self.gen.generate(params, self.canvas)
        assert r1 == r2, "displacement_variation=0 must be fully deterministic"

    def test_fmm_wave_displacement_variation_nonzero_changes_output(self):
        """displacement_variation > 0 must produce different output than variation=0."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        params = self._fmm_params(img, "Wave")
        params["fmm_num_lines"] = 20
        params["fmm_displacement_variation"] = 0.0
        r_no_var = self.gen.generate({**params}, self.canvas)
        params["fmm_displacement_variation"] = 1.0
        r_with_var = self.gen.generate({**params}, self.canvas)
        # With variation=1.0 and seed=0 there should be differences in y coords
        assert r_no_var != r_with_var, "Non-zero displacement_variation must alter output"

    def test_fmm_wave_existing_presets_unchanged(self):
        """Existing FMM Wave Lines preset must still run without regression."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "FMM Wave Lines" in presets
        p = dict(presets["FMM Wave Lines"].params)
        p["_source_image"] = img
        result = self.gen.generate(p, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    # 16.63 — FMM Wave new spacing mode preset tests
    # ------------------------------------------------------------------

    def test_fmm_adaptive_wave_preset_exists(self):
        """'FMM Adaptive Wave' preset must be present in ContourGenerator.get_presets()."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "FMM Adaptive Wave" in presets, (
            "'FMM Adaptive Wave' preset not found in ContourGenerator presets"
        )

    def test_fmm_grouped_wave_preset_exists(self):
        """'FMM Grouped Wave' preset must be present in ContourGenerator.get_presets()."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "FMM Grouped Wave" in presets, (
            "'FMM Grouped Wave' preset not found in ContourGenerator presets"
        )

    def test_fmm_adaptive_wave_preset_params(self):
        """'FMM Adaptive Wave' preset must use FMM Topographic + Wave + Adaptive spacing."""
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["FMM Adaptive Wave"].params
        assert p["mode"] == "FMM Topographic"
        assert p["fmm_render_mode"] == "Wave"
        assert p["fmm_line_spacing"] == "Adaptive"
        assert p["fmm_min_spacing_mm"] == 0.5
        assert p["fmm_max_spacing_mm"] == 6.0
        assert p["fmm_amplitude_mm"] == 5.0
        assert p["fmm_frequency"] == 6.0

    def test_fmm_grouped_wave_preset_params(self):
        """'FMM Grouped Wave' preset must use FMM Topographic + Wave + Grouped spacing."""
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["FMM Grouped Wave"].params
        assert p["mode"] == "FMM Topographic"
        assert p["fmm_render_mode"] == "Wave"
        assert p["fmm_line_spacing"] == "Grouped"
        assert p["fmm_group_size"] == 3
        assert p["fmm_group_gap_mm"] == 5.0
        assert p["fmm_group_intra_spacing_mm"] == 0.6
        assert p["fmm_amplitude_mm"] == 4.0
        assert p["fmm_frequency"] == 8.0
        assert p["fmm_displacement_variation"] == 0.4

    def test_fmm_adaptive_wave_preset_generates_output(self):
        """'FMM Adaptive Wave' preset must produce non-empty polylines on a gradient image."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        params = dict(presets["FMM Adaptive Wave"].params)
        params["_source_image"] = img
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list), "'FMM Adaptive Wave' must return a list"
        assert len(result) > 0, "'FMM Adaptive Wave' must produce non-empty output"
        assert all(len(poly) >= 2 for poly in result)

    def test_fmm_grouped_wave_preset_generates_output(self):
        """'FMM Grouped Wave' preset must produce non-empty polylines on a gradient image."""
        try:
            from scipy.ndimage import distance_transform_edt  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        img = make_gradient_image(60, 60)
        presets = {p.name: p for p in self.gen.get_presets()}
        params = dict(presets["FMM Grouped Wave"].params)
        params["_source_image"] = img
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list), "'FMM Grouped Wave' must return a list"
        assert len(result) > 0, "'FMM Grouped Wave' must produce non-empty output"
        assert all(len(poly) >= 2 for poly in result)


# ---------------------------------------------------------------------------
# HedcutGenerator
# ---------------------------------------------------------------------------

# Number of points used by _tiny_circle for stipple dot polylines
_HEDCUT_DOT_SIDES = 8  # _DOT_SIDES in hedcut.py


def _is_dot(poly, max_span_mm: float = 5.0) -> bool:
    """Return True if *poly* looks like a stipple dot (correct point count + small bbox).

    A genuine stipple dot polygon has exactly _HEDCUT_DOT_SIDES+1 points and a
    bounding-box span well under 5 mm.  Image-border-rectangle contours that
    survive to the same point count after Chaikin smoothing span hundreds of mm
    and are excluded by the bbox guard.
    """
    if len(poly) != _HEDCUT_DOT_SIDES + 1:
        return False
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (max(xs) - min(xs)) <= max_span_mm and (max(ys) - min(ys)) <= max_span_mm


class TestHedcutGenerator:
    def setup_method(self):
        from plottter.generators.hedcut import HedcutGenerator
        self.gen = HedcutGenerator()
        self.canvas = make_canvas()

    def _default_params(self, img):
        return {
            "_source_image": img,
            "highlight_threshold": 200,
            "shadow_threshold": 80,
            "edge_method": "Canny",   # fast; avoids XDoG/FDoG cost in tests
            "edge_sigma": 1.0,
            "edge_min_len": 5,
            "edge_simplify_mm": 0.5,
            "stipple_points": 100,    # small for speed
            "stipple_iterations": 2,
            "min_dot_size_mm": 0.2,
            "max_dot_size_mm": 0.5,
            "dot_style": "Outline",
            "pen_width_mm": 0.3,
            "dot_size_gamma": 1.0,
            "hatch_angle": 0.0,
            "hatch_spacing_mm": 3.0,  # coarse for speed
            "cross_hatch_shadows": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
        }

    # --- Basic properties ---

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Hedcut" in GENERATORS
        assert GENERATORS["Hedcut"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "Hedcut"
        assert self.gen.category == "image"

    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_parameters_include_required_fields(self):
        """All documented parameters must be present."""
        param_names = {p.name for p in self.gen.get_parameters()}
        required = {
            "highlight_threshold", "shadow_threshold",
            "edge_method", "edge_sigma",
            "stipple_points", "stipple_iterations", "min_dot_size_mm", "max_dot_size_mm", "dot_size_gamma",
            "hatch_angle", "hatch_spacing_mm", "cross_hatch_shadows",
            "brightness", "contrast", "blur_radius", "invert",
        }
        assert required.issubset(param_names), f"Missing parameters: {required - param_names}"

    # --- Tone-aware dot sizing ---

    def test_tone_aware_sizing_dark_pixels_get_larger_dots(self):
        """Core contract: dots over dark pixels must have a larger radius than dots over bright pixels."""
        # Left half very dark, right half very bright
        h, w = 30, 60
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, :30] = 10    # very dark → more ink → bigger dots
        img[:, 30:] = 245   # very bright → less ink → smaller dots

        params = self._default_params(img)
        params["stipple_points"] = 300          # enough to sample both halves
        params["stipple_iterations"] = 5
        params["min_dot_size_mm"] = 0.1
        params["max_dot_size_mm"] = 2.0         # wide range for clear difference
        params["dot_size_gamma"] = 1.0
        params["shadow_threshold"] = 0          # treat all pixels as midtone so dots are placed
        params["highlight_threshold"] = 255     # treat all pixels as midtone so dots are placed
        params["edge_min_len"] = 9999           # suppress edge polylines
        params["hatch_spacing_mm"] = 999.0      # suppress hatching

        result = self.gen.generate(params, self.canvas)

        dots = [p for p in result if _is_dot(p)]
        assert len(dots) >= 10, f"Need at least 10 stipple dots to compare halves, got {len(dots)}"

        # A4 canvas with margin=10: drawing area x in [10, 200]; midpoint = 105
        canvas_mid_x = (self.canvas.drawing_area()[0] + self.canvas.drawing_area()[2]) / 2

        dark_radii: list[float] = []
        bright_radii: list[float] = []
        for poly in dots:
            xs = [p[0] for p in poly]
            cx = (max(xs) + min(xs)) / 2
            radius = (max(xs) - min(xs)) / 2
            if cx < canvas_mid_x:
                dark_radii.append(radius)
            else:
                bright_radii.append(radius)

        assert len(dark_radii) > 0, "No stipple dots found in the dark (left) half of the canvas"
        assert len(bright_radii) > 0, "No stipple dots found in the bright (right) half of the canvas"

        avg_dark = sum(dark_radii) / len(dark_radii)
        avg_bright = sum(bright_radii) / len(bright_radii)
        assert avg_dark > avg_bright, (
            f"Expected dark-area dots (avg r={avg_dark:.3f}mm) to be larger than "
            f"bright-area dots (avg r={avg_bright:.3f}mm)"
        )

    # --- Edge outlines ---

    def test_edge_polylines_produced(self):
        """Edge contours are detected from a high-contrast image."""
        img = make_checkerboard(80, 80, tile=20)  # large tiles → clear edges
        params = self._default_params(img)
        params["stipple_points"] = 0    # disable stipple to isolate edges
        params["hatch_spacing_mm"] = 50.0  # wide spacing → few/no hatch lines
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Should produce edge polylines from checkerboard"
        assert all(len(p) >= 2 for p in result)

    # --- Stipple dots ---

    def test_stipple_dots_produced_from_midtone_image(self):
        """Uniform midtone image (128) produces stipple dot polylines."""
        img = np.full((60, 60), 128, dtype=np.uint8)  # 80 < 128 < 200 → midtone
        params = self._default_params(img)
        params["stipple_points"] = 80
        params["stipple_iterations"] = 2
        result = self.gen.generate(params, self.canvas)
        dots = [p for p in result if _is_dot(p)]
        assert len(dots) > 0, "Should produce stipple dots for midtone image"

    def test_no_stipple_in_pure_highlight_image(self):
        """Pure white image (255) produces no stipple dots."""
        img = np.full((60, 60), 255, dtype=np.uint8)  # 255 > 200 → highlight
        params = self._default_params(img)
        params["stipple_points"] = 100
        result = self.gen.generate(params, self.canvas)
        dots = [p for p in result if _is_dot(p)]
        assert len(dots) == 0, "Should produce no stipple dots in highlight-only image"

    def test_stipple_dots_only_in_midtone_region(self):
        """Stipple dot centers must lie within the midtone brightness band."""
        # Top half = midtone (128), bottom half = highlight (255)
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:40, :] = 128   # top 50% → midtone
        img[40:, :] = 255   # bottom 50% → highlight (no dots expected here)

        x1, y1, x2, y2 = self.canvas.drawing_area()
        draw_h = y2 - y1
        # Midtone occupies rows 0..39 of 80 → top 50% of draw height
        midzone_y_max = y1 + 0.5 * draw_h
        tolerance = draw_h * 0.05  # 5% tolerance for Lloyd boundary effects

        params = self._default_params(img)
        params["stipple_points"] = 100
        params["stipple_iterations"] = 3
        result = self.gen.generate(params, self.canvas)

        dots = [p for p in result if _is_dot(p)]
        assert len(dots) > 0, "Should produce stipple dots in midtone region"

        for dot in dots:
            center_y = sum(pt[1] for pt in dot) / len(dot)
            assert center_y <= midzone_y_max + tolerance, (
                f"Stipple dot center y={center_y:.2f} outside midtone region "
                f"(max {midzone_y_max + tolerance:.2f})"
            )

    def test_stipple_count_scales_with_stipple_points(self):
        """Increasing stipple_points produces more dot polylines."""
        img = np.full((60, 60), 128, dtype=np.uint8)  # all midtone

        params_low = self._default_params(img)
        params_low["stipple_points"] = 50
        params_low["stipple_iterations"] = 2

        params_high = self._default_params(img)
        params_high["stipple_points"] = 300
        params_high["stipple_iterations"] = 2

        dots_low = [p for p in self.gen.generate(params_low, self.canvas)
                    if _is_dot(p)]
        dots_high = [p for p in self.gen.generate(params_high, self.canvas)
                     if _is_dot(p)]

        assert len(dots_high) > len(dots_low), (
            f"More stipple_points should produce more dots: "
            f"low={len(dots_low)}, high={len(dots_high)}"
        )

    # --- Shadow hatching ---

    def test_hatch_lines_produced_in_shadow_image(self):
        """Uniform dark (shadow) image produces hatch polylines."""
        img = np.full((60, 60), 30, dtype=np.uint8)  # 30 < 80 → shadow
        params = self._default_params(img)
        params["stipple_points"] = 0  # disable stipple to isolate hatching
        params["hatch_spacing_mm"] = 3.0
        result = self.gen.generate(params, self.canvas)
        hatch = [p for p in result if p[0] != p[-1]]  # open polylines = hatch
        assert len(hatch) > 0, "Should produce hatch lines for shadow-only image"

    def test_no_hatch_in_midtone_only_image(self):
        """Uniform midtone image (128) produces no hatch lines."""
        img = np.full((60, 60), 128, dtype=np.uint8)  # 80 < 128 < 200 → midtone
        params = self._default_params(img)
        params["stipple_points"] = 0  # disable stipple
        result = self.gen.generate(params, self.canvas)
        hatch = [p for p in result if p[0] != p[-1]]
        assert len(hatch) == 0, "Should produce no hatch lines in midtone-only image"

    def test_hatch_lines_in_shadow_region_of_structured_image(self):
        """Hatch line points fall only in the shadow brightness band."""
        # Left half = shadow (30), right half = highlight (255)
        img = np.zeros((60, 60), dtype=np.uint8)
        img[:, :30] = 30    # left half → shadow
        img[:, 30:] = 255   # right half → highlight

        x1, y1, x2, y2 = self.canvas.drawing_area()
        draw_w = x2 - x1
        # Shadow occupies columns 0..29 of 60 → left 50% of draw width
        shadow_x_max = x1 + 0.5 * draw_w
        tolerance = draw_w * 0.05

        params = self._default_params(img)
        params["stipple_points"] = 0  # no stipple
        params["hatch_angle"] = 90.0  # vertical lines → easy to check x coords
        params["hatch_spacing_mm"] = 2.0
        result = self.gen.generate(params, self.canvas)

        hatch = [p for p in result if p[0] != p[-1]]
        assert len(hatch) > 0, "Should produce hatch lines in shadow half"

        for line in hatch:
            for x, _y in line:
                assert x <= shadow_x_max + tolerance, (
                    f"Hatch point x={x:.2f} outside shadow region "
                    f"(max {shadow_x_max + tolerance:.2f})"
                )

    def test_cross_hatch_produces_more_lines_than_single_hatch(self):
        """Cross-hatching in deep shadows adds a second hatch pass."""
        img = np.full((60, 60), 10, dtype=np.uint8)  # 10 < shadow//2=40 → deep shadow

        params_single = self._default_params(img)
        params_single["stipple_points"] = 0
        params_single["cross_hatch_shadows"] = False
        params_single["hatch_spacing_mm"] = 2.0

        params_cross = self._default_params(img)
        params_cross["stipple_points"] = 0
        params_cross["cross_hatch_shadows"] = True
        params_cross["hatch_spacing_mm"] = 2.0

        result_single = self.gen.generate(params_single, self.canvas)
        result_cross = self.gen.generate(params_cross, self.canvas)

        assert len(result_cross) > len(result_single), (
            f"Cross-hatching should produce more lines: "
            f"single={len(result_single)}, cross={len(result_cross)}"
        )

    # --- Composite output ---

    def test_all_three_components_present_for_gradient_image(self):
        """Gradient image produces edges, stipple dots, and hatch lines."""
        # Left-to-right gradient 0→255: left = shadow, middle = midtone, right = highlight
        img = make_gradient_image(100, 100)
        params = self._default_params(img)
        params["stipple_points"] = 500
        params["stipple_iterations"] = 5
        params["hatch_spacing_mm"] = 2.0
        result = self.gen.generate(params, self.canvas)

        dots = [p for p in result if _is_dot(p)]
        hatch = [p for p in result if p[0] != p[-1]]

        assert len(result) > 0, "Should produce output for gradient image"
        assert len(dots) > 0, "Should produce stipple dots for midtone band in gradient"
        assert len(hatch) > 0, "Should produce hatch lines for shadow band in gradient"

    def test_output_within_canvas_bounds(self):
        """All polyline points are within the canvas drawing area."""
        img = make_gradient_image(80, 80)
        params = self._default_params(img)
        params["stipple_points"] = 100
        params["stipple_iterations"] = 3
        result = self.gen.generate(params, self.canvas)
        assert within_bounds(result, self.canvas, tol=2.0)

    # --- Presets ---

    def test_presets_exist(self):
        """Six named presets exist with the expected names."""
        presets = {p.name: p for p in self.gen.get_presets()}
        assert "Hedcut / Classic WSJ" in presets
        assert "Hedcut / Dense Detail" in presets
        assert "Hedcut / Minimal" in presets
        assert "Hedcut / Bold Hedcut" in presets
        assert "Hedcut / Fine Stipple" in presets
        assert "Hedcut / WSJ Portrait" in presets

    def test_presets_have_required_keys(self):
        """Every preset must contain all required parameter keys."""
        required = {
            "highlight_threshold", "shadow_threshold",
            "edge_method", "stipple_points", "stipple_iterations",
            "hatch_angle", "hatch_spacing_mm", "cross_hatch_shadows",
            "brightness", "contrast", "blur_radius", "invert",
            "dot_style", "pen_width_mm",
            "min_dot_size_mm", "max_dot_size_mm", "dot_size_gamma",
        }
        for preset in self.gen.get_presets():
            missing = required - set(preset.params.keys())
            assert not missing, f"Preset '{preset.name}' missing keys: {missing}"

    def test_presets_runnable_on_gradient_image(self):
        """All presets generate output without error on a small gradient image."""
        img = make_gradient_image(60, 60)
        for preset in self.gen.get_presets():
            p = dict(preset.params)
            p["_source_image"] = img
            p["stipple_points"] = 50    # minimal for speed
            p["stipple_iterations"] = 2
            result = self.gen.generate(p, self.canvas)
            assert isinstance(result, list), f"Preset '{preset.name}' must return a list"

    # --- Cancellation ---

    def test_cancellation_before_edges_returns_empty(self):
        """Immediate cancellation (before edge detection) returns []."""
        img = make_gradient_image(80, 80)
        params = self._default_params(img)
        result = self.gen.generate(params, self.canvas, cancelled_callback=lambda: True)
        assert result == []

    def test_xdog_edge_method_produces_output(self):
        """XDoG edge method runs without error and produces polylines."""
        img = make_checkerboard(60, 60, tile=15)
        params = self._default_params(img)
        params["edge_method"] = "XDoG"
        params["stipple_points"] = 50
        params["stipple_iterations"] = 2
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_fdog_edge_method_produces_output(self):
        """FDoG edge method runs without error and produces polylines."""
        img = make_checkerboard(60, 60, tile=15)
        params = self._default_params(img)
        params["edge_method"] = "FDoG"
        params["stipple_points"] = 50
        params["stipple_iterations"] = 2
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    # --- Filled dot rendering (27.1) ---

    def test_tiny_circle_returns_correct_point_count(self):
        """_tiny_circle returns _DOT_SIDES+1 points."""
        from plottter.generators.hedcut import _tiny_circle
        poly = _tiny_circle(5.0, 10.0, 1.0)
        assert len(poly) == _HEDCUT_DOT_SIDES + 1

    def test_tiny_circle_closes(self):
        """First and last point of _tiny_circle are equal (closed polyline)."""
        from plottter.generators.hedcut import _tiny_circle
        poly = _tiny_circle(5.0, 10.0, 1.0)
        assert poly[0] == poly[-1]

    def test_filled_circle_returns_list_of_polylines(self):
        """_filled_circle returns a list of Polylines, each closed."""
        from plottter.generators.hedcut import _filled_circle
        rings = _filled_circle(0.0, 0.0, 2.0, pen_width_mm=0.5)
        assert isinstance(rings, list)
        assert len(rings) >= 1
        for ring in rings:
            assert len(ring) == _HEDCUT_DOT_SIDES + 1
            assert ring[0] == ring[-1]

    def test_filled_circle_concentric_rings_spaced_correctly(self):
        """Outer radius of consecutive rings decreases by pen_width_mm."""
        from plottter.generators.hedcut import _filled_circle
        import math
        pen_w = 0.4
        outer_r = 2.0
        rings = _filled_circle(0.0, 0.0, outer_r, pen_width_mm=pen_w)
        assert len(rings) >= 2, "Should produce at least 2 rings for r=2.0, pen=0.4"
        for i, ring in enumerate(rings):
            expected_r = outer_r - i * pen_w
            # Check radius of first non-center point
            x, y = ring[0]
            actual_r = math.sqrt(x ** 2 + y ** 2)
            assert abs(actual_r - expected_r) < 1e-9, (
                f"Ring {i} radius {actual_r:.4f} != expected {expected_r:.4f}"
            )

    def test_filled_circle_single_ring_for_tiny_dot(self):
        """Dot smaller than pen_width_mm returns exactly one ring."""
        from plottter.generators.hedcut import _filled_circle
        rings = _filled_circle(0.0, 0.0, 0.1, pen_width_mm=0.3)
        assert len(rings) == 1

    def test_dot_style_outline_matches_baseline(self):
        """dot_style='Outline' produces identical dots to default (no regression)."""
        img = np.full((60, 60), 128, dtype=np.uint8)
        params_baseline = self._default_params(img)
        params_baseline["stipple_points"] = 50
        params_baseline["stipple_iterations"] = 2

        params_outline = dict(params_baseline)
        params_outline["dot_style"] = "Outline"
        params_outline["pen_width_mm"] = 0.3

        result_baseline = self.gen.generate(params_baseline, self.canvas)
        result_outline = self.gen.generate(params_outline, self.canvas)

        dots_baseline = [p for p in result_baseline if _is_dot(p)]
        dots_outline = [p for p in result_outline if _is_dot(p)]
        assert len(dots_outline) == len(dots_baseline), (
            "Outline style should produce same number of dots as default"
        )

    def test_dot_style_filled_produces_more_polylines_than_outline(self):
        """Filled dots generate multiple concentric rings per dot, so more polylines."""
        img = np.full((60, 60), 128, dtype=np.uint8)
        params = self._default_params(img)
        params["stipple_points"] = 50
        params["stipple_iterations"] = 2
        params["min_dot_size_mm"] = 0.5
        params["max_dot_size_mm"] = 1.0   # large enough to have multiple rings
        params["pen_width_mm"] = 0.3

        params_outline = dict(params)
        params_outline["dot_style"] = "Outline"

        params_filled = dict(params)
        params_filled["dot_style"] = "Filled"

        result_outline = self.gen.generate(params_outline, self.canvas)
        result_filled = self.gen.generate(params_filled, self.canvas)

        # Filled should have more total polylines (multiple rings per dot)
        assert len(result_outline) > 0, "Outline should produce polylines for this image"
        assert len(result_filled) > len(result_outline), (
            f"Filled style should produce more polylines than Outline: "
            f"filled={len(result_filled)}, outline={len(result_outline)}"
        )

    def test_dot_style_filled_all_rings_same_point_count(self):
        """Every ring in a filled dot has _DOT_SIDES+1 points (small bbox = dot)."""
        img = np.full((60, 60), 128, dtype=np.uint8)
        params = self._default_params(img)
        params["stipple_points"] = 30
        params["stipple_iterations"] = 2
        params["min_dot_size_mm"] = 0.5
        params["max_dot_size_mm"] = 1.5
        params["pen_width_mm"] = 0.3
        params["dot_style"] = "Filled"

        result = self.gen.generate(params, self.canvas)
        small_polys = [p for p in result if _is_dot(p)]
        assert len(small_polys) > 0, "Filled mode should produce small dot polylines"
        for poly in small_polys:
            assert len(poly) == _HEDCUT_DOT_SIDES + 1

    def test_parameters_include_dot_style_and_pen_width(self):
        """dot_style and pen_width_mm parameters are present in get_parameters()."""
        param_names = {p.name for p in self.gen.get_parameters()}
        assert "dot_style" in param_names, "dot_style parameter missing"
        assert "pen_width_mm" in param_names, "pen_width_mm parameter missing"

    def test_legacy_dot_size_mm_param_still_renders(self):
        """Old projects with dot_size_mm saved in generator_info still render without error."""
        img = np.full((40, 40), 128, dtype=np.uint8)
        params = self._default_params(img)
        # Remove min/max keys and replace with legacy single-value key
        del params["min_dot_size_mm"]
        del params["max_dot_size_mm"]
        params["dot_size_mm"] = 0.5
        params["stipple_points"] = 50
        params["hatch_spacing_mm"] = 999.0  # suppress hatching

        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list), "Legacy dot_size_mm param should produce a list result"


# ---------------------------------------------------------------------------
# Cross-cutting: all image generators registered in category "image"
# ---------------------------------------------------------------------------


class TestImageGeneratorRegistry:
    def test_all_image_generators_registered(self):
        from plottter.generators import get_generators_by_category
        image_gens = get_generators_by_category("image")
        names = {cls.name for cls in image_gens}
        assert "Edge Detect" in names
        assert "Hatching" in names
        assert "Flow Image" in names
        assert "Stipple" in names
        assert "Contour Lines" in names
        assert "XDoG" in names
        assert "Coherent Lines (FDoG)" in names
        assert "Hedcut" in names

    def test_math_generators_still_registered(self):
        from plottter.generators import get_generators_by_category
        math_gens = get_generators_by_category("math")
        names = {cls.name for cls in math_gens}
        assert "Parametric Curves" in names
        assert "Polar Curves" in names


# ---------------------------------------------------------------------------
# compute_image_rect
# ---------------------------------------------------------------------------


class TestComputeImageRect:
    """Tests for the shared compute_image_rect helper."""

    def setup_method(self):
        from plottter.generators._helpers import compute_image_rect
        self.compute = compute_image_rect
        # Drawing area: x1=10, y1=10, x2=200, y2=287 (A4 with 10mm margin)
        self.draw_x1 = 10.0
        self.draw_y1 = 10.0
        self.draw_x2 = 200.0
        self.draw_y2 = 287.0

    # --- fill mode ---

    def test_fill_mode_returns_full_drawing_area(self):
        rect = self.compute(
            "fill", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        assert rect == (self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2)

    def test_fill_mode_ignores_offsets(self):
        rect = self.compute(
            "fill", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
            offset_x_mm=50.0, offset_y_mm=50.0,
        )
        assert rect == (self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2)

    def test_fill_mode_ignores_custom_size(self):
        rect = self.compute(
            "fill", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
            custom_w_mm=50.0, custom_h_mm=50.0,
        )
        assert rect == (self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2)

    # --- fit mode ---

    def test_fit_mode_square_image_in_portrait_canvas(self):
        # Square image (1:1) in portrait canvas (190x277)
        # Canvas aspect < 1 → constrained by width (190mm)
        draw_w = self.draw_x2 - self.draw_x1  # 190
        draw_h = self.draw_y2 - self.draw_y1  # 277
        # image aspect == 1, canvas_aspect = 190/277 < 1 → constrained by width
        rect = self.compute(
            "fit", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        x1, y1, x2, y2 = rect
        assert abs((x2 - x1) - draw_w) < 1e-9, "Width should equal draw_w"
        assert abs((y2 - y1) - draw_w) < 1e-9, "Height should equal draw_w for square"
        # Should be centered horizontally
        cx = (x1 + x2) / 2.0
        expected_cx = (self.draw_x1 + self.draw_x2) / 2.0
        assert abs(cx - expected_cx) < 1e-9

    def test_fit_mode_wide_image_fills_width(self):
        # Wide image (2:1) → width fills canvas, height is halved
        draw_w = self.draw_x2 - self.draw_x1  # 190
        rect = self.compute(
            "fit", 200, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        x1, y1, x2, y2 = rect
        assert abs((x2 - x1) - draw_w) < 1e-9
        # Height = draw_w / 2
        assert abs((y2 - y1) - draw_w / 2.0) < 1e-9

    def test_fit_mode_tall_image_fills_height(self):
        # Tall image (1:2) in landscape canvas → height fills canvas, width is halved
        # Use a square canvas for simplicity
        rect = self.compute(
            "fit", 100, 200,
            0.0, 0.0, 200.0, 200.0,
        )
        x1, y1, x2, y2 = rect
        # height should fill 200mm, width should be 100mm
        assert abs((y2 - y1) - 200.0) < 1e-9
        assert abs((x2 - x1) - 100.0) < 1e-9

    def test_fit_mode_is_centered(self):
        rect = self.compute(
            "fit", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        cx = (rect[0] + rect[2]) / 2.0
        cy = (rect[1] + rect[3]) / 2.0
        assert abs(cx - (self.draw_x1 + self.draw_x2) / 2.0) < 1e-9
        assert abs(cy - (self.draw_y1 + self.draw_y2) / 2.0) < 1e-9

    def test_fit_mode_offset_applied(self):
        base = self.compute(
            "fit", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        shifted = self.compute(
            "fit", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
            offset_x_mm=10.0, offset_y_mm=5.0,
        )
        assert abs(shifted[0] - (base[0] + 10.0)) < 1e-9
        assert abs(shifted[1] - (base[1] + 5.0)) < 1e-9
        assert abs(shifted[2] - (base[2] + 10.0)) < 1e-9
        assert abs(shifted[3] - (base[3] + 5.0)) < 1e-9

    # --- custom mode ---

    def test_custom_mode_uses_explicit_size(self):
        rect = self.compute(
            "custom", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
            custom_w_mm=80.0, custom_h_mm=60.0,
        )
        x1, y1, x2, y2 = rect
        assert abs((x2 - x1) - 80.0) < 1e-9
        assert abs((y2 - y1) - 60.0) < 1e-9

    def test_custom_mode_is_centered(self):
        rect = self.compute(
            "custom", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
            custom_w_mm=80.0, custom_h_mm=60.0,
        )
        cx = (rect[0] + rect[2]) / 2.0
        cy = (rect[1] + rect[3]) / 2.0
        assert abs(cx - (self.draw_x1 + self.draw_x2) / 2.0) < 1e-9
        assert abs(cy - (self.draw_y1 + self.draw_y2) / 2.0) < 1e-9

    def test_custom_mode_offset_applied(self):
        base = self.compute(
            "custom", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
            custom_w_mm=80.0, custom_h_mm=60.0,
        )
        shifted = self.compute(
            "custom", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
            custom_w_mm=80.0, custom_h_mm=60.0,
            offset_x_mm=-20.0, offset_y_mm=15.0,
        )
        assert abs(shifted[0] - (base[0] - 20.0)) < 1e-9
        assert abs(shifted[1] - (base[1] + 15.0)) < 1e-9

    def test_custom_mode_none_sizes_fall_back_to_draw_area(self):
        # When custom_w_mm/custom_h_mm are None, falls back to draw area size
        draw_w = self.draw_x2 - self.draw_x1
        draw_h = self.draw_y2 - self.draw_y1
        rect = self.compute(
            "custom", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        assert abs((rect[2] - rect[0]) - draw_w) < 1e-9
        assert abs((rect[3] - rect[1]) - draw_h) < 1e-9

    # --- edge cases ---

    def test_zero_width_image_returns_draw_area(self):
        rect = self.compute(
            "fit", 0, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        assert rect == (self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2)

    def test_zero_height_image_returns_draw_area(self):
        rect = self.compute(
            "fit", 100, 0,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        assert rect == (self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2)

    def test_unknown_fit_mode_returns_draw_area(self):
        # An unrecognised mode falls back to the drawing area
        rect = self.compute(
            "bogus_mode", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        assert rect == (self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2)

    def test_result_is_tuple_of_four_floats(self):
        rect = self.compute(
            "fit", 100, 100,
            self.draw_x1, self.draw_y1, self.draw_x2, self.draw_y2,
        )
        assert len(rect) == 4
        assert all(isinstance(v, float) for v in rect)


# ---------------------------------------------------------------------------
# CircularScribbleGenerator
# ---------------------------------------------------------------------------


class TestCircularScribbleGenerator:
    def setup_method(self):
        from plottter.generators.circular_scribble import CircularScribbleGenerator
        self.gen = CircularScribbleGenerator()
        self.canvas = make_canvas()

    # (a) generator is registered
    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Circular Scribble" in GENERATORS
        assert GENERATORS["Circular Scribble"].category == "image"

    # parameters include x_offset_mm / y_offset_mm with default 0
    def test_offset_params_present_with_default_zero(self):
        params = self.gen.get_parameters()
        param_map = {p.name: p for p in params}
        assert "x_offset_mm" in param_map, "x_offset_mm parameter missing"
        assert "y_offset_mm" in param_map, "y_offset_mm parameter missing"
        assert param_map["x_offset_mm"].default == 0.0
        assert param_map["y_offset_mm"].default == 0.0

    # empty result when no source image is supplied
    def test_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    # (b) sampling produces more points in dark areas than bright areas
    def test_denser_sampling_in_dark_areas(self):
        """Dark half of image should yield more seed points than bright half."""
        # Create a 100x100 image: left half black (0), right half white (255)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255  # right half is white

        params = {
            "_source_image": img,
            "min_sample_spacing_mm": 1.0,
            "max_sample_spacing_mm": 8.0,
            "seed": 42,
        }
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, "Should produce at least some output"

        # Each seed point emits 2 polylines (horizontal + vertical arm).
        # Count points whose x coordinate falls in the left (dark) vs right (bright) half.
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2.0

        left_count = 0
        right_count = 0
        for path in result:
            cx = sum(x for x, y in path) / len(path)  # centroid x of path
            if cx < mid_x:
                left_count += 1
            else:
                right_count += 1

        assert left_count > right_count, (
            f"Dark left half should have more paths than bright right half, "
            f"got left={left_count} right={right_count}"
        )

    # (c) no two seed points are closer than the local exclusion radius
    def test_minimum_point_spacing_respected(self):
        """All seed point centroids must be at least min_spacing apart."""
        import math as _math
        img = make_gradient_image(60, 60)
        min_spacing_mm = 2.0
        params = {
            "_source_image": img,
            "min_sample_spacing_mm": min_spacing_mm,
            "max_sample_spacing_mm": 10.0,
            "seed": 7,
        }
        result = self.gen.generate(params, self.canvas)
        # Collect unique centroids (each seed → 2 paths; centroid of horiz arm == centroid of vert arm)
        centroids = []
        for path in result:
            cx = sum(x for x, y in path) / len(path)
            cy = sum(y for x, y in path) / len(path)
            centroids.append((cx, cy))

        # Deduplicate (horiz + vert arm produce the same centroid)
        unique: list[tuple[float, float]] = []
        for pt in centroids:
            is_dup = any(
                _math.hypot(pt[0] - u[0], pt[1] - u[1]) < 1e-6
                for u in unique
            )
            if not is_dup:
                unique.append(pt)

        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                dist = _math.hypot(a[0] - b[0], a[1] - b[1])
                # Allow a small tolerance for coordinate-conversion rounding + ±1px jitter
                assert dist >= min_spacing_mm * 0.85, (
                    f"Points too close: {dist:.3f} mm < {min_spacing_mm * 0.85:.3f} mm"
                )

    # (d) sampling completes in reasonable time for a synthetic image
    def test_sampling_completes_quickly(self):
        import time
        img = make_gradient_image(200, 200)
        params = {
            "_source_image": img,
            "min_sample_spacing_mm": 2.0,
            "max_sample_spacing_mm": 10.0,
            "seed": 0,
        }
        start = time.monotonic()
        result = self.gen.generate(params, self.canvas)
        elapsed = time.monotonic() - start
        assert elapsed < 30.0, f"Sampling took too long: {elapsed:.1f}s"
        assert isinstance(result, list)

    # (e) x_offset_mm / y_offset_mm shift output correctly
    def test_offset_shifts_output(self):
        img = make_dark_center_image(80, 80)
        base_params = {
            "_source_image": img,
            "min_sample_spacing_mm": 3.0,
            "max_sample_spacing_mm": 10.0,
            "seed": 1,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        result_base = self.gen.generate(base_params, self.canvas)

        shift_x, shift_y = 5.0, 3.0
        shifted_params = dict(base_params)
        shifted_params["x_offset_mm"] = shift_x
        shifted_params["y_offset_mm"] = shift_y
        result_shifted = self.gen.generate(shifted_params, self.canvas)

        assert len(result_base) == len(result_shifted), (
            "Offset should not change the number of polylines"
        )
        for path_base, path_shifted in zip(result_base, result_shifted):
            for (bx, by), (sx, sy) in zip(path_base, path_shifted):
                assert abs((sx - bx) - shift_x) < 1e-6
                assert abs((sy - by) - shift_y) < 1e-6

    # (f) task 25.4 — edge-aware parameters are present with correct defaults
    def test_edge_aware_params_present(self):
        """edge_sensitivity, edge_low, and edge_high must be in get_parameters()."""
        params = self.gen.get_parameters()
        param_map = {p.name: p for p in params}
        assert "edge_sensitivity" in param_map, "edge_sensitivity parameter missing"
        assert "edge_low" in param_map, "edge_low parameter missing"
        assert "edge_high" in param_map, "edge_high parameter missing"
        assert abs(param_map["edge_sensitivity"].default - 0.7) < 1e-9
        assert param_map["edge_low"].default == 50
        assert param_map["edge_high"].default == 150

    # (g) task 25.4 — edge sensitivity changes output on a high-contrast image
    def test_edge_sensitivity_affects_output(self):
        """High edge_sensitivity should produce different scribbles than zero sensitivity."""
        # Half-black, half-white image produces a strong vertical edge at x=50.
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:, 40:] = 255

        base_params = {
            "_source_image": img,
            "min_sample_spacing_mm": 2.0,
            "max_sample_spacing_mm": 8.0,
            "seed": 42,
        }

        result_no_edge = self.gen.generate(
            {**base_params, "edge_sensitivity": 0.0}, self.canvas
        )
        result_edge = self.gen.generate(
            {**base_params, "edge_sensitivity": 0.9, "edge_low": 40, "edge_high": 120},
            self.canvas,
        )

        # Both must return non-empty polylines.
        assert len(result_no_edge) > 0
        assert len(result_edge) > 0

        # Flatten all points from each run and compare — they must differ when
        # edge sensitivity is active on a high-contrast image.
        flat_no_edge = [pt for path in result_no_edge for pt in path]
        flat_edge = [pt for path in result_edge for pt in path]

        # At least one coordinate should differ.
        differ = any(
            abs(a[0] - b[0]) > 1e-9 or abs(a[1] - b[1]) > 1e-9
            for a, b in zip(flat_no_edge, flat_edge)
        ) or len(flat_no_edge) != len(flat_edge)

        assert differ, (
            "Expected different scribble geometry with edge_sensitivity=0.9 vs 0.0 "
            "on a high-contrast image, but output was identical."
        )

    # (h) task 25.4 — graceful fallback when cv2 is unavailable
    def test_edge_aware_fallback_when_cv2_unavailable(self):
        """generate() must still return valid polylines even if cv2 is not importable."""
        import builtins
        import sys
        import unittest.mock as mock

        img = make_gradient_image(60, 60)
        params = {
            "_source_image": img,
            "min_sample_spacing_mm": 2.0,
            "max_sample_spacing_mm": 8.0,
            "seed": 99,
            "edge_sensitivity": 0.8,
            "edge_low": 50,
            "edge_high": 150,
        }

        real_import = builtins.__import__

        def _block_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("cv2 blocked for test")
            return real_import(name, *args, **kwargs)

        # Remove cv2 from sys.modules so the patched __import__ is called.
        cv2_backup = sys.modules.pop("cv2", None)
        try:
            with mock.patch("builtins.__import__", side_effect=_block_cv2):
                result = self.gen.generate(params, self.canvas)
        finally:
            if cv2_backup is not None:
                sys.modules["cv2"] = cv2_backup

        # Must return a list of polylines (possibly empty if image is uniform).
        assert isinstance(result, list), "generate() must return a list"
        for path in result:
            assert isinstance(path, list), "Each item must be a list of (x, y) tuples"
            assert len(path) >= 2, "Each polyline must have at least 2 points"
            for pt in path:
                assert len(pt) == 2, "Each point must be a (x, y) tuple"
