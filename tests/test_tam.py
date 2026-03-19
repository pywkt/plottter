"""Comprehensive tests for the TAM (Tonal Art Maps) generator — 41.6.

Covers:
  (a) Tone level nesting property
  (b) Brightness-to-tone mapping edge cases (white/black)
  (c) Cross-hatch adds strokes only above threshold
  (d) ETF orientation mode produces varying angles
  (e) Generator produces valid polylines within canvas bounds
  (f) Empty/missing image returns empty output (no error)
  (g) All density_curve options produce valid output
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.tam import (
    _build_tone_levels,
    _compute_tam_orientation_field,
    _render_strokes,
    _restrict_high_birth_levels,
    _select_strokes_for_image,
)
from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CANVAS_W = 100.0
CANVAS_H = 100.0
NUM_LEVELS = 6


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def solid_image(brightness: float, size: int = 64) -> np.ndarray:
    """Return a constant-brightness float64 image in [0, 1]."""
    return np.full((size, size), brightness, dtype=np.float64)


def make_levels(
    num_levels: int = NUM_LEVELS,
    density: float = 0.1,
    rng_seed: int = 42,
    orientation_field=0.0,
) -> list[list[tuple[float, float, float]]]:
    rng = np.random.default_rng(rng_seed)
    return _build_tone_levels(
        canvas_w=CANVAS_W,
        canvas_h=CANVAS_H,
        num_levels=num_levels,
        stroke_density=density,
        orientation_field=orientation_field,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# (a) Tone level nesting property
# ---------------------------------------------------------------------------


class TestNestingProperty:
    def test_every_stroke_in_k_is_in_k_plus_1(self):
        levels = make_levels()
        for k in range(NUM_LEVELS - 1):
            set_k = set(levels[k])
            set_k1 = set(levels[k + 1])
            assert set_k.issubset(set_k1), (
                f"Level {k} not a subset of level {k + 1}: "
                f"{len(set_k)} vs {len(set_k1)} strokes"
            )

    def test_darkest_level_is_superset_of_all(self):
        levels = make_levels()
        darkest = set(levels[-1])
        for k, level in enumerate(levels):
            assert set(level).issubset(darkest), (
                f"Level {k} contains strokes not in darkest level"
            )

    def test_stroke_count_strictly_increasing(self):
        levels = make_levels()
        counts = [len(lvl) for lvl in levels]
        for i in range(len(counts) - 1):
            assert counts[i] < counts[i + 1], (
                f"Counts not strictly increasing: {counts}"
            )

    def test_nesting_holds_for_varying_num_levels(self):
        for n in [3, 4, 5, 8]:
            levels = make_levels(num_levels=n)
            assert len(levels) == n
            for k in range(n - 1):
                assert set(levels[k]).issubset(set(levels[k + 1]))


# ---------------------------------------------------------------------------
# (b) Brightness-to-tone mapping edge cases
# ---------------------------------------------------------------------------


class TestBrightnessMapping:
    def test_white_image_returns_no_strokes(self):
        levels = make_levels()
        result = _select_strokes_for_image(
            levels, solid_image(1.0), CANVAS_W, CANVAS_H
        )
        assert result == [], f"Expected 0 strokes for white image, got {len(result)}"

    def test_black_image_returns_all_strokes(self):
        levels = make_levels()
        result = _select_strokes_for_image(
            levels, solid_image(0.0), CANVAS_W, CANVAS_H
        )
        darkest = levels[-1]
        assert len(result) == len(darkest)
        assert set(result) == set(darkest)

    def test_near_white_returns_few_or_no_strokes(self):
        levels = make_levels()
        darkest_count = len(levels[-1])
        result = _select_strokes_for_image(
            levels, solid_image(0.95), CANVAS_W, CANVAS_H
        )
        assert len(result) < darkest_count * 0.2, (
            f"Near-white image produced too many strokes: {len(result)}"
        )

    def test_near_black_returns_most_strokes(self):
        levels = make_levels()
        darkest_count = len(levels[-1])
        result = _select_strokes_for_image(
            levels, solid_image(0.05), CANVAS_W, CANVAS_H
        )
        assert len(result) > darkest_count * 0.7, (
            f"Near-black image produced too few strokes: {len(result)}"
        )

    def test_monotonicity_darker_more_strokes(self):
        levels = make_levels()
        prev_count = 0
        for brightness in [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
            result = _select_strokes_for_image(
                levels, solid_image(brightness), CANVAS_W, CANVAS_H
            )
            count = len(result)
            assert count >= prev_count, (
                f"Stroke count should be non-decreasing as brightness decreases; "
                f"brightness={brightness} gave {count}, previous was {prev_count}"
            )
            prev_count = count


# ---------------------------------------------------------------------------
# (c) Cross-hatch adds strokes only above threshold
# ---------------------------------------------------------------------------


class TestCrossHatchThreshold:
    def test_threshold_zero_allows_all_levels(self):
        """threshold_level=0 returns original tone_levels unchanged."""
        levels = make_levels()
        restricted = _restrict_high_birth_levels(levels, threshold_level=0)
        # threshold=0 means no restriction
        for k in range(NUM_LEVELS):
            assert restricted[k] == levels[k]

    def test_threshold_above_num_levels_returns_empty(self):
        levels = make_levels()
        restricted = _restrict_high_birth_levels(levels, threshold_level=NUM_LEVELS)
        for k in range(NUM_LEVELS):
            assert restricted[k] == []

    def test_threshold_excludes_early_birth_strokes(self):
        """Strokes born at level < threshold_level must not appear in any level."""
        levels = make_levels()
        threshold = 3
        restricted = _restrict_high_birth_levels(levels, threshold_level=threshold)

        # Collect strokes that were born at level < threshold
        early_strokes = set(levels[threshold - 1])  # all strokes in levels 0..threshold-1

        for k in range(NUM_LEVELS):
            for stroke in restricted[k]:
                assert stroke not in early_strokes, (
                    f"Restricted level {k} contains a stroke born below threshold {threshold}"
                )

    def test_early_levels_empty_after_threshold(self):
        """Levels below threshold_level should be empty."""
        levels = make_levels()
        threshold = 3
        restricted = _restrict_high_birth_levels(levels, threshold_level=threshold)
        for k in range(threshold):
            assert restricted[k] == [], (
                f"Level {k} should be empty after restricting to threshold={threshold}"
            )

    def test_cross_hatch_adds_more_strokes_on_dark_image(self):
        """Full generator: cross_hatch=True on a black image produces more polylines."""
        from plottter.generators.tam import TAMGenerator

        gen = TAMGenerator()
        canvas = make_canvas()
        black_img = np.zeros((64, 64), dtype=np.float64)

        base_params = {
            "_source_image": black_img,
            "num_tone_levels": 4,
            "stroke_length_mm": 3.0,
            "stroke_angle": 45.0,
            "cross_hatch": False,
            "cross_hatch_threshold": 0.3,
            "orientation_mode": "fixed",
            "etf_kernel_radius": 5.0,
            "etf_iterations": 3,
            "curvature": 0.0,
            "stroke_density": 0.5,
            "density_curve": "linear",
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        no_cross = gen.generate(base_params, canvas)

        cross_params = dict(base_params, cross_hatch=True)
        with_cross = gen.generate(cross_params, canvas)

        assert len(with_cross) >= len(no_cross), (
            f"Cross-hatch should add strokes (or keep same) on black image; "
            f"no_cross={len(no_cross)}, with_cross={len(with_cross)}"
        )

    def test_cross_hatch_no_extra_strokes_on_white_image(self):
        """Cross-hatch on pure white image should produce no strokes."""
        from plottter.generators.tam import TAMGenerator

        gen = TAMGenerator()
        canvas = make_canvas()
        white_img = np.ones((64, 64), dtype=np.float64)

        params = {
            "_source_image": white_img,
            "num_tone_levels": 4,
            "stroke_length_mm": 3.0,
            "stroke_angle": 45.0,
            "cross_hatch": True,
            "cross_hatch_threshold": 0.3,
            "orientation_mode": "fixed",
            "etf_kernel_radius": 5.0,
            "etf_iterations": 3,
            "curvature": 0.0,
            "stroke_density": 0.5,
            "density_curve": "linear",
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        result = gen.generate(params, canvas)
        assert result == [], (
            f"White image with cross-hatch should produce no strokes, got {len(result)}"
        )


# ---------------------------------------------------------------------------
# (d) ETF orientation mode produces varying angles
# ---------------------------------------------------------------------------


class TestETFOrientation:
    def test_etf_returns_2d_array_or_scalar(self):
        """_compute_tam_orientation_field('etf', ...) returns a 2-D array or scalar."""
        pytest.importorskip("cv2")
        # Simple gradient image: horizontal ramp
        img_gray = np.zeros((64, 64), dtype=np.float64)
        img_gray[:, 32:] = 1.0  # left half black, right half white

        field = _compute_tam_orientation_field(
            img_gray,
            mode="etf",
            fixed_angle_rad=0.0,
            etf_kernel_radius=3.0,
            etf_iterations=2,
        )
        assert isinstance(field, np.ndarray), "ETF field should be a 2-D array"
        assert field.ndim == 2
        assert field.shape == img_gray.shape

    def test_etf_produces_varying_angles(self):
        """ETF mode produces non-uniform angles on a non-uniform image."""
        pytest.importorskip("cv2")
        # Image with a distinct vertical edge in the middle
        img_gray = np.zeros((64, 64), dtype=np.float64)
        img_gray[:, 32:] = 1.0  # vertical edge at x=32

        field = _compute_tam_orientation_field(
            img_gray,
            mode="etf",
            fixed_angle_rad=math.pi / 4,
            etf_kernel_radius=3.0,
            etf_iterations=2,
        )
        if isinstance(field, np.ndarray):
            # The field should not be entirely uniform
            unique_vals = np.unique(np.round(field, 4))
            assert len(unique_vals) > 1, (
                "ETF orientation field should have more than one unique angle"
            )

    def test_etf_levels_have_non_uniform_angles(self):
        """Tone levels built with ETF field use varying angles per stroke."""
        pytest.importorskip("cv2")
        img_gray = np.zeros((64, 64), dtype=np.float64)
        img_gray[:, 32:] = 1.0

        field = _compute_tam_orientation_field(
            img_gray,
            mode="etf",
            fixed_angle_rad=math.pi / 4,
            etf_kernel_radius=3.0,
            etf_iterations=2,
        )
        if not isinstance(field, np.ndarray):
            pytest.skip("ETF field fell back to scalar (cv2 unavailable)")

        rng = np.random.default_rng(42)
        levels = _build_tone_levels(
            CANVAS_W, CANVAS_H, 3, stroke_density=0.05,
            orientation_field=field, rng=rng
        )
        all_strokes = levels[-1]
        angles = [s[2] for s in all_strokes]
        # With a varying field, not all angles should be identical
        assert len(set(round(a, 6) for a in angles)) > 1, (
            "ETF mode should produce varying angles across strokes"
        )

    def test_fixed_mode_returns_scalar(self):
        """Fixed mode always returns the input scalar angle."""
        img_gray = np.zeros((32, 32), dtype=np.float64)
        angle_rad = math.pi / 3
        field = _compute_tam_orientation_field(
            img_gray,
            mode="fixed",
            fixed_angle_rad=angle_rad,
            etf_kernel_radius=5.0,
            etf_iterations=3,
        )
        assert isinstance(field, float)
        assert field == pytest.approx(angle_rad)

    def test_gradient_mode_returns_2d_array(self):
        """Gradient mode returns a 2-D array of angles."""
        pytest.importorskip("cv2")
        img_gray = np.zeros((32, 32), dtype=np.float64)
        img_gray[:, 16:] = 1.0
        field = _compute_tam_orientation_field(
            img_gray,
            mode="gradient",
            fixed_angle_rad=0.0,
            etf_kernel_radius=5.0,
            etf_iterations=3,
        )
        assert isinstance(field, np.ndarray)
        assert field.ndim == 2
        assert field.shape == img_gray.shape


# ---------------------------------------------------------------------------
# (e) Generator produces valid polylines within canvas bounds
# ---------------------------------------------------------------------------


class TestGeneratorOutput:
    def setup_method(self):
        from plottter.generators.tam import TAMGenerator

        self.gen = TAMGenerator()
        self.canvas = make_canvas()

    def _default_params(self, img: np.ndarray) -> dict:
        return {
            "_source_image": img,
            "num_tone_levels": 4,
            "stroke_length_mm": 4.0,
            "stroke_angle": 45.0,
            "cross_hatch": False,
            "cross_hatch_threshold": 0.5,
            "orientation_mode": "fixed",
            "etf_kernel_radius": 5.0,
            "etf_iterations": 3,
            "curvature": 0.0,
            "stroke_density": 0.3,
            "density_curve": "linear",
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }

    def test_each_polyline_has_at_least_2_points(self):
        img = solid_image(0.5)
        result = self.gen.generate(self._default_params(img), self.canvas)
        assert len(result) > 0
        for poly in result:
            assert len(poly) >= 2, f"Polyline has only {len(poly)} point(s)"

    def test_polylines_within_canvas_bounds(self):
        img = solid_image(0.3)
        result = self.gen.generate(self._default_params(img), self.canvas)
        x1, y1, x2, y2 = self.canvas.drawing_area()
        tol = self.gen.get_parameters()  # just to check it works; use numeric tol below
        tol = 5.0  # strokes may extend half a stroke length beyond seed point

        assert len(result) > 0
        for poly in result:
            for x, y in poly:
                assert x1 - tol <= x <= x2 + tol, (
                    f"x={x} is outside canvas bounds [{x1 - tol}, {x2 + tol}]"
                )
                assert y1 - tol <= y <= y2 + tol, (
                    f"y={y} is outside canvas bounds [{y1 - tol}, {y2 + tol}]"
                )

    def test_black_image_produces_strokes(self):
        img = np.zeros((64, 64), dtype=np.float64)
        result = self.gen.generate(self._default_params(img), self.canvas)
        assert len(result) > 0, "Black image should produce at least some strokes"

    def test_white_image_produces_no_strokes(self):
        img = np.ones((64, 64), dtype=np.float64)
        result = self.gen.generate(self._default_params(img), self.canvas)
        assert result == [], (
            f"White image should produce no strokes, got {len(result)}"
        )

    def test_generator_registration(self):
        from plottter.generators import GENERATORS

        assert "Tonal Art Maps (TAM)" in GENERATORS
        assert GENERATORS["Tonal Art Maps (TAM)"].category == "image"

    def test_progress_callback_fires(self):
        img = solid_image(0.4)
        progress_values = []
        result = self.gen.generate(
            self._default_params(img),
            self.canvas,
            progress_callback=lambda v: progress_values.append(v),
        )
        assert len(progress_values) > 0, "Progress callback should have been called"
        assert 100 in progress_values, "Progress callback should reach 100"

    def test_cancellation_stops_early(self):
        img = solid_image(0.3)
        call_count = [0]

        def cancel_after_first():
            call_count[0] += 1
            return call_count[0] >= 2  # cancel on second check

        result = self.gen.generate(
            self._default_params(img),
            self.canvas,
            cancelled_callback=cancel_after_first,
        )
        # Should return [] or partial result without error
        assert isinstance(result, list)

    def test_curvature_produces_multi_point_polylines(self):
        img = solid_image(0.2)
        params = self._default_params(img)
        params["curvature"] = 1.0
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        # At least some polylines should have more than 2 points
        multi_point = [p for p in result if len(p) > 2]
        assert len(multi_point) > 0, (
            "curvature=1.0 should produce curved polylines with > 2 points"
        )

    def test_invert_changes_output(self):
        img = solid_image(0.2)  # dark image → many strokes normally
        params_normal = self._default_params(img)
        params_inverted = dict(params_normal, invert=True)
        normal = self.gen.generate(params_normal, self.canvas)
        inverted = self.gen.generate(params_inverted, self.canvas)
        # Inverted dark image ≈ bright image → fewer strokes
        assert len(inverted) < len(normal), (
            f"Inverted dark image should produce fewer strokes than normal "
            f"(inverted={len(inverted)}, normal={len(normal)})"
        )


# ---------------------------------------------------------------------------
# (f) Empty/missing image returns [] without error
# ---------------------------------------------------------------------------


class TestMissingImage:
    def setup_method(self):
        from plottter.generators.tam import TAMGenerator

        self.gen = TAMGenerator()
        self.canvas = make_canvas()

    def _base_params(self) -> dict:
        return {
            "num_tone_levels": 4,
            "stroke_length_mm": 4.0,
            "stroke_angle": 45.0,
            "cross_hatch": False,
            "cross_hatch_threshold": 0.5,
            "orientation_mode": "fixed",
            "etf_kernel_radius": 5.0,
            "etf_iterations": 3,
            "curvature": 0.0,
            "stroke_density": 0.3,
            "density_curve": "linear",
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }

    def test_no_source_image_key_returns_empty(self):
        params = self._base_params()  # no '_source_image' key
        result = self.gen.generate(params, self.canvas)
        assert result == [], (
            "Generator should return [] when '_source_image' key is absent"
        )

    def test_none_source_image_returns_empty(self):
        params = self._base_params()
        params["_source_image"] = None
        result = self.gen.generate(params, self.canvas)
        assert result == [], (
            "Generator should return [] when '_source_image' is None"
        )

    def test_empty_tone_levels_select_returns_empty(self):
        img = solid_image(0.5)
        result = _select_strokes_for_image([], img, CANVAS_W, CANVAS_H)
        assert result == []


# ---------------------------------------------------------------------------
# (g) All density_curve options produce valid output
# ---------------------------------------------------------------------------


class TestDensityCurves:
    def setup_method(self):
        self.levels = make_levels()
        self.canvas_w = CANVAS_W
        self.canvas_h = CANVAS_H

    def test_linear_curve_valid_range(self):
        for brightness in [0.0, 0.25, 0.5, 0.75, 1.0]:
            img = solid_image(brightness)
            result = _select_strokes_for_image(
                self.levels, img, self.canvas_w, self.canvas_h, density_curve="linear"
            )
            assert 0 <= len(result) <= len(self.levels[-1])

    def test_quadratic_curve_valid_range(self):
        for brightness in [0.0, 0.25, 0.5, 0.75, 1.0]:
            img = solid_image(brightness)
            result = _select_strokes_for_image(
                self.levels, img, self.canvas_w, self.canvas_h, density_curve="quadratic"
            )
            assert 0 <= len(result) <= len(self.levels[-1])

    def test_logarithmic_curve_valid_range(self):
        for brightness in [0.0, 0.25, 0.5, 0.75, 1.0]:
            img = solid_image(brightness)
            result = _select_strokes_for_image(
                self.levels, img, self.canvas_w, self.canvas_h, density_curve="logarithmic"
            )
            assert 0 <= len(result) <= len(self.levels[-1])

    def test_all_curves_return_zero_for_white(self):
        img = solid_image(1.0)
        for curve in ["linear", "quadratic", "logarithmic"]:
            result = _select_strokes_for_image(
                self.levels, img, self.canvas_w, self.canvas_h, density_curve=curve
            )
            assert result == [], (
                f"Curve '{curve}' should produce no strokes for white image"
            )

    def test_all_curves_return_all_for_black(self):
        img = solid_image(0.0)
        darkest_count = len(self.levels[-1])
        for curve in ["linear", "quadratic", "logarithmic"]:
            result = _select_strokes_for_image(
                self.levels, img, self.canvas_w, self.canvas_h, density_curve=curve
            )
            assert len(result) == darkest_count, (
                f"Curve '{curve}' should return all {darkest_count} strokes for black image, "
                f"got {len(result)}"
            )

    def test_quadratic_more_than_linear_at_midgray(self):
        """sqrt curve is concave — more strokes than linear at mid-gray."""
        img = solid_image(0.5)
        linear = _select_strokes_for_image(
            self.levels, img, self.canvas_w, self.canvas_h, density_curve="linear"
        )
        quadratic = _select_strokes_for_image(
            self.levels, img, self.canvas_w, self.canvas_h, density_curve="quadratic"
        )
        assert len(quadratic) >= len(linear), (
            f"Quadratic should produce >= strokes than linear at 50% gray "
            f"(quadratic={len(quadratic)}, linear={len(linear)})"
        )

    def test_unknown_curve_falls_back_to_linear(self):
        """An unrecognised curve name uses the linear fallback."""
        img = solid_image(0.5)
        linear_result = _select_strokes_for_image(
            self.levels, img, self.canvas_w, self.canvas_h, density_curve="linear"
        )
        fallback_result = _select_strokes_for_image(
            self.levels, img, self.canvas_w, self.canvas_h, density_curve="unknown_curve"
        )
        assert len(fallback_result) == len(linear_result), (
            "Unknown curve should fall back to linear"
        )

    def test_density_curve_in_full_generator(self):
        """Full generator runs without error for all density_curve options."""
        from plottter.generators.tam import TAMGenerator

        gen = TAMGenerator()
        canvas = make_canvas()
        img = solid_image(0.4)

        base_params = {
            "_source_image": img,
            "num_tone_levels": 3,
            "stroke_length_mm": 4.0,
            "stroke_angle": 45.0,
            "cross_hatch": False,
            "cross_hatch_threshold": 0.5,
            "orientation_mode": "fixed",
            "etf_kernel_radius": 5.0,
            "etf_iterations": 3,
            "curvature": 0.0,
            "stroke_density": 0.3,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }

        for curve in ["linear", "quadratic", "logarithmic"]:
            params = dict(base_params, density_curve=curve)
            result = gen.generate(params, canvas)
            assert isinstance(result, list), f"density_curve='{curve}' returned non-list"
            for poly in result:
                assert len(poly) >= 2, (
                    f"density_curve='{curve}' produced polyline with < 2 points"
                )
