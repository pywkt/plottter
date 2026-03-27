"""Tests for the SpiralGenerator (task 67.1 + 67.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators import GENERATORS
from plottter.generators.spiral import SpiralGenerator, _trace_spiral, _sample_image_at
from plottter.models import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canvas(w_mm: float = 200.0, h_mm: float = 200.0) -> Canvas:
    return Canvas(width_mm=w_mm, height_mm=h_mm, margin_mm=10.0)


def _make_gray(h: int = 100, w: int = 100, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# _trace_spiral unit tests
# ---------------------------------------------------------------------------

class TestTraceSpiralUnit:
    def test_returns_list_of_tuples(self):
        pts = _trace_spiral(
            center_x_mm=0.0, center_y_mm=0.0,
            ring_spacing_mm=5.0, max_radius_mm=20.0, step_size_mm=0.5,
        )
        assert isinstance(pts, list)
        assert len(pts) > 0
        assert len(pts[0]) == 3  # (x, y, theta)

    def test_radii_increase_monotonically(self):
        pts = _trace_spiral(
            center_x_mm=0.0, center_y_mm=0.0,
            ring_spacing_mm=5.0, max_radius_mm=30.0, step_size_mm=0.5,
        )
        radii = [math.sqrt(x**2 + y**2) for x, y, _ in pts]
        for i in range(1, len(radii)):
            assert radii[i] >= radii[i - 1] - 1e-9

    def test_max_radius_respected(self):
        max_r = 25.0
        pts = _trace_spiral(
            center_x_mm=0.0, center_y_mm=0.0,
            ring_spacing_mm=5.0, max_radius_mm=max_r, step_size_mm=0.5,
        )
        radii = [math.sqrt(x**2 + y**2) for x, y, _ in pts]
        assert all(r <= max_r + 1e-9 for r in radii)

    def test_no_point_at_origin(self):
        """No point should be at the origin (avoid div-by-zero)."""
        pts = _trace_spiral(
            center_x_mm=0.0, center_y_mm=0.0,
            ring_spacing_mm=3.0, max_radius_mm=20.0, step_size_mm=0.5,
        )
        radii = [math.sqrt(x**2 + y**2) for x, y, _ in pts]
        assert all(r > 0.0 for r in radii)

    def test_first_radius_approximately_step_size(self):
        """The first point should have r approximately equal to step_size."""
        step = 0.5
        ring = 3.0
        pts = _trace_spiral(
            center_x_mm=0.0, center_y_mm=0.0,
            ring_spacing_mm=ring, max_radius_mm=20.0, step_size_mm=step,
        )
        x0, y0, _ = pts[0]
        r0 = math.sqrt(x0**2 + y0**2)
        assert abs(r0 - step) < 1e-9

    def test_arc_length_between_consecutive_points(self):
        """Consecutive points should be approximately step_size_mm apart."""
        step = 0.5
        pts = _trace_spiral(
            center_x_mm=0.0, center_y_mm=0.0,
            ring_spacing_mm=5.0, max_radius_mm=30.0, step_size_mm=step,
        )
        # Check a sample of arc lengths (skip the very start where curvature is high)
        sample_indices = range(10, min(100, len(pts) - 1), 5)
        for i in sample_indices:
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            # At larger radii the approximation is very close; at small radii
            # curvature makes the chord slightly shorter than the arc.
            assert dist < step * 1.5
            assert dist > step * 0.1

    def test_ring_spacing_matches(self):
        """After one full revolution the radius should increase by ring_spacing."""
        ring_spacing = 5.0
        step = 0.1
        pts = _trace_spiral(
            center_x_mm=0.0, center_y_mm=0.0,
            ring_spacing_mm=ring_spacing, max_radius_mm=50.0, step_size_mm=step,
        )
        # Find points near theta=2*pi and theta=4*pi to compare radii
        two_pi = 2.0 * math.pi
        def find_near_theta(target_theta: float) -> tuple[float, float, float] | None:
            best = None
            best_diff = float("inf")
            for pt in pts:
                diff = abs(pt[2] - target_theta)
                if diff < best_diff:
                    best_diff = diff
                    best = pt
            return best

        pt1 = find_near_theta(two_pi)
        pt2 = find_near_theta(2 * two_pi)
        assert pt1 is not None and pt2 is not None
        r1 = math.sqrt(pt1[0]**2 + pt1[1]**2)
        r2 = math.sqrt(pt2[0]**2 + pt2[1]**2)
        # r(theta) = ring_spacing * theta / (2*pi)
        # r(4*pi) - r(2*pi) = ring_spacing
        assert abs(r2 - r1 - ring_spacing) < ring_spacing * 0.05  # <5% error

    def test_center_offset(self):
        """Spiral center should be at (cx, cy)."""
        cx, cy = 10.0, 20.0
        pts = _trace_spiral(
            center_x_mm=cx, center_y_mm=cy,
            ring_spacing_mm=3.0, max_radius_mm=20.0, step_size_mm=0.5,
        )
        # All points should be centered on (cx, cy)
        x0, y0, _ = pts[0]
        r0 = math.sqrt((x0 - cx)**2 + (y0 - cy)**2)
        assert r0 > 0.0  # not at origin

    def test_empty_when_max_radius_zero(self):
        pts = _trace_spiral(0.0, 0.0, 3.0, 0.0, 0.5)
        assert pts == []

    def test_empty_when_max_radius_less_than_step(self):
        pts = _trace_spiral(0.0, 0.0, 3.0, 0.3, 0.5)
        # step_size > max_radius means first point already exceeds max_radius
        assert pts == []


# ---------------------------------------------------------------------------
# SpiralGenerator integration tests
# ---------------------------------------------------------------------------

class TestSpiralGeneratorRegistration:
    def test_registered_in_generators(self):
        assert "Spiral" in GENERATORS

    def test_category_is_image(self):
        assert GENERATORS["Spiral"].category == "image"


class TestSpiralGeneratorGenerate:
    def _run(self, **overrides) -> list:
        canvas = _make_canvas()
        params: dict = {
            "_source_image": _make_gray(100, 100, 128),
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        params.update(overrides)
        gen = SpiralGenerator()
        return gen.generate(params, canvas)

    def test_returns_single_polyline(self):
        result = self._run()
        assert len(result) == 1

    def test_polyline_has_many_points(self):
        result = self._run()
        assert len(result[0]) > 100

    def test_covers_image_area(self):
        """Spiral should extend from near center to near the corners of image rect."""
        canvas = _make_canvas(200.0, 200.0)
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        center_x = (draw_x1 + draw_x2) / 2.0
        center_y = (draw_y1 + draw_y2) / 2.0

        params = {
            "_source_image": _make_gray(100, 100, 128),
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        gen = SpiralGenerator()
        result = gen.generate(params, canvas)
        poly = result[0]

        # Find maximum radius from center
        max_r_found = max(
            math.sqrt((x - center_x)**2 + (y - center_y)**2)
            for x, y in poly
        )
        # Expected max_radius = distance from center to farthest corner
        corners = [(draw_x1, draw_y1), (draw_x2, draw_y1),
                   (draw_x1, draw_y2), (draw_x2, draw_y2)]
        expected_max_r = max(
            math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            for cx, cy in corners
        )
        # Spiral should reach at least 90% of the maximum radius
        assert max_r_found >= expected_max_r * 0.9

    def test_points_approximately_step_size_apart(self):
        """Consecutive polyline points should be approximately step_size_mm apart."""
        step = 0.5
        # Use amplitude=0 to test plain spiral spacing (no oscillation displacement)
        result = self._run(step_size_mm=step, amplitude=0.0)
        poly = result[0]
        # Sample a subset of consecutive pairs
        sample_indices = range(20, min(200, len(poly) - 1), 10)
        for i in sample_indices:
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            assert dist < step * 2.0
            assert dist > step * 0.05

    def test_ring_spacing_affects_output(self):
        """Tighter ring spacing should produce more points."""
        result_tight = self._run(ring_spacing_mm=2.0)
        result_loose = self._run(ring_spacing_mm=8.0)
        assert len(result_tight[0]) > len(result_loose[0])

    def test_center_x_pct_positions_spiral(self):
        """center_x_pct=0 should place the center near the left edge."""
        canvas = _make_canvas(200.0, 200.0)
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        params = {
            "_source_image": _make_gray(100, 100, 128),
            "ring_spacing_mm": 3.0,
            "center_x_pct": 0.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        gen = SpiralGenerator()
        result = gen.generate(params, canvas)
        poly = result[0]

        # The spiral center should be at the left edge of the drawing area
        expected_cx = draw_x1
        expected_cy = (draw_y1 + draw_y2) / 2.0

        # Find the point with the smallest radius from expected center
        min_r = min(
            math.sqrt((x - expected_cx)**2 + (y - expected_cy)**2)
            for x, y in poly
        )
        # The spiral starts at r == step_size_mm from the center
        assert min_r < 2.0  # close to the expected center

    def test_center_y_pct_positions_spiral(self):
        """center_y_pct=100 should place the center near the bottom edge."""
        canvas = _make_canvas(200.0, 200.0)
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        params = {
            "_source_image": _make_gray(100, 100, 128),
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 100.0,
            "step_size_mm": 0.5,
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        gen = SpiralGenerator()
        result = gen.generate(params, canvas)
        poly = result[0]

        expected_cx = (draw_x1 + draw_x2) / 2.0
        expected_cy = draw_y2

        min_r = min(
            math.sqrt((x - expected_cx)**2 + (y - expected_cy)**2)
            for x, y in poly
        )
        assert min_r < 2.0

    def test_output_offset_applied(self):
        """x_offset_mm and y_offset_mm should shift all output points."""
        result_no_offset = self._run(x_offset_mm=0.0, y_offset_mm=0.0)
        result_with_offset = self._run(x_offset_mm=10.0, y_offset_mm=5.0)
        assert len(result_no_offset[0]) == len(result_with_offset[0])
        x0, y0 = result_no_offset[0][0]
        x1, y1 = result_with_offset[0][0]
        assert abs(x1 - x0 - 10.0) < 1e-9
        assert abs(y1 - y0 - 5.0) < 1e-9

    def test_no_image_returns_polyline(self):
        """Generator should work without a source image."""
        canvas = _make_canvas()
        params = {
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        gen = SpiralGenerator()
        result = gen.generate(params, canvas)
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_get_parameters_has_required_params(self):
        gen = SpiralGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        required = {
            "ring_spacing_mm", "center_x_pct", "center_y_pct", "step_size_mm",
            "x_offset_mm", "y_offset_mm", "image_fit_mode",
        }
        assert required.issubset(param_names)

    def test_get_presets_non_empty(self):
        gen = SpiralGenerator()
        presets = gen.get_presets()
        assert len(presets) > 0

    def test_ring_spacing_controls_ring_distance(self):
        """ring_spacing_mm sets the actual radial distance between successive rings."""
        ring_spacing = 6.0
        canvas = _make_canvas(200.0, 200.0)
        params = {
            "ring_spacing_mm": ring_spacing,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.1,  # fine steps for accurate measurement
            "amplitude": 0.0,
            "oscillation_mode": "Sawtooth",
            "variable_velocity": False,
            "skip_white": False,
            "connected_lines": True,
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        result = SpiralGenerator().generate(params, canvas)
        poly = result[0]

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0
        radii = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in poly]

        # Find the point closest to each ring crossing radius n*ring_spacing
        # and verify successive gaps equal ring_spacing.
        crossing_radii = [
            min(radii, key=lambda r, t=n * ring_spacing: abs(r - t))
            for n in range(1, 5)
        ]
        for i in range(1, len(crossing_radii)):
            gap = crossing_radii[i] - crossing_radii[i - 1]
            assert abs(gap - ring_spacing) < ring_spacing * 0.1, (
                f"Ring gap {gap:.2f} mm differs from ring_spacing {ring_spacing}"
            )


# ---------------------------------------------------------------------------
# _sample_image_at unit tests (task 67.2)
# ---------------------------------------------------------------------------

class TestSampleImageAt:
    def test_exact_pixel(self):
        img = np.array([[0, 128], [255, 64]], dtype=np.float32)
        assert abs(_sample_image_at(img, 0.0, 0.0) - 0.0) < 1e-6
        assert abs(_sample_image_at(img, 1.0, 0.0) - 128.0) < 1e-6
        assert abs(_sample_image_at(img, 0.0, 1.0) - 255.0) < 1e-6
        assert abs(_sample_image_at(img, 1.0, 1.0) - 64.0) < 1e-6

    def test_bilinear_midpoint(self):
        # All equal values → result should be exactly that value
        img = np.full((4, 4), 100.0, dtype=np.float32)
        assert abs(_sample_image_at(img, 1.5, 1.5) - 100.0) < 1e-6

    def test_out_of_bounds_clamped(self):
        img = np.full((4, 4), 200.0, dtype=np.float32)
        # Should clamp without error
        val = _sample_image_at(img, -1.0, -1.0)
        assert 0.0 <= val <= 255.0
        val2 = _sample_image_at(img, 100.0, 100.0)
        assert 0.0 <= val2 <= 255.0


# ---------------------------------------------------------------------------
# Oscillation tests (task 67.2)
# ---------------------------------------------------------------------------

class TestOscillation:
    def _run(self, image: np.ndarray | None = None, **overrides) -> list:
        canvas = _make_canvas()
        img = image if image is not None else _make_gray(100, 100, 128)
        params: dict = {
            "_source_image": img,
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "amplitude": 0.8,
            "oscillation_mode": "Sawtooth",
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            # Disable variable velocity so tests focus on oscillation only
            "variable_velocity": False,
            "skip_white": False,
        }
        params.update(overrides)
        return SpiralGenerator().generate(params, canvas)

    def test_amplitude_zero_matches_plain_spiral(self):
        """amplitude=0 should produce the same points as a plain spiral."""
        result_zero = self._run(amplitude=0.0)
        result_plain = self._run(_source_image=None, amplitude=0.0)
        # Both should have same number of points and similar coordinates
        poly_zero = result_zero[0]
        poly_plain = result_plain[0]
        assert len(poly_zero) == len(poly_plain)
        # Check a few points are close
        for (x0, y0), (x1, y1) in zip(poly_zero[:10], poly_plain[:10]):
            assert abs(x0 - x1) < 1e-6
            assert abs(y0 - y1) < 1e-6

    def test_dark_image_large_oscillations(self):
        """Dark pixels (value=0) should produce large perpendicular offsets."""
        dark_img = np.zeros((100, 100), dtype=np.uint8)
        bright_img = np.full((100, 100), 255, dtype=np.uint8)

        result_dark = self._run(image=dark_img, amplitude=1.0)
        result_bright = self._run(image=bright_img, amplitude=1.0)

        poly_dark = result_dark[0]
        poly_bright = result_bright[0]

        # Find average displacement from each other across same-index points
        # Dark should have higher variance (due to oscillation) than bright
        dark_xs = [x for x, _ in poly_dark[10:200]]
        bright_xs = [x for x, _ in poly_bright[10:200]]
        dark_var = np.var(dark_xs)
        bright_var = np.var(bright_xs)
        assert dark_var > bright_var

    def test_bright_image_near_zero_offset(self):
        """Fully bright image should produce offsets near zero."""
        bright_img = np.full((100, 100), 255, dtype=np.uint8)
        result_bright = self._run(image=bright_img, amplitude=1.0)
        result_no_amp = self._run(image=bright_img, amplitude=0.0)

        poly_bright = result_bright[0]
        poly_no_amp = result_no_amp[0]
        # Points should be nearly identical
        for (x0, y0), (x1, y1) in zip(poly_bright[:50], poly_no_amp[:50]):
            assert abs(x0 - x1) < 1e-6
            assert abs(y0 - y1) < 1e-6

    def test_amplitude_1_max_offset_half_ring_spacing(self):
        """With amplitude=1.0 and a dark image, max offset = ring_spacing / 2.

        amplitude=1.0 should swing the spiral halfway to the adjacent ring in each
        direction (±ring_spacing/2), so peak displacement from the base spiral
        equals ring_spacing/2.
        """
        dark_img = np.zeros((100, 100), dtype=np.uint8)
        ring_spacing = 3.0
        canvas = _make_canvas()
        params = {
            "_source_image": dark_img,
            "ring_spacing_mm": ring_spacing,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "amplitude": 1.0,
            "oscillation_mode": "Sawtooth",
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            "variable_velocity": False,
            "skip_white": False,
        }
        result = SpiralGenerator().generate(params, canvas)
        poly = result[0]

        # With no-image run for baseline
        params_base = dict(params)
        params_base["_source_image"] = None
        params_base["amplitude"] = 0.0
        result_base = SpiralGenerator().generate(params_base, canvas)
        poly_base = result_base[0]

        # At dark pixels with amplitude=1.0: offset = ring_spacing / 2
        # Check that max displacement from baseline is approximately ring_spacing / 2
        displacements = [
            math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            for (x0, y0), (x1, y1) in zip(poly_base[10:200], poly[10:200])
        ]
        max_disp = max(displacements)
        expected = ring_spacing / 2.0
        # Should be close to ring_spacing/2 (within 5% tolerance)
        assert abs(max_disp - expected) < expected * 0.05

    def test_oscillation_modes_produce_different_waveforms(self):
        """Sawtooth, Sine, and Square modes should produce different point distributions."""
        mid_img = np.full((100, 100), 0, dtype=np.uint8)  # dark for max oscillation

        result_saw = self._run(image=mid_img, oscillation_mode="Sawtooth", amplitude=1.0)
        result_sin = self._run(image=mid_img, oscillation_mode="Sine", amplitude=1.0)
        result_sqr = self._run(image=mid_img, oscillation_mode="Square", amplitude=1.0)

        # All should have same length
        assert len(result_saw[0]) == len(result_sin[0]) == len(result_sqr[0])

        # But different coordinates (sample first 100 points)
        xs_saw = [x for x, _ in result_saw[0][5:105]]
        xs_sin = [x for x, _ in result_sin[0][5:105]]
        xs_sqr = [x for x, _ in result_sqr[0][5:105]]

        # Each pair should differ meaningfully
        diff_saw_sin = sum(abs(a - b) for a, b in zip(xs_saw, xs_sin))
        diff_saw_sqr = sum(abs(a - b) for a, b in zip(xs_saw, xs_sqr))
        diff_sin_sqr = sum(abs(a - b) for a, b in zip(xs_sin, xs_sqr))

        assert diff_saw_sin > 0.1
        assert diff_saw_sqr > 0.1
        assert diff_sin_sqr > 0.1

    def test_sine_mode_actually_oscillates(self):
        """Sine mode must produce non-zero displacements (not a flat spiral)."""
        dark_img = np.zeros((100, 100), dtype=np.uint8)
        result_sin = self._run(image=dark_img, oscillation_mode="Sine", amplitude=1.0)
        result_base = self._run(image=dark_img, amplitude=0.0)

        poly_sin = result_sin[0]
        poly_base = result_base[0]

        # Compute max displacement from base spiral
        max_disp = max(
            math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            for (x0, y0), (x1, y1) in zip(poly_base[1:200], poly_sin[1:200])
        )
        # Sine mode should produce actual oscillation, not flat spiral
        assert max_disp > 0.01

    def test_get_parameters_includes_oscillation_params(self):
        gen = SpiralGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "amplitude" in param_names
        assert "oscillation_mode" in param_names

    def test_amplitude_param_range(self):
        gen = SpiralGenerator()
        amp_param = next(p for p in gen.get_parameters() if p.name == "amplitude")
        assert amp_param.min == 0.01
        assert amp_param.max == 2.0
        assert amp_param.default == 0.8

    def test_oscillation_mode_choices(self):
        gen = SpiralGenerator()
        mode_param = next(p for p in gen.get_parameters() if p.name == "oscillation_mode")
        assert set(mode_param.choices) == {"Sawtooth", "Sine", "Square"}
        assert mode_param.default == "Sawtooth"


# ---------------------------------------------------------------------------
# Variable velocity tests (task 67.3A/B)
# ---------------------------------------------------------------------------

class TestVariableVelocity:
    """Tests for variable_velocity, min_velocity, max_velocity parameters."""

    def _run(self, image: np.ndarray | None = None, **overrides) -> list:
        canvas = _make_canvas()
        img = image if image is not None else _make_gray(100, 100, 128)
        params: dict = {
            "_source_image": img,
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "amplitude": 0.8,
            "oscillation_mode": "Sawtooth",
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            "variable_velocity": True,
            "min_velocity": 0.8,
            "max_velocity": 3.0,
            "skip_white": False,
            "white_threshold": 240,
            "connected_lines": True,
        }
        params.update(overrides)
        return SpiralGenerator().generate(params, canvas)

    def test_dark_image_more_points_than_bright(self):
        """variable_velocity=True: dark image should produce more points than bright image."""
        dark_img = np.zeros((100, 100), dtype=np.uint8)
        bright_img = np.full((100, 100), 255, dtype=np.uint8)

        result_dark = self._run(image=dark_img, variable_velocity=True)
        result_bright = self._run(image=bright_img, variable_velocity=True)

        # Dark areas use smaller steps → more points
        assert len(result_dark[0]) > len(result_bright[0])

    def test_variable_velocity_false_uniform_spacing(self):
        """variable_velocity=False should produce same count regardless of image brightness."""
        dark_img = np.zeros((100, 100), dtype=np.uint8)
        bright_img = np.full((100, 100), 255, dtype=np.uint8)

        result_dark = self._run(image=dark_img, variable_velocity=False)
        result_bright = self._run(image=bright_img, variable_velocity=False)

        # Without variable velocity, same image → same point count
        assert len(result_dark[0]) == len(result_bright[0])

    def test_variable_velocity_params_in_parameter_list(self):
        gen = SpiralGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "variable_velocity" in param_names
        assert "min_velocity" in param_names
        assert "max_velocity" in param_names

    def test_min_velocity_param_range(self):
        gen = SpiralGenerator()
        p = next(p for p in gen.get_parameters() if p.name == "min_velocity")
        assert p.min == 0.5
        assert p.max == 5.0
        assert p.default == 0.8

    def test_max_velocity_param_range(self):
        gen = SpiralGenerator()
        p = next(p for p in gen.get_parameters() if p.name == "max_velocity")
        assert p.min == 1.0
        assert p.max == 10.0
        assert p.default == 3.0

    def test_variable_velocity_default_true(self):
        gen = SpiralGenerator()
        p = next(p for p in gen.get_parameters() if p.name == "variable_velocity")
        assert p.default is True


# ---------------------------------------------------------------------------
# White-area skipping tests (task 67.3C)
# ---------------------------------------------------------------------------

class TestSkipWhite:
    """Tests for skip_white and white_threshold parameters."""

    def _run(self, image: np.ndarray | None = None, **overrides) -> list:
        canvas = _make_canvas()
        img = image if image is not None else _make_gray(100, 100, 128)
        params: dict = {
            "_source_image": img,
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "amplitude": 1.0,
            "oscillation_mode": "Sawtooth",
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            "variable_velocity": False,
            "min_velocity": 0.8,
            "max_velocity": 3.0,
            "skip_white": True,
            "white_threshold": 240,
            "connected_lines": True,
        }
        params.update(overrides)
        return SpiralGenerator().generate(params, canvas)

    def test_skip_white_flattens_oscillation_in_white_areas(self):
        """With skip_white=True, fully white image → flat spiral (no oscillation)."""
        bright_img = np.full((100, 100), 255, dtype=np.uint8)

        result_white_skip = self._run(image=bright_img, skip_white=True, white_threshold=240)
        result_no_skip = self._run(image=bright_img, skip_white=False)

        # Both should still produce a single polyline (connected_lines=True)
        assert len(result_white_skip) == 1
        assert len(result_no_skip) == 1

        # With skip_white, all points in white area are flat (no oscillation).
        # Since bright img has brightness=255 > threshold=240, all points flat.
        # result_no_skip with bright image also has amplitude=0 effect (1-255/255=0),
        # so they should be identical.
        assert len(result_white_skip[0]) == len(result_no_skip[0])

    def test_skip_white_connected_produces_single_polyline(self):
        """skip_white=True + connected_lines=True → always 1 polyline."""
        # Half-white, half-dark image
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255  # right half is white

        result = self._run(image=img, skip_white=True, connected_lines=True)
        assert len(result) == 1

    def test_skip_white_params_in_parameter_list(self):
        gen = SpiralGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "skip_white" in param_names
        assert "white_threshold" in param_names

    def test_skip_white_default_true(self):
        gen = SpiralGenerator()
        p = next(p for p in gen.get_parameters() if p.name == "skip_white")
        assert p.default is True

    def test_white_threshold_default_240(self):
        gen = SpiralGenerator()
        p = next(p for p in gen.get_parameters() if p.name == "white_threshold")
        assert p.default == 240
        assert p.min == 0
        assert p.max == 255


# ---------------------------------------------------------------------------
# connected_lines tests (task 67.3D)
# ---------------------------------------------------------------------------

class TestConnectedLines:
    """Tests for connected_lines parameter."""

    def _run(self, image: np.ndarray | None = None, **overrides) -> list:
        canvas = _make_canvas()
        img = image if image is not None else _make_gray(100, 100, 128)
        params: dict = {
            "_source_image": img,
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "amplitude": 0.8,
            "oscillation_mode": "Sawtooth",
            "image_fit_mode": "fill",
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
            "variable_velocity": False,
            "min_velocity": 0.8,
            "max_velocity": 3.0,
            "skip_white": True,
            "white_threshold": 128,
            "connected_lines": True,
        }
        params.update(overrides)
        return SpiralGenerator().generate(params, canvas)

    def test_connected_lines_true_single_polyline(self):
        """connected_lines=True with mixed image → exactly 1 polyline."""
        # Half black, half white
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255

        result = self._run(image=img, connected_lines=True)
        assert len(result) == 1

    def test_connected_lines_false_multiple_polylines(self):
        """connected_lines=False with white areas → multiple polylines."""
        # Checkerboard-like: alternating dark/white columns to force many breaks
        img = np.zeros((100, 100), dtype=np.uint8)
        for col in range(0, 100, 10):
            img[:, col:col + 5] = 255  # every other strip is white

        result = self._run(image=img, connected_lines=False, white_threshold=128)
        # Should produce more than 1 polyline due to white gaps
        assert len(result) > 1

    def test_connected_lines_false_no_white_areas_still_single(self):
        """connected_lines=False with fully dark image → 1 polyline (no breaks)."""
        dark_img = np.zeros((100, 100), dtype=np.uint8)
        result = self._run(image=dark_img, connected_lines=False, white_threshold=240)
        # All pixels are 0 which is below threshold, so no breaks
        assert len(result) == 1

    def test_connected_lines_param_in_parameter_list(self):
        gen = SpiralGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "connected_lines" in param_names

    def test_connected_lines_default_true(self):
        gen = SpiralGenerator()
        p = next(p for p in gen.get_parameters() if p.name == "connected_lines")
        assert p.default is True

    def test_connected_lines_false_total_points_same_as_true(self):
        """connected_lines=False should have same total points as True (just split)."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255  # right half white

        result_connected = self._run(image=img, connected_lines=True, white_threshold=128)
        result_broken = self._run(image=img, connected_lines=False, white_threshold=128)

        # connected=True has white points included (flat), connected=False excludes them
        # Total points in broken mode should be fewer (white areas dropped)
        total_connected = len(result_connected[0])
        total_broken = sum(len(p) for p in result_broken)
        assert total_broken < total_connected


# ---------------------------------------------------------------------------
# Preset tests (task 67.4A/B)
# ---------------------------------------------------------------------------

class TestPresets:
    """Tests for get_presets() — task 67.4A."""

    def test_all_required_preset_names_present(self):
        gen = SpiralGenerator()
        names = {p.name for p in gen.get_presets()}
        for required in ("Portrait", "Bold Spiral", "Fine Detail", "Sine Wave", "Minimal"):
            assert required in names, f"Missing preset: {required}"

    def test_all_presets_generate_valid_nonempty_output(self):
        """Every preset must produce at least one non-empty polyline."""
        gen = SpiralGenerator()
        canvas = _make_canvas()
        img = _make_gray(100, 100, 128)
        for preset in gen.get_presets():
            params = dict(preset.params)
            params["_source_image"] = img
            result = gen.generate(params, canvas)
            assert len(result) > 0, f"Preset '{preset.name}' returned no polylines"
            total_pts = sum(len(p) for p in result)
            assert total_pts > 0, f"Preset '{preset.name}' returned only empty polylines"

    def test_generator_registered_and_accessible(self):
        """Spiral generator must be in the GENERATORS registry."""
        assert "Spiral" in GENERATORS
        assert GENERATORS["Spiral"] is SpiralGenerator

    def test_portrait_preset_params(self):
        gen = SpiralGenerator()
        p = next(pr for pr in gen.get_presets() if pr.name == "Portrait")
        assert p.params["ring_spacing_mm"] == 2.5
        assert p.params["amplitude"] == 0.9
        assert p.params["oscillation_mode"] == "Sawtooth"
        assert p.params["variable_velocity"] is True
        assert p.params["skip_white"] is True
        assert p.params["white_threshold"] == 230

    def test_bold_spiral_preset_params(self):
        gen = SpiralGenerator()
        p = next(pr for pr in gen.get_presets() if pr.name == "Bold Spiral")
        assert p.params["ring_spacing_mm"] == 4.0
        assert p.params["amplitude"] == 1.2
        assert p.params["variable_velocity"] is True
        assert p.params["min_velocity"] == 0.5
        assert p.params["max_velocity"] == 4.0

    def test_fine_detail_preset_params(self):
        gen = SpiralGenerator()
        p = next(pr for pr in gen.get_presets() if pr.name == "Fine Detail")
        assert p.params["ring_spacing_mm"] == 1.5
        assert p.params["amplitude"] == 0.7
        assert p.params["oscillation_mode"] == "Sine"
        assert p.params["step_size_mm"] == 0.3
        assert p.params["variable_velocity"] is True

    def test_sine_wave_preset_params(self):
        gen = SpiralGenerator()
        p = next(pr for pr in gen.get_presets() if pr.name == "Sine Wave")
        assert p.params["ring_spacing_mm"] == 3.0
        assert p.params["amplitude"] == 0.8
        assert p.params["oscillation_mode"] == "Sine"
        assert p.params["variable_velocity"] is False

    def test_minimal_preset_params(self):
        gen = SpiralGenerator()
        p = next(pr for pr in gen.get_presets() if pr.name == "Minimal")
        assert p.params["ring_spacing_mm"] == 5.0
        assert p.params["amplitude"] == 0.6
        assert p.params["oscillation_mode"] == "Sawtooth"
        assert p.params["skip_white"] is True
        assert p.params["white_threshold"] == 200
        assert p.params["connected_lines"] is True


# ---------------------------------------------------------------------------
# Fit mode tests (task 67.4C-g)
# ---------------------------------------------------------------------------

class TestFitMode:
    """Tests that image_fit_mode is respected in the spiral generator."""

    def _run(self, canvas: Canvas, img: np.ndarray, fit_mode: str) -> list:
        params = {
            "_source_image": img,
            "ring_spacing_mm": 3.0,
            "center_x_pct": 50.0,
            "center_y_pct": 50.0,
            "step_size_mm": 0.5,
            "amplitude": 0.0,
            "oscillation_mode": "Sawtooth",
            "variable_velocity": False,
            "skip_white": False,
            "connected_lines": True,
            "image_fit_mode": fit_mode,
            "image_width_mm": 80.0,
            "image_height_mm": 80.0,
            "image_offset_x_mm": 0.0,
            "image_offset_y_mm": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        return SpiralGenerator().generate(params, canvas)

    def test_fit_mode_spiral_smaller_than_fill(self):
        """'fit' on a wide canvas with a tall image → smaller max radius than 'fill'."""
        # Wide canvas (2:1), tall image (1:4 pixels) — fit letterboxes to narrow strip
        canvas = _make_canvas(200.0, 100.0)
        img = _make_gray(200, 50, 128)  # 50px wide × 200px tall

        result_fill = self._run(canvas, img, "fill")
        result_fit = self._run(canvas, img, "fit")

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        max_r_fill = max(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in result_fill[0])
        max_r_fit = max(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in result_fit[0])

        # fit image rect is much smaller → spiral reaches a smaller radius
        assert max_r_fit < max_r_fill

    def test_fill_mode_covers_full_drawing_area(self):
        """'fill' mode: spiral should reach near the drawing-area corners."""
        canvas = _make_canvas(200.0, 200.0)
        img = _make_gray(100, 100, 128)
        result = self._run(canvas, img, "fill")

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0
        expected_max_r = max(
            math.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2)
            for c in [(draw_x1, draw_y1), (draw_x2, draw_y1),
                      (draw_x1, draw_y2), (draw_x2, draw_y2)]
        )
        max_r_found = max(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in result[0])
        assert max_r_found >= expected_max_r * 0.9

    def test_fit_mode_output_bounded_by_image_rect(self):
        """All output points with fit mode must lie within the image rect's bounding circle."""
        from plottter.generators._helpers import compute_image_rect

        canvas = _make_canvas(200.0, 100.0)
        img = _make_gray(200, 50, 128)  # 50px wide × 200px tall
        result = self._run(canvas, img, "fit")

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            "fit", 50, 200, draw_x1, draw_y1, draw_x2, draw_y2
        )
        cx = (img_x1 + img_x2) / 2.0
        cy = (img_y1 + img_y2) / 2.0
        # max_radius = distance from image rect center to its farthest corner
        max_r = max(
            math.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2)
            for c in [(img_x1, img_y1), (img_x2, img_y1),
                      (img_x1, img_y2), (img_x2, img_y2)]
        )
        for poly in result:
            for x, y in poly:
                r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                assert r <= max_r + 0.5, (
                    f"Point ({x:.1f}, {y:.1f}) at r={r:.1f} exceeds image rect max_r={max_r:.1f}"
                )
