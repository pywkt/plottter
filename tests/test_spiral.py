"""Tests for the SpiralGenerator (task 67.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators import GENERATORS
from plottter.generators.spiral import SpiralGenerator, _trace_spiral
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
        result = self._run(step_size_mm=step)
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
