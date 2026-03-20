"""Tests for HalftoneGenerator scaffold — grid layouts (Task 47.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.models.canvas import Canvas


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


# ---------------------------------------------------------------------------
# Grid helpers — standalone function tests
# ---------------------------------------------------------------------------

class TestGridSquare:
    def setup_method(self):
        from plottter.generators.halftone import _grid_square
        self.fn = _grid_square

    def test_returns_ndarray(self):
        pts = self.fn(5.0, 0.0, 100.0, 100.0)
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_approximate_point_count(self):
        """Square grid at angle=0 should produce ~(w/spacing) * (h/spacing) points."""
        w, h, s = 190.0, 277.0, 5.0
        pts = self.fn(s, 0.0, w, h)
        expected = (w / s) * (h / s)
        # Allow ±10% tolerance for edge/boundary effects
        assert abs(len(pts) - expected) / expected < 0.10, (
            f"Expected ~{expected:.0f} points, got {len(pts)}"
        )

    def test_all_points_within_bounds(self):
        w, h = 150.0, 200.0
        pts = self.fn(4.0, 0.0, w, h)
        assert np.all(pts[:, 0] >= 0.0)
        assert np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0.0)
        assert np.all(pts[:, 1] <= h)

    def test_angle_rotates_positions(self):
        """Grid at angle=45 should have different point positions than angle=0."""
        w, h = 100.0, 100.0
        pts0 = self.fn(5.0, 0.0, w, h)
        pts45 = self.fn(5.0, 45.0, w, h)
        # Sort both by (x, y) and compare; they should differ significantly
        s0 = set((round(x, 3), round(y, 3)) for x, y in pts0)
        s45 = set((round(x, 3), round(y, 3)) for x, y in pts45)
        common = s0 & s45
        # Very few points should coincide between unrotated and 45°-rotated grids
        assert len(common) < 0.1 * min(len(s0), len(s45)), (
            "Rotated grid should have mostly different point positions"
        )

    def test_angle_zero_produces_axis_aligned_grid(self):
        """At angle=0, all x coordinates should be multiples of spacing from center."""
        w, h, s = 100.0, 100.0, 10.0
        pts = self.fn(s, 0.0, w, h)
        # Unique x-values should be evenly spaced at `spacing` intervals
        xs = sorted(set(round(x, 6) for x in pts[:, 0]))
        for i in range(len(xs) - 1):
            gap = xs[i + 1] - xs[i]
            assert abs(gap - s) < 1e-4, f"Column gap {gap:.6f} != spacing {s}"

    def test_different_spacings_change_count(self):
        """Smaller spacing → more points."""
        w, h = 100.0, 100.0
        pts_fine = self.fn(3.0, 0.0, w, h)
        pts_coarse = self.fn(8.0, 0.0, w, h)
        assert len(pts_fine) > len(pts_coarse)



class TestGridHexagonal:
    def setup_method(self):
        from plottter.generators.halftone import _grid_hexagonal
        self.fn = _grid_hexagonal

    def test_returns_ndarray(self):
        pts = self.fn(5.0, 0.0, 100.0, 100.0)
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_all_points_within_bounds(self):
        w, h = 150.0, 200.0
        pts = self.fn(4.0, 0.0, w, h)
        assert np.all(pts[:, 0] >= 0.0)
        assert np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0.0)
        assert np.all(pts[:, 1] <= h)

    def test_more_points_than_square_at_same_spacing(self):
        """Hex grid should produce ~15% more points than square grid at same spacing."""
        from plottter.generators.halftone import _grid_square
        w, h, s = 190.0, 277.0, 5.0
        pts_sq = _grid_square(s, 0.0, w, h)
        pts_hex = self.fn(s, 0.0, w, h)
        ratio = len(pts_hex) / max(len(pts_sq), 1)
        # Hex grid row spacing is √3/2 ≈ 0.866 of square, so ~15% more rows
        # Allow a generous range: 5%–30% more
        assert 1.05 < ratio < 1.30, (
            f"Hex/square point ratio {ratio:.3f} outside expected range [1.05, 1.30]"
        )

    def test_angle_rotates_hex_grid(self):
        """Rotated hex grid should differ from unrotated."""
        w, h = 100.0, 100.0
        pts0 = self.fn(5.0, 0.0, w, h)
        pts30 = self.fn(5.0, 30.0, w, h)
        s0 = set((round(x, 2), round(y, 2)) for x, y in pts0)
        s30 = set((round(x, 2), round(y, 2)) for x, y in pts30)
        common = s0 & s30
        assert len(common) < 0.15 * min(len(s0), len(s30)), (
            "30°-rotated hex grid should have mostly different positions"
        )

    def test_row_spacing_is_sqrt3_over_2(self):
        """At angle=0, unique y-values should be spaced at spacing*√3/2."""
        s = 10.0
        pts = self.fn(s, 0.0, 200.0, 200.0)
        row_h = s * math.sqrt(3.0) / 2.0
        ys = sorted(set(round(y, 4) for y in pts[:, 1]))
        for i in range(len(ys) - 1):
            gap = ys[i + 1] - ys[i]
            assert abs(gap - row_h) < 1e-3, (
                f"Row gap {gap:.6f} != expected {row_h:.6f}"
            )


class TestGridDiagonal:
    def setup_method(self):
        from plottter.generators.halftone import _grid_diagonal, _grid_square
        self.fn = _grid_diagonal
        self.sq = _grid_square

    def test_returns_ndarray(self):
        pts = self.fn(5.0, 100.0, 100.0)
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2

    def test_all_points_within_bounds(self):
        w, h = 150.0, 200.0
        pts = self.fn(4.0, w, h)
        assert np.all(pts[:, 0] >= 0.0)
        assert np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0.0)
        assert np.all(pts[:, 1] <= h)

    def test_equivalent_to_square_at_45(self):
        """_grid_diagonal should produce same points as _grid_square(spacing, 45)."""
        w, h, s = 100.0, 100.0, 5.0
        pts_diag = self.fn(s, w, h)
        pts_45 = self.sq(s, 45.0, w, h)
        assert len(pts_diag) == len(pts_45)
        # Sort and compare
        pts_diag_s = pts_diag[np.lexsort((pts_diag[:, 1], pts_diag[:, 0]))]
        pts_45_s = pts_45[np.lexsort((pts_45[:, 1], pts_45[:, 0]))]
        np.testing.assert_allclose(pts_diag_s, pts_45_s, atol=1e-10)

    def test_differs_from_unrotated_square(self):
        """Diagonal grid should differ from 0°-rotated square grid."""
        w, h, s = 100.0, 100.0, 5.0
        pts_diag = self.fn(s, w, h)
        pts_0 = self.sq(s, 0.0, w, h)
        s_diag = set((round(x, 3), round(y, 3)) for x, y in pts_diag)
        s_0 = set((round(x, 3), round(y, 3)) for x, y in pts_0)
        common = s_diag & s_0
        # Very few shared points between 0° and 45° grids
        assert len(common) < 0.1 * min(len(s_diag), len(s_0))


# ---------------------------------------------------------------------------
# Generator-level tests
# ---------------------------------------------------------------------------

class TestHalftoneGenerator:
    def setup_method(self):
        from plottter.generators.halftone import HalftoneGenerator
        self.gen = HalftoneGenerator()
        self.canvas = make_canvas()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Dot Grid Halftone" in GENERATORS

    def test_category(self):
        assert self.gen.category == "image"

    def test_name(self):
        assert self.gen.name == "Dot Grid Halftone"

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def test_has_grid_spacing_param(self):
        names = {p.name for p in self.gen.get_parameters()}
        assert "grid_spacing_mm" in names

    def test_has_grid_type_param(self):
        from plottter.generators.base import ChoiceParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "grid_type" in params
        assert isinstance(params["grid_type"], ChoiceParam)
        assert set(params["grid_type"].choices) == {"Square", "Hexagonal", "Diagonal"}

    def test_has_grid_angle_param(self):
        names = {p.name for p in self.gen.get_parameters()}
        assert "grid_angle_deg" in names

    def test_has_image_param(self):
        from plottter.generators.base import ImageParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "_source_image" in params
        assert isinstance(params["_source_image"], ImageParam)

    def test_has_offset_params(self):
        names = {p.name for p in self.gen.get_parameters()}
        assert "x_offset_mm" in names
        assert "y_offset_mm" in names

    def test_has_image_processing_params(self):
        names = {p.name for p in self.gen.get_parameters()}
        assert "brightness" in names
        assert "contrast" in names
        assert "blur_radius" in names
        assert "invert" in names

    def test_grid_spacing_defaults(self):
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        sp = params["grid_spacing_mm"]
        assert isinstance(sp, FloatParam)
        assert sp.default == pytest.approx(3.0)
        assert sp.min == pytest.approx(0.5)
        assert sp.max == pytest.approx(20.0)

    # ------------------------------------------------------------------
    # generate() — returns polylines for all dot shapes
    # ------------------------------------------------------------------

    def test_generate_returns_list(self):
        result = self.gen.generate({}, self.canvas)
        assert isinstance(result, list)

    def test_generate_square_grid_returns_polylines(self):
        result = self.gen.generate(
            {"grid_type": "Square", "grid_spacing_mm": 5.0, "dot_shape": "Circle"},
            self.canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_hexagonal_grid_returns_polylines(self):
        result = self.gen.generate(
            {"grid_type": "Hexagonal", "grid_spacing_mm": 5.0, "dot_shape": "Circle"},
            self.canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_diagonal_grid_returns_polylines(self):
        result = self.gen.generate(
            {"grid_type": "Diagonal", "grid_spacing_mm": 5.0, "dot_shape": "Circle"},
            self.canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_accepts_progress_callback(self):
        progress_values = []
        self.gen.generate(
            {"grid_spacing_mm": 5.0},
            self.canvas,
            progress_callback=progress_values.append,
        )
        assert 100 in progress_values

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def test_has_presets(self):
        presets = self.gen.get_presets()
        assert len(presets) > 0

    def test_preset_names_are_strings(self):
        for p in self.gen.get_presets():
            assert isinstance(p.name, str) and p.name

    # ------------------------------------------------------------------
    # Grid point geometry properties (via grid functions)
    # ------------------------------------------------------------------

    def test_square_grid_approx_count(self):
        """Square grid at default spacing should produce ~(w/s)*(h/s) points."""
        from plottter.generators.halftone import _grid_square
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        w = draw_x2 - draw_x1
        h = draw_y2 - draw_y1
        s = 5.0
        pts = _grid_square(s, 0.0, w, h)
        expected = (w / s) * (h / s)
        assert abs(len(pts) - expected) / expected < 0.10

    def test_hex_grid_more_points_than_square(self):
        """Hex grid produces ~15% more points than square at same spacing."""
        from plottter.generators.halftone import _grid_hexagonal, _grid_square
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        w = draw_x2 - draw_x1
        h = draw_y2 - draw_y1
        s = 5.0
        n_sq = len(_grid_square(s, 0.0, w, h))
        n_hex = len(_grid_hexagonal(s, 0.0, w, h))
        ratio = n_hex / n_sq
        assert 1.05 < ratio < 1.30

    def test_all_grid_points_within_canvas(self):
        """Grid points produced by all grid types should lie within the drawing area."""
        from plottter.generators.halftone import (
            _grid_diagonal, _grid_hexagonal, _grid_square,
        )
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        w = draw_x2 - draw_x1
        h = draw_y2 - draw_y1

        for fn_name, pts in [
            ("square", _grid_square(5.0, 0.0, w, h)),
            ("square_rotated", _grid_square(5.0, 30.0, w, h)),
            ("hexagonal", _grid_hexagonal(5.0, 0.0, w, h)),
            ("diagonal", _grid_diagonal(5.0, w, h)),
        ]:
            assert np.all(pts[:, 0] >= -1e-9), f"{fn_name}: x below 0"
            assert np.all(pts[:, 0] <= w + 1e-9), f"{fn_name}: x above w"
            assert np.all(pts[:, 1] >= -1e-9), f"{fn_name}: y below 0"
            assert np.all(pts[:, 1] <= h + 1e-9), f"{fn_name}: y above h"

    def test_grid_angle_changes_point_positions(self):
        """Non-zero grid_angle_deg should produce different grid point set."""
        from plottter.generators.halftone import _grid_square
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        w = draw_x2 - draw_x1
        h = draw_y2 - draw_y1
        pts0 = _grid_square(5.0, 0.0, w, h)
        pts15 = _grid_square(5.0, 15.0, w, h)
        # Centroids should differ
        c0 = pts0.mean(axis=0)
        c15 = pts15.mean(axis=0)
        # Point sets differ (not an exact translation)
        s0 = set((round(x, 2), round(y, 2)) for x, y in pts0)
        s15 = set((round(x, 2), round(y, 2)) for x, y in pts15)
        common = s0 & s15
        assert len(common) < 0.2 * min(len(s0), len(s15)), (
            "15°-rotated grid should differ substantially from 0° grid"
        )


# ---------------------------------------------------------------------------
# Task 47.2 — brightness sampling and dot size mapping
# ---------------------------------------------------------------------------

class TestBrightnessToRadius:
    """Unit tests for _brightness_to_radius mapping function."""

    def setup_method(self):
        from plottter.generators.halftone import _brightness_to_radius
        self.fn = _brightness_to_radius

    def test_black_gives_max_radius(self):
        """Brightness=0 (black) should produce max_radius for all curves."""
        for curve in ("Area-Proportional", "Linear", "Logarithmic"):
            r = self.fn(0.0, max_radius=2.0, min_radius=0.1, curve=curve, gamma=1.0)
            assert r == pytest.approx(2.0, abs=1e-6), f"curve={curve}: expected 2.0, got {r}"

    def test_white_with_min_zero_gives_skip(self):
        """Brightness=255, min_radius=0 → dot should be skipped (radius < 0)."""
        for curve in ("Area-Proportional", "Linear", "Logarithmic"):
            r = self.fn(255.0, max_radius=2.0, min_radius=0.0, curve=curve, gamma=1.0)
            assert r < 0.0, f"curve={curve}: expected skip (<0), got {r}"

    def test_area_proportional_50pct_gray(self):
        """50% gray with area-proportional and gamma=1 → dot area ≈ 50% of max area.

        area = π*r² vs π*max_r²; ratio should be ≈0.5.
        """
        r = self.fn(128.0, max_radius=2.0, min_radius=0.0, curve="Area-Proportional", gamma=1.0)
        area_ratio = (r / 2.0) ** 2
        assert abs(area_ratio - 0.5) < 0.01, f"area ratio {area_ratio:.4f} not near 0.5"

    def test_gamma_gt1_larger_midtone_dots(self):
        """gamma > 1 emphasizes shadows: midtone dots are larger (more ink) than gamma=1.

        With formula r = max_r*(1 - t^gamma), for 0 < t < 1, t^gamma < t when gamma > 1,
        so (1-t^gamma) > (1-t), giving a larger radius.
        """
        r_g1 = self.fn(128.0, max_radius=2.0, min_radius=0.0, curve="Linear", gamma=1.0)
        r_g2 = self.fn(128.0, max_radius=2.0, min_radius=0.0, curve="Linear", gamma=2.0)
        assert r_g2 > r_g1, f"gamma=2 should give larger radius ({r_g2:.4f}) than gamma=1 ({r_g1:.4f})"

    def test_monotone_decreasing(self):
        """Radius should decrease monotonically as brightness increases."""
        for curve in ("Area-Proportional", "Linear", "Logarithmic"):
            prev = None
            for b in (0, 64, 128, 192, 255):
                r = self.fn(float(b), max_radius=2.0, min_radius=0.0, curve=curve, gamma=1.0)
                if prev is not None:
                    assert r <= prev + 1e-9, f"curve={curve}: r({b})={r:.4f} > r({b-64})={prev:.4f}"
                prev = r

    def test_min_radius_clamps_output(self):
        """Output should never be below min_radius (for dots that aren't skipped)."""
        r = self.fn(200.0, max_radius=2.0, min_radius=0.5, curve="Linear", gamma=1.0)
        assert r >= 0.5 - 1e-9


class TestHalftoneImageSampling:
    """Integration tests for image sampling inside HalftoneGenerator.generate()."""

    def setup_method(self):
        from plottter.generators.halftone import HalftoneGenerator
        self.gen = HalftoneGenerator()
        self.canvas = make_canvas()

    def _make_uniform_image(self, value: int) -> np.ndarray:
        """Create a small uniform grayscale image."""
        return np.full((50, 50), value, dtype=np.uint8)

    def test_pure_black_image_all_dots_at_max(self):
        """Black image → all dots at max_radius (after generate populates _computed_dots)."""
        img = self._make_uniform_image(0)
        max_r = 1.4
        self.gen.generate(
            {
                "_source_image": img,
                "grid_spacing_mm": 10.0,
                "max_dot_radius_mm": max_r,
                "min_dot_radius_mm": 0.1,
                "size_curve": "Area-Proportional",
                "size_gamma": 1.0,
            },
            self.canvas,
        )
        dots = self.gen._computed_dots
        assert len(dots) > 0, "Expected dots for black image"
        for x, y, r in dots:
            assert r == pytest.approx(max_r, abs=1e-6), f"Expected max_r={max_r}, got {r}"

    def test_pure_white_image_min_zero_no_dots(self):
        """White image with min_radius=0 → no dots generated."""
        img = self._make_uniform_image(255)
        self.gen.generate(
            {
                "_source_image": img,
                "grid_spacing_mm": 10.0,
                "max_dot_radius_mm": 1.4,
                "min_dot_radius_mm": 0.0,
                "size_curve": "Area-Proportional",
                "size_gamma": 1.0,
            },
            self.canvas,
        )
        assert self.gen._computed_dots == [], "Expected no dots for pure white image with min_radius=0"

    def test_50pct_gray_area_proportional(self):
        """50% gray with area-proportional → dot area ≈ 50% of max area."""
        img = self._make_uniform_image(128)
        max_r = 2.0
        self.gen.generate(
            {
                "_source_image": img,
                "grid_spacing_mm": 10.0,
                "max_dot_radius_mm": max_r,
                "min_dot_radius_mm": 0.0,
                "size_curve": "Area-Proportional",
                "size_gamma": 1.0,
            },
            self.canvas,
        )
        dots = self.gen._computed_dots
        assert len(dots) > 0, "Expected dots for 50% gray image"
        for x, y, r in dots:
            area_ratio = (r / max_r) ** 2
            assert abs(area_ratio - 0.5) < 0.02, f"area ratio {area_ratio:.4f} not near 0.5"

    def test_gamma_gt1_larger_dots_than_gamma1(self):
        """gamma > 1 emphasizes shadows: midtone dots are larger (more ink) than gamma=1.

        With formula r = max_r*(1-t^gamma), gamma>1 compresses t^gamma downward for
        0 < t < 1, so midtone dots have larger radii than with gamma=1.
        """
        img = self._make_uniform_image(128)
        max_r = 2.0

        self.gen.generate(
            {
                "_source_image": img,
                "grid_spacing_mm": 10.0,
                "max_dot_radius_mm": max_r,
                "min_dot_radius_mm": 0.0,
                "size_curve": "Linear",
                "size_gamma": 1.0,
            },
            self.canvas,
        )
        radii_g1 = [r for _, _, r in self.gen._computed_dots]

        self.gen.generate(
            {
                "_source_image": img,
                "grid_spacing_mm": 10.0,
                "max_dot_radius_mm": max_r,
                "min_dot_radius_mm": 0.0,
                "size_curve": "Linear",
                "size_gamma": 2.0,
            },
            self.canvas,
        )
        radii_g2 = [r for _, _, r in self.gen._computed_dots]

        assert len(radii_g1) > 0
        assert len(radii_g2) > 0
        avg_g1 = sum(radii_g1) / len(radii_g1)
        avg_g2 = sum(radii_g2) / len(radii_g2)
        assert avg_g2 > avg_g1, f"gamma=2 avg radius {avg_g2:.4f} should be > gamma=1 {avg_g1:.4f}"

    def test_no_image_uses_max_radius(self):
        """When no source image is provided, all dots should use max_radius."""
        max_r = 1.8
        self.gen.generate(
            {
                "grid_spacing_mm": 10.0,
                "max_dot_radius_mm": max_r,
                "min_dot_radius_mm": 0.1,
            },
            self.canvas,
        )
        dots = self.gen._computed_dots
        assert len(dots) > 0
        for x, y, r in dots:
            assert r == pytest.approx(max_r, abs=1e-6)

    def test_new_dot_size_params_in_parameters(self):
        """All new dot size parameters must be present in get_parameters()."""
        names = {p.name for p in self.gen.get_parameters()}
        assert "max_dot_radius_mm" in names
        assert "min_dot_radius_mm" in names
        assert "size_curve" in names
        assert "size_gamma" in names

    def test_size_curve_choices(self):
        """size_curve must offer Area-Proportional, Linear, and Logarithmic."""
        from plottter.generators.base import ChoiceParam
        params = {p.name: p for p in self.gen.get_parameters()}
        sc = params["size_curve"]
        assert isinstance(sc, ChoiceParam)
        assert set(sc.choices) == {"Area-Proportional", "Linear", "Logarithmic"}
        assert sc.default == "Area-Proportional"


# ---------------------------------------------------------------------------
# Task 47.3 — dot shape rendering
# ---------------------------------------------------------------------------

class TestDotRenderingFunctions:
    """Unit tests for standalone dot-shape helper functions."""

    def test_dot_circle_point_count(self):
        """_dot_circle with N segments should produce N+1 points (closed)."""
        from plottter.generators.halftone import _dot_circle
        for seg in (6, 8, 16, 32):
            poly = _dot_circle(0.0, 0.0, 1.0, seg)
            assert len(poly) == seg + 1, f"segments={seg}: expected {seg+1} pts, got {len(poly)}"

    def test_dot_circle_is_closed(self):
        """First and last point of _dot_circle must be identical (closed polyline)."""
        from plottter.generators.halftone import _dot_circle
        poly = _dot_circle(5.0, 3.0, 2.0, 16)
        assert poly[0] == poly[-1], "Circle polyline should be closed (first == last)"

    def test_dot_circle_radius(self):
        """All points of _dot_circle should lie on the circle of given radius."""
        from plottter.generators.halftone import _dot_circle
        cx, cy, r = 10.0, 20.0, 3.5
        poly = _dot_circle(cx, cy, r, 32)
        for x, y in poly:
            dist = math.hypot(x - cx, y - cy)
            assert abs(dist - r) < 1e-9, f"Point ({x:.4f},{y:.4f}) not on circle of r={r}"

    def test_dot_filled_ring_count(self):
        """_dot_filled should produce ceil(r / pen_spacing) concentric rings."""
        from plottter.generators.halftone import _dot_filled
        r, spacing = 1.4, 0.3
        expected_rings = math.ceil(r / spacing)
        polys = _dot_filled(0.0, 0.0, r, spacing, 16)
        assert len(polys) == expected_rings, (
            f"Expected {expected_rings} rings, got {len(polys)}"
        )

    def test_dot_filled_multiple_polylines(self):
        """_dot_filled must return more than one polyline for r > pen_spacing."""
        from plottter.generators.halftone import _dot_filled
        polys = _dot_filled(0.0, 0.0, 2.0, 0.5, 16)
        assert len(polys) > 1, "Filled circle should produce multiple concentric rings"

    def test_dot_filled_rings_are_closed(self):
        """Every ring polyline from _dot_filled should be closed."""
        from plottter.generators.halftone import _dot_filled
        for poly in _dot_filled(0.0, 0.0, 1.5, 0.4, 12):
            assert poly[0] == poly[-1], "Each concentric ring should be closed"

    def test_dot_filled_rings_decreasing_radius(self):
        """Successive rings from _dot_filled should have strictly decreasing radii."""
        from plottter.generators.halftone import _dot_filled
        cx, cy = 5.0, 5.0
        polys = _dot_filled(cx, cy, 2.0, 0.5, 32)
        radii = [math.hypot(p[0][0] - cx, p[0][1] - cy) for p in polys]
        for a, b in zip(radii, radii[1:]):
            assert a > b - 1e-9, f"Ring radii not decreasing: {a:.4f} then {b:.4f}"

    def test_dot_spiral_single_polyline(self):
        """_dot_spiral should return a single polyline (list of points, not list of lists)."""
        from plottter.generators.halftone import _dot_spiral
        poly = _dot_spiral(0.0, 0.0, 1.5, 0.3, 16)
        assert isinstance(poly, list), "Spiral should be a list"
        assert len(poly) > 0
        # Each element should be a (float, float) tuple, not a list
        assert isinstance(poly[0], tuple), "Each element should be a (x,y) point tuple"

    def test_dot_spiral_reaches_center(self):
        """Last point of _dot_spiral should be close to (x, y)."""
        from plottter.generators.halftone import _dot_spiral
        cx, cy = 10.0, 15.0
        poly = _dot_spiral(cx, cy, 1.2, 0.3, 16)
        lx, ly = poly[-1]
        assert math.hypot(lx - cx, ly - cy) < 1e-9, (
            f"Spiral end ({lx:.4f},{ly:.4f}) not at center ({cx},{cy})"
        )

    def test_dot_spiral_starts_at_outer_radius(self):
        """First point of _dot_spiral should lie on the outer radius."""
        from plottter.generators.halftone import _dot_spiral
        cx, cy, r = 5.0, 5.0, 1.5
        poly = _dot_spiral(cx, cy, r, 0.3, 16)
        dist = math.hypot(poly[0][0] - cx, poly[0][1] - cy)
        assert abs(dist - r) < 1e-9, f"Spiral start dist={dist:.6f} != outer radius={r}"

    def test_dot_square_point_count(self):
        """_dot_square should return exactly 5 points (closed rectangle)."""
        from plottter.generators.halftone import _dot_square
        poly = _dot_square(0.0, 0.0, 1.0)
        assert len(poly) == 5, f"Square should have 5 points, got {len(poly)}"

    def test_dot_square_is_closed(self):
        """First and last point of _dot_square must be identical."""
        from plottter.generators.halftone import _dot_square
        poly = _dot_square(3.0, 4.0, 2.0)
        assert poly[0] == poly[-1], "Square polyline should be closed"

    def test_dot_square_correct_corners(self):
        """_dot_square corners should be at (x±r, y±r)."""
        from plottter.generators.halftone import _dot_square
        cx, cy, r = 5.0, 5.0, 2.0
        poly = _dot_square(cx, cy, r)
        corners = set(poly[:4])
        expected = {
            (cx - r, cy - r),
            (cx + r, cy - r),
            (cx + r, cy + r),
            (cx - r, cy + r),
        }
        assert corners == expected, f"Square corners {corners} != expected {expected}"

    def test_dot_diamond_point_count(self):
        """_dot_diamond should return exactly 5 points (closed diamond)."""
        from plottter.generators.halftone import _dot_diamond
        poly = _dot_diamond(0.0, 0.0, 1.0)
        assert len(poly) == 5

    def test_dot_diamond_is_closed(self):
        """First and last point of _dot_diamond must be identical."""
        from plottter.generators.halftone import _dot_diamond
        poly = _dot_diamond(0.0, 0.0, 1.0)
        assert poly[0] == poly[-1]

    def test_dot_diamond_cardinal_points(self):
        """_dot_diamond corners should be at cardinal positions (top/right/bottom/left)."""
        from plottter.generators.halftone import _dot_diamond
        cx, cy, r = 5.0, 5.0, 2.0
        poly = _dot_diamond(cx, cy, r)
        corners = set(poly[:4])
        expected = {
            (cx,     cy - r),
            (cx + r, cy    ),
            (cx,     cy + r),
            (cx - r, cy    ),
        }
        assert corners == expected

    def test_dot_cross_two_polylines(self):
        """_dot_cross should return exactly 2 polylines."""
        from plottter.generators.halftone import _dot_cross
        result = _dot_cross(0.0, 0.0, 1.0)
        assert len(result) == 2

    def test_dot_cross_segment_length(self):
        """Each arm of _dot_cross should span 2r (from -r to +r through center)."""
        from plottter.generators.halftone import _dot_cross
        cx, cy, r = 5.0, 5.0, 3.0
        h_seg, v_seg = _dot_cross(cx, cy, r)
        # Horizontal: y constant, x from cx-r to cx+r
        assert h_seg[0] == (cx - r, cy)
        assert h_seg[1] == (cx + r, cy)
        # Vertical: x constant, y from cy-r to cy+r
        assert v_seg[0] == (cx, cy - r)
        assert v_seg[1] == (cx, cy + r)


class TestDotShapeParameters:
    """Tests for the new dot shape parameters on HalftoneGenerator."""

    def setup_method(self):
        from plottter.generators.halftone import HalftoneGenerator
        self.gen = HalftoneGenerator()

    def test_has_dot_shape_param(self):
        from plottter.generators.base import ChoiceParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "dot_shape" in params
        assert isinstance(params["dot_shape"], ChoiceParam)

    def test_dot_shape_choices(self):
        from plottter.generators.base import ChoiceParam
        params = {p.name: p for p in self.gen.get_parameters()}
        choices = set(params["dot_shape"].choices)
        assert choices == {"Circle", "Filled Circle", "Spiral Fill", "Square", "Diamond", "Cross"}

    def test_dot_shape_default(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert params["dot_shape"].default == "Circle"

    def test_has_circle_segments_param(self):
        from plottter.generators.base import IntParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "circle_segments" in params
        assert isinstance(params["circle_segments"], IntParam)

    def test_circle_segments_range(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        cs = params["circle_segments"]
        assert cs.min == 6
        assert cs.max == 64
        assert cs.default == 16

    def test_circle_segments_visible_when(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["circle_segments"].visible_when
        assert vw is not None
        assert "dot_shape" in vw
        shapes = set(vw["dot_shape"])
        assert shapes == {"Circle", "Filled Circle", "Spiral Fill"}

    def test_has_fill_line_spacing_param(self):
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "fill_line_spacing_mm" in params
        assert isinstance(params["fill_line_spacing_mm"], FloatParam)

    def test_fill_line_spacing_range(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        fls = params["fill_line_spacing_mm"]
        assert fls.min == pytest.approx(0.1)
        assert fls.max == pytest.approx(2.0)
        assert fls.default == pytest.approx(0.3)

    def test_fill_line_spacing_visible_when(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        vw = params["fill_line_spacing_mm"].visible_when
        assert vw is not None
        assert "dot_shape" in vw
        shapes = set(vw["dot_shape"])
        assert shapes == {"Filled Circle", "Spiral Fill"}


class TestDotShapeGenerate:
    """Integration tests for dot shape rendering in HalftoneGenerator.generate()."""

    def setup_method(self):
        from plottter.generators.halftone import HalftoneGenerator
        self.gen = HalftoneGenerator()
        self.canvas = make_canvas()

    def _params(self, dot_shape: str, **kw) -> dict:
        base = {
            "grid_spacing_mm": 20.0,   # coarse grid → few dots for fast tests
            "max_dot_radius_mm": 2.0,
            "min_dot_radius_mm": 0.1,
            "dot_shape": dot_shape,
            "circle_segments": 16,
            "fill_line_spacing_mm": 0.5,
        }
        base.update(kw)
        return base

    def test_circle_dots_are_closed(self):
        """Circle dot polylines must be closed (first == last point)."""
        result = self.gen.generate(self._params("Circle"), self.canvas)
        assert len(result) > 0
        for poly in result:
            assert poly[0] == poly[-1], "Circle polyline not closed"

    def test_circle_dots_segment_count(self):
        """Circle dot polylines must have circle_segments + 1 points."""
        seg = 12
        result = self.gen.generate(self._params("Circle", circle_segments=seg), self.canvas)
        assert len(result) > 0
        for poly in result:
            assert len(poly) == seg + 1, f"Expected {seg+1} pts, got {len(poly)}"

    def test_filled_circle_multiple_polylines_per_dot(self):
        """Filled Circle mode should produce more polylines than dot count (multiple rings)."""
        result = self.gen.generate(self._params("Filled Circle"), self.canvas)
        dot_count = len(self.gen._computed_dots)
        assert len(result) > dot_count, (
            f"Filled Circle ({len(result)} polys) should exceed dot count ({dot_count})"
        )

    def test_spiral_fill_one_polyline_per_dot(self):
        """Spiral Fill mode should produce exactly one polyline per dot."""
        result = self.gen.generate(self._params("Spiral Fill"), self.canvas)
        dot_count = len(self.gen._computed_dots)
        assert len(result) == dot_count, (
            f"Spiral Fill should give 1 polyline per dot: {len(result)} polys, {dot_count} dots"
        )

    def test_square_dots_five_points(self):
        """Square dot polylines must have exactly 5 points (closed rectangle)."""
        result = self.gen.generate(self._params("Square"), self.canvas)
        assert len(result) > 0
        for poly in result:
            assert len(poly) == 5, f"Square polyline should have 5 pts, got {len(poly)}"

    def test_diamond_dots_five_points(self):
        """Diamond dot polylines must have exactly 5 points (closed)."""
        result = self.gen.generate(self._params("Diamond"), self.canvas)
        assert len(result) > 0
        for poly in result:
            assert len(poly) == 5, f"Diamond polyline should have 5 pts, got {len(poly)}"

    def test_cross_dots_two_polylines_per_dot(self):
        """Cross mode should produce exactly 2 polylines per dot."""
        result = self.gen.generate(self._params("Cross"), self.canvas)
        dot_count = len(self.gen._computed_dots)
        assert len(result) == 2 * dot_count, (
            f"Cross should give 2 polylines per dot: {len(result)} polys, {dot_count} dots"
        )

    def test_all_shapes_within_canvas_bounds(self):
        """All dot shape polylines should lie within the canvas bounds."""
        cw = self.canvas.width_mm
        ch = self.canvas.height_mm
        for shape in ("Circle", "Filled Circle", "Spiral Fill", "Square", "Diamond", "Cross"):
            result = self.gen.generate(self._params(shape), self.canvas)
            for poly in result:
                for x, y in poly:
                    assert -1e-6 <= x <= cw + 1e-6, (
                        f"shape={shape}: x={x:.4f} out of canvas [0, {cw}]"
                    )
                    assert -1e-6 <= y <= ch + 1e-6, (
                        f"shape={shape}: y={y:.4f} out of canvas [0, {ch}]"
                    )
