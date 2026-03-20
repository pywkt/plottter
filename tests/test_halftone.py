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
    # generate() scaffold — returns list (empty for now)
    # ------------------------------------------------------------------

    def test_generate_returns_list(self):
        result = self.gen.generate({}, self.canvas)
        assert isinstance(result, list)

    def test_generate_square_returns_empty(self):
        result = self.gen.generate(
            {"grid_type": "Square", "grid_spacing_mm": 5.0},
            self.canvas,
        )
        assert result == []

    def test_generate_hexagonal_returns_empty(self):
        result = self.gen.generate(
            {"grid_type": "Hexagonal", "grid_spacing_mm": 5.0},
            self.canvas,
        )
        assert result == []

    def test_generate_diagonal_returns_empty(self):
        result = self.gen.generate(
            {"grid_type": "Diagonal", "grid_spacing_mm": 5.0},
            self.canvas,
        )
        assert result == []

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
