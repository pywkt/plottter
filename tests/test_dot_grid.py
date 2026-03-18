"""Tests for the DotGridGenerator (Phase 35.3)."""

from __future__ import annotations

import math

import pytest

from plottter.models.canvas import Canvas


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


class TestDotGridGenerator:
    def setup_method(self):
        from plottter.generators.dot_grid import DotGridGenerator
        self.gen = DotGridGenerator()
        self.canvas = make_canvas()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Dot Grid" in GENERATORS

    def test_category(self):
        assert self.gen.category == "math"

    # ------------------------------------------------------------------
    # Shape correctness
    # ------------------------------------------------------------------

    def test_circle_shape_renders(self):
        paths = self.gen.generate(
            {"dot_shape": "Circle", "grid_cols": 3, "grid_rows": 3,
             "base_size_mm": 2.0, "spacing_mm": 5.0, "noise_strength": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        # Each circle is a closed polygon
        for path in paths:
            assert path[0] == path[-1], "Circle polyline should be closed"

    def test_square_shape_renders(self):
        paths = self.gen.generate(
            {"dot_shape": "Square", "grid_cols": 3, "grid_rows": 3,
             "base_size_mm": 2.0, "spacing_mm": 5.0, "noise_strength": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 5  # 4 corners + closing point
            assert path[0] == path[-1]

    def test_diamond_shape_renders(self):
        paths = self.gen.generate(
            {"dot_shape": "Diamond", "grid_cols": 3, "grid_rows": 3,
             "base_size_mm": 2.0, "spacing_mm": 5.0, "noise_strength": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 5  # 4 points + closing
            assert path[0] == path[-1]

    def test_cross_shape_renders(self):
        paths = self.gen.generate(
            {"dot_shape": "Cross", "grid_cols": 3, "grid_rows": 3,
             "base_size_mm": 2.0, "spacing_mm": 5.0, "noise_strength": 0.0},
            self.canvas,
        )
        # Cross produces 2 polylines per cell (horizontal + vertical)
        assert len(paths) > 0
        assert len(paths) == 3 * 3 * 2  # 9 cells × 2 lines

    def test_star_shape_renders(self):
        paths = self.gen.generate(
            {"dot_shape": "Star", "grid_cols": 3, "grid_rows": 3,
             "base_size_mm": 2.0, "spacing_mm": 5.0, "noise_strength": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        # Star has 10 points (5 outer + 5 inner) plus closing = 11 vertices
        for path in paths:
            assert len(path) == 11  # 5*2 + 1

    def test_star_has_alternating_radii(self):
        """Star should alternate between outer and inner radius."""
        from plottter.generators.dot_grid import _star_at_origin
        r = 3.0
        path = _star_at_origin(r)
        # Compute distance of each point from center (star is at origin, no offset)
        dists = [math.hypot(x, y) for x, y in path[:-1]]  # exclude closing point
        outer_r = r
        inner_r = r * 0.4
        for i, d in enumerate(dists):
            if i % 2 == 0:
                assert abs(d - outer_r) < 1e-9, f"Point {i} should be at outer radius"
            else:
                assert abs(d - inner_r) < 1e-9, f"Point {i} should be at inner radius"

    def test_hexagon_shape_renders(self):
        paths = self.gen.generate(
            {"dot_shape": "Hexagon", "grid_cols": 3, "grid_rows": 3,
             "base_size_mm": 2.0, "spacing_mm": 5.0, "noise_strength": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 7  # 6 vertices + closing

    # ------------------------------------------------------------------
    # Noise modulation
    # ------------------------------------------------------------------

    def test_noise_creates_size_variation(self):
        """With noise enabled, dot sizes should vary across the grid."""
        from plottter.generators.dot_grid import _NOISE_AVAILABLE
        if not _NOISE_AVAILABLE:
            pytest.skip("noise library not installed")

        # Generate a large grid so we have plenty of size variation
        paths_noisy = self.gen.generate(
            {"dot_shape": "Circle", "grid_cols": 10, "grid_rows": 10,
             "base_size_mm": 3.0, "spacing_mm": 5.0,
             "noise_scale": 0.1, "noise_strength": 0.8,
             "min_size_mm": 0.5, "max_size_mm": 10.0, "noise_seed": 42},
            self.canvas,
        )
        paths_uniform = self.gen.generate(
            {"dot_shape": "Circle", "grid_cols": 10, "grid_rows": 10,
             "base_size_mm": 3.0, "spacing_mm": 5.0,
             "noise_scale": 0.1, "noise_strength": 0.0,
             "min_size_mm": 0.5, "max_size_mm": 10.0, "noise_seed": 42},
            self.canvas,
        )
        # With noise, different circles should have different radii
        def first_radius(path):
            x0, y0 = path[0]
            cx = (min(x for x, _ in path) + max(x for x, _ in path)) / 2.0
            return abs(x0 - cx)

        radii_noisy = [first_radius(p) for p in paths_noisy]
        radii_uniform = [first_radius(p) for p in paths_uniform]

        # Noisy radii should not all be equal
        assert max(radii_noisy) - min(radii_noisy) > 0.01, \
            "Noise should produce size variation"
        # Uniform radii should all be equal
        assert max(radii_uniform) - min(radii_uniform) < 1e-6, \
            "noise_strength=0 should produce uniform dots"

    def test_noise_strength_zero_is_uniform(self):
        """noise_strength=0 should produce identical dots everywhere."""
        paths = self.gen.generate(
            {"dot_shape": "Square", "grid_cols": 5, "grid_rows": 5,
             "base_size_mm": 2.0, "spacing_mm": 5.0,
             "noise_strength": 0.0, "min_size_mm": 0.0, "max_size_mm": 20.0},
            self.canvas,
        )
        # All squares should have the same size (half-side = base_size = 2.0)
        # Check width of each square
        widths = [max(x for x, _ in p) - min(x for x, _ in p) for p in paths]
        for w in widths:
            assert abs(w - 4.0) < 1e-9, f"Expected width 4.0 mm, got {w}"

    # ------------------------------------------------------------------
    # Size clamping and min_size skip
    # ------------------------------------------------------------------

    def test_dots_below_min_size_are_omitted(self):
        """Setting min_size_mm larger than base_size with noise should skip some dots."""
        from plottter.generators.dot_grid import _NOISE_AVAILABLE
        if not _NOISE_AVAILABLE:
            pytest.skip("noise library not installed")

        # Set min_size high so some noise-reduced dots are below threshold
        # With noise_strength=1.0, size can range from base*(1-1)=0 to base*(1+1)=2*base
        paths_with_skip = self.gen.generate(
            {"dot_shape": "Circle", "grid_cols": 8, "grid_rows": 8,
             "base_size_mm": 2.0, "spacing_mm": 5.0,
             "noise_scale": 0.3, "noise_strength": 1.0,
             "min_size_mm": 2.0, "max_size_mm": 10.0, "noise_seed": 0},
            self.canvas,
        )
        paths_no_skip = self.gen.generate(
            {"dot_shape": "Circle", "grid_cols": 8, "grid_rows": 8,
             "base_size_mm": 2.0, "spacing_mm": 5.0,
             "noise_scale": 0.3, "noise_strength": 1.0,
             "min_size_mm": 0.0, "max_size_mm": 10.0, "noise_seed": 0},
            self.canvas,
        )
        # With higher min_size, strictly fewer dots should be rendered
        assert len(paths_with_skip) < len(paths_no_skip), \
            "Higher min_size should skip some dots (strict fewer)"

    def test_max_size_clamps_dots(self):
        """Dots should never exceed max_size_mm."""
        from plottter.generators.dot_grid import _NOISE_AVAILABLE
        if not _NOISE_AVAILABLE:
            pytest.skip("noise library not installed")

        max_size = 3.0
        paths = self.gen.generate(
            {"dot_shape": "Circle", "grid_cols": 6, "grid_rows": 6,
             "base_size_mm": 2.5, "spacing_mm": 6.0,
             "noise_scale": 0.1, "noise_strength": 0.8,
             "min_size_mm": 0.5, "max_size_mm": max_size, "noise_seed": 7},
            self.canvas,
        )
        for path in paths:
            # The bounding half-width of each circle should not exceed max_size
            half_w = (max(x for x, _ in path) - min(x for x, _ in path)) / 2.0
            assert half_w <= max_size + 1e-9, \
                f"Circle width {half_w * 2:.3f} exceeds max_size {max_size * 2}"

    # ------------------------------------------------------------------
    # Rotation noise
    # ------------------------------------------------------------------

    def test_rotation_noise_varies_shape_orientation(self):
        """rotation_noise > 0 should produce different orientations at different positions."""
        from plottter.generators.dot_grid import _NOISE_AVAILABLE
        if not _NOISE_AVAILABLE:
            pytest.skip("noise library not installed")

        # Square is a good test shape: rotation changes vertex angles visibly
        paths = self.gen.generate(
            {"dot_shape": "Square", "grid_cols": 5, "grid_rows": 5,
             "base_size_mm": 2.0, "spacing_mm": 8.0, "noise_strength": 0.0,
             "rotation_noise": 90.0, "jitter_mm": 0.0, "noise_seed": 42},
            self.canvas,
        )
        # For each square, compute the angle from the bounding-box center to its first vertex
        angles = []
        for path in paths:
            cx = (min(x for x, _ in path) + max(x for x, _ in path)) / 2.0
            cy = (min(y for _, y in path) + max(y for _, y in path)) / 2.0
            dx, dy = path[0][0] - cx, path[0][1] - cy
            angles.append(math.atan2(dy, dx))
        assert max(angles) - min(angles) > 0.01, \
            "rotation_noise > 0 should produce varied vertex angles across the grid"

    def test_rotation_noise_zero_no_rotation(self):
        """rotation_noise=0 should produce no rotation (all squares axis-aligned)."""
        paths = self.gen.generate(
            {"dot_shape": "Square", "grid_cols": 5, "grid_rows": 5,
             "base_size_mm": 2.0, "spacing_mm": 8.0, "noise_strength": 0.0,
             "rotation_noise": 0.0, "jitter_mm": 0.0},
            self.canvas,
        )
        # Unrotated square: first vertex (-r, -r) → angle atan2(-1, -1) = -3π/4
        expected_angle = math.atan2(-1.0, -1.0)
        for path in paths:
            cx = (min(x for x, _ in path) + max(x for x, _ in path)) / 2.0
            cy = (min(y for _, y in path) + max(y for _, y in path)) / 2.0
            dx, dy = path[0][0] - cx, path[0][1] - cy
            angle = math.atan2(dy, dx)
            assert abs(angle - expected_angle) < 1e-9, \
                f"rotation_noise=0 should produce axis-aligned squares, got angle {angle:.6f}"

    # ------------------------------------------------------------------
    # Position jitter
    # ------------------------------------------------------------------

    def test_jitter_mm_displaces_dot_centers(self):
        """jitter_mm > 0 should displace dots from their ideal grid positions."""
        from plottter.generators.dot_grid import _NOISE_AVAILABLE
        if not _NOISE_AVAILABLE:
            pytest.skip("noise library not installed")

        base_params = {
            "dot_shape": "Square", "grid_cols": 5, "grid_rows": 5,
            "base_size_mm": 1.0, "spacing_mm": 10.0, "noise_strength": 0.0,
            "rotation_noise": 0.0, "noise_seed": 42,
        }

        def center(path):
            xs = [x for x, _ in path]
            ys = [y for _, y in path]
            return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

        paths_no_jitter = self.gen.generate({**base_params, "jitter_mm": 0.0}, self.canvas)
        paths_jitter = self.gen.generate({**base_params, "jitter_mm": 3.0}, self.canvas)

        centers_no = [center(p) for p in paths_no_jitter]
        centers_jittered = [center(p) for p in paths_jitter]

        diffs = [
            math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            for c1, c2 in zip(centers_no, centers_jittered)
        ]
        assert any(d > 0.01 for d in diffs), \
            "jitter_mm > 0 should shift at least some dot centers away from grid positions"

    def test_jitter_mm_zero_keeps_on_grid(self):
        """jitter_mm=0 should keep all dots exactly on a uniform grid."""
        spacing = 10.0
        cols, rows = 4, 4
        paths = self.gen.generate(
            {"dot_shape": "Square", "grid_cols": cols, "grid_rows": rows,
             "base_size_mm": 1.0, "spacing_mm": spacing, "noise_strength": 0.0,
             "rotation_noise": 0.0, "jitter_mm": 0.0},
            self.canvas,
        )
        assert len(paths) == cols * rows

        def center(path):
            xs = [x for x, _ in path]
            ys = [y for _, y in path]
            return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

        centers = [center(p) for p in paths]
        all_xs = sorted(set(round(c[0], 6) for c in centers))
        all_ys = sorted(set(round(c[1], 6) for c in centers))

        assert len(all_xs) == cols, f"Expected {cols} unique x-positions, got {len(all_xs)}"
        assert len(all_ys) == rows, f"Expected {rows} unique y-positions, got {len(all_ys)}"

        for i in range(len(all_xs) - 1):
            gap = all_xs[i + 1] - all_xs[i]
            assert abs(gap - spacing) < 1e-4, f"Column gap {gap:.6f} != spacing {spacing}"
        for i in range(len(all_ys) - 1):
            gap = all_ys[i + 1] - all_ys[i]
            assert abs(gap - spacing) < 1e-4, f"Row gap {gap:.6f} != spacing {spacing}"

    # ------------------------------------------------------------------
    # Filled shapes
    # ------------------------------------------------------------------

    def test_filled_produces_more_polylines(self):
        """filled=True should produce more polylines (concentric copies) than a single outline."""
        base_params = {
            "dot_shape": "Circle", "grid_cols": 3, "grid_rows": 3,
            "base_size_mm": 4.0, "spacing_mm": 12.0, "noise_strength": 0.0,
            "rotation_noise": 0.0, "jitter_mm": 0.0,
        }
        paths_outline = self.gen.generate({**base_params, "filled": False}, self.canvas)
        paths_filled = self.gen.generate(
            {**base_params, "filled": True, "pen_width_mm": 0.5}, self.canvas
        )
        assert len(paths_filled) > len(paths_outline), \
            "filled=True should produce more polylines (concentric fills) than a single outline"

    def test_smaller_pen_width_more_concentric_lines(self):
        """Smaller pen_width_mm should produce more concentric fill lines per dot."""
        base_params = {
            "dot_shape": "Circle", "grid_cols": 2, "grid_rows": 2,
            "base_size_mm": 5.0, "spacing_mm": 14.0, "noise_strength": 0.0,
            "rotation_noise": 0.0, "jitter_mm": 0.0, "filled": True,
        }
        paths_narrow = self.gen.generate({**base_params, "pen_width_mm": 0.2}, self.canvas)
        paths_wide = self.gen.generate({**base_params, "pen_width_mm": 1.0}, self.canvas)
        assert len(paths_narrow) > len(paths_wide), \
            "Smaller pen_width_mm should produce more concentric lines per dot"
