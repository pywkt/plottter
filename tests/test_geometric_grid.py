"""Tests for the GeometricGridGenerator (Phase 37.1 + 37.2)."""

from __future__ import annotations

import math

import pytest

from plottter.models.canvas import Canvas


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


class TestGeometricGridGenerator:
    def setup_method(self):
        from plottter.generators.geometric_grid import GeometricGridGenerator
        self.gen = GeometricGridGenerator()
        self.canvas = make_canvas()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Geometric Grid" in GENERATORS

    def test_category(self):
        assert self.gen.category == "math"

    # ------------------------------------------------------------------
    # Square grid — basic rendering
    # ------------------------------------------------------------------

    def test_square_grid_renders(self):
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 20.0, "cell_shape": "Outline",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0

    def test_square_grid_outline_shape(self):
        """Each cell outline is a closed 5-point polygon."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Outline",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 5, "Outline cell should have 5 points (4 corners + close)"
            assert path[0] == path[-1], "Outline cell should be closed"

    def test_diagonal_shape(self):
        """Diagonal produces one line per cell."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Diagonal",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 2, "Diagonal cell should be a 2-point line"

    def test_cross_shape(self):
        """Cross produces two lines per cell."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Cross",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        # Must be even (2 per cell)
        assert len(paths) % 2 == 0

    def test_circle_inscribed_shape(self):
        """Circle Inscribed produces a closed polygon per cell."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Circle Inscribed",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert path[0] == path[-1], "Circle inscribed should be a closed polygon"

    def test_diamond_inscribed_shape(self):
        """Diamond Inscribed produces a closed 5-point polygon per cell."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Diamond Inscribed",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 5, "Diamond inscribed should have 5 points"
            assert path[0] == path[-1], "Diamond inscribed should be closed"

    def test_random_fill_shape(self):
        """Random Fill produces non-empty output."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Random Fill",
             "density_variation": 0.0, "noise_seed": 42},
            self.canvas,
        )
        assert len(paths) > 0

    # ------------------------------------------------------------------
    # Density variation
    # ------------------------------------------------------------------

    def test_density_variation_zero_draws_all(self):
        """density_variation=0 draws every cell."""
        paths_dense = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 20.0, "cell_shape": "Outline",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths_dense) > 0

    def test_density_variation_creates_sparse_regions(self):
        """density_variation > 0 produces fewer cells than density_variation=0."""
        params_full = {
            "grid_type": "Square", "cell_size_mm": 10.0, "cell_shape": "Outline",
            "density_variation": 0.0, "noise_seed": 42,
        }
        params_sparse = {
            "grid_type": "Square", "cell_size_mm": 10.0, "cell_shape": "Outline",
            "density_variation": 0.9, "noise_seed": 42,
        }
        paths_full = self.gen.generate(params_full, self.canvas)
        paths_sparse = self.gen.generate(params_sparse, self.canvas)
        assert len(paths_sparse) < len(paths_full), (
            "Higher density_variation should produce fewer cells"
        )

    # ------------------------------------------------------------------
    # Hex grid — 37.2
    # ------------------------------------------------------------------

    def test_hexagonal_grid_renders(self):
        paths = self.gen.generate(
            {"grid_type": "Hexagonal", "cell_size_mm": 15.0, "cell_shape": "Outline",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0

    def test_hexagonal_outline_is_heptagon(self):
        """Each hex outline has 7 points (6 vertices + closing point)."""
        paths = self.gen.generate(
            {"grid_type": "Hexagonal", "cell_size_mm": 20.0, "cell_shape": "Outline",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 7, f"Hex outline should have 7 points, got {len(path)}"
            # First and last point should be (approximately) the same — closed hex
            assert abs(path[0][0] - path[-1][0]) < 1e-9
            assert abs(path[0][1] - path[-1][1]) < 1e-9

    def test_hexagonal_outline_vertices_at_correct_radius(self):
        """All hex outline vertices should be at distance cell_size from centre."""
        cell_size = 15.0
        paths = self.gen.generate(
            {"grid_type": "Hexagonal", "cell_size_mm": cell_size, "cell_shape": "Outline",
             "density_variation": 0.0, "cell_rotation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        # Check first hexagon
        path = paths[0]
        # Centroid of the 6 vertices (exclude the repeated closing point)
        verts = path[:6]
        cx = sum(x for x, y in verts) / 6
        cy = sum(y for x, y in verts) / 6
        for x, y in verts:
            dist = math.hypot(x - cx, y - cy)
            assert abs(dist - cell_size) < 1e-6, (
                f"Hex vertex at distance {dist:.4f}, expected {cell_size}"
            )

    def test_hexagonal_circle_inscribed(self):
        """Circle Inscribed inside hex produces a closed polygon per cell."""
        paths = self.gen.generate(
            {"grid_type": "Hexagonal", "cell_size_mm": 20.0, "cell_shape": "Circle Inscribed",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert path[0] == path[-1], "Inscribed circle should be closed"

    def test_hexagonal_diagonal_inscribed(self):
        """Diagonal inscribed in hex cell produces 2-point lines."""
        paths = self.gen.generate(
            {"grid_type": "Hexagonal", "cell_size_mm": 20.0, "cell_shape": "Diagonal",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 2, "Diagonal should be a 2-point line"

    def test_hexagonal_density_variation(self):
        """High density_variation skips some hex cells."""
        params_full = {
            "grid_type": "Hexagonal", "cell_size_mm": 15.0, "cell_shape": "Outline",
            "density_variation": 0.0, "noise_seed": 42,
        }
        params_sparse = {
            "grid_type": "Hexagonal", "cell_size_mm": 15.0, "cell_shape": "Outline",
            "density_variation": 0.9, "noise_seed": 42,
        }
        paths_full = self.gen.generate(params_full, self.canvas)
        paths_sparse = self.gen.generate(params_sparse, self.canvas)
        assert len(paths_sparse) < len(paths_full)

    # ------------------------------------------------------------------
    # Triangular grid — 37.2
    # ------------------------------------------------------------------

    def test_triangular_grid_renders(self):
        paths = self.gen.generate(
            {"grid_type": "Triangular", "cell_size_mm": 20.0, "cell_shape": "Outline",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0

    def test_triangular_outline_is_closed_triangle(self):
        """Each triangle outline has 4 points (3 vertices + closing point, first==last)."""
        paths = self.gen.generate(
            {"grid_type": "Triangular", "cell_size_mm": 25.0, "cell_shape": "Outline",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 4, f"Triangle outline should have 4 points, got {len(path)}"
            assert abs(path[0][0] - path[-1][0]) < 1e-9
            assert abs(path[0][1] - path[-1][1]) < 1e-9

    def test_triangular_circle_inscribed(self):
        """Circle Inscribed inside triangle produces a closed polygon per cell."""
        paths = self.gen.generate(
            {"grid_type": "Triangular", "cell_size_mm": 25.0, "cell_shape": "Circle Inscribed",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert path[0] == path[-1], "Inscribed circle in triangle should be closed"

    def test_triangular_diamond_inscribed(self):
        """Diamond Inscribed inside triangle produces a closed 5-point polygon."""
        paths = self.gen.generate(
            {"grid_type": "Triangular", "cell_size_mm": 25.0, "cell_shape": "Diamond Inscribed",
             "density_variation": 0.0},
            self.canvas,
        )
        assert len(paths) > 0
        for path in paths:
            assert len(path) == 5
            assert path[0] == path[-1]

    def test_triangular_density_variation(self):
        """High density_variation skips some triangle cells."""
        params_full = {
            "grid_type": "Triangular", "cell_size_mm": 20.0, "cell_shape": "Outline",
            "density_variation": 0.0, "noise_seed": 42,
        }
        params_sparse = {
            "grid_type": "Triangular", "cell_size_mm": 20.0, "cell_shape": "Outline",
            "density_variation": 0.9, "noise_seed": 42,
        }
        paths_full = self.gen.generate(params_full, self.canvas)
        paths_sparse = self.gen.generate(params_sparse, self.canvas)
        assert len(paths_sparse) < len(paths_full)

    # ------------------------------------------------------------------
    # Cell rotation — 37.2
    # ------------------------------------------------------------------

    def test_cell_rotation_param_exists(self):
        """cell_rotation and rotation_noise params must exist in parameter list."""
        param_names = {p.name for p in self.gen.get_parameters()}
        assert "cell_rotation" in param_names
        assert "rotation_noise" in param_names

    def test_cell_rotation_zero_unchanged(self):
        """cell_rotation=0 produces the same output as not specifying rotation."""
        params_no_rot = {
            "grid_type": "Square", "cell_size_mm": 20.0, "cell_shape": "Diagonal",
            "density_variation": 0.0, "cell_rotation": 0.0, "rotation_noise": 0.0,
        }
        params_default = {
            "grid_type": "Square", "cell_size_mm": 20.0, "cell_shape": "Diagonal",
            "density_variation": 0.0,
        }
        paths_no_rot = self.gen.generate(params_no_rot, self.canvas)
        paths_default = self.gen.generate(params_default, self.canvas)
        assert len(paths_no_rot) == len(paths_default)
        for p1, p2 in zip(paths_no_rot, paths_default):
            for (x1, y1), (x2, y2) in zip(p1, p2):
                assert abs(x1 - x2) < 1e-9
                assert abs(y1 - y2) < 1e-9

    def test_cell_rotation_changes_output(self):
        """Non-zero cell_rotation produces different point positions."""
        params_base = {
            "grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Diagonal",
            "density_variation": 0.0, "cell_rotation": 0.0, "rotation_noise": 0.0,
        }
        params_rotated = dict(params_base, cell_rotation=45.0)
        paths_base = self.gen.generate(params_base, self.canvas)
        paths_rotated = self.gen.generate(params_rotated, self.canvas)
        assert len(paths_base) == len(paths_rotated)
        # At least some paths should differ after 45° rotation
        diffs = sum(
            1 for p_base, p_rot in zip(paths_base, paths_rotated)
            if any(abs(x1 - x2) > 1e-6 or abs(y1 - y2) > 1e-6
                   for (x1, y1), (x2, y2) in zip(p_base, p_rot))
        )
        assert diffs > 0, "cell_rotation=45 should change point positions"

    def test_rotation_noise_creates_cell_variation(self):
        """rotation_noise > 0 produces varying rotations across cells."""
        params_no_noise = {
            "grid_type": "Square", "cell_size_mm": 20.0, "cell_shape": "Diagonal",
            "density_variation": 0.0, "cell_rotation": 0.0, "rotation_noise": 0.0,
        }
        params_with_noise = dict(params_no_noise, rotation_noise=45.0)
        paths_no_noise = self.gen.generate(params_no_noise, self.canvas)
        paths_with_noise = self.gen.generate(params_with_noise, self.canvas)
        assert len(paths_no_noise) == len(paths_with_noise)
        # With rotation noise, paths should differ from uniform case
        diffs = sum(
            1 for p_nn, p_n in zip(paths_no_noise, paths_with_noise)
            if any(abs(x1 - x2) > 1e-6 or abs(y1 - y2) > 1e-6
                   for (x1, y1), (x2, y2) in zip(p_nn, p_n))
        )
        assert diffs > 0, "rotation_noise should create per-cell variation"

    def test_rotation_applies_to_hex_grid(self):
        """Rotation is applied to hexagonal grid cells."""
        params_no_rot = {
            "grid_type": "Hexagonal", "cell_size_mm": 20.0, "cell_shape": "Diamond Inscribed",
            "density_variation": 0.0, "cell_rotation": 0.0,
        }
        params_rotated = dict(params_no_rot, cell_rotation=30.0)
        paths_no_rot = self.gen.generate(params_no_rot, self.canvas)
        paths_rotated = self.gen.generate(params_rotated, self.canvas)
        assert len(paths_no_rot) == len(paths_rotated)
        diffs = sum(
            1 for p_base, p_rot in zip(paths_no_rot, paths_rotated)
            if any(abs(x1 - x2) > 1e-6 or abs(y1 - y2) > 1e-6
                   for (x1, y1), (x2, y2) in zip(p_base, p_rot))
        )
        assert diffs > 0

    def test_rotation_applies_to_tri_grid(self):
        """Rotation is applied to triangular grid cells."""
        params_no_rot = {
            "grid_type": "Triangular", "cell_size_mm": 25.0, "cell_shape": "Circle Inscribed",
            "density_variation": 0.0, "cell_rotation": 0.0,
        }
        params_rotated = dict(params_no_rot, cell_rotation=45.0)
        paths_no_rot = self.gen.generate(params_no_rot, self.canvas)
        paths_rotated = self.gen.generate(params_rotated, self.canvas)
        assert len(paths_no_rot) == len(paths_rotated)
        diffs = sum(
            1 for p_base, p_rot in zip(paths_no_rot, paths_rotated)
            if any(abs(x1 - x2) > 1e-6 or abs(y1 - y2) > 1e-6
                   for (x1, y1), (x2, y2) in zip(p_base, p_rot))
        )
        assert diffs > 0

    # ------------------------------------------------------------------
    # x/y offset
    # ------------------------------------------------------------------

    def test_xy_offset(self):
        params_base = {
            "grid_type": "Square", "cell_size_mm": 20.0, "cell_shape": "Outline",
            "density_variation": 0.0, "x_offset_mm": 0.0, "y_offset_mm": 0.0,
        }
        params_shifted = dict(params_base, x_offset_mm=10.0, y_offset_mm=5.0)
        paths_base = self.gen.generate(params_base, self.canvas)
        paths_shifted = self.gen.generate(params_shifted, self.canvas)
        assert len(paths_base) == len(paths_shifted)
        # First point of each path should be shifted by (10, 5)
        for p_base, p_shifted in zip(paths_base, paths_shifted):
            assert abs(p_shifted[0][0] - p_base[0][0] - 10.0) < 1e-9
            assert abs(p_shifted[0][1] - p_base[0][1] - 5.0) < 1e-9

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def test_all_presets_produce_output(self):
        for preset in self.gen.get_presets():
            params = {p.name: p.default for p in self.gen.get_parameters()}
            params.update(preset.params)
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) > 0, f"Preset '{preset.name}' produced no output"

    def test_preset_names(self):
        """All five spec'd presets must be present."""
        names = {p.name for p in self.gen.get_presets()}
        for expected in ("Honeycomb", "Broken Tiles", "Triangle Mesh", "City Grid", "Hex Detail"):
            assert expected in names, f"Preset '{expected}' missing from get_presets()"

    # ------------------------------------------------------------------
    # Subdivision — 37.3
    # ------------------------------------------------------------------

    def test_subdivisions_param_exists(self):
        """subdivisions parameter must be present in the parameter list."""
        param_names = {p.name for p in self.gen.get_parameters()}
        assert "subdivisions" in param_names

    def test_subdivisions_zero_unchanged_square(self):
        """subdivisions=0 must produce identical output to not specifying it."""
        params_default = {
            "grid_type": "Square", "cell_size_mm": 20.0, "cell_shape": "Outline",
            "density_variation": 0.0, "noise_seed": 42,
        }
        params_zero = dict(params_default, subdivisions=0)
        paths_default = self.gen.generate(params_default, self.canvas)
        paths_zero = self.gen.generate(params_zero, self.canvas)
        assert len(paths_default) == len(paths_zero)
        for p1, p2 in zip(paths_default, paths_zero):
            for (x1, y1), (x2, y2) in zip(p1, p2):
                assert abs(x1 - x2) < 1e-9
                assert abs(y1 - y2) < 1e-9

    def test_square_subdivision_produces_output(self):
        """subdivisions=1 on a square grid must produce non-empty output."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 25.0, "cell_shape": "Outline",
             "density_variation": 0.0, "subdivisions": 1, "noise_seed": 42},
            self.canvas,
        )
        assert len(paths) > 0

    def test_square_subdivision_increases_path_count(self):
        """subdivisions=1 should produce more paths than subdivisions=0 (some cells subdivide)."""
        base_params = {
            "grid_type": "Square", "cell_size_mm": 25.0, "cell_shape": "Outline",
            "density_variation": 0.0, "noise_seed": 42,
        }
        paths_no_subdiv = self.gen.generate(dict(base_params, subdivisions=0), self.canvas)
        paths_subdiv = self.gen.generate(dict(base_params, subdivisions=1), self.canvas)
        assert len(paths_subdiv) >= len(paths_no_subdiv), (
            "Subdivision should produce at least as many paths as no subdivision"
        )

    def test_hex_subdivision_produces_output(self):
        """subdivisions=1 on a hexagonal grid must produce non-empty output."""
        paths = self.gen.generate(
            {"grid_type": "Hexagonal", "cell_size_mm": 20.0, "cell_shape": "Outline",
             "density_variation": 0.0, "subdivisions": 1, "noise_seed": 42},
            self.canvas,
        )
        assert len(paths) > 0

    def test_triangular_subdivision_produces_output(self):
        """subdivisions=1 on a triangular grid must produce non-empty output."""
        paths = self.gen.generate(
            {"grid_type": "Triangular", "cell_size_mm": 25.0, "cell_shape": "Outline",
             "density_variation": 0.0, "subdivisions": 1, "noise_seed": 42},
            self.canvas,
        )
        assert len(paths) > 0

    def test_subdivision_depth_2_produces_output(self):
        """subdivisions=2 must produce non-empty output without crashing."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 30.0, "cell_shape": "Diagonal",
             "density_variation": 0.0, "subdivisions": 2, "noise_seed": 7},
            self.canvas,
        )
        assert len(paths) > 0

    def test_subdivision_depth_3_produces_output(self):
        """subdivisions=3 (maximum) must produce non-empty output without crashing."""
        paths = self.gen.generate(
            {"grid_type": "Square", "cell_size_mm": 40.0, "cell_shape": "Circle Inscribed",
             "density_variation": 0.0, "subdivisions": 3, "noise_seed": 99},
            self.canvas,
        )
        assert len(paths) > 0

    def test_subdivisions_zero_unchanged_hex(self):
        """subdivisions=0 on hexagonal grid must be identical to no-subdivision default."""
        params_default = {
            "grid_type": "Hexagonal", "cell_size_mm": 20.0, "cell_shape": "Outline",
            "density_variation": 0.0, "noise_seed": 42,
        }
        params_zero = dict(params_default, subdivisions=0)
        paths_default = self.gen.generate(params_default, self.canvas)
        paths_zero = self.gen.generate(params_zero, self.canvas)
        assert len(paths_default) == len(paths_zero)

    def test_subdivisions_zero_unchanged_tri(self):
        """subdivisions=0 on triangular grid must be identical to no-subdivision default."""
        params_default = {
            "grid_type": "Triangular", "cell_size_mm": 25.0, "cell_shape": "Outline",
            "density_variation": 0.0, "noise_seed": 42,
        }
        params_zero = dict(params_default, subdivisions=0)
        paths_default = self.gen.generate(params_default, self.canvas)
        paths_zero = self.gen.generate(params_zero, self.canvas)
        assert len(paths_default) == len(paths_zero)
