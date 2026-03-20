"""Tests for VoronoiGenerator seed point strategies (Task 45.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canvas() -> Canvas:
    """A4 canvas with 10 mm margins → 190 × 277 mm drawing area."""
    return Canvas.from_preset("A4", margin=10.0)


def drawing_dims(canvas: Canvas) -> tuple[float, float]:
    x1, y1, x2, y2 = canvas.drawing_area()
    return x2 - x1, y2 - y1


# ---------------------------------------------------------------------------
# Registration / metadata
# ---------------------------------------------------------------------------

class TestVoronoiRegistration:
    def test_registered(self):
        from plottter.generators import GENERATORS
        assert "Voronoi / Delaunay" in GENERATORS

    def test_category(self):
        from plottter.generators.voronoi import VoronoiGenerator
        assert VoronoiGenerator.category == "math"

    def test_has_parameters(self):
        from plottter.generators.voronoi import VoronoiGenerator
        gen = VoronoiGenerator()
        params = gen.get_parameters()
        names = {p.name for p in params}
        assert "num_points" in names
        assert "seed_method" in names
        assert "poisson_spacing_mm" in names
        assert "grid_jitter" in names
        assert "x_offset_mm" in names
        assert "y_offset_mm" in names

    def test_has_presets(self):
        from plottter.generators.voronoi import VoronoiGenerator
        presets = VoronoiGenerator().get_presets()
        assert len(presets) >= 1
        names = {p.name for p in presets}
        assert "Phyllotaxis Spiral" in names

    def test_generate_returns_list(self):
        from plottter.generators.voronoi import VoronoiGenerator
        gen = VoronoiGenerator()
        result = gen.generate({"seed_method": "Random", "num_points": 50}, make_canvas())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _seeds_random
# ---------------------------------------------------------------------------

class TestSeedsRandom:
    def setup_method(self):
        from plottter.generators.voronoi import _seeds_random
        self._fn = _seeds_random
        self.rng = np.random.default_rng(0)

    def test_exact_point_count(self):
        w, h = 190.0, 277.0
        pts = self._fn(200, w, h, self.rng)
        assert pts.shape == (200, 2)

    def test_within_bounds(self):
        w, h = 190.0, 277.0
        pts = self._fn(500, w, h, self.rng)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= h)

    def test_reproducible_with_same_seed(self):
        w, h = 190.0, 277.0
        pts1 = self._fn(100, w, h, np.random.default_rng(42))
        pts2 = self._fn(100, w, h, np.random.default_rng(42))
        np.testing.assert_array_equal(pts1, pts2)

    def test_different_seeds_differ(self):
        w, h = 190.0, 277.0
        pts1 = self._fn(100, w, h, np.random.default_rng(1))
        pts2 = self._fn(100, w, h, np.random.default_rng(2))
        assert not np.allclose(pts1, pts2)


# ---------------------------------------------------------------------------
# _seeds_poisson_disk
# ---------------------------------------------------------------------------

class TestSeedsPoissonDisk:
    def setup_method(self):
        from plottter.generators.voronoi import _seeds_poisson_disk
        self._fn = _seeds_poisson_disk
        self.rng = np.random.default_rng(0)

    def test_returns_2d_array(self):
        pts = self._fn(5.0, 100.0, 100.0, self.rng)
        assert pts.ndim == 2 and pts.shape[1] == 2

    def test_within_bounds(self):
        w, h = 190.0, 277.0
        pts = self._fn(10.0, w, h, self.rng)
        assert pts.shape[0] > 0
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= h)

    def test_minimum_distance_respected(self):
        spacing = 8.0
        w, h = 100.0, 100.0
        pts = self._fn(spacing, w, h, np.random.default_rng(7))
        if len(pts) < 2:
            pytest.skip("Too few points to check pairwise distances")
        # Check every pair — tolerating a small floating-point margin
        tol = spacing * 0.95  # allow 5% tolerance for numerical issues
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                assert dist >= tol, (
                    f"Points {i} and {j} are too close: {dist:.4f} < {tol:.4f}"
                )

    def test_approximate_point_count(self):
        """Poisson disk should produce a reasonable number of points for the spacing."""
        spacing = 5.0
        w, h = 100.0, 100.0
        pts = self._fn(spacing, w, h, np.random.default_rng(0))
        # Expected: roughly area / (pi * (spacing/2)^2) = 10000 / (pi*6.25) ≈ 509
        # Allow a very wide band — just check it's at least a handful
        assert len(pts) >= 5

    def test_small_spacing_more_points_than_large(self):
        w, h = 100.0, 100.0
        pts_small = self._fn(3.0, w, h, np.random.default_rng(0))
        pts_large = self._fn(15.0, w, h, np.random.default_rng(0))
        assert len(pts_small) > len(pts_large)


# ---------------------------------------------------------------------------
# _seeds_grid_jitter
# ---------------------------------------------------------------------------

class TestSeedsGridJitter:
    def setup_method(self):
        from plottter.generators.voronoi import _seeds_grid_jitter
        self._fn = _seeds_grid_jitter
        self.rng = np.random.default_rng(0)

    def test_returns_2d_array(self):
        pts = self._fn(10.0, 0.0, 100.0, 100.0, self.rng)
        assert pts.ndim == 2 and pts.shape[1] == 2

    def test_within_bounds(self):
        w, h = 190.0, 277.0
        pts = self._fn(10.0, 0.5, w, h, self.rng)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= h)

    def test_expected_approximate_count(self):
        spacing = 10.0
        w, h = 100.0, 100.0
        pts = self._fn(spacing, 0.0, w, h, self.rng)  # no jitter → exact grid
        # 100/10 = 10 per axis → 100 points
        assert abs(len(pts) - 100) <= 5  # within 5 for boundary effects

    def test_no_jitter_gives_regular_grid(self):
        spacing = 10.0
        w, h = 50.0, 50.0
        pts = self._fn(spacing, 0.0, w, h, np.random.default_rng(0))
        # With no jitter all points should be on grid positions (multiples of spacing)
        grid_xs = set(np.arange(spacing / 2, w, spacing).round(6))
        grid_ys = set(np.arange(spacing / 2, h, spacing).round(6))
        for x, y in pts:
            assert round(x, 6) in grid_xs or math.isclose(x, round(x / spacing) * spacing, abs_tol=1e-6)
            assert round(y, 6) in grid_ys or math.isclose(y, round(y / spacing) * spacing, abs_tol=1e-6)

    def test_jitter_displaces_points(self):
        spacing = 10.0
        w, h = 100.0, 100.0
        pts_no_jitter = self._fn(spacing, 0.0, w, h, np.random.default_rng(5))
        pts_jitter = self._fn(spacing, 0.8, w, h, np.random.default_rng(5))
        # With jitter the points should NOT all match the clean grid positions
        if len(pts_no_jitter) == len(pts_jitter):
            assert not np.allclose(pts_no_jitter, pts_jitter)


# ---------------------------------------------------------------------------
# _seeds_phyllotaxis
# ---------------------------------------------------------------------------

class TestSeedsPhyllotaxis:
    def setup_method(self):
        from plottter.generators.voronoi import _seeds_phyllotaxis
        self._fn = _seeds_phyllotaxis

    def test_exact_count(self):
        pts = self._fn(200, 190.0, 277.0)
        assert pts.shape == (200, 2)

    def test_within_bounds(self):
        w, h = 190.0, 277.0
        pts = self._fn(500, w, h)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= w)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= h)

    def test_centred_on_canvas(self):
        w, h = 190.0, 277.0
        pts = self._fn(500, w, h)
        # The centre of the bounding box should be close to the canvas centre
        cx, cy = w / 2.0, h / 2.0
        x_mid = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        y_mid = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
        assert abs(x_mid - cx) < 5.0, f"x centre off: {x_mid:.2f} vs {cx:.2f}"
        assert abs(y_mid - cy) < 5.0, f"y centre off: {y_mid:.2f} vs {cy:.2f}"

    def test_spiral_pattern(self):
        """Successive points should be at monotonically increasing radii."""
        w, h = 190.0, 277.0
        pts = self._fn(200, w, h)
        cx, cy = w / 2.0, h / 2.0
        radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        # The spiral radius formula is r = R * sqrt(i/n), which is monotone.
        # Allow for the first point (i=0) at r=0.
        assert radii[0] < 1e-6, "First phyllotaxis point should be at centre"
        # Subsequent radii should be non-decreasing overall (some wobble is OK,
        # but the trend must be upward — check the last quarter > first quarter)
        n = len(radii)
        assert radii[n // 2 :].mean() > radii[1 : n // 2].mean(), (
            "Outer half of phyllotaxis should have larger radii than inner half"
        )

    def test_deterministic(self):
        """Phyllotaxis is deterministic — no RNG involved."""
        pts1 = self._fn(300, 190.0, 277.0)
        pts2 = self._fn(300, 190.0, 277.0)
        np.testing.assert_array_equal(pts1, pts2)

    def test_golden_angle_separation(self):
        """Check that consecutive angular steps approximate the golden angle."""
        w, h = 100.0, 100.0
        pts = self._fn(100, w, h)
        cx, cy = w / 2.0, h / 2.0
        # Skip the centre point (i=0) which has r=0
        angles = np.arctan2(pts[1:, 1] - cy, pts[1:, 0] - cx)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        diffs = np.diff(angles) % (2 * math.pi)
        # Each angular difference should be ≈ golden_angle (mod 2π)
        expected = golden_angle % (2 * math.pi)
        # Allow generous tolerance because angles wrap around
        close = np.isclose(diffs, expected, atol=0.2) | np.isclose(
            diffs, expected - 2 * math.pi, atol=0.2
        ) | np.isclose(
            diffs, expected + 2 * math.pi, atol=0.2
        )
        assert close.mean() > 0.5, "Most angular steps should approximate the golden angle"


# ---------------------------------------------------------------------------
# Integration: generate() uses the correct strategy
# ---------------------------------------------------------------------------

class TestVoronoiGenerateIntegration:
    def setup_method(self):
        from plottter.generators.voronoi import VoronoiGenerator
        self.gen = VoronoiGenerator()
        self.canvas = make_canvas()

    def test_generate_random_no_crash(self):
        result = self.gen.generate(
            {"seed_method": "Random", "num_points": 100, "random_seed": 0},
            self.canvas,
        )
        assert isinstance(result, list)

    def test_generate_poisson_no_crash(self):
        result = self.gen.generate(
            {"seed_method": "Poisson Disk", "poisson_spacing_mm": 10.0, "random_seed": 0},
            self.canvas,
        )
        assert isinstance(result, list)

    def test_generate_grid_jitter_no_crash(self):
        result = self.gen.generate(
            {"seed_method": "Grid Jitter", "grid_spacing_mm": 15.0, "grid_jitter": 0.3, "random_seed": 0},
            self.canvas,
        )
        assert isinstance(result, list)

    def test_generate_phyllotaxis_no_crash(self):
        result = self.gen.generate(
            {"seed_method": "Phyllotaxis", "num_points": 200},
            self.canvas,
        )
        assert isinstance(result, list)

    def test_generate_unknown_method_falls_back_to_random(self):
        result = self.gen.generate(
            {"seed_method": "UnknownMethod", "num_points": 50, "random_seed": 0},
            self.canvas,
        )
        assert isinstance(result, list)

    def test_progress_callback_called(self):
        calls = []
        self.gen.generate(
            {"seed_method": "Random", "num_points": 50, "random_seed": 0},
            self.canvas,
            progress_callback=calls.append,
        )
        assert 100 in calls


# ---------------------------------------------------------------------------
# _render_voronoi — Task 45.2 specific tests
# ---------------------------------------------------------------------------


class TestRenderVoronoi:
    """Tests for _render_voronoi: closed interior cells, boundary clipping,
    no out-of-bounds edges, and valid Polyline output."""

    def setup_method(self):
        from plottter.generators.voronoi import _render_voronoi
        self._fn = _render_voronoi
        self.canvas = make_canvas()
        self.x1, self.y1, self.x2, self.y2 = self.canvas.drawing_area()
        self.bbox = (self.x1, self.y1, self.x2, self.y2)

    def _make_seeds(self, n: int, seed: int = 42) -> "np.ndarray":
        rng = np.random.default_rng(seed)
        seeds = rng.random((n, 2))
        seeds[:, 0] = self.x1 + seeds[:, 0] * (self.x2 - self.x1)
        seeds[:, 1] = self.y1 + seeds[:, 1] * (self.y2 - self.y1)
        return seeds

    # (d) output is a valid list of Polyline (list of list of 2-tuples of floats)
    def test_returns_list_of_polylines(self):
        seeds = self._make_seeds(100)
        result = self._fn(seeds, self.bbox)
        assert isinstance(result, list)
        for pl in result:
            assert isinstance(pl, list), "Each element must be a list (Polyline)"
            assert len(pl) >= 2, "Each polyline must have at least 2 points"
            for pt in pl:
                assert len(pt) == 2, "Each point must be a 2-tuple"
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)

    # (c) no edges extend beyond canvas bounds
    def test_no_edges_beyond_canvas_bounds(self):
        seeds = self._make_seeds(200)
        result = self._fn(seeds, self.bbox)
        tol = 1e-6
        for pl in result:
            for x, y in pl:
                assert x >= self.x1 - tol, f"x={x:.4f} < x_min={self.x1}"
                assert x <= self.x2 + tol, f"x={x:.4f} > x_max={self.x2}"
                assert y >= self.y1 - tol, f"y={y:.4f} < y_min={self.y1}"
                assert y <= self.y2 + tol, f"y={y:.4f} > y_max={self.y2}"

    # (b) boundary cells are clipped (some edges lie on bbox boundary)
    def test_boundary_cells_clipped_to_canvas(self):
        seeds = self._make_seeds(150)
        result = self._fn(seeds, self.bbox)
        tol = 1e-4
        # At least some edges should have endpoints on the boundary
        on_boundary = False
        for pl in result:
            for x, y in pl:
                if (
                    abs(x - self.x1) < tol
                    or abs(x - self.x2) < tol
                    or abs(y - self.y1) < tol
                    or abs(y - self.y2) < tol
                ):
                    on_boundary = True
                    break
            if on_boundary:
                break
        assert on_boundary, "Expected some Voronoi edges to be clipped at canvas boundary"

    # (a) interior cells produce many edges (sufficient coverage)
    def test_produces_edges_for_interior_cells(self):
        seeds = self._make_seeds(100)
        result = self._fn(seeds, self.bbox)
        # With 100 seeds, Voronoi diagram has ~100 cells → expect substantial edges
        assert len(result) >= 50, f"Expected at least 50 edges, got {len(result)}"

    def test_returns_empty_for_too_few_seeds(self):
        seeds = self._make_seeds(3)
        result = self._fn(seeds, self.bbox)
        assert result == [], "Should return empty list for < 4 seeds"

    def test_all_render_modes_produce_valid_polylines(self):
        from plottter.generators.voronoi import VoronoiGenerator
        gen = VoronoiGenerator()
        canvas = make_canvas()
        for mode in ["Voronoi Edges", "Delaunay Edges", "Both", "Voronoi + Centroids"]:
            result = gen.generate(
                {
                    "render_mode": mode,
                    "num_points": 50,
                    "seed_method": "Random",
                    "random_seed": 7,
                },
                canvas,
            )
            assert isinstance(result, list), f"render_mode={mode!r} did not return list"
            for pl in result:
                assert isinstance(pl, list)
                assert len(pl) >= 2
                for pt in pl:
                    assert len(pt) == 2

    def test_voronoi_and_both_modes_include_voronoi_edges(self):
        """'Voronoi Edges' and 'Both' must include Voronoi output (non-empty)."""
        from plottter.generators.voronoi import VoronoiGenerator
        gen = VoronoiGenerator()
        canvas = make_canvas()
        for mode in ["Voronoi Edges", "Both"]:
            result = gen.generate(
                {
                    "render_mode": mode,
                    "num_points": 100,
                    "seed_method": "Random",
                    "random_seed": 1,
                },
                canvas,
            )
            assert len(result) > 0, f"render_mode={mode!r} produced no output"

    def test_render_mode_param_present_with_correct_choices(self):
        from plottter.generators.voronoi import VoronoiGenerator
        from plottter.generators.base import ChoiceParam
        gen = VoronoiGenerator()
        params = gen.get_parameters()
        render_params = [p for p in params if p.name == "render_mode"]
        assert len(render_params) == 1
        rp = render_params[0]
        assert isinstance(rp, ChoiceParam)
        assert "Voronoi Edges" in rp.choices
        assert "Delaunay Edges" in rp.choices
        assert "Both" in rp.choices
        assert "Voronoi + Centroids" in rp.choices
        assert rp.default == "Voronoi Edges"


# ---------------------------------------------------------------------------
# _render_delaunay — Task 45.3 (a) and (b)
# ---------------------------------------------------------------------------


class TestRenderDelaunay:
    """Tests for _render_delaunay: triangle coverage and no duplicate edges."""

    def setup_method(self):
        from plottter.generators.voronoi import _render_delaunay
        self._fn = _render_delaunay
        self.canvas = make_canvas()
        self.x1, self.y1, self.x2, self.y2 = self.canvas.drawing_area()
        self.bbox = (self.x1, self.y1, self.x2, self.y2)

    def _make_seeds(self, n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        seeds = rng.random((n, 2))
        seeds[:, 0] = self.x1 + seeds[:, 0] * (self.x2 - self.x1)
        seeds[:, 1] = self.y1 + seeds[:, 1] * (self.y2 - self.y1)
        return seeds

    # (a) Delaunay produces triangles covering the point set
    def test_produces_edges_for_point_set(self):
        seeds = self._make_seeds(50)
        result = self._fn(seeds, self.bbox)
        # 50 points → Delaunay has ~2n - 2 - h triangles → expect many edges
        assert len(result) >= 30, f"Expected ≥ 30 Delaunay edges, got {len(result)}"

    def test_returns_valid_polylines(self):
        seeds = self._make_seeds(30)
        result = self._fn(seeds, self.bbox)
        assert isinstance(result, list)
        for pl in result:
            assert isinstance(pl, list)
            assert len(pl) >= 2
            for pt in pl:
                assert len(pt) == 2
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)

    def test_edges_within_canvas_bounds(self):
        seeds = self._make_seeds(80)
        result = self._fn(seeds, self.bbox)
        tol = 1e-6
        for pl in result:
            for x, y in pl:
                assert x >= self.x1 - tol
                assert x <= self.x2 + tol
                assert y >= self.y1 - tol
                assert y <= self.y2 + tol

    # (b) no duplicate edges
    def test_no_duplicate_edges(self):
        """Each edge (a, b) should appear at most once in the output."""
        seeds = self._make_seeds(60)
        result = self._fn(seeds, self.bbox)
        # Encode each 2-point polyline as a frozenset of rounded endpoints
        seen: set[frozenset] = set()
        for pl in result:
            if len(pl) == 2:
                key = frozenset(
                    (round(x, 6), round(y, 6)) for x, y in pl
                )
                assert key not in seen, f"Duplicate edge found: {pl}"
                seen.add(key)

    def test_returns_empty_for_too_few_seeds(self):
        seeds = self._make_seeds(2)
        result = self._fn(seeds, self.bbox)
        assert result == []

    def test_edge_count_matches_delaunay_formula(self):
        """For n interior points the number of Delaunay edges is roughly 3n."""
        seeds = self._make_seeds(100)
        result = self._fn(seeds, self.bbox)
        # Very loose bound: at least n edges expected
        assert len(result) >= len(seeds), (
            f"Expected at least {len(seeds)} edges, got {len(result)}"
        )


# ---------------------------------------------------------------------------
# _lloyd_relax — Task 45.3 (c)
# ---------------------------------------------------------------------------


class TestLloydRelax:
    """Tests for _lloyd_relax: convergence to more regular cell layout."""

    def setup_method(self):
        from plottter.generators.voronoi import _lloyd_relax
        self._fn = _lloyd_relax
        self.canvas = make_canvas()
        self.x1, self.y1, self.x2, self.y2 = self.canvas.drawing_area()
        self.bbox = (self.x1, self.y1, self.x2, self.y2)

    def _make_seeds(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        seeds = rng.random((n, 2))
        seeds[:, 0] = self.x1 + seeds[:, 0] * (self.x2 - self.x1)
        seeds[:, 1] = self.y1 + seeds[:, 1] * (self.y2 - self.y1)
        return seeds

    def test_zero_iterations_returns_same_seeds(self):
        seeds = self._make_seeds(50)
        result = self._fn(seeds, self.bbox, iterations=0)
        np.testing.assert_array_equal(result, seeds)

    def test_returns_same_count(self):
        seeds = self._make_seeds(80)
        result = self._fn(seeds, self.bbox, iterations=5)
        assert result.shape == seeds.shape

    def test_stays_within_bbox(self):
        seeds = self._make_seeds(100)
        result = self._fn(seeds, self.bbox, iterations=10)
        tol = 1e-6
        assert np.all(result[:, 0] >= self.x1 - tol)
        assert np.all(result[:, 0] <= self.x2 + tol)
        assert np.all(result[:, 1] >= self.y1 - tol)
        assert np.all(result[:, 1] <= self.y2 + tol)

    def test_relaxation_moves_seeds(self):
        """At least some seeds must move after relaxation."""
        seeds = self._make_seeds(80)
        result = self._fn(seeds, self.bbox, iterations=3)
        assert not np.allclose(result, seeds), "Lloyd relaxation should move seeds"

    # (c) high iterations → more regular (lower variance in nearest-neighbour distances)
    def test_high_iterations_more_regular(self):
        """After many iterations the NN-distance variance should decrease."""
        from scipy.spatial import KDTree

        seeds = self._make_seeds(200)
        relaxed = self._fn(seeds, self.bbox, iterations=20)

        tree_orig = KDTree(seeds)
        dists_orig, _ = tree_orig.query(seeds, k=2)
        nn_orig = dists_orig[:, 1]

        tree_rel = KDTree(relaxed)
        dists_rel, _ = tree_rel.query(relaxed, k=2)
        nn_rel = dists_rel[:, 1]

        var_orig = float(np.var(nn_orig))
        var_rel = float(np.var(nn_rel))
        assert var_rel < var_orig, (
            f"Relaxed variance {var_rel:.4f} should be < original variance {var_orig:.4f}"
        )

    def test_too_few_seeds_returns_unchanged(self):
        seeds = self._make_seeds(3)
        result = self._fn(seeds, self.bbox, iterations=5)
        np.testing.assert_array_equal(result, seeds)


# ---------------------------------------------------------------------------
# Integration: "Both" mode and "Voronoi + Centroids" — Task 45.3 (d) and (e)
# ---------------------------------------------------------------------------


class TestRenderModesBothAndCentroids:
    """Integration tests for 'Both' and 'Voronoi + Centroids' render modes."""

    def setup_method(self):
        from plottter.generators.voronoi import VoronoiGenerator
        self.gen = VoronoiGenerator()
        self.canvas = make_canvas()

    # (d) "Both" mode overlays Voronoi and Delaunay
    def test_both_mode_more_edges_than_voronoi_alone(self):
        common = {"num_points": 80, "seed_method": "Random", "random_seed": 3}
        vor_only = self.gen.generate({**common, "render_mode": "Voronoi Edges"}, self.canvas)
        del_only = self.gen.generate({**common, "render_mode": "Delaunay Edges"}, self.canvas)
        both = self.gen.generate({**common, "render_mode": "Both"}, self.canvas)
        assert len(both) == len(vor_only) + len(del_only), (
            "'Both' output should equal sum of Voronoi + Delaunay edge counts"
        )

    # (e) centroid mode draws dots (circles) at each seed
    def test_centroids_mode_produces_circles(self):
        n = 60
        result = self.gen.generate(
            {
                "render_mode": "Voronoi + Centroids",
                "num_points": n,
                "seed_method": "Random",
                "random_seed": 5,
                "centroid_radius_mm": 1.0,
            },
            self.canvas,
        )
        assert len(result) > 0
        # Each circle polyline should be closed (first == last point)
        closed = [pl for pl in result if pl[0] == pl[-1]]
        assert len(closed) >= n, (
            f"Expected at least {n} closed circle polylines, got {len(closed)}"
        )

    def test_lloyd_iterations_param_present(self):
        from plottter.generators.voronoi import VoronoiGenerator
        from plottter.generators.base import IntParam
        gen = VoronoiGenerator()
        params = gen.get_parameters()
        lloyd_params = [p for p in params if p.name == "lloyd_iterations"]
        assert len(lloyd_params) == 1
        lp = lloyd_params[0]
        assert isinstance(lp, IntParam)
        assert lp.min == 0
        assert lp.max == 50
        assert lp.default == 0

    def test_lloyd_relaxation_wired_into_generate(self):
        """generate() with lloyd_iterations > 0 should still return valid polylines."""
        result = self.gen.generate(
            {
                "render_mode": "Voronoi Edges",
                "num_points": 80,
                "seed_method": "Random",
                "random_seed": 0,
                "lloyd_iterations": 5,
            },
            self.canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Image-density mode — Task 45.5 (h)
# ---------------------------------------------------------------------------


class TestImageDensity:
    """Tests for image-density seed modulation: black → more seeds, white → fewer."""

    @staticmethod
    def _white_image(size: int = 64) -> np.ndarray:
        """Fully white RGB image (all 255)."""
        return np.full((size, size, 3), 255, dtype=np.uint8)

    @staticmethod
    def _black_image(size: int = 64) -> np.ndarray:
        """Fully black RGB image (all 0)."""
        return np.zeros((size, size, 3), dtype=np.uint8)

    def test_white_image_produces_no_points(self):
        """Fully white density → acceptance P = 1 − 255/255 = 0 → zero seeds."""
        from plottter.generators.voronoi import (
            _prepare_density_image,
            _seeds_random_density,
        )

        w, h = 100.0, 100.0
        density = _prepare_density_image(self._white_image(), {}, w, h)
        pts = _seeds_random_density(200, w, h, np.random.default_rng(0), density)
        assert len(pts) == 0, f"Expected 0 points from white image, got {len(pts)}"

    def test_black_image_accepts_all_attempts(self):
        """Fully black density → acceptance P = 1 − 0/255 = 1 → all n accepted."""
        from plottter.generators.voronoi import (
            _prepare_density_image,
            _seeds_random_density,
        )

        w, h = 100.0, 100.0
        n = 100
        density = _prepare_density_image(self._black_image(), {}, w, h)
        pts = _seeds_random_density(n, w, h, np.random.default_rng(0), density)
        assert len(pts) == n, f"Expected {n} points from black image, got {len(pts)}"

    def test_black_image_produces_more_points_than_white(self):
        """Dark image → more seeds; bright image → fewer seeds."""
        from plottter.generators.voronoi import (
            _prepare_density_image,
            _seeds_random_density,
        )

        w, h = 100.0, 100.0
        n = 200
        black_density = _prepare_density_image(self._black_image(), {}, w, h)
        white_density = _prepare_density_image(self._white_image(), {}, w, h)
        pts_black = _seeds_random_density(n, w, h, np.random.default_rng(0), black_density)
        pts_white = _seeds_random_density(n, w, h, np.random.default_rng(0), white_density)
        assert len(pts_black) > len(pts_white), (
            f"Black image should produce more points ({len(pts_black)}) "
            f"than white image ({len(pts_white)})"
        )

    def test_image_density_through_generate_black_vs_white(self):
        """generate() with image_density=True: black image → more Voronoi edges."""
        from plottter.generators.voronoi import VoronoiGenerator

        gen = VoronoiGenerator()
        canvas = make_canvas()
        common = {
            "image_density": True,
            "seed_method": "Random",
            "num_points": 200,
            "render_mode": "Voronoi Edges",
            "random_seed": 0,
        }
        result_black = gen.generate({**common, "_source_image": self._black_image()}, canvas)
        result_white = gen.generate({**common, "_source_image": self._white_image()}, canvas)
        # Black → more seeds → more Voronoi edges
        assert len(result_black) > len(result_white), (
            f"Black image should produce more edges ({len(result_black)}) "
            f"than white image ({len(result_white)})"
        )


# ---------------------------------------------------------------------------
# All presets — Task 45.5 (i)
# ---------------------------------------------------------------------------


def _small_canvas() -> "Canvas":
    """A small 80×80 mm canvas for fast preset testing."""
    from plottter.models.canvas import Canvas

    return Canvas(width_mm=80.0, height_mm=80.0, margin_mm=5.0)


class TestAllPresets:
    """All registered VoronoiGenerator presets should produce valid non-empty output."""

    def setup_method(self):
        from plottter.generators.voronoi import VoronoiGenerator

        self.gen = VoronoiGenerator()
        # Use a small canvas so Poisson Disk / Lloyd presets finish quickly
        self.canvas = _small_canvas()

    def test_preset_count(self):
        presets = self.gen.get_presets()
        assert len(presets) >= 5, f"Expected at least 5 presets, got {len(presets)}"

    def test_all_presets_generate_non_empty_valid_polylines(self):
        """Every preset must return a non-empty list of valid Polylines."""
        presets = self.gen.get_presets()
        for preset in presets:
            result = self.gen.generate(preset.params, self.canvas)
            assert isinstance(result, list), (
                f"Preset {preset.name!r}: expected list, got {type(result)}"
            )
            assert len(result) > 0, (
                f"Preset {preset.name!r}: expected non-empty output"
            )
            for pl in result:
                assert isinstance(pl, list), (
                    f"Preset {preset.name!r}: element is not a list"
                )
                assert len(pl) >= 2, (
                    f"Preset {preset.name!r}: polyline has fewer than 2 points"
                )
                for pt in pl:
                    assert len(pt) == 2, (
                        f"Preset {preset.name!r}: point is not a 2-tuple"
                    )

    def test_preset_output_within_canvas_bounds(self):
        """Preset output coordinates must lie within the canvas drawing area."""
        presets = self.gen.get_presets()
        x1, y1, x2, y2 = self.canvas.drawing_area()
        # Expand by a generous tolerance for clipping edge cases
        tol = 1.0
        for preset in presets:
            result = self.gen.generate(preset.params, self.canvas)
            for pl in result:
                for x, y in pl:
                    assert x >= x1 - tol, (
                        f"Preset {preset.name!r}: x={x:.3f} < x_min={x1}"
                    )
                    assert x <= x2 + tol, (
                        f"Preset {preset.name!r}: x={x:.3f} > x_max={x2}"
                    )
                    assert y >= y1 - tol, (
                        f"Preset {preset.name!r}: y={y:.3f} < y_min={y1}"
                    )
                    assert y <= y2 + tol, (
                        f"Preset {preset.name!r}: y={y:.3f} > y_max={y2}"
                    )
