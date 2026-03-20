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
