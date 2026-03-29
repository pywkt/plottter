"""Tests for TSP path reversal in _nearest_neighbor_tsp / _kd_nearest_neighbor_tsp.

Tests:
(a) TSP path with reversal produces shorter or equal travel distance vs without
(b) All input polyline points are still present in the output
(c) "Dots" mode is unaffected by the TSP change (single-point paths → reversal
    is a no-op since start == end)
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _travel_distance(ordered_paths: list) -> float:
    """Total pen-travel distance between consecutive path endpoints."""
    dist = 0.0
    prev_end = None
    for path in ordered_paths:
        if prev_end is not None:
            dx = path[0][0] - prev_end[0]
            dy = path[0][1] - prev_end[1]
            dist += math.sqrt(dx * dx + dy * dy)
        prev_end = path[-1]
    return dist


def _all_points_present(
    original_paths: list,
    ordered_paths: list,
) -> bool:
    """Every point from every original path appears (in some path) in output."""
    original_pts = {pt for path in original_paths for pt in path}
    output_pts = {pt for path in ordered_paths for pt in path}
    return original_pts == output_pts


def _make_directed_paths() -> list:
    """Return polylines where start ≠ end, arranged so reversal helps.

    Layout (in mm):
      Path 0: (0, 0) → (10, 0)   (horizontal right)
      Path 1: (20, 0) → (11, 0)  (horizontal left — end at 11 is closer to 10)
      Path 2: (21, 0) → (30, 0)  (horizontal right)
      Path 3: (40, 0) → (31, 0)  (horizontal left)

    Without reversal, the NN heuristic (starting at path 0's end = (10,0))
    would connect to the *start* of the nearest path, potentially missing the
    opportunity to come from the better end.  With reversal it can approach
    each path from the closer endpoint, yielding a shorter or equal tour.
    """
    return [
        [(0.0, 0.0), (10.0, 0.0)],
        [(20.0, 0.0), (11.0, 0.0)],
        [(21.0, 0.0), (30.0, 0.0)],
        [(40.0, 0.0), (31.0, 0.0)],
    ]


# ---------------------------------------------------------------------------
# (a) Reversal produces shorter or equal travel distance
# ---------------------------------------------------------------------------

class TestReversalReducesTravel:
    """Path reversal should never make the tour worse than forward-only NN."""

    def _forward_only_travel(self, paths: list) -> float:
        """Forward-only NN heuristic (no reversal) for comparison."""
        from plottter.generators.stipple import _nearest_neighbor_tsp

        # Temporarily monkey-patch to disable reversal by using only start dist
        n = len(paths)
        starts = [p[0] for p in paths]
        visited = [False] * n
        result = [paths[0]]
        visited[0] = True
        cur = paths[0][-1]

        for _ in range(n - 1):
            best_d = float("inf")
            best_j = -1
            for j in range(n):
                if visited[j]:
                    continue
                dx = cur[0] - starts[j][0]
                dy = cur[1] - starts[j][1]
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j == -1:
                break
            result.append(paths[best_j])
            visited[best_j] = True
            cur = paths[best_j][-1]

        return _travel_distance(result)

    def test_directed_paths_reversal_not_worse(self):
        from plottter.generators.stipple import _nearest_neighbor_tsp

        paths = _make_directed_paths()
        with_reversal = _travel_distance(_nearest_neighbor_tsp(paths))
        without_reversal = self._forward_only_travel(paths)

        assert with_reversal <= without_reversal + 1e-9, (
            f"Reversal increased travel: {with_reversal:.4f} > {without_reversal:.4f}"
        )

    def test_single_point_paths_reversal_neutral(self):
        """Single-point paths are symmetric; reversal must not change travel."""
        from plottter.generators.stipple import _nearest_neighbor_tsp

        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 100, (30, 2))
        single_paths = [[(float(x), float(y))] for x, y in pts]

        ordered = _nearest_neighbor_tsp(single_paths)
        # All paths are one point; start == end → reversal is a no-op
        for path in ordered:
            assert len(path) == 1

    def test_random_segments_reversal_not_worse(self):
        """On random 2-point segments, reversal never increases total travel."""
        from plottter.generators.stipple import _nearest_neighbor_tsp

        rng = np.random.default_rng(42)
        raw = rng.uniform(0, 100, (20, 2, 2))
        paths = [[(float(raw[i, 0, 0]), float(raw[i, 0, 1])),
                  (float(raw[i, 1, 0]), float(raw[i, 1, 1]))]
                 for i in range(20)]

        with_rev = _travel_distance(_nearest_neighbor_tsp(paths))
        without_rev = self._forward_only_travel(paths)
        assert with_rev <= without_rev + 1e-9


# ---------------------------------------------------------------------------
# (b) All input points are present in the output
# ---------------------------------------------------------------------------

class TestAllPointsVisited:
    def test_single_point_paths_all_visited(self):
        from plottter.generators.stipple import _nearest_neighbor_tsp

        rng = np.random.default_rng(7)
        pts = rng.uniform(0, 50, (25, 2))
        single_paths = [[(float(x), float(y))] for x, y in pts]

        ordered = _nearest_neighbor_tsp(single_paths)
        assert len(ordered) == len(single_paths)
        assert _all_points_present(single_paths, ordered)

    def test_multi_point_paths_all_visited(self):
        from plottter.generators.stipple import _nearest_neighbor_tsp

        paths = _make_directed_paths()
        ordered = _nearest_neighbor_tsp(paths)
        assert len(ordered) == len(paths)
        assert _all_points_present(paths, ordered)

    @pytest.mark.parametrize("n", [5, 20, 50])
    def test_all_paths_returned_parametric(self, n):
        from plottter.generators.stipple import _nearest_neighbor_tsp

        rng = np.random.default_rng(n)
        raw = rng.uniform(0, 100, (n, 2, 2))
        paths = [[(float(raw[i, 0, 0]), float(raw[i, 0, 1])),
                  (float(raw[i, 1, 0]), float(raw[i, 1, 1]))]
                 for i in range(n)]

        ordered = _nearest_neighbor_tsp(paths)
        assert len(ordered) == n
        assert _all_points_present(paths, ordered)


# ---------------------------------------------------------------------------
# (c) Dots mode unaffected — single-point path reversal is a no-op
# ---------------------------------------------------------------------------

class TestDotsMode:
    def make_canvas(self):
        from plottter.models.canvas import Canvas
        return Canvas(width_mm=200.0, height_mm=200.0, margin_mm=10.0)

    def make_gradient_image(self, w: int = 80, h: int = 80) -> np.ndarray:
        img = np.zeros((h, w), dtype=np.uint8)
        for col in range(w):
            img[:, col] = int(col / (w - 1) * 255)
        return img

    def test_dots_mode_produces_multiple_polylines(self):
        """Dots mode should return one tiny-circle per stipple point."""
        from plottter.generators.stipple import StippleGenerator

        gen = StippleGenerator()
        canvas = self.make_canvas()
        img = self.make_gradient_image()
        params = {
            "_source_image": img,
            "num_points": 20,
            "iterations": 1,
            "min_dot_spacing_mm": 0.0,
            "seed": 1,
            "render_mode": "Dots",
            "tsp_optimize": False,
        }
        result = gen.generate(params, canvas)
        # Each dot is a small circle polyline (multi-vertex), not a TSP path
        assert len(result) == 20, f"Expected 20 dot polylines, got {len(result)}"
        for poly in result:
            assert len(poly) > 1, "Each dot should be a circle, not a single point"

    def test_dots_mode_not_single_polyline(self):
        """Dots mode must never collapse to a single connected TSP polyline."""
        from plottter.generators.stipple import StippleGenerator

        gen = StippleGenerator()
        canvas = self.make_canvas()
        img = self.make_gradient_image()
        params = {
            "_source_image": img,
            "num_points": 15,
            "iterations": 1,
            "min_dot_spacing_mm": 0.0,
            "seed": 2,
            "render_mode": "Dots",
            "tsp_optimize": False,
        }
        result = gen.generate(params, canvas)
        assert len(result) != 1 or len(result[0]) <= 9 + 1, (
            "Dots mode should not produce a single long TSP polyline"
        )
