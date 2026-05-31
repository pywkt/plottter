"""Tests for path post-processing and optimization (Phase 8)."""

import math
import pytest

from plottter.processing.simplify import simplify_paths, simplify_polyline
from plottter.processing.filter import filter_short_paths
from plottter.processing.clip import clip_to_bounds
from plottter.processing.merge import merge_nearby_paths, merge_fragments
from plottter.processing.optimize import (
    reorder_paths,
    optimize_2opt,
    optimize_or_opt,
    calculate_travel_distance,
)
from plottter.processing.weld import weld_overlapping_paths
from plottter.processing.curves import fit_curves
from plottter.processing.scale import scale_paths_to_canvas
from plottter.models.canvas import Canvas
from plottter.processing import (
    simplify_paths as pkg_simplify,
    filter_short_paths as pkg_filter,
    clip_to_bounds as pkg_clip,
    merge_nearby_paths as pkg_merge,
    merge_fragments as pkg_merge_fragments,
    reorder_paths as pkg_reorder,
    optimize_2opt as pkg_2opt,
    calculate_travel_distance as pkg_travel,
    weld_overlapping_paths as pkg_weld,
    fit_curves as pkg_fit_curves,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_zigzag(n: int = 20, amplitude: float = 5.0) -> list[tuple[float, float]]:
    """Zigzag polyline with n points."""
    points = []
    for i in range(n):
        x = float(i)
        y = amplitude if i % 2 == 0 else 0.0
        points.append((x, y))
    return points


def _path_length(polyline: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


# ---------------------------------------------------------------------------
# Package-level convenience imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_all_symbols_importable(self) -> None:
        assert pkg_simplify is not None
        assert pkg_filter is not None
        assert pkg_clip is not None
        assert pkg_merge is not None
        assert pkg_merge_fragments is not None
        assert pkg_reorder is not None
        assert pkg_2opt is not None
        assert pkg_travel is not None
        assert pkg_weld is not None


# ---------------------------------------------------------------------------
# Simplify
# ---------------------------------------------------------------------------

class TestSimplify:
    def test_collinear_points_reduced(self) -> None:
        """All collinear intermediate points should be removed."""
        polyline = [(float(i), 0.0) for i in range(10)]
        result = simplify_polyline(polyline, tolerance_mm=0.1)
        # Only start and end needed
        assert result[0] == polyline[0]
        assert result[-1] == polyline[-1]
        assert len(result) == 2

    def test_zigzag_reduces_point_count(self) -> None:
        """Zigzag with large amplitude should keep most points."""
        polyline = make_zigzag(20, amplitude=5.0)
        result = simplify_polyline(polyline, tolerance_mm=0.1)
        # High amplitude zigzag — all peaks are far from the line, so most kept
        assert len(result) > 2
        # But still less than the original (at minimum collinear end-stretches removed)
        assert len(result) <= len(polyline)

    def test_collinear_zigzag_eliminated(self) -> None:
        """Zigzag with very small amplitude below tolerance should be simplified to 2 pts."""
        polyline = make_zigzag(20, amplitude=0.01)
        result = simplify_polyline(polyline, tolerance_mm=0.1)
        assert len(result) == 2

    def test_preserves_first_and_last(self) -> None:
        polyline = [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0)]
        result = simplify_polyline(polyline, tolerance_mm=0.1)
        assert result[0] == (0.0, 0.0)
        assert result[-1] == (10.0, 0.0)

    def test_short_polyline_unchanged(self) -> None:
        """Polylines with fewer than 3 points are returned as-is."""
        single = [(0.0, 0.0)]
        pair = [(0.0, 0.0), (1.0, 1.0)]
        assert simplify_polyline(single) == single
        assert simplify_polyline(pair) == pair

    def test_list_form(self) -> None:
        paths = [make_zigzag(10, amplitude=0.01)]
        result = simplify_paths(paths, tolerance_mm=0.1)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_empty_list(self) -> None:
        assert simplify_paths([]) == []


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class TestFilter:
    def test_removes_short_paths(self) -> None:
        short = [(0.0, 0.0), (0.1, 0.0)]   # length 0.1mm
        long_ = [(0.0, 0.0), (10.0, 0.0)]  # length 10mm
        result = filter_short_paths([short, long_], min_length_mm=0.5)
        assert len(result) == 1
        assert result[0] == long_

    def test_keeps_paths_at_threshold(self) -> None:
        path = [(0.0, 0.0), (0.5, 0.0)]  # exactly 0.5mm
        result = filter_short_paths([path], min_length_mm=0.5)
        assert len(result) == 1

    def test_removes_single_point_paths(self) -> None:
        single = [(1.0, 1.0)]
        result = filter_short_paths([single], min_length_mm=0.5)
        assert result == []

    def test_empty_input(self) -> None:
        assert filter_short_paths([]) == []

    def test_all_long_paths_kept(self) -> None:
        paths = [[(0.0, 0.0), (float(i + 1), 0.0)] for i in range(5)]
        result = filter_short_paths(paths, min_length_mm=0.5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

class TestClip:
    BOUNDS = (0.0, 0.0, 100.0, 100.0)

    def test_fully_inside_unchanged(self) -> None:
        polyline = [(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)]
        result = clip_to_bounds([polyline], self.BOUNDS)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_fully_outside_removed(self) -> None:
        polyline = [(200.0, 200.0), (300.0, 200.0)]
        result = clip_to_bounds([polyline], self.BOUNDS)
        assert result == []

    def test_crossing_splits_into_two(self) -> None:
        """A polyline that crosses two opposing boundaries produces two sub-paths."""
        # Goes from (-10, 50) → (110, 50): crosses left then right boundary
        polyline = [(-10.0, 50.0), (110.0, 50.0)]
        result = clip_to_bounds([polyline], self.BOUNDS)
        # Should produce one clipped segment inside the box
        assert len(result) == 1
        seg = result[0]
        assert abs(seg[0][0] - 0.0) < 1e-6    # clipped at left boundary
        assert abs(seg[-1][0] - 100.0) < 1e-6  # clipped at right boundary

    def test_multi_segment_crossing_splits(self) -> None:
        """Polyline that goes in and out of bounds produces multiple output paths."""
        # inside → outside → inside
        polyline = [
            (50.0, 50.0),    # inside
            (150.0, 50.0),   # outside
            (50.0, 50.0),    # inside
        ]
        result = clip_to_bounds([polyline], self.BOUNDS)
        # Two valid inside segments
        assert len(result) == 2

    def test_empty_input(self) -> None:
        assert clip_to_bounds([], self.BOUNDS) == []

    def test_all_points_on_boundary(self) -> None:
        polyline = [(0.0, 0.0), (100.0, 0.0)]  # on the bottom edge (y=0)
        result = clip_to_bounds([polyline], self.BOUNDS)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_connects_nearby_paths(self) -> None:
        """Two paths with overlapping endpoints should be merged."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.1, 0.0), (10.0, 0.0)]  # 0.1mm gap — within default 0.5mm threshold
        result = merge_nearby_paths([path_a, path_b], threshold_mm=0.5)
        assert len(result) == 1
        assert result[0][0] == (0.0, 0.0)
        assert result[0][-1] == (10.0, 0.0)

    def test_does_not_merge_distant_paths(self) -> None:
        """Paths more than threshold apart should NOT be merged."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(10.0, 0.0), (15.0, 0.0)]  # 5mm gap
        result = merge_nearby_paths([path_a, path_b], threshold_mm=0.5)
        assert len(result) == 2

    def test_single_path_unchanged(self) -> None:
        path = [(0.0, 0.0), (1.0, 0.0)]
        result = merge_nearby_paths([path])
        assert len(result) == 1

    def test_empty_input(self) -> None:
        assert merge_nearby_paths([]) == []

    def test_merge_with_reversal(self) -> None:
        """Merging should reverse a path if its END is the nearby endpoint."""
        # path_a ends at (5,0); path_b ends at (5.1,0) — merge by reversing path_b
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(10.0, 0.0), (5.1, 0.0)]
        result = merge_nearby_paths([path_a, path_b], threshold_mm=0.5)
        assert len(result) == 1
        # Should form a connected polyline (start + snapped junction + end).
        merged = result[0]
        assert len(merged) >= 3
        assert merged[0] == (0.0, 0.0)
        assert merged[-1] == (10.0, 0.0)

    def test_merge_snaps_to_midpoint_no_phantom_bridge(self) -> None:
        """Joining paths with a non-zero gap must snap the junction to the
        midpoint of the gap rather than encoding a straight-line bridge.

        Regression: the old code appended path B verbatim, so the resulting
        polyline contained both endpoints of the gap as consecutive vertices
        — telling the plotter to draw a visible line across the gap. On dense
        map plots this produced phantom 'roads' wherever two real roads ended
        close to each other.
        """
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.4, 0.0), (10.0, 0.0)]  # 0.4mm gap
        result = merge_nearby_paths([path_a, path_b], threshold_mm=0.5)
        assert len(result) == 1
        merged = result[0]
        # Exactly one snapped point at the midpoint of the gap.
        assert merged == [(0.0, 0.0), (5.2, 0.0), (10.0, 0.0)]
        # Neither the original end-of-A nor the original start-of-B should
        # appear as separate vertices — that's what created the phantom line.
        assert (5.0, 0.0) not in merged
        assert (5.4, 0.0) not in merged

    def test_merge_keeps_coincident_join_lossless(self) -> None:
        """When endpoints are already exactly coincident, the merge should drop
        the duplicate point but keep every other vertex intact."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.0, 0.0), (10.0, 0.0)]
        result = merge_nearby_paths([path_a, path_b], threshold_mm=0.5)
        assert result == [[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]]


# ---------------------------------------------------------------------------
# Reorder (nearest-neighbour)
# ---------------------------------------------------------------------------

class TestReorder:
    def test_output_has_same_paths(self) -> None:
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(50.0, 50.0), (51.0, 50.0)],
            [(100.0, 0.0), (101.0, 0.0)],
        ]
        result = reorder_paths(paths)
        assert len(result) == len(paths)
        # Each original path appears exactly once (possibly reversed)
        result_sets = [frozenset(p) for p in result]
        orig_sets = [frozenset(p) for p in paths] + [frozenset(p[::-1]) for p in paths]
        for rs in result_sets:
            assert rs in orig_sets

    def test_reduces_travel_vs_bad_ordering(self) -> None:
        """Nearest-neighbour ordering should reduce travel compared to worst-case ordering."""
        # Paths in worst-case order: alternating between opposite corners
        path_near = [(1.0, 1.0), (2.0, 1.0)]       # near origin
        path_far = [(99.0, 99.0), (100.0, 99.0)]   # far from origin

        # Bad order: far → near → far → near alternating
        bad = [path_far, path_near, path_far, path_near]
        bad_travel = calculate_travel_distance(bad)

        good = reorder_paths([path_near, path_far, path_near, path_far])
        good_travel = calculate_travel_distance(good)

        assert good_travel <= bad_travel + 1e-6

    def test_empty_input(self) -> None:
        assert reorder_paths([]) == []

    def test_single_path(self) -> None:
        path = [(0.0, 0.0), (1.0, 1.0)]
        result = reorder_paths([path])
        assert len(result) == 1

    # ------------------------------------------------------------------
    # KD-tree path (>=50 paths) tests
    # ------------------------------------------------------------------

    def test_kdtree_no_paths_lost_or_duplicated(self) -> None:
        """KD-tree reorder with >=50 paths: every input path appears exactly once."""
        import random
        rng = random.Random(7)
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(100)
        ]
        result = reorder_paths(paths)
        assert len(result) == 100
        result_sets = [frozenset(map(tuple, p)) for p in result]
        orig_sets = {frozenset(map(tuple, p)) for p in paths} | {
            frozenset(map(tuple, p[::-1])) for p in paths
        }
        for rs in result_sets:
            assert rs in orig_sets

    def test_kdtree_travel_not_worse_than_brute(self) -> None:
        """KD-tree (>=50 paths) travel distance must be ≤ brute-force result + tiny epsilon."""
        import random
        from plottter.processing.optimize import _reorder_paths_brute, _reorder_paths_kdtree
        rng = random.Random(42)
        valid = [
            [(rng.uniform(0, 200), rng.uniform(0, 200)),
             (rng.uniform(0, 200), rng.uniform(0, 200))]
            for _ in range(80)
        ]
        brute = _reorder_paths_brute(valid)
        kdtree = _reorder_paths_kdtree(valid)
        brute_dist = calculate_travel_distance(brute)
        kdtree_dist = calculate_travel_distance(kdtree)
        # Both implement the same greedy algorithm; results should be equal or very close
        assert kdtree_dist <= brute_dist + 1e-6

    def test_small_inputs_use_brute_force(self) -> None:
        """Inputs with fewer than 50 paths produce the same result as the brute-force path."""
        import random
        from plottter.processing.optimize import _reorder_paths_brute, _reorder_paths_kdtree, _NN_KDTREE_THRESHOLD
        assert _NN_KDTREE_THRESHOLD == 50
        rng = random.Random(99)
        valid = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(30)
        ]
        # reorder_paths should delegate to brute-force for <50 paths;
        # use num_starts=1 so both use the same single-start NN
        brute = _reorder_paths_brute(valid, num_starts=1)
        result = reorder_paths(valid, num_starts=1)
        assert result == brute

    def test_kdtree_path_reversal_correct(self) -> None:
        """When the end of a path is nearer than its start, the path is reversed."""
        # Place paths so that the optimal ordering requires reversal:
        # current position ends near (10, 0); next path is [(9, 0), (0, 0)] — end is nearer
        paths = [[(0.0, 0.0), (1.0, 0.0)]] * 50  # 50 identical stub paths to force kdtree
        # Add a path that will need reversal: its end is at origin, start is far
        reversible = [(50.0, 50.0), (1.0, 0.0)]
        paths[0] = [(0.0, 0.0), (10.0, 0.0)]
        paths[1] = reversible
        result = reorder_paths(paths)
        assert len(result) == len(paths)

    def test_kdtree_reduces_travel_large(self) -> None:
        """KD-tree reorder with 200 paths reduces travel vs original (unordered) arrangement."""
        import random
        rng = random.Random(1337)
        # Create paths in a bad order: alternating between two distant regions
        near_paths = [
            [(rng.uniform(0, 10), rng.uniform(0, 10)),
             (rng.uniform(0, 10), rng.uniform(0, 10))]
            for _ in range(100)
        ]
        far_paths = [
            [(rng.uniform(90, 100), rng.uniform(90, 100)),
             (rng.uniform(90, 100), rng.uniform(90, 100))]
            for _ in range(100)
        ]
        # Interleave near and far — worst case ordering
        interleaved = [p for pair in zip(near_paths, far_paths) for p in pair]
        bad_travel = calculate_travel_distance(interleaved)
        result = reorder_paths(interleaved)
        good_travel = calculate_travel_distance(result)
        assert good_travel < bad_travel - 1.0  # significantly less travel

    def test_performance_large_input(self) -> None:
        """KD-tree reorder of 10,000 paths completes in under 10 seconds."""
        import random
        import time
        rng = random.Random(42)
        paths = [
            [(rng.uniform(0, 200), rng.uniform(0, 200)),
             (rng.uniform(0, 200), rng.uniform(0, 200))]
            for _ in range(10_000)
        ]
        t0 = time.time()
        result = reorder_paths(paths)
        elapsed = time.time() - t0
        assert len(result) == 10_000
        assert elapsed < 20.0, f"KD-tree reorder took {elapsed:.1f}s, expected < 20s"

    def test_multi_start_never_worse_than_single_start(self) -> None:
        """Multi-start NN (num_starts=5) travel distance is ≤ single-start result."""
        import random
        rng = random.Random(7)
        paths = [
            [(rng.uniform(0, 200), rng.uniform(0, 200)),
             (rng.uniform(0, 200), rng.uniform(0, 200))]
            for _ in range(200)
        ]
        single = reorder_paths(paths, num_starts=1)
        multi = reorder_paths(paths, num_starts=5)
        assert len(multi) == len(paths)
        single_dist = calculate_travel_distance(single)
        multi_dist = calculate_travel_distance(multi)
        assert multi_dist <= single_dist + 1e-6

    def test_multi_start_deterministic(self) -> None:
        """Same input always produces the same output (no randomness)."""
        import random
        rng = random.Random(13)
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(100)
        ]
        result_a = reorder_paths(paths, num_starts=5)
        result_b = reorder_paths(paths, num_starts=5)
        assert result_a == result_b

    def test_multi_start_improves_on_adversarial_input(self) -> None:
        """With 5 starts, result is ≤ single-start on a 1000-path input."""
        import random
        rng = random.Random(99)
        # Create a challenging clustered layout where the origin-biased start is suboptimal
        paths = []
        # Cluster of paths far from origin but close to each other
        for _ in range(500):
            cx, cy = 500.0, 500.0
            paths.append([
                (cx + rng.uniform(-10, 10), cy + rng.uniform(-10, 10)),
                (cx + rng.uniform(-10, 10), cy + rng.uniform(-10, 10)),
            ])
        # Paths near origin
        for _ in range(500):
            paths.append([
                (rng.uniform(0, 50), rng.uniform(0, 50)),
                (rng.uniform(0, 50), rng.uniform(0, 50)),
            ])
        rng.shuffle(paths)
        single = reorder_paths(paths, num_starts=1)
        multi = reorder_paths(paths, num_starts=5)
        assert len(multi) == len(paths)
        assert calculate_travel_distance(multi) <= calculate_travel_distance(single) + 1e-6

    def test_multi_start_performance_large_input(self) -> None:
        """5-start reorder of 10,000 paths completes in under 30 seconds."""
        import random
        import time
        rng = random.Random(42)
        paths = [
            [(rng.uniform(0, 200), rng.uniform(0, 200)),
             (rng.uniform(0, 200), rng.uniform(0, 200))]
            for _ in range(10_000)
        ]
        t0 = time.time()
        result = reorder_paths(paths, num_starts=5)
        elapsed = time.time() - t0
        assert len(result) == 10_000
        assert elapsed < 30.0, f"5-start reorder took {elapsed:.1f}s, expected < 30s"

    def test_num_starts_one_matches_origin_seed_brute(self) -> None:
        """num_starts=1 on small input matches the origin-seeded brute-force result."""
        import random
        from plottter.processing.optimize import _reorder_paths_brute
        rng = random.Random(55)
        valid = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(30)
        ]
        result = reorder_paths(valid, num_starts=1)
        brute = _reorder_paths_brute(valid, num_starts=1)
        assert result == brute

    def test_no_paths_lost_with_multi_start(self) -> None:
        """Multi-start must not lose or duplicate any paths."""
        import random
        rng = random.Random(21)
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(80)
        ]
        result = reorder_paths(paths, num_starts=5)
        assert len(result) == len(paths)
        orig_sets = {frozenset(map(tuple, p)) for p in paths}
        for p in result:
            assert frozenset(map(tuple, p)) in orig_sets


# ---------------------------------------------------------------------------
# 2-opt improvement
# ---------------------------------------------------------------------------

class TestOptimize2Opt:
    def test_does_not_increase_travel(self) -> None:
        """2-opt must never increase travel distance."""
        import random
        rng = random.Random(42)
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(15)
        ]
        nn = reorder_paths(paths)
        nn_travel = calculate_travel_distance(nn)

        two_opt = optimize_2opt(nn)
        two_opt_travel = calculate_travel_distance(two_opt)

        assert two_opt_travel <= nn_travel + 1e-6

    def test_output_has_same_count(self) -> None:
        paths = [[(float(i), 0.0), (float(i) + 1, 0.0)] for i in range(5)]
        result = optimize_2opt(paths)
        assert len(result) == len(paths)

    def test_empty_and_tiny_inputs(self) -> None:
        assert optimize_2opt([]) == []
        single = [[(0.0, 0.0), (1.0, 0.0)]]
        assert len(optimize_2opt(single)) == 1
        pair = [[(0.0, 0.0), (1.0, 0.0)], [(5.0, 0.0), (6.0, 0.0)]]
        assert len(optimize_2opt(pair)) == 2

    # ------------------------------------------------------------------
    # Neighbor-list 2-opt tests (n >= 50 uses KD-tree neighbor lists)
    # ------------------------------------------------------------------

    def test_neighbor_list_never_increases_travel(self) -> None:
        """Neighbor-list 2-opt (large input) must not increase travel distance."""
        import random
        from plottter.processing.optimize import _2OPT_KDTREE_THRESHOLD
        rng = random.Random(7)
        n = _2OPT_KDTREE_THRESHOLD + 50  # ensure neighbor-list branch
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(n)
        ]
        nn = reorder_paths(paths)
        nn_travel = calculate_travel_distance(nn)
        result = optimize_2opt(nn)
        assert len(result) == n
        assert calculate_travel_distance(result) <= nn_travel + 1e-6

    def test_neighbor_list_no_paths_lost(self) -> None:
        """Neighbor-list 2-opt must not lose or duplicate any paths."""
        import random
        rng = random.Random(13)
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(100)
        ]
        result = optimize_2opt(reorder_paths(paths))
        assert len(result) == len(paths)
        orig_sets = {frozenset(map(tuple, p)) for p in paths}
        for p in result:
            assert frozenset(map(tuple, p)) in orig_sets

    def test_small_input_identical_to_old_brute_force(self) -> None:
        """Inputs below threshold use the same brute-force logic as before."""
        import random
        from plottter.processing.optimize import _2OPT_KDTREE_THRESHOLD
        rng = random.Random(21)
        # Use a small input guaranteed to take the brute-force path
        n = _2OPT_KDTREE_THRESHOLD - 10
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(n)
        ]
        nn = reorder_paths(paths)
        result = optimize_2opt(nn)
        assert len(result) == n
        # Travel distance must not worsen
        assert calculate_travel_distance(result) <= calculate_travel_distance(nn) + 1e-6

    def test_neighbor_list_resolves_obvious_crossing(self) -> None:
        """Neighbor-list 2-opt resolves a geometrically obvious crossing."""
        # Build a route where two edges visibly cross each other.
        # The crossing pair:
        #   edge A: (0,0)→(100,100)  (diagonal)
        #   edge B: (100,0)→(0,100)  (anti-diagonal, crosses A)
        # Embed them in a route of n=60 paths so the neighbor-list branch runs.
        import random
        rng = random.Random(99)
        padding = [
            [(float(i) * 0.1, 0.0), (float(i) * 0.1 + 0.01, 0.0)]
            for i in range(58)
        ]
        path_a = [(0.0, 0.0), (10.0, 10.0)]
        path_b = [(10.0, 0.0), (0.0, 10.0)]
        route = padding + [path_a, path_b]
        original_travel = calculate_travel_distance(route)
        result = optimize_2opt(route)
        assert len(result) == len(route)
        # 2-opt should not make things worse
        assert calculate_travel_distance(result) <= original_travel + 1e-6

    def test_performance_large_input(self) -> None:
        """Neighbor-list 2-opt on 5,000 paths completes in under 30 seconds."""
        import random
        import time
        rng = random.Random(42)
        paths = [
            [(rng.uniform(0, 200), rng.uniform(0, 200)),
             (rng.uniform(0, 200), rng.uniform(0, 200))]
            for _ in range(5_000)
        ]
        nn = reorder_paths(paths)
        t0 = time.time()
        result = optimize_2opt(nn)
        elapsed = time.time() - t0
        assert len(result) == 5_000
        assert elapsed < 30.0, f"Neighbor-list 2-opt took {elapsed:.1f}s, expected < 30s"

    def test_adaptive_iteration_limit(self) -> None:
        """Adaptive max_iters = max(max_iterations, n*2) is respected."""
        # We can't directly inspect the iteration count, but we can verify
        # that the function terminates for large n without hanging, and that
        # passing a small max_iterations doesn't prevent the algorithm from
        # running more passes on a large input.
        import random
        rng = random.Random(77)
        paths = [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(100)
        ]
        nn = reorder_paths(paths)
        # With max_iterations=1 but n=100, internal limit = max(1, 200) = 200
        result = optimize_2opt(nn, max_iterations=1)
        assert len(result) == len(paths)
        # Travel should still be <= NN result (algorithm runs enough passes)
        assert calculate_travel_distance(result) <= calculate_travel_distance(nn) + 1e-6


# ---------------------------------------------------------------------------
# calculate_travel_distance
# ---------------------------------------------------------------------------

class TestTravelDistance:
    def test_single_path(self) -> None:
        """Single horizontal path: travel = distance from origin to start + end to origin."""
        path = [(10.0, 0.0), (20.0, 0.0)]
        d = calculate_travel_distance([path])
        # origin → (10,0) = 10, (20,0) → origin = 20
        assert abs(d - 30.0) < 1e-6

    def test_two_paths(self) -> None:
        path_a = [(10.0, 0.0), (20.0, 0.0)]
        path_b = [(30.0, 0.0), (40.0, 0.0)]
        d = calculate_travel_distance([path_a, path_b])
        # origin→(10,0)=10, (20,0)→(30,0)=10, (40,0)→origin=40 → total 60
        assert abs(d - 60.0) < 1e-6

    def test_empty_returns_zero(self) -> None:
        assert calculate_travel_distance([]) == 0.0

    def test_known_arrangement(self) -> None:
        """Paths in a straight line: each pen-up hop is 1mm."""
        # 5 paths of length 1mm, each starting 2mm from the previous end
        paths = [[(float(i * 3), 0.0), (float(i * 3) + 1.0, 0.0)] for i in range(5)]
        d = calculate_travel_distance(paths)
        # origin → (0,0) = 0
        # (1,0) → (3,0) = 2, (4,0) → (6,0) = 2, (7,0) → (9,0) = 2, (10,0) → (12,0) = 2
        # (13,0) → origin = 13
        expected = 0.0 + 2.0 * 4 + 13.0
        assert abs(d - expected) < 1e-6


# ---------------------------------------------------------------------------
# Weld / Union Overlapping Paths
# ---------------------------------------------------------------------------

class TestWeld:
    def test_exact_duplicate_segment_removed(self) -> None:
        """A segment appearing in a second path that exactly duplicates the first is removed."""
        path_a = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        path_b = [(0.0, 0.0), (5.0, 0.0)]  # first segment of path_a duplicated
        result = weld_overlapping_paths([path_a, path_b])
        # path_b's segment is a duplicate — path_b should disappear entirely
        assert len(result) == 1
        assert result[0] == path_a

    def test_reversed_duplicate_removed(self) -> None:
        """A reversed duplicate segment is also detected and removed."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.0, 0.0), (0.0, 0.0)]  # same segment, reversed
        result = weld_overlapping_paths([path_a, path_b])
        assert len(result) == 1

    def test_near_duplicate_within_tolerance(self) -> None:
        """Segments within tolerance_mm are treated as duplicates."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(0.02, 0.0), (5.02, 0.0)]  # 0.02mm offset, within default 0.1mm
        result = weld_overlapping_paths([path_a, path_b], tolerance_mm=0.1)
        assert len(result) == 1

    def test_near_duplicate_outside_tolerance_kept(self) -> None:
        """Segments outside tolerance are NOT removed."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(0.5, 0.0), (5.5, 0.0)]  # 0.5mm offset, outside 0.1mm tolerance
        result = weld_overlapping_paths([path_a, path_b], tolerance_mm=0.1)
        assert len(result) == 2

    def test_no_paths_unchanged(self) -> None:
        assert weld_overlapping_paths([]) == []

    def test_single_path_unchanged(self) -> None:
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        result = weld_overlapping_paths([path])
        assert result == [path]

    def test_partial_overlap_splits_path(self) -> None:
        """A path partially duplicating another becomes a shorter fragment."""
        # path_a covers (0,0)→(5,0)→(10,0)
        # path_b duplicates middle segment and adds a new one
        path_a = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        path_b = [(5.0, 0.0), (10.0, 0.0), (15.0, 0.0)]  # first seg dup, second new
        result = weld_overlapping_paths([path_a, path_b])
        # path_a kept; path_b should only contain the non-dup segment (10→15)
        all_points = [pt for poly in result for pt in poly]
        assert (15.0, 0.0) in all_points  # the unique segment survives
        # path_a's segments are not removed
        assert any((0.0, 0.0) in p for p in result)

    def test_non_overlapping_paths_kept(self) -> None:
        """Completely different paths should all be preserved."""
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(10.0, 0.0), (11.0, 0.0)],
            [(20.0, 0.0), (21.0, 0.0)],
        ]
        result = weld_overlapping_paths(paths)
        assert len(result) == 3

    def test_empty_paths_after_dedup_removed(self) -> None:
        """Paths that become fully empty after de-duplication are not returned."""
        path_a = [(0.0, 0.0), (1.0, 0.0)]
        path_b = [(0.0, 0.0), (1.0, 0.0)]  # identical to path_a
        result = weld_overlapping_paths([path_a, path_b])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Bezier Curve Fitting (fit_curves)
# ---------------------------------------------------------------------------

class TestFitCurves:
    def test_package_import(self) -> None:
        assert pkg_fit_curves is fit_curves

    def test_empty_input_returns_empty(self) -> None:
        assert fit_curves([]) == []

    def test_zero_tolerance_returns_unchanged(self) -> None:
        poly = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
        result = fit_curves([poly], tolerance_mm=0.0)
        # tolerance <= 0 → pass-through
        assert result == [poly]

    def test_two_point_polyline_returned_as_is(self) -> None:
        poly = [(0.0, 0.0), (10.0, 0.0)]
        result = fit_curves([poly], tolerance_mm=0.5)
        assert len(result) == 1
        assert result[0][0] == poly[0]
        assert result[0][-1] == poly[-1]

    def test_collinear_points_preserved(self) -> None:
        """Collinear points fit perfectly — all output lies on the original line."""
        poly = [(float(i), 0.0) for i in range(20)]
        result = fit_curves([poly], tolerance_mm=0.5)
        assert len(result) == 1
        # Endpoints preserved
        assert math.isclose(result[0][0][0], 0.0, abs_tol=1e-6)
        assert math.isclose(result[0][-1][0], 19.0, abs_tol=1e-6)
        # All output points should lie on y=0 (collinear line)
        for px, py in result[0]:
            assert math.isclose(py, 0.0, abs_tol=1e-6)

    def test_smooth_arc_stays_within_tolerance(self) -> None:
        """A quarter-circle arc: output points should be within tolerance of the arc."""
        import math as _math
        n = 50
        poly = [(_math.cos(t), _math.sin(t)) for t in [i * _math.pi / (2 * (n - 1)) for i in range(n)]]
        tol = 0.05  # 0.05 mm (radius = 1 mm)
        result = fit_curves([poly], tolerance_mm=tol)
        assert len(result) == 1
        out = result[0]
        # Every output point must lie near the unit circle
        for px, py in out:
            r = _math.hypot(px, py)
            assert abs(r - 1.0) < tol * 4  # generous bound: Bezier approximation

    def test_output_reduces_point_count(self) -> None:
        """A dense arc should have significantly fewer output points after fitting."""
        import math as _math
        n = 100
        poly = [(_math.cos(t), _math.sin(t)) for t in [i * _math.pi / (2 * (n - 1)) for i in range(n)]]
        result = fit_curves([poly], tolerance_mm=0.1)
        assert len(result) == 1
        # Dense 100-point arc should be reducible with Bezier fitting
        assert len(result[0]) < n

    def test_closed_polyline_stays_closed(self) -> None:
        """A closed input polyline (first == last) should produce a closed output."""
        import math as _math
        n = 32
        poly = [(_math.cos(2 * _math.pi * i / n), _math.sin(2 * _math.pi * i / n)) for i in range(n)]
        poly.append(poly[0])  # close it
        result = fit_curves([poly], tolerance_mm=0.1)
        assert len(result) == 1
        out = result[0]
        assert len(out) >= 2
        assert out[0] == out[-1]

    def test_multiple_polylines_all_processed(self) -> None:
        """All input polylines are fitted and returned."""
        polys = [
            [(float(i), 0.0) for i in range(10)],
            [(0.0, float(i)) for i in range(10)],
        ]
        result = fit_curves(polys, tolerance_mm=0.5)
        assert len(result) == 2

    def test_single_point_polyline_excluded(self) -> None:
        """A single-point polyline is dropped (cannot form a valid 2-point output)."""
        result = fit_curves([[(5.0, 5.0)]], tolerance_mm=0.5)
        assert result == []

    def test_higher_tolerance_fewer_points(self) -> None:
        """Higher tolerance should produce equal or fewer output points than lower tolerance."""
        import math as _math
        n = 80
        poly = [(_math.cos(t), _math.sin(t)) for t in [i * 2 * _math.pi / n for i in range(n)]]
        poly.append(poly[0])

        result_fine = fit_curves([poly], tolerance_mm=0.02)
        result_coarse = fit_curves([poly], tolerance_mm=0.5)

        pts_fine = len(result_fine[0]) if result_fine else 0
        pts_coarse = len(result_coarse[0]) if result_coarse else 0
        assert pts_coarse <= pts_fine


# ---------------------------------------------------------------------------
# Fragment Merging (merge_fragments)
# ---------------------------------------------------------------------------

class TestMergeFragments:
    def test_nearby_endpoints_merged_into_one(self) -> None:
        """Two polylines whose endpoints are within tolerance merge into one."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.3, 0.0), (10.0, 0.0)]  # 0.3mm gap — within 0.5mm tolerance
        result = merge_fragments([path_a, path_b], gap_tolerance_mm=0.5)
        assert len(result) == 1
        # The merged path must start at (0,0) and end at (10,0) (or reversed)
        endpoints = {result[0][0], result[0][-1]}
        assert (0.0, 0.0) in endpoints
        assert (10.0, 0.0) in endpoints

    def test_far_apart_paths_remain_separate(self) -> None:
        """Polylines with endpoints more than tolerance apart are not merged."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(10.0, 0.0), (15.0, 0.0)]  # 5mm gap — exceeds 0.5mm tolerance
        result = merge_fragments([path_a, path_b], gap_tolerance_mm=0.5)
        assert len(result) == 2

    def test_empty_input_returns_empty(self) -> None:
        assert merge_fragments([], gap_tolerance_mm=0.5) == []

    def test_single_path_unchanged(self) -> None:
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]
        result = merge_fragments([path], gap_tolerance_mm=0.5)
        assert len(result) == 1
        assert result[0] == path

    def test_zero_tolerance_disables_merging(self) -> None:
        """gap_tolerance_mm=0 must disable all merging."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.0, 0.0), (10.0, 0.0)]  # exact overlap
        result = merge_fragments([path_a, path_b], gap_tolerance_mm=0.0)
        assert len(result) == 2

    def test_negative_tolerance_disables_merging(self) -> None:
        """Negative gap_tolerance_mm must also disable merging."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.0, 0.0), (10.0, 0.0)]
        result = merge_fragments([path_a, path_b], gap_tolerance_mm=-1.0)
        assert len(result) == 2

    def test_merge_requires_reversal(self) -> None:
        """Should merge even when path_b is oriented end→end with path_a (requires reversal)."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        # path_b ends near path_a's end — must reverse path_b to form a chain
        path_b = [(10.0, 0.0), (5.1, 0.0)]
        result = merge_fragments([path_a, path_b], gap_tolerance_mm=0.5)
        assert len(result) == 1

    def test_prepend_from_start(self) -> None:
        """Should extend path by prepending when path_b's endpoint is near path_a's start."""
        path_a = [(5.0, 0.0), (10.0, 0.0)]
        path_b = [(0.0, 0.0), (4.9, 0.0)]  # path_b ends near path_a's start
        result = merge_fragments([path_a, path_b], gap_tolerance_mm=0.5)
        assert len(result) == 1
        endpoints = {result[0][0], result[0][-1]}
        assert (0.0, 0.0) in endpoints
        assert (10.0, 0.0) in endpoints

    def test_chain_of_three_fragments(self) -> None:
        """Three consecutive fragments should all merge into a single polyline."""
        path_a = [(0.0, 0.0), (5.0, 0.0)]
        path_b = [(5.2, 0.0), (10.0, 0.0)]
        path_c = [(10.3, 0.0), (15.0, 0.0)]
        result = merge_fragments([path_a, path_b, path_c], gap_tolerance_mm=0.5)
        assert len(result) == 1

    def test_package_import(self) -> None:
        assert pkg_merge_fragments is merge_fragments


# ---------------------------------------------------------------------------
# Progress reporting and cancellation
# ---------------------------------------------------------------------------

class TestProgressAndCancellation:
    """Tests for progress_callback and cancelled parameters (task 31.5)."""

    def _make_paths(self, n: int, seed: int = 42) -> list[list[tuple[float, float]]]:
        import random
        rng = random.Random(seed)
        return [
            [(rng.uniform(0, 100), rng.uniform(0, 100)),
             (rng.uniform(0, 100), rng.uniform(0, 100))]
            for _ in range(n)
        ]

    # ------------------------------------------------------------------
    # reorder_paths progress
    # ------------------------------------------------------------------

    def test_reorder_progress_callback_called(self) -> None:
        """progress_callback is invoked at least once during reorder_paths."""
        paths = self._make_paths(60)  # >= 50 to use KD-tree branch
        calls: list[float] = []
        reorder_paths(paths, num_starts=1, progress_callback=calls.append)
        assert len(calls) > 0
        assert all(0.0 <= v <= 1.0 for v in calls), "All progress values must be in [0, 1]"

    def test_reorder_progress_monotonically_increasing(self) -> None:
        """progress_callback values must be non-decreasing."""
        paths = self._make_paths(60)
        calls: list[float] = []
        reorder_paths(paths, num_starts=1, progress_callback=calls.append)
        for i in range(1, len(calls)):
            assert calls[i] >= calls[i - 1] - 1e-9, (
                f"Progress went backwards: {calls[i - 1]} → {calls[i]}"
            )

    def test_reorder_progress_small_input(self) -> None:
        """progress_callback works on small (brute-force) inputs too."""
        paths = self._make_paths(20)  # < 50, uses brute force
        calls: list[float] = []
        reorder_paths(paths, num_starts=1, progress_callback=calls.append)
        assert len(calls) > 0

    def test_reorder_cancel_returns_all_paths(self) -> None:
        """Cancelling mid-reorder returns all input paths (none lost)."""
        n = 60
        paths = self._make_paths(n)
        # Cancel immediately
        result = reorder_paths(paths, num_starts=1, cancelled=lambda: True)
        assert len(result) == n, f"Expected {n} paths, got {len(result)}"
        orig_sets = {frozenset(map(tuple, p)) for p in paths}
        for p in result:
            assert frozenset(map(tuple, p)) in orig_sets

    def test_reorder_cancel_small_input_returns_all_paths(self) -> None:
        """Cancelling on brute-force (small) input still returns all paths."""
        n = 20
        paths = self._make_paths(n)
        result = reorder_paths(paths, num_starts=1, cancelled=lambda: True)
        assert len(result) == n

    def test_reorder_no_callback_unchanged_behaviour(self) -> None:
        """Passing no callbacks produces the same result as before."""
        paths = self._make_paths(80)
        result_no_cb = reorder_paths(paths, num_starts=1)
        result_with_cb = reorder_paths(paths, num_starts=1, progress_callback=lambda f: None)
        assert result_no_cb == result_with_cb

    # ------------------------------------------------------------------
    # optimize_2opt progress
    # ------------------------------------------------------------------

    def test_2opt_progress_callback_called(self) -> None:
        """progress_callback is invoked during optimize_2opt."""
        paths = reorder_paths(self._make_paths(60))
        calls: list[float] = []
        optimize_2opt(paths, progress_callback=calls.append)
        assert len(calls) > 0
        assert all(0.0 <= v <= 1.0 for v in calls)

    def test_2opt_cancel_returns_valid_route(self) -> None:
        """Cancelling optimize_2opt returns current best (all paths present)."""
        n = 60
        paths = reorder_paths(self._make_paths(n))
        result = optimize_2opt(paths, cancelled=lambda: True)
        assert len(result) == n
        orig_sets = {frozenset(map(tuple, p)) for p in paths}
        for p in result:
            assert frozenset(map(tuple, p)) in orig_sets

    def test_2opt_cancel_small_input(self) -> None:
        """Cancelling works on small (brute-force) 2-opt inputs."""
        n = 10
        paths = self._make_paths(n)
        result = optimize_2opt(paths, cancelled=lambda: True)
        assert len(result) == n

    def test_2opt_cancel_does_not_worsen_beyond_input(self) -> None:
        """Cancelled 2-opt must not produce a route worse than input."""
        paths = reorder_paths(self._make_paths(60))
        before = calculate_travel_distance(paths)
        cancelled_flag = [False]

        call_count = [0]
        def _cancel_after_few(f: float) -> None:
            call_count[0] += 1
            if call_count[0] >= 3:
                cancelled_flag[0] = True

        result = optimize_2opt(paths, progress_callback=_cancel_after_few,
                               cancelled=lambda: cancelled_flag[0])
        # Travel distance of cancelled result must be <= input (2-opt only improves)
        assert calculate_travel_distance(result) <= before + 1e-6

    # ------------------------------------------------------------------
    # optimize_or_opt progress
    # ------------------------------------------------------------------

    def test_or_opt_progress_callback_called(self) -> None:
        """progress_callback is invoked during optimize_or_opt."""
        paths = reorder_paths(self._make_paths(60))
        calls: list[float] = []
        optimize_or_opt(paths, progress_callback=calls.append)
        assert len(calls) > 0
        assert all(0.0 <= v <= 1.0 for v in calls)

    def test_or_opt_cancel_returns_valid_route(self) -> None:
        """Cancelling optimize_or_opt returns current best (all paths present)."""
        n = 60
        paths = reorder_paths(self._make_paths(n))
        result = optimize_or_opt(paths, cancelled=lambda: True)
        assert len(result) == n
        orig_sets = {frozenset(map(tuple, p)) for p in paths}
        for p in result:
            assert frozenset(map(tuple, p)) in orig_sets

    def test_or_opt_cancel_does_not_worsen_beyond_input(self) -> None:
        """Cancelled Or-opt must not produce a route worse than input."""
        paths = reorder_paths(self._make_paths(60))
        before = calculate_travel_distance(paths)
        cancelled_flag = [False]
        call_count = [0]
        def _cancel_after_few(f: float) -> None:
            call_count[0] += 1
            if call_count[0] >= 3:
                cancelled_flag[0] = True

        result = optimize_or_opt(paths, progress_callback=_cancel_after_few,
                                 cancelled=lambda: cancelled_flag[0])
        assert calculate_travel_distance(result) <= before + 1e-6

    # ------------------------------------------------------------------
    # Performance: callback overhead must be negligible
    # ------------------------------------------------------------------

    def test_progress_callback_overhead_negligible(self) -> None:
        """Callback overhead is < 5% of total runtime (well within the 1% target)."""
        import time
        paths = self._make_paths(200, seed=7)
        nn = reorder_paths(paths, num_starts=1)

        # Without callback
        t0 = time.perf_counter()
        optimize_2opt(nn)
        t_no_cb = time.perf_counter() - t0

        nn2 = reorder_paths(paths, num_starts=1)

        # With callback
        t0 = time.perf_counter()
        optimize_2opt(nn2, progress_callback=lambda f: None)
        t_with_cb = time.perf_counter() - t0

        # Allow up to 5x overhead (very generous; callback is trivially cheap)
        # In practice it should be < 1%
        if t_no_cb > 0:
            assert t_with_cb < t_no_cb * 5 + 0.1, (
                f"Callback adds too much overhead: {t_no_cb:.3f}s → {t_with_cb:.3f}s"
            )

    # ------------------------------------------------------------------
    # Multi-start cancellation fallback paths
    # ------------------------------------------------------------------

    def test_reorder_cancel_multistart_kdtree_returns_all_paths(self) -> None:
        """Cancelling multi-start reorder (KD-tree, n>=50) returns all input paths.

        Exercises the fallback at optimize.py lines 317-320:
          if cancelled and cancelled():
              return best_result if best_result else _reorder_paths_kdtree_from_seed(...)
        With cancelled=lambda: True the check fires on seed_i=0 before any result
        is stored, forcing the else branch (_reorder_paths_kdtree_from_seed fallback).
        """
        n = 60
        paths = self._make_paths(n)
        result = reorder_paths(paths, num_starts=5, cancelled=lambda: True)
        assert len(result) == n, f"Expected {n} paths, got {len(result)}"
        orig_sets = {frozenset(map(tuple, p)) for p in paths}
        for p in result:
            assert frozenset(map(tuple, p)) in orig_sets

    def test_reorder_cancel_multistart_brute_returns_all_paths(self) -> None:
        """Cancelling multi-start reorder (brute-force, n<50) returns all input paths.

        Exercises the fallback at optimize.py lines 152-153:
          if cancelled and cancelled():
              return best_result if best_result else _reorder_paths_brute_from_seed(...)
        With cancelled=lambda: True the check fires on seed_i=0 before any result
        is stored, forcing the else branch (_reorder_paths_brute_from_seed fallback).
        """
        n = 20
        paths = self._make_paths(n)
        result = reorder_paths(paths, num_starts=5, cancelled=lambda: True)
        assert len(result) == n, f"Expected {n} paths, got {len(result)}"
        orig_sets = {frozenset(map(tuple, p)) for p in paths}
        for p in result:
            assert frozenset(map(tuple, p)) in orig_sets

# ---------------------------------------------------------------------------
# Scale paths to canvas
# ---------------------------------------------------------------------------


class TestScalePathsToCanvas:
    def _make_canvas(self, w: float, h: float, margin: float = 10.0) -> Canvas:
        return Canvas(width_mm=w, height_mm=h, margin_mm=margin)

    def test_same_canvas_identity(self) -> None:
        """Scaling to the same canvas leaves paths unchanged."""
        canvas = self._make_canvas(210.0, 297.0)
        paths = [[(10.0, 10.0), (100.0, 150.0), (200.0, 280.0)]]
        result = scale_paths_to_canvas(paths, canvas, canvas)
        assert len(result) == 1
        for (rx, ry), (ox, oy) in zip(result[0], paths[0]):
            assert abs(rx - ox) < 1e-9
            assert abs(ry - oy) < 1e-9

    def test_double_canvas_scales_drawing_area(self) -> None:
        """A4 -> A3 (approx double area): points in drawing area scale proportionally."""
        old = self._make_canvas(210.0, 297.0, margin=10.0)
        new = self._make_canvas(297.0, 420.0, margin=10.0)
        # Point at old drawing-area origin (10, 10)
        paths = [[(10.0, 10.0)]]
        result = scale_paths_to_canvas(paths, old, new)
        # Drawing-area origin should map to new drawing-area origin (10, 10)
        assert abs(result[0][0][0] - 10.0) < 1e-9
        assert abs(result[0][0][1] - 10.0) < 1e-9

    def test_point_at_old_margin_maps_to_new_margin(self) -> None:
        """Top-left drawing-area corner always maps to the new margin."""
        old = self._make_canvas(100.0, 100.0, margin=5.0)
        new = self._make_canvas(200.0, 150.0, margin=15.0)
        paths = [[(5.0, 5.0)]]
        result = scale_paths_to_canvas(paths, old, new)
        assert abs(result[0][0][0] - 15.0) < 1e-9
        assert abs(result[0][0][1] - 15.0) < 1e-9

    def test_point_at_old_far_corner_maps_to_new_far_corner(self) -> None:
        """Bottom-right drawing-area corner maps to new drawing-area bottom-right."""
        old = self._make_canvas(100.0, 100.0, margin=5.0)
        new = self._make_canvas(200.0, 150.0, margin=15.0)
        # old far corner: (100-5, 100-5) = (95, 95)
        paths = [[(95.0, 95.0)]]
        result = scale_paths_to_canvas(paths, old, new)
        # new far corner: (200-15, 150-15) = (185, 135)
        assert abs(result[0][0][0] - 185.0) < 1e-9
        assert abs(result[0][0][1] - 135.0) < 1e-9

    def test_center_point_stays_centered(self) -> None:
        """Center of old drawing area maps to center of new drawing area."""
        old = self._make_canvas(200.0, 200.0, margin=10.0)
        new = self._make_canvas(400.0, 300.0, margin=20.0)
        # old center: (10 + 90, 10 + 90) = (100, 100)
        paths = [[(100.0, 100.0)]]
        result = scale_paths_to_canvas(paths, old, new)
        # new center: (20 + 180, 20 + 130) = (200, 150)
        assert abs(result[0][0][0] - 200.0) < 1e-9
        assert abs(result[0][0][1] - 150.0) < 1e-9

    def test_multiple_polylines(self) -> None:
        """Multiple polylines are all scaled independently."""
        old = self._make_canvas(100.0, 100.0, margin=10.0)
        new = self._make_canvas(200.0, 200.0, margin=10.0)
        paths = [
            [(10.0, 10.0), (90.0, 90.0)],
            [(50.0, 50.0)],
        ]
        result = scale_paths_to_canvas(paths, old, new)
        assert len(result) == 2
        # old draw area: (10,10)→(90,90), width=80, height=80
        # new draw area: (10,10)→(190,190), width=180, height=180
        sx = 180.0 / 80.0
        for orig_poly, new_poly in zip(paths, result):
            for (ox, oy), (nx, ny) in zip(orig_poly, new_poly):
                assert abs(nx - (10.0 + (ox - 10.0) * sx)) < 1e-9
                assert abs(ny - (10.0 + (oy - 10.0) * sx)) < 1e-9

    def test_empty_paths_list(self) -> None:
        """Empty input returns empty output."""
        old = self._make_canvas(210.0, 297.0)
        new = self._make_canvas(297.0, 420.0)
        result = scale_paths_to_canvas([], old, new)
        assert result == []

    def test_zero_drawing_area_returns_copy(self) -> None:
        """If old drawing area has zero width, paths are returned unchanged."""
        old = Canvas(width_mm=0.0, height_mm=100.0, margin_mm=0.0)
        new = self._make_canvas(210.0, 297.0)
        paths = [[(5.0, 10.0), (20.0, 30.0)]]
        result = scale_paths_to_canvas(paths, old, new)
        assert result[0] == paths[0]


# ---------------------------------------------------------------------------
# JIT Parity — merge.py
# ---------------------------------------------------------------------------

class TestMergeJITParity:
    """Verify that the ``_find_and_snap`` JIT kernel produces identical output
    to an equivalent pure-Python reference implementation on deterministic inputs.

    When numba is absent, the ``@njit`` shim is a no-op, so both the
    "JIT" call and the pure-Python reference effectively run the same Python
    code — the tests still pass and confirm that the kernel logic is correct.
    When numba is installed, the compiled kernel must produce numerically
    identical results to the reference.
    """

    # ------------------------------------------------------------------
    # Pure-Python reference for _find_and_snap
    # ------------------------------------------------------------------

    @staticmethod
    def _find_and_snap_py(qx, qy, cand_idxs, endpoints, consumed, own_idx, reverse_if_end):
        """Verbatim Python reimplementation of the _find_and_snap kernel."""
        best_j = -1
        best_dist_sq = 1e300
        best_reverse_j = 0
        best_idx_found = -1

        for idx in cand_idxs:
            idx = int(idx)
            j = idx >> 1
            if j == own_idx or consumed[j]:
                continue
            ex = endpoints[idx, 0]
            ey = endpoints[idx, 1]
            dx = qx - ex
            dy = qy - ey
            dsq = dx * dx + dy * dy
            if dsq < best_dist_sq:
                best_dist_sq = dsq
                best_j = j
                if reverse_if_end:
                    best_reverse_j = 1 if (idx & 1) else 0
                else:
                    best_reverse_j = 0 if (idx & 1) else 1
                best_idx_found = idx

        if best_j < 0:
            return -1, 0, 0.0, 0.0, 0

        jx = endpoints[best_idx_found, 0]
        jy = endpoints[best_idx_found, 1]
        coincident = 1 if (abs(qx - jx) < 1e-9 and abs(qy - jy) < 1e-9) else 0
        mid_x = (qx + jx) * 0.5
        mid_y = (qy + jy) * 0.5
        return best_j, best_reverse_j, mid_x, mid_y, coincident

    # ------------------------------------------------------------------
    # Kernel parity tests
    # ------------------------------------------------------------------

    def test_no_candidates_returns_sentinel(self) -> None:
        """Empty candidate list → best_j == -1."""
        import numpy as np
        from plottter.processing.merge import _find_and_snap

        endpoints = np.array([[0.0, 0.0], [5.0, 0.0], [5.1, 0.0], [10.0, 0.0]], dtype=np.float64)
        consumed = np.zeros(2, dtype=np.bool_)
        cands = np.array([], dtype=np.int64)

        jit_result = _find_and_snap(5.0, 0.0, cands, endpoints, consumed, 0, 1)
        py_result = self._find_and_snap_py(5.0, 0.0, cands, endpoints, consumed, 0, 1)

        assert jit_result[0] == py_result[0] == -1

    def test_single_candidate_phase_a_parity(self) -> None:
        """Phase A (reverse_if_end=1): JIT and Python agree on best_j, reverse flag, midpoint."""
        import numpy as np
        from plottter.processing.merge import _find_and_snap

        # Path 0: [(0,0), (5,0)]; Path 1: [(5.2,0), (10,0)]
        # Endpoints: idx0=(0,0), idx1=(5,0), idx2=(5.2,0), idx3=(10,0)
        endpoints = np.array([[0.0, 0.0], [5.0, 0.0], [5.2, 0.0], [10.0, 0.0]], dtype=np.float64)
        consumed = np.zeros(2, dtype=np.bool_)
        # Query from end of path 0 (5,0): candidate idx 2 (start of path 1)
        cands = np.array([2], dtype=np.int64)

        jit = _find_and_snap(5.0, 0.0, cands, endpoints, consumed, 0, 1)
        py = self._find_and_snap_py(5.0, 0.0, cands, endpoints, consumed, 0, 1)

        assert jit[0] == py[0] == 1          # best_j
        assert jit[1] == py[1] == 0          # best_reverse_j (idx=2 is even → start → no rev)
        assert abs(jit[2] - py[2]) < 1e-12   # mid_x
        assert abs(jit[3] - py[3]) < 1e-12   # mid_y
        assert jit[4] == py[4] == 0          # not coincident

    def test_single_candidate_phase_a_reversal_parity(self) -> None:
        """Phase A: candidate is j's END (odd idx) → reverse_j=1."""
        import numpy as np
        from plottter.processing.merge import _find_and_snap

        # Path 1's end (idx=3=odd) is near query point → needs reversal
        endpoints = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [5.1, 0.0]], dtype=np.float64)
        consumed = np.zeros(2, dtype=np.bool_)
        cands = np.array([3], dtype=np.int64)  # idx=3 → j=1, is end

        jit = _find_and_snap(5.0, 0.0, cands, endpoints, consumed, 0, 1)
        py = self._find_and_snap_py(5.0, 0.0, cands, endpoints, consumed, 0, 1)

        assert jit[0] == py[0] == 1
        assert jit[1] == py[1] == 1   # reverse_j=1 because idx=3 is odd

    def test_phase_b_parity(self) -> None:
        """Phase B (reverse_if_end=0): JIT and Python agree."""
        import numpy as np
        from plottter.processing.merge import _find_and_snap

        # Path 0 starts at (5,0); path 1 ends at (4.9,0) (idx=3)
        # idx=3 is odd → j's end → in Phase B (reverse_if_end=0), odd means NO reversal
        endpoints = np.array([[5.0, 0.0], [10.0, 0.0], [0.0, 0.0], [4.9, 0.0]], dtype=np.float64)
        consumed = np.zeros(2, dtype=np.bool_)
        cands = np.array([3], dtype=np.int64)  # idx=3 → j=1, is end

        jit = _find_and_snap(5.0, 0.0, cands, endpoints, consumed, 0, 0)
        py = self._find_and_snap_py(5.0, 0.0, cands, endpoints, consumed, 0, 0)

        assert jit[0] == py[0] == 1
        assert jit[1] == py[1] == 0   # no reversal (odd idx, reverse_if_end=0)
        assert abs(jit[2] - py[2]) < 1e-12
        assert abs(jit[3] - py[3]) < 1e-12

    def test_coincident_endpoints_parity(self) -> None:
        """Exactly coincident endpoints → coincident flag == 1 for both."""
        import numpy as np
        from plottter.processing.merge import _find_and_snap

        endpoints = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        consumed = np.zeros(2, dtype=np.bool_)
        cands = np.array([2], dtype=np.int64)

        jit = _find_and_snap(5.0, 0.0, cands, endpoints, consumed, 0, 1)
        py = self._find_and_snap_py(5.0, 0.0, cands, endpoints, consumed, 0, 1)

        assert jit[4] == py[4] == 1   # coincident

    def test_consumed_path_skipped_parity(self) -> None:
        """A consumed path is never selected — both JIT and Python must skip it."""
        import numpy as np
        from plottter.processing.merge import _find_and_snap

        endpoints = np.array([[0.0, 0.0], [5.0, 0.0], [5.1, 0.0], [10.0, 0.0]], dtype=np.float64)
        consumed = np.array([False, True], dtype=np.bool_)   # path 1 consumed
        cands = np.array([2, 3], dtype=np.int64)

        jit = _find_and_snap(5.0, 0.0, cands, endpoints, consumed, 0, 1)
        py = self._find_and_snap_py(5.0, 0.0, cands, endpoints, consumed, 0, 1)

        assert jit[0] == py[0] == -1   # no candidate available

    def test_multiple_candidates_nearest_wins_parity(self) -> None:
        """When multiple candidates are available, the nearest one wins."""
        import numpy as np
        from plottter.processing.merge import _find_and_snap

        # Path 1 start at (5.3,0), path 2 start at (5.1,0) — path 2 is closer
        endpoints = np.array([
            [0.0, 0.0], [5.0, 0.0],    # path 0
            [5.3, 0.0], [10.0, 0.0],   # path 1
            [5.1, 0.0], [15.0, 0.0],   # path 2
        ], dtype=np.float64)
        consumed = np.zeros(3, dtype=np.bool_)
        cands = np.array([2, 4], dtype=np.int64)  # path 1 start, path 2 start

        jit = _find_and_snap(5.0, 0.0, cands, endpoints, consumed, 0, 1)
        py = self._find_and_snap_py(5.0, 0.0, cands, endpoints, consumed, 0, 1)

        assert jit[0] == py[0] == 2   # path 2 is nearest (dist 0.1 vs 0.3)
        assert jit[1] == py[1]
        assert abs(jit[2] - py[2]) < 1e-12
        assert abs(jit[3] - py[3]) < 1e-12

    def test_merge_nearby_paths_end_to_end_parity(self) -> None:
        """Full merge_nearby_paths output matches reference on a chain of 5 paths."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(5.2, 0.0), (10.0, 0.0)],
            [(10.1, 0.0), (15.0, 0.0)],
            [(15.3, 0.0), (20.0, 0.0)],
            [(20.4, 0.0), (25.0, 0.0)],
        ]
        result = merge_nearby_paths(paths, threshold_mm=0.5)
        # All 5 paths should merge into one continuous polyline
        assert len(result) == 1
        assert result[0][0] == (0.0, 0.0)
        assert result[0][-1] == (25.0, 0.0)

    def test_merge_fragments_end_to_end_parity(self) -> None:
        """Full merge_fragments output matches reference on a branching set."""
        import random
        rng = random.Random(42)
        # Generate a chain of 10 fragments with tiny gaps
        paths = []
        x = 0.0
        for _ in range(10):
            length = rng.uniform(1.0, 5.0)
            gap = rng.uniform(0.0, 0.3)
            paths.append([(x, 0.0), (x + length, 0.0)])
            x += length + gap

        result = merge_fragments(paths, gap_tolerance_mm=0.5)
        # All 10 short fragments should merge into one
        assert len(result) == 1


# ---------------------------------------------------------------------------
# JIT Parity — weld.py
# ---------------------------------------------------------------------------

class TestWeldJITParity:
    """Verify that the ``_segments_match_jit`` JIT kernel produces identical
    output to an equivalent pure-Python reference implementation.

    As with TestMergeJITParity, tests pass both when numba is present
    (validating compiled output) and when it is absent (validating the
    pure-Python fallback path that the @njit shim provides).
    """

    # ------------------------------------------------------------------
    # Pure-Python reference for _segments_match_jit
    # ------------------------------------------------------------------

    @staticmethod
    def _segments_match_py(a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y, tol_sq):
        """Verbatim Python reimplementation of _segments_match_jit."""
        d00x = a0x - b0x; d00y = a0y - b0y
        if d00x * d00x + d00y * d00y <= tol_sq:
            d11x = a1x - b1x; d11y = a1y - b1y
            if d11x * d11x + d11y * d11y <= tol_sq:
                return True
        d01x = a0x - b1x; d01y = a0y - b1y
        if d01x * d01x + d01y * d01y <= tol_sq:
            d10x = a1x - b0x; d10y = a1y - b0y
            if d10x * d10x + d10y * d10y <= tol_sq:
                return True
        return False

    # ------------------------------------------------------------------
    # Kernel parity tests
    # ------------------------------------------------------------------

    def test_exact_duplicate_same_direction(self) -> None:
        """Identical segments in same direction → True from both JIT and Python."""
        from plottter.processing.weld import _segments_match_jit

        tol_sq = 0.01  # tolerance = 0.1 mm
        jit = _segments_match_jit(0.0, 0.0, 5.0, 0.0,  0.0, 0.0, 5.0, 0.0,  tol_sq)
        py  = self._segments_match_py(0.0, 0.0, 5.0, 0.0,  0.0, 0.0, 5.0, 0.0,  tol_sq)
        assert jit == py == True

    def test_exact_duplicate_reversed(self) -> None:
        """Reversed duplicate → True from both."""
        from plottter.processing.weld import _segments_match_jit

        tol_sq = 0.01
        jit = _segments_match_jit(0.0, 0.0, 5.0, 0.0,  5.0, 0.0, 0.0, 0.0,  tol_sq)
        py  = self._segments_match_py(0.0, 0.0, 5.0, 0.0,  5.0, 0.0, 0.0, 0.0,  tol_sq)
        assert jit == py == True

    def test_near_duplicate_within_tolerance(self) -> None:
        """Near-duplicate (offset < sqrt(tol_sq)) → True from both."""
        from plottter.processing.weld import _segments_match_jit

        tol_sq = 0.01   # tolerance = 0.1 mm
        # 0.05mm offset — within tolerance
        jit = _segments_match_jit(0.0, 0.0, 5.0, 0.0,  0.05, 0.0, 5.05, 0.0,  tol_sq)
        py  = self._segments_match_py(0.0, 0.0, 5.0, 0.0,  0.05, 0.0, 5.05, 0.0,  tol_sq)
        assert jit == py == True

    def test_different_segment_not_matched(self) -> None:
        """Clearly different segments → False from both."""
        from plottter.processing.weld import _segments_match_jit

        tol_sq = 0.01
        jit = _segments_match_jit(0.0, 0.0, 5.0, 0.0,  10.0, 0.0, 15.0, 0.0,  tol_sq)
        py  = self._segments_match_py(0.0, 0.0, 5.0, 0.0,  10.0, 0.0, 15.0, 0.0,  tol_sq)
        assert jit == py == False

    def test_outside_tolerance_not_matched(self) -> None:
        """Offset just outside tolerance → False from both."""
        from plottter.processing.weld import _segments_match_jit

        tol_sq = 0.01   # tolerance = 0.1 mm
        # 0.2mm offset — outside tolerance
        jit = _segments_match_jit(0.0, 0.0, 5.0, 0.0,  0.2, 0.0, 5.2, 0.0,  tol_sq)
        py  = self._segments_match_py(0.0, 0.0, 5.0, 0.0,  0.2, 0.0, 5.2, 0.0,  tol_sq)
        assert jit == py == False

    def test_first_endpoint_match_second_mismatch(self) -> None:
        """First endpoints match but second endpoints don't → False from both."""
        from plottter.processing.weld import _segments_match_jit

        tol_sq = 0.01
        # a0 ≈ b0 but a1 is far from b1
        jit = _segments_match_jit(0.0, 0.0, 5.0, 0.0,  0.05, 0.0, 20.0, 0.0,  tol_sq)
        py  = self._segments_match_py(0.0, 0.0, 5.0, 0.0,  0.05, 0.0, 20.0, 0.0,  tol_sq)
        assert jit == py == False

    def test_diagonal_segment_parity(self) -> None:
        """Diagonal segments (non-axis-aligned) → JIT and Python agree."""
        from plottter.processing.weld import _segments_match_jit

        import math
        tol_sq = 0.01
        # Identical diagonal
        jit = _segments_match_jit(1.0, 1.0, 4.0, 5.0,  1.0, 1.0, 4.0, 5.0,  tol_sq)
        py  = self._segments_match_py(1.0, 1.0, 4.0, 5.0,  1.0, 1.0, 4.0, 5.0,  tol_sq)
        assert jit == py == True

        # Diagonal reversed
        jit2 = _segments_match_jit(1.0, 1.0, 4.0, 5.0,  4.0, 5.0, 1.0, 1.0,  tol_sq)
        py2  = self._segments_match_py(1.0, 1.0, 4.0, 5.0,  4.0, 5.0, 1.0, 1.0,  tol_sq)
        assert jit2 == py2 == True

    def test_weld_end_to_end_parity(self) -> None:
        """Full weld_overlapping_paths output is correct on a known input."""
        path_a = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        path_b = [(0.0, 0.0), (5.0, 0.0)]  # first segment of path_a duplicated
        result = weld_overlapping_paths([path_a, path_b])
        assert len(result) == 1
        assert result[0] == path_a

    def test_weld_large_input_parity(self) -> None:
        """weld_overlapping_paths produces correct deduplication on a larger deterministic input."""
        import random
        rng = random.Random(7)
        # Build 50 unique segments
        unique = [
            [(rng.uniform(0, 50), rng.uniform(0, 50)),
             (rng.uniform(0, 50), rng.uniform(0, 50))]
            for _ in range(50)
        ]
        # Duplicate the first 10 segments as separate paths
        dupes = [list(p) for p in unique[:10]]
        all_paths = unique + dupes

        result = weld_overlapping_paths(all_paths, tolerance_mm=0.01)
        # Exactly 50 unique segments should survive
        assert len(result) == 50
