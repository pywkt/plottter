"""Tests for optimize_3opt() in processing/optimize.py."""

import math
import random
import pytest

from plottter.processing.optimize import optimize_3opt, calculate_travel_distance
from plottter.processing import optimize_3opt as pkg_optimize_3opt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_point_paths(n: int, seed: int = 42) -> list[list[tuple[float, float]]]:
    """Make n single-point paths at random positions."""
    rng = random.Random(seed)
    return [[(rng.uniform(0, 200), rng.uniform(0, 200))] * 2 for _ in range(n)]


def seg(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y1)]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestOptimize3optEdgeCases:
    def test_empty_returns_empty(self) -> None:
        assert optimize_3opt([]) == []

    def test_single_path_returned_unchanged(self) -> None:
        path = [seg(0.0, 0.0, 1.0, 0.0)]
        result = optimize_3opt(path)
        assert len(result) == 1
        assert result[0] == path[0]

    def test_two_paths_returned_unchanged(self) -> None:
        paths = [seg(0.0, 0.0, 1.0, 0.0), seg(5.0, 0.0, 6.0, 0.0)]
        result = optimize_3opt(paths)
        assert len(result) == 2
        assert result[0] == paths[0]
        assert result[1] == paths[1]

    def test_empty_returns_list(self) -> None:
        result = optimize_3opt([])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Distance reduction
# ---------------------------------------------------------------------------

class TestOptimize3optDistanceReduction:
    def test_reduces_travel_distance_random_50(self) -> None:
        """3-opt on 50 random single-point paths must strictly reduce travel distance."""
        paths = make_point_paths(50, seed=42)
        before = calculate_travel_distance(paths)
        result = optimize_3opt(paths, max_iterations=500)
        after = calculate_travel_distance(result)
        assert after < before, f"3-opt did not reduce distance: {before:.2f} → {after:.2f}"

    def test_reduces_travel_distance_larger(self) -> None:
        """3-opt on 100 random paths should not increase travel distance."""
        paths = make_point_paths(100, seed=99)
        before = calculate_travel_distance(paths)
        result = optimize_3opt(paths, max_iterations=200)
        after = calculate_travel_distance(result)
        assert after <= before + 1e-6

    def test_preserves_path_count(self) -> None:
        """Result must have the same number of paths as input."""
        paths = make_point_paths(50, seed=7)
        result = optimize_3opt(paths)
        assert len(result) == len(paths)

    def test_returns_list_of_polylines(self) -> None:
        paths = make_point_paths(20, seed=1)
        result = optimize_3opt(paths)
        assert isinstance(result, list)
        for p in result:
            assert isinstance(p, list)
            for pt in p:
                assert len(pt) == 2

    def test_already_optimal_does_not_increase_distance(self) -> None:
        """A well-ordered sequence should not be worsened."""
        # Sequential line along x-axis — already near-optimal
        paths = [seg(float(i * 10), 0.0, float(i * 10 + 5), 0.0) for i in range(20)]
        before = calculate_travel_distance(paths)
        result = optimize_3opt(paths, max_iterations=200)
        after = calculate_travel_distance(result)
        assert after <= before + 1e-6

    def test_three_paths_minimum(self) -> None:
        """Exactly 3 paths is the minimum for 3-opt to run."""
        paths = [seg(0.0, 0.0, 1.0, 0.0), seg(100.0, 0.0, 101.0, 0.0), seg(50.0, 0.0, 51.0, 0.0)]
        result = optimize_3opt(paths, max_iterations=100)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestOptimize3optCancellation:
    def test_cancellation_stops_early(self) -> None:
        """Cancelled flag causes early return."""
        call_count = [0]

        def cancel_after_first() -> bool:
            call_count[0] += 1
            return call_count[0] > 1  # cancel after first check

        paths = make_point_paths(50, seed=5)
        result = optimize_3opt(paths, max_iterations=1000, cancelled=cancel_after_first)
        # Must still return all paths
        assert len(result) == len(paths)

    def test_cancellation_returns_valid_route(self) -> None:
        """Cancelled result must still be a valid list of polylines."""
        cancelled_flag = [False]

        def cancel() -> bool:
            cancelled_flag[0] = True
            return True  # cancel immediately

        paths = make_point_paths(30, seed=3)
        result = optimize_3opt(paths, cancelled=cancel)
        assert isinstance(result, list)
        assert len(result) == len(paths)

    def test_no_cancellation_runs_to_completion(self) -> None:
        """With cancelled=None, function completes normally."""
        paths = make_point_paths(20, seed=11)
        result = optimize_3opt(paths, max_iterations=100, cancelled=None)
        assert len(result) == len(paths)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestOptimize3optProgressCallback:
    def test_progress_callback_fires(self) -> None:
        """Progress callback must be called at least once."""
        calls: list[float] = []

        def cb(v: float) -> None:
            calls.append(v)

        paths = make_point_paths(30, seed=22)
        optimize_3opt(paths, max_iterations=10, progress_callback=cb)
        assert len(calls) > 0

    def test_progress_values_in_range(self) -> None:
        """All reported progress values must be in [0.0, 1.0]."""
        calls: list[float] = []

        def cb(v: float) -> None:
            calls.append(v)

        paths = make_point_paths(20, seed=33)
        optimize_3opt(paths, max_iterations=20, progress_callback=cb)
        for v in calls:
            assert 0.0 <= v <= 1.0, f"progress out of range: {v}"

    def test_progress_non_decreasing(self) -> None:
        """Progress values must be non-decreasing."""
        calls: list[float] = []

        def cb(v: float) -> None:
            calls.append(v)

        paths = make_point_paths(30, seed=44)
        optimize_3opt(paths, max_iterations=20, progress_callback=cb)
        for a, b in zip(calls, calls[1:]):
            assert b >= a - 1e-9, f"progress went backwards: {a} → {b}"

    def test_no_callback_runs_fine(self) -> None:
        """progress_callback=None must not raise."""
        paths = make_point_paths(15, seed=55)
        result = optimize_3opt(paths, max_iterations=10, progress_callback=None)
        assert len(result) == len(paths)


# ---------------------------------------------------------------------------
# Package-level export
# ---------------------------------------------------------------------------

class TestOptimize3optPackageExport:
    def test_importable_from_package(self) -> None:
        assert pkg_optimize_3opt is not None

    def test_package_export_is_same_function(self) -> None:
        assert pkg_optimize_3opt is optimize_3opt
