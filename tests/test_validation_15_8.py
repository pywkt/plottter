"""Phase 15.8 validation: path optimization pipeline.

Verifies the full path optimization pipeline against the spec in
specs/path_optimization.md:

1. Travel distance decreases after reorder + 2-opt on a deliberately
   pessimistic (reverse-sorted) path arrangement.
2. Path count is preserved by reorder and 2-opt.
3. Total pen-down (visual) path length is unchanged after reorder/2-opt.
4. Before/after metrics (travel distance, pen-lift count, percent reduction)
   are computed correctly and reflect measurable improvements.
5. Each individual pipeline step (simplify, filter, clip, merge, weld) behaves
   correctly in isolation and their combination gives coherent results.
6. End-to-end pipeline using real generator output (ParametricGenerator) shows
   measurable improvement for a deliberately unordered layer.
7. Edge cases: empty input, single path, already-optimal input.
"""

from __future__ import annotations

import math
import random

import pytest

from plottter.processing.simplify import simplify_paths, simplify_polyline
from plottter.processing.filter import filter_short_paths
from plottter.processing.clip import clip_to_bounds
from plottter.processing.merge import merge_nearby_paths
from plottter.processing.optimize import (
    reorder_paths,
    optimize_2opt,
    calculate_travel_distance,
)
from plottter.processing.weld import weld_overlapping_paths
from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pen_down_distance(paths: list[list[tuple[float, float]]]) -> float:
    """Total length of all polylines (pen-down distance)."""
    total = 0.0
    for path in paths:
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            total += math.hypot(x2 - x1, y2 - y1)
    return total


def _pen_lift_count(paths: list[list[tuple[float, float]]]) -> int:
    """Number of pen lifts = number of paths."""
    return len(paths)


def _percent_reduction(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


def _make_grid_paths(
    rows: int = 5,
    cols: int = 5,
    spacing: float = 20.0,
    offset_x: float = 10.0,
    offset_y: float = 10.0,
    points_per_path: int = 3,
) -> list[list[tuple[float, float]]]:
    """Grid of short horizontal strokes spread across a canvas area.

    Each path is a horizontal segment at position (c*spacing, r*spacing).
    The returned order is column-major so adjacent paths are far apart —
    a worst-case arrangement for pen travel before reordering.
    """
    paths = []
    for c in range(cols):
        for r in range(rows):
            x0 = offset_x + c * spacing
            y0 = offset_y + r * spacing
            # Small horizontal segment (length = spacing/4)
            seg_len = spacing / 4.0
            step = seg_len / max(points_per_path - 1, 1)
            pts = [(x0 + i * step, y0) for i in range(points_per_path)]
            paths.append(pts)
    return paths


def _make_scattered_paths(n: int = 30, seed: int = 42) -> list[list[tuple[float, float]]]:
    """Random short strokes scattered over a 200×200 mm area."""
    rng = random.Random(seed)
    paths = []
    for _ in range(n):
        x = rng.uniform(10.0, 190.0)
        y = rng.uniform(10.0, 190.0)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(5.0, 20.0)
        x2 = x + length * math.cos(angle)
        y2 = y + length * math.sin(angle)
        paths.append([(x, y), (x2, y2)])
    return paths


def _total_point_count(paths: list[list[tuple[float, float]]]) -> int:
    return sum(len(p) for p in paths)


# ---------------------------------------------------------------------------
# 1. Travel distance decreases after reordering
# ---------------------------------------------------------------------------


class TestTravelDistanceDecreasesAfterReorder:
    """Reorder + 2-opt must reduce pen-up travel for sub-optimal arrangements."""

    def test_column_major_grid_improves_on_reorder(self) -> None:
        """Column-major grid (bad order) gets shorter travel after reorder."""
        paths = _make_grid_paths(rows=6, cols=6, spacing=20.0)
        before = calculate_travel_distance(paths)
        reordered = reorder_paths(paths)
        after = calculate_travel_distance(reordered)
        assert after < before, (
            f"Expected travel to decrease: before={before:.2f}, after={after:.2f}"
        )

    def test_2opt_further_improves_on_nn_result(self) -> None:
        """2-opt must produce travel distance <= nearest-neighbor result."""
        paths = _make_scattered_paths(n=40, seed=7)
        nn_paths = reorder_paths(paths)
        nn_travel = calculate_travel_distance(nn_paths)
        opt_paths = optimize_2opt(nn_paths, max_iterations=200)
        opt_travel = calculate_travel_distance(opt_paths)
        # 2-opt should not make things worse
        assert opt_travel <= nn_travel + 1e-6, (
            f"2-opt worsened travel: nn={nn_travel:.2f}, 2opt={opt_travel:.2f}"
        )

    def test_reverse_sorted_paths_improve_significantly(self) -> None:
        """Paths sorted in reverse-optimal order see large improvement."""
        # 10 paths along a diagonal — optimal is sequential, reversed is worst-case
        paths = [[(float(i * 10), float(i * 10)),
                  (float(i * 10 + 5), float(i * 10 + 5))]
                 for i in range(10)]
        # Reverse so pen must jump back and forth
        reversed_paths = list(reversed(paths))
        before = calculate_travel_distance(reversed_paths)
        optimized = optimize_2opt(reorder_paths(reversed_paths))
        after = calculate_travel_distance(optimized)
        reduction = _percent_reduction(before, after)
        # For a simple diagonal, reversal produces ~9x the optimal travel
        assert after < before, f"No improvement: before={before:.2f}, after={after:.2f}"
        assert reduction > 30.0, f"Expected >30% reduction, got {reduction:.1f}%"

    def test_scattered_paths_improve_on_full_pipeline(self) -> None:
        """Full pipeline (reorder + 2-opt) on scattered paths gives improvement."""
        paths = _make_scattered_paths(n=50, seed=13)
        before = calculate_travel_distance(paths)
        optimized = optimize_2opt(reorder_paths(paths))
        after = calculate_travel_distance(optimized)
        assert after < before

    def test_percent_reduction_formula(self) -> None:
        """Percent reduction formula: (before-after)/before*100."""
        assert abs(_percent_reduction(100.0, 75.0) - 25.0) < 1e-6
        assert abs(_percent_reduction(200.0, 100.0) - 50.0) < 1e-6
        assert _percent_reduction(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 2. Path count preservation
# ---------------------------------------------------------------------------


class TestPathCountPreservation:
    """Reorder and 2-opt must preserve path count (no paths lost or duplicated)."""

    def test_reorder_preserves_count(self) -> None:
        paths = _make_grid_paths(rows=4, cols=4, spacing=15.0)
        reordered = reorder_paths(paths)
        assert len(reordered) == len(paths)

    def test_2opt_preserves_count(self) -> None:
        paths = _make_scattered_paths(n=20, seed=99)
        reordered = reorder_paths(paths)
        optimized = optimize_2opt(reordered)
        assert len(optimized) == len(paths)

    def test_reorder_preserves_all_path_content(self) -> None:
        """Every path from the input must appear in the output (possibly reversed)."""
        paths = _make_grid_paths(rows=3, cols=3, spacing=10.0)
        # Represent each path as a sorted tuple of its endpoints for deterministic comparison
        def path_sig(p: list[tuple[float, float]]) -> tuple:
            return tuple(sorted([p[0], p[-1]]))

        input_sigs = sorted(path_sig(p) for p in paths)
        reordered = reorder_paths(paths)
        output_sigs = sorted(path_sig(p) for p in reordered)

        assert input_sigs == output_sigs, "Reorder changed path content"

    def test_2opt_preserves_all_path_content(self) -> None:
        """2-opt must not add, remove, or alter path drawing content."""
        paths = _make_scattered_paths(n=15, seed=5)
        reordered = reorder_paths(paths)
        optimized = optimize_2opt(reordered)
        assert len(optimized) == len(reordered)

        def path_key(p: list) -> tuple:
            return tuple(sorted([p[0], p[-1]]))

        before_keys = sorted(path_key(p) for p in reordered)
        after_keys = sorted(path_key(p) for p in optimized)
        assert before_keys == after_keys


# ---------------------------------------------------------------------------
# 3. Visual output unchanged (pen-down distance preserved)
# ---------------------------------------------------------------------------


class TestVisualOutputUnchanged:
    """Reorder and 2-opt must preserve total pen-down (drawing) distance."""

    def _pen_down_equal(
        self,
        paths_a: list[list[tuple[float, float]]],
        paths_b: list[list[tuple[float, float]]],
        tol: float = 1e-6,
    ) -> bool:
        return abs(_pen_down_distance(paths_a) - _pen_down_distance(paths_b)) < tol

    def test_reorder_preserves_pen_down_distance(self) -> None:
        paths = _make_grid_paths(rows=5, cols=5, spacing=18.0)
        reordered = reorder_paths(paths)
        assert self._pen_down_equal(paths, reordered), (
            f"Pen-down changed: {_pen_down_distance(paths):.4f} → "
            f"{_pen_down_distance(reordered):.4f}"
        )

    def test_2opt_preserves_pen_down_distance(self) -> None:
        paths = _make_scattered_paths(n=25, seed=17)
        reordered = reorder_paths(paths)
        optimized = optimize_2opt(reordered)
        assert self._pen_down_equal(reordered, optimized), (
            f"Pen-down changed: {_pen_down_distance(reordered):.4f} → "
            f"{_pen_down_distance(optimized):.4f}"
        )

    def test_full_pipeline_preserves_visual_points(self) -> None:
        """Simplify (tight tolerance) + reorder must not lose drawn content."""
        paths = _make_grid_paths(rows=4, cols=4, spacing=20.0, points_per_path=10)
        # Use a tolerance so small it shouldn't simplify any real geometry
        simplified = simplify_paths(paths, tolerance_mm=0.001)
        reordered = reorder_paths(simplified)
        # Point count may decrease (simplify) but pen-down distance should be ~preserved
        before_pd = _pen_down_distance(paths)
        after_pd = _pen_down_distance(reordered)
        # Allow up to 1% deviation due to RDP simplification on diagonal points
        assert abs(after_pd - before_pd) / max(before_pd, 1e-9) < 0.01, (
            f"Pen-down changed by >1%: {before_pd:.4f} → {after_pd:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Metrics: before/after computation matches manual calculation
# ---------------------------------------------------------------------------


class TestMetricsCalculation:
    """Verify that the metric formulas match manual calculations."""

    def test_travel_distance_origin_to_first_path(self) -> None:
        """Travel includes initial move from (0,0) to first path start."""
        paths = [[(10.0, 0.0), (20.0, 0.0)]]
        # travel = dist(origin→path0.start) + dist(path0.end→origin)
        expected = math.hypot(10.0, 0.0) + math.hypot(20.0, 0.0)
        assert abs(calculate_travel_distance(paths) - expected) < 1e-6

    def test_travel_distance_multi_path(self) -> None:
        """Travel between consecutive paths is end→start of next path."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(10.0, 0.0), (15.0, 0.0)],
        ]
        # origin→p0.start + p0.end→p1.start + p1.end→origin
        expected = (
            math.hypot(0.0, 0.0)       # origin → (0,0)
            + math.hypot(5.0, 0.0)     # (5,0) → (10,0)
            + math.hypot(15.0, 0.0)    # (15,0) → origin
        )
        assert abs(calculate_travel_distance(paths) - expected) < 1e-6

    def test_pen_lift_count_equals_path_count(self) -> None:
        """Pen lift count is simply the number of paths."""
        paths = _make_grid_paths(rows=3, cols=3, spacing=10.0)
        assert _pen_lift_count(paths) == len(paths) == 9

    def test_before_after_metrics_improve_after_reorder(self) -> None:
        """before_travel > after_travel; before_lifts == after_lifts for reorder."""
        paths = _make_grid_paths(rows=5, cols=5, spacing=20.0)
        before_travel = calculate_travel_distance(paths)
        before_lifts = _pen_lift_count(paths)

        optimized = optimize_2opt(reorder_paths(paths))
        after_travel = calculate_travel_distance(optimized)
        after_lifts = _pen_lift_count(optimized)

        assert after_travel < before_travel
        assert after_lifts == before_lifts  # reorder doesn't remove paths
        reduction = _percent_reduction(before_travel, after_travel)
        assert reduction > 0.0

    def test_merge_reduces_pen_lifts(self) -> None:
        """Merging connected paths reduces pen lift count."""
        # Two paths sharing an endpoint within threshold
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(5.0, 0.0), (10.0, 0.0)],  # shares endpoint with path 0
        ]
        before_lifts = _pen_lift_count(paths)
        merged = merge_nearby_paths(paths, threshold_mm=0.1)
        after_lifts = _pen_lift_count(merged)
        assert after_lifts < before_lifts, (
            f"Expected merge to reduce lifts: {before_lifts} → {after_lifts}"
        )

    def test_filter_reduces_path_count(self) -> None:
        """Filter removes short paths and reduces count."""
        paths = [
            [(0.0, 0.0), (0.1, 0.0)],   # length 0.1 — short (< 0.5)
            [(0.0, 0.0), (5.0, 0.0)],   # length 5.0 — kept
            [(0.0, 0.0), (0.05, 0.0)],  # length 0.05 — very short
        ]
        filtered = filter_short_paths(paths, min_length_mm=0.5)
        assert len(filtered) == 1
        assert filtered[0] == [(0.0, 0.0), (5.0, 0.0)]

    def test_percent_reduction_is_non_negative_after_optimization(self) -> None:
        """Optimization should never increase travel distance."""
        paths = _make_scattered_paths(n=20, seed=3)
        before = calculate_travel_distance(paths)
        after = calculate_travel_distance(optimize_2opt(reorder_paths(paths)))
        assert _percent_reduction(before, after) >= 0.0


# ---------------------------------------------------------------------------
# 5. Individual pipeline steps
# ---------------------------------------------------------------------------


class TestSimplify:
    """RDP simplification reduces point count while preserving shape."""

    def test_straight_line_reduces_to_two_points(self) -> None:
        poly = [(float(i), 0.0) for i in range(10)]
        result = simplify_polyline(poly, tolerance_mm=0.1)
        assert len(result) == 2
        assert result[0] == poly[0]
        assert result[-1] == poly[-1]

    def test_zigzag_mostly_preserved(self) -> None:
        """Zigzag points are far from any simplified baseline — must be kept."""
        poly = [(float(i), 10.0 if i % 2 == 0 else 0.0) for i in range(20)]
        result = simplify_polyline(poly, tolerance_mm=0.1)
        # Zigzag amplitude=10 >> tolerance=0.1, so all points should be kept
        assert len(result) == len(poly)

    def test_simplify_paths_applies_to_all_paths(self) -> None:
        paths = [
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],  # collinear
            [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)],  # has deviation
        ]
        result = simplify_paths(paths, tolerance_mm=0.1)
        assert len(result) == 2
        assert len(result[0]) == 2  # collinear → 2 points
        assert len(result[1]) == 3  # peak point preserved

    def test_simplify_preserves_endpoints(self) -> None:
        poly = [(0.0, 0.0)] + [(float(i), 0.01) for i in range(1, 9)] + [(10.0, 0.0)]
        result = simplify_polyline(poly, tolerance_mm=0.1)
        assert result[0] == (0.0, 0.0)
        assert result[-1] == (10.0, 0.0)

    def test_zero_tolerance_keeps_all_points(self) -> None:
        poly = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0), (3.0, 0.7)]
        result = simplify_polyline(poly, tolerance_mm=0.0)
        assert len(result) == len(poly)

    def test_simplify_reduces_total_point_count(self) -> None:
        """A path with many nearly-collinear points has fewer after simplify."""
        # Slightly noisy horizontal line
        rng = random.Random(1)
        poly = [(float(i), rng.uniform(-0.005, 0.005)) for i in range(100)]
        result = simplify_polyline(poly, tolerance_mm=0.05)
        assert len(result) < len(poly)


class TestFilter:
    """Short-path filter removes paths below minimum length threshold."""

    def test_removes_paths_below_threshold(self) -> None:
        paths = [[(0.0, 0.0), (0.3, 0.0)]]  # length 0.3 < 0.5
        result = filter_short_paths(paths, min_length_mm=0.5)
        assert result == []

    def test_keeps_paths_at_or_above_threshold(self) -> None:
        paths = [[(0.0, 0.0), (0.5, 0.0)]]  # exactly 0.5
        result = filter_short_paths(paths, min_length_mm=0.5)
        assert len(result) == 1

    def test_mixed_keeps_only_long_paths(self) -> None:
        paths = [
            [(0.0, 0.0), (0.1, 0.0)],   # short
            [(0.0, 0.0), (10.0, 0.0)],  # long
            [(0.0, 0.0), (0.4, 0.0)],   # short
        ]
        result = filter_short_paths(paths, min_length_mm=0.5)
        assert len(result) == 1
        assert result[0][1] == (10.0, 0.0)

    def test_filter_counts_multi_segment_length(self) -> None:
        """Path length is sum of all segment lengths."""
        # Two segments each 0.3 mm → total 0.6 mm
        path = [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0)]
        result = filter_short_paths([path], min_length_mm=0.5)
        assert len(result) == 1

    def test_empty_input_returns_empty(self) -> None:
        assert filter_short_paths([], min_length_mm=1.0) == []


class TestClip:
    """Canvas clipping splits paths crossing bounds and removes paths outside."""

    def _make_bounds(self) -> tuple[float, float, float, float]:
        return (10.0, 10.0, 190.0, 190.0)

    def test_path_fully_inside_unchanged(self) -> None:
        path = [(20.0, 20.0), (100.0, 100.0)]
        result = clip_to_bounds([path], self._make_bounds())
        assert len(result) == 1

    def test_path_fully_outside_removed(self) -> None:
        path = [(200.0, 200.0), (250.0, 250.0)]
        result = clip_to_bounds([path], self._make_bounds())
        assert result == []

    def test_crossing_path_is_split(self) -> None:
        """Path crossing the boundary produces a clipped segment inside bounds."""
        # Horizontal path crossing left bound at x=10
        path = [(0.0, 50.0), (100.0, 50.0)]
        result = clip_to_bounds([path], self._make_bounds())
        # At least one segment entirely within bounds
        assert len(result) >= 1
        for clipped_path in result:
            for x, y in clipped_path:
                assert 10.0 - 1e-6 <= x <= 190.0 + 1e-6
                assert 10.0 - 1e-6 <= y <= 190.0 + 1e-6

    def test_clip_all_paths_to_canvas_drawing_area(self) -> None:
        """Paths clipped to canvas drawing_area stay within bounds."""
        canvas = Canvas.from_preset("A4", margin=10.0)
        x1, y1, x2, y2 = canvas.drawing_area()
        paths = [
            [(0.0, 100.0), (300.0, 100.0)],  # horizontal crossing both edges
            [(105.0, 0.0), (105.0, 400.0)],  # vertical crossing both edges
            [(x1 + 5, y1 + 5), (x2 - 5, y2 - 5)],  # fully inside
        ]
        result = clip_to_bounds(paths, (x1, y1, x2, y2))
        for path in result:
            for x, y in path:
                assert x1 - 1e-6 <= x <= x2 + 1e-6, f"x={x} outside [{x1}, {x2}]"
                assert y1 - 1e-6 <= y <= y2 + 1e-6, f"y={y} outside [{y1}, {y2}]"


class TestMerge:
    """Merge nearby paths joins paths whose endpoints are within threshold."""

    def test_adjacent_paths_merged_to_one(self) -> None:
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(5.0, 0.0), (10.0, 0.0)],
        ]
        result = merge_nearby_paths(paths, threshold_mm=0.1)
        assert len(result) == 1
        assert result[0][0] == (0.0, 0.0)
        assert result[0][-1] == (10.0, 0.0)

    def test_non_adjacent_paths_not_merged(self) -> None:
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(100.0, 0.0), (105.0, 0.0)],
        ]
        result = merge_nearby_paths(paths, threshold_mm=0.5)
        assert len(result) == 2

    def test_merge_preserves_total_drawn_distance(self) -> None:
        """Total pen-down length is approximately preserved after merging."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(5.0, 0.0), (10.0, 0.0)],
        ]
        before = _pen_down_distance(paths)
        merged = merge_nearby_paths(paths, threshold_mm=0.1)
        after = _pen_down_distance(merged)
        # Allow a tiny float tolerance
        assert abs(after - before) < 1e-6

    def test_merge_reduces_travel_distance(self) -> None:
        """After merging, fewer paths means fewer pen lifts and less travel."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(5.0, 0.0), (10.0, 0.0)],
        ]
        before_travel = calculate_travel_distance(paths)
        merged = merge_nearby_paths(paths, threshold_mm=0.1)
        after_travel = calculate_travel_distance(merged)
        assert after_travel <= before_travel

    def test_empty_input_returns_empty(self) -> None:
        assert merge_nearby_paths([], threshold_mm=0.5) == []

    def test_single_path_unchanged(self) -> None:
        paths = [[(0.0, 0.0), (5.0, 5.0)]]
        result = merge_nearby_paths(paths, threshold_mm=0.5)
        assert len(result) == 1


class TestWeld:
    """Weld removes duplicate overlapping segments."""

    def test_duplicate_segment_removed(self) -> None:
        """Two identical overlapping paths reduce to one."""
        seg = [(0.0, 0.0), (5.0, 0.0)]
        paths = [seg, seg[:]]  # exact duplicate
        result = weld_overlapping_paths(paths, tolerance_mm=0.1)
        # One path should be removed (the duplicate)
        assert len(result) == 1

    def test_non_overlapping_paths_unchanged(self) -> None:
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(10.0, 0.0), (15.0, 0.0)],
        ]
        result = weld_overlapping_paths(paths, tolerance_mm=0.1)
        assert len(result) == 2

    def test_weld_reduces_total_segments(self) -> None:
        """Welding exact duplicate paths produces fewer total segments."""
        paths = []
        # 5 identical horizontal segments
        for _ in range(5):
            paths.append([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
        result = weld_overlapping_paths(paths, tolerance_mm=0.1)
        assert _total_point_count(result) < _total_point_count(paths)

    def test_reversed_duplicate_removed(self) -> None:
        """A reversed duplicate of a segment is also detected and removed."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0)],
            [(5.0, 0.0), (0.0, 0.0)],  # reversed
        ]
        result = weld_overlapping_paths(paths, tolerance_mm=0.1)
        # Should have fewer segments than input
        assert _total_point_count(result) < _total_point_count(paths)

    def test_empty_input_returns_empty(self) -> None:
        assert weld_overlapping_paths([], tolerance_mm=0.1) == []

    def test_single_path_returned_unchanged(self) -> None:
        paths = [[(0.0, 0.0), (5.0, 0.0)]]
        result = weld_overlapping_paths(paths, tolerance_mm=0.1)
        assert len(result) == 1
        assert result[0] == [(0.0, 0.0), (5.0, 0.0)]


# ---------------------------------------------------------------------------
# 6. End-to-end pipeline with real generator output
# ---------------------------------------------------------------------------


class TestEndToEndOptimizationPipeline:
    """Full optimization pipeline using ParametricGenerator output."""

    def test_parametric_generator_pipeline(self) -> None:
        """Parametric generator output shows improvement after optimization."""
        from plottter.generators.parametric import ParametricGenerator
        from plottter.models.canvas import Canvas

        canvas = Canvas.from_preset("A4", margin=10.0)
        gen = ParametricGenerator()
        # Use Lissajous preset for reproducible complex output
        presets = {p.name: p.params for p in gen.get_presets()}
        params = presets.get("Lissajous", {})
        if not params:
            params = {"num_points": 2000, "t_start": 0.0, "t_end": 6.283185}

        paths = gen.generate(params, canvas)
        assert len(paths) > 0, "Generator produced no paths"

        before_travel = calculate_travel_distance(paths)
        before_lifts = _pen_lift_count(paths)
        before_pd = _pen_down_distance(paths)

        # Run full pipeline (skip clip to avoid removing any drawing content)
        simplified = simplify_paths(paths, tolerance_mm=0.1)
        reordered = reorder_paths(simplified)
        optimized = optimize_2opt(reordered, max_iterations=100)

        after_travel = calculate_travel_distance(optimized)
        after_lifts = _pen_lift_count(optimized)
        after_pd = _pen_down_distance(optimized)

        # Travel should decrease or stay the same
        assert after_travel <= before_travel + 1e-6
        # Pen count preserved by reorder/2-opt
        assert after_lifts == before_lifts
        # Pen-down distance approximately preserved (allow 1% for simplification)
        assert abs(after_pd - before_pd) / max(before_pd, 1e-9) < 0.01

    def test_full_pipeline_with_clip_and_merge(self) -> None:
        """Full pipeline including clip and merge on a synthetic multi-path layer."""
        canvas = Canvas.from_preset("A4", margin=10.0)
        x1, y1, x2, y2 = canvas.drawing_area()

        # Create paths mostly within canvas but a few crossing the boundary
        paths = _make_grid_paths(
            rows=8, cols=8, spacing=20.0,
            offset_x=x1, offset_y=y1,
        )
        # Add some paths crossing the boundary
        paths += [
            [(0.0, 100.0), (300.0, 100.0)],  # horizontal, crosses both edges
            [(50.0, 0.0), (50.0, 400.0)],    # vertical, crosses both edges
        ]

        before_travel = calculate_travel_distance(paths)

        # Run full pipeline
        clipped = clip_to_bounds(paths, (x1, y1, x2, y2))
        filtered = filter_short_paths(clipped, min_length_mm=0.5)
        merged = merge_nearby_paths(filtered, threshold_mm=0.5)
        reordered = reorder_paths(merged)
        optimized = optimize_2opt(reordered, max_iterations=100)

        after_travel = calculate_travel_distance(optimized)

        # All resulting points must be within canvas bounds
        for path in optimized:
            for x, y in path:
                assert x1 - 1e-6 <= x <= x2 + 1e-6
                assert y1 - 1e-6 <= y <= y2 + 1e-6

        # Travel should be <= before (the extra paths extended the total dramatically)
        assert after_travel <= before_travel


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty input, single path, already-optimal."""

    def test_travel_distance_empty(self) -> None:
        assert calculate_travel_distance([]) == 0.0

    def test_travel_distance_single_path(self) -> None:
        path = [(5.0, 5.0), (10.0, 5.0)]
        dist = calculate_travel_distance([path])
        # origin→(5,5) + (10,5)→origin
        expected = math.hypot(5.0, 5.0) + math.hypot(10.0, 5.0)
        assert abs(dist - expected) < 1e-6

    def test_reorder_empty_returns_empty(self) -> None:
        assert reorder_paths([]) == []

    def test_reorder_single_path_unchanged(self) -> None:
        path = [(1.0, 2.0), (3.0, 4.0)]
        result = reorder_paths([path])
        assert len(result) == 1

    def test_2opt_empty_returns_empty(self) -> None:
        assert optimize_2opt([]) == []

    def test_2opt_single_path_returns_single(self) -> None:
        result = optimize_2opt([[(0.0, 0.0), (1.0, 0.0)]])
        assert len(result) == 1

    def test_2opt_two_paths_returns_two(self) -> None:
        paths = [[(0.0, 0.0), (1.0, 0.0)], [(2.0, 0.0), (3.0, 0.0)]]
        result = optimize_2opt(paths)
        assert len(result) == 2

    def test_already_optimal_path_unchanged(self) -> None:
        """Sequential paths on a diagonal are near-optimal; travel stays same."""
        # Paths arranged sequentially along x-axis — already optimal
        paths = [[(float(i * 10), 0.0), (float(i * 10 + 5), 0.0)] for i in range(5)]
        before = calculate_travel_distance(paths)
        after = calculate_travel_distance(optimize_2opt(reorder_paths(paths)))
        # Should not get worse
        assert after <= before + 1e-6

    def test_degenerate_single_point_path_handled(self) -> None:
        """Single-point paths are filtered by reorder (< 2 points)."""
        paths = [[(5.0, 5.0)], [(1.0, 1.0), (2.0, 2.0)]]
        # reorder_paths skips paths with < 2 points
        result = reorder_paths(paths)
        # Should contain the valid 2-point path
        assert any(len(p) == 2 for p in result)

    def test_large_path_count_runs_without_error(self) -> None:
        """100 paths run through the full pipeline without exceptions."""
        paths = _make_scattered_paths(n=100, seed=42)
        result = optimize_2opt(reorder_paths(paths))
        assert len(result) == len(paths)


# ---------------------------------------------------------------------------
# 8. Integration with Layer and Project models
# ---------------------------------------------------------------------------


class TestOptimizationWithLayerModel:
    """Verify optimization integrates cleanly with Layer and Project models."""

    def test_layer_paths_before_after_optimization(self) -> None:
        """Paths assigned to a layer can be round-tripped through optimization."""
        layer = Layer(name="Test", color="#000000")
        paths = _make_grid_paths(rows=4, cols=4, spacing=15.0)
        layer.add_paths(paths)

        assert layer.path_count() == 16

        # Simulate optimization result
        optimized = optimize_2opt(reorder_paths(layer.paths))
        layer.clear_paths()
        layer.add_paths(optimized)

        assert layer.path_count() == 16  # count preserved

    def test_optimization_metrics_from_layer_paths(self) -> None:
        """Metrics computed from layer paths are consistent."""
        layer = Layer(name="Metrics", color="#FF0000")
        paths = _make_grid_paths(rows=5, cols=5, spacing=20.0)
        layer.add_paths(paths)

        before_travel = calculate_travel_distance(layer.paths)
        before_lifts = _pen_lift_count(layer.paths)

        optimized = optimize_2opt(reorder_paths(layer.paths))
        after_travel = calculate_travel_distance(optimized)
        after_lifts = _pen_lift_count(optimized)

        reduction = _percent_reduction(before_travel, after_travel)

        assert before_travel > 0
        assert after_travel > 0
        assert after_travel < before_travel
        assert after_lifts == before_lifts
        assert reduction > 0.0

    def test_optimization_does_not_mutate_layer_paths(self) -> None:
        """Optimization functions return new lists; layer.paths is unchanged."""
        layer = Layer(name="Immutable", color="#00FF00")
        paths = _make_grid_paths(rows=3, cols=3, spacing=10.0)
        layer.add_paths(paths)

        original_count = layer.path_count()
        original_first_point = layer.paths[0][0]

        # Run optimization
        reorder_paths(layer.paths)
        optimize_2opt(layer.paths)

        # Layer must be unchanged
        assert layer.path_count() == original_count
        assert layer.paths[0][0] == original_first_point
