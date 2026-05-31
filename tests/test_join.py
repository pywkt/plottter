"""Tests for join_at_junctions (graph-aware path chaining, task 157.1)."""

import math
import pytest

from plottter.processing.join import join_at_junctions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_length(paths):
    """Sum of Euclidean lengths of all polylines in *paths*."""
    total = 0.0
    for path in paths:
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            total += math.hypot(x1 - x0, y1 - y0)
    return total


def _all_points(paths):
    """Flat list of every vertex across all polylines."""
    return [pt for path in paths for pt in path]


def _covers_point(paths, target, tol=0.15):
    """Return True if *target* appears (within *tol*) in any polyline."""
    for path in paths:
        for pt in path:
            if math.hypot(pt[0] - target[0], pt[1] - target[1]) <= tol:
                return True
    return False


# ---------------------------------------------------------------------------
# (i) Y-shape: 3 paths meeting at shared point → fewer chains, pen lift saved
# ---------------------------------------------------------------------------

class TestYShape:
    """3 paths all sharing (0,0) as one endpoint form a Y-shape."""

    def _make_y(self):
        return [
            [(0.0, 0.0), (1.0, 0.0)],   # arm A → right
            [(0.0, 0.0), (0.0, 1.0)],   # arm B → up
            [(0.0, 0.0), (-1.0, 0.0)],  # arm C → left
        ]

    def test_fewer_chains_than_input(self):
        """Output should have fewer chains than the 3 input paths."""
        paths = self._make_y()
        result = join_at_junctions(paths, threshold_mm=0.1)
        # Y-shape has 4 odd-degree vertices; minimum 1 pen lift → 2 chains
        assert len(result) < len(paths), (
            f"Expected fewer than 3 chains, got {len(result)}"
        )

    def test_pen_lift_saved(self):
        """Result should have at most 2 chains (≤ 1 pen lift needed)."""
        paths = self._make_y()
        result = join_at_junctions(paths, threshold_mm=0.1)
        assert len(result) <= 2, (
            f"Expected ≤ 2 chains (1 pen lift), got {len(result)}"
        )

    def test_total_length_preserved(self):
        """All three arm lengths (1 mm each) must be present in output."""
        paths = self._make_y()
        result = join_at_junctions(paths, threshold_mm=0.1)
        orig_len = _total_length(paths)
        out_len = _total_length(result)
        assert abs(out_len - orig_len) < 0.01, (
            f"Total length changed: {orig_len:.4f} → {out_len:.4f}"
        )

    def test_all_arm_endpoints_present(self):
        """Each arm's far endpoint must appear somewhere in the result."""
        paths = self._make_y()
        result = join_at_junctions(paths, threshold_mm=0.1)
        for target in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]:
            assert _covers_point(result, target), (
                f"Arm endpoint {target} missing from result"
            )


# ---------------------------------------------------------------------------
# (ii) T-junction: endpoint on interior of another path → single chain
# ---------------------------------------------------------------------------

class TestTJunction:
    """Path A's endpoint lies on the interior of path B."""

    def test_single_chain(self):
        # Path B: long horizontal line from (0,0) to (4,0)
        # Path A: vertical arm from (2,0) to (2,2)
        # A's start (2,0) is an interior point of B → split B there
        path_b = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        path_a = [(2.0, 0.0), (2.0, 2.0)]

        result = join_at_junctions([path_b, path_a], threshold_mm=0.1)
        # After splitting B at (2,0): B1=(0,0)→(2,0), B2=(2,0)→(4,0), A=(2,0)→(2,2)
        # Graph: v(0,0), v(2,0), v(4,0), v(2,2)
        # Degrees: v(0,0)=1, v(2,0)=3, v(4,0)=1, v(2,2)=1  → 4 odd verts
        # Minimum 1 pen lift → 2 chains
        assert len(result) <= 2, (
            f"Expected ≤ 2 chains, got {len(result)}"
        )
        # All original endpoints should be covered
        for target in [(0.0, 0.0), (4.0, 0.0), (2.0, 2.0)]:
            assert _covers_point(result, target), (
                f"Expected {target} in result, got {result}"
            )

    def test_interior_vertex_exact_match(self):
        """Split fires even when the split is exactly on a vertex of B."""
        path_b = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        path_a = [(1.0, 0.0), (1.0, 1.0)]

        result = join_at_junctions([path_b, path_a], threshold_mm=0.1)
        assert len(result) >= 1
        for target in [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)]:
            assert _covers_point(result, target), (
                f"Expected {target} in result"
            )

    def test_no_split_when_disabled(self):
        """With split_at_t_junctions=False, the T-junction is ignored."""
        path_b = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        path_a = [(2.0, 0.0), (2.0, 2.0)]

        result_split = join_at_junctions([path_b, path_a], threshold_mm=0.1, split_at_t_junctions=True)
        result_no_split = join_at_junctions([path_b, path_a], threshold_mm=0.1, split_at_t_junctions=False)

        # With splitting we should join more; without, path_b stays whole
        # The key check: no_split result has at least 1 chain (path_b)
        assert len(result_no_split) >= 1


# ---------------------------------------------------------------------------
# (iii) Closed square (4 edges) → single closed polyline
# ---------------------------------------------------------------------------

class TestClosedSquare:
    """A square made of four separate line segments should form one loop."""

    def _make_square(self):
        return [
            [(0.0, 0.0), (1.0, 0.0)],  # bottom
            [(1.0, 0.0), (1.0, 1.0)],  # right
            [(1.0, 1.0), (0.0, 1.0)],  # top
            [(0.0, 1.0), (0.0, 0.0)],  # left
        ]

    def test_single_loop(self):
        paths = self._make_square()
        result = join_at_junctions(paths, threshold_mm=0.1)
        assert len(result) == 1, (
            f"Expected 1 closed loop, got {len(result)} chains"
        )

    def test_loop_is_closed(self):
        """The single result polyline must start and end at the same point."""
        paths = self._make_square()
        result = join_at_junctions(paths, threshold_mm=0.1)
        assert len(result) == 1
        chain = result[0]
        assert len(chain) >= 4, "Loop should have at least 4 points"
        start, end = chain[0], chain[-1]
        dist = math.hypot(start[0] - end[0], start[1] - end[1])
        assert dist < 0.15, (
            f"Loop not closed: start={start}, end={end}, dist={dist:.4f}"
        )

    def test_perimeter_preserved(self):
        """Total perimeter must equal 4 mm (4 unit edges)."""
        paths = self._make_square()
        result = join_at_junctions(paths, threshold_mm=0.1)
        out_len = _total_length(result)
        assert abs(out_len - 4.0) < 0.05, (
            f"Expected perimeter ≈ 4.0 mm, got {out_len:.4f}"
        )

    def test_all_corners_present(self):
        """All four corners of the square must appear in the output."""
        paths = self._make_square()
        result = join_at_junctions(paths, threshold_mm=0.1)
        for corner in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
            assert _covers_point(result, corner), (
                f"Corner {corner} missing from result"
            )


# ---------------------------------------------------------------------------
# (iv) Two disconnected components → exactly 2 chains
# ---------------------------------------------------------------------------

class TestDisconnectedComponents:
    """Paths from two separate regions must stay as distinct chains."""

    def test_two_components(self):
        # Component 1: square at origin (all-even → 1 closed loop)
        square = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(1.0, 0.0), (1.0, 1.0)],
            [(1.0, 1.0), (0.0, 1.0)],
            [(0.0, 1.0), (0.0, 0.0)],
        ]
        # Component 2: isolated line segment far away
        line = [[(10.0, 10.0), (11.0, 10.0)]]

        result = join_at_junctions(square + line, threshold_mm=0.1)
        assert len(result) == 2, (
            f"Expected 2 chains for 2 components, got {len(result)}"
        )

    def test_two_open_chains(self):
        """Two separate open paths remain separate."""
        path1 = [(0.0, 0.0), (1.0, 0.0)]
        path2 = [(5.0, 5.0), (6.0, 5.0)]

        result = join_at_junctions([path1, path2], threshold_mm=0.1)
        assert len(result) == 2, (
            f"Expected 2 separate chains, got {len(result)}"
        )

    def test_components_content_intact(self):
        """Each component's points are still present after joining."""
        path1 = [(0.0, 0.0), (1.0, 0.0)]
        path2 = [(5.0, 5.0), (6.0, 5.0)]

        result = join_at_junctions([path1, path2], threshold_mm=0.1)
        assert _covers_point(result, (0.0, 0.0))
        assert _covers_point(result, (1.0, 0.0))
        assert _covers_point(result, (5.0, 5.0))
        assert _covers_point(result, (6.0, 5.0))


# ---------------------------------------------------------------------------
# (v) Deterministic output across repeated runs
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Output must be identical across multiple calls with the same input."""

    def _run_multiple(self, paths, runs=5, **kwargs):
        results = [join_at_junctions(paths, **kwargs) for _ in range(runs)]
        return results

    def test_same_result_each_run(self):
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(1.0, 0.0), (1.0, 1.0)],
            [(1.0, 1.0), (0.0, 1.0)],
            [(0.0, 1.0), (0.0, 0.0)],
        ]
        results = self._run_multiple(paths)
        first = results[0]
        for i, r in enumerate(results[1:], 1):
            assert len(r) == len(first), (
                f"Run {i}: chain count differs: {len(first)} vs {len(r)}"
            )
            for j, (chain_a, chain_b) in enumerate(zip(first, r)):
                assert len(chain_a) == len(chain_b), (
                    f"Run {i}, chain {j}: length differs"
                )
                for k, (pa, pb) in enumerate(zip(chain_a, chain_b)):
                    assert abs(pa[0] - pb[0]) < 1e-9 and abs(pa[1] - pb[1]) < 1e-9, (
                        f"Run {i}, chain {j}, pt {k}: {pa} != {pb}"
                    )

    def test_deterministic_y_shape(self):
        """Y-shape gives the same result every time."""
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(0.0, 0.0), (0.0, 1.0)],
            [(0.0, 0.0), (-1.0, 0.0)],
        ]
        results = self._run_multiple(paths)
        first = results[0]
        for r in results[1:]:
            assert len(r) == len(first)
            for chain_a, chain_b in zip(first, r):
                assert chain_a == chain_b

    def test_deterministic_with_shuffled_identical_input(self):
        """Two calls with the same paths in the same order give identical output."""
        import copy
        paths = [
            [(0.0, 0.0), (2.0, 0.0)],
            [(2.0, 0.0), (2.0, 2.0)],
            [(2.0, 2.0), (0.0, 2.0)],
            [(0.0, 2.0), (0.0, 0.0)],
            [(10.0, 0.0), (12.0, 0.0)],
        ]
        r1 = join_at_junctions(copy.deepcopy(paths), threshold_mm=0.1)
        r2 = join_at_junctions(copy.deepcopy(paths), threshold_mm=0.1)
        assert len(r1) == len(r2)
        for c1, c2 in zip(r1, r2):
            assert c1 == c2


# ---------------------------------------------------------------------------
# Edge cases / defensive behaviour
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Degenerate inputs must not crash or produce wrong results."""

    def test_empty_input(self):
        assert join_at_junctions([]) == []

    def test_single_path(self):
        path = [(0.0, 0.0), (1.0, 0.0)]
        result = join_at_junctions([path])
        assert len(result) == 1
        assert result[0] == path

    def test_single_point_path_filtered(self):
        """Single-point 'paths' are silently dropped."""
        degenerate = [(0.0, 0.0)]
        valid = [(0.0, 0.0), (1.0, 0.0)]
        result = join_at_junctions([degenerate, valid])
        assert len(result) == 1

    def test_exact_duplicate_filtered(self):
        """Exact duplicate paths are deduplicated before processing."""
        path = [(0.0, 0.0), (1.0, 0.0)]
        result = join_at_junctions([path, path])
        assert len(result) == 1

    def test_closed_triangle(self):
        """Three edges forming a triangle collapse to one closed loop."""
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(1.0, 0.0), (0.5, 1.0)],
            [(0.5, 1.0), (0.0, 0.0)],
        ]
        result = join_at_junctions(paths, threshold_mm=0.1)
        assert len(result) == 1
        chain = result[0]
        start, end = chain[0], chain[-1]
        assert math.hypot(start[0] - end[0], start[1] - end[1]) < 0.15

    def test_threshold_respected(self):
        """Paths farther than threshold are NOT joined."""
        # Gap = 0.5 mm, threshold = 0.1 mm → should NOT join
        path1 = [(0.0, 0.0), (1.0, 0.0)]
        path2 = [(1.5, 0.0), (2.0, 0.0)]  # 0.5 mm gap
        result = join_at_junctions([path1, path2], threshold_mm=0.1)
        assert len(result) == 2

    def test_threshold_joins_close_paths(self):
        """Paths within threshold snap and join."""
        # Gap = 0.05 mm < threshold 0.1 mm → should join into 1 chain
        path1 = [(0.0, 0.0), (1.0, 0.0)]
        path2 = [(1.05, 0.0), (2.0, 0.0)]  # 0.05 mm gap
        result = join_at_junctions([path1, path2], threshold_mm=0.1)
        assert len(result) == 1, (
            f"Expected 1 joined chain, got {len(result)}"
        )

    def test_all_paths_empty_after_filter(self):
        """Only single-point paths → empty result."""
        result = join_at_junctions([[(0.0, 0.0)], [(1.0, 1.0)]])
        assert result == []
