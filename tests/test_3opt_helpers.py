"""Tests for 3-opt helper functions in processing/optimize.py."""

import math
import pytest

from plottter.processing.optimize import (
    _3opt_cost,
    _3opt_reconnect,
    _build_3opt_neighbors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seg(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    """Create a two-point path from (x0,y0) to (x1,y1)."""
    return [(x0, y0), (x1, y1)]


def make_paths(n: int = 10) -> list[list[tuple[float, float]]]:
    """Make n simple horizontal paths spaced 10mm apart vertically."""
    return [[(float(i * 10), 0.0), (float(i * 10 + 5), 0.0)] for i in range(n)]


# ---------------------------------------------------------------------------
# _3opt_cost
# ---------------------------------------------------------------------------

class Test3optCost:
    def test_simple_known_distances(self) -> None:
        # paths: p0=(0,0)→(1,0), p1=(2,0)→(3,0), p2=(5,0)→(6,0), p3=(10,0)→(11,0)
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(2.0, 0.0), (3.0, 0.0)],
            [(5.0, 0.0), (6.0, 0.0)],
            [(10.0, 0.0), (11.0, 0.0)],
        ]
        # cost = dist(end[0], start[1]) + dist(end[1], start[2]) + dist(end[2], start[3])
        #      = dist((1,0),(2,0)) + dist((3,0),(5,0)) + dist((6,0),(10,0))
        #      = 1.0 + 2.0 + 4.0 = 7.0
        cost = _3opt_cost(paths, 0, 1, 2)
        assert math.isclose(cost, 7.0, rel_tol=1e-9)

    def test_vertical_distances(self) -> None:
        paths = [
            [(0.0, 0.0), (0.0, 3.0)],
            [(0.0, 7.0), (0.0, 10.0)],
            [(0.0, 14.0), (0.0, 20.0)],
        ]
        # dist(end[0]=3, start[1]=7) = 4
        # dist(end[1]=10, start[2]=14) = 4
        # dist(end[2]=20, start[k+1]) where k=2 is last index → 0 (no next path)
        cost = _3opt_cost(paths, 0, 1, 2)
        assert math.isclose(cost, 8.0, rel_tol=1e-9)

    def test_diagonal_distance(self) -> None:
        paths = [
            [(0.0, 0.0), (0.0, 0.0)],
            [(3.0, 4.0), (3.0, 4.0)],
            [(6.0, 8.0), (6.0, 8.0)],
            [(9.0, 12.0), (9.0, 12.0)],
        ]
        # dist((0,0),(3,4)) = 5, dist((3,4),(6,8)) = 5, dist((6,8),(9,12)) = 5
        cost = _3opt_cost(paths, 0, 1, 2)
        assert math.isclose(cost, 15.0, rel_tol=1e-9)

    def test_k_at_last_index_gives_zero_for_third_edge(self) -> None:
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(3.0, 0.0), (4.0, 0.0)],
            [(6.0, 0.0), (7.0, 0.0)],
        ]
        # k=2 (last index), so edge3 = 0
        cost = _3opt_cost(paths, 0, 1, 2)
        # dist(end[0]=(1,0), start[1]=(3,0)) = 2
        # dist(end[1]=(4,0), start[2]=(6,0)) = 2
        # k is last index → 0
        assert math.isclose(cost, 4.0, rel_tol=1e-9)

    def test_returns_float(self) -> None:
        paths = make_paths(5)
        result = _3opt_cost(paths, 0, 1, 2)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _3opt_reconnect
# ---------------------------------------------------------------------------

class Test3optReconnect:
    """Tests for all 5 move IDs of _3opt_reconnect."""

    def setup_method(self) -> None:
        # 8 paths labelled p0..p7 for easy identification by their start point
        # Each path: [(x, 0), (x+1, 0)] so path i starts at (i*10, 0)
        self.paths = [
            [(float(i * 10), 0.0), (float(i * 10 + 1), 0.0)]
            for i in range(8)
        ]
        self.n = 8
        # Cut points: i=1, j=3, k=5
        # A = paths[0..1] = [p0, p1]
        # B = paths[2..3] = [p2, p3]
        # C = paths[4..5] = [p4, p5]
        # D = paths[6..7] = [p6, p7]
        self.i, self.j, self.k = 1, 3, 5

    def _reconnect(self, move_id: int) -> list:
        return _3opt_reconnect(self.paths, self.i, self.j, self.k, move_id)

    def test_all_moves_preserve_path_count(self) -> None:
        for move_id in range(5):
            result = self._reconnect(move_id)
            assert len(result) == self.n, f"move {move_id}: expected {self.n} paths, got {len(result)}"

    def test_all_moves_preserve_total_points(self) -> None:
        total_pts = sum(len(p) for p in self.paths)
        for move_id in range(5):
            result = self._reconnect(move_id)
            got = sum(len(p) for p in result)
            assert got == total_pts, f"move {move_id}: total points mismatch"

    def test_move0_reverses_B(self) -> None:
        # move 0: A + B_rev + C + D
        result = self._reconnect(0)
        A = self.paths[:2]
        B = self.paths[2:4]
        C = self.paths[4:6]
        D = self.paths[6:]
        B_rev = [p[::-1] for p in reversed(B)]
        expected = A + B_rev + C + D
        assert result == expected

    def test_move1_reverses_C(self) -> None:
        # move 1: A + B + C_rev + D
        result = self._reconnect(1)
        A = self.paths[:2]
        B = self.paths[2:4]
        C = self.paths[4:6]
        D = self.paths[6:]
        C_rev = [p[::-1] for p in reversed(C)]
        expected = A + B + C_rev + D
        assert result == expected

    def test_move2_reverses_both(self) -> None:
        # move 2: A + B_rev + C_rev + D
        result = self._reconnect(2)
        A = self.paths[:2]
        B = self.paths[2:4]
        C = self.paths[4:6]
        D = self.paths[6:]
        B_rev = [p[::-1] for p in reversed(B)]
        C_rev = [p[::-1] for p in reversed(C)]
        expected = A + B_rev + C_rev + D
        assert result == expected

    def test_move3_swaps_BC(self) -> None:
        # move 3: A + C + B + D
        result = self._reconnect(3)
        A = self.paths[:2]
        B = self.paths[2:4]
        C = self.paths[4:6]
        D = self.paths[6:]
        expected = A + C + B + D
        assert result == expected

    def test_move4_swaps_and_reverses(self) -> None:
        # move 4: A + C_rev + B_rev + D
        result = self._reconnect(4)
        A = self.paths[:2]
        B = self.paths[2:4]
        C = self.paths[4:6]
        D = self.paths[6:]
        B_rev = [p[::-1] for p in reversed(B)]
        C_rev = [p[::-1] for p in reversed(C)]
        expected = A + C_rev + B_rev + D
        assert result == expected

    def test_move3_is_distinct_from_move4(self) -> None:
        result3 = self._reconnect(3)
        result4 = self._reconnect(4)
        assert result3 != result4

    def test_invalid_move_id_raises(self) -> None:
        with pytest.raises(ValueError):
            _3opt_reconnect(self.paths, 1, 3, 5, 5)

    def test_single_path_segments(self) -> None:
        # Each segment has exactly 1 path: i=0, j=1, k=2
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],
            [(2.0, 0.0), (3.0, 0.0)],
            [(4.0, 0.0), (5.0, 0.0)],
            [(6.0, 0.0), (7.0, 0.0)],
        ]
        for move_id in range(5):
            result = _3opt_reconnect(paths, 0, 1, 2, move_id)
            assert len(result) == 4

    def test_move0_reversal_flips_path_points(self) -> None:
        # Verify individual path points are flipped when segment is reversed
        paths = [
            [(0.0, 0.0), (1.0, 0.0)],        # A
            [(10.0, 0.0), (11.0, 5.0)],      # B
            [(20.0, 0.0), (21.0, 0.0)],      # C
        ]
        result = _3opt_reconnect(paths, 0, 1, 2, 0)
        # move 0: A + B_rev + C
        # B_rev = [paths[1][::-1]] = [[(11.0, 5.0), (10.0, 0.0)]]
        assert result[1] == [(11.0, 5.0), (10.0, 0.0)]


# ---------------------------------------------------------------------------
# _build_3opt_neighbors
# ---------------------------------------------------------------------------

class TestBuild3optNeighbors:
    def test_returns_k_neighbors_per_path(self) -> None:
        paths = make_paths(20)
        k = 5
        neighbors = _build_3opt_neighbors(paths, k=k)
        assert len(neighbors) == 20
        for i, nbrs in enumerate(neighbors):
            assert len(nbrs) == k, f"path {i}: expected {k} neighbors, got {len(nbrs)}"

    def test_no_self_reference(self) -> None:
        paths = make_paths(20)
        neighbors = _build_3opt_neighbors(paths, k=5)
        for i, nbrs in enumerate(neighbors):
            assert i not in nbrs, f"path {i} references itself"

    def test_default_k(self) -> None:
        paths = make_paths(30)
        neighbors = _build_3opt_neighbors(paths)
        assert len(neighbors) == 30
        for nbrs in neighbors:
            assert len(nbrs) == 15

    def test_k_capped_at_n_minus_1(self) -> None:
        # Only 5 paths, requesting k=20 → each path gets at most 4 neighbors
        paths = make_paths(5)
        neighbors = _build_3opt_neighbors(paths, k=20)
        for nbrs in neighbors:
            assert len(nbrs) == 4

    def test_neighbor_indices_valid(self) -> None:
        n = 15
        paths = make_paths(n)
        neighbors = _build_3opt_neighbors(paths, k=5)
        for nbrs in neighbors:
            for idx in nbrs:
                assert 0 <= idx < n

    def test_spatial_locality(self) -> None:
        # Paths clustered in two groups; neighbors should prefer same cluster
        group_a = [[(float(i), 0.0), (float(i) + 0.5, 0.0)] for i in range(10)]
        group_b = [[(float(i) + 1000.0, 0.0), (float(i) + 1000.5, 0.0)] for i in range(10)]
        paths = group_a + group_b
        neighbors = _build_3opt_neighbors(paths, k=5)
        # Neighbors of path 0 (group A) should all be in group A (indices 0-9)
        for idx in neighbors[0]:
            assert idx < 10, f"path 0 neighbor {idx} is in group B"
        # Neighbors of path 10 (group B) should all be in group B (indices 10-19)
        for idx in neighbors[10]:
            assert idx >= 10, f"path 10 neighbor {idx} is in group A"

    def test_returns_list_of_lists_of_ints(self) -> None:
        paths = make_paths(10)
        neighbors = _build_3opt_neighbors(paths, k=3)
        assert isinstance(neighbors, list)
        for nbrs in neighbors:
            assert isinstance(nbrs, list)
            for idx in nbrs:
                assert isinstance(idx, int)
