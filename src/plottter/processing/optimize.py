"""Path reordering optimizers: nearest-neighbor + 2-opt + Or-opt improvement."""

import math
from collections.abc import Callable
import numpy as np
from scipy.spatial import cKDTree
from plottter.models.path import Polyline, Point


# ---------------------------------------------------------------------------
# Travel distance helpers
# ---------------------------------------------------------------------------

def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def calculate_travel_distance(paths: list[Polyline]) -> float:
    """Return total pen-up travel distance (mm) for the given path ordering.

    Travel is measured as:
    - Distance from origin (0, 0) to the start of the first path.
    - Between each consecutive pair of paths: end of path[i] → start of path[i+1].
    - Distance from the end of the last path back to origin.

    Args:
        paths: Ordered list of polylines.

    Returns:
        Total pen-up travel distance in mm.
    """
    if not paths:
        return 0.0

    origin: Point = (0.0, 0.0)
    total = _dist(origin, paths[0][0])

    for i in range(len(paths) - 1):
        total += _dist(paths[i][-1], paths[i + 1][0])

    total += _dist(paths[-1][-1], origin)
    return total


# ---------------------------------------------------------------------------
# Nearest-neighbor reordering
# ---------------------------------------------------------------------------

_NN_KDTREE_THRESHOLD = 50  # Use brute-force for small inputs


def _reorder_paths_brute_from_seed(
    valid: list[list[Point]],
    seed_idx: int,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    progress_offset: float = 0.0,
    progress_scale: float = 1.0,
) -> list[list[Point]]:
    """Brute-force nearest-neighbour reorder starting from *seed_idx*."""
    n = len(valid)
    unvisited = list(range(n))
    result: list[list[Point]] = []

    unvisited.remove(seed_idx)
    result.append(valid[seed_idx])
    current_pos: Point = valid[seed_idx][-1]
    _report_interval = max(1, n // 20)

    while unvisited:
        if cancelled and cancelled():
            # Return partial + remaining paths to preserve all paths
            for idx in unvisited:
                result.append(valid[idx])
            return result

        n_placed = len(result)
        if progress_callback and n_placed % _report_interval == 0:
            progress_callback(progress_offset + progress_scale * n_placed / n)

        best_dist = float("inf")
        best_i = -1
        best_reversed = False

        for idx in unvisited:
            d_start = _dist(current_pos, valid[idx][0])
            d_end = _dist(current_pos, valid[idx][-1])

            if d_start <= d_end:
                if d_start < best_dist:
                    best_dist = d_start
                    best_i = idx
                    best_reversed = False
            else:
                if d_end < best_dist:
                    best_dist = d_end
                    best_i = idx
                    best_reversed = True

        path = valid[best_i]
        if best_reversed:
            path = path[::-1]

        result.append(path)
        current_pos = path[-1]
        unvisited.remove(best_i)

    return result


def _reorder_paths_brute(
    valid: list[list[Point]],
    num_starts: int = 1,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[list[Point]]:
    """Brute-force nearest-neighbour reorder (used for small inputs).

    When *num_starts* > 1, runs NN from multiple seed paths and keeps the
    result with the shortest total pen-up travel distance.
    """
    n = len(valid)
    origin: Point = (0.0, 0.0)

    # First seed: path whose start is closest to origin (current behaviour)
    origin_seed = min(range(n), key=lambda i: _dist(origin, valid[i][0]))

    seeds: list[int] = [origin_seed]
    seen: set[int] = {origin_seed}

    # Additional seeds: evenly spaced indices through the valid list
    for k in range(1, num_starts):
        idx = k * n // num_starts
        if idx not in seen:
            seeds.append(idx)
            seen.add(idx)

    n_seeds = len(seeds)

    if n_seeds == 1:
        return _reorder_paths_brute_from_seed(
            valid, seeds[0],
            progress_callback=progress_callback,
            cancelled=cancelled,
            progress_offset=0.0,
            progress_scale=1.0,
        )

    best_result: list[list[Point]] = []
    best_dist = float("inf")
    for seed_i, seed_idx in enumerate(seeds):
        if cancelled and cancelled():
            return best_result if best_result else _reorder_paths_brute_from_seed(valid, seed_idx, cancelled=cancelled)
        result = _reorder_paths_brute_from_seed(
            valid, seed_idx,
            progress_callback=progress_callback,
            cancelled=cancelled,
            progress_offset=seed_i / n_seeds,
            progress_scale=1.0 / n_seeds,
        )
        if progress_callback:
            progress_callback((seed_i + 1) / n_seeds)
        dist = calculate_travel_distance(result)
        if dist < best_dist:
            best_dist = dist
            best_result = result

    return best_result


def _reorder_paths_kdtree_from_seed(
    valid: list[list[Point]],
    endpoints: "np.ndarray",
    tree: "cKDTree",
    seed_idx: int,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    progress_offset: float = 0.0,
    progress_scale: float = 1.0,
) -> list[list[Point]]:
    """KD-tree NN reorder starting from *seed_idx*."""
    n = len(valid)
    consumed = [False] * n
    result: list[list[Point]] = []

    consumed[seed_idx] = True
    result.append(valid[seed_idx])
    current_pos = np.array(
        [valid[seed_idx][-1][0], valid[seed_idx][-1][1]], dtype=np.float64
    )

    remaining = n - 1
    k = min(2 * n, max(16, n // 10 + 16))
    _report_interval = max(1, n // 20)

    while remaining > 0:
        if cancelled and cancelled():
            # Append unconsumed paths to preserve all paths
            for pi in range(n):
                if not consumed[pi]:
                    result.append(valid[pi])
            return result

        n_placed = len(result)
        if progress_callback and n_placed % _report_interval == 0:
            progress_callback(progress_offset + progress_scale * n_placed / n)

        _, near_indices = tree.query(current_pos.reshape(1, 2), k=k)
        near_indices = near_indices[0]

        chosen_pi = -1
        chosen_reversed = False

        for idx in near_indices:
            pi = idx // 2
            if consumed[pi]:
                continue
            chosen_pi = pi
            chosen_reversed = idx % 2 == 1
            break

        if chosen_pi == -1:
            # All k candidates exhausted — full linear scan (rare).
            best_d = float("inf")
            for pi in range(n):
                if consumed[pi]:
                    continue
                d_start = math.hypot(
                    current_pos[0] - valid[pi][0][0],
                    current_pos[1] - valid[pi][0][1],
                )
                d_end = math.hypot(
                    current_pos[0] - valid[pi][-1][0],
                    current_pos[1] - valid[pi][-1][1],
                )
                d = min(d_start, d_end)
                if d < best_d:
                    best_d = d
                    chosen_pi = pi
                    chosen_reversed = d_end < d_start

        path = valid[chosen_pi]
        if chosen_reversed:
            path = path[::-1]

        result.append(path)
        consumed[chosen_pi] = True
        current_pos[0] = path[-1][0]
        current_pos[1] = path[-1][1]
        remaining -= 1

    return result


def _reorder_paths_kdtree(
    valid: list[list[Point]],
    num_starts: int = 1,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[list[Point]]:
    """KD-tree nearest-neighbour reorder (used for large inputs).

    When *num_starts* > 1, runs NN from multiple seed paths and keeps the
    result with the shortest total pen-up travel distance.
    """
    n = len(valid)
    # Build endpoint array shared across all starts:
    # index 2*i = start of path i, 2*i+1 = end of path i
    endpoints = np.empty((2 * n, 2), dtype=np.float64)
    for i, path in enumerate(valid):
        endpoints[2 * i, 0] = path[0][0]
        endpoints[2 * i, 1] = path[0][1]
        endpoints[2 * i + 1, 0] = path[-1][0]
        endpoints[2 * i + 1, 1] = path[-1][1]

    tree = cKDTree(endpoints)

    # First seed: path whose start is closest to origin — query the tree
    origin_arr = np.array([[0.0, 0.0]], dtype=np.float64)
    k_seed = min(2 * n, max(10, n // 10))
    _, seed_indices = tree.query(origin_arr, k=k_seed)
    seed_indices = seed_indices[0]

    origin_seed = -1
    for idx in seed_indices:
        pi = idx // 2
        if idx % 2 == 0:  # prefer start endpoints
            origin_seed = pi
            break
    if origin_seed == -1:
        origin_seed = seed_indices[0] // 2 if len(seed_indices) else 0

    seeds: list[int] = [origin_seed]
    seen: set[int] = {origin_seed}

    # Additional seeds: evenly spaced indices through the valid list
    for k in range(1, num_starts):
        idx = k * n // num_starts
        if idx not in seen:
            seeds.append(idx)
            seen.add(idx)

    n_seeds = len(seeds)

    if n_seeds == 1:
        return _reorder_paths_kdtree_from_seed(
            valid, endpoints, tree, seeds[0],
            progress_callback=progress_callback,
            cancelled=cancelled,
            progress_offset=0.0,
            progress_scale=1.0,
        )

    best_result: list[list[Point]] = []
    best_dist = float("inf")
    for seed_i, seed_idx in enumerate(seeds):
        if cancelled and cancelled():
            return best_result if best_result else _reorder_paths_kdtree_from_seed(
                valid, endpoints, tree, seed_idx, cancelled=cancelled
            )
        result = _reorder_paths_kdtree_from_seed(
            valid, endpoints, tree, seed_idx,
            progress_callback=progress_callback,
            cancelled=cancelled,
            progress_offset=seed_i / n_seeds,
            progress_scale=1.0 / n_seeds,
        )
        if progress_callback:
            progress_callback((seed_i + 1) / n_seeds)
        dist = calculate_travel_distance(result)
        if dist < best_dist:
            best_dist = dist
            best_result = result

    return best_result


def reorder_paths(
    paths: list[Polyline],
    num_starts: int = 5,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Polyline]:
    """Reorder paths to minimise pen-up travel using a nearest-neighbour heuristic.

    Uses a ``scipy.spatial.cKDTree`` for O(n log n) endpoint lookups when the
    input is large enough to benefit (≥ ``_NN_KDTREE_THRESHOLD`` paths).  Falls
    back to a brute-force O(n²) scan for small inputs where the KD-tree overhead
    would dominate.

    When *num_starts* > 1 (default 5), the NN algorithm is run from multiple
    seed paths and the result with the shortest pen-up travel distance is kept.
    The first seed is the path closest to the origin (greedy-NN baseline).
    Additional seeds are paths at evenly spaced indices in the input list,
    providing geographic diversity without randomness.

    Algorithm (per seed):
    1. Start with the seed path.
    2. After completing a path, find the nearest unvisited endpoint (start *or*
       end of every remaining path).
    3. If the nearest endpoint is the **end** of a candidate path, reverse that
       path so that its start aligns with our current position.
    4. Repeat until all paths are included.

    Args:
        paths: Input list of polylines. Empty paths are skipped.
        num_starts: Number of seed paths to try. Default 5.
        progress_callback: Optional callable receiving a float in [0.0, 1.0]
            as the algorithm progresses.  Called periodically; may be invoked
            from a background thread.
        cancelled: Optional callable that returns ``True`` when the caller
            wants to abort.  Checked at each outer iteration; a partially
            ordered result (all paths present) is returned if cancelled.

    Returns:
        New list of polylines in optimised order (some may be reversed).
    """
    if not paths:
        return []

    # Filter out degenerate paths
    valid = [list(p) for p in paths if len(p) >= 2]
    if not valid:
        return [list(p) for p in paths]

    if len(valid) < _NN_KDTREE_THRESHOLD:
        return _reorder_paths_brute(
            valid,
            num_starts=num_starts,
            progress_callback=progress_callback,
            cancelled=cancelled,
        )
    return _reorder_paths_kdtree(
        valid,
        num_starts=num_starts,
        progress_callback=progress_callback,
        cancelled=cancelled,
    )


# ---------------------------------------------------------------------------
# 2-opt improvement
# ---------------------------------------------------------------------------

_2OPT_KDTREE_THRESHOLD = 50  # Use brute-force for small inputs
_2OPT_NEIGHBOR_K = 20        # Number of nearest neighbors per path in the list


def _build_2opt_neighbors(route: list[list[Point]], k: int) -> list[list[int]]:
    """Build a spatial neighbor list for 2-opt using path end points.

    For each position *i* in *route*, returns the indices of the *k* closest
    positions (by end-point Euclidean distance).  These are the candidate *j*
    values checked in the 2-opt inner loop.

    Args:
        route: Current route (ordered list of polylines).
        k: Number of neighbors to compute per position.

    Returns:
        List of length n; entry i is a list of up to k position indices.
    """
    n = len(route)
    k = min(k, n - 1)

    end_pts = np.empty((n, 2), dtype=np.float64)
    for i, path in enumerate(route):
        end_pts[i, 0] = path[-1][0]
        end_pts[i, 1] = path[-1][1]

    tree = cKDTree(end_pts)
    _, indices = tree.query(end_pts, k=min(k + 1, n))  # +1 to filter self

    neighbors: list[list[int]] = []
    for i in range(n):
        nbrs = [int(idx) for idx in indices[i] if idx != i]
        neighbors.append(nbrs[:k])

    return neighbors


def optimize_2opt(
    paths: list[Polyline],
    max_iterations: int = 1000,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Polyline]:
    """Improve a nearest-neighbour ordering with 2-opt swaps.

    For small inputs (fewer than ``_2OPT_KDTREE_THRESHOLD`` paths) the
    original O(n²) brute-force scan is used unchanged.  For larger inputs a
    spatial neighbor list (KD-tree over path end points) restricts the inner
    loop to the *k* spatially nearest paths, reducing per-pass work from
    O(n²) to O(n × k).  The neighbor list is rebuilt every
    ``min(100, n // 10)`` iterations so it stays reasonably fresh as the
    route changes.

    The iteration budget adapts to input size:
    ``max_iters = max(max_iterations, n × 2)`` so larger inputs receive
    proportionally more passes.

    Args:
        paths: Ordered list of polylines (ideally already NN-ordered).
        max_iterations: Minimum number of improvement passes.  The actual
            limit is ``max(max_iterations, len(paths) * 2)``.
        progress_callback: Optional callable receiving a float in [0.0, 1.0]
            as the algorithm progresses.  Called at the start of each outer
            iteration.
        cancelled: Optional callable that returns ``True`` when the caller
            wants to abort.  Checked at the start of each outer iteration;
            the current best route is returned immediately.

    Returns:
        New list of polylines in the improved order (some may be reversed
        relative to the input).
    """
    if len(paths) < 3:
        return list(paths)

    route = [list(p) for p in paths]
    n = len(route)

    # Adaptive iteration limit
    max_iters = max(max_iterations, n * 2)

    if n < _2OPT_KDTREE_THRESHOLD:
        # ---- Brute-force O(n²) pass (small inputs, identical to old code) --
        for iteration in range(max_iters):
            if cancelled and cancelled():
                break
            if progress_callback:
                progress_callback(iteration / max_iters)

            improved = False

            for i in range(n - 1):
                for j in range(i + 2, n):
                    # Current cost of edges: i→i+1 and j→j+1
                    end_i = route[i][-1]
                    start_i1 = route[i + 1][0]
                    end_j = route[j][-1]
                    start_j1 = route[j + 1][0] if j + 1 < n else (0.0, 0.0)

                    before = _dist(end_i, start_i1) + _dist(end_j, start_j1)

                    # After reversing segment [i+1 .. j]:
                    after = _dist(end_i, end_j) + _dist(start_i1, start_j1)

                    if after < before - 1e-9:
                        segment = route[i + 1 : j + 1]
                        segment.reverse()
                        for seg_k in range(len(segment)):
                            segment[seg_k] = segment[seg_k][::-1]
                        route[i + 1 : j + 1] = segment
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return route

    # ---- Neighbor-list O(n × k) pass (large inputs) ----------------------
    k_neighbors = min(_2OPT_NEIGHBOR_K, n - 1)
    neighbors = _build_2opt_neighbors(route, k_neighbors)
    # Rebuild the neighbor list every this many outer iterations
    rebuild_interval = min(100, max(1, n // 10))

    for iteration in range(max_iters):
        if cancelled and cancelled():
            break
        if progress_callback:
            progress_callback(iteration / max_iters)

        # Rebuild periodically so the list stays fresh after route changes
        if iteration > 0 and iteration % rebuild_interval == 0:
            neighbors = _build_2opt_neighbors(route, k_neighbors)

        improved = False

        for i in range(n - 1):
            for j in neighbors[i]:
                j = int(j)
                if j <= i + 1 or j >= n:
                    continue  # 2-opt requires j > i+1

                end_i = route[i][-1]
                start_i1 = route[i + 1][0]
                end_j = route[j][-1]
                start_j1 = route[j + 1][0] if j + 1 < n else (0.0, 0.0)

                before = _dist(end_i, start_i1) + _dist(end_j, start_j1)
                after = _dist(end_i, end_j) + _dist(start_i1, start_j1)

                if after < before - 1e-9:
                    segment = route[i + 1 : j + 1]
                    segment.reverse()
                    for seg_k in range(len(segment)):
                        segment[seg_k] = segment[seg_k][::-1]
                    route[i + 1 : j + 1] = segment
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return route


# ---------------------------------------------------------------------------
# Or-opt improvement
# ---------------------------------------------------------------------------

_OR_OPT_KDTREE_THRESHOLD = 50  # Use brute-force for small inputs
_OR_OPT_NEIGHBOR_K = 20        # Number of nearest neighbors per path


def _build_or_opt_neighbors(route: list[list[Point]], k: int) -> list[list[int]]:
    """Build a spatial neighbor list for Or-opt using path midpoints.

    For each position *i* in *route*, returns the indices of the *k* closest
    positions (by midpoint Euclidean distance).  These are the candidate
    insertion positions tested in the Or-opt inner loop.

    Args:
        route: Current route (ordered list of polylines).
        k: Number of neighbors to compute per position.

    Returns:
        List of length n; entry i is a list of up to k position indices.
    """
    n = len(route)
    k = min(k, n - 1)

    mids = np.empty((n, 2), dtype=np.float64)
    for i, path in enumerate(route):
        mids[i, 0] = (path[0][0] + path[-1][0]) / 2.0
        mids[i, 1] = (path[0][1] + path[-1][1]) / 2.0

    tree = cKDTree(mids)
    _, indices = tree.query(mids, k=min(k + 1, n))

    neighbors: list[list[int]] = []
    for i in range(n):
        nbrs = [int(idx) for idx in indices[i] if idx != i]
        neighbors.append(nbrs[:k])

    return neighbors


def optimize_or_opt(
    paths: list[Polyline],
    max_iterations: int = 500,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Polyline]:
    """Apply Or-opt improvement to a path ordering.

    Or-opt tries relocating short subsequences (1, 2, or 3 consecutive paths)
    to a better position in the route.  It catches improvements that 2-opt
    cannot — for example, a single path that is in completely the wrong part
    of the sequence.

    Uses "first improvement" strategy: accepts the first move that reduces
    total travel distance, then restarts.  For large inputs
    (≥ ``_OR_OPT_KDTREE_THRESHOLD`` paths) a spatial neighbor list (KD-tree
    over path midpoints) restricts reinsertion candidates to spatially nearby
    paths, reducing per-pass work from O(n²) to O(n × k).  The neighbor list
    is rebuilt immediately after each accepted move.

    The iteration budget adapts to input size:
    ``max_iters = max(max_iterations, len(paths))``.

    Args:
        paths: Ordered list of polylines (ideally already NN + 2-opt ordered).
        max_iterations: Minimum number of improvement passes.  The actual
            limit is ``max(max_iterations, len(paths))``.
        progress_callback: Optional callable receiving a float in [0.0, 1.0]
            as the algorithm progresses.  Called at the start of each outer
            iteration.
        cancelled: Optional callable that returns ``True`` when the caller
            wants to abort.  Checked at the start of each outer iteration;
            the current best route is returned immediately.

    Returns:
        New list of polylines in improved order (paths not modified internally,
        only reordered).
    """
    if len(paths) < 4:
        return list(paths)

    route = [list(p) for p in paths]
    n = len(route)
    max_iters = max(max_iterations, n)
    origin: Point = (0.0, 0.0)
    use_neighbors = n >= _OR_OPT_KDTREE_THRESHOLD

    if use_neighbors:
        k_neighbors = min(_OR_OPT_NEIGHBOR_K, n - 1)
        neighbors = _build_or_opt_neighbors(route, k_neighbors)

    for _iteration in range(max_iters):
        if cancelled and cancelled():
            break
        if progress_callback:
            progress_callback(_iteration / max_iters)

        improved = False

        for seg_len in (1, 2, 3):
            if n < seg_len + 2:
                continue

            for i in range(n - seg_len + 1):
                seg = route[i : i + seg_len]
                seg_first_start: Point = seg[0][0]
                seg_last_end: Point = seg[-1][-1]

                prev_end: Point = route[i - 1][-1] if i > 0 else origin
                after_seg: Point = (
                    route[i + seg_len][0] if i + seg_len < n else origin
                )

                # Saving achieved by closing the gap left when the segment is removed
                gap_save = (
                    _dist(prev_end, seg_first_start)
                    + _dist(seg_last_end, after_seg)
                    - _dist(prev_end, after_seg)
                )

                # Candidate insertion positions: insert the segment AFTER route[j]
                candidates = neighbors[i] if use_neighbors else range(n)  # type: ignore[assignment]

                for j_raw in candidates:
                    j = int(j_raw)
                    if j >= n:
                        continue
                    # Skip forbidden positions: within or immediately before the segment
                    if i <= j <= i + seg_len - 1:
                        continue
                    if i > 0 and j == i - 1:
                        continue  # no-op: segment would stay in the same place

                    after_j: Point = route[j][-1]
                    next_j: Point = route[j + 1][0] if j + 1 < n else origin

                    insert_cost = (
                        _dist(after_j, seg_first_start)
                        + _dist(seg_last_end, next_j)
                        - _dist(after_j, next_j)
                    )
                    gain = gap_save - insert_cost

                    if gain > 1e-9:
                        # Remove segment from position i, reinsert after j_adj
                        j_adj = j if j < i else j - seg_len
                        tmp = route[:i] + route[i + seg_len :]
                        route = tmp[: j_adj + 1] + seg + tmp[j_adj + 1 :]
                        n = len(route)
                        if use_neighbors:
                            neighbors = _build_or_opt_neighbors(route, k_neighbors)
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return route


# ---------------------------------------------------------------------------
# 3-opt helpers
# ---------------------------------------------------------------------------

_3OPT_NEIGHBOR_K = 15  # Default number of nearest neighbors for 3-opt


def _3opt_cost(paths: list[list[Point]], i: int, j: int, k: int) -> float:
    """Compute the cost of the three edges removed in a 3-opt move.

    Returns the sum of Euclidean distances for the three edges:
        dist(end[i], start[i+1]) + dist(end[j], start[j+1]) + dist(end[k], start[k+1])

    Args:
        paths: Current ordered list of polylines.
        i: Index of the first cut point.
        j: Index of the second cut point (i < j).
        k: Index of the third cut point (j < k).

    Returns:
        Sum of the three removed edge lengths.
    """
    n = len(paths)
    d1 = _dist(paths[i][-1], paths[i + 1][0]) if i + 1 < n else 0.0
    d2 = _dist(paths[j][-1], paths[j + 1][0]) if j + 1 < n else 0.0
    d3 = _dist(paths[k][-1], paths[k + 1][0]) if k + 1 < n else 0.0
    return d1 + d2 + d3


def _rev_seg(seg: list[list[Point]]) -> list[list[Point]]:
    """Reverse a segment: reverse order of paths and flip each path."""
    return [p[::-1] for p in reversed(seg)]


def _3opt_reconnect(
    paths: list[list[Point]], i: int, j: int, k: int, move_id: int
) -> list[list[Point]]:
    """Reconnect three segments after a 3-opt cut.

    Given cut points i < j < k, the route is split into four parts:
        A = paths[0..i], B = paths[i+1..j], C = paths[j+1..k], D = paths[k+1..]

    The five non-identity reconnection moves are:
        move 0: A + B_rev + C + D        (reverse segment B)
        move 1: A + B + C_rev + D        (reverse segment C)
        move 2: A + B_rev + C_rev + D    (reverse both B and C)
        move 3: A + C + B + D            (swap B and C, no reversal)
        move 4: A + C_rev + B_rev + D    (swap B and C, both reversed)

    Each reversal flips the order of paths within the segment and also
    reverses the point order within each individual path so that pen-up
    travel is computed correctly.

    Args:
        paths: Current ordered list of polylines.
        i: Index of the first cut point.
        j: Index of the second cut point (i < j).
        k: Index of the third cut point (j < k).
        move_id: Reconnection move identifier (0–4).

    Returns:
        New list of paths with the segments reconnected according to move_id.

    Raises:
        ValueError: If move_id is not in 0–4.
    """
    A = paths[: i + 1]
    B = paths[i + 1 : j + 1]
    C = paths[j + 1 : k + 1]
    D = paths[k + 1 :]

    if move_id == 0:
        return A + _rev_seg(B) + C + D
    elif move_id == 1:
        return A + B + _rev_seg(C) + D
    elif move_id == 2:
        return A + _rev_seg(B) + _rev_seg(C) + D
    elif move_id == 3:
        return A + C + B + D
    elif move_id == 4:
        return A + _rev_seg(C) + _rev_seg(B) + D
    else:
        raise ValueError(f"Invalid move_id {move_id!r}: must be 0-4")


def _build_3opt_neighbors(
    paths: list[list[Point]], k: int = _3OPT_NEIGHBOR_K
) -> list[list[int]]:
    """Build a spatial neighbor list for 3-opt using path midpoints.

    For each position *i* in *paths*, returns the indices of the *k* closest
    positions (by midpoint Euclidean distance).  Mirrors the pattern used by
    ``_build_or_opt_neighbors``.

    Args:
        paths: Current ordered list of polylines.
        k: Number of neighbors to compute per position.

    Returns:
        List of length n; entry i is a list of up to k position indices.
    """
    n = len(paths)
    k = min(k, n - 1)

    mids = np.empty((n, 2), dtype=np.float64)
    for i, path in enumerate(paths):
        mids[i, 0] = (path[0][0] + path[-1][0]) / 2.0
        mids[i, 1] = (path[0][1] + path[-1][1]) / 2.0

    tree = cKDTree(mids)
    _, indices = tree.query(mids, k=min(k + 1, n))

    neighbors: list[list[int]] = []
    for i in range(n):
        nbrs = [int(idx) for idx in indices[i] if idx != i]
        neighbors.append(nbrs[:k])

    return neighbors


def optimize_3opt(
    paths: list[Polyline],
    max_iterations: int = 500,
    progress_callback: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Polyline]:
    """Apply 3-opt improvement to a path ordering.

    For each triple of positions (i, j, k) with i < j < k, try all 5
    non-identity reconnection moves and accept the first one that reduces
    total 3-edge cost (greedy first-improvement).  Uses a spatial neighbor
    list to restrict the inner loops to spatially nearby paths.

    Iteration budget: ``max(max_iterations, len(paths))``.
    Neighbor list is rebuilt every ~5% of iterations.

    Args:
        paths: Ordered list of polylines.
        max_iterations: Minimum number of passes.
        progress_callback: Optional callable receiving a float in [0.0, 1.0].
        cancelled: Optional callable returning True when caller wants to abort.

    Returns:
        New list of polylines in improved order.
    """
    if len(paths) < 3:
        return list(paths)

    route = [list(p) for p in paths]
    n = len(route)
    max_iters = max(max_iterations, n)

    k_neighbors = min(_3OPT_NEIGHBOR_K, n - 1)
    neighbors = _build_3opt_neighbors(route, k_neighbors)

    # Rebuild neighbor list every ~5% of iterations (at least every 10)
    rebuild_interval = max(10, max_iters // 20)

    for iteration in range(max_iters):
        if cancelled and cancelled():
            break
        if progress_callback:
            progress_callback(iteration / max_iters)

        if iteration > 0 and iteration % rebuild_interval == 0:
            neighbors = _build_3opt_neighbors(route, k_neighbors)

        improved = False

        for i in range(n - 2):
            for j in neighbors[i]:
                j = int(j)
                if j <= i or j >= n - 1:
                    continue

                for k_idx in neighbors[j]:
                    k_idx = int(k_idx)
                    if k_idx <= j or k_idx >= n:
                        continue

                    current_cost = _3opt_cost(route, i, j, k_idx)

                    for move_id in range(5):
                        candidate = _3opt_reconnect(route, i, j, k_idx, move_id)
                        new_cost = _3opt_cost(candidate, i, j, k_idx)
                        if new_cost < current_cost - 1e-9:
                            route = candidate
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return route
