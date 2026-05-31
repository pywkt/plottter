"""Merge polylines whose endpoints are within a threshold distance."""

import numpy as np
from scipy.spatial import cKDTree

from plottter.models.path import Polyline
from plottter.processing._jit import njit


@njit(cache=True)
def _find_and_snap(
    qx,            # float: query endpoint x
    qy,            # float: query endpoint y
    cand_idxs,     # int64 ndarray: candidate endpoint indices (from cKDTree)
    endpoints,     # (2n, 2) float64 ndarray: all endpoints
    consumed,      # (n,) bool ndarray: which paths are already consumed
    own_idx,       # int: index of the path being extended (excluded from match)
    reverse_if_end,  # int (1 or 0): 1 → reverse j when matched idx is odd (extend-from-end),
                     #               0 → reverse j when matched idx is even (prepend-from-start)
):
    """Find the best merge candidate and precompute the snap midpoint.

    Iterates over *cand_idxs* (the output of ``cKDTree.query_ball_point``), skipping
    the path's own endpoints and already-consumed paths, then returns the index of
    the nearest compatible path together with the pre-computed midpoint for the gap
    snap.

    Returns a 5-tuple ``(best_j, best_reverse_j, mid_x, mid_y, coincident)``
    where *best_j* is ``-1`` when no candidate was found.
    *best_reverse_j* and *coincident* are returned as ``int`` (0 or 1) for numba
    type-inference stability.
    """
    best_j = -1
    best_dist_sq = 1e300
    best_reverse_j = 0
    best_idx = -1

    for k in range(len(cand_idxs)):
        idx = int(cand_idxs[k])
        j = idx >> 1          # path index = endpoint_index // 2
        if j == own_idx:
            continue
        if consumed[j]:
            continue
        ex = endpoints[idx, 0]
        ey = endpoints[idx, 1]
        dx = qx - ex
        dy = qy - ey
        dsq = dx * dx + dy * dy
        if dsq < best_dist_sq:
            best_dist_sq = dsq
            best_j = j
            # Determine whether j needs to be reversed before appending.
            # Phase A (extend from end, reverse_if_end=1):
            #   odd  idx → j's end matched → reverse j so its start aligns
            # Phase B (prepend to start, reverse_if_end=0):
            #   even idx → j's start matched → reverse j so its end aligns
            if reverse_if_end:
                best_reverse_j = 1 if (idx & 1) else 0
            else:
                best_reverse_j = 0 if (idx & 1) else 1
            best_idx = idx

    if best_j < 0:
        return -1, 0, 0.0, 0.0, 0

    jx = endpoints[best_idx, 0]
    jy = endpoints[best_idx, 1]
    # Coincident = already exactly touching; snap = midpoint of the gap.
    coincident = 1 if (abs(qx - jx) < 1e-9 and abs(qy - jy) < 1e-9) else 0
    mid_x = (qx + jx) * 0.5
    mid_y = (qy + jy) * 0.5
    return best_j, best_reverse_j, mid_x, mid_y, coincident


def merge_nearby_paths(paths: list[Polyline], threshold_mm: float = 0.5) -> list[Polyline]:
    """Connect paths whose endpoints are within *threshold_mm* of each other.

    Uses ``scipy.spatial.cKDTree`` for efficient nearest-neighbour queries.
    Each path has two endpoints (start, end). When an endpoint from one path
    is within the threshold of an endpoint from a different path, the two
    paths are merged into a single continuous polyline.

    When the two endpoints are not exactly coincident, the join point is
    *snapped to the midpoint* of the gap (and the duplicate point dropped) so
    a polyline does not silently encode a phantom straight line across the
    gap. This is critical for map / road plots where road endpoints often sit
    a fraction of a mm apart at intersections — the old behaviour drew a
    visible "road" through every such gap.

    The algorithm is greedy and single-pass: once a path is merged into
    another it is marked as consumed and will not be merged again. This
    avoids creating long chains in one pass but reliably reduces the number
    of short disconnected strokes.

    Self-loops (merging a path's start to its own end) are avoided.

    The inner candidate-selection loop is JIT-compiled via
    ``_find_and_snap`` when numba is available, falling back to the same
    pure-Python code path otherwise.

    Args:
        paths: Input list of polylines.
        threshold_mm: Maximum endpoint distance that triggers a merge (mm).

    Returns:
        New list of polylines with nearby paths merged. The list may be
        shorter than the input if merges occurred.
    """
    if not paths:
        return []

    n = len(paths)
    # Working copy so we can reverse paths without mutating input
    working: list[list[tuple[float, float]]] = [list(p) for p in paths]
    # numpy bool array so the JIT kernel sees up-to-date consumed flags
    consumed = np.zeros(n, dtype=np.bool_)

    changed = True
    while changed:
        changed = False
        # Rebuild tree from current (possibly modified) endpoints
        endpoints = []
        for i, p in enumerate(working):
            if consumed[i]:
                endpoints.append((0.0, 0.0))
                endpoints.append((0.0, 0.0))
            else:
                endpoints.append(p[0])
                endpoints.append(p[-1])

        pts_array = np.array(endpoints, dtype=np.float64)
        tree = cKDTree(pts_array)

        for i in range(n):
            if consumed[i]:
                continue
            if len(working[i]) < 2:
                continue

            # Try to extend from the END of path i
            end_pt = working[i][-1]
            indices = tree.query_ball_point(end_pt, threshold_mm)

            if not indices:
                continue

            indices_arr = np.asarray(indices, dtype=np.int64)
            best_j, best_rev, mid_x, mid_y, coincident = _find_and_snap(
                end_pt[0], end_pt[1], indices_arr, pts_array, consumed, i, 1
            )

            if best_j >= 0:
                if best_rev:
                    working[best_j] = working[best_j][::-1]
                if coincident:
                    working[i] = working[i] + working[best_j][1:]
                else:
                    working[i] = working[i][:-1] + [(mid_x, mid_y)] + working[best_j][1:]
                consumed[best_j] = True
                changed = True

    return [working[i] for i in range(n) if not consumed[i]]


def merge_fragments(
    polylines: list[Polyline], gap_tolerance_mm: float = 0.5
) -> list[Polyline]:
    """Merge short polyline fragments by connecting nearby endpoints.

    Designed for skeleton-traced output where morphological skeletonization
    produces many disconnected short segments at branch junctions and corners.
    Unlike ``merge_nearby_paths`` (which only extends path chains from the end),
    this function considers *both* endpoints of each path, enabling longer
    continuous chains to form in either direction.

    Algorithm:
        1. Build a ``cKDTree`` of all path start/end points.
        2. For each non-consumed path *i*:

           a. Try to *extend from the end* — find the nearest endpoint of some
              path *j* within ``gap_tolerance_mm``.  Append *j* to the end of
              *i*, reversing *j* if its end (not start) was the match.

           b. If no end-extension was found, try to *prepend from the start* —
              find the nearest endpoint of *j*.  Prepend *j* to the start of
              *i*, reversing *j* if its start (not end) was the match.

        3. Repeat until no more merges occur.

    Prefers end→start connections (no reversal needed) over end→end / start→start,
    but allows reversals when that is the closest available match.

    The inner candidate-selection loops are JIT-compiled via
    ``_find_and_snap`` when numba is available.

    Args:
        polylines:        Input list of polylines.
        gap_tolerance_mm: Maximum endpoint distance to trigger a merge (mm).
                          Pass 0 (or negative) to skip all merging.

    Returns:
        New list of polylines with nearby fragments merged.  The returned list
        may be shorter than the input when merges occurred.
    """
    if not polylines or gap_tolerance_mm <= 0:
        return list(polylines)

    n = len(polylines)
    # Working copies so we can reverse fragments without mutating the input
    working: list[list[tuple[float, float]]] = [list(p) for p in polylines]
    # numpy bool array so the JIT kernel sees up-to-date consumed flags
    consumed = np.zeros(n, dtype=np.bool_)

    changed = True
    while changed:
        changed = False
        # Rebuild endpoint lookup each outer pass.
        # Index 2*i → start of path i; 2*i+1 → end of path i.
        endpoints: list[tuple[float, float]] = []
        for i, p in enumerate(working):
            if consumed[i]:
                endpoints.append((0.0, 0.0))
                endpoints.append((0.0, 0.0))
            else:
                endpoints.append(p[0])
                endpoints.append(p[-1])

        pts_array = np.array(endpoints, dtype=np.float64)
        tree = cKDTree(pts_array)

        for i in range(n):
            if consumed[i]:
                continue
            if len(working[i]) < 2:
                continue

            # ---- Phase A: extend from END of path i ----
            end_pt = working[i][-1]
            end_idxs = tree.query_ball_point(end_pt, gap_tolerance_mm)

            if end_idxs:
                end_arr = np.asarray(end_idxs, dtype=np.int64)
                best_j, best_rev, mid_x, mid_y, coincident = _find_and_snap(
                    end_pt[0], end_pt[1], end_arr, pts_array, consumed, i, 1
                )
                if best_j >= 0:
                    if best_rev:
                        working[best_j] = working[best_j][::-1]
                    if coincident:
                        working[i] = working[i] + working[best_j][1:]
                    else:
                        working[i] = working[i][:-1] + [(mid_x, mid_y)] + working[best_j][1:]
                    consumed[best_j] = True
                    changed = True
                    # Skip phase B this pass; the next outer iteration will
                    # rebuild the tree and try again.
                    continue

            # ---- Phase B: prepend to START of path i ----
            start_pt = working[i][0]
            start_idxs = tree.query_ball_point(start_pt, gap_tolerance_mm)

            if start_idxs:
                start_arr = np.asarray(start_idxs, dtype=np.int64)
                best_j, best_rev, mid_x, mid_y, coincident = _find_and_snap(
                    start_pt[0], start_pt[1], start_arr, pts_array, consumed, i, 0
                )
                if best_j >= 0:
                    if best_rev:
                        working[best_j] = working[best_j][::-1]
                    # j's end is now near i's start — prepend j to i.
                    if coincident:
                        working[i] = working[best_j][:-1] + working[i]
                    else:
                        working[i] = working[best_j][:-1] + [(mid_x, mid_y)] + working[i][1:]
                    consumed[best_j] = True
                    changed = True

    return [working[i] for i in range(n) if not consumed[i]]
