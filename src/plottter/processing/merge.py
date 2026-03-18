"""Merge polylines whose endpoints are within a threshold distance."""

import math
import numpy as np
from scipy.spatial import cKDTree

from plottter.models.path import Polyline


def merge_nearby_paths(paths: list[Polyline], threshold_mm: float = 0.5) -> list[Polyline]:
    """Connect paths whose endpoints are within *threshold_mm* of each other.

    Uses ``scipy.spatial.cKDTree`` for efficient nearest-neighbour queries.
    Each path has two endpoints (start, end). When an endpoint from one path
    is within the threshold of an endpoint from a different path, the two
    paths are merged into a single continuous polyline.

    The algorithm is greedy and single-pass: once a path is merged into
    another it is marked as consumed and will not be merged again. This
    avoids creating long chains in one pass but reliably reduces the number
    of short disconnected strokes.

    Self-loops (merging a path's start to its own end) are avoided.

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
    consumed = [False] * n

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

            best_j: int | None = None
            best_dist = float("inf")
            best_reverse_j = False

            for idx in indices:
                j = idx // 2
                is_end = idx % 2 == 1

                if j == i or consumed[j]:
                    continue
                # Skip if this is path i's own start (2*i)
                if idx == 2 * i:
                    continue

                ex, ey = endpoints[idx]
                dist = math.hypot(end_pt[0] - ex, end_pt[1] - ey)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    # If we match the END of j, we must reverse j to append
                    best_reverse_j = is_end

            if best_j is not None:
                j = best_j
                if best_reverse_j:
                    working[j] = working[j][::-1]
                # Merge: drop duplicate junction point if exactly equal
                junction_i = working[i][-1]
                junction_j = working[j][0]
                if (abs(junction_i[0] - junction_j[0]) < 1e-9 and
                        abs(junction_i[1] - junction_j[1]) < 1e-9):
                    working[i] = working[i] + working[j][1:]
                else:
                    working[i] = working[i] + working[j]
                consumed[j] = True
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
    consumed = [False] * n

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

            best_j: int | None = None
            best_dist = float("inf")
            best_rev_j = False

            for idx in end_idxs:
                j = idx // 2
                if j == i or consumed[j]:
                    continue
                if idx == 2 * i:  # own start — would create a self-loop
                    continue
                ex, ey = endpoints[idx]
                dist = math.hypot(end_pt[0] - ex, end_pt[1] - ey)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    # Matching j's end (odd index) → must reverse j so its
                    # start aligns with i's end.
                    best_rev_j = (idx % 2 == 1)

            if best_j is not None:
                j = best_j
                if best_rev_j:
                    working[j] = working[j][::-1]
                ix, iy = working[i][-1]
                jx, jy = working[j][0]
                if abs(ix - jx) < 1e-9 and abs(iy - jy) < 1e-9:
                    working[i] = working[i] + working[j][1:]
                else:
                    working[i] = working[i] + working[j]
                consumed[j] = True
                changed = True
                # Skip phase B this pass; the next outer iteration will
                # rebuild the tree and try again.
                continue

            # ---- Phase B: prepend to START of path i ----
            start_pt = working[i][0]
            start_idxs = tree.query_ball_point(start_pt, gap_tolerance_mm)

            best_j = None
            best_dist = float("inf")
            best_rev_j = False

            for idx in start_idxs:
                j = idx // 2
                if j == i or consumed[j]:
                    continue
                if idx == 2 * i + 1:  # own end — would create a self-loop
                    continue
                ex, ey = endpoints[idx]
                dist = math.hypot(start_pt[0] - ex, start_pt[1] - ey)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    # Matching j's start (even index) → must reverse j so its
                    # end aligns with i's start.
                    best_rev_j = (idx % 2 == 0)

            if best_j is not None:
                j = best_j
                if best_rev_j:
                    working[j] = working[j][::-1]
                # j's end is now near i's start — prepend j to i
                jx, jy = working[j][-1]
                ix, iy = working[i][0]
                if abs(jx - ix) < 1e-9 and abs(jy - iy) < 1e-9:
                    working[i] = working[j][:-1] + working[i]
                else:
                    working[i] = working[j] + working[i]
                consumed[j] = True
                changed = True

    return [working[i] for i in range(n) if not consumed[i]]
