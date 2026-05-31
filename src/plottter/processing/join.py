"""Graph-aware path chaining via Eulerian path tracing (join_at_junctions)."""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import cKDTree

from plottter.models.path import Point, Polyline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _snap_key(pt: Point, inv_snap: float) -> tuple[int, int]:
    """Return the integer grid key for *pt* at the given snap resolution."""
    return (round(pt[0] * inv_snap), round(pt[1] * inv_snap))


def _filter_degenerate(paths: list[Polyline]) -> list[Polyline]:
    """Remove single-point paths and exact duplicates."""
    seen: set[tuple[Point, ...]] = set()
    result: list[Polyline] = []
    for p in paths:
        if len(p) < 2:
            continue
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def _split_at_t_junctions(
    paths: list[Polyline], threshold_mm: float
) -> list[Polyline]:
    """Split paths wherever another path's endpoint touches an interior vertex.

    Builds a ``cKDTree`` over every *interior* vertex (not the first or last
    point) of every path.  For each path's two endpoints, any interior vertex
    of another path within *threshold_mm* triggers a split of that path at the
    matching interior vertex index.  This converts T-junctions into
    endpoint↔endpoint connections so the graph builder only needs to look at
    path endpoints.

    Total neighbor-pair checks are capped at ``max(len(paths) * 8, 64)`` so
    pathological dense-point-cloud inputs cannot cause O(n²) hangs.
    """
    if not paths:
        return []

    # Collect all interior vertices with back-references
    interior_pts: list[tuple[float, float]] = []
    interior_refs: list[tuple[int, int]] = []  # (path_idx, vertex_idx)

    for i, path in enumerate(paths):
        for vi in range(1, len(path) - 1):
            interior_pts.append(path[vi])
            interior_refs.append((i, vi))

    if not interior_pts:
        return list(paths)

    arr = np.array(interior_pts, dtype=np.float64)
    tree = cKDTree(arr)

    # Collect splits per path: path_idx -> set of vertex indices to split at
    splits_needed: dict[int, set[int]] = defaultdict(set)

    # Cap: limit total (endpoint, interior-vertex) pairs examined so dense
    # inputs with many coincident vertices cannot cause O(n²) work.
    max_checks = max(len(paths) * 8, 64)
    checks_done = 0

    for i, path in enumerate(paths):
        if checks_done >= max_checks:
            break
        for endpoint in (path[0], path[-1]):
            if checks_done >= max_checks:
                break
            ep_arr = np.array(endpoint, dtype=np.float64)
            neighbors = tree.query_ball_point(ep_arr, threshold_mm)
            for ni in neighbors:
                if checks_done >= max_checks:
                    break
                checks_done += 1
                pi, vi = interior_refs[ni]
                if pi != i:
                    splits_needed[pi].add(vi)

    if not splits_needed:
        return list(paths)

    result: list[Polyline] = []
    for i, path in enumerate(paths):
        if i not in splits_needed:
            result.append(path)
        else:
            split_idxs = sorted(splits_needed[i])
            prev = 0
            for si in split_idxs:
                segment = path[prev : si + 1]
                if len(segment) >= 2:
                    result.append(segment)
                prev = si
            tail = path[prev:]
            if len(tail) >= 2:
                result.append(tail)

    return result


def _join_segments(segments: list[list[Point]]) -> list[Point]:
    """Concatenate point-sequences with midpoint snapping at joins."""
    if not segments:
        return []
    result: list[Point] = list(segments[0])
    for seg in segments[1:]:
        if not seg:
            continue
        last = result[-1]
        first = seg[0]
        if abs(last[0] - first[0]) < 1e-9 and abs(last[1] - first[1]) < 1e-9:
            result.extend(seg[1:])
        else:
            mid: Point = (
                (last[0] + first[0]) * 0.5,
                (last[1] + first[1]) * 0.5,
            )
            result[-1] = mid
            result.extend(seg[1:])
    return result


def _hierholzer(
    adj: dict[int, list[tuple[int, int]]],
    edge_data: list[tuple[int, int, list[Point], bool]],
    edge_used: list[bool],
    start: int,
) -> list[tuple[int, int]]:
    """Trace an Eulerian walk from *start* using Hierholzer's algorithm.

    Returns the trail as a list of (vertex_id, incoming_edge_idx) tuples.
    The first element always has incoming_edge_idx == -1 (no incoming edge).
    The trail is returned in traversal order (start → … → end).
    """
    adj_pos: dict[int, int] = {v: 0 for v in adj}

    stack: list[tuple[int, int]] = [(start, -1)]
    trail: list[tuple[int, int]] = []

    while stack:
        v, _incoming = stack[-1]
        found = False
        while adj_pos.get(v, 0) < len(adj.get(v, [])):
            u, ei = adj[v][adj_pos[v]]
            adj_pos[v] += 1
            if not edge_used[ei]:
                edge_used[ei] = True
                stack.append((u, ei))
                found = True
                break
        if not found:
            trail.append(stack.pop())

    trail.reverse()
    return trail


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def join_at_junctions(
    paths: list[Polyline],
    threshold_mm: float = 0.1,
    split_at_t_junctions: bool = True,
) -> list[Polyline]:
    """Chain paths into Eulerian walks using graph-aware junction analysis.

    The algorithm:

    1. **Filter** degenerate input (single-point paths, exact duplicates).
    2. **T-junction splitting** — when an endpoint of path A lies within
       *threshold_mm* of an *interior* vertex of path B, B is split at that
       vertex so the junction becomes endpoint↔endpoint.
    3. **Multigraph construction** — every (post-split) path becomes one edge;
       its two endpoints are snapped to integer grid keys (resolution =
       1 / threshold_mm) to discover shared vertices.
    4. **Connected components** — graph vertices are partitioned into disjoint
       components via BFS; each component is processed independently.
    5. **Eulerian walk** — Hierholzer's algorithm traces each component:

       - 0 odd-degree vertices → single closed loop.
       - 2 odd-degree vertices → single open chain from odd vertex to odd.
       - >2 odd-degree vertices → greedily pair odd vertices by nearest
         distance (vertex ID as tie-breaker) and inject phantom "pen-lift"
         edges until 2 odd vertices remain; the resulting walk is split at
         each pen-lift, yielding multiple chains.

    Join points that are close but not coincident are snapped to the midpoint
    of the gap, matching the behaviour of :func:`merge_nearby_paths`.

    Args:
        paths: Input list of polylines.
        threshold_mm: Vertex-proximity threshold used for snapping and
            T-junction detection (mm).  Defaults to 0.1 mm.
        split_at_t_junctions: When ``True`` (the default), split paths at
            T-junction interior vertices before building the graph.

    Returns:
        List of polylines representing the traced chains.  Paths in the
        same connected component are joined; disconnected components remain
        as separate entries in the returned list.
    """
    # ------------------------------------------------------------------
    # Step 1: filter degenerate input
    # ------------------------------------------------------------------
    working: list[list[Point]] = _filter_degenerate([list(p) for p in paths])
    if not working:
        return []

    # ------------------------------------------------------------------
    # Step 2: T-junction splitting
    # ------------------------------------------------------------------
    if split_at_t_junctions:
        working = _split_at_t_junctions(working, threshold_mm)
        working = _filter_degenerate(working)
    if not working:
        return []

    # ------------------------------------------------------------------
    # Step 3: build multigraph
    # ------------------------------------------------------------------
    inv_snap = 1.0 / max(threshold_mm, 1e-9)

    # vertex_id (int) ↔ snapped grid key; vertex_coords[id] = mm position
    snap_to_vid: dict[tuple[int, int], int] = {}
    vertex_coords: list[Point] = []

    def _get_vid(pt: Point) -> int:
        key = _snap_key(pt, inv_snap)
        if key not in snap_to_vid:
            vid = len(vertex_coords)
            snap_to_vid[key] = vid
            vertex_coords.append(pt)
        return snap_to_vid[key]

    # edge_data: (v1, v2, points, is_phantom)
    edge_data: list[tuple[int, int, list[Point], bool]] = []
    edge_used: list[bool] = []

    # adj[v] = list of (neighbor_vid, edge_idx)
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def _add_edge(
        v1: int, v2: int, pts: list[Point], phantom: bool = False
    ) -> int:
        ei = len(edge_data)
        edge_data.append((v1, v2, pts, phantom))
        edge_used.append(False)
        adj[v1].append((v2, ei))
        adj[v2].append((v1, ei))
        return ei

    # One pass: register each path's endpoints and add a graph edge.
    # _get_vid is O(1) dict lookup — no cap needed here.
    for path_idx, path in enumerate(working):
        v1 = _get_vid(path[0])
        v2 = _get_vid(path[-1])
        _add_edge(v1, v2, list(path))

    # ------------------------------------------------------------------
    # Step 4: find connected components (BFS)
    # ------------------------------------------------------------------
    all_vids: set[int] = set(adj.keys())
    visited: set[int] = set()
    components: list[list[int]] = []

    for start_vid in sorted(all_vids):  # sorted → deterministic order
        if start_vid in visited:
            continue
        comp: list[int] = []
        queue: deque[int] = deque([start_vid])
        in_queue: set[int] = {start_vid}
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            for u, _ei in adj[v]:
                if u not in visited and u not in in_queue:
                    in_queue.add(u)
                    queue.append(u)
        components.append(comp)

    # ------------------------------------------------------------------
    # Step 5: Eulerian walk per component
    # ------------------------------------------------------------------
    result: list[Polyline] = []

    for comp_vids in components:
        # Degree of each vertex in this component (before phantom edges)
        degree: dict[int, int] = {v: len(adj[v]) for v in comp_vids}

        # Odd-degree vertices sorted for determinism
        odd_verts: list[int] = sorted(
            v for v in comp_vids if degree[v] % 2 == 1
        )

        # Greedily pair odd vertices (injecting phantom pen-lift edges)
        # until at most 2 odd vertices remain → exactly one Eulerian path.
        while len(odd_verts) > 2:
            v0 = odd_verts[0]
            c0 = vertex_coords[v0]
            best_idx = 1
            best_dist = math.hypot(
                c0[0] - vertex_coords[odd_verts[1]][0],
                c0[1] - vertex_coords[odd_verts[1]][1],
            )
            for i in range(2, len(odd_verts)):
                vi = odd_verts[i]
                ci = vertex_coords[vi]
                d = math.hypot(c0[0] - ci[0], c0[1] - ci[1])
                # Tie-break by vertex ID for determinism
                if d < best_dist or (d == best_dist and vi < odd_verts[best_idx]):
                    best_dist = d
                    best_idx = i

            paired = odd_verts[best_idx]
            _add_edge(v0, paired, [], phantom=True)
            # Remove both from odd list (higher index first to keep indices valid)
            del odd_verts[best_idx]
            odd_verts.pop(0)

        # Choose start vertex for Hierholzer's
        if odd_verts:
            start_v = min(odd_verts)  # deterministic: smallest odd-vertex ID
        else:
            start_v = min(comp_vids)  # deterministic: smallest vertex ID

        # Run Hierholzer's
        trail = _hierholzer(adj, edge_data, edge_used, start_v)

        # Reconstruct polylines, splitting at phantom pen-lift edges
        current_segs: list[list[Point]] = []

        for step_idx in range(1, len(trail)):
            prev_v, _ = trail[step_idx - 1]
            _curr_v, ei = trail[step_idx]

            e_v1, _e_v2, e_pts, is_phantom = edge_data[ei]

            if is_phantom:
                # Pen lift: flush current chain and start fresh
                if current_segs:
                    chain = _join_segments(current_segs)
                    if len(chain) >= 2:
                        result.append(chain)
                current_segs = []
                continue

            # Determine traversal direction
            seg = list(e_pts) if e_v1 == prev_v else list(reversed(e_pts))
            if seg:
                current_segs.append(seg)

        if current_segs:
            chain = _join_segments(current_segs)
            if len(chain) >= 2:
                result.append(chain)

    return result
