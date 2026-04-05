"""Plane-mesh intersection (mesh slicing) utilities.

Provides:
- _slice_mesh:      Intersect a triangle mesh with a plane → raw 3D segments.
- _chain_segments:  Chain segments into closed polylines (tolerance-based).
- _project_to_2d:   Drop the dominant normal axis to get 2D contours.
- slice_mesh:       All-in-one public function.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Core intersection
# ---------------------------------------------------------------------------

def _slice_mesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Intersect a triangle mesh with a plane and return 3D line segments.

    For each triangle that straddles the plane, computes the two edge-crossing
    intersection points and returns them as a (P1, P2) tuple.

    Parameters
    ----------
    vertices:       (N, 3) float64 vertex positions.
    faces:          (M, 3) int32 vertex indices (triangles).
    plane_origin:   A point on the plane, shape (3,).
    plane_normal:   Plane normal vector, shape (3,).  Need not be unit length.

    Returns
    -------
    List of (P1, P2) tuples where P1 and P2 are 3D numpy float64 arrays.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    origin = np.asarray(plane_origin, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Signed distances for all vertices: d[i] = dot(vertices[i] - origin, normal)
    vdists = (vertices - origin) @ normal  # (N,)

    # Per-face signed distances: shape (M, 3)
    d = vdists[faces]
    # Per-face vertex positions: shape (M, 3, 3)
    v = vertices[faces]

    segments: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    for i in range(len(faces)):
        da, db, dc = float(d[i, 0]), float(d[i, 1]), float(d[i, 2])

        # Skip if all three vertices are on the same side (all positive or all negative)
        if (da > 0 and db > 0 and dc > 0) or (da < 0 and db < 0 and dc < 0):
            continue

        va, vb, vc = v[i, 0], v[i, 1], v[i, 2]

        # Find intersection points on crossing edges
        pts: list[NDArray[np.float64]] = []
        edge_dists = [(da, db, va, vb), (db, dc, vb, vc), (dc, da, vc, va)]
        for d1, d2, p1, p2 in edge_dists:
            if (d1 > 0) == (d2 > 0):
                # Both on the same side (both positive, or both non-positive):
                # edge does not strictly cross the plane.
                # Vertices with d=0 are captured via their adjacent crossing edge.
                continue
            denom = d1 - d2
            if abs(denom) < 1e-14:
                continue
            t = d1 / denom
            pts.append(p1 + t * (p2 - p1))

        if len(pts) == 2:
            segments.append((pts[0], pts[1]))

    return segments


# ---------------------------------------------------------------------------
# Segment chaining
# ---------------------------------------------------------------------------

def _chain_edges_by_index(
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    """Chain (a, b) integer edge pairs into polylines.

    Same greedy algorithm as ``Mesh._chain_edges``:
    walks from branch/endpoint vertices first, then handles remaining loops.

    Returns a list of vertex-index chains.  Closed loops have the same
    start and end index.
    """
    if not edges:
        return []

    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    unvisited: set[tuple[int, int]] = set()
    for a, b in edges:
        unvisited.add((min(a, b), max(a, b)))

    chains: list[list[int]] = []

    def walk(start: int, nxt: int) -> list[int]:
        key = (min(start, nxt), max(start, nxt))
        if key not in unvisited:
            return []
        unvisited.discard(key)
        chain = [start, nxt]
        prev, curr = start, nxt
        while True:
            if len(adj[curr]) != 2:
                break
            next_node: int | None = None
            for n in adj[curr]:
                if n != prev:
                    k = (min(curr, n), max(curr, n))
                    if k in unvisited:
                        next_node = n
                        break
            if next_node is None:
                break
            k = (min(curr, next_node), max(curr, next_node))
            unvisited.discard(k)
            chain.append(next_node)
            prev, curr = curr, next_node
        return chain

    # Walk from branch/endpoint vertices (degree ≠ 2)
    branch_vertices = [v for v, nbrs in adj.items() if len(nbrs) != 2]
    for bv in branch_vertices:
        for neighbor in adj[bv]:
            key = (min(bv, neighbor), max(bv, neighbor))
            if key in unvisited:
                chain = walk(bv, neighbor)
                if len(chain) >= 2:
                    chains.append(chain)

    # Handle remaining unvisited edges (closed loops)
    while unvisited:
        key = next(iter(unvisited))
        a, b = key
        chain = walk(a, b)
        if len(chain) >= 2:
            chains.append(chain)

    return chains


def _chain_segments(
    segments: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    tol: float = 1e-6,
) -> list[list[NDArray[np.float64]]]:
    """Chain 3D line segments into polylines by matching shared endpoints.

    Uses a tolerance-based spatial hash to identify matching endpoints, then
    runs the same greedy chaining algorithm as ``Mesh._chain_edges``.

    Parameters
    ----------
    segments:   List of (P1, P2) endpoint pairs in 3D.
    tol:        Coordinate matching tolerance.  Points within this distance
                are considered identical.

    Returns
    -------
    List of polylines, each a list of 3D numpy arrays.
    Closed loops have the same first and last point.
    """
    if not segments:
        return []

    # Map floating-point 3D points → integer indices using a grid hash
    scale = 1.0 / tol
    point_to_idx: dict[tuple[int, int, int], int] = {}
    points: list[NDArray[np.float64]] = []

    def get_idx(pt: NDArray[np.float64]) -> int:
        key = (
            int(round(float(pt[0]) * scale)),
            int(round(float(pt[1]) * scale)),
            int(round(float(pt[2]) * scale)),
        )
        if key not in point_to_idx:
            idx = len(points)
            point_to_idx[key] = idx
            points.append(pt.copy())
        return point_to_idx[key]

    # Build integer edge list
    edges: list[tuple[int, int]] = []
    for p1, p2 in segments:
        i1 = get_idx(p1)
        i2 = get_idx(p2)
        if i1 != i2:
            edges.append((i1, i2))

    # Chain edges
    chains_idx = _chain_edges_by_index(edges)

    # Convert index chains back to 3D point lists
    result: list[list[NDArray[np.float64]]] = []
    for chain in chains_idx:
        result.append([points[i] for i in chain])
    return result


# ---------------------------------------------------------------------------
# 2D projection
# ---------------------------------------------------------------------------

def _project_to_2d(
    contours_3d: list[list[NDArray[np.float64]]],
    plane_normal: NDArray[np.float64],
) -> list[list[tuple[float, float]]]:
    """Project 3D contour points to 2D by dropping the dominant normal axis.

    For a Z-slicing plane (normal = [0, 0, 1]), drops Z and keeps (X, Y).
    More generally, drops the axis with the largest absolute component in the
    normalised plane normal.

    Parameters
    ----------
    contours_3d:    List of polylines, each a list of (3,) numpy arrays.
    plane_normal:   Plane normal vector; used to determine which axis to drop.

    Returns
    -------
    List of polylines, each a list of (x, y) float tuples.
    """
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    drop_axis = int(np.argmax(np.abs(n)))
    keep_axes = [i for i in range(3) if i != drop_axis]

    result: list[list[tuple[float, float]]] = []
    for contour_3d in contours_3d:
        contour_2d = [
            (float(pt[keep_axes[0]]), float(pt[keep_axes[1]]))
            for pt in contour_3d
        ]
        result.append(contour_2d)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def slice_mesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    tol: float = 1e-6,
    project: bool = True,
) -> list[list[tuple[float, float]]] | list[list[NDArray[np.float64]]]:
    """Slice a triangle mesh with a plane and return closed 2D (or 3D) contours.

    Combines :func:`_slice_mesh`, :func:`_chain_segments`, and optionally
    :func:`_project_to_2d` into a single convenience function.

    Parameters
    ----------
    vertices:       (N, 3) float64 vertex positions.
    faces:          (M, 3) int32 vertex indices.
    plane_origin:   A point on the cutting plane.
    plane_normal:   Normal vector of the cutting plane.
    tol:            Endpoint-matching tolerance for chaining.
    project:        If True (default), project contours to 2D using the plane
                    normal to determine the drop axis.  If False, return 3D.

    Returns
    -------
    List of closed polylines.  When ``project=True``, each polyline is a list
    of ``(x, y)`` float tuples.  When ``project=False``, a list of 3D numpy
    arrays.
    """
    segments = _slice_mesh(vertices, faces, plane_origin, plane_normal)
    contours_3d = _chain_segments(segments, tol=tol)
    if project:
        return _project_to_2d(contours_3d, plane_normal)
    return contours_3d
