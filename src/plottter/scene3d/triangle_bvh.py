"""Per-mesh BVH for fast triangle intersection.

TriangleBVH accelerates ray-triangle intersection for a single Mesh from
O(N) brute-force to O(log N) by organizing triangles in a BVH.

Key optimizations:
- Edge vectors e1/e2 and base vertex v0 are precomputed for all faces at build time.
- Leaf nodes store their triangle data in compact NumPy arrays (no index indirection
  during traversal).
- Leaf intersection uses fully vectorized Möller-Trumbore over all K leaf triangles
  in a single NumPy call, eliminating per-triangle Python overhead.
- Traversal uses an iterative stack instead of Python recursion.
- Backface culling is done with a vectorized dot-product over the leaf's normals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .ray import Ray, Hit, EPSILON


@dataclass
class _TriNode:
    """Node in the TriangleBVH tree."""

    bbox_min: NDArray[np.float64]      # shape (3,)
    bbox_max: NDArray[np.float64]      # shape (3,)
    left: "_TriNode | None" = None
    right: "_TriNode | None" = None
    # Leaf data — only set for leaf nodes (is_leaf returns True):
    leaf_v0: NDArray[np.float64] | None = None       # (K, 3)
    leaf_e1: NDArray[np.float64] | None = None       # (K, 3)
    leaf_e2: NDArray[np.float64] | None = None       # (K, 3)
    leaf_normals: NDArray[np.float64] | None = None  # (K, 3) or None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


MAX_LEAF_TRIS = 8  # split if a node would hold more than this many triangles


class TriangleBVH:
    """BVH acceleration structure for a triangle mesh.

    Usage
    -----
    bvh = TriangleBVH()
    bvh.build(vertices, faces, backface_cull=True)
    hit = bvh.intersect(ray)
    blocked = bvh.intersect_any(ray, t_max=segment_length)

    Parameters
    ----------
    vertices : (N, 3) float64 array
    faces    : (M, 3) int array (triangle vertex indices)
    backface_cull : pre-compute face normals and skip back-facing triangles
    """

    def __init__(self) -> None:
        self._backface_cull: bool = False
        self._root: _TriNode | None = None
        # Precomputed per-face arrays (built once, indexed by face index)
        self._v0: NDArray[np.float64] | None = None          # (M, 3)
        self._e1: NDArray[np.float64] | None = None          # (M, 3)
        self._e2: NDArray[np.float64] | None = None          # (M, 3)
        self._face_normals: NDArray[np.float64] | None = None  # (M, 3)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int32],
        backface_cull: bool = True,
    ) -> None:
        """Build the BVH from raw vertex/face arrays."""
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        self._backface_cull = backface_cull

        if len(faces) == 0:
            self._root = None
            return

        # Precompute per-face geometry — done once, reused by all leaves.
        self._v0 = vertices[faces[:, 0]]              # (M, 3)
        self._e1 = vertices[faces[:, 1]] - self._v0  # (M, 3)
        self._e2 = vertices[faces[:, 2]] - self._v0  # (M, 3)

        if backface_cull:
            normals = np.cross(self._e1, self._e2)  # (M, 3)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms = np.where(norms < EPSILON, 1.0, norms)
            self._face_normals = normals / norms
        else:
            self._face_normals = None

        all_indices = np.arange(len(faces), dtype=np.int32)
        self._root = self._build_node(all_indices)

    def _build_node(self, indices: NDArray[np.int32]) -> _TriNode:
        v0s = self._v0[indices]                  # (K, 3)
        v1s = v0s + self._e1[indices]            # (K, 3)
        v2s = v0s + self._e2[indices]            # (K, 3)

        # Combined AABB
        mins = np.minimum(np.minimum(v0s, v1s), v2s).min(axis=0)
        maxs = np.maximum(np.maximum(v0s, v1s), v2s).max(axis=0)

        node = _TriNode(bbox_min=mins, bbox_max=maxs)

        if len(indices) <= MAX_LEAF_TRIS:
            self._make_leaf(node, indices)
            return node

        # Longest-axis median split using triangle centroids
        size = maxs - mins
        axis = int(np.argmax(size))
        centroids = (v0s[:, axis] + v1s[:, axis] + v2s[:, axis]) / 3.0
        order = np.argsort(centroids)
        sorted_indices = indices[order]

        mid = len(sorted_indices) // 2
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]

        if len(left_indices) == 0 or len(right_indices) == 0:
            # Degenerate split: make leaf regardless of triangle count
            self._make_leaf(node, indices)
            return node

        node.left = self._build_node(left_indices)
        node.right = self._build_node(right_indices)
        return node

    def _make_leaf(self, node: _TriNode, indices: NDArray[np.int32]) -> None:
        """Populate a leaf node with precomputed per-triangle arrays."""
        node.leaf_v0 = self._v0[indices]
        node.leaf_e1 = self._e1[indices]
        node.leaf_e2 = self._e2[indices]
        if self._backface_cull and self._face_normals is not None:
            node.leaf_normals = self._face_normals[indices]

    # ------------------------------------------------------------------
    # Ray-AABB slab test (scalar, no object overhead)
    # ------------------------------------------------------------------

    @staticmethod
    def _aabb_hit(
        bmin: NDArray[np.float64],
        bmax: NDArray[np.float64],
        origin: NDArray[np.float64],
        inv_dir: NDArray[np.float64],
    ) -> bool:
        t1 = (bmin - origin) * inv_dir
        t2 = (bmax - origin) * inv_dir
        tmin = float(np.max(np.minimum(t1, t2)))
        tmax = float(np.min(np.maximum(t1, t2)))
        return tmax >= max(tmin, 0.0)

    # ------------------------------------------------------------------
    # Vectorized Möller-Trumbore over K leaf triangles
    # ------------------------------------------------------------------

    @staticmethod
    def _mt_batch(
        v0: NDArray[np.float64],        # (K, 3)
        e1: NDArray[np.float64],        # (K, 3)
        e2: NDArray[np.float64],        # (K, 3)
        origin: NDArray[np.float64],    # (3,)
        direction: NDArray[np.float64], # (3,)
    ) -> NDArray[np.float64]:
        """Return t values for K triangles; np.inf for misses/parallel/back-face."""
        # h = cross(direction, e2): broadcast (3,) × (K,3) → (K,3)
        h = np.cross(direction, e2)
        # a = dot(e1[k], h[k]): (K,)
        a = np.einsum("ki,ki->k", e1, h)
        valid = np.abs(a) >= EPSILON
        inv_a = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)

        # s = origin - v0: (K,3)
        s = origin - v0
        # u = inv_a * dot(s[k], h[k]): (K,)
        u = inv_a * np.einsum("ki,ki->k", s, h)
        valid &= (u >= 0.0) & (u <= 1.0)

        # q = cross(s, e1): (K,3)
        q = np.cross(s, e1)
        # v_coord = inv_a * dot(direction, q[k]): (K,)
        v_coord = inv_a * np.einsum("i,ki->k", direction, q)
        valid &= (v_coord >= 0.0) & ((u + v_coord) <= 1.0)

        # t = inv_a * dot(e2[k], q[k]): (K,)
        t = inv_a * np.einsum("ki,ki->k", e2, q)
        valid &= t > EPSILON

        return np.where(valid, t, np.inf)

    # ------------------------------------------------------------------
    # Public intersection API
    # ------------------------------------------------------------------

    def intersect(self, ray: Ray) -> Hit | None:
        """Find the closest triangle intersection. Returns Hit or None."""
        if self._root is None:
            return None
        origin = np.asarray(ray.origin, dtype=np.float64)
        direction = np.asarray(ray.direction, dtype=np.float64)
        inv_dir = np.where(np.abs(direction) < EPSILON, np.inf, 1.0 / direction)
        return self._intersect_iterative(origin, direction, inv_dir)

    def _intersect_iterative(
        self,
        origin: NDArray,
        direction: NDArray,
        inv_dir: NDArray,
    ) -> Hit | None:
        t_max = np.inf
        closest: Hit | None = None
        stack: list[_TriNode] = [self._root]

        while stack:
            node = stack.pop()
            if not self._aabb_hit(node.bbox_min, node.bbox_max, origin, inv_dir):
                continue

            if node.is_leaf:
                v0 = node.leaf_v0
                e1 = node.leaf_e1
                e2 = node.leaf_e2

                # Vectorized backface culling
                if node.leaf_normals is not None:
                    dots = node.leaf_normals @ direction  # (K,)
                    mask = dots <= 0.0
                    if not np.any(mask):
                        continue
                    v0 = v0[mask]
                    e1 = e1[mask]
                    e2 = e2[mask]

                t_arr = self._mt_batch(v0, e1, e2, origin, direction)
                # Keep only hits closer than current best
                t_arr = np.where(t_arr < t_max, t_arr, np.inf)
                if np.all(np.isinf(t_arr)):
                    continue
                k = int(np.argmin(t_arr))
                t_hit = float(t_arr[k])
                closest = Hit(shape=None, t=t_hit, point=origin + direction * t_hit)
                t_max = t_hit
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)

        return closest

    def intersect_any(self, ray: Ray, t_max: float = float("inf")) -> bool:
        """Return True if ray hits any triangle within t_max (early exit)."""
        if self._root is None:
            return False
        origin = np.asarray(ray.origin, dtype=np.float64)
        direction = np.asarray(ray.direction, dtype=np.float64)
        inv_dir = np.where(np.abs(direction) < EPSILON, np.inf, 1.0 / direction)
        return self._intersect_any_iterative(origin, direction, inv_dir, t_max)

    def _intersect_any_iterative(
        self,
        origin: NDArray,
        direction: NDArray,
        inv_dir: NDArray,
        t_max: float,
    ) -> bool:
        stack: list[_TriNode] = [self._root]

        while stack:
            node = stack.pop()
            if not self._aabb_hit(node.bbox_min, node.bbox_max, origin, inv_dir):
                continue

            if node.is_leaf:
                v0 = node.leaf_v0
                e1 = node.leaf_e1
                e2 = node.leaf_e2

                if node.leaf_normals is not None:
                    dots = node.leaf_normals @ direction  # (K,)
                    mask = dots <= 0.0
                    if not np.any(mask):
                        continue
                    v0 = v0[mask]
                    e1 = e1[mask]
                    e2 = e2[mask]

                t_arr = self._mt_batch(v0, e1, e2, origin, direction)
                if np.any(t_arr < t_max):
                    return True
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)

        return False
