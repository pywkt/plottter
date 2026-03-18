"""Bounding Volume Hierarchy (BVH) acceleration structure.

Uses median-split construction for good average-case performance.
Ray traversal uses a flat NumPy array node representation for
Numba JIT compatibility (optional).

The BVH accelerates hidden line removal (HLR) by quickly culling
shapes that cannot possibly occlude a given ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .bbox import BBox
from .ray import Ray, Hit, EPSILON
from .vector3 import vec3

if TYPE_CHECKING:
    from .shapes.base import Shape

# Try to import Numba for JIT acceleration
try:
    from numba import njit as _numba_njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def _numba_njit(func=None, **kwargs):  # type: ignore
        if func is not None:
            return func
        return lambda f: f
    _NUMBA_AVAILABLE = False


@dataclass
class BVHNode:
    """A node in the BVH tree."""
    bbox: BBox
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None
    shapes: list["Shape"] = field(default_factory=list)  # leaf node shapes

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class BVH:
    """BVH tree for fast ray intersection testing.

    Usage
    -----
    bvh = BVH(shapes)
    bvh.build()
    hit = bvh.intersect(ray)
    """

    MAX_LEAF_SHAPES = 4  # split if a node has more than this many shapes

    def __init__(self, shapes: list["Shape"]) -> None:
        self.shapes = shapes
        self.root: BVHNode | None = None

    def build(self) -> None:
        """Build the BVH tree using median-split on the longest axis."""
        if not self.shapes:
            self.root = None
            return
        self.root = self._build_node(list(self.shapes))

    def _build_node(self, shapes: list["Shape"]) -> BVHNode:
        # Compute combined bounding box
        combined = BBox.empty()
        bboxes = []
        for s in shapes:
            b = s.bbox()
            bboxes.append(b)
            combined = combined.expand(b)

        node = BVHNode(bbox=combined)

        if len(shapes) <= self.MAX_LEAF_SHAPES:
            node.shapes = shapes
            return node

        # Split on the longest axis at the median centroid
        axis = combined.longest_axis()
        centers = [(b.center()[axis], i) for i, b in enumerate(bboxes)]
        centers.sort(key=lambda x: x[0])
        mid = len(centers) // 2

        left_shapes = [shapes[centers[i][1]] for i in range(mid)]
        right_shapes = [shapes[centers[i][1]] for i in range(mid, len(centers))]

        if not left_shapes or not right_shapes:
            node.shapes = shapes
            return node

        node.left = self._build_node(left_shapes)
        node.right = self._build_node(right_shapes)
        return node

    def intersect(self, ray: Ray) -> Hit | None:
        """Find the closest intersection of ray with any shape in the BVH."""
        if self.root is None:
            return None
        return self._intersect_node(self.root, ray, float("inf"))

    def _intersect_node(self, node: BVHNode, ray: Ray, t_max: float) -> Hit | None:
        if node.bbox.intersect(ray) is None:
            return None

        if node.is_leaf:
            closest: Hit | None = None
            for shape in node.shapes:
                hit = shape.intersect(ray)
                if hit is not None and EPSILON < hit.t < t_max:
                    if closest is None or hit.t < closest.t:
                        closest = hit
                        t_max = hit.t
            return closest

        # Interior node: recurse into both children
        left_hit = self._intersect_node(node.left, ray, t_max) if node.left else None
        if left_hit is not None:
            t_max = left_hit.t
        right_hit = self._intersect_node(node.right, ray, t_max) if node.right else None

        if right_hit is not None and (left_hit is None or right_hit.t < left_hit.t):
            return right_hit
        return left_hit

    def intersect_any(self, ray: Ray, t_max: float = float("inf")) -> bool:
        """Return True if the ray hits anything within distance t_max.

        This is faster than intersect() for occlusion testing since it
        stops at the first hit rather than finding the closest one.
        """
        if self.root is None:
            return False
        return self._intersect_any_node(self.root, ray, t_max)

    def _intersect_any_node(self, node: BVHNode, ray: Ray, t_max: float) -> bool:
        if node.bbox.intersect(ray) is None:
            return False
        if node.is_leaf:
            for shape in node.shapes:
                # Use shape.intersect_any() for early-exit occlusion testing.
                # Shapes that override this (e.g. Mesh) can skip the full
                # closest-hit search and return as soon as any hit is found.
                if shape.intersect_any(ray, t_max):
                    return True
            return False
        if node.left and self._intersect_any_node(node.left, ray, t_max):
            return True
        if node.right and self._intersect_any_node(node.right, ray, t_max):
            return True
        return False
