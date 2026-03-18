"""CSG (Constructive Solid Geometry) difference and intersection shapes.

Note: Ported from Viewport.js which notes these as 'buggy'. Use with caution.
They work for simple cases but may produce artifacts with complex geometry.
"""

from __future__ import annotations

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit
from .base import Shape


class CSGDifference(Shape):
    """CSG difference: shape A minus shape B.

    Returns the parts of A that are not inside B.
    Note: This is an approximation — ray intersection checks are used for HLR,
    but path filtering is not implemented (paths are just A's paths).
    """

    def __init__(self, shape_a: Shape, shape_b: Shape) -> None:
        self.shape_a = shape_a
        self.shape_b = shape_b

    def paths(self) -> list[Path3D]:
        # Filter A's paths to only those outside B
        result = []
        for path in self.shape_a.paths():
            filtered = path.filter_outside(self.shape_b)
            result.extend(filtered)
        return result

    def intersect(self, ray: Ray) -> Hit | None:
        hit_a = self.shape_a.intersect(ray)
        hit_b = self.shape_b.intersect(ray)
        if hit_a is None:
            return None
        if hit_b is None:
            hit_a.shape = self
            return hit_a
        # Return the farther intersection (the "outside" of B)
        if hit_a.t > hit_b.t:
            hit_a.shape = self
            return hit_a
        return None

    def bbox(self) -> BBox:
        return self.shape_a.bbox()


class CSGIntersection(Shape):
    """CSG intersection: the overlapping region of shapes A and B."""

    def __init__(self, shape_a: Shape, shape_b: Shape) -> None:
        self.shape_a = shape_a
        self.shape_b = shape_b

    def paths(self) -> list[Path3D]:
        result = []
        for path in self.shape_a.paths():
            filtered = path.filter_inside(self.shape_b)
            result.extend(filtered)
        return result

    def intersect(self, ray: Ray) -> Hit | None:
        hit_a = self.shape_a.intersect(ray)
        hit_b = self.shape_b.intersect(ray)
        if hit_a is None or hit_b is None:
            return None
        # Use the farther hit (entering the intersection)
        if hit_a.t > hit_b.t:
            hit_a.shape = self
            return hit_a
        hit_b.shape = self
        return hit_b

    def bbox(self) -> BBox:
        a = self.shape_a.bbox()
        b = self.shape_b.bbox()
        import numpy as np
        return BBox(np.maximum(a.min, b.min), np.minimum(a.max, b.max))
