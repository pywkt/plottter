"""Axis-aligned bounding box (AABB) with ray intersection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .vector3 import Vec3, vec3
from .ray import Ray, EPSILON


class BBox:
    """Axis-aligned bounding box."""

    def __init__(self, min_pt: Vec3, max_pt: Vec3) -> None:
        self.min = np.asarray(min_pt, dtype=np.float64)
        self.max = np.asarray(max_pt, dtype=np.float64)

    @classmethod
    def empty(cls) -> "BBox":
        return cls(
            vec3(float("inf"), float("inf"), float("inf")),
            vec3(float("-inf"), float("-inf"), float("-inf")),
        )

    @classmethod
    def from_points(cls, points: NDArray[np.float64]) -> "BBox":
        """Create a bounding box enclosing all given points (N, 3)."""
        return cls(points.min(axis=0), points.max(axis=0))

    def expand(self, other: "BBox") -> "BBox":
        """Return the union of this box and another."""
        return BBox(
            np.minimum(self.min, other.min),
            np.maximum(self.max, other.max),
        )

    def pad(self, amount: float = 1e-4) -> "BBox":
        """Expand the box by a small amount in all directions."""
        d = np.full(3, amount, dtype=np.float64)
        return BBox(self.min - d, self.max + d)

    def intersect(self, ray: Ray) -> tuple[float, float] | None:
        """Slab-based ray-AABB intersection test.

        Returns (t_min, t_max) if the ray intersects, else None.
        t_min may be negative (origin inside box).
        """
        inv = np.where(np.abs(ray.direction) < EPSILON, np.inf, 1.0 / ray.direction)
        t1 = (self.min - ray.origin) * inv
        t2 = (self.max - ray.origin) * inv
        t_min = float(np.max(np.minimum(t1, t2)))
        t_max = float(np.min(np.maximum(t1, t2)))
        if t_max < t_min or t_max < 0:
            return None
        return t_min, t_max

    def center(self) -> Vec3:
        return (self.min + self.max) * 0.5

    def size(self) -> Vec3:
        return self.max - self.min

    def longest_axis(self) -> int:
        """Return the index (0=x, 1=y, 2=z) of the longest axis."""
        s = self.size()
        return int(np.argmax(s))

    def contains_point(self, p: Vec3) -> bool:
        return bool(np.all(p >= self.min) and np.all(p <= self.max))

    def __repr__(self) -> str:
        return f"BBox(min={self.min}, max={self.max})"
