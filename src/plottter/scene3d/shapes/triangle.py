"""Single triangle shape."""

from __future__ import annotations

import numpy as np

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import Vec3, vec3
from .base import Shape


class Triangle(Shape):
    """A single triangle in 3D space.

    Parameters
    ----------
    v0, v1, v2: Triangle vertices in world space.
    draw_edges: If True, draw all 3 edges (default).
    """

    def __init__(
        self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        draw_edges: bool = True,
    ) -> None:
        self.v0 = np.asarray(v0, dtype=np.float64)
        self.v1 = np.asarray(v1, dtype=np.float64)
        self.v2 = np.asarray(v2, dtype=np.float64)
        self.draw_edges = draw_edges

    def paths(self) -> list[Path3D]:
        if self.draw_edges:
            return [Path3D([self.v0, self.v1, self.v2, self.v0])]
        return []

    def intersect(self, ray: Ray) -> Hit | None:
        """Moller-Trumbore ray-triangle intersection."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = np.cross(ray.direction, edge2)
        a = float(np.dot(edge1, h))
        if abs(a) < EPSILON:
            return None  # parallel
        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * float(np.dot(s, h))
        if u < 0.0 or u > 1.0:
            return None
        q = np.cross(s, edge1)
        v = f * float(np.dot(ray.direction, q))
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * float(np.dot(edge2, q))
        if t < EPSILON:
            return None
        return Hit(shape=self, t=t, point=ray.at(t))

    def bbox(self) -> BBox:
        pts = np.stack([self.v0, self.v1, self.v2])
        return BBox(pts.min(axis=0), pts.max(axis=0)).pad()
