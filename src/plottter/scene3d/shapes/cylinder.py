"""Cylinder shape variants."""

from __future__ import annotations

import math
import numpy as np

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import vec3, Vec3
from .base import Shape


class Cylinder(Shape):
    """Wireframe cylinder.

    Parameters
    ----------
    bottom:  Center of the bottom circle.
    top:     Center of the top circle.
    radius:  Cylinder radius.
    lines:   Number of vertical lines.
    """

    def __init__(
        self,
        bottom: Vec3 | None = None,
        top: Vec3 | None = None,
        radius: float = 1.0,
        lines: int = 12,
    ) -> None:
        self.bottom = vec3(0, -1, 0) if bottom is None else np.asarray(bottom, dtype=np.float64)
        self.top = vec3(0, 1, 0) if top is None else np.asarray(top, dtype=np.float64)
        self.radius = radius
        self.lines = lines

    def _local_frame(self) -> tuple:
        axis = self.top - self.bottom
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-9:
            axis = vec3(0, 1, 0)
        else:
            axis = axis / axis_len
        up = vec3(0, 1, 0)
        if abs(np.dot(up, axis)) > 0.99:
            up = vec3(1, 0, 0)
        r1 = np.cross(axis, up)
        r1 /= np.linalg.norm(r1)
        r2 = np.cross(axis, r1)
        return r1, r2

    def _circle_pts(self, center: Vec3, steps: int = 48) -> list[Vec3]:
        r1, r2 = self._local_frame()
        pts = []
        for i in range(steps + 1):
            theta = 2 * math.pi * i / steps
            p = center + self.radius * (math.cos(theta) * r1 + math.sin(theta) * r2)
            pts.append(p)
        return pts

    def paths(self) -> list[Path3D]:
        paths: list[Path3D] = []

        # Compute axis direction (bottom → top) for cap normals
        axis = self.top - self.bottom
        axis_len = float(np.linalg.norm(axis))
        axis_dir = axis / max(axis_len, 1e-9)

        # Bottom cap: normal points away from top (downward)
        p_bottom = Path3D(self._circle_pts(self.bottom))
        p_bottom.face_normal = -axis_dir
        paths.append(p_bottom)

        # Top cap: normal points away from bottom (upward)
        p_top = Path3D(self._circle_pts(self.top))
        p_top.face_normal = axis_dir
        paths.append(p_top)

        # Lateral lines: normal is radially outward
        r1, r2 = self._local_frame()
        for i in range(self.lines):
            theta = 2 * math.pi * i / self.lines
            radial_dir = math.cos(theta) * r1 + math.sin(theta) * r2
            offset = self.radius * radial_dir
            p = Path3D([self.bottom + offset, self.top + offset])
            p.face_normal = radial_dir  # already unit length (r1, r2 are unit vectors)
            paths.append(p)
        return paths

    def intersect(self, ray: Ray) -> Hit | None:
        result = self.bbox().intersect(ray)
        if result is None:
            return None
        t_min, t_max = result
        t = t_min if t_min >= EPSILON else t_max
        if t < EPSILON:
            return None
        return Hit(shape=self, t=t, point=ray.at(t))

    def bbox(self) -> BBox:
        r = self.radius
        min_pt = np.minimum(self.bottom, self.top) - r
        max_pt = np.maximum(self.bottom, self.top) + r
        return BBox(min_pt, max_pt)


class OutlineCylinder(Cylinder):
    """Cylinder that only renders silhouette lines (2 vertical + 2 circles)."""

    def paths(self) -> list[Path3D]:
        paths: list[Path3D] = []

        axis = self.top - self.bottom
        axis_len = float(np.linalg.norm(axis))
        axis_dir = axis / max(axis_len, 1e-9)

        p_bottom = Path3D(self._circle_pts(self.bottom))
        p_bottom.face_normal = -axis_dir
        paths.append(p_bottom)

        p_top = Path3D(self._circle_pts(self.top))
        p_top.face_normal = axis_dir
        paths.append(p_top)

        r1, r2 = self._local_frame()
        for sign in (-1, 1):
            radial_dir = sign * r1
            offset = self.radius * radial_dir
            p = Path3D([self.bottom + offset, self.top + offset])
            p.face_normal = radial_dir
            paths.append(p)
        return paths
