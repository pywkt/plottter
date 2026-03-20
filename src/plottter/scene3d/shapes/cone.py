"""Cone shape variants."""

from __future__ import annotations

import math
import numpy as np

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import vec3, Vec3
from .base import Shape


class Cone(Shape):
    """Wireframe cone.

    Parameters
    ----------
    apex:    Tip of the cone.
    base:    Center of the base circle.
    radius:  Radius of the base circle.
    lines:   Number of lines from apex to base.
    """

    def __init__(
        self,
        apex: Vec3 | None = None,
        base: Vec3 | None = None,
        radius: float = 1.0,
        lines: int = 12,
    ) -> None:
        self.apex = vec3(0, 1, 0) if apex is None else np.asarray(apex, dtype=np.float64)
        self.base = vec3(0, 0, 0) if base is None else np.asarray(base, dtype=np.float64)
        self.radius = radius
        self.lines = lines

    def _base_points(self, steps: int | None = None) -> list[Vec3]:
        if steps is None:
            steps = max(32, self.lines * 4)
        pts = []
        for i in range(steps + 1):
            theta = 2 * math.pi * i / steps
            # Build local frame: need up and right vectors in base plane
            axis = self.apex - self.base
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-9:
                axis = vec3(0, 1, 0)
            else:
                axis = axis / axis_len
            # Two orthogonal vectors in the base plane
            up = vec3(0, 1, 0)
            if abs(np.dot(up, axis)) > 0.99:
                up = vec3(1, 0, 0)
            r1 = np.cross(axis, up)
            r1 /= np.linalg.norm(r1)
            r2 = np.cross(axis, r1)
            p = self.base + self.radius * (math.cos(theta) * r1 + math.sin(theta) * r2)
            pts.append(p)
        return pts

    def paths(self) -> list[Path3D]:
        paths: list[Path3D] = []
        base_pts = self._base_points()

        # Compute local frame once
        axis = self.apex - self.base
        height = float(np.linalg.norm(axis))
        axis_dir = axis / max(height, 1e-9)  # unit vector: base → apex
        up = vec3(0, 1, 0)
        if abs(np.dot(up, axis_dir)) > 0.99:
            up = vec3(1, 0, 0)
        r1 = np.cross(axis_dir, up)
        r1 /= max(float(np.linalg.norm(r1)), 1e-9)
        r2 = np.cross(axis_dir, r1)

        # Base circle: normal points away from apex (opposite axis direction)
        p_base = Path3D(base_pts)
        p_base.face_normal = -axis_dir
        paths.append(p_base)

        # Lines from apex to base
        for i in range(self.lines):
            theta = 2 * math.pi * i / self.lines
            radial_dir = math.cos(theta) * r1 + math.sin(theta) * r2
            base_pt = self.base + self.radius * radial_dir
            p = Path3D([self.apex, base_pt])
            # Outward surface normal: tilted radially out and toward the apex
            # n = normalize(height * radial_dir + radius * axis_dir)
            n = height * radial_dir + self.radius * axis_dir
            n_len = float(np.linalg.norm(n))
            if n_len > 1e-9:
                p.face_normal = n / n_len
            paths.append(p)
        return paths

    def intersect(self, ray: Ray) -> Hit | None:
        # Use bounding box as conservative test
        result = self.bbox().intersect(ray)
        if result is None:
            return None
        t_min, t_max = result
        t = t_min if t_min >= EPSILON else t_max
        if t < EPSILON:
            return None
        return Hit(shape=self, t=t, point=ray.at(t))

    def surface_triangles(self) -> list[tuple]:
        """Return 48 world-space triangles: 24 side triangles + 24 base cap triangles."""
        n = 24
        triangles = []

        # Compute local frame (same logic as _base_points but without the loop)
        axis = self.apex - self.base
        axis_len = float(np.linalg.norm(axis))
        axis_dir = axis / max(axis_len, 1e-9)
        up = vec3(0, 1, 0)
        if abs(float(np.dot(up, axis_dir))) > 0.99:
            up = vec3(1, 0, 0)
        r1 = np.cross(axis_dir, up)
        r1 = r1 / max(float(np.linalg.norm(r1)), 1e-9)
        r2 = np.cross(axis_dir, r1)

        for i in range(n):
            theta0 = 2 * math.pi * i / n
            theta1 = 2 * math.pi * (i + 1) / n
            b0 = self.base + self.radius * (math.cos(theta0) * r1 + math.sin(theta0) * r2)
            b1 = self.base + self.radius * (math.cos(theta1) * r1 + math.sin(theta1) * r2)
            # Side triangle: apex → b0 → b1
            triangles.append((self.apex, b0, b1))
            # Base cap triangle (reversed winding for downward normal)
            triangles.append((self.base, b1, b0))

        return triangles

    def bbox(self) -> BBox:
        r = self.radius
        base = self.base
        apex = self.apex
        min_pt = np.minimum(apex, base - r)
        max_pt = np.maximum(apex, base + r)
        return BBox(min_pt, max_pt)


class OutlineCone(Cone):
    """Cone that only renders the two silhouette lines from apex to base edge."""

    def paths(self) -> list[Path3D]:
        paths: list[Path3D] = []
        base_pts = self._base_points()

        # Compute local frame
        axis = self.apex - self.base
        height = float(np.linalg.norm(axis))
        axis_dir = axis / max(height, 1e-9)
        up = vec3(0, 1, 0)
        if abs(np.dot(up, axis_dir)) > 0.99:
            up = vec3(1, 0, 0)
        r1 = np.cross(axis_dir, up)
        r1 /= max(float(np.linalg.norm(r1)), 1e-9)

        # Base circle: normal points away from apex
        p_base = Path3D(base_pts)
        p_base.face_normal = -axis_dir
        paths.append(p_base)

        # Only 2 outline lines (left and right silhouette)
        for sign in (-1, 1):
            radial_dir = sign * r1
            base_pt = self.base + self.radius * radial_dir
            p = Path3D([self.apex, base_pt])
            n = height * radial_dir + self.radius * axis_dir
            n_len = float(np.linalg.norm(n))
            if n_len > 1e-9:
                p.face_normal = n / n_len
            paths.append(p)
        return paths
