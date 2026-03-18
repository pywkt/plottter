"""Sphere shape variants: wireframe, outline-only, and shaded."""

from __future__ import annotations

import math
import numpy as np

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import vec3, Vec3, normalize, dot
from .base import Shape


class Sphere(Shape):
    """Wireframe sphere with latitude/longitude lines.

    Parameters
    ----------
    center:   Center point in world space.
    radius:   Sphere radius.
    lat_lines: Number of latitude circles (horizontal).
    lng_lines: Number of longitude lines (vertical).
    """

    def __init__(
        self,
        center: Vec3 | None = None,
        radius: float = 1.0,
        lat_lines: int = 10,
        lng_lines: int = 10,
    ) -> None:
        self.center = vec3(0, 0, 0) if center is None else np.asarray(center, dtype=np.float64)
        self.radius = radius
        self.lat_lines = lat_lines
        self.lng_lines = lng_lines

    def paths(self) -> list[Path3D]:
        paths: list[Path3D] = []
        # Latitude circles (excluding poles)
        for i in range(1, self.lat_lines):
            phi = math.pi * i / self.lat_lines  # 0..pi
            r = self.radius * math.sin(phi)
            z = self.center[2] + self.radius * math.cos(phi)
            pts = []
            steps = max(32, self.lng_lines * 4)
            for j in range(steps + 1):
                theta = 2 * math.pi * j / steps
                x = self.center[0] + r * math.cos(theta)
                y = self.center[1] + r * math.sin(theta)
                pts.append(vec3(x, y, z))
            p = Path3D(pts)
            # Approximate face normal: outward surface normal at path midpoint.
            mid = pts[len(pts) // 2]
            diff = mid - self.center
            diff_len = float(np.linalg.norm(diff))
            if diff_len > 1e-12:
                p.face_normal = diff / diff_len
            paths.append(p)
        # Longitude arcs
        for j in range(self.lng_lines):
            theta = 2 * math.pi * j / self.lng_lines
            pts = []
            steps = max(32, self.lat_lines * 4)
            for i in range(steps + 1):
                phi = math.pi * i / steps
                x = self.center[0] + self.radius * math.sin(phi) * math.cos(theta)
                y = self.center[1] + self.radius * math.sin(phi) * math.sin(theta)
                z = self.center[2] + self.radius * math.cos(phi)
                pts.append(vec3(x, y, z))
            p = Path3D(pts)
            # Approximate face normal: outward surface normal at path midpoint.
            mid = pts[len(pts) // 2]
            diff = mid - self.center
            diff_len = float(np.linalg.norm(diff))
            if diff_len > 1e-12:
                p.face_normal = diff / diff_len
            paths.append(p)
        return paths

    def intersect(self, ray: Ray) -> Hit | None:
        """Analytic ray-sphere intersection."""
        oc = ray.origin - self.center
        a = float(np.dot(ray.direction, ray.direction))
        b = 2.0 * float(np.dot(oc, ray.direction))
        c = float(np.dot(oc, oc)) - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t = (-b - sqrt_disc) / (2 * a)
        if t < EPSILON:
            t = (-b + sqrt_disc) / (2 * a)
        if t < EPSILON:
            return None
        return Hit(shape=self, t=t, point=ray.at(t))

    def bbox(self) -> BBox:
        r = self.radius
        return BBox(self.center - r, self.center + r)


class OutlineSphere(Sphere):
    """Sphere that only renders the silhouette outline relative to a view direction."""

    def __init__(
        self,
        center: Vec3 | None = None,
        radius: float = 1.0,
        detail: int = 64,
        view_dir: Vec3 | None = None,
    ) -> None:
        super().__init__(center, radius)
        self.detail = detail
        self.view_dir = normalize(view_dir) if view_dir is not None else vec3(0, 0, -1)

    def paths(self) -> list[Path3D]:
        # Silhouette is a circle perpendicular to the view direction
        # Find two orthogonal vectors in the silhouette plane
        up = vec3(0, 1, 0)
        if abs(dot(up, self.view_dir)) > 0.99:
            up = vec3(1, 0, 0)
        r1 = normalize(np.cross(self.view_dir, up))
        r2 = normalize(np.cross(self.view_dir, r1))
        pts = []
        for i in range(self.detail + 1):
            theta = 2 * math.pi * i / self.detail
            p = self.center + self.radius * (math.cos(theta) * r1 + math.sin(theta) * r2)
            pts.append(p)
        return [Path3D(pts)]


class ShadedSphere(Shape):
    """Sphere with line-density shading based on a light direction.

    Brighter (lit) areas have sparser lines; darker (shadow) areas have
    denser lines. All lines are latitude-like circles.
    """

    def __init__(
        self,
        center: Vec3 | None = None,
        radius: float = 1.0,
        light_dir: Vec3 | None = None,
        min_lines: int = 10,
        max_lines: int = 40,
    ) -> None:
        self.center = vec3(0, 0, 0) if center is None else np.asarray(center, dtype=np.float64)
        self.radius = radius
        self.light_dir = normalize(light_dir) if light_dir is not None else normalize(vec3(1, 1, -1))
        self.min_lines = min_lines
        self.max_lines = max_lines

    def set_light_dir(self, light_dir: Vec3) -> None:
        """Update the light direction vector (normalized in-place)."""
        self.light_dir = normalize(np.asarray(light_dir, dtype=np.float64))

    def paths(self) -> list[Path3D]:
        paths: list[Path3D] = []
        total = self.max_lines
        # Find the "equator" axis perpendicular to light
        up = vec3(0, 1, 0)
        if abs(dot(up, self.light_dir)) > 0.99:
            up = vec3(1, 0, 0)
        axis = normalize(np.cross(self.light_dir, up))  # rotation axis

        for i in range(total + 1):
            t = i / total
            # Map t to angle -pi/2..pi/2 (from shadow to lit hemisphere)
            phi = math.pi * t - math.pi / 2
            # Compute circle on sphere at this "latitude" relative to light axis
            r = self.radius * math.cos(phi)
            if r < 1e-6:
                continue
            center_offset = self.light_dir * self.radius * math.sin(phi)
            c = self.center + center_offset

            # Compute line density: more lines in shadow (t near 0), fewer near t=1
            density = 1.0 - t  # 1.0 at shadow, 0.0 at highlight
            if density < 0.05:
                continue  # skip near-highlight lines

            steps = max(16, int(32 + 32 * density))
            r2 = normalize(np.cross(self.light_dir, axis))
            pts = []
            for j in range(steps + 1):
                theta = 2 * math.pi * j / steps
                p = c + r * (math.cos(theta) * axis + math.sin(theta) * r2)
                pts.append(p)
            paths.append(Path3D(pts))
        return paths

    def intersect(self, ray: Ray) -> Hit | None:
        oc = ray.origin - self.center
        a = float(np.dot(ray.direction, ray.direction))
        b = 2.0 * float(np.dot(oc, ray.direction))
        c = float(np.dot(oc, oc)) - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t = (-b - sqrt_disc) / (2 * a)
        if t < EPSILON:
            t = (-b + sqrt_disc) / (2 * a)
        if t < EPSILON:
            return None
        return Hit(shape=self, t=t, point=ray.at(t))

    def bbox(self) -> BBox:
        r = self.radius
        return BBox(self.center - r, self.center + r)
