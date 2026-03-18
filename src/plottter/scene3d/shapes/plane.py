"""Plane shape variants: flat grid and terrain heightfield."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import vec3, Vec3
from .base import Shape


class Plane(Shape):
    """Flat grid plane in the XZ plane.

    Parameters
    ----------
    center:  Center of the plane.
    size:    Width and depth (half-extent in each direction).
    steps:   Number of grid divisions in each direction.
    """

    def __init__(
        self,
        center: Vec3 | None = None,
        size: float = 4.0,
        steps: int = 8,
    ) -> None:
        self.center = vec3(0, 0, 0) if center is None else np.asarray(center, dtype=np.float64)
        self.size = size
        self.steps = steps

    def paths(self) -> list[Path3D]:
        paths: list[Path3D] = []
        h = self.size / 2
        n = self.steps
        c = self.center
        # Flat plane lies in XZ at y=c[1]; normal points upward (+Y)
        normal = np.array([0.0, 1.0, 0.0])
        # Lines parallel to X axis
        for i in range(n + 1):
            t = i / n
            z = c[2] - h + 2 * h * t
            p = Path3D([
                vec3(c[0] - h, c[1], z),
                vec3(c[0] + h, c[1], z),
            ])
            p.face_normal = normal
            paths.append(p)
        # Lines parallel to Z axis
        for i in range(n + 1):
            t = i / n
            x = c[0] - h + 2 * h * t
            p = Path3D([
                vec3(x, c[1], c[2] - h),
                vec3(x, c[1], c[2] + h),
            ])
            p.face_normal = normal
            paths.append(p)
        return paths

    def intersect(self, ray: Ray) -> Hit | None:
        # Infinite plane at y = center[1]
        if abs(ray.direction[1]) < EPSILON:
            return None
        t = (self.center[1] - ray.origin[1]) / ray.direction[1]
        if t < EPSILON:
            return None
        p = ray.at(t)
        h = self.size / 2
        if (abs(p[0] - self.center[0]) <= h and abs(p[2] - self.center[2]) <= h):
            return Hit(shape=self, t=t, point=p)
        return None

    def bbox(self) -> BBox:
        h = self.size / 2
        c = self.center
        return BBox(vec3(c[0] - h, c[1] - 0.001, c[2] - h),
                    vec3(c[0] + h, c[1] + 0.001, c[2] + h))


class TerrainPlane(Shape):
    """Terrain height field rendered as a grid with noise-based height.

    Parameters
    ----------
    center:     Center of the plane.
    size:       Width and depth.
    steps:      Grid resolution.
    height_fn:  Optional callable (x, z) -> y. If None, uses Perlin noise.
    max_height: Maximum terrain height.
    """

    def __init__(
        self,
        center: Vec3 | None = None,
        size: float = 4.0,
        steps: int = 20,
        height_fn=None,
        max_height: float = 1.0,
    ) -> None:
        self.center = vec3(0, 0, 0) if center is None else np.asarray(center, dtype=np.float64)
        self.size = size
        self.steps = steps
        self.max_height = max_height
        self._height_fn = height_fn

    def _height(self, x: float, z: float) -> float:
        if self._height_fn is not None:
            return self._height_fn(x, z)
        # Fallback to simple sine-based terrain
        return self.max_height * 0.5 * (
            math.sin(x * 1.2) * math.cos(z * 0.9) +
            math.sin(x * 2.3 + 0.5) * math.sin(z * 2.1)
        )

    def _grid_pts(self) -> NDArray[np.float64]:
        h = self.size / 2
        n = self.steps
        xs = np.linspace(-h, h, n + 1) + self.center[0]
        zs = np.linspace(-h, h, n + 1) + self.center[2]
        pts = np.zeros((n + 1, n + 1, 3), dtype=np.float64)
        for i, x in enumerate(xs):
            for j, z in enumerate(zs):
                y = self._height(x, z) + self.center[1]
                pts[i, j] = [x, y, z]
        return pts

    def _compute_normals(self, grid: "NDArray[np.float64]") -> "NDArray[np.float64]":
        """Compute upward-pointing surface normal at each grid vertex via central differences."""
        n = self.steps
        normals = np.zeros((n + 1, n + 1, 3), dtype=np.float64)
        for i in range(n + 1):
            for j in range(n + 1):
                dx = grid[min(i + 1, n), j] - grid[max(i - 1, 0), j]
                dz = grid[i, min(j + 1, n)] - grid[i, max(j - 1, 0)]
                # cross(dz, dx) points upward for a terrain in the XZ plane
                nv = np.cross(dz, dx)
                nv_len = float(np.linalg.norm(nv))
                normals[i, j] = nv / nv_len if nv_len > 1e-9 else np.array([0.0, 1.0, 0.0])
        return normals

    def paths(self) -> list[Path3D]:
        grid = self._grid_pts()
        normals = self._compute_normals(grid)
        n = self.steps
        paths: list[Path3D] = []
        # Lines along X direction (varying i, fixed j)
        for j in range(n + 1):
            pts = [grid[i, j] for i in range(n + 1)]
            p = Path3D(pts)
            # Average normal across all vertices on this path
            avg_n = normals[:, j, :].mean(axis=0)
            avg_len = float(np.linalg.norm(avg_n))
            p.face_normal = avg_n / avg_len if avg_len > 1e-9 else np.array([0.0, 1.0, 0.0])
            paths.append(p)
        # Lines along Z direction (varying j, fixed i)
        for i in range(n + 1):
            pts = [grid[i, j] for j in range(n + 1)]
            p = Path3D(pts)
            avg_n = normals[i, :, :].mean(axis=0)
            avg_len = float(np.linalg.norm(avg_n))
            p.face_normal = avg_n / avg_len if avg_len > 1e-9 else np.array([0.0, 1.0, 0.0])
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
        h = self.size / 2
        c = self.center
        return BBox(
            vec3(c[0] - h, c[1] - abs(self.max_height) - 0.001, c[2] - h),
            vec3(c[0] + h, c[1] + abs(self.max_height) + 0.001, c[2] + h),
        )
