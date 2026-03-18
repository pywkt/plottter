"""Cube shape variants: wireframe and striped."""

from __future__ import annotations

import math
import numpy as np

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import vec3, Vec3
from .base import Shape


class Cube(Shape):
    """Wireframe cube (12 edges).

    Parameters
    ----------
    center: Center of the cube.
    size:   Side length.
    """

    def __init__(
        self,
        center: Vec3 | None = None,
        size: float = 1.0,
    ) -> None:
        self.center = vec3(0, 0, 0) if center is None else np.asarray(center, dtype=np.float64)
        self.size = size

    def _corners(self) -> list[Vec3]:
        h = self.size / 2
        c = self.center
        return [
            c + vec3(dx, dy, dz)
            for dx in (-h, h) for dy in (-h, h) for dz in (-h, h)
        ]

    def paths(self) -> list[Path3D]:
        """Return 12 edges of the cube as individual 2-point paths.

        Each edge is shared by two faces.  The ``face_normal`` on each
        ``Path3D`` is the normalised average of the two adjacent face normals:
          - X-axis edge at (sy, sz): adjacent faces are the Y-face (sign sy)
            and Z-face (sign sz) → average normal = normalize(0, sy, sz)
          - Y-axis edge at (sx, sz): adjacent X-face and Z-face
            → normalize(sx, 0, sz)
          - Z-axis edge at (sx, sy): adjacent X-face and Y-face
            → normalize(sx, sy, 0)
        """
        h = self.size / 2
        c = self.center
        corners = {
            (sx, sy, sz): c + vec3(sx * h, sy * h, sz * h)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        }
        edges = []
        # 4 edges along X axis (adjacent Y-face + Z-face)
        for sy in (-1, 1):
            for sz in (-1, 1):
                p = Path3D([corners[(-1, sy, sz)], corners[(1, sy, sz)]])
                n = np.array([0.0, float(sy), float(sz)], dtype=np.float64)
                p.face_normal = n / float(np.linalg.norm(n))
                edges.append(p)
        # 4 edges along Y axis (adjacent X-face + Z-face)
        for sx in (-1, 1):
            for sz in (-1, 1):
                p = Path3D([corners[(sx, -1, sz)], corners[(sx, 1, sz)]])
                n = np.array([float(sx), 0.0, float(sz)], dtype=np.float64)
                p.face_normal = n / float(np.linalg.norm(n))
                edges.append(p)
        # 4 edges along Z axis (adjacent X-face + Y-face)
        for sx in (-1, 1):
            for sy in (-1, 1):
                p = Path3D([corners[(sx, sy, -1)], corners[(sx, sy, 1)]])
                n = np.array([float(sx), float(sy), 0.0], dtype=np.float64)
                p.face_normal = n / float(np.linalg.norm(n))
                edges.append(p)
        return edges

    def intersect(self, ray: Ray) -> Hit | None:
        from ..bbox import BBox
        box = self.bbox()
        result = box.intersect(ray)
        if result is None:
            return None
        t_min, t_max = result
        t = t_min if t_min >= EPSILON else t_max
        if t < EPSILON:
            return None
        return Hit(shape=self, t=t, point=ray.at(t))

    def bbox(self) -> BBox:
        h = self.size / 2
        return BBox(self.center - h, self.center + h)


class StripedCube(Cube):
    """Cube with face stripes for a more artistic plotter look.

    Parameters
    ----------
    stripe_count: Number of stripes per face.
    """

    def __init__(
        self,
        center: Vec3 | None = None,
        size: float = 1.0,
        stripe_count: int = 5,
    ) -> None:
        super().__init__(center, size)
        self.stripe_count = stripe_count

    def paths(self) -> list[Path3D]:
        paths = super().paths()  # start with the 12 edges
        h = self.size / 2
        n = self.stripe_count
        c = self.center

        # Add horizontal stripes on each of the 6 faces
        # Each face is defined by a fixed axis and sign
        # face_defs: (normal_axis, normal_sign, u_axis, v_axis)
        face_defs = [
            (0, +1, 1, 2),  # +X face: u=Y, v=Z
            (0, -1, 1, 2),  # -X face
            (1, +1, 0, 2),  # +Y face: u=X, v=Z
            (1, -1, 0, 2),  # -Y face
            (2, +1, 0, 1),  # +Z face: u=X, v=Y
            (2, -1, 0, 1),  # -Z face
        ]
        for normal_axis, normal_sign, u_axis, v_axis in face_defs:
            face_center = c.copy()
            face_center[normal_axis] += normal_sign * h
            for i in range(1, n):
                t = i / n  # 0..1
                v_val = -h + 2 * h * t
                pt_a = face_center.copy()
                pt_a[v_axis] = c[v_axis] + v_val
                pt_a[u_axis] = c[u_axis] - h
                pt_b = face_center.copy()
                pt_b[v_axis] = c[v_axis] + v_val
                pt_b[u_axis] = c[u_axis] + h
                paths.append(Path3D([pt_a, pt_b]))
        return paths
