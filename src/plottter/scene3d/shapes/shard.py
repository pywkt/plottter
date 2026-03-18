"""Shard shape — a double pyramid (bipyramid)."""

from __future__ import annotations

import math
import numpy as np

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import vec3, Vec3
from .base import Shape
from .triangle import Triangle


class Shard(Shape):
    """Double pyramid (bipyramid) shape.

    Two pyramids base-to-base, creating a diamond/shard shape.

    Parameters
    ----------
    center:  Center point.
    radius:  Equatorial radius.
    height:  Total height (split equally above/below center).
    sides:   Number of sides on the equatorial polygon.
    """

    def __init__(
        self,
        center: Vec3 | None = None,
        radius: float = 1.0,
        height: float = 2.0,
        sides: int = 6,
    ) -> None:
        self.center = vec3(0, 0, 0) if center is None else np.asarray(center, dtype=np.float64)
        self.radius = radius
        self.height = height
        self.sides = sides

    def _equatorial_pts(self) -> list[Vec3]:
        c = self.center
        pts = []
        for i in range(self.sides):
            theta = 2 * math.pi * i / self.sides
            pts.append(vec3(
                c[0] + self.radius * math.cos(theta),
                c[1],
                c[2] + self.radius * math.sin(theta),
            ))
        return pts

    def paths(self) -> list[Path3D]:
        c = self.center
        apex_top = c + vec3(0, self.height / 2, 0)
        apex_bot = c - vec3(0, self.height / 2, 0)
        eq_pts = self._equatorial_pts()
        n = len(eq_pts)
        paths: list[Path3D] = []

        # Equatorial ring: leave face_normal as None (edge shared between both pyramids)
        paths.append(Path3D(eq_pts + [eq_pts[0]]))

        # Top pyramid edges: apex_top → eq_pts[i]
        # Each edge borders two triangular faces of the top pyramid.
        # Outward normal for face (apex_top, eq_pts[i], eq_pts[i+1]):
        #   n = cross(eq_pts[i+1] - apex_top, eq_pts[i] - apex_top)
        # Average the normals of the left and right adjacent faces.
        for i in range(n):
            ep = eq_pts[i]
            v_cur = ep - apex_top
            v_next = eq_pts[(i + 1) % n] - apex_top
            v_prev = eq_pts[(i - 1) % n] - apex_top
            # Right face (apex_top, eq_pts[i], eq_pts[(i+1)%n])
            n_right = np.cross(v_next, v_cur)
            nr_len = float(np.linalg.norm(n_right))
            if nr_len > 1e-9:
                n_right = n_right / nr_len
            # Left face (apex_top, eq_pts[(i-1)%n], eq_pts[i])
            n_left = np.cross(v_cur, v_prev)
            nl_len = float(np.linalg.norm(n_left))
            if nl_len > 1e-9:
                n_left = n_left / nl_len
            avg = n_right + n_left
            avg_len = float(np.linalg.norm(avg))
            path = Path3D([apex_top, ep])
            if avg_len > 1e-9:
                path.face_normal = avg / avg_len
            paths.append(path)

        # Bottom pyramid edges: apex_bot → eq_pts[i]
        # Outward normal for face (apex_bot, eq_pts[i], eq_pts[i+1]):
        #   n = cross(eq_pts[i] - apex_bot, eq_pts[i+1] - apex_bot)
        for i in range(n):
            ep = eq_pts[i]
            v_cur = ep - apex_bot
            v_next = eq_pts[(i + 1) % n] - apex_bot
            v_prev = eq_pts[(i - 1) % n] - apex_bot
            # Right face (apex_bot, eq_pts[i], eq_pts[(i+1)%n])
            n_right = np.cross(v_cur, v_next)
            nr_len = float(np.linalg.norm(n_right))
            if nr_len > 1e-9:
                n_right = n_right / nr_len
            # Left face (apex_bot, eq_pts[(i-1)%n], eq_pts[i])
            n_left = np.cross(v_prev, v_cur)
            nl_len = float(np.linalg.norm(n_left))
            if nl_len > 1e-9:
                n_left = n_left / nl_len
            avg = n_right + n_left
            avg_len = float(np.linalg.norm(avg))
            path = Path3D([apex_bot, ep])
            if avg_len > 1e-9:
                path.face_normal = avg / avg_len
            paths.append(path)

        return paths

    def _triangles(self) -> list[Triangle]:
        c = self.center
        apex_top = c + vec3(0, self.height / 2, 0)
        apex_bot = c - vec3(0, self.height / 2, 0)
        eq_pts = self._equatorial_pts()
        tris = []
        n = len(eq_pts)
        for i in range(n):
            a = eq_pts[i]
            b = eq_pts[(i + 1) % n]
            tris.append(Triangle(apex_top, a, b, draw_edges=False))
            tris.append(Triangle(apex_bot, a, b, draw_edges=False))
        return tris

    def intersect(self, ray: Ray) -> Hit | None:
        closest: Hit | None = None
        for tri in self._triangles():
            hit = tri.intersect(ray)
            if hit is not None:
                hit.shape = self
                if closest is None or hit.t < closest.t:
                    closest = hit
        return closest

    def bbox(self) -> BBox:
        c = self.center
        h = self.height / 2
        r = self.radius
        return BBox(vec3(c[0] - r, c[1] - h, c[2] - r),
                    vec3(c[0] + r, c[1] + h, c[2] + r))
