"""TransformedShape — applies a 4x4 matrix transform to any shape."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..matrix4 import Mat4, transform_point, transform_direction, transform_points_batch
from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import Vec3, normalize
from .base import Shape


class TransformedShape(Shape):
    """Wraps any shape with a 4×4 world transform matrix.

    Parameters
    ----------
    shape:     The wrapped shape (defined in local/object space).
    transform: 4×4 matrix transforming object space → world space.
    """

    def __init__(self, shape: Shape, transform: Mat4) -> None:
        self.shape = shape
        self.transform = np.asarray(transform, dtype=np.float64)
        try:
            self.inv_transform = np.linalg.inv(self.transform)
        except np.linalg.LinAlgError:
            # Singular matrix: use identity as a safe no-op fallback so that
            # ray intersection still runs (rays pass through unchanged) rather
            # than being incorrectly transformed by the original matrix.
            self.inv_transform = np.eye(4, dtype=np.float64)

    def paths(self) -> list[Path3D]:
        result = []
        for path in self.shape.paths():
            pts_array = np.array(path.points, dtype=np.float64)
            if len(pts_array) == 0:
                continue
            transformed = transform_points_batch(self.transform, pts_array)
            result.append(Path3D(list(transformed)))
        return result

    def intersect(self, ray: Ray) -> Hit | None:
        # Transform ray to object space
        local_origin = transform_point(self.inv_transform, ray.origin)
        local_dir = transform_direction(self.inv_transform, ray.direction)
        local_dir_len = np.linalg.norm(local_dir)
        if local_dir_len < EPSILON:
            return None
        local_dir_norm = local_dir / local_dir_len

        local_ray = Ray(origin=local_origin, direction=local_dir_norm)
        hit = self.shape.intersect(local_ray)
        if hit is None:
            return None

        # Transform hit back to world space
        t_world = hit.t / local_dir_len
        world_point = transform_point(self.transform, hit.point)
        return Hit(shape=self, t=t_world, point=world_point)

    def bbox(self) -> BBox:
        inner = self.shape.bbox()
        # Transform all 8 corners
        corners = np.array([
            [inner.min[0], inner.min[1], inner.min[2]],
            [inner.max[0], inner.min[1], inner.min[2]],
            [inner.min[0], inner.max[1], inner.min[2]],
            [inner.max[0], inner.max[1], inner.min[2]],
            [inner.min[0], inner.min[1], inner.max[2]],
            [inner.max[0], inner.min[1], inner.max[2]],
            [inner.min[0], inner.max[1], inner.max[2]],
            [inner.max[0], inner.max[1], inner.max[2]],
        ], dtype=np.float64)
        transformed = transform_points_batch(self.transform, corners)
        return BBox(transformed.min(axis=0), transformed.max(axis=0))
