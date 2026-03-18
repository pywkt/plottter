"""Parametric math function shape."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit
from ..vector3 import vec3, Vec3
from .base import Shape


class FunctionShape(Shape):
    """Parametric function rendered as a 3D line art grid.

    The function f(u, v) -> (x, y, z) is evaluated over a 2D parameter space.
    Lines are drawn along constant-u and constant-v curves.

    Parameters
    ----------
    fn:        Callable (u, v) -> (x, y, z) or np.ndarray of shape (3,).
    u_range:   (u_min, u_max) parameter range.
    v_range:   (v_min, v_max) parameter range.
    u_steps:   Number of u parameter samples.
    v_steps:   Number of v parameter samples.
    """

    def __init__(
        self,
        fn: Callable[[float, float], tuple[float, float, float]] | None = None,
        u_range: tuple[float, float] = (0.0, 1.0),
        v_range: tuple[float, float] = (0.0, 1.0),
        u_steps: int = 20,
        v_steps: int = 20,
    ) -> None:
        if fn is None:
            # Default: saddle surface
            def fn(u: float, v: float) -> tuple[float, float, float]:
                return (u, u * v, v)
        self.fn = fn
        self.u_range = u_range
        self.v_range = v_range
        self.u_steps = u_steps
        self.v_steps = v_steps
        self._cached_grid: np.ndarray | None = None

    def _build_grid(self) -> np.ndarray:
        if self._cached_grid is not None:
            return self._cached_grid
        us = np.linspace(self.u_range[0], self.u_range[1], self.u_steps + 1)
        vs = np.linspace(self.v_range[0], self.v_range[1], self.v_steps + 1)
        grid = np.zeros((len(us), len(vs), 3), dtype=np.float64)
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                try:
                    result = self.fn(u, v)
                except Exception as exc:
                    raise ValueError(
                        f"FunctionShape: fn({u!r}, {v!r}) raised an exception: {exc}"
                    ) from exc
                result_arr = np.asarray(result, dtype=np.float64)
                if result_arr.shape != (3,):
                    raise ValueError(
                        f"FunctionShape: fn({u!r}, {v!r}) must return a sequence of length 3, "
                        f"got shape {result_arr.shape}"
                    )
                if not np.all(np.isfinite(result_arr)):
                    raise ValueError(
                        f"FunctionShape: fn({u!r}, {v!r}) returned non-finite values: {result_arr}"
                    )
                grid[i, j] = result_arr
        self._cached_grid = grid
        return grid

    def paths(self) -> list[Path3D]:
        grid = self._build_grid()
        nu, nv = grid.shape[:2]
        paths: list[Path3D] = []
        # Lines along constant u
        for i in range(nu):
            pts = [grid[i, j] for j in range(nv)]
            paths.append(Path3D(pts))
        # Lines along constant v
        for j in range(nv):
            pts = [grid[i, j] for i in range(nu)]
            paths.append(Path3D(pts))
        return paths

    def intersect(self, ray: Ray) -> Hit | None:
        # Conservative bounding box test only
        result = self.bbox().intersect(ray)
        if result is None:
            return None
        t_min, t_max = result
        from ..ray import EPSILON
        t = t_min if t_min >= EPSILON else t_max
        if t < EPSILON:
            return None
        return Hit(shape=self, t=t, point=ray.at(t))

    def bbox(self) -> BBox:
        grid = self._build_grid()
        pts = grid.reshape(-1, 3)
        return BBox(pts.min(axis=0), pts.max(axis=0)).pad()
