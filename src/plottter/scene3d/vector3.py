"""3D vector operations using NumPy arrays.

Vectors are plain np.ndarray of shape (3,). This module provides
convenience functions for common 3D operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Vec3 = NDArray[np.float64]  # shape (3,)


def vec3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=np.float64)


def normalize(v: Vec3) -> Vec3:
    n = np.linalg.norm(v)
    if n == 0.0:
        return v.copy()
    return v / n


def dot(a: Vec3, b: Vec3) -> float:
    return float(np.dot(a, b))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return np.cross(a, b)


def length(v: Vec3) -> float:
    return float(np.linalg.norm(v))


def lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a + (b - a) * t
