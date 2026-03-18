"""4x4 matrix math for 3D transformations.

Matrices are np.ndarray of shape (4, 4). Column-major convention
(OpenGL-style): points are row vectors multiplied on the right.
Transform a point p with: p_h = [px, py, pz, 1] @ M
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .vector3 import Vec3, normalize, cross, dot

Mat4 = NDArray[np.float64]  # shape (4, 4)


def identity() -> Mat4:
    return np.eye(4, dtype=np.float64)


def translate(tx: float, ty: float, tz: float) -> Mat4:
    m = np.eye(4, dtype=np.float64)
    m[3, 0] = tx
    m[3, 1] = ty
    m[3, 2] = tz
    return m


def scale(sx: float, sy: float, sz: float) -> Mat4:
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_x(angle_rad: float) -> Mat4:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[1, 1] = c;  m[1, 2] = s
    m[2, 1] = -s; m[2, 2] = c
    return m


def rotate_y(angle_rad: float) -> Mat4:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c;  m[0, 2] = -s
    m[2, 0] = s;  m[2, 2] = c
    return m


def rotate_z(angle_rad: float) -> Mat4:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c;  m[0, 1] = s
    m[1, 0] = -s; m[1, 1] = c
    return m


def rotate_xyz(rx: float, ry: float, rz: float) -> Mat4:
    """Compose X then Y then Z rotation (Euler angles in radians)."""
    return rotate_x(rx) @ rotate_y(ry) @ rotate_z(rz)


def look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4:
    """Build a view matrix (world→camera transform).

    Parameters
    ----------
    eye:    Camera position in world space.
    center: Point the camera is looking at.
    up:     World up vector (usually (0,1,0)).

    Returns
    -------
    4×4 view matrix such that p_view = p_world @ M.
    """
    f = normalize(center - eye)  # forward
    r = normalize(cross(f, up))  # right
    u = cross(r, f)              # recomputed up

    m = np.array([
        [r[0],  r[1],  r[2],  -dot(r, eye)],
        [u[0],  u[1],  u[2],  -dot(u, eye)],
        [-f[0], -f[1], -f[2],  dot(f, eye)],
        [0.0,   0.0,   0.0,   1.0],
    ], dtype=np.float64).T  # transpose so it's row-vector convention
    return m


def perspective(fov_rad: float, aspect: float, near: float, far: float) -> Mat4:
    """Build a perspective projection matrix (clip-space convention: z in [-1, 1]).

    Parameters
    ----------
    fov_rad: Vertical field of view in radians.
    aspect:  Width / height.
    near:    Near clip distance (positive).
    far:     Far clip distance (positive).
    """
    if fov_rad <= 0 or fov_rad >= math.pi:
        raise ValueError(f"perspective: fov_rad must be in (0, pi), got {fov_rad}")
    if abs(near - far) < 1e-9:
        raise ValueError(f"perspective: near and far must be distinct, got near={near}, far={far}")
    f = 1.0 / math.tan(fov_rad / 2.0)
    nf = near - far
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / nf
    m[2, 3] = -1.0
    m[3, 2] = (2 * far * near) / nf
    return m


def orthographic(left: float, right: float, bottom: float, top: float,
                 near: float, far: float) -> Mat4:
    """Build an orthographic projection matrix."""
    if abs(right - left) < 1e-9:
        raise ValueError(f"orthographic: left and right must be distinct, got left={left}, right={right}")
    if abs(top - bottom) < 1e-9:
        raise ValueError(f"orthographic: top and bottom must be distinct, got top={top}, bottom={bottom}")
    if abs(far - near) < 1e-9:
        raise ValueError(f"orthographic: near and far must be distinct, got near={near}, far={far}")
    rl = right - left
    tb = top - bottom
    fn = far - near
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 2.0 / rl
    m[1, 1] = 2.0 / tb
    m[2, 2] = -2.0 / fn
    m[3, 0] = -(right + left) / rl
    m[3, 1] = -(top + bottom) / tb
    m[3, 2] = -(far + near) / fn
    m[3, 3] = 1.0
    return m


def multiply(*matrices: Mat4) -> Mat4:
    """Multiply a sequence of matrices left-to-right."""
    result = matrices[0]
    for m in matrices[1:]:
        result = result @ m
    return result


def transform_point(m: Mat4, p: Vec3) -> Vec3:
    """Apply a 4×4 matrix to a 3D point (w=1)."""
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    result = ph @ m
    if result[3] != 0.0:
        return result[:3] / result[3]
    return result[:3]


def transform_direction(m: Mat4, d: Vec3) -> Vec3:
    """Apply a 4×4 matrix to a direction vector (w=0)."""
    ph = np.array([d[0], d[1], d[2], 0.0], dtype=np.float64)
    return (ph @ m)[:3]


def transform_points_batch(m: Mat4, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform an (N, 3) array of points by matrix m.

    Returns an (N, 3) array of transformed points.
    """
    n = len(points)
    homogeneous = np.ones((n, 4), dtype=np.float64)
    homogeneous[:, :3] = points
    result = homogeneous @ m
    w = result[:, 3:4]
    # Avoid division by zero
    w = np.where(np.abs(w) < 1e-12, 1.0, w)
    return result[:, :3] / w
