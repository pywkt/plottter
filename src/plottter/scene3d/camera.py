"""Camera setup for the scene3d renderer.

Supports perspective and orthographic projections.
Camera can be configured either via explicit eye/center/up or
via orbit parameters (azimuth, elevation, distance).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .vector3 import Vec3, vec3, normalize
from .matrix4 import Mat4, look_at, perspective, orthographic, multiply


ProjectionType = Literal["perspective", "orthographic"]


class Camera:
    """Camera with perspective or orthographic projection.

    Parameters
    ----------
    projection:  "perspective" or "orthographic".
    fov_deg:     Vertical field of view in degrees (perspective only).
    ortho_scale: Half-height of the orthographic view volume.
    near:        Near clip plane distance.
    far:         Far clip plane distance.
    aspect:      Width / height ratio of the viewport.
    """

    def __init__(
        self,
        projection: ProjectionType = "perspective",
        fov_deg: float = 45.0,
        ortho_scale: float = 3.0,
        near: float = 0.1,
        far: float = 1000.0,
        aspect: float = 1.0,
    ) -> None:
        self.projection = projection
        self.fov_deg = fov_deg
        self.ortho_scale = ortho_scale
        self.near = near
        self.far = far
        self.aspect = aspect

        # Will be set via look_at or orbit
        self.eye = vec3(0, 0, 5)
        self.center = vec3(0, 0, 0)
        self.up = vec3(0, 1, 0)

    def set_look_at(self, eye: Vec3, center: Vec3, up: Vec3) -> "Camera":
        self.eye = np.asarray(eye, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        return self

    def set_orbit(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        center: Vec3 | None = None,
    ) -> "Camera":
        """Set camera from spherical coordinates.

        Parameters
        ----------
        azimuth_deg:   Horizontal angle in degrees (0 = +Z axis, 90 = +X axis).
        elevation_deg: Vertical angle in degrees (-90 = below, 90 = above).
        distance:      Distance from center.
        center:        Look-at point. Defaults to origin.
        """
        if center is not None:
            self.center = np.asarray(center, dtype=np.float64)

        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        x = distance * math.cos(el) * math.sin(az)
        y = distance * math.sin(el)
        z = distance * math.cos(el) * math.cos(az)
        self.eye = self.center + vec3(x, y, z)
        # Compute a stable up vector
        if abs(elevation_deg) >= 89.9:
            # At poles, use a different up to avoid gimbal lock
            self.up = vec3(math.cos(az), 0, math.sin(az)) if elevation_deg > 0 else vec3(-math.cos(az), 0, -math.sin(az))
        else:
            self.up = vec3(0, 1, 0)
        return self

    def view_matrix(self) -> Mat4:
        return look_at(self.eye, self.center, self.up)

    def projection_matrix(self) -> Mat4:
        if self.projection == "perspective":
            return perspective(math.radians(self.fov_deg), self.aspect, self.near, self.far)
        else:
            h = self.ortho_scale
            w = h * self.aspect
            return orthographic(-w, w, -h, h, self.near, self.far)

    def view_proj_matrix(self) -> Mat4:
        """Combined view × projection matrix."""
        return multiply(self.view_matrix(), self.projection_matrix())

    @classmethod
    def default(cls, aspect: float = 1.0) -> "Camera":
        """Return a camera with standard perspective view from slightly above."""
        cam = cls(projection="perspective", fov_deg=45.0, aspect=aspect)
        cam.set_orbit(azimuth_deg=30.0, elevation_deg=20.0, distance=8.0)
        return cam

    def to_dict(self) -> dict:
        """Serialize camera params for project file storage."""
        return {
            "projection": self.projection,
            "fov_deg": self.fov_deg,
            "ortho_scale": self.ortho_scale,
            "near": self.near,
            "far": self.far,
            "aspect": self.aspect,
            "eye": list(self.eye),
            "center": list(self.center),
            "up": list(self.up),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        """Deserialize camera from dict."""
        cam = cls(
            projection=d.get("projection", "perspective"),
            fov_deg=d.get("fov_deg", 45.0),
            ortho_scale=d.get("ortho_scale", 3.0),
            near=d.get("near", 0.1),
            far=d.get("far", 1000.0),
            aspect=d.get("aspect", 1.0),
        )
        cam.eye = np.array(d.get("eye", [0, 0, 5]), dtype=np.float64)
        cam.center = np.array(d.get("center", [0, 0, 0]), dtype=np.float64)
        cam.up = np.array(d.get("up", [0, 1, 0]), dtype=np.float64)
        return cam
