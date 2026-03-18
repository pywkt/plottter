"""Ray and intersection result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .vector3 import Vec3, vec3

if TYPE_CHECKING:
    pass

EPSILON = 1e-8


@dataclass
class Ray:
    """A ray defined by an origin point and a unit direction vector."""

    origin: Vec3
    direction: Vec3  # should be unit length

    def at(self, t: float) -> Vec3:
        """Return the point at parameter t along the ray."""
        return self.origin + self.direction * t

    @classmethod
    def from_to(cls, start: Vec3, end: Vec3) -> "Ray":
        """Create a ray from start to end. Direction is normalized."""
        d = end - start
        n = np.linalg.norm(d)
        if n < EPSILON:
            d = vec3(0.0, 0.0, 1.0)
        else:
            d = d / n
        return cls(origin=start.copy(), direction=d)


@dataclass
class Hit:
    """Result of a ray intersection test."""

    shape: Any  # the Shape that was hit (avoid circular import)
    t: float    # ray parameter at the intersection point
    point: Vec3  # world-space intersection point

    @property
    def distance(self) -> float:
        return self.t


NO_HIT: Hit | None = None
