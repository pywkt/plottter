"""Abstract base class for all 3D shapes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..path3d import Path3D
    from ..bbox import BBox
    from ..ray import Ray, Hit
    from ..vector3 import Vec3


class Shape(ABC):
    """Abstract base for all renderable 3D shapes.

    Every shape must implement three methods:
    - paths(): generate the 3D polylines to render
    - intersect(): ray intersection for hidden line removal
    - bbox(): axis-aligned bounding box for BVH construction

    Shapes that support surface fill rendering may also implement:
    - surface_triangles(): world-space triangles for surface fill
    """

    @abstractmethod
    def paths(self) -> list["Path3D"]:
        """Return the 3D polylines that define this shape's visual representation."""
        ...

    @abstractmethod
    def intersect(self, ray: "Ray") -> "Hit | None":
        """Test for ray intersection. Return Hit or None."""
        ...

    def surface_triangles(self) -> "list[tuple[Vec3, Vec3, Vec3]]":
        """Return world-space triangles for surface fill rendering.

        Each element is a tuple of three Vec3 vertices forming a triangle.
        Shapes that don't override this fall back to wireframe-only rendering.
        """
        return []

    def intersect_any(self, ray: "Ray", t_max: float) -> bool:
        """Return True if ray hits this shape within t_max.

        Default implementation wraps intersect().  Override for faster occlusion
        testing (e.g. Mesh uses TriangleBVH.intersect_any for early-exit).
        """
        from ..ray import EPSILON
        hit = self.intersect(ray)
        return hit is not None and EPSILON < hit.t < t_max

    @abstractmethod
    def bbox(self) -> "BBox":
        """Return the axis-aligned bounding box of this shape."""
        ...
