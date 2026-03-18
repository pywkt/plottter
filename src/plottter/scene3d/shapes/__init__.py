"""3D shape primitives for the scene3d renderer."""

from .base import Shape
from .sphere import Sphere, OutlineSphere, ShadedSphere
from .cube import Cube, StripedCube
from .cone import Cone, OutlineCone
from .cylinder import Cylinder, OutlineCylinder
from .plane import Plane, TerrainPlane
from .triangle import Triangle
from .shard import Shard
from .mesh import Mesh
from .function_shape import FunctionShape
from .csg import CSGDifference, CSGIntersection
from .transformed import TransformedShape

__all__ = [
    "Shape",
    "Sphere", "OutlineSphere", "ShadedSphere",
    "Cube", "StripedCube",
    "Cone", "OutlineCone",
    "Cylinder", "OutlineCylinder",
    "Plane", "TerrainPlane",
    "Triangle",
    "Shard",
    "Mesh",
    "FunctionShape",
    "CSGDifference", "CSGIntersection",
    "TransformedShape",
]
