"""scene3d — 3D line art renderer for Plottter.

Port of Viewport.js (JavaScript) / Ln (Go) to Python with NumPy.
Produces polyline output suitable for pen plotters.

Usage
-----
from plottter.scene3d import Scene, Camera
from plottter.scene3d.shapes import Sphere, Cube, Plane

scene = Scene()
scene.add(Sphere(radius=1.0))
scene.add(Cube(center=[2, 0, 0], size=1.5))
scene.add(Plane(center=[0, -1, 0], size=6))
scene.compile()

camera = Camera.default(aspect=1.0)
polylines = scene.render(camera, canvas_w_mm=100, canvas_h_mm=100)
# polylines: list[list[tuple[float, float]]] — mm coordinates
"""

from .scene import Scene
from .camera import Camera
from .path3d import Path3D
from .ray import Ray, Hit
from .bbox import BBox
from .bvh import BVH
from .vector3 import vec3, normalize, Vec3
from .matrix4 import identity, look_at, perspective, orthographic

__all__ = [
    "Scene",
    "Camera",
    "Path3D",
    "Ray",
    "Hit",
    "BBox",
    "BVH",
    "vec3",
    "normalize",
    "Vec3",
    "identity",
    "look_at",
    "perspective",
    "orthographic",
]
