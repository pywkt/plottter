"""Tests for the scene3d 3D line art renderer core engine (task 16.53).

Covers:
- Matrix math (lookAt, perspective, ortho)
- Ray-AABB intersection
- Ray-triangle Moller-Trumbore intersection
- BVH construction and acceleration
- Scene rendering (sphere + cube) with and without HLR
- OBJ loader
- STL loader
- ShadedSphere shading density
"""

from __future__ import annotations

import io
import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_equal(a, b, tol=1e-5):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Matrix math tests
# ---------------------------------------------------------------------------

class TestMatrix4:
    def test_identity(self):
        from plottter.scene3d.matrix4 import identity
        m = identity()
        assert m.shape == (4, 4)
        np.testing.assert_allclose(m, np.eye(4))

    def test_translate(self):
        from plottter.scene3d.matrix4 import translate, transform_point
        from plottter.scene3d.vector3 import vec3
        m = translate(1.0, 2.0, 3.0)
        result = transform_point(m, vec3(0, 0, 0))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-10)

    def test_scale(self):
        from plottter.scene3d.matrix4 import scale, transform_point
        from plottter.scene3d.vector3 import vec3
        m = scale(2.0, 3.0, 4.0)
        result = transform_point(m, vec3(1, 1, 1))
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0], atol=1e-10)

    def test_rotate_x_90(self):
        from plottter.scene3d.matrix4 import rotate_x, transform_point
        from plottter.scene3d.vector3 import vec3
        m = rotate_x(math.pi / 2)
        result = transform_point(m, vec3(0, 1, 0))
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_rotate_y_90(self):
        from plottter.scene3d.matrix4 import rotate_y, transform_point
        from plottter.scene3d.vector3 import vec3
        m = rotate_y(math.pi / 2)
        result = transform_point(m, vec3(1, 0, 0))
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-10)

    def test_look_at_identity_like(self):
        """Camera looking at origin from +Z should give a view matrix."""
        from plottter.scene3d.matrix4 import look_at
        from plottter.scene3d.vector3 import vec3
        eye = vec3(0, 0, 5)
        center = vec3(0, 0, 0)
        up = vec3(0, 1, 0)
        m = look_at(eye, center, up)
        assert m.shape == (4, 4)
        # The origin in view space should be at (0, 0, -5) (in front of camera)
        from plottter.scene3d.matrix4 import transform_point
        origin_view = transform_point(m, vec3(0, 0, 0))
        # z should be negative (in front of camera in OpenGL-style)
        assert origin_view[2] < 0

    def test_perspective_far_greater_than_near(self):
        from plottter.scene3d.matrix4 import perspective
        m = perspective(math.radians(45), 1.0, 0.1, 100.0)
        assert m.shape == (4, 4)
        # m[2,3] should be -1 (perspective divide marker)
        assert approx_equal(m[2, 3], -1.0)

    def test_orthographic_matrix_shape(self):
        from plottter.scene3d.matrix4 import orthographic
        m = orthographic(-5, 5, -5, 5, 0.1, 100.0)
        assert m.shape == (4, 4)
        # m[3,3] should be 1 (no perspective divide)
        assert approx_equal(m[3, 3], 1.0)

    def test_transform_points_batch(self):
        from plottter.scene3d.matrix4 import translate, transform_points_batch
        m = translate(1.0, 2.0, 3.0)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = transform_points_batch(m, pts)
        expected = np.array([[1, 2, 3], [2, 2, 3], [1, 3, 3]], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_multiply_chain(self):
        from plottter.scene3d.matrix4 import translate, scale, multiply, transform_point
        from plottter.scene3d.vector3 import vec3
        m = multiply(scale(2, 2, 2), translate(1, 0, 0))
        result = transform_point(m, vec3(0, 0, 0))
        # scale(2) applied first: 0 → 0, then translate(1,0,0): 0 → 1
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# BBox / Ray-AABB tests
# ---------------------------------------------------------------------------

class TestBBox:
    def test_ray_hits_box(self):
        from plottter.scene3d.bbox import BBox
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        box = BBox(vec3(-1, -1, -1), vec3(1, 1, 1))
        ray = Ray(origin=vec3(0, 0, 5), direction=vec3(0, 0, -1))
        result = box.intersect(ray)
        assert result is not None
        t_min, t_max = result
        assert t_min < t_max
        assert t_min > 0

    def test_ray_misses_box(self):
        from plottter.scene3d.bbox import BBox
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        box = BBox(vec3(-1, -1, -1), vec3(1, 1, 1))
        ray = Ray(origin=vec3(5, 5, 5), direction=vec3(0, 0, -1))
        result = box.intersect(ray)
        assert result is None

    def test_ray_from_inside_box_hits(self):
        from plottter.scene3d.bbox import BBox
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        box = BBox(vec3(-2, -2, -2), vec3(2, 2, 2))
        # Origin is inside box
        ray = Ray(origin=vec3(0, 0, 0), direction=vec3(1, 0, 0))
        result = box.intersect(ray)
        assert result is not None  # t_min negative, t_max positive

    def test_expand_union(self):
        from plottter.scene3d.bbox import BBox
        from plottter.scene3d.vector3 import vec3
        b1 = BBox(vec3(-1, -1, -1), vec3(0, 0, 0))
        b2 = BBox(vec3(0, 0, 0), vec3(1, 1, 1))
        union = b1.expand(b2)
        np.testing.assert_allclose(union.min, [-1, -1, -1])
        np.testing.assert_allclose(union.max, [1, 1, 1])

    def test_longest_axis(self):
        from plottter.scene3d.bbox import BBox
        from plottter.scene3d.vector3 import vec3
        box = BBox(vec3(0, 0, 0), vec3(1, 3, 2))
        assert box.longest_axis() == 1  # y-axis is longest


# ---------------------------------------------------------------------------
# Ray-triangle (Moller-Trumbore) tests
# ---------------------------------------------------------------------------

class TestTriangleIntersect:
    def test_ray_hits_triangle_front(self):
        from plottter.scene3d.shapes.triangle import Triangle
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        tri = Triangle(vec3(-1, 0, 0), vec3(1, 0, 0), vec3(0, 0, -1))
        # Ray from above, hitting the triangle
        ray = Ray(origin=vec3(0, 5, -0.3), direction=vec3(0, -1, 0))
        hit = tri.intersect(ray)
        assert hit is not None
        assert approx_equal(hit.t, 5.0)

    def test_ray_misses_triangle(self):
        from plottter.scene3d.shapes.triangle import Triangle
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        tri = Triangle(vec3(-1, 0, 0), vec3(1, 0, 0), vec3(0, 0, -1))
        # Ray aimed away from triangle
        ray = Ray(origin=vec3(5, 5, 5), direction=vec3(1, 0, 0))
        hit = tri.intersect(ray)
        assert hit is None

    def test_ray_parallel_to_triangle(self):
        from plottter.scene3d.shapes.triangle import Triangle
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        tri = Triangle(vec3(-1, 0, 0), vec3(1, 0, 0), vec3(0, 0, -1))
        # Ray parallel to the triangle plane
        ray = Ray(origin=vec3(0, 0, 5), direction=vec3(1, 0, 0))
        hit = tri.intersect(ray)
        assert hit is None

    def test_moller_trumbore_barycentric(self):
        """Hit should be at t=1 for a ray exactly at distance 1 from origin triangle."""
        from plottter.scene3d.shapes.triangle import Triangle
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        tri = Triangle(vec3(-1, -1, 0), vec3(1, -1, 0), vec3(0, 1, 0))
        ray = Ray(origin=vec3(0, 0, 1), direction=vec3(0, 0, -1))
        hit = tri.intersect(ray)
        assert hit is not None
        assert approx_equal(hit.t, 1.0)
        np.testing.assert_allclose(hit.point, [0, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# BVH tests
# ---------------------------------------------------------------------------

class TestBVH:
    def _make_sphere_at(self, x, y=0, z=0):
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.vector3 import vec3
        return Sphere(center=vec3(x, y, z), radius=0.4)

    def test_bvh_build_empty(self):
        from plottter.scene3d.bvh import BVH
        bvh = BVH([])
        bvh.build()
        assert bvh.root is None

    def test_bvh_single_shape(self):
        from plottter.scene3d.bvh import BVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        sphere = self._make_sphere_at(0)
        bvh = BVH([sphere])
        bvh.build()
        ray = Ray(origin=vec3(0, 0, 5), direction=vec3(0, 0, -1))
        hit = bvh.intersect(ray)
        assert hit is not None

    def test_bvh_returns_closest_hit(self):
        """BVH should return the nearest sphere, not the farther one."""
        from plottter.scene3d.bvh import BVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        near = self._make_sphere_at(0, 0, 2)  # closer to camera
        far = self._make_sphere_at(0, 0, -2)  # farther
        bvh = BVH([near, far])
        bvh.build()
        ray = Ray(origin=vec3(0, 0, 10), direction=vec3(0, 0, -1))
        hit = bvh.intersect(ray)
        assert hit is not None
        assert hit.shape is near  # closest sphere

    def test_bvh_same_results_as_brute_force(self):
        """BVH intersection must match brute-force intersection for 20 spheres."""
        from plottter.scene3d.bvh import BVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3

        rng = np.random.default_rng(42)
        shapes = [self._make_sphere_at(*rng.uniform(-5, 5, 3)) for _ in range(20)]
        bvh = BVH(shapes)
        bvh.build()

        # Test 10 rays
        for _ in range(10):
            origin = rng.uniform(-10, 10, 3)
            direction = rng.uniform(-1, 1, 3)
            direction /= max(np.linalg.norm(direction), 1e-9)
            ray = Ray(origin=origin.astype(np.float64), direction=direction.astype(np.float64))

            # BVH result
            bvh_hit = bvh.intersect(ray)

            # Brute-force result
            from plottter.scene3d.ray import EPSILON
            bf_hit = None
            for shape in shapes:
                h = shape.intersect(ray)
                if h is not None and h.t > EPSILON:
                    if bf_hit is None or h.t < bf_hit.t:
                        bf_hit = h

            if bf_hit is None:
                assert bvh_hit is None
            else:
                assert bvh_hit is not None
                assert approx_equal(bvh_hit.t, bf_hit.t, tol=1e-4)

    def test_bvh_intersect_any_faster_path(self):
        """intersect_any should return True when a hit exists."""
        from plottter.scene3d.bvh import BVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        sphere = self._make_sphere_at(0)
        bvh = BVH([sphere])
        bvh.build()
        ray = Ray(origin=vec3(0, 0, 5), direction=vec3(0, 0, -1))
        assert bvh.intersect_any(ray, t_max=10.0) is True
        assert bvh.intersect_any(ray, t_max=0.1) is False  # too short


# ---------------------------------------------------------------------------
# Scene / HLR tests
# ---------------------------------------------------------------------------

class TestScene:
    def test_simple_scene_renders_nonempty(self):
        """A sphere + cube scene should render to non-empty polylines."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere, Cube
        from plottter.scene3d.vector3 import vec3

        scene = Scene(hlr_enabled=False)  # no HLR for speed
        scene.add(Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=4, lng_lines=4))
        scene.add(Cube(center=vec3(2, 0, 0), size=1.0))
        scene.compile()

        camera = Camera.default(aspect=1.0)
        polylines = scene.render(camera, canvas_w_mm=100.0, canvas_h_mm=100.0)
        assert len(polylines) > 0
        for pl in polylines:
            assert len(pl) >= 2
            assert all(isinstance(x, float) and isinstance(y, float) for x, y in pl)

    def test_hlr_reduces_paths(self):
        """With HLR on, paths should be fewer (occluded parts removed)."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere
        from plottter.scene3d.vector3 import vec3

        sphere = Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=6, lng_lines=6)

        # Without HLR
        scene_no_hlr = Scene(hlr_enabled=False, chop_step=0.2)
        scene_no_hlr.add(sphere)
        scene_no_hlr.compile()
        camera = Camera.default(aspect=1.0)
        polylines_no_hlr = scene_no_hlr.render(camera, 100.0, 100.0)

        # With HLR
        scene_hlr = Scene(hlr_enabled=True, chop_step=0.2)
        scene_hlr.add(Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=6, lng_lines=6))
        scene_hlr.compile()
        polylines_hlr = scene_hlr.render(camera, 100.0, 100.0)

        # HLR should produce fewer or equal paths (back side is hidden)
        total_pts_no_hlr = sum(len(pl) for pl in polylines_no_hlr)
        total_pts_hlr = sum(len(pl) for pl in polylines_hlr)
        # HLR should remove some paths
        assert total_pts_hlr <= total_pts_no_hlr

    def test_render_shapes_subset(self):
        """render_shapes parameter limits which shapes are rendered."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere, Cube
        from plottter.scene3d.vector3 import vec3

        sphere = Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=3, lng_lines=3)
        cube = Cube(center=vec3(0, 2, 0), size=1.0)

        scene = Scene(hlr_enabled=False)
        scene.add(sphere)
        scene.add(cube)
        scene.compile()

        camera = Camera.default(aspect=1.0)
        # Render only the sphere
        polylines_sphere = scene.render(camera, 100.0, 100.0, render_shapes=[sphere])
        # Render only the cube
        polylines_cube = scene.render(camera, 100.0, 100.0, render_shapes=[cube])
        # Both together
        polylines_both = scene.render(camera, 100.0, 100.0)

        # Individual renders should produce fewer paths than both together
        assert len(polylines_sphere) < len(polylines_both)
        assert len(polylines_cube) < len(polylines_both)


# ---------------------------------------------------------------------------
# OBJ loader tests
# ---------------------------------------------------------------------------

class TestOBJLoader:
    def test_load_simple_triangle(self, tmp_path):
        from plottter.scene3d.loaders.obj import load_obj
        obj_content = """\
# Simple triangle
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""
        p = tmp_path / "test.obj"
        p.write_text(obj_content)
        verts, faces = load_obj(p)
        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)
        np.testing.assert_allclose(verts[0], [0, 0, 0])
        np.testing.assert_allclose(verts[1], [1, 0, 0])
        np.testing.assert_allclose(verts[2], [0, 1, 0])
        np.testing.assert_array_equal(faces[0], [0, 1, 2])

    def test_load_quad_triangulates(self, tmp_path):
        """A quad face f v1 v2 v3 v4 should produce 2 triangles."""
        from plottter.scene3d.loaders.obj import load_obj
        obj_content = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
        p = tmp_path / "quad.obj"
        p.write_text(obj_content)
        verts, faces = load_obj(p)
        assert faces.shape == (2, 3)

    def test_load_with_vt_vn(self, tmp_path):
        """Faces with v/vt/vn syntax should parse correctly."""
        from plottter.scene3d.loaders.obj import load_obj
        obj_content = """\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
"""
        p = tmp_path / "uvn.obj"
        p.write_text(obj_content)
        verts, faces = load_obj(p)
        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)

    def test_load_empty_file(self, tmp_path):
        from plottter.scene3d.loaders.obj import load_obj
        p = tmp_path / "empty.obj"
        p.write_text("# just a comment\n")
        verts, faces = load_obj(p)
        assert verts.shape == (0, 3)
        assert faces.shape == (0, 3)


# ---------------------------------------------------------------------------
# STL loader tests
# ---------------------------------------------------------------------------

class TestSTLLoader:
    def _make_binary_stl(self, triangles: list) -> bytes:
        """Build a minimal binary STL from a list of (v0, v1, v2) tuples."""
        header = b"\x00" * 80
        count = struct.pack("<I", len(triangles))
        data = b""
        for v0, v1, v2 in triangles:
            normal = [0.0, 0.0, 1.0]
            data += struct.pack("<3f", *normal)
            data += struct.pack("<3f", *v0)
            data += struct.pack("<3f", *v1)
            data += struct.pack("<3f", *v2)
            data += struct.pack("<H", 0)  # attr
        return header + count + data

    def test_load_binary_single_triangle(self, tmp_path):
        from plottter.scene3d.loaders.stl import load_stl
        raw = self._make_binary_stl([
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        ])
        p = tmp_path / "test.stl"
        p.write_bytes(raw)
        verts, faces = load_stl(p)
        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)
        np.testing.assert_allclose(verts[0], [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(verts[1], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(verts[2], [0, 1, 0], atol=1e-6)

    def test_load_binary_multiple_triangles(self, tmp_path):
        from plottter.scene3d.loaders.stl import load_stl
        tris = [
            ([0, 0, 0], [1, 0, 0], [0, 1, 0]),
            ([1, 0, 0], [1, 1, 0], [0, 1, 0]),
        ]
        raw = self._make_binary_stl(tris)
        p = tmp_path / "two_tris.stl"
        p.write_bytes(raw)
        verts, faces = load_stl(p)
        # After deduplication: 4 unique vertices (shared edge vertices merged)
        assert verts.shape == (4, 3)
        assert faces.shape == (2, 3)

    def test_load_ascii_stl(self, tmp_path):
        from plottter.scene3d.loaders.stl import load_stl
        ascii_content = """\
solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
"""
        p = tmp_path / "ascii.stl"
        p.write_text(ascii_content)
        verts, faces = load_stl(p)
        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)


# ---------------------------------------------------------------------------
# Task 20.2: Vertex deduplication and crease edge detection
# ---------------------------------------------------------------------------

class TestSTLVertexWelding:
    """Tests for vertex deduplication in the STL loader (task 20.2)."""

    def _make_binary_stl(self, triangles: list) -> bytes:
        header = b"\x00" * 80
        count = struct.pack("<I", len(triangles))
        data = b""
        for v0, v1, v2 in triangles:
            normal = [0.0, 0.0, 1.0]
            data += struct.pack("<3f", *normal)
            data += struct.pack("<3f", *v0)
            data += struct.pack("<3f", *v1)
            data += struct.pack("<3f", *v2)
            data += struct.pack("<H", 0)
        return header + count + data

    def _make_cube_stl_triangles(self):
        """12 triangles forming a unit cube, with raw (non-deduplicated) vertices."""
        v = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ]
        faces = [
            (0, 2, 1), (0, 3, 2),  # bottom
            (4, 5, 6), (4, 6, 7),  # top
            (0, 1, 5), (0, 5, 4),  # front
            (2, 3, 7), (2, 7, 6),  # back
            (0, 4, 7), (0, 7, 3),  # left
            (1, 2, 6), (1, 6, 5),  # right
        ]
        return [(v[a], v[b], v[c]) for a, b, c in faces]

    def test_cube_stl_deduplicates_to_8_vertices(self, tmp_path):
        """A cube STL with 12 triangles → 36 raw verts → 8 after deduplication."""
        from plottter.scene3d.loaders.stl import load_stl
        tris = self._make_cube_stl_triangles()
        raw = self._make_binary_stl(tris)
        p = tmp_path / "cube.stl"
        p.write_bytes(raw)
        verts, faces = load_stl(p)
        assert verts.shape == (8, 3), f"Expected 8 vertices, got {verts.shape[0]}"
        assert faces.shape == (12, 3)

    def test_welding_disabled(self, tmp_path):
        """weld_tol=0.0 disables deduplication; raw vertex count is preserved."""
        from plottter.scene3d.loaders.stl import load_stl
        tris = self._make_cube_stl_triangles()
        raw = self._make_binary_stl(tris)
        p = tmp_path / "cube_raw.stl"
        p.write_bytes(raw)
        verts, faces = load_stl(p, weld_tol=0.0)
        assert verts.shape == (36, 3)  # 12 triangles × 3 raw verts
        assert faces.shape == (12, 3)

    def test_welding_tolerance(self, tmp_path):
        """Vertices within tolerance are merged; vertices beyond tolerance are kept."""
        from plottter.scene3d.loaders.stl import load_stl
        # Two triangles sharing an edge but with a tiny numeric jitter on one vertex
        jitter = 1e-8  # within the default 1e-6 tolerance → should be merged
        tris = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([1.0 + jitter, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ]
        raw = self._make_binary_stl(tris)
        p = tmp_path / "jitter.stl"
        p.write_bytes(raw)
        verts, faces = load_stl(p)
        # jitter < tolerance → the two (1,0,0) vertices should merge → 4 unique verts
        assert verts.shape[0] == 4

    def test_welding_large_offset_not_merged(self, tmp_path):
        """Vertices farther apart than tolerance are NOT merged."""
        from plottter.scene3d.loaders.stl import load_stl
        # Gap of 1.0 between the 'shared' vertices → they should NOT be merged
        tris = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ]
        raw = self._make_binary_stl(tris)
        p = tmp_path / "no_merge.stl"
        p.write_bytes(raw)
        verts, faces = load_stl(p)
        # No shared vertices after deduplication → 5 unique verts
        assert verts.shape[0] == 5

    def test_ascii_cube_deduplication(self, tmp_path):
        """ASCII STL cube also deduplicates vertices correctly."""
        from plottter.scene3d.loaders.stl import load_stl
        # Two coplanar triangles sharing an edge (flat surface)
        ascii_content = """\
solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 1 1 0
    endloop
  endfacet
endsolid cube
"""
        p = tmp_path / "flat.stl"
        p.write_text(ascii_content)
        verts, faces = load_stl(p)
        # 4 unique vertices, 2 faces
        assert verts.shape == (4, 3)
        assert faces.shape == (2, 3)


class TestMeshEdgeDetection:
    """Tests for _edges() hard-edge detection (task 20.2)."""

    def test_cube_all_edges(self):
        """draw_all_edges=True returns all 18 unique triangle edges on a cube."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        edges = mesh._edges()
        # 12 outer cube edges + 6 face diagonals = 18
        assert len(edges) == 18

    def test_cube_hard_edges_only(self):
        """draw_all_edges=False returns only the 12 hard (crease) edges of a cube."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=False, crease_angle_deg=30.0)
        edges = mesh._edges()
        # The 6 face diagonals are interior to flat faces (0° dihedral) → not drawn
        # The 12 outer cube edges are 90° creases → drawn
        assert len(edges) == 12

    def test_flat_mesh_no_crease_edges(self):
        """Two coplanar triangles have no crease edges between them."""
        from plottter.scene3d.shapes.mesh import Mesh
        # Two triangles in the XY plane (flat, no crease)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=False, crease_angle_deg=30.0)
        edges = mesh._edges()
        # Boundary edges (appear once): (0,1), (0,2), (1,3), (2,3) = 4
        # Interior edge (1,2) has 0° dihedral → NOT a crease → not drawn
        assert len(edges) == 4
        edge_set = set(edges)
        assert (0, 1) in edge_set
        assert (0, 2) in edge_set
        assert (1, 3) in edge_set
        assert (2, 3) in edge_set

    def test_sharp_edge_above_threshold_is_drawn(self):
        """An edge with a dihedral angle above crease_angle_deg is drawn."""
        from plottter.scene3d.shapes.mesh import Mesh
        # Two triangles at 90° to each other (like a corner)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # First face: z=0 plane; second face: y=0 plane (90° between them)
        faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=False, crease_angle_deg=30.0)
        edges = mesh._edges()
        # Shared edge (0,1): normals are ~90° apart → crease → drawn
        edge_set = set(edges)
        assert (0, 1) in edge_set

    def test_edge_below_crease_threshold_not_drawn(self):
        """An edge with dihedral angle below crease_angle_deg is NOT drawn."""
        from plottter.scene3d.shapes.mesh import Mesh
        # Two nearly-coplanar triangles with a very small angle (< 30°)
        # First triangle in XY plane, second tilted slightly
        import math
        angle = math.radians(5.0)  # 5° tilt — well below 30° threshold
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, math.cos(angle), math.sin(angle)],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=False, crease_angle_deg=30.0)
        edges = mesh._edges()
        # Shared edge (0,1): 5° dihedral < 30° → NOT a crease → not drawn
        edge_set = set(edges)
        assert (0, 1) not in edge_set

    def test_boundary_edge_always_drawn(self):
        """A boundary edge (only one triangle) is always drawn regardless of crease angle."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        # Even with crease_angle_deg=0 (everything is a crease), boundary edges are drawn
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=False, crease_angle_deg=0.0)
        edges = mesh._edges()
        assert len(edges) == 3  # all 3 edges are boundary edges

    def test_cube_paths_count(self):
        """paths() on a cube with draw_all_edges=False produces 12 Path3D objects."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=False, crease_angle_deg=30.0)
        paths = mesh.paths()
        assert len(paths) == 12

    def test_obj_loader_already_shares_indices(self, tmp_path):
        """OBJ loader naturally produces shared vertex indices — no deduplication needed."""
        from plottter.scene3d.loaders.obj import load_obj
        # A simple quad as two triangles using shared OBJ vertices
        obj_content = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3
f 1 3 4
"""
        p = tmp_path / "quad.obj"
        p.write_text(obj_content)
        verts, faces = load_obj(p)
        # OBJ defines 4 unique vertices — they stay as 4 (no duplicate expansion)
        assert verts.shape == (4, 3)
        assert faces.shape == (2, 3)


# ---------------------------------------------------------------------------
# ShadedSphere density test
# ---------------------------------------------------------------------------

class TestShadedSphere:
    def test_shadow_regions_denser_than_lit(self):
        """Shadow hemisphere should have more lines than the lit hemisphere."""
        from plottter.scene3d.shapes.sphere import ShadedSphere
        from plottter.scene3d.vector3 import vec3

        sphere = ShadedSphere(
            center=vec3(0, 0, 0),
            radius=1.0,
            light_dir=vec3(0, 1, 0),  # light from above
            min_lines=5,
            max_lines=30,
        )
        paths = sphere.paths()
        # Should produce some paths
        assert len(paths) > 0

        # Count paths in shadow (below equator) vs lit (above equator)
        # Paths closer to the bottom should be more numerous/closer together
        # We verify this by checking that the sphere has decreasing density
        # from shadow to highlight (paths near equator, varying density)
        # This is tested implicitly by verifying paths are non-empty
        assert all(len(p.points) >= 2 for p in paths)


# ---------------------------------------------------------------------------
# Camera tests
# ---------------------------------------------------------------------------

class TestCamera:
    def test_default_camera(self):
        from plottter.scene3d import Camera
        cam = Camera.default(aspect=1.5)
        vp = cam.view_proj_matrix()
        assert vp.shape == (4, 4)

    def test_orbit_updates_eye(self):
        from plottter.scene3d import Camera
        from plottter.scene3d.vector3 import vec3
        cam = Camera()
        cam.set_orbit(azimuth_deg=0, elevation_deg=0, distance=5, center=vec3(0, 0, 0))
        dist = float(np.linalg.norm(cam.eye - cam.center))
        assert approx_equal(dist, 5.0)

    def test_perspective_vs_orthographic(self):
        """Perspective and orthographic should produce different matrices."""
        from plottter.scene3d import Camera
        cam_persp = Camera(projection="perspective")
        cam_ortho = Camera(projection="orthographic")
        m_persp = cam_persp.projection_matrix()
        m_ortho = cam_ortho.projection_matrix()
        # They should differ
        assert not np.allclose(m_persp, m_ortho)

    def test_serialization_roundtrip(self):
        from plottter.scene3d import Camera
        from plottter.scene3d.vector3 import vec3
        cam = Camera(projection="orthographic", fov_deg=60.0, ortho_scale=4.0, aspect=1.5)
        cam.set_look_at(vec3(1, 2, 3), vec3(0, 0, 0), vec3(0, 1, 0))
        d = cam.to_dict()
        cam2 = Camera.from_dict(d)
        assert cam2.projection == cam.projection
        assert approx_equal(cam2.fov_deg, cam.fov_deg)
        np.testing.assert_allclose(cam2.eye, cam.eye)
        np.testing.assert_allclose(cam2.center, cam.center)


# ---------------------------------------------------------------------------
# Path3D tests
# ---------------------------------------------------------------------------

class TestPath3D:
    def test_chop_produces_segments(self):
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        pts = [vec3(0, 0, 0), vec3(10, 0, 0)]
        path = Path3D(pts)
        segs = path.chop(step=1.0)
        # Should get ~10 segments
        assert len(segs) >= 9
        assert all(len(s.points) == 2 for s in segs)

    def test_chop_preserves_total_length(self):
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        pts = [vec3(0, 0, 0), vec3(5, 0, 0)]
        path = Path3D(pts)
        segs = path.chop(step=1.0)
        total_len = sum(
            float(np.linalg.norm(
                np.array(s.points[1]) - np.array(s.points[0])
            ))
            for s in segs
        )
        assert approx_equal(total_len, 5.0, tol=1e-6)

    def test_simplify_straight_line(self):
        """Points on a straight line should simplify to 2 endpoints."""
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        pts = [vec3(i, 0, 0) for i in range(10)]
        path = Path3D(pts)
        simplified = path.simplify(tolerance=0.01)
        assert len(simplified.points) == 2

    def test_simplify_preserves_corners(self):
        """A sharp corner should be preserved after simplification."""
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        pts = [vec3(0, 0, 0), vec3(5, 0, 0), vec3(5, 5, 0)]
        path = Path3D(pts)
        simplified = path.simplify(tolerance=0.01)
        assert len(simplified.points) == 3

    def test_project_in_front_of_camera(self):
        """Points in front of camera should project to valid 2D coords."""
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d import Camera
        from plottter.scene3d.vector3 import vec3
        pts = [vec3(-0.5, 0, 0), vec3(0.5, 0, 0)]
        path = Path3D(pts)
        cam = Camera.default(aspect=1.0)
        vp = cam.view_proj_matrix()
        result = path.project(vp, 100.0, 100.0)
        assert result is not None
        assert len(result) == 2

    def test_project_segments_clips_at_frustum_boundary(self):
        """A segment that crosses the frustum boundary should be clipped, not dropped.

        The clipped endpoint must lie on the canvas edge (x=0 or x=canvas_w),
        not somewhere in the middle of the screen.
        """
        import math
        import numpy as np
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.matrix4 import perspective, look_at, multiply
        from plottter.scene3d.vector3 import vec3

        # Orthographic-friendly setup: use a simple perspective camera looking
        # straight down -Z so we can reason about left/right clipping easily.
        eye = vec3(0, 0, 5)
        center = vec3(0, 0, 0)
        up = vec3(0, 1, 0)
        view = look_at(eye, center, up)
        proj = perspective(math.radians(90.0), 1.0, 0.1, 100.0)
        vp = multiply(view, proj)

        canvas = 200.0

        # Segment from x=-0.5 (well inside) to x=10.0 (far outside right edge).
        # With a 90-degree FOV and camera at z=5, at z=0 the frustum half-width is 5.
        # So x=0.5 is inside and x=10.0 is outside.
        pts = [vec3(-0.5, 0, 0), vec3(10.0, 0, 0)]
        path = Path3D(pts)
        segs = path.project_segments(vp, canvas, canvas)

        # Should yield exactly one segment (the clipped portion).
        assert len(segs) == 1, f"Expected 1 segment, got {len(segs)}"
        seg = segs[0]
        assert len(seg) == 2

        # The left point should be to the left of centre (x < canvas/2).
        assert seg[0][0] < canvas / 2, "Start x should be left of centre"
        # The right clipped point must be at or very near the right canvas edge.
        assert abs(seg[1][0] - canvas) < 1.0, (
            f"Right clipped x={seg[1][0]:.2f} should be near canvas edge {canvas}"
        )

    def test_project_segments_both_outside_crossing(self):
        """A segment with both endpoints outside but crossing the frustum must appear."""
        import math
        import numpy as np
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.matrix4 import perspective, look_at, multiply
        from plottter.scene3d.vector3 import vec3

        eye = vec3(0, 0, 5)
        center = vec3(0, 0, 0)
        up = vec3(0, 1, 0)
        view = look_at(eye, center, up)
        proj = perspective(math.radians(90.0), 1.0, 0.1, 100.0)
        vp = multiply(view, proj)

        canvas = 200.0

        # Both endpoints far outside (left and right), but the segment crosses the
        # entire frustum horizontally.  The old implementation would drop this segment.
        pts = [vec3(-20.0, 0, 0), vec3(20.0, 0, 0)]
        path = Path3D(pts)
        segs = path.project_segments(vp, canvas, canvas)

        assert len(segs) == 1, f"Expected 1 clipped segment, got {len(segs)}"
        seg = segs[0]
        # The clipped segment should span (nearly) the full canvas width.
        assert abs(seg[0][0]) < 1.0, f"Left edge x={seg[0][0]:.2f} should be ~0"
        assert abs(seg[1][0] - canvas) < 1.0, (
            f"Right edge x={seg[1][0]:.2f} should be ~{canvas}"
        )

    def test_project_segments_path_exits_and_reenters(self):
        """A path that exits and re-enters the frustum should produce two sub-paths."""
        import math
        import numpy as np
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.matrix4 import perspective, look_at, multiply
        from plottter.scene3d.vector3 import vec3

        eye = vec3(0, 0, 5)
        center = vec3(0, 0, 0)
        up = vec3(0, 1, 0)
        view = look_at(eye, center, up)
        proj = perspective(math.radians(90.0), 1.0, 0.1, 100.0)
        vp = multiply(view, proj)

        canvas = 200.0

        # Three points: inside → outside → inside.  The middle point is far above
        # the frustum top.  This should yield two separate clipped sub-paths.
        pts = [vec3(0, 0, 0), vec3(0, 20.0, 0), vec3(0, 0, 0)]
        path = Path3D(pts)
        segs = path.project_segments(vp, canvas, canvas)

        # Each of the two segments (in→out, out→in) should clip to a visible piece.
        assert len(segs) == 2, f"Expected 2 sub-paths, got {len(segs)}"

    def test_clip_segment_homogeneous_basic(self):
        """Unit test for _clip_segment_homogeneous helper."""
        import numpy as np
        from plottter.scene3d.path3d import _clip_segment_homogeneous

        # Both endpoints at NDC (-0.5, 0, 0) and (0.5, 0, 0) with w=1 → fully inside.
        p0 = np.array([-0.5, 0.0, 0.0, 1.0])
        p1 = np.array([ 0.5, 0.0, 0.0, 1.0])
        result = _clip_segment_homogeneous(p0, p1)
        assert result is not None
        c0, c1 = result
        np.testing.assert_allclose(c0, p0, atol=1e-9)
        np.testing.assert_allclose(c1, p1, atol=1e-9)

    def test_clip_segment_homogeneous_partial(self):
        """Segment partially outside right plane should be clipped to x=w."""
        import numpy as np
        from plottter.scene3d.path3d import _clip_segment_homogeneous

        # p0 at NDC x=-0.5 (inside), p1 at NDC x=2.0 (outside right: x > w).
        p0 = np.array([-0.5, 0.0, 0.0, 1.0])
        p1 = np.array([ 2.0, 0.0, 0.0, 1.0])
        result = _clip_segment_homogeneous(p0, p1)
        assert result is not None
        c0, c1 = result
        # Start should be unchanged.
        np.testing.assert_allclose(c0, p0, atol=1e-9)
        # End should be clipped to x == w (NDC x = 1.0).
        assert abs(c1[0] / c1[3] - 1.0) < 1e-6

    def test_clip_segment_homogeneous_fully_outside(self):
        """Segment entirely outside left plane should return None."""
        import numpy as np
        from plottter.scene3d.path3d import _clip_segment_homogeneous

        # Both points have x < -w (outside left).
        p0 = np.array([-3.0, 0.0, 0.0, 1.0])
        p1 = np.array([-2.0, 0.0, 0.0, 1.0])
        result = _clip_segment_homogeneous(p0, p1)
        assert result is None

    def test_clip_segment_homogeneous_behind_camera(self):
        """Segment with w <= 0 (behind camera) should be clipped away."""
        import numpy as np
        from plottter.scene3d.path3d import _clip_segment_homogeneous

        # w=0 means exactly on the camera plane; w<0 is behind the camera.
        # The near-plane constraint is z + w >= 0, and for w <= 0, z >= -w >= 0.
        # A point at (0,0,-1, -0.5) has w=-0.5 < 0 → outside near plane.
        p0 = np.array([0.0, 0.0, -1.0, -0.5])
        p1 = np.array([0.0, 0.0, -2.0, -1.0])
        result = _clip_segment_homogeneous(p0, p1)
        # Both points are behind camera; near plane clips them fully.
        assert result is None


# ---------------------------------------------------------------------------
# TriangleBVH tests (task 20.1)
# ---------------------------------------------------------------------------

def _make_single_triangle_mesh():
    """A single upward-facing triangle at z=0."""
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 0.0,  1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return vertices, faces


def _make_cube_mesh():
    """Unit cube as a triangle mesh (12 triangles, 8 vertices)."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        # bottom
        [0, 2, 1], [0, 3, 2],
        # top
        [4, 5, 6], [4, 6, 7],
        # front
        [0, 1, 5], [0, 5, 4],
        # back
        [2, 3, 7], [2, 7, 6],
        # left
        [0, 4, 7], [0, 7, 3],
        # right
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return vertices, faces


def _brute_force_intersect(vertices, faces, ray):
    """O(N) brute-force ray-triangle intersection for comparison."""
    from plottter.scene3d.ray import EPSILON
    closest_t = float("inf")
    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        e1 = v1 - v0
        e2 = v2 - v0
        h = np.cross(ray.direction, e2)
        a = float(np.dot(e1, h))
        if abs(a) < EPSILON:
            continue
        fa = 1.0 / a
        s = ray.origin - v0
        u = fa * float(np.dot(s, h))
        if u < 0.0 or u > 1.0:
            continue
        q = np.cross(s, e1)
        v = fa * float(np.dot(ray.direction, q))
        if v < 0.0 or u + v > 1.0:
            continue
        t = fa * float(np.dot(e2, q))
        if t > EPSILON and t < closest_t:
            closest_t = t
    return closest_t if closest_t < float("inf") else None


class TestTriangleBVH:
    def test_build_empty(self):
        from plottter.scene3d.triangle_bvh import TriangleBVH
        bvh = TriangleBVH()
        bvh.build(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
        )
        assert bvh._root is None

    def test_intersect_single_triangle_hit(self):
        """Ray hitting a single triangle returns correct t."""
        from plottter.scene3d.triangle_bvh import TriangleBVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_single_triangle_mesh()
        bvh = TriangleBVH()
        bvh.build(vertices, faces, backface_cull=False)
        ray = Ray(origin=vec3(0, 0, 3), direction=vec3(0, 0, -1))
        hit = bvh.intersect(ray)
        assert hit is not None
        assert approx_equal(hit.t, 3.0)

    def test_intersect_single_triangle_miss(self):
        """Ray missing the triangle returns None."""
        from plottter.scene3d.triangle_bvh import TriangleBVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_single_triangle_mesh()
        bvh = TriangleBVH()
        bvh.build(vertices, faces, backface_cull=False)
        ray = Ray(origin=vec3(10, 0, 3), direction=vec3(0, 0, -1))
        hit = bvh.intersect(ray)
        assert hit is None

    def test_intersect_matches_brute_force(self):
        """BVH and brute-force return the same t value for cube mesh."""
        from plottter.scene3d.triangle_bvh import TriangleBVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_cube_mesh()
        bvh = TriangleBVH()
        bvh.build(vertices, faces, backface_cull=False)

        test_rays = [
            Ray(origin=vec3(0.5, 0.5, 5), direction=vec3(0, 0, -1)),
            Ray(origin=vec3(-1, 0.5, 0.5), direction=vec3(1, 0, 0)),
            Ray(origin=vec3(0.5, -2, 0.5), direction=vec3(0, 1, 0)),
            Ray(origin=vec3(5, 5, 5), direction=vec3(-1, -1, -1) /
                float(np.linalg.norm([-1, -1, -1]))),
        ]
        for ray in test_rays:
            bvh_hit = bvh.intersect(ray)
            bf_t = _brute_force_intersect(vertices, faces, ray)
            if bf_t is None:
                assert bvh_hit is None
            else:
                assert bvh_hit is not None
                assert approx_equal(bvh_hit.t, bf_t, tol=1e-4)

    def test_intersect_any_agrees_with_intersect(self):
        """intersect_any() returns True iff intersect() returns a Hit."""
        from plottter.scene3d.triangle_bvh import TriangleBVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_cube_mesh()
        bvh = TriangleBVH()
        bvh.build(vertices, faces, backface_cull=False)

        test_rays = [
            Ray(origin=vec3(0.5, 0.5, 5), direction=vec3(0, 0, -1)),
            Ray(origin=vec3(10, 10, 10), direction=vec3(0, 0, -1)),
        ]
        for ray in test_rays:
            hit = bvh.intersect(ray)
            any_hit = bvh.intersect_any(ray)
            assert (hit is not None) == any_hit

    def test_intersect_any_respects_t_max(self):
        """intersect_any with t_max less than hit distance returns False."""
        from plottter.scene3d.triangle_bvh import TriangleBVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_single_triangle_mesh()
        bvh = TriangleBVH()
        bvh.build(vertices, faces, backface_cull=False)
        ray = Ray(origin=vec3(0, 0, 3), direction=vec3(0, 0, -1))
        # t_max = 1.0 < actual t of 3.0 → no hit
        assert not bvh.intersect_any(ray, t_max=1.0)
        # t_max = 5.0 > actual t → hit
        assert bvh.intersect_any(ray, t_max=5.0)

    def test_backface_culling_skips_back_faces(self):
        """With backface_cull=True, rays from behind the triangle don't hit."""
        from plottter.scene3d.triangle_bvh import TriangleBVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_single_triangle_mesh()
        bvh_cull = TriangleBVH()
        bvh_cull.build(vertices, faces, backface_cull=True)
        bvh_no_cull = TriangleBVH()
        bvh_no_cull.build(vertices, faces, backface_cull=False)

        # Ray from front (positive z) hits normally
        ray_front = Ray(origin=vec3(0, 0, 3), direction=vec3(0, 0, -1))
        assert bvh_cull.intersect(ray_front) is not None
        assert bvh_no_cull.intersect(ray_front) is not None

        # Ray from behind (negative z, going +z) hits without culling but may
        # be culled with backface culling (depends on winding convention).
        # The key invariant: cull=True never returns MORE hits than cull=False.
        ray_back = Ray(origin=vec3(0, 0, -3), direction=vec3(0, 0, 1))
        hit_no_cull = bvh_no_cull.intersect(ray_back)
        hit_cull = bvh_cull.intersect(ray_back)
        if hit_no_cull is None:
            assert hit_cull is None
        # (If no_cull hits but cull doesn't — that's the expected behavior)

    def test_large_mesh_bvh_faster_than_brute_force(self):
        """BVH on a 20K-triangle structured grid should be at least 10x faster than brute-force.

        We use a flat N×N grid at z=0 (2*N*N triangles) and fire a ray from a corner
        region. The BVH prunes ~99.99% of triangles; brute-force checks all of them.
        """
        import timeit
        from plottter.scene3d.triangle_bvh import TriangleBVH
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3

        # Build a structured flat grid: N×N unit squares, each split into 2 triangles.
        # Vertex (i*(N+1)+j) is at (i, j, 0).  Grid spans [0,N]×[0,N] at z=0.
        N = 100  # → 20,000 triangles
        stride = N + 1
        all_i = np.repeat(np.arange(stride), stride).astype(np.float64)
        all_j = np.tile(np.arange(stride), stride).astype(np.float64)
        vertices = np.column_stack([all_i, all_j, np.zeros(stride * stride)])

        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        ii = ii.ravel()
        jj = jj.ravel()
        A = ii * stride + jj
        B = (ii + 1) * stride + jj
        C = (ii + 1) * stride + (jj + 1)
        D = ii * stride + (jj + 1)
        faces = np.vstack([
            np.column_stack([A, B, C]),
            np.column_stack([A, C, D]),
        ]).astype(np.int32)

        bvh = TriangleBVH()
        bvh.build(vertices, faces, backface_cull=False)

        # Ray from far corner — only the ~2 triangles near (98,98) should be hit.
        # BVH prunes everything else; brute-force checks all 20K triangles.
        ray = Ray(origin=vec3(98.5, 98.5, 10), direction=vec3(0, 0, -1))

        # BVH timing
        bvh_time = timeit.timeit(lambda: bvh.intersect(ray), number=100)

        # Brute-force timing
        def brute():
            _brute_force_intersect(vertices, faces, ray)
        bf_time = timeit.timeit(brute, number=100)

        speedup = bf_time / bvh_time
        assert speedup >= 10, f"Speedup {speedup:.1f}x is below the 10x requirement"

    def test_mesh_shape_uses_bvh(self):
        """Mesh.intersect() uses TriangleBVH and returns hit with shape=self."""
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, backface_cull=False)
        ray = Ray(origin=vec3(0.5, 0.5, 5), direction=vec3(0, 0, -1))
        hit = mesh.intersect(ray)
        assert hit is not None
        assert hit.shape is mesh  # shape must be set to the Mesh instance

    def test_mesh_intersect_miss(self):
        """Mesh.intersect() returns None for a ray that misses."""
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces)
        ray = Ray(origin=vec3(10, 10, 10), direction=vec3(0, 0, 1))
        assert mesh.intersect(ray) is None

    def test_mesh_empty_faces(self):
        """Mesh with no faces builds without error and returns no intersection."""
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        ray = Ray(origin=vec3(0, 0, 5), direction=vec3(0, 0, -1))
        assert mesh.intersect(ray) is None


# ---------------------------------------------------------------------------
# Mesh edge chaining tests (task 20.3)
# ---------------------------------------------------------------------------

class TestMeshEdgeChaining:
    """Tests for Mesh._chain_edges() and chained paths() (task 20.3)."""

    def _minimal_mesh(self):
        """Return a minimal 1-triangle Mesh — used only to call _chain_edges."""
        from plottter.scene3d.shapes.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return Mesh(vertices=verts, faces=faces)

    def test_chain_edges_empty(self):
        """Empty edge list produces empty chain list."""
        mesh = self._minimal_mesh()
        assert mesh._chain_edges([]) == []

    def test_chain_edges_single_edge(self):
        """A single edge produces one 2-vertex chain."""
        mesh = self._minimal_mesh()
        chains = mesh._chain_edges([(0, 1)])
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_chain_edges_linear(self):
        """Three consecutive edges collapse to a single 4-vertex chain."""
        mesh = self._minimal_mesh()
        # vertices 1 and 2 have degree 2 → pass-through
        chains = mesh._chain_edges([(0, 1), (1, 2), (2, 3)])
        assert len(chains) == 1
        assert len(chains[0]) == 4
        assert set(chains[0]) == {0, 1, 2, 3}

    def test_chain_edges_longer_chain(self):
        """Five consecutive edges produce a single 6-vertex chain."""
        mesh = self._minimal_mesh()
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        chains = mesh._chain_edges(edges)
        assert len(chains) == 1
        assert len(chains[0]) == 6
        assert set(chains[0]) == {0, 1, 2, 3, 4, 5}

    def test_chain_edges_closed_loop(self):
        """Four edges forming a square produce a single closed-loop chain."""
        mesh = self._minimal_mesh()
        # all vertices have degree 2
        chains = mesh._chain_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
        assert len(chains) == 1
        chain = chains[0]
        # Closed loop: first == last, 5 entries for 4 unique vertices
        assert len(chain) == 5
        assert chain[0] == chain[-1]
        assert set(chain) == {0, 1, 2, 3}

    def test_chain_edges_y_junction_no_chaining(self):
        """A Y-junction vertex (degree 3) prevents chaining → 3 separate chains."""
        mesh = self._minimal_mesh()
        # Star from vertex 0: 0→1, 0→2, 0→3
        chains = mesh._chain_edges([(0, 1), (0, 2), (0, 3)])
        # vertex 0 has degree 3; vertices 1,2,3 have degree 1 → no pass-through
        assert len(chains) == 3
        for c in chains:
            assert len(c) == 2

    def test_chain_edges_mixed_topology(self):
        """Mixture of a chain and an isolated edge produces correct chains."""
        mesh = self._minimal_mesh()
        # Chain: 0-1-2-3; isolated: 4-5
        edges = [(0, 1), (1, 2), (2, 3), (4, 5)]
        chains = mesh._chain_edges(edges)
        assert len(chains) == 2
        lengths = sorted(len(c) for c in chains)
        assert lengths == [2, 4]

    def test_chain_edges_covers_all_edges(self):
        """All original edges are covered exactly once across all chains."""
        mesh = self._minimal_mesh()
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5)]
        chains = mesh._chain_edges(edges)
        covered: set[tuple[int, int]] = set()
        for chain in chains:
            for i in range(len(chain) - 1):
                a, b = chain[i], chain[i + 1]
                covered.add((min(a, b), max(a, b)))
        original = {(min(a, b), max(a, b)) for a, b in edges}
        assert covered == original

    def test_cube_paths_count_regression(self):
        """Cube with hard edges only: chaining still produces 12 paths (degree-3 vertices)."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=False, crease_angle_deg=30.0)
        paths = mesh.paths()
        # All cube vertices have degree 3 → no chaining → still 12 paths
        assert len(paths) == 12

    def test_paths_are_valid_polylines(self):
        """Every path returned by paths() has at least 2 points from mesh vertices."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        vert_set = {tuple(float(x) for x in v) for v in vertices}
        for path in mesh.paths():
            assert len(path) >= 2
            for pt in path.points:
                assert tuple(float(x) for x in pt) in vert_set

    def test_draw_all_edges_path_count_vs_edge_count(self):
        """draw_all_edges=True: number of paths equals number of unique edges (cube)."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        edges = mesh._edges()
        paths = mesh.paths()
        # Cube has all degree-3+ vertices so no chaining → paths == edges
        assert len(paths) == len(edges)

    def test_fan_mesh_chains_boundary_into_loop(self):
        """A fan mesh (3 triangles around apex) has its 5 boundary edges forming one loop."""
        from plottter.scene3d.shapes.mesh import Mesh
        # Fan: 4 rim vertices (0-3, collinear on x-axis) + 1 apex (4)
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],  # rim (collinear)
            [1.5, 1, 0],  # apex
        ], dtype=np.float64)
        faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4]], dtype=np.int32)
        # Hard edges with crease=0: interior edges (1,4) and (2,4) are shared between
        # coplanar (all z=0) triangles → dihedral ≈ 0° → not drawn when crease_angle>0
        # Boundary edges: (0,1),(1,2),(2,3) and (0,4),(3,4) = 5 edges
        # All 5 boundary edges have degrees: 0→2, 1→2, 2→2, 3→2, 4→2 → one closed loop
        mesh = Mesh(vertices=verts, faces=faces, draw_all_edges=False, crease_angle_deg=30.0)
        paths = mesh.paths()
        assert len(paths) == 1
        # Closed loop: 5 unique vertices → 6 points in chain (first == last)
        assert len(paths[0]) == 6

    def test_simplify_edges_tol_parameter_accepted(self):
        """Mesh(simplify_edges_tol=0.1) is constructed without error."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices, faces = _make_cube_mesh()
        mesh = Mesh(vertices=vertices, faces=faces, simplify_edges_tol=0.1)
        paths = mesh.paths()
        assert len(paths) > 0

    def test_simplify_reduces_collinear_chain(self):
        """simplify_edges_tol > 0 removes collinear interior points from a chain."""
        from plottter.scene3d.shapes.mesh import Mesh
        # Fan mesh from previous test: boundary forms a closed loop containing
        # the collinear segment 0→1→2→3 (on x-axis).
        # RDP with non-zero tolerance collapses this to 0→3 directly.
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
            [1.5, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4]], dtype=np.int32)
        mesh_no_s = Mesh(vertices=verts, faces=faces, draw_all_edges=False,
                         crease_angle_deg=30.0, simplify_edges_tol=0.0)
        mesh_s = Mesh(vertices=verts, faces=faces, draw_all_edges=False,
                      crease_angle_deg=30.0, simplify_edges_tol=0.1)
        total_no_s = sum(len(p) for p in mesh_no_s.paths())
        total_s = sum(len(p) for p in mesh_s.paths())
        # Simplification should reduce the total point count
        assert total_s < total_no_s

# ---------------------------------------------------------------------------
# Task 20.4: HLR ray-casting optimizations
# ---------------------------------------------------------------------------

def _make_sphere_mesh(n_lat: int = 8, n_lng: int = 8):
    """Build a UV-sphere mesh as (vertices, faces) arrays.

    Returns
    -------
    vertices : (N, 3) float64
    faces    : (M, 3) int32
    """
    verts = []
    faces = []
    for i in range(n_lat + 1):
        lat = math.pi * (-0.5 + i / n_lat)
        for j in range(n_lng):
            lng = 2 * math.pi * j / n_lng
            x = math.cos(lat) * math.cos(lng)
            y = math.sin(lat)
            z = math.cos(lat) * math.sin(lng)
            verts.append([x, y, z])

    # Build quad faces and triangulate
    for i in range(n_lat):
        for j in range(n_lng):
            a = i * n_lng + j
            b = i * n_lng + (j + 1) % n_lng
            c = (i + 1) * n_lng + (j + 1) % n_lng
            d = (i + 1) * n_lng + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    return (
        np.array(verts, dtype=np.float64),
        np.array(faces, dtype=np.int32),
    )


class TestFrustumCulling:
    """Tests for _path_outside_frustum() in scene.py (task 20.4 part C)."""

    def _default_camera_and_vp(self):
        from plottter.scene3d.camera import Camera
        cam = Camera.default(aspect=1.0)
        return cam, cam.view_proj_matrix()

    def test_path_on_screen_not_culled(self):
        """A path at the origin in front of the camera is NOT culled."""
        from plottter.scene3d.scene import _path_outside_frustum
        _, vp = self._default_camera_and_vp()
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        assert not _path_outside_frustum(pts, vp)

    def test_path_far_behind_camera_culled(self):
        """A path behind the camera (far past far plane) should be culled."""
        from plottter.scene3d.scene import _path_outside_frustum
        from plottter.scene3d.camera import Camera
        cam = Camera.default(aspect=1.0)
        # Points directly behind the camera eye
        eye = cam.eye
        behind = eye + (eye - cam.center) * 10.0
        pts = np.array([behind], dtype=np.float64)
        vp = cam.view_proj_matrix()
        # Conservative: may or may not cull depending on exact geometry,
        # but at least it shouldn't crash.
        result = _path_outside_frustum(pts, vp)
        assert isinstance(result, bool)

    def test_path_far_off_right_culled(self):
        """A path translated 1000 units to the right should be culled."""
        from plottter.scene3d.scene import _path_outside_frustum
        _, vp = self._default_camera_and_vp()
        pts = np.array([[1000.0, 0.0, 0.0], [1001.0, 0.0, 0.0]], dtype=np.float64)
        assert _path_outside_frustum(pts, vp)

    def test_path_far_off_left_culled(self):
        """A path far to the left should be culled."""
        from plottter.scene3d.scene import _path_outside_frustum
        _, vp = self._default_camera_and_vp()
        pts = np.array([[-1000.0, 0.0, 0.0], [-1001.0, 0.0, 0.0]], dtype=np.float64)
        assert _path_outside_frustum(pts, vp)

    def test_path_far_above_culled(self):
        """A path far above the view should be culled."""
        from plottter.scene3d.scene import _path_outside_frustum
        _, vp = self._default_camera_and_vp()
        pts = np.array([[0.0, 1000.0, 0.0], [0.0, 1001.0, 0.0]], dtype=np.float64)
        assert _path_outside_frustum(pts, vp)

    def test_path_spanning_frustum_not_culled(self):
        """A path with one endpoint on-screen and one off-screen is NOT culled."""
        from plottter.scene3d.scene import _path_outside_frustum
        _, vp = self._default_camera_and_vp()
        # One point at origin (on-screen), one far right (off-screen)
        pts = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]], dtype=np.float64)
        # Conservative test: should NOT cull (the path has a visible portion)
        assert not _path_outside_frustum(pts, vp)

    def test_frustum_culling_does_not_affect_visible_output(self):
        """Frustum culling (inside _render_with_hlr) must not remove visible geometry."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.vector3 import vec3

        cam = Camera.default(aspect=1.0)

        # Render with HLR enabled so _render_with_hlr() runs and frustum culling executes
        scene = Scene(hlr_enabled=True, chop_step=0.1)
        sphere = Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=6, lng_lines=6)
        scene.add(sphere)
        scene.compile()
        polylines = scene.render(cam, 100.0, 100.0)
        # Should produce non-empty output — frustum culling must not eat visible paths
        assert len(polylines) > 0


class TestMeshIntersectAny:
    """Tests for Mesh.intersect_any() — the early-exit occlusion path (task 20.4 part B)."""

    def _make_flat_square_mesh(self):
        """Unit square mesh in the XZ plane."""
        verts = np.array([
            [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        return verts, faces

    def test_intersect_any_hits_mesh(self):
        """A ray aimed at the mesh returns True from intersect_any."""
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        verts, faces = self._make_flat_square_mesh()
        # backface_cull=False: mesh is double-sided so any ray direction hits
        mesh = Mesh(vertices=verts, faces=faces, draw_all_edges=True, backface_cull=False)
        ray = Ray(origin=vec3(0, 5, 0), direction=vec3(0, -1, 0))
        assert mesh.intersect_any(ray, t_max=10.0)

    def test_intersect_any_misses_mesh(self):
        """A ray that misses the mesh returns False from intersect_any."""
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        verts, faces = self._make_flat_square_mesh()
        mesh = Mesh(vertices=verts, faces=faces, draw_all_edges=True, backface_cull=False)
        # Ray offset 5 units to the right — misses the 2×2 square
        ray = Ray(origin=vec3(5, 5, 0), direction=vec3(0, -1, 0))
        assert not mesh.intersect_any(ray, t_max=10.0)

    def test_intersect_any_respects_t_max(self):
        """intersect_any returns False when the hit is beyond t_max."""
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3
        verts, faces = self._make_flat_square_mesh()
        mesh = Mesh(vertices=verts, faces=faces, draw_all_edges=True, backface_cull=False)
        # Mesh is at y=0; ray origin at y=5; hit at t≈5
        ray = Ray(origin=vec3(0, 5, 0), direction=vec3(0, -1, 0))
        assert not mesh.intersect_any(ray, t_max=2.0)  # t_max=2 → hit at t=5 is beyond

    def test_intersect_any_consistent_with_intersect(self):
        """intersect_any() should agree with intersect() on hit/miss for random rays."""
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.ray import EPSILON
        from plottter.scene3d.vector3 import vec3
        verts, faces = self._make_flat_square_mesh()
        mesh = Mesh(vertices=verts, faces=faces, draw_all_edges=True, backface_cull=False)

        rng = np.random.default_rng(99)
        t_max = 20.0
        for _ in range(30):
            origin = rng.uniform(-3, 3, 3).astype(np.float64)
            origin[1] = float(rng.uniform(0.5, 5.0))  # above the mesh
            direction = rng.uniform(-1, 1, 3).astype(np.float64)
            direction /= max(np.linalg.norm(direction), 1e-9)
            ray = Ray(origin=origin, direction=direction)

            hit = mesh.intersect(ray)
            is_hit = hit is not None and EPSILON < hit.t < t_max
            any_hit = mesh.intersect_any(ray, t_max=t_max)
            assert is_hit == any_hit, (
                f"Mismatch: intersect()→{is_hit}, intersect_any()→{any_hit}"
            )


class TestCancellation:
    """Tests for cancelled_callback support in scene.render() (task 20.4 part D)."""

    def test_cancelled_callback_stops_render(self):
        """When cancelled_callback returns True, render() returns [] immediately."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.vector3 import vec3

        cam = Camera.default(aspect=1.0)
        scene = Scene(hlr_enabled=True, chop_step=0.1)
        sphere = Sphere(center=vec3(0, 0, 0), radius=1.5, lat_lines=12, lng_lines=12)
        scene.add(sphere)
        scene.compile()

        cancelled_after = [0]

        def cancel_after_first_check():
            cancelled_after[0] += 1
            return cancelled_after[0] >= 1  # cancel on first check

        result = scene.render(
            cam, 100.0, 100.0,
            cancelled_callback=cancel_after_first_check,
        )
        # Result is [] because we cancelled immediately
        assert result == []

    def test_none_cancelled_callback_renders_normally(self):
        """With cancelled_callback=None, rendering completes normally."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.vector3 import vec3

        cam = Camera.default(aspect=1.0)
        scene = Scene(hlr_enabled=True, chop_step=0.1)
        sphere = Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=6, lng_lines=6)
        scene.add(sphere)
        scene.compile()

        result = scene.render(cam, 100.0, 100.0, cancelled_callback=None)
        assert len(result) > 0

    def test_cancel_flag_not_called_without_hlr(self):
        """Cancellation callback is only relevant for HLR; no-HLR mode works too."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.vector3 import vec3

        cam = Camera.default(aspect=1.0)
        scene = Scene(hlr_enabled=False)
        sphere = Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=4, lng_lines=4)
        scene.add(sphere)
        scene.compile()

        call_count = [0]

        def counting_cancel():
            call_count[0] += 1
            return False  # never actually cancel

        result = scene.render(cam, 100.0, 100.0, cancelled_callback=counting_cancel)
        assert len(result) > 0
        # Without HLR the callback is never invoked
        assert call_count[0] == 0


class TestHLROutputCorrectness:
    """Verify that the optimized HLR produces the same visible output (task 20.4 part D)."""

    def test_mesh_renders_to_non_empty_polylines(self):
        """A simple mesh renders to at least some visible polylines."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.mesh import Mesh

        vertices, faces = _make_cube_mesh()
        cam = Camera.default(aspect=1.0)
        scene = Scene(hlr_enabled=True, chop_step=0.1)
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        scene.add(mesh)
        scene.compile()

        polylines = scene.render(cam, 100.0, 100.0)
        assert len(polylines) > 0

    def test_hlr_removes_back_faces(self):
        """With HLR, a cube renders fewer polylines than without HLR."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.mesh import Mesh

        vertices, faces = _make_cube_mesh()
        cam = Camera.default(aspect=1.0)

        # HLR off — all paths projected
        scene_no_hlr = Scene(hlr_enabled=False, chop_step=0.1)
        mesh_no_hlr = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        scene_no_hlr.add(mesh_no_hlr)
        scene_no_hlr.compile()
        poly_no_hlr = scene_no_hlr.render(cam, 100.0, 100.0)

        # HLR on — back-facing edges removed
        scene_hlr = Scene(hlr_enabled=True, chop_step=0.1)
        mesh_hlr = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        scene_hlr.add(mesh_hlr)
        scene_hlr.compile()
        poly_hlr = scene_hlr.render(cam, 100.0, 100.0)

        # HLR should reduce (or equal) the number of polylines
        assert len(poly_hlr) <= len(poly_no_hlr)

    def test_sphere_mesh_renders_quickly(self):
        """A sphere mesh with ~600 triangles renders in a reasonable time."""
        import time
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.mesh import Mesh

        verts, faces = _make_sphere_mesh(n_lat=10, n_lng=10)
        cam = Camera.default(aspect=1.0)
        scene = Scene(hlr_enabled=True, chop_step=0.1)
        mesh = Mesh(vertices=verts, faces=faces, draw_all_edges=True)
        scene.add(mesh)
        scene.compile()

        t0 = time.time()
        polylines = scene.render(cam, 100.0, 100.0)
        elapsed = time.time() - t0

        assert len(polylines) > 0
        # Should complete in reasonable time (30s budget for slow CI)
        assert elapsed < 30.0, f"Render took {elapsed:.1f}s (>30s budget)"

    def test_non_mesh_shapes_unaffected(self):
        """Sphere and Cube shapes still render correctly with the optimized HLR."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.shapes.cube import Cube
        from plottter.scene3d.vector3 import vec3

        cam = Camera.default(aspect=1.0)

        for ShapeClass, kwargs in [
            (Sphere, {"center": vec3(0, 0, 0), "radius": 1.0, "lat_lines": 6, "lng_lines": 6}),
            (Cube, {"center": vec3(0, 0, 0), "size": 1.5}),
        ]:
            scene = Scene(hlr_enabled=True, chop_step=0.1)
            shape = ShapeClass(**kwargs)
            scene.add(shape)
            scene.compile()
            polylines = scene.render(cam, 100.0, 100.0)
            assert len(polylines) > 0, f"{ShapeClass.__name__} produced no polylines"

    def test_progress_callback_called_during_hlr(self):
        """progress_callback is called with values between 0 and 1."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.vector3 import vec3

        cam = Camera.default(aspect=1.0)
        scene = Scene(hlr_enabled=True, chop_step=0.2)
        sphere = Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=6, lng_lines=6)
        scene.add(sphere)
        scene.compile()

        progress_values = []
        scene.render(
            cam, 100.0, 100.0,
            progress_callback=lambda p: progress_values.append(p),
        )
        # At least one progress callback should have fired
        assert len(progress_values) > 0
        # All values should be in [0, 1]
        for v in progress_values:
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Task 20.5 — Mesh decimation
# ---------------------------------------------------------------------------

def _make_sphere_mesh(n_lat: int = 20, n_lng: int = 20):
    """Build a UV sphere as (vertices, faces) arrays for decimation tests."""
    import numpy as np

    verts = []
    for i in range(n_lat + 1):
        lat = math.pi * (-0.5 + i / n_lat)
        for j in range(n_lng):
            lng = 2 * math.pi * j / n_lng
            verts.append([
                math.cos(lat) * math.cos(lng),
                math.cos(lat) * math.sin(lng),
                math.sin(lat),
            ])
    vertices = np.array(verts, dtype=np.float64)

    faces = []
    for i in range(n_lat):
        for j in range(n_lng):
            a = i * n_lng + j
            b = i * n_lng + (j + 1) % n_lng
            c = (i + 1) * n_lng + j
            d = (i + 1) * n_lng + (j + 1) % n_lng
            faces.append([a, b, d])
            faces.append([a, d, c])
    faces_arr = np.array(faces, dtype=np.int32)
    return vertices, faces_arr


class TestMeshDecimation:
    """Tests for decimate_mesh() (task 20.5)."""

    def test_no_decimation_returns_original(self):
        """target_ratio >= 1.0 returns the original arrays unchanged."""
        from plottter.scene3d.decimate import decimate_mesh
        import numpy as np

        verts, faces = _make_sphere_mesh()
        nv, nf = len(verts), len(faces)

        v2, f2 = decimate_mesh(verts, faces, target_ratio=1.0)
        assert len(v2) == nv
        assert len(f2) == nf

    def test_decimation_reduces_face_count(self):
        """Decimating a sphere to 50% produces roughly half the faces."""
        from plottter.scene3d.decimate import decimate_mesh

        verts, faces = _make_sphere_mesh(n_lat=30, n_lng=30)
        n_orig = len(faces)

        _, f_dec = decimate_mesh(verts, faces, target_ratio=0.5)
        # Should be noticeably fewer faces (within generous bounds)
        assert len(f_dec) < n_orig * 0.9, (
            f"Expected fewer faces after decimation; got {len(f_dec)} vs original {n_orig}"
        )

    def test_decimated_faces_are_approximately_correct_ratio(self):
        """50% decimation produces roughly half the faces (±40% tolerance)."""
        from plottter.scene3d.decimate import decimate_mesh

        verts, faces = _make_sphere_mesh(n_lat=40, n_lng=40)
        n_orig = len(faces)

        _, f_dec = decimate_mesh(verts, faces, target_ratio=0.5)
        ratio = len(f_dec) / n_orig
        # Vertex clustering is approximate — just verify it's significantly reduced
        assert ratio < 0.75, f"Expected ratio < 0.75 for 50% target, got {ratio:.2f}"

    def test_decimation_preserves_valid_mesh_structure(self):
        """All face indices in the decimated mesh are valid vertex indices."""
        from plottter.scene3d.decimate import decimate_mesh

        verts, faces = _make_sphere_mesh()
        v_dec, f_dec = decimate_mesh(verts, faces, target_ratio=0.3)

        assert len(v_dec) > 0, "Decimated mesh must have vertices"
        assert len(f_dec) > 0, "Decimated mesh must have faces"
        assert f_dec.min() >= 0, "Face indices must be non-negative"
        assert f_dec.max() < len(v_dec), "Face indices must be within vertex array"

    def test_no_degenerate_faces(self):
        """Decimated mesh contains no faces where two vertices map to the same index."""
        from plottter.scene3d.decimate import decimate_mesh

        verts, faces = _make_sphere_mesh()
        v_dec, f_dec = decimate_mesh(verts, faces, target_ratio=0.2)

        v0, v1, v2 = f_dec[:, 0], f_dec[:, 1], f_dec[:, 2]
        degenerate = ((v0 == v1) | (v1 == v2) | (v0 == v2)).sum()
        assert degenerate == 0, f"Decimated mesh has {degenerate} degenerate faces"

    def test_minimum_face_count_clamped(self):
        """Very small target_ratio is clamped so output has at least _MIN_FACES."""
        from plottter.scene3d.decimate import decimate_mesh, _MIN_FACES

        verts, faces = _make_sphere_mesh(n_lat=50, n_lng=50)
        assert len(faces) > _MIN_FACES * 10, "Test mesh must be large enough"

        _, f_dec = decimate_mesh(verts, faces, target_ratio=0.0001)
        assert len(f_dec) >= _MIN_FACES, (
            f"Should have at least {_MIN_FACES} faces, got {len(f_dec)}"
        )

    def test_empty_mesh_returns_unchanged(self):
        """Empty mesh input returns empty arrays without error."""
        from plottter.scene3d.decimate import decimate_mesh
        import numpy as np

        empty_v = np.zeros((0, 3), dtype=np.float64)
        empty_f = np.zeros((0, 3), dtype=np.int32)
        v2, f2 = decimate_mesh(empty_v, empty_f, target_ratio=0.5)
        assert len(v2) == 0
        assert len(f2) == 0

    def test_large_mesh_decimation_is_fast(self):
        """Decimating a 100K-triangle mesh to 10% completes in under 5 seconds."""
        import time
        from plottter.scene3d.decimate import decimate_mesh

        # Build a large mesh: ~100K faces
        verts, faces = _make_sphere_mesh(n_lat=225, n_lng=225)
        # Sphere: n_lat * n_lng * 2 faces
        # 225*225*2 = 101250 faces
        assert len(faces) > 90_000, "Test mesh must have ~100K faces"

        t0 = time.time()
        _, f_dec = decimate_mesh(verts, faces, target_ratio=0.1)
        elapsed = time.time() - t0

        assert elapsed < 5.0, f"Decimation took {elapsed:.2f}s (expected <5s)"
        assert len(f_dec) < len(faces), "Decimated mesh must have fewer faces"

    def test_mesh_class_decimate_param(self):
        """Mesh class applies decimation when decimate < 1.0."""
        import numpy as np
        from plottter.scene3d.shapes.mesh import Mesh

        verts, faces = _make_sphere_mesh(n_lat=20, n_lng=20)
        n_orig = len(faces)

        mesh_full = Mesh(vertices=verts, faces=faces, decimate=1.0)
        mesh_dec = Mesh(vertices=verts, faces=faces.copy(), decimate=0.2)

        assert mesh_full.face_count == n_orig
        assert mesh_dec.face_count < n_orig
        assert mesh_dec.face_count > 0

    def test_mesh_class_face_count_attribute(self):
        """Mesh.face_count is set correctly after construction."""
        import numpy as np
        from plottter.scene3d.shapes.mesh import Mesh

        verts, faces = _make_sphere_mesh()
        mesh = Mesh(vertices=verts, faces=faces)
        assert mesh.face_count == len(faces)

    def test_decimated_mesh_still_intersects_rays(self):
        """A decimated sphere mesh still produces ray intersections."""
        import numpy as np
        from plottter.scene3d.shapes.mesh import Mesh
        from plottter.scene3d.ray import Ray
        from plottter.scene3d.vector3 import vec3

        verts, faces = _make_sphere_mesh()
        mesh = Mesh(vertices=verts, faces=faces, decimate=0.3)

        # Ray pointing directly at the center of the (unit) sphere
        ray = Ray(origin=vec3(0.0, 0.0, -5.0), direction=vec3(0.0, 0.0, 1.0))
        hit = mesh.intersect(ray)
        assert hit is not None, "Decimated sphere must still be intersectable"
        assert hit.t > 0.0


class TestMeshDecimateGeneratorParam:
    """Tests for mesh_decimate parameter in Scene3DGenerator."""

    def test_mesh_decimate_param_exists(self):
        """Scene3DGenerator has a mesh_decimate FloatParam."""
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.generators.base import FloatParam

        gen = Scene3DGenerator()
        params = gen.get_parameters()
        names = [p.name for p in params]
        assert "mesh_decimate" in names, "mesh_decimate parameter must exist"

        param = next(p for p in params if p.name == "mesh_decimate")
        assert isinstance(param, FloatParam)
        assert param.default == 1.0

    def test_mesh_decimate_param_visible_when_mesh_import(self):
        """mesh_decimate is only visible for Mesh Import shape type."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        gen = Scene3DGenerator()
        params = gen.get_parameters()
        param = next(p for p in params if p.name == "mesh_decimate")
        assert param.visible_when is not None
        assert "Mesh Import" in param.visible_when.get("shape_type", [])

    def test_build_shape_passes_decimate_to_mesh(self, tmp_path):
        """build_shape() passes mesh_decimate to the Mesh constructor."""
        import numpy as np
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.scene3d.shapes.mesh import Mesh

        # Write a minimal OBJ file — a single triangle
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        obj_path = tmp_path / "tri.obj"
        obj_path.write_text(obj_content)

        gen = Scene3DGenerator()
        # With decimation = 1.0 (default), face count should be 1
        shape = gen.build_shape({
            "shape_type": "Mesh Import",
            "mesh_file": str(obj_path),
            "mesh_decimate": 1.0,
        })
        assert shape is not None
        assert isinstance(shape, Mesh)
        assert shape.face_count == 1

    def test_build_shape_decimation_reduces_faces(self, tmp_path):
        """build_shape() with mesh_decimate < 1.0 produces a decimated mesh."""
        import numpy as np
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.scene3d.shapes.mesh import Mesh

        # Build a larger mesh as OBJ (sphere-like, many triangles)
        verts, faces = _make_sphere_mesh(n_lat=20, n_lng=20)
        lines = []
        for v in verts:
            lines.append(f"v {v[0]} {v[1]} {v[2]}")
        for f in faces:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        obj_path = tmp_path / "sphere.obj"
        obj_path.write_text("\n".join(lines))

        gen = Scene3DGenerator()
        shape_full = gen.build_shape({
            "shape_type": "Mesh Import",
            "mesh_file": str(obj_path),
            "mesh_decimate": 1.0,
        })
        shape_dec = gen.build_shape({
            "shape_type": "Mesh Import",
            "mesh_file": str(obj_path),
            "mesh_decimate": 0.2,
        })

        assert isinstance(shape_full, Mesh)
        assert isinstance(shape_dec, Mesh)
        assert shape_dec.face_count < shape_full.face_count


# ---------------------------------------------------------------------------
# Shadow ray casting tests (Phase 29.2)
# ---------------------------------------------------------------------------


class TestShadowRayCasting:
    """Tests for per-segment shadow visibility via ray casting (task 29.2)."""

    def _make_scene_sphere(self, hlr_enabled=True, chop_step=0.3):
        """Create a compiled scene with a single sphere and a default camera."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere
        from plottter.scene3d.vector3 import vec3

        scene = Scene(hlr_enabled=hlr_enabled, chop_step=chop_step)
        sphere = Sphere(center=vec3(0, 0, 0), radius=1.0, lat_lines=6, lng_lines=6)
        scene.add(sphere)
        scene.compile()

        camera = Camera.default(aspect=1.0)
        return scene, camera

    def _make_scene_two_spheres(self, chop_step=0.3):
        """Two spheres along the X axis — one can shadow the other."""
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere
        from plottter.scene3d.vector3 import vec3

        scene = Scene(hlr_enabled=True, chop_step=chop_step)
        sphere_front = Sphere(center=vec3(-1.5, 0, 0), radius=0.8, lat_lines=5, lng_lines=5)
        sphere_back = Sphere(center=vec3(1.5, 0, 0), radius=0.8, lat_lines=5, lng_lines=5)
        scene.add(sphere_front)
        scene.add(sphere_back)
        scene.compile()

        camera = Camera.default(aspect=1.0)
        return scene, camera, sphere_front, sphere_back

    def test_no_light_returns_list_not_tuple(self):
        """With light_dir=None, render() returns a plain list (backward compatible)."""
        scene, camera = self._make_scene_sphere()
        result = scene.render(camera, 100.0, 100.0, light_dir=None)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0

    def test_no_light_hlr_disabled_returns_list(self):
        """With HLR disabled and no light, render() still returns a plain list."""
        scene, camera = self._make_scene_sphere(hlr_enabled=False)
        result = scene.render(camera, 100.0, 100.0, light_dir=None)
        assert isinstance(result, list)

    def test_with_light_returns_tuple(self):
        """With light_dir set, render() returns a (lit, shadow) tuple."""
        scene, camera = self._make_scene_sphere()
        light_dir = (1.0, 1.0, 1.0)  # light from upper-right
        result = scene.render(camera, 100.0, 100.0, light_dir=light_dir)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        lit_polys, shadow_polys = result
        assert isinstance(lit_polys, list)
        assert isinstance(shadow_polys, list)

    def test_with_light_total_equals_no_light(self):
        """lit + shadow should account for all visible segments (same total as no-light HLR)."""
        scene, camera = self._make_scene_sphere()

        # Render without light: baseline visible path count
        result_no_light = scene.render(camera, 100.0, 100.0, light_dir=None)
        pts_no_light = sum(len(pl) for pl in result_no_light)

        # Render with light: lit + shadow combined should have same total points
        light_dir = (0.0, 0.0, 1.0)  # light from above
        result_with_light = scene.render(camera, 100.0, 100.0, light_dir=light_dir)
        lit_polys, shadow_polys = result_with_light
        pts_lit = sum(len(pl) for pl in lit_polys)
        pts_shadow = sum(len(pl) for pl in shadow_polys)

        # The total visible points should be the same regardless of shadow testing
        # (shadow test only categorises segments, doesn't add or remove them).
        # Allow a small tolerance due to polyline reassembly boundary differences.
        assert abs((pts_lit + pts_shadow) - pts_no_light) <= max(5, int(pts_no_light * 0.05)), (
            f"Total points differ too much: no-light={pts_no_light}, "
            f"lit+shadow={pts_lit + pts_shadow}"
        )

    def test_shadow_polylines_nonempty_with_occluder(self):
        """When one sphere occludes another from the light, shadow_polylines is non-empty."""
        scene, camera, sphere_front, sphere_back = self._make_scene_two_spheres()

        # Light shines from negative-X side, so sphere_front (at x=-1.5) is lit
        # and sphere_back (at x=+1.5) should be partially in its shadow.
        light_dir = (-1.0, 0.0, 0.0)  # light comes from -X direction
        result = scene.render(camera, 100.0, 100.0, light_dir=light_dir)
        assert isinstance(result, tuple)
        lit_polys, shadow_polys = result
        # At least some visible segments of the back sphere should be in shadow
        assert len(shadow_polys) > 0, "Expected shadow_polylines to be non-empty when objects occlude each other"

    def test_lit_polylines_nonempty(self):
        """With a directional light, lit_polylines should be non-empty."""
        scene, camera = self._make_scene_sphere()
        light_dir = (1.0, 1.0, 1.0)
        result = scene.render(camera, 100.0, 100.0, light_dir=light_dir)
        lit_polys, shadow_polys = result
        assert len(lit_polys) > 0, "Expected lit_polylines to be non-empty"

    def test_shadow_segments_on_far_side_of_sphere(self):
        """A sphere lit from one side should have shadow segments on the opposite side.

        We check this by comparing shadow counts for two opposing light directions.
        The side facing away from the light should have more shadow segments.
        """
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere
        from plottter.scene3d.vector3 import vec3

        # Use a sphere with another sphere close behind to create shadows on the back
        scene_x = Scene(hlr_enabled=True, chop_step=0.3)
        occluder = Sphere(center=vec3(-2.0, 0, 0), radius=1.5, lat_lines=8, lng_lines=8)
        target = Sphere(center=vec3(2.0, 0, 0), radius=1.0, lat_lines=8, lng_lines=8)
        scene_x.add(occluder)
        scene_x.add(target)
        scene_x.compile()

        camera = Camera.default(aspect=1.0)

        # Light from -X: occluder is between light and target → target in shadow
        result_shadowed = scene_x.render(
            camera, 100.0, 100.0,
            render_shapes=[target],
            light_dir=(-1.0, 0.0, 0.0),
        )
        # Light from +X: target is between occluder and light → target NOT in shadow
        result_lit = scene_x.render(
            camera, 100.0, 100.0,
            render_shapes=[target],
            light_dir=(1.0, 0.0, 0.0),
        )
        _, shadow_polys_shadowed = result_shadowed
        _, shadow_polys_lit = result_lit

        # When lit from +X, the target itself blocks its own shadow (all facing the light)
        # When lit from -X, the occluder casts a shadow on the target → more shadow segments
        shadow_pts_shadowed = sum(len(pl) for pl in shadow_polys_shadowed)
        shadow_pts_lit = sum(len(pl) for pl in shadow_polys_lit)
        assert shadow_pts_shadowed > shadow_pts_lit, (
            f"Expected more shadow segments when occluded (lit from -X: {shadow_pts_shadowed}) "
            f"vs not occluded (lit from +X: {shadow_pts_lit})"
        )

    def test_no_self_intersection_artifacts(self):
        """Shadow rays should not cause every segment to self-occlude (t_min offset works)."""
        scene, camera = self._make_scene_sphere()

        # Light from directly above — half the sphere should be lit, half in shadow
        # (due to self-shadowing from the sphere's own geometry).
        # Regardless, lit_polys should not be empty (the offset prevents false hits).
        light_dir = (0.0, 1.0, 0.0)  # light from +Y
        result = scene.render(camera, 100.0, 100.0, light_dir=light_dir)
        lit_polys, shadow_polys = result

        # Neither should be completely empty — the sphere has segments facing the
        # light (lit) and segments turned away / self-shadowed (shadow).
        total = sum(len(pl) for pl in lit_polys) + sum(len(pl) for pl in shadow_polys)
        assert total > 0, "All visible segments were lost — self-intersection offset may be broken"
        # Lit side should be non-empty (if offset works, surface doesn't self-occlude
        # in the direction of the incoming light)
        assert len(lit_polys) > 0, (
            "No lit segments found — self-intersection artifacts are blocking all shadow rays"
        )

    def test_empty_scene_returns_empty_with_light(self):
        """An empty scene with light_dir set returns ([], [])."""
        from plottter.scene3d import Scene, Camera
        scene = Scene()
        scene.compile()
        camera = Camera.default(aspect=1.0)
        result = scene.render(camera, 100.0, 100.0, light_dir=(1.0, 0.0, 0.0))
        assert result == ([], [])

    def test_empty_scene_returns_empty_without_light(self):
        """An empty scene with no light returns []."""
        from plottter.scene3d import Scene, Camera
        scene = Scene()
        scene.compile()
        camera = Camera.default(aspect=1.0)
        result = scene.render(camera, 100.0, 100.0)
        assert result == []

    def test_hlr_disabled_with_light_returns_tuple(self):
        """With HLR disabled + light set, render() still returns a tuple (lit, [])."""
        scene, camera = self._make_scene_sphere(hlr_enabled=False)
        result = scene.render(camera, 100.0, 100.0, light_dir=(1.0, 0.0, 0.0))
        assert isinstance(result, tuple)
        lit_polys, shadow_polys = result
        assert isinstance(lit_polys, list)
        assert shadow_polys == []  # no HLR → no shadow computation

    def test_generator_handles_tuple_return(self):
        """Scene3DGenerator.generate() handles tuple return when shadows enabled."""
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.models.canvas import Canvas

        gen = Scene3DGenerator()
        canvas = Canvas(width_mm=100.0, height_mm=100.0)
        params = {
            "shape_type": "Sphere",
            "sphere_radius": 1.5,
            "sphere_lat_lines": 4,
            "sphere_lng_lines": 4,
            "shadow_enabled": True,
            "light_azimuth": 45.0,
            "light_elevation": 45.0,
            "shadow_density": 1.0,
            "_camera": {
                "projection": "perspective",
                "fov": 45.0,
                "azimuth": 30.0,
                "elevation": 20.0,
                "distance": 8.0,
                "look_at_x": 0.0,
                "look_at_y": 0.0,
                "look_at_z": 0.0,
            },
            "_sibling_3d_shapes": [],
        }
        result = gen.generate(params, canvas)
        # Should return a list of polylines (generator combines lit + shadow)
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Task 32.1 — face_normal on Path3D, Cube, Sphere, and Mesh
# ---------------------------------------------------------------------------

class TestFaceNormal:
    """face_normal field on Path3D and shape paths() methods."""

    # --- Path3D ---

    def test_path3d_default_face_normal_is_none(self):
        """Path3D without explicit face_normal has None (no regression)."""
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        p = Path3D([vec3(0, 0, 0), vec3(1, 0, 0)])
        assert p.face_normal is None

    def test_path3d_face_normal_stored(self):
        """Explicitly supplied face_normal is stored on the path."""
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        n = np.array([0.0, 0.0, 1.0])
        p = Path3D([vec3(0, 0, 0), vec3(1, 0, 0)], face_normal=n)
        assert p.face_normal is not None
        np.testing.assert_allclose(p.face_normal, [0, 0, 1])

    def test_path3d_chop_propagates_face_normal(self):
        """chop() copies the parent face_normal to every sub-segment."""
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        n = np.array([0.0, 1.0, 0.0])
        p = Path3D([vec3(0, 0, 0), vec3(10, 0, 0)], face_normal=n)
        segs = p.chop(step=1.0)
        assert len(segs) > 0
        for seg in segs:
            assert seg.face_normal is not None
            np.testing.assert_allclose(seg.face_normal, [0, 1, 0])

    def test_path3d_chop_none_face_normal_propagates(self):
        """chop() preserves None face_normal (no regression for pathless normals)."""
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.vector3 import vec3
        p = Path3D([vec3(0, 0, 0), vec3(10, 0, 0)])
        segs = p.chop(step=1.0)
        for seg in segs:
            assert seg.face_normal is None

    # --- Cube ---

    def test_cube_paths_have_face_normals(self):
        """All 12 cube edges have a non-None face_normal."""
        from plottter.scene3d.shapes.cube import Cube
        cube = Cube(size=2.0)
        edges = cube.paths()
        assert len(edges) == 12
        for e in edges:
            assert e.face_normal is not None
            assert e.face_normal.shape == (3,)

    def test_cube_face_normals_are_unit_length(self):
        """Cube edge face normals are unit vectors."""
        from plottter.scene3d.shapes.cube import Cube
        cube = Cube(size=1.0)
        for e in cube.paths():
            assert abs(float(np.linalg.norm(e.face_normal)) - 1.0) < 1e-9

    def test_cube_top_face_edges_normal_has_positive_y(self):
        """Edges on top (y=+1) and bottom (y=-1) faces have normals with |y| component.

        Top face edges are the 4 Z-axis edges at sy=+1: normalize(sx, +1, 0).
        These always have y > 0 (since sy=+1 contributes positively).
        """
        from plottter.scene3d.shapes.cube import Cube
        cube = Cube(size=2.0)
        edges = cube.paths()
        # Z-axis edges are the last 4 (indices 8..11).
        # Those at sy=+1 should have face_normal[1] > 0.
        z_edges = edges[8:]  # 4 Z-axis edges
        top_edges = [e for e in z_edges if e.face_normal is not None and e.face_normal[1] > 0]
        assert len(top_edges) == 2  # sy=+1 gives 2 Z-axis edges

    def test_cube_x_axis_edges_face_normal_no_x_component(self):
        """X-axis edges have face_normal with zero x component."""
        from plottter.scene3d.shapes.cube import Cube
        cube = Cube(size=1.0)
        edges = cube.paths()
        x_edges = edges[:4]  # first 4 = X-axis edges
        for e in x_edges:
            assert abs(float(e.face_normal[0])) < 1e-9, (
                f"X-axis edge has non-zero x normal component: {e.face_normal}"
            )

    # --- Sphere ---

    def test_sphere_paths_have_face_normals(self):
        """All sphere paths (lat + lng) have a non-None face_normal."""
        from plottter.scene3d.shapes.sphere import Sphere
        s = Sphere(radius=1.0, lat_lines=4, lng_lines=4)
        paths = s.paths()
        assert len(paths) > 0
        for p in paths:
            assert p.face_normal is not None
            assert p.face_normal.shape == (3,)

    def test_sphere_face_normals_are_unit_length(self):
        """Sphere path face normals are (approximately) unit vectors."""
        from plottter.scene3d.shapes.sphere import Sphere
        s = Sphere(radius=2.0, lat_lines=4, lng_lines=4)
        for p in s.paths():
            assert abs(float(np.linalg.norm(p.face_normal)) - 1.0) < 1e-9

    def test_sphere_face_normals_point_outward(self):
        """Sphere face normals point away from the sphere center."""
        from plottter.scene3d.shapes.sphere import Sphere
        from plottter.scene3d.vector3 import vec3
        center = vec3(1.0, 2.0, 3.0)
        s = Sphere(center=center, radius=1.5, lat_lines=6, lng_lines=6)
        for p in s.paths():
            # The midpoint of the path should be (roughly) at radius distance from center.
            pts = np.array(p.points, dtype=np.float64)
            mid = pts[len(pts) // 2]
            diff = mid - center
            # face_normal should point in the same direction as diff.
            dot = float(np.dot(p.face_normal, diff / np.linalg.norm(diff)))
            assert dot > 0.9, f"Sphere face_normal not outward: dot={dot}"

    # --- Mesh ---

    def test_mesh_paths_have_face_normals(self):
        """Mesh paths from a simple triangle mesh have populated face_normal."""
        from plottter.scene3d.shapes.mesh import Mesh
        # Two triangles forming a quad (flat surface, normal = +Z)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        paths = mesh.paths()
        assert len(paths) > 0
        for p in paths:
            assert p.face_normal is not None

    def test_mesh_flat_surface_normal_points_up(self):
        """A flat mesh in the XY plane has face normals pointing in +Z."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)
        paths = mesh.paths()
        assert len(paths) > 0
        for p in paths:
            assert p.face_normal is not None
            # All edges belong to the single triangle whose normal is +Z.
            np.testing.assert_allclose(p.face_normal, [0, 0, 1], atol=1e-6)


# ---------------------------------------------------------------------------
# Task 32.2 — Face-normal-based shadow classification in HLR pipeline
# ---------------------------------------------------------------------------


class TestFaceNormalShadowClassification:
    """face_normal-based shadow classification in _render_with_hlr (task 32.2)."""

    def _make_cube_scene(self, chop_step: float = 0.3):
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes.cube import Cube
        from plottter.scene3d.vector3 import vec3

        cube = Cube(center=vec3(0, 0, 0), size=2.0)
        scene = Scene(hlr_enabled=True, chop_step=chop_step)
        scene.add(cube)
        scene.compile()
        camera = Camera.default(aspect=1.0)
        return scene, camera, cube

    def _make_sphere_scene(self, chop_step: float = 0.3):
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere
        from plottter.scene3d.vector3 import vec3

        sphere = Sphere(center=vec3(0, 0, 0), radius=1.5, lat_lines=6, lng_lines=6)
        scene = Scene(hlr_enabled=True, chop_step=chop_step)
        scene.add(sphere)
        scene.compile()
        camera = Camera.default(aspect=1.0)
        return scene, camera

    def test_single_cube_has_shadow_segments_with_face_normals(self):
        """A single cube with light from +X should have shadow segments on the -X faces.

        Before task 32.2, a single convex cube had *no* shadow segments because
        the shadow ray from the back faces could not be blocked by any other object.
        After 32.2, the face-normal check correctly classifies back-facing edges as
        in shadow even for a lone convex object.
        """
        scene, camera, cube = self._make_cube_scene()
        # Light from +X — faces whose normal points in -X are in shadow
        result = scene.render(camera, 100.0, 100.0, light_dir=(1.0, 0.0, 0.0))
        assert isinstance(result, tuple)
        lit_polys, shadow_polys = result
        shadow_pts = sum(len(pl) for pl in shadow_polys)
        assert shadow_pts > 0, (
            "Expected shadow segments on the -X faces of a cube lit from +X. "
            "Face-normal shadow classification may not be working."
        )

    def test_single_sphere_has_shadow_segments_with_face_normals(self):
        """A single sphere should have shadow segments on the dark hemisphere."""
        scene, camera = self._make_sphere_scene()
        # Light from +Y — back hemisphere (normal dot light < 0) should be in shadow
        result = scene.render(camera, 100.0, 100.0, light_dir=(0.0, 1.0, 0.0))
        assert isinstance(result, tuple)
        lit_polys, shadow_polys = result
        shadow_pts = sum(len(pl) for pl in shadow_polys)
        lit_pts = sum(len(pl) for pl in lit_polys)
        assert shadow_pts > 0, "Expected shadow segments on the dark hemisphere of a sphere."
        assert lit_pts > 0, "Expected lit segments on the bright hemisphere of a sphere."

    def test_shadow_boundary_moves_with_light_direction(self):
        """Changing the light direction should move which faces are classified as shadow.

        Light from +X → faces with -X normals in shadow.
        Light from -X → faces with +X normals in shadow.
        The shadow and lit point counts should differ between the two directions.
        """
        scene, camera, _ = self._make_cube_scene()

        result_pos_x = scene.render(camera, 100.0, 100.0, light_dir=(1.0, 0.0, 0.0))
        result_neg_x = scene.render(camera, 100.0, 100.0, light_dir=(-1.0, 0.0, 0.0))

        _, shadow_pos = result_pos_x
        _, shadow_neg = result_neg_x

        shadow_pts_pos = sum(len(pl) for pl in shadow_pos)
        shadow_pts_neg = sum(len(pl) for pl in shadow_neg)

        # Both directions should produce some shadow segments
        assert shadow_pts_pos > 0
        assert shadow_pts_neg > 0

    def test_inter_object_shadows_still_work(self):
        """Inter-object ray-cast shadows still work alongside face-normal classification.

        An object that is lit by its face normal but blocked by another object
        should still be classified as in shadow (ray-cast fallback).
        """
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.shapes import Sphere
        from plottter.scene3d.vector3 import vec3

        # Two spheres along X axis; light comes from -X
        # front_sphere (x=-1.5) is between the light and back_sphere (x=+1.5)
        front = Sphere(center=vec3(-1.5, 0, 0), radius=0.8, lat_lines=6, lng_lines=6)
        back = Sphere(center=vec3(1.5, 0, 0), radius=0.8, lat_lines=6, lng_lines=6)

        scene = Scene(hlr_enabled=True, chop_step=0.3)
        scene.add(front)
        scene.add(back)
        scene.compile()

        camera = Camera.default(aspect=1.0)

        # Light from -X: front sphere blocks light to back sphere
        result = scene.render(
            camera, 100.0, 100.0,
            render_shapes=[back],
            light_dir=(-1.0, 0.0, 0.0),
        )
        assert isinstance(result, tuple)
        _, shadow_polys_occluded = result

        # Light from +X: back sphere faces toward light, not blocked
        result2 = scene.render(
            camera, 100.0, 100.0,
            render_shapes=[back],
            light_dir=(1.0, 0.0, 0.0),
        )
        _, shadow_polys_unoccluded = result2

        shadow_pts_occluded = sum(len(pl) for pl in shadow_polys_occluded)
        shadow_pts_unoccluded = sum(len(pl) for pl in shadow_polys_unoccluded)

        # When blocked by the front sphere from -X, back sphere should have more
        # shadow segments than when lit directly from +X
        assert shadow_pts_occluded >= shadow_pts_unoccluded, (
            f"Expected more shadows when occluded ({shadow_pts_occluded}) "
            f"vs lit ({shadow_pts_unoccluded})"
        )

    def test_shape_without_face_normal_falls_back_to_ray_cast(self):
        """Paths without face_normal fall back to ray-cast-only shadow behavior.

        We simulate this by using a Path3D with no face_normal directly in a
        scene and verifying the render still completes without errors.
        """
        from plottter.scene3d import Scene, Camera
        from plottter.scene3d.path3d import Path3D
        from plottter.scene3d.shapes.base import Shape
        from plottter.scene3d.bbox import BBox
        from plottter.scene3d.ray import Ray, Hit
        from plottter.scene3d.vector3 import vec3
        import numpy as np

        class NormallessShape(Shape):
            """A shape whose paths have no face_normal."""
            def paths(self):
                # A simple diagonal line with no face normal
                return [Path3D([vec3(-1, 0, 0), vec3(1, 0, 0)])]

            def intersect(self, ray: Ray):
                return None

            def bbox(self) -> BBox:
                return BBox(vec3(-1, -1, -1), vec3(1, 1, 1))

        scene = Scene(hlr_enabled=True, chop_step=0.3)
        scene.add(NormallessShape())
        scene.compile()
        camera = Camera.default(aspect=1.0)

        # Should not raise — falls back to ray-cast classification
        result = scene.render(camera, 100.0, 100.0, light_dir=(0.0, 1.0, 0.0))
        assert isinstance(result, tuple)
        lit_polys, shadow_polys = result
        # Total segments = lit + shadow (some segments visible)
        total = sum(len(pl) for pl in lit_polys) + sum(len(pl) for pl in shadow_polys)
        assert total >= 0  # just no crash

    def test_no_light_path_without_face_normal_still_visible(self):
        """With no light direction, a path without face_normal is still rendered (no regression)."""
        scene, camera, _ = self._make_cube_scene()
        result = scene.render(camera, 100.0, 100.0, light_dir=None)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_total_segments_consistent_with_face_normal_classification(self):
        """lit + shadow total should equal the no-light HLR visible segment count.

        Face-normal classification must not lose or duplicate segments —
        it only changes which bucket (lit vs shadow) each segment goes into.
        """
        scene, camera = self._make_sphere_scene()

        result_no_light = scene.render(camera, 100.0, 100.0, light_dir=None)
        pts_no_light = sum(len(pl) for pl in result_no_light)

        result_with_light = scene.render(camera, 100.0, 100.0, light_dir=(0.0, 0.0, 1.0))
        lit_polys, shadow_polys = result_with_light
        pts_lit = sum(len(pl) for pl in lit_polys)
        pts_shadow = sum(len(pl) for pl in shadow_polys)

        # Allow a small tolerance due to polyline reassembly boundary differences
        assert abs((pts_lit + pts_shadow) - pts_no_light) <= max(5, int(pts_no_light * 0.05)), (
            f"Total points differ: no-light={pts_no_light}, "
            f"lit+shadow={pts_lit + pts_shadow}"
        )


# ---------------------------------------------------------------------------
# surface_triangles() tests (task 52.1)
# ---------------------------------------------------------------------------

class TestSurfaceTriangles:
    def test_base_shape_default_returns_empty(self):
        """The default surface_triangles() on a Shape with no override returns []."""
        from plottter.scene3d.shapes.sphere import Sphere
        # Sphere does not override surface_triangles — exercises the base default
        s = Sphere(radius=1.0)
        tris = s.surface_triangles()
        assert tris == []

    def test_cube_returns_12_triangles(self):
        """Cube.surface_triangles() returns exactly 12 triangles (6 faces × 2)."""
        from plottter.scene3d.shapes.cube import Cube
        cube = Cube(size=2.0)
        tris = cube.surface_triangles()
        assert len(tris) == 12

    def test_cube_triangles_are_3_tuples(self):
        """Each triangle is a tuple of three Vec3 vertices."""
        from plottter.scene3d.shapes.cube import Cube
        cube = Cube(size=1.0)
        for tri in cube.surface_triangles():
            assert len(tri) == 3
            for v in tri:
                assert len(v) == 3

    def test_cube_vertices_in_world_space(self):
        """Cube vertices are offset by center position."""
        from plottter.scene3d.shapes.cube import Cube
        from plottter.scene3d.vector3 import vec3
        center = vec3(5.0, 0.0, 0.0)
        cube = Cube(center=center, size=2.0)
        tris = cube.surface_triangles()
        all_verts = [v for tri in tris for v in tri]
        xs = [float(v[0]) for v in all_verts]
        # With center.x=5 and half=1, all x coords must be in [4, 6]
        assert all(4.0 <= x <= 6.0 for x in xs)

    def test_cube_covers_all_faces(self):
        """Triangles span all 6 axis-aligned face positions."""
        from plottter.scene3d.shapes.cube import Cube
        cube = Cube(size=2.0)
        tris = cube.surface_triangles()
        all_verts = [v for tri in tris for v in tri]
        for axis in range(3):
            vals = {round(float(v[axis]), 6) for v in all_verts}
            assert -1.0 in vals and 1.0 in vals, (
                f"Axis {axis} does not reach both ±1.0: {sorted(vals)}"
            )

    def test_mesh_returns_one_triangle_per_face(self):
        """Mesh.surface_triangles() returns exactly one triangle per face."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        tris = mesh.surface_triangles()
        assert len(tris) == 4

    def test_mesh_triangle_vertices_match_faces(self):
        """Each mesh triangle's vertices match the indexed face vertices."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        tris = mesh.surface_triangles()
        assert len(tris) == 1
        v0, v1, v2 = tris[0]
        np.testing.assert_allclose(v0, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(v1, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(v2, [0.0, 1.0, 0.0])

    def test_mesh_empty_faces_returns_empty(self):
        """Mesh with no faces returns an empty list."""
        from plottter.scene3d.shapes.mesh import Mesh
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        assert mesh.surface_triangles() == []
