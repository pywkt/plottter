"""Tests for plane-mesh intersection / mesh slicing (task 90.1).

Covers:
(a) Slicing a unit cube at z=0.5 produces a single square contour.
(b) Slicing a sphere produces circular-ish contours.
(c) Slicing below/above the mesh produces no contours.
(d) Segments are chained into closed polylines.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _unit_cube_vf():
    """Unit cube [0,1]^3 — 8 vertices, 12 triangles.

    Slicing at z=0.5 hits no vertices (clean cut).
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=np.float64)

    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # -Z face
        [4, 5, 6], [4, 6, 7],  # +Z face
        [0, 1, 5], [0, 5, 4],  # -Y face
        [2, 3, 7], [2, 7, 6],  # +Y face
        [0, 4, 7], [0, 7, 3],  # -X face
        [1, 2, 6], [1, 6, 5],  # +X face
    ], dtype=np.int32)

    return vertices, faces


def _uv_sphere_vf(radius: float = 1.0, lat_steps: int = 16, lng_steps: int = 32):
    """Generate a UV-sphere triangle mesh centred at the origin."""
    vertices = []
    faces = []

    # Poles
    vertices.append([0.0, 0.0, radius])   # north pole = 0
    vertices.append([0.0, 0.0, -radius])  # south pole = 1

    # Latitude rings (excluding poles)
    for i in range(1, lat_steps):
        phi = math.pi * i / lat_steps  # 0 → π
        z = radius * math.cos(phi)
        r = radius * math.sin(phi)
        for j in range(lng_steps):
            theta = 2 * math.pi * j / lng_steps
            vertices.append([r * math.cos(theta), r * math.sin(theta), z])

    verts = np.array(vertices, dtype=np.float64)

    def ring_idx(lat: int, lng: int) -> int:
        """Index of vertex at latitude ring `lat` (1..lat_steps-1), longitude `lng`."""
        return 2 + (lat - 1) * lng_steps + (lng % lng_steps)

    # North-pole triangles
    for j in range(lng_steps):
        faces.append([0, ring_idx(1, j), ring_idx(1, j + 1)])

    # Middle quads → 2 triangles each
    for i in range(1, lat_steps - 1):
        for j in range(lng_steps):
            a = ring_idx(i, j)
            b = ring_idx(i, j + 1)
            c = ring_idx(i + 1, j + 1)
            d = ring_idx(i + 1, j)
            faces.append([a, b, c])
            faces.append([a, c, d])

    # South-pole triangles
    for j in range(lng_steps):
        faces.append([1, ring_idx(lat_steps - 1, j + 1), ring_idx(lat_steps - 1, j)])

    return verts, np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# (a) Unit cube at z=0.5 → single square contour
# ---------------------------------------------------------------------------

class TestCubeSlice:
    def test_single_contour(self):
        """Slicing the unit cube at z=0.5 yields exactly one contour."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 1, (
            f"Expected 1 contour for cube slice at z=0.5, got {len(contours)}"
        )

    def test_contour_is_closed(self):
        """The contour returned for the cube slice should form a closed loop."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 1
        contour = contours[0]
        # A closed loop has its first and last points identical (same index)
        # OR the chain contains 4+ unique corners forming a loop.
        # _chain_edges_by_index produces closed loops with first == last index.
        assert len(contour) >= 4, "Square contour should have at least 4 points"
        # Verify first and last points are the same (closed loop)
        x0, y0 = contour[0]
        xn, yn = contour[-1]
        assert abs(x0 - xn) < 1e-6 and abs(y0 - yn) < 1e-6, (
            "Contour should be a closed loop (first == last point)"
        )

    def test_contour_is_square(self):
        """The contour should lie on the edges of the unit square at z=0.5.

        Each cube face is split into 2 triangles by a diagonal, so the plane
        intersection may include midpoints on the square edges (e.g. x=0.5
        on an edge where y=0).  All points must lie on one of the 4 edges of
        the unit square: {x=0}, {x=1}, {y=0}, {y=1}.
        """
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 1
        pts = np.array(contours[0])
        xs = pts[:, 0]
        ys = pts[:, 1]
        # Every point must be on one of the 4 edges: x∈{0,1} or y∈{0,1}
        on_edge = (
            (np.abs(xs) < 1e-6) | (np.abs(xs - 1.0) < 1e-6) |
            (np.abs(ys) < 1e-6) | (np.abs(ys - 1.0) < 1e-6)
        )
        assert on_edge.all(), (
            f"Some points are not on the unit square edge: "
            f"{pts[~on_edge].tolist()}"
        )
        # All coordinates must be within [0, 1]
        assert xs.min() >= -1e-6 and xs.max() <= 1.0 + 1e-6
        assert ys.min() >= -1e-6 and ys.max() <= 1.0 + 1e-6

    def test_contour_perimeter(self):
        """The square contour should have perimeter ≈ 4.0."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        pts = np.array(contours[0])
        perimeter = 0.0
        for i in range(len(pts) - 1):
            dx = pts[i + 1, 0] - pts[i, 0]
            dy = pts[i + 1, 1] - pts[i, 1]
            perimeter += math.sqrt(dx * dx + dy * dy)
        assert abs(perimeter - 4.0) < 1e-5, (
            f"Square perimeter should be 4.0, got {perimeter:.6f}"
        )


# ---------------------------------------------------------------------------
# (b) Slicing a sphere produces a roughly circular contour
# ---------------------------------------------------------------------------

class TestSphereSlice:
    def test_equatorial_slice_single_contour(self):
        """Slicing a sphere at z=0 (equator) yields exactly one contour."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _uv_sphere_vf(radius=1.0, lat_steps=16, lng_steps=32)
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 1, (
            f"Expected 1 contour for equatorial sphere slice, got {len(contours)}"
        )

    def test_equatorial_slice_is_closed(self):
        """Sphere equatorial contour should be a closed loop."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _uv_sphere_vf(radius=1.0, lat_steps=16, lng_steps=32)
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 1
        contour = contours[0]
        x0, y0 = contour[0]
        xn, yn = contour[-1]
        assert abs(x0 - xn) < 1e-5 and abs(y0 - yn) < 1e-5, (
            "Sphere equatorial contour should be a closed loop"
        )

    def test_equatorial_slice_roughly_circular(self):
        """All points on the equatorial contour should be approximately radius 1.0."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _uv_sphere_vf(radius=1.0, lat_steps=16, lng_steps=32)
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        pts = np.array(contours[0])
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        # All points should be within 10% of the true radius
        assert radii.min() > 0.8, f"Min radius too small: {radii.min():.4f}"
        assert radii.max() < 1.1, f"Max radius too large: {radii.max():.4f}"

    def test_off_centre_slice(self):
        """Slicing a sphere off-centre should also produce one contour."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _uv_sphere_vf(radius=1.0, lat_steps=16, lng_steps=32)
        # Slice at z=0.5 — intersection should be a circle of radius ~0.866
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 1
        pts = np.array(contours[0])
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        expected_r = math.sqrt(1.0 - 0.5 ** 2)  # ≈ 0.866
        assert all(abs(r - expected_r) < 0.15 for r in radii), (
            f"Off-centre sphere slice radii deviating too far from {expected_r:.3f}"
        )


# ---------------------------------------------------------------------------
# (c) Slicing outside the mesh → no contours
# ---------------------------------------------------------------------------

class TestOutOfBoundsSlice:
    def test_cube_slice_above(self):
        """Slicing above the unit cube (z=2.0) produces no contours."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 2.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 0, (
            f"Expected 0 contours for slice above cube, got {len(contours)}"
        )

    def test_cube_slice_below(self):
        """Slicing below the unit cube (z=-1.0) produces no contours."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, -1.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 0, (
            f"Expected 0 contours for slice below cube, got {len(contours)}"
        )

    def test_sphere_slice_outside_radius(self):
        """Slicing beyond the sphere radius produces no contours."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _uv_sphere_vf(radius=1.0, lat_steps=16, lng_steps=32)
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 1.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        assert len(contours) == 0, (
            f"Expected 0 contours beyond sphere, got {len(contours)}"
        )


# ---------------------------------------------------------------------------
# (d) Chaining: segments form closed polylines
# ---------------------------------------------------------------------------

class TestSegmentChaining:
    def test_raw_segments_are_chained(self):
        """_slice_mesh returns raw segments; _chain_segments produces polylines."""
        from plottter.generators.mesh_slicer import _slice_mesh, _chain_segments
        vertices, faces = _unit_cube_vf()
        segments = _slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        # Should have 4 raw segments for a square cross-section (one per pair of
        # triangles on the 4 side faces that the plane crosses).
        assert len(segments) > 0, "Expected at least one raw segment"
        contours = _chain_segments(segments)
        assert len(contours) == 1, f"Expected 1 chained contour, got {len(contours)}"

    def test_chained_loop_closed(self):
        """_chain_segments closes the loop so first point == last point."""
        from plottter.generators.mesh_slicer import _slice_mesh, _chain_segments
        vertices, faces = _unit_cube_vf()
        segments = _slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        contours = _chain_segments(segments)
        chain = contours[0]
        assert np.allclose(chain[0], chain[-1], atol=1e-6), (
            "Chained loop should be closed (first == last point)"
        )

    def test_independent_segments_not_chained(self):
        """Two disjoint loops produce two separate polylines."""
        from plottter.generators.mesh_slicer import _chain_segments
        # Two squares, one in XY plane at z=0, one at z=2
        # Each forms a closed 4-segment loop
        def square_segments(z_offset):
            return [
                (np.array([0.0, 0.0, z_offset]), np.array([1.0, 0.0, z_offset])),
                (np.array([1.0, 0.0, z_offset]), np.array([1.0, 1.0, z_offset])),
                (np.array([1.0, 1.0, z_offset]), np.array([0.0, 1.0, z_offset])),
                (np.array([0.0, 1.0, z_offset]), np.array([0.0, 0.0, z_offset])),
            ]

        segs = square_segments(0.0) + square_segments(2.0)
        contours = _chain_segments(segs)
        assert len(contours) == 2, (
            f"Two disjoint loops should produce 2 contours, got {len(contours)}"
        )

    def test_empty_segments(self):
        """Empty input produces empty output."""
        from plottter.generators.mesh_slicer import _chain_segments
        assert _chain_segments([]) == []

    def test_3d_contours_returned_when_no_project(self):
        """slice_mesh with project=False returns 3D contours."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
            project=False,
        )
        assert len(contours) == 1
        # Each point should be a 3D array
        pt = contours[0][0]
        assert hasattr(pt, '__len__') and len(pt) == 3, (
            "project=False should return 3D points"
        )
        # z-coordinate should be 0.5 for all points (on the plane)
        for chain in contours:
            for p in chain:
                assert abs(p[2] - 0.5) < 1e-6, (
                    f"All points should lie on the z=0.5 plane, got z={p[2]}"
                )


# ---------------------------------------------------------------------------
# (e) Project-to-2D axis selection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_z_slice_keeps_xy(self):
        """Z-slicing: projection drops Z, keeps X and Y."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours_2d = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.0, 0.0, 0.5]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        # Each point should be a 2-tuple
        pt = contours_2d[0][0]
        assert len(pt) == 2, "Projected point should have 2 coordinates"

    def test_x_slice(self):
        """X-slicing at x=0.5 should produce a square in (Y, Z)."""
        from plottter.generators.mesh_slicer import slice_mesh
        vertices, faces = _unit_cube_vf()
        contours = slice_mesh(
            vertices, faces,
            plane_origin=np.array([0.5, 0.0, 0.0]),
            plane_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert len(contours) == 1, (
            f"X-slice of cube should give 1 contour, got {len(contours)}"
        )
        pts = np.array(contours[0])
        # After dropping X, Y and Z remain as axes 0 and 1.
        # Points must lie on the unit-square edges in (Y, Z) space.
        ys = pts[:, 0]
        zs = pts[:, 1]
        on_edge = (
            (np.abs(ys) < 1e-6) | (np.abs(ys - 1.0) < 1e-6) |
            (np.abs(zs) < 1e-6) | (np.abs(zs - 1.0) < 1e-6)
        )
        assert on_edge.all(), (
            f"Some points are not on the unit square edge: "
            f"{pts[~on_edge].tolist()}"
        )


# ---------------------------------------------------------------------------
# Helpers for task 90.2 tests
# ---------------------------------------------------------------------------

def _write_stl_ascii(path: str, vertices, faces) -> None:
    """Write a minimal ASCII STL file from vertex/face arrays."""
    with open(path, "w") as f:
        f.write("solid mesh\n")
        for tri in faces:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            nd = np.linalg.norm(n)
            if nd > 1e-12:
                n = n / nd
            else:
                n = np.array([0.0, 0.0, 1.0])
            f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid mesh\n")


# ---------------------------------------------------------------------------
# (f) _slice_mesh_multi
# ---------------------------------------------------------------------------

class TestSliceMeshMulti:
    def test_returns_correct_number_of_slices(self):
        """_slice_mesh_multi returns exactly num_slices lists."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi
        vertices, faces = _unit_cube_vf()
        result = _slice_mesh_multi(vertices, faces, axis="Z", num_slices=5,
                                   z_min=0.0, z_max=1.0)
        assert len(result) == 5, f"Expected 5 slices, got {len(result)}"

    def test_cube_each_slice_has_one_contour(self):
        """For a convex mesh like a unit cube, each interior slice has exactly 1 contour."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi
        vertices, faces = _unit_cube_vf()
        result = _slice_mesh_multi(vertices, faces, axis="Z", num_slices=5,
                                   z_min=0.0, z_max=1.0)
        for i, contours in enumerate(result):
            assert len(contours) == 1, (
                f"Slice {i}: expected 1 contour for cube, got {len(contours)}"
            )

    def test_x_axis_slicing(self):
        """_slice_mesh_multi works along the X axis."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi
        vertices, faces = _unit_cube_vf()
        result = _slice_mesh_multi(vertices, faces, axis="X", num_slices=3,
                                   z_min=0.0, z_max=1.0)
        assert len(result) == 3
        for contours in result:
            assert len(contours) == 1

    def test_y_axis_slicing(self):
        """_slice_mesh_multi works along the Y axis."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi
        vertices, faces = _unit_cube_vf()
        result = _slice_mesh_multi(vertices, faces, axis="Y", num_slices=4,
                                   z_min=0.0, z_max=1.0)
        assert len(result) == 4

    def test_contours_are_2d(self):
        """Each contour point from _slice_mesh_multi should be a 2-tuple."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi
        vertices, faces = _unit_cube_vf()
        result = _slice_mesh_multi(vertices, faces, axis="Z", num_slices=3,
                                   z_min=0.0, z_max=1.0)
        for contours in result:
            for contour in contours:
                for pt in contour:
                    assert len(pt) == 2, f"Expected 2D point, got {pt}"


# ---------------------------------------------------------------------------
# (g) MeshSlicerGenerator
# ---------------------------------------------------------------------------

class TestMeshSlicerGenerator:
    def test_registered(self):
        """MeshSlicerGenerator is registered in GENERATORS under 'Mesh Slicer'."""
        from plottter.generators import GENERATORS
        assert "Mesh Slicer" in GENERATORS, (
            f"'Mesh Slicer' not found. Keys: {list(GENERATORS.keys())}"
        )

    def test_category(self):
        """MeshSlicerGenerator has category '3d'."""
        from plottter.generators import GENERATORS
        cls = GENERATORS["Mesh Slicer"]
        assert cls.category == "3d"

    def test_empty_path_returns_empty(self):
        """generate() with no mesh_file returns an empty list."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas
        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        result = gen.generate({"mesh_file": ""}, canvas)
        assert result == []

    def test_missing_file_returns_empty(self):
        """generate() with a non-existent file path returns an empty list."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas
        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        result = gen.generate({"mesh_file": "/nonexistent/file.stl"}, canvas)
        assert result == []

    def test_cube_stl_produces_polylines(self, tmp_path):
        """Slicing a cube STL produces a non-empty list of polylines."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        result = gen.generate({
            "mesh_file": stl_path,
            "slice_axis": "Z",
            "num_slices": 5,
            "slice_spacing_mm": 2.0,
            "scale": 10.0,
        }, canvas)
        assert len(result) > 0, "Expected polylines from cube STL slice"

    def test_num_slices_controls_count(self, tmp_path):
        """num_slices parameter controls the number of output polylines."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)

        # Unit cube: each interior Z-slice gives exactly 1 contour
        for n in (5, 10, 20):
            result = gen.generate({
                "mesh_file": stl_path,
                "slice_axis": "Z",
                "num_slices": n,
                "slice_spacing_mm": 2.0,
                "scale": 10.0,
            }, canvas)
            assert len(result) == n, (
                f"num_slices={n} should produce {n} polylines, got {len(result)}"
            )

    def test_output_centered_on_canvas(self, tmp_path):
        """Output polylines should be centered on the canvas."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        result = gen.generate({
            "mesh_file": stl_path,
            "slice_axis": "Z",
            "num_slices": 10,
            "slice_spacing_mm": 2.0,
            "scale": 10.0,
        }, canvas)

        assert len(result) > 0
        all_xs = [pt[0] for poly in result for pt in poly]
        all_ys = [pt[1] for poly in result for pt in poly]
        cx = (min(all_xs) + max(all_xs)) / 2.0
        cy = (min(all_ys) + max(all_ys)) / 2.0

        assert abs(cx - canvas.width_mm / 2.0) < 0.5, (
            f"X center {cx:.2f} should be near canvas center {canvas.width_mm / 2.0:.2f}"
        )
        assert abs(cy - canvas.height_mm / 2.0) < 0.5, (
            f"Y center {cy:.2f} should be near canvas center {canvas.height_mm / 2.0:.2f}"
        )


# ---------------------------------------------------------------------------
# (h) view_mode: Plan View vs Stacked (task 90.3)
# ---------------------------------------------------------------------------

class TestViewMode:
    """Tests for the view_mode parameter added in task 90.3."""

    def test_stacked_view_produces_y_offsets(self, tmp_path):
        """Stacked view: successive slices should have increasing Y centroids."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        result = gen.generate({
            "mesh_file": stl_path,
            "slice_axis": "Z",
            "num_slices": 5,
            "view_mode": "Stacked",
            "slice_spacing_mm": 5.0,
            "scale": 10.0,
        }, canvas)

        assert len(result) == 5

        # Each poly is a single closed contour from the unit cube.
        # Their Y centroids should be monotonically increasing (stacked).
        centroids_y = [sum(pt[1] for pt in poly) / len(poly) for poly in result]
        for i in range(1, len(centroids_y)):
            assert centroids_y[i] > centroids_y[i - 1], (
                f"Stacked: slice {i} Y centroid ({centroids_y[i]:.2f}) should be "
                f"above slice {i-1} ({centroids_y[i-1]:.2f})"
            )

    def test_plan_view_contours_overlap(self, tmp_path):
        """Plan View: contours should NOT be artificially stacked (no slice_spacing_mm offset).

        We verify this by comparing Y-centroid spread: plan view spread should be much
        smaller than stacked view spread (which equals (n-1)*spacing = 4*5 = 20 mm).
        """
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)

        # Plan view run
        result_plan = gen.generate({
            "mesh_file": stl_path,
            "slice_axis": "Z",
            "num_slices": 5,
            "view_mode": "Plan View",
            "slice_spacing_mm": 5.0,
            "scale": 10.0,
        }, canvas)

        # Stacked view run (same params, different view_mode)
        result_stacked = gen.generate({
            "mesh_file": stl_path,
            "slice_axis": "Z",
            "num_slices": 5,
            "view_mode": "Stacked",
            "slice_spacing_mm": 5.0,
            "scale": 10.0,
        }, canvas)

        assert len(result_plan) == 5
        assert len(result_stacked) == 5

        def y_spread(polys):
            centroids = [sum(pt[1] for pt in poly) / len(poly) for poly in polys]
            return max(centroids) - min(centroids)

        plan_spread = y_spread(result_plan)
        stacked_spread = y_spread(result_stacked)

        # Stacked spread should be ~(n-1)*spacing = 4*5 = 20 mm
        # Plan view spread should be a tiny fraction of that (only from mesh geometry, ~scale=10mm)
        assert stacked_spread > 15.0, (
            f"Stacked view Y spread too small: {stacked_spread:.2f} (expected ~20)"
        )
        assert plan_spread < stacked_spread / 4.0, (
            f"Plan View Y spread ({plan_spread:.2f}) should be much smaller than "
            f"stacked spread ({stacked_spread:.2f})"
        )

    def test_plan_view_default_is_stacked(self, tmp_path):
        """When view_mode is omitted the generator uses Stacked (default) behaviour."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        # No view_mode key in params — should default to Stacked
        result = gen.generate({
            "mesh_file": stl_path,
            "slice_axis": "Z",
            "num_slices": 3,
            "slice_spacing_mm": 5.0,
            "scale": 10.0,
        }, canvas)

        assert len(result) == 3
        centroids_y = [sum(pt[1] for pt in poly) / len(poly) for poly in result]
        # Stacked: each successive centroid should be higher
        assert centroids_y[1] > centroids_y[0]
        assert centroids_y[2] > centroids_y[1]


# ---------------------------------------------------------------------------
# (i) Presets (task 90.3)
# ---------------------------------------------------------------------------

class TestPresets:
    """Tests for the new presets added in task 90.3."""

    def _preset_params(self, preset_name: str) -> dict:
        from plottter.generators import GENERATORS
        gen = GENERATORS["Mesh Slicer"]()
        for preset in gen.get_presets():
            if preset.name == preset_name:
                return preset.params
        raise KeyError(f"Preset '{preset_name}' not found")

    def test_topographic_map_preset_exists(self):
        """'Topographic Map' preset is registered."""
        params = self._preset_params("Topographic Map")
        assert params["view_mode"] == "Plan View"
        assert params["slice_axis"] == "Z"
        assert params["num_slices"] == 40

    def test_side_profile_preset_exists(self):
        """'Side Profile' preset is registered."""
        params = self._preset_params("Side Profile")
        assert params["view_mode"] == "Stacked"
        assert params["slice_axis"] == "Z"
        assert params["num_slices"] == 30
        assert params["slice_spacing_mm"] == 2.0

    def test_cross_sections_preset_exists(self):
        """'Cross Sections' preset is registered."""
        params = self._preset_params("Cross Sections")
        assert params["view_mode"] == "Stacked"
        assert params["slice_axis"] == "X"
        assert params["num_slices"] == 20
        assert params["slice_spacing_mm"] == 3.0

    def test_topographic_map_generates_valid_output(self, tmp_path):
        """'Topographic Map' preset produces valid polylines on a cube mesh."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        params = self._preset_params("Topographic Map")
        params["mesh_file"] = stl_path
        result = gen.generate(params, canvas)

        assert len(result) == params["num_slices"], (
            f"Topographic Map: expected {params['num_slices']} polylines, got {len(result)}"
        )
        # Plan view: Y spread should be tiny (no stacking offset applied)
        centroids_y = [sum(pt[1] for pt in poly) / len(poly) for poly in result]
        y_spread = max(centroids_y) - min(centroids_y)
        # Without stacking the spread is purely geometric: << scale (10 mm)
        assert y_spread < params.get("scale", 10.0), (
            f"Plan View Y spread ({y_spread:.2f}) should be < scale ({params.get('scale', 10.0)})"
        )

    def test_side_profile_generates_valid_output(self, tmp_path):
        """'Side Profile' preset produces stacked polylines on a cube mesh."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        params = self._preset_params("Side Profile")
        params["mesh_file"] = stl_path
        result = gen.generate(params, canvas)

        assert len(result) == params["num_slices"]
        # Stacked: each centroid Y should be monotonically increasing
        centroids_y = [sum(pt[1] for pt in poly) / len(poly) for poly in result]
        for i in range(1, len(centroids_y)):
            assert centroids_y[i] > centroids_y[i - 1]

    def test_cross_sections_generates_valid_output(self, tmp_path):
        """'Cross Sections' preset produces stacked polylines on a cube mesh."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _unit_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=210.0, height_mm=297.0)
        params = self._preset_params("Cross Sections")
        params["mesh_file"] = stl_path
        result = gen.generate(params, canvas)

        assert len(result) == params["num_slices"]
        # Stacked: each centroid Y should be monotonically increasing
        centroids_y = [sum(pt[1] for pt in poly) / len(poly) for poly in result]
        for i in range(1, len(centroids_y)):
            assert centroids_y[i] > centroids_y[i - 1]


# ---------------------------------------------------------------------------
# (j) _slice_mesh_multi_3d (task 98.1)
# ---------------------------------------------------------------------------

class TestSliceMeshMulti3D:
    """Tests for the _slice_mesh_multi_3d helper (returns 3D contours + normal)."""

    def test_returns_tuple(self):
        """_slice_mesh_multi_3d returns a tuple (normal, per_slice_contours)."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        result = _slice_mesh_multi_3d(vertices, faces, axis="Z", num_slices=5,
                                      z_min=0.0, z_max=1.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_normal_z_axis(self):
        """Normal vector for axis='Z' should be (0, 0, 1)."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        normal, _ = _slice_mesh_multi_3d(vertices, faces, axis="Z", num_slices=5,
                                         z_min=0.0, z_max=1.0)
        assert normal.shape == (3,)
        assert np.allclose(normal, [0.0, 0.0, 1.0]), (
            f"Z-axis normal should be (0,0,1), got {normal}"
        )

    def test_normal_x_axis(self):
        """Normal vector for axis='X' should be (1, 0, 0)."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        normal, _ = _slice_mesh_multi_3d(vertices, faces, axis="X", num_slices=3,
                                         z_min=0.0, z_max=1.0)
        assert np.allclose(normal, [1.0, 0.0, 0.0]), (
            f"X-axis normal should be (1,0,0), got {normal}"
        )

    def test_normal_y_axis(self):
        """Normal vector for axis='Y' should be (0, 1, 0)."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        normal, _ = _slice_mesh_multi_3d(vertices, faces, axis="Y", num_slices=3,
                                         z_min=0.0, z_max=1.0)
        assert np.allclose(normal, [0.0, 1.0, 0.0]), (
            f"Y-axis normal should be (0,1,0), got {normal}"
        )

    def test_returns_correct_number_of_slices(self):
        """_slice_mesh_multi_3d returns exactly num_slices lists."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        _, per_slice = _slice_mesh_multi_3d(vertices, faces, axis="Z", num_slices=5,
                                            z_min=0.0, z_max=1.0)
        assert len(per_slice) == 5, f"Expected 5 slices, got {len(per_slice)}"

    def test_cube_each_slice_has_one_contour(self):
        """For a convex cube, each interior slice has exactly 1 contour."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        _, per_slice = _slice_mesh_multi_3d(vertices, faces, axis="Z", num_slices=5,
                                            z_min=0.0, z_max=1.0)
        for i, contours in enumerate(per_slice):
            assert len(contours) == 1, (
                f"Slice {i}: expected 1 contour for cube, got {len(contours)}"
            )

    def test_contour_points_are_3d_arrays(self):
        """Each point in the 3D contours is a numpy array with shape (3,)."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        _, per_slice = _slice_mesh_multi_3d(vertices, faces, axis="Z", num_slices=5,
                                            z_min=0.0, z_max=1.0)
        for contours in per_slice:
            for contour in contours:
                for pt in contour:
                    assert isinstance(pt, np.ndarray), (
                        f"Expected np.ndarray, got {type(pt)}"
                    )
                    assert pt.shape == (3,), (
                        f"Expected shape (3,), got {pt.shape}"
                    )

    def test_contour_points_lie_on_plane(self):
        """All 3D contour points should have their Z coordinate at the slice plane."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        num_slices = 5
        z_min, z_max = 0.0, 1.0
        _, per_slice = _slice_mesh_multi_3d(vertices, faces, axis="Z",
                                            num_slices=num_slices,
                                            z_min=z_min, z_max=z_max)
        # Compute expected Z positions
        expected_zs = np.linspace(z_min, z_max, num_slices + 2)[1:-1]
        for i, (contours, expected_z) in enumerate(zip(per_slice, expected_zs)):
            for contour in contours:
                for pt in contour:
                    assert abs(pt[2] - expected_z) < 1e-5, (
                        f"Slice {i}: point z={pt[2]:.6f} should be {expected_z:.6f}"
                    )

    def test_3d_contours_differ_from_2d(self):
        """3D contours have shape-(3,) points; 2D contours have len-2 tuples."""
        from plottter.generators.mesh_slicer import _slice_mesh_multi, _slice_mesh_multi_3d
        vertices, faces = _unit_cube_vf()
        result_2d = _slice_mesh_multi(vertices, faces, axis="Z", num_slices=3,
                                      z_min=0.0, z_max=1.0)
        _, result_3d = _slice_mesh_multi_3d(vertices, faces, axis="Z", num_slices=3,
                                            z_min=0.0, z_max=1.0)
        # 2D: each point is a 2-tuple
        for contours in result_2d:
            for contour in contours:
                for pt in contour:
                    assert len(pt) == 2
        # 3D: each point is a (3,) array
        for contours in result_3d:
            for contour in contours:
                for pt in contour:
                    assert len(pt) == 3
