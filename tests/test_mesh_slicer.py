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
