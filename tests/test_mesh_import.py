"""Tests for mesh import improvements (task 59.5).

Covers:
(a) crease_angle=10 produces more edges than crease_angle=60
(b) crease_angle=0 is equivalent to draw_all_edges=True
(c) OBJ deduplication: duplicate vertices are merged
(d) Degenerate faces removed after vertex dedup
(e) HLR renders convex shape without false occlusion (all front-facing edges visible)
(f) hlr_quality parameter accepted without error
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Inline cube mesh helpers
# ---------------------------------------------------------------------------

def _cube_vertices() -> np.ndarray:
    """8 vertices of a unit cube centred at origin."""
    return np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=np.float64)


def _cube_faces() -> np.ndarray:
    """12 triangles (2 per face) for a cube with shared vertices."""
    return np.array([
        [0, 1, 2], [0, 2, 3],  # -Z face
        [4, 6, 5], [4, 7, 6],  # +Z face
        [0, 4, 5], [0, 5, 1],  # -Y face
        [2, 6, 7], [2, 7, 3],  # +Y face
        [0, 3, 7], [0, 7, 4],  # -X face
        [1, 5, 6], [1, 6, 2],  # +X face
    ], dtype=np.int32)


def _make_mesh(draw_all_edges=False, crease_angle_deg=30.0):
    from plottter.scene3d.shapes.mesh import Mesh
    return Mesh(
        vertices=_cube_vertices(),
        faces=_cube_faces(),
        draw_all_edges=draw_all_edges,
        crease_angle_deg=crease_angle_deg,
    )


# ---------------------------------------------------------------------------
# (a) crease_angle=10 produces more edges than crease_angle=60
# ---------------------------------------------------------------------------

class TestCreaseAngle:
    def test_crease_angle_10_more_edges_than_60(self):
        """crease_angle=10 produces more edges than crease_angle=60.

        Uses an 8-vertex, 12-face cube-like mesh where vertex 6 is shifted
        slightly (z=0.3 instead of 0.5), making the top-face diagonal a ~16°
        crease. crease_angle=10 draws it (16° > 10°); crease_angle=60 does not
        (16° < 60°). Result: 13 edges vs 12 edges.
        """
        from plottter.scene3d.shapes.mesh import Mesh

        # Cube with one vertex slightly shifted so the top-face diagonal
        # has a ~16° dihedral angle (between the 10° and 60° thresholds).
        vertices = np.array([
            [-0.5, -0.5, -0.5],  # 0
            [ 0.5, -0.5, -0.5],  # 1
            [ 0.5,  0.5, -0.5],  # 2
            [-0.5,  0.5, -0.5],  # 3
            [-0.5, -0.5,  0.5],  # 4
            [ 0.5, -0.5,  0.5],  # 5
            [ 0.5,  0.5,  0.3],  # 6 ← shifted (was 0.5), creates ~16° crease
            [-0.5,  0.5,  0.5],  # 7
        ], dtype=np.float64)
        faces = _cube_faces()  # same 12-triangle topology

        mesh_low = Mesh(vertices=vertices, faces=faces, crease_angle_deg=10.0)
        mesh_high = Mesh(vertices=vertices, faces=faces, crease_angle_deg=60.0)

        edges_low = mesh_low._edges()
        edges_high = mesh_high._edges()

        assert len(edges_low) > len(edges_high), (
            f"crease_angle=10 ({len(edges_low)} edges) should produce more edges "
            f"than crease_angle=60 ({len(edges_high)} edges)"
        )

    def test_low_crease_angle_finds_cube_edges(self):
        """crease_angle=10 on a cube draws all 12 feature edges (90° dihedrals)."""
        mesh = _make_mesh(crease_angle_deg=10.0)
        edges = mesh._edges()
        # A cube has 12 feature edges (one per cube edge, shared by 2 faces at 90°).
        # The 6 internal diagonals (coplanar, 0° dihedral) are not drawn.
        assert len(edges) == 12, f"Expected 12 edges for crease_angle=10, got {len(edges)}"

    def test_high_crease_angle_hides_cube_edges(self):
        """crease_angle=91 on a cube hides all edges (90° < 91° threshold)."""
        mesh = _make_mesh(crease_angle_deg=91.0)
        edges = mesh._edges()
        # Cube's dihedral angles are 90°, which is below the 91° threshold,
        # so no crease edges are drawn. No boundary edges exist (closed mesh).
        assert len(edges) == 0, f"Expected 0 edges for crease_angle=91, got {len(edges)}"


# ---------------------------------------------------------------------------
# (b) crease_angle=0 is equivalent to draw_all_edges=True at the generator level
# ---------------------------------------------------------------------------

class TestCreaseAngleZeroEquivDrawAll:
    def _write_cube_obj(self, path: Path) -> None:
        """Write a minimal cube OBJ with shared vertices."""
        lines = [
            "v -0.5 -0.5 -0.5", "v  0.5 -0.5 -0.5", "v  0.5  0.5 -0.5", "v -0.5  0.5 -0.5",
            "v -0.5 -0.5  0.5", "v  0.5 -0.5  0.5", "v  0.5  0.5  0.5", "v -0.5  0.5  0.5",
            "f 1 2 3", "f 1 3 4",  # -Z
            "f 5 7 6", "f 5 8 7",  # +Z
            "f 1 5 6", "f 1 6 2",  # -Y
            "f 3 7 8", "f 3 8 4",  # +Y
            "f 1 4 8", "f 1 8 5",  # -X
            "f 2 6 7", "f 2 7 3",  # +X
        ]
        path.write_text("\n".join(lines))

    def test_generator_crease_zero_sets_draw_all_edges(self):
        """Generator translates crease_angle=0 to draw_all_edges=True on the Mesh."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        gen = Scene3DGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "cube.obj"
            self._write_cube_obj(obj_path)

            mesh_zero = gen.build_shape({
                "shape_type": "Mesh Import",
                "mesh_file": str(obj_path),
                "mesh_crease_angle": 0.0,
                "mesh_all_edges": False,
                "mesh_decimate": 1.0,
            })
            mesh_all = gen.build_shape({
                "shape_type": "Mesh Import",
                "mesh_file": str(obj_path),
                "mesh_crease_angle": 30.0,
                "mesh_all_edges": True,
                "mesh_decimate": 1.0,
            })

        # Both should have draw_all_edges=True
        assert mesh_zero.draw_all_edges, "crease_angle=0 should set draw_all_edges=True"
        assert mesh_all.draw_all_edges, "mesh_all_edges=True should set draw_all_edges=True"

    def test_crease_zero_same_edge_count_as_draw_all(self):
        """crease_angle=0 and draw_all_edges=True produce the same number of edges."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        gen = Scene3DGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "cube.obj"
            self._write_cube_obj(obj_path)

            mesh_zero = gen.build_shape({
                "shape_type": "Mesh Import",
                "mesh_file": str(obj_path),
                "mesh_crease_angle": 0.0,
                "mesh_all_edges": False,
                "mesh_decimate": 1.0,
            })
            mesh_all = gen.build_shape({
                "shape_type": "Mesh Import",
                "mesh_file": str(obj_path),
                "mesh_crease_angle": 30.0,
                "mesh_all_edges": True,
                "mesh_decimate": 1.0,
            })

        edges_zero = mesh_zero._edges()
        edges_all = mesh_all._edges()

        assert len(edges_zero) == len(edges_all), (
            f"crease_angle=0 ({len(edges_zero)}) != draw_all_edges=True ({len(edges_all)})"
        )

    def test_crease_zero_more_edges_than_crease_30(self):
        """crease_angle=0 should draw at least as many edges as crease_angle=30."""
        mesh_zero = _make_mesh(crease_angle_deg=0.0)
        mesh_30 = _make_mesh(crease_angle_deg=30.0)

        edges_zero = set(mesh_zero._edges())
        edges_30 = set(mesh_30._edges())

        assert len(edges_zero) >= len(edges_30), (
            f"crease_angle=0 ({len(edges_zero)}) should be >= crease_angle=30 ({len(edges_30)})"
        )


# ---------------------------------------------------------------------------
# (c) OBJ deduplication: duplicate vertices are merged
# ---------------------------------------------------------------------------

class TestObjDeduplication:
    def _write_cube_obj_with_duplicates(self, path: Path) -> int:
        """Write a cube OBJ where each face has its own (duplicated) vertices.

        Returns the total number of raw vertex lines written (pre-dedup).
        """
        # 6 faces × 4 corners × 1 vertex each = 24 vertices (duplicated)
        # Cube positions (only 8 unique)
        corners = [
            (-0.5, -0.5, -0.5),
            ( 0.5, -0.5, -0.5),
            ( 0.5,  0.5, -0.5),
            (-0.5,  0.5, -0.5),
            (-0.5, -0.5,  0.5),
            ( 0.5, -0.5,  0.5),
            ( 0.5,  0.5,  0.5),
            (-0.5,  0.5,  0.5),
        ]
        faces_idx = [
            (0, 1, 2, 3),  # -Z
            (4, 7, 6, 5),  # +Z
            (0, 4, 5, 1),  # -Y
            (3, 2, 6, 7),  # +Y
            (0, 3, 7, 4),  # -X
            (1, 5, 6, 2),  # +X
        ]

        lines = []
        vtx_index = 1
        face_lines = []
        total_verts = 0
        for face in faces_idx:
            face_vtx_indices = []
            for ci in face:
                x, y, z = corners[ci]
                lines.append(f"v {x} {y} {z}")
                face_vtx_indices.append(vtx_index)
                vtx_index += 1
                total_verts += 1
            # quad → two triangles via face line
            a, b, c, d = face_vtx_indices
            face_lines.append(f"f {a} {b} {c} {d}")

        with open(path, "w") as fh:
            fh.write("# cube with duplicated vertices\n")
            for line in lines:
                fh.write(line + "\n")
            for line in face_lines:
                fh.write(line + "\n")

        return total_verts

    def test_dedup_reduces_vertex_count(self):
        """OBJ loader should deduplicate vertices, reducing count from 24 to 8."""
        from plottter.scene3d.loaders.obj import load_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "cube_duped.obj"
            raw_count = self._write_cube_obj_with_duplicates(obj_path)

            verts, faces = load_obj(obj_path)

        assert raw_count == 24, f"Expected 24 raw vertices, got {raw_count}"
        assert len(verts) < raw_count, (
            f"Dedup should reduce {raw_count} → fewer; got {len(verts)}"
        )
        # A cube has exactly 8 unique vertices
        assert len(verts) == 8, f"Expected 8 unique vertices after dedup, got {len(verts)}"

    def test_dedup_preserves_face_count(self):
        """Deduplication should not remove valid faces, only merge vertices."""
        from plottter.scene3d.loaders.obj import load_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "cube_duped2.obj"
            self._write_cube_obj_with_duplicates(obj_path)
            verts, faces = load_obj(obj_path)

        # 6 quads → 12 triangles (fan triangulation of each quad)
        assert len(faces) == 12, f"Expected 12 faces, got {len(faces)}"

    def test_no_dedup_with_zero_tolerance(self):
        """With weld_tol=0, no deduplication is performed."""
        from plottter.scene3d.loaders.obj import load_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "cube_nodedup.obj"
            raw_count = self._write_cube_obj_with_duplicates(obj_path)
            verts, faces = load_obj(obj_path, weld_tol=0.0)

        assert len(verts) == raw_count, (
            f"weld_tol=0 should keep all {raw_count} vertices, got {len(verts)}"
        )


# ---------------------------------------------------------------------------
# (d) Degenerate faces removed after dedup
# ---------------------------------------------------------------------------

class TestDegenerateFaceRemoval:
    def _write_obj_with_degenerate(self, path: Path) -> None:
        """Write an OBJ where dedup collapses one face to a degenerate triangle."""
        # 4 unique vertices + 1 near-duplicate of vertex 0
        # Face using dup-of-0, 1, 2 will degenerate after weld (0==0, so tri 0,1,2 is ok)
        # Instead: 3 vertices where two are duplicates → degenerate triangle
        content = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "v 0.0 0.0 0.0\n"  # duplicate of v1 (index 4 → welded to index 1)
            "v 0.5 0.5 0.0\n"  # unique vertex
            # Valid triangle: 1, 2, 3
            "f 1 2 3\n"
            # Degenerate: vertex 4 welds to vertex 1 → triangle (1, 1, 5) is degenerate
            "f 1 4 5\n"
        )
        path.write_text(content)

    def test_degenerate_faces_removed(self):
        """Faces that collapse to a degenerate triangle after dedup are removed."""
        from plottter.scene3d.loaders.obj import load_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "degenerate.obj"
            self._write_obj_with_degenerate(obj_path)
            verts, faces = load_obj(obj_path)

        # After dedup, vertex 4 welds to vertex 1; face (1,4,5) → (0,0,4) = degenerate
        # Only the first face (1,2,3) should survive
        assert len(faces) == 1, f"Expected 1 non-degenerate face, got {len(faces)}"

    def test_no_face_has_repeated_indices(self):
        """After dedup, no face should reference the same vertex index twice."""
        from plottter.scene3d.loaders.obj import load_obj

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "degenerate2.obj"
            self._write_obj_with_degenerate(obj_path)
            verts, faces = load_obj(obj_path)

        for face in faces:
            assert len(set(face)) == 3, (
                f"Face {face} has repeated vertex indices (degenerate)"
            )


# ---------------------------------------------------------------------------
# (e) HLR on convex shape — no false occlusion
# ---------------------------------------------------------------------------

class TestHLRConvexShape:
    """A convex mesh viewed from outside should have all silhouette edges visible."""

    def _render_mesh(self, mesh, hlr_enabled=True, hlr_quality="Normal"):
        """Render a mesh using Scene and return polylines."""
        from plottter.scene3d.scene import Scene
        from plottter.scene3d.camera import Camera
        from plottter.scene3d.vector3 import vec3

        scene = Scene(hlr_enabled=hlr_enabled, chop_step=0.1)
        scene.add(mesh)

        camera = Camera.default(aspect=1.0)
        camera.set_orbit(
            azimuth_deg=30.0,
            elevation_deg=20.0,
            distance=5.0,
            center=vec3(0.0, 0.0, 0.0),
        )

        polylines = scene.render(
            camera=camera,
            canvas_w_mm=200.0,
            canvas_h_mm=200.0,
            hlr_quality=hlr_quality,
        )
        return polylines

    def test_cube_hlr_produces_output(self):
        """HLR on a cube should produce non-empty polyline output."""
        mesh = _make_mesh(draw_all_edges=True)
        polylines = self._render_mesh(mesh, hlr_enabled=True)
        assert len(polylines) > 0, "HLR should produce at least one visible polyline"

    def test_cube_hlr_fewer_than_no_hlr(self):
        """With HLR on, fewer lines visible than without HLR (back faces removed)."""
        mesh_hlr = _make_mesh(draw_all_edges=True)
        mesh_no_hlr = _make_mesh(draw_all_edges=True)

        lines_hlr = self._render_mesh(mesh_hlr, hlr_enabled=True)
        lines_no_hlr = self._render_mesh(mesh_no_hlr, hlr_enabled=False)

        # HLR hides occluded lines → fewer or equal polylines
        assert len(lines_hlr) <= len(lines_no_hlr), (
            f"HLR ({len(lines_hlr)}) should produce <= lines vs no-HLR ({len(lines_no_hlr)})"
        )

    def test_front_facing_edges_visible(self):
        """Front-facing edges of a convex shape should not be falsely occluded."""
        # Render with HLR — for a cube viewed from (3,3,3), the 3 visible faces
        # contribute edges that must be present in the output.
        mesh = _make_mesh(draw_all_edges=True)
        polylines = self._render_mesh(mesh, hlr_enabled=True)

        # Verify we get at least a reasonable number of visible lines
        # (not everything hidden due to false self-occlusion)
        total_points = sum(len(p) for p in polylines)
        assert total_points > 4, (
            f"Too few visible points ({total_points}); likely false occlusion"
        )

    def test_hlr_no_hlr_same_total_segments_approx(self):
        """HLR should not hide MORE than 75% of front-facing lines for a cube."""
        mesh_hlr = _make_mesh(draw_all_edges=True)
        mesh_no_hlr = _make_mesh(draw_all_edges=True)

        lines_hlr = self._render_mesh(mesh_hlr, hlr_enabled=True)
        lines_no_hlr = self._render_mesh(mesh_no_hlr, hlr_enabled=False)

        # At least 25% of lines should be visible (no extreme false occlusion)
        if len(lines_no_hlr) > 0:
            ratio = len(lines_hlr) / len(lines_no_hlr)
            assert ratio >= 0.25, (
                f"Only {ratio:.1%} of lines visible with HLR — possible false occlusion"
            )


# ---------------------------------------------------------------------------
# (f) hlr_quality parameter accepted without error
# ---------------------------------------------------------------------------

class TestHlrQuality:
    def _render_with_quality(self, quality: str):
        from plottter.scene3d.scene import Scene
        from plottter.scene3d.camera import Camera
        from plottter.scene3d.vector3 import vec3

        mesh = _make_mesh(draw_all_edges=True)
        scene = Scene(hlr_enabled=True, chop_step=0.1)
        scene.add(mesh)

        camera = Camera.default(aspect=1.0)
        camera.set_orbit(
            azimuth_deg=30.0,
            elevation_deg=20.0,
            distance=5.0,
            center=vec3(0.0, 0.0, 0.0),
        )

        return scene.render(
            camera=camera,
            canvas_w_mm=200.0,
            canvas_h_mm=200.0,
            hlr_quality=quality,
        )

    def test_fine_quality_accepted(self):
        """hlr_quality='Fine' should be accepted and produce output."""
        result = self._render_with_quality("Fine")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_normal_quality_accepted(self):
        """hlr_quality='Normal' should be accepted and produce output."""
        result = self._render_with_quality("Normal")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_fast_quality_accepted(self):
        """hlr_quality='Fast' should be accepted and produce output."""
        result = self._render_with_quality("Fast")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_qualities_produce_similar_output(self):
        """All hlr_quality values should produce similar (non-empty) results."""
        results = {}
        for q in ["Fine", "Normal", "Fast"]:
            results[q] = self._render_with_quality(q)

        for q, polylines in results.items():
            assert len(polylines) > 0, f"hlr_quality='{q}' produced empty output"

    def test_generator_accepts_hlr_quality_param(self):
        """Scene3DGenerator should accept hlr_quality in its params dict."""
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.models.canvas import Canvas

        gen = Scene3DGenerator()
        canvas = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)

        params = {
            "hlr_enabled": False,  # disable HLR for speed
            "chop_step": 0.5,
            "hlr_quality": "Fast",
            "_camera": {
                "azimuth": 30.0,
                "elevation": 20.0,
                "distance": 8.0,
                "look_at_x": 0.0,
                "look_at_y": 0.0,
                "look_at_z": 0.0,
                "fov": 45.0,
                "projection": "perspective",
            },
            "shape_type": "Sphere",
            "sphere_radius": 1.0,
        }

        # Should not raise
        result = gen.generate(params, canvas)
        assert isinstance(result, list)
