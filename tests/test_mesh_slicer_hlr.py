"""Tests for Camera mode HLR rendering in MeshSlicerGenerator (task 98.8).

Covers:
(1) test_camera_mode_basic: Camera mode with a simple inline cube mesh returns polylines.
(2) test_camera_mode_hlr_reduces_output: HLR should produce <= points vs. non-HLR.
(3) test_camera_mode_no_mesh_file: Camera mode with no mesh_file returns [] without crash.
(4) test_camera_mode_cancelled: cancelled_callback returning True → empty list.
(5) test_stacked_mode_unchanged: Stacked mode still produces correct stacked output.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _centered_cube_vf():
    """Cube centred at origin with vertices at (±1, ±1, ±1) — 8 vertices, 12 triangles."""
    vertices = np.array([
        [-1.0, -1.0, -1.0],  # 0
        [ 1.0, -1.0, -1.0],  # 1
        [ 1.0,  1.0, -1.0],  # 2
        [-1.0,  1.0, -1.0],  # 3
        [-1.0, -1.0,  1.0],  # 4
        [ 1.0, -1.0,  1.0],  # 5
        [ 1.0,  1.0,  1.0],  # 6
        [-1.0,  1.0,  1.0],  # 7
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


def _default_camera() -> dict:
    """Return a camera dict with default azimuth/elevation/distance."""
    return {
        "projection": "perspective",
        "fov": 45.0,
        "azimuth": 30.0,
        "elevation": 20.0,
        "distance": 8.0,
        "look_at_x": 0.0,
        "look_at_y": 0.0,
        "look_at_z": 0.0,
    }


# ---------------------------------------------------------------------------
# Camera mode tests
# ---------------------------------------------------------------------------

class TestCameraMode:
    """Tests for Camera view_mode added alongside HLR rendering (task 98.x)."""

    def test_camera_mode_basic(self, tmp_path):
        """Camera mode with a simple inline cube mesh returns a non-empty list of polylines."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _centered_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=200.0, height_mm=200.0)
        result = gen.generate({
            "mesh_file": stl_path,
            "view_mode": "Camera",
            "num_slices": 10,
            "slice_axis": "Z",
            "hlr_enabled": True,
            "_camera": _default_camera(),
        }, canvas)

        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Camera mode should produce at least one polyline"
        for poly in result:
            assert isinstance(poly, list)
            assert len(poly) >= 2
            for pt in poly:
                assert len(pt) == 2, f"Each point should be 2D, got {pt}"

    def test_camera_mode_hlr_reduces_output(self, tmp_path):
        """HLR-enabled output should have <= total polyline points vs. HLR-disabled."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _centered_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=200.0, height_mm=200.0)
        base_params = {
            "mesh_file": stl_path,
            "view_mode": "Camera",
            "num_slices": 10,
            "slice_axis": "Z",
            "_camera": _default_camera(),
            "chop_step": 0.05,
        }

        non_hlr_result = gen.generate({**base_params, "hlr_enabled": False}, canvas)
        hlr_result = gen.generate({**base_params, "hlr_enabled": True}, canvas)

        non_hlr_total = sum(len(p) for p in non_hlr_result)
        hlr_total = sum(len(p) for p in hlr_result)

        assert non_hlr_total > 0, "Non-HLR result should be non-empty"
        # HLR removes back-side slice segments, so total points should be <= non-HLR
        assert hlr_total <= non_hlr_total, (
            f"HLR ({hlr_total} pts) should have <= points than non-HLR ({non_hlr_total} pts)"
            " because back-side slices are removed"
        )

    def test_camera_mode_no_mesh_file(self):
        """Camera mode with no mesh_file returns empty list without crashing."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=200.0, height_mm=200.0)
        result = gen.generate({
            "mesh_file": "",
            "view_mode": "Camera",
            "_camera": _default_camera(),
        }, canvas)

        assert result == [], f"Expected [] for missing mesh_file, got {result}"

    def test_camera_mode_cancelled(self, tmp_path):
        """A cancelled_callback returning True immediately causes generate() to return []."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _centered_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=200.0, height_mm=200.0)
        result = gen.generate(
            {
                "mesh_file": stl_path,
                "view_mode": "Camera",
                "num_slices": 10,
                "slice_axis": "Z",
                "_camera": _default_camera(),
            },
            canvas,
            cancelled_callback=lambda: True,
        )

        assert result == [], "Cancelled generate() should return empty list"

    def test_stacked_mode_unchanged(self, tmp_path):
        """Stacked mode still produces correctly stacked polylines after Camera mode additions."""
        from plottter.generators import GENERATORS
        from plottter.models import Canvas

        vertices, faces = _centered_cube_vf()
        stl_path = str(tmp_path / "cube.stl")
        _write_stl_ascii(stl_path, vertices, faces)

        gen = GENERATORS["Mesh Slicer"]()
        canvas = Canvas(width_mm=200.0, height_mm=200.0)
        num_slices = 5
        result = gen.generate({
            "mesh_file": stl_path,
            "view_mode": "Stacked",
            "num_slices": num_slices,
            "slice_axis": "Z",
            "scale": 10.0,
            "slice_spacing_mm": 2.0,
        }, canvas)

        # Each interior Z-slice of a cube produces exactly one contour
        assert len(result) == num_slices, (
            f"Stacked mode with {num_slices} slices should produce {num_slices} polylines, "
            f"got {len(result)}"
        )

        # Verify polyline structure
        for poly in result:
            assert isinstance(poly, list)
            assert len(poly) >= 2
            for pt in poly:
                assert len(pt) == 2

        # Output should be centred on canvas (within 5 mm tolerance)
        all_xs = [pt[0] for poly in result for pt in poly]
        all_ys = [pt[1] for poly in result for pt in poly]
        cx = (min(all_xs) + max(all_xs)) / 2.0
        cy = (min(all_ys) + max(all_ys)) / 2.0
        assert abs(cx - canvas.width_mm / 2.0) < 5.0, (
            f"Output should be centred on canvas X, cx={cx:.2f}, canvas_cx={canvas.width_mm / 2.0:.2f}"
        )
        assert abs(cy - canvas.height_mm / 2.0) < 5.0, (
            f"Output should be centred on canvas Y, cy={cy:.2f}, canvas_cy={canvas.height_mm / 2.0:.2f}"
        )
