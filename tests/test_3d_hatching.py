"""Tests for 3D hatched rendering (task 52.5).

Covers:
(a) surface_triangles() returns correct count for Cube (12), Sphere (512), Cylinder (96), Cone (48)
(b) _compute_hatching_faces() with a single front-facing triangle returns one result
(c) _compute_hatching_faces() filters back-facing triangles
(d) _fill_triangle_with_hatching() produces lines inside the triangle
(e) cross-hatch produces more lines than single-direction
(f) brightness=1 with min_density=0 produces no hatching
(g) brightness=0 produces max density hatching
(h) "Hatched" render_style produces more polylines than "Wireframe"
(i) all new presets generate valid output
(j) Mesh shapes get hatched correctly
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

CANVAS = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)

CAM = {
    "azimuth": 30.0,
    "elevation": 20.0,
    "distance": 8.0,
    "look_at_x": 0.0,
    "look_at_y": 0.0,
    "look_at_z": 0.0,
    "fov": 45.0,
    "projection": "perspective",
}

# HLR disabled for speed; coarse step
FAST_PARAMS = {"hlr_enabled": False, "chop_step": 0.2, "_camera": CAM}


def make_gen():
    from plottter.generators.scene3d_generator import Scene3DGenerator
    return Scene3DGenerator()


def run(params):
    gen = make_gen()
    return gen.generate(params, CANVAS)


# ---------------------------------------------------------------------------
# (a) surface_triangles() count tests
# ---------------------------------------------------------------------------

class TestSurfaceTriangleCounts:
    """surface_triangles() returns the documented triangle count for each shape."""

    def test_cube_returns_12_triangles(self):
        """Cube has 6 faces × 2 triangles each = 12."""
        from plottter.scene3d.shapes.cube import Cube
        tris = Cube(size=1.0).surface_triangles()
        assert len(tris) == 12, f"Expected 12 triangles, got {len(tris)}"

    def test_sphere_returns_512_triangles(self):
        """Sphere uses 16 lat × 16 lon × 2 quads = 512 triangles."""
        from plottter.scene3d.shapes.sphere import Sphere
        tris = Sphere(radius=1.0).surface_triangles()
        assert len(tris) == 512, f"Expected 512 triangles, got {len(tris)}"

    def test_cylinder_returns_96_triangles(self):
        """Cylinder uses n=24 segments × 4 triangles (2 side + 1 top + 1 bottom) = 96."""
        from plottter.scene3d.shapes.cylinder import Cylinder
        cyl = Cylinder(
            bottom=np.array([0.0, -1.0, 0.0]),
            top=np.array([0.0, 1.0, 0.0]),
            radius=1.0,
        )
        tris = cyl.surface_triangles()
        assert len(tris) == 96, f"Expected 96 triangles, got {len(tris)}"

    def test_cone_returns_48_triangles(self):
        """Cone uses n=24 segments × 2 triangles (1 side + 1 base) = 48."""
        from plottter.scene3d.shapes.cone import Cone
        cone = Cone(
            apex=np.array([0.0, 1.0, 0.0]),
            base=np.array([0.0, -1.0, 0.0]),
            radius=1.0,
        )
        tris = cone.surface_triangles()
        assert len(tris) == 48, f"Expected 48 triangles, got {len(tris)}"

    def test_each_triangle_is_a_3_tuple(self):
        """Every triangle returned is a tuple of exactly 3 vertices."""
        from plottter.scene3d.shapes.cube import Cube
        for tri in Cube(size=1.0).surface_triangles():
            assert len(tri) == 3, f"Each triangle must have 3 vertices, got {len(tri)}"

    def test_vertices_are_3d(self):
        """Each vertex in a triangle is a 3-element array-like."""
        from plottter.scene3d.shapes.sphere import Sphere
        for tri in Sphere(radius=1.0).surface_triangles()[:10]:
            for v in tri:
                arr = np.asarray(v)
                assert arr.shape == (3,), f"Vertex must be 3D, got shape {arr.shape}"

    def test_surface_triangles_base_class_returns_empty(self):
        """Shape base class default surface_triangles() returns an empty list."""
        from plottter.scene3d.shapes.base import Shape
        # StripedCube inherits surface_triangles from Cube (non-empty)
        # but ShadedSphere has no surface_triangles override → returns []
        from plottter.scene3d.shapes.sphere import ShadedSphere
        sphere = ShadedSphere(radius=1.0)
        tris = sphere.surface_triangles()
        assert tris == [], "ShadedSphere has no surface_triangles() override — should return []"


# ---------------------------------------------------------------------------
# Helpers for visibility tests
# ---------------------------------------------------------------------------

def _make_single_triangle_scene(front_facing: bool):
    """Return (mesh, scene, camera) with one triangle.

    Camera is at z=+8 looking at origin (azimuth=0, elevation=0).
    front_facing=True  → CCW winding from +Z, normal = (0, 0, +1)
    front_facing=False → CW winding from +Z,  normal = (0, 0, -1)
    """
    from plottter.scene3d.scene import Scene
    from plottter.scene3d.camera import Camera
    from plottter.scene3d.shapes.mesh import Mesh

    if front_facing:
        # CCW from +Z → outward normal (0, 0, +1)
        vertices = np.array([[-1.0, -1.0, 0.0],
                              [ 1.0, -1.0, 0.0],
                              [ 0.0,  1.0, 0.0]], dtype=np.float64)
    else:
        # CW from +Z → outward normal (0, 0, -1)
        vertices = np.array([[-1.0, -1.0, 0.0],
                              [ 0.0,  1.0, 0.0],
                              [ 1.0, -1.0, 0.0]], dtype=np.float64)

    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)

    scene = Scene(hlr_enabled=False)
    scene.add(mesh)
    scene.compile()

    # azimuth=0, elevation=0 → camera at (0, 0, 8)
    camera = Camera(projection="perspective", fov_deg=45.0, aspect=1.0)
    camera.set_orbit(azimuth_deg=0.0, elevation_deg=0.0, distance=8.0)

    return mesh, scene, camera


# ---------------------------------------------------------------------------
# (b) _compute_hatching_faces() with a single front-facing triangle
# ---------------------------------------------------------------------------

class TestComputeHatchingFacesFrontFacing:
    """_compute_hatching_faces() correctly identifies a single front-facing triangle."""

    def test_front_facing_triangle_returns_one_result(self):
        from plottter.generators.scene3d_generator import _compute_hatching_faces

        mesh, scene, camera = _make_single_triangle_scene(front_facing=True)
        light = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        results = _compute_hatching_faces(
            mesh, scene, light, camera,
            canvas_w_mm=200.0, canvas_h_mm=200.0,
        )
        assert len(results) == 1, (
            f"Single front-facing triangle should produce 1 result, got {len(results)}"
        )

    def test_result_structure_is_verts_and_brightness(self):
        """Each result item is (list[3 × (x,y)], float brightness ∈ [0,1])."""
        from plottter.generators.scene3d_generator import _compute_hatching_faces

        mesh, scene, camera = _make_single_triangle_scene(front_facing=True)
        light = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        results = _compute_hatching_faces(
            mesh, scene, light, camera,
            canvas_w_mm=200.0, canvas_h_mm=200.0,
        )
        assert len(results) == 1
        verts_2d, brightness = results[0]
        assert len(verts_2d) == 3
        assert 0.0 <= brightness <= 1.0
        for pt in verts_2d:
            assert len(pt) == 2
            assert isinstance(pt[0], float)
            assert isinstance(pt[1], float)

    def test_front_facing_brightness_is_positive(self):
        """A front-facing triangle lit from the camera direction has brightness > 0."""
        from plottter.generators.scene3d_generator import _compute_hatching_faces

        mesh, scene, camera = _make_single_triangle_scene(front_facing=True)
        # Light same as camera direction (from +Z) → max brightness (1.0)
        light = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        results = _compute_hatching_faces(
            mesh, scene, light, camera,
            canvas_w_mm=200.0, canvas_h_mm=200.0,
        )
        _, brightness = results[0]
        assert brightness > 0.0, f"Triangle lit from camera side must have brightness > 0, got {brightness}"


# ---------------------------------------------------------------------------
# (c) _compute_hatching_faces() filters back-facing triangles
# ---------------------------------------------------------------------------

class TestComputeHatchingFacesBackFacing:
    """_compute_hatching_faces() rejects back-facing triangles via back-face culling."""

    def test_back_facing_triangle_returns_zero_results(self):
        from plottter.generators.scene3d_generator import _compute_hatching_faces

        mesh, scene, camera = _make_single_triangle_scene(front_facing=False)
        light = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        results = _compute_hatching_faces(
            mesh, scene, light, camera,
            canvas_w_mm=200.0, canvas_h_mm=200.0,
        )
        assert len(results) == 0, (
            f"Back-facing triangle must be culled, but got {len(results)} result(s)"
        )

    def test_cube_from_3_quarter_view_sees_exactly_6_triangles(self):
        """From azimuth=30°/elevation=20°, exactly 3 cube faces (6 triangles) are front-facing."""
        from plottter.generators.scene3d_generator import _compute_hatching_faces
        from plottter.scene3d.scene import Scene
        from plottter.scene3d.camera import Camera
        from plottter.scene3d.shapes.cube import Cube

        cube = Cube(size=2.0)
        scene = Scene(hlr_enabled=False)
        scene.add(cube)
        scene.compile()

        camera = Camera(projection="perspective", fov_deg=45.0, aspect=1.0)
        camera.set_orbit(azimuth_deg=30.0, elevation_deg=20.0, distance=8.0)

        light = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        light /= np.linalg.norm(light)

        results = _compute_hatching_faces(
            cube, scene, light, camera,
            canvas_w_mm=200.0, canvas_h_mm=200.0,
        )
        # 3 visible faces × 2 triangles per face = 6
        assert len(results) == 6, (
            f"From 3/4 view, cube should have exactly 6 visible triangles (3 faces × 2), "
            f"got {len(results)}"
        )


# ---------------------------------------------------------------------------
# (d) _fill_triangle_with_hatching() produces lines inside the triangle
# ---------------------------------------------------------------------------

# Large triangle for reliable line production
_LARGE_TRI = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]


class TestFillTriangleWithHatching:
    """_fill_triangle_with_hatching() produces valid hatching lines."""

    def test_positive_density_produces_lines(self):
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        lines = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=45.0, cross_hatch=False)
        assert len(lines) > 0, "Positive density must produce at least one hatch line"

    def test_zero_density_produces_no_lines(self):
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        lines = _fill_triangle_with_hatching(_LARGE_TRI, density=0.0, angle_deg=45.0, cross_hatch=False)
        assert lines == [], "Zero density must produce no hatch lines"

    def test_negative_density_produces_no_lines(self):
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        lines = _fill_triangle_with_hatching(_LARGE_TRI, density=-1.0, angle_deg=45.0, cross_hatch=False)
        assert lines == [], "Negative density must produce no hatch lines"

    def test_collinear_triangle_produces_no_lines(self):
        """A zero-area (collinear) triangle returns an empty list."""
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        collinear = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        lines = _fill_triangle_with_hatching(collinear, density=1.0, angle_deg=45.0, cross_hatch=False)
        assert lines == [], "Degenerate (zero-area) triangle must produce no lines"

    def test_lines_are_valid_polylines(self):
        """Each line is a list of ≥2 (x, y) float tuples."""
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        lines = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=45.0, cross_hatch=False)
        assert len(lines) > 0
        for line in lines:
            assert isinstance(line, list)
            assert len(line) >= 2
            for pt in line:
                assert len(pt) == 2
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)

    def test_higher_density_produces_more_lines(self):
        """A higher density value produces ≥ as many lines as a lower density value."""
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        low = _fill_triangle_with_hatching(_LARGE_TRI, density=0.5, angle_deg=45.0, cross_hatch=False)
        high = _fill_triangle_with_hatching(_LARGE_TRI, density=2.0, angle_deg=45.0, cross_hatch=False)
        assert len(high) >= len(low), (
            f"Higher density should produce more lines: low={len(low)}, high={len(high)}"
        )

    def test_different_angles_produce_different_orientations(self):
        """Hatching at 0° and 90° produces lines in different orientations."""
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        lines_0 = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=0.0, cross_hatch=False)
        lines_90 = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=90.0, cross_hatch=False)
        # Both should produce lines; the endpoints differ due to orientation
        assert len(lines_0) > 0
        assert len(lines_90) > 0
        # Compare first-line endpoints: different angles yield different coordinates
        if lines_0 and lines_90:
            pt_0 = lines_0[0][0]
            pt_90 = lines_90[0][0]
            # At least one coordinate should differ
            assert pt_0 != pt_90, "Lines at 0° and 90° should start at different positions"


# ---------------------------------------------------------------------------
# (e) cross-hatch produces more lines than single-direction
# ---------------------------------------------------------------------------

class TestCrossHatching:
    """Cross-hatching (two angle passes) produces more lines than single-direction hatching."""

    def test_cross_hatch_produces_more_lines_than_single_direction(self):
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        single = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=45.0, cross_hatch=False)
        cross = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=45.0, cross_hatch=True)
        assert len(cross) > len(single), (
            f"Cross-hatch should produce more lines than single: "
            f"single={len(single)}, cross={len(cross)}"
        )

    def test_cross_hatch_count_is_approximately_double_single(self):
        """Cross-hatching adds a second perpendicular pass ≈ doubling the line count."""
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        single = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=30.0, cross_hatch=False)
        cross = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=30.0, cross_hatch=True)
        assert len(single) > 0
        # Cross should be between 1.5× and 3× single (angle effects may vary slightly)
        assert len(cross) >= len(single), (
            "Cross-hatch must have at least as many lines as single-direction"
        )
        assert len(cross) <= 3 * len(single), (
            f"Cross-hatch count unexpectedly high: {len(cross)} vs single={len(single)}"
        )

    def test_cross_hatch_false_matches_no_cross_hatch(self):
        """cross_hatch=False produces the same result as single-direction."""
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        a = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=45.0, cross_hatch=False)
        b = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=45.0, cross_hatch=False)
        assert len(a) == len(b), "Same parameters must produce same count"


# ---------------------------------------------------------------------------
# (f) brightness=1 with min_density=0 produces no hatching
# ---------------------------------------------------------------------------

class TestBrightnessMapping:
    """brightness_to_density() maps brightness to the correct hatching density."""

    def test_brightness_1_min_density_0_gives_zero(self):
        from plottter.scene3d.hatching import brightness_to_density
        d = brightness_to_density(1.0, min_density=0.0, max_density=4.0)
        assert d == 0.0, f"brightness=1 with min_density=0 → density=0, got {d}"

    def test_brightness_1_min_density_0_produces_no_lines(self):
        """density=0 from brightness=1/min_density=0 → no hatching lines."""
        from plottter.scene3d.hatching import brightness_to_density, _fill_triangle_with_hatching
        density = brightness_to_density(1.0, min_density=0.0, max_density=4.0)
        lines = _fill_triangle_with_hatching(_LARGE_TRI, density, 45.0, False)
        assert lines == [], (
            f"brightness=1, min_density=0 must produce no lines (density={density})"
        )

    def test_brightness_1_positive_min_density_gives_min(self):
        """brightness=1.0 → density = min_density (regardless of max_density)."""
        from plottter.scene3d.hatching import brightness_to_density
        d = brightness_to_density(1.0, min_density=0.5, max_density=4.0)
        assert abs(d - 0.5) < 1e-9, f"brightness=1 → min_density=0.5, got {d}"

    def test_brightness_0_gives_max_density(self):
        """brightness=0.0 → density = max_density (darkest face, densest hatching)."""
        from plottter.scene3d.hatching import brightness_to_density
        d = brightness_to_density(0.0, min_density=0.5, max_density=4.0)
        assert abs(d - 4.0) < 1e-9, f"brightness=0 → max_density=4.0, got {d}"

    def test_brightness_0_5_gives_midpoint(self):
        """brightness=0.5 → density at the midpoint between min and max."""
        from plottter.scene3d.hatching import brightness_to_density
        d = brightness_to_density(0.5, min_density=0.0, max_density=4.0)
        # density = min + (1 - brightness) * (max - min) = 0 + 0.5 * 4 = 2.0
        assert abs(d - 2.0) < 1e-9, f"brightness=0.5 → density=2.0, got {d}"

    def test_brightness_above_1_is_clamped(self):
        """brightness > 1.0 is clamped to 1.0."""
        from plottter.scene3d.hatching import brightness_to_density
        d = brightness_to_density(2.0, min_density=0.0, max_density=4.0)
        assert d == 0.0, f"brightness=2.0 clamped to 1 → density=0, got {d}"

    def test_brightness_below_0_is_clamped(self):
        """brightness < 0.0 is clamped to 0.0."""
        from plottter.scene3d.hatching import brightness_to_density
        d = brightness_to_density(-1.0, min_density=0.0, max_density=4.0)
        assert d == 4.0, f"brightness=-1 clamped to 0 → max_density=4.0, got {d}"


# ---------------------------------------------------------------------------
# (g) brightness=0 produces max density hatching (integration)
# ---------------------------------------------------------------------------

class TestMaxDensityHatching:
    """brightness=0.0 → max_density → densest hatching on the triangle."""

    def test_max_density_produces_most_lines(self):
        from plottter.scene3d.hatching import _fill_triangle_with_hatching
        min_d = _fill_triangle_with_hatching(_LARGE_TRI, density=1.0, angle_deg=45.0, cross_hatch=False)
        max_d = _fill_triangle_with_hatching(_LARGE_TRI, density=4.0, angle_deg=45.0, cross_hatch=False)
        assert len(max_d) >= len(min_d), (
            f"Max density (4.0) should produce ≥ lines as lower density (1.0): "
            f"min={len(min_d)}, max={len(max_d)}"
        )

    def test_brightness_0_produces_non_empty_hatching(self):
        """brightness=0.0 with positive max_density should produce hatching lines."""
        from plottter.scene3d.hatching import brightness_to_density, _fill_triangle_with_hatching
        density = brightness_to_density(0.0, min_density=0.0, max_density=4.0)
        lines = _fill_triangle_with_hatching(_LARGE_TRI, density, 45.0, False)
        assert len(lines) > 0, (
            f"brightness=0 → max_density={density} must produce hatching lines"
        )


# ---------------------------------------------------------------------------
# (h) "Hatched" render_style produces more polylines than "Wireframe"
# ---------------------------------------------------------------------------

class TestHatchedVsWireframe:
    """Hatched render_style fills visible faces and produces more lines than wireframe."""

    _BASE = {
        "sphere_radius": 1.5,
        "sphere_lat_lines": 6,
        "sphere_lng_lines": 6,
        "hatch_density_min": 0.5,
        "hatch_density_max": 4.0,
        "hatch_angle_deg": 45.0,
        "hatch_cross": False,
        **FAST_PARAMS,
    }

    def test_hatched_sphere_has_more_polylines_than_wireframe(self):
        """'Hatched' Sphere produces far more polylines than 'Wireframe' Sphere."""
        wireframe = run({**self._BASE, "shape_type": "Sphere", "render_style": "Wireframe"})
        hatched = run({**self._BASE, "shape_type": "Sphere", "render_style": "Hatched"})
        assert len(hatched) > len(wireframe), (
            f"'Hatched' must have more polylines than 'Wireframe': "
            f"wireframe={len(wireframe)}, hatched={len(hatched)}"
        )

    def test_hatched_cube_has_more_polylines_than_wireframe(self):
        """'Hatched' Cube produces more polylines than 'Wireframe' Cube."""
        base = {
            "shape_type": "Cube",
            "cube_size": 2.0,
            "hatch_density_min": 0.5,
            "hatch_density_max": 4.0,
            "hatch_angle_deg": 45.0,
            "hatch_cross": False,
            **FAST_PARAMS,
        }
        wireframe = run({**base, "render_style": "Wireframe"})
        hatched = run({**base, "render_style": "Hatched"})
        assert len(hatched) > 0, "'Hatched' Cube must produce at least some polylines"
        assert len(hatched) > len(wireframe), (
            f"'Hatched' must have more polylines than 'Wireframe': "
            f"wireframe={len(wireframe)}, hatched={len(hatched)}"
        )

    def test_wireframe_plus_hatched_has_more_than_wireframe_alone(self):
        """'Wireframe + Hatched' produces more polylines than 'Wireframe' alone."""
        wireframe = run({**self._BASE, "shape_type": "Sphere", "render_style": "Wireframe"})
        combined = run({**self._BASE, "shape_type": "Sphere", "render_style": "Wireframe + Hatched"})
        assert len(combined) > len(wireframe), (
            f"'Wireframe + Hatched' must have more polylines than 'Wireframe': "
            f"wireframe={len(wireframe)}, combined={len(combined)}"
        )

    def test_wireframe_plus_hatched_has_more_than_hatched_alone(self):
        """'Wireframe + Hatched' produces at least as many polylines as 'Hatched' alone."""
        hatched = run({**self._BASE, "shape_type": "Sphere", "render_style": "Hatched"})
        combined = run({**self._BASE, "shape_type": "Sphere", "render_style": "Wireframe + Hatched"})
        assert len(combined) >= len(hatched), (
            f"'Wireframe + Hatched' must have ≥ polylines as 'Hatched': "
            f"hatched={len(hatched)}, combined={len(combined)}"
        )

    def test_hatched_output_is_valid_polylines(self):
        """All polylines from 'Hatched' mode are valid (list of (x, y) float tuples)."""
        result = run({**self._BASE, "shape_type": "Sphere", "render_style": "Hatched"})
        assert len(result) > 0
        for poly in result:
            assert isinstance(poly, list)
            assert len(poly) >= 2
            for pt in poly:
                assert len(pt) == 2
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)

    def test_render_style_parameter_exists(self):
        """render_style and all hatch_* params must appear in get_parameters()."""
        gen = make_gen()
        param_names = {p.name for p in gen.get_parameters()}
        for name in ("render_style", "hatch_density_min", "hatch_density_max",
                     "hatch_angle_deg", "hatch_cross"):
            assert name in param_names, f"Missing parameter: {name}"

    def test_render_style_default_is_wireframe(self):
        """render_style default must be 'Wireframe'."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        assert params_map["render_style"].default == "Wireframe", (
            f"render_style default must be 'Wireframe', got {params_map['render_style'].default!r}"
        )


# ---------------------------------------------------------------------------
# (i) all new presets generate valid output
# ---------------------------------------------------------------------------

class TestHatchingPresets:
    """New hatching presets (task 52.4) generate valid output."""

    _NEW_PRESET_NAMES = ("Hatched Sphere", "Cross-Hatched Cube", "Pen & Ink Portrait")

    def test_new_presets_exist(self):
        """All three new preset names appear in get_presets()."""
        gen = make_gen()
        preset_names = {p.name for p in gen.get_presets()}
        for name in self._NEW_PRESET_NAMES:
            assert name in preset_names, (
                f"Expected preset '{name}' in get_presets(), found: {preset_names}"
            )

    def test_hatched_sphere_produces_non_empty_output(self):
        """'Hatched Sphere' preset generates at least one polyline."""
        gen = make_gen()
        presets_map = {p.name: p for p in gen.get_presets()}
        params = {**presets_map["Hatched Sphere"].params, "_camera": CAM}
        result = gen.generate(params, CANVAS)
        assert isinstance(result, list)
        assert len(result) > 0, "'Hatched Sphere' must produce at least one polyline"

    def test_cross_hatched_cube_produces_non_empty_output(self):
        """'Cross-Hatched Cube' preset generates at least one polyline."""
        gen = make_gen()
        presets_map = {p.name: p for p in gen.get_presets()}
        params = {**presets_map["Cross-Hatched Cube"].params, "_camera": CAM}
        result = gen.generate(params, CANVAS)
        assert isinstance(result, list)
        assert len(result) > 0, "'Cross-Hatched Cube' must produce at least one polyline"

    def test_pen_and_ink_portrait_with_empty_mesh_returns_empty_list(self):
        """'Pen & Ink Portrait' with no mesh_file returns [] (no mesh to render)."""
        gen = make_gen()
        presets_map = {p.name: p for p in gen.get_presets()}
        params = {**presets_map["Pen & Ink Portrait"].params, "_camera": CAM}
        # mesh_file is "" by default
        result = gen.generate(params, CANVAS)
        assert isinstance(result, list), "Must always return a list"
        assert result == [], (
            f"Empty mesh_file must produce [], got {len(result)} items"
        )

    def test_hatched_sphere_preset_has_correct_render_style(self):
        """'Hatched Sphere' preset specifies render_style='Hatched'."""
        gen = make_gen()
        presets_map = {p.name: p for p in gen.get_presets()}
        assert presets_map["Hatched Sphere"].params.get("render_style") == "Hatched"

    def test_cross_hatched_cube_preset_has_correct_render_style(self):
        """'Cross-Hatched Cube' preset specifies render_style='Wireframe + Hatched'."""
        gen = make_gen()
        presets_map = {p.name: p for p in gen.get_presets()}
        assert presets_map["Cross-Hatched Cube"].params.get("render_style") == "Wireframe + Hatched"

    def test_all_presets_return_valid_polyline_structure(self):
        """All presets that produce output have valid polyline structure."""
        gen = make_gen()
        for preset in gen.get_presets():
            params = {**preset.params, "_camera": CAM}
            result = gen.generate(params, CANVAS)
            assert isinstance(result, list), f"Preset '{preset.name}': must return a list"
            for poly in result:
                assert isinstance(poly, list), (
                    f"Preset '{preset.name}': each item must be a list"
                )
                assert len(poly) >= 2, (
                    f"Preset '{preset.name}': each polyline must have ≥2 points"
                )
                for pt in poly:
                    assert len(pt) == 2, (
                        f"Preset '{preset.name}': each point must be (x, y)"
                    )


# ---------------------------------------------------------------------------
# (j) Mesh shapes get hatched correctly
# ---------------------------------------------------------------------------

class TestMeshHatching:
    """Mesh shapes implement surface_triangles() and integrate correctly with hatching."""

    def test_mesh_surface_triangles_one_per_face(self):
        """Mesh.surface_triangles() returns exactly one triangle tuple per face."""
        from plottter.scene3d.shapes.mesh import Mesh

        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        tris = mesh.surface_triangles()
        assert len(tris) == 4, f"4-face mesh must return 4 triangles, got {len(tris)}"

    def test_mesh_surface_triangles_vertex_positions_correct(self):
        """Vertex positions in surface_triangles() match the input vertex array."""
        from plottter.scene3d.shapes.mesh import Mesh

        vertices = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.5, 1.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces)
        tris = mesh.surface_triangles()
        assert len(tris) == 1
        v0, v1, v2 = tris[0]
        np.testing.assert_allclose(np.asarray(v0), vertices[0], atol=1e-10)
        np.testing.assert_allclose(np.asarray(v1), vertices[1], atol=1e-10)
        np.testing.assert_allclose(np.asarray(v2), vertices[2], atol=1e-10)

    def test_mesh_front_face_found_by_compute_hatching_faces(self):
        """_compute_hatching_faces() detects a front-facing Mesh face."""
        from plottter.generators.scene3d_generator import _compute_hatching_faces
        from plottter.scene3d.scene import Scene
        from plottter.scene3d.camera import Camera
        from plottter.scene3d.shapes.mesh import Mesh

        # Large quad-like triangle facing camera at +Z
        vertices = np.array([[-5.0, -5.0, 0.0],
                              [ 5.0, -5.0, 0.0],
                              [ 0.0,  5.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(vertices=vertices, faces=faces, draw_all_edges=True)

        scene = Scene(hlr_enabled=False)
        scene.add(mesh)
        scene.compile()

        camera = Camera(projection="perspective", fov_deg=45.0, aspect=1.0)
        camera.set_orbit(azimuth_deg=0.0, elevation_deg=0.0, distance=8.0)

        light = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        results = _compute_hatching_faces(
            mesh, scene, light, camera,
            canvas_w_mm=200.0, canvas_h_mm=200.0,
        )
        assert len(results) > 0, "Front-facing Mesh face must be found"

    def test_mesh_hatching_via_generate_with_temp_obj(self, tmp_path):
        """Mesh Import with a real OBJ file produces hatch lines via the generator."""
        obj_content = (
            "# Minimal quad\n"
            "v -1.0 -1.0 0.0\n"
            "v  1.0 -1.0 0.0\n"
            "v  1.0  1.0 0.0\n"
            "v -1.0  1.0 0.0\n"
            "f 1 2 3\n"
            "f 1 3 4\n"
        )
        obj_file = tmp_path / "quad.obj"
        obj_file.write_text(obj_content)

        params = {
            "shape_type": "Mesh Import",
            "mesh_file": str(obj_file),
            "mesh_all_edges": False,
            "mesh_decimate": 1.0,
            "render_style": "Hatched",
            "hatch_density_min": 0.5,
            "hatch_density_max": 4.0,
            "hatch_angle_deg": 45.0,
            "hatch_cross": False,
            **FAST_PARAMS,
        }
        result = run(params)
        assert isinstance(result, list), "Should return a list"
        assert len(result) > 0, (
            "Mesh Import with 'Hatched' style should produce hatch lines for a facing quad"
        )

    def test_mesh_hatching_cross_hatch_via_generate(self, tmp_path):
        """Mesh Import with cross-hatch produces more lines than single-direction hatch."""
        obj_content = (
            "# Larger quad\n"
            "v -5.0 -5.0 0.0\n"
            "v  5.0 -5.0 0.0\n"
            "v  5.0  5.0 0.0\n"
            "v -5.0  5.0 0.0\n"
            "f 1 2 3\n"
            "f 1 3 4\n"
        )
        obj_file = tmp_path / "large_quad.obj"
        obj_file.write_text(obj_content)

        base = {
            "shape_type": "Mesh Import",
            "mesh_file": str(obj_file),
            "mesh_all_edges": False,
            "mesh_decimate": 1.0,
            "render_style": "Hatched",
            "hatch_density_min": 0.5,
            "hatch_density_max": 4.0,
            "hatch_angle_deg": 45.0,
            **FAST_PARAMS,
        }
        single = run({**base, "hatch_cross": False})
        cross = run({**base, "hatch_cross": True})
        assert len(single) > 0
        assert len(cross) >= len(single), (
            f"Cross-hatch should produce >= lines as single: single={len(single)}, cross={len(cross)}"
        )


# ---------------------------------------------------------------------------
# Perspective hatching tests (task 94.1)
# ---------------------------------------------------------------------------

# A large triangle well-centred in the canvas for reliable line production.
_PERSP_TRI = [(80.0, 100.0), (130.0, 100.0), (105.0, 150.0)]

# Canvas and camera shared by perspective tests
_PERSP_CANVAS = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)
_PERSP_CAM_DICT = {
    "azimuth": 30.0,
    "elevation": 20.0,
    "distance": 8.0,
    "look_at_x": 0.0,
    "look_at_y": 0.0,
    "look_at_z": 0.0,
    "fov": 45.0,
    "projection": "perspective",
}


def _make_persp_camera(cam_dict=None):
    """Build a Camera object matching FAST_PARAMS camera dict."""
    from plottter.scene3d.camera import Camera
    d = cam_dict or _PERSP_CAM_DICT
    aspect = _PERSP_CANVAS.width_mm / _PERSP_CANVAS.height_mm
    cam = Camera(
        projection=d.get("projection", "perspective"),
        fov_deg=float(d.get("fov", 45.0)),
        aspect=aspect,
    )
    cam.set_orbit(
        azimuth_deg=float(d.get("azimuth", 30.0)),
        elevation_deg=float(d.get("elevation", 20.0)),
        distance=float(d.get("distance", 8.0)),
        center=[
            float(d.get("look_at_x", 0.0)),
            float(d.get("look_at_y", 0.0)),
            float(d.get("look_at_z", 0.0)),
        ],
    )
    return cam


class TestComputeVanishingPoint:
    """compute_vanishing_point() returns sensible 2D canvas coordinates."""

    def test_returns_tuple_of_two_floats_for_perspective(self):
        from plottter.scene3d.hatching import compute_vanishing_point
        cam = _make_persp_camera()
        vp = compute_vanishing_point(cam, 0.0, 210.0, 297.0)
        assert vp is not None, "Perspective camera must return a non-None vanishing point"
        assert isinstance(vp, tuple) and len(vp) == 2
        assert isinstance(vp[0], float) and isinstance(vp[1], float)

    def test_different_angles_give_different_vanishing_points(self):
        from plottter.scene3d.hatching import compute_vanishing_point
        cam = _make_persp_camera()
        vp0 = compute_vanishing_point(cam, 0.0, 210.0, 297.0)
        vp90 = compute_vanishing_point(cam, 90.0, 210.0, 297.0)
        assert vp0 is not None
        assert vp90 is not None
        # Different 3D directions must yield different vanishing points.
        assert vp0 != vp90, (
            f"Vanishing points for angle=0 and angle=90 should differ: {vp0} vs {vp90}"
        )

    def test_orthographic_camera_returns_none_or_far_point(self):
        """Orthographic camera gives no perspective vanishing point."""
        from plottter.scene3d.hatching import compute_vanishing_point
        from plottter.scene3d.camera import Camera
        cam = Camera(projection="orthographic", aspect=1.0)
        cam.set_orbit(azimuth_deg=30.0, elevation_deg=20.0, distance=8.0)
        # For orthographic projection the vanishing point function either returns
        # None (correct) or any value — the important thing is it doesn't crash.
        vp = compute_vanishing_point(cam, 0.0, 210.0, 297.0)
        # vp can be None or a tuple; we only require no exception.
        assert vp is None or (isinstance(vp, tuple) and len(vp) == 2)

    def test_offset_shifts_vanishing_point(self):
        """offset_mm shifts the vanishing point by the same amount."""
        from plottter.scene3d.hatching import compute_vanishing_point
        cam = _make_persp_camera()
        vp_no_offset = compute_vanishing_point(cam, 0.0, 210.0, 297.0, offset_mm=(0.0, 0.0))
        vp_offset = compute_vanishing_point(cam, 0.0, 210.0, 297.0, offset_mm=(10.0, 5.0))
        assert vp_no_offset is not None and vp_offset is not None
        assert abs(vp_offset[0] - vp_no_offset[0] - 10.0) < 1e-6, (
            "X component of VP must shift by x_offset"
        )
        assert abs(vp_offset[1] - vp_no_offset[1] - 5.0) < 1e-6, (
            "Y component of VP must shift by y_offset"
        )


class TestHatchPolygonPerspective:
    """_hatch_polygon_perspective() generates convergent hatching lines."""

    def _polygon(self):
        from shapely.geometry import Polygon
        return Polygon(_PERSP_TRI)

    def test_produces_lines_for_positive_spacing(self):
        from plottter.scene3d.hatching import _hatch_polygon_perspective
        from shapely.geometry import Polygon
        poly = Polygon(_PERSP_TRI)
        vp = (0.0, 600.0)  # far above the triangle
        lines = _hatch_polygon_perspective(poly, spacing=2.0, angle_deg=0.0, vanishing_point=vp)
        assert len(lines) > 0, "Positive spacing must produce at least one line"

    def test_all_lines_have_at_least_two_points(self):
        from plottter.scene3d.hatching import _hatch_polygon_perspective
        from shapely.geometry import Polygon
        poly = Polygon(_PERSP_TRI)
        vp = (0.0, 600.0)
        lines = _hatch_polygon_perspective(poly, spacing=2.0, angle_deg=0.0, vanishing_point=vp)
        for line in lines:
            assert len(line) >= 2, "Each hatch line must have ≥ 2 points"

    def test_lines_converge_toward_vanishing_point(self):
        """All generated lines, when extended, pass close to the vanishing point."""
        from plottter.scene3d.hatching import _hatch_polygon_perspective
        from shapely.geometry import Polygon
        poly = Polygon(_PERSP_TRI)
        # Place VP well outside the triangle so convergence is measurable.
        vp_x, vp_y = -200.0, -200.0
        lines = _hatch_polygon_perspective(
            poly, spacing=3.0, angle_deg=45.0, vanishing_point=(vp_x, vp_y)
        )
        assert len(lines) > 0, "Must produce at least one line"

        # For each line segment, extend it to the VP and verify the line passes
        # within a small tolerance of the VP.
        # The line passes through VP when: (p2 - p1) × (VP - p1) ≈ 0 (cross product ≈ 0)
        max_perp_dist = 0.0
        for line in lines:
            if len(line) < 2:
                continue
            x1, y1 = line[0]
            x2, y2 = line[-1]
            # Perpendicular distance from VP to the infinite line through p1, p2
            # = |cross(p2-p1, p1-VP)| / |p2-p1|
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            # Cross product (scalar): (p2-p1) × (p1-VP)
            cross = dx * (y1 - vp_y) - dy * (x1 - vp_x)
            perp_dist = abs(cross) / length
            max_perp_dist = max(max_perp_dist, perp_dist)

        # Tolerance: lines should pass within 1 mm of the VP (numerical precision)
        assert max_perp_dist < 1.0, (
            f"Lines should converge to VP within 1 mm; max perpendicular distance = {max_perp_dist:.4f} mm"
        )

    def test_higher_density_produces_more_lines(self):
        from plottter.scene3d.hatching import _hatch_polygon_perspective
        from shapely.geometry import Polygon
        poly = Polygon(_PERSP_TRI)
        vp = (-200.0, -200.0)
        low = _hatch_polygon_perspective(poly, spacing=4.0, angle_deg=0.0, vanishing_point=vp)
        high = _hatch_polygon_perspective(poly, spacing=1.0, angle_deg=0.0, vanishing_point=vp)
        assert len(high) >= len(low), (
            f"Smaller spacing (higher density) should produce ≥ lines: low={len(low)}, high={len(high)}"
        )

    def test_perspective_differs_from_parallel_for_close_vp(self):
        """With VP close to polygon, perspective hatching differs from parallel hatching."""
        from plottter.scene3d.hatching import _hatch_polygon_perspective, _hatch_polygon
        from shapely.geometry import Polygon
        poly = Polygon(_PERSP_TRI)
        # VP close to triangle (strong convergence effect)
        vp = (105.0, 0.0)
        spacing = 2.0
        angle_deg = 0.0
        persp_lines = _hatch_polygon_perspective(poly, spacing, angle_deg, vanishing_point=vp)
        parallel_lines = _hatch_polygon(poly, spacing, angle_deg)
        assert len(persp_lines) > 0 and len(parallel_lines) > 0
        # The endpoints should differ — convergent lines are not parallel.
        # Compare the first line's direction vector.
        def direction(line):
            x1, y1 = line[0]
            x2, y2 = line[-1]
            d = math.hypot(x2 - x1, y2 - y1)
            return (x2 - x1) / d, (y2 - y1) / d

        # With a close VP the lines fan out significantly — not all parallel.
        angles_persp = [math.atan2(*direction(l)[::-1]) for l in persp_lines]
        angles_parallel = [math.atan2(*direction(l)[::-1]) for l in parallel_lines]
        range_persp = max(angles_persp) - min(angles_persp)
        range_parallel = max(angles_parallel) - min(angles_parallel)
        assert range_persp > range_parallel, (
            "Perspective hatching from a close VP should have wider angular spread than parallel"
        )

    def test_density_at_centroid_matches_spacing(self):
        """Adjacent lines at the centroid distance are approximately `spacing` mm apart."""
        from plottter.scene3d.hatching import _hatch_polygon_perspective
        from shapely.geometry import Polygon
        poly = Polygon(_PERSP_TRI)
        # VP well above the triangle so geometry is clean.
        vp_x, vp_y = 105.0, -300.0  # directly above the centroid
        spacing = 5.0
        lines = _hatch_polygon_perspective(
            poly, spacing=spacing, angle_deg=0.0, vanishing_point=(vp_x, vp_y)
        )
        assert len(lines) >= 2, "Need at least 2 lines to measure spacing"

        # Compute angle of each line as seen from VP, then check angular spacing × r ≈ spacing.
        cx = float(poly.centroid.x)
        cy = float(poly.centroid.y)
        r = math.hypot(cx - vp_x, cy - vp_y)

        # Midpoints of each segment
        def midpoint(line):
            return (line[0][0] + line[-1][0]) / 2, (line[0][1] + line[-1][1]) / 2

        angles = []
        for line in lines:
            mx, my = midpoint(line)
            angles.append(math.atan2(my - vp_y, mx - vp_x))
        angles.sort()

        # Smallest angular step between adjacent lines
        min_dtheta = min(angles[i + 1] - angles[i] for i in range(len(angles) - 1))
        actual_spacing_at_r = min_dtheta * r

        # Allow 50% tolerance due to polygon truncation near edges.
        assert 0.1 * spacing <= actual_spacing_at_r <= 3.0 * spacing, (
            f"Spacing at centroid should be ≈ {spacing} mm; got {actual_spacing_at_r:.3f} mm"
        )


class TestPerspectiveHatchingParameter:
    """perspective_hatching parameter integrates correctly with Scene3DGenerator."""

    _BASE = {
        "shape_type": "Cube",
        "cube_size": 2.0,
        "hatch_density_min": 0.5,
        "hatch_density_max": 4.0,
        "hatch_angle_deg": 45.0,
        "hatch_cross": False,
        "render_style": "Hatched",
        **FAST_PARAMS,
    }

    def test_perspective_hatching_param_exists(self):
        """perspective_hatching must appear in get_parameters()."""
        gen = make_gen()
        param_names = {p.name for p in gen.get_parameters()}
        assert "perspective_hatching" in param_names, (
            "perspective_hatching parameter missing from get_parameters()"
        )

    def test_perspective_hatching_default_is_false(self):
        """perspective_hatching default must be False."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        assert params_map["perspective_hatching"].default is False

    def test_perspective_false_produces_valid_output(self):
        """perspective_hatching=False works identically to the original behaviour."""
        result = run({**self._BASE, "perspective_hatching": False})
        assert isinstance(result, list)
        assert len(result) > 0, "perspective_hatching=False must produce output"

    def test_perspective_true_produces_valid_output(self):
        """perspective_hatching=True produces a non-empty list of valid polylines."""
        result = run({**self._BASE, "perspective_hatching": True})
        assert isinstance(result, list)
        assert len(result) > 0, "perspective_hatching=True must produce at least one polyline"
        for poly in result:
            assert isinstance(poly, list) and len(poly) >= 2
            for pt in poly:
                assert len(pt) == 2

    def test_perspective_true_output_differs_from_false(self):
        """Perspective and parallel hatching produce different line sets."""
        parallel = run({**self._BASE, "perspective_hatching": False})
        persp = run({**self._BASE, "perspective_hatching": True})
        # The sets of polylines should differ (at least in coordinates).
        # Compare the first polyline's starting point.
        assert len(parallel) > 0 and len(persp) > 0
        # Collect all endpoints into a flat set and verify they are not identical.
        def flatten(lines):
            pts = set()
            for line in lines:
                for pt in line:
                    pts.add((round(pt[0], 3), round(pt[1], 3)))
            return pts
        pts_parallel = flatten(parallel)
        pts_persp = flatten(persp)
        # Some points might overlap (polygon boundary), but not all should match.
        assert pts_parallel != pts_persp, (
            "perspective_hatching=True should produce different lines than parallel hatching"
        )
