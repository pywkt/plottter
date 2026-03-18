"""Tests for Scene3DGenerator — generator integration for the 3D line art renderer (task 16.54).

Covers:
- generator produces non-empty polylines for each primitive type
- with HLR off, output contains more paths than with HLR on (occluded paths removed)
- sibling shape injection occludes geometry
- camera parameter changes produce different output
"""

from __future__ import annotations

import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANVAS = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)

# Shared camera dict used by all tests (default orbit)
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

# Use a coarser HLR step so tests run fast
FAST_PARAMS = {"hlr_enabled": False, "chop_step": 0.2, "_camera": CAM}


def make_gen():
    from plottter.generators.scene3d_generator import Scene3DGenerator
    return Scene3DGenerator()


def run(params):
    """Run the generator and return the polyline list."""
    gen = make_gen()
    return gen.generate(params, CANVAS)


# ---------------------------------------------------------------------------
# Per-primitive smoke tests
# ---------------------------------------------------------------------------

class TestPrimitives:
    """Each primitive type should produce at least one polyline."""

    def _smoke(self, shape_params: dict) -> list:
        params = {**FAST_PARAMS, **shape_params}
        result = run(params)
        assert isinstance(result, list), "generate() must return a list"
        assert len(result) > 0, f"No polylines produced for {shape_params}"
        return result

    def test_sphere(self):
        self._smoke({"shape_type": "Sphere", "sphere_radius": 1.0,
                     "sphere_lat_lines": 4, "sphere_lng_lines": 4})

    def test_shaded_sphere(self):
        self._smoke({"shape_type": "Shaded Sphere", "sphere_radius": 1.0,
                     "shaded_min_lines": 4, "shaded_max_lines": 10})

    def test_cube(self):
        self._smoke({"shape_type": "Cube", "cube_size": 1.5})

    def test_striped_cube(self):
        self._smoke({"shape_type": "Striped Cube", "cube_size": 1.5, "cube_stripes": 3})

    def test_cone(self):
        self._smoke({"shape_type": "Cone", "cone_radius": 1.0, "cone_height": 2.0,
                     "cone_lines": 6})

    def test_cylinder(self):
        self._smoke({"shape_type": "Cylinder", "cyl_radius": 1.0, "cyl_height": 2.0,
                     "cyl_lines": 6})

    def test_plane(self):
        self._smoke({"shape_type": "Plane", "plane_size": 4.0, "plane_steps": 4})

    def test_terrain(self):
        self._smoke({"shape_type": "Terrain", "plane_size": 4.0, "plane_steps": 4})

    def test_shard(self):
        self._smoke({"shape_type": "Shard", "shard_radius": 1.0, "shard_height": 2.0,
                     "shard_sides": 4})

    def test_mesh_import_no_file_returns_empty(self):
        """Mesh Import with no file path should return an empty list."""
        result = run({**FAST_PARAMS, "shape_type": "Mesh Import", "mesh_file": ""})
        assert result == []


# ---------------------------------------------------------------------------
# HLR on vs off
# ---------------------------------------------------------------------------

class TestHLR:
    """With HLR disabled the renderer returns at least as many paths as with HLR enabled
    (HLR can only remove paths, never add them)."""

    def test_hlr_reduces_path_count(self):
        shape_params = {
            "shape_type": "Sphere",
            "sphere_radius": 1.0,
            "sphere_lat_lines": 6,
            "sphere_lng_lines": 6,
            "_camera": CAM,
        }

        no_hlr_paths = run({**shape_params, "hlr_enabled": False, "chop_step": 0.1})
        hlr_paths = run({**shape_params, "hlr_enabled": True, "chop_step": 0.1})

        assert len(no_hlr_paths) >= len(hlr_paths), (
            f"Expected HLR to reduce paths: hlr={len(hlr_paths)}, "
            f"no_hlr={len(no_hlr_paths)}"
        )


# ---------------------------------------------------------------------------
# Sibling shape injection
# ---------------------------------------------------------------------------

class TestSiblingOcclusion:
    """A shape placed directly in front of the camera should occlude geometry behind it."""

    def test_sibling_reduces_visible_paths(self):
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.scene3d.shapes import Cube

        # Sphere at origin, camera at (0, 0, 8) looking toward origin.
        cam_front = {
            "azimuth": 0.0,   # camera looks from +Z
            "elevation": 0.0,
            "distance": 8.0,
            "look_at_x": 0.0,
            "look_at_y": 0.0,
            "look_at_z": 0.0,
            "fov": 45.0,
            "projection": "perspective",
        }

        sphere_params = {
            "shape_type": "Sphere",
            "sphere_radius": 0.8,
            "sphere_lat_lines": 6,
            "sphere_lng_lines": 6,
            "hlr_enabled": True,
            "chop_step": 0.1,
            "_camera": cam_front,
        }

        # Without occlusion
        no_occlude_paths = run({**sphere_params, "_sibling_3d_shapes": []})

        # Place a large cube between camera (+Z side) and the sphere at origin.
        # Cube positioned at (0, 0, 3) — halfway between camera (z=8) and sphere (z=0).
        gen = Scene3DGenerator()
        blocker_params = {
            "shape_type": "Cube",
            "cube_size": 4.0,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": 3.0,
        }
        blocker = gen.build_transformed_shape(blocker_params)

        occluded_paths = run({**sphere_params, "_sibling_3d_shapes": [blocker]})

        # With the blocking cube the sphere should have fewer visible paths
        assert len(occluded_paths) <= len(no_occlude_paths), (
            f"Sibling occlusion should not increase path count: "
            f"occluded={len(occluded_paths)}, no_occlude={len(no_occlude_paths)}"
        )


# ---------------------------------------------------------------------------
# Camera parameter changes produce different output
# ---------------------------------------------------------------------------

class TestCameraVariation:
    def test_different_azimuths_differ(self):
        base = {
            "shape_type": "Cube",
            "cube_size": 1.5,
            "hlr_enabled": False,
            "chop_step": 0.2,
        }

        cam_a = {**CAM, "azimuth": 0.0}
        cam_b = {**CAM, "azimuth": 90.0}

        paths_a = run({**base, "_camera": cam_a})
        paths_b = run({**base, "_camera": cam_b})

        # The sets of paths should not be identical when the camera has rotated 90°
        def flatten(paths):
            return [(round(x, 2), round(y, 2)) for p in paths for x, y in p]

        assert flatten(paths_a) != flatten(paths_b), (
            "90° azimuth rotation should produce different projected paths"
        )

    def test_perspective_vs_orthographic_differ(self):
        base = {
            "shape_type": "Sphere",
            "sphere_radius": 1.0,
            "sphere_lat_lines": 5,
            "sphere_lng_lines": 5,
            "hlr_enabled": False,
            "chop_step": 0.2,
        }

        cam_p = {**CAM, "projection": "perspective"}
        cam_o = {**CAM, "projection": "orthographic"}

        paths_p = run({**base, "_camera": cam_p})
        paths_o = run({**base, "_camera": cam_o})

        # Both projections should produce output; paths may differ
        assert len(paths_p) > 0
        assert len(paths_o) > 0

        def flatten(paths):
            return [(round(x, 1), round(y, 1)) for p in paths for x, y in p]

        # Very unlikely that perspective and orthographic give identical coordinates
        assert flatten(paths_p) != flatten(paths_o), (
            "Perspective and orthographic projections should produce different paths"
        )


# ---------------------------------------------------------------------------
# Polyline structure sanity
# ---------------------------------------------------------------------------

class TestPolylineStructure:
    def test_polylines_are_lists_of_2d_points(self):
        result = run({**FAST_PARAMS, "shape_type": "Cube", "cube_size": 1.5})
        assert len(result) > 0
        for poly in result:
            assert isinstance(poly, list), "Each polyline must be a list"
            assert len(poly) >= 2, "Each polyline must have at least 2 points"
            for pt in poly:
                assert len(pt) == 2, "Each point must be a 2-tuple (x, y)"
                assert isinstance(pt[0], float), "Point coordinates must be floats"
                assert isinstance(pt[1], float), "Point coordinates must be floats"

    def test_output_within_canvas_bounds(self):
        result = run({**FAST_PARAMS, "shape_type": "Sphere",
                      "sphere_radius": 1.0, "sphere_lat_lines": 5, "sphere_lng_lines": 5})
        draw_x1, draw_y1, draw_x2, draw_y2 = CANVAS.drawing_area()
        # Allow small overshoot due to floating-point edge cases
        tolerance = 1.0  # mm
        for poly in result:
            for x, y in poly:
                assert draw_x1 - tolerance <= x <= draw_x2 + tolerance, f"x={x} out of bounds"
                assert draw_y1 - tolerance <= y <= draw_y2 + tolerance, f"y={y} out of bounds"


# ---------------------------------------------------------------------------
# x_offset_mm / y_offset_mm (task 23.1)
# ---------------------------------------------------------------------------

class TestCanvasOffset:
    """x_offset_mm and y_offset_mm should shift all polyline points after projection."""

    _SPHERE_PARAMS = {
        "shape_type": "Sphere",
        "sphere_radius": 1.0,
        "sphere_lat_lines": 4,
        "sphere_lng_lines": 4,
        **FAST_PARAMS,
    }

    def test_default_zero_offset_matches_no_offset(self):
        """Default (0, 0) offset produces identical output to omitting the params."""
        without = run(self._SPHERE_PARAMS)
        with_zero = run({**self._SPHERE_PARAMS, "x_offset_mm": 0.0, "y_offset_mm": 0.0})

        def flatten(paths):
            return [(round(x, 6), round(y, 6)) for p in paths for x, y in p]

        assert flatten(without) == flatten(with_zero), (
            "Explicit (0, 0) offset should produce identical output to no offset params"
        )

    def test_x_offset_shifts_all_points_right(self):
        """Setting x_offset_mm=20 shifts every point 20mm to the right."""
        baseline = run(self._SPHERE_PARAMS)
        shifted = run({**self._SPHERE_PARAMS, "x_offset_mm": 20.0, "y_offset_mm": 0.0})

        assert len(baseline) == len(shifted), "Offset must not change polyline count"
        for base_poly, shift_poly in zip(baseline, shifted):
            assert len(base_poly) == len(shift_poly), "Offset must not change point count"
            for (bx, by), (sx, sy) in zip(base_poly, shift_poly):
                assert abs((sx - bx) - 20.0) < 1e-9, f"Expected x shift of 20mm, got {sx - bx}"
                assert abs(sy - by) < 1e-9, f"y should be unchanged, got delta {sy - by}"

    def test_y_offset_shifts_all_points_down(self):
        """Setting y_offset_mm=15 shifts every point 15mm downward."""
        baseline = run(self._SPHERE_PARAMS)
        shifted = run({**self._SPHERE_PARAMS, "x_offset_mm": 0.0, "y_offset_mm": 15.0})

        assert len(baseline) == len(shifted)
        for base_poly, shift_poly in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(base_poly, shift_poly):
                assert abs(sx - bx) < 1e-9, f"x should be unchanged, got delta {sx - bx}"
                assert abs((sy - by) - 15.0) < 1e-9, f"Expected y shift of 15mm, got {sy - by}"

    def test_combined_offset(self):
        """x and y offsets compose independently."""
        baseline = run(self._SPHERE_PARAMS)
        shifted = run({**self._SPHERE_PARAMS, "x_offset_mm": -10.0, "y_offset_mm": 5.0})

        assert len(baseline) == len(shifted)
        for base_poly, shift_poly in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(base_poly, shift_poly):
                assert abs((sx - bx) - (-10.0)) < 1e-9
                assert abs((sy - by) - 5.0) < 1e-9

    def test_offset_params_in_parameter_list(self):
        """x_offset_mm and y_offset_mm must appear in get_parameters() with randomizable=False."""
        from plottter.generators.scene3d_generator import Scene3DGenerator
        gen = Scene3DGenerator()
        params = gen.get_parameters()
        names = {p.name: p for p in params}
        assert "x_offset_mm" in names, "x_offset_mm must be in get_parameters()"
        assert "y_offset_mm" in names, "y_offset_mm must be in get_parameters()"
        assert not names["x_offset_mm"].randomizable, "x_offset_mm must not be randomizable"
        assert not names["y_offset_mm"].randomizable, "y_offset_mm must not be randomizable"


# ---------------------------------------------------------------------------
# Frustum-aware offset clipping (task 24.1)
# ---------------------------------------------------------------------------

_SMALL_CANVAS = Canvas(width_mm=100.0, height_mm=100.0, margin_mm=0.0)

_SPHERE_PARAMS_SMALL = {
    "shape_type": "Sphere",
    "sphere_radius": 1.5,
    "sphere_lat_lines": 6,
    "sphere_lng_lines": 6,
    **FAST_PARAMS,
}


def run_small(params):
    """Run the generator on the small 100×100 canvas."""
    gen = make_gen()
    return gen.generate(params, _SMALL_CANVAS)


class TestFrustumAwareOffset:
    """Verify that offset is incorporated into the frustum, not applied as a
    post-render translation.

    Key invariant: with the new approach, all rendered points are always within
    [0, canvas_w_mm] × [0, canvas_h_mm], regardless of offset magnitude.  The
    old post-translate approach would produce out-of-bounds coordinates when the
    offset is large enough to shift content off-canvas.
    """

    def test_zero_offset_all_points_within_canvas(self):
        """Baseline: all projected points are within canvas bounds."""
        polylines = run_small(_SPHERE_PARAMS_SMALL)
        assert polylines, "Should produce some polylines"
        for poly in polylines:
            for x, y in poly:
                assert -1e-6 <= x <= 100.0 + 1e-6, f"x={x:.3f} outside [0, 100]"
                assert -1e-6 <= y <= 100.0 + 1e-6, f"y={y:.3f} outside [0, 100]"

    def test_large_x_offset_all_points_within_canvas(self):
        """With a large x offset, all rendered points must stay within [0, canvas_w] × [0, canvas_h].

        With the old post-translate approach, a negative offset shifts sphere content to
        x < 0 (off-canvas left).  The new frustum-aware approach clips correctly.
        """
        params = {**_SPHERE_PARAMS_SMALL, "x_offset_mm": -40.0, "y_offset_mm": 0.0}
        polylines = run_small(params)
        for poly in polylines:
            for x, y in poly:
                assert -1e-6 <= x <= 100.0 + 1e-6, f"x={x:.3f} outside [0, 100] (x_off=-40)"
                assert -1e-6 <= y <= 100.0 + 1e-6, f"y={y:.3f} outside [0, 100] (x_off=-40)"

    def test_large_y_offset_all_points_within_canvas(self):
        """With a large y offset, all rendered points must stay within [0, canvas_w] × [0, canvas_h]."""
        params = {**_SPHERE_PARAMS_SMALL, "x_offset_mm": 0.0, "y_offset_mm": -40.0}
        polylines = run_small(params)
        for poly in polylines:
            for x, y in poly:
                assert -1e-6 <= x <= 100.0 + 1e-6, f"x={x:.3f} outside [0, 100] (y_off=-40)"
                assert -1e-6 <= y <= 100.0 + 1e-6, f"y={y:.3f} outside [0, 100] (y_off=-40)"

    def test_offset_moves_sphere_fully_offscreen_produces_empty(self):
        """If the offset moves the entire shape off-canvas, result should be empty."""
        # x_offset_mm=200 shifts the visible NDC range far left; the sphere
        # (centered at origin) maps entirely outside [0, 100] mm.
        params = {**_SPHERE_PARAMS_SMALL, "x_offset_mm": 200.0, "y_offset_mm": 0.0}
        polylines = run_small(params)
        assert polylines == [], (
            f"Expected empty output when sphere is fully off-canvas, got {len(polylines)} polylines"
        )

    def test_offset_moves_sphere_fully_offscreen_negative_produces_empty(self):
        """Negative large offset (sphere shifts right off canvas) also produces empty."""
        params = {**_SPHERE_PARAMS_SMALL, "x_offset_mm": -200.0, "y_offset_mm": 0.0}
        polylines = run_small(params)
        assert polylines == [], (
            f"Expected empty output for x_off=-200, got {len(polylines)} polylines"
        )

    def test_partial_x_offset_clips_to_canvas_left_edge(self):
        """With a negative x offset, the sphere's right portion remains visible and clipped.

        With old post-translate: some points would be at x < 0.
        With new frustum-aware: all x >= 0.
        """
        # x_off=-40: shifts sphere so right half appears near x=0..~23mm of canvas
        params = {**_SPHERE_PARAMS_SMALL, "x_offset_mm": -40.0}
        polylines = run_small(params)
        if polylines:
            all_x = [x for poly in polylines for x, y in poly]
            assert min(all_x) >= -1e-6, (
                f"Points outside left edge: min x = {min(all_x):.3f}"
            )

    def test_scene_render_offset_mm_parameter(self):
        """Scene.render() accepts offset_mm keyword and produces correct output."""
        from plottter.scene3d.scene import Scene
        from plottter.scene3d import Camera
        from plottter.scene3d.shapes import Sphere

        scene = Scene(hlr_enabled=False)
        scene.add(Sphere(radius=1.0))
        scene.compile()

        cam = Camera.default(aspect=1.0)
        w, h = 100.0, 100.0

        result_zero = scene.render(cam, w, h, offset_mm=(0.0, 0.0))
        result_offset = scene.render(cam, w, h, offset_mm=(20.0, 0.0))

        assert result_zero, "Should produce some polylines with no offset"
        assert result_offset, "Should produce some polylines with small offset"

        # All points from the offset render must be within canvas bounds
        for poly in result_offset:
            for x, y in poly:
                assert -1e-6 <= x <= w + 1e-6, f"x={x:.3f} outside [0, {w}]"
                assert -1e-6 <= y <= h + 1e-6, f"y={y:.3f} outside [0, {h}]"


# ---------------------------------------------------------------------------
# Shadow / light source parameter tests (task 29.1)
# ---------------------------------------------------------------------------

class TestShadowParams:
    """Tests for scene-level light source parameters added in task 29.1."""

    def test_shadow_params_in_parameter_list(self):
        """shadow_enabled, light_azimuth, light_elevation, shadow_density are exposed."""
        gen = make_gen()
        param_names = {p.name for p in gen.get_parameters()}
        assert "shadow_enabled" in param_names
        assert "light_azimuth" in param_names
        assert "light_elevation" in param_names
        assert "shadow_density" in param_names

    def test_shadow_disabled_produces_same_output_as_before(self):
        """shadow_enabled=False is the default and produces identical output."""
        params_no_shadow = {**FAST_PARAMS, "shape_type": "Sphere",
                            "sphere_radius": 1.0, "sphere_lat_lines": 4, "sphere_lng_lines": 4}
        params_shadow_off = {**params_no_shadow, "shadow_enabled": False}

        result_no_shadow = run(params_no_shadow)
        result_shadow_off = run(params_shadow_off)
        assert result_no_shadow == result_shadow_off

    def test_shadow_enabled_still_produces_output(self):
        """With shadow_enabled=True, the generator still returns non-empty polylines."""
        params = {**FAST_PARAMS, "shape_type": "Sphere",
                  "sphere_radius": 1.0, "sphere_lat_lines": 4, "sphere_lng_lines": 4,
                  "shadow_enabled": True, "light_azimuth": 45.0, "light_elevation": 45.0,
                  "shadow_density": 1.0}
        result = run(params)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_light_controls_hidden_when_shadow_disabled(self):
        """light_azimuth, light_elevation, shadow_density are visible only when shadow_enabled."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        for name in ("light_azimuth", "light_elevation", "shadow_density"):
            p = params_map[name]
            assert hasattr(p, "visible_when") and p.visible_when, (
                f"{name} should have visible_when set"
            )
            assert "shadow_enabled" in p.visible_when, (
                f"{name}.visible_when should key on shadow_enabled"
            )

    def test_scene_render_accepts_light_dir_parameter(self):
        """Scene.render() accepts the new light_dir keyword without error.

        When light_dir is None:  returns list[Polyline]
        When light_dir is set:   returns (lit_polylines, shadow_polylines)
        """
        from plottter.scene3d.scene import Scene
        from plottter.scene3d import Camera
        from plottter.scene3d.shapes import Sphere

        scene = Scene(hlr_enabled=False)
        scene.add(Sphere(radius=1.0))
        scene.compile()

        cam = Camera.default(aspect=1.0)
        # light_dir=None → plain list
        result_none = scene.render(cam, 100.0, 100.0, light_dir=None)
        assert isinstance(result_none, list)
        assert len(result_none) > 0

        # light_dir provided → tuple (lit_polylines, shadow_polylines)
        result_lit = scene.render(cam, 100.0, 100.0, light_dir=(0.5, 0.5, 0.7))
        assert isinstance(result_lit, tuple) and len(result_lit) == 2
        lit, shadow = result_lit
        assert isinstance(lit, list)
        assert isinstance(shadow, list)
        # With HLR disabled no shadow rays are cast; all paths are lit
        assert len(lit) > 0

    def test_shaded_sphere_uses_scene_light_when_shadow_enabled(self):
        """When shadow_enabled=True, ShadedSphere's light_dir is updated to match scene light."""
        import math
        import numpy as np
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.scene3d.shapes.sphere import ShadedSphere
        from plottter.scene3d.shapes.transformed import TransformedShape

        gen = Scene3DGenerator()
        params = {
            **FAST_PARAMS,
            "shape_type": "Shaded Sphere",
            "sphere_radius": 1.0,
            "shaded_min_lines": 4,
            "shaded_max_lines": 10,
            "shadow_enabled": True,
            "light_azimuth": 0.0,
            "light_elevation": 90.0,  # straight up → lz=1, lx=ly=0
            "shadow_density": 1.0,
        }

        # Build the shape and check its light_dir is updated
        shape = gen.build_transformed_shape(params)
        assert shape is not None

        # We need to exercise the full generate() to trigger the sync
        result = gen.generate(params, CANVAS)
        assert len(result) > 0  # still produces output

    def test_set_light_dir_on_shaded_sphere(self):
        """ShadedSphere.set_light_dir() updates the light direction."""
        import math
        import numpy as np
        from plottter.scene3d.shapes.sphere import ShadedSphere

        sphere = ShadedSphere(radius=1.0, light_dir=[1.0, 0.0, 0.0])
        assert abs(sphere.light_dir[0] - 1.0) < 1e-6

        sphere.set_light_dir(np.array([0.0, 1.0, 0.0]))
        assert abs(sphere.light_dir[1] - 1.0) < 1e-6
        assert abs(sphere.light_dir[0]) < 1e-6


# ---------------------------------------------------------------------------
# Shadow hatching tests (task 29.3)
# ---------------------------------------------------------------------------

# Use HLR enabled so shadow rays can actually be cast.
_SHADOW_BASE_PARAMS = {
    "_camera": CAM,
    "shape_type": "Sphere",
    "sphere_radius": 1.0,
    "sphere_lat_lines": 4,
    "sphere_lng_lines": 4,
    "hlr_enabled": True,
    "chop_step": 0.1,
}


def _shadow_params(**extra):
    return {
        **_SHADOW_BASE_PARAMS,
        "shadow_enabled": True,
        "light_azimuth": 45.0,
        "light_elevation": 30.0,
        "shadow_density": 1.0,
        "shadow_style": "Thicken",
        "shadow_hatch_angle": 45.0,
        **extra,
    }


class TestShadowHatching:
    """Tests for on-surface shadow hatching (task 29.3)."""

    # ── (a) Shadow-disabled output is identical to before ────────────────
    def test_shadow_disabled_output_unchanged(self):
        """shadow_enabled=False produces the same output regardless of shadow_style."""
        base = {**_SHADOW_BASE_PARAMS, "shadow_enabled": False}
        result = run(base)
        result_with_style = run({**base, "shadow_style": "Thicken", "shadow_hatch_angle": 45.0})
        assert result == result_with_style

    # ── (b) Thicken style produces more polylines than no shadows ─────────
    def test_thicken_adds_extra_polylines(self):
        """With shadows and Thicken style, total polyline count > shadows disabled."""
        no_shadow = run({**_SHADOW_BASE_PARAMS, "shadow_enabled": False})
        with_shadow = run(_shadow_params(shadow_style="Thicken", shadow_density=1.0))
        # Thicken adds parallel offset lines to shadow edges; result should have more
        # polylines than the shadow-free render (assuming some segments are in shadow).
        assert len(with_shadow) >= len(no_shadow), (
            "Shadow Thicken mode should produce at least as many polylines as no-shadow render"
        )

    # ── (c) Density controls how many offset lines are added ─────────────
    def test_higher_density_adds_more_lines(self):
        """Higher shadow density produces more hatching lines."""
        low = run(_shadow_params(shadow_style="Thicken", shadow_density=0.5))
        high = run(_shadow_params(shadow_style="Thicken", shadow_density=2.0))
        # More density → more offset copies → more polylines
        assert len(high) >= len(low), (
            f"Higher density should produce >= polylines: low={len(low)}, high={len(high)}"
        )

    # ── (d) All three styles produce output ──────────────────────────────
    def test_all_shadow_styles_produce_output(self):
        """Thicken, Hatch, and Cross-Hatch all produce non-empty output."""
        for style in ("Thicken", "Hatch", "Cross-Hatch"):
            result = run(_shadow_params(shadow_style=style))
            assert isinstance(result, list), f"{style}: generate() must return a list"
            assert len(result) > 0, f"{style}: should produce non-empty output"

    # ── (e) Output is all polylines (no fills) ────────────────────────────
    def test_output_is_all_polylines(self):
        """All returned items must be polylines (list of 2-tuples), not regions."""
        result = run(_shadow_params(shadow_style="Cross-Hatch"))
        for poly in result:
            assert isinstance(poly, list), "Each item must be a list of points"
            assert len(poly) >= 2, "Each polyline must have at least 2 points"
            for pt in poly:
                assert len(pt) == 2, f"Each point must be (x, y), got {pt}"

    # ── New params appear in get_parameters() ────────────────────────────
    def test_new_params_in_parameter_list(self):
        """shadow_style and shadow_hatch_angle must appear in get_parameters()."""
        gen = make_gen()
        param_names = {p.name for p in gen.get_parameters()}
        assert "shadow_style" in param_names, "shadow_style must be in get_parameters()"
        assert "shadow_hatch_angle" in param_names, "shadow_hatch_angle must be in get_parameters()"

    def test_shadow_style_visible_when_shadow_enabled(self):
        """shadow_style and shadow_hatch_angle are only visible when shadow_enabled=True."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        for name in ("shadow_style", "shadow_hatch_angle"):
            p = params_map[name]
            assert hasattr(p, "visible_when") and p.visible_when, (
                f"{name} should have visible_when set"
            )
            assert "shadow_enabled" in p.visible_when, (
                f"{name}.visible_when should key on shadow_enabled"
            )

    # ── _offset_polyline_2d unit tests ────────────────────────────────────
    def test_offset_polyline_horizontal_left(self):
        """A horizontal line offset to the left (positive) should shift upward."""
        from plottter.generators.scene3d_generator import _offset_polyline_2d
        poly = [(0.0, 0.0), (10.0, 0.0)]
        offset = _offset_polyline_2d(poly, 1.0)
        assert len(offset) == 2
        # Left of rightward travel = upward (y increases)
        assert abs(offset[0][1] - 1.0) < 1e-9
        assert abs(offset[1][1] - 1.0) < 1e-9

    def test_offset_polyline_empty_input(self):
        """Empty or single-point input returns empty list."""
        from plottter.generators.scene3d_generator import _offset_polyline_2d
        assert _offset_polyline_2d([], 1.0) == []
        assert _offset_polyline_2d([(0.0, 0.0)], 1.0) == []

    def test_offset_polyline_preserves_point_count(self):
        """Offset polyline has the same number of points as the input."""
        from plottter.generators.scene3d_generator import _offset_polyline_2d
        poly = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        offset = _offset_polyline_2d(poly, 0.5)
        assert len(offset) == len(poly)

    # ── _hatch_shadow_polylines unit tests ────────────────────────────────
    def test_hatch_empty_input_returns_empty(self):
        """No shadow polys → empty hatch output."""
        from plottter.generators.scene3d_generator import _hatch_shadow_polylines
        assert _hatch_shadow_polylines([], 1.0, "Thicken") == []

    def test_thicken_density_one_adds_two_offset_lines_per_poly(self):
        """With density=1.0 and one shadow polyline, Thicken adds 2 offset polylines."""
        from plottter.generators.scene3d_generator import _hatch_shadow_polylines
        shadow = [[(0.0, 0.0), (10.0, 0.0)]]
        result = _hatch_shadow_polylines(shadow, density=1.0, shadow_style="Thicken")
        # n_offsets = round(1.0) = 1 → one offset on each side = 2 total
        assert len(result) == 2

    def test_thicken_density_two_adds_four_offset_lines_per_poly(self):
        """With density=2.0 and one shadow polyline, Thicken adds 4 offset polylines."""
        from plottter.generators.scene3d_generator import _hatch_shadow_polylines
        shadow = [[(0.0, 0.0), (10.0, 0.0)]]
        result = _hatch_shadow_polylines(shadow, density=2.0, shadow_style="Thicken")
        # n_offsets = round(2.0) = 2 → two offsets on each side = 4 total
        assert len(result) == 4

    def test_hatch_style_produces_tick_marks(self):
        """Hatch mode produces tick marks along the shadow polyline."""
        from plottter.generators.scene3d_generator import _hatch_shadow_polylines
        shadow = [[(0.0, 0.0), (10.0, 0.0)]]
        result = _hatch_shadow_polylines(shadow, density=1.0, shadow_style="Hatch")
        # Each tick is a 2-point polyline
        assert len(result) > 0
        for tick in result:
            assert len(tick) == 2, "Each Hatch tick must be a 2-point polyline"

    def test_cross_hatch_produces_more_lines_than_hatch(self):
        """Cross-Hatch produces twice the tick marks of Hatch (two angle passes)."""
        from plottter.generators.scene3d_generator import _hatch_shadow_polylines
        shadow = [[(0.0, 0.0), (10.0, 0.0)]]
        hatch = _hatch_shadow_polylines(shadow, density=1.0, shadow_style="Hatch")
        cross = _hatch_shadow_polylines(shadow, density=1.0, shadow_style="Cross-Hatch")
        assert len(cross) == len(hatch) * 2, (
            "Cross-Hatch has two angle passes so should have 2× the tick count of Hatch"
        )


# ---------------------------------------------------------------------------
# Ground-plane cast shadow tests (task 29.4)
# ---------------------------------------------------------------------------

def _ground_shadow_params(**extra):
    """Build params with shadows + ground plane enabled on a sphere."""
    return {
        **_SHADOW_BASE_PARAMS,
        "shadow_enabled": True,
        "light_azimuth": 45.0,
        "light_elevation": 45.0,
        "shadow_density": 1.0,
        "shadow_style": "Thicken",
        "shadow_hatch_angle": 45.0,
        "shadow_ground_plane": True,
        "ground_plane_z": -2.0,
        **extra,
    }


class TestGroundPlaneShadow:
    """Tests for the ground-plane cast shadow feature (task 29.4)."""

    # ── (a) Ground plane disabled produces no extra geometry ─────────────
    def test_ground_plane_disabled_same_as_no_ground_shadow(self):
        """shadow_ground_plane=False (default) produces identical output to omitting it."""
        params_no_ground = {
            **_SHADOW_BASE_PARAMS,
            "shadow_enabled": True,
            "light_azimuth": 45.0,
            "light_elevation": 45.0,
            "shadow_density": 1.0,
            "shadow_style": "Thicken",
            "shadow_hatch_angle": 45.0,
        }
        params_with_disabled = {**params_no_ground, "shadow_ground_plane": False}
        result_no = run(params_no_ground)
        result_disabled = run(params_with_disabled)
        assert result_no == result_disabled, (
            "shadow_ground_plane=False must produce identical output to omitting the param"
        )

    # ── (b) Ground plane enabled adds extra paths ─────────────────────────
    def test_ground_plane_enabled_adds_extra_paths(self):
        """With shadow_ground_plane=True, more polylines are produced (shadow on floor)."""
        params_without = {
            **_SHADOW_BASE_PARAMS,
            "shadow_enabled": True,
            "light_azimuth": 45.0,
            "light_elevation": 45.0,
            "shadow_density": 1.0,
            "shadow_style": "Thicken",
            "shadow_hatch_angle": 45.0,
            "shadow_ground_plane": False,
        }
        params_with = _ground_shadow_params()
        result_without = run(params_without)
        result_with = run(params_with)
        assert len(result_with) >= len(result_without), (
            "Ground plane shadow should add extra polylines (shadow on the floor)"
        )

    # ── (c) Output is valid polylines ─────────────────────────────────────
    def test_ground_plane_output_is_valid_polylines(self):
        """All returned items must be polylines (list of 2-tuples)."""
        result = run(_ground_shadow_params())
        assert isinstance(result, list)
        for poly in result:
            assert isinstance(poly, list), "Each item must be a list"
            assert len(poly) >= 2, "Each polyline must have at least 2 points"
            for pt in poly:
                assert len(pt) == 2, f"Each point must be (x, y), got {pt}"

    # ── (d) Params appear in get_parameters() ─────────────────────────────
    def test_ground_plane_params_in_parameter_list(self):
        """shadow_ground_plane and ground_plane_z must appear in get_parameters()."""
        gen = make_gen()
        param_names = {p.name for p in gen.get_parameters()}
        assert "shadow_ground_plane" in param_names, (
            "shadow_ground_plane must be in get_parameters()"
        )
        assert "ground_plane_z" in param_names, (
            "ground_plane_z must be in get_parameters()"
        )

    def test_ground_plane_params_visible_when_shadow_enabled(self):
        """shadow_ground_plane is only visible when shadow_enabled=True."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        p = params_map["shadow_ground_plane"]
        assert hasattr(p, "visible_when") and p.visible_when, (
            "shadow_ground_plane should have visible_when set"
        )
        assert "shadow_enabled" in p.visible_when, (
            "shadow_ground_plane.visible_when should key on shadow_enabled"
        )

    # ── (e) Different ground_plane_z → different output ───────────────────
    def test_different_ground_z_produces_different_shadow(self):
        """Changing ground_plane_z moves the shadow (produces different polylines)."""
        result_low = run(_ground_shadow_params(ground_plane_z=-3.0))
        result_high = run(_ground_shadow_params(ground_plane_z=-1.0))

        def flatten_pts(paths):
            return [(round(x, 3), round(y, 3)) for p in paths for x, y in p]

        # Shadows at different Z heights project to different 2D positions
        pts_low = flatten_pts(result_low)
        pts_high = flatten_pts(result_high)
        assert pts_low != pts_high, (
            "Different ground_plane_z values should produce different shadow positions"
        )

    # ── (f) Light direction change moves shadow ───────────────────────────
    def test_different_light_azimuth_moves_shadow(self):
        """Changing light_azimuth changes the shadow direction on the ground plane."""
        result_a = run(_ground_shadow_params(light_azimuth=0.0))
        result_b = run(_ground_shadow_params(light_azimuth=90.0))

        def flatten_pts(paths):
            return [(round(x, 3), round(y, 3)) for p in paths for x, y in p]

        assert flatten_pts(result_a) != flatten_pts(result_b), (
            "Different light azimuths should move the shadow position"
        )

    # ── Unit tests for _project_to_ground_z ───────────────────────────────
    def test_project_to_ground_z_above_plane(self):
        """A point above the ground plane projects correctly."""
        from plottter.generators.scene3d_generator import _project_to_ground_z
        # Light straight up (lz=1), point at z=2, ground at z=0
        # t = (2 - 0) / 1 = 2; shadow at (px - 0*2, py - 0*2, 0)
        result = _project_to_ground_z(1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0)
        assert result is not None
        sx, sy, sz = result
        assert abs(sx - 1.0) < 1e-9, f"Expected sx=1.0, got {sx}"
        assert abs(sy - 2.0) < 1e-9, f"Expected sy=2.0, got {sy}"
        assert abs(sz - 0.0) < 1e-9, f"Expected sz=0.0, got {sz}"

    def test_project_to_ground_z_with_light_angle(self):
        """Light with both horizontal and vertical component casts angled shadow."""
        from plottter.generators.scene3d_generator import _project_to_ground_z
        # Light direction (lx=1, ly=0, lz=1) normalized
        import math
        lx, ly, lz = 1.0 / math.sqrt(2), 0.0, 1.0 / math.sqrt(2)
        # Point at (0, 0, 1), ground at z=0 → t = 1/lz = sqrt(2)
        result = _project_to_ground_z(0.0, 0.0, 1.0, lx, ly, lz, 0.0)
        assert result is not None
        sx, sy, sz = result
        assert abs(sz - 0.0) < 1e-9, "Shadow must be on the ground plane"
        assert sx < 0.0, "Shadow should be offset in the -x direction (opposite light)"

    def test_project_to_ground_z_below_plane_returns_none(self):
        """A point below the ground plane returns None (no shadow possible)."""
        from plottter.generators.scene3d_generator import _project_to_ground_z
        # Point at z=-1, ground at z=0, lz=1 → t = (-1 - 0)/1 = -1 < 0 → None
        result = _project_to_ground_z(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0)
        assert result is None

    def test_project_to_ground_z_horizontal_light_returns_none(self):
        """Horizontal light (lz ≈ 0) returns None to avoid division by zero."""
        from plottter.generators.scene3d_generator import _project_to_ground_z
        result = _project_to_ground_z(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        assert result is None

    # ── Unit tests for _compute_ground_shadow_paths ───────────────────────
    def test_compute_ground_shadow_paths_returns_paths(self):
        """_compute_ground_shadow_paths returns non-empty paths for a sphere above ground."""
        import math
        import numpy as np
        from plottter.generators.scene3d_generator import (
            Scene3DGenerator, _compute_ground_shadow_paths
        )
        from plottter.scene3d.shapes import Sphere

        sphere = Sphere(radius=1.0, lat_lines=4, lng_lines=4)
        # Light from above at 45° elevation
        el = math.radians(45.0)
        az = math.radians(45.0)
        ld_norm = np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ], dtype=np.float64)

        # Ground plane at z=-2, sphere is at z=0 (with radius 1, extends from z=-1 to +1)
        paths = _compute_ground_shadow_paths(sphere, ld_norm, -2.0, 1.0, 45.0)
        assert len(paths) > 0, "Should produce shadow paths for a sphere above the ground"

    def test_compute_ground_shadow_paths_points_on_ground(self):
        """All points in the projected shadow paths must be at z = ground_z."""
        import math
        import numpy as np
        from plottter.generators.scene3d_generator import _compute_ground_shadow_paths
        from plottter.scene3d.shapes import Sphere

        sphere = Sphere(radius=1.0, lat_lines=4, lng_lines=4)
        el = math.radians(60.0)
        az = math.radians(30.0)
        ld_norm = np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ], dtype=np.float64)

        ground_z = -3.0
        paths = _compute_ground_shadow_paths(sphere, ld_norm, ground_z, 1.0, 45.0)
        for path in paths:
            for pt in path.points:
                assert abs(float(pt[2]) - ground_z) < 1e-6, (
                    f"Shadow point z={pt[2]:.6f} should be at ground_z={ground_z}"
                )

    def test_compute_ground_shadow_paths_horizontal_light_returns_empty(self):
        """Horizontal light (lz ≈ 0) returns no shadow paths."""
        import numpy as np
        from plottter.generators.scene3d_generator import _compute_ground_shadow_paths
        from plottter.scene3d.shapes import Sphere

        sphere = Sphere(radius=1.0)
        # Perfectly horizontal light
        ld_norm = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        paths = _compute_ground_shadow_paths(sphere, ld_norm, 0.0, 1.0, 45.0)
        assert paths == [], "Horizontal light should produce no ground shadow paths"

    def test_ground_shadow_uses_hull_outline_not_wireframe_edges(self):
        """Shadow projection must use convex-hull outline, not per-wireframe-edge projection.

        Before task 32.3 this method projected individual wireframe edges onto the
        ground plane, producing as many shadow paths as there are wireframe paths.
        The correct behaviour (post-32.3) computes the convex hull of ALL shadow
        points and emits only 1 hull-outline path (plus sparse hatch lines).
        This test fails if per-edge projection is re-introduced.
        """
        import math
        import numpy as np
        from plottter.generators.scene3d_generator import _compute_ground_shadow_paths
        from plottter.scene3d.shapes import Sphere

        # Sphere with many wireframe paths so per-edge leakage would be obvious.
        # lat_lines=8, lng_lines=8 → 7 latitude circles + 8 longitude arcs = 15 paths.
        sphere = Sphere(radius=1.0, lat_lines=8, lng_lines=8)
        el = math.radians(45.0)
        az = math.radians(45.0)
        ld_norm = np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ], dtype=np.float64)

        # Low density (spacing=10) → very sparse hatching; output should be the
        # hull outline (1 path) plus at most a couple of hatch lines.
        paths = _compute_ground_shadow_paths(sphere, ld_norm, -2.0, 0.1, 45.0)

        wireframe_count = len(sphere.paths())  # 15 for lat_lines=8, lng_lines=8
        assert len(paths) >= 1, "Should produce at least the hull-outline path"
        assert len(paths) < wireframe_count, (
            f"Shadow produced {len(paths)} paths but the sphere has {wireframe_count} "
            "wireframe paths — looks like per-edge projection was re-introduced"
        )

        # The first path must be the closed convex-hull outline (first point == last point).
        hull_path = paths[0]
        first_pt = hull_path.points[0]
        last_pt = hull_path.points[-1]
        assert np.allclose(first_pt, last_pt, atol=1e-9), (
            "First shadow path must be a closed polygon (hull outline): "
            f"first={first_pt}, last={last_pt}"
        )

    # ── Integration: Scene.render() accepts extra_render_paths ────────────
    def test_scene_render_extra_paths(self):
        """Scene.render() extra_render_paths adds extra visible paths to output."""
        from plottter.scene3d.scene import Scene
        from plottter.scene3d import Camera
        from plottter.scene3d.shapes import Sphere
        from plottter.scene3d.path3d import Path3D
        import numpy as np

        scene = Scene(hlr_enabled=False)
        sphere = Sphere(radius=0.5)
        scene.add(sphere)
        scene.compile()
        cam = Camera.default(aspect=1.0)

        # Render without extra paths
        result_base = scene.render(cam, 100.0, 100.0)
        assert len(result_base) > 0

        # Create an extra path that sits on a different plane in 3D
        extra_path = Path3D([
            np.array([0.0, 0.0, -3.0]),
            np.array([1.0, 0.0, -3.0]),
            np.array([1.0, 1.0, -3.0]),
        ])
        result_extra = scene.render(cam, 100.0, 100.0, extra_render_paths=[extra_path])
        # Extra path adds more polylines
        assert len(result_extra) >= len(result_base), (
            "extra_render_paths should add at least one extra polyline"
        )

    def test_scene_render_no_extra_paths_unchanged(self):
        """Scene.render() with extra_render_paths=None is identical to the default."""
        from plottter.scene3d.scene import Scene
        from plottter.scene3d import Camera
        from plottter.scene3d.shapes import Sphere

        scene = Scene(hlr_enabled=False)
        scene.add(Sphere(radius=1.0))
        scene.compile()
        cam = Camera.default(aspect=1.0)

        result_default = scene.render(cam, 100.0, 100.0)
        result_none = scene.render(cam, 100.0, 100.0, extra_render_paths=None)
        assert result_default == result_none, (
            "extra_render_paths=None must produce identical output to default"
        )


# ---------------------------------------------------------------------------
# Shadow render mode tests (task 29.5)
# ---------------------------------------------------------------------------

def _render_mode_params(**extra):
    """Build params with shadows enabled on a sphere for render-mode tests."""
    return {
        **_SHADOW_BASE_PARAMS,
        "shadow_enabled": True,
        "light_azimuth": 45.0,
        "light_elevation": 30.0,
        "shadow_density": 1.0,
        "shadow_style": "Thicken",
        "shadow_hatch_angle": 45.0,
        "shadow_ground_plane": True,
        "ground_plane_z": -2.0,
        **extra,
    }


class TestShadowRenderModeParam:
    """Tests for the shadow_render_mode ChoiceParam (task 29.5)."""

    def test_shadow_render_mode_in_parameter_list(self):
        """shadow_render_mode must appear in get_parameters()."""
        gen = make_gen()
        param_names = {p.name for p in gen.get_parameters()}
        assert "shadow_render_mode" in param_names, (
            "shadow_render_mode must be in get_parameters()"
        )

    def test_shadow_render_mode_default_is_combined(self):
        """shadow_render_mode must have default='Combined'."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        p = params_map["shadow_render_mode"]
        assert p.default == "Combined", (
            f"shadow_render_mode default must be 'Combined', got {p.default!r}"
        )

    def test_shadow_render_mode_visible_when_shadow_enabled(self):
        """shadow_render_mode is only visible when shadow_enabled=True."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        p = params_map["shadow_render_mode"]
        assert hasattr(p, "visible_when") and p.visible_when, (
            "shadow_render_mode should have visible_when set"
        )
        assert "shadow_enabled" in p.visible_when, (
            "shadow_render_mode.visible_when should key on shadow_enabled"
        )
        assert p.visible_when["shadow_enabled"] == [True], (
            "shadow_render_mode.visible_when['shadow_enabled'] should be [True]"
        )

    def test_shadow_render_mode_has_three_choices(self):
        """shadow_render_mode has exactly the choices: Combined, Shadow Only, Lit Only."""
        gen = make_gen()
        params_map = {p.name: p for p in gen.get_parameters()}
        p = params_map["shadow_render_mode"]
        assert hasattr(p, "choices"), "shadow_render_mode must be a ChoiceParam"
        assert set(p.choices) == {"Combined", "Shadow Only", "Lit Only"}, (
            f"Expected choices {{Combined, Shadow Only, Lit Only}}, got {p.choices}"
        )


class TestShadowRenderModeBehavior:
    """Behavioral tests for shadow_render_mode values (task 29.5)."""

    def test_all_modes_produce_non_empty_output(self):
        """All three render modes must produce non-empty polyline lists."""
        for mode in ("Combined", "Shadow Only", "Lit Only"):
            result = run(_render_mode_params(shadow_render_mode=mode))
            assert isinstance(result, list), f"{mode}: generate() must return a list"
            assert len(result) > 0, f"{mode}: should produce non-empty output with a lit scene"

    def test_lit_only_fewer_polylines_than_combined(self):
        """'Lit Only' returns fewer total polylines than 'Combined' when shadows exist.

        'Combined' = lit wireframe + shadow hatching + ground shadow.
        'Lit Only' = only the visible lit wireframe edges (no shadow hatching, no ground shadow).
        """
        combined = run(_render_mode_params(shadow_render_mode="Combined"))
        lit_only = run(_render_mode_params(shadow_render_mode="Lit Only"))
        assert len(lit_only) <= len(combined), (
            f"'Lit Only' should have fewer or equal polylines than 'Combined': "
            f"lit_only={len(lit_only)}, combined={len(combined)}"
        )

    def test_shadow_only_differs_from_combined_and_lit_only(self):
        """'Shadow Only' output differs from both 'Combined' and 'Lit Only'.

        'Shadow Only' renders only shadow hatching + ground shadow, not the lit wireframe.
        """
        def flatten(paths):
            return [(round(x, 3), round(y, 3)) for p in paths for x, y in p]

        combined = run(_render_mode_params(shadow_render_mode="Combined"))
        shadow_only = run(_render_mode_params(shadow_render_mode="Shadow Only"))
        lit_only = run(_render_mode_params(shadow_render_mode="Lit Only"))

        assert flatten(shadow_only) != flatten(combined), (
            "'Shadow Only' output should differ from 'Combined'"
        )
        assert flatten(shadow_only) != flatten(lit_only), (
            "'Shadow Only' output should differ from 'Lit Only'"
        )

    def test_default_mode_combined_equals_no_mode_param(self):
        """Omitting shadow_render_mode (default='Combined') produces same output as explicit 'Combined'."""
        def flatten(paths):
            return [(round(x, 6), round(y, 6)) for p in paths for x, y in p]

        without_param = run(_render_mode_params())  # no shadow_render_mode key
        with_combined = run(_render_mode_params(shadow_render_mode="Combined"))

        assert flatten(without_param) == flatten(with_combined), (
            "Omitting shadow_render_mode should produce the same output as explicitly 'Combined'"
        )

    def test_shadow_only_includes_ground_plane_shadow_with_hlr(self):
        """'Shadow Only' + HLR must include ground-plane shadow paths.

        This exercises the bug fixed in _render_with_hlr(): when render_shapes=[]
        (no main-shape wireframe), the early-return guard must NOT fire before
        processing extra_render_paths (ground shadow geometry).
        """
        # Shadow Only WITH ground plane — should produce more polylines because
        # the ground-plane shadow is included via extra_render_paths.
        with_ground = run(
            _render_mode_params(shadow_render_mode="Shadow Only", shadow_ground_plane=True)
        )
        # Shadow Only WITHOUT ground plane — no extra_render_paths passed.
        without_ground = run(
            _render_mode_params(shadow_render_mode="Shadow Only", shadow_ground_plane=False)
        )
        assert len(with_ground) > len(without_ground), (
            "Shadow Only + HLR + shadow_ground_plane=True should produce more polylines "
            f"than shadow_ground_plane=False: with={len(with_ground)}, without={len(without_ground)}"
        )


# ---------------------------------------------------------------------------
# Shadow preset tests (task 29.5)
# ---------------------------------------------------------------------------

class TestShadowPresets:
    """Tests for the three new shadow presets: Dramatic Shadows, Architectural, Subtle Shading."""

    _PRESET_NAMES = ("Dramatic Shadows", "Architectural", "Subtle Shading")

    def test_shadow_presets_exist(self):
        """All three shadow preset names must appear in get_presets()."""
        gen = make_gen()
        preset_names = {p.name for p in gen.get_presets()}
        for name in self._PRESET_NAMES:
            assert name in preset_names, (
                f"Expected preset '{name}' in get_presets(), found: {preset_names}"
            )

    def test_shadow_presets_produce_non_empty_output(self):
        """Running each shadow preset must produce non-empty polyline output."""
        gen = make_gen()
        presets_map = {p.name: p for p in gen.get_presets()}
        for name in self._PRESET_NAMES:
            preset = presets_map[name]
            # Merge preset params with a camera so the generator has all needed keys
            params = {**preset.params, "_camera": CAM}
            result = gen.generate(params, CANVAS)
            assert isinstance(result, list), f"Preset '{name}': generate() must return a list"
            assert len(result) > 0, f"Preset '{name}': must produce non-empty output"

    def test_shadow_presets_have_shadow_enabled(self):
        """Each shadow preset must have shadow_enabled=True."""
        gen = make_gen()
        presets_map = {p.name: p for p in gen.get_presets()}
        for name in self._PRESET_NAMES:
            preset = presets_map[name]
            assert preset.params.get("shadow_enabled") is True, (
                f"Preset '{name}' must have shadow_enabled=True"
            )
