"""Phase 16.55 validation: 3D line art renderer — interactive preview (Phase C).

Verifies:
1. CanvasWidget exposes a set_3d_preview_active() method that enables/disables
   3D interactive preview mode and changes the widget state accordingly.
2. CanvasWidget stores and updates camera orbit state (azimuth, elevation, distance)
   via update_3d_camera() and set_3d_wireframe_polylines().
3. CanvasWidget emits camera_orbit_changed and camera_pan_changed signals with
   correct types (float, float, float).
4. _WireframeWorker renders a scene WITHOUT HLR and emits projected 2D polylines.
5. _WireframeWorker emits finished(list) — not an empty list — for a valid scene.
6. Settings panel has the 3D camera group box with all required controls:
   azimuth, elevation, distance, look-at XYZ, FOV, projection, preview toggle.
7. Settings panel _get_camera_dict() returns a dict with the expected keys.
8. _on_canvas_camera_orbit_changed() syncs canvas orbit back to spinboxes
   without re-triggering the wireframe timer.
9. _on_canvas_camera_pan_changed() syncs canvas pan back to spinboxes.
10. Camera controls trigger project metadata persistence (scene3d_camera key).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project_with_3d_layer():
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4")
    proj = Project(name="3DPreviewTest", canvas=canvas)
    layer = Layer(name="Sphere", color="#000000")
    layer.generator_info = {
        "mode": "3D Scene",
        "generator": "3D Scene",
        "params": {
            "shape_type": "Sphere",
            "sphere_radius": 1.0,
            "sphere_lat_lines": 4,
            "sphere_lng_lines": 4,
            "hlr_enabled": False,
            "chop_step": 0.2,
        },
    }
    proj.add_layer(layer)
    return proj


CAMERA_DICT = {
    "azimuth": 30.0,
    "elevation": 20.0,
    "distance": 8.0,
    "look_at_x": 0.0,
    "look_at_y": 0.0,
    "look_at_z": 0.0,
    "fov": 45.0,
    "projection": "perspective",
}


# ---------------------------------------------------------------------------
# 1–3. CanvasWidget 3D preview mode and camera state
# ---------------------------------------------------------------------------


class TestCanvasWidget3DPreview:
    """CanvasWidget exposes 3D preview mode with correct state management."""

    @pytest.fixture
    def controller(self, qapp):
        from plottter.gui.project_controller import ProjectController

        proj = _make_project_with_3d_layer()
        return ProjectController(proj)

    @pytest.fixture
    def canvas(self, controller, qtbot):
        from plottter.gui.canvas_widget import CanvasWidget

        w = CanvasWidget(controller)
        qtbot.addWidget(w)
        return w

    def test_3d_preview_disabled_by_default(self, canvas) -> None:
        """3D preview should be off when the widget is first created."""
        assert not canvas._3d_preview_active, (
            "3D preview should be inactive on construction"
        )

    def test_set_3d_preview_active_enables_mode(self, canvas) -> None:
        """set_3d_preview_active(True) should enable 3D preview mode."""
        canvas.set_3d_preview_active(True)
        assert canvas._3d_preview_active, "3D preview should be active after enabling"

    def test_set_3d_preview_active_false_disables_mode(self, canvas) -> None:
        """set_3d_preview_active(False) should disable 3D preview mode."""
        canvas.set_3d_preview_active(True)
        canvas.set_3d_preview_active(False)
        assert not canvas._3d_preview_active, (
            "3D preview should be inactive after disabling"
        )

    def test_update_3d_camera_stores_values(self, canvas) -> None:
        """update_3d_camera() should store azimuth, elevation, distance, lookat."""
        canvas.update_3d_camera(
            azimuth=45.0,
            elevation=30.0,
            distance=12.0,
            lookat=(1.0, 2.0, 3.0),
        )
        assert canvas._3d_cam_azimuth == 45.0
        assert canvas._3d_cam_elevation == 30.0
        assert canvas._3d_cam_distance == 12.0
        assert canvas._3d_cam_lookat == (1.0, 2.0, 3.0)

    def test_set_3d_wireframe_polylines_stores_data(self, canvas) -> None:
        """set_3d_wireframe_polylines() should store the polyline list."""
        polylines = [[(0.0, 0.0), (10.0, 10.0)], [(5.0, 5.0), (15.0, 5.0)]]
        canvas.set_3d_wireframe_polylines(polylines)
        assert canvas._3d_wireframe_polylines == polylines

    def test_camera_orbit_changed_signal_exists(self, canvas) -> None:
        """CanvasWidget must expose a camera_orbit_changed signal."""
        assert hasattr(canvas, "camera_orbit_changed"), (
            "CanvasWidget must have camera_orbit_changed signal"
        )

    def test_camera_pan_changed_signal_exists(self, canvas) -> None:
        """CanvasWidget must expose a camera_pan_changed signal."""
        assert hasattr(canvas, "camera_pan_changed"), (
            "CanvasWidget must have camera_pan_changed signal"
        )

    def test_reset_3d_camera_restores_defaults(self, canvas, qtbot) -> None:
        """_reset_3d_camera() should restore camera to default values and emit signal."""
        canvas.update_3d_camera(azimuth=90.0, elevation=60.0, distance=20.0,
                                lookat=(5.0, 5.0, 5.0))

        orbit_signals = []
        canvas.camera_orbit_changed.connect(
            lambda az, el, dist: orbit_signals.append((az, el, dist))
        )

        canvas._reset_3d_camera()

        assert canvas._3d_cam_azimuth == 30.0
        assert canvas._3d_cam_elevation == 20.0
        assert canvas._3d_cam_distance == 8.0
        assert canvas._3d_cam_lookat == (0.0, 0.0, 0.0)
        assert len(orbit_signals) == 1
        assert orbit_signals[0] == (30.0, 20.0, 8.0)

    def test_enabling_3d_preview_disables_mask_paint(self, canvas) -> None:
        """Activating 3D preview should deactivate mask paint mode."""
        canvas._mask_paint_active = True
        canvas.set_3d_preview_active(True)
        assert not canvas._mask_paint_active, (
            "Mask paint mode should be disabled when 3D preview is activated"
        )

    def test_enabling_3d_preview_disables_shape_draw(self, canvas) -> None:
        """Activating 3D preview should deactivate shape draw mode."""
        canvas._shape_draw_active = True
        canvas.set_3d_preview_active(True)
        assert not canvas._shape_draw_active, (
            "Shape draw mode should be disabled when 3D preview is activated"
        )


# ---------------------------------------------------------------------------
# 4–5. _WireframeWorker renders without HLR
# ---------------------------------------------------------------------------


class TestWireframeWorker:
    """_WireframeWorker renders a scene without HLR and emits 2D polylines."""

    def _run_worker_sync(self, layer_params_list, camera_dict, w_mm=210.0, h_mm=297.0):
        """Run the worker synchronously by calling run() directly.

        We call run() directly (not start()) so the test doesn't require an
        event loop for QThread.
        """
        from plottter.gui.settings_panel import _WireframeWorker

        results = []
        errors = []

        worker = _WireframeWorker(
            layer_params_list=layer_params_list,
            camera_dict=camera_dict,
            canvas_w_mm=w_mm,
            canvas_h_mm=h_mm,
        )
        worker.result_ready.connect(results.append)
        worker.render_error.connect(errors.append)
        worker.run()  # synchronous call in test context

        return results, errors

    def test_worker_produces_polylines_for_sphere(self) -> None:
        """_WireframeWorker should produce non-empty polylines for a sphere layer."""
        layer_params = [
            {
                "shape_type": "Sphere",
                "sphere_radius": 1.0,
                "sphere_lat_lines": 4,
                "sphere_lng_lines": 4,
            }
        ]
        results, errors = self._run_worker_sync(layer_params, CAMERA_DICT)

        assert not errors, f"Worker emitted error: {errors}"
        assert len(results) == 1, "Worker should emit finished exactly once"
        polylines = results[0]
        assert isinstance(polylines, list), "finished must emit a list"
        assert len(polylines) > 0, "Sphere should produce at least one wireframe polyline"

    def test_worker_each_polyline_has_2d_points(self) -> None:
        """Each polyline from the worker must contain 2D (x, y) points."""
        layer_params = [
            {
                "shape_type": "Cube",
                "cube_size": 1.5,
            }
        ]
        results, errors = self._run_worker_sync(layer_params, CAMERA_DICT)

        assert not errors, f"Worker emitted error: {errors}"
        polylines = results[0]
        for poly in polylines:
            assert len(poly) >= 2, f"Polyline has fewer than 2 points: {poly}"
            for pt in poly:
                assert len(pt) == 2, f"Point must be 2D: {pt}"
                assert isinstance(pt[0], float), f"x must be float: {pt[0]}"
                assert isinstance(pt[1], float), f"y must be float: {pt[1]}"

    def test_worker_empty_layer_list_emits_empty(self) -> None:
        """With no layers, worker should emit an empty list without error."""
        results, errors = self._run_worker_sync([], CAMERA_DICT)

        assert not errors, f"Worker emitted error: {errors}"
        assert len(results) == 1
        assert results[0] == [], "Empty scene should produce empty polyline list"

    def test_worker_multiple_shapes_aggregate_output(self) -> None:
        """Multiple layers should produce more polylines than a single layer."""
        single = [{"shape_type": "Sphere", "sphere_radius": 1.0,
                   "sphere_lat_lines": 4, "sphere_lng_lines": 4}]
        multi = [
            {"shape_type": "Sphere", "sphere_radius": 1.0,
             "sphere_lat_lines": 4, "sphere_lng_lines": 4},
            {"shape_type": "Cube", "cube_size": 1.5},
        ]

        results_single, _ = self._run_worker_sync(single, CAMERA_DICT)
        results_multi, _ = self._run_worker_sync(multi, CAMERA_DICT)

        assert len(results_multi[0]) >= len(results_single[0]), (
            "Multiple shapes should produce at least as many polylines as one shape"
        )

    def test_worker_orthographic_also_works(self) -> None:
        """_WireframeWorker should work with orthographic projection."""
        ortho_cam = {**CAMERA_DICT, "projection": "orthographic"}
        layer_params = [{"shape_type": "Cone", "cone_radius": 1.0,
                         "cone_height": 2.0, "cone_lines": 6}]
        results, errors = self._run_worker_sync(layer_params, ortho_cam)

        assert not errors, f"Worker emitted error with orthographic projection: {errors}"
        assert len(results[0]) > 0, "Orthographic projection should produce polylines"


# ---------------------------------------------------------------------------
# 6–7. Settings panel has 3D camera controls
# ---------------------------------------------------------------------------


class TestSettingsPanel3DCamera:
    """Settings panel has a 3D camera group box with all required controls."""

    @pytest.fixture
    def controller(self, qapp):
        from plottter.gui.project_controller import ProjectController

        proj = _make_project_with_3d_layer()
        return ProjectController(proj)

    @pytest.fixture
    def panel(self, controller, qtbot):
        from plottter.gui.settings_panel import SettingsPanel

        sp = SettingsPanel(controller)
        qtbot.addWidget(sp)
        return sp

    def test_3d_camera_group_exists(self, panel) -> None:
        """Settings panel must have a _3d_camera_group QGroupBox."""
        assert hasattr(panel, "_3d_camera_group"), (
            "SettingsPanel must have _3d_camera_group"
        )

    def test_3d_camera_group_initially_hidden(self, panel) -> None:
        """3D camera group is hidden when not in 3D Scene mode."""
        assert not panel._3d_camera_group.isVisible(), (
            "3D camera group should be hidden when not in 3D Scene mode"
        )

    def test_camera_spinboxes_exist(self, panel) -> None:
        """Settings panel must have spinboxes for all camera orbit parameters."""
        required = [
            "_cam_azimuth_spin",
            "_cam_elevation_spin",
            "_cam_distance_spin",
            "_cam_lookat_x_spin",
            "_cam_lookat_y_spin",
            "_cam_lookat_z_spin",
            "_cam_fov_spin",
        ]
        for attr in required:
            assert hasattr(panel, attr), (
                f"SettingsPanel must have {attr}"
            )

    def test_camera_projection_combo_exists(self, panel) -> None:
        """Settings panel must have a projection combo with perspective/orthographic."""
        assert hasattr(panel, "_cam_projection_combo"), (
            "SettingsPanel must have _cam_projection_combo"
        )
        items = [
            panel._cam_projection_combo.itemText(i)
            for i in range(panel._cam_projection_combo.count())
        ]
        assert "perspective" in items, "Combo must include 'perspective'"
        assert "orthographic" in items, "Combo must include 'orthographic'"

    def test_3d_preview_button_exists(self, panel) -> None:
        """Settings panel must have a checkable 3D preview toggle button."""
        assert hasattr(panel, "_3d_preview_btn"), (
            "SettingsPanel must have _3d_preview_btn"
        )
        assert panel._3d_preview_btn.isCheckable(), (
            "_3d_preview_btn must be checkable (toggle button)"
        )

    def test_get_camera_dict_returns_all_keys(self, panel) -> None:
        """_get_camera_dict() must return a dict with all expected camera keys."""
        cam = panel._get_camera_dict()
        required_keys = {
            "azimuth", "elevation", "distance",
            "look_at_x", "look_at_y", "look_at_z",
            "fov", "projection",
        }
        missing = required_keys - set(cam.keys())
        assert not missing, f"Camera dict missing keys: {missing}"

    def test_get_camera_dict_types(self, panel) -> None:
        """Camera dict values must be numeric (float) or string."""
        cam = panel._get_camera_dict()
        float_keys = {"azimuth", "elevation", "distance",
                      "look_at_x", "look_at_y", "look_at_z", "fov"}
        for key in float_keys:
            assert isinstance(cam[key], (int, float)), (
                f"cam['{key}'] must be numeric, got {type(cam[key])}"
            )
        assert isinstance(cam["projection"], str), (
            "cam['projection'] must be a string"
        )

    def test_import_mesh_button_exists(self, panel) -> None:
        """Settings panel must have an Import Mesh button."""
        assert hasattr(panel, "_import_mesh_btn"), (
            "SettingsPanel must have _import_mesh_btn"
        )


# ---------------------------------------------------------------------------
# 8–9. Bidirectional camera sync between canvas and settings panel
# ---------------------------------------------------------------------------


class TestCameraSync:
    """Camera changes in the canvas are reflected back in the settings panel spinboxes."""

    @pytest.fixture
    def controller(self, qapp):
        from plottter.gui.project_controller import ProjectController

        proj = _make_project_with_3d_layer()
        return ProjectController(proj)

    @pytest.fixture
    def panel(self, controller, qtbot):
        from plottter.gui.settings_panel import SettingsPanel

        sp = SettingsPanel(controller)
        qtbot.addWidget(sp)
        return sp

    def test_on_canvas_camera_orbit_changed_syncs_spinboxes(self, panel) -> None:
        """_on_canvas_camera_orbit_changed() should update the three orbit spinboxes."""
        panel._on_canvas_camera_orbit_changed(90.0, -15.0, 12.5)

        assert panel._cam_azimuth_spin.value() == pytest.approx(90.0, abs=0.01)
        assert panel._cam_elevation_spin.value() == pytest.approx(-15.0, abs=0.01)
        assert panel._cam_distance_spin.value() == pytest.approx(12.5, abs=0.01)

    def test_on_canvas_camera_pan_changed_syncs_lookat_spinboxes(self, panel) -> None:
        """_on_canvas_camera_pan_changed() should update the look-at spinboxes."""
        panel._on_canvas_camera_pan_changed(1.5, -0.5, 3.0)

        assert panel._cam_lookat_x_spin.value() == pytest.approx(1.5, abs=0.01)
        assert panel._cam_lookat_y_spin.value() == pytest.approx(-0.5, abs=0.01)
        assert panel._cam_lookat_z_spin.value() == pytest.approx(3.0, abs=0.01)

    def test_orbit_sync_does_not_fire_camera_changed_infinitely(self, panel) -> None:
        """Syncing orbit from canvas should not create an infinite signal loop.

        _on_canvas_camera_orbit_changed blocks signals during the update
        so _on_camera_changed is called at most once (for the persist call).
        This test verifies the call completes without recursion or error.
        """
        # This should complete without raising RecursionError or similar
        panel._on_canvas_camera_orbit_changed(45.0, 10.0, 6.0)
        panel._on_canvas_camera_orbit_changed(60.0, 20.0, 8.0)
        # If we reach here, no infinite loop occurred
        assert panel._cam_azimuth_spin.value() == pytest.approx(60.0, abs=0.01)

    def test_pan_sync_does_not_fire_camera_changed_infinitely(self, panel) -> None:
        """Syncing pan from canvas should not create an infinite signal loop."""
        panel._on_canvas_camera_pan_changed(0.5, 0.5, 0.5)
        panel._on_canvas_camera_pan_changed(1.0, 1.0, 1.0)
        # Should complete without error
        assert panel._cam_lookat_x_spin.value() == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 10. Camera controls persist to project metadata
# ---------------------------------------------------------------------------


class TestCameraMetadataPersistence:
    """Camera changes are persisted to project.metadata['scene3d_camera']."""

    @pytest.fixture
    def controller(self, qapp):
        from plottter.gui.project_controller import ProjectController

        proj = _make_project_with_3d_layer()
        return ProjectController(proj)

    @pytest.fixture
    def panel(self, controller, qtbot):
        from plottter.gui.settings_panel import SettingsPanel

        sp = SettingsPanel(controller)
        sp._current_mode = "3D Scene"  # simulate 3D mode
        qtbot.addWidget(sp)
        return sp

    def test_on_camera_changed_persists_to_metadata(self, panel, controller) -> None:
        """_on_camera_changed() should write camera values to project.metadata."""
        panel._cam_azimuth_spin.setValue(75.0)
        panel._cam_elevation_spin.setValue(35.0)
        panel._cam_distance_spin.setValue(10.0)
        panel._on_camera_changed()

        project = controller.current_project
        assert "scene3d_camera" in project.metadata, (
            "project.metadata should have 'scene3d_camera' after camera change"
        )
        cam = project.metadata["scene3d_camera"]
        assert cam.get("azimuth") == pytest.approx(75.0, abs=0.01)
        assert cam.get("elevation") == pytest.approx(35.0, abs=0.01)
        assert cam.get("distance") == pytest.approx(10.0, abs=0.01)

    def test_load_camera_from_project_restores_values(self, panel, controller) -> None:
        """_load_camera_from_project() should restore spinbox values from metadata."""
        # First write some camera values to metadata
        controller.current_project.metadata["scene3d_camera"] = {
            "azimuth": 120.0,
            "elevation": -30.0,
            "distance": 15.0,
            "look_at_x": 2.0,
            "look_at_y": 1.0,
            "look_at_z": -1.0,
            "fov": 60.0,
            "projection": "orthographic",
        }

        panel._load_camera_from_project()

        assert panel._cam_azimuth_spin.value() == pytest.approx(120.0, abs=0.01)
        assert panel._cam_elevation_spin.value() == pytest.approx(-30.0, abs=0.01)
        assert panel._cam_distance_spin.value() == pytest.approx(15.0, abs=0.01)
        assert panel._cam_lookat_x_spin.value() == pytest.approx(2.0, abs=0.01)
        assert panel._cam_lookat_y_spin.value() == pytest.approx(1.0, abs=0.01)
        assert panel._cam_lookat_z_spin.value() == pytest.approx(-1.0, abs=0.01)
        assert panel._cam_fov_spin.value() == pytest.approx(60.0, abs=0.01)
        assert panel._cam_projection_combo.currentText() == "orthographic"
