"""_Scene3dMixin — 3D camera, preview, wireframe, and auto-regen methods."""

from __future__ import annotations

from PyQt6.QtWidgets import QFileDialog

from .workers import _WireframeWorker


class _Scene3dMixin:
    """Mixin for 3D scene camera, wireframe preview, and auto-regeneration methods."""

    def _on_auto_regen_3d_toggled(self, _state: int) -> None:
        """Persist the auto-regenerate checkbox state to QSettings."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("3d/auto_regenerate", self._auto_regen_3d_cb.isChecked())

    def _trigger_auto_regen_siblings(self, generated_layer_id: str) -> None:
        """Start sequential regeneration of all 3D layers *except* the one just generated."""
        # Guard: don't start a new chain while one is already in progress
        if self._auto_regen_layers:
            return

        try:
            project = self._controller.current_project
        except Exception:
            return
        if project is None:
            return

        siblings = [
            layer for layer in project.layers
            if layer.id != generated_layer_id
            and isinstance(layer.generator_info, dict)
            and layer.generator_info.get("mode") == "3D Scene"
        ]
        if not siblings:
            return

        n = len(siblings)
        self._auto_regen_layers = siblings
        self._auto_regen_idx = 0

        # Show status message
        mw = self.window()
        if hasattr(mw, "statusBar"):
            mw.statusBar().showMessage(
                f"Auto-regenerating {n} other 3D layer{'s' if n != 1 else ''}…"
            )

        self._start_auto_regen_next()

    def _start_auto_regen_next(self) -> None:
        """Start (or continue) the auto-regen chain for sibling 3D layers."""
        if self._auto_regen_idx >= len(self._auto_regen_layers):
            self._finish_auto_regen()
            return

        layer = self._auto_regen_layers[self._auto_regen_idx]
        info = layer.generator_info
        params = dict(info.get("params", {}))

        # Inject shared camera
        try:
            project = self._controller.current_project
        except Exception:
            self._finish_auto_regen()
            return
        cam = project.metadata.get("scene3d_camera", {})
        if cam:
            params["_camera"] = cam

        # Inject sibling shapes for HLR occlusion
        params["_sibling_3d_shapes"] = self._build_sibling_3d_shapes(layer.id)

        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.gui.generator_worker import GeneratorWorker

        generator = Scene3DGenerator()
        canvas = project.canvas
        layer_id = layer.id

        worker = GeneratorWorker(generator, params, canvas, parent=self)

        def on_finished(paths: list, lid: str = layer_id) -> None:
            self._controller.set_layer_paths(lid, paths, "Auto-regenerate 3D Layer")
            self._auto_regen_idx += 1
            self._start_auto_regen_next()
            worker.deleteLater()

        def on_error(_msg: str) -> None:
            # Skip failed layer and continue
            self._auto_regen_idx += 1
            self._start_auto_regen_next()
            worker.deleteLater()

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._auto_regen_worker = worker
        worker.start()

    def _finish_auto_regen(self) -> None:
        """Called when auto-regen chain completes."""
        n = len(self._auto_regen_layers)
        done = min(self._auto_regen_idx, n)
        mw = self.window()
        if hasattr(mw, "statusBar"):
            mw.statusBar().showMessage(
                f"Auto-regenerated {done} 3D layer{'s' if done != 1 else ''} successfully.",
                5000,
            )
        self._auto_regen_layers = []
        self._auto_regen_idx = 0
        self._auto_regen_worker = None

    # ------------------------------------------------------------------
    # 3D Camera helpers
    # ------------------------------------------------------------------

    def _get_camera_dict(self) -> dict:
        """Return the current camera settings as a plain dict."""
        return {
            "azimuth": self._cam_azimuth_spin.value(),
            "elevation": self._cam_elevation_spin.value(),
            "distance": self._cam_distance_spin.value(),
            "look_at_x": self._cam_lookat_x_spin.value(),
            "look_at_y": self._cam_lookat_y_spin.value(),
            "look_at_z": self._cam_lookat_z_spin.value(),
            "fov": self._cam_fov_spin.value(),
            "projection": self._cam_projection_combo.currentText(),
        }

    def _on_camera_changed(self) -> None:
        """Persist current camera settings to project metadata whenever a spin/combo changes."""
        if self._controller is None:
            return
        try:
            project = self._controller.current_project
        except Exception:
            return
        if project is None:
            return
        project.metadata["scene3d_camera"] = self._get_camera_dict()
        # If 3D preview is active, debounce a wireframe refresh
        if (
            self._canvas_ref is not None
            and self._3d_preview_btn.isChecked()
            and self._current_mode == "3D Scene"
        ):
            self._wireframe_timer.start()

    def _load_camera_from_project(self) -> None:
        """Populate camera controls from project metadata (called on project load)."""
        if self._controller is None:
            return
        try:
            project = self._controller.current_project
        except Exception:
            return
        if project is None:
            return
        cam = project.metadata.get("scene3d_camera", {})
        # Block signals so we don't trigger _on_camera_changed during restore
        for widget in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
            self._cam_fov_spin,
        ):
            widget.blockSignals(True)
        self._cam_projection_combo.blockSignals(True)

        self._cam_azimuth_spin.setValue(float(cam.get("azimuth", 30.0)))
        self._cam_elevation_spin.setValue(float(cam.get("elevation", 20.0)))
        self._cam_distance_spin.setValue(float(cam.get("distance", 8.0)))
        self._cam_lookat_x_spin.setValue(float(cam.get("look_at_x", 0.0)))
        self._cam_lookat_y_spin.setValue(float(cam.get("look_at_y", 0.0)))
        self._cam_lookat_z_spin.setValue(float(cam.get("look_at_z", 0.0)))
        self._cam_fov_spin.setValue(float(cam.get("fov", 45.0)))
        proj_text = cam.get("projection", "perspective")
        idx = self._cam_projection_combo.findText(proj_text)
        if idx >= 0:
            self._cam_projection_combo.setCurrentIndex(idx)

        for widget in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
            self._cam_fov_spin,
        ):
            widget.blockSignals(False)
        self._cam_projection_combo.blockSignals(False)

    def _build_sibling_3d_shapes(self, current_layer_id: str) -> list:
        """Collect transformed Shape objects from all other 3D Scene layers for HLR occlusion."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        shapes: list = []
        try:
            project = self._controller.current_project
        except Exception:
            return shapes
        if project is None:
            return shapes

        gen = Scene3DGenerator()
        for layer in project.layers:
            if layer.id == current_layer_id:
                continue
            info = layer.generator_info
            if not isinstance(info, dict):
                continue
            if info.get("mode") != "3D Scene":
                continue
            params = info.get("params", {})
            try:
                shape = gen.build_transformed_shape(params)
                if shape is not None:
                    shapes.append(shape)
            except Exception:
                pass  # skip broken sibling layers silently

        return shapes

    # ------------------------------------------------------------------
    # 3D preview event handlers
    # ------------------------------------------------------------------

    def _on_3d_preview_toggled(self, checked: bool) -> None:
        """Toggle real-time 3D wireframe preview on the canvas."""
        if self._canvas_ref is None:
            return
        self._3d_preview_btn.setText(
            "Disable 3D Preview" if checked else "Enable 3D Preview"
        )
        self._canvas_ref.set_3d_preview_active(checked)
        if checked:
            # Sync canvas camera state from spinboxes, then kick off a wireframe render
            self._canvas_ref.update_3d_camera(
                azimuth=self._cam_azimuth_spin.value(),
                elevation=self._cam_elevation_spin.value(),
                distance=self._cam_distance_spin.value(),
                lookat=(
                    self._cam_lookat_x_spin.value(),
                    self._cam_lookat_y_spin.value(),
                    self._cam_lookat_z_spin.value(),
                ),
            )
            self._wireframe_timer.start()
        else:
            self._wireframe_timer.stop()
            if self._wireframe_worker is not None:
                self._wireframe_worker.cancel()
                self._wireframe_worker.wait()
                self._wireframe_worker = None

    def _on_canvas_camera_orbit_changed(self, az: float, el: float, dist: float) -> None:
        """Sync canvas orbit drag result back to settings panel spinboxes."""
        # Block signals to avoid re-triggering _on_camera_changed while updating
        for spin in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
        ):
            spin.blockSignals(True)
        self._cam_azimuth_spin.setValue(az)
        self._cam_elevation_spin.setValue(el)
        self._cam_distance_spin.setValue(dist)
        for spin in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
        ):
            spin.blockSignals(False)
        # Persist and refresh
        self._on_camera_changed()

    def _on_canvas_camera_pan_changed(self, lx: float, ly: float, lz: float) -> None:
        """Sync canvas pan (look-at) result back to settings panel spinboxes."""
        for spin in (
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
        ):
            spin.blockSignals(True)
        self._cam_lookat_x_spin.setValue(lx)
        self._cam_lookat_y_spin.setValue(ly)
        self._cam_lookat_z_spin.setValue(lz)
        for spin in (
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
        ):
            spin.blockSignals(False)
        self._on_camera_changed()

    def _on_canvas_projection_toggle(self) -> None:
        """Toggle projection combo between perspective and orthographic."""
        current = self._cam_projection_combo.currentText()
        new_text = "orthographic" if current == "perspective" else "perspective"
        idx = self._cam_projection_combo.findText(new_text)
        if idx >= 0:
            self._cam_projection_combo.setCurrentIndex(idx)

    def _start_wireframe_worker(self) -> None:
        """Render all 3D layers without HLR for the live wireframe preview."""
        if self._canvas_ref is None or not self._3d_preview_btn.isChecked():
            return
        if self._current_mode != "3D Scene":
            return
        try:
            project = self._controller.current_project
        except Exception:  # noqa: BLE001
            return
        if project is None:
            return

        canvas = project.canvas
        cam_dict = self._get_camera_dict()

        # Collect params from all 3D Scene layers
        layer_params_list: list[dict] = []
        current_layer_id = self.current_layer_id()

        for layer in project.layers:
            if layer.id == current_layer_id:
                # Use live params from the settings panel for the active layer
                snapshot = self._get_settings_snapshot()
                if snapshot is not None and snapshot.get("mode") == "3D Scene":
                    layer_params_list.append(dict(snapshot.get("params", {})))
                continue
            info = layer.generator_info
            if not isinstance(info, dict):
                continue
            if info.get("mode") != "3D Scene":
                continue
            params = dict(info.get("params", {}))
            layer_params_list.append(params)

        # If a previous worker exists, cancel and wait for it to stop
        if self._wireframe_worker is not None:
            self._wireframe_worker.cancel()
            try:
                self._wireframe_worker.result_ready.disconnect()
                self._wireframe_worker.render_error.disconnect()
            except Exception:  # noqa: BLE001
                pass
            if self._wireframe_worker.isRunning():
                # Give it a moment to finish; if still running, re-queue and bail
                if not self._wireframe_worker.wait(50):
                    self._wireframe_timer.start()
                    return
            self._wireframe_worker = None

        worker = _WireframeWorker(
            layer_params_list=layer_params_list,
            camera_dict=cam_dict,
            canvas_w_mm=canvas.width_mm,
            canvas_h_mm=canvas.height_mm,
        )
        worker.result_ready.connect(self._on_wireframe_finished)
        worker.render_error.connect(self._on_wireframe_error)
        self._wireframe_worker = worker
        worker.start()

    def _on_wireframe_finished(self, polylines: list) -> None:
        """Receive rendered wireframe polylines and push them to the canvas."""
        # Worker ref kept alive until thread done — don't None it until wait() confirms
        if self._wireframe_worker is not None:
            self._wireframe_worker.wait()
            self._wireframe_worker = None
        if self._canvas_ref is not None and self._3d_preview_btn.isChecked():
            self._canvas_ref.set_3d_wireframe_polylines(polylines)

    def _on_wireframe_error(self, error_msg: str) -> None:
        """Handle a wireframe render error (preview is best-effort; no dialog shown)."""
        if self._wireframe_worker is not None:
            self._wireframe_worker.wait()
            self._wireframe_worker = None

    def _on_import_mesh(self) -> None:
        """Open an OBJ or STL file and create a new 3D Scene layer using that mesh."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import 3D Mesh",
            "",
            "3D Mesh Files (*.obj *.stl);;OBJ Files (*.obj);;STL Files (*.stl);;All Files (*)",
        )
        if not file_path:
            return

        try:
            project = self._controller.current_project
        except Exception:  # noqa: BLE001
            return
        if project is None:
            return

        import os
        from plottter.models import Layer

        name = os.path.splitext(os.path.basename(file_path))[0]
        layer = Layer(name=name or "Mesh", color="#3264C8")
        layer.generator_info = {
            "mode": "3D Scene",
            "generator": "3D Scene",
            "params": {
                "shape_type": "Mesh Import",
                "mesh_file": file_path,
                "mesh_all_edges": False,
            },
        }
        self._controller.add_layer(layer)

        # Refresh wireframe if preview is active
        if self._3d_preview_btn.isChecked():
            self._wireframe_timer.start()

    def _apply_shared_transforms(self, paths: list) -> list:
        """Apply scale, rotation, translate, mirror, rotational symmetry, and tiling to paths."""
        import math

        canvas = self._controller.current_project.canvas
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        # Scale (around canvas center)
        scale = self._transform_scale_spin.value()
        if scale != 1.0:
            paths = [
                [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in path]
                for path in paths
            ]

        # Rotation (around canvas center)
        rot_deg = self._transform_rotation_spin.value()
        if rot_deg != 0.0:
            theta = math.radians(rot_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            paths = [
                [
                    (
                        cx + (x - cx) * cos_t - (y - cy) * sin_t,
                        cy + (x - cx) * sin_t + (y - cy) * cos_t,
                    )
                    for x, y in path
                ]
                for path in paths
            ]

        # Translate
        tx = self._transform_x_spin.value()
        ty = self._transform_y_spin.value()
        if tx != 0.0 or ty != 0.0:
            paths = [[(x + tx, y + ty) for x, y in path] for path in paths]

        # Mirror Horizontal (flip around vertical center axis)
        if self._mirror_h_check.isChecked():
            mirrored = [[(2.0 * cx - x, y) for x, y in path] for path in paths]
            paths = list(paths) + mirrored

        # Mirror Vertical (flip around horizontal center axis)
        if self._mirror_v_check.isChecked():
            mirrored = [[(x, 2.0 * cy - y) for x, y in path] for path in paths]
            paths = list(paths) + mirrored

        # Rotational n-fold symmetry
        n_fold = self._n_fold_spin.value()
        if n_fold > 1:
            original = list(paths)
            for k in range(1, n_fold):
                angle = 2.0 * math.pi * k / n_fold
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                rotated = [
                    [
                        (
                            cx + (x - cx) * cos_a - (y - cy) * sin_a,
                            cy + (x - cx) * sin_a + (y - cy) * cos_a,
                        )
                        for x, y in path
                    ]
                    for path in original
                ]
                paths = paths + rotated

        # Tile repeat
        tile_rows = self._tile_rows_spin.value()
        tile_cols = self._tile_cols_spin.value()
        if tile_rows > 1 or tile_cols > 1:
            draw_w = draw_x2 - draw_x1
            draw_h = draw_y2 - draw_y1
            cell_w = draw_w / tile_cols
            cell_h = draw_h / tile_rows

            # Scale original paths to fit one tile cell (centered at first cell center)
            if paths:
                all_pts = [pt for path in paths for pt in path]
                if all_pts:
                    xs = [p[0] for p in all_pts]
                    ys = [p[1] for p in all_pts]
                    content_w = (max(xs) - min(xs)) or 1.0
                    content_h = (max(ys) - min(ys)) or 1.0
                    scale_factor = min(cell_w / content_w, cell_h / content_h) * 0.9
                    pcx = (min(xs) + max(xs)) / 2.0
                    pcy = (min(ys) + max(ys)) / 2.0

                    tiled: list = []
                    for row in range(tile_rows):
                        for col in range(tile_cols):
                            tile_cx = draw_x1 + (col + 0.5) * cell_w
                            tile_cy = draw_y1 + (row + 0.5) * cell_h
                            for path in paths:
                                new_path = [
                                    (
                                        (x - pcx) * scale_factor + tile_cx,
                                        (y - pcy) * scale_factor + tile_cy,
                                    )
                                    for x, y in path
                                ]
                                tiled.append(new_path)
                    paths = tiled

        return paths
