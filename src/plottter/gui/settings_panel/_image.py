"""_ImageMixin — image source, preprocessing, and depth map methods."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
)

from .workers import _DepthMapWorker


class _ImageMixin:
    """Mixin for image loading, preprocessing, and depth map methods."""

    def _refresh_source_layer_combo(self, *_args: Any) -> None:
        """Refresh the rasterize source layer combo, excluding the current target layer."""
        # Get the current target layer id to exclude (prevent self-referencing)
        target_idx = self._layer_combo.currentIndex()
        target_layer_id = self._layer_combo.itemData(target_idx) if target_idx >= 0 else None

        current_src = self._source_layer_combo.currentData()
        self._source_layer_combo.blockSignals(True)
        self._source_layer_combo.clear()
        for layer in self._controller.current_project.layers:
            if layer.id != target_layer_id:
                self._source_layer_combo.addItem(layer.name, layer.id)
        # Restore previous selection if still available
        idx = self._source_layer_combo.findData(current_src)
        if idx >= 0:
            self._source_layer_combo.setCurrentIndex(idx)
        self._source_layer_combo.blockSignals(False)

    def _on_image_source_type_changed(self, checked: bool = True) -> None:
        """Toggle between file, layer, and AI depth map source UI.

        Connected to all 3 radio buttons' toggled signals.  When a button
        becomes *un*checked (checked=False), we skip processing — the handler
        for the button that just became *checked* will run immediately after.
        """
        if not checked:
            return

        # Determine which source type is now active
        if self._src_type_file_radio.isChecked():
            new_type = "file"
        elif self._src_type_layer_radio.isChecked():
            new_type = "layer"
        elif self._src_type_depth_radio.isChecked():
            new_type = "depth_map"
        else:
            return

        prev_type = self._image_source_type
        self._image_source_type = new_type

        self._file_src_widget.setVisible(new_type == "file")
        self._layer_src_widget.setVisible(new_type == "layer")
        self._depth_src_widget.setVisible(new_type == "depth_map")

        if new_type == "layer":
            # Switching to layer mode — refresh the combo and auto-rasterize if possible
            self._refresh_source_layer_combo()
            self._on_rasterize_layer()
        elif new_type == "file":
            # Switching back to file mode — restore original file-based image
            if prev_type == "depth_map" and self._original_raw_image is not None:
                self._raw_image = self._original_raw_image
                self._original_raw_image = None
                self._update_image_preview()
            elif self._image_source_path:
                try:
                    from plottter.io.image_import import load_image
                    self._raw_image = load_image(self._image_source_path)
                    self._update_image_preview()
                except Exception:
                    pass
            else:
                self._raw_image = None
                self._update_image_preview()
        elif new_type == "depth_map":
            # Switching to depth map mode — save original image if needed
            if prev_type == "file" and self._raw_image is not None:
                self._original_raw_image = self._raw_image
            # Check if we already have a cached depth map for this image
            cache_key = self._image_source_path
            if cache_key and cache_key in self._depth_map_cache:
                depth = self._depth_map_cache[cache_key]
                if self._depth_invert_check.isChecked():
                    depth = 1.0 - depth
                self._apply_depth_map(depth)
                self._depth_status_label.setText("Depth map ready (cached)")
            else:
                self._depth_status_label.setText("No depth map generated")

    def _on_source_layer_combo_changed(self, _index: int = 0) -> None:
        """Auto-rasterize when the source layer selection changes."""
        if self._image_source_type == "layer":
            self._on_rasterize_layer()

    def _on_rasterize_layer(self) -> None:
        """Rasterize the selected source layer and use it as the raw image."""
        if self._image_source_type != "layer":
            return

        idx = self._source_layer_combo.currentIndex()
        if idx < 0:
            self._layer_src_status_label.setText("No source layer selected.")
            return

        layer_id = self._source_layer_combo.itemData(idx)
        layer = self._controller.get_layer(layer_id)
        if layer is None:
            self._layer_src_status_label.setText("Source layer not found.")
            return

        if not layer.paths:
            self._layer_src_status_label.setText("Warning: source layer has no paths.")
            return

        canvas = self._controller.current_project.canvas
        dpi = self._rasterize_dpi_spin.value()
        stroke_mm = self._rasterize_stroke_spin.value()

        try:
            from plottter.processing.rasterize import rasterize_layer
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rasterized = rasterize_layer(layer, canvas, resolution_dpi=dpi, stroke_width_mm=stroke_mm)
            if caught:
                self._layer_src_status_label.setText(str(caught[0].message))
            else:
                h, w = rasterized.shape
                self._layer_src_status_label.setText(f"Rasterized: {w}×{h} px")
        except Exception as exc:
            self._layer_src_status_label.setText(f"Error: {exc}")
            return

        self._source_layer_id = layer_id
        self._raw_image = rasterized
        self._ai_bg_rgba = None
        self._update_image_preview()

    def _on_source_layer_paths_changed(self, layer_id: str) -> None:
        """Re-rasterize when the source layer's paths change."""
        if self._image_source_type == "layer" and layer_id == self._source_layer_id:
            self._on_rasterize_layer()

    # ------------------------------------------------------------------
    # AI Depth Map source methods
    # ------------------------------------------------------------------

    def _on_generate_depth_map(self) -> None:
        """Generate a depth map for the currently loaded image via Replicate AI."""
        # Use the original file image as source (not a previously-computed depth map)
        source_image = (
            self._original_raw_image
            if self._original_raw_image is not None
            else self._raw_image
        )
        if source_image is None:
            QMessageBox.information(
                self,
                "No Image Loaded",
                "Please load a source image first (use the File source type to load an image, "
                "then switch to AI Depth Map).",
            )
            return

        # Check in-memory cache first
        cache_key = self._image_source_path
        if cache_key and cache_key in self._depth_map_cache:
            depth = self._depth_map_cache[cache_key]
            if self._depth_invert_check.isChecked():
                depth = 1.0 - depth
            self._apply_depth_map(depth)
            self._depth_status_label.setText("Depth map ready (cached)")
            return

        # Read API key and cache directory from QSettings
        from PyQt6.QtCore import QSettings
        settings = QSettings("Plottter", "Plottter")
        api_key = str(settings.value("replicate/api_key", "") or "")
        raw_cache_dir = (
            settings.value("ai/cache_dir", "") or
            settings.value("ai/depth_cache_dir", "") or ""
        )
        cache_dir: "str | None" = raw_cache_dir.strip() or None
        if cache_dir is None:
            import pathlib
            cache_dir = str(pathlib.Path.home() / ".plottter" / "ai_cache")

        if not api_key:
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please configure your Replicate API key in Preferences (Ctrl+,) before "
                "generating a depth map.",
            )
            return

        # Guard against starting a second worker while one is still running
        if self._depth_map_worker is not None and self._depth_map_worker.isRunning():
            return

        self._depth_status_label.setText("Generating…")
        self._gen_depth_btn.setEnabled(False)

        self._depth_map_worker = _DepthMapWorker(api_key, cache_dir, source_image)
        self._depth_map_worker.progress.connect(lambda p: None)  # optional: update status
        self._depth_map_worker.finished.connect(self._on_depth_map_ready)
        self._depth_map_worker.error.connect(self._on_depth_map_error)
        self._depth_map_worker.finished.connect(self._depth_map_worker.deleteLater)
        self._depth_map_worker.error.connect(self._depth_map_worker.deleteLater)
        self._depth_map_worker.start()

    def _on_depth_map_ready(self, depth_map: "np.ndarray") -> None:
        """Called when the depth map worker finishes successfully."""
        if self._depth_map_worker is not None:
            self._depth_map_worker.wait()
            self._depth_map_worker = None
        self._gen_depth_btn.setEnabled(True)
        # Store in in-memory cache before applying inversion
        cache_key = self._image_source_path
        if cache_key:
            self._depth_map_cache[cache_key] = depth_map

        if self._depth_invert_check.isChecked():
            depth_map = 1.0 - depth_map
        self._apply_depth_map(depth_map)
        self._depth_status_label.setText("Depth map ready")

    def _on_depth_map_error(self, error_msg: str) -> None:
        """Called when the depth map worker fails."""
        if self._depth_map_worker is not None:
            self._depth_map_worker.wait()
            self._depth_map_worker = None
        self._gen_depth_btn.setEnabled(True)
        self._depth_status_label.setText(f"Error: {error_msg[:80]}")
        QMessageBox.warning(
            self,
            "Depth Map Error",
            f"Failed to generate depth map:\n\n{error_msg}",
        )

    def _apply_depth_map(self, depth_map: "np.ndarray") -> None:
        """Convert float32 depth map to 3-channel uint8 and set as the raw image."""
        depth_uint8 = (depth_map * 255.0).clip(0, 255).astype("uint8")
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        self._raw_image = depth_rgb
        self._ai_bg_rgba = None
        self._update_image_preview()

    def _on_depth_invert_changed(self, checked: bool) -> None:
        """Re-apply the depth map with updated inversion when the checkbox is toggled."""
        if self._image_source_type != "depth_map":
            return
        cache_key = self._image_source_path
        if cache_key and cache_key in self._depth_map_cache:
            depth = self._depth_map_cache[cache_key]
            if checked:
                depth = 1.0 - depth
            self._apply_depth_map(depth)
            self._depth_status_label.setText("Depth map ready (inverted)" if checked else "Depth map ready")

    def _get_cache_dir(self) -> str:
        """Return the AI disk cache directory path (creates default path if not configured)."""
        import pathlib
        from PyQt6.QtCore import QSettings

        settings = QSettings("Plottter", "Plottter")
        raw_cache_dir = (
            settings.value("ai/cache_dir", "") or
            settings.value("ai/depth_cache_dir", "") or ""
        )
        cache_dir = raw_cache_dir.strip() or str(pathlib.Path.home() / ".plottter" / "ai_cache")
        return cache_dir

    def _check_ai_cache_for_image(self, image: "np.ndarray", path: str) -> None:
        """Pre-load AI results from disk cache for *image* without applying them.

        Populates ``self._ai_bg_rgba`` if a cached BG-removal result exists, and
        ``self._depth_map_cache[path]`` if a cached depth map exists.  Updates the
        UI indicators accordingly.  Does NOT auto-apply or auto-enable checkboxes.
        """
        import hashlib
        import os

        cache_dir = self._get_cache_dir()
        img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # --- BG removal cache ---
        bg_cache_path = os.path.join(cache_dir, "bg_removal", f"{img_hash}.png")
        if os.path.exists(bg_cache_path):
            try:
                from PIL import Image as _PIL_Image

                pil = _PIL_Image.open(bg_cache_path).convert("RGBA")
                result = np.array(pil)
                if result.shape[:2] == image.shape[:2]:
                    self._ai_bg_rgba = result
                    self._ai_bg_cached_label.setVisible(True)
            except Exception:
                pass

        # --- Depth map cache ---
        flat_path = os.path.join(cache_dir, f"{img_hash}.png")
        subdir_path = os.path.join(cache_dir, "depth", f"{img_hash}.png")
        depth_cache_path = flat_path if os.path.exists(flat_path) else subdir_path
        if os.path.exists(depth_cache_path):
            try:
                from PIL import Image as _PIL_Image

                pil = _PIL_Image.open(depth_cache_path)
                arr = np.array(pil).astype(np.float32)
                if arr.max() > 1.0:
                    arr = arr / 65535.0
                if arr.shape == tuple(image.shape[:2]):
                    self._depth_map_cache[path] = arr
                    self._depth_status_label.setText("Depth map ready (cached)")
            except Exception:
                pass

    def _on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.gif);;All Files (*)",
        )
        if not path:
            return
        try:
            from plottter.io.image_import import load_image

            self._raw_image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Image Load Error", str(exc))
            return

        self._image_source_path = path

        # Invalidate any cached AI background removal result for the previous image.
        # Also uncheck the AI BG checkbox so that cached results are NOT auto-applied
        # when the preview renders — the user must explicitly re-enable it.
        self._ai_bg_rgba = None
        self._ai_bg_cached_label.setVisible(False)
        self._ai_bg_check.blockSignals(True)
        self._ai_bg_check.setChecked(False)
        self._ai_bg_check.blockSignals(False)
        self._depth_status_label.setText("No depth map generated")

        # Pre-load any existing AI cache results for this image (without auto-applying)
        self._check_ai_cache_for_image(self._raw_image, path)

        # Update the AI BG checkbox enabled state in case cached result availability changed
        self.update_ai_availability()

        # Update custom size spinboxes to match the canvas drawing area by default
        self._reset_image_size_to_canvas()

        import os

        self._image_filename_label.setText(os.path.basename(path))
        self._update_ai_mask_image_label()
        self._update_image_preview()

    def _on_preprocessing_changed(self, *_args: Any) -> None:
        self._gamma_val_label.setText(f"{self._gamma_slider.value() / 100:.2f}")
        self._unsharp_val_label.setText(f"{self._unsharp_slider.value() / 10:.1f}")
        self._preprocess_timer.start()

    def _on_reset_preprocessing(self) -> None:
        """Restore the image-adjustment controls to their defaults.

        Layout controls (fit mode, custom size, offsets) are intentionally
        left alone — only the visual-adjustment widgets are reset.
        """
        widgets = (
            self._auto_contrast_check,
            self._bright_slider,
            self._contrast_slider,
            self._gamma_slider,
            self._blur_slider,
            self._unsharp_slider,
            self._threshold_check,
            self._threshold_slider,
            self._invert_check,
            self._remove_bg_check,
            self._bg_tolerance_spin,
            self._ai_bg_check,
            self._crop_to_canvas_check,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self._auto_contrast_check.setChecked(True)
            self._bright_slider.setValue(0)
            self._contrast_slider.setValue(0)
            self._gamma_slider.setValue(100)
            self._blur_slider.setValue(0)
            self._unsharp_slider.setValue(0)
            self._threshold_check.setChecked(False)
            self._threshold_slider.setValue(128)
            self._threshold_slider.setEnabled(False)
            self._invert_check.setChecked(False)
            self._remove_bg_check.setChecked(False)
            self._bg_tolerance_spin.setValue(20.0)
            self._bg_tolerance_spin.setEnabled(False)
            self._ai_bg_check.setChecked(False)
            self._crop_to_canvas_check.setChecked(True)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._on_preprocessing_changed()

    def _reset_image_size_to_canvas(self) -> None:
        """Set custom size spinboxes to match the canvas drawing area dimensions."""
        try:
            canvas = self._controller.current_project.canvas
            draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
            self._image_width_spin.blockSignals(True)
            self._image_height_spin.blockSignals(True)
            self._image_width_spin.setValue(round(draw_x2 - draw_x1, 1))
            self._image_height_spin.setValue(round(draw_y2 - draw_y1, 1))
            self._image_width_spin.blockSignals(False)
            self._image_height_spin.blockSignals(False)
        except AttributeError:
            pass

    def _on_image_fit_mode_changed(self, _index: int = 0) -> None:
        """Show/hide custom size and offset controls based on fit mode."""
        mode = self._image_fit_mode()
        is_custom = mode == "custom"
        is_fill = mode == "fill"
        self._custom_size_widget.setVisible(is_custom)
        self._image_offset_widget.setVisible(not is_fill)
        # Hide crop-to-canvas when not in fill mode since explicit sizing handles it
        self._crop_to_canvas_check.setVisible(is_fill)
        self._on_preprocessing_changed()

    def _image_fit_mode(self) -> str:
        """Return the current fit mode as a string: 'fill', 'fit', or 'custom'."""
        text = self._image_fit_combo.currentText()
        if text == "Fit (Keep Aspect)":
            return "fit"
        if text == "Custom Size":
            return "custom"
        return "fill"

    # ------------------------------------------------------------------
    # Direct-manipulation image positioning (drag/zoom on canvas)
    # ------------------------------------------------------------------

    def _on_position_image_toggled(self, checked: bool) -> None:
        """Toggle interactive image-positioning on the canvas widget.

        When enabling, switch Fit Mode to 'Custom Size' so the drag has a
        stable model to write back into (width/height + X/Y offset). The
        currently-displayed rect is captured into the spinboxes first so
        the image doesn't jump on toggle.
        """
        if self._canvas_ref is None:
            self._position_image_btn.setChecked(False)
            return
        if checked:
            current_rect = self._canvas_ref.get_image_overlay_rect_mm()
            if current_rect is None or self._raw_image is None:
                self._position_image_btn.setChecked(False)
                return
            # Capture the current rect into the spinboxes before switching
            # modes so the visible image stays exactly where it is.
            self._write_rect_to_custom_spinboxes(current_rect)
            if self._image_fit_combo.currentText() != "Custom Size":
                self._image_fit_combo.blockSignals(True)
                self._image_fit_combo.setCurrentText("Custom Size")
                self._image_fit_combo.blockSignals(False)
                self._on_image_fit_mode_changed()
        self._canvas_ref.set_image_position_active(checked)

    def _on_reset_image_position(self) -> None:
        """Restore the default fit-to-canvas placement."""
        # Exit position mode if active.
        if self._position_image_btn.isChecked():
            self._position_image_btn.setChecked(False)
        self._image_offset_x_spin.blockSignals(True)
        self._image_offset_y_spin.blockSignals(True)
        self._image_offset_x_spin.setValue(0.0)
        self._image_offset_y_spin.setValue(0.0)
        self._image_offset_x_spin.blockSignals(False)
        self._image_offset_y_spin.blockSignals(False)
        self._image_fit_combo.setCurrentText("Fit (Keep Aspect)")
        # _on_image_fit_mode_changed will fire from the combo signal and
        # trigger a preview refresh.

    def _write_rect_to_custom_spinboxes(
        self, rect_mm: tuple[float, float, float, float]
    ) -> None:
        """Invert ``rect_mm`` back to (custom_w, custom_h, offset_x, offset_y).

        Mirrors :func:`compute_image_rect` for ``fit_mode == 'custom'``:
        width/height = rect size, and offset is the rect centre minus the
        drawing-area centre. Spinbox writes block signals so the 200 ms
        preprocessing debounce doesn't fire; the canvas overlay is already
        showing the new rect from the drag/wheel.
        """
        x1, y1, x2, y2 = rect_mm
        try:
            canvas = self._controller.current_project.canvas
        except AttributeError:
            return
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_cx = (draw_x1 + draw_x2) / 2.0
        draw_cy = (draw_y1 + draw_y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        rect_cx = (x1 + x2) / 2.0
        rect_cy = (y1 + y2) / 2.0
        offset_x = rect_cx - draw_cx
        offset_y = rect_cy - draw_cy
        for s, v in (
            (self._image_width_spin, w),
            (self._image_height_spin, h),
            (self._image_offset_x_spin, offset_x),
            (self._image_offset_y_spin, offset_y),
        ):
            s.blockSignals(True)
            s.setValue(round(v, 2))
            s.blockSignals(False)

    # 8-pixel halo added to the cropped source image so Category C generators
    # (Edge Detect, XDoG, FDoG) still have neighbourhood pixels available
    # near the canvas boundary. The halo is in source-image pixel space, not
    # mm — small enough to cost almost nothing, big enough to keep edge
    # responses correct at the canvas margin.
    _AUTO_CROP_PADDING_PX = 8

    def compute_visible_image_crop(
        self,
        image: "np.ndarray | None",
        params: dict,
    ) -> "tuple[np.ndarray, dict] | None":
        """Crop the source image + image-rect params to the visible canvas area.

        Returns ``None`` when no crop is needed (rect already inside the
        drawing area), when the image and canvas have no overlap (caller can
        skip generation), or when the necessary inputs are missing.

        Otherwise returns ``(cropped_image, override_params)`` where
        ``override_params`` contains adjusted ``image_fit_mode`` /
        ``image_width_mm`` / ``image_height_mm`` / ``image_offset_x_mm`` /
        ``image_offset_y_mm`` such that
        ``compute_image_rect("custom", ...)`` on the cropped image yields
        exactly the intersection of the original rect with the canvas
        drawing area.
        """
        if image is None or image.size == 0:
            return None
        try:
            canvas = self._controller.current_project.canvas
        except AttributeError:
            return None
        from plottter.generators._helpers import compute_image_rect

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        if image.ndim < 2:
            return None
        img_h, img_w = image.shape[:2]
        fit_mode = str(params.get("image_fit_mode", "fit"))
        full_rect = compute_image_rect(
            fit_mode=fit_mode,
            image_w_px=img_w,
            image_h_px=img_h,
            draw_x1=draw_x1,
            draw_y1=draw_y1,
            draw_x2=draw_x2,
            draw_y2=draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        rx1, ry1, rx2, ry2 = full_rect
        # Visible intersection.
        vx1 = max(rx1, draw_x1)
        vy1 = max(ry1, draw_y1)
        vx2 = min(rx2, draw_x2)
        vy2 = min(ry2, draw_y2)
        if vx2 <= vx1 or vy2 <= vy1:
            # Image rect is entirely off-canvas; let caller skip generation.
            return None
        # No-op when the rect already lives inside the drawing area (with a
        # tolerance for float wobble). Saves cost on the common case.
        if (
            vx1 <= rx1 + 1e-6
            and vy1 <= ry1 + 1e-6
            and vx2 >= rx2 - 1e-6
            and vy2 >= ry2 - 1e-6
        ):
            return None

        rect_w = rx2 - rx1
        rect_h = ry2 - ry1
        if rect_w <= 0 or rect_h <= 0:
            return None

        # mm-fraction within the original rect → source pixel coords.
        frac_x1 = (vx1 - rx1) / rect_w
        frac_y1 = (vy1 - ry1) / rect_h
        frac_x2 = (vx2 - rx1) / rect_w
        frac_y2 = (vy2 - ry1) / rect_h
        pad = self._AUTO_CROP_PADDING_PX
        px_x1 = max(0, int(frac_x1 * img_w) - pad)
        px_y1 = max(0, int(frac_y1 * img_h) - pad)
        px_x2 = min(img_w, int(round(frac_x2 * img_w)) + pad)
        px_y2 = min(img_h, int(round(frac_y2 * img_h)) + pad)
        if px_x2 <= px_x1 or px_y2 <= px_y1:
            return None
        cropped = image[px_y1:px_y2, px_x1:px_x2]

        # Recompute the *actual* mm rect that this crop occupies — the
        # padding may have nudged it outward beyond ``visible_rect``.
        actual_x1 = rx1 + (px_x1 / img_w) * rect_w
        actual_y1 = ry1 + (px_y1 / img_h) * rect_h
        actual_x2 = rx1 + (px_x2 / img_w) * rect_w
        actual_y2 = ry1 + (px_y2 / img_h) * rect_h
        new_w = actual_x2 - actual_x1
        new_h = actual_y2 - actual_y1
        draw_cx = (draw_x1 + draw_x2) / 2.0
        draw_cy = (draw_y1 + draw_y2) / 2.0
        new_cx = (actual_x1 + actual_x2) / 2.0
        new_cy = (actual_y1 + actual_y2) / 2.0
        override = {
            "image_fit_mode": "custom",
            "image_width_mm": new_w,
            "image_height_mm": new_h,
            "image_offset_x_mm": new_cx - draw_cx,
            "image_offset_y_mm": new_cy - draw_cy,
        }
        return cropped, override

    def _on_canvas_image_view_changed(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        """Mirror a canvas-side drag/zoom into spinboxes + project metadata."""
        rect = (x1, y1, x2, y2)
        self._write_rect_to_custom_spinboxes(rect)
        # Persist so the placement survives a save/reload (mirrors map_view).
        try:
            project = self._controller.current_project
            if project is not None:
                project.metadata["image_view"] = {
                    "fit_mode": "custom",
                    "custom_w_mm": self._image_width_spin.value(),
                    "custom_h_mm": self._image_height_spin.value(),
                    "offset_x_mm": self._image_offset_x_spin.value(),
                    "offset_y_mm": self._image_offset_y_spin.value(),
                }
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass

    def _on_image_width_changed(self, value: float) -> None:
        """If lock aspect ratio is checked, update height proportionally."""
        if self._lock_aspect_check.isChecked() and self._raw_image is not None:
            h_px, w_px = self._raw_image.shape[:2]
            if w_px > 0:
                aspect = h_px / w_px
                self._image_height_spin.blockSignals(True)
                self._image_height_spin.setValue(round(value * aspect, 2))
                self._image_height_spin.blockSignals(False)
        self._on_preprocessing_changed()

    def _on_image_height_changed(self, value: float) -> None:
        """If lock aspect ratio is checked, update width proportionally."""
        if self._lock_aspect_check.isChecked() and self._raw_image is not None:
            h_px, w_px = self._raw_image.shape[:2]
            if h_px > 0:
                aspect = w_px / h_px
                self._image_width_spin.blockSignals(True)
                self._image_width_spin.setValue(round(value * aspect, 2))
                self._image_width_spin.blockSignals(False)
        self._on_preprocessing_changed()

    def _get_preprocessing_params(self) -> dict:
        params: dict[str, Any] = {}
        params["auto_contrast"] = self._auto_contrast_check.isChecked()
        brightness = self._bright_slider.value()
        if brightness != 0:
            params["brightness"] = brightness
        contrast = self._contrast_slider.value()
        if contrast != 0:
            params["contrast"] = contrast
        gamma = self._gamma_slider.value() / 100.0
        if abs(gamma - 1.0) > 1e-6:
            params["gamma"] = gamma
        blur = self._blur_slider.value()
        if blur > 0:
            params["blur"] = float(blur)
        unsharp = self._unsharp_slider.value() / 10.0
        if unsharp > 0:
            params["unsharp_amount"] = unsharp
        if self._threshold_check.isChecked():
            params["threshold"] = float(self._threshold_slider.value())
        if self._invert_check.isChecked():
            params["invert"] = True
        if self._remove_bg_check.isChecked():
            params["remove_background"] = float(self._bg_tolerance_spin.value())
        # ai_bg_removal is handled directly in _update_image_preview() via _ai_bg_rgba cache
        # Crop to canvas is skipped when using a rasterized layer as source: the rasterized
        # image already covers exactly the drawing area and has the correct aspect ratio/content.
        # Applying crop_to_aspect would shift or scale the content, breaking coordinate alignment.
        # Also skip when not in "Fill Canvas" mode since explicit sizing handles mapping.
        fit_mode = self._image_fit_mode()
        if (
            self._crop_to_canvas_check.isChecked()
            and self._image_source_type != "layer"
            and fit_mode == "fill"
        ):
            canvas = self._controller.current_project.canvas
            params["crop_width"] = canvas.width_mm * 5
            params["crop_height"] = canvas.height_mm * 5

        # Image size & position params (used by generators via compute_image_rect)
        params["image_fit_mode"] = fit_mode
        if fit_mode != "fill":
            params["image_offset_x_mm"] = self._image_offset_x_spin.value()
            params["image_offset_y_mm"] = self._image_offset_y_spin.value()
        if fit_mode == "custom":
            params["image_width_mm"] = self._image_width_spin.value()
            params["image_height_mm"] = self._image_height_spin.value()
        return params

    def _update_image_preview(self) -> None:
        if self._raw_image is None:
            self._preprocessed_image = None
            self._preprocessed_color = None
            self._thumbnail_label.clear()
            self.image_preprocessed.emit(None)
            self.image_rect_changed.emit(None)
            # No image → exit + disable Position Image.
            if self._position_image_btn.isChecked():
                self._position_image_btn.setChecked(False)
            self._position_image_btn.setEnabled(False)
            return
        self._position_image_btn.setEnabled(True)

        try:
            from plottter.io.image_import import preprocess, to_grayscale

            params = self._get_preprocessing_params()
            # If AI BG removal is active and we have a cached RGBA result, composite
            # onto white to produce an RGB base image before normal preprocessing.
            source = self._raw_image
            if (
                self._ai_bg_check.isChecked()
                and self._ai_bg_rgba is not None
            ):
                rgba = self._ai_bg_rgba
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                rgb = rgba[:, :, :3].astype(np.float32)
                white = np.full_like(rgb, 255.0)
                source = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
            preprocessed = preprocess(source, params)
            gray = to_grayscale(preprocessed)
            self._preprocessed_image = gray
            # Keep the pre-grayscale image for generators that opt in to colour
            # (PixelArt's palette quantizer, etc.).  preprocess() may emit a 2D
            # array when threshold is enabled; in that case there's no colour
            # info to preserve so we fall back to None.
            self._preprocessed_color = preprocessed if preprocessed.ndim == 3 else None
        except Exception as exc:
            QMessageBox.warning(self, "Preprocessing Error", str(exc))
            return

        # Update thumbnail
        self._update_thumbnail(self._preprocessed_image)
        # Notify canvas of new image and its placement rect
        self.image_preprocessed.emit(self._preprocessed_image)
        self._emit_image_rect()

    def _emit_image_rect(self) -> None:
        """Compute and emit the mm rect where the image overlay should be drawn."""
        gray = self._preprocessed_image
        if gray is None:
            self.image_rect_changed.emit(None)
            return
        from plottter.generators._helpers import compute_image_rect
        canvas = self._controller.current_project.canvas
        margin = canvas.margin_mm
        draw_x1 = margin
        draw_y1 = margin
        draw_x2 = canvas.width_mm - margin
        draw_y2 = canvas.height_mm - margin
        h, w = gray.shape[:2]
        fit_mode = self._image_fit_mode()
        custom_w = self._image_width_spin.value() if fit_mode == "custom" else None
        custom_h = self._image_height_spin.value() if fit_mode == "custom" else None
        offset_x = self._image_offset_x_spin.value() if fit_mode != "fill" else 0.0
        offset_y = self._image_offset_y_spin.value() if fit_mode != "fill" else 0.0
        rect = compute_image_rect(
            fit_mode=fit_mode,
            image_w_px=w,
            image_h_px=h,
            draw_x1=draw_x1,
            draw_y1=draw_y1,
            draw_x2=draw_x2,
            draw_y2=draw_y2,
            custom_w_mm=custom_w,
            custom_h_mm=custom_h,
            offset_x_mm=offset_x,
            offset_y_mm=offset_y,
        )
        self.image_rect_changed.emit(rect)

    def _update_thumbnail(self, gray: np.ndarray) -> None:
        from PyQt6.QtGui import QImage

        arr = np.ascontiguousarray(gray)
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        label_w = self._thumbnail_label.width() or 200
        label_h = self._thumbnail_label.height() or 120
        scaled = pixmap.scaled(
            label_w,
            label_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail_label.setPixmap(scaled)
