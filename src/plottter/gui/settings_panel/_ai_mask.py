"""_AiMaskMixin — AI mask generation handlers."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from .workers import _AiMaskWorker


class _AiMaskMixin:
    """Mixin for AI mask generation handlers."""

    def _update_ai_mask_image_label(self) -> None:
        """Refresh the image status label in the AI mask group."""
        if self._raw_image is not None:
            h, w = self._raw_image.shape[:2]
            self._ai_mask_image_label.setText(f"{w}×{h} px")
        else:
            self._ai_mask_image_label.setText("No image loaded")

    def _on_ai_mask_mode_changed(self, _index: int = 0) -> None:
        """Handle AI mask mode combo change: update canvas interaction and instructions."""
        mode_text = self._ai_mask_mode_combo.currentText()
        is_manual = mode_text == "Manual Brush"
        is_text = mode_text == "Text Prompt"
        is_point = mode_text == "Point Prompt"
        is_box = mode_text == "Box Prompt"

        self._ai_mask_text_input.setVisible(is_text)

        if is_point:
            self._ai_mask_instructions.setText(
                "Left-click to mark areas to include.\n"
                "Right-click to mark areas to exclude from the selection."
            )
            self._ai_mask_instructions.setVisible(True)
        elif is_box:
            self._ai_mask_instructions.setText("Left-click and drag to draw a bounding box.")
            self._ai_mask_instructions.setVisible(True)
        else:
            self._ai_mask_instructions.setVisible(False)

        # Erase is active in any Manual Brush mode; size/hardness depend on tool too
        self._erase_check.setEnabled(is_manual)
        self._update_mask_control_states()

        # Generate Mask button is only meaningful for AI modes, not Manual Brush
        self._ai_mask_generate_btn.setEnabled(not is_manual and self._ai_key_available)

        if self._canvas_ref is None or self._current_mode != "Mask Paint":
            return

        if is_manual:
            self._canvas_ref.set_ai_mask_mode(None)
            self._canvas_ref.set_mask_paint_active(True)
        elif is_point:
            self._canvas_ref.set_ai_mask_mode("point")
            self._canvas_ref.set_mask_paint_active(False)
        elif is_box:
            self._canvas_ref.set_ai_mask_mode("box")
            self._canvas_ref.set_mask_paint_active(False)
        else:
            # Text mode: AI generates from text prompt; no canvas brush interaction
            self._canvas_ref.set_ai_mask_mode(None)
            self._canvas_ref.set_mask_paint_active(False)

    def _on_ai_mask_point_selected(self, x_mm: float, y_mm: float, positive: bool) -> None:
        """Update status label when a point prompt is added."""
        if self._canvas_ref is None:
            return
        pos_count = len(self._canvas_ref.get_ai_mask_positive_points())
        neg_count = len(self._canvas_ref.get_ai_mask_negative_points())
        kind = "positive" if positive else "negative"
        self._ai_mask_status.setText(
            f"Added {kind} point — {pos_count} positive, {neg_count} negative"
        )

    def _on_ai_mask_box_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Update status label when a box prompt is drawn."""
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        self._ai_mask_status.setText(f"Box drawn: {w:.1f}×{h:.1f} mm. Click Generate Mask.")

    def _on_ai_mask_clear(self) -> None:
        """Clear all AI prompt points/box from the canvas."""
        if self._canvas_ref is not None:
            self._canvas_ref.clear_ai_mask_points()
        self._ai_mask_status.setText("Prompts cleared.")

    def _on_ai_mask_generate(self) -> None:
        """Start a background worker to generate an AI mask from the current prompts."""
        mode_text = self._ai_mask_mode_combo.currentText()
        if mode_text == "Manual Brush":
            return  # Manual Brush uses canvas painting, not AI generation

        if self._raw_image is None:
            QMessageBox.warning(
                self,
                "No Image",
                "Please load an image first (use the Load Image button or load one in Image to Lines mode).",
            )
            return

        if self._ai_mask_worker is not None and self._ai_mask_worker.isRunning():
            return

        from PyQt6.QtCore import QSettings
        settings = QSettings("Plottter", "Plottter")
        api_key = settings.value("replicate/api_key", "") or ""
        if mode_text == "Point Prompt":
            canvas_mode = "point"
        elif mode_text == "Box Prompt":
            canvas_mode = "box"
        else:
            canvas_mode = "text"

        # Validate prompts before starting the worker
        if canvas_mode == "point":
            if self._canvas_ref is None or not self._canvas_ref.get_ai_mask_positive_points():
                QMessageBox.warning(
                    self,
                    "No Points",
                    "Add at least one positive point (left-click on the image).",
                )
                return
        elif canvas_mode == "box":
            if self._canvas_ref is None or self._canvas_ref.get_ai_mask_box() is None:
                QMessageBox.warning(
                    self,
                    "No Box",
                    "Draw a bounding box by left-clicking and dragging on the canvas.",
                )
                return
        elif canvas_mode == "text":
            if not self._ai_mask_text_input.text().strip():
                QMessageBox.warning(
                    self,
                    "No Text Prompt",
                    "Enter a text description of the object to segment.",
                )
                return

        # Use the preprocessed image (which matches what's displayed on canvas,
        # including fill/fit stretching) rather than the raw image. This ensures
        # the AI mask aligns with the visible image. The preprocessed image is
        # grayscale, so convert to RGB for the AI model.
        if self._preprocessed_image is not None:
            gray = self._preprocessed_image
            source_img = np.stack([gray, gray, gray], axis=-1)
        else:
            source_img = self._raw_image
            if source_img.ndim == 2:
                source_img = np.stack([source_img] * 3, axis=-1)
            elif source_img.ndim == 3 and source_img.shape[2] == 4:
                source_img = source_img[:, :, :3]

        canvas = self._controller.current_project.canvas
        pos_pts = self._canvas_ref.get_ai_mask_positive_points() if self._canvas_ref else []
        neg_pts = self._canvas_ref.get_ai_mask_negative_points() if self._canvas_ref else []
        box_mm = self._canvas_ref.get_ai_mask_box() if self._canvas_ref else None

        self._ai_mask_generate_btn.setEnabled(False)
        self._ai_mask_progress.setMaximum(100)
        self._ai_mask_progress.setValue(0)
        self._ai_mask_progress.setVisible(True)
        self._ai_mask_status.setText("Generating AI mask…")

        # Pass drawing area dimensions (not full canvas) for mm→pixel conversion,
        # since the image fills the drawing area (inside margins).
        margin = canvas.margin_mm
        draw_w = canvas.width_mm - 2 * margin
        draw_h = canvas.height_mm - 2 * margin

        # Offset click coordinates from canvas-origin to drawing-area-origin
        # (subtract margin so (margin, margin) maps to image pixel (0, 0))
        offset_pos = [(x - margin, y - margin) for x, y in pos_pts]
        offset_neg = [(x - margin, y - margin) for x, y in neg_pts]
        offset_box = None
        if box_mm is not None:
            bx1, by1, bx2, by2 = box_mm
            offset_box = (bx1 - margin, by1 - margin, bx2 - margin, by2 - margin)

        self._ai_mask_worker = _AiMaskWorker(
            api_key=api_key,
            image=source_img,
            mode=canvas_mode,
            positive_points=offset_pos,
            negative_points=offset_neg,
            box_xyxy_mm=offset_box,
            text_prompt=self._ai_mask_text_input.text().strip(),
            canvas_width_mm=draw_w,
            canvas_height_mm=draw_h,
        )
        self._ai_mask_worker.progress.connect(self._ai_mask_progress.setValue)
        self._ai_mask_worker.finished.connect(self._on_ai_mask_result)
        self._ai_mask_worker.error.connect(self._on_ai_mask_error)
        self._ai_mask_worker.start()

    def _on_ai_mask_result(self, mask: "np.ndarray") -> None:
        """Apply the AI-generated mask to the canvas."""
        self._ai_mask_progress.setVisible(False)
        self._ai_mask_generate_btn.setEnabled(
            self._ai_mask_mode_combo.currentText() != "Manual Brush"
            and self._ai_key_available
        )

        if self._canvas_ref is None:
            return

        # Convert binary uint8 (0/255) → float32 (0.0/1.0)
        float_mask = mask.astype(np.float32) / 255.0

        # The AI mask has the source image's dimensions/aspect ratio.
        # It must be placed at the same position as the image overlay
        # on the canvas, not stretched to fill the entire canvas.
        _PX_PER_MM = 5
        canvas = self._controller.current_project.canvas
        target_h = int(canvas.height_mm * _PX_PER_MM)
        target_w = int(canvas.width_mm * _PX_PER_MM)

        # Get the image overlay rect that the canvas widget is using.
        # This is the authoritative rect — it's what the user sees.
        img_rect = self._canvas_ref.get_image_overlay_rect_mm()
        if img_rect is None:
            # Fallback: use the drawing area (fill mode)
            margin = canvas.margin_mm
            img_rect = (margin, margin,
                        canvas.width_mm - margin, canvas.height_mm - margin)

        rx1, ry1, rx2, ry2 = img_rect

        # Convert image rect from mm to mask-pixel coordinates
        px_x1 = max(0, int(round(rx1 * _PX_PER_MM)))
        px_y1 = max(0, int(round(ry1 * _PX_PER_MM)))
        px_x2 = min(target_w, int(round(rx2 * _PX_PER_MM)))
        px_y2 = min(target_h, int(round(ry2 * _PX_PER_MM)))
        region_w = px_x2 - px_x1
        region_h = px_y2 - px_y1

        if region_w > 0 and region_h > 0:
            # Resize mask to fit the image region, preserving its content
            from PIL import Image as _PIL_Image
            pil = _PIL_Image.fromarray((float_mask * 255).astype(np.uint8))
            pil = pil.resize((region_w, region_h), _PIL_Image.NEAREST)
            region_mask = np.array(pil).astype(np.float32) / 255.0

            # Place into a canvas-sized mask of zeros
            canvas_mask = np.zeros((target_h, target_w), dtype=np.float32)
            canvas_mask[px_y1:px_y2, px_x1:px_x2] = region_mask
            float_mask = canvas_mask

        self._canvas_ref.set_mask(float_mask)
        # Switch to manual brush mode so the mask overlay is visible
        # and brush controls are re-enabled for refinement.
        self._ai_mask_mode_combo.setCurrentText("Manual Brush")
        # _on_ai_mask_mode_changed fires via signal and handles:
        # - set_mask_paint_active(True)
        # - set_ai_mask_mode(None)
        # - re-enabling brush size/hardness/erase controls
        self._ai_mask_status.setText(
            "AI mask applied. Use the brush to refine, or click Apply to Layer."
        )

    def _on_ai_mask_error(self, msg: str) -> None:
        """Handle AI mask generation error."""
        self._ai_mask_progress.setVisible(False)
        self._ai_mask_generate_btn.setEnabled(
            self._ai_mask_mode_combo.currentText() != "Manual Brush"
            and self._ai_key_available
        )
        self._ai_mask_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "AI Mask Error", msg)
