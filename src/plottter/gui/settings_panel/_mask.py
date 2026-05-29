"""_MaskMixin — mask paint handlers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QInputDialog,
    QListWidgetItem,
    QMessageBox,
)
from PIL import Image as _PilImage


# Mask resolution — must match canvas_widget._MASK_PX_PER_MM.  One mask
# pixel is 1 / 5 = 0.2 mm, which sets the achievable clip-boundary precision.
_MASK_PX_PER_MM = 5


def _clip_paths_to_mask(paths: list, mask) -> list:  # type: ignore[no-untyped-def]
    """Clip *paths* to the in-mask region (mask value > 0.5).

    Each segment is super-sampled at one-mask-pixel intervals so that the
    boundary cut happens along the segment itself rather than at the next
    vertex — see :meth:`_MaskMixin._clip_paths_to_mask` for context.

    Module-level so it can be unit-tested without spinning up the panel.
    """
    h, w = mask.shape
    step_mm = 1.0 / _MASK_PX_PER_MM

    def _in_mask(x_mm: float, y_mm: float) -> bool:
        px = int(x_mm * _MASK_PX_PER_MM)
        py = int(y_mm * _MASK_PX_PER_MM)
        return 0 <= px < w and 0 <= py < h and bool(mask[py, px] > 0.5)

    def _supersample(polyline):
        """Yield ``(x, y)`` along *polyline* spaced no more than ``step_mm`` apart.

        The first vertex is always emitted; each subsequent segment is
        subdivided into ``ceil(length / step_mm)`` equal sub-segments so
        that the endpoint of every sub-segment lands on either the next
        vertex or a fraction along the way — never overshooting.
        """
        if not polyline:
            return
        prev = polyline[0]
        yield prev
        for nxt in polyline[1:]:
            dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
            seg_len = math.hypot(dx, dy)
            n = max(1, int(math.ceil(seg_len / step_mm)))
            for i in range(1, n + 1):
                t = i / n
                yield (prev[0] + dx * t, prev[1] + dy * t)
            prev = nxt

    result: list = []
    for polyline in paths:
        if len(polyline) < 2:
            continue
        current_seg: list = []
        for x_mm, y_mm in _supersample(polyline):
            if _in_mask(x_mm, y_mm):
                current_seg.append((x_mm, y_mm))
            else:
                if len(current_seg) >= 2:
                    result.append(current_seg)
                current_seg = []
        if len(current_seg) >= 2:
            result.append(current_seg)
    return result


class _MaskMixin:
    """Mixin for mask paint handlers."""

    def _on_clear_mask(self) -> None:
        if self._canvas_ref is not None:
            self._canvas_ref.clear_mask()
        self._mask_status_label.setText("Mask cleared.")

    def _on_invert_mask(self) -> None:
        if self._canvas_ref is None:
            return
        before, after = self._canvas_ref.invert_mask()
        from plottter.gui.commands import MaskPaintCommand
        cmd = MaskPaintCommand(self._canvas_ref, before, after, "Invert Mask")
        self._controller.undo_stack.push(cmd)

    def _on_mask_stroke_done(self, mask) -> None:  # type: ignore[no-untyped-def]
        if mask is None:
            return
        painted_px = int((mask > 0.5).sum())
        total_px = int(mask.shape[0] * mask.shape[1])
        pct = painted_px / max(1, total_px) * 100.0
        self._mask_status_label.setText(f"Painted area: {pct:.1f}%")

    def _on_mask_op_done(self, before, after) -> None:  # type: ignore[no-untyped-def]
        """Push a MaskPaintCommand to the undo stack after any mask operation."""
        from plottter.gui.commands import MaskPaintCommand
        tool_text = self._mask_tool_combo.currentText()
        description = f"Mask {tool_text}"
        cmd = MaskPaintCommand(self._canvas_ref, before, after, description)
        self._controller.undo_stack.push(cmd)

    def _on_apply_refinement(self) -> None:
        """Apply feather and grow/shrink refinement to the current mask."""
        if self._canvas_ref is None:
            return
        mask = self._canvas_ref.get_mask()
        if mask is None or not mask.any():
            QMessageBox.warning(
                self, "Apply Refinement", "No mask to refine. Paint a mask first."
            )
            return

        from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

        # PX_PER_MM must match canvas_widget._MASK_PX_PER_MM
        PX_PER_MM = 5

        feather_mm = self._feather_spin.value()
        grow_shrink_mm = self._grow_shrink_spin.value()

        # Nothing to do if both are zero
        if feather_mm == 0.0 and grow_shrink_mm == 0.0:
            return

        before = mask.copy()
        refined = mask.astype(np.float32)

        # Apply grow/shrink BEFORE feather so feathering softens the grown/shrunk edge
        if grow_shrink_mm > 0:
            # Grow: dilate with maximum filter then re-threshold
            size = int(abs(grow_shrink_mm) * PX_PER_MM * 2 + 1)
            refined = maximum_filter(refined, size=size)
            refined = (refined > 0.5).astype(np.float32)
        elif grow_shrink_mm < 0:
            # Shrink: erode with minimum filter then re-threshold
            size = int(abs(grow_shrink_mm) * PX_PER_MM * 2 + 1)
            refined = minimum_filter(refined, size=size)
            refined = (refined > 0.5).astype(np.float32)

        # Apply feather (Gaussian blur)
        if feather_mm > 0:
            sigma = feather_mm * PX_PER_MM
            refined = gaussian_filter(refined, sigma=sigma)

        # Set the refined mask
        self._canvas_ref.set_mask(refined)
        after = refined.copy()

        # Push undo command
        from plottter.gui.commands import MaskPaintCommand
        cmd = MaskPaintCommand(self._canvas_ref, before, after, "Refine Mask")
        self._controller.undo_stack.push(cmd)

        # Update status
        self._mask_status_label.setText("Mask refinement applied.")

    def _refresh_mask_list(self, *_args: Any) -> None:
        """Repopulate the saved-masks list from the controller."""
        self._mask_list.blockSignals(True)
        current = self._mask_list.currentItem()
        current_name = current.text() if current else None
        self._mask_list.clear()
        for name in self._controller.mask_names():
            item = QListWidgetItem(name)
            # Build a 32x32 thumbnail icon from the mask array
            try:
                mask_arr = self._controller.load_mask(name)
                pil_img = _PilImage.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                pil_img = pil_img.resize((32, 32), _PilImage.LANCZOS)
                data = pil_img.tobytes()
                qimg = QImage(data, 32, 32, 32, QImage.Format.Format_Grayscale8)
                item.setIcon(QPixmap.fromImage(qimg))
            except Exception:  # noqa: BLE001
                pass
            self._mask_list.addItem(item)
        # Restore selection
        if current_name is not None:
            items = self._mask_list.findItems(current_name, Qt.MatchFlag.MatchExactly)
            if items:
                self._mask_list.setCurrentItem(items[0])
        self._mask_list.blockSignals(False)

    def _on_save_mask(self) -> None:
        """Prompt for a name and save the current canvas mask."""
        if self._canvas_ref is None:
            return
        mask = self._canvas_ref.get_mask()
        if mask is None or not mask.any():
            QMessageBox.warning(self, "Save Mask", "No mask to save. Paint a mask first.")
            return
        name, ok = QInputDialog.getText(self, "Save Mask", "Mask name:")
        if not ok or not name.strip():
            return
        self._controller.save_mask(name.strip(), mask)

    def _on_load_mask(self, *_args: Any) -> None:
        """Load the selected mask from the list and apply it to the canvas."""
        item = self._mask_list.currentItem()
        if item is None:
            return
        name = item.text()
        try:
            mask = self._controller.load_mask(name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Load Mask", f"Could not load mask '{name}': {exc}")
            return
        if self._canvas_ref is not None:
            self._canvas_ref.set_mask(mask)
            # Activate mask paint mode so the overlay is visible
            self._canvas_ref.set_mask_paint_active(True)
        self._mask_status_label.setText(f"Loaded mask: {name}")

    def _on_delete_mask(self) -> None:
        """Delete the selected mask after confirmation."""
        item = self._mask_list.currentItem()
        if item is None:
            return
        name = item.text()
        reply = QMessageBox.question(
            self,
            "Delete Mask",
            f"Delete mask '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._controller.delete_mask(name)

    def _on_rename_mask(self) -> None:
        """Rename the selected mask via a text prompt."""
        item = self._mask_list.currentItem()
        if item is None:
            return
        old_name = item.text()
        new_name, ok = QInputDialog.getText(
            self, "Rename Mask", "New name:", text=old_name
        )
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        self._controller.rename_mask(old_name, new_name.strip())

    _MASK_TOOL_MAP: dict[str, str] = {
        "Brush": "brush",
        "Rectangle": "rectangle",
        "Ellipse": "circle",
        "Polygon": "polygon",
        "Pen/Lasso": "pen",
    }

    _SD_TOOL_MAP: dict[str, str] = {
        "Rectangle": "rectangle",
        "Ellipse": "ellipse",
        "Polygon": "polygon",
        "Freehand": "freehand",
        "Line/Polyline": "line",
    }

    def _update_mask_control_states(self) -> None:
        """Sync enabled state of brush size/hardness with AI mode + mask tool."""
        ai_mode_text = self._ai_mask_mode_combo.currentText()
        is_manual = ai_mode_text == "Manual Brush"
        tool_text = self._mask_tool_combo.currentText()
        tool = self._MASK_TOOL_MAP.get(tool_text, "brush")
        brush_only = is_manual and tool == "brush"
        self._brush_size_label.setEnabled(brush_only)
        self._brush_size_spin.setEnabled(brush_only)
        self._brush_hardness_form_label.setEnabled(brush_only)
        self._brush_hardness_slider.setEnabled(brush_only)
        self._brush_hardness_label.setEnabled(brush_only)

    def _on_mask_tool_changed(self, _index: int = 0) -> None:
        """Handle mask tool combo change: update canvas and brush control state."""
        text = self._mask_tool_combo.currentText()
        tool = self._MASK_TOOL_MAP.get(text, "brush")
        if self._canvas_ref is not None:
            self._canvas_ref.set_mask_tool(tool)
        self._update_mask_control_states()

    def _on_apply_mask(self) -> None:
        """Clip the target layer's paths to the painted mask region."""
        if self._canvas_ref is None:
            return
        mask = self._canvas_ref.get_mask()
        if mask is None:
            self._mask_status_label.setText("No mask painted yet.")
            return

        idx = self._mask_target_layer_combo.currentIndex()
        if idx < 0:
            self._mask_status_label.setText("No target layer selected.")
            return
        layer_id = self._mask_target_layer_combo.itemData(idx)

        project = self._controller.current_project
        layer = next((lyr for lyr in project.layers if lyr.id == layer_id), None)
        if layer is None:
            return

        before_count = layer.path_count()
        new_paths = self._clip_paths_to_mask(layer.paths, mask)
        self._controller.set_layer_paths(layer_id, new_paths, "Apply Mask")
        self._mask_status_label.setText(
            f"Applied to '{layer.name}': {len(new_paths)} paths (was {before_count})"
        )

    def _clip_paths_to_mask(self, paths: list, mask) -> list:  # type: ignore[no-untyped-def]
        """Return paths clipped to the painted mask region (mask value > 0.5).

        Each segment between two consecutive vertices is sampled at roughly
        one mask-pixel intervals (~0.2 mm) so that the clip splits at the
        actual mask boundary rather than at the next vertex.  Without this
        a sparse polyline — e.g. a 2-point hatch fill line — would be
        evaluated only at its endpoints; if both endpoints fell outside the
        mask (or both inside an inverted mask), the entire line would be
        dropped (or kept) even when it crossed the boundary in between.

        Segments shorter than 2 points after clipping are discarded.
        """
        return _clip_paths_to_mask(paths, mask)
