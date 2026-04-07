"""_FmmMixin — FMM source point pick handlers."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox


class _FmmMixin:
    """Mixin for FMM (Fast Marching Method) source point pick handlers."""

    def _fmm_btn_alive(self) -> bool:
        """Return True if the FMM pick button still exists as a live Qt object."""
        if self._pick_fmm_source_btn is None:
            return False
        try:
            self._pick_fmm_source_btn.isVisible()  # type: ignore[union-attr]
            return True
        except RuntimeError:
            self._pick_fmm_source_btn = None
            return False

    def _on_pick_fmm_source_clicked(self) -> None:
        """Activate FMM source pick mode on the canvas and update button text."""
        if self._canvas_ref is None:
            return
        self._canvas_ref.set_fmm_source_mode(True)
        if self._fmm_btn_alive():
            self._pick_fmm_source_btn.setText("Click on image…")  # type: ignore[union-attr]

    def _on_fmm_source_point_set(self, x_mm: float, y_mm: float) -> None:
        """Convert canvas mm click to image-relative percentages and update spinboxes."""
        if self._fmm_btn_alive():
            self._pick_fmm_source_btn.setText("Pick on Canvas")  # type: ignore[union-attr]

        # Get the image rect in mm so we can convert to image-space percentages.
        rect_mm = None
        if self._canvas_ref is not None:
            rect_mm = self._canvas_ref.get_image_overlay_rect_mm()

        if rect_mm is None:
            # Fall back to the canvas drawing area
            canvas = self._controller.current_project.canvas
            margin = canvas.margin_mm
            rect_mm = (
                margin,
                margin,
                canvas.width_mm - margin,
                canvas.height_mm - margin,
            )

        ix1, iy1, ix2, iy2 = rect_mm
        w_mm = ix2 - ix1
        h_mm = iy2 - iy1
        if w_mm <= 0 or h_mm <= 0:
            return

        x_pct = max(0.0, min(100.0, (x_mm - ix1) / w_mm * 100.0))
        y_pct = max(0.0, min(100.0, (y_mm - iy1) / h_mm * 100.0))

        # Update the fmm_source_x_pct / fmm_source_y_pct spinboxes.
        x_widget = self._param_widgets.get("fmm_source_x_pct")
        y_widget = self._param_widgets.get("fmm_source_y_pct")
        if isinstance(x_widget, QDoubleSpinBox):
            x_widget.setValue(x_pct)
        if isinstance(y_widget, QDoubleSpinBox):
            y_widget.setValue(y_pct)

        # Update the canvas marker to show where the source point was placed.
        if self._canvas_ref is not None:
            self._canvas_ref.set_fmm_source_marker(x_mm, y_mm)

    def _update_fmm_marker(self) -> None:
        """Sync the FMM source point marker on the canvas from the current spinbox values.

        Called when the user edits the fmm_source_x_pct / fmm_source_y_pct spinboxes
        directly, or when a layer snapshot is applied, so the marker always reflects
        the current parameter state.
        """
        if self._canvas_ref is None:
            return

        # Only show the marker when "Custom" source point is selected.
        source_widget = self._param_widgets.get("fmm_source_point")
        if not (isinstance(source_widget, QComboBox) and source_widget.currentText() == "Custom"):
            return

        x_widget = self._param_widgets.get("fmm_source_x_pct")
        y_widget = self._param_widgets.get("fmm_source_y_pct")
        if not (isinstance(x_widget, QDoubleSpinBox) and isinstance(y_widget, QDoubleSpinBox)):
            return

        x_pct = x_widget.value()
        y_pct = y_widget.value()

        # Convert percentage → mm using the image overlay rect (or drawing area fallback).
        rect_mm = self._canvas_ref.get_image_overlay_rect_mm()
        if rect_mm is None:
            canvas = self._controller.current_project.canvas
            margin = canvas.margin_mm
            rect_mm = (margin, margin, canvas.width_mm - margin, canvas.height_mm - margin)

        ix1, iy1, ix2, iy2 = rect_mm
        w_mm = ix2 - ix1
        h_mm = iy2 - iy1
        if w_mm <= 0 or h_mm <= 0:
            return

        x_mm = ix1 + x_pct / 100.0 * w_mm
        y_mm = iy1 + y_pct / 100.0 * h_mm
        self._canvas_ref.set_fmm_source_marker(x_mm, y_mm)
