"""_ShapeDrawMixin — shape drawing tool handlers."""

from __future__ import annotations


class _ShapeDrawMixin:
    """Mixin for shape drawing tool handlers."""

    def _on_sd_fill_changed(self, _index: int = 0) -> None:
        """Show/hide fill spacing and angle controls based on the selected fill type."""
        fill_text = self._sd_fill_combo.currentText()
        has_fill = fill_text != "None"
        has_angle = fill_text in ("Hatching", "Cross-hatch")
        self._sd_fill_spacing_label.setVisible(has_fill)
        self._sd_fill_spacing_spin.setVisible(has_fill)
        self._sd_fill_angle_label.setVisible(has_angle)
        self._sd_fill_angle_spin.setVisible(has_angle)

    def _on_sd_tool_changed(self, _index: int = 0) -> None:
        """Handle shape drawing tool combo change: update canvas tool."""
        if self._canvas_ref is None or self._current_mode != "Shape Drawing":
            return
        tool_text = self._sd_tool_combo.currentText()
        tool = self._SD_TOOL_MAP.get(tool_text, "rectangle")
        self._canvas_ref.set_shape_draw_tool(tool)

    def _on_shape_drawn(self, polyline: list) -> None:
        """Handle a completed shape from the canvas.

        Applies Chaikin smoothing (if requested), generates fill polylines
        based on the selected fill type, and appends all resulting polylines
        to the target layer (not replacing existing paths).
        """
        if not polyline or len(polyline) < 2:
            return

        # Apply Chaikin smoothing to the shape outline
        smooth_passes = self._sd_smooth_spin.value()
        if smooth_passes > 0:
            try:
                from plottter.generators.contour import _chaikin_smooth
                is_closed = polyline[0] == polyline[-1]
                polyline = _chaikin_smooth(list(polyline), smooth_passes, closed=is_closed)
                # Re-close if it was closed and smoothing opened it
                if is_closed and len(polyline) >= 2 and polyline[0] != polyline[-1]:
                    polyline.append(polyline[0])
            except Exception:
                pass  # smoothing is best-effort

        new_paths: list = []

        # Stroke (outline) polyline
        if self._sd_stroke_check.isChecked():
            new_paths.append(list(polyline))

        # Fill polylines (only meaningful for closed shapes)
        fill_text = self._sd_fill_combo.currentText()
        is_closed_shape = len(polyline) >= 3 and polyline[0] == polyline[-1]

        if fill_text != "None" and is_closed_shape:
            spacing = self._sd_fill_spacing_spin.value()
            angle = self._sd_fill_angle_spin.value()
            try:
                from plottter.generators.contour import (
                    _fill_polygon_hatch,
                    _fill_polygon_concentric,
                )
                from shapely.validation import make_valid
                from shapely.geometry import Polygon

                # Normalize the polygon via make_valid to handle self-intersections
                raw_poly = Polygon(polyline)
                valid_geom = make_valid(raw_poly)

                def _polygons_from_geom(geom) -> list:
                    """Extract individual Polygon objects from any Shapely geometry."""
                    if geom.geom_type == "Polygon":
                        return [geom]
                    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
                        return [g for g in geom.geoms if g.geom_type == "Polygon"]
                    return []

                for poly in _polygons_from_geom(valid_geom):
                    outer_pts: list = list(poly.exterior.coords)
                    hole_pts_list: list = [list(h.coords) for h in poly.interiors]

                    if fill_text == "Hatching":
                        fill_lines = _fill_polygon_hatch(outer_pts, hole_pts_list, angle, spacing)
                        new_paths.extend(fill_lines)
                    elif fill_text == "Cross-hatch":
                        fill_lines = _fill_polygon_hatch(outer_pts, hole_pts_list, angle, spacing)
                        fill_lines2 = _fill_polygon_hatch(outer_pts, hole_pts_list, (angle + 90.0) % 180.0, spacing)
                        new_paths.extend(fill_lines)
                        new_paths.extend(fill_lines2)
                    elif fill_text == "Concentric":
                        fill_rings = _fill_polygon_concentric(outer_pts, hole_pts_list, spacing)
                        new_paths.extend(fill_rings)
            except Exception:
                pass  # fill is best-effort; always at least keep the stroke

        if not new_paths:
            return

        # Get the target layer
        layer_id = self._sd_target_layer_combo.currentData()
        if not layer_id:
            return

        self._controller.add_paths_to_layer(layer_id, new_paths, "Draw Shape")
