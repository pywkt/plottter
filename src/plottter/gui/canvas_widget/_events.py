"""_EventsMixin — mouse, wheel, and keyboard event handlers for CanvasWidget."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

from .enums import MaskTool, ShapeDrawTool


class _EventsMixin:
    """Mixin providing Qt event handler methods for CanvasWidget.

    Must not inherit from QObject.  Designed to be used as a leftmost base:
        class CanvasWidget(_EventsMixin, ..., QWidget): ...
    """

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._3d_preview_active:
            # Scroll to zoom: shrink/grow orbit distance
            delta = event.angleDelta().y()
            factor = 0.9 if delta > 0 else (1.0 / 0.9)
            new_dist = max(0.5, min(50.0, self._3d_cam_distance * factor))
            if new_dist != self._3d_cam_distance:
                self._3d_cam_distance = new_dist
                self.camera_orbit_changed.emit(
                    self._3d_cam_azimuth, self._3d_cam_elevation, self._3d_cam_distance
                )
                self.update()
            return

        # Map positioning mode: wheel zooms the map about the cursor position.
        # Modifier-based screen pan (Ctrl/Alt) falls through to the existing paths.
        if self._map_position_active and self._map_view is not None:
            modifiers = event.modifiers()
            if not (modifiers & Qt.KeyboardModifier.ControlModifier) and \
               not (modifiers & Qt.KeyboardModifier.AltModifier):
                from plottter.osm.geometry import mercator, inverse_mercator, clamp_map_view
                angle_delta = event.angleDelta().y()
                factor = 1.1 if angle_delta > 0 else (1.0 / 1.1)
                scale = self._map_view["scale"]
                new_scale = scale * factor

                # Printable-area centre in mm.
                canvas = self._controller.current_project.canvas
                left, top, right, bottom = canvas.drawing_area()
                ccx = (left + right) / 2
                ccy = (top + bottom) / 2

                # Cursor position in mm (via screen→mm transform).
                cursor_mm_x, cursor_mm_y = self.pixel_to_mm(event.position())

                # Adjust centre so the geographic point under the cursor is preserved.
                # Derivation: x_mm = ccx + (px - mcx)*scale → new_mcx keeps cursor fixed.
                mcx, mcy = mercator(self._map_view["center_lat"], self._map_view["center_lon"])
                inv = 1.0 / scale - 1.0 / new_scale
                new_mcx = mcx + (cursor_mm_x - ccx) * inv
                new_mcy = mcy - (cursor_mm_y - ccy) * inv

                new_lat, new_lon = inverse_mercator(new_mcx, new_mcy)
                new_view = {"center_lat": new_lat, "center_lon": new_lon, "scale": new_scale}
                if self._map_features:
                    new_view = clamp_map_view(new_view, self._map_features, canvas)
                self._map_view = new_view
                self.map_view_changed.emit(
                    new_view["center_lat"], new_view["center_lon"], new_view["scale"]
                )
                self.update()
                return

        # Image positioning mode: wheel zooms the overlay rect about the
        # cursor. Modifier-based screen pan (Ctrl/Alt) falls through.
        if self._image_position_active and self._image_overlay_rect_mm is not None:
            modifiers = event.modifiers()
            if not (modifiers & Qt.KeyboardModifier.ControlModifier) and \
               not (modifiers & Qt.KeyboardModifier.AltModifier):
                angle_delta = event.angleDelta().y()
                factor = 1.1 if angle_delta > 0 else (1.0 / 1.1)
                x1, y1, x2, y2 = self._image_overlay_rect_mm
                cursor_mm_x, cursor_mm_y = self.pixel_to_mm(event.position())
                # Scale the rect about the cursor point so the mm point under
                # the cursor stays fixed: new_x = cursor + (old - cursor) * f
                new_x1 = cursor_mm_x + (x1 - cursor_mm_x) * factor
                new_y1 = cursor_mm_y + (y1 - cursor_mm_y) * factor
                new_x2 = cursor_mm_x + (x2 - cursor_mm_x) * factor
                new_y2 = cursor_mm_y + (y2 - cursor_mm_y) * factor
                # Clamp to a sensible minimum size so the user can't zoom away.
                if (new_x2 - new_x1) >= 1.0 and (new_y2 - new_y1) >= 1.0:
                    self._image_overlay_rect_mm = (new_x1, new_y1, new_x2, new_y2)
                    self.image_view_changed.emit(new_x1, new_y1, new_x2, new_y2)
                    self.update()
                return

        # Modifier-based pan must be checked BEFORE falling through to zoom.
        # Use angleDelta exclusively — pixelDelta-based heuristics caused
        # simultaneous zoom+pan on mice that emit both deltas per click.
        modifiers = event.modifiers()
        angle_delta = event.angleDelta().y()

        # Ctrl+wheel: pan vertically
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._pan_offset += QPointF(0.0, angle_delta / 120.0 * 40)
            self._clamp_pan_offset()
            self.update()
            return

        # Alt+wheel: pan horizontally
        if modifiers & Qt.KeyboardModifier.AltModifier:
            self._pan_offset += QPointF(angle_delta / 120.0 * 40, 0.0)
            self._clamp_pan_offset()
            self.update()
            return

        # Default: zoom centered on cursor
        factor = 1.15 if angle_delta > 0 else (1.0 / 1.15)
        center = event.position()
        self._apply_zoom(factor, center)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # ── 3D preview mode ──────────────────────────────────────────────
        if self._3d_preview_active:
            if event.button() == Qt.MouseButton.LeftButton:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    # Shift+Left → pan look-at point
                    self._3d_pan_drag_start = event.pos()
                    self._3d_pan_start_lookat = self._3d_cam_lookat
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    # Left → orbit
                    self._3d_orbit_drag_start = event.pos()
                    self._3d_orbit_start_az = self._3d_cam_azimuth
                    self._3d_orbit_start_el = self._3d_cam_elevation
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif event.button() == Qt.MouseButton.MiddleButton:
                self._3d_pan_drag_start = event.pos()
                self._3d_pan_start_lookat = self._3d_cam_lookat
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif event.button() == Qt.MouseButton.RightButton:
                self._show_3d_context_menu(event.pos())
            return

        # ── Map positioning mode: left drag pans the map ─────────────────
        if self._map_position_active:
            if event.button() == Qt.MouseButton.LeftButton and self._map_view is not None:
                from plottter.osm.geometry import mercator
                self._map_pan_drag_start = event.pos()
                mcx, mcy = mercator(
                    self._map_view["center_lat"], self._map_view["center_lon"]
                )
                self._map_pan_start_merc = (mcx, mcy)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # ── Image positioning mode: left drag pans the image overlay ─────
        if self._image_position_active:
            if (
                event.button() == Qt.MouseButton.LeftButton
                and self._image_overlay_rect_mm is not None
            ):
                self._image_pan_drag_start = event.pos()
                self._image_pan_start_rect = self._image_overlay_rect_mm
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # AI mask point mode: left = positive point, right = negative point
        if self._ai_mask_mode == "point":
            pos_mm = self.pixel_to_mm(QPointF(event.pos()))
            if event.button() == Qt.MouseButton.LeftButton:
                self._ai_mask_positive_points.append(pos_mm)
                self.ai_mask_point_selected.emit(pos_mm[0], pos_mm[1], True)
                self.update()
            elif event.button() == Qt.MouseButton.RightButton:
                self._ai_mask_negative_points.append(pos_mm)
                self.ai_mask_point_selected.emit(pos_mm[0], pos_mm[1], False)
                self.update()
            return

        # AI mask box mode: left drag to draw bounding box
        if self._ai_mask_mode == "box":
            if event.button() == Qt.MouseButton.LeftButton:
                pos_mm = self.pixel_to_mm(QPointF(event.pos()))
                self._ai_box_start = pos_mm
                self._ai_box_end = None
                self.update()
            return

        # Drag-to-move mode: left drag translates the active layer's paths
        if self._drag_move_active and event.button() == Qt.MouseButton.LeftButton:
            active_id = self._controller.active_layer_id
            layer = self._controller.get_layer(active_id) if active_id else None
            if layer is not None and not layer.locked and layer.paths:
                pos_mm = self.pixel_to_mm(QPointF(event.pos()))
                self._drag_move_start_mm = pos_mm
                self._drag_move_offset_mm = (0.0, 0.0)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # FMM source point pick mode: single left-click sets the source position
        if self._fmm_source_mode and event.button() == Qt.MouseButton.LeftButton:
            pos_mm = self.pixel_to_mm(QPointF(event.pos()))
            self._fmm_source_mode = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.fmm_source_point_set.emit(pos_mm[0], pos_mm[1])
            return

        # Shape drawing mode
        if self._shape_draw_active and event.button() == Qt.MouseButton.LeftButton:
            pos_mm = self.pixel_to_mm(QPointF(event.pos()))
            if self._shape_draw_tool in (ShapeDrawTool.RECTANGLE, ShapeDrawTool.ELLIPSE):
                self._sd_start_mm = pos_mm
                self._sd_end_mm = pos_mm
            elif self._shape_draw_tool == ShapeDrawTool.POLYGON:
                if not self._sd_polygon_vertices:
                    self._sd_polygon_vertices = [pos_mm]
                else:
                    self._sd_polygon_vertices.append(pos_mm)
            elif self._shape_draw_tool == ShapeDrawTool.FREEHAND:
                self._sd_pen_points = [pos_mm]
            elif self._shape_draw_tool == ShapeDrawTool.LINE:
                if not self._sd_line_vertices:
                    self._sd_line_vertices = [pos_mm]
                else:
                    self._sd_line_vertices.append(pos_mm)
            self.update()
            return
        if self._shape_draw_active and event.button() == Qt.MouseButton.RightButton:
            # Right-click cancels in-progress polygon or line
            if self._shape_draw_tool == ShapeDrawTool.POLYGON and self._sd_polygon_vertices:
                self._sd_polygon_vertices.clear()
                self._sd_polygon_cursor_mm = None
            elif self._shape_draw_tool == ShapeDrawTool.LINE and self._sd_line_vertices:
                self._sd_line_vertices.clear()
                self._sd_line_cursor_mm = None
            self.update()
            return

        if self._mask_paint_active and event.button() == Qt.MouseButton.LeftButton:
            pos_mm = self.pixel_to_mm(QPointF(event.pos()))
            if self._mask_tool == MaskTool.BRUSH:
                self._pre_op_mask = self._snapshot_mask()
                self._paint_at(*pos_mm)
                self._last_brush_pos = pos_mm
            elif self._mask_tool in (MaskTool.RECTANGLE, MaskTool.CIRCLE):
                self._pre_op_mask = self._snapshot_mask()
                self._shape_start_mm = pos_mm
                self._shape_end_mm = pos_mm
            elif self._mask_tool == MaskTool.POLYGON:
                self._handle_polygon_press(pos_mm)
            elif self._mask_tool == MaskTool.PEN:
                self._pre_op_mask = self._snapshot_mask()
                self._pen_points = [pos_mm]
            return
        if self._mask_paint_active and event.button() == Qt.MouseButton.RightButton:
            if self._mask_tool == MaskTool.POLYGON and self._polygon_vertices:
                # Right-click cancels in-progress polygon
                self._polygon_vertices.clear()
                self._pre_op_mask = None
                self.update()
            return
        if (
            self._space_held
            and event.button() == Qt.MouseButton.LeftButton
            and not self._3d_preview_active
        ):
            self._last_pan_pos = event.pos()
            self._hand_pan_active = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # ── 3D preview mode: orbit and pan ──────────────────────────────
        if self._3d_preview_active:
            if self._3d_orbit_drag_start is not None:
                delta = event.pos() - self._3d_orbit_drag_start
                sensitivity = 0.3  # degrees per pixel
                az = (self._3d_orbit_start_az - delta.x() * sensitivity) % 360.0
                el = max(-89.9, min(89.9, self._3d_orbit_start_el + delta.y() * sensitivity))
                self._3d_cam_azimuth = az
                self._3d_cam_elevation = el
                self.camera_orbit_changed.emit(az, el, self._3d_cam_distance)
                self.update()
            elif self._3d_pan_drag_start is not None:
                delta = event.pos() - self._3d_pan_drag_start
                # Pan sensitivity scales with orbit distance
                pan_scale = self._3d_cam_distance * 0.003
                # Compute camera right vector from azimuth (horizontal pan)
                az_rad = math.radians(self._3d_cam_azimuth)
                right_x = -math.cos(az_rad)
                right_z = math.sin(az_rad)
                lx = self._3d_pan_start_lookat[0] + delta.x() * pan_scale * right_x
                ly = self._3d_pan_start_lookat[1] - delta.y() * pan_scale
                lz = self._3d_pan_start_lookat[2] + delta.x() * pan_scale * right_z
                self._3d_cam_lookat = (lx, ly, lz)
                self.camera_pan_changed.emit(lx, ly, lz)
                self.update()
            return

        # ── Map positioning mode: left-drag pans the map ─────────────────
        if self._map_position_active:
            if (
                self._map_pan_drag_start is not None
                and self._map_pan_start_merc is not None
                and self._map_view is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                from plottter.osm.geometry import inverse_mercator, clamp_map_view
                delta = event.pos() - self._map_pan_drag_start
                scale = self._map_view["scale"]
                # Pixel delta → Mercator delta.
                # Dragging right → centre moves west (−mcx); dragging down → north (+mcy).
                # Derivation: view_transform places mcx/mcy at canvas centre (ccx/ccy);
                # x_mm = ccx + (px − mcx)·scale  →  new_mcx = start_mcx − Δpx/(zoom·scale)
                # y_mm = ccy − (py − mcy)·scale  →  new_mcy = start_mcy + Δpy/(zoom·scale)
                dx_merc = -delta.x() / (self._zoom * scale)
                dy_merc = delta.y() / (self._zoom * scale)
                new_mcx = self._map_pan_start_merc[0] + dx_merc
                new_mcy = self._map_pan_start_merc[1] + dy_merc
                new_lat, new_lon = inverse_mercator(new_mcx, new_mcy)
                new_view = {"center_lat": new_lat, "center_lon": new_lon, "scale": scale}
                if self._map_features:
                    canvas = self._controller.current_project.canvas
                    new_view = clamp_map_view(new_view, self._map_features, canvas)
                self._map_view = new_view
                self.map_view_changed.emit(
                    new_view["center_lat"], new_view["center_lon"], new_view["scale"]
                )
                self.update()
            x_mm, y_mm = self.pixel_to_mm(QPointF(event.pos()))
            self.mouse_position_mm.emit(x_mm, y_mm)
            return

        # ── Image positioning mode: left-drag pans the overlay rect ──────
        if self._image_position_active:
            if (
                self._image_pan_drag_start is not None
                and self._image_pan_start_rect is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                delta = event.pos() - self._image_pan_drag_start
                # Pixel delta → mm delta. self._zoom mm = 1 pixel.
                dx_mm = delta.x() / self._zoom
                dy_mm = delta.y() / self._zoom
                sx1, sy1, sx2, sy2 = self._image_pan_start_rect
                new_rect = (sx1 + dx_mm, sy1 + dy_mm, sx2 + dx_mm, sy2 + dy_mm)
                self._image_overlay_rect_mm = new_rect
                self.image_view_changed.emit(*new_rect)
                self.update()
            x_mm, y_mm = self.pixel_to_mm(QPointF(event.pos()))
            self.mouse_position_mm.emit(x_mm, y_mm)
            return

        # FMM source pick mode: update live crosshair preview on mouse move
        if self._fmm_source_mode:
            self._fmm_cursor_preview_mm = self.pixel_to_mm(QPointF(event.pos()))
            self.update()
            x_mm, y_mm = self._fmm_cursor_preview_mm
            self.mouse_position_mm.emit(x_mm, y_mm)
            return

        # AI mask box: track rubber-band while dragging
        if self._ai_mask_mode == "box" and self._ai_box_start is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._ai_box_end = self.pixel_to_mm(QPointF(event.pos()))
                self.update()
            x_mm, y_mm = self.pixel_to_mm(QPointF(event.pos()))
            self.mouse_position_mm.emit(x_mm, y_mm)
            return

        # Shape drawing — update rubber-band / freehand tracking
        if self._shape_draw_active:
            pos_mm = self.pixel_to_mm(QPointF(event.pos()))
            if self._shape_draw_tool in (ShapeDrawTool.RECTANGLE, ShapeDrawTool.ELLIPSE):
                if self._sd_start_mm is not None and (event.buttons() & Qt.MouseButton.LeftButton):
                    end_mm = pos_mm
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        dx = end_mm[0] - self._sd_start_mm[0]
                        dy = end_mm[1] - self._sd_start_mm[1]
                        d = min(abs(dx), abs(dy))
                        end_mm = (
                            self._sd_start_mm[0] + (d if dx >= 0 else -d),
                            self._sd_start_mm[1] + (d if dy >= 0 else -d),
                        )
                    self._sd_end_mm = end_mm
            elif self._shape_draw_tool == ShapeDrawTool.POLYGON:
                self._sd_polygon_cursor_mm = pos_mm
            elif self._shape_draw_tool == ShapeDrawTool.FREEHAND:
                if self._sd_pen_points and (event.buttons() & Qt.MouseButton.LeftButton):
                    self._sd_pen_points.append(pos_mm)
            elif self._shape_draw_tool == ShapeDrawTool.LINE:
                self._sd_line_cursor_mm = pos_mm
            self.update()
            x_mm, y_mm = pos_mm
            self.mouse_position_mm.emit(x_mm, y_mm)
            return

        # Mask paint — all tools
        if self._mask_paint_active:
            pos_mm = self.pixel_to_mm(QPointF(event.pos()))
            if self._mask_tool == MaskTool.BRUSH:
                if self._last_brush_pos is not None:
                    self._interpolate_stroke(self._last_brush_pos, pos_mm)
                    self._last_brush_pos = pos_mm
            elif self._mask_tool in (MaskTool.RECTANGLE, MaskTool.CIRCLE):
                if self._shape_start_mm is not None and (event.buttons() & Qt.MouseButton.LeftButton):
                    end_mm = pos_mm
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        dx = end_mm[0] - self._shape_start_mm[0]
                        dy = end_mm[1] - self._shape_start_mm[1]
                        d = min(abs(dx), abs(dy))
                        end_mm = (
                            self._shape_start_mm[0] + (d if dx >= 0 else -d),
                            self._shape_start_mm[1] + (d if dy >= 0 else -d),
                        )
                    self._shape_end_mm = end_mm
            elif self._mask_tool == MaskTool.POLYGON:
                self._polygon_cursor_mm = pos_mm
            elif self._mask_tool == MaskTool.PEN:
                if self._pen_points and (event.buttons() & Qt.MouseButton.LeftButton):
                    self._pen_points.append(pos_mm)
            self._brush_cursor_pos = (float(event.pos().x()), float(event.pos().y()))
            self.update()
            x_mm, y_mm = pos_mm
            self.mouse_position_mm.emit(x_mm, y_mm)
            return

        # Drag-to-move: update live preview offset
        if self._drag_move_active and self._drag_move_start_mm is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                cur_mm = self.pixel_to_mm(QPointF(event.pos()))
                self._drag_move_offset_mm = (
                    cur_mm[0] - self._drag_move_start_mm[0],
                    cur_mm[1] - self._drag_move_start_mm[1],
                )
                self.update()
            x_mm, y_mm = self.pixel_to_mm(QPointF(event.pos()))
            self.mouse_position_mm.emit(x_mm, y_mm)
            return

        # Pan
        if self._last_pan_pos is not None:
            delta = event.pos() - self._last_pan_pos
            self._pan_offset += QPointF(delta.x(), delta.y())
            self._last_pan_pos = event.pos()
            self._clamp_pan_offset()
            self.update()

        # Emit mm position for status bar
        x_mm, y_mm = self.pixel_to_mm(QPointF(event.pos()))
        self.mouse_position_mm.emit(x_mm, y_mm)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # ── 3D preview mode ──────────────────────────────────────────────
        if self._3d_preview_active:
            if event.button() in (
                Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton
            ):
                self._3d_orbit_drag_start = None
                self._3d_pan_drag_start = None
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

        # ── Map positioning mode: end pan drag ───────────────────────────
        if self._map_position_active:
            if event.button() == Qt.MouseButton.LeftButton:
                if self._map_pan_drag_start is not None:
                    self._map_pan_drag_start = None
                    self._map_pan_start_merc = None
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

        # ── Image positioning mode: end pan drag ─────────────────────────
        if self._image_position_active:
            if event.button() == Qt.MouseButton.LeftButton:
                if self._image_pan_drag_start is not None:
                    self._image_pan_drag_start = None
                    self._image_pan_start_rect = None
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return

        # Drag-to-move: finalize on left-button release
        if self._drag_move_active and event.button() == Qt.MouseButton.LeftButton:
            if self._drag_move_start_mm is not None:
                dx, dy = self._drag_move_offset_mm
                self._drag_move_start_mm = None
                self._drag_move_offset_mm = (0.0, 0.0)
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    self.layer_move_finished.emit(dx, dy)
                self.update()
            return

        # AI mask box: complete the box on left-button release
        if self._ai_mask_mode == "box" and event.button() == Qt.MouseButton.LeftButton:
            if self._ai_box_start is not None:
                self._ai_box_end = self.pixel_to_mm(QPointF(event.pos()))
                box = self.get_ai_mask_box()
                if box is not None:
                    self.ai_mask_box_drawn.emit(*box)
                self.update()
            return

        if self._mask_paint_active and event.button() == Qt.MouseButton.LeftButton:
            if self._mask_tool == MaskTool.BRUSH:
                if self._last_brush_pos is not None:
                    after = self._snapshot_mask()
                    self._last_brush_pos = None
                    self.mask_stroke_done.emit(self._mask_array)
                    self.mask_op_done.emit(self._pre_op_mask, after)
                    self._pre_op_mask = None
            elif self._mask_tool in (MaskTool.RECTANGLE, MaskTool.CIRCLE):
                if self._shape_start_mm is not None and self._shape_end_mm is not None:
                    sx, sy = self._shape_start_mm
                    ex, ey = self._shape_end_mm
                    # Skip zero-size clicks to avoid spurious undo entries.
                    from .enums import _MASK_PX_PER_MM
                    px_w = abs(int(round(ex * _MASK_PX_PER_MM)) - int(round(sx * _MASK_PX_PER_MM)))
                    px_h = abs(int(round(ey * _MASK_PX_PER_MM)) - int(round(sy * _MASK_PX_PER_MM)))
                    if px_w > 0 or px_h > 0:
                        if self._mask_tool == MaskTool.RECTANGLE:
                            self._apply_rectangle_mask()
                        else:
                            self._apply_ellipse_mask()
                        after = self._snapshot_mask()
                        self.mask_op_done.emit(self._pre_op_mask, after)
                    self._pre_op_mask = None
                self._shape_start_mm = None
                self._shape_end_mm = None
            elif self._mask_tool == MaskTool.PEN:
                if len(self._pen_points) >= 3:
                    self._apply_pen_mask()
                    after = self._snapshot_mask()
                    self.mask_op_done.emit(self._pre_op_mask, after)
                    self._pre_op_mask = None
                self._pen_points.clear()
            # Polygon is completed by double-click or Enter key
            self.update()
            return
        # Shape drawing mode: complete the shape on mouse release (rect/ellipse/freehand)
        if self._shape_draw_active and event.button() == Qt.MouseButton.LeftButton:
            polyline: list[tuple[float, float]] | None = None
            if self._shape_draw_tool in (ShapeDrawTool.RECTANGLE, ShapeDrawTool.ELLIPSE):
                if self._sd_start_mm is not None and self._sd_end_mm is not None:
                    sx, sy = self._sd_start_mm
                    ex, ey = self._sd_end_mm
                    dx = abs(ex - sx)
                    dy = abs(ey - sy)
                    # Ignore degenerate zero-size clicks
                    if dx > 0.01 or dy > 0.01:
                        if self._shape_draw_tool == ShapeDrawTool.RECTANGLE:
                            polyline = self._sd_make_rectangle_polyline(sx, sy, ex, ey)
                        else:
                            polyline = self._sd_make_ellipse_polyline(sx, sy, ex, ey)
                self._sd_start_mm = None
                self._sd_end_mm = None
                self.update()
            elif self._shape_draw_tool == ShapeDrawTool.FREEHAND:
                if len(self._sd_pen_points) >= 3:
                    # Auto-close freehand: append first point
                    pts = list(self._sd_pen_points)
                    if pts[-1] != pts[0]:
                        pts.append(pts[0])
                    polyline = pts
                self._sd_pen_points.clear()
                self.update()
            if polyline is not None:
                self.shape_drawn.emit(polyline)
            return

        if event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ) and self._last_pan_pos is not None:
            self._last_pan_pos = None
            if self._hand_pan_active:
                self._hand_pan_active = False
                self.setCursor(
                    Qt.CursorShape.OpenHandCursor if self._space_held else Qt.CursorShape.ArrowCursor
                )
            else:
                self.setCursor(Qt.CursorShape.CrossCursor if (self._mask_paint_active or self._shape_draw_active) else Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click in Polygon mode: close and fill the polygon."""
        # Shape drawing: double-click completes polygon (closed) or line (open)
        if self._shape_draw_active and event.button() == Qt.MouseButton.LeftButton:
            if self._shape_draw_tool == ShapeDrawTool.POLYGON:
                # Qt fires press → release → doubleClick, so the double-click press
                # already added a duplicate vertex — remove it
                if len(self._sd_polygon_vertices) > 3:
                    self._sd_polygon_vertices.pop()
                if len(self._sd_polygon_vertices) >= 3:
                    pts = list(self._sd_polygon_vertices)
                    if pts[-1] != pts[0]:
                        pts.append(pts[0])  # close
                    self.shape_drawn.emit(pts)
                self._sd_polygon_vertices.clear()
                self._sd_polygon_cursor_mm = None
                self.update()
                return
            elif self._shape_draw_tool == ShapeDrawTool.LINE:
                # Double-click finishes the line (open polyline, no close)
                if len(self._sd_line_vertices) > 1:
                    self._sd_line_vertices.pop()  # remove duplicate from double-click press
                if len(self._sd_line_vertices) >= 2:
                    self.shape_drawn.emit(list(self._sd_line_vertices))
                self._sd_line_vertices.clear()
                self._sd_line_cursor_mm = None
                self.update()
                return

        if (
            self._mask_paint_active
            and self._mask_tool == MaskTool.POLYGON
            and event.button() == Qt.MouseButton.LeftButton
        ):
            # Qt fires press → release → doubleClick → release, so the first click
            # of the double-click already added a vertex. Remove that duplicate.
            if len(self._polygon_vertices) > 3:
                self._polygon_vertices.pop()
            if len(self._polygon_vertices) >= 3:
                self._apply_polygon_mask()
                after = self._snapshot_mask()
                self.mask_op_done.emit(self._pre_op_mask, after)
                self._pre_op_mask = None
                self._polygon_vertices.clear()
                self._polygon_cursor_mm = None
                self.update()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard shortcuts for shape tools."""
        # Shape drawing keyboard shortcuts
        if self._shape_draw_active:
            if self._shape_draw_tool == ShapeDrawTool.POLYGON:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    if len(self._sd_polygon_vertices) >= 3:
                        pts = list(self._sd_polygon_vertices)
                        if pts[-1] != pts[0]:
                            pts.append(pts[0])
                        self.shape_drawn.emit(pts)
                    self._sd_polygon_vertices.clear()
                    self._sd_polygon_cursor_mm = None
                    self.update()
                    return
                elif event.key() == Qt.Key.Key_Escape:
                    self._sd_polygon_vertices.clear()
                    self._sd_polygon_cursor_mm = None
                    self.update()
                    return
            elif self._shape_draw_tool == ShapeDrawTool.LINE:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    if len(self._sd_line_vertices) >= 2:
                        self.shape_drawn.emit(list(self._sd_line_vertices))
                    self._sd_line_vertices.clear()
                    self._sd_line_cursor_mm = None
                    self.update()
                    return
                elif event.key() == Qt.Key.Key_Escape:
                    self._sd_line_vertices.clear()
                    self._sd_line_cursor_mm = None
                    self.update()
                    return

        if self._mask_paint_active and self._mask_tool == MaskTool.POLYGON:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if len(self._polygon_vertices) >= 3:
                    self._apply_polygon_mask()
                    after = self._snapshot_mask()
                    self.mask_op_done.emit(self._pre_op_mask, after)
                    self._pre_op_mask = None
                    self._polygon_vertices.clear()
                    self._polygon_cursor_mm = None
                    self.update()
                return
            elif event.key() == Qt.Key.Key_Escape:
                self._polygon_vertices.clear()
                self._polygon_cursor_mm = None
                self._pre_op_mask = None
                self.update()
                return

        # Arrow-key panning — skip in 3D preview mode (camera controls handle arrows there).
        # Canvas receives keyboard focus via StrongFocus (set in __init__); clicking the
        # canvas focuses it so subsequent arrow presses work without re-clicking.
        if not self._3d_preview_active and event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            step = 160.0 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 40.0
            key = event.key()
            if key == Qt.Key.Key_Left:
                self._pan_offset += QPointF(step, 0.0)
            elif key == Qt.Key.Key_Right:
                self._pan_offset += QPointF(-step, 0.0)
            elif key == Qt.Key.Key_Up:
                self._pan_offset += QPointF(0.0, step)
            elif key == Qt.Key.Key_Down:
                self._pan_offset += QPointF(0.0, -step)
            self._clamp_pan_offset()
            self.update()
            return

        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            _exclusive = (
                self._3d_preview_active
                or self._mask_paint_active
                or self._shape_draw_active
                or self._drag_move_active
                or self._fmm_source_mode
                or self._ai_mask_mode
            )
            if not _exclusive:
                self._space_held = True
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._hand_pan_active = False
            _exclusive = (
                self._3d_preview_active
                or self._mask_paint_active
                or self._shape_draw_active
                or self._drag_move_active
                or self._fmm_source_mode
                or self._ai_mask_mode
            )
            if not _exclusive:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:
        if self._space_held:
            self._space_held = False
            self._hand_pan_active = False
            self._last_pan_pos = None
            _exclusive = (
                self._mask_paint_active
                or self._shape_draw_active
                or self._drag_move_active
                or self._3d_preview_active
            )
            if not _exclusive:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().focusOutEvent(event)
