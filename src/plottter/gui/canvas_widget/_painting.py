"""_PaintingMixin — paintEvent and all _draw_* helpers for CanvasWidget."""
from __future__ import annotations

import math
import random

import numpy as np

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
)

from .enums import MaskTool, ShapeDrawTool


class _PaintingMixin:
    """Mixin providing paintEvent and all drawing helpers for CanvasWidget.

    Must not inherit from QObject.
    """

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#808080"))

        project = self._controller.current_project
        canvas = project.canvas
        w_mm = canvas.width_mm
        h_mm = canvas.height_mm

        # Paper boundary — force pure white in Ink Preview so the multiply
        # blend math (paper × ink₁ × ink₂ × …) gives true subtractive colour.
        paper_tl = self.mm_to_pixel((0.0, 0.0))
        paper_br = self.mm_to_pixel((w_mm, h_mm))
        paper_rect = QRectF(paper_tl, paper_br)
        if self._ink_preview:
            paper_color = QColor("white")
        else:
            paper_color = QColor("#FAFAFA") if self._show_paper_texture else QColor("white")
        painter.fillRect(paper_rect, paper_color)
        pen = QPen(QColor("black"), 1.0)
        painter.setPen(pen)
        painter.drawRect(paper_rect)

        # Margin boundary (dashed gray)
        margin = canvas.margin_mm
        margin_tl = self.mm_to_pixel((margin, margin))
        margin_br = self.mm_to_pixel((w_mm - margin, h_mm - margin))
        margin_rect = QRectF(margin_tl, margin_br)
        dash_pen = QPen(QColor("#AAAAAA"), 0.5, Qt.PenStyle.DashLine)
        painter.setPen(dash_pen)
        painter.drawRect(margin_rect)

        # Image overlay (semi-transparent, behind paths, within drawing area to
        # match where generators produce output)
        if self._show_image_overlay and self._image_overlay is not None:
            if self._image_overlay_rect_mm is not None:
                ix1, iy1, ix2, iy2 = self._image_overlay_rect_mm
                overlay_tl = self.mm_to_pixel((ix1, iy1))
                overlay_br = self.mm_to_pixel((ix2, iy2))
                overlay_rect = QRectF(overlay_tl, overlay_br)
            else:
                overlay_rect = margin_rect
            painter.setOpacity(0.4)
            painter.drawPixmap(
                overlay_rect,
                self._image_overlay,
                QRectF(self._image_overlay.rect()),
            )
            painter.setOpacity(1.0)

        # Mask paint overlay (above paper, below paths)
        if self._mask_paint_active and self._mask_array is not None:
            self._draw_mask_overlay(painter, canvas)

        # Grid overlay
        if self._show_grid:
            self._draw_grid(painter, canvas)

        # Registration marks
        if self._show_reg_marks and project.registration_marks:
            self._draw_registration_marks(painter, canvas, project.reg_mark_style)

        # 3D preview mode: overlay dark viewport and render wireframe
        if self._3d_preview_active:
            self._draw_3d_preview(painter, canvas)
        else:
            # Paths.  In Ink Preview mode, switch to multiply blending so
            # stacked layers combine like real ink on paper (cyan + yellow =
            # green); restore the default mode afterwards so overlays
            # (travel lines, registration marks, brush cursor, etc.) draw
            # normally on top.
            if self._ink_preview:
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
            try:
                if self._anim_mode:
                    self._draw_animated_paths(painter)
                else:
                    active_id = self._controller.active_layer_id
                    for layer in project.layers:
                        if not layer.visible:
                            continue
                        if (
                            self._drag_move_active
                            and self._drag_move_start_mm is not None
                            and layer.id == active_id
                        ):
                            self._draw_layer(painter, layer, offset=self._drag_move_offset_mm)
                        else:
                            self._draw_layer(painter, layer)
            finally:
                if self._ink_preview:
                    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Pen-up travel visualization (normal mode only)
            if self._show_travel:
                self._draw_travel_lines(painter, project)

        # Map positioning preview overlay (faded lines + data-bounds outline)
        if self._map_position_active:
            self._draw_map_preview(painter, canvas)

        # AI mask overlays (points and bounding box) — shown whenever prompts exist
        if (
            self._ai_mask_mode is not None
            or self._ai_mask_positive_points
            or self._ai_mask_negative_points
            or (self._ai_box_start is not None and self._ai_box_end is not None)
        ):
            self._draw_ai_mask_overlays(painter)

        # FMM source point marker (persistent, shown after pick or manual spinbox edit)
        if self._fmm_source_marker_mm is not None:
            self._draw_fmm_source_marker(painter, *self._fmm_source_marker_mm)

        # FMM live cursor preview (shown while pick mode is active)
        if self._fmm_source_mode and self._fmm_cursor_preview_mm is not None:
            self._draw_fmm_source_marker(painter, *self._fmm_cursor_preview_mm, is_preview=True)

        # Shape tool rubber-band / in-progress feedback
        if self._mask_paint_active:
            self._draw_shape_feedback(painter)

        # Shape draw in-progress feedback
        if self._shape_draw_active:
            self._draw_shape_draw_feedback(painter)

        # Brush cursor ring — only shown for the Brush tool
        if (
            self._mask_paint_active
            and self._mask_tool == MaskTool.BRUSH
            and self._brush_cursor_pos is not None
        ):
            self._draw_brush_cursor(painter, *self._brush_cursor_pos)

    # ------------------------------------------------------------------
    # 3D preview helpers
    # ------------------------------------------------------------------

    def _draw_3d_preview(self, painter: QPainter, canvas) -> None:  # type: ignore[no-untyped-def]
        """Draw a dark viewport overlay with wireframe polylines in 3D preview mode."""
        # Dark background over the paper area
        paper_tl = self.mm_to_pixel((0.0, 0.0))
        paper_br = self.mm_to_pixel((canvas.width_mm, canvas.height_mm))
        paper_rect = QRectF(paper_tl, paper_br)
        painter.fillRect(paper_rect, QColor("#1A1A2E"))

        # Wireframe lines — bright cyan on dark background
        wire_pen = QPen(QColor("#00E5FF"), max(0.5, self._zoom * 0.25))
        painter.setPen(wire_pen)

        vp_left, vp_top = self.pixel_to_mm(QPointF(0.0, 0.0))
        vp_right, vp_bottom = self.pixel_to_mm(QPointF(float(self.width()), float(self.height())))

        for polyline in self._3d_wireframe_polylines:
            if len(polyline) < 2:
                continue
            # Viewport culling
            min_x = min_y = float("inf")
            max_x = max_y = float("-inf")
            for px, py in polyline:
                if px < min_x:
                    min_x = px
                if px > max_x:
                    max_x = px
                if py < min_y:
                    min_y = py
                if py > max_y:
                    max_y = py
            if max_x < vp_left or min_x > vp_right or max_y < vp_top or min_y > vp_bottom:
                continue
            pts = [self.mm_to_pixel(pt) for pt in polyline]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        # Info label: camera params
        info_pen = QPen(QColor("#AAAAAA"), 1.0)
        painter.setPen(info_pen)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        label = (
            f"3D Preview  Az:{self._3d_cam_azimuth:.0f}°  "
            f"El:{self._3d_cam_elevation:.0f}°  "
            f"Dist:{self._3d_cam_distance:.1f}  "
            f"[Left drag=orbit  Shift+drag/Middle=pan  Scroll=zoom]"
        )
        margin_px = self.mm_to_pixel((canvas.margin_mm, canvas.margin_mm))
        painter.drawText(
            QPointF(margin_px.x() + 4, paper_tl.y() + 16), label
        )

        # Loading indicator when no wireframe yet
        if not self._3d_wireframe_polylines:
            loading_pen = QPen(QColor("#666688"), 1.5)
            painter.setPen(loading_pen)
            font2 = painter.font()
            font2.setPointSize(14)
            painter.setFont(font2)
            cx = (paper_tl.x() + paper_br.x()) / 2.0
            cy = (paper_tl.y() + paper_br.y()) / 2.0
            painter.drawText(
                QPointF(cx - 80, cy), "Generating wireframe…"
            )

    def _draw_map_preview(self, painter: QPainter, canvas) -> None:  # type: ignore[no-untyped-def]
        """Draw decimated map preview polylines and data-bounds outline.

        Projects the stored Mercator preview polylines through the current
        map view transform, then draws them as faded mid-grey lines so the
        printable-area rectangle reads as the visible crop boundary.

        Also draws a faint dashed rectangle showing the extent of the fetched
        data, so the user can see pan/zoom limits.
        """
        if self._map_view is None:
            return

        from plottter.osm.geometry import mercator, view_transform

        transform = view_transform(
            self._map_view["center_lat"],
            self._map_view["center_lon"],
            self._map_view["scale"],
            canvas,
        )

        # Viewport bounds in mm for culling
        vp_left, vp_top = self.pixel_to_mm(QPointF(0.0, 0.0))
        vp_right, vp_bottom = self.pixel_to_mm(
            QPointF(float(self.width()), float(self.height()))
        )

        # Faded mid-grey lines for preview polylines
        line_pen = QPen(QColor(120, 120, 120, 180), max(0.5, self._zoom * 0.2))
        painter.setPen(line_pen)

        for polyline in self._map_preview_polylines:
            if len(polyline) < 2:
                continue
            # Project Mercator → mm
            pts_mm = [
                (
                    transform.x_origin + mx * transform.scale,
                    transform.y_origin - my * transform.scale,
                )
                for mx, my in polyline
            ]
            # Viewport culling (bounding-box check)
            min_x = min(p[0] for p in pts_mm)
            max_x = max(p[0] for p in pts_mm)
            min_y = min(p[1] for p in pts_mm)
            max_y = max(p[1] for p in pts_mm)
            if max_x < vp_left or min_x > vp_right or max_y < vp_top or min_y > vp_bottom:
                continue
            pts_px = [self.mm_to_pixel(p) for p in pts_mm]
            for i in range(len(pts_px) - 1):
                painter.drawLine(pts_px[i], pts_px[i + 1])

        # Dashed outline of the fetched-data bounds
        if self._map_data_bounds is not None:
            min_lat, min_lon, max_lat, max_lon = self._map_data_bounds
            corners_latlon = [
                (min_lat, min_lon),
                (min_lat, max_lon),
                (max_lat, max_lon),
                (max_lat, min_lon),
            ]
            corners_px = []
            for lat, lon in corners_latlon:
                mx, my = mercator(lat, lon)
                x_mm = transform.x_origin + mx * transform.scale
                y_mm = transform.y_origin - my * transform.scale
                corners_px.append(self.mm_to_pixel((x_mm, y_mm)))

            bounds_pen = QPen(QColor(80, 140, 200, 160), max(0.5, self._zoom * 0.3))
            bounds_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(bounds_pen)

            n = len(corners_px)
            for i in range(n):
                painter.drawLine(corners_px[i], corners_px[(i + 1) % n])

    def _show_3d_context_menu(self, pos: QPoint) -> None:
        """Show a context menu for 3D preview interactions."""
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)

        reset_act = menu.addAction("Reset Camera")
        reset_act.triggered.connect(self._reset_3d_camera)

        toggle_proj_act = menu.addAction("Toggle Projection (Perspective/Ortho)")
        toggle_proj_act.triggered.connect(self._toggle_3d_projection)

        menu.exec(self.mapToGlobal(pos))

    def _reset_3d_camera(self) -> None:
        """Reset camera to default orbit position and emit signals."""
        self._3d_cam_azimuth = 30.0
        self._3d_cam_elevation = 20.0
        self._3d_cam_distance = 8.0
        self._3d_cam_lookat = (0.0, 0.0, 0.0)
        self.camera_orbit_changed.emit(30.0, 20.0, 8.0)
        self.camera_pan_changed.emit(0.0, 0.0, 0.0)
        self.update()

    def _toggle_3d_projection(self) -> None:
        """Emit signal so the settings panel toggles its projection combo."""
        self.camera_projection_toggle_requested.emit()

    def _draw_grid(self, painter: QPainter, canvas) -> None:  # type: ignore[no-untyped-def]
        grid_pen = QPen(QColor("#DDDDDD"), 0.5)
        painter.setPen(grid_pen)
        spacing = self.GRID_SPACING_MM
        x = spacing
        while x < canvas.width_mm:
            p1 = self.mm_to_pixel((x, 0))
            p2 = self.mm_to_pixel((x, canvas.height_mm))
            painter.drawLine(p1, p2)
            x += spacing
        y = spacing
        while y < canvas.height_mm:
            p1 = self.mm_to_pixel((0, y))
            p2 = self.mm_to_pixel((canvas.width_mm, y))
            painter.drawLine(p1, p2)
            y += spacing

    def _draw_registration_marks(
        self, painter: QPainter, canvas, style: str = "corners"  # type: ignore[no-untyped-def]
    ) -> None:
        """Draw registration marks per style: 'corners', 'center', or 'both'."""
        arm = 3.0  # mm
        reg_pen = QPen(QColor("black"), 0.5)
        painter.setPen(reg_pen)

        if style in ("corners", "both"):
            corners = [
                (0.0, 0.0),
                (canvas.width_mm, 0.0),
                (0.0, canvas.height_mm),
                (canvas.width_mm, canvas.height_mm),
            ]
            for cx, cy in corners:
                sign_x = 1.0 if cx == 0.0 else -1.0
                sign_y = 1.0 if cy == 0.0 else -1.0
                p_center = self.mm_to_pixel((cx, cy))
                p_h = self.mm_to_pixel((cx + sign_x * arm, cy))
                p_v = self.mm_to_pixel((cx, cy + sign_y * arm))
                painter.drawLine(p_center, p_h)
                painter.drawLine(p_center, p_v)

        if style in ("center", "both"):
            cx = canvas.width_mm / 2
            cy = canvas.height_mm / 2
            painter.drawLine(
                self.mm_to_pixel((cx - arm, cy)),
                self.mm_to_pixel((cx + arm, cy)),
            )
            painter.drawLine(
                self.mm_to_pixel((cx, cy - arm)),
                self.mm_to_pixel((cx, cy + arm)),
            )

    def _jitter_point(self, pt: tuple[float, float]) -> QPointF:
        """Convert a mm point to pixel, adding jitter noise when enabled.

        Jitter is applied in pixel space so the wobble is always visible
        regardless of zoom level. Sigma scales with intensity:
        at intensity=1.0 the standard deviation is 0.8 px.
        This is preview-only and does not touch model data.
        """
        qpt = self.mm_to_pixel(pt)
        if self._jitter_enabled:
            sigma = 0.8 * self._jitter_intensity
            qpt = QPointF(
                qpt.x() + random.gauss(0.0, sigma),
                qpt.y() + random.gauss(0.0, sigma),
            )
        return qpt

    def _draw_layer(
        self,
        painter: QPainter,
        layer,  # type: ignore[no-untyped-def]
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        color = QColor(layer.color)
        # In Ink Preview the multiply blend depends on full-alpha sources to
        # produce true subtractive colour — a 50% opacity cyan × magenta would
        # otherwise come back faded.  Editing view keeps the opacity slider.
        color.setAlphaF(1.0 if self._ink_preview else layer.opacity)
        # Pen width tracks the configured preview pen width (mm) scaled by
        # zoom (px/mm); never below 0.5 px so hairlines stay visible.
        pen = QPen(color, max(0.5, self._zoom * self._preview_pen_width_mm))
        painter.setPen(pen)

        # Compute viewport bounds in mm for culling — paths whose bounding box
        # is entirely outside the visible area can be skipped entirely.
        vp_left, vp_top = self.pixel_to_mm(QPointF(0.0, 0.0))
        vp_right, vp_bottom = self.pixel_to_mm(QPointF(float(self.width()), float(self.height())))

        ox, oy = offset
        for polyline in layer.paths:
            if len(polyline) < 2:
                continue
            # Viewport culling: skip paths entirely outside the visible area.
            min_x = min_y = float("inf")
            max_x = max_y = float("-inf")
            for px, py in polyline:
                opx, opy = px + ox, py + oy
                if opx < min_x:
                    min_x = opx
                if opx > max_x:
                    max_x = opx
                if opy < min_y:
                    min_y = opy
                if opy > max_y:
                    max_y = opy
            if max_x < vp_left or min_x > vp_right or max_y < vp_top or min_y > vp_bottom:
                continue
            pts = [self._jitter_point((px + ox, py + oy)) for px, py in polyline]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

    def _draw_animated_paths(self, painter: QPainter) -> None:
        """Render paths in animation mode: completed=full, current=partial, future=hidden."""
        all_paths = self._anim_all_paths
        current_idx = self._anim_current_path

        for i, (color_str, opacity, polyline) in enumerate(all_paths):
            if i > current_idx:
                break  # future paths hidden

            color = QColor(color_str)
            color.setAlphaF(opacity)
            # Match _draw_layer so animation playback uses the same preview
            # width as static rendering.
            pen = QPen(color, max(0.5, self._zoom * self._preview_pen_width_mm))
            painter.setPen(pen)

            if i < current_idx:
                # Completed path — draw fully
                if len(polyline) < 2:
                    continue
                pts = [self._jitter_point(pt) for pt in polyline]
                for j in range(len(pts) - 1):
                    painter.drawLine(pts[j], pts[j + 1])
            else:
                # Current partial path — draw up to current point
                end_pt = min(self._anim_current_point + 1, len(polyline))
                partial = polyline[:end_pt]
                if len(partial) >= 2:
                    pts = [self._jitter_point(pt) for pt in partial]
                    for j in range(len(pts) - 1):
                        painter.drawLine(pts[j], pts[j + 1])

                # Pen position indicator (red crosshair)
                pt_idx = min(self._anim_current_point, len(polyline) - 1)
                if polyline:
                    pos = self.mm_to_pixel(polyline[pt_idx])
                    indicator_pen = QPen(QColor("#FF4444"), 1.5)
                    painter.setPen(indicator_pen)
                    r = 5.0  # pixel radius
                    painter.drawLine(
                        QPointF(pos.x() - r, pos.y()), QPointF(pos.x() + r, pos.y())
                    )
                    painter.drawLine(
                        QPointF(pos.x(), pos.y() - r), QPointF(pos.x(), pos.y() + r)
                    )

    def _draw_travel_lines(self, painter: QPainter, project) -> None:  # type: ignore[no-untyped-def]
        """Draw pen-up travel moves as dotted gray lines."""
        travel_pen = QPen(QColor("#AAAAAA"), 0.5, Qt.PenStyle.DotLine)
        painter.setPen(travel_pen)

        last_end: tuple[float, float] | None = None
        for layer in project.layers:
            if not layer.visible:
                continue
            for polyline in layer.paths:
                if not polyline:
                    continue
                if last_end is not None:
                    p1 = self.mm_to_pixel(last_end)
                    p2 = self.mm_to_pixel(polyline[0])
                    painter.drawLine(p1, p2)
                last_end = polyline[-1]

    # ------------------------------------------------------------------
    # Mask paint rendering helpers
    # ------------------------------------------------------------------

    def _draw_mask_overlay(self, painter: QPainter, canvas) -> None:  # type: ignore[no-untyped-def]
        """Render the mask as a semi-transparent red overlay scaled to the paper area."""
        assert self._mask_array is not None
        h, w = self._mask_array.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = 180   # red
        rgba[:, :, 1] = 30    # slight green tint → warm red
        rgba[:, :, 3] = (self._mask_array * 150).astype(np.uint8)

        arr_c = np.ascontiguousarray(rgba)
        qimg = QImage(arr_c.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg.copy())

        paper_tl = self.mm_to_pixel((0.0, 0.0))
        paper_br = self.mm_to_pixel((canvas.width_mm, canvas.height_mm))
        paper_rect = QRectF(paper_tl, paper_br)
        painter.drawPixmap(paper_rect, pixmap, QRectF(pixmap.rect()))

    def _draw_brush_cursor(self, painter: QPainter, px: float, py: float) -> None:
        """Draw a dashed ring showing the current brush footprint."""
        radius_screen = self._mask_brush_size_mm * self._zoom / 2.0
        color = QColor(220, 50, 50, 200) if not self._mask_erase else QColor(50, 120, 220, 200)
        pen = QPen(color, 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(px, py), radius_screen, radius_screen)

    def _draw_shape_feedback(self, painter: QPainter) -> None:
        """Draw rubber-band / in-progress feedback for shape tools."""
        color = QColor(220, 50, 50, 220) if not self._mask_erase else QColor(50, 120, 220, 220)
        dash_pen = QPen(color, 1.5, Qt.PenStyle.DashLine)
        solid_pen = QPen(color, 1.5, Qt.PenStyle.SolidLine)

        if self._mask_tool in (MaskTool.RECTANGLE, MaskTool.CIRCLE):
            if self._shape_start_mm is not None and self._shape_end_mm is not None:
                tl = self.mm_to_pixel(self._shape_start_mm)
                br = self.mm_to_pixel(self._shape_end_mm)
                rect = QRectF(tl, br)
                painter.setPen(dash_pen)
                fill_color = QColor(color.red(), color.green(), color.blue(), 30)
                painter.setBrush(fill_color)
                if self._mask_tool == MaskTool.RECTANGLE:
                    painter.drawRect(rect)
                else:
                    painter.drawEllipse(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        elif self._mask_tool == MaskTool.POLYGON and self._polygon_vertices:
            painter.setPen(solid_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Draw completed segments
            for i in range(len(self._polygon_vertices) - 1):
                p1 = self.mm_to_pixel(self._polygon_vertices[i])
                p2 = self.mm_to_pixel(self._polygon_vertices[i + 1])
                painter.drawLine(p1, p2)
            # Rubber-band line from last vertex to cursor
            if self._polygon_cursor_mm is not None:
                last = self.mm_to_pixel(self._polygon_vertices[-1])
                cursor = self.mm_to_pixel(self._polygon_cursor_mm)
                painter.setPen(dash_pen)
                painter.drawLine(last, cursor)
                # Faint closing line from cursor to first vertex
                if len(self._polygon_vertices) >= 2:
                    first = self.mm_to_pixel(self._polygon_vertices[0])
                    close_pen = QPen(
                        QColor(color.red(), color.green(), color.blue(), 80),
                        1.0,
                        Qt.PenStyle.DotLine,
                    )
                    painter.setPen(close_pen)
                    painter.drawLine(cursor, first)
            # Vertex markers
            painter.setPen(solid_pen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 180))
            for vx, vy in self._polygon_vertices:
                p = self.mm_to_pixel((vx, vy))
                painter.drawEllipse(p, 3.5, 3.5)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        elif self._mask_tool == MaskTool.PEN and len(self._pen_points) >= 2:
            painter.setPen(solid_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._pen_points) - 1):
                p1 = self.mm_to_pixel(self._pen_points[i])
                p2 = self.mm_to_pixel(self._pen_points[i + 1])
                painter.drawLine(p1, p2)
            # Dotted closing line back to first point
            if len(self._pen_points) >= 3:
                last = self.mm_to_pixel(self._pen_points[-1])
                first = self.mm_to_pixel(self._pen_points[0])
                close_pen = QPen(
                    QColor(color.red(), color.green(), color.blue(), 140),
                    1.0,
                    Qt.PenStyle.DotLine,
                )
                painter.setPen(close_pen)
                painter.drawLine(last, first)

    # ------------------------------------------------------------------
    # Shape drawing helpers
    # ------------------------------------------------------------------

    def _sd_make_rectangle_polyline(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> list[tuple[float, float]]:
        """Return a closed rectangle polyline (5 points, last = first) in mm."""
        lx, rx = min(x1, x2), max(x1, x2)
        ty, by = min(y1, y2), max(y1, y2)
        return [(lx, ty), (rx, ty), (rx, by), (lx, by), (lx, ty)]

    def _sd_make_ellipse_polyline(
        self, x1: float, y1: float, x2: float, y2: float, n: int = 64
    ) -> list[tuple[float, float]]:
        """Return an approximate closed ellipse polyline with n points in mm."""
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx = abs(x2 - x1) / 2.0
        ry = abs(y2 - y1) / 2.0
        pts = []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            pts.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
        pts.append(pts[0])  # close
        return pts

    def _draw_shape_draw_feedback(self, painter: QPainter) -> None:
        """Draw in-progress shape feedback for Shape Drawing mode (dashed layer color)."""
        color = QColor(self._shape_draw_color)
        dash_pen = QPen(color, 1.5, Qt.PenStyle.DashLine)
        solid_pen = QPen(color, 1.5, Qt.PenStyle.SolidLine)

        if self._shape_draw_tool in (ShapeDrawTool.RECTANGLE, ShapeDrawTool.ELLIPSE):
            if self._sd_start_mm is not None and self._sd_end_mm is not None:
                tl = self.mm_to_pixel(self._sd_start_mm)
                br = self.mm_to_pixel(self._sd_end_mm)
                rect = QRectF(tl, br)
                painter.setPen(dash_pen)
                fill_color = QColor(color.red(), color.green(), color.blue(), 30)
                painter.setBrush(fill_color)
                if self._shape_draw_tool == ShapeDrawTool.RECTANGLE:
                    painter.drawRect(rect)
                else:
                    painter.drawEllipse(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        elif self._shape_draw_tool == ShapeDrawTool.POLYGON and self._sd_polygon_vertices:
            painter.setPen(solid_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._sd_polygon_vertices) - 1):
                p1 = self.mm_to_pixel(self._sd_polygon_vertices[i])
                p2 = self.mm_to_pixel(self._sd_polygon_vertices[i + 1])
                painter.drawLine(p1, p2)
            if self._sd_polygon_cursor_mm is not None:
                last = self.mm_to_pixel(self._sd_polygon_vertices[-1])
                cursor = self.mm_to_pixel(self._sd_polygon_cursor_mm)
                painter.setPen(dash_pen)
                painter.drawLine(last, cursor)
                if len(self._sd_polygon_vertices) >= 2:
                    first = self.mm_to_pixel(self._sd_polygon_vertices[0])
                    close_pen = QPen(
                        QColor(color.red(), color.green(), color.blue(), 80),
                        1.0,
                        Qt.PenStyle.DotLine,
                    )
                    painter.setPen(close_pen)
                    painter.drawLine(cursor, first)
            painter.setPen(solid_pen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 180))
            for vx, vy in self._sd_polygon_vertices:
                p = self.mm_to_pixel((vx, vy))
                painter.drawEllipse(p, 3.5, 3.5)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        elif self._shape_draw_tool == ShapeDrawTool.FREEHAND and len(self._sd_pen_points) >= 2:
            painter.setPen(solid_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._sd_pen_points) - 1):
                p1 = self.mm_to_pixel(self._sd_pen_points[i])
                p2 = self.mm_to_pixel(self._sd_pen_points[i + 1])
                painter.drawLine(p1, p2)
            if len(self._sd_pen_points) >= 3:
                last = self.mm_to_pixel(self._sd_pen_points[-1])
                first = self.mm_to_pixel(self._sd_pen_points[0])
                close_pen = QPen(
                    QColor(color.red(), color.green(), color.blue(), 140),
                    1.0,
                    Qt.PenStyle.DotLine,
                )
                painter.setPen(close_pen)
                painter.drawLine(last, first)

        elif self._shape_draw_tool == ShapeDrawTool.LINE and len(self._sd_line_vertices) >= 1:
            painter.setPen(solid_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._sd_line_vertices) - 1):
                p1 = self.mm_to_pixel(self._sd_line_vertices[i])
                p2 = self.mm_to_pixel(self._sd_line_vertices[i + 1])
                painter.drawLine(p1, p2)
            if self._sd_line_cursor_mm is not None and self._sd_line_vertices:
                last = self.mm_to_pixel(self._sd_line_vertices[-1])
                cursor = self.mm_to_pixel(self._sd_line_cursor_mm)
                painter.setPen(dash_pen)
                painter.drawLine(last, cursor)
            painter.setPen(solid_pen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 180))
            for vx, vy in self._sd_line_vertices:
                p = self.mm_to_pixel((vx, vy))
                painter.drawEllipse(p, 3.5, 3.5)
            painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_ai_mask_overlays(self, painter: QPainter) -> None:
        """Render AI mask interaction handles: positive/negative points and bounding box."""
        r = 7.0  # radius in pixels

        # Positive points — green circle with plus sign
        for x_mm, y_mm in self._ai_mask_positive_points:
            pos = self.mm_to_pixel((x_mm, y_mm))
            painter.setPen(QPen(QColor(0, 200, 60), 2.0))
            painter.setBrush(QColor(0, 200, 60, 160))
            painter.drawEllipse(pos, r, r)
            painter.setPen(QPen(QColor("white"), 1.5))
            painter.drawLine(
                QPointF(pos.x() - r * 0.6, pos.y()),
                QPointF(pos.x() + r * 0.6, pos.y()),
            )
            painter.drawLine(
                QPointF(pos.x(), pos.y() - r * 0.6),
                QPointF(pos.x(), pos.y() + r * 0.6),
            )

        # Negative points — red circle with minus sign
        for x_mm, y_mm in self._ai_mask_negative_points:
            pos = self.mm_to_pixel((x_mm, y_mm))
            painter.setPen(QPen(QColor(220, 40, 40), 2.0))
            painter.setBrush(QColor(220, 40, 40, 160))
            painter.drawEllipse(pos, r, r)
            painter.setPen(QPen(QColor("white"), 1.5))
            painter.drawLine(
                QPointF(pos.x() - r * 0.6, pos.y()),
                QPointF(pos.x() + r * 0.6, pos.y()),
            )

        # Bounding box — blue dashed rectangle
        if self._ai_box_start is not None and self._ai_box_end is not None:
            tl = self.mm_to_pixel(self._ai_box_start)
            br = self.mm_to_pixel(self._ai_box_end)
            rect = QRectF(tl, br)
            painter.setPen(QPen(QColor(30, 130, 255), 2.0, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(30, 130, 255, 30))
            painter.drawRect(rect)

    def _draw_fmm_source_marker(
        self, painter: QPainter, x_mm: float, y_mm: float, is_preview: bool = False
    ) -> None:
        """Draw a crosshair + circle at the given mm position as FMM source feedback.

        Args:
            painter: Active QPainter for this paint event.
            x_mm: Marker X coordinate in canvas mm.
            y_mm: Marker Y coordinate in canvas mm.
            is_preview: When True, renders semi-transparent (live cursor preview).
        """
        pos = self.mm_to_pixel((x_mm, y_mm))
        arm_px = 12.0   # arm length in pixels
        gap_px = 5.0    # gap between centre and arm start / circle radius

        alpha = 160 if is_preview else 255
        fg_color = QColor(255, 140, 0, alpha)    # orange foreground
        outline_color = QColor(0, 0, 0, alpha)   # black outline for contrast

        painter.setBrush(Qt.BrushStyle.NoBrush)

        # --- black outline pass (thicker, drawn first for contrast) ---
        painter.setPen(QPen(outline_color, 2.5))
        # Horizontal arms
        painter.drawLine(
            QPointF(pos.x() - arm_px - gap_px, pos.y()),
            QPointF(pos.x() - gap_px, pos.y()),
        )
        painter.drawLine(
            QPointF(pos.x() + gap_px, pos.y()),
            QPointF(pos.x() + arm_px + gap_px, pos.y()),
        )
        # Vertical arms
        painter.drawLine(
            QPointF(pos.x(), pos.y() - arm_px - gap_px),
            QPointF(pos.x(), pos.y() - gap_px),
        )
        painter.drawLine(
            QPointF(pos.x(), pos.y() + gap_px),
            QPointF(pos.x(), pos.y() + arm_px + gap_px),
        )
        # Circle
        painter.drawEllipse(pos, gap_px, gap_px)

        # --- orange foreground pass (thinner) ---
        painter.setPen(QPen(fg_color, 1.5))
        painter.drawLine(
            QPointF(pos.x() - arm_px - gap_px, pos.y()),
            QPointF(pos.x() - gap_px, pos.y()),
        )
        painter.drawLine(
            QPointF(pos.x() + gap_px, pos.y()),
            QPointF(pos.x() + arm_px + gap_px, pos.y()),
        )
        painter.drawLine(
            QPointF(pos.x(), pos.y() - arm_px - gap_px),
            QPointF(pos.x(), pos.y() - gap_px),
        )
        painter.drawLine(
            QPointF(pos.x(), pos.y() + gap_px),
            QPointF(pos.x(), pos.y() + arm_px + gap_px),
        )
        painter.drawEllipse(pos, gap_px, gap_px)
