"""CanvasWidget — zoomable, pannable vector canvas rendered with QPainter."""

from __future__ import annotations

import enum
import math
import random
from typing import TYPE_CHECKING

import numpy as np

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from plottter.gui.project_controller import ProjectController

# Mask resolution: pixels per millimetre (must match image-pipeline PX_PER_MM = 5)
_MASK_PX_PER_MM: int = 5


class MaskTool(str, enum.Enum):
    """Active mask-painting tool."""

    BRUSH = "brush"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"
    PEN = "pen"


class ShapeDrawTool(str, enum.Enum):
    """Active shape-drawing tool."""

    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEHAND = "freehand"
    LINE = "line"


class CanvasWidget(QWidget):
    """Renders project layers with zoom/pan support."""

    # Emitted with (x_mm, y_mm) on mouse move for status bar
    mouse_position_mm = pyqtSignal(float, float)

    # Emitted when animation state changes: (is_playing, current_path_idx, total_paths)
    anim_state_changed = pyqtSignal(bool, int, int)

    # Emitted when a brush stroke finishes (mouse release) with the current mask array.
    # Passes the mask as a float32 ndarray (H×W, values 0–1) or None.
    mask_stroke_done = pyqtSignal(object)

    # Emitted when any mask paint operation completes (brush stroke or shape fill).
    # Args: (mask_before, mask_after) — both float32 ndarray or None; used for undo.
    mask_op_done = pyqtSignal(object, object)

    # Emitted when the user clicks a point in AI mask point-prompt mode.
    # Args: (x_mm, y_mm, is_positive) — True for positive, False for negative.
    ai_mask_point_selected = pyqtSignal(float, float, bool)

    # Emitted when the user finishes drawing a box in AI mask box-prompt mode.
    # Args: (x1_mm, y1_mm, x2_mm, y2_mm) — top-left and bottom-right in mm.
    ai_mask_box_drawn = pyqtSignal(float, float, float, float)

    # Emitted when the user finishes drawing a shape in Shape Drawing mode.
    # Args: list of (x_mm, y_mm) tuples — closed or open polyline.
    shape_drawn = pyqtSignal(list)

    # Emitted when the user orbits the camera in 3D preview mode.
    # Args: (azimuth_deg, elevation_deg, distance)
    camera_orbit_changed = pyqtSignal(float, float, float)

    # Emitted when the user pans the camera look-at point in 3D preview mode.
    # Args: (look_at_x, look_at_y, look_at_z)
    camera_pan_changed = pyqtSignal(float, float, float)

    # Emitted when the user selects "Toggle Projection" from the 3D context menu.
    camera_projection_toggle_requested = pyqtSignal()

    # Emitted when the user finishes a drag-to-move operation on the active layer.
    # Args: (dx_mm, dy_mm) — translation applied to all paths in the active layer.
    layer_move_finished = pyqtSignal(float, float)

    # Emitted when the user clicks to set the FMM source point.
    # Args: (x_mm, y_mm) — click position in canvas mm coordinates.
    fmm_source_point_set = pyqtSignal(float, float)

    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    GRID_SPACING_MM = 10.0
    ANIM_TIMER_INTERVAL_MS = 50  # ~20 fps

    def __init__(self, controller: ProjectController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._last_pan_pos: QPoint | None = None
        self._show_grid = False
        self._show_reg_marks = True
        self._fitted = False
        self._image_overlay: QPixmap | None = None
        self._image_overlay_rect_mm: tuple[float, float, float, float] | None = None
        self._show_travel = False
        self._show_paper_texture = False
        self._show_image_overlay = True

        # Pen jitter (preview-only): slight random displacement per rendered point
        self._jitter_enabled = False
        self._jitter_intensity = 1.0  # relative intensity, 0.1–5.0

        # Mask paint state
        self._mask_paint_active = False
        self._mask_array: np.ndarray | None = None   # float32 H×W, values 0–1
        self._mask_brush_size_mm: float = 5.0        # brush diameter in mm
        self._mask_brush_hardness: float = 0.8       # 0.0 (soft) → 1.0 (hard)
        self._mask_erase: bool = False
        self._last_brush_pos: tuple[float, float] | None = None
        self._brush_cursor_pos: tuple[float, float] | None = None  # screen px

        # Active mask tool
        self._mask_tool: MaskTool = MaskTool.BRUSH

        # Rectangle / Circle tool state (rubber-band drag)
        self._shape_start_mm: tuple[float, float] | None = None
        self._shape_end_mm: tuple[float, float] | None = None

        # Polygon tool state
        self._polygon_vertices: list[tuple[float, float]] = []
        self._polygon_cursor_mm: tuple[float, float] | None = None  # for rubber-band

        # Pen/Lasso tool state
        self._pen_points: list[tuple[float, float]] = []

        # Pre-operation mask snapshot (for undo — saved at press time)
        self._pre_op_mask: np.ndarray | None = None

        # AI mask interaction state
        self._ai_mask_mode: str | None = None  # 'point', 'box', or None
        self._ai_mask_positive_points: list[tuple[float, float]] = []
        self._ai_mask_negative_points: list[tuple[float, float]] = []
        self._ai_box_start: tuple[float, float] | None = None
        self._ai_box_end: tuple[float, float] | None = None

        # Shape drawing state
        self._shape_draw_active = False
        self._shape_draw_tool: ShapeDrawTool = ShapeDrawTool.RECTANGLE
        self._shape_draw_color: str = "#3264C8"  # default blue; overridable
        # Shared rubber-band / vertex state (mutually exclusive with mask paint)
        self._sd_start_mm: tuple[float, float] | None = None  # rect/ellipse start
        self._sd_end_mm: tuple[float, float] | None = None    # rect/ellipse end
        self._sd_polygon_vertices: list[tuple[float, float]] = []
        self._sd_polygon_cursor_mm: tuple[float, float] | None = None
        self._sd_pen_points: list[tuple[float, float]] = []
        self._sd_line_vertices: list[tuple[float, float]] = []
        self._sd_line_cursor_mm: tuple[float, float] | None = None

        # Animation state
        self._anim_mode = False   # True = animation rendering active
        self._anim_playing = False
        # list of (color: str, opacity: float, polyline: list[tuple])
        self._anim_all_paths: list[tuple[str, float, list]] = []
        self._anim_current_path = 0   # index into _anim_all_paths
        self._anim_current_point = 0  # point index within current path
        self._anim_speed = 1.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(self.ANIM_TIMER_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._anim_tick)

        # 3D preview mode state
        self._3d_preview_active: bool = False
        self._3d_wireframe_polylines: list = []
        # Camera state stored for mouse-based orbit/pan/zoom
        self._3d_cam_azimuth: float = 30.0
        self._3d_cam_elevation: float = 20.0
        self._3d_cam_distance: float = 8.0
        self._3d_cam_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Orbit drag tracking
        self._3d_orbit_drag_start: QPoint | None = None
        self._3d_orbit_start_az: float = 30.0
        self._3d_orbit_start_el: float = 20.0
        # Pan drag tracking
        self._3d_pan_drag_start: QPoint | None = None
        self._3d_pan_start_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Drag-to-move tool state
        self._drag_move_active: bool = False
        self._drag_move_start_mm: tuple[float, float] | None = None
        self._drag_move_offset_mm: tuple[float, float] = (0.0, 0.0)

        # FMM source point pick mode
        self._fmm_source_mode: bool = False
        # FMM source point marker (persistent crosshair shown after pick/manual edit)
        self._fmm_source_marker_mm: tuple[float, float] | None = None
        # Live cursor preview crosshair shown while pick mode is active
        self._fmm_cursor_preview_mm: tuple[float, float] | None = None

        # Space+drag hand-pan state
        self._space_held: bool = False
        self._hand_pan_active: bool = False

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        # Connect controller signals
        controller.project_loaded.connect(self._on_project_loaded)
        controller.paths_changed.connect(self._on_paths_changed)
        controller.layer_changed.connect(self._on_layer_changed)
        controller.layers_reordered.connect(self.update)
        controller.canvas_changed.connect(self.update)
        controller.layer_added.connect(self._on_layer_changed)
        controller.layer_removed.connect(self._on_layer_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_show_grid(self, visible: bool) -> None:
        self._show_grid = visible
        self.update()

    def set_show_reg_marks(self, visible: bool) -> None:
        self._show_reg_marks = visible
        self.update()

    def set_show_travel(self, visible: bool) -> None:
        self._show_travel = visible
        self.update()

    def set_paper_texture(self, enabled: bool) -> None:
        self._show_paper_texture = enabled
        self.update()

    def set_show_image_overlay(self, visible: bool) -> None:
        self._show_image_overlay = visible
        self.update()

    def set_jitter_enabled(self, enabled: bool) -> None:
        """Enable/disable pen jitter simulation (preview-only)."""
        self._jitter_enabled = enabled
        self.update()

    def set_jitter_intensity(self, intensity: float) -> None:
        """Set pen jitter intensity (0.1–5.0). Higher values = more wobble."""
        self._jitter_intensity = max(0.1, min(5.0, intensity))
        self.update()

    def get_jitter_intensity(self) -> float:
        """Return the current pen jitter intensity."""
        return self._jitter_intensity

    # -- Mask paint public API --

    def set_mask_paint_active(self, enabled: bool) -> None:
        """Enable or disable mask-painting mode."""
        self._mask_paint_active = enabled
        self._last_brush_pos = None
        self._brush_cursor_pos = None
        # Cancel any in-progress shapes
        self._shape_start_mm = None
        self._shape_end_mm = None
        self._polygon_vertices.clear()
        self._polygon_cursor_mm = None
        self._pen_points.clear()
        if enabled:
            # Disable shape drawing when entering mask paint mode
            self._shape_draw_active = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_mask_tool(self, tool: str) -> None:
        """Set the active mask-painting tool (brush/rectangle/circle/polygon/pen)."""
        self._mask_tool = MaskTool(tool)
        # Cancel any in-progress shapes when switching tools
        self._shape_start_mm = None
        self._shape_end_mm = None
        self._polygon_vertices.clear()
        self._polygon_cursor_mm = None
        self._pen_points.clear()
        self.update()

    def get_mask_tool(self) -> str:
        """Return the current mask tool name."""
        return self._mask_tool.value

    def set_brush_size_mm(self, size_mm: float) -> None:
        """Set the brush diameter in millimetres."""
        self._mask_brush_size_mm = max(0.5, size_mm)

    def set_brush_hardness(self, hardness: float) -> None:
        """Set brush hardness [0.0 = fully soft gaussian, 1.0 = hard circle]."""
        self._mask_brush_hardness = max(0.0, min(1.0, hardness))

    def set_erase_mode(self, erase: bool) -> None:
        self._mask_erase = erase

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Load a pre-existing mask (float32 H×W) or None to clear."""
        self._mask_array = mask
        self.update()

    def get_mask(self) -> np.ndarray | None:
        """Return the current mask array (float32 H×W) or None."""
        return self._mask_array

    def clear_mask(self) -> None:
        """Erase the entire mask."""
        self._mask_array = None
        self.update()

    def invert_mask(self) -> tuple:
        """Invert the mask — painted areas become unpainted and vice versa.

        If no mask has been painted yet, initialises a full mask (all 1.0)
        since inverting an empty (all-zero) mask produces a fully-masked canvas.

        Returns:
            (before, after) tuple of numpy arrays (float32 H×W) representing
            the mask state before and after inversion.  The caller is
            responsible for pushing an undo command.
        """
        before = self._mask_array.copy() if self._mask_array is not None else None
        self._ensure_mask()
        assert self._mask_array is not None
        self._mask_array = 1.0 - self._mask_array
        after = self._mask_array.copy()
        self.update()
        return before, after

    # -- AI mask interaction public API --

    def set_ai_mask_mode(self, mode: str | None) -> None:
        """Set AI mask interaction mode.

        Args:
            mode: ``'point'`` for point-prompt mode, ``'box'`` for box-prompt
                mode, or ``None`` to disable AI mask interaction.
        """
        self._ai_mask_mode = mode
        self._ai_box_start = None
        self._ai_box_end = None
        if mode is not None:
            # Disable brush painting while AI mode is active
            self.set_mask_paint_active(False)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def clear_ai_mask_points(self) -> None:
        """Clear all stored point/box prompts."""
        self._ai_mask_positive_points.clear()
        self._ai_mask_negative_points.clear()
        self._ai_box_start = None
        self._ai_box_end = None
        self.update()

    def get_ai_mask_positive_points(self) -> list[tuple[float, float]]:
        """Return the current list of positive (foreground) point prompts in mm."""
        return list(self._ai_mask_positive_points)

    def get_ai_mask_negative_points(self) -> list[tuple[float, float]]:
        """Return the current list of negative (background) point prompts in mm."""
        return list(self._ai_mask_negative_points)

    def get_ai_mask_box(self) -> tuple[float, float, float, float] | None:
        """Return the current box prompt as (x1, y1, x2, y2) in mm, or None."""
        if self._ai_box_start is None or self._ai_box_end is None:
            return None
        x1, y1 = self._ai_box_start
        x2, y2 = self._ai_box_end
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    # -- Shape drawing public API --

    def set_shape_draw_active(self, active: bool) -> None:
        """Enable or disable shape-drawing mode."""
        self._shape_draw_active = active
        # Cancel any in-progress shape
        self._sd_start_mm = None
        self._sd_end_mm = None
        self._sd_polygon_vertices.clear()
        self._sd_polygon_cursor_mm = None
        self._sd_pen_points.clear()
        self._sd_line_vertices.clear()
        self._sd_line_cursor_mm = None
        if active:
            # Disable mask paint when entering shape draw mode
            self._mask_paint_active = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_shape_draw_tool(self, tool: str) -> None:
        """Set active shape-drawing tool (rectangle/ellipse/polygon/freehand/line)."""
        self._shape_draw_tool = ShapeDrawTool(tool)
        # Cancel any in-progress shape on tool switch
        self._sd_start_mm = None
        self._sd_end_mm = None
        self._sd_polygon_vertices.clear()
        self._sd_polygon_cursor_mm = None
        self._sd_pen_points.clear()
        self._sd_line_vertices.clear()
        self._sd_line_cursor_mm = None
        self.update()

    def set_shape_draw_color(self, color: str) -> None:
        """Set the color used for shape draw feedback and emitted shapes."""
        self._shape_draw_color = color
        self.update()

    # -- 3D preview public API --

    def set_3d_preview_active(self, active: bool) -> None:
        """Enable or disable 3D interactive preview mode.

        When active the canvas shows a real-time wireframe rendered with
        QPainter.  Mouse interactions control the 3D camera:
          - Left drag       → orbit (azimuth / elevation)
          - Middle drag or
            Shift+Left drag → pan (look-at point)
          - Scroll wheel    → zoom (orbit distance)
          - Right-click     → context menu (reset / toggle projection)
        """
        self._3d_preview_active = active
        if active:
            self._mask_paint_active = False
            self._shape_draw_active = False
            self._3d_orbit_drag_start = None
            self._3d_pan_drag_start = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._3d_orbit_drag_start = None
            self._3d_pan_drag_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def update_3d_camera(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat: tuple[float, float, float],
    ) -> None:
        """Sync 3D camera state from the settings panel (no signal emission)."""
        self._3d_cam_azimuth = azimuth
        self._3d_cam_elevation = elevation
        self._3d_cam_distance = distance
        self._3d_cam_lookat = lookat
        if self._3d_preview_active:
            self.update()

    def set_3d_wireframe_polylines(self, polylines: list) -> None:
        """Store pre-rendered wireframe polylines and repaint when in 3D mode."""
        self._3d_wireframe_polylines = polylines
        if self._3d_preview_active:
            self.update()

    # -- Drag-to-move public API --

    def set_drag_move_active(self, active: bool) -> None:
        """Enable or disable drag-to-move tool for repositioning active layer content."""
        self._drag_move_active = active
        self._drag_move_start_mm = None
        self._drag_move_offset_mm = (0.0, 0.0)
        if active:
            # Disable conflicting modes
            self._mask_paint_active = False
            self._shape_draw_active = False
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def is_drag_move_active(self) -> bool:
        """Return True if drag-to-move mode is currently active."""
        return self._drag_move_active

    def set_fmm_source_mode(self, active: bool) -> None:
        """Enable or disable FMM source point pick mode.

        When active, the next left-click emits ``fmm_source_point_set``
        with the click position in mm and then deactivates the mode.
        """
        self._fmm_source_mode = active
        if active:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._fmm_cursor_preview_mm = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_fmm_source_marker(self, x_mm: float, y_mm: float) -> None:
        """Show a persistent crosshair marker at the given canvas position (mm).

        Called after the user picks a source point or edits the percentage
        spinboxes manually so the marker stays visible as visual feedback.
        """
        self._fmm_source_marker_mm = (x_mm, y_mm)
        self.update()

    def clear_fmm_source_marker(self) -> None:
        """Remove the FMM source point marker and live cursor preview from the canvas."""
        self._fmm_source_marker_mm = None
        self._fmm_cursor_preview_mm = None
        self.update()

    def get_image_overlay_rect_mm(self) -> tuple[float, float, float, float] | None:
        """Return the current image overlay rectangle in mm, or None if not set."""
        return self._image_overlay_rect_mm

    def fit_to_window(self) -> None:
        self._fit_to_window()
        self.update()

    def zoom_in(self) -> None:
        self._apply_zoom(1.25, self.rect().center())

    def zoom_out(self) -> None:
        self._apply_zoom(0.8, self.rect().center())

    def set_image_overlay(self, image) -> None:  # type: ignore[no-untyped-def]
        """Set a grayscale numpy array as the semi-transparent canvas overlay.

        Pass None to clear the overlay.
        The image is converted to QPixmap once and cached for painting.
        """
        import numpy as np

        if image is None:
            self._image_overlay = None
        else:
            arr: np.ndarray = np.ascontiguousarray(image)
            if arr.ndim == 2:
                h, w = arr.shape
                qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                h, w = arr.shape[:2]
                qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
            self._image_overlay = QPixmap.fromImage(qimg.copy())
        self.update()

    def set_image_overlay_rect(self, rect_mm: tuple[float, float, float, float] | None) -> None:
        """Set the mm rectangle where the image overlay is drawn.

        Pass None to fall back to the full drawing-area (margin) rectangle.
        This should match the rect returned by compute_image_rect() so the
        overlay aligns with where generators produce output.
        """
        self._image_overlay_rect_mm = rect_mm
        self.update()

    # -- Animation public API --

    def toggle_animation(self) -> None:
        """Toggle animation play/pause. Starts from beginning if not in animation mode."""
        if not self._anim_mode:
            self._rebuild_anim_paths()
            if not self._anim_all_paths:
                return  # nothing to animate
            self._anim_current_path = 0
            self._anim_current_point = 0
            self._anim_mode = True
        if self._anim_playing:
            self._pause_animation()
        else:
            if self._anim_current_path >= len(self._anim_all_paths):
                # Rewind to start
                self._anim_current_path = 0
                self._anim_current_point = 0
            self._play_animation()

    def step_anim_forward(self) -> None:
        """Advance animation by one complete path."""
        if not self._anim_mode:
            self._rebuild_anim_paths()
            if not self._anim_all_paths:
                return
            self._anim_current_path = 0
            self._anim_current_point = 0
            self._anim_mode = True
        if self._anim_current_path < len(self._anim_all_paths):
            self._anim_current_path += 1
            self._anim_current_point = 0
        self._emit_anim_state()
        self.update()

    def step_anim_backward(self) -> None:
        """Go back to the start of the previous path."""
        if not self._anim_mode:
            return
        if self._anim_current_path > 0:
            self._anim_current_path -= 1
            self._anim_current_point = 0
        self._emit_anim_state()
        self.update()

    def seek_animation(self, path_idx: int) -> None:
        """Jump to a specific path index."""
        if not self._anim_mode:
            self._rebuild_anim_paths()
            self._anim_mode = True
        self._anim_current_path = max(0, min(path_idx, len(self._anim_all_paths)))
        self._anim_current_point = 0
        self._emit_anim_state()
        self.update()

    def set_anim_speed(self, speed: float) -> None:
        self._anim_speed = max(0.1, min(10.0, speed))

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def mm_to_pixel(self, point: tuple[float, float]) -> QPointF:
        """Convert mm coordinates to widget pixel coordinates."""
        x_px = point[0] * self._zoom + self._pan_offset.x()
        y_px = point[1] * self._zoom + self._pan_offset.y()
        return QPointF(x_px, y_px)

    def pixel_to_mm(self, point: QPointF) -> tuple[float, float]:
        """Convert widget pixel coordinates to mm coordinates."""
        x_mm = (point.x() - self._pan_offset.x()) / self._zoom
        y_mm = (point.y() - self._pan_offset.y()) / self._zoom
        return (x_mm, y_mm)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_project_loaded(self) -> None:
        self._fitted = True  # already visible; fit immediately
        self._fit_to_window()
        self._reset_animation()
        self.update()

    def _on_paths_changed(self, _layer_id: str) -> None:
        if self._anim_mode:
            self._reset_animation()
        self.update()

    def _on_layer_changed(self, _layer_id: str) -> None:
        if self._anim_mode:
            self._reset_animation()
        self.update()

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
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
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

    def _clamp_pan_offset(self) -> None:
        """Clamp _pan_offset so the paper cannot scroll completely off-screen.

        A full implementation is provided by task 96.6; this stub is a no-op
        placeholder so callers compile without errors until that task runs.
        """

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#808080"))

        project = self._controller.current_project
        canvas = project.canvas
        w_mm = canvas.width_mm
        h_mm = canvas.height_mm

        # Paper boundary
        paper_tl = self.mm_to_pixel((0.0, 0.0))
        paper_br = self.mm_to_pixel((w_mm, h_mm))
        paper_rect = QRectF(paper_tl, paper_br)
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
            # Paths
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

                # Pen-up travel visualization (normal mode only)
                if self._show_travel:
                    self._draw_travel_lines(painter, project)

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
        color.setAlphaF(layer.opacity)
        pen = QPen(color, max(0.5, self._zoom * 0.3))
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
            pen = QPen(color, max(0.5, self._zoom * 0.3))
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
    # Mask paint helpers
    # ------------------------------------------------------------------

    def _ensure_mask(self) -> None:
        """Lazily create the mask array if it doesn't exist yet."""
        if self._mask_array is None:
            canvas = self._controller.current_project.canvas
            h = int(canvas.height_mm * _MASK_PX_PER_MM)
            w = int(canvas.width_mm * _MASK_PX_PER_MM)
            self._mask_array = np.zeros((h, w), dtype=np.float32)

    def _paint_at(self, x_mm: float, y_mm: float) -> None:
        """Stamp the brush at a mm position onto the mask array."""
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        cx = x_mm * _MASK_PX_PER_MM
        cy = y_mm * _MASK_PX_PER_MM
        radius_px = max(0.5, self._mask_brush_size_mm * _MASK_PX_PER_MM / 2.0)

        # Bounding box for the brush stamp
        x1 = max(0, int(cx - radius_px - 1))
        y1 = max(0, int(cy - radius_px - 1))
        x2 = min(w, int(cx + radius_px + 2))
        y2 = min(h, int(cy + radius_px + 2))
        if x1 >= x2 or y1 >= y2:
            return

        xs = np.arange(x1, x2, dtype=np.float32) - cx
        ys = np.arange(y1, y2, dtype=np.float32) - cy
        X, Y = np.meshgrid(xs, ys)
        dist = np.sqrt(X * X + Y * Y)

        hardness = self._mask_brush_hardness
        if hardness >= 0.999:
            stamp = (dist <= radius_px).astype(np.float32)
        else:
            # Gaussian falloff; sigma shrinks as hardness increases
            sigma = radius_px * (1.0 - hardness * 0.9) * 0.5
            if sigma < 0.01:
                stamp = (dist <= radius_px).astype(np.float32)
            else:
                stamp = np.exp(-(dist * dist) / (2.0 * sigma * sigma))
                # Scale so the centre has value 1.0 and clip
                stamp = np.clip(stamp / max(stamp.max(), 1e-6), 0.0, 1.0)

        patch = self._mask_array[y1:y2, x1:x2]
        if self._mask_erase:
            self._mask_array[y1:y2, x1:x2] = np.maximum(patch - stamp, 0.0)
        else:
            self._mask_array[y1:y2, x1:x2] = np.minimum(patch + stamp, 1.0)

        self.update()

    def _interpolate_stroke(
        self, last_pos: tuple[float, float], pos: tuple[float, float]
    ) -> None:
        """Paint brush stamps along the line from last_pos to pos."""
        dx = pos[0] - last_pos[0]
        dy = pos[1] - last_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        step = max(self._mask_brush_size_mm / 4.0, 0.1)
        if dist <= step:
            self._paint_at(*pos)
            return
        n_steps = max(1, int(dist / step))
        for i in range(1, n_steps + 1):
            t = i / n_steps
            self._paint_at(last_pos[0] + t * dx, last_pos[1] + t * dy)

    def _snapshot_mask(self) -> np.ndarray | None:
        """Return a copy of the current mask array, or None if no mask exists."""
        if self._mask_array is None:
            return None
        return self._mask_array.copy()

    def _handle_polygon_press(self, pos_mm: tuple[float, float]) -> None:
        """Add a vertex to the in-progress polygon."""
        if not self._polygon_vertices:
            # First vertex: save snapshot for undo
            self._pre_op_mask = self._snapshot_mask()
        self._polygon_vertices.append(pos_mm)
        self.update()

    def _apply_rectangle_mask(self) -> None:
        """Fill a hard-edged rectangle into the mask array (or erase)."""
        if self._shape_start_mm is None or self._shape_end_mm is None:
            return

        x1_mm, y1_mm = self._shape_start_mm
        x2_mm, y2_mm = self._shape_end_mm

        # Compute unclamped pixel bounds; reject degenerate shapes before
        # allocating the mask (avoids creating a zeros-array on a bare click).
        raw_col1 = int(round(min(x1_mm, x2_mm) * _MASK_PX_PER_MM))
        raw_col2 = int(round(max(x1_mm, x2_mm) * _MASK_PX_PER_MM))
        raw_row1 = int(round(min(y1_mm, y2_mm) * _MASK_PX_PER_MM))
        raw_row2 = int(round(max(y1_mm, y2_mm) * _MASK_PX_PER_MM))
        if raw_col1 >= raw_col2 or raw_row1 >= raw_row2:
            return

        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        col1 = max(0, raw_col1)
        row1 = max(0, raw_row1)
        col2 = min(w, raw_col2)
        row2 = min(h, raw_row2)

        if col1 >= col2 or row1 >= row2:
            return

        if self._mask_erase:
            self._mask_array[row1:row2, col1:col2] = 0.0
        else:
            self._mask_array[row1:row2, col1:col2] = 1.0

    def _apply_ellipse_mask(self) -> None:
        """Fill a hard-edged ellipse into the mask array (or erase)."""
        if self._shape_start_mm is None or self._shape_end_mm is None:
            return
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        import cv2  # available project-wide dependency

        x1_mm, y1_mm = self._shape_start_mm
        x2_mm, y2_mm = self._shape_end_mm

        x1 = min(x1_mm, x2_mm) * _MASK_PX_PER_MM
        y1 = min(y1_mm, y2_mm) * _MASK_PX_PER_MM
        x2 = max(x1_mm, x2_mm) * _MASK_PX_PER_MM
        y2 = max(y1_mm, y2_mm) * _MASK_PX_PER_MM

        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        ax = max(1, int(round((x2 - x1) / 2)))
        ay = max(1, int(round((y2 - y1) / 2)))

        ellipse_buf = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ellipse_buf, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        if self._mask_erase:
            self._mask_array[ellipse_buf > 0] = 0.0
        else:
            self._mask_array[ellipse_buf > 0] = 1.0

    def _apply_polygon_mask(self) -> None:
        """Fill a hard-edged polygon into the mask array (or erase)."""
        if len(self._polygon_vertices) < 3:
            return
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        import cv2

        pts = np.array(
            [
                (int(round(x * _MASK_PX_PER_MM)), int(round(y * _MASK_PX_PER_MM)))
                for x, y in self._polygon_vertices
            ],
            dtype=np.int32,
        )

        poly_buf = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_buf, [pts], 255)

        if self._mask_erase:
            self._mask_array[poly_buf > 0] = 0.0
        else:
            self._mask_array[poly_buf > 0] = 1.0

    def _apply_pen_mask(self) -> None:
        """Fill a hard-edged lasso (freeform closed shape) into the mask array (or erase)."""
        if len(self._pen_points) < 3:
            return
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        import cv2

        pts = np.array(
            [
                (int(round(x * _MASK_PX_PER_MM)), int(round(y * _MASK_PX_PER_MM)))
                for x, y in self._pen_points
            ],
            dtype=np.int32,
        )

        pen_buf = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pen_buf, [pts], 255)

        if self._mask_erase:
            self._mask_array[pen_buf > 0] = 0.0
        else:
            self._mask_array[pen_buf > 0] = 1.0

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

    # ------------------------------------------------------------------
    # Animation internal helpers
    # ------------------------------------------------------------------

    def _rebuild_anim_paths(self) -> None:
        """Collect all visible paths for animation, in layer/path order."""
        self._anim_all_paths = []
        for layer in self._controller.current_project.layers:
            if not layer.visible:
                continue
            for polyline in layer.paths:
                if len(polyline) >= 2:
                    self._anim_all_paths.append(
                        (layer.color, layer.opacity, list(polyline))
                    )

    def _play_animation(self) -> None:
        self._anim_playing = True
        self._anim_timer.start()
        self._emit_anim_state()

    def _pause_animation(self) -> None:
        self._anim_playing = False
        self._anim_timer.stop()
        self._emit_anim_state()

    def _reset_animation(self) -> None:
        """Exit animation mode and return to normal rendering."""
        self._anim_playing = False
        self._anim_timer.stop()
        self._anim_mode = False
        self._anim_all_paths = []
        self._anim_current_path = 0
        self._anim_current_point = 0
        self._emit_anim_state()

    def _emit_anim_state(self) -> None:
        self.anim_state_changed.emit(
            self._anim_playing,
            self._anim_current_path,
            len(self._anim_all_paths),
        )

    def _anim_tick(self) -> None:
        """Advance animation state on each timer tick.

        Advancement is distance-based: the pen moves ~80 mm/s (default plotter
        speed) along the path per real-time second, scaled by ``_anim_speed``.
        At 1× speed and a 50 ms tick the budget is 80 × 0.05 = 4 mm per tick,
        so sparse paths (long segments) and dense paths (many short segments)
        animate at the same physical rate.
        """
        if not self._anim_mode or not self._anim_playing:
            return

        _PLOTTER_SPEED_MM_S = 80.0
        tick_s = self.ANIM_TIMER_INTERVAL_MS / 1000.0
        distance_budget = _PLOTTER_SPEED_MM_S * self._anim_speed * tick_s
        changed_path = False
        completed = False

        while distance_budget > 0:
            if self._anim_current_path >= len(self._anim_all_paths):
                self._pause_animation()
                completed = True
                break

            current_polyline = self._anim_all_paths[self._anim_current_path][2]
            next_pt = self._anim_current_point + 1

            if next_pt >= len(current_polyline):
                # Finished this path; advance to the next one
                self._anim_current_path += 1
                self._anim_current_point = 0
                changed_path = True
                continue

            p0 = current_polyline[self._anim_current_point]
            p1 = current_polyline[next_pt]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            seg_dist = (dx * dx + dy * dy) ** 0.5
            # Use at least 1 µm so degenerate (duplicate) points are consumed
            # without causing an infinite loop.
            distance_budget -= max(seg_dist, 1e-3)
            self._anim_current_point = next_pt

        if changed_path and not completed:
            self._emit_anim_state()
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_zoom(self, factor: float, center) -> None:  # type: ignore[no-untyped-def]
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        # Adjust pan so the zoom is centered on `center`
        cx = center.x()
        cy = center.y()
        scale = new_zoom / self._zoom
        self._pan_offset = QPointF(
            cx - scale * (cx - self._pan_offset.x()),
            cy - scale * (cy - self._pan_offset.y()),
        )
        self._zoom = new_zoom
        self.update()

    def _fit_to_window(self) -> None:
        """Scale and center the canvas to fill the widget."""
        if self.width() == 0 or self.height() == 0:
            return
        canvas = self._controller.current_project.canvas
        margin_px = 20
        available_w = self.width() - 2 * margin_px
        available_h = self.height() - 2 * margin_px
        scale_x = available_w / canvas.width_mm
        scale_y = available_h / canvas.height_mm
        self._zoom = min(scale_x, scale_y)
        paper_px_w = canvas.width_mm * self._zoom
        paper_px_h = canvas.height_mm * self._zoom
        self._pan_offset = QPointF(
            (self.width() - paper_px_w) / 2,
            (self.height() - paper_px_h) / 2,
        )

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self._fit_to_window()
            self.update()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
