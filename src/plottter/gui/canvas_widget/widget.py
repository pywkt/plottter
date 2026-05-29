"""CanvasWidget — zoomable, pannable vector canvas rendered with QPainter."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from plottter.gui.project_controller import ProjectController

from ._animation import _AnimationMixin
from ._events import _EventsMixin
from ._mask_ops import _MaskOpsMixin
from ._painting import _PaintingMixin
from .enums import MaskTool, ShapeDrawTool, _MASK_PX_PER_MM



def _simplify_polylines(
    polylines: list[list[tuple[float, float]]],
    tolerance: float = 1e-5,
) -> list[list[tuple[float, float]]]:
    """Apply Douglas-Peucker simplification to a list of Mercator polylines.

    Uses Shapely's ``simplify`` for robustness.  Falls back to the original
    polyline if Shapely is unavailable or raises.

    Args:
        polylines:  List of polylines, each a list of (x, y) Mercator pairs.
        tolerance:  Simplification tolerance in Mercator units.

    Returns:
        Simplified polylines (same order, never fewer than 2 points each).
    """
    result = []
    for pl in polylines:
        if len(pl) < 2:
            continue
        try:
            from shapely.geometry import LineString  # type: ignore
            simp = LineString(pl).simplify(tolerance, preserve_topology=False)
            coords = list(simp.coords)
            result.append(coords if len(coords) >= 2 else pl)
        except Exception:
            result.append(pl)
    return result


def _cap_polylines(
    polylines: list[list[tuple[float, float]]],
    max_points: int,
) -> list[list[tuple[float, float]]]:
    """Drop the smallest polylines first until total point count ≤ *max_points*.

    Large features (by point count) are retained preferentially so the
    most-visible roads/outlines survive the budget cut.

    Args:
        polylines:  List of polylines (already simplified).
        max_points: Maximum total number of points to keep.

    Returns:
        Subset of *polylines* with total points ≤ *max_points*.
    """
    total = sum(len(pl) for pl in polylines)
    if total <= max_points:
        return polylines
    # Sort largest first, greedily include until budget is exhausted.
    sorted_pls = sorted(polylines, key=len, reverse=True)
    result: list[list[tuple[float, float]]] = []
    used = 0
    for pl in sorted_pls:
        if used + len(pl) <= max_points:
            result.append(pl)
            used += len(pl)
        if used >= max_points:
            break
    return result


class CanvasWidget(_EventsMixin, _PaintingMixin, _MaskOpsMixin, _AnimationMixin, QWidget):
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

    # Emitted when the map view changes during interactive positioning.
    # Args: (center_lat, center_lon, scale)
    map_view_changed = pyqtSignal(float, float, float)

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
        self._last_pan_pos = None
        self._show_grid = False
        self._show_reg_marks = True
        self._fitted = False
        self._image_overlay: QPixmap | None = None
        self._image_overlay_rect_mm: tuple[float, float, float, float] | None = None
        self._show_travel = False
        self._show_paper_texture = False
        self._show_image_overlay = True
        # Ink Preview mode — flip the canvas to multiply-blended layer
        # rendering so stacked CMYK / colour-separated layers combine
        # subtractively (cyan + yellow = green, etc.).  Preview-only: no
        # effect on project data or exports.
        self._ink_preview = False

        # Pen jitter (preview-only): slight random displacement per rendered point
        self._jitter_enabled = False
        self._jitter_intensity = 1.0  # relative intensity, 0.1–5.0

        # Preview pen width in mm — purely a display setting used by
        # _draw_layer to size the QPen so you can eyeball how thick a real
        # marker / ballpoint stroke will be relative to the path layout.
        # Does not affect export.  0.3 mm ≈ a fine-tip pen; bump to ~1.2 mm
        # for marker-style preview.
        self._preview_pen_width_mm: float = 0.3

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
        self._3d_orbit_drag_start = None
        self._3d_orbit_start_az: float = 30.0
        self._3d_orbit_start_el: float = 20.0
        # Pan drag tracking
        self._3d_pan_drag_start = None
        self._3d_pan_start_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Map positioning preview state
        self._map_position_active: bool = False
        self._map_preview_polylines: list = []  # decimated polylines in Mercator coords
        self._map_data_bounds: tuple[float, float, float, float] | None = None  # (min_lat, min_lon, max_lat, max_lon)
        self._map_view: dict | None = None  # {center_lat, center_lon, scale}
        self._map_features: list | None = None  # MapFeature list for clamp_map_view
        # Map pan drag tracking (mirroring 3D orbit/pan)
        self._map_pan_drag_start = None   # QPoint where left-drag began
        self._map_pan_start_merc: tuple[float, float] | None = None  # centre in Mercator at drag start  # {center_lat, center_lon, scale}

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

    def set_ink_preview(self, enabled: bool) -> None:
        """Toggle multiply-blended layer rendering for colour-mixing preview."""
        self._ink_preview = enabled
        self.update()

    def set_preview_pen_width_mm(self, width_mm: float) -> None:
        """Set the on-canvas stroke width in millimetres (display only).

        Clamped to ``[0.05, 5.0]`` to keep the preview useful — sub-pixel
        widths fall back to one pixel anyway, and very wide widths drown
        the path geometry.
        """
        self._preview_pen_width_mm = max(0.05, min(5.0, float(width_mm)))
        self.update()

    def get_preview_pen_width_mm(self) -> float:
        return self._preview_pen_width_mm

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

    # ------------------------------------------------------------------
    # Map positioning preview API
    # ------------------------------------------------------------------

    def set_map_position_active(self, active: bool) -> None:
        """Enable or disable map-positioning interactive preview mode.

        When active, the canvas shows a faded vector preview of the fetched
        map data.  Mouse interactions control the map view:
          - Left drag  → pan (shift center lat/lon)
          - Scroll     → zoom (change scale)
        """
        self._map_position_active = active
        if active:
            self._mask_paint_active = False
            self._shape_draw_active = False
            self._3d_preview_active = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_map_preview_data(
        self,
        map_data: object,
        data_bounds: tuple[float, float, float, float],
        enabled_categories: list[str] | None = None,
    ) -> None:
        """Store fetched MapData and extract decimated Mercator preview polylines.

        Args:
            map_data:    A MapData instance (``map_data.features`` dict).
            data_bounds: Geographic bounds ``(min_lat, min_lon, max_lat, max_lon)``
                         used to draw the faint bounds outline and for clamping.
            enabled_categories: If provided, only features from these category
                ids contribute to the preview (visible polylines AND the
                clamping feature set). When None (legacy callers), every
                fetched category is included. Callers should pass the same
                category list used to drive generation so the preview matches
                what Generate will produce — otherwise the user sees water /
                parks in the preview that the generator (with those toggles
                off) doesn't draw, and ``clamp_map_view`` allows panning into
                feature-empty regions.
        """
        from plottter.osm.geometry import mercator

        self._map_data_bounds = data_bounds

        if enabled_categories is None:
            items = list(map_data.features.items())
        else:
            allow = set(enabled_categories)
            items = [(c, f) for c, f in map_data.features.items() if c in allow]

        # Store flattened feature list for clamp_map_view during interaction.
        self._map_features = [feature for _, features in items for feature in features]

        # Convert all features to Mercator-coord polylines.
        # Include lines and area outlines; skip inner rings (building fills etc.)
        raw: list[list[tuple[float, float]]] = []
        for _category, features in items:
            for feature in features:
                if len(feature.coords) < 2:
                    continue
                merc_pts = [mercator(lat, lon) for lat, lon in feature.coords]
                raw.append(merc_pts)

        # Decimate: simplify then cap total point count.
        MAX_POINTS = 15_000
        simplified = _simplify_polylines(raw)
        self._map_preview_polylines = _cap_polylines(simplified, MAX_POINTS)

        if self._map_position_active:
            self.update()

    def update_map_view(self, view: dict) -> None:
        """Sync map view state from the settings panel (no signal emission).

        Args:
            view: ``{center_lat, center_lon, scale}`` dict (same format as the
                  ``_map_view`` param injected into the generator).
        """
        # Store a copy — several call sites pass their own ``self._map_view``
        # by reference, so aliasing would let later panel-side replacements
        # drift away from the canvas's view (and vice-versa). The pan handler
        # reassigns ``self._map_view`` to a fresh dict per drag so it's safe
        # on that side, but defensive copying here removes one whole class of
        # subtle desync bugs.
        self._map_view = dict(view) if view is not None else None
        if self._map_position_active:
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

    def center_view(self) -> None:
        """Re-center the paper in the widget without changing zoom."""
        if self.width() == 0 or self.height() == 0:
            return
        canvas = self._controller.current_project.canvas
        paper_px_w = canvas.width_mm * self._zoom
        paper_px_h = canvas.height_mm * self._zoom
        self._pan_offset = QPointF(
            (self.width() - paper_px_w) / 2,
            (self.height() - paper_px_h) / 2,
        )
        self.update()

    def pan_left(self) -> None:
        self._pan_offset += QPointF(40.0, 0.0)
        self._clamp_pan_offset()
        self.update()

    def pan_right(self) -> None:
        self._pan_offset += QPointF(-40.0, 0.0)
        self._clamp_pan_offset()
        self.update()

    def pan_up(self) -> None:
        self._pan_offset += QPointF(0.0, 40.0)
        self._clamp_pan_offset()
        self.update()

    def pan_down(self) -> None:
        self._pan_offset += QPointF(0.0, -40.0)
        self._clamp_pan_offset()
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
    # Event handlers (controller signals)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_pan_offset(self) -> None:
        """Clamp _pan_offset so the paper cannot scroll completely off-screen.

        At least 50 px of paper remains visible on each edge.
        """
        project = self._controller.current_project
        if project is None or project.canvas is None:
            return
        canvas = project.canvas
        paper_w_px = canvas.width_mm * self._zoom
        paper_h_px = canvas.height_mm * self._zoom
        margin_px = 50.0
        x = self._pan_offset.x()
        y = self._pan_offset.y()
        x = max(margin_px - paper_w_px, min(x, self.width() - margin_px))
        y = max(margin_px - paper_h_px, min(y, self.height() - margin_px))
        self._pan_offset = QPointF(x, y)

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
