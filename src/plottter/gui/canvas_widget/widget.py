"""CanvasWidget — zoomable, pannable vector canvas rendered with QPainter."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPainterPath, QPixmap
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from plottter.gui.project_controller import ProjectController

from ._animation import _AnimationMixin
from ._events import _EventsMixin
from ._mask_ops import _MaskOpsMixin
from ._painting import _PaintingMixin
from ._perf_hud import PerfHud
from ._render_cache import LayerPathCache, ScenePixmapCache, TravelPathCache
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

    # Emitted while the user drags or zooms the source image overlay in
    # image-position mode. Args: (x1_mm, y1_mm, x2_mm, y2_mm) — the new
    # mm rectangle of the image overlay.
    image_view_changed = pyqtSignal(float, float, float, float)

    # Emitted when the user finishes a drag-to-move operation on the active layer.
    # Args: (dx_mm, dy_mm) — translation applied to all paths in the active layer.
    layer_move_finished = pyqtSignal(float, float)

    # Emitted when the user clicks to set the FMM source point.
    # Args: (x_mm, y_mm) — click position in canvas mm coordinates.
    fmm_source_point_set = pyqtSignal(float, float)

    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    GRID_SPACING_MM = 10.0
    #: Idle delay (ms) after the last zoom step before the soft scaled blit is
    #: replaced by a crisp re-render of the scene pixmap (spec §7.3 / §2.2).
    ZOOM_IDLE_MS = 120
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
        # Persistent premultiplied-ARGB overlay cache (§8.1). Built lazily from
        # ``_mask_array``; ``_paint_at`` writes only its stamp bbox, full
        # rebuilds happen on set/clear/invert/first-creation/shape fills.
        self._mask_overlay_qimage: QImage | None = None
        # Test hook: total pixels written into the overlay cache so a test can
        # assert a brush stamp touches no more than its bbox.
        self._mask_overlay_pixels_written: int = 0
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
        # Completed-path geometry, one cached QPainterPath per (color, opacity),
        # appended as paths finish (spec §8.4); rebuilt on backward jumps.
        self._anim_done_paths: dict[tuple[str, float], QPainterPath] = {}
        self._anim_speed = 1.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(self.ANIM_TIMER_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._anim_tick)

        # 3D preview mode state
        self._3d_preview_active: bool = False
        self._3d_wireframe_polylines: list = []
        # One QPainterPath holding every wireframe polyline in mm coords, rebuilt
        # only by ``set_3d_wireframe_polylines``. ``_draw_3d_preview`` draws it
        # through the world (mm→px) transform so orbit/pan/zoom repaints never
        # re-cull or re-project per point (§8.3).
        self._3d_wire_path: QPainterPath = QPainterPath()
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
        # One QPainterPath holding every preview polyline in Mercator coords,
        # rebuilt only by ``set_map_preview_data``. ``_draw_map_preview`` draws
        # it through a composed merc→mm→px QTransform so panning/zooming the map
        # view never rebuilds the path (§8.2).
        self._map_merc_path: QPainterPath = QPainterPath()
        self._map_data_bounds: tuple[float, float, float, float] | None = None  # (min_lat, min_lon, max_lat, max_lon)
        self._map_view: dict | None = None  # {center_lat, center_lon, scale}
        self._map_features: list | None = None  # MapFeature list for clamp_map_view
        # Map pan drag tracking (mirroring 3D orbit/pan)
        self._map_pan_drag_start = None   # QPoint where left-drag began
        self._map_pan_start_merc: tuple[float, float] | None = None  # centre in Mercator at drag start  # {center_lat, center_lon, scale}

        # Image positioning interactive mode state
        self._image_position_active: bool = False
        self._image_pan_drag_start = None  # QPoint where left-drag began
        # The image overlay rect at the moment the drag began; mm tuple
        self._image_pan_start_rect: tuple[float, float, float, float] | None = None

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

        # Env-gated paint-time performance HUD (spec §5.1). Constructed only
        # when PLOTTTER_PERF_HUD=1 at construction; otherwise None, so the
        # per-paint cost is a single ``if self._perf_hud is None`` check.
        self._perf_hud: PerfHud | None = (
            PerfHud() if os.environ.get("PLOTTTER_PERF_HUD") == "1" else None
        )

        # Cached render geometry (spec §6). ``_path_cache`` holds one mm
        # ``QPainterPath`` per layer; ``_travel_cache`` holds the single travel
        # path spanning all visible layers. Disable caching (build fresh per
        # frame, store nothing) via ``PLOTTTER_NO_CANVAS_CACHE=1`` or by setting
        # ``_render_cache_enabled`` directly in tests / the bench --no-cache run.
        self._path_cache = LayerPathCache()
        self._travel_cache = TravelPathCache()
        self._render_cache_enabled: bool = (
            os.environ.get("PLOTTTER_NO_CANVAS_CACHE") != "1"
        )

        # Baked scene pixmap cache (spec §7). ``scene_revision`` is a single
        # monotonically increasing integer bumped by EVERY trigger that changes
        # the static scene content (grid / reg marks / layer paths / travel);
        # the pixmap entry stores the revision it was built at, so a mismatch
        # forces a rebuild. Every §7.4 setter/signal/event calls
        # ``_bump_scene_revision`` — when in doubt, bump (a spurious rebuild is
        # ~60 ms once; a stale cache is a visible bug).
        self._scene_cache = ScenePixmapCache()
        self.scene_revision: int = 0

        # Soft-zoom idle timer (spec §7.3). During a wheel-zoom gesture frames
        # are a scaled blit of the last crisp pixmap; this single-shot timer is
        # (re)armed on every zoom step and, ``ZOOM_IDLE_MS`` after the gesture
        # settles, rebuilds the pixmap crisp and repaints.
        self._zoom_idle_timer = QTimer(self)
        self._zoom_idle_timer.setSingleShot(True)
        self._zoom_idle_timer.setInterval(self.ZOOM_IDLE_MS)
        self._zoom_idle_timer.timeout.connect(self._on_zoom_idle_rebuild)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        # Connect controller signals. Cache invalidation is wired into these
        # existing handlers per spec §6.3 (no new connections):
        controller.project_loaded.connect(self._on_project_loaded)
        controller.paths_changed.connect(self._on_paths_changed)
        controller.layer_changed.connect(self._on_layer_changed)
        controller.layers_reordered.connect(self._on_layers_reordered)
        controller.canvas_changed.connect(self._on_canvas_changed)
        controller.layer_added.connect(self._on_layer_added)
        controller.layer_removed.connect(self._on_layer_removed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _bump_scene_revision(self) -> None:
        """Invalidate the baked scene pixmap by advancing ``scene_revision`` (§7.4).

        Every trigger that changes the static scene content (grid / reg marks /
        layer paths / travel, or the view geometry the pixmap was rendered for)
        calls this so the next paint rebuilds rather than blitting stale pixels.
        """
        self.scene_revision += 1

    def set_show_grid(self, visible: bool) -> None:
        self._show_grid = visible
        self._bump_scene_revision()
        self.update()

    def set_show_reg_marks(self, visible: bool) -> None:
        self._show_reg_marks = visible
        self._bump_scene_revision()
        self.update()

    def set_show_travel(self, visible: bool) -> None:
        self._show_travel = visible
        self._bump_scene_revision()
        self.update()

    def set_paper_texture(self, enabled: bool) -> None:
        self._show_paper_texture = enabled
        self._bump_scene_revision()
        self.update()

    def set_ink_preview(self, enabled: bool) -> None:
        """Toggle multiply-blended layer rendering for colour-mixing preview."""
        self._ink_preview = enabled
        self._bump_scene_revision()
        self.update()

    def set_preview_pen_width_mm(self, width_mm: float) -> None:
        """Set the on-canvas stroke width in millimetres (display only).

        Clamped to ``[0.05, 5.0]`` to keep the preview useful — sub-pixel
        widths fall back to one pixel anyway, and very wide widths drown
        the path geometry.
        """
        self._preview_pen_width_mm = max(0.05, min(5.0, float(width_mm)))
        self._bump_scene_revision()
        self.update()

    def get_preview_pen_width_mm(self) -> float:
        return self._preview_pen_width_mm

    def set_show_image_overlay(self, visible: bool) -> None:
        self._show_image_overlay = visible
        self.update()

    def set_jitter_enabled(self, enabled: bool) -> None:
        """Enable/disable pen jitter simulation (preview-only)."""
        self._jitter_enabled = enabled
        # Baked-jitter variants depend on the enabled flag; drop them so the
        # next paint rebuilds (or skips) them. Un-jittered paths are kept (§6.4).
        self._path_cache.invalidate_jitter()
        self._bump_scene_revision()
        self.update()

    def set_jitter_intensity(self, intensity: float) -> None:
        """Set pen jitter intensity (0.1–5.0). Higher values = more wobble."""
        self._jitter_intensity = max(0.1, min(5.0, intensity))
        # Intensity sets the baked displacement sigma; invalidate variants so
        # they rebuild at the new intensity on the next paint (§6.4).
        self._path_cache.invalidate_jitter()
        self._bump_scene_revision()
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
        self._invalidate_mask_overlay()
        self.update()

    def get_mask(self) -> np.ndarray | None:
        """Return the current mask array (float32 H×W) or None."""
        return self._mask_array

    def clear_mask(self) -> None:
        """Erase the entire mask."""
        self._mask_array = None
        self._invalidate_mask_overlay()
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
        self._invalidate_mask_overlay()
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
        # Build one mm-coordinate QPainterPath for the whole wireframe so
        # repaints only need to set the world transform (§8.3) rather than
        # culling + projecting every point per frame.
        wire_path = QPainterPath()
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            wire_path.moveTo(polyline[0][0], polyline[0][1])
            for px, py in polyline[1:]:
                wire_path.lineTo(px, py)
        self._3d_wire_path = wire_path
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
            self._image_position_active = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_image_position_active(self, active: bool) -> None:
        """Enable or disable image-positioning interactive mode.

        When active, mouse interactions transform the imported source-image
        overlay rect in place:
          - Left drag → translate the overlay
          - Scroll    → scale about the cursor
        Mutates ``_image_overlay_rect_mm`` directly during the gesture and
        emits ``image_view_changed`` so the settings panel can mirror the
        new rect into its spinboxes and persist it.
        """
        self._image_position_active = active
        if active:
            self._mask_paint_active = False
            self._shape_draw_active = False
            self._3d_preview_active = False
            self._map_position_active = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self._image_pan_drag_start = None
            self._image_pan_start_rect = None
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

        # Build one Mercator-coordinate QPainterPath for the whole preview so
        # repaints only need to compose a QTransform (§8.2) rather than
        # re-projecting every point per frame.
        merc_path = QPainterPath()
        for polyline in self._map_preview_polylines:
            if len(polyline) < 2:
                continue
            merc_path.moveTo(polyline[0][0], polyline[0][1])
            for mx, my in polyline[1:]:
                merc_path.lineTo(mx, my)
        self._map_merc_path = merc_path

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
        # Entering/exiting drag-move changes which layer is excluded from the
        # baked pixmap (§7.5), so the scene must rebuild.
        self._bump_scene_revision()
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

    def view_zoom(self) -> float:
        """Current zoom factor in pixels per mm (read-only accessor for rulers)."""
        return self._zoom

    def view_pan_offset(self) -> QPointF:
        """Current pan offset in widget pixels (read-only accessor for rulers)."""
        return QPointF(self._pan_offset)

    # ------------------------------------------------------------------
    # Event handlers (controller signals)
    # ------------------------------------------------------------------

    def _on_canvas_changed(self) -> None:
        # Canvas size / margin / registration-mark style changed — all baked
        # into the scene pixmap, so bump the revision (§7.4) and repaint.
        self._bump_scene_revision()
        self.update()

    def _on_project_loaded(self) -> None:
        # Whole project replaced — every cached path is stale (§6.3).
        self._path_cache.invalidate_all()
        self._travel_cache.invalidate()
        self._bump_scene_revision()
        self._fitted = True  # already visible; fit immediately
        self._fit_to_window()
        self._reset_animation()
        self.update()

    def _on_paths_changed(self, layer_id: str) -> None:
        # The layer's geometry changed → rebuild its path and the travel path
        # (travel hops depend on every layer's endpoints).
        self._path_cache.invalidate(layer_id)
        self._travel_cache.invalidate()
        self._bump_scene_revision()
        if self._anim_mode:
            self._reset_animation()
        self.update()

    def _on_layer_changed(self, layer_id: str) -> None:
        # Colour/opacity changes don't strictly need it, but paths may have
        # changed too — keep it simple (§6.3). Visibility toggles arrive here
        # too, which also affects the travel path (§6.5).
        self._path_cache.invalidate(layer_id)
        self._travel_cache.invalidate()
        self._bump_scene_revision()
        if self._anim_mode:
            self._reset_animation()
        self.update()

    def _on_layer_added(self, _layer_id: str) -> None:
        # A new layer has no cached path yet; only the travel path (which spans
        # all visible layers) needs rebuilding (§6.3).
        self._travel_cache.invalidate()
        self._bump_scene_revision()
        if self._anim_mode:
            self._reset_animation()
        self.update()

    def _on_layer_removed(self, layer_id: str) -> None:
        # Drop the removed layer's entry and rebuild the travel path (§6.3).
        self._path_cache.invalidate(layer_id)
        self._travel_cache.invalidate()
        self._bump_scene_revision()
        if self._anim_mode:
            self._reset_animation()
        self.update()

    def _on_layers_reordered(self) -> None:
        # Per-layer paths are unchanged, but travel order follows layer order
        # so the travel path is rebuilt (§6.3).
        self._travel_cache.invalidate()
        self._bump_scene_revision()
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
        # Arm the soft-zoom idle timer: this frame will blit the cached pixmap
        # scaled (if the ratio is in range), and ``_on_zoom_idle_rebuild`` will
        # swap in a crisp rebuild ~120 ms after the gesture stops (spec §7.3).
        self._zoom_idle_timer.start(self.ZOOM_IDLE_MS)
        self.update()

    def _on_zoom_idle_rebuild(self) -> None:
        """Rebuild the scene pixmap crisp after a soft-zoom gesture settles (§7.3).

        Fired ``ZOOM_IDLE_MS`` after the last zoom step. While the cache is
        bypassed (ink / animation / 3D / no-cache) there is no pixmap to refine,
        so this is a no-op. Otherwise, if the cached pixmap is still at the old
        zoom (the soft state), rebuild it at the now-stable zoom and repaint so
        the next frame blits a crisp 1:1 pixmap instead of the scaled preview.
        """
        if not self._scene_cache_active():
            return
        if not self._scene_cache.is_valid(self):
            self._scene_cache.rebuild(self)
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
        # The scene pixmap is sized to the viewport (+ slop); a resize changes
        # the covered region, so the cache must rebuild (§7.4).
        self._bump_scene_revision()
