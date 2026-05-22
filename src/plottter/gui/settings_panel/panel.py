"""SettingsPanel — dynamically built parameter controls from a generator definition."""

from __future__ import annotations

from typing import Any

import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from plottter.gui.project_controller import ProjectController

from .workers import _AiBgWorker, _AiMaskWorker, _AiSegmentWorker, _DepthMapWorker, _WireframeWorker
from ._ui_build import _UIBuildMixin
from ._image import _ImageMixin
from ._mask import _MaskMixin
from ._ai_mask import _AiMaskMixin
from ._fmm import _FmmMixin
from ._presets import _PresetsMixin
from ._generate import _GenerateMixin
from ._scene3d import _Scene3dMixin
from ._colorsep import _ColorSepMixin
from ._snapshot import _SnapshotMixin
from ._shape_draw import _ShapeDrawMixin


class SettingsPanel(
    _UIBuildMixin,
    _ImageMixin,
    _MaskMixin,
    _AiMaskMixin,
    _FmmMixin,
    _PresetsMixin,
    _GenerateMixin,
    _Scene3dMixin,
    _ColorSepMixin,
    _SnapshotMixin,
    _ShapeDrawMixin,
    QScrollArea,
):
    """Dynamically builds parameter controls from a generator's get_parameters() list."""

    # Emitted with the preprocessed grayscale image (H×W uint8) or None to clear.
    image_preprocessed = pyqtSignal(object)
    # Emitted with the mm rect (x1, y1, x2, y2) where the image overlay should be drawn,
    # or None when no image is loaded.
    image_rect_changed = pyqtSignal(object)
    # Emitted to request a mode switch (e.g. when restoring a layer's saved mode).
    mode_change_requested = pyqtSignal(str)

    def __init__(self, controller: ProjectController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._generator = None
        self._param_widgets: dict[str, QWidget] = {}
        self._param_labels: dict[str, QLabel] = {}
        self._post_proc_widgets: dict[str, QWidget] = {}
        self._post_proc_labels: dict[str, QLabel] = {}
        self._worker = None
        self._current_mode: str = "Math Art"
        self._raw_image: np.ndarray | None = None
        self._preprocessed_image: np.ndarray | None = None
        self._canvas_ref = None  # set via set_canvas()
        self._ai_bg_rgba: np.ndarray | None = None  # cached RGBA result from remove_background()
        self._ai_bg_worker: _AiBgWorker | None = None
        self._ai_segment_worker: _AiSegmentWorker | None = None
        self._ai_sep_preprocessed: np.ndarray | None = None  # preprocessed image for async segment
        self._ai_mask_worker: _AiMaskWorker | None = None
        self._ai_key_available: bool = False  # True when Replicate API key is valid
        # Singleton ReplicateClient — recreated only when API key or cache_dir changes
        self._replicate_client: object | None = None  # ReplicateClient instance or None
        self._replicate_client_api_key: str = ""  # API key used to build the current client
        self._replicate_client_cache_dir: str | None = None  # cache_dir used to build client
        self._image_source_path: str = ""  # Full path of the loaded source image
        self._image_source_type: str = "file"  # "file", "layer", or "depth_map"
        self._source_layer_id: str | None = None  # layer id used as rasterize source
        # AI depth map state
        self._original_raw_image: np.ndarray | None = None  # pre-depth-map original
        self._depth_map_cache: dict[str, np.ndarray] = {}  # keyed by image_source_path
        self._depth_map_worker: _DepthMapWorker | None = None
        # Cached user presets for the current generator (refreshed in _rebuild_preset_combo)
        self._user_presets: list = []

        # FMM source point pick button (injected by set_generator for ContourGenerator)
        self._pick_fmm_source_btn: object | None = None  # QPushButton or None
        self._pick_fmm_source_label: object | None = None  # QLabel or None

        # Auto-regenerate other 3D layers state (task 62.2)
        self._auto_regen_layers: list = []   # pending layers for auto-regen chain
        self._auto_regen_idx: int = 0
        self._auto_regen_worker: object | None = None  # active GeneratorWorker or None

        # 3D wireframe preview state
        self._wireframe_worker: _WireframeWorker | None = None
        # Debounce timer: 250ms after last camera change, fire wireframe update
        self._wireframe_timer = QTimer(self)
        self._wireframe_timer.setSingleShot(True)
        self._wireframe_timer.setInterval(250)
        self._wireframe_timer.timeout.connect(self._start_wireframe_worker)

        # Debounce timer for preprocessing slider changes (200ms)
        self._preprocess_timer = QTimer(self)
        self._preprocess_timer.setSingleShot(True)
        self._preprocess_timer.setInterval(200)
        self._preprocess_timer.timeout.connect(self._update_image_preview)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setWidget(self._container)
        self.setWidgetResizable(True)
        self.setMinimumWidth(250)

        self._build_initial_ui()

    def _refresh_layer_combo(self, *_args: Any) -> None:
        current = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        for layer in self._controller.current_project.layers:
            self._layer_combo.addItem(layer.name, layer.id)
        # Restore previous selection if possible
        idx = self._layer_combo.findText(current)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._layer_combo.blockSignals(False)

        # Also refresh the mask target layer combo
        current_mask = self._mask_target_layer_combo.currentText()
        self._mask_target_layer_combo.blockSignals(True)
        self._mask_target_layer_combo.clear()
        for layer in self._controller.current_project.layers:
            self._mask_target_layer_combo.addItem(layer.name, layer.id)
        idx_mask = self._mask_target_layer_combo.findText(current_mask)
        if idx_mask >= 0:
            self._mask_target_layer_combo.setCurrentIndex(idx_mask)
        self._mask_target_layer_combo.blockSignals(False)

        # Also refresh the shape draw target layer combo
        self._refresh_sd_layer_combo()

        # Also refresh the rasterize source layer combo
        self._refresh_source_layer_combo()

    def _refresh_sd_layer_combo(self, *_args: Any) -> None:
        """Refresh the Shape Drawing target-layer combo box."""
        current_sd = self._sd_target_layer_combo.currentText()
        self._sd_target_layer_combo.blockSignals(True)
        self._sd_target_layer_combo.clear()
        for layer in self._controller.current_project.layers:
            self._sd_target_layer_combo.addItem(layer.name, layer.id)
        idx_sd = self._sd_target_layer_combo.findText(current_sd)
        if idx_sd >= 0:
            self._sd_target_layer_combo.setCurrentIndex(idx_sd)
        self._sd_target_layer_combo.blockSignals(False)

    def on_mode_changed(self, mode: str) -> None:
        """Called when the mode panel changes mode."""
        self._current_mode = mode
        is_image_mode = mode == "Image to Lines"
        is_color_sep = mode == "Color Separation"
        is_mask_paint = mode == "Mask Paint"
        is_shape_draw = mode == "Shape Drawing"
        is_3d = mode == "3D Scene"

        self._3d_camera_group.setVisible(is_3d)

        # Image source and preprocessing are shared between Image-to-Lines and Color Separation
        self._image_source_group.setVisible(is_image_mode or is_color_sep)
        self._preprocessing_group.setVisible(is_image_mode or is_color_sep)
        self._color_sep_group.setVisible(is_color_sep)

        # Mask paint controls
        self._mask_paint_group.setVisible(is_mask_paint)
        self._saved_masks_group.setVisible(is_mask_paint)
        self._ai_mask_group.setVisible(is_mask_paint)
        self._mask_refine_group.setVisible(is_mask_paint)

        if is_mask_paint:
            self._update_ai_mask_image_label()
            # Apply current AI prompt mode to canvas (point/box enable canvas interaction)
            self._on_ai_mask_mode_changed()
        else:
            # Leaving mask paint mode: disable all AI canvas interaction
            if self._canvas_ref is not None:
                self._canvas_ref.set_ai_mask_mode(None)

        # Deactivate canvas brush painting when leaving mask paint mode
        if self._canvas_ref is not None and not is_mask_paint:
            self._canvas_ref.set_mask_paint_active(False)

        # Deactivate FMM source pick mode when leaving Image to Lines mode
        if self._canvas_ref is not None and not is_image_mode:
            self._canvas_ref.set_fmm_source_mode(False)
            self._canvas_ref.clear_fmm_source_marker()
        if self._fmm_btn_alive():
            self._pick_fmm_source_btn.setText("Pick on Canvas")  # type: ignore[union-attr]

        # Shape drawing controls
        self._shape_draw_group.setVisible(is_shape_draw)
        if self._canvas_ref is not None:
            if is_shape_draw:
                # Sync canvas tool to current combo selection
                tool_text = self._sd_tool_combo.currentText()
                tool = self._SD_TOOL_MAP.get(tool_text, "rectangle")
                self._canvas_ref.set_shape_draw_tool(tool)
                # Set canvas color from the active layer
                layer_id = self._sd_target_layer_combo.currentData()
                if layer_id:
                    layer = self._controller.get_layer(layer_id)
                    if layer is not None:
                        self._canvas_ref.set_shape_draw_color(layer.color)
                self._canvas_ref.set_shape_draw_active(True)
            else:
                self._canvas_ref.set_shape_draw_active(False)

        try:
            from plottter.generators import get_generators_by_category
        except ImportError:
            self._generator_type_group.setVisible(False)
            return

        if mode == "Math Art":
            generators = get_generators_by_category("math")
        elif mode == "Image to Lines":
            generators = get_generators_by_category("image")
        elif mode == "3D Scene":
            generators = get_generators_by_category("3d")
        else:
            generators = []

        self._generator_type_combo.blockSignals(True)
        self._generator_type_combo.clear()
        for gen_cls in generators:
            self._generator_type_combo.addItem(gen_cls.name, gen_cls)
        self._generator_type_combo.blockSignals(False)

        visible = bool(generators) and not is_color_sep and not is_mask_paint and not is_shape_draw
        self._generator_type_group.setVisible(visible)

        if generators and not is_color_sep and not is_mask_paint and not is_shape_draw:
            self._on_generator_type_changed(0)
        else:
            self.set_generator(None)

        # Hide/show standard generator controls for Color Separation, Mask Paint, and Shape Drawing
        show_gen_controls = not is_color_sep and not is_mask_paint and not is_shape_draw
        self._preset_group.setVisible(show_gen_controls)
        self._layer_group.setVisible(show_gen_controls)
        self._params_group.setVisible(show_gen_controls)
        # Post-generation 2D transforms are not applicable in 3D Scene mode
        self._transforms_group.setVisible(show_gen_controls and not is_3d)
        self._post_proc_group.setVisible(show_gen_controls and not is_3d)
        self._generate_btn.setVisible(show_gen_controls)
        self._randomize_btn.setVisible(show_gen_controls)

    def set_canvas(self, canvas) -> None:  # type: ignore[no-untyped-def]
        """Wire up the mask-brush controls and AI mask controls to the canvas widget."""
        self._canvas_ref = canvas

        # Brush size → canvas
        self._brush_size_spin.valueChanged.connect(canvas.set_brush_size_mm)

        # Brush hardness slider (0–100) → canvas (0.0–1.0)
        self._brush_hardness_slider.valueChanged.connect(
            lambda v: canvas.set_brush_hardness(v / 100.0)
        )

        # Erase mode checkbox → canvas
        self._erase_check.stateChanged.connect(
            lambda state: canvas.set_erase_mode(bool(state))
        )

        # Buttons
        self._clear_mask_btn.clicked.connect(self._on_clear_mask)
        self._invert_mask_btn.clicked.connect(self._on_invert_mask)
        self._apply_mask_btn.clicked.connect(self._on_apply_mask)

        # Saved masks list buttons and double-click
        self._save_mask_btn.clicked.connect(self._on_save_mask)
        self._load_mask_btn.clicked.connect(self._on_load_mask)
        self._rename_mask_btn.clicked.connect(self._on_rename_mask)
        self._delete_mask_btn.clicked.connect(self._on_delete_mask)
        self._mask_list.itemDoubleClicked.connect(self._on_load_mask)

        # Canvas stroke-done signal → status label
        canvas.mask_stroke_done.connect(self._on_mask_stroke_done)

        # Canvas mask op done (brush + shapes) → undo stack
        canvas.mask_op_done.connect(self._on_mask_op_done)

        # Mask tool combo → canvas + brush control visibility
        self._mask_tool_combo.currentIndexChanged.connect(self._on_mask_tool_changed)

        # Mask refinement
        self._apply_refinement_btn.clicked.connect(self._on_apply_refinement)

        # AI mask controls → canvas and handlers
        self._ai_mask_clear_btn.clicked.connect(self._on_ai_mask_clear)
        self._ai_mask_generate_btn.clicked.connect(self._on_ai_mask_generate)

        # Canvas AI signals → update status label
        canvas.ai_mask_point_selected.connect(self._on_ai_mask_point_selected)
        canvas.ai_mask_box_drawn.connect(self._on_ai_mask_box_drawn)

        # FMM source point pick signal
        canvas.fmm_source_point_set.connect(self._on_fmm_source_point_set)

        # Shape draw tool combo → canvas
        self._sd_tool_combo.currentIndexChanged.connect(self._on_sd_tool_changed)

        # Canvas shape drawn → handler
        canvas.shape_drawn.connect(self._on_shape_drawn)

        # 3D preview button → toggle canvas 3D mode
        self._3d_preview_btn.toggled.connect(self._on_3d_preview_toggled)

        # Canvas camera signals → sync settings panel spinboxes (bidirectional)
        canvas.camera_orbit_changed.connect(self._on_canvas_camera_orbit_changed)
        canvas.camera_pan_changed.connect(self._on_canvas_camera_pan_changed)
        canvas.camera_projection_toggle_requested.connect(self._on_canvas_projection_toggle)

    def _rebuild_dynamic_params(self) -> None:
        """Clear the dynamic params sub-layout.

        Will be wired up in task 135 to populate adjustable-var widgets.
        For now it only ensures the layout is empty so the layout split
        is observable by tests.
        """
        while self._dynamic_params_layout.rowCount() > 0:
            self._dynamic_params_layout.removeRow(0)
