"""SettingsPanel — dynamically built parameter controls from a generator definition."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image as _PilImage

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtCore import QThread, pyqtSignal as _pyqtSignal

from plottter.gui.project_controller import ProjectController


class _AiBgWorker(QThread):
    """Runs ReplicateClient.remove_background() off the main GUI thread."""

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(object)  # emits RGBA np.ndarray
    error = _pyqtSignal(str)

    def __init__(
        self, api_key: str, image: "np.ndarray", cache_dir: "str | None" = None, parent: Any = None
    ) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._image = image
        self._cache_dir = cache_dir

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key, cache_dir=self._cache_dir)
            result = client.remove_background(
                self._image, progress_callback=self.progress.emit
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _AiSegmentWorker(QThread):
    """Runs ReplicateClient.segment_image() off the main GUI thread."""

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(list)  # emits list[(mask, hex_color)]
    error = _pyqtSignal(str)

    def __init__(
        self, api_key: str, image: "np.ndarray", num_segments: int, parent: Any = None
    ) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._image = image
        self._num_segments = num_segments

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key)
            results = client.segment_image(
                self._image,
                num_segments=self._num_segments,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class _AiMaskWorker(QThread):
    """Runs AI point/box/text mask generation off the main GUI thread.

    Args:
        api_key: Replicate API key.
        image: RGB source image (H×W×3 uint8) to segment.
        mode: ``'point'``, ``'box'``, or ``'text'``.
        positive_points: (x_mm, y_mm) foreground points (point mode).
        negative_points: (x_mm, y_mm) background points (point mode).
        box_xyxy_mm: Bounding box as (x1, y1, x2, y2) in mm (box mode).
        text_prompt: Natural-language object description (text mode).
        canvas_width_mm: Canvas width in mm (used for mm→pixel conversion).
        canvas_height_mm: Canvas height in mm (used for mm→pixel conversion).
    """

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(object)  # emits binary mask (H×W uint8)
    error = _pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        image: "np.ndarray",
        mode: str,
        positive_points: list | None = None,
        negative_points: list | None = None,
        box_xyxy_mm: tuple | None = None,
        text_prompt: str = "",
        canvas_width_mm: float = 0.0,
        canvas_height_mm: float = 0.0,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._image = image
        self._mode = mode
        self._positive_points = positive_points or []
        self._negative_points = negative_points or []
        self._box_xyxy_mm = box_xyxy_mm
        self._text_prompt = text_prompt
        self._canvas_width_mm = canvas_width_mm
        self._canvas_height_mm = canvas_height_mm

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key)
            img_h, img_w = self._image.shape[:2]

            # Scale factor: mm → image pixels
            # Assumes the image fills the full canvas area.
            sx = img_w / self._canvas_width_mm if self._canvas_width_mm > 0 else 1.0
            sy = img_h / self._canvas_height_mm if self._canvas_height_mm > 0 else 1.0

            if self._mode == "point":
                pos_px = [(int(x * sx), int(y * sy)) for x, y in self._positive_points]
                neg_px = [(int(x * sx), int(y * sy)) for x, y in self._negative_points]
                mask = client.segment_by_point(
                    self._image,
                    pos_px,
                    neg_px if neg_px else None,
                    progress_callback=self.progress.emit,
                )
            elif self._mode == "box":
                if self._box_xyxy_mm is None:
                    raise ValueError("No bounding box specified.")
                x1, y1, x2, y2 = self._box_xyxy_mm
                box_px = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                mask = client.segment_by_box(
                    self._image,
                    box_px,
                    progress_callback=self.progress.emit,
                )
            elif self._mode == "text":
                if not self._text_prompt.strip():
                    raise ValueError("Text prompt must not be empty.")
                mask = client.segment_by_text(
                    self._image,
                    self._text_prompt,
                    progress_callback=self.progress.emit,
                )
            else:
                raise ValueError(f"Unknown AI mask mode: {self._mode!r}")

            self.finished.emit(mask)
        except Exception as exc:
            self.error.emit(str(exc))


class _DepthMapWorker(QThread):
    """Runs ReplicateClient.estimate_depth() off the main GUI thread."""

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(object)  # emits float32 np.ndarray (H×W)
    error = _pyqtSignal(str)

    def __init__(
        self, api_key: str, cache_dir: "str | None", image: "np.ndarray"
    ) -> None:
        super().__init__()  # no parent — prevent Qt parent-child destruction
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._image = image

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key, cache_dir=self._cache_dir)
            result = client.estimate_depth(
                self._image, progress_callback=self.progress.emit
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _VisibilityTrackedLabel(QLabel):
    """QLabel whose isVisible() returns the intended visibility set via setVisible().

    Qt's isVisible() requires the entire parent hierarchy to be shown, which makes
    it unsuitable for unit tests that don't show the top-level window. This subclass
    tracks the explicit show/hide state so isVisible() works independently of the
    parent chain's mapped state.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._explicitly_visible: bool = True  # QWidget default: not hidden until told

    def setVisible(self, visible: bool) -> None:  # type: ignore[override]
        self._explicitly_visible = visible
        super().setVisible(visible)

    def isVisible(self) -> bool:  # type: ignore[override]
        return self._explicitly_visible


class _WireframeWorker(QThread):
    """Render all 3D layers without HLR for a fast wireframe preview.

    Collects all 3D layers' shape params, builds the scene with
    ``hlr_enabled=False``, renders, and emits the projected 2D polylines.

    IMPORTANT: Custom result signals are named ``result_ready`` / ``render_error``
    so they do NOT shadow ``QThread.finished`` — Qt needs the built-in
    ``finished`` signal to work correctly for thread lifecycle management.
    """

    result_ready = _pyqtSignal(list)  # list[Polyline]
    render_error = _pyqtSignal(str)

    def __init__(
        self,
        layer_params_list: list[dict],
        camera_dict: dict,
        canvas_w_mm: float,
        canvas_h_mm: float,
    ) -> None:
        super().__init__()  # no parent — prevent Qt parent-child destruction of running thread
        self._layer_params_list = layer_params_list
        self._camera_dict = camera_dict
        self._canvas_w_mm = canvas_w_mm
        self._canvas_h_mm = canvas_h_mm
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation — checked between expensive steps."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            from plottter.scene3d import Scene, Camera
            from plottter.generators.scene3d_generator import Scene3DGenerator

            if self._cancelled:
                return

            gen = Scene3DGenerator()
            cam_dict = self._camera_dict
            aspect = self._canvas_w_mm / max(self._canvas_h_mm, 1e-6)

            camera = Camera(
                projection=cam_dict.get("projection", "perspective"),
                fov_deg=float(cam_dict.get("fov", 45.0)),
                aspect=aspect,
            )
            camera.set_orbit(
                azimuth_deg=float(cam_dict.get("azimuth", 30.0)),
                elevation_deg=float(cam_dict.get("elevation", 20.0)),
                distance=float(cam_dict.get("distance", 8.0)),
                center=[
                    float(cam_dict.get("look_at_x", 0.0)),
                    float(cam_dict.get("look_at_y", 0.0)),
                    float(cam_dict.get("look_at_z", 0.0)),
                ],
            )

            if self._cancelled:
                return

            scene = Scene(hlr_enabled=False)
            for params in self._layer_params_list:
                if self._cancelled:
                    return
                shape = gen.build_transformed_shape(params)
                if shape is not None:
                    scene.add(shape)

            if not scene.shapes:
                self.result_ready.emit([])
                return

            if self._cancelled:
                return

            polylines = scene.render(
                camera,
                canvas_w_mm=self._canvas_w_mm,
                canvas_h_mm=self._canvas_h_mm,
            )
            if not self._cancelled:
                self.result_ready.emit(polylines)
        except Exception as exc:  # noqa: BLE001
            if not self._cancelled:
                self.render_error.emit(str(exc))


class SettingsPanel(QScrollArea):
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
        self._original_raw_image: "np.ndarray | None" = None  # pre-depth-map original
        self._depth_map_cache: "dict[str, np.ndarray]" = {}  # keyed by image_source_path
        self._depth_map_worker: "_DepthMapWorker | None" = None
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

    def _build_initial_ui(self) -> None:
        # Generator type selector
        self._generator_type_group = QGroupBox("Generator")
        gen_type_layout = QVBoxLayout(self._generator_type_group)
        self._generator_type_combo = QComboBox()
        self._generator_type_combo.currentIndexChanged.connect(self._on_generator_type_changed)
        gen_type_layout.addWidget(self._generator_type_combo)
        self._layout.addWidget(self._generator_type_group)
        self._generator_type_group.setVisible(False)

        # ----------------------------------------------------------------
        # Image Source group (Image-to-Lines mode only)
        # ----------------------------------------------------------------
        self._image_source_group = QGroupBox("Image Source")
        img_src_layout = QVBoxLayout(self._image_source_group)

        # Source type selector: File or Layer
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        self._src_type_file_radio = QRadioButton("File")
        self._src_type_layer_radio = QRadioButton("Use Layer as Image Source")
        self._src_type_layer_radio.setToolTip(
            "Generator chaining: rasterize the paths of an existing layer and use\n"
            "the result as the source image for this generator.\n"
            "\n"
            "This lets you chain generators together — for example, rasterize a\n"
            "stipple layer and run edge detection on it for a different style, or\n"
            "feed a flow-field output into hatching. The rasterized image covers\n"
            "the exact same area as the source layer, so the output aligns 1:1."
        )
        self._src_type_file_radio.setChecked(True)
        self._src_type_btn_group = QButtonGroup(self)
        self._src_type_depth_radio = QRadioButton("AI Depth Map")
        self._src_type_depth_radio.setToolTip(
            "Generates a depth map from the loaded image using AI (requires Replicate API\n"
            "key). The depth map replaces the source image so any generator can use it —\n"
            "darker areas represent closer surfaces, lighter areas represent farther surfaces."
        )
        self._src_type_btn_group.addButton(self._src_type_file_radio)
        self._src_type_btn_group.addButton(self._src_type_layer_radio)
        self._src_type_btn_group.addButton(self._src_type_depth_radio)
        src_type_row = QHBoxLayout()
        src_type_row.addWidget(self._src_type_file_radio)
        src_type_row.addWidget(self._src_type_layer_radio)
        src_type_row.addWidget(self._src_type_depth_radio)
        src_type_row.addStretch()
        img_src_layout.addLayout(src_type_row)

        # --- File source controls ---
        self._file_src_widget = QWidget()
        file_src_layout = QVBoxLayout(self._file_src_widget)
        file_src_layout.setContentsMargins(0, 0, 0, 0)

        load_btn_row = QHBoxLayout()
        self._load_image_btn = QPushButton("Load Image…")
        self._load_image_btn.clicked.connect(self._on_load_image)
        load_btn_row.addWidget(self._load_image_btn)
        load_btn_row.addStretch()
        file_src_layout.addLayout(load_btn_row)

        self._image_filename_label = QLabel("(no image loaded)")
        self._image_filename_label.setWordWrap(True)
        file_src_layout.addWidget(self._image_filename_label)

        img_src_layout.addWidget(self._file_src_widget)

        # --- Layer source controls ---
        self._layer_src_widget = QWidget()
        layer_src_layout = QVBoxLayout(self._layer_src_widget)
        layer_src_layout.setContentsMargins(0, 0, 0, 0)

        # Brief explanation of generator chaining
        _chain_desc = QLabel(
            "Rasterize another layer's paths and use the result as the source\n"
            "image for this generator."
        )
        _chain_desc.setWordWrap(True)
        _chain_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        layer_src_layout.addWidget(_chain_desc)

        layer_src_form = QFormLayout()
        layer_src_form.setContentsMargins(0, 0, 0, 0)

        self._source_layer_combo = QComboBox()
        self._source_layer_combo.setToolTip("Select a layer whose paths will be rasterized as the source image")
        layer_src_form.addRow(QLabel("Source Layer"), self._source_layer_combo)

        # DPI is computed automatically (300 DPI) — not exposed in the UI.
        # The attribute is kept for internal use and backward-compatibility.
        self._rasterize_dpi_spin = QSpinBox()
        self._rasterize_dpi_spin.setRange(72, 600)
        self._rasterize_dpi_spin.setValue(300)
        self._rasterize_dpi_spin.setSuffix(" DPI")

        self._rasterize_stroke_spin = QDoubleSpinBox()
        self._rasterize_stroke_spin.setRange(0.1, 3.0)
        self._rasterize_stroke_spin.setValue(0.3)
        self._rasterize_stroke_spin.setSingleStep(0.1)
        self._rasterize_stroke_spin.setDecimals(1)
        self._rasterize_stroke_spin.setSuffix(" mm")
        layer_src_form.addRow(QLabel("Stroke Width"), self._rasterize_stroke_spin)

        layer_src_layout.addLayout(layer_src_form)

        self._rasterize_refresh_btn = QPushButton("Refresh")
        self._rasterize_refresh_btn.setToolTip("Re-rasterize the source layer")
        self._rasterize_refresh_btn.clicked.connect(self._on_rasterize_layer)
        layer_src_layout.addWidget(self._rasterize_refresh_btn)

        self._layer_src_status_label = QLabel("")
        self._layer_src_status_label.setWordWrap(True)
        layer_src_layout.addWidget(self._layer_src_status_label)

        img_src_layout.addWidget(self._layer_src_widget)
        self._layer_src_widget.setVisible(False)

        # --- AI Depth Map source controls ---
        self._depth_src_widget = QWidget()
        depth_src_layout = QVBoxLayout(self._depth_src_widget)
        depth_src_layout.setContentsMargins(0, 0, 0, 0)

        _depth_desc = QLabel(
            "Generates a depth map using AI and uses it as the source image.\n"
            "Load a photo first, then click Generate Depth Map."
        )
        _depth_desc.setWordWrap(True)
        _depth_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        depth_src_layout.addWidget(_depth_desc)

        depth_btn_row = QHBoxLayout()
        self._gen_depth_btn = QPushButton("Generate Depth Map")
        self._gen_depth_btn.clicked.connect(self._on_generate_depth_map)
        depth_btn_row.addWidget(self._gen_depth_btn)
        depth_btn_row.addStretch()
        depth_src_layout.addLayout(depth_btn_row)

        self._depth_status_label = QLabel("No depth map generated")
        self._depth_status_label.setWordWrap(True)
        depth_src_layout.addWidget(self._depth_status_label)

        self._depth_invert_check = QCheckBox("Invert Depth")
        self._depth_invert_check.setToolTip(
            "Invert the depth map so that farther objects appear darker (closer = lighter)."
        )
        self._depth_invert_check.toggled.connect(self._on_depth_invert_changed)
        depth_src_layout.addWidget(self._depth_invert_check)

        img_src_layout.addWidget(self._depth_src_widget)
        self._depth_src_widget.setVisible(False)

        # Shared thumbnail
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail_label.setMinimumHeight(80)
        self._thumbnail_label.setMaximumHeight(150)
        self._thumbnail_label.setStyleSheet("background: #555; border: 1px solid #888;")
        img_src_layout.addWidget(self._thumbnail_label)

        self._layout.addWidget(self._image_source_group)
        self._image_source_group.setVisible(False)

        # Connect source-type radio buttons (all 3 — handler ignores unchecked signals)
        self._src_type_file_radio.toggled.connect(self._on_image_source_type_changed)
        self._src_type_layer_radio.toggled.connect(self._on_image_source_type_changed)
        self._src_type_depth_radio.toggled.connect(self._on_image_source_type_changed)
        self._source_layer_combo.currentIndexChanged.connect(self._on_source_layer_combo_changed)

        # ----------------------------------------------------------------
        # Preprocessing group (Image-to-Lines mode only)
        # ----------------------------------------------------------------
        self._preprocessing_group = QGroupBox("Preprocessing")
        prep_form = QFormLayout(self._preprocessing_group)

        self._bright_slider = QSlider(Qt.Orientation.Horizontal)
        self._bright_slider.setRange(-100, 100)
        self._bright_slider.setValue(0)
        self._bright_slider.valueChanged.connect(self._on_preprocessing_changed)
        prep_form.addRow(QLabel("Brightness"), self._bright_slider)

        self._contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_slider.setRange(-100, 100)
        self._contrast_slider.setValue(0)
        self._contrast_slider.valueChanged.connect(self._on_preprocessing_changed)
        prep_form.addRow(QLabel("Contrast"), self._contrast_slider)

        # Gamma: slider value 10–500 → gamma = value/100 (0.10–5.00)
        self._gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self._gamma_slider.setRange(10, 500)
        self._gamma_slider.setValue(100)
        self._gamma_slider.valueChanged.connect(self._on_preprocessing_changed)
        self._gamma_val_label = QLabel("1.00")
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(self._gamma_slider)
        gamma_row.addWidget(self._gamma_val_label)
        prep_form.addRow(QLabel("Gamma"), gamma_row)

        self._blur_slider = QSlider(Qt.Orientation.Horizontal)
        self._blur_slider.setRange(0, 20)
        self._blur_slider.setValue(0)
        self._blur_slider.valueChanged.connect(self._on_preprocessing_changed)
        prep_form.addRow(QLabel("Blur"), self._blur_slider)

        # Threshold: checkbox enables it; slider sets value 0–255
        threshold_row_widget = QWidget()
        threshold_row = QHBoxLayout(threshold_row_widget)
        threshold_row.setContentsMargins(0, 0, 0, 0)
        self._threshold_check = QCheckBox()
        self._threshold_check.setChecked(False)
        self._threshold_check.stateChanged.connect(self._on_preprocessing_changed)
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 255)
        self._threshold_slider.setValue(128)
        self._threshold_slider.setEnabled(False)
        self._threshold_slider.valueChanged.connect(self._on_preprocessing_changed)
        self._threshold_check.stateChanged.connect(
            lambda state: self._threshold_slider.setEnabled(bool(state))
        )
        threshold_row.addWidget(self._threshold_check)
        threshold_row.addWidget(self._threshold_slider)
        prep_form.addRow(QLabel("Threshold"), threshold_row_widget)

        self._invert_check = QCheckBox("Invert")
        self._invert_check.stateChanged.connect(self._on_preprocessing_changed)
        prep_form.addRow(QLabel(""), self._invert_check)

        # Remove background: checkbox enables it; spinbox sets tolerance 0–50
        remove_bg_row_widget = QWidget()
        remove_bg_row = QHBoxLayout(remove_bg_row_widget)
        remove_bg_row.setContentsMargins(0, 0, 0, 0)
        self._remove_bg_check = QCheckBox()
        self._remove_bg_check.setChecked(False)
        self._remove_bg_check.stateChanged.connect(self._on_preprocessing_changed)
        self._bg_tolerance_spin = QDoubleSpinBox()
        self._bg_tolerance_spin.setRange(0.0, 50.0)
        self._bg_tolerance_spin.setValue(20.0)
        self._bg_tolerance_spin.setSingleStep(1.0)
        self._bg_tolerance_spin.setEnabled(False)
        self._bg_tolerance_spin.valueChanged.connect(self._on_preprocessing_changed)
        self._remove_bg_check.stateChanged.connect(
            lambda state: self._bg_tolerance_spin.setEnabled(bool(state))
        )
        remove_bg_row.addWidget(self._remove_bg_check)
        remove_bg_row.addWidget(self._bg_tolerance_spin)
        prep_form.addRow(QLabel("Remove Background"), remove_bg_row_widget)

        # AI Background Removal toggle + Apply button
        self._ai_bg_check = QCheckBox("AI Background Removal")
        self._ai_bg_check.setChecked(False)
        self._ai_bg_check.setEnabled(False)
        self._ai_bg_check.setToolTip(
            "Enter a Replicate API key in Preferences > AI Integration to enable"
        )
        self._ai_bg_check.stateChanged.connect(self._on_ai_bg_changed)
        self._apply_ai_bg_btn = QPushButton("Apply")
        self._apply_ai_bg_btn.setEnabled(False)
        self._apply_ai_bg_btn.setToolTip("Call AI to remove background from the current image")
        self._apply_ai_bg_btn.clicked.connect(self._on_apply_ai_bg)
        self._ai_bg_cached_label = _VisibilityTrackedLabel("(cached)")
        self._ai_bg_cached_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
        self._ai_bg_cached_label.setToolTip("Result loaded from cache — no API call needed")
        self._ai_bg_cached_label.setVisible(False)
        ai_bg_row = QWidget()
        ai_bg_row_layout = QHBoxLayout(ai_bg_row)
        ai_bg_row_layout.setContentsMargins(0, 0, 0, 0)
        ai_bg_row_layout.addWidget(self._ai_bg_check)
        ai_bg_row_layout.addWidget(self._apply_ai_bg_btn)
        ai_bg_row_layout.addWidget(self._ai_bg_cached_label)
        prep_form.addRow(QLabel(""), ai_bg_row)

        # Crop to canvas: resize/crop image to match canvas aspect ratio
        self._crop_to_canvas_check = QCheckBox("Crop to Canvas")
        self._crop_to_canvas_check.setChecked(True)
        self._crop_to_canvas_check.stateChanged.connect(self._on_preprocessing_changed)
        prep_form.addRow(QLabel(""), self._crop_to_canvas_check)

        # ------------------------------------------------------------------
        # Image Size & Position
        # ------------------------------------------------------------------
        self._image_fit_combo = QComboBox()
        self._image_fit_combo.addItems(["Fill Canvas", "Fit (Keep Aspect)", "Custom Size"])
        self._image_fit_combo.setToolTip(
            "Fill Canvas: image fills the entire drawing area (default).\n"
            "Fit (Keep Aspect): scale image to fit within drawing area, preserving aspect ratio.\n"
            "Custom Size: set explicit width and height in mm."
        )
        self._image_fit_combo.currentIndexChanged.connect(self._on_image_fit_mode_changed)
        prep_form.addRow(QLabel("Fit Mode"), self._image_fit_combo)

        # Custom size controls (visible only in Custom Size mode)
        self._custom_size_widget = QWidget()
        custom_size_layout = QFormLayout(self._custom_size_widget)
        custom_size_layout.setContentsMargins(0, 0, 0, 0)

        self._image_width_spin = QDoubleSpinBox()
        self._image_width_spin.setRange(1.0, 2000.0)
        self._image_width_spin.setValue(190.0)
        self._image_width_spin.setSingleStep(1.0)
        self._image_width_spin.setSuffix(" mm")
        self._image_width_spin.setToolTip("Output width in mm")
        self._image_width_spin.valueChanged.connect(self._on_image_width_changed)
        custom_size_layout.addRow(QLabel("Width"), self._image_width_spin)

        self._image_height_spin = QDoubleSpinBox()
        self._image_height_spin.setRange(1.0, 2000.0)
        self._image_height_spin.setValue(277.0)
        self._image_height_spin.setSingleStep(1.0)
        self._image_height_spin.setSuffix(" mm")
        self._image_height_spin.setToolTip("Output height in mm")
        self._image_height_spin.valueChanged.connect(self._on_image_height_changed)
        custom_size_layout.addRow(QLabel("Height"), self._image_height_spin)

        self._lock_aspect_check = QCheckBox("Lock Aspect Ratio")
        self._lock_aspect_check.setChecked(True)
        self._lock_aspect_check.setToolTip("Changing width or height automatically updates the other to maintain aspect ratio")
        custom_size_layout.addRow(QLabel(""), self._lock_aspect_check)

        self._custom_size_widget.setVisible(False)
        prep_form.addRow(self._custom_size_widget)

        # Offset controls (visible in Fit and Custom modes)
        self._image_offset_widget = QWidget()
        offset_layout = QFormLayout(self._image_offset_widget)
        offset_layout.setContentsMargins(0, 0, 0, 0)

        self._image_offset_x_spin = QDoubleSpinBox()
        self._image_offset_x_spin.setRange(-1000.0, 1000.0)
        self._image_offset_x_spin.setValue(0.0)
        self._image_offset_x_spin.setSingleStep(1.0)
        self._image_offset_x_spin.setSuffix(" mm")
        self._image_offset_x_spin.setToolTip("Horizontal offset from centered position (mm)")
        self._image_offset_x_spin.valueChanged.connect(self._on_preprocessing_changed)
        offset_layout.addRow(QLabel("X Offset"), self._image_offset_x_spin)

        self._image_offset_y_spin = QDoubleSpinBox()
        self._image_offset_y_spin.setRange(-1000.0, 1000.0)
        self._image_offset_y_spin.setValue(0.0)
        self._image_offset_y_spin.setSingleStep(1.0)
        self._image_offset_y_spin.setSuffix(" mm")
        self._image_offset_y_spin.setToolTip("Vertical offset from centered position (mm)")
        self._image_offset_y_spin.valueChanged.connect(self._on_preprocessing_changed)
        offset_layout.addRow(QLabel("Y Offset"), self._image_offset_y_spin)

        self._image_offset_widget.setVisible(False)
        prep_form.addRow(self._image_offset_widget)

        self._layout.addWidget(self._preprocessing_group)
        self._preprocessing_group.setVisible(False)

        # ----------------------------------------------------------------
        # Color Separation group (Color Separation mode only)
        # ----------------------------------------------------------------
        self._color_sep_group = QGroupBox("Color Separation")
        color_sep_layout = QVBoxLayout(self._color_sep_group)

        # Method selector (signal connected after all widgets are created to avoid AttributeError)
        method_form = QFormLayout()
        self._color_sep_method_combo = QComboBox()
        self._color_sep_method_combo.addItems(["K-Means", "Luminance", "RGB", "CMYK", "AI Layer Separation"])
        method_form.addRow(QLabel("Method"), self._color_sep_method_combo)

        # num_colors spinner (K-means / Luminance)
        self._color_sep_num_colors_spin = QSpinBox()
        self._color_sep_num_colors_spin.setRange(2, 8)
        self._color_sep_num_colors_spin.setValue(3)
        self._color_sep_num_colors_label = QLabel("Colors / Bands")
        method_form.addRow(self._color_sep_num_colors_label, self._color_sep_num_colors_spin)

        color_sep_layout.addLayout(method_form)

        # Channel checkboxes (RGB / CMYK mode)
        self._channel_check_widget = QWidget()
        channel_layout = QVBoxLayout(self._channel_check_widget)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_checks: dict[str, QCheckBox] = {}
        color_sep_layout.addWidget(self._channel_check_widget)
        self._channel_check_widget.setVisible(False)

        # Line-art algorithm for separated layers
        gen_form = QFormLayout()
        self._color_sep_gen_combo = QComboBox()
        try:
            from plottter.generators import get_generators_by_category
            for gen_cls in get_generators_by_category("image"):
                self._color_sep_gen_combo.addItem(gen_cls.name, gen_cls)
        except ImportError:
            pass
        if self._color_sep_gen_combo.count() == 0:
            self._color_sep_gen_combo.addItem("(no generators)")
        gen_form.addRow(QLabel("Line-Art Algorithm"), self._color_sep_gen_combo)

        # Preset combo for the selected line-art algorithm
        self._color_sep_preset_combo = QComboBox()
        gen_form.addRow(QLabel("Preset"), self._color_sep_preset_combo)

        color_sep_layout.addLayout(gen_form)

        # Connect algorithm change to rebuild preset combo, then populate it
        self._color_sep_gen_combo.currentIndexChanged.connect(
            self._rebuild_color_sep_preset_combo
        )
        self._rebuild_color_sep_preset_combo()

        # Separate and Generate Lines buttons
        self._separate_btn = QPushButton("Separate into Layers")
        self._separate_btn.clicked.connect(self._on_separate)
        color_sep_layout.addWidget(self._separate_btn)

        self._gen_lines_btn = QPushButton("Generate Lines for Separated Layers")
        self._gen_lines_btn.clicked.connect(self._on_generate_lines)
        self._gen_lines_btn.setEnabled(False)
        color_sep_layout.addWidget(self._gen_lines_btn)

        # Progress bar for color sep (separate from main one)
        self._color_sep_progress = QProgressBar()
        self._color_sep_progress.setVisible(False)
        color_sep_layout.addWidget(self._color_sep_progress)

        # Connect method changed signal now that all dependent widgets exist
        self._color_sep_method_combo.currentTextChanged.connect(self._on_color_sep_method_changed)
        # Initialise label/range for the current (default) method
        self._on_color_sep_method_changed(self._color_sep_method_combo.currentText())

        self._layout.addWidget(self._color_sep_group)
        self._color_sep_group.setVisible(False)

        # Internal state for separated layer IDs
        self._separated_layer_ids: list[str] = []
        # In-memory store for numpy arrays (mask, source_image) keyed by layer ID
        # Kept separate from generator_info so the project remains JSON-serializable.
        self._layer_masks: dict[str, tuple] = {}

        # ----------------------------------------------------------------
        # Mask Paint group (Mask Paint mode only)
        # ----------------------------------------------------------------
        self._mask_paint_group = QGroupBox("Mask Paint")
        mask_form = QFormLayout(self._mask_paint_group)

        self._mask_tool_combo = QComboBox()
        self._mask_tool_combo.addItems(["Brush", "Rectangle", "Ellipse", "Polygon", "Pen/Lasso"])
        mask_form.addRow(QLabel("Tool"), self._mask_tool_combo)

        self._brush_size_label = QLabel("Size")
        self._brush_size_spin = QDoubleSpinBox()
        self._brush_size_spin.setRange(0.5, 50.0)
        self._brush_size_spin.setSingleStep(0.5)
        self._brush_size_spin.setValue(5.0)
        self._brush_size_spin.setSuffix(" mm")
        mask_form.addRow(self._brush_size_label, self._brush_size_spin)

        self._brush_hardness_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_hardness_slider.setRange(0, 100)
        self._brush_hardness_slider.setValue(80)
        self._brush_hardness_label = QLabel("80%")
        self._brush_hardness_slider.valueChanged.connect(
            lambda v: self._brush_hardness_label.setText(f"{v}%")
        )
        hardness_row = QWidget()
        hardness_row_layout = QHBoxLayout(hardness_row)
        hardness_row_layout.setContentsMargins(0, 0, 0, 0)
        hardness_row_layout.addWidget(self._brush_hardness_slider)
        hardness_row_layout.addWidget(self._brush_hardness_label)
        self._brush_hardness_form_label = QLabel("Hardness")
        mask_form.addRow(self._brush_hardness_form_label, hardness_row)

        self._erase_check = QCheckBox("Erase Mode")
        mask_form.addRow(QLabel(""), self._erase_check)

        self._mask_target_layer_combo = QComboBox()
        mask_form.addRow(QLabel("Target Layer"), self._mask_target_layer_combo)

        mask_btn_row = QWidget()
        mask_btn_layout = QHBoxLayout(mask_btn_row)
        mask_btn_layout.setContentsMargins(0, 0, 0, 0)
        self._clear_mask_btn = QPushButton("Clear Mask")
        self._invert_mask_btn = QPushButton("Invert Mask")
        self._apply_mask_btn = QPushButton("Apply to Layer")
        mask_btn_layout.addWidget(self._clear_mask_btn)
        mask_btn_layout.addWidget(self._invert_mask_btn)
        mask_btn_layout.addWidget(self._apply_mask_btn)
        mask_form.addRow(mask_btn_row)

        self._mask_status_label = QLabel("No mask painted yet.")
        self._mask_status_label.setWordWrap(True)
        mask_form.addRow(self._mask_status_label)

        self._layout.addWidget(self._mask_paint_group)
        self._mask_paint_group.setVisible(False)

        # ----------------------------------------------------------------
        # Saved Masks group (Mask Paint mode only)
        # ----------------------------------------------------------------
        self._saved_masks_group = QGroupBox("Saved Masks")
        saved_masks_layout = QVBoxLayout(self._saved_masks_group)

        self._mask_list = QListWidget()
        self._mask_list.setMaximumHeight(140)
        self._mask_list.setIconSize(QSize(32, 32))
        saved_masks_layout.addWidget(self._mask_list)

        saved_masks_btn_row = QWidget()
        saved_masks_btn_layout = QHBoxLayout(saved_masks_btn_row)
        saved_masks_btn_layout.setContentsMargins(0, 0, 0, 0)
        self._save_mask_btn = QPushButton("Save Current")
        self._load_mask_btn = QPushButton("Load")
        self._rename_mask_btn = QPushButton("Rename")
        self._delete_mask_btn = QPushButton("Delete")
        saved_masks_btn_layout.addWidget(self._save_mask_btn)
        saved_masks_btn_layout.addWidget(self._load_mask_btn)
        saved_masks_btn_layout.addWidget(self._rename_mask_btn)
        saved_masks_btn_layout.addWidget(self._delete_mask_btn)
        saved_masks_layout.addWidget(saved_masks_btn_row)

        self._layout.addWidget(self._saved_masks_group)
        self._saved_masks_group.setVisible(False)

        # ----------------------------------------------------------------
        # AI Mask Generation group (Mask Paint mode only, requires API key)
        # ----------------------------------------------------------------
        self._ai_mask_group = QGroupBox("AI Mask Generation")
        ai_mask_layout = QVBoxLayout(self._ai_mask_group)

        # Image load row (AI mask uses same _raw_image as image-to-lines)
        ai_img_row = QHBoxLayout()
        self._ai_mask_load_image_btn = QPushButton("Load Image…")
        self._ai_mask_load_image_btn.clicked.connect(self._on_load_image)
        self._ai_mask_image_label = QLabel("No image loaded")
        self._ai_mask_image_label.setWordWrap(True)
        ai_img_row.addWidget(self._ai_mask_load_image_btn)
        ai_img_row.addWidget(self._ai_mask_image_label, stretch=1)
        ai_mask_layout.addLayout(ai_img_row)

        # Mode selector
        ai_mode_form = QFormLayout()
        self._ai_mask_mode_combo = QComboBox()
        self._ai_mask_mode_combo.addItems(["Manual Brush", "Point Prompt", "Box Prompt", "Text Prompt"])
        ai_mode_form.addRow(QLabel("Mode"), self._ai_mask_mode_combo)
        ai_mask_layout.addLayout(ai_mode_form)

        # Instructions label (updated based on mode)
        self._ai_mask_instructions = _VisibilityTrackedLabel(
            "Left-click to mark areas to include.\n"
            "Right-click to mark areas to exclude from the selection."
        )
        self._ai_mask_instructions.setWordWrap(True)
        self._ai_mask_instructions.setStyleSheet("color: #666; font-size: 11px;")
        self._ai_mask_instructions.setVisible(False)
        ai_mask_layout.addWidget(self._ai_mask_instructions)

        # Text prompt input (shown only in Text Prompt mode)
        self._ai_mask_text_input = QLineEdit()
        self._ai_mask_text_input.setPlaceholderText("e.g. 'the person', 'the red car'")
        self._ai_mask_text_input.setVisible(False)
        ai_mask_layout.addWidget(self._ai_mask_text_input)

        # Buttons row
        ai_btn_row = QHBoxLayout()
        self._ai_mask_clear_btn = QPushButton("Clear")
        self._ai_mask_clear_btn.setToolTip("Clear all point/box prompts")
        self._ai_mask_generate_btn = QPushButton("Generate Mask")
        ai_btn_row.addWidget(self._ai_mask_clear_btn)
        ai_btn_row.addWidget(self._ai_mask_generate_btn)
        ai_mask_layout.addLayout(ai_btn_row)

        # Progress and status
        self._ai_mask_progress = QProgressBar()
        self._ai_mask_progress.setVisible(False)
        ai_mask_layout.addWidget(self._ai_mask_progress)

        self._ai_mask_status = QLabel("")
        self._ai_mask_status.setWordWrap(True)
        ai_mask_layout.addWidget(self._ai_mask_status)

        # Mode combo signal (connect after all widgets are built)
        self._ai_mask_mode_combo.currentIndexChanged.connect(self._on_ai_mask_mode_changed)

        self._layout.addWidget(self._ai_mask_group)
        self._ai_mask_group.setVisible(False)

        # ----------------------------------------------------------------
        # Mask Refinement group (Mask Paint mode only)
        # ----------------------------------------------------------------
        self._mask_refine_group = QGroupBox("Mask Refinement")
        refine_form = QFormLayout(self._mask_refine_group)

        self._feather_spin = QDoubleSpinBox()
        self._feather_spin.setRange(0.0, 5.0)
        self._feather_spin.setSingleStep(0.1)
        self._feather_spin.setValue(0.0)
        self._feather_spin.setSuffix(" mm")
        self._feather_spin.setToolTip(
            "Blur the mask edges by this radius — softens the selection boundary"
        )
        refine_form.addRow(QLabel("Feather"), self._feather_spin)

        self._grow_shrink_spin = QDoubleSpinBox()
        self._grow_shrink_spin.setRange(-5.0, 5.0)
        self._grow_shrink_spin.setSingleStep(0.1)
        self._grow_shrink_spin.setValue(0.0)
        self._grow_shrink_spin.setSuffix(" mm")
        self._grow_shrink_spin.setToolTip(
            "Positive = expand mask outward, Negative = contract mask inward"
        )
        refine_form.addRow(QLabel("Grow / Shrink"), self._grow_shrink_spin)

        self._apply_refinement_btn = QPushButton("Apply Refinement")
        refine_form.addRow(self._apply_refinement_btn)

        self._layout.addWidget(self._mask_refine_group)
        self._mask_refine_group.setVisible(False)

        # ----------------------------------------------------------------
        # Shape Drawing group (Shape Drawing mode only)
        # ----------------------------------------------------------------
        self._shape_draw_group = QGroupBox("Shape Drawing")
        sd_form = QFormLayout(self._shape_draw_group)

        self._sd_tool_combo = QComboBox()
        self._sd_tool_combo.addItems(["Rectangle", "Ellipse", "Polygon", "Freehand", "Line/Polyline"])
        sd_form.addRow(QLabel("Tool"), self._sd_tool_combo)

        self._sd_fill_combo = QComboBox()
        self._sd_fill_combo.addItems(["None", "Hatching", "Cross-hatch", "Concentric"])
        sd_form.addRow(QLabel("Fill"), self._sd_fill_combo)

        self._sd_fill_spacing_spin = QDoubleSpinBox()
        self._sd_fill_spacing_spin.setRange(0.1, 5.0)
        self._sd_fill_spacing_spin.setSingleStep(0.1)
        self._sd_fill_spacing_spin.setValue(0.3)
        self._sd_fill_spacing_spin.setSuffix(" mm")
        self._sd_fill_spacing_label = QLabel("Spacing")
        sd_form.addRow(self._sd_fill_spacing_label, self._sd_fill_spacing_spin)

        self._sd_fill_angle_spin = QDoubleSpinBox()
        self._sd_fill_angle_spin.setRange(0.0, 180.0)
        self._sd_fill_angle_spin.setSingleStep(5.0)
        self._sd_fill_angle_spin.setValue(45.0)
        self._sd_fill_angle_spin.setSuffix("°")
        self._sd_fill_angle_label = QLabel("Angle")
        sd_form.addRow(self._sd_fill_angle_label, self._sd_fill_angle_spin)

        self._sd_stroke_check = QCheckBox("Include outline stroke")
        self._sd_stroke_check.setChecked(True)
        sd_form.addRow(QLabel(""), self._sd_stroke_check)

        self._sd_target_layer_combo = QComboBox()
        self._refresh_sd_layer_combo()
        sd_form.addRow(QLabel("Target Layer"), self._sd_target_layer_combo)

        self._sd_smooth_spin = QSpinBox()
        self._sd_smooth_spin.setRange(0, 5)
        self._sd_smooth_spin.setValue(0)
        sd_form.addRow(QLabel("Smooth Passes"), self._sd_smooth_spin)

        self._layout.addWidget(self._shape_draw_group)
        self._shape_draw_group.setVisible(False)

        # Connect fill type combo to show/hide spacing and angle
        self._sd_fill_combo.currentIndexChanged.connect(self._on_sd_fill_changed)
        self._on_sd_fill_changed()

        # 3D Camera controls (shown only in 3D Scene mode)
        self._3d_camera_group = QGroupBox("3D Camera")
        cam_form = QFormLayout(self._3d_camera_group)

        self._cam_azimuth_spin = QDoubleSpinBox()
        self._cam_azimuth_spin.setRange(0.0, 360.0)
        self._cam_azimuth_spin.setSingleStep(5.0)
        self._cam_azimuth_spin.setValue(30.0)
        self._cam_azimuth_spin.setSuffix("°")
        cam_form.addRow(QLabel("Azimuth"), self._cam_azimuth_spin)

        self._cam_elevation_spin = QDoubleSpinBox()
        self._cam_elevation_spin.setRange(-90.0, 90.0)
        self._cam_elevation_spin.setSingleStep(5.0)
        self._cam_elevation_spin.setValue(20.0)
        self._cam_elevation_spin.setSuffix("°")
        cam_form.addRow(QLabel("Elevation"), self._cam_elevation_spin)

        self._cam_distance_spin = QDoubleSpinBox()
        self._cam_distance_spin.setRange(0.1, 50.0)
        self._cam_distance_spin.setSingleStep(0.5)
        self._cam_distance_spin.setValue(8.0)
        cam_form.addRow(QLabel("Distance"), self._cam_distance_spin)

        self._cam_lookat_x_spin = QDoubleSpinBox()
        self._cam_lookat_x_spin.setRange(-20.0, 20.0)
        self._cam_lookat_x_spin.setSingleStep(0.1)
        self._cam_lookat_x_spin.setValue(0.0)
        cam_form.addRow(QLabel("Look-at X"), self._cam_lookat_x_spin)

        self._cam_lookat_y_spin = QDoubleSpinBox()
        self._cam_lookat_y_spin.setRange(-20.0, 20.0)
        self._cam_lookat_y_spin.setSingleStep(0.1)
        self._cam_lookat_y_spin.setValue(0.0)
        cam_form.addRow(QLabel("Look-at Y"), self._cam_lookat_y_spin)

        self._cam_lookat_z_spin = QDoubleSpinBox()
        self._cam_lookat_z_spin.setRange(-20.0, 20.0)
        self._cam_lookat_z_spin.setSingleStep(0.1)
        self._cam_lookat_z_spin.setValue(0.0)
        cam_form.addRow(QLabel("Look-at Z"), self._cam_lookat_z_spin)

        self._cam_fov_spin = QDoubleSpinBox()
        self._cam_fov_spin.setRange(5.0, 120.0)
        self._cam_fov_spin.setSingleStep(5.0)
        self._cam_fov_spin.setValue(45.0)
        self._cam_fov_spin.setSuffix("°")
        cam_form.addRow(QLabel("FOV"), self._cam_fov_spin)

        self._cam_projection_combo = QComboBox()
        self._cam_projection_combo.addItems(["perspective", "orthographic"])
        cam_form.addRow(QLabel("Projection"), self._cam_projection_combo)

        # "3D Preview" toggle button — enables real-time wireframe on canvas
        self._3d_preview_btn = QPushButton("Enable 3D Preview")
        self._3d_preview_btn.setCheckable(True)
        self._3d_preview_btn.setToolTip(
            "Show a real-time wireframe preview in the canvas.\n"
            "Left drag = orbit  |  Middle/Shift+drag = pan  |  Scroll = zoom"
        )
        cam_form.addRow(self._3d_preview_btn)

        # "Import Mesh…" button — convenience shortcut to load OBJ/STL as new layer
        self._import_mesh_btn = QPushButton("Import Mesh File…")
        self._import_mesh_btn.setToolTip(
            "Open an OBJ or STL file and create a new 3D layer using that mesh"
        )
        self._import_mesh_btn.clicked.connect(self._on_import_mesh)
        cam_form.addRow(self._import_mesh_btn)

        # Auto-regenerate checkbox (task 62.2)
        self._auto_regen_3d_cb = QCheckBox("Auto-regenerate other 3D layers")
        self._auto_regen_3d_cb.setToolTip(
            "After generating this layer, automatically regenerate all other 3D Scene\n"
            "layers so cross-layer hidden-line removal stays accurate.\n"
            "Default: off (avoids unexpected slowdowns)."
        )
        from PyQt6.QtCore import QSettings
        self._auto_regen_3d_cb.setChecked(
            QSettings("Plottter", "Plottter").value("3d/auto_regenerate", False, type=bool)
        )
        self._auto_regen_3d_cb.stateChanged.connect(self._on_auto_regen_3d_toggled)
        cam_form.addRow(self._auto_regen_3d_cb)

        self._layout.addWidget(self._3d_camera_group)
        self._3d_camera_group.setVisible(False)

        # Connect camera controls to persist to project metadata and trigger wireframe refresh
        for _cam_spin in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
            self._cam_fov_spin,
        ):
            _cam_spin.valueChanged.connect(self._on_camera_changed)
        self._cam_projection_combo.currentIndexChanged.connect(self._on_camera_changed)

        # Preset dropdown
        self._preset_group = QGroupBox("Preset")
        preset_layout = QVBoxLayout(self._preset_group)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Custom")
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._preset_combo.view().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._preset_combo.view().customContextMenuRequested.connect(self._on_preset_context_menu)
        preset_layout.addWidget(self._preset_combo)
        self._layout.addWidget(self._preset_group)

        # Target layer
        self._layer_group = QGroupBox("Target Layer")
        layer_layout = QVBoxLayout(self._layer_group)
        self._layer_combo = QComboBox()
        self._refresh_layer_combo()
        layer_layout.addWidget(self._layer_combo)
        self._layout.addWidget(self._layer_group)

        # Params container (rebuilt when generator is set)
        self._params_group = QGroupBox("Parameters")
        self._params_form = QFormLayout()
        self._params_group.setLayout(self._params_form)
        self._layout.addWidget(self._params_group)

        # Shared post-generation transforms group
        self._transforms_group = QGroupBox("Post-Generation Transforms")
        transforms_form = QFormLayout(self._transforms_group)

        self._transform_scale_spin = QDoubleSpinBox()
        self._transform_scale_spin.setMinimum(0.01)
        self._transform_scale_spin.setMaximum(100.0)
        self._transform_scale_spin.setSingleStep(0.1)
        self._transform_scale_spin.setValue(1.0)
        self._transform_scale_spin.setDecimals(3)
        transforms_form.addRow(QLabel("Scale"), self._transform_scale_spin)

        self._transform_rotation_spin = QDoubleSpinBox()
        self._transform_rotation_spin.setMinimum(-360.0)
        self._transform_rotation_spin.setMaximum(360.0)
        self._transform_rotation_spin.setSingleStep(1.0)
        self._transform_rotation_spin.setValue(0.0)
        self._transform_rotation_spin.setDecimals(2)
        transforms_form.addRow(QLabel("Rotation (deg)"), self._transform_rotation_spin)

        self._transform_x_spin = QDoubleSpinBox()
        self._transform_x_spin.setMinimum(-500.0)
        self._transform_x_spin.setMaximum(500.0)
        self._transform_x_spin.setSingleStep(1.0)
        self._transform_x_spin.setValue(0.0)
        self._transform_x_spin.setDecimals(2)
        transforms_form.addRow(QLabel("Translate X (mm)"), self._transform_x_spin)

        self._transform_y_spin = QDoubleSpinBox()
        self._transform_y_spin.setMinimum(-500.0)
        self._transform_y_spin.setMaximum(500.0)
        self._transform_y_spin.setSingleStep(1.0)
        self._transform_y_spin.setValue(0.0)
        self._transform_y_spin.setDecimals(2)
        transforms_form.addRow(QLabel("Translate Y (mm)"), self._transform_y_spin)

        self._mirror_h_check = QCheckBox()
        self._mirror_v_check = QCheckBox()
        transforms_form.addRow(QLabel("Mirror Horizontal"), self._mirror_h_check)
        transforms_form.addRow(QLabel("Mirror Vertical"), self._mirror_v_check)

        self._n_fold_spin = QSpinBox()
        self._n_fold_spin.setMinimum(1)
        self._n_fold_spin.setMaximum(12)
        self._n_fold_spin.setValue(1)
        transforms_form.addRow(QLabel("Rotational Symmetry (n)"), self._n_fold_spin)

        self._tile_rows_spin = QSpinBox()
        self._tile_rows_spin.setMinimum(1)
        self._tile_rows_spin.setMaximum(20)
        self._tile_rows_spin.setValue(1)
        self._tile_cols_spin = QSpinBox()
        self._tile_cols_spin.setMinimum(1)
        self._tile_cols_spin.setMaximum(20)
        self._tile_cols_spin.setValue(1)
        transforms_form.addRow(QLabel("Tile Rows"), self._tile_rows_spin)
        transforms_form.addRow(QLabel("Tile Columns"), self._tile_cols_spin)

        self._layout.addWidget(self._transforms_group)

        # Post-Processing group (brush effects, same for all generators)
        self._post_proc_group = QGroupBox("Post-Processing")
        self._post_proc_form = QFormLayout()
        self._post_proc_group.setLayout(self._post_proc_form)

        try:
            from plottter.generators.base import (
                Generator as _Gen,
                FloatParam as _FP,
                IntParam as _IP,
                ChoiceParam as _CP,
            )
            for _pp_param in _Gen.get_post_processing_parameters():
                _pp_label = QLabel(_pp_param.label)
                if isinstance(_pp_param, _FP):
                    _pp_widget: QWidget = QDoubleSpinBox()
                    _pp_widget.setMinimum(_pp_param.min)  # type: ignore[attr-defined]
                    _pp_widget.setMaximum(_pp_param.max)  # type: ignore[attr-defined]
                    _pp_widget.setSingleStep(_pp_param.step)  # type: ignore[attr-defined]
                    _pp_widget.setValue(_pp_param.default)  # type: ignore[attr-defined]
                    _pp_widget.setDecimals(4)  # type: ignore[attr-defined]
                elif isinstance(_pp_param, _IP):
                    _pp_widget = QSpinBox()
                    _pp_widget.setMinimum(_pp_param.min)  # type: ignore[attr-defined]
                    _pp_widget.setMaximum(_pp_param.max)  # type: ignore[attr-defined]
                    _pp_widget.setSingleStep(_pp_param.step)  # type: ignore[attr-defined]
                    _pp_widget.setValue(_pp_param.default)  # type: ignore[attr-defined]
                elif isinstance(_pp_param, _CP):
                    _pp_widget = QComboBox()
                    _pp_widget.addItems(_pp_param.choices)  # type: ignore[attr-defined]
                    _pp_idx = (
                        _pp_param.choices.index(_pp_param.default)
                        if _pp_param.default in _pp_param.choices
                        else 0
                    )
                    _pp_widget.setCurrentIndex(_pp_idx)  # type: ignore[attr-defined]
                    _pp_widget.currentTextChanged.connect(self._update_post_proc_visibility)  # type: ignore[attr-defined]
                else:
                    _pp_widget = QLineEdit(str(getattr(_pp_param, "default", "")))
                if _pp_param.description:
                    _pp_widget.setToolTip(_pp_param.description)
                    _pp_label.setToolTip(_pp_param.description)
                self._post_proc_widgets[_pp_param.name] = _pp_widget
                self._post_proc_labels[_pp_param.name] = _pp_label
                self._post_proc_form.addRow(_pp_label, _pp_widget)
        except ImportError:
            pass

        self._layout.addWidget(self._post_proc_group)
        self._update_post_proc_visibility()

        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._layout.addWidget(self._progress_bar)

        # Cancel button (shown during generation)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._layout.addWidget(self._cancel_btn)

        # Generate / Randomize buttons
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._on_generate)
        self._randomize_btn = QPushButton("Randomize")
        self._randomize_btn.clicked.connect(self._on_randomize)
        self._layout.addWidget(self._generate_btn)
        self._layout.addWidget(self._randomize_btn)
        self._layout.addStretch()

        # Update AI control availability based on current settings
        self.update_ai_availability()

        # Connect controller signals
        self._controller.layer_added.connect(self._refresh_layer_combo)
        self._controller.layer_removed.connect(self._refresh_layer_combo)
        self._controller.layer_changed.connect(self._refresh_layer_combo)
        self._controller.layers_reordered.connect(self._refresh_layer_combo)
        self._controller.project_loaded.connect(self._refresh_layer_combo)
        self._controller.project_loaded.connect(self._on_project_loaded)
        self._controller.masks_changed.connect(self._refresh_mask_list)
        self._controller.project_loaded.connect(self._refresh_mask_list)
        self._controller.active_layer_changed.connect(self._on_active_layer_changed)
        self._controller.paths_changed.connect(self._on_source_layer_paths_changed)
        self._controller.generator_info_changed.connect(self._on_generator_info_changed)
        self._layer_combo.currentIndexChanged.connect(self._refresh_source_layer_combo)

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

    def _refresh_source_layer_combo(self, *_args: Any) -> None:
        """Refresh the rasterize source layer combo, excluding the current target layer."""
        # Get the current target layer id to exclude (prevent self-referencing)
        target_idx = self._layer_combo.currentIndex()
        target_layer_id = self._layer_combo.itemData(target_idx) if target_idx >= 0 else None

        current_src = self._source_layer_combo.currentData()
        self._source_layer_combo.blockSignals(True)
        self._source_layer_combo.clear()
        for layer in self._controller.current_project.layers:
            if layer.id != target_layer_id:
                self._source_layer_combo.addItem(layer.name, layer.id)
        # Restore previous selection if still available
        idx = self._source_layer_combo.findData(current_src)
        if idx >= 0:
            self._source_layer_combo.setCurrentIndex(idx)
        self._source_layer_combo.blockSignals(False)

    def _on_image_source_type_changed(self, checked: bool = True) -> None:
        """Toggle between file, layer, and AI depth map source UI.

        Connected to all 3 radio buttons' toggled signals.  When a button
        becomes *un*checked (checked=False), we skip processing — the handler
        for the button that just became *checked* will run immediately after.
        """
        if not checked:
            return

        # Determine which source type is now active
        if self._src_type_file_radio.isChecked():
            new_type = "file"
        elif self._src_type_layer_radio.isChecked():
            new_type = "layer"
        elif self._src_type_depth_radio.isChecked():
            new_type = "depth_map"
        else:
            return

        prev_type = self._image_source_type
        self._image_source_type = new_type

        self._file_src_widget.setVisible(new_type == "file")
        self._layer_src_widget.setVisible(new_type == "layer")
        self._depth_src_widget.setVisible(new_type == "depth_map")

        if new_type == "layer":
            # Switching to layer mode — refresh the combo and auto-rasterize if possible
            self._refresh_source_layer_combo()
            self._on_rasterize_layer()
        elif new_type == "file":
            # Switching back to file mode — restore original file-based image
            if prev_type == "depth_map" and self._original_raw_image is not None:
                self._raw_image = self._original_raw_image
                self._original_raw_image = None
                self._update_image_preview()
            elif self._image_source_path:
                try:
                    from plottter.io.image_import import load_image
                    self._raw_image = load_image(self._image_source_path)
                    self._update_image_preview()
                except Exception:
                    pass
            else:
                self._raw_image = None
                self._update_image_preview()
        elif new_type == "depth_map":
            # Switching to depth map mode — save original image if needed
            if prev_type == "file" and self._raw_image is not None:
                self._original_raw_image = self._raw_image
            # Check if we already have a cached depth map for this image
            cache_key = self._image_source_path
            if cache_key and cache_key in self._depth_map_cache:
                depth = self._depth_map_cache[cache_key]
                if self._depth_invert_check.isChecked():
                    depth = 1.0 - depth
                self._apply_depth_map(depth)
                self._depth_status_label.setText("Depth map ready (cached)")
            else:
                self._depth_status_label.setText("No depth map generated")

    def _on_source_layer_combo_changed(self, _index: int = 0) -> None:
        """Auto-rasterize when the source layer selection changes."""
        if self._image_source_type == "layer":
            self._on_rasterize_layer()

    def _on_rasterize_layer(self) -> None:
        """Rasterize the selected source layer and use it as the raw image."""
        if self._image_source_type != "layer":
            return

        idx = self._source_layer_combo.currentIndex()
        if idx < 0:
            self._layer_src_status_label.setText("No source layer selected.")
            return

        layer_id = self._source_layer_combo.itemData(idx)
        layer = self._controller.get_layer(layer_id)
        if layer is None:
            self._layer_src_status_label.setText("Source layer not found.")
            return

        if not layer.paths:
            self._layer_src_status_label.setText("Warning: source layer has no paths.")
            return

        canvas = self._controller.current_project.canvas
        dpi = self._rasterize_dpi_spin.value()
        stroke_mm = self._rasterize_stroke_spin.value()

        try:
            from plottter.processing.rasterize import rasterize_layer
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rasterized = rasterize_layer(layer, canvas, resolution_dpi=dpi, stroke_width_mm=stroke_mm)
            if caught:
                self._layer_src_status_label.setText(str(caught[0].message))
            else:
                h, w = rasterized.shape
                self._layer_src_status_label.setText(f"Rasterized: {w}×{h} px")
        except Exception as exc:
            self._layer_src_status_label.setText(f"Error: {exc}")
            return

        self._source_layer_id = layer_id
        self._raw_image = rasterized
        self._ai_bg_rgba = None
        self._update_image_preview()

    def _on_source_layer_paths_changed(self, layer_id: str) -> None:
        """Re-rasterize when the source layer's paths change."""
        if self._image_source_type == "layer" and layer_id == self._source_layer_id:
            self._on_rasterize_layer()

    # ------------------------------------------------------------------
    # AI Depth Map source methods
    # ------------------------------------------------------------------

    def _on_generate_depth_map(self) -> None:
        """Generate a depth map for the currently loaded image via Replicate AI."""
        # Use the original file image as source (not a previously-computed depth map)
        source_image = (
            self._original_raw_image
            if self._original_raw_image is not None
            else self._raw_image
        )
        if source_image is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "No Image Loaded",
                "Please load a source image first (use the File source type to load an image, "
                "then switch to AI Depth Map).",
            )
            return

        # Check in-memory cache first
        cache_key = self._image_source_path
        if cache_key and cache_key in self._depth_map_cache:
            depth = self._depth_map_cache[cache_key]
            if self._depth_invert_check.isChecked():
                depth = 1.0 - depth
            self._apply_depth_map(depth)
            self._depth_status_label.setText("Depth map ready (cached)")
            return

        # Read API key and cache directory from QSettings
        from PyQt6.QtCore import QSettings
        settings = QSettings("Plottter", "Plottter")
        api_key = str(settings.value("replicate/api_key", "") or "")
        raw_cache_dir = (
            settings.value("ai/cache_dir", "") or
            settings.value("ai/depth_cache_dir", "") or ""
        )
        cache_dir: "str | None" = raw_cache_dir.strip() or None
        if cache_dir is None:
            import pathlib
            cache_dir = str(pathlib.Path.home() / ".plottter" / "ai_cache")

        if not api_key:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please configure your Replicate API key in Preferences (Ctrl+,) before "
                "generating a depth map.",
            )
            return

        # Guard against starting a second worker while one is still running
        if self._depth_map_worker is not None and self._depth_map_worker.isRunning():
            return

        self._depth_status_label.setText("Generating…")
        self._gen_depth_btn.setEnabled(False)

        self._depth_map_worker = _DepthMapWorker(api_key, cache_dir, source_image)
        self._depth_map_worker.progress.connect(lambda p: None)  # optional: update status
        self._depth_map_worker.finished.connect(self._on_depth_map_ready)
        self._depth_map_worker.error.connect(self._on_depth_map_error)
        self._depth_map_worker.finished.connect(self._depth_map_worker.deleteLater)
        self._depth_map_worker.error.connect(self._depth_map_worker.deleteLater)
        self._depth_map_worker.start()

    def _on_depth_map_ready(self, depth_map: "np.ndarray") -> None:
        """Called when the depth map worker finishes successfully."""
        if self._depth_map_worker is not None:
            self._depth_map_worker.wait()
            self._depth_map_worker = None
        self._gen_depth_btn.setEnabled(True)
        # Store in in-memory cache before applying inversion
        cache_key = self._image_source_path
        if cache_key:
            self._depth_map_cache[cache_key] = depth_map

        if self._depth_invert_check.isChecked():
            depth_map = 1.0 - depth_map
        self._apply_depth_map(depth_map)
        self._depth_status_label.setText("Depth map ready")

    def _on_depth_map_error(self, error_msg: str) -> None:
        """Called when the depth map worker fails."""
        if self._depth_map_worker is not None:
            self._depth_map_worker.wait()
            self._depth_map_worker = None
        self._gen_depth_btn.setEnabled(True)
        self._depth_status_label.setText(f"Error: {error_msg[:80]}")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "Depth Map Error",
            f"Failed to generate depth map:\n\n{error_msg}",
        )

    def _apply_depth_map(self, depth_map: "np.ndarray") -> None:
        """Convert float32 depth map to 3-channel uint8 and set as the raw image."""
        depth_uint8 = (depth_map * 255.0).clip(0, 255).astype("uint8")
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        self._raw_image = depth_rgb
        self._ai_bg_rgba = None
        self._update_image_preview()

    def _on_depth_invert_changed(self, checked: bool) -> None:
        """Re-apply the depth map with updated inversion when the checkbox is toggled."""
        if self._image_source_type != "depth_map":
            return
        cache_key = self._image_source_path
        if cache_key and cache_key in self._depth_map_cache:
            depth = self._depth_map_cache[cache_key]
            if checked:
                depth = 1.0 - depth
            self._apply_depth_map(depth)
            self._depth_status_label.setText("Depth map ready (inverted)" if checked else "Depth map ready")

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

    # ------------------------------------------------------------------
    # Mask paint handlers
    # ------------------------------------------------------------------

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

        Polylines are split wherever they pass through unpainted regions;
        segments shorter than 2 points are discarded.
        """
        # Import constant at runtime to avoid any circular-import risk
        _MASK_PX_PER_MM = 5  # must match canvas_widget._MASK_PX_PER_MM
        h, w = mask.shape
        result: list = []

        for polyline in paths:
            current_seg: list = []
            for pt in polyline:
                x_mm, y_mm = pt
                px = int(x_mm * _MASK_PX_PER_MM)
                py = int(y_mm * _MASK_PX_PER_MM)
                in_mask = 0 <= px < w and 0 <= py < h and mask[py, px] > 0.5
                if in_mask:
                    current_seg.append(pt)
                else:
                    if len(current_seg) >= 2:
                        result.append(current_seg)
                    current_seg = []
            if len(current_seg) >= 2:
                result.append(current_seg)

        return result

    # ------------------------------------------------------------------
    # AI Mask Generation handlers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # FMM source point pick handlers
    # ------------------------------------------------------------------

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

    def _on_generator_type_changed(self, _index: int = 0) -> None:
        idx = self._generator_type_combo.currentIndex()
        if idx < 0:
            self.set_generator(None)
            return
        gen_cls = self._generator_type_combo.itemData(idx)
        if gen_cls is not None:
            self.set_generator(gen_cls())

    def set_generator(self, generator: Any) -> None:
        """Rebuild the parameter UI for a new generator."""
        # Deactivate FMM pick mode whenever the generator changes
        if self._canvas_ref is not None:
            self._canvas_ref.set_fmm_source_mode(False)
            self._canvas_ref.clear_fmm_source_marker()
        if self._fmm_btn_alive():
            self._pick_fmm_source_btn.setText("Pick on Canvas")  # type: ignore[union-attr]

        self._generator = generator
        self._param_widgets.clear()
        self._param_labels.clear()

        # Clear params form
        while self._params_form.rowCount() > 0:
            self._params_form.removeRow(0)

        # Populate preset combo
        self._rebuild_preset_combo()

        if generator is None:
            return

        # Build param widgets
        try:
            from plottter.generators.base import (
                FloatParam, IntParam, ExpressionParam, ChoiceParam, BoolParam,
                StringParam, FontParam, ImageParam,
            )
            from plottter.gui.widgets.font_picker import FontPicker
        except ImportError:
            FontParam = None  # type: ignore[assignment,misc]
            FontPicker = None  # type: ignore[assignment,misc]

        for param in generator.get_parameters():
            label = QLabel(param.label)
            if isinstance(param, FloatParam):
                widget: QWidget = QDoubleSpinBox()
                widget.setMinimum(param.min)  # type: ignore[attr-defined]
                widget.setMaximum(param.max)  # type: ignore[attr-defined]
                widget.setSingleStep(param.step)  # type: ignore[attr-defined]
                widget.setValue(param.default)  # type: ignore[attr-defined]
                widget.setDecimals(4)  # type: ignore[attr-defined]
            elif isinstance(param, IntParam):
                widget = QSpinBox()
                widget.setMinimum(param.min)  # type: ignore[attr-defined]
                widget.setMaximum(param.max)  # type: ignore[attr-defined]
                widget.setSingleStep(param.step)  # type: ignore[attr-defined]
                widget.setValue(param.default)  # type: ignore[attr-defined]
            elif isinstance(param, StringParam):
                if param.multiline:
                    widget = QPlainTextEdit(str(param.default))
                    widget.setFixedHeight(80)  # type: ignore[attr-defined]
                else:
                    widget = QLineEdit(str(param.default))
            elif isinstance(param, ExpressionParam):
                widget = QLineEdit(str(param.default))  # type: ignore[attr-defined]
            elif isinstance(param, ChoiceParam):
                widget = QComboBox()
                widget.addItems(param.choices)  # type: ignore[attr-defined]
                idx = param.choices.index(param.default) if param.default in param.choices else 0  # type: ignore[attr-defined]
                widget.setCurrentIndex(idx)  # type: ignore[attr-defined]
                # When a choice changes, update visibility of conditional params
                widget.currentTextChanged.connect(self._update_param_visibility)  # type: ignore[attr-defined]
                # When a choice changes, update tooltip if choice_descriptions provided
                if param.choice_descriptions:
                    _choice_descs = param.choice_descriptions

                    def _make_choice_tooltip_updater(combo: QComboBox, descs: dict[str, str]) -> None:
                        def _update_choice_tooltip(text: str) -> None:
                            tip = descs.get(text, "")
                            combo.setToolTip(tip)
                        combo.currentTextChanged.connect(_update_choice_tooltip)
                        # Set initial tooltip
                        _update_choice_tooltip(combo.currentText())

                    _make_choice_tooltip_updater(widget, _choice_descs)  # type: ignore[arg-type]
            elif isinstance(param, BoolParam):
                widget = QCheckBox()
                widget.setChecked(param.default)  # type: ignore[attr-defined]
                # When a bool changes, update visibility of conditional params
                widget.stateChanged.connect(self._update_param_visibility)  # type: ignore[attr-defined]
            elif FontParam is not None and FontPicker is not None and isinstance(param, FontParam):
                widget = FontPicker()
                if param.default:
                    widget.set_font_path(param.default)  # type: ignore[attr-defined]
            elif isinstance(param, ImageParam):
                from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton
                import functools
                container = QWidget()
                row_layout = QHBoxLayout(container)
                row_layout.setContentsMargins(0, 0, 0, 0)
                line_edit = QLineEdit(str(param.default) if param.default else "")
                browse_btn = QPushButton("Browse…")
                browse_btn.setFixedWidth(70)

                def _browse(le: QLineEdit) -> None:
                    path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select Image",
                        "",
                        "Images (*.jpg *.jpeg *.png *.webp *.gif *.bmp *.tiff);;All Files (*)",
                    )
                    if path:
                        le.setText(path)

                browse_btn.clicked.connect(functools.partial(_browse, line_edit))
                row_layout.addWidget(line_edit)
                row_layout.addWidget(browse_btn)
                # Store container reference so _update_param_visibility can
                # hide/show the whole row (QLineEdit + Browse button) together.
                line_edit.setProperty("_image_container", container)
                if param.description:
                    container.setToolTip(param.description)
                    label.setToolTip(param.description)
                self._param_widgets[param.name] = line_edit
                self._param_labels[param.name] = label
                self._params_form.addRow(label, container)
                continue
            else:
                widget = QLineEdit(str(getattr(param, "default", "")))

            # Apply description as tooltip on both the widget and its label.
            # For ChoiceParam with choice_descriptions, the combo tooltip is already
            # set per-choice by _make_choice_tooltip_updater; only set the generic
            # description on the label so we don't overwrite the initial choice tooltip.
            if param.description:
                has_choice_descs = isinstance(param, ChoiceParam) and bool(param.choice_descriptions)
                if not has_choice_descs:
                    widget.setToolTip(param.description)
                label.setToolTip(param.description)

            self._param_widgets[param.name] = widget
            self._param_labels[param.name] = label
            self._params_form.addRow(label, widget)

        # If the generator has fmm_source_x_pct / fmm_source_y_pct parameters, inject
        # a "Pick on Canvas" button so the user can click to set the FMM source point.
        self._pick_fmm_source_btn = None
        self._pick_fmm_source_label = None
        if "fmm_source_x_pct" in self._param_widgets:
            from PyQt6.QtWidgets import QPushButton
            btn = QPushButton("Pick on Canvas")
            btn.setToolTip(
                "Click this button, then click on the image to set the FMM wave origin."
            )
            btn.clicked.connect(self._on_pick_fmm_source_clicked)
            lbl = QLabel("")
            self._pick_fmm_source_btn = btn
            self._pick_fmm_source_label = lbl
            self._params_form.addRow(lbl, btn)

            # Keep the canvas marker in sync when the user edits spinboxes directly.
            x_widget = self._param_widgets.get("fmm_source_x_pct")
            y_widget = self._param_widgets.get("fmm_source_y_pct")
            if isinstance(x_widget, QDoubleSpinBox):
                x_widget.valueChanged.connect(self._update_fmm_marker)
            if isinstance(y_widget, QDoubleSpinBox):
                y_widget.valueChanged.connect(self._update_fmm_marker)

        # Apply initial visibility for params with visible_when conditions
        self._update_param_visibility()

        # For Math Art generators: show image source + preprocessing panels when the
        # generator has any ImageParam (any future math generator with image input gets
        # the full panel automatically).
        if self._current_mode == "Math Art":
            from plottter.generators.base import ImageParam as _ImageParam
            has_image_param = any(
                isinstance(p, _ImageParam) for p in generator.get_parameters()
            )
            self._image_source_group.setVisible(has_image_param)
            self._preprocessing_group.setVisible(has_image_param)

    def _update_param_visibility(self, *_args: Any) -> None:
        """Show/hide parameter rows based on their visible_when conditions."""
        if self._generator is None:
            return
        for param in self._generator.get_parameters():
            if param.visible_when is None:
                continue
            # Param is visible only when ALL conditions are satisfied
            visible = True
            for dep_name, allowed_values in param.visible_when.items():
                dep_widget = self._param_widgets.get(dep_name)
                if dep_widget is None:
                    continue
                if isinstance(dep_widget, QComboBox):
                    current = dep_widget.currentText()
                    if current not in allowed_values:
                        visible = False
                        break
                elif isinstance(dep_widget, QCheckBox):
                    current = dep_widget.isChecked()
                    if current not in allowed_values:
                        visible = False
                        break
            label_w = self._param_labels.get(param.name)
            field_w = self._param_widgets.get(param.name)
            if label_w is not None:
                label_w.setVisible(visible)
            if field_w is not None:
                field_w.setVisible(visible)

        # Show/hide the "Pick on Canvas" button based on fmm_source_point == "Custom"
        if self._fmm_btn_alive():
            source_widget = self._param_widgets.get("fmm_source_point")
            show_btn = (
                isinstance(source_widget, QComboBox)
                and source_widget.currentText() == "Custom"
            )
            self._pick_fmm_source_btn.setVisible(show_btn)  # type: ignore[union-attr]
            if self._pick_fmm_source_label is not None:
                self._pick_fmm_source_label.setVisible(show_btn)  # type: ignore[union-attr]
            if not show_btn and self._canvas_ref is not None:
                self._canvas_ref.set_fmm_source_mode(False)
                self._canvas_ref.clear_fmm_source_marker()

    def _update_post_proc_visibility(self, *_args: Any) -> None:
        """Show/hide post-processing parameter rows based on their visible_when conditions."""
        try:
            from plottter.generators.base import Generator as _Gen
            post_proc_params = _Gen.get_post_processing_parameters()
        except ImportError:
            return
        for param in post_proc_params:
            if param.visible_when is None:
                continue
            visible = True
            for dep_name, allowed_values in param.visible_when.items():
                dep_widget = self._post_proc_widgets.get(dep_name)
                if dep_widget is None:
                    continue
                if isinstance(dep_widget, QComboBox):
                    if dep_widget.currentText() not in allowed_values:
                        visible = False
                        break
            label_w = self._post_proc_labels.get(param.name)
            field_w = self._post_proc_widgets.get(param.name)
            if label_w is not None:
                label_w.setVisible(visible)
            if field_w is not None:
                field_w.setVisible(visible)

    # ------------------------------------------------------------------
    # Preset combo helpers
    # ------------------------------------------------------------------

    _SAVE_PRESET_ACTION = "Save Current as Preset\u2026"  # "Save Current as Preset…"

    def _rebuild_preset_combo(self) -> None:
        """Repopulate the preset combo from the current generator's presets."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem("Custom")
        if self._generator is not None:
            for preset in self._generator.get_presets():
                idx = self._preset_combo.count()
                self._preset_combo.addItem(preset.name)
                self._preset_combo.setItemData(idx, "builtin", Qt.ItemDataRole.UserRole)
                if preset.description:
                    self._preset_combo.setItemData(idx, preset.description, Qt.ItemDataRole.ToolTipRole)

            # Load and cache user presets for this generator.
            try:
                from plottter.presets.user_presets import load_user_presets
                self._user_presets = load_user_presets(self._generator.name)
            except Exception:
                self._user_presets = []

            if self._user_presets:
                self._preset_combo.insertSeparator(self._preset_combo.count())
                self._preset_combo.addItem("— User Presets —")
                # Make the section header non-selectable.
                header_idx = self._preset_combo.count() - 1
                model = self._preset_combo.model()
                if model is not None:
                    header_item = model.item(header_idx)
                    if header_item is not None:
                        from PyQt6.QtCore import Qt as _Qt
                        header_item.setFlags(
                            header_item.flags()
                            & ~_Qt.ItemFlag.ItemIsEnabled
                            & ~_Qt.ItemFlag.ItemIsSelectable
                        )
                for preset in self._user_presets:
                    idx = self._preset_combo.count()
                    self._preset_combo.addItem(preset.name)
                    # Tag item as a user preset for future context-menu support.
                    self._preset_combo.setItemData(idx, "user", Qt.ItemDataRole.UserRole)

            self._preset_combo.insertSeparator(self._preset_combo.count())
            self._preset_combo.addItem(self._SAVE_PRESET_ACTION)
        else:
            self._user_presets = []
        self._preset_combo.blockSignals(False)

    def _gather_current_params(self) -> dict[str, Any]:
        """Collect serialisable parameter values from the current widgets."""
        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FPGather
        except ImportError:
            _FPGather = None  # type: ignore[assignment,misc]

        result: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                result[name] = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                result[name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                sentinel = widget.property("_sentinel")
                result[name] = sentinel if sentinel is not None else widget.text()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif _FPGather is not None and isinstance(widget, _FPGather):
                result[name] = widget.font_path()
        return result

    def _save_current_as_preset(self) -> None:
        """Prompt the user for a name and persist the current params as a user preset."""
        if self._generator is None:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
            return

        name, ok = QInputDialog.getText(
            self,
            "Save Preset",
            "Enter a name for this preset:",
        )
        if not ok or not name.strip():
            # User cancelled — restore combo to "Custom"
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
            return

        name = name.strip()
        params = self._gather_current_params()

        try:
            from plottter.generators.base import Preset
            from plottter.presets.user_presets import save_user_preset
            save_user_preset(self._generator.name, Preset(name=name, params=params))
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save preset: {exc}")
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
            return

        # Refresh the combo to include the newly saved user preset.
        self._rebuild_preset_combo()
        # Select the newly saved preset name if it appears in the combo, else "Custom"
        self._preset_combo.blockSignals(True)
        idx = self._preset_combo.findText(name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        else:
            self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)

    def _on_preset_context_menu(self, pos) -> None:
        """Show a context menu with rename/delete for user preset items."""
        idx = self._preset_combo.view().indexAt(pos).row()
        if idx < 0:
            return
        item_data = self._preset_combo.itemData(idx, Qt.ItemDataRole.UserRole)
        if item_data != "user":
            return
        preset_name = self._preset_combo.itemText(idx)
        if not preset_name:
            return

        menu = QMenu(self)
        rename_action = menu.addAction("Rename Preset")
        delete_action = menu.addAction("Delete Preset")
        chosen = menu.exec(self._preset_combo.view().mapToGlobal(pos))
        if chosen is rename_action:
            self._rename_user_preset_action(preset_name)
        elif chosen is delete_action:
            self._delete_user_preset_action(preset_name)

    def _delete_user_preset_action(self, preset_name: str) -> None:
        """Ask for confirmation and delete a user preset."""
        if self._generator is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from plottter.presets.user_presets import delete_user_preset
            delete_user_preset(self._generator.name, preset_name)
        except Exception as exc:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete preset: {exc}")
            return
        self._rebuild_preset_combo()
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)

    def _rename_user_preset_action(self, old_name: str) -> None:
        """Prompt the user for a new name and rename a user preset."""
        if self._generator is None:
            return
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Preset",
            "Enter a new name for this preset:",
            text=old_name,
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == old_name:
            return
        try:
            from plottter.presets.user_presets import rename_user_preset
            rename_user_preset(self._generator.name, old_name, new_name)
        except Exception as exc:
            QMessageBox.warning(self, "Rename Failed", f"Could not rename preset: {exc}")
            return
        self._rebuild_preset_combo()
        self._preset_combo.blockSignals(True)
        idx = self._preset_combo.findText(new_name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        else:
            self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)

    def _on_preset_changed(self, preset_name: str) -> None:
        if preset_name == self._SAVE_PRESET_ACTION:
            self._save_current_as_preset()
            return
        if preset_name == "Custom":
            # Reset all expression fields to editable, clear sentinel values, and
            # restore default text so users don't see the "(ODE — not editable)" placeholder.
            if self._generator is not None:
                try:
                    from plottter.generators.base import ExpressionParam
                    param_defaults = {
                        p.name: p.default
                        for p in self._generator.get_parameters()
                        if isinstance(p, ExpressionParam)
                    }
                except ImportError:
                    param_defaults = {}
            else:
                param_defaults = {}
            for name, widget in self._param_widgets.items():
                if isinstance(widget, QLineEdit):
                    if widget.property("_sentinel") is not None:
                        widget.setReadOnly(False)
                        widget.setProperty("_sentinel", None)
                        widget.setPlaceholderText("")
                        if name in param_defaults:
                            widget.setText(str(param_defaults[name]))
            return
        if self._generator is None:
            return
        for preset in self._generator.get_presets():
            if preset.name == preset_name:
                self._apply_preset_params(preset.params)
                return
        # Check user presets if no built-in preset matched.
        for preset in self._user_presets:
            if preset.name == preset_name:
                self._apply_preset_params(preset.params)
                return

    def _apply_preset_params(self, params: dict[str, Any]) -> None:
        # Reset all expression fields to editable before applying preset values.
        for widget in self._param_widgets.values():
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(False)

        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FPPreset
        except ImportError:
            _FPPreset = None  # type: ignore[assignment,misc]

        for name, value in params.items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QLineEdit):
                str_val = str(value)
                # Sentinel values (e.g. "__lorenz__") indicate ODE-driven params
                # that are not user-editable. Show a human-readable placeholder.
                if str_val.startswith("__") and str_val.endswith("__"):
                    widget.setText("(ODE — not editable)")
                    widget.setReadOnly(True)
                    widget.setProperty("_sentinel", str_val)
                else:
                    widget.setText(str_val)
                    widget.setProperty("_sentinel", None)
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif _FPPreset is not None and isinstance(widget, _FPPreset):
                widget.set_font_path(str(value))
        self._update_param_visibility()

    def apply_generator_preset(self, gen_cls: type, preset_name: str) -> None:
        """Switch to the given generator and apply the named preset.

        Can be called externally (e.g. from PresetGalleryDialog) while the
        settings panel is in Math Art mode.
        """
        # Select the generator in the type combo if it's listed there
        for i in range(self._generator_type_combo.count()):
            if self._generator_type_combo.itemData(i) is gen_cls:
                self._generator_type_combo.setCurrentIndex(i)
                break
        else:
            # Generator not in combo (e.g. mode mismatch) — force-set it
            self.set_generator(gen_cls())
        # Apply the named preset
        idx = self._preset_combo.findText(preset_name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    @property
    def current_mode(self) -> str:
        """The currently active mode string (e.g. '3D Scene', 'Math Art')."""
        return self._current_mode

    def trigger_generate(self) -> None:
        """Public entry point — can be called from a menu action."""
        self._on_generate()

    def trigger_randomize(self) -> None:
        """Public entry point — can be called from a menu action."""
        self._on_randomize()

    def trigger_surprise_me(self) -> None:
        """Pick a random math generator, randomize its params, and generate."""
        import random
        try:
            from plottter.generators import get_generators_by_category
        except ImportError:
            return
        generators = get_generators_by_category("math")
        if not generators:
            return
        gen_cls = random.choice(generators)
        # Find and select this generator in the combo (if currently in Math Art mode)
        for i in range(self._generator_type_combo.count()):
            if self._generator_type_combo.itemData(i) is gen_cls:
                self._generator_type_combo.setCurrentIndex(i)
                break
        else:
            # Force-set the generator regardless of combo state
            self.set_generator(gen_cls())
        self._on_randomize()
        self._on_generate()

    def _on_generate(self) -> None:
        # Flush current UI state to model before reading sibling layers' generator_info
        self.flush_current_snapshot()

        if self._generator is None:
            QMessageBox.warning(
                self,
                "No Generator",
                "Please select a mode and generator first.",
            )
            return

        layer_id = self.current_layer_id()
        if layer_id is None:
            QMessageBox.warning(
                self,
                "No Target Layer",
                "Please add a layer to the project before generating.",
            )
            return

        params = self.get_params()
        canvas = self._controller.current_project.canvas

        from plottter.gui.generator_worker import GeneratorWorker

        if self._worker is not None and self._worker.isRunning() and not self._worker.is_cancelled():
            return  # already running

        # Inject sibling shapes for 3D HLR occlusion
        from plottter.generators.scene3d_generator import Scene3DGenerator
        if isinstance(self._generator, Scene3DGenerator):
            params["_sibling_3d_shapes"] = self._build_sibling_3d_shapes(layer_id)

        self._worker = GeneratorWorker(self._generator, params, canvas)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(lambda paths: self._on_generation_finished(paths, layer_id))
        self._worker.metadata_ready.connect(
            lambda meta: self._on_generation_metadata(meta, layer_id)
        )
        self._worker.error.connect(self._on_generation_error)
        self._worker.finished.connect(self._cleanup_generation_ui)
        self._worker.error.connect(self._cleanup_generation_ui)

        self._generate_btn.setEnabled(False)
        self._randomize_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)

        self._worker.start()

    def _on_progress(self, percent: int) -> None:
        self._progress_bar.setValue(percent)

    def _on_generation_finished(self, paths: list, layer_id: str) -> None:
        paths = self._apply_shared_transforms(paths)
        # Apply brush post-processing if a brush type is selected
        brush_widget = self._post_proc_widgets.get("brush_type")
        if isinstance(brush_widget, QComboBox):
            brush_type = brush_widget.currentText()
            if brush_type and brush_type != "None":
                brush_params: dict[str, Any] = {}
                for _bname, _bwidget in self._post_proc_widgets.items():
                    if _bname == "brush_type":
                        continue
                    if isinstance(_bwidget, (QDoubleSpinBox, QSpinBox)):
                        brush_params[_bname] = _bwidget.value()
                    elif isinstance(_bwidget, QComboBox):
                        brush_params[_bname] = _bwidget.currentText()
                try:
                    from plottter.processing.brush import apply_brush
                    paths = apply_brush(paths, brush_type, brush_params)
                except Exception:
                    pass
        self._controller.set_layer_paths(layer_id, paths, "Generate")

        # Auto-regenerate other 3D layers if enabled (task 62.2)
        if (
            self._current_mode == "3D Scene"
            and self._auto_regen_3d_cb.isChecked()
        ):
            self._trigger_auto_regen_siblings(layer_id)

    def _on_generation_metadata(self, meta: dict, source_layer_id: str) -> None:
        """Handle side-channel metadata emitted by GeneratorWorker after generation.

        The auto-created depth map preview layer was removed in task 16.57 — the
        depth map is now a first-class image source and is visible as the canvas
        overlay rather than a separate layer.  This handler is kept as a no-op so
        the GeneratorWorker.metadata_ready signal still has a valid connection.
        """

    # ------------------------------------------------------------------
    # Auto-regenerate other 3D layers (task 62.2)
    # ------------------------------------------------------------------

    def _on_auto_regen_3d_toggled(self, _state: int) -> None:
        """Persist the auto-regenerate checkbox state to QSettings."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("3d/auto_regenerate", self._auto_regen_3d_cb.isChecked())

    def _trigger_auto_regen_siblings(self, generated_layer_id: str) -> None:
        """Start sequential regeneration of all 3D layers *except* the one just generated."""
        # Guard: don't start a new chain while one is already in progress
        if self._auto_regen_layers:
            return

        try:
            project = self._controller.current_project
        except Exception:
            return
        if project is None:
            return

        siblings = [
            layer for layer in project.layers
            if layer.id != generated_layer_id
            and isinstance(layer.generator_info, dict)
            and layer.generator_info.get("mode") == "3D Scene"
        ]
        if not siblings:
            return

        n = len(siblings)
        self._auto_regen_layers = siblings
        self._auto_regen_idx = 0

        # Show status message
        mw = self.window()
        if hasattr(mw, "statusBar"):
            mw.statusBar().showMessage(
                f"Auto-regenerating {n} other 3D layer{'s' if n != 1 else ''}…"
            )

        self._start_auto_regen_next()

    def _start_auto_regen_next(self) -> None:
        """Start (or continue) the auto-regen chain for sibling 3D layers."""
        if self._auto_regen_idx >= len(self._auto_regen_layers):
            self._finish_auto_regen()
            return

        layer = self._auto_regen_layers[self._auto_regen_idx]
        info = layer.generator_info
        params = dict(info.get("params", {}))

        # Inject shared camera
        try:
            project = self._controller.current_project
        except Exception:
            self._finish_auto_regen()
            return
        cam = project.metadata.get("scene3d_camera", {})
        if cam:
            params["_camera"] = cam

        # Inject sibling shapes for HLR occlusion
        params["_sibling_3d_shapes"] = self._build_sibling_3d_shapes(layer.id)

        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.gui.generator_worker import GeneratorWorker

        generator = Scene3DGenerator()
        canvas = project.canvas
        layer_id = layer.id

        worker = GeneratorWorker(generator, params, canvas, parent=self)

        def on_finished(paths: list, lid: str = layer_id) -> None:
            self._controller.set_layer_paths(lid, paths, "Auto-regenerate 3D Layer")
            self._auto_regen_idx += 1
            self._start_auto_regen_next()
            worker.deleteLater()

        def on_error(_msg: str) -> None:
            # Skip failed layer and continue
            self._auto_regen_idx += 1
            self._start_auto_regen_next()
            worker.deleteLater()

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._auto_regen_worker = worker
        worker.start()

    def _finish_auto_regen(self) -> None:
        """Called when auto-regen chain completes."""
        n = len(self._auto_regen_layers)
        done = min(self._auto_regen_idx, n)
        mw = self.window()
        if hasattr(mw, "statusBar"):
            mw.statusBar().showMessage(
                f"Auto-regenerated {done} 3D layer{'s' if done != 1 else ''} successfully.",
                5000,
            )
        self._auto_regen_layers = []
        self._auto_regen_idx = 0
        self._auto_regen_worker = None

    # ------------------------------------------------------------------
    # 3D Camera helpers
    # ------------------------------------------------------------------

    def _get_camera_dict(self) -> dict:
        """Return the current camera settings as a plain dict."""
        return {
            "azimuth": self._cam_azimuth_spin.value(),
            "elevation": self._cam_elevation_spin.value(),
            "distance": self._cam_distance_spin.value(),
            "look_at_x": self._cam_lookat_x_spin.value(),
            "look_at_y": self._cam_lookat_y_spin.value(),
            "look_at_z": self._cam_lookat_z_spin.value(),
            "fov": self._cam_fov_spin.value(),
            "projection": self._cam_projection_combo.currentText(),
        }

    def _on_camera_changed(self) -> None:
        """Persist current camera settings to project metadata whenever a spin/combo changes."""
        if self._controller is None:
            return
        try:
            project = self._controller.current_project
        except Exception:
            return
        if project is None:
            return
        project.metadata["scene3d_camera"] = self._get_camera_dict()
        # If 3D preview is active, debounce a wireframe refresh
        if (
            self._canvas_ref is not None
            and self._3d_preview_btn.isChecked()
            and self._current_mode == "3D Scene"
        ):
            self._wireframe_timer.start()

    def _load_camera_from_project(self) -> None:
        """Populate camera controls from project metadata (called on project load)."""
        if self._controller is None:
            return
        try:
            project = self._controller.current_project
        except Exception:
            return
        if project is None:
            return
        cam = project.metadata.get("scene3d_camera", {})
        # Block signals so we don't trigger _on_camera_changed during restore
        for widget in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
            self._cam_fov_spin,
        ):
            widget.blockSignals(True)
        self._cam_projection_combo.blockSignals(True)

        self._cam_azimuth_spin.setValue(float(cam.get("azimuth", 30.0)))
        self._cam_elevation_spin.setValue(float(cam.get("elevation", 20.0)))
        self._cam_distance_spin.setValue(float(cam.get("distance", 8.0)))
        self._cam_lookat_x_spin.setValue(float(cam.get("look_at_x", 0.0)))
        self._cam_lookat_y_spin.setValue(float(cam.get("look_at_y", 0.0)))
        self._cam_lookat_z_spin.setValue(float(cam.get("look_at_z", 0.0)))
        self._cam_fov_spin.setValue(float(cam.get("fov", 45.0)))
        proj_text = cam.get("projection", "perspective")
        idx = self._cam_projection_combo.findText(proj_text)
        if idx >= 0:
            self._cam_projection_combo.setCurrentIndex(idx)

        for widget in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
            self._cam_fov_spin,
        ):
            widget.blockSignals(False)
        self._cam_projection_combo.blockSignals(False)

    def _build_sibling_3d_shapes(self, current_layer_id: str) -> list:
        """Collect transformed Shape objects from all other 3D Scene layers for HLR occlusion."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        shapes: list = []
        try:
            project = self._controller.current_project
        except Exception:
            return shapes
        if project is None:
            return shapes

        gen = Scene3DGenerator()
        for layer in project.layers:
            if layer.id == current_layer_id:
                continue
            info = layer.generator_info
            if not isinstance(info, dict):
                continue
            if info.get("mode") != "3D Scene":
                continue
            params = info.get("params", {})
            try:
                shape = gen.build_transformed_shape(params)
                if shape is not None:
                    shapes.append(shape)
            except Exception:
                pass  # skip broken sibling layers silently

        return shapes

    # ------------------------------------------------------------------
    # 3D preview event handlers
    # ------------------------------------------------------------------

    def _on_3d_preview_toggled(self, checked: bool) -> None:
        """Toggle real-time 3D wireframe preview on the canvas."""
        if self._canvas_ref is None:
            return
        self._3d_preview_btn.setText(
            "Disable 3D Preview" if checked else "Enable 3D Preview"
        )
        self._canvas_ref.set_3d_preview_active(checked)
        if checked:
            # Sync canvas camera state from spinboxes, then kick off a wireframe render
            self._canvas_ref.update_3d_camera(
                azimuth=self._cam_azimuth_spin.value(),
                elevation=self._cam_elevation_spin.value(),
                distance=self._cam_distance_spin.value(),
                lookat=(
                    self._cam_lookat_x_spin.value(),
                    self._cam_lookat_y_spin.value(),
                    self._cam_lookat_z_spin.value(),
                ),
            )
            self._wireframe_timer.start()
        else:
            self._wireframe_timer.stop()
            if self._wireframe_worker is not None:
                self._wireframe_worker.cancel()
                self._wireframe_worker.wait()
                self._wireframe_worker = None

    def _on_canvas_camera_orbit_changed(self, az: float, el: float, dist: float) -> None:
        """Sync canvas orbit drag result back to settings panel spinboxes."""
        # Block signals to avoid re-triggering _on_camera_changed while updating
        for spin in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
        ):
            spin.blockSignals(True)
        self._cam_azimuth_spin.setValue(az)
        self._cam_elevation_spin.setValue(el)
        self._cam_distance_spin.setValue(dist)
        for spin in (
            self._cam_azimuth_spin,
            self._cam_elevation_spin,
            self._cam_distance_spin,
        ):
            spin.blockSignals(False)
        # Persist and refresh
        self._on_camera_changed()

    def _on_canvas_camera_pan_changed(self, lx: float, ly: float, lz: float) -> None:
        """Sync canvas pan (look-at) result back to settings panel spinboxes."""
        for spin in (
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
        ):
            spin.blockSignals(True)
        self._cam_lookat_x_spin.setValue(lx)
        self._cam_lookat_y_spin.setValue(ly)
        self._cam_lookat_z_spin.setValue(lz)
        for spin in (
            self._cam_lookat_x_spin,
            self._cam_lookat_y_spin,
            self._cam_lookat_z_spin,
        ):
            spin.blockSignals(False)
        self._on_camera_changed()

    def _on_canvas_projection_toggle(self) -> None:
        """Toggle projection combo between perspective and orthographic."""
        current = self._cam_projection_combo.currentText()
        new_text = "orthographic" if current == "perspective" else "perspective"
        idx = self._cam_projection_combo.findText(new_text)
        if idx >= 0:
            self._cam_projection_combo.setCurrentIndex(idx)

    def _start_wireframe_worker(self) -> None:
        """Render all 3D layers without HLR for the live wireframe preview."""
        if self._canvas_ref is None or not self._3d_preview_btn.isChecked():
            return
        if self._current_mode != "3D Scene":
            return
        try:
            project = self._controller.current_project
        except Exception:  # noqa: BLE001
            return
        if project is None:
            return

        canvas = project.canvas
        cam_dict = self._get_camera_dict()

        # Collect params from all 3D Scene layers
        layer_params_list: list[dict] = []
        current_layer_id = self.current_layer_id()
        for layer in project.layers:
            if layer.id == current_layer_id:
                # Use live params from the settings panel for the active layer
                snapshot = self._get_settings_snapshot()
                if snapshot is not None and snapshot.get("mode") == "3D Scene":
                    layer_params_list.append(dict(snapshot.get("params", {})))
                continue
            info = layer.generator_info
            if not isinstance(info, dict):
                continue
            if info.get("mode") != "3D Scene":
                continue
            params = dict(info.get("params", {}))
            layer_params_list.append(params)

        # If a previous worker exists, cancel and wait for it to stop
        if self._wireframe_worker is not None:
            self._wireframe_worker.cancel()
            try:
                self._wireframe_worker.result_ready.disconnect()
                self._wireframe_worker.render_error.disconnect()
            except Exception:  # noqa: BLE001
                pass
            if self._wireframe_worker.isRunning():
                # Give it a moment to finish; if still running, re-queue and bail
                if not self._wireframe_worker.wait(50):
                    self._wireframe_timer.start()
                    return
            self._wireframe_worker = None

        worker = _WireframeWorker(
            layer_params_list=layer_params_list,
            camera_dict=cam_dict,
            canvas_w_mm=canvas.width_mm,
            canvas_h_mm=canvas.height_mm,
        )
        worker.result_ready.connect(self._on_wireframe_finished)
        worker.render_error.connect(self._on_wireframe_error)
        self._wireframe_worker = worker
        worker.start()

    def _on_wireframe_finished(self, polylines: list) -> None:
        """Receive rendered wireframe polylines and push them to the canvas."""
        # Worker ref kept alive until thread done — don't None it until wait() confirms
        if self._wireframe_worker is not None:
            self._wireframe_worker.wait()
            self._wireframe_worker = None
        if self._canvas_ref is not None and self._3d_preview_btn.isChecked():
            self._canvas_ref.set_3d_wireframe_polylines(polylines)

    def _on_wireframe_error(self, error_msg: str) -> None:
        """Handle a wireframe render error (preview is best-effort; no dialog shown)."""
        if self._wireframe_worker is not None:
            self._wireframe_worker.wait()
            self._wireframe_worker = None

    def _on_import_mesh(self) -> None:
        """Open an OBJ or STL file and create a new 3D Scene layer using that mesh."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import 3D Mesh",
            "",
            "3D Mesh Files (*.obj *.stl);;OBJ Files (*.obj);;STL Files (*.stl);;All Files (*)",
        )
        if not file_path:
            return

        try:
            project = self._controller.current_project
        except Exception:  # noqa: BLE001
            return
        if project is None:
            return

        import os
        from plottter.models import Layer

        name = os.path.splitext(os.path.basename(file_path))[0]
        layer = Layer(name=name or "Mesh", color="#3264C8")
        layer.generator_info = {
            "mode": "3D Scene",
            "generator": "3D Scene",
            "params": {
                "shape_type": "Mesh Import",
                "mesh_file": file_path,
                "mesh_all_edges": False,
            },
        }
        self._controller.add_layer(layer)

        # Refresh wireframe if preview is active
        if self._3d_preview_btn.isChecked():
            self._wireframe_timer.start()

    def _apply_shared_transforms(self, paths: list) -> list:
        """Apply scale, rotation, translate, mirror, rotational symmetry, and tiling to paths."""
        import math

        canvas = self._controller.current_project.canvas
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        # Scale (around canvas center)
        scale = self._transform_scale_spin.value()
        if scale != 1.0:
            paths = [
                [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in path]
                for path in paths
            ]

        # Rotation (around canvas center)
        rot_deg = self._transform_rotation_spin.value()
        if rot_deg != 0.0:
            theta = math.radians(rot_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            paths = [
                [
                    (
                        cx + (x - cx) * cos_t - (y - cy) * sin_t,
                        cy + (x - cx) * sin_t + (y - cy) * cos_t,
                    )
                    for x, y in path
                ]
                for path in paths
            ]

        # Translate
        tx = self._transform_x_spin.value()
        ty = self._transform_y_spin.value()
        if tx != 0.0 or ty != 0.0:
            paths = [[(x + tx, y + ty) for x, y in path] for path in paths]

        # Mirror Horizontal (flip around vertical center axis)
        if self._mirror_h_check.isChecked():
            mirrored = [[(2.0 * cx - x, y) for x, y in path] for path in paths]
            paths = list(paths) + mirrored

        # Mirror Vertical (flip around horizontal center axis)
        if self._mirror_v_check.isChecked():
            mirrored = [[(x, 2.0 * cy - y) for x, y in path] for path in paths]
            paths = list(paths) + mirrored

        # Rotational n-fold symmetry
        n_fold = self._n_fold_spin.value()
        if n_fold > 1:
            original = list(paths)
            for k in range(1, n_fold):
                angle = 2.0 * math.pi * k / n_fold
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                rotated = [
                    [
                        (
                            cx + (x - cx) * cos_a - (y - cy) * sin_a,
                            cy + (x - cx) * sin_a + (y - cy) * cos_a,
                        )
                        for x, y in path
                    ]
                    for path in original
                ]
                paths = paths + rotated

        # Tile repeat
        tile_rows = self._tile_rows_spin.value()
        tile_cols = self._tile_cols_spin.value()
        if tile_rows > 1 or tile_cols > 1:
            draw_w = draw_x2 - draw_x1
            draw_h = draw_y2 - draw_y1
            cell_w = draw_w / tile_cols
            cell_h = draw_h / tile_rows

            # Scale original paths to fit one tile cell (centered at first cell center)
            if paths:
                all_pts = [pt for path in paths for pt in path]
                if all_pts:
                    xs = [p[0] for p in all_pts]
                    ys = [p[1] for p in all_pts]
                    content_w = (max(xs) - min(xs)) or 1.0
                    content_h = (max(ys) - min(ys)) or 1.0
                    scale_factor = min(cell_w / content_w, cell_h / content_h) * 0.9
                    pcx = (min(xs) + max(xs)) / 2.0
                    pcy = (min(ys) + max(ys)) / 2.0

                    tiled: list = []
                    for row in range(tile_rows):
                        for col in range(tile_cols):
                            tile_cx = draw_x1 + (col + 0.5) * cell_w
                            tile_cy = draw_y1 + (row + 0.5) * cell_h
                            for path in paths:
                                new_path = [
                                    (
                                        (x - pcx) * scale_factor + tile_cx,
                                        (y - pcy) * scale_factor + tile_cy,
                                    )
                                    for x, y in path
                                ]
                                tiled.append(new_path)
                    paths = tiled

        return paths

    def _on_generation_error(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "Generation Error",
            f"An error occurred during generation:\n\n{message}",
        )

    def _cleanup_generation_ui(self, *_args: Any) -> None:
        self._generate_btn.setEnabled(True)
        self._randomize_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._cleanup_generation_ui()

    # ------------------------------------------------------------------
    # Color Separation
    # ------------------------------------------------------------------

    def _on_color_sep_method_changed(self, method: str) -> None:
        is_kmeans = method == "K-Means"
        is_lum = method == "Luminance"
        is_rgb = method == "RGB"
        is_cmyk = method == "CMYK"
        is_ai = method == "AI Layer Separation"

        self._color_sep_num_colors_spin.setVisible(is_kmeans or is_lum or is_ai)
        self._color_sep_num_colors_label.setVisible(is_kmeans or is_lum or is_ai)
        if is_kmeans:
            self._color_sep_num_colors_spin.setRange(2, 8)
            self._color_sep_num_colors_label.setText("Colors")
        elif is_lum:
            self._color_sep_num_colors_spin.setRange(2, 5)
            self._color_sep_num_colors_label.setText("Bands")
        elif is_ai:
            self._color_sep_num_colors_spin.setRange(2, 8)
            self._color_sep_num_colors_label.setText("Segments")

        # Build channel checkboxes
        self._channel_check_widget.setVisible(is_rgb or is_cmyk)
        layout = self._channel_check_widget.layout()
        # Clear existing checkboxes
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._channel_checks.clear()

        if is_rgb:
            for ch in ("Red", "Green", "Blue"):
                cb = QCheckBox(ch)
                cb.setChecked(True)
                layout.addWidget(cb)
                self._channel_checks[ch] = cb
        elif is_cmyk:
            for ch in ("Cyan", "Magenta", "Yellow", "Key (Black)"):
                cb = QCheckBox(ch)
                cb.setChecked(True)
                layout.addWidget(cb)
                self._channel_checks[ch] = cb

    def _rebuild_color_sep_preset_combo(self) -> None:
        """Rebuild the color separation preset combo based on the selected generator."""
        self._color_sep_preset_combo.blockSignals(True)
        self._color_sep_preset_combo.clear()

        # Always add "Default" as first item with None data
        self._color_sep_preset_combo.addItem("Default", None)

        # Get the currently selected generator class
        gen_cls = self._color_sep_gen_combo.currentData()
        if gen_cls is None:
            self._color_sep_preset_combo.blockSignals(False)
            return

        try:
            # Instantiate the generator to get its presets
            gen_instance = gen_cls()
            presets = gen_instance.get_presets()

            # Add built-in presets
            for preset in presets:
                self._color_sep_preset_combo.addItem(preset.name, preset.params)

            # Load and add user presets
            try:
                from plottter.presets.user_presets import load_user_presets

                user_presets = load_user_presets(gen_cls.name)
                if user_presets:
                    self._color_sep_preset_combo.insertSeparator(
                        self._color_sep_preset_combo.count()
                    )
                    self._color_sep_preset_combo.addItem("— User Presets —")
                    # Make the section header non-selectable
                    header_idx = self._color_sep_preset_combo.count() - 1
                    model = self._color_sep_preset_combo.model()
                    if model is not None:
                        header_item = model.item(header_idx)
                        if header_item is not None:
                            header_item.setFlags(
                                header_item.flags()
                                & ~Qt.ItemFlag.ItemIsEnabled
                                & ~Qt.ItemFlag.ItemIsSelectable
                            )
                    for user_preset in user_presets:
                        self._color_sep_preset_combo.addItem(
                            user_preset.name, user_preset.params
                        )
            except Exception:
                pass  # User presets are optional; ignore failures

        except Exception:
            pass  # If generator instantiation fails, just show Default

        self._color_sep_preset_combo.blockSignals(False)

    def _on_ai_bg_changed(self, state: int) -> None:
        """Handle AI Background Removal toggle: disable manual BG removal when AI is on."""
        ai_on = bool(state)
        self._remove_bg_check.setEnabled(not ai_on)
        if ai_on:
            self._remove_bg_check.setChecked(False)
            self._bg_tolerance_spin.setEnabled(False)
        # Enable Apply button only when checkbox is on and API key is available
        self._apply_ai_bg_btn.setEnabled(ai_on and self._ai_key_available)
        self._on_preprocessing_changed()

    def update_ai_availability(self) -> None:
        """Enable/disable AI controls based on whether a Replicate API key is configured."""
        try:
            from PyQt6.QtCore import QSettings
            from plottter.ai.replicate_client import ReplicateClient
            settings = QSettings("Plottter", "Plottter")
            api_key = settings.value("replicate/api_key", "") or ""
            client = ReplicateClient(api_key=api_key)
            ai_available = client.is_available()
        except Exception:
            ai_available = False

        _no_key_tip = "Enter a Replicate API key in Preferences > AI Integration to enable"

        self._ai_key_available = ai_available
        has_cached_bg = self._ai_bg_rgba is not None

        # Update cached indicator visibility
        self._ai_bg_cached_label.setVisible(has_cached_bg)

        if ai_available:
            self._ai_bg_check.setEnabled(True)
            self._ai_bg_check.setToolTip("")
            self._apply_ai_bg_btn.setEnabled(self._ai_bg_check.isChecked())
            # AI mask generation — disabled in Manual Brush mode since no AI call is needed
            is_manual_mode = self._ai_mask_mode_combo.currentText() == "Manual Brush"
            self._ai_mask_generate_btn.setEnabled(not is_manual_mode)
            self._ai_mask_generate_btn.setToolTip("")
        else:
            # When no API key, allow enabling the checkbox if a cached result is available
            # so the user can activate BG removal without an API call.
            if has_cached_bg:
                self._ai_bg_check.setEnabled(True)
                self._ai_bg_check.setToolTip(
                    "Cached result available — no API key needed to use it"
                )
            else:
                self._ai_bg_check.setChecked(False)
                self._ai_bg_check.setEnabled(False)
                self._ai_bg_check.setToolTip(_no_key_tip)
            self._apply_ai_bg_btn.setEnabled(False)
            # AI mask generation
            self._ai_mask_generate_btn.setEnabled(False)
            self._ai_mask_generate_btn.setToolTip(_no_key_tip)

    def _on_apply_ai_bg(self) -> None:
        """Start a background thread to call AI background removal on the current image."""
        if self._raw_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        if self._ai_bg_worker is not None and self._ai_bg_worker.isRunning():
            return  # already running

        from PyQt6.QtCore import QSettings

        settings = QSettings("Plottter", "Plottter")
        api_key = settings.value("replicate/api_key", "") or ""

        source_img = self._raw_image
        if source_img.ndim == 2:
            source_img = np.stack([source_img] * 3, axis=-1)
        elif source_img.ndim == 3 and source_img.shape[2] == 4:
            source_img = source_img[:, :, :3]

        cache_dir = self._get_cache_dir()
        self._apply_ai_bg_btn.setEnabled(False)
        self._ai_bg_worker = _AiBgWorker(api_key=api_key, image=source_img, cache_dir=cache_dir)
        self._ai_bg_worker.finished.connect(self._on_ai_bg_result)
        self._ai_bg_worker.error.connect(self._on_ai_bg_error)
        self._ai_bg_worker.finished.connect(
            lambda _: self._apply_ai_bg_btn.setEnabled(self._ai_key_available and self._ai_bg_check.isChecked())
        )
        self._ai_bg_worker.error.connect(
            lambda _: self._apply_ai_bg_btn.setEnabled(self._ai_key_available and self._ai_bg_check.isChecked())
        )
        self._ai_bg_worker.start()

    def _on_ai_bg_result(self, rgba: "np.ndarray") -> None:
        """Store the AI background removal result and refresh the preview."""
        self._ai_bg_rgba = rgba
        self._ai_bg_cached_label.setVisible(True)
        self._update_image_preview()

    def _on_ai_bg_error(self, msg: str) -> None:
        QMessageBox.critical(self, "AI Background Removal Error", msg)

    def _on_separate(self) -> None:
        """Run color separation and create one layer per cluster/channel."""
        if self._raw_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            from plottter.io.image_import import preprocess
            params = self._get_preprocessing_params()
            # If AI BG removal is active, composite onto white before
            # preprocessing — same logic as _update_image_preview().
            source = self._raw_image
            if (
                self._ai_bg_check.isChecked()
                and self._ai_bg_rgba is not None
            ):
                rgba = self._ai_bg_rgba
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                rgb = rgba[:, :, :3].astype(np.float32)
                white = np.full_like(rgb, 255.0)
                source = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
            preprocessed = preprocess(source, params)
        except Exception as exc:
            QMessageBox.critical(self, "Preprocessing Error", str(exc))
            return

        method = self._color_sep_method_combo.currentText()
        num = self._color_sep_num_colors_spin.value()

        if method == "AI Layer Separation":
            # Network call — run in a background QThread to keep the GUI responsive.
            from PyQt6.QtCore import QSettings
            from plottter.ai.replicate_client import ReplicateClient

            settings = QSettings("Plottter", "Plottter")
            api_key = settings.value("replicate/api_key", "") or ""
            client = ReplicateClient(api_key=api_key)
            if not client.is_available():
                QMessageBox.warning(
                    self,
                    "AI Unavailable",
                    "AI Layer Separation requires a Replicate API key.\n"
                    "Set your Replicate API key in Preferences > AI Integration.",
                )
                return

            source_img = source
            if source_img.ndim == 2:
                source_img = np.stack([source_img] * 3, axis=-1)
            elif source_img.ndim == 3 and source_img.shape[2] == 4:
                source_img = source_img[:, :, :3]

            # Store preprocessed so the finished callback can use it for mask association
            self._ai_sep_preprocessed = preprocessed

            self._separate_btn.setEnabled(False)
            self._color_sep_progress.setMaximum(0)  # indeterminate while waiting for AI
            self._color_sep_progress.setVisible(True)

            self._ai_segment_worker = _AiSegmentWorker(
                api_key=api_key, image=source_img, num_segments=num
            )
            self._ai_segment_worker.progress.connect(
                lambda p: self._color_sep_progress.setValue(p)
            )
            self._ai_segment_worker.finished.connect(
                lambda results: self._on_ai_segment_finished(results, method)
            )
            self._ai_segment_worker.error.connect(self._on_ai_segment_error)
            self._ai_segment_worker.start()
            return  # layer creation happens asynchronously in _on_ai_segment_finished

        try:
            if method == "K-Means":
                from plottter.color import kmeans_separate
                # K-Means requires an RGB image; apply only spatial transforms
                # (crop/resize) to the raw image, not grayscale conversion or
                # threshold — those would destroy the color information.
                spatial_params = {
                    k: v for k, v in params.items()
                    if k in ("crop_width", "crop_height")
                }
                raw_rgb = preprocess(source, spatial_params)
                if raw_rgb.ndim == 2:
                    raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
                elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                    raw_rgb = raw_rgb[:, :, :3]
                results = kmeans_separate(raw_rgb, num_colors=num)
                layer_names = [f"Cluster {i + 1}" for i in range(len(results))]
            elif method == "Luminance":
                from plottter.color import luminance_separate
                results = luminance_separate(preprocessed, num_bands=num)
                band_names = ["Shadows", "Midtones", "Highlights", "Highlights 2", "Highlights 3"]
                layer_names = [band_names[i] if i < len(band_names) else f"Band {i + 1}" for i in range(len(results))]
            elif method == "RGB":
                from plottter.color import rgb_separate
                # RGB/CMYK separation requires an RGB image, not the
                # grayscale-preprocessed one.  Use source (with BG removal applied).
                raw_rgb = source
                if raw_rgb.ndim == 2:
                    raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
                elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                    raw_rgb = raw_rgb[:, :, :3]
                results = rgb_separate(raw_rgb)
                layer_names = ["Red Channel", "Green Channel", "Blue Channel"]
                channel_names = ["Red", "Green", "Blue"]
                filtered = []
                filtered_names = []
                for i, (mask, color) in enumerate(results):
                    ch = channel_names[i]
                    if ch not in self._channel_checks or self._channel_checks[ch].isChecked():
                        filtered.append((mask, color))
                        filtered_names.append(layer_names[i])
                results = filtered
                layer_names = filtered_names
            elif method == "CMYK":
                from plottter.color import cmyk_separate
                # CMYK separation requires an RGB image.
                raw_rgb = source
                if raw_rgb.ndim == 2:
                    raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
                elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                    raw_rgb = raw_rgb[:, :, :3]
                results = cmyk_separate(raw_rgb)
                layer_names = ["Cyan Channel", "Magenta Channel", "Yellow Channel", "Key (Black) Channel"]
                channel_names_list = ["Cyan", "Magenta", "Yellow", "Key (Black)"]
                filtered = []
                filtered_names = []
                for i, (mask, color) in enumerate(results):
                    ch = channel_names_list[i]
                    if ch not in self._channel_checks or self._channel_checks[ch].isChecked():
                        filtered.append((mask, color))
                        filtered_names.append(layer_names[i])
                results = filtered
                layer_names = filtered_names
            else:
                return
        except Exception as exc:
            QMessageBox.critical(self, "Separation Error", str(exc))
            return

        self._apply_separation_results(results, layer_names, method, preprocessed)

    def _on_ai_segment_finished(
        self, results: list, method: str
    ) -> None:
        """Called on the main thread when the AI segmentation worker succeeds."""
        self._separate_btn.setEnabled(True)
        self._color_sep_progress.setMaximum(100)
        self._color_sep_progress.setVisible(False)

        layer_names = [f"AI Segment {i + 1}" for i in range(len(results))]
        preprocessed = self._ai_sep_preprocessed
        self._ai_sep_preprocessed = None
        self._apply_separation_results(results, layer_names, method, preprocessed)

    def _on_ai_segment_error(self, msg: str) -> None:
        """Called on the main thread when the AI segmentation worker fails."""
        self._separate_btn.setEnabled(True)
        self._color_sep_progress.setMaximum(100)
        self._color_sep_progress.setVisible(False)
        self._ai_sep_preprocessed = None
        QMessageBox.critical(self, "AI Segmentation Error", msg)

    def _apply_separation_results(
        self,
        results: list,
        layer_names: list,
        method: str,
        preprocessed: "np.ndarray",
    ) -> None:
        """Create layers from separation results (called from both sync and async paths)."""
        # Remove previous separation layers before creating new ones
        self._controller.undo_stack.beginMacro("Separate Into Layers")
        for old_lid in list(self._separated_layer_ids):
            self._controller.remove_layer(old_lid)
            self._layer_masks.pop(old_lid, None)
        self._separated_layer_ids.clear()

        from plottter.models import Layer
        for (mask, hex_color), lname in zip(results, layer_names):
            display_name = f"{lname} — {hex_color}"
            layer = Layer(
                name=display_name,
                color=hex_color,
                generator_info={
                    "type": "color_separation",
                    "method": method,
                },
            )
            added = self._controller.add_layer(layer)
            self._separated_layer_ids.append(added.id)
            self._layer_masks[added.id] = (mask, preprocessed)
        self._controller.undo_stack.endMacro()

        self._gen_lines_btn.setEnabled(len(self._separated_layer_ids) > 0)
        QMessageBox.information(
            self,
            "Color Separation",
            f"Created {len(self._separated_layer_ids)} layer(s) from color separation.",
        )

    def _on_generate_lines(self) -> None:
        """Generate line art for each separated layer using the selected algorithm."""
        if not self._separated_layer_ids:
            return

        idx = self._color_sep_gen_combo.currentIndex()
        if idx < 0:
            return
        gen_cls = self._color_sep_gen_combo.itemData(idx)
        if gen_cls is None:
            return

        canvas = self._controller.current_project.canvas

        # Gather layers with masks
        layers_to_process: list[tuple[str, object, object]] = []
        for lid in self._separated_layer_ids:
            if lid not in self._layer_masks:
                continue
            mask, src_img = self._layer_masks[lid]
            layers_to_process.append((lid, mask, src_img))

        if not layers_to_process:
            return

        from plottter.gui.generator_worker import GeneratorWorker

        self._gen_lines_btn.setEnabled(False)
        self._color_sep_progress.setMaximum(len(layers_to_process))
        self._color_sep_progress.setValue(0)
        self._color_sep_progress.setVisible(True)

        self._lines_queue = list(layers_to_process)
        self._lines_done = 0
        self._lines_canvas = canvas
        self._lines_gen_cls = gen_cls
        self._lines_worker: object = None
        self._controller.undo_stack.beginMacro("Generate Lines")
        self._process_next_lines_layer()

    def _process_next_lines_layer(self) -> None:
        if not self._lines_queue:
            self._color_sep_progress.setVisible(False)
            self._gen_lines_btn.setEnabled(True)
            self._controller.undo_stack.endMacro()
            return

        layer_id, mask, src_img = self._lines_queue.pop(0)
        import numpy as np

        # Determine grayscale image to feed the generator
        if mask.dtype == np.bool_:
            # K-Means / Luminance: boolean mask — apply it to the source image
            if src_img.ndim == 3:
                from plottter.io.image_import import to_grayscale
                gray = to_grayscale(src_img)
            else:
                gray = src_img.copy()
            masked_gray = gray.copy()
            masked_gray[~mask] = 255  # pixels outside the cluster → white
        else:
            # RGB / CMYK: mask IS the grayscale channel image (uint8)
            masked_gray = mask.copy()

        gen = self._lines_gen_cls()

        # Check if a preset is selected in the color sep preset combo
        preset_params = self._color_sep_preset_combo.currentData()
        if preset_params is not None:
            # Use preset params as base (copy to avoid mutation)
            gen_params: dict = dict(preset_params)
        else:
            # Default: build params from generator defaults
            gen_params = {}
            for p in gen.get_parameters():
                if hasattr(p, "default"):
                    gen_params[p.name] = p.default

        # Always set _source_image and image placement params regardless of preset
        gen_params["_source_image"] = masked_gray
        gen_params["image_fit_mode"] = self._image_fit_mode()
        fit_mode = gen_params["image_fit_mode"]
        if fit_mode == "custom":
            gen_params["image_width_mm"] = self._image_width_spin.value()
            gen_params["image_height_mm"] = self._image_height_spin.value()
        if fit_mode != "fill":
            gen_params["image_offset_x_mm"] = self._image_offset_x_spin.value()
            gen_params["image_offset_y_mm"] = self._image_offset_y_spin.value()

        from plottter.gui.generator_worker import GeneratorWorker
        worker = GeneratorWorker(gen, gen_params, self._lines_canvas)

        def on_finished(paths, lid=layer_id):
            self._controller.set_layer_paths(lid, paths, "Generate Lines")
            self._lines_done += 1
            self._color_sep_progress.setValue(self._lines_done)
            self._process_next_lines_layer()

        def on_error(msg):
            QMessageBox.warning(self, "Generate Lines Error", msg)
            self._lines_done += 1
            self._color_sep_progress.setValue(self._lines_done)
            self._process_next_lines_layer()

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._lines_worker = worker
        worker.start()

    def _on_randomize(self) -> None:
        """Randomize all parameter values within their ranges."""
        import random
        if self._generator is None:
            return
        try:
            from plottter.generators.base import FloatParam, IntParam, ChoiceParam, BoolParam
        except ImportError:
            return

        for param in self._generator.get_parameters():
            if not param.randomizable:
                continue
            widget = self._param_widgets.get(param.name)
            if widget is None:
                continue
            if isinstance(param, FloatParam) and isinstance(widget, QDoubleSpinBox):
                widget.setValue(random.uniform(param.min, param.max))
            elif isinstance(param, IntParam) and isinstance(widget, QSpinBox):
                widget.setValue(random.randint(param.min, param.max))
            elif isinstance(param, ChoiceParam) and isinstance(widget, QComboBox):
                widget.setCurrentIndex(random.randrange(widget.count()))
            elif isinstance(param, BoolParam) and isinstance(widget, QCheckBox):
                widget.setChecked(random.choice([True, False]))

    # ------------------------------------------------------------------
    # Image source and preprocessing
    # ------------------------------------------------------------------

    def _get_cache_dir(self) -> str:
        """Return the AI disk cache directory path (creates default path if not configured)."""
        import pathlib
        from PyQt6.QtCore import QSettings

        settings = QSettings("Plottter", "Plottter")
        raw_cache_dir = (
            settings.value("ai/cache_dir", "") or
            settings.value("ai/depth_cache_dir", "") or ""
        )
        cache_dir = raw_cache_dir.strip() or str(pathlib.Path.home() / ".plottter" / "ai_cache")
        return cache_dir

    def _check_ai_cache_for_image(self, image: "np.ndarray", path: str) -> None:
        """Pre-load AI results from disk cache for *image* without applying them.

        Populates ``self._ai_bg_rgba`` if a cached BG-removal result exists, and
        ``self._depth_map_cache[path]`` if a cached depth map exists.  Updates the
        UI indicators accordingly.  Does NOT auto-apply or auto-enable checkboxes.
        """
        import hashlib
        import os

        cache_dir = self._get_cache_dir()
        img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # --- BG removal cache ---
        bg_cache_path = os.path.join(cache_dir, "bg_removal", f"{img_hash}.png")
        if os.path.exists(bg_cache_path):
            try:
                from PIL import Image as _PIL_Image

                pil = _PIL_Image.open(bg_cache_path).convert("RGBA")
                result = np.array(pil)
                if result.shape[:2] == image.shape[:2]:
                    self._ai_bg_rgba = result
                    self._ai_bg_cached_label.setVisible(True)
            except Exception:
                pass

        # --- Depth map cache ---
        flat_path = os.path.join(cache_dir, f"{img_hash}.png")
        subdir_path = os.path.join(cache_dir, "depth", f"{img_hash}.png")
        depth_cache_path = flat_path if os.path.exists(flat_path) else subdir_path
        if os.path.exists(depth_cache_path):
            try:
                from PIL import Image as _PIL_Image

                pil = _PIL_Image.open(depth_cache_path)
                arr = np.array(pil).astype(np.float32)
                if arr.max() > 1.0:
                    arr = arr / 65535.0
                if arr.shape == tuple(image.shape[:2]):
                    self._depth_map_cache[path] = arr
                    self._depth_status_label.setText("Depth map ready (cached)")
            except Exception:
                pass

    def _on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.gif);;All Files (*)",
        )
        if not path:
            return
        try:
            from plottter.io.image_import import load_image

            self._raw_image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Image Load Error", str(exc))
            return

        self._image_source_path = path

        # Invalidate any cached AI background removal result for the previous image.
        # Also uncheck the AI BG checkbox so that cached results are NOT auto-applied
        # when the preview renders — the user must explicitly re-enable it.
        self._ai_bg_rgba = None
        self._ai_bg_cached_label.setVisible(False)
        self._ai_bg_check.blockSignals(True)
        self._ai_bg_check.setChecked(False)
        self._ai_bg_check.blockSignals(False)
        self._depth_status_label.setText("No depth map generated")

        # Pre-load any existing AI cache results for this image (without auto-applying)
        self._check_ai_cache_for_image(self._raw_image, path)

        # Update the AI BG checkbox enabled state in case cached result availability changed
        self.update_ai_availability()

        # Update custom size spinboxes to match the canvas drawing area by default
        self._reset_image_size_to_canvas()

        import os

        self._image_filename_label.setText(os.path.basename(path))
        self._update_ai_mask_image_label()
        self._update_image_preview()

    def _on_preprocessing_changed(self, *_args: Any) -> None:
        self._gamma_val_label.setText(f"{self._gamma_slider.value() / 100:.2f}")
        self._preprocess_timer.start()

    def _reset_image_size_to_canvas(self) -> None:
        """Set custom size spinboxes to match the canvas drawing area dimensions."""
        try:
            canvas = self._controller.current_project.canvas
            draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
            self._image_width_spin.blockSignals(True)
            self._image_height_spin.blockSignals(True)
            self._image_width_spin.setValue(round(draw_x2 - draw_x1, 1))
            self._image_height_spin.setValue(round(draw_y2 - draw_y1, 1))
            self._image_width_spin.blockSignals(False)
            self._image_height_spin.blockSignals(False)
        except AttributeError:
            pass

    def _on_image_fit_mode_changed(self, _index: int = 0) -> None:
        """Show/hide custom size and offset controls based on fit mode."""
        mode = self._image_fit_mode()
        is_custom = mode == "custom"
        is_fill = mode == "fill"
        self._custom_size_widget.setVisible(is_custom)
        self._image_offset_widget.setVisible(not is_fill)
        # Hide crop-to-canvas when not in fill mode since explicit sizing handles it
        self._crop_to_canvas_check.setVisible(is_fill)
        self._on_preprocessing_changed()

    def _image_fit_mode(self) -> str:
        """Return the current fit mode as a string: 'fill', 'fit', or 'custom'."""
        text = self._image_fit_combo.currentText()
        if text == "Fit (Keep Aspect)":
            return "fit"
        if text == "Custom Size":
            return "custom"
        return "fill"

    def _on_image_width_changed(self, value: float) -> None:
        """If lock aspect ratio is checked, update height proportionally."""
        if self._lock_aspect_check.isChecked() and self._raw_image is not None:
            h_px, w_px = self._raw_image.shape[:2]
            if w_px > 0:
                aspect = h_px / w_px
                self._image_height_spin.blockSignals(True)
                self._image_height_spin.setValue(round(value * aspect, 2))
                self._image_height_spin.blockSignals(False)
        self._on_preprocessing_changed()

    def _on_image_height_changed(self, value: float) -> None:
        """If lock aspect ratio is checked, update width proportionally."""
        if self._lock_aspect_check.isChecked() and self._raw_image is not None:
            h_px, w_px = self._raw_image.shape[:2]
            if h_px > 0:
                aspect = w_px / h_px
                self._image_width_spin.blockSignals(True)
                self._image_width_spin.setValue(round(value * aspect, 2))
                self._image_width_spin.blockSignals(False)
        self._on_preprocessing_changed()

    def _get_preprocessing_params(self) -> dict:
        params: dict[str, Any] = {}
        brightness = self._bright_slider.value()
        if brightness != 0:
            params["brightness"] = brightness
        contrast = self._contrast_slider.value()
        if contrast != 0:
            params["contrast"] = contrast
        gamma = self._gamma_slider.value() / 100.0
        if abs(gamma - 1.0) > 1e-6:
            params["gamma"] = gamma
        blur = self._blur_slider.value()
        if blur > 0:
            params["blur"] = float(blur)
        if self._threshold_check.isChecked():
            params["threshold"] = float(self._threshold_slider.value())
        if self._invert_check.isChecked():
            params["invert"] = True
        if self._remove_bg_check.isChecked():
            params["remove_background"] = float(self._bg_tolerance_spin.value())
        # ai_bg_removal is handled directly in _update_image_preview() via _ai_bg_rgba cache
        # Crop to canvas is skipped when using a rasterized layer as source: the rasterized
        # image already covers exactly the drawing area and has the correct aspect ratio/content.
        # Applying crop_to_aspect would shift or scale the content, breaking coordinate alignment.
        # Also skip when not in "Fill Canvas" mode since explicit sizing handles mapping.
        fit_mode = self._image_fit_mode()
        if (
            self._crop_to_canvas_check.isChecked()
            and self._image_source_type != "layer"
            and fit_mode == "fill"
        ):
            canvas = self._controller.current_project.canvas
            params["crop_width"] = canvas.width_mm * 5
            params["crop_height"] = canvas.height_mm * 5

        # Image size & position params (used by generators via compute_image_rect)
        params["image_fit_mode"] = fit_mode
        if fit_mode != "fill":
            params["image_offset_x_mm"] = self._image_offset_x_spin.value()
            params["image_offset_y_mm"] = self._image_offset_y_spin.value()
        if fit_mode == "custom":
            params["image_width_mm"] = self._image_width_spin.value()
            params["image_height_mm"] = self._image_height_spin.value()
        return params

    def _update_image_preview(self) -> None:
        if self._raw_image is None:
            self._preprocessed_image = None
            self._thumbnail_label.clear()
            self.image_preprocessed.emit(None)
            self.image_rect_changed.emit(None)
            return

        try:
            from plottter.io.image_import import preprocess, to_grayscale

            params = self._get_preprocessing_params()
            # If AI BG removal is active and we have a cached RGBA result, composite
            # onto white to produce an RGB base image before normal preprocessing.
            source = self._raw_image
            if (
                self._ai_bg_check.isChecked()
                and self._ai_bg_rgba is not None
            ):
                rgba = self._ai_bg_rgba
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                rgb = rgba[:, :, :3].astype(np.float32)
                white = np.full_like(rgb, 255.0)
                source = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
            preprocessed = preprocess(source, params)
            gray = to_grayscale(preprocessed)
            self._preprocessed_image = gray
        except Exception as exc:
            QMessageBox.warning(self, "Preprocessing Error", str(exc))
            return

        # Update thumbnail
        self._update_thumbnail(self._preprocessed_image)
        # Notify canvas of new image and its placement rect
        self.image_preprocessed.emit(self._preprocessed_image)
        self._emit_image_rect()

    def _emit_image_rect(self) -> None:
        """Compute and emit the mm rect where the image overlay should be drawn."""
        gray = self._preprocessed_image
        if gray is None:
            self.image_rect_changed.emit(None)
            return
        from plottter.generators._helpers import compute_image_rect
        canvas = self._controller.current_project.canvas
        margin = canvas.margin_mm
        draw_x1 = margin
        draw_y1 = margin
        draw_x2 = canvas.width_mm - margin
        draw_y2 = canvas.height_mm - margin
        h, w = gray.shape[:2]
        fit_mode = self._image_fit_mode()
        custom_w = self._image_width_spin.value() if fit_mode == "custom" else None
        custom_h = self._image_height_spin.value() if fit_mode == "custom" else None
        offset_x = self._image_offset_x_spin.value() if fit_mode != "fill" else 0.0
        offset_y = self._image_offset_y_spin.value() if fit_mode != "fill" else 0.0
        rect = compute_image_rect(
            fit_mode=fit_mode,
            image_w_px=w,
            image_h_px=h,
            draw_x1=draw_x1,
            draw_y1=draw_y1,
            draw_x2=draw_x2,
            draw_y2=draw_y2,
            custom_w_mm=custom_w,
            custom_h_mm=custom_h,
            offset_x_mm=offset_x,
            offset_y_mm=offset_y,
        )
        self.image_rect_changed.emit(rect)

    def _update_thumbnail(self, gray: np.ndarray) -> None:
        from PyQt6.QtGui import QImage

        arr = np.ascontiguousarray(gray)
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        label_w = self._thumbnail_label.width() or 200
        label_h = self._thumbnail_label.height() or 120
        scaled = pixmap.scaled(
            label_w,
            label_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Params with image injection
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, Any]:
        """Collect current parameter values from the widgets."""
        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FontPicker
        except ImportError:
            _FontPicker = None  # type: ignore[assignment,misc]

        result: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                result[name] = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                result[name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                sentinel = widget.property("_sentinel")
                result[name] = sentinel if sentinel is not None else widget.text()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif _FontPicker is not None and isinstance(widget, _FontPicker):
                result[name] = widget.font_path()

        # Inject preprocessed image for image generators
        if self._current_mode == "Image to Lines" and self._preprocessed_image is not None:
            result["_source_image"] = self._preprocessed_image
            result.update(self._get_preprocessing_params())

        # Inject shared camera for 3D Scene generators
        if self._current_mode == "3D Scene":
            result["_camera"] = self._get_camera_dict()

        return result

    def current_layer_id(self) -> str | None:
        """Return the currently selected target layer id."""
        idx = self._layer_combo.currentIndex()
        if idx >= 0:
            return self._layer_combo.itemData(idx)
        return None

    def flush_current_snapshot(self) -> None:
        """Save current UI state to the active layer's generator_info."""
        snapshot = self._get_settings_snapshot()
        layer_id = self.current_layer_id()
        if snapshot is not None and layer_id:
            self._controller.set_layer_generator_info(layer_id, snapshot)

    # ------------------------------------------------------------------
    # Per-layer generator settings memory
    # ------------------------------------------------------------------

    def _get_settings_snapshot(self) -> dict | None:
        """Capture the current generator type, params, and transforms as a dict.

        Returns None if there is no active generator (e.g. Color Separation mode).
        """
        if self._generator is None:
            return None
        if self._current_mode not in ("Math Art", "Image to Lines", "3D Scene"):
            return None

        gen_name = self._generator_type_combo.currentText()

        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FontPickerSnap
        except ImportError:
            _FontPickerSnap = None  # type: ignore[assignment,misc]

        params: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                params[name] = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                params[name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                sentinel = widget.property("_sentinel")
                params[name] = sentinel if sentinel is not None else widget.text()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif _FontPickerSnap is not None and isinstance(widget, _FontPickerSnap):
                params[name] = widget.font_path()

        transforms = {
            "scale": self._transform_scale_spin.value(),
            "rotation": self._transform_rotation_spin.value(),
            "translate_x": self._transform_x_spin.value(),
            "translate_y": self._transform_y_spin.value(),
            "mirror_h": self._mirror_h_check.isChecked(),
            "mirror_v": self._mirror_v_check.isChecked(),
            "n_fold": self._n_fold_spin.value(),
            "tile_rows": self._tile_rows_spin.value(),
            "tile_cols": self._tile_cols_spin.value(),
        }

        snapshot: dict = {
            "generator_name": gen_name,
            "mode": self._current_mode,
            "params": params,
            "transforms": transforms,
            "image_source_type": self._image_source_type,
        }
        # Persist depth map invert state alongside the source type
        try:
            snapshot["depth_map_invert"] = self._depth_invert_check.isChecked()
        except AttributeError:
            pass

        # Persist image size & position settings
        try:
            snapshot["image_fit_mode"] = self._image_fit_mode()
            snapshot["image_width_mm"] = self._image_width_spin.value()
            snapshot["image_height_mm"] = self._image_height_spin.value()
            snapshot["image_offset_x_mm"] = self._image_offset_x_spin.value()
            snapshot["image_offset_y_mm"] = self._image_offset_y_spin.value()
            snapshot["image_lock_aspect"] = self._lock_aspect_check.isChecked()
        except AttributeError:
            pass

        # Persist post-processing (brush) settings
        post_proc_params: dict[str, Any] = {}
        for _ppname, _ppwidget in self._post_proc_widgets.items():
            if isinstance(_ppwidget, (QDoubleSpinBox, QSpinBox)):
                post_proc_params[_ppname] = _ppwidget.value()
            elif isinstance(_ppwidget, QComboBox):
                post_proc_params[_ppname] = _ppwidget.currentText()
        if post_proc_params:
            snapshot["post_processing"] = post_proc_params

        return snapshot

    def _apply_settings_snapshot(self, info: dict) -> None:
        """Apply a saved generator settings snapshot to the UI.

        Restores the mode (via mode_change_requested signal), generator type,
        parameter values, and shared transforms.
        """
        mode = info.get("mode", "")
        gen_name = info.get("generator_name", "")
        params = info.get("params", {})
        transforms = info.get("transforms", {})

        # Switch mode if needed (ModePanel listens to mode_change_requested)
        if mode and mode != self._current_mode:
            self.mode_change_requested.emit(mode)
            # on_mode_changed() is called synchronously (direct connection),
            # which resets the generator combo to the first item.

        # Select the saved generator by name.
        # Always rebuild the parameter UI even if the same generator is already
        # selected — the parameter values differ per layer.
        if gen_name:
            idx = self._generator_type_combo.findText(gen_name)
            if idx >= 0:
                self._generator_type_combo.blockSignals(True)
                self._generator_type_combo.setCurrentIndex(idx)
                self._generator_type_combo.blockSignals(False)
                self._on_generator_type_changed()

        # Restore parameter values
        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FPApply
        except ImportError:
            _FPApply = None  # type: ignore[assignment,misc]

        for name, value in params.items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QComboBox):
                combo_idx = widget.findText(str(value))
                if combo_idx >= 0:
                    widget.setCurrentIndex(combo_idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif _FPApply is not None and isinstance(widget, _FPApply):
                widget.set_font_path(str(value))

        # Restore shared transform values
        if "scale" in transforms:
            self._transform_scale_spin.setValue(transforms["scale"])
        if "rotation" in transforms:
            self._transform_rotation_spin.setValue(transforms["rotation"])
        if "translate_x" in transforms:
            self._transform_x_spin.setValue(transforms["translate_x"])
        if "translate_y" in transforms:
            self._transform_y_spin.setValue(transforms["translate_y"])
        if "mirror_h" in transforms:
            self._mirror_h_check.setChecked(transforms["mirror_h"])
        if "mirror_v" in transforms:
            self._mirror_v_check.setChecked(transforms["mirror_v"])
        if "n_fold" in transforms:
            self._n_fold_spin.setValue(transforms["n_fold"])
        if "tile_rows" in transforms:
            self._tile_rows_spin.setValue(transforms["tile_rows"])
        if "tile_cols" in transforms:
            self._tile_cols_spin.setValue(transforms["tile_cols"])

        # Re-evaluate conditional visibility after restoring params
        self._update_param_visibility()

        # Restore image source type (file / layer / depth_map)
        src_type = info.get("image_source_type", "file")
        if src_type == "depth_map":
            self._src_type_depth_radio.setChecked(True)
        elif src_type == "layer":
            self._src_type_layer_radio.setChecked(True)
        else:
            self._src_type_file_radio.setChecked(True)

        # Restore depth map invert state
        if "depth_map_invert" in info:
            try:
                self._depth_invert_check.setChecked(bool(info["depth_map_invert"]))
            except AttributeError:
                pass

        # Restore image size & position settings
        try:
            if "image_fit_mode" in info:
                mode = info["image_fit_mode"]
                if mode == "fit":
                    idx = self._image_fit_combo.findText("Fit (Keep Aspect)")
                elif mode == "custom":
                    idx = self._image_fit_combo.findText("Custom Size")
                else:
                    idx = self._image_fit_combo.findText("Fill Canvas")
                if idx >= 0:
                    self._image_fit_combo.blockSignals(True)
                    self._image_fit_combo.setCurrentIndex(idx)
                    self._image_fit_combo.blockSignals(False)
                    self._on_image_fit_mode_changed()
            if "image_width_mm" in info:
                self._image_width_spin.setValue(float(info["image_width_mm"]))
            if "image_height_mm" in info:
                self._image_height_spin.setValue(float(info["image_height_mm"]))
            if "image_offset_x_mm" in info:
                self._image_offset_x_spin.setValue(float(info["image_offset_x_mm"]))
            if "image_offset_y_mm" in info:
                self._image_offset_y_spin.setValue(float(info["image_offset_y_mm"]))
            if "image_lock_aspect" in info:
                self._lock_aspect_check.setChecked(bool(info["image_lock_aspect"]))
        except AttributeError:
            pass

        # Restore post-processing (brush) settings
        post_proc = info.get("post_processing", {})
        for _ppname, _ppvalue in post_proc.items():
            _ppwidget = self._post_proc_widgets.get(_ppname)
            if _ppwidget is None:
                continue
            if isinstance(_ppwidget, (QDoubleSpinBox, QSpinBox)):
                _ppwidget.setValue(_ppvalue)
            elif isinstance(_ppwidget, QComboBox):
                _pp_idx = _ppwidget.findText(str(_ppvalue))
                if _pp_idx >= 0:
                    _ppwidget.setCurrentIndex(_pp_idx)
        self._update_post_proc_visibility()

    def _on_active_layer_changed(self, layer_id: str) -> None:
        """Handle active layer change: save current settings to old layer, restore new."""
        # Deactivate FMM pick mode when switching layers
        if self._canvas_ref is not None:
            self._canvas_ref.set_fmm_source_mode(False)
            self._canvas_ref.clear_fmm_source_marker()
        if self._fmm_btn_alive():
            self._pick_fmm_source_btn.setText("Pick on Canvas")  # type: ignore[union-attr]

        # Save current settings to the layer currently shown in the target combo
        prev_layer_id = self.current_layer_id()
        if prev_layer_id and prev_layer_id != layer_id:
            snapshot = self._get_settings_snapshot()
            if snapshot is not None:
                self._controller.set_layer_generator_info(prev_layer_id, snapshot)

        # Switch the target layer combo to the new active layer (no signal to avoid loop)
        idx = self._layer_combo.findData(layer_id)
        if idx >= 0:
            self._layer_combo.blockSignals(True)
            self._layer_combo.setCurrentIndex(idx)
            self._layer_combo.blockSignals(False)

        # Restore the new layer's saved settings (if any)
        new_layer = self._controller.get_layer(layer_id)
        if new_layer is not None and isinstance(new_layer.generator_info, dict):
            info = new_layer.generator_info
            if info.get("mode") in ("Math Art", "Image to Lines", "3D Scene"):
                self._apply_settings_snapshot(info)

    def _on_generator_info_changed(self, layer_id: str) -> None:
        """Refresh offset param widgets when generator_info changes for the active layer.

        Called after an undoable operation (e.g. MoveLayerCommand) updates
        ``generator_info`` on the active layer.  Only the ``x_offset_mm`` and
        ``y_offset_mm`` spinboxes are touched — the rest of the UI is left intact
        to avoid disrupting the user's workflow.
        """
        if layer_id != self._controller.active_layer_id:
            return
        layer = self._controller.get_layer(layer_id)
        if layer is None or not isinstance(layer.generator_info, dict):
            return
        params = layer.generator_info.get("params", {})
        for name in ("x_offset_mm", "y_offset_mm", "pos_x", "pos_y"):
            widget = self._param_widgets.get(name)
            if widget is not None and isinstance(widget, QDoubleSpinBox) and name in params:
                widget.blockSignals(True)
                widget.setValue(float(params[name]))
                widget.blockSignals(False)

    def _on_project_loaded(self) -> None:
        """Reset per-layer settings memory tracking when a new project is loaded."""
        # Clear any stale layer selection so _on_active_layer_changed starts fresh
        self._load_camera_from_project()

    # ------------------------------------------------------------------
    # Shape Drawing handlers
    # ------------------------------------------------------------------

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
