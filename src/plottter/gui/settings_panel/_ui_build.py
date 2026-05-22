"""_UIBuildMixin — builds the initial SettingsPanel UI in _build_initial_ui()."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .workers import _VisibilityTrackedLabel


class _UIBuildMixin:
    """Mixin that builds the initial SettingsPanel UI."""

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

        self._auto_contrast_check = QCheckBox("Auto Contrast")
        self._auto_contrast_check.setChecked(True)
        self._auto_contrast_check.toggled.connect(self._on_preprocessing_changed)
        prep_form.addRow(self._auto_contrast_check)

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

        # Unsharp mask: slider value 0–50 → amount 0.0–5.0
        self._unsharp_slider = QSlider(Qt.Orientation.Horizontal)
        self._unsharp_slider.setRange(0, 50)
        self._unsharp_slider.setValue(0)
        self._unsharp_slider.valueChanged.connect(self._on_preprocessing_changed)
        self._unsharp_val_label = QLabel("0.0")
        unsharp_row = QHBoxLayout()
        unsharp_row.addWidget(self._unsharp_slider)
        unsharp_row.addWidget(self._unsharp_val_label)
        unsharp_row_widget = QWidget()
        unsharp_row_widget.setLayout(unsharp_row)
        prep_form.addRow(QLabel("Unsharp Mask"), unsharp_row_widget)

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

        self._gen_lines_btn = QPushButton("Generate Lines (All Layers)")
        self._gen_lines_btn.clicked.connect(self._on_generate_lines)
        self._gen_lines_btn.setEnabled(False)
        color_sep_layout.addWidget(self._gen_lines_btn)

        self._gen_lines_selected_btn = QPushButton("Generate Lines (Selected Layer)")
        self._gen_lines_selected_btn.clicked.connect(self._on_generate_lines_selected)
        self._gen_lines_selected_btn.setEnabled(False)
        self._gen_lines_selected_btn.setToolTip(
            "Generate line art for only the selected layer — allows using "
            "a different generator/preset for each separated layer"
        )
        color_sep_layout.addWidget(self._gen_lines_selected_btn)

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

        from PyQt6.QtCore import QSize
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
        from PyQt6.QtWidgets import QLineEdit
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
                    from PyQt6.QtWidgets import QLineEdit
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
        self._generate_btn.setToolTip("Generate (Ctrl+Enter)")
        self._generate_btn.clicked.connect(self._on_generate)
        self._randomize_btn = QPushButton("Randomize")
        self._randomize_btn.clicked.connect(self._on_randomize)
        self._layout.addWidget(self._generate_btn)
        self._layout.addWidget(self._randomize_btn)
        self._layout.addStretch()

        # Ctrl+Return / Ctrl+Enter triggers Generate. Window-scoped so it
        # fires from anywhere — including from within a multi-line code
        # textarea (TurtleToy) where the textbox would otherwise eat the key.
        from PyQt6.QtGui import QKeySequence, QShortcut
        for keyseq in ("Ctrl+Return", "Ctrl+Enter"):
            sc = QShortcut(QKeySequence(keyseq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(self._on_generate)

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
