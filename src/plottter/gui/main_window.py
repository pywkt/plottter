"""MainWindow — the top-level application window."""

from __future__ import annotations

import copy

from PyQt6.QtCore import Qt, QSize, QThread, QSettings, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from plottter.gui.animation_bar import AnimationBar
from plottter.gui.canvas_widget import CanvasWidget
from plottter.gui.layer_panel import LayerPanel
from plottter.gui.mode_panel import ModePanel
from plottter.gui.project_controller import ProjectController
from plottter.gui.settings_panel import SettingsPanel
from plottter.models import Canvas, Layer, Project
from plottter.models.path import Polyline


class _WeldWorker(QThread):
    """QThread that runs weld_overlapping_paths on a layer's paths."""

    finished = pyqtSignal(list, int, int)  # (new_paths, before_count, after_count)
    progress = pyqtSignal(int, int)        # (current_index, total)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        paths: list[Polyline],
        tolerance_mm: float = 0.1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._tolerance_mm = tolerance_mm
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from plottter.processing.weld import weld_overlapping_paths

            before_count = len(self._paths)
            new_paths = weld_overlapping_paths(
                self._paths,
                tolerance_mm=self._tolerance_mm,
                cancelled_callback=lambda: self._cancelled,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            if self._cancelled:
                self.cancelled.emit()
            else:
                after_count = len(new_paths)
                self.finished.emit(new_paths, before_count, after_count)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _OptimizeWorker(QThread):
    """QThread that runs the full path optimization pipeline on a layer's paths."""

    finished = pyqtSignal(list, float, float, int, int)  # (new_paths, before_travel, after_travel, before_lifts, after_lifts)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100 within-layer progress

    def __init__(
        self,
        paths: list[Polyline],
        run_weld: bool = False,
        weld_tolerance: float = 0.1,
        run_simplify: bool = True,
        simplify_tolerance: float = 0.1,
        run_filter: bool = True,
        filter_min_length: float = 0.5,
        run_clip: bool = True,
        clip_bounds: tuple[float, float, float, float] | None = None,
        run_merge: bool = True,
        merge_threshold: float = 0.5,
        run_2opt: bool = True,
        run_3opt: bool = False,
        run_or_opt: bool = True,
        num_starts: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._run_weld = run_weld
        self._weld_tolerance = weld_tolerance
        self._run_simplify = run_simplify
        self._simplify_tolerance = simplify_tolerance
        self._run_filter = run_filter
        self._filter_min_length = filter_min_length
        self._run_clip = run_clip
        self._clip_bounds = clip_bounds
        self._run_merge = run_merge
        self._merge_threshold = merge_threshold
        self._run_2opt = run_2opt
        self._run_3opt = run_3opt
        self._run_or_opt = run_or_opt
        self._num_starts = num_starts
        self._cancelled = False

    def request_stop(self) -> None:
        """Request cancellation.  The worker will stop at the next safe checkpoint."""
        self._cancelled = True

    def run(self) -> None:
        try:
            from plottter.processing import (
                weld_overlapping_paths,
                simplify_paths,
                filter_short_paths,
                clip_to_bounds,
                merge_nearby_paths,
                reorder_paths,
                optimize_2opt,
                optimize_3opt,
                optimize_or_opt,
                calculate_travel_distance,
            )

            paths = self._paths
            before_travel = calculate_travel_distance(paths)
            before_lifts = len(paths)

            # --- Preprocessing steps (0-10%) ---
            self.progress.emit(0)
            if self._run_weld:
                paths = weld_overlapping_paths(paths, tolerance_mm=self._weld_tolerance)
            if self._run_simplify:
                paths = simplify_paths(paths, tolerance_mm=self._simplify_tolerance)
            if self._run_filter:
                paths = filter_short_paths(paths, min_length_mm=self._filter_min_length)
            if self._run_clip and self._clip_bounds is not None:
                paths = clip_to_bounds(paths, self._clip_bounds)
            if self._run_merge:
                paths = merge_nearby_paths(paths, threshold_mm=self._merge_threshold)
            self.progress.emit(10)

            if self._cancelled:
                after_travel = calculate_travel_distance(paths)
                self.finished.emit(paths, before_travel, after_travel, before_lifts, len(paths))
                return

            # --- Nearest-neighbour reordering (10-35%) ---
            def _nn_progress(f: float) -> None:
                self.progress.emit(10 + int(f * 25))

            paths = reorder_paths(
                paths,
                num_starts=self._num_starts,
                progress_callback=_nn_progress,
                cancelled=lambda: self._cancelled,
            )
            self.progress.emit(35)

            if self._run_2opt and not self._cancelled:
                # --- 2-opt improvement (35-55%) ---
                def _2opt_progress(f: float) -> None:
                    self.progress.emit(35 + int(f * 20))

                paths = optimize_2opt(
                    paths,
                    progress_callback=_2opt_progress,
                    cancelled=lambda: self._cancelled,
                )
                self.progress.emit(55)

            if self._run_3opt and not self._cancelled:
                # --- 3-opt improvement (55-75%) ---
                def _3opt_progress(f: float) -> None:
                    self.progress.emit(55 + int(f * 20))

                paths = optimize_3opt(
                    paths,
                    progress_callback=_3opt_progress,
                    cancelled=lambda: self._cancelled,
                )
                self.progress.emit(75)

            if self._run_or_opt and not self._cancelled:
                # --- Or-opt improvement (75-100%) ---
                def _oropt_progress(f: float) -> None:
                    self.progress.emit(75 + int(f * 25))

                paths = optimize_or_opt(
                    paths,
                    progress_callback=_oropt_progress,
                    cancelled=lambda: self._cancelled,
                )
                self.progress.emit(100)

            after_travel = calculate_travel_distance(paths)
            after_lifts = len(paths)
            self.finished.emit(paths, before_travel, after_travel, before_lifts, after_lifts)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _BrushWorker(QThread):
    """QThread that runs apply_brush on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        brush_type: str,
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._brush_type = brush_type
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.brush import apply_brush
            self.progress.emit(10)
            result = apply_brush(self._paths, self._brush_type, self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _TaperWorker(QThread):
    """QThread that runs taper_paths on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.taper import taper_paths
            self.progress.emit(10)
            result = taper_paths(self._paths, **self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _OffsetWorker(QThread):
    """QThread that runs offset_paths on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.offset import offset_paths
            self.progress.emit(10)
            result = offset_paths(self._paths, **self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _BrushDialog:
    """Factory that builds and shows the Apply Brush dialog.

    Returns (brush_type, params) on accept, or (None, None) on cancel.
    Uses a QStackedWidget so each brush type shows its own parameter page.
    """

    BRUSH_TYPES = ["None", "Stippled", "Multi-Stroke", "Calligraphic"]
    # Maps brush name → stack page index (0 = empty "None" page)
    _PAGE_INDEX = {"None": 0, "Stippled": 1, "Multi-Stroke": 2, "Calligraphic": 3}

    @staticmethod
    def run(parent, sample_paths: list[Polyline]) -> tuple[str | None, dict | None]:
        """Show the dialog and return (brush_type, params) or (None, None)."""
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QComboBox,
            QGroupBox,
            QVBoxLayout,
            QSpinBox,
            QStackedWidget,
            QWidget,
        )
        from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen
        from PyQt6.QtCore import Qt

        dialog = QDialog(parent)
        dialog.setWindowTitle("Apply Brush to Layer")
        dialog.setMinimumWidth(460)

        main_layout = QVBoxLayout(dialog)
        main_layout.setSpacing(10)

        # --- Brush type selector ---
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Brush type:"))
        type_combo = QComboBox()
        type_combo.addItems(_BrushDialog.BRUSH_TYPES)
        type_layout.addWidget(type_combo)
        type_layout.addStretch()
        main_layout.addLayout(type_layout)

        # --- Stacked parameter pages ---
        stack = QStackedWidget()
        main_layout.addWidget(stack)

        # Page 0: None (empty placeholder)
        stack.addWidget(QWidget())

        # Page 1: Stipple parameters
        stipple_group = QGroupBox("Stipple Parameters")
        stipple_form = QFormLayout(stipple_group)

        spacing_spin = QDoubleSpinBox()
        spacing_spin.setRange(0.1, 50.0)
        spacing_spin.setSingleStep(0.1)
        spacing_spin.setValue(1.0)
        spacing_spin.setSuffix(" mm")
        stipple_form.addRow("Spacing:", spacing_spin)

        size_spin = QDoubleSpinBox()
        size_spin.setRange(0.05, 5.0)
        size_spin.setSingleStep(0.05)
        size_spin.setValue(0.3)
        size_spin.setSuffix(" mm")
        stipple_form.addRow("Dot size:", size_spin)

        randomness_spin = QDoubleSpinBox()
        randomness_spin.setRange(0.0, 1.0)
        randomness_spin.setSingleStep(0.05)
        randomness_spin.setValue(0.2)
        stipple_form.addRow("Randomness:", randomness_spin)

        stack.addWidget(stipple_group)  # index 1

        # Page 2: Multi-Stroke parameters
        multi_group = QGroupBox("Multi-Stroke Parameters")
        multi_form = QFormLayout(multi_group)

        stroke_count_spin = QSpinBox()
        stroke_count_spin.setRange(1, 20)
        stroke_count_spin.setValue(3)
        multi_form.addRow("Stroke count:", stroke_count_spin)

        spread_spin = QDoubleSpinBox()
        spread_spin.setRange(0.0, 10.0)
        spread_spin.setSingleStep(0.1)
        spread_spin.setValue(0.5)
        spread_spin.setSuffix(" mm")
        multi_form.addRow("Spread:", spread_spin)

        stroke_noise_spin = QDoubleSpinBox()
        stroke_noise_spin.setRange(0.0, 1.0)
        stroke_noise_spin.setSingleStep(0.05)
        stroke_noise_spin.setValue(0.3)
        multi_form.addRow("Noise:", stroke_noise_spin)

        stack.addWidget(multi_group)  # index 2

        # Page 3: Calligraphic parameters
        calli_group = QGroupBox("Calligraphic Parameters")
        calli_form = QFormLayout(calli_group)

        nib_angle_spin = QDoubleSpinBox()
        nib_angle_spin.setRange(0.0, 180.0)
        nib_angle_spin.setSingleStep(5.0)
        nib_angle_spin.setValue(45.0)
        nib_angle_spin.setSuffix("°")
        calli_form.addRow("Nib angle:", nib_angle_spin)

        nib_width_spin = QDoubleSpinBox()
        nib_width_spin.setRange(0.1, 20.0)
        nib_width_spin.setSingleStep(0.1)
        nib_width_spin.setValue(1.5)
        nib_width_spin.setSuffix(" mm")
        calli_form.addRow("Max width:", nib_width_spin)

        min_width_spin = QDoubleSpinBox()
        min_width_spin.setRange(0.01, 10.0)
        min_width_spin.setSingleStep(0.05)
        min_width_spin.setValue(0.2)
        min_width_spin.setSuffix(" mm")
        calli_form.addRow("Min width:", min_width_spin)

        stack.addWidget(calli_group)  # index 3

        # --- Preview area ---
        preview_label = QLabel()
        preview_label.setFixedSize(400, 180)
        preview_label.setStyleSheet("border: 1px solid gray; background: white;")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(preview_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        main_layout.addWidget(buttons)

        # --------------- helpers ---------------
        def _get_params() -> dict:
            return {
                # Stipple
                "stipple_spacing_mm": spacing_spin.value(),
                "stipple_size_mm": size_spin.value(),
                "stipple_randomness": randomness_spin.value(),
                # Multi-stroke
                "stroke_count": stroke_count_spin.value(),
                "stroke_spread_mm": spread_spin.value(),
                "stroke_noise": stroke_noise_spin.value(),
                # Calligraphic
                "nib_angle": nib_angle_spin.value(),
                "nib_width_mm": nib_width_spin.value(),
                "min_width_mm": min_width_spin.value(),
            }

        def _update_page() -> None:
            brush = type_combo.currentText()
            page = _BrushDialog._PAGE_INDEX.get(brush, 0)
            stack.setCurrentIndex(page)

        def _render_preview() -> None:
            from plottter.processing.brush import apply_brush

            brush = type_combo.currentText()
            params = _get_params()
            preview_input = sample_paths[:50]
            try:
                brushed = apply_brush(preview_input, brush, params)
            except Exception:
                brushed = preview_input

            all_paths = list(brushed)
            if not all_paths:
                preview_label.setText("(no output)")
                return

            # Compute bounding box
            xs = [x for p in all_paths for x, _ in p]
            ys = [y for p in all_paths for _, y in p]
            if not xs:
                preview_label.setText("(no output)")
                return
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w_data = x_max - x_min or 1.0
            h_data = y_max - y_min or 1.0

            pw, ph = 400, 180
            margin = 10
            scale = min((pw - 2 * margin) / w_data, (ph - 2 * margin) / h_data)

            pixmap = QPixmap(pw, ph)
            pixmap.fill(QColor("white"))
            painter = QPainter(pixmap)
            pen = QPen(QColor("black"))
            pen.setWidthF(0.5)
            painter.setPen(pen)

            def to_px(x: float, y: float):
                px = margin + (x - x_min) * scale
                py = ph - margin - (y - y_min) * scale
                return px, py

            for path in all_paths:
                if len(path) < 2:
                    continue
                for i in range(len(path) - 1):
                    x1, y1 = to_px(*path[i])
                    x2, y2 = to_px(*path[i + 1])
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            painter.end()
            preview_label.setPixmap(pixmap)
            preview_label.setText("")

        def _on_change(*_) -> None:
            _update_page()
            _render_preview()

        type_combo.currentIndexChanged.connect(_on_change)
        # Stipple controls
        spacing_spin.valueChanged.connect(_on_change)
        size_spin.valueChanged.connect(_on_change)
        randomness_spin.valueChanged.connect(_on_change)
        # Multi-stroke controls
        stroke_count_spin.valueChanged.connect(_on_change)
        spread_spin.valueChanged.connect(_on_change)
        stroke_noise_spin.valueChanged.connect(_on_change)
        # Calligraphic controls
        nib_angle_spin.valueChanged.connect(_on_change)
        nib_width_spin.valueChanged.connect(_on_change)
        min_width_spin.valueChanged.connect(_on_change)

        # Initial state
        _update_page()
        _render_preview()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None, None

        return type_combo.currentText(), _get_params()


class MainWindow(QMainWindow):
    """Main application window with menus, toolbar, splitter layout."""

    def __init__(self, controller: ProjectController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._current_file: str | None = None

        self.setWindowTitle("Plottter")
        self.setMinimumSize(900, 600)

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._connect_signals()
        self._update_title()
        self._restore_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: mode + layer
        left_panel = QWidget()
        left_panel.setMinimumWidth(180)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._mode_panel = ModePanel()
        left_layout.addWidget(self._mode_panel)

        self._layer_panel = LayerPanel(self._controller)
        left_layout.addWidget(self._layer_panel, stretch=1)

        splitter.addWidget(left_panel)

        # Center: canvas + animation bar
        canvas_container = QWidget()
        canvas_vbox = QVBoxLayout(canvas_container)
        canvas_vbox.setContentsMargins(0, 0, 0, 0)
        canvas_vbox.setSpacing(0)

        self._canvas = CanvasWidget(self._controller)
        canvas_vbox.addWidget(self._canvas, stretch=1)

        self._anim_bar = AnimationBar()
        canvas_vbox.addWidget(self._anim_bar)

        splitter.addWidget(canvas_container)

        # Right panel: settings
        self._settings_panel = SettingsPanel(self._controller)
        self._settings_panel.setMinimumWidth(250)
        splitter.addWidget(self._settings_panel)

        # Splitter proportions: left=1, center=3, right=2
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        self.setCentralWidget(splitter)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        # --- File ---
        file_menu = menu_bar.addMenu("&File")

        self._act_new = QAction("&New", self)
        self._act_new.setShortcut(QKeySequence.StandardKey.New)
        self._act_new.triggered.connect(self._on_new)
        file_menu.addAction(self._act_new)

        self._act_open = QAction("&Open…", self)
        self._act_open.setShortcut(QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self._on_open)
        file_menu.addAction(self._act_open)

        self._recent_menu = file_menu.addMenu("Recent &Projects")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        self._act_save = QAction("&Save", self)
        self._act_save.setShortcut(QKeySequence.StandardKey.Save)
        self._act_save.triggered.connect(self._on_save)
        file_menu.addAction(self._act_save)

        self._act_save_as = QAction("Save &As…", self)
        self._act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._act_save_as.triggered.connect(self._on_save_as)
        file_menu.addAction(self._act_save_as)

        file_menu.addSeparator()

        self._act_export_current = QAction("Export Current Layer…", self)
        self._act_export_current.setShortcut(QKeySequence("Ctrl+E"))
        self._act_export_current.triggered.connect(self._on_export_current)
        file_menu.addAction(self._act_export_current)

        self._act_export_all = QAction("Export All Layers…", self)
        self._act_export_all.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self._act_export_all.triggered.connect(self._on_export_all)
        file_menu.addAction(self._act_export_all)

        file_menu.addSeparator()

        self._act_quit = QAction("&Quit", self)
        self._act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self._act_quit.triggered.connect(self.close)
        file_menu.addAction(self._act_quit)

        # --- Edit ---
        edit_menu = menu_bar.addMenu("&Edit")

        self._act_undo = QAction("&Undo", self)
        self._act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._act_undo.setEnabled(False)
        self._act_undo.triggered.connect(self._controller.undo_stack.undo)
        edit_menu.addAction(self._act_undo)

        self._act_redo = QAction("&Redo", self)
        self._act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self._act_redo.setEnabled(False)
        self._act_redo.triggered.connect(self._controller.undo_stack.redo)
        edit_menu.addAction(self._act_redo)

        edit_menu.addSeparator()

        self._act_canvas_settings = QAction("&Canvas Settings…", self)
        self._act_canvas_settings.triggered.connect(self._on_canvas_settings)
        edit_menu.addAction(self._act_canvas_settings)

        self._act_rotate_canvas = QAction("&Rotate Canvas (Swap Dimensions)", self)
        self._act_rotate_canvas.triggered.connect(self._on_rotate_canvas)
        edit_menu.addAction(self._act_rotate_canvas)

        self._act_preferences = QAction("&Preferences…", self)
        self._act_preferences.setShortcut(QKeySequence("Ctrl+,"))
        self._act_preferences.triggered.connect(self._on_preferences)
        edit_menu.addAction(self._act_preferences)

        # --- View ---
        view_menu = menu_bar.addMenu("&View")

        self._act_zoom_in = QAction("Zoom &In", self)
        self._act_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        self._act_zoom_in.triggered.connect(self._canvas.zoom_in)
        view_menu.addAction(self._act_zoom_in)

        self._act_zoom_out = QAction("Zoom &Out", self)
        self._act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        self._act_zoom_out.triggered.connect(self._canvas.zoom_out)
        view_menu.addAction(self._act_zoom_out)

        self._act_fit = QAction("&Fit to Window", self)
        self._act_fit.setShortcut(QKeySequence("Ctrl+0"))
        self._act_fit.triggered.connect(self._canvas.fit_to_window)
        view_menu.addAction(self._act_fit)

        view_menu.addSeparator()

        self._act_grid = QAction("Toggle &Grid", self)
        self._act_grid.setShortcut(QKeySequence("G"))
        self._act_grid.setCheckable(True)
        self._act_grid.toggled.connect(self._canvas.set_show_grid)
        view_menu.addAction(self._act_grid)

        self._act_reg_marks = QAction("Toggle &Registration Marks", self)
        self._act_reg_marks.setShortcut(QKeySequence("R"))
        self._act_reg_marks.setCheckable(True)
        self._act_reg_marks.setChecked(True)
        self._act_reg_marks.toggled.connect(self._canvas.set_show_reg_marks)
        view_menu.addAction(self._act_reg_marks)

        self._act_travel = QAction("Toggle &Travel Moves", self)
        self._act_travel.setShortcut(QKeySequence("T"))
        self._act_travel.setCheckable(True)
        self._act_travel.toggled.connect(self._canvas.set_show_travel)
        view_menu.addAction(self._act_travel)

        self._act_image_overlay = QAction("Toggle &Image Overlay", self)
        self._act_image_overlay.setShortcut(QKeySequence("I"))
        self._act_image_overlay.setCheckable(True)
        self._act_image_overlay.setChecked(True)
        self._act_image_overlay.toggled.connect(self._canvas.set_show_image_overlay)
        view_menu.addAction(self._act_image_overlay)

        view_menu.addSeparator()

        self._act_paper_texture = QAction("Toggle &Paper Texture", self)
        self._act_paper_texture.setCheckable(True)
        self._act_paper_texture.toggled.connect(self._canvas.set_paper_texture)
        view_menu.addAction(self._act_paper_texture)

        self._act_jitter = QAction("Toggle Pen &Jitter", self)
        self._act_jitter.setCheckable(True)
        self._act_jitter.setToolTip("Simulate organic pen wobble in the preview (does not affect export)")
        self._act_jitter.toggled.connect(self._canvas.set_jitter_enabled)
        view_menu.addAction(self._act_jitter)

        self._act_jitter_intensity = QAction("Pen Jitter Intensity…", self)
        self._act_jitter_intensity.setToolTip("Set the amount of pen jitter wobble (0.1 = subtle, 5.0 = heavy)")
        self._act_jitter_intensity.triggered.connect(self._on_jitter_intensity)
        view_menu.addAction(self._act_jitter_intensity)

        # --- Generate ---
        generate_menu = menu_bar.addMenu("&Generate")
        self._act_generate_now = QAction("Generate Now", self)
        self._act_generate_now.setShortcut(QKeySequence("Ctrl+G"))
        self._act_generate_now.triggered.connect(self._on_generate_now)
        generate_menu.addAction(self._act_generate_now)

        self._act_randomize = QAction("Randomize Parameters", self)
        self._act_randomize.setShortcut(QKeySequence("Ctrl+R"))
        self._act_randomize.triggered.connect(self._on_randomize)
        generate_menu.addAction(self._act_randomize)

        generate_menu.addSeparator()

        self._act_surprise_me = QAction("Surprise Me!", self)
        self._act_surprise_me.setToolTip("Pick a random math generator, randomize its parameters, and generate")
        self._act_surprise_me.triggered.connect(self._on_surprise_me)
        generate_menu.addAction(self._act_surprise_me)

        self._act_browse_presets = QAction("Browse Presets…", self)
        self._act_browse_presets.setToolTip("Browse all math art presets with thumbnail previews")
        self._act_browse_presets.triggered.connect(self._on_browse_presets)
        generate_menu.addAction(self._act_browse_presets)

        # --- Tools ---
        tools_menu = menu_bar.addMenu("&Tools")

        self._act_optimize_layer = QAction("Optimize Current Layer", self)
        self._act_optimize_layer.triggered.connect(self._on_optimize_layer)
        tools_menu.addAction(self._act_optimize_layer)

        self._act_optimize_all = QAction("Optimize All Layers", self)
        self._act_optimize_all.triggered.connect(self._on_optimize_all)
        tools_menu.addAction(self._act_optimize_all)

        self._act_regen_all_3d = QAction("Regenerate All 3D Layers", self)
        self._act_regen_all_3d.setShortcut(QKeySequence("Ctrl+Shift+G"))
        self._act_regen_all_3d.setToolTip(
            "Sequentially regenerate all 3D Scene layers with up-to-date sibling occlusion"
        )
        self._act_regen_all_3d.triggered.connect(self._on_regenerate_all_3d)
        tools_menu.addAction(self._act_regen_all_3d)

        tools_menu.addSeparator()

        self._act_simplify = QAction("Simplify Paths", self)
        self._act_simplify.triggered.connect(self._on_simplify_layer)
        tools_menu.addAction(self._act_simplify)

        self._act_merge = QAction("Merge Nearby Paths", self)
        self._act_merge.triggered.connect(self._on_merge_layer)
        tools_menu.addAction(self._act_merge)

        self._act_clip = QAction("Clip to Canvas", self)
        self._act_clip.triggered.connect(self._on_clip_layer)
        tools_menu.addAction(self._act_clip)

        self._act_weld = QAction("Weld Overlapping Paths", self)
        self._act_weld.setToolTip("Remove duplicate overlapping segments across paths in the active layer")
        self._act_weld.triggered.connect(self._on_weld_layer)
        tools_menu.addAction(self._act_weld)

        self._act_apply_brush = QAction("Apply Brush to Layer…", self)
        self._act_apply_brush.setToolTip("Replace paths with a stylized brush effect (stippled dots, multi-stroke, calligraphic)")
        self._act_apply_brush.triggered.connect(self._on_apply_brush_layer)
        tools_menu.addAction(self._act_apply_brush)

        self._act_taper = QAction("Taper Paths…", self)
        self._act_taper.setToolTip("Replace paths with tapered stroke outlines that fade in and out")
        self._act_taper.triggered.connect(self._on_taper_layer)
        tools_menu.addAction(self._act_taper)

        self._act_offset = QAction("Offset Paths…", self)
        self._act_offset.setToolTip("Generate parallel offset copies of paths at a specified distance")
        self._act_offset.triggered.connect(self._on_offset_layer)
        tools_menu.addAction(self._act_offset)

        tools_menu.addSeparator()

        self._act_plot_axidraw = QAction("Plot with AxiDraw…", self)
        self._act_plot_axidraw.setToolTip("Send the current project directly to an AxiDraw plotter via USB")
        self._act_plot_axidraw.triggered.connect(self._on_plot_axidraw)
        tools_menu.addAction(self._act_plot_axidraw)

        tools_menu.addSeparator()

        self._act_plugins = QAction("Manage Plugins…", self)
        self._act_plugins.setToolTip("Load custom generator plugins and view plugin directories")
        self._act_plugins.triggered.connect(self._on_manage_plugins)
        tools_menu.addAction(self._act_plugins)

        # --- Help ---
        help_menu = menu_bar.addMenu("&Help")
        _act_about = QAction("&About Plottter", self)
        _act_about.triggered.connect(self._on_about)
        help_menu.addAction(_act_about)

        _act_kbd_shortcuts = QAction("&Keyboard Shortcuts…", self)
        _act_kbd_shortcuts.triggered.connect(self._on_kbd_shortcuts)
        help_menu.addAction(_act_kbd_shortcuts)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        tb.addAction(self._act_new)
        tb.addAction(self._act_open)
        tb.addAction(self._act_save)
        tb.addAction(self._act_export_current)
        tb.addSeparator()
        tb.addAction(self._act_undo)
        tb.addAction(self._act_redo)
        tb.addSeparator()
        self._act_drag_move = QAction("Move Tool", self)
        self._act_drag_move.setCheckable(True)
        self._act_drag_move.setShortcut(QKeySequence("V"))
        self._act_drag_move.setToolTip("Move Tool (V) — drag to reposition active layer content")
        tb.addAction(self._act_drag_move)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()

        def _make_label(min_width: int) -> QLabel:
            lbl = QLabel()
            lbl.setMinimumWidth(min_width)
            lbl.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            return lbl

        # Canvas dimensions and paper preset (e.g. "297 × 420 mm (A3)")
        self._status_canvas = _make_label(200)

        # Path count (e.g. "Paths: 1,234")
        self._status_paths = _make_label(130)

        # Pen travel distances and efficiency (e.g. "Draw: 1,234 mm  Travel: 567 mm  Eff: 69%")
        self._status_travel = _make_label(320)

        # Cursor position in mm (e.g. "123.4, 456.7 mm")
        self._status_cursor = _make_label(160)

        def _make_sep() -> QFrame:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            return sep

        # All info widgets are permanent (right-aligned), leaving the left
        # area free for transient showMessage() notifications.
        sb.addPermanentWidget(self._status_canvas)
        sb.addPermanentWidget(_make_sep())
        sb.addPermanentWidget(self._status_paths)
        sb.addPermanentWidget(_make_sep())
        sb.addPermanentWidget(self._status_travel)
        sb.addPermanentWidget(_make_sep())
        sb.addPermanentWidget(self._status_cursor)

        sb.showMessage("Ready")
        self._update_status_bar()

    def _connect_signals(self) -> None:
        c = self._controller
        c.project_loaded.connect(self._on_project_loaded)
        c.canvas_changed.connect(self._update_status_bar)
        c.paths_changed.connect(self._update_status_bar)
        c.layers_reordered.connect(self._update_status_bar)
        c.layer_added.connect(self._update_status_bar)
        c.layer_removed.connect(self._update_status_bar)
        c.layer_changed.connect(self._update_status_bar)
        c.modified_changed.connect(self._update_title)

        # Undo/redo stack — enable/disable actions and update their text
        stack = c.undo_stack
        stack.canUndoChanged.connect(self._act_undo.setEnabled)
        stack.canRedoChanged.connect(self._act_redo.setEnabled)
        stack.undoTextChanged.connect(self._on_undo_text_changed)
        stack.redoTextChanged.connect(self._on_redo_text_changed)
        self._canvas.mouse_position_mm.connect(self._on_cursor_moved)
        self._layer_panel.pre_duplicate.connect(self._settings_panel.flush_current_snapshot)
        self._mode_panel.mode_changed.connect(self._settings_panel.on_mode_changed)
        self._settings_panel.mode_change_requested.connect(self._on_mode_change_requested)
        self._settings_panel.image_preprocessed.connect(self._canvas.set_image_overlay)
        self._settings_panel.image_rect_changed.connect(self._canvas.set_image_overlay_rect)

        # Animation bar ↔ canvas
        self._anim_bar.play_pause_toggled.connect(self._canvas.toggle_animation)
        self._anim_bar.step_back_requested.connect(self._canvas.step_anim_backward)
        self._anim_bar.step_forward_requested.connect(self._canvas.step_anim_forward)
        self._anim_bar.seek_requested.connect(self._canvas.seek_animation)
        self._anim_bar.speed_changed.connect(self._canvas.set_anim_speed)
        self._canvas.anim_state_changed.connect(self._on_anim_state_changed)

        # Keyboard shortcuts for animation step (advertised in AnimationBar tooltips)
        QShortcut(QKeySequence("Shift+Left"), self).activated.connect(
            self._canvas.step_anim_backward
        )
        QShortcut(QKeySequence("Shift+Right"), self).activated.connect(
            self._canvas.step_anim_forward
        )

        # Drag-to-move tool
        self._act_drag_move.toggled.connect(self._canvas.set_drag_move_active)
        self._canvas.layer_move_finished.connect(self._on_layer_move_finished)

        # Wire mask-paint brush controls to canvas
        self._settings_panel.set_canvas(self._canvas)

        # Trigger initial population of the generator type combo for the default mode
        self._settings_panel.on_mode_changed(self._mode_panel.current_mode())

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_project_loaded(self) -> None:
        self._update_title()
        self._update_status_bar()

    def _on_mode_change_requested(self, mode: str) -> None:
        """Handle a mode change requested by the settings panel (e.g. when restoring layer settings)."""
        self._mode_panel.set_mode(mode)
        self._settings_panel.on_mode_changed(mode)

    def _on_cursor_moved(self, x_mm: float, y_mm: float) -> None:
        self._status_cursor.setText(f"  {x_mm:.1f}, {y_mm:.1f} mm  ")

    def _on_undo_text_changed(self, text: str) -> None:
        try:
            self._act_undo.setText(f"&Undo {text}" if text else "&Undo")
        except (AttributeError, RuntimeError):
            pass

    def _on_redo_text_changed(self, text: str) -> None:
        try:
            self._act_redo.setText(f"&Redo {text}" if text else "&Redo")
        except (AttributeError, RuntimeError):
            pass

    def _on_anim_state_changed(self, is_playing: bool, current_path: int, total: int) -> None:
        self._anim_bar.set_playing(is_playing)
        self._anim_bar.set_total_paths(total)
        self._anim_bar.set_position(current_path)

    def _on_layer_move_finished(self, dx_mm: float, dy_mm: float) -> None:
        """Apply a completed drag-to-move offset to the active layer's paths.

        If the layer's generator has ``x_offset_mm`` / ``y_offset_mm`` params,
        those are updated in generator_info so re-generating preserves the new
        position.  Both the path translation and the param update are bundled
        into a single undoable ``MoveLayerCommand``.
        """
        layer_id = self._controller.active_layer_id
        if not layer_id:
            return
        layer = self._controller.get_layer(layer_id)
        if layer is None or not layer.paths:
            return

        old_paths = [list(p) for p in layer.paths]
        new_paths = [[(x + dx_mm, y + dy_mm) for x, y in path] for path in layer.paths]

        # Check if the generator exposes x_offset_mm / y_offset_mm params.
        # generator_info may be None if the user hasn't switched away from this
        # layer yet (it's only persisted on layer switch).  Grab a live snapshot
        # from the settings panel in that case.
        old_gen_info = layer.generator_info
        if old_gen_info is None:
            old_gen_info = self._settings_panel._get_settings_snapshot()
            if old_gen_info is not None:
                layer.generator_info = old_gen_info
        new_gen_info: dict | None = None
        if (
            old_gen_info is not None
            and isinstance(old_gen_info.get("params"), dict)
            and "x_offset_mm" in old_gen_info["params"]
            and "y_offset_mm" in old_gen_info["params"]
        ):
            new_gen_info = copy.deepcopy(old_gen_info)
            new_gen_info["params"]["x_offset_mm"] = (
                old_gen_info["params"]["x_offset_mm"] + dx_mm
            )
            new_gen_info["params"]["y_offset_mm"] = (
                old_gen_info["params"]["y_offset_mm"] + dy_mm
            )
        elif (
            old_gen_info is not None
            and old_gen_info.get("mode") == "3D Scene"
            and isinstance(old_gen_info.get("params"), dict)
            and "pos_x" in old_gen_info["params"]
            and "pos_y" in old_gen_info["params"]
        ):
            # 3D Scene: pos_x/pos_y are in 3D world units.  Canvas X maps
            # directly to 3D X; canvas Y increases downward but 3D Y is up,
            # so the sign is inverted.
            new_gen_info = copy.deepcopy(old_gen_info)
            new_gen_info["params"]["pos_x"] = (
                old_gen_info["params"]["pos_x"] + dx_mm
            )
            new_gen_info["params"]["pos_y"] = (
                old_gen_info["params"]["pos_y"] - dy_mm
            )

        from plottter.gui.commands import MoveLayerCommand
        cmd = MoveLayerCommand(
            self._controller,
            layer_id,
            new_paths,
            old_paths,
            new_gen_info,
            copy.deepcopy(old_gen_info) if new_gen_info is not None else None,
        )
        self._controller.undo_stack.push(cmd)

    def _update_title(self, *_args) -> None:  # type: ignore[no-untyped-def]
        name = self._controller.current_project.name
        modified = self._controller.modified
        title = f"Plottter — {name}{'*' if modified else ''}"
        self.setWindowTitle(title)

    def _update_status_bar(self, *_args) -> None:  # type: ignore[no-untyped-def]
        project = self._controller.current_project
        canvas = project.canvas
        # Determine preset name
        preset = canvas.paper_preset if canvas.paper_preset != "Custom" else "Custom"
        self._status_canvas.setText(
            f"  {canvas.width_mm:.0f} \u00d7 {canvas.height_mm:.0f} mm ({preset})  "
        )

        total_paths = self._calculate_pen_lifts()
        self._status_paths.setText(f"  Paths: {total_paths:,}  ")

        pen_down = self._calculate_pen_down()
        pen_up = self._calculate_travel()
        total_dist = pen_down + pen_up
        efficiency = (pen_down / total_dist * 100) if total_dist > 0 else 0.0
        self._status_travel.setText(
            f"  Draw: {pen_down:,.0f} mm  Travel: {pen_up:,.0f} mm  Eff: {efficiency:.0f}%  "
        )

    def _calculate_pen_down(self) -> float:
        """Total pen-down (drawing) distance across all visible layers."""
        project = self._controller.current_project
        total = 0.0
        for layer in project.layers:
            if not layer.visible:
                continue
            for polyline in layer.paths:
                for i in range(len(polyline) - 1):
                    dx = polyline[i + 1][0] - polyline[i][0]
                    dy = polyline[i + 1][1] - polyline[i][1]
                    total += (dx * dx + dy * dy) ** 0.5
        return total

    def _calculate_travel(self) -> float:
        """Total pen-up (travel) distance between paths across all visible layers."""
        project = self._controller.current_project
        total = 0.0
        last_end: tuple[float, float] | None = None
        for layer in project.layers:
            if not layer.visible:
                continue
            for polyline in layer.paths:
                if not polyline:
                    continue
                if last_end is not None:
                    dx = polyline[0][0] - last_end[0]
                    dy = polyline[0][1] - last_end[1]
                    total += (dx * dx + dy * dy) ** 0.5
                last_end = polyline[-1]
        return total

    def _calculate_pen_lifts(self) -> int:
        """Number of pen lifts (= number of paths across all visible layers)."""
        project = self._controller.current_project
        return sum(
            layer.path_count() for layer in project.layers if layer.visible
        )

    # ------------------------------------------------------------------
    # File menu actions
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        if not self._prompt_save_if_modified():
            return
        from plottter.gui.dialogs.new_project import NewProjectDialog
        dialog = NewProjectDialog(self)
        if dialog.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        canvas = dialog.get_canvas()
        project = Project(name="Untitled", canvas=canvas)
        project.add_layer(Layer(name="Layer 1", color="#000000"))
        self._current_file = None
        self._controller.new_project(project)

    def _on_open(self) -> None:
        if not self._prompt_save_if_modified():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", self.last_file_dir(), "Plottter Files (*.plottter);;All Files (*)"
        )
        if not path:
            return
        try:
            from plottter.io.project_file import load_project
            project = load_project(path)
            self._current_file = path
            self.save_last_file_dir(path)
            self._add_recent_project(path)
            self._controller.load_project(project)
        except Exception as exc:
            QMessageBox.critical(self, "Error Opening File", str(exc))

    def _on_save(self) -> None:
        if self._current_file:
            self._save_to(self._current_file)
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", self.last_file_dir(), "Plottter Files (*.plottter);;All Files (*)"
        )
        if path:
            if not path.endswith(".plottter"):
                path += ".plottter"
            self._save_to(path)

    def _save_to(self, path: str) -> None:
        try:
            from plottter.io.project_file import save_project
            save_project(self._controller.current_project, path)
            self._current_file = path
            self.save_last_file_dir(path)
            self._add_recent_project(path)
            self._controller.mark_saved()
        except Exception as exc:
            QMessageBox.critical(self, "Error Saving File", str(exc))

    def _on_export_current(self) -> None:
        from plottter.gui.dialogs.export import ExportDialog
        dialog = ExportDialog(self)
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self._do_export(settings, "current")

    def _on_export_all(self) -> None:
        from plottter.gui.dialogs.export import ExportDialog
        dialog = ExportDialog(self)
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self._do_export(settings, settings.get("layer_mode", "all_separate"))

    def _do_export(self, settings: dict, mode: str) -> None:
        project = self._controller.current_project
        path = settings.get("output_path", "")
        if not path:
            QMessageBox.warning(self, "Export", "Please specify an output path.")
            return
        fmt = settings.get("format", "SVG")
        if mode == "current":
            layer_id = self._controller.active_layer_id
            active = self._controller.get_layer(layer_id) if layer_id else None
            if active is None:
                QMessageBox.warning(self, "Export", "No active layer to export.")
                return
        try:
            if fmt == "SVG":
                from plottter.export.svg import (
                    export_layer_svg,
                    export_all_layers_svg,
                    export_combined_svg,
                )
                if mode == "current":
                    export_layer_svg(active, project.canvas, path, settings)
                elif mode == "all_separate":
                    export_all_layers_svg(project, path, settings)
                else:
                    export_combined_svg(project, path, settings)
            elif fmt == "HPGL":
                from plottter.export.hpgl import (
                    export_layer_hpgl,
                    export_all_layers_hpgl,
                )
                if mode == "current":
                    export_layer_hpgl(active, project.canvas, path, settings)
                else:
                    export_all_layers_hpgl(project, path, settings)
            elif fmt == "G-code":
                from plottter.export.gcode import (
                    export_layer_gcode,
                    export_all_layers_gcode,
                )
                if mode == "current":
                    export_layer_gcode(active, project.canvas, path, settings)
                else:
                    export_all_layers_gcode(project, path, settings)
            elif fmt == "Mural":
                from plottter.export.mural import (
                    export_layer_mural,
                    export_all_layers_mural,
                )
                if mode == "current":
                    mural_warnings = export_layer_mural(active, project.canvas, path, settings)
                else:
                    mural_warnings = export_all_layers_mural(project, path, settings)
                if mural_warnings:
                    unique = list(dict.fromkeys(mural_warnings))
                    warn_text = "\n".join(unique[:10])
                    if len(unique) > 10:
                        warn_text += f"\n… and {len(unique) - 10} more"
                    QMessageBox.warning(
                        self,
                        "Mural Export — Out-of-Bounds Coordinates",
                        f"Some coordinates fall outside the valid drawing area:\n\n{warn_text}",
                    )
            QMessageBox.information(self, "Export", f"Exported successfully to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Edit menu actions
    # ------------------------------------------------------------------

    def _on_canvas_settings(self) -> None:
        from plottter.gui.dialogs.new_project import NewProjectDialog
        project = self._controller.current_project
        old_canvas = project.canvas
        dialog = NewProjectDialog(self, initial_canvas=old_canvas)
        if dialog.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        new_canvas = dialog.get_canvas()

        # Only offer scaling when there is art to scale
        layers_with_paths = [layer for layer in project.layers if layer.paths]
        scale_art = False
        if layers_with_paths:
            reply = QMessageBox.question(
                self,
                "Scale Art to New Canvas?",
                "Scale existing art to fit the new canvas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            scale_art = reply == QMessageBox.StandardButton.Yes

        if not scale_art:
            self._controller.set_canvas(new_canvas)
            return

        # Pre-compute scaled paths and updated generator_info for each layer
        from plottter.processing.scale import scale_paths_to_canvas
        old_left, old_top, old_right, old_bottom = old_canvas.drawing_area()
        new_left, new_top, new_right, new_bottom = new_canvas.drawing_area()
        old_draw_w = old_right - old_left
        old_draw_h = old_bottom - old_top
        new_draw_w = new_right - new_left
        new_draw_h = new_bottom - new_top
        sx = new_draw_w / old_draw_w if old_draw_w else 1.0
        sy = new_draw_h / old_draw_h if old_draw_h else 1.0

        scale_data = []
        for layer in project.layers:
            if not layer.paths:
                continue
            old_paths = [list(p) for p in layer.paths]
            new_paths = scale_paths_to_canvas(layer.paths, old_canvas, new_canvas)
            old_gen_info = layer.generator_info
            new_gen_info = None
            if old_gen_info is not None and isinstance(old_gen_info.get("params"), dict):
                params = old_gen_info["params"]
                if "x_offset_mm" in params or "y_offset_mm" in params:
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "x_offset_mm" in params:
                        new_gen_info["params"]["x_offset_mm"] = params["x_offset_mm"] * sx
                    if "y_offset_mm" in params:
                        new_gen_info["params"]["y_offset_mm"] = params["y_offset_mm"] * sy
                elif (
                    old_gen_info.get("mode") == "3D Scene"
                    and ("pos_x" in params or "pos_y" in params)
                ):
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "pos_x" in params:
                        new_gen_info["params"]["pos_x"] = params["pos_x"] * sx
                    if "pos_y" in params:
                        new_gen_info["params"]["pos_y"] = params["pos_y"] * sy
            scale_data.append(
                (
                    layer.id,
                    old_paths,
                    new_paths,
                    copy.deepcopy(old_gen_info) if new_gen_info is not None else None,
                    new_gen_info,
                )
            )

        # Push canvas change + all path scalings as one undoable macro
        from plottter.gui.commands import MoveLayerCommand, SetCanvasCommand
        self._controller.undo_stack.beginMacro("Canvas Resize with Scale")
        try:
            cmd = SetCanvasCommand(
                self._controller, new_canvas, copy.copy(old_canvas), description="Canvas Settings"
            )
            self._controller.undo_stack.push(cmd)
            for layer_id, old_paths, new_paths, old_gen_info, new_gen_info in scale_data:
                cmd = MoveLayerCommand(
                    self._controller,
                    layer_id,
                    new_paths,
                    old_paths,
                    new_gen_info,
                    old_gen_info,
                )
                self._controller.undo_stack.push(cmd)
        finally:
            self._controller.undo_stack.endMacro()

    def _ask_rotate_scale_mode(self) -> str | None:
        """Show a dialog asking how to handle art when rotating the canvas.

        Returns one of ``"stretch"``, ``"keep_aspect"``, ``"none"``, or
        ``None`` if the user cancelled.
        """
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Rotate Canvas — Scale Art?")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("How should existing art be handled after rotation?"))
        rb_stretch = QRadioButton("Scale to fit  (stretch to fill new dimensions)")
        rb_keep = QRadioButton("Scale to fit, keep aspect  (uniform scale, centered)")
        rb_none = QRadioButton("Don't scale  (keep art at original mm positions)")
        rb_keep.setChecked(True)
        layout.addWidget(rb_stretch)
        layout.addWidget(rb_keep)
        layout.addWidget(rb_none)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        if rb_stretch.isChecked():
            return "stretch"
        if rb_keep.isChecked():
            return "keep_aspect"
        return "none"

    def _on_rotate_canvas(self) -> None:
        project = self._controller.current_project
        old_canvas = project.canvas
        new_canvas = Canvas(
            width_mm=old_canvas.height_mm,
            height_mm=old_canvas.width_mm,
            margin_mm=old_canvas.margin_mm,
            paper_preset=old_canvas.paper_preset,
        )

        layers_with_paths = [layer for layer in project.layers if layer.paths]

        scale_mode = "none"  # default when no art exists
        if layers_with_paths:
            result = self._ask_rotate_scale_mode()
            if result is None:
                return
            scale_mode = result

        if scale_mode == "none":
            self._controller.set_canvas(new_canvas, description="Rotate Canvas")
            return

        # Build scaled path data for each layer
        from plottter.processing.scale import scale_paths_to_canvas, scale_paths_keep_aspect

        old_left, old_top, old_right, old_bottom = old_canvas.drawing_area()
        new_left, new_top, new_right, new_bottom = new_canvas.drawing_area()
        old_draw_w = old_right - old_left
        old_draw_h = old_bottom - old_top
        new_draw_w = new_right - new_left
        new_draw_h = new_bottom - new_top

        if scale_mode == "stretch":
            sx = new_draw_w / old_draw_w if old_draw_w else 1.0
            sy = new_draw_h / old_draw_h if old_draw_h else 1.0
        else:  # keep_aspect
            s = min(new_draw_w / old_draw_w, new_draw_h / old_draw_h) if old_draw_w and old_draw_h else 1.0
            sx = sy = s

        scale_data = []
        for layer in project.layers:
            if not layer.paths:
                continue
            old_paths = [list(p) for p in layer.paths]
            if scale_mode == "stretch":
                new_paths = scale_paths_to_canvas(layer.paths, old_canvas, new_canvas)
            else:
                new_paths = scale_paths_keep_aspect(layer.paths, old_canvas, new_canvas)
            old_gen_info = layer.generator_info
            new_gen_info = None
            if old_gen_info is not None and isinstance(old_gen_info.get("params"), dict):
                params = old_gen_info["params"]
                if "x_offset_mm" in params or "y_offset_mm" in params:
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "x_offset_mm" in params:
                        new_gen_info["params"]["x_offset_mm"] = params["x_offset_mm"] * sx
                    if "y_offset_mm" in params:
                        new_gen_info["params"]["y_offset_mm"] = params["y_offset_mm"] * sy
                elif (
                    old_gen_info.get("mode") == "3D Scene"
                    and ("pos_x" in params or "pos_y" in params)
                ):
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "pos_x" in params:
                        new_gen_info["params"]["pos_x"] = params["pos_x"] * sx
                    if "pos_y" in params:
                        new_gen_info["params"]["pos_y"] = params["pos_y"] * sy
            scale_data.append((
                layer.id,
                old_paths,
                new_paths,
                copy.deepcopy(old_gen_info) if new_gen_info is not None else None,
                new_gen_info,
            ))

        from plottter.gui.commands import MoveLayerCommand, SetCanvasCommand
        self._controller.undo_stack.beginMacro("Rotate Canvas with Scale")
        try:
            cmd = SetCanvasCommand(
                self._controller, new_canvas, copy.copy(old_canvas), description="Rotate Canvas"
            )
            self._controller.undo_stack.push(cmd)
            for layer_id, old_paths, new_paths, old_gen_info, new_gen_info in scale_data:
                cmd = MoveLayerCommand(
                    self._controller,
                    layer_id,
                    new_paths,
                    old_paths,
                    new_gen_info,
                    old_gen_info,
                )
                self._controller.undo_stack.push(cmd)
        finally:
            self._controller.undo_stack.endMacro()

    def _on_preferences(self) -> None:
        from plottter.gui.dialogs.preferences import PreferencesDialog
        dialog = PreferencesDialog(self)
        dialog.exec()
        # Refresh AI control availability in case the API key was changed
        self._settings_panel.update_ai_availability()

    # ------------------------------------------------------------------
    # Generate menu actions
    # ------------------------------------------------------------------

    def _on_generate_now(self) -> None:
        self._settings_panel.trigger_generate()

    def _on_randomize(self) -> None:
        self._settings_panel.trigger_randomize()

    def _on_jitter_intensity(self) -> None:
        """Open a dialog to set the pen jitter intensity."""
        from PyQt6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(
            self,
            "Pen Jitter Intensity",
            "Intensity (0.1 = subtle wobble, 5.0 = heavy jitter):",
            self._canvas.get_jitter_intensity(),
            0.1,
            5.0,
            1,
        )
        if ok:
            self._canvas.set_jitter_intensity(value)

    def _on_surprise_me(self) -> None:
        """Switch to Math Art mode, pick a random generator, randomize params, generate."""
        self._mode_panel.set_mode("Math Art")
        self._settings_panel.on_mode_changed("Math Art")
        self._settings_panel.trigger_surprise_me()

    def _on_browse_presets(self) -> None:
        """Open the preset gallery; apply chosen preset to the settings panel."""
        from PyQt6.QtWidgets import QDialog as _QDialog
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog

        dialog = PresetGalleryDialog(parent=self)
        if dialog.exec() == _QDialog.DialogCode.Accepted:
            gen_cls, preset_name = dialog.selected_preset()
            if gen_cls is not None:
                self._mode_panel.set_mode("Math Art")
                self._settings_panel.on_mode_changed("Math Art")
                self._settings_panel.apply_generator_preset(gen_cls, preset_name)

    # ------------------------------------------------------------------
    # Tools menu actions
    # ------------------------------------------------------------------

    def _on_optimize_layer(self) -> None:
        """Run the full optimization pipeline on the selected layer."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Optimize", "No selected layer to optimize.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Optimize", "Selected layer has no paths.")
            return

        from plottter.gui.dialogs.optimize_dialog import OptimizeSettingsDialog

        dlg = OptimizeSettingsDialog(parent=self)
        if dlg.exec() != OptimizeSettingsDialog.DialogCode.Accepted:
            return

        bounds = self._controller.current_project.canvas.drawing_area()
        self._run_optimization([layer], bounds, settings=dlg.get_settings())

    def _on_optimize_all(self) -> None:
        """Run the full optimization pipeline on all unlocked layers."""
        project = self._controller.current_project
        layers = [l for l in project.layers if not l.locked and l.paths]
        if not layers:
            QMessageBox.information(self, "Optimize All", "No unlocked layers with paths.")
            return
        bounds = project.canvas.drawing_area()
        self._run_optimization(layers, bounds)

    def _on_regenerate_all_3d(self) -> None:
        """Sequentially regenerate all 3D Scene layers with up-to-date sibling occlusion."""
        # Only flush when the settings panel is actually showing 3D controls.
        # Flushing while the panel is in a different mode (e.g. Math Art) would
        # overwrite a 3D layer's generator_info with the wrong mode's UI state.
        if self._settings_panel.current_mode == "3D Scene":
            self._settings_panel.flush_current_snapshot()

        project = self._controller.current_project
        d3_layers = [
            layer for layer in project.layers
            if isinstance(layer.generator_info, dict)
            and layer.generator_info.get("mode") == "3D Scene"
        ]

        if not d3_layers:
            QMessageBox.information(
                self,
                "Regenerate All 3D Layers",
                "No 3D Scene layers found in the project.",
            )
            return

        n = len(d3_layers)
        self._regen3d_layers = d3_layers
        self._regen3d_idx = 0

        progress = QProgressDialog(
            f"Generating 3D layer 1 of {n}…", "Cancel", 0, n * 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        self._regen3d_progress = progress

        # Wrap all path changes in a single undo macro
        self._controller.undo_stack.beginMacro("Regenerate All 3D Layers")
        self._start_next_3d_regen()

    def _start_next_3d_regen(self) -> None:
        if self._regen3d_idx >= len(self._regen3d_layers):
            self._finish_3d_regen()
            return

        if self._regen3d_progress.wasCanceled():
            self._finish_3d_regen(cancelled=True)
            return

        layer = self._regen3d_layers[self._regen3d_idx]
        n = len(self._regen3d_layers)
        base_progress = self._regen3d_idx * 100

        self._regen3d_progress.setLabelText(
            f"Generating 3D layer {self._regen3d_idx + 1} of {n}: '{layer.name}'…"
        )
        self._regen3d_progress.setValue(base_progress)

        info = layer.generator_info
        params = dict(info.get("params", {}))

        # Inject shared camera from project metadata
        project = self._controller.current_project
        cam = project.metadata.get("scene3d_camera", {})
        if cam:
            params["_camera"] = cam

        # Inject sibling shapes for HLR occlusion using up-to-date generator_info
        params["_sibling_3d_shapes"] = self._settings_panel._build_sibling_3d_shapes(layer.id)

        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.gui.generator_worker import GeneratorWorker

        generator = Scene3DGenerator()
        canvas = project.canvas
        layer_id = layer.id

        worker = GeneratorWorker(generator, params, canvas, parent=self)

        def on_progress(pct: int) -> None:
            self._regen3d_progress.setValue(base_progress + pct)

        def on_finished(paths: list, lid: str = layer_id) -> None:
            self._controller.set_layer_paths(lid, paths, "Regenerate 3D Layer")
            self._regen3d_idx += 1
            self._regen3d_progress.setValue(self._regen3d_idx * 100)
            self._start_next_3d_regen()
            worker.deleteLater()

        def on_error(msg: str) -> None:
            QMessageBox.critical(self, "3D Regeneration Error", msg)
            self._regen3d_idx += 1
            self._regen3d_progress.setValue(self._regen3d_idx * 100)
            self._start_next_3d_regen()
            worker.deleteLater()

        # Disconnect previous layer's cancel connection to avoid stacking
        prev = getattr(self, "_regen3d_worker", None)
        if prev is not None:
            try:
                self._regen3d_progress.canceled.disconnect(prev.cancel)
            except (RuntimeError, TypeError):
                pass
        self._regen3d_progress.canceled.connect(worker.cancel)

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._regen3d_worker = worker
        worker.start()

    def _finish_3d_regen(self, cancelled: bool = False) -> None:
        self._controller.undo_stack.endMacro()
        self._regen3d_progress.close()
        n = len(self._regen3d_layers)
        if cancelled:
            done = self._regen3d_idx
            self.statusBar().showMessage(
                f"3D regeneration cancelled after {done}/{n} layers.", 4000
            )
        else:
            self.statusBar().showMessage(
                f"Regenerated {n} 3D layer{'s' if n != 1 else ''} successfully.", 5000
            )

    def _on_simplify_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return
        from plottter.gui.dialogs.simplify_dialog import SimplifyDialog
        dialog = SimplifyDialog(list(layer.paths), parent=self)
        if dialog.exec() != SimplifyDialog.DialogCode.Accepted:
            return
        from plottter.processing import simplify_paths
        new_paths = simplify_paths(layer.paths, dialog.get_tolerance())
        self._controller.set_layer_paths(layer.id, new_paths, "Simplify Paths")

    def _on_merge_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return
        from plottter.gui.dialogs.merge_dialog import MergeDialog
        dialog = MergeDialog(list(layer.paths), parent=self)
        if dialog.exec() != MergeDialog.DialogCode.Accepted:
            return
        from plottter.processing import merge_nearby_paths
        new_paths = merge_nearby_paths(layer.paths, dialog.get_threshold())
        self._controller.set_layer_paths(layer.id, new_paths, "Merge Nearby Paths")

    def _on_clip_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return
        bounds = self._controller.current_project.canvas.drawing_area()
        from plottter.processing import clip_to_bounds
        new_paths = clip_to_bounds(layer.paths, bounds)
        self._controller.set_layer_paths(layer.id, new_paths, "Clip to Canvas")

    def _on_weld_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return

        from plottter.gui.dialogs.weld_dialog import WeldDialog

        dlg = WeldDialog(parent=self)
        if dlg.exec() != WeldDialog.DialogCode.Accepted:
            return
        tolerance_mm = dlg.get_tolerance()

        total = len(layer.paths)
        progress = QProgressDialog(
            f"Welding overlapping paths in '{layer.name}'…", "Cancel", 0, total, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        layer_id = layer.id
        worker = _WeldWorker(paths=list(layer.paths), tolerance_mm=tolerance_mm, parent=self)

        def on_progress(cur: int, tot: int) -> None:
            if tot > 0:
                progress.setValue(cur)

        def on_finished(new_paths: list, before_count: int, after_count: int) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Weld Overlapping Paths")
            removed = before_count - after_count
            self.statusBar().showMessage(
                f"Weld complete: {before_count} → {after_count} paths "
                f"({removed} removed).",
                5000,
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Weld Error", msg)
            worker.deleteLater()

        def on_cancelled() -> None:
            worker.cancel()

        def on_weld_cancelled() -> None:
            progress.close()
            self.statusBar().showMessage("Weld cancelled.", 3000)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.cancelled.connect(on_weld_cancelled)
        worker.error.connect(on_error)
        progress.canceled.connect(on_cancelled)
        self._weld_worker = worker
        worker.start()

    def _on_apply_brush_layer(self) -> None:
        """Show the Apply Brush dialog and replace the selected layer's paths."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Apply Brush", "No selected layer to apply brush to.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Apply Brush", "Selected layer has no paths.")
            return

        brush_type, params = _BrushDialog.run(self, list(layer.paths))
        if brush_type is None or brush_type == "None":
            return  # Cancelled or no-op

        total = len(layer.paths)
        progress = QProgressDialog(
            f"Applying '{brush_type}' brush to '{layer.name}'…", "", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = _BrushWorker(paths=list(layer.paths), brush_type=brush_type, params=params, parent=self)

        def on_progress(value: int) -> None:
            progress.setValue(value)

        def on_finished(new_paths: list) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Apply Brush")
            self.statusBar().showMessage(
                f"Brush applied: {total} → {len(new_paths)} paths.", 4000
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Brush Error", msg)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._brush_worker = worker
        worker.start()

    def _on_taper_layer(self) -> None:
        """Show the Taper Paths dialog and replace the selected layer's paths."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Taper Paths", "No selected layer to apply taper to.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Taper Paths", "Selected layer has no paths.")
            return

        from plottter.gui.dialogs.taper_dialog import TaperSettingsDialog

        dlg = TaperSettingsDialog(list(layer.paths), parent=self)
        if dlg.exec() != TaperSettingsDialog.DialogCode.Accepted:
            return

        params = dlg.get_params()
        total = len(layer.paths)
        progress = QProgressDialog(
            f"Applying taper to '{layer.name}'…", "", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = _TaperWorker(paths=list(layer.paths), params=params, parent=self)

        def on_progress(value: int) -> None:
            progress.setValue(value)

        def on_finished(new_paths: list) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Taper Paths")
            self.statusBar().showMessage(
                f"Taper applied: {total} → {len(new_paths)} paths.", 4000
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Taper Error", msg)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._taper_worker = worker
        worker.start()

    def _on_offset_layer(self) -> None:
        """Show the Offset Paths dialog and replace the selected layer's paths."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Offset Paths", "No selected layer. Please select a layer first.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Offset Paths", "Selected layer has no paths.")
            return

        from plottter.gui.dialogs.offset_dialog import OffsetSettingsDialog
        dlg = OffsetSettingsDialog(list(layer.paths), parent=self)
        if dlg.exec() != OffsetSettingsDialog.DialogCode.Accepted:
            return

        params = dlg.get_params()
        total = len(layer.paths)
        progress = QProgressDialog(
            f"Applying offset to '{layer.name}'…", "", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = _OffsetWorker(paths=list(layer.paths), params=params, parent=self)

        def on_progress(value: int) -> None:
            progress.setValue(value)

        def on_finished(new_paths: list) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Offset Paths")
            self.statusBar().showMessage(
                f"Offset applied: {total} → {len(new_paths)} paths.", 4000
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Offset Error", msg)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._offset_worker = worker
        worker.start()

    def _on_plot_axidraw(self) -> None:
        """Open AxiDraw plot dialog for direct USB plotting."""
        project = self._controller.current_project
        # Build SVG of all visible layers
        from plottter.export.axidraw import project_to_svg_string
        svg_data = project_to_svg_string(project, None, {"stroke_width_mm": 0.3})
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dlg = AxiDrawDialog(svg_data, parent=self)
        dlg.plot_started.connect(lambda: self.statusBar().showMessage("Plotting…"))
        dlg.plot_finished.connect(lambda: self.statusBar().showMessage("Plot complete."))
        dlg.exec()

    def _on_manage_plugins(self) -> None:
        """Show the plugin management dialog."""
        from plottter.generators.plugin_loader import (
            create_user_plugin_dir,
            get_plugin_dirs,
            load_plugins,
        )
        from plottter.generators import GENERATORS

        # Ensure user plugin directory exists, then reload plugins
        user_dir = create_user_plugin_dir()
        new_names = load_plugins()
        plugin_dirs = get_plugin_dirs()

        dir_list = "\n".join(f"  • {d}" for d in plugin_dirs)
        gen_list = "\n".join(
            f"  • {name}" for name in sorted(GENERATORS.keys())
        ) or "  (none)"

        if new_names:
            newly = "\n".join(f"  + {n}" for n in new_names)
            msg = (
                f"Newly loaded plugins:\n{newly}\n\n"
                f"All registered generators:\n{gen_list}\n\n"
                f"Plugin directories searched:\n{dir_list}"
            )
        else:
            msg = (
                f"No new plugins found.\n\n"
                f"All registered generators:\n{gen_list}\n\n"
                f"Plugin directories:\n{dir_list}\n\n"
                f"Place .py files in the plugin directory to add custom generators.\n"
                f"User plugin directory: {user_dir}"
            )

        QMessageBox.information(self, "Plugin Manager", msg)

    def _run_optimization(
        self,
        layers: list[Layer],
        bounds: tuple[float, float, float, float],
        settings: dict | None = None,
    ) -> None:
        """Run optimization on each layer sequentially (one worker per layer)."""
        self._opt_layers = list(layers)
        self._opt_bounds = bounds
        self._opt_settings = settings  # None → use worker defaults
        self._opt_layer_idx = 0
        self._opt_results: list[tuple[Layer, list[Polyline], float, float, int, int]] = []

        # Range is 0..100*n_layers so within-layer progress drives the bar smoothly
        progress = QProgressDialog(
            "Optimizing paths…", "Cancel", 0, max(1, len(layers)) * 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        self._opt_progress = progress

        self._start_next_opt_layer()

    def _start_next_opt_layer(self) -> None:
        if self._opt_layer_idx >= len(self._opt_layers):
            self._finish_optimization()
            return

        if self._opt_progress.wasCanceled():
            self._finish_optimization(cancelled=True)
            return

        layer = self._opt_layers[self._opt_layer_idx]
        n_layers = len(self._opt_layers)
        base_progress = self._opt_layer_idx * 100

        self._opt_progress.setLabelText(
            f"Optimizing '{layer.name}' ({self._opt_layer_idx + 1}/{n_layers})…\n"
            f"Step: Preprocessing"
        )

        s = self._opt_settings or {}
        worker = _OptimizeWorker(
            paths=list(layer.paths),
            run_weld=s.get("run_weld", False),
            weld_tolerance=s.get("weld_tolerance", 0.1),
            run_simplify=s.get("run_simplify", True),
            simplify_tolerance=s.get("simplify_tolerance", 0.1),
            run_filter=s.get("run_filter", True),
            filter_min_length=s.get("filter_min_length", 0.5),
            run_clip=s.get("run_clip", True),
            clip_bounds=self._opt_bounds,
            run_merge=s.get("run_merge", True),
            merge_threshold=s.get("merge_threshold", 0.5),
            run_2opt=s.get("run_2opt", True),
            run_3opt=s.get("run_3opt", False),
            run_or_opt=s.get("run_or_opt", True),
            num_starts=5,
            parent=self,
        )

        def on_progress(value: int) -> None:
            self._opt_progress.setValue(base_progress + value)
            if value < 10:
                step = "Preprocessing"
            elif value < 35:
                step = "Reordering paths"
            elif value < 55:
                step = "Running 2-opt"
            elif value < 75:
                step = "Running 3-opt"
            else:
                step = "Running Or-opt"
            self._opt_progress.setLabelText(
                f"Optimizing '{layer.name}' ({self._opt_layer_idx + 1}/{n_layers})…\n"
                f"Step: {step}"
            )

        def on_finished(new_paths, before, after, before_lifts, after_lifts):
            self._opt_results.append((layer, new_paths, before, after, before_lifts, after_lifts))
            self._opt_layer_idx += 1
            self._opt_progress.setValue(self._opt_layer_idx * 100)
            self._start_next_opt_layer()
            worker.deleteLater()

        def on_error(msg):
            QMessageBox.critical(self, "Optimization Error", msg)
            self._opt_layer_idx += 1
            self._opt_progress.setValue(self._opt_layer_idx * 100)
            self._start_next_opt_layer()
            worker.deleteLater()

        # Wire cancel button to stop the worker gracefully.
        # Disconnect the previous layer's worker first to avoid accumulating
        # cancel connections across layers (each layer creates a new worker).
        prev_worker = getattr(self, "_opt_worker", None)
        if prev_worker is not None:
            try:
                self._opt_progress.canceled.disconnect(prev_worker.request_stop)
            except (RuntimeError, TypeError):
                pass  # already disconnected or deleted
        self._opt_progress.canceled.connect(worker.request_stop)

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.progress.connect(on_progress)
        self._opt_worker = worker
        worker.start()

    def _finish_optimization(self, cancelled: bool = False) -> None:
        self._opt_progress.close()

        if not self._opt_results:
            return

        # Apply results
        for layer, new_paths, _before, _after, _bl, _al in self._opt_results:
            self._controller.set_layer_paths(layer.id, new_paths, "Optimize Paths")

        if cancelled:
            return

        # Build metrics report
        lines = []
        total_before = 0.0
        total_after = 0.0
        total_lifts_before = 0
        total_lifts_after = 0
        for layer, _, before, after, lifts_before, lifts_after in self._opt_results:
            reduction = ((before - after) / before * 100) if before > 0 else 0.0
            lifts_delta = lifts_before - lifts_after
            lines.append(
                f"<b>{layer.name}</b>: {before:.0f} mm → {after:.0f} mm "
                f"({reduction:.1f}% reduction), "
                f"pen lifts {lifts_before} → {lifts_after} ({lifts_delta:+d})"
            )
            total_before += before
            total_after += after
            total_lifts_before += lifts_before
            total_lifts_after += lifts_after

        if len(self._opt_results) > 1:
            total_reduction = (
                ((total_before - total_after) / total_before * 100)
                if total_before > 0 else 0.0
            )
            total_lifts_delta = total_lifts_before - total_lifts_after
            lines.append(
                f"<br><b>Total</b>: {total_before:.0f} mm → {total_after:.0f} mm "
                f"({total_reduction:.1f}% reduction), "
                f"pen lifts {total_lifts_before} → {total_lifts_after} ({total_lifts_delta:+d})"
            )

        QMessageBox.information(
            self,
            "Optimization Complete",
            "<b>Pen-up travel distance &amp; pen lift count:</b><br><br>" + "<br>".join(lines),
        )

    # ------------------------------------------------------------------
    # Help menu actions
    # ------------------------------------------------------------------

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Plottter",
            "<b>Plottter v0.1.0</b><br><br>"
            "A desktop application for generating plotter-ready vector art "
            "from mathematical equations and raster images.<br><br>"
            "<b>Features:</b><br>"
            "• Math art generators: parametric curves, polar equations, "
            "L-systems, flow fields, grid patterns<br>"
            "• Image-to-lines: edge detection, hatching, flow fields, stippling<br>"
            "• Multi-layer system with color separation<br>"
            "• Path optimization for pen travel minimization<br>"
            "• SVG, HPGL, and G-code export<br><br>"
            "<b>Credits:</b><br>"
            "Inspired by DrawingBot V3 (open-source plotter art generator)<br>"
            "Built with Python 3.12, PyQt6, NumPy, OpenCV, Shapely, SciPy, "
            "svgwrite, and Pillow.<br><br>"
            "License: MIT",
        )

    def _on_kbd_shortcuts(self) -> None:
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        shortcuts = [
            ("New project", "Ctrl+N"),
            ("Open project", "Ctrl+O"),
            ("Save project", "Ctrl+S"),
            ("Save as", "Ctrl+Shift+S"),
            ("Export current layer", "Ctrl+E"),
            ("Export all layers", "Ctrl+Shift+E"),
            ("Quit", "Ctrl+Q"),
            ("Undo", "Ctrl+Z"),
            ("Redo", "Ctrl+Y"),
            ("Generate", "Ctrl+G"),
            ("Regenerate all 3D layers", "Ctrl+Shift+G"),
            ("Randomize parameters", "Ctrl+R"),
            ("Zoom in", "Ctrl+="),
            ("Zoom out", "Ctrl+-"),
            ("Zoom to fit", "Ctrl+0"),
            ("Toggle grid", "G"),
            ("Toggle registration marks", "R"),
            ("Toggle travel moves", "T"),
            ("Toggle image overlay", "I"),
            ("Step animation back", "Shift+Left"),
            ("Step animation forward", "Shift+Right"),
        ]

        table = QTableWidget(len(shortcuts), 2, dialog)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.verticalHeader().setVisible(False)

        for i, (action, shortcut) in enumerate(shortcuts):
            table.setItem(i, 0, QTableWidgetItem(action))
            table.setItem(i, 1, QTableWidgetItem(shortcut))

        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.exec()

    # ------------------------------------------------------------------
    # Window state persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("splitter_state", self._splitter.saveState())

    def _restore_state(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = settings.value("splitter_state")
        if splitter_state is not None:
            self._splitter.restoreState(splitter_state)

    def last_file_dir(self) -> str:
        """Return the last used file dialog directory (for open/save dialogs)."""
        settings = QSettings("Plottter", "Plottter")
        return settings.value("last_file_dir", "") or ""

    def save_last_file_dir(self, path: str) -> None:
        """Persist the directory of *path* so the next file dialog opens there."""
        import os
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("last_file_dir", os.path.dirname(path))

    def _recent_projects(self) -> list[str]:
        settings = QSettings("Plottter", "Plottter")
        return list(settings.value("recent_projects", []) or [])

    def _add_recent_project(self, path: str) -> None:
        settings = QSettings("Plottter", "Plottter")
        recent: list[str] = list(settings.value("recent_projects", []) or [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:10]
        settings.setValue("recent_projects", recent)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._recent_projects()
        if not recent:
            act = QAction("(empty)", self)
            act.setEnabled(False)
            self._recent_menu.addAction(act)
            return
        for path in recent:
            import os
            label = os.path.basename(path)
            act = QAction(label, self)
            act.setToolTip(path)
            act.triggered.connect(lambda checked, p=path: self._open_recent_project(p))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clear_act = QAction("Clear Recent Projects", self)
        clear_act.triggered.connect(self._clear_recent_projects)
        self._recent_menu.addAction(clear_act)

    def _open_recent_project(self, path: str) -> None:
        if not self._prompt_save_if_modified():
            return
        import os
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"File not found:\n{path}")
            return
        try:
            from plottter.io.project_file import load_project
            project = load_project(path)
            self._current_file = path
            self.save_last_file_dir(path)
            self._add_recent_project(path)
            self._controller.load_project(project)
        except Exception as exc:
            QMessageBox.critical(self, "Error Opening File", str(exc))

    def _clear_recent_projects(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.remove("recent_projects")
        self._rebuild_recent_menu()

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._prompt_save_if_modified():
            self._save_state()
            event.accept()
        else:
            event.ignore()

    def _prompt_save_if_modified(self) -> bool:
        """Ask the user to save if there are unsaved changes. Returns True to proceed."""
        if not self._controller.modified:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Save before proceeding?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._on_save()
            return not self._controller.modified  # True if save succeeded
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel
