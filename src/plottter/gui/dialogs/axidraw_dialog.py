"""AxiDraw plotter control dialog.

Provides settings for direct USB plotting via pyaxidraw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QMutex, QSettings, Qt, QThread, QWaitCondition, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from plottter.models.canvas import PAPER_PRESETS

if TYPE_CHECKING:
    from plottter.models.project import Project


# ---------------------------------------------------------------------------
# Background plot workers
# ---------------------------------------------------------------------------

class _PlotWorker(QThread):
    """Runs a single combined-SVG plot job in a background thread.

    Supports a software pause: ``pause()`` (called from the GUI thread) asks
    pyaxidraw to stop at the next safe point. The run then emits ``paused``
    with a resume-SVG instead of ``finished``; feeding that SVG back with
    ``resume=True`` continues the plot.
    """

    progress = pyqtSignal(float)          # 0–100
    finished = pyqtSignal()
    paused = pyqtSignal(str)              # resume SVG (may be empty)
    error = pyqtSignal(str)

    def __init__(self, svg_data: str, settings: dict, resume: bool = False, parent=None, transport=None):
        super().__init__(parent)
        self._svg_data = svg_data
        self._settings = settings
        self._resume = resume
        self._ad = None  # set by on_ready once plotting is configured
        from plottter.export.transport import LocalUsbTransport
        self._transport = transport or LocalUsbTransport()

    def pause(self) -> None:
        """Request a pause from the GUI thread (no-op if not yet plotting)."""
        ad = self._ad
        if ad is None:
            return
        try:
            ad.transmit_pause_request()
        except Exception:
            pass

    def _store_ad(self, ad) -> None:
        self._ad = ad

    def run(self) -> None:
        try:
            outcome = self._transport.plot_svg(
                self._svg_data,
                self._settings,
                self.progress.emit,
                on_ready=self._store_ad,
                resume=self._resume,
            )
            if outcome.paused:
                self.paused.emit(outcome.resume_svg or "")
            else:
                self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class _MultiLayerPlotWorker(QThread):
    """Plot a list of layer SVG jobs sequentially, pausing for pen swap."""

    progress = pyqtSignal(float)                    # 0–100 overall
    layer_started = pyqtSignal(int, str, str)       # idx, name, color
    pen_swap_requested = pyqtSignal(int, str, str)  # next_idx, name, color
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        layer_jobs: list[tuple[str, str, str]],
        settings: dict,
        parent=None,
        transport=None,
    ):
        super().__init__(parent)
        self._jobs = layer_jobs  # list of (name, color, svg)
        self._settings = settings
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._continue = False
        self._cancelled = False
        from plottter.export.transport import LocalUsbTransport
        self._transport = transport or LocalUsbTransport()

    def continue_plot(self) -> None:
        """Wake the worker after a pen swap."""
        self._mutex.lock()
        self._continue = True
        self._cond.wakeAll()
        self._mutex.unlock()

    def cancel(self) -> None:
        """Abort the plot — sets the cancel flag and wakes the worker."""
        self._mutex.lock()
        self._cancelled = True
        self._continue = True
        self._cond.wakeAll()
        self._mutex.unlock()

    def run(self) -> None:
        try:
            n = max(1, len(self._jobs))
            span = 100.0 / n
            for idx, (name, color, svg) in enumerate(self._jobs):
                if self._cancelled:
                    break
                self.layer_started.emit(idx, name, color)

                base = idx * span

                def cb(p: float, base=base, span=span) -> None:
                    self.progress.emit(base + p * span / 100.0)

                self._transport.plot_svg(svg, self._settings, cb)

                # Pause before next layer for pen swap.
                if idx < len(self._jobs) - 1:
                    next_name, next_color, _ = self._jobs[idx + 1]
                    self.pen_swap_requested.emit(idx + 1, next_name, next_color)
                    self._mutex.lock()
                    self._continue = False
                    while not self._continue:
                        self._cond.wait(self._mutex)
                    self._mutex.unlock()

            if self._cancelled:
                self.error.emit("Plot cancelled.")
            else:
                self.progress.emit(100.0)
                self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class _ManualCommandWorker(QThread):
    """Runs a one-off manual AxiDraw command (raise/lower pen, release motors)."""

    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, command: str, settings: dict, parent=None, transport=None):
        super().__init__(parent)
        self._command = command
        self._settings = settings
        from plottter.export.transport import LocalUsbTransport
        self._transport = transport or LocalUsbTransport()

    def run(self) -> None:
        try:
            self._transport.run_manual(self._command, self._settings)
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Helper: build a colour swatch pixmap for the pen-swap modal
# ---------------------------------------------------------------------------

def _make_color_swatch(color_hex: str, size: int = 48) -> QPixmap:
    """Return a square pixmap filled with *color_hex* and a thin border."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    qcolor = QColor(color_hex) if color_hex else QColor("#000000")
    if not qcolor.isValid():
        qcolor = QColor("#000000")
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(2, 2, size - 4, size - 4, qcolor)
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawRect(2, 2, size - 5, size - 5)
    finally:
        painter.end()
    return pix


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class AxiDrawDialog(QDialog):
    """Settings dialog and plot launcher for the AxiDraw plotter."""

    # Emitted when a plot job starts (for status bar messaging)
    plot_started = pyqtSignal()
    plot_finished = pyqtSignal()

    _MODEL_NAMES = [
        "AxiDraw V2 / V3",          # model=1
        "AxiDraw V3/A3 or SE/A3",   # model=2
        "AxiDraw V3 XLX",           # model=3
        "AxiDraw MiniKit",           # model=4
        "AxiDraw SE/A1",             # model=5
        "AxiDraw SE/A2",             # model=6
    ]

    # QSettings key + fallback for the remembered model selection.
    _MODEL_SETTINGS_KEY = "axidraw/model_index"
    _DEFAULT_MODEL_INDEX = 1  # AxiDraw V3/A3

    # Remote plotter (network) device settings — persisted across sessions.
    _REMOTE_ENABLED_KEY = "remote_plotter/enabled"
    _REMOTE_URL_KEY = "remote_plotter/url"
    _REMOTE_TOKEN_KEY = "remote_plotter/token"

    # Plot-orientation options (index -> (flip_x, flip_y)). Persisted so a
    # machine whose origin differs from the app's top-left only needs setting
    # once.
    _ORIENTATION_NAMES = [
        "Normal",
        "Flip horizontal (X) — mirrors, reverses text",
        "Flip vertical (Y) — mirrors, reverses text",
        "Rotate 180° — repositions, keeps text readable",
    ]
    _ORIENTATION_FLIPS = [
        (False, False),  # Normal
        (True, False),   # Flip X
        (False, True),   # Flip Y
        (True, True),    # Rotate 180°
    ]
    _ORIENTATION_SETTINGS_KEY = "axidraw/plot_orientation"

    # Plot bed size — pad every plot to one fixed document size so a given
    # coordinate maps to the same physical spot regardless of canvas size.
    _BED_MATCH_CANVAS = "Match canvas (no padding)"
    _BED_SETTINGS_KEY = "axidraw/bed_size"

    # Step (in pen-position %) applied by each live pressure-tuning nudge.
    _PRESSURE_STEP = 2

    def __init__(
        self,
        project: "Project",
        active_layer_id: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._active_layer_id = active_layer_id
        self._worker: QThread | None = None
        self._is_multilayer_worker = False
        self._resume_svg: str | None = None
        # The active plotter transport (USB today; network later). All plot /
        # manual / availability calls route through this.
        from plottter.export.transport import LocalUsbTransport
        self._transport = LocalUsbTransport()
        self.setWindowTitle("Plot with AxiDraw")
        self.setMinimumWidth(480)
        self._build_ui()
        self._apply_initial_size()
        self._update_layer_summary()
        self._resolve_transport()
        self._check_availability()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Status banner
        self._status_label = QLabel("Checking for pyaxidraw…")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # The dialog is sized to fit its content (see _apply_initial_size), so
        # horizontal scrolling should never be needed — disable it outright and
        # keep the vertical scrollbar only as a fallback on very small screens.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        scroll.setWidget(inner)
        self._scroll_inner = inner
        form_root = QVBoxLayout(inner)
        root.addWidget(scroll)

        # --- Layers group ---
        layers_group = QGroupBox("Layers")
        layers_layout = QVBoxLayout(layers_group)
        form_root.addWidget(layers_group)

        self._scope_all_radio = QRadioButton("All visible layers")
        self._scope_all_radio.setChecked(True)
        self._scope_single_radio = QRadioButton("Single layer:")
        self._scope_single_radio.setToolTip(
            "Plot exactly one layer — useful when stepping through a multi-pen "
            "project one pen at a time. Picking from the dropdown overrides "
            "the layer's visibility for this plot only."
        )

        # Dropdown of every layer in the project (with colour swatch icons).
        # Defaults to whichever layer was active when the dialog opened so the
        # common "plot the currently selected layer" workflow stays one click.
        self._layer_combo = QComboBox()
        self._populate_layer_combo()
        # Picking a layer from the dropdown implies the user wants to plot that
        # one layer — auto-switch the radio so they don't have to click twice.
        self._layer_combo.activated.connect(
            lambda _i: self._scope_single_radio.setChecked(True)
        )
        self._layer_combo.currentIndexChanged.connect(self._update_layer_summary)

        # Disable the single-layer mode only when the project has no layers at
        # all — with the dropdown there's no "no active layer" failure mode.
        if not self._project.layers:
            self._scope_single_radio.setEnabled(False)
            self._layer_combo.setEnabled(False)
            self._scope_single_radio.setToolTip("Project has no layers.")

        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self._scope_all_radio)
        self._scope_group.addButton(self._scope_single_radio)
        self._scope_all_radio.toggled.connect(self._update_layer_summary)
        self._scope_single_radio.toggled.connect(self._update_layer_summary)

        layers_layout.addWidget(self._scope_all_radio)

        # Put the radio + combo on one row so the dropdown reads as "this radio's
        # value", not as a separate setting below it.
        single_row = QHBoxLayout()
        single_row.setContentsMargins(0, 0, 0, 0)
        single_row.addWidget(self._scope_single_radio)
        single_row.addWidget(self._layer_combo, 1)
        layers_layout.addLayout(single_row)

        self._pause_check = QCheckBox("Pause for pen swap between layers")
        self._pause_check.setToolTip(
            "After each layer finishes, the dialog will pause and prompt you "
            "to insert the next pen before continuing.  Has no effect when "
            "only one layer will be plotted."
        )
        layers_layout.addWidget(self._pause_check)

        self._layer_summary_label = QLabel("")
        self._layer_summary_label.setWordWrap(True)
        self._layer_summary_label.setStyleSheet("color: #888; font-size: 11px;")
        layers_layout.addWidget(self._layer_summary_label)

        # --- Device group ---
        dev_group = QGroupBox("Device")
        dev_layout = QFormLayout(dev_group)
        form_root.addWidget(dev_group)

        self._model_combo = QComboBox()
        for name in self._MODEL_NAMES:
            self._model_combo.addItem(name)
        # Restore the last-used model, falling back to the V3/A3 default.
        settings = QSettings("Plottter", "Plottter")
        saved_idx = settings.value(
            self._MODEL_SETTINGS_KEY, self._DEFAULT_MODEL_INDEX, type=int
        )
        if not 0 <= saved_idx < len(self._MODEL_NAMES):
            saved_idx = self._DEFAULT_MODEL_INDEX
        self._model_combo.setCurrentIndex(saved_idx)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        dev_layout.addRow("Model:", self._model_combo)

        # Plot orientation — corrects for plotters whose physical origin sits
        # on a different corner than the app's top-left (0, 0).
        self._orientation_combo = QComboBox()
        for name in self._ORIENTATION_NAMES:
            self._orientation_combo.addItem(name)
        saved_orient = settings.value(
            self._ORIENTATION_SETTINGS_KEY, 0, type=int
        )
        if not 0 <= saved_orient < len(self._ORIENTATION_NAMES):
            saved_orient = 0
        self._orientation_combo.setCurrentIndex(saved_orient)
        self._orientation_combo.setToolTip(
            "Mirror or rotate the plot to match your plotter's physical origin "
            "if it isn't the app's top-left corner. Does not change the on-screen "
            "drawing — only what is sent to the device."
        )
        self._orientation_combo.currentIndexChanged.connect(
            self._on_orientation_changed
        )
        dev_layout.addRow("Plot orientation:", self._orientation_combo)

        # Plot bed size — pads every plot to one fixed size (with art anchored
        # at the top-left) so the same coordinate lands at the same physical
        # spot no matter what canvas the design came from. Set this to your
        # plotter's bed for the paper-alignment workflow to line up.
        self._bed_combo = QComboBox()
        self._bed_combo.addItem(self._BED_MATCH_CANVAS)
        for _bed_name in PAPER_PRESETS:
            self._bed_combo.addItem(_bed_name)
        saved_bed = settings.value(
            self._BED_SETTINGS_KEY, self._BED_MATCH_CANVAS, type=str
        )
        bed_idx = self._bed_combo.findText(saved_bed)
        self._bed_combo.setCurrentIndex(bed_idx if bed_idx >= 0 else 0)
        self._bed_combo.setToolTip(
            "Pad every plot to this fixed size so a coordinate always maps to "
            "the same physical point — needed for the paper-size alignment "
            "guide to line up with designs from differently-sized canvases. "
            "Set it to your plotter's bed size. 'Match canvas' plots at the "
            "project canvas size (the original behaviour)."
        )
        self._bed_combo.currentIndexChanged.connect(self._on_bed_changed)
        dev_layout.addRow("Plot bed size:", self._bed_combo)

        self._port_label = QLabel("Auto-detect")
        dev_layout.addRow("USB Port:", self._port_label)

        self._preview_check = QCheckBox("Preview mode (no device required)")
        self._preview_check.setToolTip(
            "Simulate the plot without sending commands to a device. "
            "Useful for checking settings."
        )
        dev_layout.addRow("", self._preview_check)

        # --- Remote plotter (network) group ---
        remote_group = QGroupBox("Remote Plotter (network)")
        remote_layout = QFormLayout(remote_group)
        form_root.addWidget(remote_group)

        settings = QSettings("Plottter", "Plottter")
        self._remote_enabled_check = QCheckBox("Send to remote device instead of USB")
        self._remote_enabled_check.setToolTip(
            "Offload plotting to a networked plot daemon (e.g. a Raspberry Pi "
            "connected to the plotter) so this laptop is free during long plots."
        )
        self._remote_enabled_check.setChecked(
            settings.value(self._REMOTE_ENABLED_KEY, False, type=bool)
        )
        remote_layout.addRow("", self._remote_enabled_check)

        self._remote_url_edit = QLineEdit(
            str(settings.value(self._REMOTE_URL_KEY, "") or "")
        )
        self._remote_url_edit.setPlaceholderText("http://plotter-pi.local:8080")
        remote_layout.addRow("Device URL:", self._remote_url_edit)

        self._remote_token_edit = QLineEdit(
            str(settings.value(self._REMOTE_TOKEN_KEY, "") or "")
        )
        self._remote_token_edit.setPlaceholderText("optional — leave blank if the daemon has no token")
        remote_layout.addRow("Token:", self._remote_token_edit)

        self._remote_refresh_btn = QPushButton("Refresh connection")
        self._remote_refresh_btn.setToolTip(
            "Re-check which plotter is connected after changing these settings "
            "or starting the daemon."
        )
        remote_layout.addRow("", self._remote_refresh_btn)

        self._remote_enabled_check.toggled.connect(self._on_remote_settings_changed)
        self._remote_url_edit.editingFinished.connect(self._on_remote_settings_changed)
        self._remote_token_edit.editingFinished.connect(self._on_remote_settings_changed)
        self._remote_refresh_btn.clicked.connect(self._on_remote_settings_changed)

        # --- Speed group ---
        speed_group = QGroupBox("Speed")
        speed_layout = QFormLayout(speed_group)
        form_root.addWidget(speed_group)

        self._speed_pendown = QSpinBox()
        self._speed_pendown.setRange(1, 100)
        self._speed_pendown.setValue(25)
        self._speed_pendown.setSuffix(" %")
        speed_layout.addRow("Drawing speed:", self._speed_pendown)

        self._speed_penup = QSpinBox()
        self._speed_penup.setRange(1, 100)
        self._speed_penup.setValue(75)
        self._speed_penup.setSuffix(" %")
        speed_layout.addRow("Travel speed:", self._speed_penup)

        self._const_speed = QCheckBox("Constant speed (ignore acceleration)")
        speed_layout.addRow("", self._const_speed)

        # --- Pen group ---
        pen_group = QGroupBox("Pen")
        pen_layout = QFormLayout(pen_group)
        form_root.addWidget(pen_group)

        self._pen_pos_down = QSpinBox()
        self._pen_pos_down.setRange(0, 100)
        self._pen_pos_down.setValue(40)
        self._pen_pos_down.setSuffix(" %")
        pen_layout.addRow("Pen-down position:", self._pen_pos_down)

        self._pen_pos_up = QSpinBox()
        self._pen_pos_up.setRange(0, 100)
        self._pen_pos_up.setValue(60)
        self._pen_pos_up.setSuffix(" %")
        pen_layout.addRow("Pen-up position:", self._pen_pos_up)

        self._pen_delay_down = QSpinBox()
        self._pen_delay_down.setRange(0, 2000)
        self._pen_delay_down.setValue(125)
        self._pen_delay_down.setSuffix(" ms")
        self._pen_delay_down.setToolTip(
            "Pause after lowering the pen, before the carriage starts moving. "
            "A short delay (~100–150 ms) lets the servo fully seat so strokes "
            "don't skip at the start. Set to 0 for the fastest plotting."
        )
        pen_layout.addRow("Delay after pen-down:", self._pen_delay_down)

        self._pen_delay_up = QSpinBox()
        self._pen_delay_up.setRange(0, 2000)
        self._pen_delay_up.setValue(0)
        self._pen_delay_up.setSuffix(" ms")
        pen_layout.addRow("Delay after pen-up:", self._pen_delay_up)

        # --- Manual controls: useful for locking in the pen position ---
        manual_label = QLabel(
            "Manual controls (use these to position the pen before plotting):"
        )
        manual_label.setStyleSheet("color: #aaa; font-size: 11px;")
        pen_layout.addRow(manual_label)

        manual_btn_row = QHBoxLayout()
        self._raise_pen_btn = QPushButton("Raise Pen")
        self._raise_pen_btn.setToolTip(
            "Lift the pen to the pen-up position. Uses the current Pen-up "
            "position setting above."
        )
        self._raise_pen_btn.clicked.connect(
            lambda: self._run_manual("raise_pen")
        )
        manual_btn_row.addWidget(self._raise_pen_btn)

        self._lower_pen_btn = QPushButton("Lower Pen")
        self._lower_pen_btn.setToolTip(
            "Lower the pen to the pen-down position. Useful for checking "
            "where the pen will contact the paper before locking it in. "
            "Uses the current Pen-down position setting above."
        )
        self._lower_pen_btn.clicked.connect(
            lambda: self._run_manual("lower_pen")
        )
        manual_btn_row.addWidget(self._lower_pen_btn)

        self._release_motors_btn = QPushButton("Release Motors")
        self._release_motors_btn.setToolTip(
            "Disengage the X/Y stepper motors so you can move the carriage "
            "by hand. The motors re-engage automatically on the next plot."
        )
        self._release_motors_btn.clicked.connect(
            lambda: self._run_manual("disable_xy")
        )
        manual_btn_row.addWidget(self._release_motors_btn)

        self._return_home_btn = QPushButton("Return Home")
        self._return_home_btn.setToolTip(
            "Move the pen carriage back to the home corner (0,0). Handy after "
            "stopping a plot with the plotter's pause button — it re-engages "
            "the motors and raises the pen for you. Only homes correctly if the "
            "carriage hasn't been pushed by hand since it stopped, and needs "
            "plotter firmware 2.6.2 or newer."
        )
        self._return_home_btn.clicked.connect(
            lambda: self._run_manual("walk_home")
        )
        manual_btn_row.addWidget(self._return_home_btn)
        pen_layout.addRow(manual_btn_row)

        # --- Live pressure tuning ---
        # Lower the pen, then nudge the pen-down position a couple % at a time
        # while the pen stays on the paper, so the user can dial in contact
        # pressure by feel without re-clamping the pen. A lower pen-down
        # position means the carriage drops the pen further = more pressure.
        pressure_label = QLabel(
            "Pressure tuning (lower the pen first, then nudge until it draws "
            "cleanly without skipping):"
        )
        pressure_label.setWordWrap(True)
        pressure_label.setStyleSheet("color: #aaa; font-size: 11px;")
        pen_layout.addRow(pressure_label)

        pressure_btn_row = QHBoxLayout()
        self._less_pressure_btn = QPushButton("− Less Pressure")
        self._less_pressure_btn.setToolTip(
            "Raise the pen-down position by "
            f"{self._PRESSURE_STEP}% and re-lower the pen — less contact force."
        )
        self._less_pressure_btn.clicked.connect(
            lambda: self._nudge_pressure(self._PRESSURE_STEP)
        )
        pressure_btn_row.addWidget(self._less_pressure_btn)

        self._more_pressure_btn = QPushButton("More Pressure +")
        self._more_pressure_btn.setToolTip(
            "Lower the pen-down position by "
            f"{self._PRESSURE_STEP}% and re-lower the pen — more contact force."
        )
        self._more_pressure_btn.clicked.connect(
            lambda: self._nudge_pressure(-self._PRESSURE_STEP)
        )
        pressure_btn_row.addWidget(self._more_pressure_btn)
        pen_layout.addRow(pressure_btn_row)

        self._manual_buttons = [
            self._raise_pen_btn,
            self._lower_pen_btn,
            self._release_motors_btn,
            self._return_home_btn,
            self._less_pressure_btn,
            self._more_pressure_btn,
        ]

        form_root.addStretch()

        # --- Progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        root.addLayout(btn_row)

        self._plot_btn = QPushButton("Plot Now")
        self._plot_btn.setDefault(True)
        self._plot_btn.clicked.connect(self._on_plot)
        btn_row.addWidget(self._plot_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setToolTip(
            "Pause the plot at the next safe point. The pen lifts and the "
            "carriage stops; click Resume to continue from where it left off."
        )
        self._pause_btn.setVisible(False)
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._resume_btn = QPushButton("Resume")
        self._resume_btn.setToolTip(
            "Continue a paused plot from where it stopped. Works whether you "
            "paused from here or with the plotter's physical button."
        )
        self._resume_btn.setVisible(False)
        self._resume_btn.clicked.connect(self._on_resume)
        btn_row.addWidget(self._resume_btn)

        self._cancel_btn = QPushButton("Close")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

    # ------------------------------------------------------------------
    # Initial sizing
    # ------------------------------------------------------------------

    def _apply_initial_size(self) -> None:
        """Resize so all controls fit without scrolling on a typical screen.

        The scrollable content is the tallest part of the dialog; we size to
        its natural ``sizeHint`` plus the surrounding chrome (status banner,
        progress bar, button row, margins), then cap to the available screen
        so the dialog never opens larger than the display.
        """
        inner_hint = self._scroll_inner.sizeHint()
        chrome_h = (
            self._status_label.sizeHint().height()
            + self._progress_bar.sizeHint().height()
            + self._plot_btn.sizeHint().height()
            + 48  # layout margins + spacing
        )
        target_w = max(self.minimumWidth(), inner_hint.width() + 24)
        target_h = inner_hint.height() + chrome_h

        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = min(target_w, avail.width() - 80)
            target_h = min(target_h, avail.height() - 80)

        self.resize(target_w, target_h)

    def _on_model_changed(self, index: int) -> None:
        """Persist the selected plotter model so it's the default next time."""
        if 0 <= index < len(self._MODEL_NAMES):
            settings = QSettings("Plottter", "Plottter")
            settings.setValue(self._MODEL_SETTINGS_KEY, index)

    def _on_orientation_changed(self, index: int) -> None:
        """Persist the plot-orientation choice so it sticks across sessions."""
        if 0 <= index < len(self._ORIENTATION_NAMES):
            settings = QSettings("Plottter", "Plottter")
            settings.setValue(self._ORIENTATION_SETTINGS_KEY, index)

    def _on_bed_changed(self, index: int) -> None:
        """Persist the plot bed-size choice so it sticks across sessions."""
        settings = QSettings("Plottter", "Plottter")
        settings.setValue(self._BED_SETTINGS_KEY, self._bed_combo.currentText())

    def _bed_size_mm(self) -> tuple[float | None, float | None]:
        """Return the configured bed (width, height) in mm, or (None, None).

        ``None`` means "match canvas" — the export then uses the canvas size.
        """
        label = self._bed_combo.currentText()
        if label == self._BED_MATCH_CANVAS:
            return None, None
        dims = PAPER_PRESETS.get(label)
        if not dims:
            return None, None
        return float(dims[0]), float(dims[1])

    # ------------------------------------------------------------------
    # Layer selection helpers
    # ------------------------------------------------------------------

    def _find_layer(self, layer_id: str | None):
        if layer_id is None:
            return None
        for lyr in self._project.layers:
            if lyr.id == layer_id:
                return lyr
        return None

    def _populate_layer_combo(self) -> None:
        """Fill the single-layer dropdown with every layer + colour swatch.

        Defaults the selection to ``self._active_layer_id`` so the dialog opens
        on the currently selected layer — preserving the old "Active layer
        only" one-click workflow.
        """
        self._layer_combo.clear()
        for lyr in self._project.layers:
            icon = QIcon(_make_color_swatch(lyr.color, size=16))
            label = lyr.name if lyr.visible else f"{lyr.name} (hidden)"
            self._layer_combo.addItem(icon, label, lyr.id)
        # Default to the layer that was active when the dialog opened.
        if self._active_layer_id is not None:
            for i in range(self._layer_combo.count()):
                if self._layer_combo.itemData(i) == self._active_layer_id:
                    self._layer_combo.setCurrentIndex(i)
                    break

    def _selected_layer_ids(self) -> list[str] | None:
        """Return explicit layer-id list, or None for 'all visible'."""
        if self._scope_single_radio.isChecked():
            layer_id = self._layer_combo.currentData()
            return [layer_id] if layer_id else []
        return None

    def _layers_to_plot(self) -> list:
        """Compute the actual list of Layer objects that will be plotted."""
        ids = self._selected_layer_ids()
        if ids is None:
            return [lyr for lyr in self._project.layers if lyr.visible]
        id_set = set(ids)
        return [lyr for lyr in self._project.layers if lyr.id in id_set]

    def _update_layer_summary(self) -> None:
        """Refresh the summary label and pause-checkbox enabled state."""
        layers = self._layers_to_plot()
        n = len(layers)
        total_paths = sum(len(lyr.paths) for lyr in layers)
        if n == 0:
            self._layer_summary_label.setText("⚠ No layers will be plotted.")
        elif n == 1:
            self._layer_summary_label.setText(
                f"Will plot 1 layer (“{layers[0].name}”) — {total_paths} paths."
            )
        else:
            preview = ", ".join(f"“{lyr.name}”" for lyr in layers[:3])
            if n > 3:
                preview += f", … (+{n - 3} more)"
            self._layer_summary_label.setText(
                f"Will plot {n} layers ({preview}) — {total_paths} paths total."
            )

        # Pause only matters when there are 2+ layers.
        self._pause_check.setEnabled(n >= 2)
        if n < 2:
            self._pause_check.setChecked(False)

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def _resolve_transport(self) -> None:
        """Pick the active transport from the remote-device settings.

        Remote enabled + a URL → ``NetworkTransport``; otherwise ``LocalUsbTransport``.
        ``_check_availability`` then reports whether that transport is reachable.
        """
        from plottter.export.transport import LocalUsbTransport, NetworkTransport

        use_remote = self._remote_enabled_check.isChecked()
        url = self._remote_url_edit.text().strip()
        token = self._remote_token_edit.text().strip() or None
        if use_remote and url:
            self._transport = NetworkTransport(url, token)
        else:
            self._transport = LocalUsbTransport()

    def _on_remote_settings_changed(self) -> None:
        """Persist remote-device settings, then re-resolve + re-check.

        No-op while a plot is running so the transport never swaps mid-plot.
        """
        if self._worker is not None and self._worker.isRunning():
            return
        settings = QSettings("Plottter", "Plottter")
        settings.setValue(self._REMOTE_ENABLED_KEY, self._remote_enabled_check.isChecked())
        settings.setValue(self._REMOTE_URL_KEY, self._remote_url_edit.text().strip())
        settings.setValue(self._REMOTE_TOKEN_KEY, self._remote_token_edit.text().strip())
        self._resolve_transport()
        self._check_availability()

    def _check_availability(self) -> None:
        status = self._transport.health()
        self._status_label.setText(status.detail)
        if status.connected:
            self._status_label.setStyleSheet("")
        else:
            self._status_label.setStyleSheet("color: #cc4400;")
            self._preview_check.setChecked(True)

    # ------------------------------------------------------------------
    # Plot action
    # ------------------------------------------------------------------

    def _build_settings(self) -> dict:
        flip_x, flip_y = self._ORIENTATION_FLIPS[
            self._orientation_combo.currentIndex()
        ]
        bed_w, bed_h = self._bed_size_mm()
        return {
            "model": self._model_combo.currentIndex() + 1,
            "flip_x": flip_x,
            "flip_y": flip_y,
            "bed_width_mm": bed_w,
            "bed_height_mm": bed_h,
            "speed_pendown": self._speed_pendown.value(),
            "speed_penup": self._speed_penup.value(),
            "pen_pos_down": self._pen_pos_down.value(),
            "pen_pos_up": self._pen_pos_up.value(),
            "pen_delay_down": self._pen_delay_down.value(),
            "pen_delay_up": self._pen_delay_up.value(),
            "const_speed": self._const_speed.isChecked(),
            "preview": self._preview_check.isChecked(),
            "report_time": True,
        }

    def _on_plot(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        layers = self._layers_to_plot()
        if not layers:
            QMessageBox.warning(
                self,
                "Nothing to Plot",
                "No layers are selected for plotting. Toggle visibility on "
                "the layers you want to plot, or pick “Active layer only”.",
            )
            return

        settings = self._build_settings()
        layer_ids = self._selected_layer_ids()
        pen_swap = self._pause_check.isChecked() and len(layers) >= 2

        self._resume_svg = None
        self._progress_bar.setValue(0)
        self.plot_started.emit()

        if pen_swap:
            # Multi-layer pen-swap mode handles its own between-layer pausing;
            # software pause/resume is not offered here.
            from plottter.export.axidraw import project_to_layer_svg_list
            jobs = project_to_layer_svg_list(self._project, layer_ids, settings)
            worker = _MultiLayerPlotWorker(jobs, settings, parent=None, transport=self._transport)
            worker.progress.connect(lambda p: self._progress_bar.setValue(int(p)))
            worker.pen_swap_requested.connect(self._on_pen_swap_requested)
            worker.finished.connect(self._on_plot_finished)
            worker.error.connect(self._on_plot_error)
            self._worker = worker
            self._is_multilayer_worker = True
            self._set_plot_ui_state("plotting", allow_pause=False)
            worker.start()
        else:
            from plottter.export.axidraw import project_to_svg_string
            svg_data = project_to_svg_string(self._project, layer_ids, settings)
            worker = _PlotWorker(svg_data, settings, parent=None, transport=self._transport)
            worker.progress.connect(lambda p: self._progress_bar.setValue(int(p)))
            worker.finished.connect(self._on_plot_finished)
            worker.paused.connect(self._on_plot_paused)
            worker.error.connect(self._on_plot_error)
            self._worker = worker
            self._is_multilayer_worker = False
            self._set_plot_ui_state("plotting", allow_pause=True)
            worker.start()

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def _set_plot_ui_state(self, state: str, allow_pause: bool = True) -> None:
        """Toggle button visibility/enabled for 'idle', 'plotting', 'paused'."""
        plotting = state == "plotting"
        paused = state == "paused"
        idle = state == "idle"
        self._plot_btn.setEnabled(idle)
        # Manual controls are usable while idle or paused, not while plotting.
        self._set_manual_buttons_enabled(not plotting)
        self._progress_bar.setVisible(plotting or paused)
        self._pause_btn.setVisible(plotting and allow_pause)
        self._pause_btn.setEnabled(plotting and allow_pause)
        self._pause_btn.setText("Pause")
        self._resume_btn.setVisible(paused)
        self._resume_btn.setEnabled(paused)

    def _on_pause(self) -> None:
        """Ask the running plot worker to pause."""
        worker = self._worker
        if isinstance(worker, _PlotWorker) and worker.isRunning():
            worker.pause()
            self._pause_btn.setEnabled(False)
            self._pause_btn.setText("Pausing…")

    def _on_plot_paused(self, resume_svg: str) -> None:
        """Plot stopped early (software pause or physical button)."""
        self._resume_svg = resume_svg or None
        if not self._resume_svg:
            # Stopped with no resume data — can't continue; reset to idle.
            self._set_plot_ui_state("idle")
            self.plot_finished.emit()
            QMessageBox.warning(
                self,
                "Plot Stopped",
                "The plot stopped but no resume data was returned, so it can't "
                "be continued. You'll need to start the plot again.",
            )
            return
        self._set_plot_ui_state("paused")
        QMessageBox.information(
            self,
            "Plot Paused",
            "Plot paused. Click Resume to continue from where it stopped, "
            "or Close to stop.",
        )

    def _on_resume(self) -> None:
        """Continue a paused plot from its saved resume SVG."""
        if not self._resume_svg or (self._worker and self._worker.isRunning()):
            return
        settings = self._build_settings()
        self._progress_bar.setValue(0)
        worker = _PlotWorker(self._resume_svg, settings, resume=True, parent=None, transport=self._transport)
        worker.progress.connect(lambda p: self._progress_bar.setValue(int(p)))
        worker.finished.connect(self._on_plot_finished)
        worker.paused.connect(self._on_plot_paused)
        worker.error.connect(self._on_plot_error)
        self._worker = worker
        self._is_multilayer_worker = False
        self._set_plot_ui_state("plotting", allow_pause=True)
        worker.start()

    # ------------------------------------------------------------------
    # Pen-swap pause handling
    # ------------------------------------------------------------------

    def _on_pen_swap_requested(self, next_idx: int, name: str, color: str) -> None:
        """Show modal prompting the user to swap pens before the next layer."""
        worker = self._worker
        if not isinstance(worker, _MultiLayerPlotWorker):
            return

        box = QMessageBox(self)
        box.setWindowTitle("Swap Pen")
        box.setIconPixmap(_make_color_swatch(color, size=56))
        box.setText(f"<b>Insert the pen for layer “{name}”</b>")
        box.setInformativeText(
            f"Colour: <span style='font-family: monospace;'>{color}</span><br><br>"
            "The pen carriage has returned home. Swap the pen, then click "
            "<b>Continue</b> to plot the next layer."
        )
        cont_btn = box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton("Cancel Plot", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cont_btn)
        box.exec()

        if box.clickedButton() is cancel_btn:
            worker.cancel()
        else:
            worker.continue_plot()

    # ------------------------------------------------------------------
    # Manual pen / motor controls
    # ------------------------------------------------------------------

    def _set_manual_buttons_enabled(self, enabled: bool) -> None:
        for btn in getattr(self, "_manual_buttons", []):
            btn.setEnabled(enabled)

    def _run_manual(self, command: str) -> None:
        """Fire a one-off manual AxiDraw command in a background thread."""
        if self._worker and self._worker.isRunning():
            return
        # Disable all manual buttons during the command so the user can't
        # queue up two commands while one is in flight.
        self._set_manual_buttons_enabled(False)
        self._plot_btn.setEnabled(False)

        settings = self._build_settings()
        self._manual_worker = _ManualCommandWorker(command, settings, parent=None, transport=self._transport)
        self._manual_worker.finished.connect(self._on_manual_finished)
        self._manual_worker.error.connect(self._on_manual_error)
        self._manual_worker.start()

    def _nudge_pressure(self, delta: int) -> None:
        """Adjust the pen-down position by *delta* % and re-lower the pen.

        Negative *delta* drops the pen further (more pressure); positive lifts
        it slightly (less pressure). Re-lowering immediately lets the user feel
        the new contact force while tuning. No-ops at the 0–100 % limits and
        while another command/plot is in flight.
        """
        if self._worker and self._worker.isRunning():
            return
        current = self._pen_pos_down.value()
        new_val = max(0, min(100, current + delta))
        if new_val == current:
            return  # already at the limit — nothing to apply
        self._pen_pos_down.setValue(new_val)
        self._run_manual("lower_pen")

    def _on_manual_finished(self) -> None:
        self._set_manual_buttons_enabled(True)
        self._plot_btn.setEnabled(True)

    def _on_manual_error(self, msg: str) -> None:
        self._set_manual_buttons_enabled(True)
        self._plot_btn.setEnabled(True)
        QMessageBox.warning(self, "Manual Command Failed", msg)

    def _on_plot_finished(self) -> None:
        self._progress_bar.setValue(100)
        self._resume_svg = None
        self._set_plot_ui_state("idle")
        self.plot_finished.emit()
        QMessageBox.information(
            self,
            "Plot Complete",
            "Plot job completed successfully.",
        )

    def _on_plot_error(self, msg: str) -> None:
        self._resume_svg = None
        self._set_plot_ui_state("idle")
        self.plot_finished.emit()
        # User-cancelled plots are not really errors — use an info dialog.
        if msg == "Plot cancelled.":
            QMessageBox.information(self, "Plot Cancelled", msg)
        else:
            QMessageBox.critical(self, "Plot Error", msg)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        worker = self._worker
        if worker is not None and worker.isRunning():
            if isinstance(worker, _MultiLayerPlotWorker):
                worker.cancel()
            elif isinstance(worker, _PlotWorker):
                worker.pause()  # stop the plot cleanly before closing
            worker.wait(3000)
        super().closeEvent(event)
