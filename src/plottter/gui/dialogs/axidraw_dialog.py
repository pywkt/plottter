"""AxiDraw plotter control dialog.

Provides settings for direct USB plotting via pyaxidraw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QMutex, Qt, QThread, QWaitCondition, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

if TYPE_CHECKING:
    from plottter.models.project import Project


# ---------------------------------------------------------------------------
# Background plot workers
# ---------------------------------------------------------------------------

class _PlotWorker(QThread):
    """Runs a single combined-SVG plot job in a background thread."""

    progress = pyqtSignal(float)          # 0–100
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, svg_data: str, settings: dict, parent=None):
        super().__init__(parent)
        self._svg_data = svg_data
        self._settings = settings

    def run(self) -> None:
        try:
            from plottter.export.axidraw import plot_svg_string
            plot_svg_string(self._svg_data, self._settings, self.progress.emit)
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
    ):
        super().__init__(parent)
        self._jobs = layer_jobs  # list of (name, color, svg)
        self._settings = settings
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._continue = False
        self._cancelled = False

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
            from plottter.export.axidraw import plot_svg_string

            n = max(1, len(self._jobs))
            span = 100.0 / n
            for idx, (name, color, svg) in enumerate(self._jobs):
                if self._cancelled:
                    break
                self.layer_started.emit(idx, name, color)

                base = idx * span

                def cb(p: float, base=base, span=span) -> None:
                    self.progress.emit(base + p * span / 100.0)

                plot_svg_string(svg, self._settings, cb)

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

    def __init__(self, command: str, settings: dict, parent=None):
        super().__init__(parent)
        self._command = command
        self._settings = settings

    def run(self) -> None:
        try:
            from plottter.export.axidraw import run_manual_command
            run_manual_command(self._command, self._settings)
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
        self.setWindowTitle("Plot with AxiDraw")
        self.setMinimumWidth(420)
        self._build_ui()
        self._update_layer_summary()
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
        inner = QWidget()
        scroll.setWidget(inner)
        form_root = QVBoxLayout(inner)
        root.addWidget(scroll)

        # --- Layers group ---
        layers_group = QGroupBox("Layers")
        layers_layout = QVBoxLayout(layers_group)
        form_root.addWidget(layers_group)

        self._scope_all_radio = QRadioButton("All visible layers")
        self._scope_all_radio.setChecked(True)
        self._scope_active_radio = QRadioButton("Active layer only")
        # Disable "Active layer only" when there's no active layer to plot.
        active_layer = self._find_layer(self._active_layer_id)
        if active_layer is None:
            self._scope_active_radio.setEnabled(False)
            self._scope_active_radio.setToolTip("No active layer selected.")

        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self._scope_all_radio)
        self._scope_group.addButton(self._scope_active_radio)
        self._scope_all_radio.toggled.connect(self._update_layer_summary)
        self._scope_active_radio.toggled.connect(self._update_layer_summary)

        layers_layout.addWidget(self._scope_all_radio)
        layers_layout.addWidget(self._scope_active_radio)

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
        self._model_combo.setCurrentIndex(1)  # V3/A3 default
        dev_layout.addRow("Model:", self._model_combo)

        self._port_label = QLabel("Auto-detect")
        dev_layout.addRow("USB Port:", self._port_label)

        self._preview_check = QCheckBox("Preview mode (no device required)")
        self._preview_check.setToolTip(
            "Simulate the plot without sending commands to a device. "
            "Useful for checking settings."
        )
        dev_layout.addRow("", self._preview_check)

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
        self._pen_delay_down.setValue(0)
        self._pen_delay_down.setSuffix(" ms")
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
        pen_layout.addRow(manual_btn_row)

        self._manual_buttons = [
            self._raise_pen_btn,
            self._lower_pen_btn,
            self._release_motors_btn,
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

        self._cancel_btn = QPushButton("Close")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

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

    def _selected_layer_ids(self) -> list[str] | None:
        """Return explicit layer-id list, or None for 'all visible'."""
        if self._scope_active_radio.isChecked():
            active = self._find_layer(self._active_layer_id)
            return [active.id] if active is not None else []
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

    def _check_availability(self) -> None:
        from plottter.export.axidraw import check_axidraw_available
        if check_axidraw_available():
            self._status_label.setText(
                "pyaxidraw is installed. Connect your AxiDraw via USB and click Plot Now."
            )
            self._status_label.setStyleSheet("")
        else:
            self._status_label.setText(
                "pyaxidraw is NOT installed.\n"
                "Install it with:\n"
                "  pip install https://cdn.evilmadscientist.com/dl/ad/public/AxiDraw_API.zip\n"
                "You can still use Preview mode to test settings without a device."
            )
            self._status_label.setStyleSheet("color: #cc4400;")
            self._preview_check.setChecked(True)

    # ------------------------------------------------------------------
    # Plot action
    # ------------------------------------------------------------------

    def _build_settings(self) -> dict:
        return {
            "model": self._model_combo.currentIndex() + 1,
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
        pause = self._pause_check.isChecked() and len(layers) >= 2

        self._plot_btn.setEnabled(False)
        self._set_manual_buttons_enabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self.plot_started.emit()

        if pause:
            from plottter.export.axidraw import project_to_layer_svg_list
            jobs = project_to_layer_svg_list(self._project, layer_ids, settings)
            worker = _MultiLayerPlotWorker(jobs, settings, parent=None)
            worker.progress.connect(lambda p: self._progress_bar.setValue(int(p)))
            worker.pen_swap_requested.connect(self._on_pen_swap_requested)
            worker.finished.connect(self._on_plot_finished)
            worker.error.connect(self._on_plot_error)
            self._worker = worker
            self._is_multilayer_worker = True
            worker.start()
        else:
            from plottter.export.axidraw import project_to_svg_string
            svg_data = project_to_svg_string(self._project, layer_ids, settings)
            worker = _PlotWorker(svg_data, settings, parent=None)
            worker.progress.connect(lambda p: self._progress_bar.setValue(int(p)))
            worker.finished.connect(self._on_plot_finished)
            worker.error.connect(self._on_plot_error)
            self._worker = worker
            self._is_multilayer_worker = False
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
        self._manual_worker = _ManualCommandWorker(command, settings, parent=None)
        self._manual_worker.finished.connect(self._on_manual_finished)
        self._manual_worker.error.connect(self._on_manual_error)
        self._manual_worker.start()

    def _on_manual_finished(self) -> None:
        self._set_manual_buttons_enabled(True)
        self._plot_btn.setEnabled(True)

    def _on_manual_error(self, msg: str) -> None:
        self._set_manual_buttons_enabled(True)
        self._plot_btn.setEnabled(True)
        QMessageBox.warning(self, "Manual Command Failed", msg)

    def _on_plot_finished(self) -> None:
        self._progress_bar.setValue(100)
        self._plot_btn.setEnabled(True)
        self._set_manual_buttons_enabled(True)
        self.plot_finished.emit()
        QMessageBox.information(
            self,
            "Plot Complete",
            "Plot job completed successfully.",
        )

    def _on_plot_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._plot_btn.setEnabled(True)
        self._set_manual_buttons_enabled(True)
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
            worker.wait(3000)
        super().closeEvent(event)
