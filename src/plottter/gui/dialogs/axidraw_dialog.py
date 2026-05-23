"""AxiDraw plotter control dialog.

Provides settings for direct USB plotting via pyaxidraw.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Background plot worker
# ---------------------------------------------------------------------------

class _PlotWorker(QThread):
    """Runs the AxiDraw plot job in a background thread."""

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

    def __init__(self, svg_data: str = "", parent=None):
        super().__init__(parent)
        self._svg_data = svg_data
        self._worker: _PlotWorker | None = None
        self.setWindowTitle("Plot with AxiDraw")
        self.setMinimumWidth(400)
        self._build_ui()
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

        settings = self._build_settings()

        self._plot_btn.setEnabled(False)
        self._set_manual_buttons_enabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self.plot_started.emit()

        self._worker = _PlotWorker(self._svg_data, settings, parent=None)
        self._worker.progress.connect(lambda p: self._progress_bar.setValue(int(p)))
        self._worker.finished.connect(self._on_plot_finished)
        self._worker.error.connect(self._on_plot_error)
        self._worker.start()

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
        QMessageBox.critical(self, "Plot Error", msg)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._worker and self._worker.isRunning():
            self._worker.wait(3000)
        super().closeEvent(event)
