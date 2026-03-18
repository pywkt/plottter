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
                "Install it with:  pip install pyaxidraw\n"
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
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self.plot_started.emit()

        self._worker = _PlotWorker(self._svg_data, settings, parent=None)
        self._worker.progress.connect(lambda p: self._progress_bar.setValue(int(p)))
        self._worker.finished.connect(self._on_plot_finished)
        self._worker.error.connect(self._on_plot_error)
        self._worker.start()

    def _on_plot_finished(self) -> None:
        self._progress_bar.setValue(100)
        self._plot_btn.setEnabled(True)
        self.plot_finished.emit()
        QMessageBox.information(
            self,
            "Plot Complete",
            "Plot job completed successfully.",
        )

    def _on_plot_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._plot_btn.setEnabled(True)
        QMessageBox.critical(self, "Plot Error", msg)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._worker and self._worker.isRunning():
            self._worker.wait(3000)
        super().closeEvent(event)
