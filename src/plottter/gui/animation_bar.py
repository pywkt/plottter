"""AnimationBar — playback controls for stroke-order animation."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class AnimationBar(QWidget):
    """Horizontal control bar for stroke-order animation playback.

    Signals:
        play_pause_toggled: emitted when the play/pause button is clicked.
        step_back_requested: emitted when step-back is clicked.
        step_forward_requested: emitted when step-forward is clicked.
        seek_requested(int): emitted when the seek slider is moved; carries the path index.
        speed_changed(float): emitted when the speed spinner value changes.
    """

    play_pause_toggled = pyqtSignal()
    step_back_requested = pyqtSignal()
    step_forward_requested = pyqtSignal()
    seek_requested = pyqtSignal(int)
    speed_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_playing = False
        self._updating_slider = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        self._btn_step_back = QPushButton("◀")
        self._btn_step_back.setFixedWidth(28)
        self._btn_step_back.setToolTip("Step backward one stroke (Shift+Left)")
        self._btn_step_back.clicked.connect(self.step_back_requested)
        layout.addWidget(self._btn_step_back)

        self._btn_play_pause = QPushButton("▶")
        self._btn_play_pause.setFixedWidth(32)
        self._btn_play_pause.setToolTip("Play / Pause animation")
        self._btn_play_pause.clicked.connect(self.play_pause_toggled)
        layout.addWidget(self._btn_play_pause)

        self._btn_step_forward = QPushButton("▶|")
        self._btn_step_forward.setFixedWidth(32)
        self._btn_step_forward.setToolTip("Step forward one stroke (Shift+Right)")
        self._btn_step_forward.clicked.connect(self.step_forward_requested)
        layout.addWidget(self._btn_step_forward)

        layout.addWidget(QLabel("|"))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setValue(0)
        self._slider.setToolTip("Seek to stroke")
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        layout.addWidget(QLabel("|"))

        layout.addWidget(QLabel("Speed:"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 10.0)
        self._speed_spin.setSingleStep(0.1)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setDecimals(1)
        self._speed_spin.setSuffix("x")
        self._speed_spin.setFixedWidth(65)
        self._speed_spin.setToolTip("Playback speed multiplier (0.1x – 10x)")
        self._speed_spin.valueChanged.connect(self.speed_changed)
        layout.addWidget(self._speed_spin)

    # ------------------------------------------------------------------
    # Slots called from CanvasWidget to update UI state
    # ------------------------------------------------------------------

    def set_playing(self, playing: bool) -> None:
        """Update play/pause button icon to reflect current state."""
        self._is_playing = playing
        self._btn_play_pause.setText("⏸" if playing else "▶")

    def set_total_paths(self, total: int) -> None:
        """Update the seek slider range to match total path count."""
        self._updating_slider = True
        self._slider.setRange(0, max(0, total - 1) if total > 0 else 0)
        self._updating_slider = False

    def set_position(self, path_idx: int) -> None:
        """Update the seek slider position without emitting seek_requested."""
        self._updating_slider = True
        self._slider.setValue(path_idx)
        self._updating_slider = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value: int) -> None:
        if not self._updating_slider:
            self.seek_requested.emit(value)
