"""MainWindow — the top-level application window."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from plottter.gui.animation_bar import AnimationBar
from plottter.gui.canvas_widget import CanvasWidget
from plottter.gui.layer_panel import LayerPanel
from plottter.gui.mode_panel import ModePanel
from plottter.gui.project_controller import ProjectController
from plottter.gui.settings_panel import SettingsPanel

from ._canvas_ops import _CanvasOpsMixin
from ._file_ops import _FileOpsMixin
from ._generator_ops import _GeneratorOpsMixin
from ._menus import _MenusMixin
from ._plugin_ops import _PluginOpsMixin
from ._processing_ops import _ProcessingOpsMixin


class MainWindow(
    _FileOpsMixin,
    _MenusMixin,
    _ProcessingOpsMixin,
    _CanvasOpsMixin,
    _GeneratorOpsMixin,
    _PluginOpsMixin,
    QMainWindow,
):
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
        self._canvas.setToolTip(
            "Wheel to zoom · Ctrl+wheel to pan vertically · Alt+wheel to pan horizontally · "
            "Middle-drag or Space+drag to pan · Ctrl+= / Ctrl+\u2212 to zoom · Ctrl+0 to fit"
        )
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

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._prompt_save_if_modified():
            self._save_state()
            event.accept()
        else:
            event.ignore()
