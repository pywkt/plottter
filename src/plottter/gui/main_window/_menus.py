"""_MenusMixin — menu bar, toolbar, status bar construction and related handlers."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolBar,
)


class _MenusMixin:
    """Mixin providing menu/toolbar/status-bar construction for MainWindow."""

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

        self._act_center = QAction("&Center View", self)
        self._act_center.setShortcut(QKeySequence("Ctrl+Shift+0"))
        self._act_center.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self._act_center.triggered.connect(self._canvas.center_view)
        view_menu.addAction(self._act_center)

        # Pan actions without keyboard shortcuts — arrow keys are handled by
        # canvas keyPressEvent (task 96.1) so they don't conflict with the
        # layer panel's own arrow-key navigation.
        self._act_pan_left = QAction("Pan &Left", self)
        self._act_pan_left.triggered.connect(self._canvas.pan_left)
        view_menu.addAction(self._act_pan_left)

        self._act_pan_right = QAction("Pan &Right", self)
        self._act_pan_right.triggered.connect(self._canvas.pan_right)
        view_menu.addAction(self._act_pan_right)

        self._act_pan_up = QAction("Pan &Up", self)
        self._act_pan_up.triggered.connect(self._canvas.pan_up)
        view_menu.addAction(self._act_pan_up)

        self._act_pan_down = QAction("Pan &Down", self)
        self._act_pan_down.triggered.connect(self._canvas.pan_down)
        view_menu.addAction(self._act_pan_down)

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
        tools_menu = self._tools_menu = menu_bar.addMenu("&Tools")

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

        # Processing plugins submenu (populated dynamically by _rebuild_processing_plugins_menu)
        self._processing_plugins_menu = tools_menu.addMenu("Processing Plugins")
        self._processing_plugins_menu.setToolTipsVisible(True)
        self._rebuild_processing_plugins_menu()

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

    def _rebuild_processing_plugins_menu(self) -> None:
        """Repopulate the Processing Plugins submenu from PROCESSING_PLUGINS."""
        from plottter.processing.plugin import PROCESSING_PLUGINS

        menu = self._processing_plugins_menu
        menu.clear()

        if not PROCESSING_PLUGINS:
            placeholder = menu.addAction("(No processing plugins found)")
            placeholder.setEnabled(False)
            return

        for plugin_name in sorted(PROCESSING_PLUGINS.keys()):
            plugin_cls = PROCESSING_PLUGINS[plugin_name]
            act = QAction(plugin_name, self)
            if plugin_cls.description:
                act.setToolTip(plugin_cls.description)
            # Use a default argument to capture plugin_cls in the closure
            act.triggered.connect(
                lambda checked=False, cls=plugin_cls: self._on_run_processing_plugin(cls)
            )
            menu.addAction(act)

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._recent_projects()
        if not recent:
            act = QAction("(empty)", self)
            act.setEnabled(False)
            self._recent_menu.addAction(act)
            return
        for path in recent:
            label = os.path.basename(path)
            act = QAction(label, self)
            act.setToolTip(path)
            act.triggered.connect(lambda checked, p=path: self._open_recent_project(p))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clear_act = QAction("Clear Recent Projects", self)
        clear_act.triggered.connect(self._clear_recent_projects)
        self._recent_menu.addAction(clear_act)

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
