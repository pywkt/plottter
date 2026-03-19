"""LayerPanel — list-based UI for managing project layers."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize, QEvent
from PyQt6.QtGui import QColor, QIcon, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from plottter.gui.dialogs.export import ExportDialog
from plottter.gui.project_controller import ProjectController


def _color_swatch_icon(hex_color: str, size: int = 16) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(hex_color))
    return QIcon(pixmap)


class _LayerItem(QWidget):
    """Custom widget for a single layer row in the list."""

    def __init__(
        self,
        layer_id: str,
        layer_name: str,
        layer_color: str,
        visible: bool,
        locked: bool,
        path_count: int,
        controller: ProjectController,
        opacity: float = 1.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layer_id = layer_id
        self._controller = controller
        self._edit_original_name = ""
        self._setup_ui(layer_name, layer_color, visible, locked, path_count, opacity)

    def _setup_ui(
        self, name: str, color: str, visible: bool, locked: bool, path_count: int, opacity: float = 1.0
    ) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Color swatch
        self._color_btn = QPushButton()
        self._color_btn.setIcon(_color_swatch_icon(color))
        self._color_btn.setIconSize(QSize(14, 14))
        self._color_btn.setFixedSize(22, 22)
        self._color_btn.setToolTip("Change layer color")
        self._color_btn.clicked.connect(self._on_color_click)
        layout.addWidget(self._color_btn)

        # Inline editable name — read-only by default; double-click to rename.
        # Explicitly use WindowText so Qt's selection cascade (which can force
        # white) does not bleed into the label.
        self._name_edit = QLineEdit(name)
        self._name_edit.setReadOnly(True)
        self._name_edit.setMinimumWidth(60)
        self._name_edit.setStyleSheet(
            "QLineEdit { border: none; background: transparent; color: palette(windowText); }"
            "QLineEdit:focus { border: 1px solid palette(highlight); background: palette(base); color: palette(text); }"
        )
        self._name_edit.editingFinished.connect(self._on_name_edited)
        self._name_edit.installEventFilter(self)
        layout.addWidget(self._name_edit, stretch=1)

        # Path count badge
        self._count_label = QLabel(f"[{path_count}]")
        self._count_label.setStyleSheet("color: palette(placeholderText); font-size: 10px;")
        layout.addWidget(self._count_label)

        # Visibility toggle
        self._vis_btn = QPushButton("👁" if visible else "◻")
        self._vis_btn.setFixedSize(22, 22)
        self._vis_btn.setToolTip("Toggle visibility")
        self._vis_btn.clicked.connect(self._on_vis_click)
        layout.addWidget(self._vis_btn)

        # Lock toggle
        self._lock_btn = QPushButton("🔒" if locked else "🔓")
        self._lock_btn.setFixedSize(22, 22)
        self._lock_btn.setToolTip("Toggle lock")
        self._lock_btn.clicked.connect(self._on_lock_click)
        layout.addWidget(self._lock_btn)

        # Opacity slider
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(round(opacity * 100))
        self._opacity_slider.setFixedWidth(60)
        self._opacity_slider.setToolTip("Layer opacity")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self._opacity_slider)

        self._opacity_label = QLabel(f"{round(opacity * 100)}%")
        self._opacity_label.setFixedWidth(34)
        self._opacity_label.setStyleSheet("color: palette(placeholderText); font-size: 10px;")
        layout.addWidget(self._opacity_label)

        self._visible = visible
        self._locked = locked
        self._color = color

    def _on_color_click(self) -> None:
        new_color = QColorDialog.getColor(QColor(self._color), self, "Layer Color")
        if new_color.isValid():
            self._color = new_color.name()
            self._color_btn.setIcon(_color_swatch_icon(self._color))
            self._controller.set_layer_color(self.layer_id, self._color)

    def _on_vis_click(self) -> None:
        self._visible = not self._visible
        self._vis_btn.setText("👁" if self._visible else "◻")
        self._controller.set_layer_visible(self.layer_id, self._visible)

    def _on_lock_click(self) -> None:
        self._locked = not self._locked
        self._lock_btn.setText("🔒" if self._locked else "🔓")
        self._controller.set_layer_locked(self.layer_id, self._locked)

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_label.setText(f"{value}%")
        self._controller.set_layer_opacity(self.layer_id, value / 100.0)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self._enter_edit_mode()
        super().mouseDoubleClickEvent(event)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj is self._name_edit:
            if event.type() == QEvent.Type.MouseButtonDblClick:
                self._enter_edit_mode()
                return True
            elif event.type() == QEvent.Type.FocusOut:
                self._exit_edit_mode(save=True)
            elif event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape:
                    self._exit_edit_mode(save=False)
                    return True
        return super().eventFilter(obj, event)

    def _enter_edit_mode(self) -> None:
        """Enter inline rename mode: make the name field editable and select all."""
        self._edit_original_name = self._name_edit.text()
        self._name_edit.setReadOnly(False)
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _exit_edit_mode(self, save: bool = True) -> None:
        """Exit inline rename mode. If save=False, restore the original name."""
        if self._name_edit.isReadOnly():
            return
        self._name_edit.setReadOnly(True)
        if save:
            new_name = self._name_edit.text().strip()
            if new_name:
                self._controller.set_layer_name(self.layer_id, new_name)
            else:
                # Restore previous name from model if blank
                layer = self._controller.get_layer(self.layer_id)
                if layer:
                    self._name_edit.setText(layer.name)
        else:
            self._name_edit.setText(self._edit_original_name)

    def _on_name_edited(self) -> None:
        self._exit_edit_mode(save=True)

    def update_from_layer(self, name: str, color: str, visible: bool, locked: bool, path_count: int, opacity: float = 1.0) -> None:
        self._name_edit.setText(name)
        self._color = color
        self._color_btn.setIcon(_color_swatch_icon(color))
        self._visible = visible
        self._vis_btn.setText("👁" if visible else "◻")
        self._locked = locked
        self._lock_btn.setText("🔒" if locked else "🔓")
        self._count_label.setText(f"[{path_count}]")
        slider_val = round(opacity * 100)
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(slider_val)
        self._opacity_slider.blockSignals(False)
        self._opacity_label.setText(f"{slider_val}%")

    def set_selected(self, active: bool, in_selection: bool = False) -> None:
        """Adjust text colors and background for three visual states.

        active: this is the "current" layer — full selection highlight.
        in_selection: part of a multi-selection group but not the active layer —
            subtle blue tint to indicate membership without full highlight.
        Neither: normal appearance.

        Qt's stylesheet cascade can force white text onto child widgets inside a
        selected QListWidgetItem, making text unreadable against light highlight
        colours.  We override this by reading the correct foreground colour
        directly from the palette and applying it via an explicit stylesheet rule.
        """
        pal = self.palette()
        if active:
            text_color = pal.color(QPalette.ColorRole.HighlightedText).name()
            count_color = text_color
            bg = ""
        elif in_selection:
            text_color = pal.color(QPalette.ColorRole.WindowText).name()
            count_color = pal.color(QPalette.ColorRole.PlaceholderText).name()
            bg = "background: rgba(100, 160, 220, 0.25);"
        else:
            text_color = pal.color(QPalette.ColorRole.WindowText).name()
            count_color = pal.color(QPalette.ColorRole.PlaceholderText).name()
            bg = ""

        self._name_edit.setStyleSheet(
            f"QLineEdit {{ border: none; background: transparent; color: {text_color}; }}"
            "QLineEdit:focus { border: 1px solid palette(highlight); background: palette(base); color: palette(text); }"
        )
        self._count_label.setStyleSheet(f"color: {count_color}; font-size: 10px;")
        self._opacity_label.setStyleSheet(f"color: {count_color}; font-size: 10px;")

        # Apply subtle tint for in_selection state
        if in_selection and not active:
            self.setAutoFillBackground(True)
            p = self.palette()
            p.setColor(QPalette.ColorRole.Window, QColor(100, 160, 220, 60))
            self.setPalette(p)
        else:
            self.setAutoFillBackground(False)
            self.setPalette(self.style().standardPalette())


class LayerPanel(QWidget):
    """Panel with a list of layers and management buttons."""

    def __init__(self, controller: ProjectController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._setup_ui()
        self._connect_signals()
        self._rebuild_list()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._list = QListWidget()
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)

        self._add_btn = QPushButton("+")
        self._add_btn.setToolTip("Add Layer")
        self._add_btn.setFixedWidth(28)
        self._add_btn.clicked.connect(self._on_add)

        self._del_btn = QPushButton("−")
        self._del_btn.setToolTip("Delete Layer")
        self._del_btn.setFixedWidth(28)
        self._del_btn.clicked.connect(self._on_delete)

        self._dup_btn = QPushButton("⧉")
        self._dup_btn.setToolTip("Duplicate Layer")
        self._dup_btn.setFixedWidth(28)
        self._dup_btn.clicked.connect(self._on_duplicate)

        self._merge_btn = QPushButton("⊕")
        self._merge_btn.setToolTip("Merge Selected Layers (Ctrl+Click to select multiple)")
        self._merge_btn.setFixedWidth(28)
        self._merge_btn.clicked.connect(self._on_merge)

        self._up_btn = QPushButton("▲")
        self._up_btn.setToolTip("Move Up")
        self._up_btn.setFixedWidth(28)
        self._up_btn.clicked.connect(self._on_move_up)

        self._down_btn = QPushButton("▼")
        self._down_btn.setToolTip("Move Down")
        self._down_btn.setFixedWidth(28)
        self._down_btn.clicked.connect(self._on_move_down)

        for btn in [self._add_btn, self._del_btn, self._dup_btn, self._merge_btn, self._up_btn, self._down_btn]:
            btn_layout.addWidget(btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        c = self._controller
        c.project_loaded.connect(self._rebuild_list)
        c.layer_added.connect(self._rebuild_list)
        c.layer_removed.connect(self._rebuild_list)
        c.layers_reordered.connect(self._rebuild_list)
        c.layer_changed.connect(self._on_layer_changed)
        c.paths_changed.connect(self._on_paths_changed)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        self._list.itemSelectionChanged.connect(self._update_selection_visuals)

    def _rebuild_list(self, *_args) -> None:  # type: ignore[no-untyped-def]
        self._list.clear()
        for layer in self._controller.current_project.layers:
            item = QListWidgetItem(self._list)
            widget = _LayerItem(
                layer.id, layer.name, layer.color, layer.visible,
                layer.locked, layer.path_count(), self._controller,
                opacity=layer.opacity,
            )
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, layer.id)
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)

        # Restore active layer selection after rebuilding the list
        active_id = self._controller.active_layer_id
        restored = False
        if active_id:
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == active_id:
                    self._list.setCurrentRow(i)
                    widget = self._list.itemWidget(item)
                    if isinstance(widget, _LayerItem):
                        widget.set_selected(True)
                    restored = True
                    break
        if not restored and self._list.count() > 0:
            self._list.setCurrentRow(0)
            item = self._list.item(0)
            if item:
                widget = self._list.itemWidget(item)
                if isinstance(widget, _LayerItem):
                    widget.set_selected(True)
        self._update_selection_visuals()

    def _on_layer_changed(self, layer_id: str) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == layer_id:
                widget = self._list.itemWidget(item)
                layer = self._controller.get_layer(layer_id)
                if layer and isinstance(widget, _LayerItem):
                    widget.update_from_layer(
                        layer.name, layer.color, layer.visible, layer.locked, layer.path_count(), layer.opacity
                    )
                break

    def _on_paths_changed(self, layer_id: str) -> None:
        self._on_layer_changed(layer_id)

    def _on_current_item_changed(self, current, _previous) -> None:  # type: ignore[no-untyped-def]
        """Notify the controller when the active (most recently clicked) layer changes."""
        if current is not None:
            layer_id = current.data(Qt.ItemDataRole.UserRole)
            if layer_id:
                self._controller.set_active_layer(layer_id)
        self._update_selection_visuals()

    def _update_selection_visuals(self) -> None:
        """Refresh the visual state of every item based on current/selected state."""
        current = self._list.currentItem()
        selected_items = self._list.selectedItems()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            widget = self._list.itemWidget(item)
            if isinstance(widget, _LayerItem):
                is_active = item is current
                is_in_sel = any(item is s for s in selected_items) and not is_active
                widget.set_selected(is_active, in_selection=is_in_sel)

    def _current_layer_id(self) -> str | None:
        item = self._list.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    def _selected_layer_ids(self) -> list[str]:
        ids = []
        for item in self._list.selectedItems():
            lid = item.data(Qt.ItemDataRole.UserRole)
            if lid:
                ids.append(lid)
        return ids

    def _on_add(self) -> None:
        self._controller.add_layer()

    def _on_delete(self) -> None:
        layer_id = self._current_layer_id()
        if layer_id:
            if len(self._controller.current_project.layers) <= 1:
                QMessageBox.warning(self, "Cannot Delete", "A project must have at least one layer.")
                return
            self._controller.remove_layer(layer_id)

    def _on_duplicate(self) -> None:
        layer_id = self._current_layer_id()
        if layer_id:
            self._controller.duplicate_layer(layer_id)

    def _on_merge(self) -> None:
        ids = self._selected_layer_ids()
        if len(ids) < 2:
            QMessageBox.information(self, "Merge Layers", "Select at least 2 layers to merge (Ctrl+Click to select multiple).")
            return
        merged = self._controller.merge_layers(ids)
        # Select the resulting merged layer as active
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == merged.id:
                self._list.clearSelection()
                self._list.setCurrentRow(i)
                item.setSelected(True)
                break

    def _on_move_up(self) -> None:
        row = self._list.currentRow()
        if row > 0:
            layer_id = self._current_layer_id()
            if layer_id:
                self._controller.reorder_layer(layer_id, row - 1)
                self._list.setCurrentRow(row - 1)

    def _on_move_down(self) -> None:
        row = self._list.currentRow()
        if row < self._list.count() - 1:
            layer_id = self._current_layer_id()
            if layer_id:
                self._controller.reorder_layer(layer_id, row + 1)
                self._list.setCurrentRow(row + 1)

    def _on_rows_moved(self, _parent, _start, _end, _dest, _dest_row) -> None:  # type: ignore[no-untyped-def]
        """Sync drag-drop reorder to the model."""
        new_order: list[str] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item:
                new_order.append(item.data(Qt.ItemDataRole.UserRole))
        # Reorder model to match the new_order
        for new_idx, lid in enumerate(new_order):
            current_idx = next(
                (j for j, l in enumerate(self._controller.current_project.layers) if l.id == lid),
                None,
            )
            if current_idx is not None and current_idx != new_idx:
                self._controller.reorder_layer(lid, new_idx)

    def _show_context_menu(self, pos) -> None:  # type: ignore[no-untyped-def]
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        color_action = menu.addAction("Change Color")
        menu.addSeparator()
        export_action = menu.addAction("Export Layer")
        action = menu.exec(self._list.mapToGlobal(pos))
        if action == rename_action:
            self._rename_layer(layer_id)
        elif action == color_action:
            self._change_layer_color(layer_id)
        elif action == export_action:
            self._export_layer(layer_id)

    def _rename_layer(self, layer_id: str) -> None:
        layer = self._controller.get_layer(layer_id)
        if layer is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Layer", "Layer name:", text=layer.name)
        if ok and new_name.strip():
            self._controller.set_layer_name(layer_id, new_name.strip())

    def _change_layer_color(self, layer_id: str) -> None:
        layer = self._controller.get_layer(layer_id)
        if layer is None:
            return
        new_color = QColorDialog.getColor(QColor(layer.color), self, "Layer Color")
        if new_color.isValid():
            self._controller.set_layer_color(layer_id, new_color.name())

    def _export_layer(self, layer_id: str) -> None:
        layer = self._controller.get_layer(layer_id)
        if layer is None:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        settings = dlg.get_settings()
        path = settings.get("output_path", "")
        if not path:
            QMessageBox.warning(self, "Export", "Please specify an output path.")
            return
        try:
            canvas = self._controller.current_project.canvas
            fmt = settings.get("format", "SVG")
            if fmt == "SVG":
                from plottter.export.svg import export_layer_svg
                export_layer_svg(layer, canvas, path, settings)
            elif fmt == "HPGL":
                from plottter.export.hpgl import export_layer_hpgl
                export_layer_hpgl(layer, canvas, path, settings)
            elif fmt == "G-code":
                from plottter.export.gcode import export_layer_gcode
                export_layer_gcode(layer, canvas, path, settings)
            else:
                QMessageBox.warning(self, "Export", f"Unknown format: {fmt}")
                return
            QMessageBox.information(self, "Export Layer", f"Exported '{layer.name}' to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
