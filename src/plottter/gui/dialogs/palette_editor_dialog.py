"""Standalone palette editor dialog for creating and editing user PenPalettes."""
from __future__ import annotations

import re
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from plottter.color.palette import PenPalette, save_user_palette

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

_SWATCH_SIZE = 16  # px


def _make_swatch_icon(hex_color: str) -> QIcon:
    """Return a small filled square QIcon for the given #RRGGBB hex colour."""
    pixmap = QPixmap(_SWATCH_SIZE, _SWATCH_SIZE)
    pixmap.fill(QColor(hex_color))
    return QIcon(pixmap)


class PaletteEditorDialog(QDialog):
    """Modal dialog for creating or editing a user PenPalette.

    Parameters
    ----------
    parent:
        Optional parent widget.
    initial:
        If provided, the dialog opens pre-filled with that palette's data.
        Otherwise it opens blank (create mode).
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        initial: Optional[PenPalette] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Palette Editor")
        self.setMinimumWidth(400)
        self._result_palette: Optional[PenPalette] = None
        self._setup_ui()

        if initial is not None:
            self._name_edit.setText(initial.name)
            self._desc_edit.setText(initial.description)
            for color in initial.colors:
                self._append_color_row(color)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Name / Description fields ---
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. My Watercolours")
        form.addRow("Name:", self._name_edit)

        self._desc_edit = QLineEdit()
        self._desc_edit.setPlaceholderText("Optional one-line description")
        form.addRow("Description:", self._desc_edit)

        layout.addLayout(form)

        # --- Colour list + side buttons ---
        list_row = QHBoxLayout()
        list_row.setSpacing(6)

        self._color_list = QListWidget()
        self._color_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        list_row.addWidget(self._color_list, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)
        btn_col.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._add_btn = QPushButton("Add")
        self._remove_btn = QPushButton("Remove")
        self._edit_btn = QPushButton("Edit…")
        self._up_btn = QPushButton("Move Up")
        self._down_btn = QPushButton("Move Down")

        for btn in (self._add_btn, self._remove_btn, self._edit_btn, self._up_btn, self._down_btn):
            btn_col.addWidget(btn)

        list_row.addLayout(btn_col)
        layout.addLayout(list_row)

        # --- Save / Cancel ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # --- Wire signals ---
        self._add_btn.clicked.connect(self._on_add)
        self._remove_btn.clicked.connect(self._on_remove)
        self._edit_btn.clicked.connect(self._on_edit)
        self._up_btn.clicked.connect(self._on_move_up)
        self._down_btn.clicked.connect(self._on_move_down)
        self._color_list.currentRowChanged.connect(self._update_button_states)

        self._update_button_states()

    # ------------------------------------------------------------------
    # List helpers
    # ------------------------------------------------------------------

    def _append_color_row(self, hex_color: str) -> None:
        """Add a new row for *hex_color* at the end of the list."""
        item = QListWidgetItem()
        item.setIcon(_make_swatch_icon(hex_color))
        item.setText(hex_color.upper())
        item.setData(Qt.ItemDataRole.UserRole, hex_color.upper())
        self._color_list.addItem(item)

    def _update_row(self, row: int, hex_color: str) -> None:
        """Update an existing list row in place."""
        item = self._color_list.item(row)
        if item is None:
            return
        item.setIcon(_make_swatch_icon(hex_color))
        item.setText(hex_color.upper())
        item.setData(Qt.ItemDataRole.UserRole, hex_color.upper())

    def _update_button_states(self) -> None:
        """Enable / disable side buttons depending on selection and list size."""
        row = self._color_list.currentRow()
        count = self._color_list.count()
        has_sel = row >= 0
        self._remove_btn.setEnabled(has_sel)
        self._edit_btn.setEnabled(has_sel)
        self._up_btn.setEnabled(has_sel and row > 0)
        self._down_btn.setEnabled(has_sel and row < count - 1)

    def _colors(self) -> list[str]:
        """Return current hex colours from the list, in order."""
        return [
            self._color_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._color_list.count())
        ]

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        """Open QColorDialog; append the chosen colour."""
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            hex_color = color.name().upper()
            self._append_color_row(hex_color)
            self._color_list.setCurrentRow(self._color_list.count() - 1)
            self._update_button_states()

    def _on_remove(self) -> None:
        row = self._color_list.currentRow()
        if row < 0:
            return
        self._color_list.takeItem(row)
        # Select the nearest remaining row
        new_count = self._color_list.count()
        if new_count > 0:
            self._color_list.setCurrentRow(min(row, new_count - 1))
        self._update_button_states()

    def _on_edit(self) -> None:
        row = self._color_list.currentRow()
        if row < 0:
            return
        current_hex = self._color_list.item(row).data(Qt.ItemDataRole.UserRole)
        initial_color = QColor(current_hex)
        color = QColorDialog.getColor(initial_color, parent=self)
        if color.isValid():
            self._update_row(row, color.name().upper())

    def _on_move_up(self) -> None:
        row = self._color_list.currentRow()
        if row <= 0:
            return
        item = self._color_list.takeItem(row)
        self._color_list.insertItem(row - 1, item)
        self._color_list.setCurrentRow(row - 1)
        self._update_button_states()

    def _on_move_down(self) -> None:
        row = self._color_list.currentRow()
        if row < 0 or row >= self._color_list.count() - 1:
            return
        item = self._color_list.takeItem(row)
        self._color_list.insertItem(row + 1, item)
        self._color_list.setCurrentRow(row + 1)
        self._update_button_states()

    # ------------------------------------------------------------------
    # Save / validation
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """Validate fields and, if OK, persist the palette then accept."""
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Name must not be empty.")
            self._name_edit.setFocus()
            return

        colors = self._colors()
        if not colors:
            QMessageBox.warning(
                self, "Validation Error", "Add at least one colour before saving."
            )
            return

        invalid = [c for c in colors if not _HEX_RE.match(c)]
        if invalid:
            QMessageBox.warning(
                self,
                "Validation Error",
                f"Invalid hex colour(s): {', '.join(invalid)}",
            )
            return

        palette = PenPalette(
            name=name,
            colors=tuple(colors),
            description=self._desc_edit.text().strip(),
        )
        save_user_palette(palette)
        self._result_palette = palette
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_result(self) -> Optional[PenPalette]:
        """Return the saved PenPalette, or None if the dialog was cancelled."""
        return self._result_palette
