"""Font picker widget for Plottter.

A self-contained widget that lets users select a font by family and style,
with a live preview label and a button to open the Google Fonts browser.

The selected value exposed to the rest of the application is the absolute
path to the ``.ttf`` / ``.otf`` font file, identical to what the old
``system_font_path`` StringParam used to hold.

Usage::

    picker = FontPicker()
    picker.font_changed.connect(lambda path: print("Selected:", path))
    print(picker.font_path())          # current file path (str, may be "")
    picker.set_font_path("/usr/share/fonts/FreeSans.ttf")
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FontPicker(QWidget):
    """Compact font-selection widget.

    Displays a family combo box (filterable by typing), a style combo box,
    and a small preview label.  Optionally opens the Google Fonts browser
    dialog when the "Browse Google Fonts…" button is clicked.

    Signals
    -------
    font_changed(file_path: str)
        Emitted whenever the resolved font file path changes.  ``file_path``
        is an absolute path or an empty string when no font is selected.
    """

    font_changed = pyqtSignal(str)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._font_path: str = ""
        self._font_info_by_family: dict[str, list] = {}  # family -> [FontInfo]
        self._populated: bool = False

        self._build_ui()
        # Defer the (potentially slow) font scan to the first time the
        # widget is actually shown so that creating the widget is cheap.
        self._populate_fonts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def font_path(self) -> str:
        """Return the absolute path to the selected font file (may be "")."""
        return self._font_path

    def set_font_path(self, path: str) -> None:
        """Select the font that corresponds to *path*.

        If *path* matches a known font in the catalog the family and style
        dropdowns are updated accordingly.  Otherwise the path is stored as-is
        (e.g. a manually typed path that hasn't been catalogued yet) and the
        family combo is left unchanged.

        Emits :attr:`font_changed` if the path changes.
        """
        if path == self._font_path:
            return
        self._font_path = path
        self._sync_combos_from_path(path)
        self._update_preview()
        self.font_changed.emit(path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # --- Row 1: family combo + browse button ---
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(4)

        self._family_combo = QComboBox()
        self._family_combo.setEditable(True)
        self._family_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._family_combo.setMinimumWidth(160)
        self._family_combo.setToolTip(
            "Font family — type to filter the list"
        )
        self._family_combo.currentTextChanged.connect(self._on_family_changed)

        self._browse_btn = QPushButton("Browse Google Fonts…")
        self._browse_btn.setToolTip(
            "Search and download fonts from Google Fonts"
        )
        self._browse_btn.clicked.connect(self._on_browse_google_fonts)

        row1.addWidget(self._family_combo, stretch=1)
        row1.addWidget(self._browse_btn, stretch=0)

        # --- Row 2: style combo ---
        self._style_combo = QComboBox()
        self._style_combo.setToolTip("Font style (weight / variant)")
        self._style_combo.currentTextChanged.connect(self._on_style_changed)

        # --- Row 3: preview ---
        self._preview_label = QLabel("AaBbCc 123")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._preview_label.setMinimumHeight(28)
        self._preview_label.setToolTip("Preview of the selected font")

        layout.addLayout(row1)
        layout.addWidget(self._style_combo)
        layout.addWidget(self._preview_label)

    # ------------------------------------------------------------------
    # Font catalog population
    # ------------------------------------------------------------------

    def _populate_fonts(self) -> None:
        """Load the system font catalog and populate the family combo."""
        if self._populated:
            return
        self._populated = True

        try:
            from plottter.fonts.discovery import discover_system_fonts
        except ImportError:
            return

        fonts = discover_system_fonts()

        # Build family → list[FontInfo] map
        family_map: dict[str, list] = {}
        for info in fonts:
            family_map.setdefault(info.family, []).append(info)
        self._font_info_by_family = family_map

        families = sorted(family_map.keys(), key=str.lower)

        self._family_combo.blockSignals(True)
        self._family_combo.clear()
        self._family_combo.addItem("")  # empty / no-selection sentinel
        self._family_combo.addItems(families)

        # Add case-insensitive completer for keyboard-driven filtering
        completer = QCompleter([""] + families, self._family_combo)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._family_combo.setCompleter(completer)

        self._family_combo.blockSignals(False)

        # If we already have a path set before population completed,
        # try to reconcile it with the catalog now.
        if self._font_path:
            self._sync_combos_from_path(self._font_path)
        else:
            self._refresh_style_combo(None)

    # ------------------------------------------------------------------
    # Combo interaction handlers
    # ------------------------------------------------------------------

    def _on_family_changed(self, family: str) -> None:
        self._refresh_style_combo(family or None)
        self._resolve_path_from_combos()

    def _on_style_changed(self, _style: str) -> None:
        self._resolve_path_from_combos()

    def _refresh_style_combo(self, family: str | None) -> None:
        """Repopulate the style combo for *family* (or clear it if None)."""
        self._style_combo.blockSignals(True)
        self._style_combo.clear()

        if family and family in self._font_info_by_family:
            styles: list[str] = sorted(
                {info.style for info in self._font_info_by_family[family]},
                key=lambda s: (s != "Regular", s),
            )
            self._style_combo.addItems(styles)
            self._style_combo.setEnabled(len(styles) > 1)
        else:
            self._style_combo.setEnabled(False)

        self._style_combo.blockSignals(False)

    def _resolve_path_from_combos(self) -> None:
        """Update ``_font_path`` from the current combo selections and emit."""
        family = self._family_combo.currentText().strip()
        style = self._style_combo.currentText().strip() or "Regular"

        if not family:
            new_path = ""
        else:
            # Look up from the in-memory catalog so that mock-populated widgets
            # (e.g. in tests) work correctly without hitting the system cache.
            infos = self._font_info_by_family.get(family, [])
            match = next((i for i in infos if i.style == style), None)
            if match is None:
                match = next((i for i in infos if i.style == "Regular"), None)
            if match is None and infos:
                match = infos[0]
            new_path = match.file_path if match is not None else ""

        if new_path != self._font_path:
            self._font_path = new_path
            self.font_changed.emit(new_path)
        self._update_preview()

    def _sync_combos_from_path(self, path: str) -> None:
        """Select the family/style entries that correspond to *path*."""
        if not path:
            self._family_combo.blockSignals(True)
            self._family_combo.setCurrentIndex(0)
            self._family_combo.blockSignals(False)
            self._refresh_style_combo(None)
            return

        # Find which FontInfo has this file_path
        target_info = None
        for infos in self._font_info_by_family.values():
            for info in infos:
                if info.file_path == path:
                    target_info = info
                    break
            if target_info is not None:
                break

        if target_info is None:
            # Unknown path — show it as-is in the family combo
            self._family_combo.blockSignals(True)
            self._family_combo.setEditText(path)
            self._family_combo.blockSignals(False)
            self._refresh_style_combo(None)
            return

        # Select family
        idx = self._family_combo.findText(target_info.family)
        self._family_combo.blockSignals(True)
        if idx >= 0:
            self._family_combo.setCurrentIndex(idx)
        else:
            self._family_combo.setEditText(target_info.family)
        self._family_combo.blockSignals(False)

        # Refresh style options for this family, then select style
        self._refresh_style_combo(target_info.family)
        style_idx = self._style_combo.findText(target_info.style)
        self._style_combo.blockSignals(True)
        if style_idx >= 0:
            self._style_combo.setCurrentIndex(style_idx)
        self._style_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        """Render the preview label with the currently selected font."""
        if self._font_path:
            try:
                # Use QFont with the family name for the preview (fast, no TTF load)
                family = self._family_combo.currentText().strip()
                if family:
                    qfont = QFont(family, 14)
                    self._preview_label.setFont(qfont)
                    self._preview_label.setText("AaBbCc 123")
                    return
            except Exception:
                pass
        # Reset to default font with placeholder text
        self._preview_label.setFont(QFont())
        self._preview_label.setText(
            "AaBbCc 123" if not self._font_path else self._font_path
        )

    # ------------------------------------------------------------------
    # Google Fonts browser
    # ------------------------------------------------------------------

    def _on_browse_google_fonts(self) -> None:
        """Open the Google Fonts browser dialog (task 18.5)."""
        try:
            from plottter.gui.dialogs.google_fonts import GoogleFontsDialog
            dialog = GoogleFontsDialog(self)
            if dialog.exec():
                selected_path = dialog.selected_font_path()
                if selected_path:
                    # Refresh catalog so the new font appears in the combo
                    try:
                        from plottter.fonts.discovery import invalidate_font_cache
                        invalidate_font_cache()
                    except ImportError:
                        pass
                    self._populated = False
                    self._populate_fonts()
                    self.set_font_path(selected_path)
        except ImportError:
            # Google Fonts dialog not yet implemented (task 18.5)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Google Fonts Browser",
                "The Google Fonts browser will be available in a future update "
                "(task 18.5).\n\nYou can also download fonts manually via "
                "the terminal:\n  python -c \"from plottter.fonts import "
                "download_google_font; download_google_font('Roboto')\"",
            )
