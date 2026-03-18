"""Google Fonts browser dialog for Plottter.

Lets users search, preview, and download fonts from the Google Fonts catalog.
Downloaded fonts are cached to ``~/.plottter/fonts/google/`` and
automatically appear alongside system fonts in the FontPicker widget.

Usage (from FontPicker or anywhere else)::

    dialog = GoogleFontsDialog(parent)
    if dialog.exec():
        path = dialog.selected_font_path()
        if path:
            print("Downloaded:", path)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QSortFilterProxyModel,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QFont, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
)

# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

_GOOGLE_FONT_CACHE_DIR = Path.home() / ".plottter" / "fonts" / "google"


class _DownloadWorker(QThread):
    """Background thread that downloads a single Google Font.

    Signals
    -------
    progress(int, int)
        Emitted during download with (bytes_downloaded, total_bytes).
        total_bytes may be 0 when the server doesn't send Content-Length.
    finished_ok(str)
        Emitted on success with the absolute path to the cached file.
    finished_err(str)
        Emitted on failure with a human-readable error message.
    """

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(
        self,
        family: str,
        style: str = "regular",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._family = family
        self._style = style

    def run(self) -> None:
        try:
            from plottter.fonts.google_fonts import download_google_font

            path = download_google_font(
                self._family,
                self._style,
                progress_callback=lambda done, total: self.progress.emit(done, total),
            )
            self.finished_ok.emit(path)
        except Exception as exc:  # noqa: BLE001
            self.finished_err.emit(str(exc))


# ---------------------------------------------------------------------------
# List model
# ---------------------------------------------------------------------------

# Columns in the internal model (all visible in QTableView)
_COL_NAME = 0
_COL_CATEGORY = 1
_COL_DOWNLOADED = 2  # ✓ badge column

# Custom role for font family name (so we can get it back from selected index)
_ROLE_FAMILY = Qt.ItemDataRole.UserRole + 1


def _is_font_cached(family: str) -> bool:
    """Return True if the regular-weight TTF is already in the local cache."""
    safe_family = family.replace(" ", "-")
    return (_GOOGLE_FONT_CACHE_DIR / f"{safe_family}-regular.ttf").exists()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class GoogleFontsDialog(QDialog):
    """Searchable, downloadable Google Fonts browser.

    Opens the bundled Google Fonts catalog, lets the user search by name
    and filter by category, shows a checkmark on already-cached fonts, and
    allows downloading the selected font in the background.

    After :meth:`exec` returns ``QDialog.Accepted``, call
    :meth:`selected_font_path` to get the path to the downloaded font file.

    Parameters
    ----------
    parent:
        Parent widget.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Browse Google Fonts")
        self.setMinimumSize(520, 540)
        self.resize(620, 600)

        self._selected_path: str = ""
        self._download_worker: Optional[_DownloadWorker] = None

        self._build_ui()
        self._load_catalog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def selected_font_path(self) -> str:
        """Return the absolute path to the downloaded font file, or ""."""
        return self._selected_path

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ---- Search + filter row ----
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search Google Fonts…")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._on_search_changed)

        self._category_combo = QComboBox()
        self._category_combo.addItems(
            ["All categories", "Serif", "Sans-Serif", "Display", "Handwriting", "Monospace"]
        )
        self._category_combo.setMinimumWidth(130)
        self._category_combo.currentIndexChanged.connect(self._apply_filter)

        filter_row.addWidget(self._search_edit, stretch=1)
        filter_row.addWidget(self._category_combo, stretch=0)

        # ---- Font list ----
        self._model = QStandardItemModel(0, 3)
        self._model.setHorizontalHeaderLabels(["Family", "Category", "✓"])

        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._proxy.setFilterKeyColumn(0)  # filter on family name column

        self._list_view = QTableView()
        self._list_view.setModel(self._proxy)
        self._list_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._list_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._list_view.verticalHeader().setVisible(False)
        self._list_view.setShowGrid(False)
        self._list_view.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QHeaderView.ResizeMode.Stretch
        )
        self._list_view.horizontalHeader().setSectionResizeMode(
            _COL_CATEGORY, QHeaderView.ResizeMode.Fixed
        )
        self._list_view.setColumnWidth(_COL_CATEGORY, 110)
        self._list_view.horizontalHeader().setSectionResizeMode(
            _COL_DOWNLOADED, QHeaderView.ResizeMode.Fixed
        )
        self._list_view.setColumnWidth(_COL_DOWNLOADED, 28)
        self._list_view.selectionModel().currentChanged.connect(self._on_selection_changed)
        self._list_view.doubleClicked.connect(self._on_double_click)

        # ---- Info row ----
        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self._open_folder_btn = QPushButton("Open Fonts Folder")
        self._open_folder_btn.setToolTip(
            "Open ~/.plottter/fonts/google/ in the system file manager"
        )
        self._open_folder_btn.clicked.connect(self._on_open_folder)

        info_row.addWidget(self._info_label, stretch=1)
        info_row.addWidget(self._open_folder_btn, stretch=0)

        # ---- Progress bar (hidden until download starts) ----
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(14)

        # ---- Download button ----
        self._download_btn = QPushButton("Download Selected Font")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._on_download)

        # ---- Dialog buttons ----
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        self._ok_btn = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(False)

        # ---- Assemble layout ----
        root.addLayout(filter_row)
        root.addWidget(self._list_view, stretch=1)
        root.addLayout(info_row)
        root.addWidget(self._progress_bar)
        root.addWidget(self._download_btn)
        root.addWidget(self._button_box)

        # Debounce timer for search input
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._apply_filter)

    # ------------------------------------------------------------------
    # Catalog loading
    # ------------------------------------------------------------------

    def _load_catalog(self) -> None:
        """Populate the model from the bundled Google Fonts catalog."""
        try:
            from plottter.fonts.google_fonts import get_google_fonts_catalog
        except ImportError:
            self._info_label.setText(
                "Google Fonts catalog is not available (import error)."
            )
            return

        catalog = get_google_fonts_catalog()
        self._model.removeRows(0, self._model.rowCount())

        for font in catalog:
            cached = _is_font_cached(font.family)
            badge = "✓" if cached else ""

            name_item = QStandardItem(font.family)
            name_item.setData(font.family, _ROLE_FAMILY)
            name_item.setEditable(False)
            # Show a checkmark decoration when cached
            if cached:
                name_item.setToolTip(f"{font.family} — already downloaded")
            else:
                name_item.setToolTip(f"{font.family} ({font.category})")

            cat_item = QStandardItem(font.category.title())
            cat_item.setEditable(False)

            badge_item = QStandardItem(badge)
            badge_item.setEditable(False)

            self._model.appendRow([name_item, cat_item, badge_item])

        self._proxy.invalidate()
        self._info_label.setText(
            f"{self._proxy.rowCount()} fonts  "
            f"(✓ = already downloaded)"
        )

    # ------------------------------------------------------------------
    # Search / filter
    # ------------------------------------------------------------------

    def _on_search_changed(self, _text: str) -> None:
        """Restart the debounce timer on each keystroke."""
        self._search_timer.start()

    def _apply_filter(self) -> None:
        """Apply the current search text and category filter to the proxy."""
        query = self._search_edit.text().strip()
        category_text = self._category_combo.currentText()

        if category_text == "All categories":
            category_filter = ""
        else:
            category_filter = category_text.lower()

        if category_filter:
            # Multi-column filter: need custom filtering
            # We'll use a simple approach: filter by name via proxy, then hide
            # rows that don't match the category.
            # A cleaner alternative is to subclass QSortFilterProxyModel.
            self._proxy.setFilterKeyColumn(0)
            self._proxy.setFilterFixedString(query)
            # Now hide rows whose category doesn't match
            for row in range(self._proxy.rowCount()):
                proxy_idx = self._proxy.index(row, _COL_CATEGORY)
                source_idx = self._proxy.mapToSource(proxy_idx)
                cat = (
                    self._model.item(source_idx.row(), _COL_CATEGORY)
                    .text()
                    .lower()
                )
                is_visible = (cat == category_filter)
                self._list_view.setRowHidden(row, not is_visible)
        else:
            # Re-show all rows (clear any category hiding)
            for row in range(self._proxy.rowCount()):
                self._list_view.setRowHidden(row, False)
            self._proxy.setFilterKeyColumn(0)
            self._proxy.setFilterFixedString(query)

        visible = sum(
            1
            for row in range(self._proxy.rowCount())
            if not self._list_view.isRowHidden(row)
        )
        self._info_label.setText(
            f"{visible} fonts  (✓ = already downloaded)"
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _current_family(self) -> str | None:
        """Return the family name currently selected in the list, or None."""
        idx = self._list_view.currentIndex()
        if not idx.isValid():
            return None
        source_idx = self._proxy.mapToSource(idx)
        item = self._model.item(source_idx.row(), _COL_NAME)
        if item is None:
            return None
        return item.data(_ROLE_FAMILY)

    def _on_selection_changed(self, current, _previous) -> None:
        """Update UI state when the selection changes."""
        family = self._current_family()
        if not family:
            self._download_btn.setEnabled(False)
            self._ok_btn.setEnabled(False)
            return

        cached = _is_font_cached(family)
        self._download_btn.setEnabled(not cached)
        self._download_btn.setText(
            "Already Downloaded" if cached else "Download Selected Font"
        )
        # OK button is enabled only if the font is already cached
        self._ok_btn.setEnabled(cached)
        if cached:
            safe = family.replace(" ", "-")
            self._selected_path = str(
                _GOOGLE_FONT_CACHE_DIR / f"{safe}-regular.ttf"
            )
        else:
            self._selected_path = ""

    def _on_double_click(self, _index) -> None:
        """Double-clicking a font triggers download (or accept if cached)."""
        family = self._current_family()
        if not family:
            return
        if _is_font_cached(family):
            self._on_accept()
        else:
            self._on_download()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _on_download(self) -> None:
        """Start downloading the selected font in a background thread."""
        family = self._current_family()
        if not family:
            return

        if self._download_worker and self._download_worker.isRunning():
            return  # already busy

        self._download_btn.setEnabled(False)
        self._download_btn.setText("Downloading…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # indeterminate

        self._download_worker = _DownloadWorker(family, "regular")
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished_ok.connect(self._on_download_ok)
        self._download_worker.finished_err.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(done * 100 / total))
        else:
            self._progress_bar.setRange(0, 0)  # keep indeterminate

    def _on_download_ok(self, path: str) -> None:
        """Called on successful download."""
        self._progress_bar.setVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(100)

        self._selected_path = path

        # Update badge in the model
        family = self._current_family()
        if family:
            self._refresh_row_badge(family)

        self._download_btn.setText("Downloaded ✓")
        self._download_btn.setEnabled(False)
        self._ok_btn.setEnabled(True)

        self._info_label.setText(
            f"Downloaded: {Path(path).name}"
        )

    def _on_download_error(self, message: str) -> None:
        """Called when the download fails."""
        self._progress_bar.setVisible(False)
        self._download_btn.setText("Download Selected Font")
        self._download_btn.setEnabled(True)
        QMessageBox.warning(
            self,
            "Download Failed",
            f"Could not download the selected font:\n\n{message}\n\n"
            "Check your internet connection or try again later.",
        )

    def _refresh_row_badge(self, family: str) -> None:
        """Update the ✓ badge for *family* in the model after download."""
        for row in range(self._model.rowCount()):
            item = self._model.item(row, _COL_NAME)
            if item and item.data(_ROLE_FAMILY) == family:
                badge_item = self._model.item(row, _COL_DOWNLOADED)
                if badge_item:
                    badge_item.setText("✓")
                item.setToolTip(f"{family} — already downloaded")
                break

    # ------------------------------------------------------------------
    # Open folder
    # ------------------------------------------------------------------

    def _on_open_folder(self) -> None:
        """Open the Google Fonts cache directory in the system file manager."""
        folder = _GOOGLE_FONT_CACHE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        folder_str = str(folder)

        if sys.platform == "darwin":
            subprocess.Popen(["open", folder_str])
        elif sys.platform == "win32":
            os.startfile(folder_str)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", folder_str])

    # ------------------------------------------------------------------
    # Accept / reject
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        """Accept the dialog, returning the selected font path."""
        family = self._current_family()
        if family and _is_font_cached(family) and not self._selected_path:
            safe = family.replace(" ", "-")
            self._selected_path = str(
                _GOOGLE_FONT_CACHE_DIR / f"{safe}-regular.ttf"
            )
        self.accept()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Ensure background thread is stopped when the dialog closes."""
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.wait(2000)
        super().closeEvent(event)
