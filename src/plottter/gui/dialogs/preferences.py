"""Preferences dialog — application-wide settings.

Currently contains:
- AI Integration: Replicate.com API key entry and connection test.
- AI Results Cache: unified cache directory for depth maps, background removal, and masks.
"""

from __future__ import annotations

import glob
import os
import pathlib

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_DEFAULT_CACHE_DIR = str(pathlib.Path.home() / ".plottter" / "ai_cache")
_CACHE_SUBDIRS = ("depth", "bg_removal", "masks")


def _cache_dir_from_settings(settings: QSettings) -> str:
    """Return the configured cache dir, migrating the old key if needed."""
    # New key takes priority
    value = settings.value("ai/cache_dir", "") or ""
    if value:
        return value
    # Fall back to old key (migration)
    old = settings.value("ai/depth_cache_dir", "") or ""
    if old:
        # Migrate: write under the new key and remove the old one
        settings.setValue("ai/cache_dir", old)
        settings.remove("ai/depth_cache_dir")
        return old
    return ""


def _compute_cache_size(cache_dir: str) -> int:
    """Return total bytes of all PNG files under cache_dir (all subdirs)."""
    if not os.path.isdir(cache_dir):
        return 0
    total = 0
    for subdir in _CACHE_SUBDIRS:
        subpath = os.path.join(cache_dir, subdir)
        if os.path.isdir(subpath):
            for f in glob.glob(os.path.join(subpath, "*.png")):
                try:
                    total += os.path.getsize(f)
                except OSError:
                    pass
    # Also count PNGs directly in the root (legacy depth_cache layout)
    for f in glob.glob(os.path.join(cache_dir, "*.png")):
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    return total


def _format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    if bytes_ < 1024 ** 2:
        return f"{bytes_ / 1024:.1f} KB"
    if bytes_ < 1024 ** 3:
        return f"{bytes_ / 1024 ** 2:.1f} MB"
    return f"{bytes_ / 1024 ** 3:.2f} GB"


class PreferencesDialog(QDialog):
    """Modal preferences dialog with AI Integration settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_ai_group())
        layout.addWidget(self._build_cache_group())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_settings()
        self._update_cache_size_label()

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _build_ai_group(self) -> QGroupBox:
        group = QGroupBox("AI Integration")
        form = QFormLayout(group)

        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("r8_…")
        self._api_key_edit.setToolTip(
            "Your Replicate.com API key.  "
            "Get one at https://replicate.com/account/api-tokens"
        )
        form.addRow("API Key:", self._api_key_edit)

        test_row = QWidget()
        test_layout = QHBoxLayout(test_row)
        test_layout.setContentsMargins(0, 0, 0, 0)

        self._test_btn = QPushButton("Test Connection")
        self._test_btn.setToolTip("Verify that the API key is valid")
        self._test_btn.clicked.connect(self._on_test_connection)
        test_layout.addWidget(self._test_btn)
        test_layout.addStretch()

        form.addRow("", test_row)

        note = QLabel(
            "Leave blank to disable AI features.  "
            "Get a free API key at <a href=\"https://replicate.com\">replicate.com</a>."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setOpenExternalLinks(True)
        form.addRow(note)

        return group

    def _build_cache_group(self) -> QGroupBox:
        group = QGroupBox("AI Results Cache")
        form = QFormLayout(group)

        # Path picker row
        cache_row = QWidget()
        cache_layout = QHBoxLayout(cache_row)
        cache_layout.setContentsMargins(0, 0, 0, 0)

        self._cache_dir_edit = QLineEdit()
        self._cache_dir_edit.setPlaceholderText(_DEFAULT_CACHE_DIR)
        self._cache_dir_edit.setToolTip(
            "Directory where AI results are cached as PNG files.\n"
            "Subdirectories: depth/, bg_removal/, masks/\n"
            "Leave blank to use the default location."
        )
        cache_layout.addWidget(self._cache_dir_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(self._on_browse_cache_dir)
        cache_layout.addWidget(browse_btn)

        form.addRow("Cache Directory:", cache_row)

        # Cache size label
        self._cache_size_label = QLabel("Cache size: —")
        form.addRow("", self._cache_size_label)

        # Clear cache button
        clear_row = QWidget()
        clear_layout = QHBoxLayout(clear_row)
        clear_layout.setContentsMargins(0, 0, 0, 0)

        self._clear_cache_btn = QPushButton("Clear Cache")
        self._clear_cache_btn.setToolTip(
            "Delete all cached AI result PNG files from all subdirectories "
            "(depth/, bg_removal/, masks/)."
        )
        self._clear_cache_btn.clicked.connect(self._on_clear_cache)
        clear_layout.addWidget(self._clear_cache_btn)
        clear_layout.addStretch()

        form.addRow("", clear_row)

        note = QLabel(
            "AI results (depth maps, background removal, segmentation masks) are "
            "cached to avoid repeated Replicate API calls when only rendering "
            "parameters change.  Cache entries are keyed by source image content "
            "(SHA-256)."
        )
        note.setWordWrap(True)
        form.addRow(note)

        return group

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        key = settings.value("replicate/api_key", "") or ""
        self._api_key_edit.setText(key)
        cache_dir = _cache_dir_from_settings(settings)
        self._cache_dir_edit.setText(cache_dir)

    def _save_settings(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("replicate/api_key", self._api_key_edit.text().strip())
        settings.setValue("ai/cache_dir", self._cache_dir_edit.text().strip())
        # Remove old key if present
        settings.remove("ai/depth_cache_dir")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolved_cache_dir(self) -> str:
        d = self._cache_dir_edit.text().strip()
        return d if d else _DEFAULT_CACHE_DIR

    def _update_cache_size_label(self) -> None:
        cache_dir = self._resolved_cache_dir()
        size = _compute_cache_size(cache_dir)
        self._cache_size_label.setText(f"Cache size: {_format_size(size)}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        self._save_settings()
        self.accept()

    def _on_browse_cache_dir(self) -> None:
        """Open a directory picker for the AI cache directory."""
        current = self._cache_dir_edit.text().strip() or _DEFAULT_CACHE_DIR
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select AI Cache Directory",
            current,
        )
        if chosen:
            self._cache_dir_edit.setText(chosen)
            self._update_cache_size_label()

    def _on_clear_cache(self) -> None:
        """Delete all .png files from all subdirectories of the cache directory."""
        cache_dir = self._resolved_cache_dir()

        if not os.path.isdir(cache_dir):
            QMessageBox.information(
                self,
                "Clear Cache",
                f"Cache directory does not exist:\n{cache_dir}",
            )
            return

        # Gather files per category
        categories: dict[str, list[str]] = {}
        for subdir in _CACHE_SUBDIRS:
            subpath = os.path.join(cache_dir, subdir)
            if os.path.isdir(subpath):
                files = glob.glob(os.path.join(subpath, "*.png"))
                if files:
                    categories[subdir] = files
        # Also root-level PNGs (legacy)
        root_pngs = glob.glob(os.path.join(cache_dir, "*.png"))
        if root_pngs:
            categories["(root)"] = root_pngs

        total_files = sum(len(v) for v in categories.values())
        if total_files == 0:
            QMessageBox.information(
                self,
                "Clear Cache",
                "No cached AI result files found.",
            )
            return

        summary_lines = "\n".join(
            f"  {name}/: {len(files)} file(s)"
            for name, files in sorted(categories.items())
        )
        reply = QMessageBox.question(
            self,
            "Clear Cache",
            f"Delete {total_files} cached file(s) from:\n{cache_dir}\n\n{summary_lines}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed_by_cat: dict[str, int] = {}
        for name, files in categories.items():
            count = 0
            for f in files:
                try:
                    os.remove(f)
                    count += 1
                except OSError:
                    pass
            removed_by_cat[name] = count

        result_lines = "\n".join(
            f"  {name}/: {count} removed"
            for name, count in sorted(removed_by_cat.items())
        )
        total_removed = sum(removed_by_cat.values())
        QMessageBox.information(
            self,
            "Clear Cache",
            f"Removed {total_removed} cached file(s):\n\n{result_lines}",
        )
        self._update_cache_size_label()

    def _on_test_connection(self) -> None:
        """Test the Replicate API key by calling is_available() and a lightweight check."""
        key = self._api_key_edit.text().strip()
        if not key:
            QMessageBox.warning(
                self,
                "Test Connection",
                "Please enter an API key before testing.",
            )
            return

        # Attempt a lightweight REST API call to verify the key
        import urllib.request
        import urllib.error
        import json

        try:
            req = urllib.request.Request(
                "https://api.replicate.com/v1/account",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
            username = data.get("username", "")
            status_msg = f"Connection successful! Signed in as: {username}" if username else "Connection successful! API key is valid."
            success = True
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            if exc.code == 401:
                status_msg = "Authentication failed: invalid API key."
            elif "rate limit" in body_text.lower():
                status_msg = f"Rate limited — key may be valid, but you are being throttled.\n{body_text}"
            else:
                status_msg = f"Connection error (HTTP {exc.code}): {body_text}"
            success = False
        except Exception as exc:
            status_msg = f"Connection error: {exc}"
            success = False

        if success:
            # Also update the status bar of the parent window if possible
            parent = self.parent()
            if hasattr(parent, "statusBar"):
                parent.statusBar().showMessage("Replicate API: connected", 5000)
            QMessageBox.information(self, "Test Connection", status_msg)
        else:
            QMessageBox.warning(self, "Test Connection", status_msg)
