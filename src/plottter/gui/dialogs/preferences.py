"""Preferences dialog — application-wide settings.

Currently contains:
- AI Integration: Replicate.com API key entry and connection test.
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
            "Install the optional dependency with "
            "<code>pip install plottter[ai]</code>."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(note)

        return group

    def _build_cache_group(self) -> QGroupBox:
        group = QGroupBox("Depth Map Cache")
        form = QFormLayout(group)

        # Path picker row
        cache_row = QWidget()
        cache_layout = QHBoxLayout(cache_row)
        cache_layout.setContentsMargins(0, 0, 0, 0)

        self._cache_dir_edit = QLineEdit()
        self._cache_dir_edit.setPlaceholderText(
            str(pathlib.Path.home() / ".plottter" / "depth_cache")
        )
        self._cache_dir_edit.setToolTip(
            "Directory where AI depth maps are cached as 16-bit PNG files.\n"
            "Leave blank to use the default location."
        )
        cache_layout.addWidget(self._cache_dir_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(self._on_browse_cache_dir)
        cache_layout.addWidget(browse_btn)

        form.addRow("Cache Directory:", cache_row)

        # Clear cache button
        clear_row = QWidget()
        clear_layout = QHBoxLayout(clear_row)
        clear_layout.setContentsMargins(0, 0, 0, 0)

        self._clear_cache_btn = QPushButton("Clear Cache")
        self._clear_cache_btn.setToolTip(
            "Delete all cached depth map PNG files from the cache directory."
        )
        self._clear_cache_btn.clicked.connect(self._on_clear_cache)
        clear_layout.addWidget(self._clear_cache_btn)
        clear_layout.addStretch()

        form.addRow("", clear_row)

        note = QLabel(
            "Depth maps are cached to avoid repeated Replicate API calls when "
            "only contour parameters change.  Cache entries are keyed by source "
            "image content (SHA-256)."
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
        cache_dir = settings.value("ai/depth_cache_dir", "") or ""
        self._cache_dir_edit.setText(cache_dir)

    def _save_settings(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("replicate/api_key", self._api_key_edit.text().strip())
        settings.setValue("ai/depth_cache_dir", self._cache_dir_edit.text().strip())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        self._save_settings()
        self.accept()

    def _on_browse_cache_dir(self) -> None:
        """Open a directory picker for the depth map cache directory."""
        current = self._cache_dir_edit.text().strip()
        if not current:
            current = str(pathlib.Path.home() / ".plottter" / "depth_cache")
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Depth Map Cache Directory",
            current,
        )
        if chosen:
            self._cache_dir_edit.setText(chosen)

    def _on_clear_cache(self) -> None:
        """Delete all .png files from the configured cache directory."""
        cache_dir = self._cache_dir_edit.text().strip()
        if not cache_dir:
            cache_dir = str(pathlib.Path.home() / ".plottter" / "depth_cache")

        if not os.path.isdir(cache_dir):
            QMessageBox.information(
                self,
                "Clear Cache",
                f"Cache directory does not exist:\n{cache_dir}",
            )
            return

        png_files = glob.glob(os.path.join(cache_dir, "*.png"))
        if not png_files:
            QMessageBox.information(
                self,
                "Clear Cache",
                "No cached depth maps found.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Clear Cache",
            f"Delete {len(png_files)} cached depth map file(s) from:\n{cache_dir}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        for f in png_files:
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass

        QMessageBox.information(
            self,
            "Clear Cache",
            f"Removed {removed} cached depth map file(s).",
        )

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

        from plottter.ai.replicate_client import ReplicateClient, ReplicateAPIError

        client = ReplicateClient(api_key=key)
        if not client.is_available():
            QMessageBox.warning(
                self,
                "Test Connection",
                "The 'replicate' package is not installed.\n"
                "Run: pip install plottter[ai]",
            )
            return

        # Attempt a lightweight API list call to verify the key
        try:
            import replicate  # type: ignore[import]
            replicate_client = replicate.Client(api_token=key)
            # A simple models list call; catches 401 Unauthorized quickly
            _ = list(replicate_client.models.list())
            status_msg = "Connection successful! API key is valid."
            success = True
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "Unauthorized" in msg or "authentication" in msg.lower():
                status_msg = "Authentication failed: invalid API key."
            elif "rate limit" in msg.lower():
                status_msg = f"Rate limited — key is valid, but you are being throttled.\n{msg}"
            else:
                status_msg = f"Connection error: {msg}"
            success = False

        if success:
            # Also update the status bar of the parent window if possible
            parent = self.parent()
            if hasattr(parent, "statusBar"):
                parent.statusBar().showMessage("Replicate API: connected", 5000)
            QMessageBox.information(self, "Test Connection", status_msg)
        else:
            QMessageBox.warning(self, "Test Connection", status_msg)
