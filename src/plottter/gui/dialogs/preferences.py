"""Preferences dialog — application-wide settings.

Laid out as a sidebar of sections (left) over a switching panel (right), so each
group gets its full width and the dialog never grows past one screen as more
settings are added. Sections:

- AI Integration: Replicate.com API key entry and connection test.
- AI Cache: unified cache directory for depth maps, background removal, and masks.
- Map: Overpass API endpoint.
- Remote Optimization: SSH host/command for off-box path optimization.
- Remote Plotter: networked plot-daemon device (URL + token) for wireless plotting.
"""

from __future__ import annotations

import glob
import os
import pathlib

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
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
        self.setMinimumSize(640, 460)

        root = QVBoxLayout(self)

        # Section sidebar (left) + switching panel (right). Each section shows
        # one group at full width, so the long help text never gets clipped.
        body = QHBoxLayout()
        root.addLayout(body, 1)

        self._nav = QListWidget()
        self._nav.setFixedWidth(180)
        # Give the section rows vertical breathing room so they don't feel
        # crammed: a couple of px between rows plus internal padding per item.
        self._nav.setSpacing(2)
        self._nav.setStyleSheet("QListWidget::item { padding: 6px 4px; }")
        self._nav.currentRowChanged.connect(self._on_section_changed)
        body.addWidget(self._nav)

        self._stack = QStackedWidget()
        body.addWidget(self._stack, 1)

        self._add_section("AI Integration", self._build_ai_group())
        self._add_section("AI Cache", self._build_cache_group())
        self._add_section("Map", self._build_map_group())
        self._add_section("Remote Optimization", self._build_remote_optimize_group())
        self._add_section("Remote Plotter", self._build_remote_plotter_group())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._nav.setCurrentRow(0)

        self._load_settings()
        self._update_cache_size_label()

    # ------------------------------------------------------------------
    # Section plumbing
    # ------------------------------------------------------------------

    def _add_section(self, title: str, group: QGroupBox) -> None:
        """Register one settings section: a sidebar entry + a stacked page.

        The group is wrapped in a scroll area so an individual section that
        outgrows the dialog scrolls rather than clipping its content.
        """
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.Shape.NoFrame)
        page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(group)
        inner_layout.addStretch(1)
        page.setWidget(inner)

        self._stack.addWidget(page)
        self._nav.addItem(title)

    def _on_section_changed(self, row: int) -> None:
        if row >= 0:
            self._stack.setCurrentIndex(row)

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

    def _build_map_group(self) -> QGroupBox:
        group = QGroupBox("Map")
        form = QFormLayout(group)

        self._overpass_endpoint_edit = QLineEdit()
        self._overpass_endpoint_edit.setPlaceholderText(
            "https://overpass-api.de/api/interpreter"
        )
        self._overpass_endpoint_edit.setToolTip(
            "Overpass API endpoint used to fetch OpenStreetMap data.\n"
            "Leave blank to use the default (overpass-api.de).\n"
            "Switch to a mirror (e.g. overpass.kumi.systems) if you hit rate limits."
        )
        form.addRow("Overpass Endpoint:", self._overpass_endpoint_edit)

        note = QLabel(
            "Leave blank to use the default endpoint. "
            "Try <tt>https://overpass.kumi.systems/api/interpreter</tt> "
            "if you receive 429 or 504 errors."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(note)

        return group

    def _build_remote_optimize_group(self) -> QGroupBox:
        group = QGroupBox("Remote Optimization")
        form = QFormLayout(group)

        self._remote_optimize_host_edit = QLineEdit()
        self._remote_optimize_host_edit.setPlaceholderText("user@fastbox")
        self._remote_optimize_host_edit.setToolTip(
            "SSH target for the 'Optimize Current Layer Remotely' tool.\n"
            "Accepts any form ssh understands: 'host', 'user@host', or an\n"
            "alias defined in ~/.ssh/config. SSH keys / agent auth are\n"
            "recommended so the call doesn't block on a password prompt."
        )
        form.addRow("Remote Host:", self._remote_optimize_host_edit)

        self._remote_optimize_cmd_edit = QLineEdit()
        self._remote_optimize_cmd_edit.setPlaceholderText("plottter")
        self._remote_optimize_cmd_edit.setToolTip(
            "Command to invoke on the remote host. Leave blank to use\n"
            "'plottter' (works if the binary is on the non-interactive\n"
            "SSH PATH, e.g. symlinked to /usr/local/bin). Otherwise put\n"
            "an absolute path, typically the binary in a venv:\n"
            "  /home/USER/path/to/repo/.venv/bin/plottter"
        )
        form.addRow("Remote Command:", self._remote_optimize_cmd_edit)

        note = QLabel(
            "Leave Host blank to be prompted on each Optimize Remotely call. "
            "Leave Command blank to default to <tt>plottter</tt>. "
            "Tip: enable SSH ControlMaster in <tt>~/.ssh/config</tt> for the "
            "host to skip the connection handshake on repeat calls."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(note)

        return group

    def _build_remote_plotter_group(self) -> QGroupBox:
        group = QGroupBox("Remote Plotter")
        form = QFormLayout(group)

        self._remote_plotter_enabled = QCheckBox("Send plots to a remote device instead of USB")
        self._remote_plotter_enabled.setToolTip(
            "Offload plotting to a networked plot daemon (e.g. a Raspberry Pi\n"
            "wired to the plotter) so this machine is free during long plots.\n"
            "When enabled, the 'Plot with AxiDraw' dialog uses the device below."
        )
        form.addRow("", self._remote_plotter_enabled)

        self._remote_plotter_url = QLineEdit()
        self._remote_plotter_url.setPlaceholderText("http://plotter-pi.local:8080")
        self._remote_plotter_url.setToolTip(
            "Base URL of the plot daemon running on the remote device."
        )
        form.addRow("Device URL:", self._remote_plotter_url)

        self._remote_plotter_token = QLineEdit()
        self._remote_plotter_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._remote_plotter_token.setPlaceholderText("optional — leave blank if the daemon has no token")
        self._remote_plotter_token.setToolTip(
            "Bearer token the daemon requires, if any. Leave blank when the "
            "daemon is unauthenticated (e.g. on a trusted LAN / Tailscale)."
        )
        form.addRow("Token:", self._remote_plotter_token)

        test_row = QWidget()
        test_layout = QHBoxLayout(test_row)
        test_layout.setContentsMargins(0, 0, 0, 0)
        self._remote_plotter_test_btn = QPushButton("Test Connection")
        self._remote_plotter_test_btn.setToolTip(
            "Ping the daemon's health endpoint and report which plotter it sees."
        )
        self._remote_plotter_test_btn.clicked.connect(self._on_test_remote_plotter)
        test_layout.addWidget(self._remote_plotter_test_btn)
        test_layout.addStretch()
        form.addRow("", test_row)

        self._remote_plotter_status = QLabel("")
        self._remote_plotter_status.setWordWrap(True)
        form.addRow("", self._remote_plotter_status)

        note = QLabel(
            "Configure the wireless plotter (a Raspberry Pi running the plot "
            "daemon) here. The <b>Plot with AxiDraw</b> dialog then shows whether "
            "it's plotting via USB or over the network."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
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
        endpoint = settings.value("map/overpass_endpoint", "") or ""
        self._overpass_endpoint_edit.setText(endpoint)
        host = settings.value("optimize/remote_host", "") or ""
        self._remote_optimize_host_edit.setText(host)
        cmd = settings.value("optimize/remote_command", "") or ""
        self._remote_optimize_cmd_edit.setText(cmd)
        self._remote_plotter_enabled.setChecked(
            settings.value("remote_plotter/enabled", False, type=bool)
        )
        self._remote_plotter_url.setText(
            str(settings.value("remote_plotter/url", "") or "")
        )
        self._remote_plotter_token.setText(
            str(settings.value("remote_plotter/token", "") or "")
        )

    def _save_settings(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("replicate/api_key", self._api_key_edit.text().strip())
        settings.setValue("ai/cache_dir", self._cache_dir_edit.text().strip())
        # Remove old key if present
        settings.remove("ai/depth_cache_dir")
        endpoint = self._overpass_endpoint_edit.text().strip()
        settings.setValue("map/overpass_endpoint", endpoint)
        settings.setValue(
            "optimize/remote_host", self._remote_optimize_host_edit.text().strip()
        )
        settings.setValue(
            "optimize/remote_command", self._remote_optimize_cmd_edit.text().strip()
        )
        settings.setValue(
            "remote_plotter/enabled", self._remote_plotter_enabled.isChecked()
        )
        settings.setValue(
            "remote_plotter/url", self._remote_plotter_url.text().strip()
        )
        settings.setValue(
            "remote_plotter/token", self._remote_plotter_token.text().strip()
        )

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

    def _on_test_remote_plotter(self) -> None:
        """Ping the configured plot daemon and report what it sees.

        Builds a throwaway ``NetworkTransport`` from the current field values
        (not the saved settings) so the user can verify a URL before clicking
        OK. ``health()`` never raises — it returns ``connected=False`` on any
        network error — so a short timeout keeps the UI responsive.
        """
        url = self._remote_plotter_url.text().strip()
        if not url:
            QMessageBox.warning(
                self,
                "Test Connection",
                "Enter a Device URL before testing.",
            )
            return

        from plottter.export.transport import NetworkTransport

        token = self._remote_plotter_token.text().strip() or None
        transport = NetworkTransport(url, token, timeout=5.0)
        self._remote_plotter_test_btn.setEnabled(False)
        self._remote_plotter_status.setText("Testing…")
        try:
            status = transport.health()
        finally:
            self._remote_plotter_test_btn.setEnabled(True)

        if status.connected:
            self._remote_plotter_status.setStyleSheet("color: #2e7d32;")
            self._remote_plotter_status.setText(f"✓ {status.detail}")
        else:
            self._remote_plotter_status.setStyleSheet("color: #cc4400;")
            self._remote_plotter_status.setText(f"✗ {status.detail}")

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
            from plottter.ai.replicate_client import _USER_AGENT
            req = urllib.request.Request(
                "https://api.replicate.com/v1/account",
                headers={
                    "Authorization": f"Bearer {key}",
                    "User-Agent": _USER_AGENT,
                },
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
