"""Tests for the Preferences dialog — focused on the Overpass endpoint setting."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Helper: isolate QSettings so tests don't pollute real user config
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Redirect QSettings to a temporary INI file for each test."""
    ini = str(tmp_path / "test_prefs.ini")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    # Override the application + organisation so tests use a fresh store
    import PyQt6.QtCore as qtcore
    orig_app_name = QSettings.defaultFormat
    settings = QSettings(ini, QSettings.Format.IniFormat)
    monkeypatch.setattr(
        "plottter.gui.dialogs.preferences.QSettings",
        lambda *args, **kwargs: QSettings(ini, QSettings.Format.IniFormat),
    )
    yield settings
    settings.sync()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

_DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"


def _make_dialog(qtbot):
    from plottter.gui.dialogs.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# Tests: default endpoint
# ---------------------------------------------------------------------------

class TestOverpassEndpointDefault:
    """When no value is stored in QSettings, defaults are applied correctly."""

    def test_field_exists(self, qtbot):
        dlg = _make_dialog(qtbot)
        assert hasattr(dlg, "_overpass_endpoint_edit")

    def test_placeholder_is_default_url(self, qtbot):
        dlg = _make_dialog(qtbot)
        assert dlg._overpass_endpoint_edit.placeholderText() == _DEFAULT_ENDPOINT

    def test_field_empty_when_no_setting(self, qtbot):
        """When QSettings has no stored value the field is empty (placeholder shown)."""
        dlg = _make_dialog(qtbot)
        assert dlg._overpass_endpoint_edit.text() == ""

    def test_get_params_uses_default_when_field_empty(self, qtbot, _isolated_settings):
        """Reading QSettings with no value returns the default endpoint."""
        val = _isolated_settings.value(
            "map/overpass_endpoint",
            _DEFAULT_ENDPOINT,
            type=str,
        )
        assert val == _DEFAULT_ENDPOINT


# ---------------------------------------------------------------------------
# Tests: saving and reading back a custom endpoint
# ---------------------------------------------------------------------------

class TestOverpassEndpointPersistence:
    """A saved value persists and is read back via QSettings."""

    def test_save_stores_custom_endpoint(self, qtbot, _isolated_settings):
        custom = "https://overpass.kumi.systems/api/interpreter"
        dlg = _make_dialog(qtbot)
        dlg._overpass_endpoint_edit.setText(custom)
        # Trigger save via _save_settings (same path as OK button)
        dlg._save_settings()
        stored = _isolated_settings.value("map/overpass_endpoint", "", type=str)
        assert stored == custom

    def test_saved_value_is_loaded_on_reopen(self, qtbot, _isolated_settings):
        custom = "https://overpass.kumi.systems/api/interpreter"
        _isolated_settings.setValue("map/overpass_endpoint", custom)
        _isolated_settings.sync()
        dlg = _make_dialog(qtbot)
        assert dlg._overpass_endpoint_edit.text() == custom

    def test_blank_save_stores_empty_string(self, qtbot, _isolated_settings):
        dlg = _make_dialog(qtbot)
        dlg._overpass_endpoint_edit.setText("")
        dlg._save_settings()
        stored = _isolated_settings.value("map/overpass_endpoint", "MISSING", type=str)
        assert stored == ""

    def test_whitespace_is_stripped_on_save(self, qtbot, _isolated_settings):
        custom = "  https://overpass.kumi.systems/api/interpreter  "
        dlg = _make_dialog(qtbot)
        dlg._overpass_endpoint_edit.setText(custom)
        dlg._save_settings()
        stored = _isolated_settings.value("map/overpass_endpoint", "", type=str)
        assert stored == custom.strip()
