"""Tests for task 18.4 — FontParam and FontPicker widget.

These tests verify:
(a) FontPicker widget populates with system fonts
(b) selecting a font emits font_changed with a valid file path
(c) style dropdown updates when family changes
(d) Text generator has a FontParam for system_font_path (not randomizable)
(e) FontParam is in generators/base.py and is not randomizable by default
(f) FontPicker.set_font_path / font_path round-trip
(g) FontPicker handles unknown paths gracefully

GUI tests (a)-(c) and (f)-(g) require a QApplication.  We use the
``qapp`` fixture from pytest-qt when available, or create one manually.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# QApplication setup (headless-safe)
# ---------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    """Return (or create) a QApplication for GUI tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ---------------------------------------------------------------------------
# FontParam unit tests — no GUI needed
# ---------------------------------------------------------------------------

class TestFontParam:
    def test_font_param_exists(self):
        """FontParam can be imported from generators.base."""
        from plottter.generators.base import FontParam
        assert FontParam is not None

    def test_font_param_not_randomizable_by_default(self):
        """FontParam.randomizable is False by default."""
        from plottter.generators.base import FontParam
        p = FontParam(name="f", label="Font")
        assert p.randomizable is False

    def test_font_param_default_is_empty_string(self):
        """FontParam default is an empty string."""
        from plottter.generators.base import FontParam
        p = FontParam(name="f", label="Font")
        assert p.default == ""

    def test_font_param_custom_default(self):
        """FontParam accepts a custom file path as default."""
        from plottter.generators.base import FontParam
        p = FontParam(name="f", label="Font", default="/tmp/test.ttf")
        assert p.default == "/tmp/test.ttf"


# ---------------------------------------------------------------------------
# TextGenerator parameter checks — no GUI needed
# ---------------------------------------------------------------------------

class TestTextGeneratorFontParam:
    def _get_gen(self):
        from plottter.generators.text import TextGenerator
        return TextGenerator()

    def test_system_font_path_is_font_param(self):
        """The system_font_path parameter in TextGenerator is a FontParam."""
        from plottter.generators.base import FontParam
        gen = self._get_gen()
        params = {p.name: p for p in gen.get_parameters()}
        assert "system_font_path" in params
        assert isinstance(params["system_font_path"], FontParam)

    def test_system_font_path_not_randomizable(self):
        """The system_font_path FontParam is not randomizable."""
        gen = self._get_gen()
        params = {p.name: p for p in gen.get_parameters()}
        assert not params["system_font_path"].randomizable

    def test_system_font_path_visible_when_system_font(self):
        """system_font_path is only visible when font_type == 'System Font'."""
        gen = self._get_gen()
        params = {p.name: p for p in gen.get_parameters()}
        vw = params["system_font_path"].visible_when
        assert vw is not None
        assert "System Font" in vw.get("font_type", [])


# ---------------------------------------------------------------------------
# FontPicker widget tests — require QApplication
# ---------------------------------------------------------------------------

def _make_font_info(family: str, style: str, path: str, source: str = "system"):
    from plottter.fonts.discovery import FontInfo
    return FontInfo(family=family, style=style, file_path=path, source=source)


class TestFontPickerWidget:
    """Tests for the FontPicker widget."""

    def _make_picker(self, fonts=None):
        """Create a FontPicker pre-populated with *fonts* (mocked catalog)."""
        from plottter.gui.widgets.font_picker import FontPicker

        if fonts is None:
            fonts = [
                _make_font_info("DejaVu Sans", "Regular", "/f/DejaVuSans.ttf"),
                _make_font_info("DejaVu Sans", "Bold", "/f/DejaVuSans-Bold.ttf"),
                _make_font_info("Liberation Mono", "Regular", "/f/LiberationMono-Regular.ttf"),
            ]

        with patch(
            "plottter.gui.widgets.font_picker.FontPicker._populate_fonts",
            lambda self: _stub_populate(self, fonts),
        ):
            picker = FontPicker()
        return picker

    def test_picker_creates_without_error(self, qapp):
        """FontPicker can be instantiated without raising."""
        from plottter.gui.widgets.font_picker import FontPicker
        with patch("plottter.gui.widgets.font_picker.FontPicker._populate_fonts"):
            picker = FontPicker()
        assert picker is not None

    def test_font_path_initially_empty(self, qapp):
        """font_path() returns '' before any selection."""
        picker = self._make_picker()
        assert picker.font_path() == ""

    def test_set_font_path_round_trip(self, qapp):
        """set_font_path(path) then font_path() returns the same path."""
        picker = self._make_picker()
        test_path = "/f/DejaVuSans.ttf"
        picker.set_font_path(test_path)
        assert picker.font_path() == test_path

    def test_font_changed_signal_emitted(self, qapp):
        """font_changed is emitted when set_font_path changes the path."""
        picker = self._make_picker()
        received = []
        picker.font_changed.connect(received.append)
        picker.set_font_path("/f/DejaVuSans.ttf")
        assert len(received) == 1
        assert received[0] == "/f/DejaVuSans.ttf"

    def test_font_changed_not_emitted_for_same_path(self, qapp):
        """font_changed is NOT emitted when the path doesn't change."""
        picker = self._make_picker()
        picker.set_font_path("/f/DejaVuSans.ttf")
        received = []
        picker.font_changed.connect(received.append)
        picker.set_font_path("/f/DejaVuSans.ttf")  # same path again
        assert received == []

    def test_family_combo_populated(self, qapp):
        """Family combo contains the families from the mock catalog."""
        picker = self._make_picker()
        families = [
            picker._family_combo.itemText(i)
            for i in range(picker._family_combo.count())
        ]
        assert "DejaVu Sans" in families
        assert "Liberation Mono" in families

    def test_style_combo_updates_when_family_selected(self, qapp):
        """Selecting a family populates the style dropdown."""
        picker = self._make_picker()
        idx = picker._family_combo.findText("DejaVu Sans")
        assert idx >= 0
        picker._family_combo.setCurrentIndex(idx)
        styles = [
            picker._style_combo.itemText(i)
            for i in range(picker._style_combo.count())
        ]
        assert "Regular" in styles
        assert "Bold" in styles

    def test_style_combo_single_style_disabled(self, qapp):
        """Style combo is disabled when only one style is available."""
        picker = self._make_picker()
        idx = picker._family_combo.findText("Liberation Mono")
        assert idx >= 0
        picker._family_combo.setCurrentIndex(idx)
        assert not picker._style_combo.isEnabled()

    def test_set_font_path_syncs_combos(self, qapp):
        """set_font_path selects the correct family/style in the combos."""
        picker = self._make_picker()
        picker.set_font_path("/f/DejaVuSans-Bold.ttf")
        assert picker._family_combo.currentText() == "DejaVu Sans"
        assert picker._style_combo.currentText() == "Bold"

    def test_unknown_path_stored(self, qapp):
        """set_font_path with an unknown path stores the path without crashing."""
        picker = self._make_picker()
        picker.set_font_path("/unknown/font.ttf")
        assert picker.font_path() == "/unknown/font.ttf"

    def test_clear_path(self, qapp):
        """set_font_path('') clears the selection."""
        picker = self._make_picker()
        received = []
        picker.set_font_path("/f/DejaVuSans.ttf")
        picker.font_changed.connect(received.append)
        picker.set_font_path("")
        assert picker.font_path() == ""
        assert "" in received


# ---------------------------------------------------------------------------
# Helpers for mocking _populate_fonts
# ---------------------------------------------------------------------------

def _stub_populate(picker, fonts):
    """Replace FontPicker._populate_fonts with one that uses *fonts*."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QCompleter

    picker._populated = True

    family_map = {}
    for info in fonts:
        family_map.setdefault(info.family, []).append(info)
    picker._font_info_by_family = family_map

    families = sorted(family_map.keys(), key=str.lower)

    picker._family_combo.blockSignals(True)
    picker._family_combo.clear()
    picker._family_combo.addItem("")
    picker._family_combo.addItems(families)

    completer = QCompleter([""] + families, picker._family_combo)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    picker._family_combo.setCompleter(completer)
    picker._family_combo.blockSignals(False)

    if picker._font_path:
        picker._sync_combos_from_path(picker._font_path)
    else:
        picker._refresh_style_combo(None)
