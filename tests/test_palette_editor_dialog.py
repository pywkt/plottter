"""Tests for PaletteEditorDialog (standalone, no panel integration).

Run with:
    QT_QPA_PLATFORM=offscreen pytest tests/test_palette_editor_dialog.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Qt application fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ---------------------------------------------------------------------------
# palette_dir monkeypatch
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_palette_dir(tmp_path, monkeypatch):
    """Redirect palette_dir() to a tmp directory so tests don't touch ~/.plottter."""
    fake_dir = tmp_path / "palettes"
    fake_dir.mkdir()
    monkeypatch.setattr(
        "plottter.color.palette.palette_dir",
        lambda: fake_dir,
    )
    monkeypatch.setattr(
        "plottter.gui.dialogs.palette_editor_dialog.save_user_palette",
        _make_save_user_palette(fake_dir),
    )
    return fake_dir


def _make_save_user_palette(palette_dir: Path):
    """Return a save_user_palette that writes to *palette_dir*."""
    from plottter.color.palette import palette_slug, palette_to_dict
    import json

    def _save(p):
        from plottter.color.palette import PenPalette  # noqa: F401
        slug = palette_slug(p.name)
        fp = palette_dir / f"{slug}.json"
        fp.write_text(json.dumps(palette_to_dict(p), indent=2))
        return fp

    return _save


# ---------------------------------------------------------------------------
# Dialog fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dlg(qapp, _patch_palette_dir):
    from plottter.gui.dialogs.palette_editor_dialog import PaletteEditorDialog

    d = PaletteEditorDialog()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Helper to get the tmp palette dir from the monkeypatched save function
# ---------------------------------------------------------------------------


def _saved_files(dlg) -> list[Path]:
    """Return all JSON files written by the patched save_user_palette."""
    # The dlg fixture uses the autouse-patched function that writes to tmp_path.
    # We can introspect via the module-level monkeypatch.
    import plottter.gui.dialogs.palette_editor_dialog as mod
    # The patched function captures the fake_dir in its closure.
    # Recover it by calling palette_dir from the palette module (also patched).
    import plottter.color.palette as palette_mod
    return list(palette_mod.palette_dir().glob("*.json"))


# ---------------------------------------------------------------------------
# Test: adding colours and Save round-trips a 2-colour palette to disk
# ---------------------------------------------------------------------------


class TestSaveRoundtrip:
    def test_add_two_colors_and_save(self, dlg, tmp_path):
        """Add two colours via _append_color_row (bypassing QColorDialog),
        fill in name, click Save, verify JSON on disk."""
        from plottter.color.palette import palette_from_dict
        import plottter.color.palette as palette_mod

        dlg._name_edit.setText("Test Palette")
        dlg._desc_edit.setText("A test")
        dlg._append_color_row("#FF0000")
        dlg._append_color_row("#0000FF")

        assert dlg._color_list.count() == 2

        # Trigger save without QColorDialog interaction
        dlg._on_save()

        assert dlg._result_palette is not None
        result = dlg.get_result()
        assert result is not None
        assert result.name == "Test Palette"
        assert result.description == "A test"
        assert "#FF0000" in result.colors
        assert "#0000FF" in result.colors

        # Verify JSON persisted on disk
        files = list(palette_mod.palette_dir().glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        loaded = palette_from_dict(data)
        assert loaded.name == "Test Palette"
        assert set(loaded.colors) == {"#FF0000", "#0000FF"}


# ---------------------------------------------------------------------------
# Test: Cancel writes nothing
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_writes_nothing(self, dlg):
        import plottter.color.palette as palette_mod

        dlg._name_edit.setText("Will Not Save")
        dlg._append_color_row("#AABBCC")

        dlg.reject()

        assert dlg.get_result() is None
        files = list(palette_mod.palette_dir().glob("*.json"))
        assert len(files) == 0


# ---------------------------------------------------------------------------
# Test: Validation errors — empty name
# ---------------------------------------------------------------------------


class TestValidationEmptyName:
    def test_empty_name_shows_warning_and_does_not_accept(self, dlg, qapp):
        """Saving with an empty name must show a warning, not accept the dialog."""
        from PyQt6.QtWidgets import QMessageBox

        dlg._name_edit.setText("")
        dlg._append_color_row("#112233")

        accepted = []

        # Intercept QMessageBox.warning so it doesn't block the test
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok):
            dlg._on_save()

        # Dialog should NOT have been accepted
        assert dlg.get_result() is None


# ---------------------------------------------------------------------------
# Test: Validation errors — zero colours
# ---------------------------------------------------------------------------


class TestValidationZeroColors:
    def test_zero_colors_shows_warning_and_does_not_accept(self, dlg):
        """Saving with no colours must show a warning, not accept the dialog."""
        from PyQt6.QtWidgets import QMessageBox

        dlg._name_edit.setText("No Colors Palette")

        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok):
            dlg._on_save()

        assert dlg.get_result() is None


# ---------------------------------------------------------------------------
# Test: Editing a colour updates the swatch in the list
# ---------------------------------------------------------------------------


class TestEditColor:
    def test_edit_color_updates_list_row(self, dlg, qapp):
        """After editing a row's colour, the list item must show the new hex."""
        dlg._append_color_row("#AABBCC")
        dlg._color_list.setCurrentRow(0)

        # Simulate picking a new colour via _update_row (what _on_edit calls)
        dlg._update_row(0, "#112233")

        item = dlg._color_list.item(0)
        from PyQt6.QtCore import Qt

        assert item.text() == "#112233"
        assert item.data(Qt.ItemDataRole.UserRole) == "#112233"

    def test_on_edit_via_mocked_color_dialog(self, dlg, qapp):
        """_on_edit with a mocked QColorDialog updates the selected row."""
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtCore import Qt

        dlg._append_color_row("#FFFFFF")
        dlg._color_list.setCurrentRow(0)

        with patch.object(
            QColorDialog,
            "getColor",
            return_value=QColor("#FF00FF"),
        ):
            dlg._on_edit()

        item = dlg._color_list.item(0)
        assert item.text() == "#FF00FF"
        assert item.data(Qt.ItemDataRole.UserRole) == "#FF00FF"


# ---------------------------------------------------------------------------
# Test: Add colour via mocked QColorDialog
# ---------------------------------------------------------------------------


class TestAddColor:
    def test_add_appends_color_row(self, dlg, qapp):
        """_on_add with a mocked dialog must append a new row."""
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        from PyQt6.QtCore import Qt

        assert dlg._color_list.count() == 0

        with patch.object(
            QColorDialog,
            "getColor",
            return_value=QColor("#CAFE00"),
        ):
            dlg._on_add()

        assert dlg._color_list.count() == 1
        item = dlg._color_list.item(0)
        assert item.text() == "#CAFE00"
        assert item.data(Qt.ItemDataRole.UserRole) == "#CAFE00"


# ---------------------------------------------------------------------------
# Test: Move Up / Move Down
# ---------------------------------------------------------------------------


class TestMoveButtons:
    def test_move_up(self, dlg):
        dlg._append_color_row("#111111")
        dlg._append_color_row("#222222")
        dlg._color_list.setCurrentRow(1)
        dlg._on_move_up()
        assert dlg._color_list.item(0).text() == "#222222"
        assert dlg._color_list.item(1).text() == "#111111"
        assert dlg._color_list.currentRow() == 0

    def test_move_down(self, dlg):
        dlg._append_color_row("#AAAAAA")
        dlg._append_color_row("#BBBBBB")
        dlg._color_list.setCurrentRow(0)
        dlg._on_move_down()
        assert dlg._color_list.item(0).text() == "#BBBBBB"
        assert dlg._color_list.item(1).text() == "#AAAAAA"
        assert dlg._color_list.currentRow() == 1


# ---------------------------------------------------------------------------
# Test: Remove colour
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_deletes_selected_row(self, dlg):
        dlg._append_color_row("#DEADBE")
        dlg._append_color_row("#EFFACE")
        dlg._color_list.setCurrentRow(0)
        dlg._on_remove()
        assert dlg._color_list.count() == 1
        assert dlg._color_list.item(0).text() == "#EFFACE"


# ---------------------------------------------------------------------------
# Test: initial palette pre-fills fields
# ---------------------------------------------------------------------------


class TestInitialPalette:
    def test_initial_fills_name_and_colors(self, qapp, _patch_palette_dir):
        from plottter.color.palette import PenPalette
        from plottter.gui.dialogs.palette_editor_dialog import PaletteEditorDialog

        palette = PenPalette(
            name="Pre-filled",
            colors=("#AABBCC", "#112233"),
            description="Hello",
        )
        d = PaletteEditorDialog(initial=palette)
        assert d._name_edit.text() == "Pre-filled"
        assert d._desc_edit.text() == "Hello"
        assert d._color_list.count() == 2
        d.close()
