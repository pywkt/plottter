"""Tests for task 26.4 — Delete and rename for user presets.

Verifies:
(a) Right-clicking a user preset shows Rename and Delete options.
(b) Right-clicking a built-in preset or "Custom" shows no context menu.
(c) Deleting a preset removes it from the combo and from disk.
(d) Renaming a preset updates the combo and the file on disk.
(e) After delete, combo reverts to "Custom".
(f) Rename dialog is pre-filled with current name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─── headless Qt ────────────────────────────────────────────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ─── minimal mock generator ─────────────────────────────────────────────────

def _make_mock_generator(name: str = "Test Generator", builtin_presets=None):
    """Return a minimal generator-like object usable with SettingsPanel."""
    from plottter.generators.base import FloatParam, Preset

    if builtin_presets is None:
        builtin_presets = [
            Preset(name="Built-in Preset A", params={"radius": 5.0}),
        ]

    gen = MagicMock()
    gen.name = name
    gen.get_presets.return_value = builtin_presets
    gen.get_parameters.return_value = [
        FloatParam(name="radius", label="Radius (mm)", min=0.1, max=50.0, default=5.0),
    ]
    return gen


# ─── SettingsPanel factory ───────────────────────────────────────────────────

@pytest.fixture
def panel(qapp, tmp_path):
    """Create a SettingsPanel with a minimal mock controller."""
    from plottter.gui.settings_panel import SettingsPanel
    from plottter.models import Canvas, Layer, Project
    from plottter.gui.project_controller import ProjectController

    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="Test", canvas=canvas)
    project.add_layer(Layer(name="L1", color="#000000"))
    controller = ProjectController(project)

    sp = SettingsPanel(controller)
    sp._current_mode = "Math Art"
    return sp, tmp_path


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_real_preset(generator_name: str, preset_name: str, params: dict, presets_dir: Path):
    from plottter.presets.user_presets import save_user_preset
    from plottter.generators.base import Preset

    save_user_preset(generator_name, Preset(name=preset_name, params=params),
                     presets_dir=presets_dir)


def _patch_load(presets_dir: Path):
    """Context manager that redirects load_user_presets to use presets_dir."""
    from plottter.presets.user_presets import load_user_presets as real_load

    def patched(name):
        return real_load(name, presets_dir=presets_dir)

    return patch(
        "plottter.presets.user_presets.load_user_presets",
        side_effect=patched,
    )


def _patch_ops(presets_dir: Path):
    """Patch all user_presets I/O operations to use presets_dir."""
    from plottter.presets import user_presets as _up

    real_load = _up.load_user_presets
    real_delete = _up.delete_user_preset
    real_rename = _up.rename_user_preset

    class _MultiPatch:
        def __enter__(self):
            self._p1 = patch.object(_up, "load_user_presets",
                                    side_effect=lambda name: real_load(name, presets_dir=presets_dir))
            self._p2 = patch.object(_up, "delete_user_preset",
                                    side_effect=lambda gen, name: real_delete(gen, name, presets_dir=presets_dir))
            self._p3 = patch.object(_up, "rename_user_preset",
                                    side_effect=lambda gen, old, new: real_rename(gen, old, new, presets_dir=presets_dir))
            self._p1.__enter__()
            self._p2.__enter__()
            self._p3.__enter__()
            return self

        def __exit__(self, *args):
            self._p3.__exit__(*args)
            self._p2.__exit__(*args)
            self._p1.__exit__(*args)

    return _MultiPatch()


def _find_item_row(combo, text: str) -> int:
    """Return the combo row index for the given item text, or -1."""
    for i in range(combo.count()):
        if combo.itemText(i) == text:
            return i
    return -1


def _make_view_index(row: int):
    """Return a mock QModelIndex that reports the given row."""
    mock_index = MagicMock()
    mock_index.row.return_value = row
    return mock_index


# ─── (a) Right-click user preset shows rename/delete ─────────────────────────

class TestContextMenuUserPreset:
    def test_user_preset_shows_rename_and_delete_actions(self, panel, tmp_path):
        """Right-clicking a user preset instantiates QMenu with Rename and Delete."""
        from PyQt6.QtCore import QPoint

        sp, _ = panel
        gen = _make_mock_generator("CtxMenuUser")
        _save_real_preset("CtxMenuUser", "My Preset", {"radius": 3.0}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        user_row = _find_item_row(sp._preset_combo, "My Preset")
        assert user_row >= 0, "User preset must be in combo"

        mock_index = _make_view_index(user_row)
        with patch.object(sp._preset_combo.view(), "indexAt", return_value=mock_index):
            with patch("plottter.gui.settings_panel.QMenu") as MockQMenu:
                mock_menu = MagicMock()
                MockQMenu.return_value = mock_menu
                mock_menu.exec.return_value = None  # user dismisses

                sp._on_preset_context_menu(QPoint(0, 0))

                assert MockQMenu.called, "QMenu should have been instantiated for a user preset"
                action_labels = [c.args[0] for c in mock_menu.addAction.call_args_list]
                assert "Rename Preset" in action_labels
                assert "Delete Preset" in action_labels


# ─── (b) Right-click built-in or "Custom" shows nothing ──────────────────────

class TestContextMenuBuiltinAndCustom:
    def test_builtin_preset_shows_no_menu(self, panel, tmp_path):
        """Right-clicking a built-in preset must NOT show a context menu."""
        from PyQt6.QtCore import QPoint

        sp, _ = panel
        gen = _make_mock_generator("CtxMenuBuiltin")

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        builtin_row = _find_item_row(sp._preset_combo, "Built-in Preset A")
        assert builtin_row >= 0

        mock_index = _make_view_index(builtin_row)
        with patch.object(sp._preset_combo.view(), "indexAt", return_value=mock_index):
            with patch("plottter.gui.settings_panel.QMenu") as MockQMenu:
                sp._on_preset_context_menu(QPoint(0, 0))
                assert not MockQMenu.called, "QMenu must NOT appear for built-in presets"

    def test_custom_item_shows_no_menu(self, panel, tmp_path):
        """Right-clicking 'Custom' must NOT show a context menu."""
        from PyQt6.QtCore import QPoint

        sp, _ = panel
        gen = _make_mock_generator("CtxMenuCustom")

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        custom_row = _find_item_row(sp._preset_combo, "Custom")
        assert custom_row >= 0

        mock_index = _make_view_index(custom_row)
        with patch.object(sp._preset_combo.view(), "indexAt", return_value=mock_index):
            with patch("plottter.gui.settings_panel.QMenu") as MockQMenu:
                sp._on_preset_context_menu(QPoint(0, 0))
                assert not MockQMenu.called, "QMenu must NOT appear for 'Custom'"


# ─── (c) Delete removes from combo and from disk ─────────────────────────────

class TestDeletePreset:
    def test_delete_removes_preset_from_combo(self, panel, tmp_path):
        """After deletion the preset is absent from the combo."""
        from PyQt6.QtWidgets import QMessageBox

        sp, _ = panel
        gen = _make_mock_generator("DelComboTest")
        _save_real_preset("DelComboTest", "Gone Preset", {"radius": 7.0}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        assert _find_item_row(sp._preset_combo, "Gone Preset") >= 0

        with _patch_ops(tmp_path):
            with patch.object(QMessageBox, "question",
                              return_value=QMessageBox.StandardButton.Yes):
                sp._delete_user_preset_action("Gone Preset")

        assert _find_item_row(sp._preset_combo, "Gone Preset") == -1

    def test_delete_removes_preset_from_disk(self, panel, tmp_path):
        """After deletion the preset no longer exists in the JSON file."""
        from PyQt6.QtWidgets import QMessageBox
        from plottter.presets.user_presets import load_user_presets

        sp, _ = panel
        gen = _make_mock_generator("DelDiskTest")
        _save_real_preset("DelDiskTest", "Disk Gone", {"radius": 4.0}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        presets_before = load_user_presets("DelDiskTest", presets_dir=tmp_path)
        assert any(p.name == "Disk Gone" for p in presets_before)

        with _patch_ops(tmp_path):
            with patch.object(QMessageBox, "question",
                              return_value=QMessageBox.StandardButton.Yes):
                sp._delete_user_preset_action("Disk Gone")

        presets_after = load_user_presets("DelDiskTest", presets_dir=tmp_path)
        assert not any(p.name == "Disk Gone" for p in presets_after)

    def test_cancel_delete_keeps_preset(self, panel, tmp_path):
        """Cancelling the delete confirmation keeps the preset intact."""
        from PyQt6.QtWidgets import QMessageBox

        sp, _ = panel
        gen = _make_mock_generator("DelCancelTest")
        _save_real_preset("DelCancelTest", "Keep Me", {"radius": 2.5}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        with _patch_ops(tmp_path):
            with patch.object(QMessageBox, "question",
                              return_value=QMessageBox.StandardButton.No):
                sp._delete_user_preset_action("Keep Me")

        assert _find_item_row(sp._preset_combo, "Keep Me") >= 0


# ─── (e) After delete, combo reverts to "Custom" ─────────────────────────────

class TestDeleteRevertsToCustom:
    def test_combo_reverts_to_custom_after_delete(self, panel, tmp_path):
        """After deleting a user preset the combo selection reverts to 'Custom'."""
        from PyQt6.QtWidgets import QMessageBox

        sp, _ = panel
        gen = _make_mock_generator("RevertCustomTest")
        _save_real_preset("RevertCustomTest", "Revert Me", {"radius": 8.0}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        idx = _find_item_row(sp._preset_combo, "Revert Me")
        sp._preset_combo.blockSignals(True)
        sp._preset_combo.setCurrentIndex(idx)
        sp._preset_combo.blockSignals(False)
        assert sp._preset_combo.currentText() == "Revert Me"

        with _patch_ops(tmp_path):
            with patch.object(QMessageBox, "question",
                              return_value=QMessageBox.StandardButton.Yes):
                sp._delete_user_preset_action("Revert Me")

        assert sp._preset_combo.currentText() == "Custom"


# ─── (d) Rename updates combo and disk ───────────────────────────────────────

class TestRenamePreset:
    def test_rename_updates_combo_items(self, panel, tmp_path):
        """After rename, new name is in the combo and old name is absent."""
        from PyQt6.QtWidgets import QInputDialog

        sp, _ = panel
        gen = _make_mock_generator("RenameComboTest")
        _save_real_preset("RenameComboTest", "Old Name", {"radius": 2.0}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        with _patch_ops(tmp_path):
            with patch.object(QInputDialog, "getText", return_value=("New Name", True)):
                sp._rename_user_preset_action("Old Name")

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "New Name" in items
        assert "Old Name" not in items

    def test_rename_updates_disk(self, panel, tmp_path):
        """After rename, the JSON file reflects the new name."""
        from PyQt6.QtWidgets import QInputDialog
        from plottter.presets.user_presets import load_user_presets

        sp, _ = panel
        gen = _make_mock_generator("RenameDiskTest")
        _save_real_preset("RenameDiskTest", "Before Rename", {"radius": 5.5}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        with _patch_ops(tmp_path):
            with patch.object(QInputDialog, "getText",
                              return_value=("After Rename", True)):
                sp._rename_user_preset_action("Before Rename")

        presets = load_user_presets("RenameDiskTest", presets_dir=tmp_path)
        names = [p.name for p in presets]
        assert "After Rename" in names
        assert "Before Rename" not in names

    def test_rename_selects_new_name_in_combo(self, panel, tmp_path):
        """After rename the combo current selection is the new name."""
        from PyQt6.QtWidgets import QInputDialog

        sp, _ = panel
        gen = _make_mock_generator("RenameSelectTest")
        _save_real_preset("RenameSelectTest", "Before", {"radius": 1.0}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        with _patch_ops(tmp_path):
            with patch.object(QInputDialog, "getText", return_value=("After", True)):
                sp._rename_user_preset_action("Before")

        assert sp._preset_combo.currentText() == "After"

    def test_cancel_rename_keeps_original_name(self, panel, tmp_path):
        """Cancelling the rename dialog leaves the preset name unchanged."""
        from PyQt6.QtWidgets import QInputDialog

        sp, _ = panel
        gen = _make_mock_generator("RenameCancelTest")
        _save_real_preset("RenameCancelTest", "Stay Same", {"radius": 3.0}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        with _patch_ops(tmp_path):
            with patch.object(QInputDialog, "getText", return_value=("", False)):
                sp._rename_user_preset_action("Stay Same")

        assert _find_item_row(sp._preset_combo, "Stay Same") >= 0


# ─── (f) Rename dialog pre-filled with current preset name ───────────────────

class TestRenameDialogPrefilled:
    def test_rename_dialog_text_is_prefilled_with_current_name(self, panel, tmp_path):
        """The QInputDialog for rename is pre-filled with the existing preset name."""
        from PyQt6.QtWidgets import QInputDialog

        sp, _ = panel
        gen = _make_mock_generator("PrefillTest")
        _save_real_preset("PrefillTest", "My Exact Name", {"radius": 3.3}, tmp_path)

        with _patch_ops(tmp_path):
            sp.set_generator(gen)

        captured: dict = {}

        def capture_getText(parent, title, label, text="", **kwargs):
            captured["text"] = text
            return ("My Exact Name", False)  # cancel — we only want to inspect the call

        with _patch_ops(tmp_path):
            with patch.object(QInputDialog, "getText", side_effect=capture_getText):
                sp._rename_user_preset_action("My Exact Name")

        assert captured.get("text") == "My Exact Name", (
            f"Rename dialog should be pre-filled with 'My Exact Name', got {captured.get('text')!r}"
        )
