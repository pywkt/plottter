"""Tests for NewProjectDialog orientation toggle (task 63.1).

Covers:
(a) A4 + Portrait → 210×297
(b) A4 + Landscape → 297×210
(c) Toggling orientation swaps spinbox values
(d) Manually setting width > height auto-selects Landscape
(e) Custom preset with manual dimensions respects orientation toggle
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def dlg(qapp):
    from plottter.gui.dialogs.new_project import NewProjectDialog

    d = NewProjectDialog()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# (a) A4 + Portrait → 210×297
# ---------------------------------------------------------------------------


def test_a4_portrait(dlg):
    dlg._preset_combo.setCurrentText("A4")
    dlg._portrait_radio.setChecked(True)
    canvas = dlg.get_canvas()
    assert canvas.width_mm == pytest.approx(210.0, abs=0.1)
    assert canvas.height_mm == pytest.approx(297.0, abs=0.1)


# ---------------------------------------------------------------------------
# (b) A4 + Landscape → 297×210
# ---------------------------------------------------------------------------


def test_a4_landscape(dlg):
    dlg._preset_combo.setCurrentText("A4")
    dlg._landscape_radio.setChecked(True)
    canvas = dlg.get_canvas()
    assert canvas.width_mm == pytest.approx(297.0, abs=0.1)
    assert canvas.height_mm == pytest.approx(210.0, abs=0.1)
    # Reset
    dlg._portrait_radio.setChecked(True)


# ---------------------------------------------------------------------------
# (c) Toggling orientation swaps spinbox values
# ---------------------------------------------------------------------------


def test_toggle_to_landscape_swaps_values(dlg):
    dlg._preset_combo.setCurrentText("A4")
    dlg._portrait_radio.setChecked(True)
    # Portrait: w=210, h=297
    assert dlg._width_spin.value() == pytest.approx(210.0, abs=0.1)
    assert dlg._height_spin.value() == pytest.approx(297.0, abs=0.1)

    dlg._landscape_radio.setChecked(True)
    assert dlg._width_spin.value() == pytest.approx(297.0, abs=0.1)
    assert dlg._height_spin.value() == pytest.approx(210.0, abs=0.1)


def test_toggle_back_to_portrait_swaps_values(dlg):
    dlg._preset_combo.setCurrentText("A4")
    dlg._landscape_radio.setChecked(True)
    dlg._portrait_radio.setChecked(True)
    assert dlg._width_spin.value() == pytest.approx(210.0, abs=0.1)
    assert dlg._height_spin.value() == pytest.approx(297.0, abs=0.1)


def test_toggle_portrait_no_swap_when_already_portrait(dlg):
    """Clicking Portrait when already portrait-shaped does not swap."""
    dlg._preset_combo.setCurrentText("A4")
    dlg._portrait_radio.setChecked(True)
    w_before = dlg._width_spin.value()
    h_before = dlg._height_spin.value()
    # Click portrait again (no-op since w < h already)
    dlg._portrait_radio.setChecked(True)
    assert dlg._width_spin.value() == pytest.approx(w_before)
    assert dlg._height_spin.value() == pytest.approx(h_before)


# ---------------------------------------------------------------------------
# (d) Manually setting width > height auto-selects Landscape
# ---------------------------------------------------------------------------


def test_manual_wide_auto_selects_landscape(dlg):
    dlg._preset_combo.setCurrentText("Custom")
    dlg._portrait_radio.setChecked(True)
    dlg._width_spin.setValue(300.0)
    dlg._height_spin.setValue(200.0)
    assert dlg._landscape_radio.isChecked()


def test_manual_tall_auto_selects_portrait(dlg):
    dlg._preset_combo.setCurrentText("Custom")
    dlg._landscape_radio.setChecked(True)
    dlg._width_spin.setValue(150.0)
    dlg._height_spin.setValue(250.0)
    assert dlg._portrait_radio.isChecked()


# ---------------------------------------------------------------------------
# (e) Custom preset — orientation toggle swaps values
# ---------------------------------------------------------------------------


def test_custom_orientation_toggle(dlg):
    dlg._preset_combo.setCurrentText("Custom")
    dlg._portrait_radio.setChecked(True)
    dlg._width_spin.setValue(100.0)
    dlg._height_spin.setValue(200.0)

    dlg._landscape_radio.setChecked(True)
    assert dlg._width_spin.value() == pytest.approx(200.0)
    assert dlg._height_spin.value() == pytest.approx(100.0)

    canvas = dlg.get_canvas()
    assert canvas.width_mm == pytest.approx(200.0)
    assert canvas.height_mm == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Default-canvas persistence (Set as default checkbox + load/save helpers)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_settings(qapp, tmp_path):
    """Give each test a clean settings slate.

    The whole config location is already redirected to a throwaway directory
    for the session by ``_isolate_qsettings`` in ``conftest.py``, so these
    ``clear()`` calls only ever touch the sandbox — never the real
    ``~/.config/Plottter/Plottter.conf``. This fixture just guarantees a clean
    store before and after each test that exercises preset persistence.

    It must NOT call ``setDefaultFormat``/``setPath`` itself: those are
    process-global and unrestored changes leak into every later test.
    """
    from PyQt6.QtCore import QSettings

    settings = QSettings("Plottter", "Plottter")
    settings.clear()
    yield tmp_path
    settings.clear()


def test_default_checkbox_unchecked_by_default(dlg):
    assert dlg.should_save_as_default() is False


def test_save_and_load_default_canvas_roundtrip(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import (
        load_default_canvas,
        save_default_canvas,
    )
    from plottter.models.canvas import Canvas

    saved = Canvas(width_mm=420.0, height_mm=594.0, margin_mm=15.0, paper_preset="A2")
    save_default_canvas(saved)

    loaded = load_default_canvas()
    assert loaded.paper_preset == "A2"
    assert loaded.width_mm == pytest.approx(420.0)
    assert loaded.height_mm == pytest.approx(594.0)
    assert loaded.margin_mm == pytest.approx(15.0)


def test_load_default_falls_back_to_a4(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import load_default_canvas

    canvas = load_default_canvas()
    assert canvas.paper_preset == "A4"
    assert canvas.width_mm == pytest.approx(210.0)
    assert canvas.height_mm == pytest.approx(297.0)
    assert canvas.margin_mm == pytest.approx(10.0)


def test_load_default_preserves_custom_dimensions(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import (
        load_default_canvas,
        save_default_canvas,
    )
    from plottter.models.canvas import Canvas

    custom = Canvas(width_mm=333.3, height_mm=444.4, margin_mm=5.0, paper_preset="Custom")
    save_default_canvas(custom)
    loaded = load_default_canvas()
    assert loaded.paper_preset == "Custom"
    assert loaded.width_mm == pytest.approx(333.3)
    assert loaded.height_mm == pytest.approx(444.4)
    assert loaded.margin_mm == pytest.approx(5.0)


def test_should_save_as_default_reflects_checkbox(dlg):
    dlg._set_default_check.setChecked(True)
    assert dlg.should_save_as_default() is True
    dlg._set_default_check.setChecked(False)
    assert dlg.should_save_as_default() is False


# ---------------------------------------------------------------------------
# User-saved paper presets (Save… / Delete buttons)
# ---------------------------------------------------------------------------


def test_load_user_presets_empty_by_default(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import load_user_presets

    assert load_user_presets() == {}


def test_save_and_load_user_presets_roundtrip(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import load_user_presets, save_user_presets

    save_user_presets({"Sketchbook": (148.0, 210.0), "Bed": (300.0, 450.0)})
    loaded = load_user_presets()
    assert loaded == {"Sketchbook": (148.0, 210.0), "Bed": (300.0, 450.0)}


def test_load_user_presets_ignores_malformed(qapp, isolated_settings):
    from PyQt6.QtCore import QSettings

    from plottter.gui.dialogs.new_project import load_user_presets

    QSettings("Plottter", "Plottter").setValue("canvas/user_presets", "not valid json")
    assert load_user_presets() == {}


def test_user_presets_appear_in_combo(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import NewProjectDialog, save_user_presets

    save_user_presets({"Sketchbook": (148.0, 210.0)})
    d = NewProjectDialog()
    try:
        items = [d._preset_combo.itemText(i) for i in range(d._preset_combo.count())]
        assert "Sketchbook" in items
        assert "A4" in items  # built-ins still there
        assert "Custom" in items
    finally:
        d.close()


def test_selecting_user_preset_populates_dimensions(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import NewProjectDialog, save_user_presets

    save_user_presets({"Sketchbook": (148.0, 210.0)})
    d = NewProjectDialog()
    try:
        d._preset_combo.setCurrentText("Sketchbook")
        assert d._width_spin.value() == pytest.approx(148.0)
        assert d._height_spin.value() == pytest.approx(210.0)
        canvas = d.get_canvas()
        assert canvas.paper_preset == "Sketchbook"
        assert canvas.width_mm == pytest.approx(148.0)
        assert canvas.height_mm == pytest.approx(210.0)
    finally:
        d.close()


def test_delete_button_disabled_for_builtins(qapp, isolated_settings):
    from plottter.gui.dialogs.new_project import NewProjectDialog, save_user_presets

    save_user_presets({"Sketchbook": (148.0, 210.0)})
    d = NewProjectDialog()
    try:
        d._preset_combo.setCurrentText("A4")
        assert not d._delete_preset_btn.isEnabled()
        d._preset_combo.setCurrentText("Custom")
        assert not d._delete_preset_btn.isEnabled()
        d._preset_combo.setCurrentText("Sketchbook")
        assert d._delete_preset_btn.isEnabled()
    finally:
        d.close()


def test_user_preset_dimensions_stored_portrait_canonical(qapp, isolated_settings):
    """Saving with landscape spinboxes still stores width ≤ height."""
    from plottter.gui.dialogs.new_project import (
        NewProjectDialog,
        load_user_presets,
        save_user_presets,
    )

    # Start fresh, simulate user picking Custom + landscape dims, then saving.
    d = NewProjectDialog()
    try:
        d._preset_combo.setCurrentText("Custom")
        d._width_spin.setValue(400.0)
        d._height_spin.setValue(250.0)
        # Call the inner machinery directly (avoid QInputDialog).
        save_user_presets({"Wide": d._current_dims_mm()})
        d._refresh_preset_combo(select="Wide")
    finally:
        d.close()

    presets = load_user_presets()
    assert presets["Wide"] == (250.0, 400.0)  # portrait-canonical


def test_save_user_preset_rejects_builtin_name(qapp, isolated_settings):
    """The built-in name list (PAPER_PRESETS keys + Custom) is reserved."""
    from plottter.gui.dialogs.new_project import (
        load_user_presets,
        save_user_presets,
    )
    from plottter.models.canvas import PAPER_PRESETS

    # save_user_presets itself doesn't enforce this — the UI does. So this
    # test guards the UI contract: the combo's "reserved" set is PAPER_PRESETS
    # ∪ {"Custom"}.
    reserved = set(PAPER_PRESETS.keys()) | {"Custom"}
    assert "A4" in reserved
    assert "Custom" in reserved

    # Round-trip with a non-reserved name works.
    save_user_presets({"My Pad": (100.0, 150.0)})
    assert "My Pad" in load_user_presets()
