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
