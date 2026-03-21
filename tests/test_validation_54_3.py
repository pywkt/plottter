"""Tests for task 54.3 — ProjectController mask operations.

Covers:
(a) save_mask stores mask and emits masks_changed signal
(b) delete_mask removes mask and emits masks_changed signal
(c) undo after delete_mask restores the mask
(d) mask_names() returns sorted list
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _make_controller():
    from plottter.gui.project_controller import ProjectController
    from plottter.models import Canvas, Project

    canvas = Canvas.from_preset("A4")
    proj = Project(name="Test", canvas=canvas)
    return ProjectController(proj)


# ---------------------------------------------------------------------------
# (a) save_mask stores mask and emits masks_changed
# ---------------------------------------------------------------------------


def test_save_mask_stores_data(qapp):
    ctrl = _make_controller()
    arr = np.ones((10, 10), dtype=np.float32) * 0.5

    ctrl.save_mask("alpha", arr)

    assert "alpha" in ctrl.current_project.masks
    # Verify round-trip is approximately correct
    loaded = ctrl.load_mask("alpha")
    np.testing.assert_allclose(loaded, arr, atol=1 / 255)


def test_save_mask_emits_signal(qapp):
    ctrl = _make_controller()
    arr = np.zeros((8, 8), dtype=np.float32)
    emitted = []
    ctrl.masks_changed.connect(lambda: emitted.append(1))

    ctrl.save_mask("beta", arr)

    assert len(emitted) == 1


def test_save_mask_overwrite_preserves_old_on_undo(qapp):
    ctrl = _make_controller()
    arr1 = np.ones((4, 4), dtype=np.float32) * 0.2
    arr2 = np.ones((4, 4), dtype=np.float32) * 0.8

    ctrl.save_mask("m", arr1)
    ctrl.save_mask("m", arr2)

    loaded_after = ctrl.load_mask("m")
    np.testing.assert_allclose(loaded_after, arr2, atol=1 / 255)

    # Undo overwrite → should restore arr1
    ctrl.undo_stack.undo()
    loaded_restored = ctrl.load_mask("m")
    np.testing.assert_allclose(loaded_restored, arr1, atol=1 / 255)


# ---------------------------------------------------------------------------
# (b) delete_mask removes mask and emits masks_changed
# ---------------------------------------------------------------------------


def test_delete_mask_removes_entry(qapp):
    ctrl = _make_controller()
    arr = np.zeros((6, 6), dtype=np.float32)
    ctrl.save_mask("to_delete", arr)

    ctrl.delete_mask("to_delete")

    assert "to_delete" not in ctrl.current_project.masks


def test_delete_mask_emits_signal(qapp):
    ctrl = _make_controller()
    arr = np.zeros((6, 6), dtype=np.float32)
    ctrl.save_mask("sig_mask", arr)

    emitted = []
    ctrl.masks_changed.connect(lambda: emitted.append(1))
    ctrl.delete_mask("sig_mask")

    assert len(emitted) == 1


# ---------------------------------------------------------------------------
# (c) undo after delete_mask restores the mask
# ---------------------------------------------------------------------------


def test_undo_delete_mask_restores(qapp):
    ctrl = _make_controller()
    arr = np.ones((5, 5), dtype=np.float32) * 0.7
    ctrl.save_mask("restorable", arr)

    ctrl.delete_mask("restorable")
    assert "restorable" not in ctrl.current_project.masks

    ctrl.undo_stack.undo()
    assert "restorable" in ctrl.current_project.masks
    restored = ctrl.load_mask("restorable")
    np.testing.assert_allclose(restored, arr, atol=1 / 255)


def test_undo_save_mask_deletes_if_was_new(qapp):
    ctrl = _make_controller()
    arr = np.zeros((4, 4), dtype=np.float32)

    ctrl.save_mask("brand_new", arr)
    assert "brand_new" in ctrl.current_project.masks

    ctrl.undo_stack.undo()
    assert "brand_new" not in ctrl.current_project.masks


# ---------------------------------------------------------------------------
# (d) mask_names() returns sorted list
# ---------------------------------------------------------------------------


def test_mask_names_sorted(qapp):
    ctrl = _make_controller()
    arr = np.zeros((4, 4), dtype=np.float32)
    ctrl.save_mask("zebra", arr)
    ctrl.save_mask("apple", arr)
    ctrl.save_mask("mango", arr)

    names = ctrl.mask_names()
    assert names == ["apple", "mango", "zebra"]


def test_mask_names_empty_when_no_masks(qapp):
    ctrl = _make_controller()
    assert ctrl.mask_names() == []


def test_mask_names_updates_after_delete(qapp):
    ctrl = _make_controller()
    arr = np.zeros((4, 4), dtype=np.float32)
    ctrl.save_mask("keep", arr)
    ctrl.save_mask("remove", arr)

    ctrl.delete_mask("remove")
    assert ctrl.mask_names() == ["keep"]


# ---------------------------------------------------------------------------
# rename_mask undo/redo
# ---------------------------------------------------------------------------


def test_rename_mask_basic(qapp):
    ctrl = _make_controller()
    arr = np.ones((4, 4), dtype=np.float32) * 0.3
    ctrl.save_mask("old_name", arr)

    ctrl.rename_mask("old_name", "new_name")

    assert "old_name" not in ctrl.current_project.masks
    assert "new_name" in ctrl.current_project.masks


def test_undo_rename_mask(qapp):
    ctrl = _make_controller()
    arr = np.ones((4, 4), dtype=np.float32) * 0.3
    ctrl.save_mask("first", arr)

    ctrl.rename_mask("first", "second")
    ctrl.undo_stack.undo()

    assert "first" in ctrl.current_project.masks
    assert "second" not in ctrl.current_project.masks
