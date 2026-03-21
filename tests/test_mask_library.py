"""Tests for mask library — save/load, round-trips, undo, and edge cases.

Covers:
(a) Project.save_mask + load_mask round-trip preserves data
(b) PNG encoding is lossless for uint8 (values within ±1/255 of original float32)
(c) project file save/load preserves masks
(d) backward compatibility — old project files without masks load correctly
(e) delete_mask removes the entry
(f) rename_mask changes the key but preserves data
(g) saving a mask with a duplicate name overwrites the old one
(h) mask_names returns sorted list
(i) large mask (A4 at 5px/mm) saves and loads correctly
(j) undo after delete_mask restores the mask
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project():
    from plottter.models.canvas import Canvas
    from plottter.models.project import Project

    canvas = Canvas.from_preset("A4")
    return Project(name="Test", canvas=canvas)


def _make_controller():
    from plottter.gui.project_controller import ProjectController

    proj = _make_project()
    return ProjectController(proj)


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


# ---------------------------------------------------------------------------
# (a) Project.save_mask + load_mask round-trip preserves data
# ---------------------------------------------------------------------------


def test_save_load_mask_round_trip():
    proj = _make_project()
    rng = np.random.default_rng(0)
    mask = rng.random((20, 30), dtype=np.float32)

    proj.save_mask("test", mask)
    restored = proj.load_mask("test")

    assert restored.shape == mask.shape
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, mask, atol=1 / 255.0)


def test_save_mask_stores_png_bytes():
    """save_mask stores non-empty bytes in project.masks."""
    proj = _make_project()
    mask = np.ones((5, 5), dtype=np.float32) * 0.5
    proj.save_mask("m", mask)

    assert "m" in proj.masks
    assert isinstance(proj.masks["m"], bytes)
    assert len(proj.masks["m"]) > 0


# ---------------------------------------------------------------------------
# (b) PNG encoding is lossless for uint8 (values within ±1/255 of original float32)
# ---------------------------------------------------------------------------


def test_png_encoding_exact_uint8_values():
    """Values that map exactly to uint8 boundaries round-trip with zero error."""
    proj = _make_project()
    uint8_vals = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    mask = (uint8_vals.astype(np.float32) / 255.0).reshape(1, 5)

    proj.save_mask("exact", mask)
    restored = proj.load_mask("exact")

    np.testing.assert_allclose(restored, mask, atol=0.0)


def test_png_quantization_error_bounded():
    """Arbitrary float32 values have at most 1/255 quantization error."""
    proj = _make_project()
    rng = np.random.default_rng(1)
    mask = rng.random((50, 50), dtype=np.float32)

    proj.save_mask("q", mask)
    restored = proj.load_mask("q")

    diff = np.abs(restored - mask)
    # PNG is lossless; only quantization from float32 → uint8 → float32
    assert float(diff.max()) <= 1.0 / 255.0 + 1e-7


# ---------------------------------------------------------------------------
# (c) project file save/load preserves masks
# ---------------------------------------------------------------------------


def test_project_file_preserves_single_mask(tmp_path):
    from plottter.io.project_file import save_project, load_project

    proj = _make_project()
    rng = np.random.default_rng(2)
    mask = rng.random((10, 15), dtype=np.float32)
    proj.save_mask("my_mask", mask)

    filepath = str(tmp_path / "proj.plottter")
    save_project(proj, filepath)
    loaded = load_project(filepath)

    assert "my_mask" in loaded.masks
    restored = loaded.load_mask("my_mask")
    assert restored.shape == mask.shape
    np.testing.assert_allclose(restored, mask, atol=1 / 255.0)


def test_project_file_preserves_multiple_masks(tmp_path):
    from plottter.io.project_file import save_project, load_project

    proj = _make_project()
    rng = np.random.default_rng(3)
    mask_a = rng.random((10, 15), dtype=np.float32)
    mask_b = rng.random((8, 12), dtype=np.float32)
    proj.save_mask("alpha", mask_a)
    proj.save_mask("beta", mask_b)

    filepath = str(tmp_path / "multi.plottter")
    save_project(proj, filepath)
    loaded = load_project(filepath)

    assert set(loaded.masks.keys()) == {"alpha", "beta"}
    np.testing.assert_allclose(loaded.load_mask("alpha"), mask_a, atol=1 / 255.0)
    np.testing.assert_allclose(loaded.load_mask("beta"), mask_b, atol=1 / 255.0)


# ---------------------------------------------------------------------------
# (d) backward compatibility — old project files without masks load correctly
# ---------------------------------------------------------------------------


def test_old_project_file_no_masks_key(tmp_path):
    """Old .plottter files without a 'masks' key should load with empty masks."""
    from plottter.io.project_file import load_project

    old_format = {
        "version": 1,
        "name": "OldProject",
        "canvas": {"width_mm": 210.0, "height_mm": 297.0},
        "layers": [],
    }
    filepath = str(tmp_path / "old.plottter")
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(old_format, fh)

    loaded = load_project(filepath)

    assert loaded.masks == {}
    assert loaded.name == "OldProject"
    assert loaded.layers == []


def test_old_project_file_layers_still_load(tmp_path):
    """Old project files retain layer data when masks key is absent."""
    from plottter.io.project_file import load_project

    old_format = {
        "version": 1,
        "name": "Legacy",
        "canvas": {"width_mm": 148.5, "height_mm": 210.0},
        "layers": [
            {"id": "abc", "name": "Ink", "color": "#000000", "paths": [], "visible": True, "locked": False, "opacity": 1.0},
        ],
    }
    filepath = str(tmp_path / "legacy.plottter")
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(old_format, fh)

    loaded = load_project(filepath)

    assert loaded.masks == {}
    assert len(loaded.layers) == 1
    assert loaded.layers[0].name == "Ink"


# ---------------------------------------------------------------------------
# (e) delete_mask removes the entry
# ---------------------------------------------------------------------------


def test_delete_mask_removes_entry():
    proj = _make_project()
    proj.save_mask("to_delete", np.zeros((5, 5), dtype=np.float32))
    assert "to_delete" in proj.masks

    proj.delete_mask("to_delete")

    assert "to_delete" not in proj.masks


def test_delete_mask_unknown_name_no_error():
    """Deleting a non-existent mask should silently do nothing."""
    proj = _make_project()
    proj.delete_mask("nonexistent")  # must not raise


def test_delete_mask_only_removes_target():
    """Deleting one mask leaves others intact."""
    proj = _make_project()
    arr = np.zeros((4, 4), dtype=np.float32)
    proj.save_mask("keep", arr)
    proj.save_mask("remove", arr)

    proj.delete_mask("remove")

    assert "keep" in proj.masks
    assert "remove" not in proj.masks


# ---------------------------------------------------------------------------
# (f) rename_mask changes the key but preserves data
# ---------------------------------------------------------------------------


def test_rename_mask_changes_key_preserves_data():
    proj = _make_project()
    rng = np.random.default_rng(4)
    mask = rng.random((6, 8), dtype=np.float32)
    proj.save_mask("old_name", mask)

    proj.rename_mask("old_name", "new_name")

    assert "old_name" not in proj.masks
    assert "new_name" in proj.masks
    restored = proj.load_mask("new_name")
    np.testing.assert_allclose(restored, mask, atol=1 / 255.0)


def test_rename_mask_bytes_identical():
    """rename_mask must not re-encode — raw bytes should be identical."""
    proj = _make_project()
    mask = np.ones((3, 3), dtype=np.float32) * 0.6
    proj.save_mask("src", mask)
    original_bytes = proj.masks["src"]

    proj.rename_mask("src", "dst")

    assert proj.masks["dst"] == original_bytes


# ---------------------------------------------------------------------------
# (g) saving a mask with a duplicate name overwrites the old one
# ---------------------------------------------------------------------------


def test_save_mask_duplicate_name_overwrites():
    proj = _make_project()
    arr1 = np.ones((4, 4), dtype=np.float32) * 0.2
    arr2 = np.ones((4, 4), dtype=np.float32) * 0.8
    proj.save_mask("dup", arr1)
    proj.save_mask("dup", arr2)

    # Exactly one entry
    assert list(proj.masks.keys()).count("dup") == 1
    # Value is arr2
    restored = proj.load_mask("dup")
    np.testing.assert_allclose(restored, arr2, atol=1 / 255.0)


# ---------------------------------------------------------------------------
# (h) mask_names returns sorted list
# ---------------------------------------------------------------------------


def test_mask_names_sorted(qapp):
    ctrl = _make_controller()
    arr = np.zeros((4, 4), dtype=np.float32)
    ctrl.save_mask("zebra", arr)
    ctrl.save_mask("apple", arr)
    ctrl.save_mask("mango", arr)

    assert ctrl.mask_names() == ["apple", "mango", "zebra"]


def test_mask_names_empty_project(qapp):
    ctrl = _make_controller()
    assert ctrl.mask_names() == []


def test_mask_names_updates_after_delete(qapp):
    ctrl = _make_controller()
    arr = np.zeros((4, 4), dtype=np.float32)
    ctrl.save_mask("keep", arr)
    ctrl.save_mask("gone", arr)

    ctrl.delete_mask("gone")

    assert ctrl.mask_names() == ["keep"]


# ---------------------------------------------------------------------------
# (i) large mask (A4 at 5px/mm) saves and loads correctly
# ---------------------------------------------------------------------------


def test_large_mask_a4_5px_per_mm(tmp_path):
    """A4 at 5 px/mm → 1050×1485 pixels — full project file round-trip."""
    from plottter.io.project_file import save_project, load_project

    proj = _make_project()
    # A4: 210 mm × 297 mm at 5 px/mm
    h = int(297 * 5)  # 1485
    w = int(210 * 5)  # 1050
    rng = np.random.default_rng(5)
    mask = rng.random((h, w), dtype=np.float32)

    proj.save_mask("large", mask)

    filepath = str(tmp_path / "large.plottter")
    save_project(proj, filepath)
    loaded = load_project(filepath)

    restored = loaded.load_mask("large")
    assert restored.shape == (h, w)
    np.testing.assert_allclose(restored, mask, atol=1 / 255.0)


# ---------------------------------------------------------------------------
# (j) undo after delete_mask restores the mask
# ---------------------------------------------------------------------------


def test_undo_delete_mask_restores(qapp):
    ctrl = _make_controller()
    rng = np.random.default_rng(6)
    mask = rng.random((8, 10), dtype=np.float32)
    ctrl.save_mask("undoable", mask)

    ctrl.delete_mask("undoable")
    assert "undoable" not in ctrl.current_project.masks

    ctrl.undo_stack.undo()

    assert "undoable" in ctrl.current_project.masks
    restored = ctrl.load_mask("undoable")
    np.testing.assert_allclose(restored, mask, atol=1 / 255.0)


def test_undo_delete_mask_data_exact(qapp):
    """Undo restores the exact same PNG bytes (no re-encoding)."""
    ctrl = _make_controller()
    arr = np.ones((5, 5), dtype=np.float32) * 0.4
    ctrl.save_mask("exact_undo", arr)
    original_bytes = ctrl.current_project.masks["exact_undo"]

    ctrl.delete_mask("exact_undo")
    ctrl.undo_stack.undo()

    assert ctrl.current_project.masks["exact_undo"] == original_bytes
