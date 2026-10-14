"""Per-layer preview in Color Separation mode.

Selecting a separated layer swaps the canvas image overlay (the panel's
``image_preprocessed`` signal) for that layer's generator input — the mask as
``_separation_mask_to_luminance`` builds it — so the user can see what each
layer actually got.  Selecting any other layer, or leaving the mode, restores
the normal preprocessed image.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _make_project():
    from plottter.models import Canvas, Layer, Project

    proj = Project(name="Test", canvas=Canvas.from_preset("A4"))
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


def _gradient(h=120, w=160):
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    return np.repeat(ramp[None, :], h, axis=0)[:, :, None].repeat(3, axis=2)


@pytest.fixture
def separated(qapp, qtbot):
    """Panel in Color Separation mode with three Luminance layers and a
    recorder of every image_preprocessed emission."""
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel

    ctrl = ProjectController(_make_project())
    sp = SettingsPanel(ctrl)
    qtbot.addWidget(sp)
    sp.on_mode_changed("Color Separation")
    sp._raw_image = _gradient()
    sp._preprocessed_image = sp._raw_image[:, :, 0].copy()
    sp._color_sep_method_combo.setCurrentText("Luminance")
    sp._color_sep_num_colors_spin.setValue(3)
    with patch("PyQt6.QtWidgets.QMessageBox.information"):
        sp._on_separate()
        qtbot.waitUntil(lambda: len(sp._separated_layer_ids) == 3, timeout=10000)
    emitted: list = []
    sp.image_preprocessed.connect(lambda img: emitted.append(img))
    return sp, ctrl, emitted


def test_selecting_separated_layer_shows_its_generator_input(separated):
    sp, _ctrl, emitted = separated
    lid = sp._separated_layer_ids[0]  # Shadows band
    sp._on_active_layer_changed(lid)
    assert len(emitted) == 1
    preview = emitted[0]
    from plottter.io.image_import import to_grayscale

    mask, src = sp._layer_masks[lid]
    gray = to_grayscale(src) if src.ndim == 3 else src
    assert preview.shape == mask.shape
    # Outside the band: white. Inside: the image's own gray values.
    assert np.all(preview[~mask] == 255)
    assert np.array_equal(preview[mask], gray[mask])
    assert preview[mask].max() < 255


def test_each_separated_layer_gets_its_own_preview(separated):
    sp, _ctrl, emitted = separated
    for lid in sp._separated_layer_ids:
        sp._on_active_layer_changed(lid)
    assert len(emitted) == 3
    # Bands are disjoint, so no two previews are identical.
    assert not np.array_equal(emitted[0], emitted[1])
    assert not np.array_equal(emitted[1], emitted[2])


def test_selecting_plain_layer_restores_preprocessed_image(separated):
    sp, ctrl, emitted = separated
    sp._on_active_layer_changed(sp._separated_layer_ids[1])
    plain = ctrl.current_project.layers[0].id
    sp._on_active_layer_changed(plain)
    assert len(emitted) == 2
    assert emitted[1] is sp._preprocessed_image


def test_plain_layer_without_prior_preview_emits_nothing(separated):
    sp, ctrl, emitted = separated
    sp._on_active_layer_changed(ctrl.current_project.layers[0].id)
    assert emitted == []


def test_leaving_mode_restores_and_entering_reapplies(separated):
    sp, ctrl, emitted = separated
    lid = sp._separated_layer_ids[2]
    ctrl.set_active_layer(lid)  # controller signal drives the panel hook
    assert len(emitted) == 1

    sp.on_mode_changed("Mask Paint")
    assert len(emitted) == 2
    assert emitted[1] is sp._preprocessed_image

    sp.on_mode_changed("Color Separation")
    assert len(emitted) == 3
    assert np.array_equal(emitted[2], emitted[0])


def test_no_preview_outside_color_separation_mode(separated):
    sp, _ctrl, emitted = separated
    sp.on_mode_changed("Image to Lines")
    emitted.clear()
    sp._on_active_layer_changed(sp._separated_layer_ids[0])
    assert emitted == []
