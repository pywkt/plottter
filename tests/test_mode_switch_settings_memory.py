"""Generator settings must survive a round-trip through a non-generator mode.

Bug: Image to Lines → tweak Hatching → Mask Paint → back to Image to Lines
came back as Edge Detect with defaults, and the next layer click then
overwrote the layer's saved snapshot with those defaults, losing the real
settings for good.

Fix: on_mode_changed flushes the panel snapshot to the active layer when
leaving a generator mode, and re-applies the layer's snapshot (if taken in
the target mode) when entering one, instead of always selecting index 0.
"""

from __future__ import annotations

import pytest


def _make_project(n_layers=1):
    from plottter.models import Canvas, Layer, Project

    proj = Project(name="Test", canvas=Canvas.from_preset("A4"))
    for i in range(n_layers):
        proj.add_layer(Layer(name=f"L{i + 1}", color="#000000"))
    return proj


@pytest.fixture
def harness(qapp, qtbot):
    """SettingsPanel wired the way MainWindow wires it for mode changes."""
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel

    controller = ProjectController(_make_project(2))
    sp = SettingsPanel(controller)
    qtbot.addWidget(sp)
    # MainWindow._on_mode_change_requested -> settings_panel.on_mode_changed
    sp.mode_change_requested.connect(sp.on_mode_changed)
    layers = controller.current_project.layers
    sp._on_active_layer_changed(layers[0].id)
    return sp, controller, layers


def _pick_hatching(sp, angle=63.0):
    sp.on_mode_changed("Image to Lines")
    sp._generator_type_combo.setCurrentText("Hatching")
    sp._param_widgets["angle_deg"].setValue(angle)


def _shown(sp):
    gen = sp._generator_type_combo.currentText()
    w = sp._param_widgets.get("angle_deg")
    return gen, (w.value() if w is not None else None)


def test_unsaved_tweak_survives_mask_paint_round_trip(harness):
    sp, _ctrl, _layers = harness
    _pick_hatching(sp, 63.0)
    sp.on_mode_changed("Mask Paint")
    sp.on_mode_changed("Image to Lines")
    assert _shown(sp) == ("Hatching", 63.0)


def test_round_trip_through_every_non_generator_mode(harness):
    sp, _ctrl, _layers = harness
    _pick_hatching(sp, 42.0)
    for other in ("Color Separation", "Mask Paint", "Shape Drawing"):
        sp.on_mode_changed(other)
        sp.on_mode_changed("Image to Lines")
        assert _shown(sp) == ("Hatching", 42.0), other


def test_generated_snapshot_is_restored_not_reset(harness):
    sp, _ctrl, layers = harness
    _pick_hatching(sp, 63.0)
    sp.flush_current_snapshot()  # what Generate does
    assert layers[0].generator_info["generator_name"] == "Hatching"
    sp.on_mode_changed("Mask Paint")
    sp.on_mode_changed("Image to Lines")
    assert _shown(sp) == ("Hatching", 63.0)


def test_next_layer_click_no_longer_clobbers_saved_snapshot(harness):
    sp, _ctrl, layers = harness
    _pick_hatching(sp, 63.0)
    sp.flush_current_snapshot()
    sp.on_mode_changed("Mask Paint")
    sp.on_mode_changed("Image to Lines")
    sp._on_active_layer_changed(layers[1].id)  # saves panel state onto L1
    assert layers[0].generator_info["generator_name"] == "Hatching"
    assert layers[0].generator_info["params"]["angle_deg"] == 63.0
    sp._on_active_layer_changed(layers[0].id)
    assert _shown(sp) == ("Hatching", 63.0)


def test_fresh_layer_still_gets_first_generator_defaults(harness):
    sp, _ctrl, layers = harness
    sp.on_mode_changed("Image to Lines")
    first = sp._generator_type_combo.itemText(0)
    sp.on_mode_changed("Mask Paint")
    sp.on_mode_changed("Image to Lines")
    assert sp._generator_type_combo.currentText() == first
    assert layers[0].generator_info is not None  # flushed on leave
    assert layers[0].generator_info["generator_name"] == first


def test_snapshot_from_other_mode_does_not_hijack_entry(harness):
    """A Math Art snapshot on the layer must not be applied when entering
    Image to Lines; only a same-mode snapshot is restored."""
    sp, _ctrl, layers = harness
    sp.on_mode_changed("Math Art")
    sp._generator_type_combo.setCurrentText("Spirograph" if sp._generator_type_combo.findText("Spirograph") >= 0 else sp._generator_type_combo.itemText(1))
    math_gen = sp._generator_type_combo.currentText()
    sp.on_mode_changed("Image to Lines")  # flushes the Math Art snapshot
    assert layers[0].generator_info["mode"] == "Math Art"
    assert layers[0].generator_info["generator_name"] == math_gen
    assert sp._current_mode == "Image to Lines"
    assert sp._generator_type_combo.currentText() == sp._generator_type_combo.itemText(0)


def test_layer_click_restore_does_not_write_outgoing_state_onto_new_layer(harness):
    """Regression guard for the restore path: clicking a layer whose snapshot
    lives in another mode triggers mode_change_requested -> on_mode_changed.
    That nested call must not flush the *previous* layer's panel state onto
    the newly selected layer."""
    sp, _ctrl, layers = harness
    # L1: Image to Lines / Hatching. L2: Math Art snapshot.
    _pick_hatching(sp, 63.0)
    sp.flush_current_snapshot()
    sp._on_active_layer_changed(layers[1].id)
    sp.on_mode_changed("Math Art")
    math_gen = sp._generator_type_combo.currentText()
    sp.flush_current_snapshot()
    assert layers[1].generator_info["mode"] == "Math Art"

    # Click back to L1 while in Math Art: restore switches mode to Image to
    # Lines. L1's snapshot must remain Hatching, not the Math Art state.
    sp._on_active_layer_changed(layers[0].id)
    assert sp._current_mode == "Image to Lines"
    assert _shown(sp) == ("Hatching", 63.0)
    assert layers[0].generator_info["generator_name"] == "Hatching"
    assert layers[1].generator_info["generator_name"] == math_gen


def test_mode_switch_with_unchanged_state_does_not_mark_modified(harness):
    sp, ctrl, _layers = harness
    _pick_hatching(sp, 63.0)
    sp.flush_current_snapshot()
    ctrl.mark_saved()
    assert not ctrl.modified
    sp.on_mode_changed("Mask Paint")
    sp.on_mode_changed("Image to Lines")
    assert not ctrl.modified
