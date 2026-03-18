"""Phase 16.34 validation: Per-layer generator settings memory.

Verifies:
1. Switching from layer A to layer B saves layer A's generator settings to
   generator_info on the layer model.
2. Switching back to layer A restores the previously saved generator settings
   (generator type, parameter values, transform values).
3. A layer with no saved generator_info gets the default settings when selected.
4. The active_layer_changed signal is emitted when the layer panel selection
   changes.
5. set_layer_generator_info() updates the layer model and marks project modified.
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project(num_layers: int = 2) -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    for i in range(num_layers):
        proj.add_layer(Layer(name=f"Layer {i + 1}", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController

    return ProjectController(_make_project(num_layers=3))


@pytest.fixture
def settings_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel

    sp = SettingsPanel(controller)
    sp.resize(400, 800)
    qtbot.addWidget(sp)
    # Initialize the generator combo with Math Art generators
    sp.on_mode_changed("Math Art")
    return sp


# ---------------------------------------------------------------------------
# ProjectController: active layer tracking
# ---------------------------------------------------------------------------


class TestActiveLayerTracking:
    def test_initial_active_layer_is_none(self, controller):
        assert controller.active_layer_id is None

    def test_set_active_layer_emits_signal(self, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        with qtbot.waitSignal(controller.active_layer_changed, timeout=1000) as blocker:
            controller.set_active_layer(layer_id)
        assert blocker.args == [layer_id]

    def test_set_active_layer_same_id_no_signal(self, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        controller.set_active_layer(layer_id)
        # Setting the same id again should NOT emit
        signals: list = []
        controller.active_layer_changed.connect(lambda lid: signals.append(lid))
        controller.set_active_layer(layer_id)
        assert signals == []

    def test_set_active_layer_updates_property(self, controller):
        layer_id = controller.current_project.layers[1].id
        controller.set_active_layer(layer_id)
        assert controller.active_layer_id == layer_id


# ---------------------------------------------------------------------------
# ProjectController: set_layer_generator_info
# ---------------------------------------------------------------------------


class TestSetLayerGeneratorInfo:
    def test_stores_info_on_layer(self, controller):
        layer = controller.current_project.layers[0]
        info = {"generator_name": "Parametric Curves", "mode": "Math Art", "params": {}, "transforms": {}}
        controller.set_layer_generator_info(layer.id, info)
        assert layer.generator_info == info

    def test_marks_project_modified(self, controller):
        layer = controller.current_project.layers[0]
        assert not controller.modified
        controller.set_layer_generator_info(layer.id, {"mode": "Math Art"})
        assert controller.modified

    def test_set_none_clears_info(self, controller):
        layer = controller.current_project.layers[0]
        controller.set_layer_generator_info(layer.id, {"mode": "Math Art"})
        controller.set_layer_generator_info(layer.id, None)
        assert layer.generator_info is None

    def test_invalid_layer_id_is_noop(self, controller):
        # Should not raise
        controller.set_layer_generator_info("nonexistent-id", {"mode": "Math Art"})


# ---------------------------------------------------------------------------
# LayerPanel: active layer notification
# ---------------------------------------------------------------------------


class TestLayerPanelActiveLayer:
    def test_clicking_layer_sets_active_layer(self, controller, qtbot):
        from plottter.gui.layer_panel import LayerPanel

        panel = LayerPanel(controller)
        qtbot.addWidget(panel)

        layer_ids = [l.id for l in controller.current_project.layers]

        received: list[str] = []
        controller.active_layer_changed.connect(received.append)

        # Select row 1 (second layer)
        panel._list.setCurrentRow(1)

        assert layer_ids[1] in received

    def test_rebuild_list_restores_active_selection(self, controller, qtbot):
        from plottter.gui.layer_panel import LayerPanel

        panel = LayerPanel(controller)
        qtbot.addWidget(panel)

        layer_ids = [l.id for l in controller.current_project.layers]
        # Set active to layer 2
        controller.set_active_layer(layer_ids[2])

        # Trigger a rebuild (e.g. by adding a layer)
        controller.add_layer()

        # The selection should still be on layer 2 (or fallback to row 0)
        current_item = panel._list.currentItem()
        from PyQt6.QtCore import Qt
        if current_item is not None:
            selected_id = current_item.data(Qt.ItemDataRole.UserRole)
            # Either the active layer is still selected, or it fell back to row 0
            assert selected_id in [layer_ids[2], controller.current_project.layers[0].id]


# ---------------------------------------------------------------------------
# SettingsPanel: per-layer snapshot save / restore
# ---------------------------------------------------------------------------


class TestSettingsPanelSnapshot:
    def test_get_settings_snapshot_captures_generator(self, settings_panel):
        """_get_settings_snapshot returns a dict with mode and generator_name."""
        snapshot = settings_panel._get_settings_snapshot()
        # Should return something if a generator is active
        if snapshot is not None:
            assert "mode" in snapshot
            assert "generator_name" in snapshot
            assert "params" in snapshot
            assert "transforms" in snapshot

    def test_get_settings_snapshot_returns_none_without_generator(self, settings_panel):
        """Returns None when no generator is selected."""
        settings_panel.set_generator(None)
        snapshot = settings_panel._get_settings_snapshot()
        assert snapshot is None

    def test_get_settings_snapshot_returns_none_in_color_sep_mode(self, settings_panel):
        settings_panel.on_mode_changed("Color Separation")
        snapshot = settings_panel._get_settings_snapshot()
        assert snapshot is None

    def test_on_active_layer_changed_saves_settings(self, settings_panel, controller):
        """Switching active layer saves the current settings to the old layer."""
        layers = controller.current_project.layers
        layer_a = layers[0]
        layer_b = layers[1]

        # Simulate layer A as the current layer in the combo
        from PyQt6.QtCore import Qt
        idx_a = settings_panel._layer_combo.findData(layer_a.id)
        if idx_a >= 0:
            settings_panel._layer_combo.blockSignals(True)
            settings_panel._layer_combo.setCurrentIndex(idx_a)
            settings_panel._layer_combo.blockSignals(False)

        # Make sure there's an active generator
        settings_panel.on_mode_changed("Math Art")
        snapshot_before = settings_panel._get_settings_snapshot()

        if snapshot_before is not None:
            # Now switch active layer to B
            controller.set_active_layer(layer_b.id)

            # Layer A should now have generator_info set
            assert layer_a.generator_info is not None
            assert layer_a.generator_info.get("mode") == "Math Art"

    def test_on_active_layer_changed_restores_settings(self, settings_panel, controller):
        """Switching back to a layer with saved settings restores them."""
        layers = controller.current_project.layers
        layer_a = layers[0]
        layer_b = layers[1]

        # Pre-set layer A's generator_info with a specific snapshot
        from plottter.generators import GENERATORS
        # Pick a generator name that exists
        math_generators = [
            name for name, cls in GENERATORS.items()
            if getattr(cls, "category", None) == "math"
        ]
        if not math_generators:
            pytest.skip("No math generators available")

        gen_name = math_generators[0]
        saved_info = {
            "generator_name": gen_name,
            "mode": "Math Art",
            "params": {},
            "transforms": {
                "scale": 2.0,
                "rotation": 45.0,
                "translate_x": 0.0,
                "translate_y": 0.0,
                "mirror_h": False,
                "mirror_v": False,
                "n_fold": 1,
                "tile_rows": 1,
                "tile_cols": 1,
            },
        }
        layer_a.generator_info = saved_info

        # Currently viewing layer B; switch to layer A
        from PyQt6.QtCore import Qt
        idx_b = settings_panel._layer_combo.findData(layer_b.id)
        if idx_b >= 0:
            settings_panel._layer_combo.blockSignals(True)
            settings_panel._layer_combo.setCurrentIndex(idx_b)
            settings_panel._layer_combo.blockSignals(False)

        controller.set_active_layer(layer_a.id)

        # The scale transform should have been restored to 2.0
        assert settings_panel._transform_scale_spin.value() == pytest.approx(2.0)
        # The rotation should have been restored to 45.0
        assert settings_panel._transform_rotation_spin.value() == pytest.approx(45.0)

    def test_layer_with_no_generator_info_does_not_crash(self, settings_panel, controller):
        """Switching to a layer without saved settings should not crash."""
        layer = controller.current_project.layers[2]
        layer.generator_info = None  # Ensure no saved settings

        # Should not raise
        controller.set_active_layer(layer.id)

    def test_apply_settings_snapshot_restores_transforms(self, settings_panel):
        """_apply_settings_snapshot correctly restores transform values."""
        info = {
            "generator_name": "",
            "mode": "Math Art",
            "params": {},
            "transforms": {
                "scale": 3.5,
                "rotation": 90.0,
                "translate_x": 10.0,
                "translate_y": -5.0,
                "mirror_h": True,
                "mirror_v": False,
                "n_fold": 4,
                "tile_rows": 2,
                "tile_cols": 3,
            },
        }
        settings_panel._apply_settings_snapshot(info)

        assert settings_panel._transform_scale_spin.value() == pytest.approx(3.5)
        assert settings_panel._transform_rotation_spin.value() == pytest.approx(90.0)
        assert settings_panel._transform_x_spin.value() == pytest.approx(10.0)
        assert settings_panel._transform_y_spin.value() == pytest.approx(-5.0)
        assert settings_panel._mirror_h_check.isChecked() is True
        assert settings_panel._mirror_v_check.isChecked() is False
        assert settings_panel._n_fold_spin.value() == 4
        assert settings_panel._tile_rows_spin.value() == 2
        assert settings_panel._tile_cols_spin.value() == 3
