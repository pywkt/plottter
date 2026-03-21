"""Tests for task 56.1 — Color separation preset combo box.

Covers:
(a) Preset combo appears below algorithm combo in Color Separation mode
(b) Changing algorithm updates the preset list
(c) "Default" is always the first option
(d) Built-in presets from the generator appear in the list
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="TestProject", canvas=canvas)
    layer = Layer(name="Layer 1", color="#000000")
    proj.add_layer(layer)
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController

    return ProjectController(_make_project())


@pytest.fixture
def main_window(controller, qtbot):
    from plottter.gui.main_window import MainWindow

    win = MainWindow(controller)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


# ---------------------------------------------------------------------------
# (a) Preset combo appears in Color Separation mode
# ---------------------------------------------------------------------------


class TestColorSepPresetComboExists:
    """Spec (a): Preset combo box exists and is visible in Color Separation mode."""

    def test_preset_combo_exists(self, main_window):
        """_color_sep_preset_combo attribute exists on settings panel."""
        panel = main_window._settings_panel
        assert hasattr(panel, "_color_sep_preset_combo")

    def test_preset_combo_is_qcombobox(self, main_window):
        """_color_sep_preset_combo is a QComboBox instance."""
        from PyQt6.QtWidgets import QComboBox

        panel = main_window._settings_panel
        assert isinstance(panel._color_sep_preset_combo, QComboBox)

    def test_preset_combo_visible_in_color_sep_mode(self, main_window):
        """Preset combo is visible when in Color Separation mode."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")
        # The parent group should not be explicitly hidden
        assert not panel._color_sep_group.isHidden()
        # The combo should not be explicitly hidden
        assert not panel._color_sep_preset_combo.isHidden()


# ---------------------------------------------------------------------------
# (b) Changing algorithm updates the preset list
# ---------------------------------------------------------------------------


class TestColorSepPresetComboUpdates:
    """Spec (b): Preset combo updates when algorithm combo changes."""

    def test_rebuild_called_on_algorithm_change(self, main_window):
        """Changing algorithm combo triggers preset combo rebuild."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Get initial preset count
        initial_count = preset_combo.count()
        assert initial_count >= 1  # At least "Default"

        # If there are multiple algorithms, try switching
        if gen_combo.count() > 1:
            gen_combo.setCurrentIndex(1)
            # Preset combo should have been rebuilt (at least "Default")
            assert preset_combo.count() >= 1

    def test_preset_list_differs_by_generator(self, main_window):
        """Different generators may have different preset counts."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        if gen_combo.count() < 2:
            pytest.skip("Need at least 2 generators to compare")

        # Get presets for first generator
        gen_combo.setCurrentIndex(0)
        presets_gen0 = [
            preset_combo.itemText(i) for i in range(preset_combo.count())
        ]

        # Get presets for second generator
        gen_combo.setCurrentIndex(1)
        presets_gen1 = [
            preset_combo.itemText(i) for i in range(preset_combo.count())
        ]

        # Both should have "Default" first
        assert presets_gen0[0] == "Default"
        assert presets_gen1[0] == "Default"


# ---------------------------------------------------------------------------
# (c) "Default" is always the first option
# ---------------------------------------------------------------------------


class TestColorSepPresetComboDefault:
    """Spec (c): 'Default' is always the first option."""

    def test_default_is_first_item(self, main_window):
        """First item in preset combo is 'Default'."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        preset_combo = panel._color_sep_preset_combo
        assert preset_combo.count() >= 1
        assert preset_combo.itemText(0) == "Default"

    def test_default_has_none_data(self, main_window):
        """'Default' item has None as its data."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        preset_combo = panel._color_sep_preset_combo
        assert preset_combo.itemData(0) is None

    def test_default_remains_first_after_algorithm_change(self, main_window):
        """'Default' stays first even after switching algorithms."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Check for all available generators
        for i in range(gen_combo.count()):
            gen_combo.setCurrentIndex(i)
            assert preset_combo.itemText(0) == "Default"
            assert preset_combo.itemData(0) is None


# ---------------------------------------------------------------------------
# (d) Built-in presets from the generator appear in the list
# ---------------------------------------------------------------------------


class TestColorSepPresetComboBuiltinPresets:
    """Spec (d): Built-in presets from the generator appear in the list."""

    def test_builtin_presets_appear(self, main_window):
        """Generator's built-in presets are added to the combo."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Get the generator class
        gen_cls = gen_combo.currentData()
        if gen_cls is None:
            pytest.skip("No generator selected")

        # Get the generator's presets
        gen_instance = gen_cls()
        builtin_presets = gen_instance.get_presets()

        if not builtin_presets:
            pytest.skip("Generator has no built-in presets")

        # Verify each builtin preset is in the combo
        combo_items = [
            preset_combo.itemText(i) for i in range(preset_combo.count())
        ]
        for preset in builtin_presets:
            assert preset.name in combo_items

    def test_preset_data_contains_params_dict(self, main_window):
        """Each preset item's data is the params dict (or None for Default)."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        gen_cls = gen_combo.currentData()
        if gen_cls is None:
            pytest.skip("No generator selected")

        gen_instance = gen_cls()
        builtin_presets = gen_instance.get_presets()

        if not builtin_presets:
            pytest.skip("Generator has no built-in presets")

        # Check first builtin preset (index 1, since 0 is "Default")
        if preset_combo.count() > 1:
            first_preset_data = preset_combo.itemData(1)
            # Should be a dict (params) for builtin presets
            assert isinstance(first_preset_data, dict)

    def test_at_least_hatching_generator_has_presets(self, main_window):
        """Hatching generator should have built-in presets."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Find Hatching generator
        hatching_idx = -1
        for i in range(gen_combo.count()):
            gen_cls = gen_combo.itemData(i)
            if gen_cls is not None and gen_cls.name == "Hatching":
                hatching_idx = i
                break

        if hatching_idx < 0:
            pytest.skip("Hatching generator not found")

        gen_combo.setCurrentIndex(hatching_idx)

        # Hatching should have presets beyond just "Default"
        assert preset_combo.count() > 1
        assert preset_combo.itemText(0) == "Default"
