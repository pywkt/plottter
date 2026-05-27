"""Tests for Map mode wiring (spec §10.1).

Covers:
(a) "Map" is present in ModePanel.MODES
(b) Selecting "Map" mode populates the generator combo with the Map generator
(c) Selecting "Map" mode hides the image source group
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project():
    canvas = Canvas.from_preset("A4", margin=10.0)
    p = Project(name="MapTest", canvas=canvas)
    p.add_layer(Layer(name="Layer 1", color="#000000"))
    return p


@pytest.fixture
def controller(project, qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(project)


@pytest.fixture
def settings_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel
    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    panel.show()
    return panel


# ---------------------------------------------------------------------------
# (a) "Map" is present in ModePanel.MODES
# ---------------------------------------------------------------------------


class TestMapModeEntry:
    def test_map_in_modes_list(self):
        from plottter.gui.mode_panel import ModePanel
        assert "Map" in ModePanel.MODES, "'Map' must be in ModePanel.MODES"

    def test_mode_panel_has_map_radio_button(self, qapp, qtbot):
        from plottter.gui.mode_panel import ModePanel
        panel = ModePanel()
        qtbot.addWidget(panel)
        assert "Map" in panel._radio_buttons, "ModePanel must have a 'Map' radio button"


# ---------------------------------------------------------------------------
# (b) Selecting "Map" lists the Map generator
# ---------------------------------------------------------------------------


class TestMapGeneratorListing:
    def test_map_mode_lists_map_generator(self, settings_panel):
        panel = settings_panel
        panel.on_mode_changed("Map")

        items = [
            panel._generator_type_combo.itemText(i)
            for i in range(panel._generator_type_combo.count())
        ]
        assert "Map" in items, (
            f"Generator combo should contain 'Map' when in Map mode; got: {items}"
        )

    def test_map_mode_generator_combo_visible(self, settings_panel):
        panel = settings_panel
        panel.on_mode_changed("Map")
        assert panel._generator_type_group.isVisible(), (
            "_generator_type_group must be visible in Map mode"
        )

    def test_map_mode_only_lists_map_generators(self, settings_panel):
        """Only generators with category='map' appear in Map mode."""
        from plottter.generators import get_generators_by_category
        panel = settings_panel
        panel.on_mode_changed("Map")

        expected = {cls.name for cls in get_generators_by_category("map")}
        actual = {
            panel._generator_type_combo.itemText(i)
            for i in range(panel._generator_type_combo.count())
        }
        assert actual == expected, (
            f"Generator combo should contain exactly map generators; "
            f"expected {expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# (c) Selecting "Map" hides the image source group
# ---------------------------------------------------------------------------


class TestMapModeGroupVisibility:
    def test_image_source_group_hidden_in_map_mode(self, settings_panel):
        panel = settings_panel
        panel.on_mode_changed("Map")
        assert not panel._image_source_group.isVisible(), (
            "_image_source_group must be hidden in Map mode"
        )

    def test_preprocessing_group_hidden_in_map_mode(self, settings_panel):
        panel = settings_panel
        panel.on_mode_changed("Map")
        assert not panel._preprocessing_group.isVisible(), (
            "_preprocessing_group must be hidden in Map mode"
        )

    def test_3d_camera_group_hidden_in_map_mode(self, settings_panel):
        panel = settings_panel
        panel.on_mode_changed("Map")
        assert not panel._3d_camera_group.isVisible(), (
            "_3d_camera_group must be hidden in Map mode"
        )

    def test_color_sep_group_hidden_in_map_mode(self, settings_panel):
        panel = settings_panel
        panel.on_mode_changed("Map")
        assert not panel._color_sep_group.isVisible(), (
            "_color_sep_group must be hidden in Map mode"
        )

    def test_image_source_shown_again_after_leaving_map_mode(self, settings_panel):
        """Switching from Map → Image to Lines restores the image source group."""
        panel = settings_panel
        panel.on_mode_changed("Map")
        assert not panel._image_source_group.isVisible()
        panel.on_mode_changed("Image to Lines")
        assert panel._image_source_group.isVisible(), (
            "_image_source_group must be visible after switching to Image to Lines"
        )
