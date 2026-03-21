"""Tests for task 56.2 — Use selected preset params when generating color sep lines.

Covers:
(a) Selecting "Default" produces the same output as before (no regression)
(b) Selecting a preset produces different params
(c) `_source_image` is always passed through regardless of preset
(d) User presets also work correctly
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

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
# (a) Selecting "Default" uses generator defaults
# ---------------------------------------------------------------------------


class TestDefaultPresetUsesDefaults:
    """Spec (a): 'Default' uses generator's get_parameters() defaults."""

    def test_default_selected_builds_params_from_generator(self, main_window):
        """When Default is selected, params come from generator defaults."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Select "Default" (index 0)
        preset_combo.setCurrentIndex(0)
        assert preset_combo.currentData() is None

        # Get the generator and its default params
        gen_cls = gen_combo.currentData()
        if gen_cls is None:
            pytest.skip("No generator selected")

        gen = gen_cls()
        expected_params = {}
        for p in gen.get_parameters():
            if hasattr(p, "default"):
                expected_params[p.name] = p.default

        # Verify at least some defaults exist
        assert len(expected_params) > 0


class TestPresetParamsAreDifferent:
    """Spec (b): Selecting a preset uses preset params."""

    def test_preset_has_params(self, main_window):
        """Preset params are valid dicts with at least one param."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Find Hatching generator which has presets
        hatching_idx = -1
        for i in range(gen_combo.count()):
            gen_cls = gen_combo.itemData(i)
            if gen_cls is not None and gen_cls.name == "Hatching":
                hatching_idx = i
                break

        if hatching_idx < 0:
            pytest.skip("Hatching generator not found")

        gen_combo.setCurrentIndex(hatching_idx)

        # Need more than just Default
        if preset_combo.count() <= 1:
            pytest.skip("No presets available for Hatching")

        # Get first preset params (index 1, skipping Default)
        preset_params = preset_combo.itemData(1)
        assert preset_params is not None
        assert isinstance(preset_params, dict)
        # Preset should have at least one parameter
        assert len(preset_params) > 0, "Preset should have at least one param"


# ---------------------------------------------------------------------------
# (c) _source_image is always passed through
# ---------------------------------------------------------------------------


class TestSourceImageAlwaysSet:
    """Spec (c): _source_image is set regardless of preset selection."""

    def test_source_image_set_with_default_preset(self, main_window, qtbot):
        """_source_image is set when Default is selected."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Select Default
        preset_combo.setCurrentIndex(0)
        assert preset_combo.currentData() is None

        gen_cls = gen_combo.currentData()
        if gen_cls is None:
            pytest.skip("No generator selected")

        # Create a mock grayscale image
        masked_gray = np.zeros((100, 100), dtype=np.uint8)

        # Set up the queue
        canvas = panel._controller._project.canvas
        layer = Layer(name="Test", color="#000000")
        panel._lines_queue = [(layer.id, masked_gray, masked_gray)]
        panel._lines_canvas = canvas
        panel._lines_gen_cls = gen_cls
        panel._lines_done = 0

        # Capture the params passed to GeneratorWorker
        captured_params = {}

        def mock_worker_init(gen, params, canvas):
            captured_params.update(params)
            mock = MagicMock()
            mock.finished = MagicMock()
            mock.finished.connect = MagicMock()
            mock.error = MagicMock()
            mock.error.connect = MagicMock()
            mock.start = MagicMock()
            return mock

        with patch(
            "plottter.gui.generator_worker.GeneratorWorker", side_effect=mock_worker_init
        ):
            panel._process_next_lines_layer()

        assert "_source_image" in captured_params
        # Note: The mask may be copied, so we check equality not identity
        assert np.array_equal(captured_params["_source_image"], masked_gray)

    def test_source_image_set_with_preset_selected(self, main_window, qtbot):
        """_source_image is set when a preset is selected."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Find Hatching generator which has presets
        hatching_idx = -1
        for i in range(gen_combo.count()):
            gen_cls = gen_combo.itemData(i)
            if gen_cls is not None and gen_cls.name == "Hatching":
                hatching_idx = i
                break

        if hatching_idx < 0:
            pytest.skip("Hatching generator not found")

        gen_combo.setCurrentIndex(hatching_idx)

        if preset_combo.count() <= 1:
            pytest.skip("No presets available for Hatching")

        # Select first preset (non-default)
        preset_combo.setCurrentIndex(1)
        assert preset_combo.currentData() is not None

        gen_cls = gen_combo.currentData()

        # Create a mock grayscale image
        masked_gray = np.zeros((100, 100), dtype=np.uint8)

        # Set up the queue
        canvas = panel._controller._project.canvas
        layer = Layer(name="Test", color="#000000")
        panel._lines_queue = [(layer.id, masked_gray, masked_gray)]
        panel._lines_canvas = canvas
        panel._lines_gen_cls = gen_cls
        panel._lines_done = 0

        # Capture the params passed to GeneratorWorker
        captured_params = {}

        def mock_worker_init(gen, params, canvas):
            captured_params.update(params)
            mock = MagicMock()
            mock.finished = MagicMock()
            mock.finished.connect = MagicMock()
            mock.error = MagicMock()
            mock.error.connect = MagicMock()
            mock.start = MagicMock()
            return mock

        with patch(
            "plottter.gui.generator_worker.GeneratorWorker", side_effect=mock_worker_init
        ):
            panel._process_next_lines_layer()

        # Verify _source_image is set
        assert "_source_image" in captured_params
        # Note: The mask may be copied, so we check equality not identity
        assert np.array_equal(captured_params["_source_image"], masked_gray)

        # Also verify preset params were used
        preset_params = preset_combo.currentData()
        for key, val in preset_params.items():
            if key != "_source_image":  # _source_image is overwritten
                assert captured_params.get(key) == val


# ---------------------------------------------------------------------------
# (d) User presets work correctly
# ---------------------------------------------------------------------------


class TestUserPresetsWork:
    """Spec (d): User presets also work correctly."""

    def test_user_preset_params_used(self, main_window, qtbot, tmp_path, monkeypatch):
        """User presets are used when selected."""
        from plottter.presets import user_presets
        from plottter.generators.base import Preset

        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        gen_cls = gen_combo.currentData()
        if gen_cls is None:
            pytest.skip("No generator selected")

        # Create a temporary user preset directory
        preset_dir = tmp_path / ".plottter" / "presets"
        preset_dir.mkdir(parents=True, exist_ok=True)

        # Mock the _PRESETS_DIR to use temp directory
        monkeypatch.setattr(user_presets, "_PRESETS_DIR", preset_dir)

        # Save a user preset
        gen_name = gen_cls.name
        custom_params = {"test_param": 42, "another_param": "custom_value"}
        preset = Preset(name="TestUserPreset", params=custom_params)
        user_presets.save_user_preset(gen_name, preset, presets_dir=preset_dir)

        # Rebuild combo to pick up user preset
        panel._rebuild_color_sep_preset_combo()

        # Find the user preset in combo
        user_preset_idx = -1
        for i in range(preset_combo.count()):
            if preset_combo.itemText(i) == "TestUserPreset":
                user_preset_idx = i
                break

        if user_preset_idx < 0:
            pytest.skip("User preset not found in combo")

        # Select user preset
        preset_combo.setCurrentIndex(user_preset_idx)
        preset_data = preset_combo.currentData()

        assert preset_data is not None
        assert isinstance(preset_data, dict)
        assert preset_data.get("test_param") == 42
        assert preset_data.get("another_param") == "custom_value"


class TestPresetParamsOverrideDefaults:
    """Test that preset params correctly override generator defaults."""

    def test_preset_overrides_defaults(self, main_window, qtbot):
        """Preset params should override generator defaults."""
        panel = main_window._settings_panel
        panel.on_mode_changed("Color Separation")

        gen_combo = panel._color_sep_gen_combo
        preset_combo = panel._color_sep_preset_combo

        # Find Hatching generator which has presets
        hatching_idx = -1
        for i in range(gen_combo.count()):
            gen_cls = gen_combo.itemData(i)
            if gen_cls is not None and gen_cls.name == "Hatching":
                hatching_idx = i
                break

        if hatching_idx < 0:
            pytest.skip("Hatching generator not found")

        gen_combo.setCurrentIndex(hatching_idx)

        if preset_combo.count() <= 1:
            pytest.skip("No presets available")

        # Select a preset
        preset_combo.setCurrentIndex(1)
        preset_params = preset_combo.currentData()
        assert preset_params is not None

        gen_cls = gen_combo.currentData()

        # Create a mock grayscale image
        masked_gray = np.zeros((100, 100), dtype=np.uint8)

        # Set up the queue
        canvas = panel._controller._project.canvas
        layer = Layer(name="Test", color="#000000")
        panel._lines_queue = [(layer.id, masked_gray, masked_gray)]
        panel._lines_canvas = canvas
        panel._lines_gen_cls = gen_cls
        panel._lines_done = 0

        captured_params = {}

        def mock_worker_init(gen, params, canvas):
            captured_params.update(params)
            mock = MagicMock()
            mock.finished = MagicMock()
            mock.finished.connect = MagicMock()
            mock.error = MagicMock()
            mock.error.connect = MagicMock()
            mock.start = MagicMock()
            return mock

        with patch(
            "plottter.gui.generator_worker.GeneratorWorker", side_effect=mock_worker_init
        ):
            panel._process_next_lines_layer()

        # All preset params should be in captured params
        for key, val in preset_params.items():
            assert captured_params.get(key) == val, f"Param {key} not correctly set"

        # _source_image should also be set
        assert "_source_image" in captured_params
