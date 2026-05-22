"""Tests for task 134.1 — three named sub-layouts in the parameter container.

Covers:
(a) _static_params_layout exists and is a QFormLayout within _params_group
(b) _dynamic_params_layout exists and is a QFormLayout within _params_group
(c) _post_proc_layout exists and is a QFormLayout within _post_proc_group
(d) _rebuild_dynamic_params() stub exists and clears the dynamic layout
"""

from __future__ import annotations

import os
import sys

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


@pytest.fixture
def panel(qapp):
    """Create a SettingsPanel with a minimal controller."""
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="Test", canvas=canvas)
    project.add_layer(Layer(name="L1", color="#000000"))
    controller = ProjectController(project)
    return SettingsPanel(controller)


# ---------------------------------------------------------------------------
# test_layout_split_creates_three_sub_layouts
# ---------------------------------------------------------------------------


class TestLayoutSplitCreatesThreeSubLayouts:
    def test_static_params_layout_exists(self, panel):
        from PyQt6.QtWidgets import QFormLayout

        assert hasattr(panel, "_static_params_layout")
        assert isinstance(panel._static_params_layout, QFormLayout)

    def test_dynamic_params_layout_exists(self, panel):
        from PyQt6.QtWidgets import QFormLayout

        assert hasattr(panel, "_dynamic_params_layout")
        assert isinstance(panel._dynamic_params_layout, QFormLayout)

    def test_post_proc_layout_exists(self, panel):
        from PyQt6.QtWidgets import QFormLayout

        assert hasattr(panel, "_post_proc_layout")
        assert isinstance(panel._post_proc_layout, QFormLayout)

    def test_static_and_dynamic_within_params_group(self, panel):
        """Both static and dynamic layouts are nested inside _params_group."""
        params_group = panel._params_group
        # Walk the layout hierarchy: params_group → outer VBoxLayout → sub-layouts
        outer = params_group.layout()
        assert outer is not None, "_params_group should have a layout"

        sub_layouts = []
        for i in range(outer.count()):
            item = outer.itemAt(i)
            if item is not None:
                sub_layouts.append(item.layout())

        assert panel._static_params_layout in sub_layouts
        assert panel._dynamic_params_layout in sub_layouts

    def test_post_proc_layout_within_post_proc_group(self, panel):
        """_post_proc_layout is the layout of _post_proc_group."""
        assert panel._post_proc_group.layout() is panel._post_proc_layout

    def test_dynamic_layout_starts_empty(self, panel):
        """_dynamic_params_layout begins with zero rows (nothing wired yet)."""
        assert panel._dynamic_params_layout.rowCount() == 0

    def test_rebuild_dynamic_params_stub_exists(self, panel):
        """_rebuild_dynamic_params() is callable and clears the dynamic layout."""
        assert callable(getattr(panel, "_rebuild_dynamic_params", None))
        # Should not raise even when layout is already empty
        panel._rebuild_dynamic_params()
        assert panel._dynamic_params_layout.rowCount() == 0


# ---------------------------------------------------------------------------
# test_overrides_merged_into_params (task 134.2)
# ---------------------------------------------------------------------------


class TestDynamicOverridesMergedIntoParams:
    """_dynamic_overrides are passed into params under the reserved key."""

    def test_overrides_merged_into_params(self, panel):
        """When _on_generate fires, params['_dynamic_overrides'] equals the panel's
        stored _dynamic_overrides dict (a shallow copy)."""
        from unittest.mock import MagicMock, patch

        # Minimal stub generator — no special flags set
        mock_gen = MagicMock()
        mock_gen.name = "MockGen"
        mock_gen.emits_multiple_layers = False
        mock_gen.uses_source_image = False
        panel._generator = mock_gen
        panel._current_mode = "Math Art"

        # Put some overrides in the panel state
        panel._dynamic_overrides = {"speed": 7, "density": 0.25}

        captured: dict = {}

        def fake_worker(gen, params, canvas):
            captured["params"] = dict(params)  # snapshot before .start()
            w = MagicMock()
            w.isRunning.return_value = False
            w.is_cancelled.return_value = False
            return w

        with patch(
            "plottter.gui.generator_worker.GeneratorWorker",
            side_effect=fake_worker,
        ):
            panel._on_generate()

        assert "_dynamic_overrides" in captured["params"], (
            "params must contain the '_dynamic_overrides' key"
        )
        assert captured["params"]["_dynamic_overrides"] == {"speed": 7, "density": 0.25}

    def test_empty_overrides_when_no_dynamic_params(self, panel):
        """A generator that provides no dynamic params receives _dynamic_overrides={}."""
        from unittest.mock import MagicMock, patch

        mock_gen = MagicMock()
        mock_gen.name = "StaticGen"
        mock_gen.emits_multiple_layers = False
        mock_gen.uses_source_image = False
        panel._generator = mock_gen
        panel._current_mode = "Math Art"
        panel._dynamic_overrides = {}  # empty — as if get_dynamic_parameters() returned []

        captured: dict = {}

        def fake_worker(gen, params, canvas):
            captured["params"] = dict(params)
            w = MagicMock()
            w.isRunning.return_value = False
            w.is_cancelled.return_value = False
            return w

        with patch(
            "plottter.gui.generator_worker.GeneratorWorker",
            side_effect=fake_worker,
        ):
            panel._on_generate()

        assert captured["params"]["_dynamic_overrides"] == {}
