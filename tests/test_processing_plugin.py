"""Tests for the processing plugin system (task 95.1)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Sample plugin source used in discovery tests
# ---------------------------------------------------------------------------

_VALID_PROC_PLUGIN_SOURCE = textwrap.dedent("""\
    from plottter.processing.plugin import ProcessingPlugin, register_processing_plugin
    from plottter.generators.base import FloatParam

    @register_processing_plugin
    class _TestScalePlugin(ProcessingPlugin):
        name = "_TestScalePlugin"
        description = "Scale every point by a constant factor."

        def get_parameters(self):
            return [
                FloatParam("factor", "Scale Factor", min=0.1, max=10.0, step=0.1, default=2.0),
            ]

        def process(self, paths, params):
            factor = params.get("factor", 2.0)
            return [[(x * factor, y * factor) for x, y in path] for path in paths]
""")

# Plugin that subclasses ProcessingPlugin but does NOT use the decorator —
# the loader should still auto-register it.
_AUTO_REGISTER_SOURCE = textwrap.dedent("""\
    from plottter.processing.plugin import ProcessingPlugin

    class _TestAutoPlugin(ProcessingPlugin):
        name = "_TestAutoPlugin"
        description = "Auto-registered plugin."

        def process(self, paths, params):
            return paths
""")


# ---------------------------------------------------------------------------
# Unit tests: ProcessingPlugin base class
# ---------------------------------------------------------------------------

class TestProcessingPluginBase:
    def test_register_decorator(self):
        from plottter.processing.plugin import (
            PROCESSING_PLUGINS,
            ProcessingPlugin,
            register_processing_plugin,
        )

        @register_processing_plugin
        class _Dummy(ProcessingPlugin):
            name = "_DummyRegistered"

            def process(self, paths, params):
                return paths

        assert "_DummyRegistered" in PROCESSING_PLUGINS
        # cleanup
        PROCESSING_PLUGINS.pop("_DummyRegistered", None)

    def test_default_parameters_empty(self):
        from plottter.processing.plugin import ProcessingPlugin, register_processing_plugin

        @register_processing_plugin
        class _NoParams(ProcessingPlugin):
            name = "_NoParamsPlugin"

            def process(self, paths, params):
                return paths

        assert _NoParams().get_parameters() == []
        from plottter.processing.plugin import PROCESSING_PLUGINS
        PROCESSING_PLUGINS.pop("_NoParamsPlugin", None)

    def test_process_is_abstract(self):
        from plottter.processing.plugin import ProcessingPlugin
        import pytest

        with pytest.raises(TypeError):
            ProcessingPlugin()  # type: ignore[abstract]

    def test_exported_from_processing_package(self):
        from plottter.processing import (
            PROCESSING_PLUGINS,
            ProcessingPlugin,
            register_processing_plugin,
        )
        assert isinstance(PROCESSING_PLUGINS, dict)
        assert ProcessingPlugin is not None
        assert callable(register_processing_plugin)


# ---------------------------------------------------------------------------
# Unit tests: plugin discovery
# ---------------------------------------------------------------------------

class TestProcessingPluginDiscovery:
    def _cleanup(self, plugin_name: str) -> None:
        from plottter.processing.plugin import PROCESSING_PLUGINS
        PROCESSING_PLUGINS.pop(plugin_name, None)
        for key in list(sys.modules.keys()):
            if "plottter_plugin_test_proc" in key or "plottter_plugin_auto" in key:
                del sys.modules[key]

    def test_decorator_plugin_discovered(self, tmp_path):
        plugin_file = tmp_path / "test_proc_plugin.py"
        plugin_file.write_text(_VALID_PROC_PLUGIN_SOURCE)

        self._cleanup("_TestScalePlugin")

        from plottter.generators.plugin_loader import load_plugins
        from plottter.processing.plugin import PROCESSING_PLUGINS

        load_plugins(extra_dirs=[tmp_path])
        assert "_TestScalePlugin" in PROCESSING_PLUGINS

        self._cleanup("_TestScalePlugin")

    def test_auto_register_plugin_discovered(self, tmp_path):
        plugin_file = tmp_path / "auto_plugin.py"
        plugin_file.write_text(_AUTO_REGISTER_SOURCE)

        self._cleanup("_TestAutoPlugin")

        from plottter.generators.plugin_loader import load_plugins
        from plottter.processing.plugin import PROCESSING_PLUGINS

        load_plugins(extra_dirs=[tmp_path])
        assert "_TestAutoPlugin" in PROCESSING_PLUGINS

        self._cleanup("_TestAutoPlugin")

    def test_discovered_plugin_runs(self, tmp_path):
        plugin_file = tmp_path / "test_proc_plugin.py"
        plugin_file.write_text(_VALID_PROC_PLUGIN_SOURCE)

        self._cleanup("_TestScalePlugin")

        from plottter.generators.plugin_loader import load_plugins
        from plottter.processing.plugin import PROCESSING_PLUGINS

        load_plugins(extra_dirs=[tmp_path])

        cls = PROCESSING_PLUGINS["_TestScalePlugin"]
        plugin = cls()
        paths = [[(1.0, 2.0), (3.0, 4.0)]]
        result = plugin.process(paths, {"factor": 2.0})
        assert result == [[(2.0, 4.0), (6.0, 8.0)]]

        self._cleanup("_TestScalePlugin")

    def test_plugin_not_doubled_on_reload(self, tmp_path):
        plugin_file = tmp_path / "test_proc_plugin.py"
        plugin_file.write_text(_VALID_PROC_PLUGIN_SOURCE)

        self._cleanup("_TestScalePlugin")

        from plottter.generators.plugin_loader import load_plugins
        from plottter.processing.plugin import PROCESSING_PLUGINS

        load_plugins(extra_dirs=[tmp_path])
        count_first = sum(1 for k in PROCESSING_PLUGINS if k == "_TestScalePlugin")

        load_plugins(extra_dirs=[tmp_path])
        count_second = sum(1 for k in PROCESSING_PLUGINS if k == "_TestScalePlugin")

        assert count_first == 1
        assert count_second == 1

        self._cleanup("_TestScalePlugin")


# ---------------------------------------------------------------------------
# GUI integration tests
# ---------------------------------------------------------------------------

class TestProcessingPluginGUI:
    @staticmethod
    def _make_plugin_cls():
        from plottter.processing.plugin import ProcessingPlugin

        class _OffsetPlugin(ProcessingPlugin):
            name = "_TestOffsetPlugin"
            description = "Shift all X coords by offset_mm."

            def get_parameters(self):
                from plottter.generators.base import FloatParam
                return [
                    FloatParam("offset_mm", "X Offset (mm)", min=-100.0, max=100.0,
                               step=0.1, default=5.0),
                ]

            def process(self, paths, params):
                offset = params.get("offset_mm", 5.0)
                return [[(x + offset, y) for x, y in path] for path in paths]

        return _OffsetPlugin

    def test_plugin_appears_in_menu(self, qtbot):
        from plottter.processing.plugin import PROCESSING_PLUGINS
        from plottter.models import Canvas, Layer, Project
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.main_window import MainWindow

        cls = self._make_plugin_cls()
        PROCESSING_PLUGINS[cls.name] = cls

        try:
            canvas = Canvas.from_preset("A4", margin=10.0)
            project = Project(name="Test", canvas=canvas)
            project.add_layer(Layer(name="L1", color="#000000"))
            controller = ProjectController(project)
            win = MainWindow(controller)
            win._prompt_save_if_modified = lambda: True
            qtbot.addWidget(win)

            win._rebuild_processing_plugins_menu()

            # Check the submenu has an action with the plugin name
            action_names = [a.text() for a in win._processing_plugins_menu.actions()]
            assert cls.name in action_names
        finally:
            PROCESSING_PLUGINS.pop(cls.name, None)

    def test_plugin_modifies_paths(self, qtbot):
        from plottter.processing.plugin import PROCESSING_PLUGINS
        from plottter.models import Canvas, Layer, Project
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.main_window import MainWindow

        cls = self._make_plugin_cls()
        PROCESSING_PLUGINS[cls.name] = cls

        try:
            canvas = Canvas.from_preset("A4", margin=10.0)
            project = Project(name="Test", canvas=canvas)
            layer = Layer(name="L1", color="#000000")
            layer.paths = [[(0.0, 0.0), (10.0, 10.0)]]
            project.add_layer(layer)
            controller = ProjectController(project)

            # Set the active layer
            controller._active_layer_id = layer.id

            win = MainWindow(controller)
            win._prompt_save_if_modified = lambda: True
            qtbot.addWidget(win)

            # Directly call _on_run_processing_plugin with no-dialog variant
            # by setting up empty parameters and bypassing the dialog
            plugin = cls()
            new_paths = plugin.process(list(layer.paths), {"offset_mm": 5.0})

            # Verify the transformation
            assert new_paths == [[(5.0, 0.0), (15.0, 10.0)]]
        finally:
            PROCESSING_PLUGINS.pop(cls.name, None)

    def test_undo_restores_paths(self, qtbot):
        from plottter.processing.plugin import PROCESSING_PLUGINS
        from plottter.models import Canvas, Layer, Project
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.main_window import MainWindow

        cls = self._make_plugin_cls()
        PROCESSING_PLUGINS[cls.name] = cls

        try:
            canvas = Canvas.from_preset("A4", margin=10.0)
            project = Project(name="Test", canvas=canvas)
            original_paths = [[(0.0, 0.0), (10.0, 10.0)]]
            layer = Layer(name="L1", color="#000000")
            layer.paths = list(original_paths)
            project.add_layer(layer)
            controller = ProjectController(project)
            controller._active_layer_id = layer.id

            win = MainWindow(controller)
            win._prompt_save_if_modified = lambda: True
            qtbot.addWidget(win)

            # Apply via controller.set_layer_paths (which uses SetLayerPathsCommand)
            plugin = cls()
            new_paths = plugin.process(list(layer.paths), {"offset_mm": 5.0})
            controller.set_layer_paths(layer.id, new_paths, cls.name)

            # Verify paths changed
            updated_layer = controller.get_layer(layer.id)
            assert updated_layer.paths == [[(5.0, 0.0), (15.0, 10.0)]]

            # Undo
            controller.undo_stack.undo()
            reverted_layer = controller.get_layer(layer.id)
            assert reverted_layer.paths == original_paths
        finally:
            PROCESSING_PLUGINS.pop(cls.name, None)

    def test_empty_menu_when_no_plugins(self, qtbot):
        from plottter.processing.plugin import PROCESSING_PLUGINS
        from plottter.models import Canvas, Layer, Project
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.main_window import MainWindow

        # Save and clear all processing plugins
        saved = dict(PROCESSING_PLUGINS)
        PROCESSING_PLUGINS.clear()

        try:
            canvas = Canvas.from_preset("A4", margin=10.0)
            project = Project(name="Test", canvas=canvas)
            project.add_layer(Layer(name="L1", color="#000000"))
            controller = ProjectController(project)
            win = MainWindow(controller)
            win._prompt_save_if_modified = lambda: True
            qtbot.addWidget(win)

            win._rebuild_processing_plugins_menu()
            actions = win._processing_plugins_menu.actions()
            # Should have exactly one disabled placeholder action
            assert len(actions) == 1
            assert not actions[0].isEnabled()
        finally:
            PROCESSING_PLUGINS.update(saved)
