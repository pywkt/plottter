"""Tests for the ExportPlugin system (task 95.2)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_EXPORT_PLUGIN_SOURCE = textwrap.dedent("""\
    from plottter.export.plugin import ExportPlugin, register_export_plugin

    @register_export_plugin
    class _TestCSVExportPlugin(ExportPlugin):
        name = "_TestCSVExport"
        file_extension = ".csv"
        description = "Test CSV export plugin."

        def export(self, paths_by_layer, canvas, file_path):
            with open(file_path, "w") as f:
                f.write("layer,color,x_mm,y_mm\\n")
                for layer_name, hex_color, paths in paths_by_layer:
                    for path in paths:
                        for x, y in path:
                            f.write(f"{layer_name},{hex_color},{x:.4f},{y:.4f}\\n")
""")

_AUTO_REGISTER_PLUGIN_SOURCE = textwrap.dedent("""\
    from plottter.export.plugin import ExportPlugin

    # No decorator — relies on auto-registration in plugin_loader
    class _TestAutoExportPlugin(ExportPlugin):
        name = "_TestAutoExport"
        file_extension = ".txt"
        description = "Auto-registered export plugin."

        def export(self, paths_by_layer, canvas, file_path):
            with open(file_path, "w") as f:
                f.write("auto\\n")
""")


def _cleanup_export_plugin(name: str) -> None:
    """Remove a test export plugin from the EXPORT_PLUGINS registry."""
    from plottter.export.plugin import EXPORT_PLUGINS
    EXPORT_PLUGINS.pop(name, None)
    for key in list(sys.modules.keys()):
        if f"plottter_plugin_" in key and name.lower().replace(" ", "_") in key.lower():
            del sys.modules[key]


# ---------------------------------------------------------------------------
# ExportPlugin ABC
# ---------------------------------------------------------------------------


class TestExportPluginABC:
    def test_export_plugin_cannot_instantiate_directly(self):
        """ExportPlugin is abstract and cannot be instantiated."""
        from plottter.export.plugin import ExportPlugin
        try:
            ExportPlugin()
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_register_decorator(self):
        """register_export_plugin adds the class to EXPORT_PLUGINS."""
        from plottter.export.plugin import EXPORT_PLUGINS, ExportPlugin, register_export_plugin

        @register_export_plugin
        class _RegTestPlugin(ExportPlugin):
            name = "_RegTestPlugin"
            file_extension = ".tst"
            description = "Test."

            def export(self, paths_by_layer, canvas, file_path):
                pass

        assert "_RegTestPlugin" in EXPORT_PLUGINS
        assert EXPORT_PLUGINS["_RegTestPlugin"] is _RegTestPlugin
        # Cleanup
        EXPORT_PLUGINS.pop("_RegTestPlugin", None)

    def test_export_plugin_attrs(self):
        """ExportPlugin subclass exposes name, file_extension, description."""
        from plottter.export.plugin import ExportPlugin, register_export_plugin

        @register_export_plugin
        class _AttrPlugin(ExportPlugin):
            name = "_AttrPlugin"
            file_extension = ".xyz"
            description = "Attr test."

            def export(self, paths_by_layer, canvas, file_path):
                pass

        inst = _AttrPlugin()
        assert inst.name == "_AttrPlugin"
        assert inst.file_extension == ".xyz"
        assert inst.description == "Attr test."
        from plottter.export.plugin import EXPORT_PLUGINS
        EXPORT_PLUGINS.pop("_AttrPlugin", None)


# ---------------------------------------------------------------------------
# ExportPlugin re-exported from plottter.export
# ---------------------------------------------------------------------------


class TestExportPackageExports:
    def test_export_plugins_importable_from_package(self):
        """EXPORT_PLUGINS and ExportPlugin are importable from plottter.export."""
        from plottter.export import EXPORT_PLUGINS, ExportPlugin, register_export_plugin
        assert isinstance(EXPORT_PLUGINS, dict)
        assert ExportPlugin is not None
        assert callable(register_export_plugin)


# ---------------------------------------------------------------------------
# Plugin loader discovers ExportPlugin subclasses
# ---------------------------------------------------------------------------


class TestPluginLoaderExportDiscovery:
    def _cleanup(self, name: str) -> None:
        _cleanup_export_plugin(name)
        for key in list(sys.modules.keys()):
            if "plottter_plugin_test_export" in key:
                del sys.modules[key]

    def test_load_export_plugin_from_dir(self, tmp_path):
        """A plugin file with an ExportPlugin subclass is auto-registered."""
        plugin_file = tmp_path / "test_export_plugin.py"
        plugin_file.write_text(_SAMPLE_EXPORT_PLUGIN_SOURCE)

        self._cleanup("_TestCSVExport")

        from plottter.generators.plugin_loader import load_plugins
        from plottter.export.plugin import EXPORT_PLUGINS

        loaded = load_plugins(extra_dirs=[tmp_path])
        assert "_TestCSVExport" in loaded
        assert "_TestCSVExport" in EXPORT_PLUGINS

        self._cleanup("_TestCSVExport")

    def test_auto_register_without_decorator(self, tmp_path):
        """ExportPlugin subclass with non-empty name is auto-registered even without decorator."""
        plugin_file = tmp_path / "test_auto_export.py"
        plugin_file.write_text(_AUTO_REGISTER_PLUGIN_SOURCE)

        self._cleanup("_TestAutoExport")
        for key in list(sys.modules.keys()):
            if "plottter_plugin_test_auto_export" in key:
                del sys.modules[key]

        from plottter.generators.plugin_loader import load_plugins
        from plottter.export.plugin import EXPORT_PLUGINS

        loaded = load_plugins(extra_dirs=[tmp_path])
        assert "_TestAutoExport" in loaded
        assert "_TestAutoExport" in EXPORT_PLUGINS

        self._cleanup("_TestAutoExport")
        for key in list(sys.modules.keys()):
            if "plottter_plugin_test_auto_export" in key:
                del sys.modules[key]


# ---------------------------------------------------------------------------
# Export dialog includes plugin in format dropdown
# ---------------------------------------------------------------------------


class TestExportDialogWithPlugin:
    def test_plugin_appears_in_format_combo(self, qapp):
        """A registered export plugin appears in the ExportDialog format combo."""
        from plottter.export.plugin import EXPORT_PLUGINS, ExportPlugin, register_export_plugin

        @register_export_plugin
        class _DialogTestPlugin(ExportPlugin):
            name = "_DialogTestPlugin"
            file_extension = ".tst"
            description = "Dialog test plugin."

            def export(self, paths_by_layer, canvas, file_path):
                pass

        try:
            from plottter.gui.dialogs.export import ExportDialog
            dialog = ExportDialog()
            items = [
                dialog._format_combo.itemText(i)
                for i in range(dialog._format_combo.count())
            ]
            assert "_DialogTestPlugin" in items
        finally:
            EXPORT_PLUGINS.pop("_DialogTestPlugin", None)

    def test_plugin_format_shows_description(self, qapp):
        """Selecting a plugin format shows its description on the info page."""
        from plottter.export.plugin import EXPORT_PLUGINS, ExportPlugin, register_export_plugin

        @register_export_plugin
        class _DescTestPlugin(ExportPlugin):
            name = "_DescTestPlugin"
            file_extension = ".desc"
            description = "Hello from test plugin."

            def export(self, paths_by_layer, canvas, file_path):
                pass

        try:
            from plottter.gui.dialogs.export import ExportDialog
            dialog = ExportDialog()
            idx = dialog._format_combo.findText("_DescTestPlugin")
            assert idx >= 0
            dialog._format_combo.setCurrentIndex(idx)
            # Stack should be on index 4 (plugin info page)
            assert dialog._format_stack.currentIndex() == 4
            assert "Hello from test plugin." in dialog._plugin_desc_label.text()
        finally:
            EXPORT_PLUGINS.pop("_DescTestPlugin", None)

    def test_plugin_get_settings_returns_format_name(self, qapp):
        """get_settings() returns the plugin's name as the format key."""
        from plottter.export.plugin import EXPORT_PLUGINS, ExportPlugin, register_export_plugin

        @register_export_plugin
        class _SettingsTestPlugin(ExportPlugin):
            name = "_SettingsTestPlugin"
            file_extension = ".st"
            description = ""

            def export(self, paths_by_layer, canvas, file_path):
                pass

        try:
            from plottter.gui.dialogs.export import ExportDialog
            dialog = ExportDialog()
            idx = dialog._format_combo.findText("_SettingsTestPlugin")
            assert idx >= 0
            dialog._format_combo.setCurrentIndex(idx)
            dialog._path_edit.setText("/tmp/out.st")
            settings = dialog.get_settings()
            assert settings["format"] == "_SettingsTestPlugin"
        finally:
            EXPORT_PLUGINS.pop("_SettingsTestPlugin", None)


# ---------------------------------------------------------------------------
# ExportPlugin.export() is called and writes correctly
# ---------------------------------------------------------------------------


class TestExportPluginExecution:
    def test_export_writes_file(self, tmp_path):
        """A sample export plugin writes the expected content to a file."""
        from plottter.export.plugin import ExportPlugin, register_export_plugin
        from plottter.models.canvas import Canvas

        @register_export_plugin
        class _WriteTestPlugin(ExportPlugin):
            name = "_WriteTestPlugin"
            file_extension = ".csv"
            description = ""

            def export(self, paths_by_layer, canvas, file_path):
                with open(file_path, "w") as f:
                    f.write("layer,color,x_mm,y_mm\n")
                    for layer_name, hex_color, paths in paths_by_layer:
                        for path in paths:
                            for x, y in path:
                                f.write(f"{layer_name},{hex_color},{x:.4f},{y:.4f}\n")

        canvas = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)
        paths_by_layer = [
            ("Red", "#ff0000", [[(10.0, 20.0), (30.0, 40.0)]]),
        ]
        out_file = tmp_path / "output.csv"
        plugin = _WriteTestPlugin()
        plugin.export(paths_by_layer, canvas, str(out_file))

        assert out_file.exists()
        content = out_file.read_text()
        assert "layer,color,x_mm,y_mm" in content
        assert "Red,#ff0000,10.0000,20.0000" in content
        assert "Red,#ff0000,30.0000,40.0000" in content

        from plottter.export.plugin import EXPORT_PLUGINS
        EXPORT_PLUGINS.pop("_WriteTestPlugin", None)

    def test_export_plugin_via_plugin_file(self, tmp_path):
        """A plugin loaded from a file writes the correct output."""
        plugin_file = tmp_path / "test_csv_export.py"
        plugin_file.write_text(_SAMPLE_EXPORT_PLUGIN_SOURCE)

        # Cleanup from previous runs
        _cleanup_export_plugin("_TestCSVExport")
        for key in list(sys.modules.keys()):
            if "plottter_plugin_test_csv_export" in key:
                del sys.modules[key]

        from plottter.generators.plugin_loader import load_plugins
        load_plugins(extra_dirs=[tmp_path])

        from plottter.export.plugin import EXPORT_PLUGINS
        assert "_TestCSVExport" in EXPORT_PLUGINS

        from plottter.models.canvas import Canvas
        plugin_cls = EXPORT_PLUGINS["_TestCSVExport"]
        plugin = plugin_cls()
        canvas = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)
        paths_by_layer = [("Blue", "#0000ff", [[(5.0, 6.0)]])]
        out_file = tmp_path / "result.csv"
        plugin.export(paths_by_layer, canvas, str(out_file))

        assert out_file.exists()
        content = out_file.read_text()
        assert "Blue,#0000ff,5.0000,6.0000" in content

        _cleanup_export_plugin("_TestCSVExport")
        for key in list(sys.modules.keys()):
            if "plottter_plugin_test_csv_export" in key:
                del sys.modules[key]


# ---------------------------------------------------------------------------
# Extension helpers in ExportDialog
# ---------------------------------------------------------------------------


class TestExportDialogExtensionHelper:
    def test_ensure_extension_plugin_appends_ext(self, qapp):
        """_ensure_extension appends plugin extension when path has none."""
        from plottter.export.plugin import EXPORT_PLUGINS, ExportPlugin, register_export_plugin

        @register_export_plugin
        class _ExtPlugin(ExportPlugin):
            name = "_ExtPlugin"
            file_extension = ".xyz"
            description = ""

            def export(self, paths_by_layer, canvas, file_path):
                pass

        try:
            from plottter.gui.dialogs.export import ExportDialog
            dialog = ExportDialog()
            result = dialog._ensure_extension("/tmp/myfile", "_ExtPlugin")
            assert result == "/tmp/myfile.xyz"
        finally:
            EXPORT_PLUGINS.pop("_ExtPlugin", None)

    def test_ensure_extension_plugin_keeps_correct_ext(self, qapp):
        """_ensure_extension does not double-append correct extension."""
        from plottter.export.plugin import EXPORT_PLUGINS, ExportPlugin, register_export_plugin

        @register_export_plugin
        class _ExtPlugin2(ExportPlugin):
            name = "_ExtPlugin2"
            file_extension = ".xyz"
            description = ""

            def export(self, paths_by_layer, canvas, file_path):
                pass

        try:
            from plottter.gui.dialogs.export import ExportDialog
            dialog = ExportDialog()
            result = dialog._ensure_extension("/tmp/myfile.xyz", "_ExtPlugin2")
            assert result == "/tmp/myfile.xyz"
        finally:
            EXPORT_PLUGINS.pop("_ExtPlugin2", None)
