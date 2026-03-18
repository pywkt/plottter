"""Tests for the plugin system (Phase 13.8)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VALID_PLUGIN_SOURCE = textwrap.dedent("""\
    from plottter.generators import register_generator
    from plottter.generators.base import FloatParam, Generator, IntParam, Preset
    from plottter.models import Canvas
    import math

    @register_generator
    class _TestPluginCircles(Generator):
        name = "_TestPluginCircles"
        category = "math"

        def get_parameters(self):
            return [
                IntParam("count", "Circle Count", min=1, max=10, step=1, default=3),
            ]

        def get_presets(self):
            return []

        def generate(self, params, canvas, progress_callback=None):
            cx, cy = canvas.width_mm / 2, canvas.height_mm / 2
            count = params.get("count", 3)
            paths = []
            for i in range(1, count + 1):
                r = i * 5.0
                n = 32
                pts = [(cx + r * math.cos(2 * math.pi * k / n),
                        cy + r * math.sin(2 * math.pi * k / n))
                       for k in range(n + 1)]
                paths.append(pts)
            return paths
""")

_BROKEN_PLUGIN_SOURCE = textwrap.dedent("""\
    # This plugin has a syntax error
    def broken(:
        pass
""")

_NON_GENERATOR_PLUGIN_SOURCE = textwrap.dedent("""\
    # Valid Python but no generator class registered
    MY_CONSTANT = 42
""")


class TestLoadPlugins:
    def _cleanup_test_plugin(self):
        """Remove any test plugin from the GENERATORS registry and sys.modules."""
        from plottter.generators import GENERATORS
        GENERATORS.pop("_TestPluginCircles", None)
        for key in list(sys.modules.keys()):
            if "plottter_plugin_test_plugin" in key or "plottter_plugin__test" in key:
                del sys.modules[key]

    def test_load_plugin_from_dir(self, tmp_path):
        """Plugin in a directory is discovered and registers its generator."""
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text(_VALID_PLUGIN_SOURCE)

        self._cleanup_test_plugin()

        from plottter.generators.plugin_loader import load_plugins
        from plottter.generators import GENERATORS

        loaded = load_plugins(extra_dirs=[tmp_path])
        assert "_TestPluginCircles" in loaded
        assert "_TestPluginCircles" in GENERATORS

        self._cleanup_test_plugin()

    def test_broken_plugin_does_not_crash(self, tmp_path):
        """A plugin with a syntax error is skipped; no exception propagates."""
        plugin_file = tmp_path / "bad_plugin.py"
        plugin_file.write_text(_BROKEN_PLUGIN_SOURCE)

        from plottter.generators.plugin_loader import load_plugins
        # Should not raise
        loaded = load_plugins(extra_dirs=[tmp_path])
        assert loaded == []

    def test_non_generator_plugin_returns_empty(self, tmp_path):
        """A valid Python file that does not register any generator returns nothing."""
        plugin_file = tmp_path / "empty_plugin.py"
        plugin_file.write_text(_NON_GENERATOR_PLUGIN_SOURCE)

        from plottter.generators.plugin_loader import load_plugins
        loaded = load_plugins(extra_dirs=[tmp_path])
        assert loaded == []

    def test_underscore_files_skipped(self, tmp_path):
        """Files starting with _ (like __init__.py) are not loaded."""
        (tmp_path / "__init__.py").write_text("# init")
        (tmp_path / "_private.py").write_text(_NON_GENERATOR_PLUGIN_SOURCE)

        from plottter.generators.plugin_loader import load_plugins
        loaded = load_plugins(extra_dirs=[tmp_path])
        assert loaded == []

    def test_nonexistent_dir_is_ignored(self, tmp_path):
        """A directory that does not exist is silently skipped."""
        missing = tmp_path / "does_not_exist"

        from plottter.generators.plugin_loader import load_plugins
        loaded = load_plugins(extra_dirs=[missing])
        assert isinstance(loaded, list)

    def test_already_loaded_plugin_not_doubled(self, tmp_path):
        """Running load_plugins twice for the same file does not double-register."""
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text(_VALID_PLUGIN_SOURCE)

        self._cleanup_test_plugin()

        from plottter.generators.plugin_loader import load_plugins
        from plottter.generators import GENERATORS

        loaded1 = load_plugins(extra_dirs=[tmp_path])
        count_after_first = sum(1 for k in GENERATORS if k == "_TestPluginCircles")

        loaded2 = load_plugins(extra_dirs=[tmp_path])
        count_after_second = sum(1 for k in GENERATORS if k == "_TestPluginCircles")

        assert count_after_first == 1
        assert count_after_second == 1  # no duplication

        self._cleanup_test_plugin()


class TestPluginDirs:
    def test_get_plugin_dirs_returns_list(self):
        from plottter.generators.plugin_loader import get_plugin_dirs
        dirs = get_plugin_dirs()
        assert isinstance(dirs, list)
        for d in dirs:
            assert isinstance(d, Path)
            assert d.exists()

    def test_create_user_plugin_dir_creates_dir(self, tmp_path, monkeypatch):
        """create_user_plugin_dir creates the directory if missing."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

        from plottter.generators import plugin_loader
        # Force reimport of the function to pick up the monkeypatched home
        import importlib
        importlib.reload(plugin_loader)

        result = plugin_loader.create_user_plugin_dir()
        assert result.exists()
        assert result.is_dir()


class TestPluginLoadPublicAPI:
    def test_generators_load_plugins_wrapper(self):
        """generators.load_plugins() delegates to plugin_loader.load_plugins()."""
        from plottter import generators
        result = generators.load_plugins()
        assert isinstance(result, list)
