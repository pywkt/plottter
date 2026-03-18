"""Plugin system for loading custom generators from a plugins directory.

Plottter scans a ``plugins/`` directory (relative to the project root or the
user's home config directory) for Python modules.  Any module that defines a
class inheriting from :class:`plottter.generators.base.Generator` and
decorated with :func:`plottter.generators.register_generator` is automatically
discovered and added to the :data:`plottter.generators.GENERATORS` registry.

Plugins can be placed in any of the following locations (searched in order):

1. ``~/.config/plottter/plugins/`` (user-level plugins)
2. ``<current_working_directory>/plugins/`` (project-level plugins)

Each plugin file must be a valid Python module (``*.py``) at the top level of
the plugins directory.  Sub-packages are not scanned.

Example plugin (``~/.config/plottter/plugins/my_circles.py``)::

    from plottter.generators import register_generator
    from plottter.generators.base import FloatParam, Generator, IntParam, Preset
    from plottter.models import Canvas, Polyline
    import math

    @register_generator
    class ConcentricCirclesGenerator(Generator):
        name = "Concentric Circles"
        category = "math"

        def get_parameters(self):
            return [
                IntParam("count", "Circle Count", min=1, max=100, step=1, default=10),
                FloatParam("spacing_mm", "Spacing (mm)", min=0.5, max=50.0,
                           step=0.5, default=5.0),
            ]

        def get_presets(self):
            return []

        def generate(self, params, canvas, progress_callback=None):
            cx, cy = canvas.width_mm / 2, canvas.height_mm / 2
            count = params.get("count", 10)
            spacing = params.get("spacing_mm", 5.0)
            paths = []
            for i in range(1, count + 1):
                r = i * spacing
                n = max(64, int(2 * math.pi * r / 0.5))
                pts = [(cx + r * math.cos(2 * math.pi * k / n),
                        cy + r * math.sin(2 * math.pi * k / n))
                       for k in range(n + 1)]
                paths.append(pts)
            return paths
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_plugin_dirs() -> list[Path]:
    """Return the list of directories to search for plugins, in priority order."""
    dirs: list[Path] = []

    # 1. User-level config directory
    home_config = Path.home() / ".config" / "plottter" / "plugins"
    dirs.append(home_config)

    # 2. Current working directory / plugins
    cwd_plugins = Path.cwd() / "plugins"
    dirs.append(cwd_plugins)

    return dirs


def load_plugins(extra_dirs: list[str | Path] | None = None) -> list[str]:
    """Scan plugin directories and load any valid generator plugins.

    Parameters
    ----------
    extra_dirs:
        Additional directories to scan (e.g. from a project-level setting).

    Returns
    -------
    list[str]
        Names of generator classes that were successfully loaded.
    """
    from plottter.generators import GENERATORS

    scan_dirs: list[Path] = _get_plugin_dirs()
    if extra_dirs:
        scan_dirs.extend(Path(d) for d in extra_dirs)

    loaded_names: list[str] = []
    before_names = set(GENERATORS.keys())

    for plugin_dir in scan_dirs:
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            continue

        for py_file in sorted(plugin_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue  # Skip __init__.py and private modules

            module_name = f"plottter_plugin_{py_file.stem}"
            if module_name in sys.modules:
                # Already loaded in this session
                continue

            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    logger.warning("Could not create module spec for %s", py_file)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                logger.info("Loaded plugin: %s from %s", module_name, py_file)
            except Exception as exc:
                logger.error("Failed to load plugin %s: %s", py_file, exc)
                # Remove from sys.modules if partially loaded
                sys.modules.pop(module_name, None)
                continue

    # Report which generators were newly registered
    after_names = set(GENERATORS.keys())
    new_names = sorted(after_names - before_names)
    loaded_names.extend(new_names)

    if new_names:
        logger.info("Plugin system registered generators: %s", new_names)

    return loaded_names


def get_plugin_dirs() -> list[Path]:
    """Return the currently active plugin directories (for display in settings)."""
    return [d for d in _get_plugin_dirs() if d.exists()]


def create_user_plugin_dir() -> Path:
    """Create the user plugin directory if it does not exist and return its path."""
    user_dir = Path.home() / ".config" / "plottter" / "plugins"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir
