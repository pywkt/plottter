"""Generator registry and discovery utilities."""

from __future__ import annotations

GENERATORS: dict[str, type] = {}


def register_generator(cls: type) -> type:
    """Decorator to register a generator class by its name."""
    GENERATORS[cls.name] = cls
    return cls


def get_generators_by_category(category: str) -> list[type]:
    """Return all registered generators with the given category."""
    return [cls for cls in GENERATORS.values() if cls.category == category]


def _import_builtin_generators() -> None:
    """Import all built-in generator modules so their @register_generator decorators fire."""
    # Each import triggers @register_generator, populating GENERATORS.
    # This must happen after register_generator is defined above.
    from plottter.generators import parametric as _parametric  # noqa: F401
    from plottter.generators import polar as _polar  # noqa: F401
    from plottter.generators import modular_mult as _modular_mult  # noqa: F401
    from plottter.generators import flow_field as _flow_field  # noqa: F401
    from plottter.generators import lsystem as _lsystem  # noqa: F401
    from plottter.generators import grid_pattern as _grid_pattern  # noqa: F401
    # Image-to-lines generators
    from plottter.generators import edge_detect as _edge_detect  # noqa: F401
    from plottter.generators import hatching as _hatching  # noqa: F401
    from plottter.generators import flow_image as _flow_image  # noqa: F401
    from plottter.generators import stipple as _stipple  # noqa: F401
    from plottter.generators import contour as _contour  # noqa: F401
    from plottter.generators import xdog as _xdog  # noqa: F401
    from plottter.generators import fdog as _fdog  # noqa: F401
    from plottter.generators import text as _text  # noqa: F401
    from plottter.generators import hedcut as _hedcut  # noqa: F401
    from plottter.generators import scene3d_generator as _scene3d  # noqa: F401
    from plottter.generators import circular_scribble as _circular_scribble  # noqa: F401
    from plottter.generators import scanline_halftone as _scanline_halftone  # noqa: F401
    from plottter.generators import dot_grid as _dot_grid  # noqa: F401
    from plottter.generators import concentric_rings as _concentric_rings  # noqa: F401
    from plottter.generators import geometric_grid as _geometric_grid  # noqa: F401
    from plottter.generators import tam as _tam  # noqa: F401
    from plottter.generators import lic as _lic  # noqa: F401
    from plottter.generators import voronoi as _voronoi  # noqa: F401
    from plottter.generators import penrose as _penrose  # noqa: F401
    from plottter.generators.halftone import HalftoneGenerator  # noqa: F401
    from plottter.generators.triangulated_hatch import MosaicHatchGenerator  # noqa: F401
    from plottter.generators.spiral import SpiralGenerator  # noqa: F401
    from plottter.generators import sketch as _sketch  # noqa: F401
    from plottter.generators import mesh_slicer as _mesh_slicer  # noqa: F401
    from plottter.generators import ascii_art as _ascii_art  # noqa: F401
    from plottter.generators import audio_waveform as _audio_waveform  # noqa: F401


def load_plugins(extra_dirs=None) -> list[str]:
    """Load custom generators from the plugins directory.

    See :mod:`plottter.generators.plugin_loader` for full documentation.

    Returns
    -------
    list[str]
        Names of newly registered generator classes.
    """
    from plottter.generators.plugin_loader import load_plugins as _load
    return _load(extra_dirs=extra_dirs)


_import_builtin_generators()
# Auto-load user plugins on import (failures are logged, never fatal)
try:
    load_plugins()
except Exception:  # pragma: no cover
    pass
