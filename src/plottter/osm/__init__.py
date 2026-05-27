"""OSM data-access subpackage.

Re-exports the public API surface used by MapGenerator and _MapMixin.
Individual sub-modules (geocode, overpass, categories, geometry, cache)
are imported lazily in the functions that need them so that the subpackage
can be imported in a headless/test context without network calls.
"""

from plottter.osm.types import MapData, MapFeature
from plottter.osm.categories import FEATURE_CATEGORIES

__all__ = [
    "MapData",
    "MapFeature",
    "FEATURE_CATEGORIES",
]
