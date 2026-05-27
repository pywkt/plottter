"""Disk cache for OSM map data (§11).

Cache directory: ~/.plottter/maps/
Cache key:       sha256(f"{location}|{round(radius_km,3)}|{extent_mode}|{sorted(selectors)}")[:16]
Cache value:     MapData serialised to JSON via MapData.to_json / from_json

Corrupt or unreadable cache files are silently ignored (return None).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plottter.osm.types import MapData


def cache_dir() -> pathlib.Path:
    """Return (and create) the map cache directory ``~/.plottter/maps/``."""
    d = pathlib.Path.home() / ".plottter" / "maps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(
    location: str,
    radius_km: float,
    extent_mode: str,
    selectors: list[str],
) -> str:
    """Return a 16-character hex cache key for the given fetch parameters.

    The key is the first 16 hex digits of the SHA-256 hash of the canonical
    string ``"{location}|{round(radius_km,3)}|{extent_mode}|{sorted_selectors}"``.
    """
    canonical = f"{location}|{round(radius_km, 3)}|{extent_mode}|{sorted(selectors)}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load(key: str) -> "MapData | None":
    """Load a cached :class:`~plottter.osm.types.MapData` for *key*.

    Returns ``None`` if the entry does not exist or the file is corrupt.
    Never raises.
    """
    from plottter.osm.types import MapData  # local import — avoid circular at module level

    try:
        path = cache_dir() / f"{key}.json"
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return MapData.from_json(data)
    except Exception:  # noqa: BLE001 — silently ignore corrupt / missing / inaccessible files
        return None


def store(key: str, map_data: "MapData") -> None:
    """Serialise *map_data* and write it to the cache under *key*.

    Writes atomically-ish: writes to a ``<key>.json.tmp`` file then renames
    so a partial write does not leave a corrupt cache entry.
    """
    tmp: pathlib.Path | None = None
    try:
        path = cache_dir() / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(map_data.to_json(), fh)
        tmp.replace(path)
    except Exception:  # noqa: BLE001 — best-effort write; do not crash on disk errors
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
