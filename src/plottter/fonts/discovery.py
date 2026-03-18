"""System font discovery for Plottter.

Scans OS font directories and builds a catalog of available fonts with
metadata read from font files using fontTools.  Results are cached to
``~/.plottter/font_cache.json`` keyed by directory modification times
so repeated calls are fast.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# FontInfo dataclass
# ---------------------------------------------------------------------------


@dataclass
class FontInfo:
    """Metadata for a single font face."""

    family: str
    style: str  # e.g. "Regular", "Bold", "Italic", "Bold Italic"
    file_path: str
    source: str = "system"  # "system" or "google"


# ---------------------------------------------------------------------------
# Platform-specific font directories
# ---------------------------------------------------------------------------

def _font_dirs() -> list[Path]:
    """Return platform-specific directories to scan for font files."""
    dirs: list[Path] = []
    if sys.platform.startswith("linux"):
        dirs = [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".local" / "share" / "fonts",
            Path.home() / ".fonts",
        ]
    elif sys.platform == "darwin":
        dirs = [
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
    elif sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        dirs = [
            Path(windir) / "Fonts",
            Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
        ]
    # Always include downloaded Google Fonts cache
    dirs.append(Path.home() / ".plottter" / "fonts" / "google")
    return [d for d in dirs if d.exists()]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_PATH = Path.home() / ".plottter" / "font_cache.json"


def _dir_mtime(directory: Path) -> float:
    """Return the modification time of a directory (0 if not present)."""
    try:
        return directory.stat().st_mtime
    except OSError:
        return 0.0


def _load_cache() -> tuple[dict[str, float], list[FontInfo]] | None:
    """Load ``font_cache.json``.

    Returns ``(mtime_map, fonts)`` or ``None`` if the cache is missing or
    corrupt.
    """
    if not _CACHE_PATH.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        mtime_map: dict[str, float] = data.get("mtimes", {})
        fonts = [
            FontInfo(
                family=f["family"],
                style=f["style"],
                file_path=f["file_path"],
                source=f.get("source", "system"),
            )
            for f in data.get("fonts", [])
        ]
        return mtime_map, fonts
    except Exception:
        return None


def _save_cache(mtime_map: dict[str, float], fonts: list[FontInfo]) -> None:
    """Write the font catalog to ``~/.plottter/font_cache.json``."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mtimes": mtime_map,
        "fonts": [asdict(f) for f in fonts],
        "saved_at": time.time(),
    }
    _CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _cache_is_valid(cached_mtimes: dict[str, float], dirs: list[Path]) -> bool:
    """Return True if all scanned directories have the same mtime as cached."""
    current_keys = {str(d): _dir_mtime(d) for d in dirs}
    return current_keys == cached_mtimes


# ---------------------------------------------------------------------------
# Font metadata extraction
# ---------------------------------------------------------------------------

_NAMEID_FAMILY = 1
_NAMEID_SUBFAMILY = 2


def _read_font_metadata(path: Path) -> list[tuple[str, str]]:
    """Return ``[(family, style), ...]`` from a font file.

    Handles .ttf, .otf (single face) and .ttc (TrueType collection).
    Returns an empty list if the file cannot be read.
    """
    try:
        from fontTools.ttLib import TTFont, TTCollection  # type: ignore[import]
    except ImportError:
        return []

    results: list[tuple[str, str]] = []

    def _extract(tt: Any) -> None:
        try:
            name_table = tt["name"]
            family = (name_table.getDebugName(_NAMEID_FAMILY) or "").strip()
            style = (name_table.getDebugName(_NAMEID_SUBFAMILY) or "Regular").strip()
            if family:
                results.append((family, style or "Regular"))
        except Exception:
            pass

    suffix = path.suffix.lower()
    try:
        if suffix == ".ttc":
            collection = TTCollection(str(path))
            try:
                for font in collection.fonts:
                    _extract(font)
            finally:
                collection.close()
        else:
            tt = TTFont(str(path), lazy=True)
            try:
                _extract(tt)
            finally:
                tt.close()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}

# Module-level cache so repeated calls in the same session are instant.
_in_memory_cache: list[FontInfo] | None = None


def discover_system_fonts(*, force_rescan: bool = False) -> list[FontInfo]:
    """Scan OS font directories and return a list of :class:`FontInfo` objects.

    Results are cached to ``~/.plottter/font_cache.json``.  The cache is
    invalidated automatically when any font directory's modification time
    changes.

    Parameters
    ----------
    force_rescan:
        If ``True``, bypass both the in-memory and on-disk cache and perform a
        full directory scan regardless of mtime.
    """
    global _in_memory_cache

    if not force_rescan and _in_memory_cache is not None:
        return _in_memory_cache

    dirs = _font_dirs()

    if not force_rescan:
        cached = _load_cache()
        if cached is not None:
            mtime_map, fonts = cached
            if _cache_is_valid(mtime_map, dirs):
                _in_memory_cache = fonts
                return fonts

    # Full scan
    fonts: list[FontInfo] = []
    for directory in dirs:
        source = "google" if "google" in str(directory) else "system"
        for root, _subdirs, files in os.walk(directory):
            for filename in files:
                if Path(filename).suffix.lower() not in _FONT_EXTENSIONS:
                    continue
                fpath = Path(root) / filename
                for family, style in _read_font_metadata(fpath):
                    fonts.append(
                        FontInfo(
                            family=family,
                            style=style,
                            file_path=str(fpath),
                            source=source,
                        )
                    )

    # Sort by family then style for consistent ordering
    fonts.sort(key=lambda f: (f.family.lower(), f.style.lower()))

    mtime_map = {str(d): _dir_mtime(d) for d in dirs}
    _save_cache(mtime_map, fonts)
    _in_memory_cache = fonts
    return fonts


def invalidate_font_cache() -> None:
    """Force the next call to :func:`discover_system_fonts` to do a full rescan.

    Deletes ``~/.plottter/font_cache.json`` and clears the in-memory cache.
    """
    global _in_memory_cache
    _in_memory_cache = None
    if _CACHE_PATH.exists():
        _CACHE_PATH.unlink()


def get_font_families() -> list[str]:
    """Return a sorted list of unique font family names from the system."""
    fonts = discover_system_fonts()
    seen: set[str] = set()
    families: list[str] = []
    for f in fonts:
        if f.family not in seen:
            seen.add(f.family)
            families.append(f.family)
    return sorted(families, key=str.lower)


def get_font_path(family: str, style: str = "Regular") -> str | None:
    """Return the file path for a specific font family and style.

    Falls back to "Regular" if the exact style is not found, then to the
    first available style for that family.  Returns ``None`` if the family
    does not exist at all.

    Parameters
    ----------
    family:
        Font family name (case-sensitive, as returned by
        :func:`get_font_families`).
    style:
        Subfamily/style name (e.g. "Regular", "Bold", "Italic").
    """
    fonts = discover_system_fonts()

    # Collect all faces for this family
    family_faces = [f for f in fonts if f.family == family]
    if not family_faces:
        return None

    # Exact style match
    for f in family_faces:
        if f.style == style:
            return f.file_path

    # Fall back to "Regular"
    for f in family_faces:
        if f.style == "Regular":
            return f.file_path

    # Fall back to first available style
    return family_faces[0].file_path
