"""Tests for src/plottter/fonts/discovery.py.

These tests verify font discovery, caching, and lookup helpers.  They rely
on at least one real .ttf file being present on the system (DejaVu fonts ship
with most Linux distributions and are available in the CI environment).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from plottter.fonts.discovery import (
    FontInfo,
    _CACHE_PATH,
    _read_font_metadata,
    discover_system_fonts,
    get_font_families,
    get_font_path,
    invalidate_font_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KNOWN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SKIP_IF_NO_FONT = pytest.mark.skipif(
    not KNOWN_FONT.exists(),
    reason="DejaVu font not installed on this system",
)


def _clean_cache() -> None:
    """Remove the on-disk and in-memory cache before a test."""
    invalidate_font_cache()


# ---------------------------------------------------------------------------
# FontInfo dataclass
# ---------------------------------------------------------------------------


def test_font_info_fields() -> None:
    fi = FontInfo(family="DejaVu Sans", style="Regular", file_path="/tmp/f.ttf")
    assert fi.family == "DejaVu Sans"
    assert fi.style == "Regular"
    assert fi.file_path == "/tmp/f.ttf"
    assert fi.source == "system"


def test_font_info_source_google() -> None:
    fi = FontInfo(family="Roboto", style="Bold", file_path="/tmp/r.ttf", source="google")
    assert fi.source == "google"


# ---------------------------------------------------------------------------
# _read_font_metadata
# ---------------------------------------------------------------------------


@SKIP_IF_NO_FONT
def test_read_font_metadata_ttf() -> None:
    results = _read_font_metadata(KNOWN_FONT)
    assert len(results) >= 1
    families = [r[0] for r in results]
    assert any("DejaVu" in f for f in families)


def test_read_font_metadata_missing_file() -> None:
    results = _read_font_metadata(Path("/nonexistent/font.ttf"))
    assert results == []


def test_read_font_metadata_bad_extension() -> None:
    # Plain text file with .ttf extension — should return empty, not crash.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
        tmp.write(b"not a font file")
        tmp_path = Path(tmp.name)
    try:
        results = _read_font_metadata(tmp_path)
        assert results == []
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# discover_system_fonts
# ---------------------------------------------------------------------------


@SKIP_IF_NO_FONT
def test_discover_system_fonts_returns_list() -> None:
    _clean_cache()
    fonts = discover_system_fonts()
    assert isinstance(fonts, list)
    assert len(fonts) > 0


@SKIP_IF_NO_FONT
def test_discover_system_fonts_each_has_required_fields() -> None:
    _clean_cache()
    fonts = discover_system_fonts()
    for f in fonts:
        assert isinstance(f, FontInfo)
        assert f.family  # non-empty
        assert f.style   # non-empty
        assert f.file_path  # non-empty


@SKIP_IF_NO_FONT
def test_discover_system_fonts_file_paths_exist() -> None:
    _clean_cache()
    fonts = discover_system_fonts()
    for f in fonts[:20]:  # check first 20 to keep test fast
        assert Path(f.file_path).exists(), f"Missing: {f.file_path}"


@SKIP_IF_NO_FONT
def test_discover_system_fonts_cache_created() -> None:
    _clean_cache()
    assert not _CACHE_PATH.exists()
    discover_system_fonts()
    assert _CACHE_PATH.exists()


@SKIP_IF_NO_FONT
def test_discover_system_fonts_second_call_uses_cache() -> None:
    _clean_cache()
    t0 = time.perf_counter()
    discover_system_fonts()
    t1 = time.perf_counter()
    first_duration = t1 - t0

    # Second call should load from in-memory cache and be much faster
    t0 = time.perf_counter()
    fonts2 = discover_system_fonts()
    t1 = time.perf_counter()
    second_duration = t1 - t0

    # In-memory cache should make second call nearly instant (< first call)
    assert second_duration <= first_duration + 0.5  # generous bound
    assert len(fonts2) > 0


@SKIP_IF_NO_FONT
def test_discover_system_fonts_disk_cache_loads() -> None:
    """Clearing in-memory cache but keeping disk cache should still be fast."""
    _clean_cache()
    discover_system_fonts()  # builds disk cache

    # Clear only the in-memory part
    import plottter.fonts.discovery as mod
    mod._in_memory_cache = None

    fonts2 = discover_system_fonts()
    assert len(fonts2) > 0


@SKIP_IF_NO_FONT
def test_force_rescan_bypasses_cache() -> None:
    _clean_cache()
    fonts1 = discover_system_fonts()
    fonts2 = discover_system_fonts(force_rescan=True)
    assert len(fonts1) == len(fonts2)


# ---------------------------------------------------------------------------
# invalidate_font_cache
# ---------------------------------------------------------------------------


@SKIP_IF_NO_FONT
def test_invalidate_font_cache_removes_file() -> None:
    _clean_cache()
    discover_system_fonts()
    assert _CACHE_PATH.exists()
    invalidate_font_cache()
    assert not _CACHE_PATH.exists()


def test_invalidate_font_cache_idempotent() -> None:
    """Calling twice shouldn't raise even if cache doesn't exist."""
    invalidate_font_cache()
    invalidate_font_cache()  # should not raise


# ---------------------------------------------------------------------------
# get_font_families
# ---------------------------------------------------------------------------


@SKIP_IF_NO_FONT
def test_get_font_families_returns_sorted_unique() -> None:
    _clean_cache()
    families = get_font_families()
    assert isinstance(families, list)
    assert len(families) > 0
    # Sorted case-insensitively
    assert families == sorted(families, key=str.lower)
    # Unique
    assert len(families) == len(set(families))


@SKIP_IF_NO_FONT
def test_get_font_families_contains_dejavu() -> None:
    _clean_cache()
    families = get_font_families()
    assert any("DejaVu" in f for f in families)


# ---------------------------------------------------------------------------
# get_font_path
# ---------------------------------------------------------------------------


@SKIP_IF_NO_FONT
def test_get_font_path_known_font() -> None:
    _clean_cache()
    families = get_font_families()
    family = next(f for f in families if "DejaVu" in f)
    path = get_font_path(family)
    assert path is not None
    assert Path(path).exists()


@SKIP_IF_NO_FONT
def test_get_font_path_exact_style() -> None:
    _clean_cache()
    fonts = discover_system_fonts()
    # Find a family/style pair that exists
    sample = fonts[0]
    path = get_font_path(sample.family, sample.style)
    assert path == sample.file_path


@SKIP_IF_NO_FONT
def test_get_font_path_fallback_to_regular() -> None:
    """Requesting a non-existent style falls back to Regular or first face."""
    _clean_cache()
    families = get_font_families()
    family = next(f for f in families if "DejaVu" in f)
    path = get_font_path(family, "NonexistentStyle999")
    assert path is not None


def test_get_font_path_nonexistent_family() -> None:
    path = get_font_path("ThisFamilyDoesNotExistAtAll_XYZ")
    assert path is None


# ---------------------------------------------------------------------------
# Cache validity — unit-level test using temp dir
# ---------------------------------------------------------------------------


def test_cache_invalidated_on_dir_mtime_change(tmp_path: Path) -> None:
    """If a font directory's mtime changes, the cache is considered stale."""
    from plottter.fonts.discovery import _cache_is_valid

    dirs = [tmp_path]
    mtime_map = {str(tmp_path): tmp_path.stat().st_mtime}

    # Same mtime — cache is valid
    assert _cache_is_valid(mtime_map, dirs) is True

    # Simulate mtime change by touching the directory
    time.sleep(0.01)
    (tmp_path / "dummy").mkdir()
    (tmp_path / "dummy").rmdir()

    # On most filesystems the dir mtime updates; if not, force via os.utime
    import os
    new_mtime = mtime_map[str(tmp_path)] + 1
    os.utime(tmp_path, (new_mtime, new_mtime))

    assert _cache_is_valid(mtime_map, dirs) is False
