"""Tests for osm/cache.py (§11)."""

from __future__ import annotations

import json

import pytest

from plottter.osm.types import MapData, MapFeature
import plottter.osm.cache as cache_mod


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_map_data() -> MapData:
    """Return a minimal but realistic MapData for round-trip testing."""
    feat = MapFeature(
        tags={"highway": "primary", "name": "Main St"},
        coords=[(35.0, 135.0), (35.001, 135.001)],
        is_area=False,
        inner_coords=[],
    )
    return MapData(
        location="Kyoto, Japan",
        center=(35.0116, 135.7681),
        bbox=(35.0, 135.75, 35.02, 135.79),
        features={"roads_major": [feat]},
        attribution="© OpenStreetMap contributors",
    )


# ---------------------------------------------------------------------------
# cache_dir
# ---------------------------------------------------------------------------

class TestCacheDir:
    def test_returns_maps_subdir_of_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        d = cache_mod.cache_dir()
        assert d == tmp_path / ".plottter" / "maps"

    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        d = cache_mod.cache_dir()
        assert d.is_dir()

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        cache_mod.cache_dir()
        # Second call must not raise even though directory exists
        d = cache_mod.cache_dir()
        assert d.is_dir()


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_returns_16_hex_chars(self):
        key = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major"])
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_stable_for_same_inputs(self):
        k1 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major", "water"])
        k2 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major", "water"])
        assert k1 == k2

    def test_selector_order_does_not_matter(self):
        k1 = cache_mod.cache_key("Paris", 1.5, "radius", ["water", "roads_major"])
        k2 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major", "water"])
        assert k1 == k2

    def test_changes_with_different_location(self):
        k1 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major"])
        k2 = cache_mod.cache_key("Tokyo", 1.5, "radius", ["roads_major"])
        assert k1 != k2

    def test_changes_with_different_radius(self):
        k1 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major"])
        k2 = cache_mod.cache_key("Paris", 2.0, "radius", ["roads_major"])
        assert k1 != k2

    def test_changes_with_different_extent_mode(self):
        k1 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major"])
        k2 = cache_mod.cache_key("Paris", 1.5, "nominatim_bbox", ["roads_major"])
        assert k1 != k2

    def test_changes_with_different_selectors(self):
        k1 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major"])
        k2 = cache_mod.cache_key("Paris", 1.5, "radius", ["roads_major", "water"])
        assert k1 != k2

    def test_radius_rounded_to_three_decimals(self):
        """Floating-point noise beyond 3 dp should not produce different keys."""
        k1 = cache_mod.cache_key("Paris", 1.5000000001, "radius", [])
        k2 = cache_mod.cache_key("Paris", 1.5, "radius", [])
        assert k1 == k2


# ---------------------------------------------------------------------------
# store / load round-trip
# ---------------------------------------------------------------------------

class TestStoreLoad:
    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        md = _make_map_data()
        key = cache_mod.cache_key(md.location, 1.5, "radius", list(md.features.keys()))

        cache_mod.store(key, md)
        loaded = cache_mod.load(key)

        assert loaded is not None
        assert loaded.location == md.location
        assert loaded.center == md.center
        assert loaded.bbox == md.bbox
        assert loaded.attribution == md.attribution
        assert set(loaded.features.keys()) == set(md.features.keys())

        orig_feat = md.features["roads_major"][0]
        loaded_feat = loaded.features["roads_major"][0]
        assert loaded_feat.tags == orig_feat.tags
        assert loaded_feat.coords == orig_feat.coords
        assert loaded_feat.is_area == orig_feat.is_area
        assert loaded_feat.inner_coords == orig_feat.inner_coords

    def test_load_returns_none_for_missing_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        result = cache_mod.load("nonexistent_key_xx")
        assert result is None

    def test_load_returns_none_for_corrupt_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        key = "corruptkey1234ab"
        maps_dir = tmp_path / ".plottter" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        (maps_dir / f"{key}.json").write_text("this is not valid json {{{{")

        result = cache_mod.load(key)
        assert result is None

    def test_load_returns_none_for_valid_json_but_wrong_schema(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        key = "wrongschema1234a"
        maps_dir = tmp_path / ".plottter" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        (maps_dir / f"{key}.json").write_text(json.dumps({"unexpected": "structure"}))

        result = cache_mod.load(key)
        assert result is None

    def test_load_rejects_outdated_schema_version(self, tmp_path, monkeypatch):
        """A structurally valid payload from an older parser (lower
        schema_version) is discarded so the caller re-fetches with the current
        parser. This is the stale-cache scenario that caused mis-stitched water."""
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        key = "oldschema12345ab"
        maps_dir = tmp_path / ".plottter" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        payload = _make_map_data().to_json()
        payload["schema_version"] = 1  # older than current
        (maps_dir / f"{key}.json").write_text(json.dumps(payload))

        assert cache_mod.load(key) is None

    def test_store_writes_current_schema_version(self, tmp_path, monkeypatch):
        from plottter.osm.types import MAPDATA_SCHEMA_VERSION

        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        md = _make_map_data()
        key = "schemaverkey1234"
        cache_mod.store(key, md)
        on_disk = json.loads((tmp_path / ".plottter" / "maps" / f"{key}.json").read_text())
        assert on_disk["schema_version"] == MAPDATA_SCHEMA_VERSION
        # And it round-trips through load (version matches).
        assert cache_mod.load(key) is not None

    def test_store_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        md = _make_map_data()
        key = "teststorekey1234"
        cache_mod.store(key, md)
        assert (tmp_path / ".plottter" / "maps" / f"{key}.json").exists()

    def test_load_never_raises_on_mkdir_failure(self, monkeypatch):
        """load() must return None even when the cache directory cannot be created."""
        def _raise(*_a, **_kw):
            raise OSError("permission denied")

        monkeypatch.setattr(cache_mod.pathlib.Path, "mkdir", _raise)
        monkeypatch.setattr(
            cache_mod.pathlib.Path, "home", staticmethod(lambda: cache_mod.pathlib.Path("/nonexistent_root"))
        )
        result = cache_mod.load("anykey1234567890")
        assert result is None

    def test_store_overwrites_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod.pathlib.Path, "home", staticmethod(lambda: tmp_path))
        md1 = _make_map_data()
        md2 = MapData(
            location="Tokyo",
            center=(35.6762, 139.6503),
            bbox=(35.6, 139.6, 35.7, 139.7),
            features={},
        )
        key = "overwritekey1234"
        cache_mod.store(key, md1)
        cache_mod.store(key, md2)
        loaded = cache_mod.load(key)
        assert loaded is not None
        assert loaded.location == "Tokyo"
