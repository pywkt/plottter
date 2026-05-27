"""Tests for plottter.osm.types — MapFeature / MapData dataclasses + JSON round-trip."""

import pytest

from plottter.osm.types import MapData, MapFeature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_map_data() -> MapData:
    """Build a MapData with two categories for use across tests."""
    road = MapFeature(
        tags={"highway": "primary", "name": "Main St"},
        coords=[(35.0, 135.0), (35.001, 135.001), (35.002, 135.002)],
        is_area=False,
    )
    park = MapFeature(
        tags={"leisure": "park", "name": "Central Park"},
        coords=[(35.01, 135.01), (35.02, 135.01), (35.02, 135.02), (35.01, 135.01)],
        is_area=True,
    )
    return MapData(
        location="Kyoto, Japan",
        center=(35.0116, 135.7681),
        bbox=(34.96, 135.72, 35.06, 135.82),
        features={
            "roads_major": [road],
            "parks": [park],
        },
    )


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestMapFeature:
    def test_fields(self):
        feat = MapFeature(
            tags={"highway": "primary"},
            coords=[(1.0, 2.0), (3.0, 4.0)],
            is_area=False,
        )
        assert feat.tags == {"highway": "primary"}
        assert feat.coords == [(1.0, 2.0), (3.0, 4.0)]
        assert feat.is_area is False

    def test_area_flag(self):
        feat = MapFeature(tags={"building": "yes"}, coords=[(0, 0), (1, 0), (1, 1), (0, 0)], is_area=True)
        assert feat.is_area is True


class TestMapData:
    def test_default_attribution(self):
        md = _make_map_data()
        assert md.attribution == "© OpenStreetMap contributors"

    def test_custom_attribution(self):
        md = MapData(
            location="Test",
            center=(0.0, 0.0),
            bbox=(0.0, 0.0, 1.0, 1.0),
            features={},
            attribution="Custom attribution",
        )
        assert md.attribution == "Custom attribution"

    def test_center_and_bbox(self):
        md = _make_map_data()
        assert md.center == (35.0116, 135.7681)
        assert md.bbox == (34.96, 135.72, 35.06, 135.82)

    def test_two_categories(self):
        md = _make_map_data()
        assert set(md.features.keys()) == {"roads_major", "parks"}
        assert len(md.features["roads_major"]) == 1
        assert len(md.features["parks"]) == 1


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    def test_to_json_is_serialisable(self):
        """to_json() must produce only JSON-native types (no tuples, no custom objs)."""
        import json
        md = _make_map_data()
        d = md.to_json()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_center_round_trip(self):
        md = _make_map_data()
        d = md.to_json()
        md2 = MapData.from_json(d)
        assert md2.center == md.center

    def test_bbox_round_trip(self):
        md = _make_map_data()
        d = md.to_json()
        md2 = MapData.from_json(d)
        assert md2.bbox == md.bbox

    def test_location_round_trip(self):
        md = _make_map_data()
        md2 = MapData.from_json(md.to_json())
        assert md2.location == "Kyoto, Japan"

    def test_attribution_round_trip(self):
        md = _make_map_data()
        md2 = MapData.from_json(md.to_json())
        assert md2.attribution == md.attribution

    def test_feature_categories_preserved(self):
        md = _make_map_data()
        md2 = MapData.from_json(md.to_json())
        assert set(md2.features.keys()) == {"roads_major", "parks"}

    def test_feature_coords_round_trip(self):
        md = _make_map_data()
        md2 = MapData.from_json(md.to_json())
        orig_coords = md.features["roads_major"][0].coords
        rt_coords = md2.features["roads_major"][0].coords
        assert len(rt_coords) == len(orig_coords)
        for orig, rt in zip(orig_coords, rt_coords):
            assert pytest.approx(orig[0]) == rt[0]
            assert pytest.approx(orig[1]) == rt[1]

    def test_feature_is_area_round_trip(self):
        md = _make_map_data()
        md2 = MapData.from_json(md.to_json())
        assert md2.features["roads_major"][0].is_area is False
        assert md2.features["parks"][0].is_area is True

    def test_feature_tags_round_trip(self):
        md = _make_map_data()
        md2 = MapData.from_json(md.to_json())
        assert md2.features["roads_major"][0].tags == {"highway": "primary", "name": "Main St"}
        assert md2.features["parks"][0].tags == {"leisure": "park", "name": "Central Park"}

    def test_to_json_coords_are_lists_not_tuples(self):
        """JSON requires lists; confirm to_json() stores coords as lists."""
        md = _make_map_data()
        d = md.to_json()
        for feats in d["features"].values():
            for feat in feats:
                assert isinstance(feat["coords"], list)
                for c in feat["coords"]:
                    assert isinstance(c, list)

    def test_from_json_coords_are_tuples(self):
        """from_json() should restore coords as tuples to match the dataclass signature."""
        md = _make_map_data()
        md2 = MapData.from_json(md.to_json())
        for feats in md2.features.values():
            for feat in feats:
                for c in feat.coords:
                    assert isinstance(c, tuple)

    def test_empty_features_round_trip(self):
        md = MapData(
            location="Empty",
            center=(0.0, 0.0),
            bbox=(-1.0, -1.0, 1.0, 1.0),
            features={},
        )
        md2 = MapData.from_json(md.to_json())
        assert md2.features == {}
        assert md2.location == "Empty"

    def test_missing_attribution_defaults(self):
        """from_json() with no 'attribution' key should fall back to the default."""
        d = {
            "location": "Nowhere",
            "center": [0.0, 0.0],
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "features": {},
            # no 'attribution' key
        }
        md = MapData.from_json(d)
        assert md.attribution == "© OpenStreetMap contributors"
