"""Tests for CLI inline fetch path of MapGenerator (task 147.4).

Covers:
- generate_layers with a ``location`` param and no ``_map_data`` performs an
  inline fetch so the CLI path works headlessly.
- geocode() and fetch_overpass() are invoked during that inline fetch.
- Returns a non-empty list of LayerSpec when features are present.
- Without location (and without _map_data), still returns [].
"""

from __future__ import annotations

from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_canvas():
    from plottter.models import Canvas

    return Canvas.from_preset("A4", margin=10.0)


def _make_geocode_result():
    """Minimal GeocodeResult for Kyoto (matches nominatim_kyoto.json fixture)."""
    from plottter.osm.geocode import GeocodeResult

    return GeocodeResult(
        display_name="Kyoto, Japan",
        lat=35.0116,
        lon=135.7681,
        bbox=(34.8891, 35.1329, 135.6138, 135.9196),
    )


def _make_road_feature():
    """A single open-way road feature inside the Kyoto bbox."""
    from plottter.osm.types import MapFeature

    return MapFeature(
        tags={"highway": "primary"},
        coords=[
            (35.000, 135.760),
            (35.005, 135.765),
            (35.010, 135.770),
        ],
        is_area=False,
    )


def _make_map_data(features=None):
    """Return a minimal MapData with one road feature."""
    from plottter.osm.types import MapData

    if features is None:
        features = {"roads_major": [_make_road_feature()]}

    return MapData(
        location="Kyoto, Japan",
        center=(35.0116, 135.7681),
        bbox=(34.9616, 135.7181, 35.0616, 135.8181),
        features=features,
    )


def _base_params(**overrides):
    """Generator params with location set; all categories off except roads_major."""
    params = {
        "location": "Kyoto, Japan",
        "radius_km": 1.5,
        "extent_mode": "radius",
        "road_detail": "standard",
        "include_roads": True,
        "include_rail": False,
        "include_water": False,
        "include_waterways": False,
        "include_parks": False,
        "include_buildings": False,
        "include_coastline": False,
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "area_fill": "none",
        "fill_spacing_mm": 2.0,
        "fill_angle_deg": 45.0,
        "major_road_strokes": 1,
        "include_attribution": False,
    }
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# Tests: no inline fetch triggered
# ---------------------------------------------------------------------------


class TestNoFetch:
    """Existing behaviour: no location → no inline fetch → []."""

    def setup_method(self):
        from plottter.generators.map_generator import MapGenerator

        self.gen = MapGenerator()
        self.canvas = _make_canvas()

    def test_no_location_no_map_data_returns_empty(self):
        """Empty params dict → no map data → []."""
        result = self.gen.generate_layers({}, self.canvas)
        assert result == []

    def test_explicit_map_data_none_no_location_returns_empty(self):
        """_map_data=None without location param → []."""
        result = self.gen.generate_layers({"_map_data": None}, self.canvas)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: inline fetch path
# ---------------------------------------------------------------------------


class TestInlineFetch:
    """generate_layers performs inline fetch when location is present."""

    def setup_method(self):
        from plottter.generators.map_generator import MapGenerator

        self.gen = MapGenerator()
        self.canvas = _make_canvas()

    # --- fetch_map_data is called -------------------------------------------

    def test_fetch_map_data_called(self):
        """fetch_map_data is invoked once when location is given and _map_data absent."""
        import plottter.osm as osm_mod

        fake = _make_map_data()

        with patch.object(osm_mod, "fetch_map_data", return_value=fake) as mock_fetch:
            result = self.gen.generate_layers(_base_params(), self.canvas)

        mock_fetch.assert_called_once()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_fetch_uses_location_param(self):
        """The location string from params is forwarded to fetch_map_data."""
        import plottter.osm as osm_mod

        fake = _make_map_data()

        with patch.object(osm_mod, "fetch_map_data", return_value=fake) as mock_fetch:
            self.gen.generate_layers(_base_params(location="Kyoto, Japan"), self.canvas)

        call_args = mock_fetch.call_args
        assert call_args[0][0] == "Kyoto, Japan"

    # --- geocode() and fetch_overpass() invoked at lower level --------------

    def test_geocode_and_overpass_invoked(self):
        """geocode() and fetch_overpass() are both called during inline fetch."""
        import plottter.osm.geocode as geocode_mod
        import plottter.osm.overpass as overpass_mod

        geo_result = _make_geocode_result()
        road_feature = _make_road_feature()

        with (
            patch.object(geocode_mod, "geocode", return_value=geo_result) as mock_geo,
            patch.object(
                overpass_mod, "fetch_overpass", return_value=[road_feature]
            ) as mock_overpass,
            patch("time.sleep"),  # suppress Nominatim throttle delay in tests
        ):
            result = self.gen.generate_layers(_base_params(), self.canvas)

        mock_geo.assert_called_once()
        mock_overpass.assert_called()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_geocode_receives_location_string(self):
        """geocode() receives the exact location string from params."""
        import plottter.osm.geocode as geocode_mod
        import plottter.osm.overpass as overpass_mod

        geo_result = _make_geocode_result()
        road_feature = _make_road_feature()

        with (
            patch.object(geocode_mod, "geocode", return_value=geo_result) as mock_geo,
            patch.object(overpass_mod, "fetch_overpass", return_value=[road_feature]),
            patch("time.sleep"),
        ):
            self.gen.generate_layers(_base_params(location="Kyoto, Japan"), self.canvas)

        positional_query = mock_geo.call_args[0][0]
        assert positional_query == "Kyoto, Japan"

    # --- output shape -------------------------------------------------------

    def test_layer_names_match_categories(self):
        """Inline-fetched result includes the Roads (major) layer."""
        import plottter.osm as osm_mod

        fake = _make_map_data()

        with patch.object(osm_mod, "fetch_map_data", return_value=fake):
            specs = self.gen.generate_layers(_base_params(), self.canvas)

        names = [s.name for s in specs]
        assert "Roads (major)" in names

    def test_layers_have_paths(self):
        """Each returned LayerSpec contains at least one path."""
        import plottter.osm as osm_mod

        fake = _make_map_data()

        with patch.object(osm_mod, "fetch_map_data", return_value=fake):
            specs = self.gen.generate_layers(_base_params(), self.canvas)

        for spec in specs:
            assert len(spec.paths) > 0, f"Layer '{spec.name}' has no paths"

    def test_empty_features_returns_empty_list(self):
        """If inline fetch returns MapData with no features, result is []."""
        import plottter.osm as osm_mod

        empty_map_data = _make_map_data(features={})

        with patch.object(osm_mod, "fetch_map_data", return_value=empty_map_data):
            result = self.gen.generate_layers(_base_params(), self.canvas)

        assert result == []

    # --- pre-fetched _map_data takes precedence -----------------------------

    def test_pre_fetched_map_data_not_overridden(self):
        """If _map_data is already populated, no inline fetch occurs."""
        import plottter.osm as osm_mod

        pre_fetched = _make_map_data()
        params = _base_params()
        params["_map_data"] = pre_fetched

        with patch.object(osm_mod, "fetch_map_data") as mock_fetch:
            result = self.gen.generate_layers(params, self.canvas)

        mock_fetch.assert_not_called()
        assert isinstance(result, list)
