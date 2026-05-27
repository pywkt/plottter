"""Phase 145 — Re-generation without re-fetch.

Verifies that changing a styling parameter (``area_fill``) and re-running
``MapGenerator.generate_layers()`` over the same cached ``MapData`` object
produces different output **without** making any network calls.

``urllib.request.urlopen`` is patched to raise ``RuntimeError`` if called, so
any accidental network access will fail the test immediately.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from plottter.models import Canvas
from plottter.generators.map_generator import MapGenerator
from plottter.osm.types import MapData, MapFeature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def _make_area_map_data() -> MapData:
    """Fixture MapData with a park polygon large enough to produce hatch lines."""
    # A rectangular park near Paris (explicitly closed: first == last).
    park_coords = [
        (48.850, 2.340),
        (48.870, 2.340),
        (48.870, 2.370),
        (48.850, 2.370),
        (48.850, 2.340),
    ]
    parks = [
        MapFeature(
            tags={"leisure": "park"},
            coords=park_coords,
            is_area=True,
        )
    ]
    return MapData(
        location="Paris, France",
        center=(48.860, 2.355),
        bbox=(48.845, 2.335, 48.875, 2.380),
        features={"parks": parks},
    )


def _urlopen_must_not_be_called(*args, **kwargs):
    raise RuntimeError(
        "urllib.request.urlopen was called during generate_layers() — "
        "re-generation must not hit the network."
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestRegenWithoutRefetch:
    """generate_layers() uses only cached MapData — no network."""

    def test_different_area_fill_produces_different_output_without_network(self):
        """Changing area_fill from 'none' to 'hatch' yields more paths, zero urlopen calls."""
        canvas = _make_canvas()
        map_data = _make_area_map_data()
        gen = MapGenerator()

        base_params = {
            "_map_data": map_data,
            "include_water": False,
            "include_parks": True,
            "include_buildings": False,
            "include_waterways": False,
            "include_roads": False,
            "include_rail": False,
            "include_coastline": False,
            "include_attribution": False,
            "simplify_mm": 0.0,
            "min_feature_mm": 0.0,
        }

        mock_urlopen = MagicMock(side_effect=_urlopen_must_not_be_called)
        with patch("urllib.request.urlopen", mock_urlopen):
            params_none = {**base_params, "area_fill": "none"}
            specs_none = gen.generate_layers(params_none, canvas)

            params_hatch = {**base_params, "area_fill": "hatch", "fill_spacing_mm": 3.0, "fill_angle_deg": 45.0}
            specs_hatch = gen.generate_layers(params_hatch, canvas)

            # Assert urlopen was never called during either generate_layers() run.
            mock_urlopen.assert_not_called()

        # Both runs must produce at least one layer.
        assert specs_none, "generate_layers(area_fill='none') returned no layers"
        assert specs_hatch, "generate_layers(area_fill='hatch') returned no layers"

        # Count total paths across all layers.
        paths_none = sum(len(spec.paths) for spec in specs_none)
        paths_hatch = sum(len(spec.paths) for spec in specs_hatch)

        # Hatch fill adds interior lines on top of the outline — must have more paths.
        assert paths_hatch > paths_none, (
            f"Expected hatch to produce more paths than outline-only "
            f"(hatch={paths_hatch}, none={paths_none})"
        )

    def test_urlopen_not_called_during_generate(self):
        """urlopen is never invoked — generate_layers is purely offline."""
        canvas = _make_canvas()
        map_data = _make_area_map_data()
        gen = MapGenerator()

        params = {
            "_map_data": map_data,
            "include_water": False,
            "include_parks": True,
            "include_buildings": False,
            "include_waterways": False,
            "include_roads": False,
            "include_rail": False,
            "include_coastline": False,
            "include_attribution": False,
            "area_fill": "none",
            "simplify_mm": 0.0,
            "min_feature_mm": 0.0,
        }

        mock_urlopen = MagicMock(side_effect=_urlopen_must_not_be_called)
        with patch("urllib.request.urlopen", mock_urlopen):
            gen.generate_layers(params, canvas)
            mock_urlopen.assert_not_called()

    def test_none_map_data_returns_empty(self):
        """When _map_data is absent, generate_layers returns [] without network."""
        canvas = _make_canvas()
        gen = MapGenerator()

        with patch("urllib.request.urlopen", side_effect=_urlopen_must_not_be_called):
            result = gen.generate_layers({"_map_data": None}, canvas)

        assert result == []
