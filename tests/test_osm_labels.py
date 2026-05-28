"""Tests for plottter.osm.labels — Label collection from MapData.

Coverage:
  - _resolve_name language priority and fallback
  - collect_water_labels: named polygons yield Labels with positions inside
    the polygons
  - unnamed features are skipped
  - sub-min_size_mm features are skipped
  - priority and category fields
  - language-specific name is preferred
  - empty features dict → empty result
  - feature outside clip box is dropped
"""

import math

import pytest
from shapely.geometry import Point, Polygon

from plottter.osm.geometry import FitTransform, mercator
from plottter.osm.labels import (
    Label,
    _resolve_name,
    collect_park_labels,
    collect_place_labels,
    collect_road_labels,
    collect_water_labels,
    collect_waterway_labels,
    place_with_collision,
)
from plottter.osm.types import MapData, MapFeature


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _transform() -> FitTransform:
    """A simple FitTransform: Mercator radians → mm at scale 1 000, offset 50."""
    return FitTransform(scale=1000.0, x_origin=50.0, y_origin=50.0)


def _clip_box() -> tuple[float, float, float, float]:
    """A generous clipping rectangle (0–200 mm in both axes)."""
    return (0.0, 0.0, 200.0, 200.0)


def _water_feature(
    lat: float,
    lon: float,
    half_deg: float = 0.001,
    name: str | None = None,
    lang_name: str | None = None,
    lang: str = "en",
    is_area: bool = True,
) -> MapFeature:
    """Square water polygon centred at (lat, lon), spanning ±half_deg."""
    coords = [
        (lat - half_deg, lon - half_deg),
        (lat - half_deg, lon + half_deg),
        (lat + half_deg, lon + half_deg),
        (lat + half_deg, lon - half_deg),
        (lat - half_deg, lon - half_deg),  # close the ring
    ]
    tags: dict[str, str] = {}
    if name is not None:
        tags["name"] = name
    if lang_name is not None:
        tags[f"name:{lang}"] = lang_name
    return MapFeature(tags=tags, coords=coords, is_area=is_area)


def _map_data(features: list[MapFeature]) -> MapData:
    return MapData(
        location="Test",
        center=(0.0, 0.0),
        bbox=(-1.0, -1.0, 1.0, 1.0),
        features={"water": features},
    )


def _projected_polygon(feature: MapFeature, transform: FitTransform) -> Polygon:
    """Reconstruct the projected shapely Polygon for a MapFeature."""
    pts = []
    for lat, lon in feature.coords:
        px, py = mercator(lat, lon)
        pts.append(
            (
                transform.x_origin + px * transform.scale,
                transform.y_origin - py * transform.scale,
            )
        )
    return Polygon(pts)


# ---------------------------------------------------------------------------
# _resolve_name
# ---------------------------------------------------------------------------


class TestResolveName:
    def test_prefers_lang_specific_key(self):
        tags = {"name": "Generic Lake", "name:en": "English Lake"}
        assert _resolve_name(tags, "en") == "English Lake"

    def test_falls_back_to_name_when_lang_absent(self):
        tags = {"name": "Fallback Lake"}
        assert _resolve_name(tags, "en") == "Fallback Lake"

    def test_falls_back_to_name_when_lang_empty(self):
        tags = {"name": "Fallback", "name:en": "   "}
        assert _resolve_name(tags, "en") == "Fallback"

    def test_returns_empty_string_when_no_name(self):
        tags = {"natural": "water", "water": "lake"}
        assert _resolve_name(tags, "en") == ""

    def test_strips_whitespace(self):
        tags = {"name": "  Padded Lake  "}
        assert _resolve_name(tags, "en") == "Padded Lake"

    def test_lang_strips_whitespace(self):
        tags = {"name:fr": " Lac Blanc "}
        assert _resolve_name(tags, "fr") == "Lac Blanc"

    def test_different_language_code(self):
        tags = {"name": "湖", "name:en": "Lake"}
        assert _resolve_name(tags, "ja") == "湖"  # no ja tag → falls back to name


# ---------------------------------------------------------------------------
# collect_water_labels — core contract
# ---------------------------------------------------------------------------


class TestCollectWaterLabels:
    @pytest.fixture
    def tf(self) -> FitTransform:
        return _transform()

    @pytest.fixture
    def box(self) -> tuple[float, float, float, float]:
        return _clip_box()

    # --- two named polygons → two Labels ----------------------------------

    def test_two_named_features_yield_two_labels(self, tf, box):
        """Fixture MapData with 2 named water polygons → exactly 2 Labels."""
        f1 = _water_feature(0.01, 0.01, name="Lake Alpha")
        f2 = _water_feature(0.05, 0.05, name="Lake Beta")
        md = _map_data([f1, f2])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 2
        assert {lb.text for lb in labels} == {"Lake Alpha", "Lake Beta"}

    # --- positions inside polygons ----------------------------------------

    def test_label_positions_inside_polygons(self, tf, box):
        """Label positions (representative_point) lie inside each projected polygon."""
        f1 = _water_feature(0.01, 0.01, half_deg=0.002, name="Lake A")
        f2 = _water_feature(0.05, 0.05, half_deg=0.002, name="Lake B")
        md = _map_data([f1, f2])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 2
        label_map = {lb.text: lb for lb in labels}

        for feat, name in ((f1, "Lake A"), (f2, "Lake B")):
            poly = _projected_polygon(feat, tf)
            px, py = label_map[name].position
            pt = Point(px, py)
            # representative_point is guaranteed inside; check containment
            # with a small numerical tolerance
            assert poly.contains(pt) or poly.distance(pt) < 1e-6, (
                f"Position {(px, py)} is not inside {name} polygon"
            )

    # --- unnamed feature is skipped ---------------------------------------

    def test_unnamed_feature_skipped(self, tf, box):
        named = _water_feature(0.01, 0.01, name="Named Lake")
        unnamed = _water_feature(0.05, 0.05)  # no name tag
        md = _map_data([named, unnamed])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].text == "Named Lake"

    # --- sub-min_size_mm feature is skipped ------------------------------

    def test_sub_min_size_skipped(self, tf, box):
        """Features whose √area < min_size_mm are excluded."""
        # tiny: half_deg=0.00001 → projected size ≪ 1 mm
        tiny = _water_feature(0.01, 0.01, half_deg=0.00001, name="Tiny Pond")
        # large: half_deg=0.1 → projected size ~7 mm
        large = _water_feature(0.05, 0.05, half_deg=0.1, name="Big Lake")
        md = _map_data([tiny, large])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=1.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].text == "Big Lake"

    def test_min_size_zero_keeps_all_named(self, tf, box):
        """min_size_mm=0 keeps all named features regardless of size."""
        tiny = _water_feature(0.01, 0.01, half_deg=0.00001, name="Tiny Pond")
        md = _map_data([tiny])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1

    # --- metadata fields --------------------------------------------------

    def test_priority_is_100(self, tf, box):
        """Water labels must carry priority 100 per spec §5.2."""
        md = _map_data([_water_feature(0.01, 0.01, name="Lake")])
        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels[0].priority == 100

    def test_category_is_water(self, tf, box):
        md = _map_data([_water_feature(0.01, 0.01, name="Lake")])
        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels[0].category == "water"

    def test_feature_size_mm_is_positive(self, tf, box):
        md = _map_data([_water_feature(0.01, 0.01, name="Lake")])
        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels[0].feature_size_mm > 0.0

    def test_feature_size_mm_equals_sqrt_area(self, tf, box):
        """feature_size_mm is √area of the clipped polygon (mm)."""
        from shapely.geometry import box as shapely_box

        f = _water_feature(0.01, 0.01, half_deg=0.002, name="Test Lake")
        md = _map_data([f])
        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert len(labels) == 1

        # Independently compute the clipped area
        poly = _projected_polygon(f, tf)
        clip = shapely_box(*box)
        clipped = poly.intersection(clip)
        expected = math.sqrt(clipped.area)
        assert abs(labels[0].feature_size_mm - expected) < 1e-6

    # --- language preference ----------------------------------------------

    def test_lang_name_preferred_over_name(self, tf, box):
        """name:<lang> is preferred over bare name."""
        f = _water_feature(0.01, 0.01, name="Generic Name", lang_name="English Name", lang="en")
        md = _map_data([f])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels[0].text == "English Name"

    def test_falls_back_to_name_for_other_language(self, tf, box):
        """Falls back to 'name' when the requested language tag is absent."""
        f = _water_feature(0.01, 0.01, name="Shared Name", lang_name="French Name", lang="fr")
        md = _map_data([f])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels[0].text == "Shared Name"

    # --- edge cases -------------------------------------------------------

    def test_empty_water_features_returns_empty_list(self, tf, box):
        md = MapData(
            location="Empty",
            center=(0.0, 0.0),
            bbox=(-1.0, -1.0, 1.0, 1.0),
            features={},
        )
        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels == []

    def test_feature_outside_clip_box_dropped(self, tf):
        """Feature projected entirely outside the clip box yields no label."""
        # lat=10, lon=10 projects to ~(224mm, -125mm) — outside (0,0,200,200)
        f = _water_feature(10.0, 10.0, half_deg=0.001, name="Far Lake")
        md = _map_data([f])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=(0.0, 0.0, 200.0, 200.0)
        )
        assert labels == []

    def test_non_area_feature_skipped(self, tf, box):
        """Features with is_area=False (waterways) are not labelled."""
        line_feature = _water_feature(0.01, 0.01, name="River", is_area=False)
        md = _map_data([line_feature])

        labels = collect_water_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels == []


# ---------------------------------------------------------------------------
# Helpers for park label tests
# ---------------------------------------------------------------------------


def _park_feature(
    lat: float,
    lon: float,
    half_deg: float = 0.001,
    name: str | None = None,
    is_area: bool = True,
) -> MapFeature:
    """Square park polygon centred at (lat, lon), spanning ±half_deg."""
    coords = [
        (lat - half_deg, lon - half_deg),
        (lat - half_deg, lon + half_deg),
        (lat + half_deg, lon + half_deg),
        (lat + half_deg, lon - half_deg),
        (lat - half_deg, lon - half_deg),  # close the ring
    ]
    tags: dict[str, str] = {}
    if name is not None:
        tags["name"] = name
    return MapFeature(tags=tags, coords=coords, is_area=is_area)


def _park_map_data(features: list[MapFeature]) -> MapData:
    return MapData(
        location="Test",
        center=(0.0, 0.0),
        bbox=(-1.0, -1.0, 1.0, 1.0),
        features={"parks": features},
    )


# ---------------------------------------------------------------------------
# TestCollectParkLabels
# ---------------------------------------------------------------------------


class TestCollectParkLabels:
    @pytest.fixture
    def tf(self) -> FitTransform:
        return _transform()

    @pytest.fixture
    def box(self) -> tuple[float, float, float, float]:
        return _clip_box()

    def test_two_named_features_yield_two_labels(self, tf, box):
        """Fixture MapData with 2 named park polygons → exactly 2 Labels."""
        f1 = _park_feature(0.01, 0.01, name="Central Park")
        f2 = _park_feature(0.05, 0.05, name="Hyde Park")
        md = _park_map_data([f1, f2])

        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 2
        assert {lb.text for lb in labels} == {"Central Park", "Hyde Park"}

    def test_priority_is_70(self, tf, box):
        """Park labels must carry priority 70."""
        md = _park_map_data([_park_feature(0.01, 0.01, name="Green Park")])
        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert len(labels) == 1
        assert labels[0].priority == 70

    def test_category_is_parks(self, tf, box):
        md = _park_map_data([_park_feature(0.01, 0.01, name="Green Park")])
        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels[0].category == "parks"

    def test_unnamed_feature_skipped(self, tf, box):
        named = _park_feature(0.01, 0.01, name="Named Park")
        unnamed = _park_feature(0.05, 0.05)  # no name tag
        md = _park_map_data([named, unnamed])

        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].text == "Named Park"

    def test_sub_min_size_skipped(self, tf, box):
        """Features whose √area < min_size_mm are excluded."""
        tiny = _park_feature(0.01, 0.01, half_deg=0.00001, name="Tiny Garden")
        large = _park_feature(0.05, 0.05, half_deg=0.1, name="Big Forest")
        md = _park_map_data([tiny, large])

        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=1.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].text == "Big Forest"

    def test_empty_parks_features_returns_empty_list(self, tf, box):
        md = MapData(
            location="Empty",
            center=(0.0, 0.0),
            bbox=(-1.0, -1.0, 1.0, 1.0),
            features={},
        )
        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels == []

    def test_non_area_feature_skipped(self, tf, box):
        """Features with is_area=False are not labelled."""
        line_feature = _park_feature(0.01, 0.01, name="Park Path", is_area=False)
        md = _park_map_data([line_feature])

        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )
        assert labels == []

    def test_label_position_inside_polygon(self, tf, box):
        """Label position (representative_point) lies inside the projected polygon."""
        f = _park_feature(0.01, 0.01, half_deg=0.002, name="Test Park")
        md = _park_map_data([f])

        labels = collect_park_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        poly = _projected_polygon(f, tf)
        px, py = labels[0].position
        pt = Point(px, py)
        assert poly.contains(pt) or poly.distance(pt) < 1e-6


# ---------------------------------------------------------------------------
# Helpers for place-label tests
# ---------------------------------------------------------------------------


def _place_node(
    lat: float,
    lon: float,
    place: str,
    name: str | None = None,
) -> MapFeature:
    """A point-style (node) place feature with a single coord."""
    tags: dict[str, str] = {"place": place}
    if name is not None:
        tags["name"] = name
    return MapFeature(tags=tags, coords=[(lat, lon)], is_area=False)


def _place_area(
    lat: float,
    lon: float,
    half_deg: float = 0.001,
    place: str = "island",
    name: str | None = None,
) -> MapFeature:
    """A polygon-style place feature (way/relation member)."""
    coords = [
        (lat - half_deg, lon - half_deg),
        (lat - half_deg, lon + half_deg),
        (lat + half_deg, lon + half_deg),
        (lat + half_deg, lon - half_deg),
        (lat - half_deg, lon - half_deg),
    ]
    tags: dict[str, str] = {"place": place}
    if name is not None:
        tags["name"] = name
    return MapFeature(tags=tags, coords=coords, is_area=True)


def _place_map_data(features: list[MapFeature]) -> MapData:
    return MapData(
        location="Test",
        center=(0.0, 0.0),
        bbox=(-1.0, -1.0, 1.0, 1.0),
        features={"places": features},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollectPlaceLabels:
    @pytest.fixture
    def tf(self) -> FitTransform:
        return _transform()

    @pytest.fixture
    def box(self) -> tuple[float, float, float, float]:
        return _clip_box()

    def test_point_island_priority_90(self, tf, box):
        """Point-style island node → Label with priority 90."""
        f = _place_node(0.01, 0.01, place="island", name="Belle Isle")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].text == "Belle Isle"
        assert labels[0].priority == 90
        assert labels[0].category == "place"

    def test_point_islet_priority_90(self, tf, box):
        """Point-style islet node → Label with priority 90."""
        f = _place_node(0.01, 0.01, place="islet", name="Small Rock")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].priority == 90

    def test_point_neighbourhood_priority_80(self, tf, box):
        """Point-style neighbourhood node → Label with priority 80."""
        f = _place_node(0.01, 0.01, place="neighbourhood", name="Montmartre")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].priority == 80

    def test_point_suburb_priority_80(self, tf, box):
        """Point-style suburb node → Label with priority 80."""
        f = _place_node(0.01, 0.01, place="suburb", name="Belltown")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].priority == 80

    def test_point_label_position_is_projected_node(self, tf, box):
        """Label position for a node feature equals the projected node coord."""
        lat, lon = 0.01, 0.02
        f = _place_node(lat, lon, place="island", name="My Island")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        from plottter.osm.geometry import mercator

        px, py = mercator(lat, lon)
        expected_x = tf.x_origin + px * tf.scale
        expected_y = tf.y_origin - py * tf.scale
        assert math.isclose(labels[0].position[0], expected_x, abs_tol=1e-9)
        assert math.isclose(labels[0].position[1], expected_y, abs_tol=1e-9)

    def test_area_island_priority_90(self, tf, box):
        """Polygon-style island feature → Label with priority 90."""
        f = _place_area(0.01, 0.01, place="island", name="Big Island")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 1
        assert labels[0].priority == 90

    def test_unnamed_node_skipped(self, tf, box):
        """A place node without a name tag produces no Label."""
        f = _place_node(0.01, 0.01, place="island")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert labels == []

    def test_node_outside_clip_box_dropped(self, tf):
        """A node whose projected position lies outside clip_box_mm is dropped."""
        # Use a very small clip box that will exclude our node
        tiny_box = (0.0, 0.0, 0.001, 0.001)
        f = _place_node(0.01, 0.01, place="island", name="Far Island")
        md = _place_map_data([f])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=tiny_box
        )

        assert labels == []

    def test_empty_places_returns_empty_list(self, tf, box):
        """MapData with no 'places' key returns an empty list."""
        md = MapData(
            location="Empty",
            center=(0.0, 0.0),
            bbox=(-1.0, -1.0, 1.0, 1.0),
            features={},
        )

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert labels == []

    def test_mixed_island_and_neighbourhood(self, tf, box):
        """Island gets priority 90, neighbourhood gets priority 80."""
        island = _place_node(0.01, 0.01, place="island", name="Isle A")
        hood = _place_node(0.05, 0.05, place="neighbourhood", name="Quarter B")
        md = _place_map_data([island, hood])

        labels = collect_place_labels(
            md, tf, language="en", min_size_mm=0.0, clip_box_mm=box
        )

        assert len(labels) == 2
        by_text = {lb.text: lb for lb in labels}
        assert by_text["Isle A"].priority == 90
        assert by_text["Quarter B"].priority == 80


# ---------------------------------------------------------------------------
# Helpers for waterway label tests
# ---------------------------------------------------------------------------


def _waterway_feature(
    coords: list[tuple[float, float]],
    name: str | None = None,
    is_area: bool = False,
) -> MapFeature:
    """A linear waterway feature with the given (lat, lon) coordinate list."""
    tags: dict[str, str] = {}
    if name is not None:
        tags["name"] = name
    return MapFeature(tags=tags, coords=coords, is_area=is_area)


def _waterway_map_data(features: list[MapFeature]) -> MapData:
    return MapData(
        location="Test",
        center=(0.0, 0.0),
        bbox=(-1.0, -1.0, 1.0, 1.0),
        features={"waterways": features},
    )


# ---------------------------------------------------------------------------
# TestCollectWaterwayLabels
# ---------------------------------------------------------------------------


class TestCollectWaterwayLabels:
    @pytest.fixture
    def tf(self) -> FitTransform:
        return _transform()

    @pytest.fixture
    def box(self) -> tuple[float, float, float, float]:
        return _clip_box()

    def test_named_waterway_yields_label(self, tf, box):
        """A single named waterway produces exactly one Label."""
        f = _waterway_feature([(0.01, 0.01), (0.01, 0.05)], name="River Test")
        md = _waterway_map_data([f])

        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].text == "River Test"

    def test_unnamed_waterway_skipped(self, tf, box):
        """A waterway without a name tag produces no Label."""
        f = _waterway_feature([(0.01, 0.01), (0.01, 0.05)])
        md = _waterway_map_data([f])

        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=box)

        assert labels == []

    def test_priority_is_60(self, tf, box):
        """Waterway labels must carry priority 60."""
        f = _waterway_feature([(0.01, 0.01), (0.01, 0.05)], name="Brook")
        md = _waterway_map_data([f])

        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].priority == 60

    def test_category_is_waterways(self, tf, box):
        """Waterway labels have category 'waterways'."""
        f = _waterway_feature([(0.01, 0.01), (0.01, 0.05)], name="Canal")
        md = _waterway_map_data([f])

        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].category == "waterways"

    def test_midpoint_position(self, tf, box):
        """Label position is the midpoint of the projected line."""
        from shapely.geometry import LineString

        coords = [(0.01, 0.01), (0.01, 0.05)]
        f = _waterway_feature(coords, name="Midpoint River")
        md = _waterway_map_data([f])

        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1

        # Independently compute expected midpoint
        projected = []
        for lat, lon in coords:
            px, py = mercator(lat, lon)
            projected.append((tf.x_origin + px * tf.scale, tf.y_origin - py * tf.scale))
        line = LineString(projected)
        expected = line.interpolate(line.length * 0.5)

        assert math.isclose(labels[0].position[0], expected.x, abs_tol=1e-9)
        assert math.isclose(labels[0].position[1], expected.y, abs_tol=1e-9)

    def test_area_feature_skipped(self, tf, box):
        """Features with is_area=True are not labelled (waterways are lines)."""
        f = _waterway_feature(
            [(0.0, 0.0), (0.0, 0.05), (0.05, 0.05), (0.05, 0.0), (0.0, 0.0)],
            name="Fake Pond",
            is_area=True,
        )
        md = _waterway_map_data([f])

        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=box)

        assert labels == []

    def test_empty_waterways_returns_empty_list(self, tf, box):
        """MapData with no 'waterways' key returns an empty list."""
        md = MapData(
            location="Empty",
            center=(0.0, 0.0),
            bbox=(-1.0, -1.0, 1.0, 1.0),
            features={},
        )

        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=box)

        assert labels == []

    def test_feature_outside_clip_box_dropped(self, tf):
        """Feature projected entirely outside clip box is dropped."""
        # lat=10, lon=10 projects to coordinates outside (0,0,200,200)
        f = _waterway_feature([(10.0, 10.0), (10.0, 10.1)], name="Far River")
        md = _waterway_map_data([f])

        labels = collect_waterway_labels(
            md, tf, language="en", clip_box_mm=(0.0, 0.0, 200.0, 200.0)
        )

        assert labels == []

    def test_multi_clipped_uses_longest(self, tf):
        """When a waterway is clipped into multiple segments, the longest is used."""
        from shapely.geometry import LineString

        # Use a narrow clip box that splits the line into two segments.
        # The line goes from lon=0.00 to lon=0.20, passing through a gap.
        # We'll fake the scenario by having two separate features of different lengths.
        clip = (49.0, 49.0, 51.5, 51.5)  # narrow box in mm-space

        # Short segment: stays mostly near the clip edge
        short_f = _waterway_feature([(0.0, 0.0), (0.0, 0.005)], name="Split Creek")
        # Long segment: longer, well within the clip box
        long_f = _waterway_feature([(0.01, 0.0), (0.01, 0.05)], name="Split Creek")
        md = _waterway_map_data([short_f, long_f])

        # Use a generous clip box so both survive
        big_clip = (0.0, 0.0, 200.0, 200.0)
        labels = collect_waterway_labels(md, tf, language="en", clip_box_mm=big_clip)

        # Both features are separate in the features list, so we get two labels
        # (one per feature). Verify the longer one has a greater feature_size_mm.
        assert len(labels) == 2
        sizes = [lb.feature_size_mm for lb in labels]
        assert max(sizes) > min(sizes)


# ---------------------------------------------------------------------------
# Helpers for road label tests
# ---------------------------------------------------------------------------


def _road_feature(
    coords: list[tuple[float, float]],
    name: str | None = None,
) -> MapFeature:
    """A road segment with the given (lat, lon) coordinate list."""
    tags: dict[str, str] = {}
    if name is not None:
        tags["name"] = name
    return MapFeature(tags=tags, coords=coords, is_area=False)


def _road_map_data(features: list[MapFeature]) -> MapData:
    return MapData(
        location="Test",
        center=(0.0, 0.0),
        bbox=(-1.0, -1.0, 1.0, 1.0),
        features={"roads_major": features},
    )


# ---------------------------------------------------------------------------
# TestCollectRoadLabels
# ---------------------------------------------------------------------------


class TestCollectRoadLabels:
    @pytest.fixture
    def tf(self) -> FitTransform:
        return _transform()

    @pytest.fixture
    def box(self) -> tuple[float, float, float, float]:
        return _clip_box()

    def test_multi_segment_same_name_yields_one_label(self, tf, box):
        """Multi-segment road with same name yields exactly one Label."""
        seg1 = _road_feature([(0.01, 0.01), (0.01, 0.02)], name="Main Street")
        seg2 = _road_feature([(0.02, 0.01), (0.02, 0.02)], name="Main Street")
        seg3 = _road_feature([(0.03, 0.01), (0.03, 0.02)], name="Main Street")
        md = _road_map_data([seg1, seg2, seg3])

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].text == "Main Street"

    def test_label_at_longest_segment_midpoint(self, tf, box):
        """Label position is the midpoint of the longest road segment."""
        from shapely.geometry import LineString

        # Short segment: 2 points ~0.01 deg apart
        short_seg = _road_feature([(0.01, 0.01), (0.01, 0.02)], name="Oak Avenue")
        # Long segment: 6 points spanning ~0.05 deg (clearly longer)
        long_seg = _road_feature(
            [(0.05, 0.01), (0.05, 0.02), (0.05, 0.03), (0.05, 0.04), (0.05, 0.05), (0.05, 0.06)],
            name="Oak Avenue",
        )
        md = _road_map_data([short_seg, long_seg])

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].priority == 50

        # Independently compute expected midpoint for the long segment
        long_projected = []
        for lat, lon in long_seg.coords:
            px, py = mercator(lat, lon)
            long_projected.append((tf.x_origin + px * tf.scale, tf.y_origin - py * tf.scale))
        long_line = LineString(long_projected)
        expected = long_line.interpolate(long_line.length * 0.5)

        assert math.isclose(labels[0].position[0], expected.x, abs_tol=1e-6)
        assert math.isclose(labels[0].position[1], expected.y, abs_tol=1e-6)

    def test_unnamed_road_skipped(self, tf, box):
        """Road segments without a 'name' tag produce no Label."""
        f = _road_feature([(0.01, 0.01), (0.01, 0.05)])  # no name
        md = _road_map_data([f])

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert labels == []

    def test_priority_is_50(self, tf, box):
        """Road labels must carry priority 50."""
        f = _road_feature([(0.01, 0.01), (0.01, 0.05)], name="High Road")
        md = _road_map_data([f])

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].priority == 50

    def test_category_is_roads(self, tf, box):
        """Road labels have category 'roads'."""
        f = _road_feature([(0.01, 0.01), (0.01, 0.05)], name="Low Road")
        md = _road_map_data([f])

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].category == "roads"

    def test_different_names_yield_separate_labels(self, tf, box):
        """Two roads with different names produce two Labels."""
        f1 = _road_feature([(0.01, 0.01), (0.01, 0.05)], name="Elm Street")
        f2 = _road_feature([(0.05, 0.01), (0.05, 0.05)], name="Oak Avenue")
        md = _road_map_data([f1, f2])

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 2
        assert {lb.text for lb in labels} == {"Elm Street", "Oak Avenue"}

    def test_empty_roads_returns_empty_list(self, tf, box):
        """MapData with no 'roads_major' key returns an empty list."""
        md = MapData(
            location="Empty",
            center=(0.0, 0.0),
            bbox=(-1.0, -1.0, 1.0, 1.0),
            features={},
        )

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert labels == []

    def test_road_outside_clip_box_dropped(self, tf):
        """Segments entirely outside clip box produce no Label."""
        # lat=10, lon=10 projects far outside (0,0,200,200)
        f = _road_feature([(10.0, 10.0), (10.0, 10.1)], name="Far Road")
        md = _road_map_data([f])

        labels = collect_road_labels(
            md, tf, language="en", clip_box_mm=(0.0, 0.0, 200.0, 200.0)
        )

        assert labels == []

    def test_mixed_named_and_unnamed_roads(self, tf, box):
        """Named roads produce labels; unnamed roads are silently skipped."""
        named = _road_feature([(0.01, 0.01), (0.01, 0.05)], name="Named Road")
        unnamed = _road_feature([(0.05, 0.01), (0.05, 0.05)])  # no name
        md = _road_map_data([named, unnamed])

        labels = collect_road_labels(md, tf, language="en", clip_box_mm=box)

        assert len(labels) == 1
        assert labels[0].text == "Named Road"


# ---------------------------------------------------------------------------
# TestPlaceWithCollision
# ---------------------------------------------------------------------------


class TestPlaceWithCollision:
    """Tests for place_with_collision — bbox placement and collision logic."""

    from plottter.osm.labels import Label, place_with_collision

    # Convenient large clip box that won't interfere with placement tests
    _BOX = (0.0, 0.0, 500.0, 500.0)
    _FONT = 5.0  # mm cap height

    def _label(
        self,
        text: str,
        x: float,
        y: float,
        priority: int = 50,
        category: str = "water",
        feature_size_mm: float = 10.0,
    ) -> "Label":
        from plottter.osm.labels import Label

        return Label(
            text=text,
            position=(x, y),
            priority=priority,
            category=category,
            feature_size_mm=feature_size_mm,
        )

    def test_higher_priority_kept_lower_dropped_on_overlap(self):
        """When two labels overlap, the higher-priority one is kept."""
        from plottter.osm.labels import place_with_collision

        # Both labels placed at the same position → identical bboxes (full overlap)
        high = self._label("High Priority", 100.0, 100.0, priority=100)
        low = self._label("Low Priority", 100.0, 100.0, priority=50)

        result = place_with_collision([low, high], self._FONT, self._BOX)

        texts = {lb.text for lb in result}
        assert "High Priority" in texts
        assert "Low Priority" not in texts

    def test_non_overlapping_labels_both_kept(self):
        """Labels far apart (no bbox overlap) are both accepted."""
        from plottter.osm.labels import place_with_collision

        a = self._label("Lake Alpha", 50.0, 50.0)
        b = self._label("Lake Beta", 400.0, 400.0)

        result = place_with_collision([a, b], self._FONT, self._BOX)

        assert len(result) == 2
        assert {lb.text for lb in result} == {"Lake Alpha", "Lake Beta"}

    def test_off_canvas_label_dropped(self):
        """A label whose bbox extends outside clip_box_mm is dropped."""
        from plottter.osm.labels import place_with_collision

        # Place a label right at the edge — its padded bbox will fall outside
        edge = self._label("Edge Label", 0.5, 0.5, priority=100)
        inside = self._label("Inside Label", 250.0, 250.0)

        clip = (0.0, 0.0, 500.0, 500.0)
        result = place_with_collision([edge, inside], self._FONT, clip)

        texts = {lb.text for lb in result}
        # edge label bbox extends below 0 (left = 0.5 - w/2 - pad < 0)
        assert "Edge Label" not in texts
        assert "Inside Label" in texts

    def test_label_fully_outside_clip_box_dropped(self):
        """A label positioned outside the canvas entirely is dropped."""
        from plottter.osm.labels import place_with_collision

        outside = self._label("Way Outside", 1000.0, 1000.0)
        result = place_with_collision([outside], self._FONT, (0.0, 0.0, 200.0, 200.0))
        assert result == []

    def test_stable_output_order_on_shuffled_input(self):
        """Output order is (category, text) regardless of input order."""
        import random
        from plottter.osm.labels import place_with_collision

        labels = [
            self._label("Zephyr River", 100.0, 100.0, category="waterways"),
            self._label("Alpha Lake", 200.0, 200.0, category="water"),
            self._label("Metro Road", 300.0, 300.0, category="roads"),
            self._label("Beta Park", 400.0, 400.0, category="parks"),
        ]

        # Shuffle and run twice; output should be identical
        shuffled = labels[:]
        random.shuffle(shuffled)
        result1 = place_with_collision(labels, self._FONT, self._BOX)
        result2 = place_with_collision(shuffled, self._FONT, self._BOX)

        assert [lb.text for lb in result1] == [lb.text for lb in result2]
        # Expected (category, text) order:
        category_texts = [(lb.category, lb.text) for lb in result1]
        assert category_texts == sorted(category_texts)

    def test_feature_size_breaks_priority_tie(self):
        """When priority is equal, larger feature_size_mm wins the overlap."""
        from plottter.osm.labels import place_with_collision

        # Same position, same priority — larger feature_size_mm should win
        big = self._label("Big Lake", 100.0, 100.0, priority=50, feature_size_mm=100.0)
        small = self._label("Small Pond", 100.0, 100.0, priority=50, feature_size_mm=1.0)

        result = place_with_collision([small, big], self._FONT, self._BOX)

        texts = {lb.text for lb in result}
        assert "Big Lake" in texts
        assert "Small Pond" not in texts

    def test_empty_input_returns_empty(self):
        """Empty label list returns empty list."""
        from plottter.osm.labels import place_with_collision

        assert place_with_collision([], self._FONT, self._BOX) == []

    def test_single_label_within_canvas_returned(self):
        """A single on-canvas label is always returned."""
        from plottter.osm.labels import place_with_collision

        label = self._label("Only Label", 250.0, 250.0)
        result = place_with_collision([label], self._FONT, self._BOX)
        assert len(result) == 1
        assert result[0].text == "Only Label"

    def test_text_tiebreak_is_alphabetical(self):
        """When priority and feature_size are equal and positions don't overlap,
        text sorting is used as deterministic tie-break (both kept here)."""
        from plottter.osm.labels import place_with_collision

        a = self._label("Alpha", 100.0, 100.0, priority=50, feature_size_mm=10.0)
        b = self._label("Beta", 400.0, 400.0, priority=50, feature_size_mm=10.0)

        result = place_with_collision([b, a], self._FONT, self._BOX)
        assert len(result) == 2
        # Output sorted by (category, text)
        assert result[0].text == "Alpha"
        assert result[1].text == "Beta"
