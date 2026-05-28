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
from plottter.osm.labels import Label, _resolve_name, collect_water_labels
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
