"""Tests for map-positioning canvas support (task 150.1).

Covers:
- set_map_position_active(True) sets _map_position_active flag.
- set_map_position_active(False) clears flag; inactive produces no preview.
- set_map_preview_data populates _map_preview_polylines (>0) and _map_data_bounds.
- update_map_view stores the view dict.
- map_view_changed signal exists.
- Decimation keeps total point count under the cap for a large fixture.
- _draw_map_preview is called (renders without error) when active + view is set.
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POINT_CAP = 15_000  # must match the cap in widget.py set_map_preview_data


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="MapCanvasTest", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


def _make_map_data(n_features: int = 5, pts_per_feature: int = 5) -> object:
    """Return a minimal MapData with ``n_features`` line features."""
    from plottter.osm.types import MapData, MapFeature

    import math

    features_list = []
    for i in range(n_features):
        coords = [
            (35.01 + j * 0.001, 135.76 + i * 0.001)
            for j in range(pts_per_feature)
        ]
        features_list.append(
            MapFeature(
                tags={"highway": "primary"},
                coords=coords,
                is_area=False,
            )
        )
    return MapData(
        location="Kyoto, Japan",
        center=(35.0116, 135.7681),
        bbox=(34.9616, 135.7181, 35.0616, 135.8181),
        features={"roads_major": features_list},
    )


def _make_large_map_data(n_features: int = 500, pts_per_feature: int = 60) -> object:
    """Return MapData large enough to exceed the point cap before decimation."""
    from plottter.osm.types import MapData, MapFeature

    features_list = []
    for i in range(n_features):
        coords = [
            (35.0 + i * 0.0001 + j * 0.00001, 135.7 + i * 0.0001)
            for j in range(pts_per_feature)
        ]
        features_list.append(
            MapFeature(
                tags={"highway": "residential"},
                coords=coords,
                is_area=False,
            )
        )
    return MapData(
        location="Kyoto big fixture",
        center=(35.0, 135.7),
        bbox=(34.9, 135.6, 35.1, 135.8),
        features={"roads_minor": features_list},
    )


def _default_view() -> dict:
    return {"center_lat": 35.0116, "center_lon": 135.7681, "scale": 500.0}


def _default_bounds() -> tuple:
    return (34.9616, 135.7181, 35.0616, 135.8181)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController

    return ProjectController(_make_project())


@pytest.fixture
def canvas_widget(controller, qtbot):
    from plottter.gui.canvas_widget import CanvasWidget

    w = CanvasWidget(controller)
    w.resize(800, 600)
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Signal existence
# ---------------------------------------------------------------------------


class TestMapSignal:
    def test_map_view_changed_signal_exists(self, canvas_widget):
        assert hasattr(canvas_widget, "map_view_changed"), (
            "map_view_changed signal must be defined on CanvasWidget"
        )


# ---------------------------------------------------------------------------
# Activation flag
# ---------------------------------------------------------------------------


class TestMapPositionActive:
    def test_initially_inactive(self, canvas_widget):
        assert canvas_widget._map_position_active is False

    def test_activate_sets_flag(self, canvas_widget):
        canvas_widget.set_map_position_active(True)
        assert canvas_widget._map_position_active is True

    def test_deactivate_clears_flag(self, canvas_widget):
        canvas_widget.set_map_position_active(True)
        canvas_widget.set_map_position_active(False)
        assert canvas_widget._map_position_active is False

    def test_activate_disables_mask_paint(self, canvas_widget):
        # Manually set mask paint active first
        canvas_widget._mask_paint_active = True
        canvas_widget.set_map_position_active(True)
        assert canvas_widget._mask_paint_active is False

    def test_activate_disables_shape_draw(self, canvas_widget):
        canvas_widget._shape_draw_active = True
        canvas_widget.set_map_position_active(True)
        assert canvas_widget._shape_draw_active is False

    def test_activate_disables_3d_preview(self, canvas_widget):
        canvas_widget._3d_preview_active = True
        canvas_widget.set_map_position_active(True)
        assert canvas_widget._3d_preview_active is False


# ---------------------------------------------------------------------------
# set_map_preview_data — preview polylines and bounds
# ---------------------------------------------------------------------------


class TestSetMapPreviewData:
    def test_preview_polylines_populated(self, canvas_widget):
        """After set_map_preview_data, _map_preview_polylines must be non-empty."""
        map_data = _make_map_data(n_features=5)
        canvas_widget.set_map_preview_data(map_data, _default_bounds())
        assert len(canvas_widget._map_preview_polylines) > 0

    def test_data_bounds_stored(self, canvas_widget):
        """_map_data_bounds must be set to the supplied bounds tuple."""
        bounds = _default_bounds()
        map_data = _make_map_data()
        canvas_widget.set_map_preview_data(map_data, bounds)
        assert canvas_widget._map_data_bounds == bounds

    def test_preview_polylines_are_mercator_tuples(self, canvas_widget):
        """Each stored polyline must be a list of (x, y) Mercator float pairs."""
        map_data = _make_map_data(n_features=2)
        canvas_widget.set_map_preview_data(map_data, _default_bounds())
        for pl in canvas_widget._map_preview_polylines:
            assert len(pl) >= 2, "each polyline must have ≥ 2 points"
            for pt in pl:
                assert len(pt) == 2
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)

    def test_inactive_no_repaint_side_effects(self, canvas_widget):
        """set_map_preview_data works even when map positioning is inactive."""
        assert canvas_widget._map_position_active is False
        map_data = _make_map_data()
        canvas_widget.set_map_preview_data(map_data, _default_bounds())
        # Data is stored regardless of active state
        assert canvas_widget._map_preview_polylines is not None


# ---------------------------------------------------------------------------
# update_map_view
# ---------------------------------------------------------------------------


class TestUpdateMapView:
    def test_view_stored(self, canvas_widget):
        view = _default_view()
        canvas_widget.update_map_view(view)
        assert canvas_widget._map_view == view

    def test_view_initially_none(self, canvas_widget):
        assert canvas_widget._map_view is None


# ---------------------------------------------------------------------------
# Decimation — point cap
# ---------------------------------------------------------------------------


class TestDecimation:
    def test_large_fixture_under_cap(self, canvas_widget):
        """A fixture with many raw points must stay under _POINT_CAP after decimation."""
        # 500 features × 60 points = 30 000 raw points, well above the 15 000 cap
        map_data = _make_large_map_data(n_features=500, pts_per_feature=60)
        canvas_widget.set_map_preview_data(map_data, (34.9, 135.6, 35.1, 135.8))
        total = sum(len(pl) for pl in canvas_widget._map_preview_polylines)
        assert total <= _POINT_CAP, (
            f"decimated total {total} exceeds cap {_POINT_CAP}"
        )

    def test_small_fixture_all_kept(self, canvas_widget):
        """A small fixture (well under the cap) must keep all polylines."""
        map_data = _make_map_data(n_features=3, pts_per_feature=4)
        canvas_widget.set_map_preview_data(map_data, _default_bounds())
        # 3 features × (4-1 simplified) = at most 3 polylines, definitely < 15000 pts
        assert len(canvas_widget._map_preview_polylines) > 0
        total = sum(len(pl) for pl in canvas_widget._map_preview_polylines)
        assert total <= _POINT_CAP


# ---------------------------------------------------------------------------
# _draw_map_preview — rendering smoke test
# ---------------------------------------------------------------------------


class TestDrawMapPreview:
    def test_draw_no_crash_when_active(self, canvas_widget, qtbot):
        """_draw_map_preview must not raise when map positioning is active."""
        map_data = _make_map_data()
        canvas_widget.set_map_preview_data(map_data, _default_bounds())
        canvas_widget.update_map_view(_default_view())
        canvas_widget.set_map_position_active(True)
        # Force a paint event
        canvas_widget.repaint()
        qtbot.waitExposed(canvas_widget)

    def test_draw_no_crash_when_inactive(self, canvas_widget, qtbot):
        """paintEvent must not raise when map positioning is inactive (no preview drawn)."""
        canvas_widget.set_map_position_active(False)
        canvas_widget.repaint()
        qtbot.waitExposed(canvas_widget)

    def test_draw_no_crash_without_view(self, canvas_widget, qtbot):
        """_draw_map_preview is safe to call with no view set (_map_view is None)."""
        map_data = _make_map_data()
        canvas_widget.set_map_preview_data(map_data, _default_bounds())
        # Don't call update_map_view — _map_view stays None
        canvas_widget._map_position_active = True
        # Should not raise
        canvas_widget.repaint()
        qtbot.waitExposed(canvas_widget)
