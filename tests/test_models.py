"""Tests for core data models: path helpers, Canvas, Layer, Project."""

import math
import pytest

from plottter.models.path import Point, Polyline, polyline_length, polyline_bounds
from plottter.models.canvas import Canvas, PAPER_PRESETS
from plottter.models.layer import Layer
from plottter.models.project import Project


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPolylineLength:
    def test_empty_returns_zero(self):
        assert polyline_length([]) == 0.0

    def test_single_point_returns_zero(self):
        assert polyline_length([(1.0, 2.0)]) == 0.0

    def test_horizontal_segment(self):
        assert polyline_length([(0.0, 0.0), (3.0, 0.0)]) == pytest.approx(3.0)

    def test_vertical_segment(self):
        assert polyline_length([(0.0, 0.0), (0.0, 4.0)]) == pytest.approx(4.0)

    def test_diagonal_segment(self):
        # 3-4-5 right triangle
        assert polyline_length([(0.0, 0.0), (3.0, 4.0)]) == pytest.approx(5.0)

    def test_multiple_segments(self):
        # two segments of length 1 each
        pts: Polyline = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        assert polyline_length(pts) == pytest.approx(2.0)


class TestPolylineBounds:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polyline_bounds([])

    def test_single_point(self):
        min_pt, max_pt = polyline_bounds([(5.0, 3.0)])
        assert min_pt == (5.0, 3.0)
        assert max_pt == (5.0, 3.0)

    def test_rectangle(self):
        pts: Polyline = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]
        min_pt, max_pt = polyline_bounds(pts)
        assert min_pt == (0.0, 0.0)
        assert max_pt == (10.0, 5.0)

    def test_negative_coords(self):
        pts: Polyline = [(-3.0, -4.0), (2.0, 1.0)]
        min_pt, max_pt = polyline_bounds(pts)
        assert min_pt == (-3.0, -4.0)
        assert max_pt == (2.0, 1.0)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


class TestCanvas:
    def test_from_preset_a4(self):
        canvas = Canvas.from_preset("A4")
        assert canvas.width_mm == 210.0
        assert canvas.height_mm == 297.0
        assert canvas.paper_preset == "A4"
        assert canvas.margin_mm == 10.0

    def test_from_preset_custom_margin(self):
        canvas = Canvas.from_preset("A3", margin=20.0)
        assert canvas.margin_mm == 20.0
        assert canvas.width_mm == 297.0

    def test_from_preset_unknown_raises(self):
        with pytest.raises(ValueError):
            Canvas.from_preset("A1")

    def test_drawing_area(self):
        canvas = Canvas(width_mm=200.0, height_mm=300.0, margin_mm=10.0)
        left, top, right, bottom = canvas.drawing_area()
        assert left == 10.0
        assert top == 10.0
        assert right == 190.0
        assert bottom == 290.0

    def test_paper_presets_all_present(self):
        for name in ("A4", "A3", "A2", "Letter", "Legal"):
            assert name in PAPER_PRESETS

    def test_custom_canvas(self):
        canvas = Canvas(width_mm=100.0, height_mm=150.0, margin_mm=5.0, paper_preset="Custom")
        assert canvas.paper_preset == "Custom"


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


class TestLayer:
    def test_default_id_is_uuid(self):
        layer = Layer(name="Test")
        import uuid
        uuid.UUID(layer.id)  # should not raise

    def test_two_layers_have_different_ids(self):
        a = Layer(name="A")
        b = Layer(name="B")
        assert a.id != b.id

    def test_path_count(self):
        layer = Layer(name="L", paths=[[(0.0, 0.0), (1.0, 1.0)], [(2.0, 2.0)]])
        assert layer.path_count() == 2

    def test_total_point_count(self):
        layer = Layer(name="L", paths=[[(0.0, 0.0), (1.0, 1.0)], [(2.0, 2.0)]])
        assert layer.total_point_count() == 3

    def test_empty_layer_counts(self):
        layer = Layer(name="L")
        assert layer.path_count() == 0
        assert layer.total_point_count() == 0

    def test_add_paths(self):
        layer = Layer(name="L")
        layer.add_paths([[(0.0, 0.0), (1.0, 0.0)]])
        assert layer.path_count() == 1

    def test_clear_paths(self):
        layer = Layer(name="L", paths=[[(0.0, 0.0)]])
        layer.clear_paths()
        assert layer.path_count() == 0

    def test_generator_info_stored(self):
        info = {"type": "parametric", "x_expr": "sin(t)"}
        layer = Layer(name="L", generator_info=info)
        assert layer.generator_info == info

    def test_defaults(self):
        layer = Layer(name="X")
        assert layer.color == "#000000"
        assert layer.visible is True
        assert layer.locked is False
        assert layer.opacity == 1.0
        assert layer.generator_info is None


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class TestProject:
    def _make_project(self) -> Project:
        canvas = Canvas.from_preset("A4")
        return Project(name="Test", canvas=canvas)

    def test_add_layer(self):
        p = self._make_project()
        layer = Layer(name="L1")
        p.add_layer(layer)
        assert len(p.layers) == 1
        assert p.layers[0] is layer

    def test_remove_layer(self):
        p = self._make_project()
        l1 = Layer(name="L1")
        l2 = Layer(name="L2")
        p.add_layer(l1)
        p.add_layer(l2)
        p.remove_layer(l1.id)
        assert len(p.layers) == 1
        assert p.layers[0].name == "L2"

    def test_remove_unknown_is_noop(self):
        p = self._make_project()
        p.remove_layer("nonexistent-id")  # should not raise

    def test_reorder_layer(self):
        p = self._make_project()
        l1, l2, l3 = Layer(name="A"), Layer(name="B"), Layer(name="C")
        for l in (l1, l2, l3):
            p.add_layer(l)
        p.reorder_layer(l1.id, 2)
        assert [l.name for l in p.layers] == ["B", "C", "A"]

    def test_get_layer(self):
        p = self._make_project()
        l1 = Layer(name="L1")
        p.add_layer(l1)
        assert p.get_layer(l1.id) is l1

    def test_get_layer_missing(self):
        p = self._make_project()
        assert p.get_layer("no-such-id") is None

    def test_duplicate_layer(self):
        p = self._make_project()
        l1 = Layer(name="L1", paths=[[(0.0, 0.0), (1.0, 1.0)]])
        p.add_layer(l1)
        dup = p.duplicate_layer(l1.id)
        assert dup.id != l1.id
        assert dup.name == "L1 copy"
        assert dup.paths == l1.paths
        # Ensure deep copy — mutating original doesn't affect duplicate
        l1.paths[0].append((99.0, 99.0))
        assert len(dup.paths[0]) == 2

    def test_duplicate_layer_unknown_raises(self):
        p = self._make_project()
        with pytest.raises(ValueError):
            p.duplicate_layer("no-such-id")

    def test_merge_layers(self):
        p = self._make_project()
        l1 = Layer(name="L1", color="#FF0000", paths=[[(0.0, 0.0), (1.0, 1.0)]])
        l2 = Layer(name="L2", color="#0000FF", paths=[[(2.0, 2.0), (3.0, 3.0)]])
        p.add_layer(l1)
        p.add_layer(l2)
        merged = p.merge_layers([l1.id, l2.id])
        assert merged.path_count() == 2
        assert merged.color == "#FF0000"  # first layer color

    def test_active_layer_prefers_visible_unlocked(self):
        p = self._make_project()
        l1 = Layer(name="locked", locked=True)
        l2 = Layer(name="free")
        p.add_layer(l1)
        p.add_layer(l2)
        assert p.active_layer is l2

    def test_active_layer_falls_back_to_first(self):
        p = self._make_project()
        l1 = Layer(name="locked", locked=True)
        p.add_layer(l1)
        assert p.active_layer is l1

    def test_active_layer_none_when_empty(self):
        p = self._make_project()
        assert p.active_layer is None
