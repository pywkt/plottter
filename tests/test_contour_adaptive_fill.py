"""Solid (Adaptive) fill for Contour Lines / Line Art Trace.

Perimeter rings follow each shape's edge, a zigzag fills the remaining
core, and everything belonging to one shape is chained into as few paths as
possible.  Thin strokes are covered by the rings alone.  The point of the
feature is stroke count: a line drawing that Solid Fill cut into ~38k
dashes comes out at a tenth of that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from plottter.generators.contour._adaptive_fill import (
    _zigzag_fill,
    fill_polygon_adaptive,
)
from plottter.generators.contour._fills import _fill_polygon_hatch


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _ink_length(paths):
    return sum(math.dist(p[i], p[i + 1]) for p in paths for i in range(len(p) - 1))


def _all_inside(paths, poly, tol=1e-6):
    shape = poly.buffer(tol)
    return all(shape.covers(LineString(p)) for p in paths if len(p) >= 2)


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def test_broad_rectangle_is_rings_plus_one_zigzag():
    paths = fill_polygon_adaptive(_rect(0, 0, 40, 20), [], 0.5, perimeter_rings=2)
    # Two rings and the zigzag are chained into a single path.
    assert len(paths) == 1
    poly = Polygon(_rect(0, 0, 40, 20))
    assert _all_inside(paths, poly)
    # Coverage: the fill reaches close to every interior point.
    probe = [(x, y) for x in np.arange(1, 39, 1.3) for y in np.arange(1, 19, 1.1)]
    lines = [LineString(p) for p in paths]
    assert all(min(ln.distance(Point(pt)) for ln in lines) <= 0.5 for pt in probe)


def test_thin_stroke_is_covered_by_rings_alone_as_one_path():
    # 1 mm wide, 30 mm long stroke at 0.3 mm spacing with 2 rings:
    # offsets at 0.3 and 0.6 exist, the core at 0.6 inset is 0 wide -> gone.
    paths = fill_polygon_adaptive(_rect(0, 0, 30, 1.0), [], 0.3, perimeter_rings=2)
    assert len(paths) == 1
    assert _all_inside(paths, Polygon(_rect(0, 0, 30, 1.0)))
    # Rings run *along* the stroke, so the path is long and has few points
    # relative to a scanline fill of the same shape.
    scan = _fill_polygon_hatch(_rect(0, 0, 30, 1.0), [], 0.0, 0.3)
    assert len(scan) >= 3 and len(paths) < len(scan)


def test_diagonal_thin_stroke_far_fewer_paths_than_solid():
    """The case that motivated the feature: a thin diagonal stroke that
    horizontal scanlines chop into hundreds of dashes."""
    # 1.2 mm wide stroke at 30 degrees, 60 mm long.
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    base = [(0, 0), (60, 0), (60, 1.2), (0, 1.2), (0, 0)]
    stroke = [(c * x - s * y, s * x + c * y) for x, y in base]
    solid = _fill_polygon_hatch(stroke, [], 0.0, 0.3)
    adaptive = fill_polygon_adaptive(stroke, [], 0.3, perimeter_rings=2)
    assert len(solid) > 50
    assert len(adaptive) <= 3
    assert _all_inside(adaptive, Polygon(stroke))


def test_holes_are_respected_and_hole_rings_drawn():
    outer = _rect(0, 0, 30, 30)
    hole = _rect(10, 10, 20, 20)
    paths = fill_polygon_adaptive(outer, [hole], 0.5, perimeter_rings=1)
    poly = Polygon(outer, [hole])
    assert _all_inside(paths, poly)
    # Something runs along the hole's inner edge (the ring offset into the ink).
    hole_ring = LineString(_rect(9.5, 9.5, 20.5, 20.5))
    assert any(LineString(p).distance(hole_ring) < 1e-6 for p in paths)


def test_zero_rings_is_zigzag_only_and_chains_rows():
    paths = fill_polygon_adaptive(_rect(0, 0, 20, 10), [], 0.5, perimeter_rings=0)
    assert len(paths) == 1
    ys = sorted({round(y, 6) for x, y in paths[0]})
    assert len(ys) >= 18  # ~ 10 / 0.5 rows
    # Boustrophedon: consecutive rows alternate direction.
    xs = [x for x, y in paths[0]]
    dirs = [np.sign(xs[i + 1] - xs[i]) for i in range(0, len(xs) - 1, 2)]
    assert all(dirs[i] != dirs[i + 1] for i in range(len(dirs) - 1))


def test_zigzag_does_not_bridge_across_a_gap():
    """A U shape: rows at the top have two separate spans. Chaining must not
    draw a connector across the white gap between the arms."""
    u = [(0, 0), (30, 0), (30, 30), (20, 30), (20, 10), (10, 10), (10, 30), (0, 30), (0, 0)]
    poly = Polygon(u)
    chains = _zigzag_fill(poly, poly, 0.5, 0.0)
    assert _all_inside(chains, poly)
    assert len(chains) <= 3  # base + two arms at most


def test_angle_rotates_the_zigzag():
    paths = fill_polygon_adaptive(_rect(0, 0, 20, 20), [], 0.5, perimeter_rings=0, angle_deg=90.0)
    assert _all_inside(paths, Polygon(_rect(0, 0, 20, 20)))
    xs = sorted({round(x, 6) for x, y in paths[0]})
    assert len(xs) >= 38  # columns, not rows


def test_shallow_edge_rows_still_chain():
    """An edge nearly parallel to the rows makes consecutive rows differ
    wildly in length; the next row's nearest end is then far from the chain
    end.  That connector runs inside the ink, so it must still be taken."""
    slanted = [(0, 0), (140, 1.0), (140, 60), (0, 61), (0, 0)]
    paths = fill_polygon_adaptive(slanted, [], 0.5, perimeter_rings=0)
    assert len(paths) == 1
    assert _all_inside(paths, Polygon(slanted))


def test_degenerate_inputs():
    assert fill_polygon_adaptive([(0, 0), (1, 1)], [], 0.3) == []
    assert fill_polygon_adaptive(_rect(0, 0, 5, 5), [], 0.0) == []
    # Tiny shape smaller than one ring: rings consume it, no crash, no output.
    assert fill_polygon_adaptive(_rect(0, 0, 0.2, 0.2), [], 0.3, perimeter_rings=1) == []


def test_multipolygon_from_self_intersection_is_handled():
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)]
    paths = fill_polygon_adaptive(bowtie, [], 0.5, perimeter_rings=1)
    assert paths
    fixed = Polygon(bowtie).buffer(0)
    assert _all_inside(paths, fixed)


# ----------------------------------------------------------------------
# Generator wiring
# ----------------------------------------------------------------------


@pytest.fixture
def gen():
    from plottter.generators.contour import ContourGenerator

    return ContourGenerator()


def _blob_image(h=120, w=160):
    import cv2

    img = np.full((h, w), 255, np.uint8)
    cv2.circle(img, (50, 60), 30, 0, -1)          # broad blob
    cv2.line(img, (90, 20), (150, 100), 0, 3)     # thin diagonal stroke
    return img


def test_fill_choice_and_rings_param_exist(gen):
    params = {p.name: p for p in gen.get_parameters()}
    assert "Solid (Adaptive)" in params["fill"].choices
    rings = params["fill_perimeter_rings"]
    assert rings.default == 2 and rings.min == 0
    assert rings.visible_when == {"mode": ["Line Art Trace"], "fill": ["Solid (Adaptive)"]}
    assert "Solid (Adaptive)" in params["fill_angle"].visible_when["fill"]
    assert "Solid (Adaptive)" in params["fill_spacing_mm"].visible_when["fill"]


def test_preset_exists_and_original_solid_preset_untouched(gen):
    presets = {p.name: p for p in gen.get_presets()}
    new = presets["Line Art / Solid Fill (Adaptive)"]
    old = presets["Line Art / Solid Fill"]
    assert new.params["fill"] == "Solid (Adaptive)"
    assert new.params["fill_perimeter_rings"] == 2
    assert old.params["fill"] == "Solid"
    assert "fill_perimeter_rings" not in old.params
    # Same settings otherwise.
    skip = ("fill", "fill_perimeter_rings")
    assert {k: v for k, v in new.params.items() if k not in skip} == {k: v for k, v in old.params.items() if k not in skip}


def test_generator_adaptive_uses_far_fewer_paths_than_solid(gen):
    from plottter.models import Canvas

    canvas = Canvas.from_preset("A4")
    base = {
        "mode": "Line Art Trace", "trace_threshold": 128, "smooth_iterations": 1,
        "simplify_mm": 0.2, "min_contour_px": 5, "fill_spacing_mm": 0.3,
        "fill_angle": 0.0, "_source_image": _blob_image(), "image_fit_mode": "fill",
    }
    solid = gen.generate({**base, "fill": "Solid"}, canvas)
    adaptive = gen.generate({**base, "fill": "Solid (Adaptive)", "fill_perimeter_rings": 2}, canvas)
    assert len(solid) > 100
    assert len(adaptive) < len(solid) / 5
    # Same ink coverage within ~15%.
    assert abs(_ink_length(adaptive) - _ink_length(solid)) / _ink_length(solid) < 0.15


# ----------------------------------------------------------------------
# "Block Fill (Zigzag)" preset — colour-separation blocks
# ----------------------------------------------------------------------


def test_block_fill_preset_fills_light_and_dark_layers(gen):
    """A separated layer arrives as its own tone on white. The preset's
    threshold of 254 must trace a pale layer (tone ~226) as well as a dark one."""
    from plottter.models import Canvas

    preset = {p.name: p for p in gen.get_presets()}["Block Fill (Zigzag)"]
    assert preset.params["fill"] == "Solid (Adaptive)"
    assert preset.params["trace_threshold"] == 254
    assert preset.params["fill_perimeter_rings"] == 1
    canvas = Canvas.from_preset("A4")
    import cv2

    counts = {}
    for tone in (226, 78):
        img = np.full((120, 160), 255, np.uint8)
        cv2.rectangle(img, (20, 20), (140, 100), int(tone), -1)
        paths = gen.generate({**preset.params, "_source_image": img, "image_fit_mode": "fill"}, canvas)
        counts[tone] = len(paths)
        assert 2 <= len(paths) <= 6, (tone, len(paths))  # outline + ring/zigzag chain
        assert _ink_length(paths) > 2000  # the block is actually filled
    assert counts[226] == counts[78]


def test_trace_outline_off_gives_bare_fill(gen):
    """Perimeter Rings 0 alone still left an edge line: the traced outline is
    added for every fill style.  fill_outline=False removes it."""
    from plottter.models import Canvas
    import cv2

    canvas = Canvas.from_preset("A4")
    img = np.full((120, 160), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (140, 100), 0, -1)
    base = {
        "mode": "Line Art Trace", "trace_threshold": 128, "smooth_iterations": 0,
        "simplify_mm": 0.2, "min_contour_px": 5, "fill": "Solid (Adaptive)",
        "fill_perimeter_rings": 0, "fill_spacing_mm": 0.5, "fill_angle": 0.0,
        "_source_image": img, "image_fit_mode": "fill",
    }
    with_outline = gen.generate({**base, "fill_outline": True}, canvas)
    bare = gen.generate({**base, "fill_outline": False}, canvas)
    assert len(with_outline) == 2  # outline + one zigzag
    assert len(bare) == 1
    # The remaining path is the zigzag (many rows), not the closed outline.
    assert bare[0][0] != bare[0][-1]
    assert len({round(y, 3) for _, y in bare[0]}) > 20
    params = {p.name: p for p in gen.get_parameters()}
    assert params["fill_outline"].default is True
    assert all("fill_outline" in p.params for p in gen.get_presets() if p.params.get("mode") == "Line Art Trace")
