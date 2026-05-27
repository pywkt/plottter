"""Tests for plottter.osm.geometry — Web Mercator projection."""

import math
import pytest

from plottter.osm.geometry import mercator


def test_mercator():
    """Verify the Web Mercator projection function per spec §6.1."""

    # --- x strictly increases with longitude ---
    x0, _ = mercator(0.0, -90.0)
    x1, _ = mercator(0.0, 0.0)
    x2, _ = mercator(0.0, 90.0)
    assert x0 < x1 < x2, "x must strictly increase with longitude"

    # --- y strictly increases with latitude ---
    _, y0 = mercator(-45.0, 0.0)
    _, y1 = mercator(0.0, 0.0)
    _, y2 = mercator(45.0, 0.0)
    assert y0 < y1 < y2, "y must strictly increase with latitude"

    # --- clamping prevents domain error at ±90° ---
    # math.log(math.tan(math.pi/4 + math.radians(90)/2)) would be log(tan(pi/2)) = log(inf) = inf
    # With clamping, these must return finite values without raising.
    x_north, y_north = mercator(90.0, 0.0)
    x_south, y_south = mercator(-90.0, 0.0)
    assert math.isfinite(y_north), "y must be finite when lat=90 (clamped)"
    assert math.isfinite(y_south), "y must be finite when lat=-90 (clamped)"

    # --- hand-computed reference: (lat=0, lon=0) → (0, 0) ---
    x_ref, y_ref = mercator(0.0, 0.0)
    assert abs(x_ref - 0.0) < 1e-6, f"x at equator/meridian: expected 0, got {x_ref}"
    assert abs(y_ref - 0.0) < 1e-6, f"y at equator/meridian: expected 0, got {y_ref}"

    # --- hand-computed reference: (lat=45, lon=90) ---
    # x = radians(90) = pi/2
    # y = log(tan(pi/4 + pi/8)) = log(tan(3*pi/8)) = log(1 + sqrt(2))
    expected_x = math.pi / 2
    expected_y = math.log(1.0 + math.sqrt(2.0))
    x_45_90, y_45_90 = mercator(45.0, 90.0)
    assert abs(x_45_90 - expected_x) < 1e-6, (
        f"x at (45, 90): expected {expected_x}, got {x_45_90}"
    )
    assert abs(y_45_90 - expected_y) < 1e-6, (
        f"y at (45, 90): expected {expected_y}, got {y_45_90}"
    )


def test_fit_to_canvas():
    """Verify fit_transform() and project_feature() per spec §6.2.

    Uses a small square of features with a known geographic extent and checks:
    - All projected mm coordinates fall inside canvas.drawing_area().
    - Aspect ratio is preserved (equal x/y scale).
    - Northernmost point maps to a smaller canvas-y than southernmost (north-up).
    """
    from plottter.osm.geometry import fit_transform, project_feature
    from plottter.osm.types import MapFeature
    from plottter.models.canvas import Canvas

    south, north = 10.0, 10.01
    west, east = 20.0, 20.01

    # A closed square feature whose corners span the geographic bbox.
    feature = MapFeature(
        tags={},
        coords=[
            (south, west),
            (south, east),
            (north, east),
            (north, west),
            (south, west),  # closed ring
        ],
        is_area=True,
    )

    # 200 × 150 mm canvas with 10 mm margins → drawing area (10, 10, 190, 140)
    canvas = Canvas(width_mm=200.0, height_mm=150.0, margin_mm=10.0)
    left, top, right, bottom = canvas.drawing_area()

    transform = fit_transform([feature], canvas)
    mm_coords = project_feature(feature, transform)

    xs = [p[0] for p in mm_coords]
    ys = [p[1] for p in mm_coords]

    # ── All points inside the printable area ──────────────────────────────────
    tol = 1e-6
    assert min(xs) >= left - tol, f"left edge violated: {min(xs)} < {left}"
    assert max(xs) <= right + tol, f"right edge violated: {max(xs)} > {right}"
    assert min(ys) >= top - tol, f"top edge violated: {min(ys)} < {top}"
    assert max(ys) <= bottom + tol, f"bottom edge violated: {max(ys)} > {bottom}"

    # ── Aspect ratio preserved (uniform scale) ────────────────────────────────
    # Ratio of canvas spans must equal ratio of Mercator spans.
    px_w, _ = mercator(south, west)
    px_e, _ = mercator(south, east)
    _, py_s = mercator(south, west)
    _, py_n = mercator(north, west)
    merc_span_x = px_e - px_w
    merc_span_y = py_n - py_s

    proj_span_x = max(xs) - min(xs)
    proj_span_y = max(ys) - min(ys)

    # Both spans are non-zero for this square input.
    ratio_merc = merc_span_x / merc_span_y
    ratio_proj = proj_span_x / proj_span_y
    assert abs(ratio_proj - ratio_merc) < 1e-6, (
        f"aspect ratio not preserved: canvas ratio={ratio_proj:.6f}, "
        f"mercator ratio={ratio_merc:.6f}"
    )

    # ── North-up: northern coords → smaller canvas-y ──────────────────────────
    north_ys = [mm_coords[i][1] for i, (lat, _) in enumerate(feature.coords) if lat == north]
    south_ys = [mm_coords[i][1] for i, (lat, _) in enumerate(feature.coords) if lat == south]

    assert north_ys and south_ys, "Could not identify northern/southern points"
    assert min(north_ys) < max(south_ys), (
        f"y-flip wrong: northernmost canvas-y {min(north_ys):.3f} "
        f"should be less than southernmost {max(south_ys):.3f}"
    )


def test_clip():
    """Verify clip_lines() per spec §6.3.

    Three invariants are checked:

    1. A line crossing the bbox edge is trimmed to the boundary.
    2. A sub-min_len_mm fragment is dropped.
    3. A multipart intersection yields >1 polyline.
    """
    from plottter.osm.geometry import clip_lines, clip_polygons

    # bbox (left, top, right, bottom) in canvas mm
    bbox = (0.0, 0.0, 100.0, 100.0)

    # ── 1. Line crossing bbox edge is trimmed ─────────────────────────────────
    # Horizontal line from x=-10 to x=110 at y=50 — crosses both left and right edges.
    line_crossing = [(-10.0, 50.0), (110.0, 50.0)]
    result = clip_lines([line_crossing], bbox, min_len_mm=0.0)
    assert len(result) == 1, f"Expected 1 trimmed segment, got {len(result)}"
    xs = [p[0] for p in result[0]]
    assert min(xs) >= 0.0 - 1e-9, f"Left edge not trimmed: min x = {min(xs)}"
    assert max(xs) <= 100.0 + 1e-9, f"Right edge not trimmed: max x = {max(xs)}"

    # ── 2. Sub-min_len_mm fragment is dropped ─────────────────────────────────
    # 0.5 mm line well inside the bbox — shorter than min_len_mm=1.0
    tiny_line = [(50.0, 50.0), (50.5, 50.0)]
    result = clip_lines([tiny_line], bbox, min_len_mm=1.0)
    assert len(result) == 0, (
        f"Expected 0 results (fragment shorter than min_len_mm dropped), got {len(result)}"
    )

    # ── 3. Multipart intersection yields >1 polyline ──────────────────────────
    # A polyline that starts inside the bbox, exits through the bottom edge
    # (y > 100), then re-enters — producing two separate clipped segments.
    #
    # Path: (10, 50) → (50, 150) → (90, 50)
    #   Segment A exits at (30, 100); segment B re-enters at (70, 100).
    multi_line = [(10.0, 50.0), (50.0, 150.0), (90.0, 50.0)]
    result = clip_lines([multi_line], bbox, min_len_mm=0.0)
    assert len(result) > 1, (
        f"Expected >1 polylines from multipart clip, got {len(result)}"
    )

    # ── 4. clip_polygons: a polygon extending outside bbox is trimmed ─────────
    # A large square centred on the bbox — extends 20 mm beyond every edge.
    large_square = [
        (-20.0, -20.0),
        (120.0, -20.0),
        (120.0, 120.0),
        (-20.0, 120.0),
        (-20.0, -20.0),
    ]
    result_poly = clip_polygons([large_square], bbox, min_len_mm=0.0)
    assert len(result_poly) == 1, f"Expected 1 clipped polygon, got {len(result_poly)}"
    all_xs = [p[0] for p in result_poly[0]]
    all_ys = [p[1] for p in result_poly[0]]
    assert min(all_xs) >= 0.0 - 1e-9, "Polygon left edge not clipped"
    assert max(all_xs) <= 100.0 + 1e-9, "Polygon right edge not clipped"
    assert min(all_ys) >= 0.0 - 1e-9, "Polygon top edge not clipped"
    assert max(all_ys) <= 100.0 + 1e-9, "Polygon bottom edge not clipped"
