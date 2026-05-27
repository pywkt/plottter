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


def test_assembly():
    """Verify assemble() per spec §6.4.

    Three invariants are checked:

    1. An open way (is_area=False) stays open — first point != last point.
    2. A closed way (is_area=True) returns a ring where first == last point.
    3. A multipolygon relation with an inner ring returns that inner ring in
       the ``inner_holes`` list.
    """
    from plottter.osm.geometry import assemble
    from plottter.osm.types import MapFeature

    # ── 1. Open way stays open ────────────────────────────────────────────────
    open_feature = MapFeature(
        tags={},
        coords=[(35.0, 135.0), (35.1, 135.1), (35.2, 135.0)],
        is_area=False,
    )
    rings, holes = assemble(open_feature)
    assert len(rings) == 1, f"Expected 1 polyline from open way, got {len(rings)}"
    assert rings[0][0] != rings[0][-1], "Open way must NOT be closed (first != last)"
    assert holes == [], "Open way must produce no inner holes"

    # ── 2. Closed way: first == last ──────────────────────────────────────────
    # Provide coords that are NOT yet closed; assemble() must close them.
    closed_feature = MapFeature(
        tags={"building": "yes"},
        coords=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        is_area=True,
    )
    rings, holes = assemble(closed_feature)
    assert len(rings) == 1, f"Expected 1 ring from closed way, got {len(rings)}"
    assert rings[0][0] == rings[0][-1], "Closed way ring must have first == last point"
    assert holes == [], "Simple closed way must produce no inner holes"

    # Also works when coords are already closed (idempotent).
    pre_closed = MapFeature(
        tags={"building": "yes"},
        coords=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
        is_area=True,
    )
    rings2, _ = assemble(pre_closed)
    assert rings2[0][0] == rings2[0][-1], "Pre-closed ring must remain closed"

    # ── 3. Relation with an inner ring → inner_holes ──────────────────────────
    outer_coords = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    inner_ring = [(2.0, 2.0), (2.0, 5.0), (5.0, 5.0), (5.0, 2.0), (2.0, 2.0)]
    relation_feature = MapFeature(
        tags={"natural": "water"},
        coords=outer_coords,
        is_area=True,
        inner_coords=[inner_ring],
    )
    rings, holes = assemble(relation_feature)
    assert len(rings) == 1, f"Expected 1 outer ring from relation, got {len(rings)}"
    assert rings[0][0] == rings[0][-1], "Relation outer ring must be closed"
    assert len(holes) == 1, f"Expected 1 inner hole from relation, got {len(holes)}"
    assert holes[0] == inner_ring, "Inner hole must match the stored inner_coords ring"


def test_inverse_mercator_round_trip():
    """inverse_mercator() is the exact inverse of mercator() to within 1e-9."""
    from plottter.osm.geometry import inverse_mercator

    test_cases = [
        (0.0, 0.0),
        (45.0, 90.0),
        (-45.0, -90.0),
        (85.0, 180.0),
        (-85.0, -180.0),
        (85.05112877980659, 0.0),   # lat clamp boundary
        (-85.05112877980659, 0.0),  # lat clamp boundary
        (33.4489, -112.0741),       # Phoenix
        (51.5074, -0.1278),         # London
    ]

    for lat, lon in test_cases:
        x, y = mercator(lat, lon)
        lat2, lon2 = inverse_mercator(x, y)
        # lon round-trips exactly (no clamping on lon)
        assert abs(lon2 - lon) < 1e-9, (
            f"lon round-trip failed for ({lat}, {lon}): got {lon2}"
        )
        # lat may be clamped by mercator; round-trip from clamped value
        lat_clamped = max(-85.05112877980659, min(85.05112877980659, lat))
        assert abs(lat2 - lat_clamped) < 1e-9, (
            f"lat round-trip failed for ({lat}, {lon}): "
            f"expected {lat_clamped}, got {lat2}"
        )


def test_view_transform_centre_placement():
    """view_transform() places the centre lat/lon at the printable-area centre."""
    from plottter.osm.geometry import view_transform, inverse_mercator
    from plottter.models.canvas import Canvas

    canvas = Canvas(width_mm=200.0, height_mm=150.0, margin_mm=10.0)
    left, top, right, bottom = canvas.drawing_area()
    ccx = (left + right) / 2
    ccy = (top + bottom) / 2

    center_lat, center_lon = 48.8566, 2.3522  # Paris
    scale = 50.0  # arbitrary mm per Mercator unit

    transform = view_transform(center_lat, center_lon, scale, canvas)

    # Apply the transform formula to (center_lat, center_lon)
    mcx, mcy = mercator(center_lat, center_lon)
    canvas_x = transform.x_origin + mcx * transform.scale
    canvas_y = transform.y_origin - mcy * transform.scale

    assert abs(canvas_x - ccx) < 1e-9, (
        f"centre x not at printable-area centre: got {canvas_x}, expected {ccx}"
    )
    assert abs(canvas_y - ccy) < 1e-9, (
        f"centre y not at printable-area centre: got {canvas_y}, expected {ccy}"
    )


def test_view_transform_scale_and_offset():
    """view_transform() applies scale correctly for a known nearby point."""
    from plottter.osm.geometry import view_transform
    from plottter.models.canvas import Canvas

    canvas = Canvas(width_mm=200.0, height_mm=150.0, margin_mm=10.0)
    left, top, right, bottom = canvas.drawing_area()
    ccx = (left + right) / 2
    ccy = (top + bottom) / 2

    center_lat, center_lon = 0.0, 0.0  # equator / prime meridian
    scale = 100.0

    transform = view_transform(center_lat, center_lon, scale, canvas)

    # A point 1 radian east and 1 radian north in Mercator space should be
    # offset by exactly (scale, -scale) from the canvas centre (y-flipped).
    # mercator(lat2, lon2) = (1.0, 1.0) when lon2 = degrees(1) and lat is chosen
    # such that the Mercator y == 1.  We just use the transform directly.
    test_mx, test_my = 1.0, 1.0
    canvas_x = transform.x_origin + test_mx * transform.scale
    canvas_y = transform.y_origin - test_my * transform.scale

    assert abs(canvas_x - (ccx + scale * 1.0)) < 1e-9, (
        f"x offset wrong: got {canvas_x}, expected {ccx + scale}"
    )
    assert abs(canvas_y - (ccy - scale * 1.0)) < 1e-9, (
        f"y offset wrong (north up): got {canvas_y}, expected {ccy - scale}"
    )


def test_default_map_view_fit_equivalence():
    """default_map_view() + view_transform() frames features identically to fit_transform().

    Spec §3.3 / test §8 (phase 149.2): all projected coordinates produced by
    the view_transform built from default_map_view must match those produced
    directly by fit_transform() within 1e-6 mm.
    """
    from plottter.osm.geometry import (
        default_map_view,
        fit_transform,
        project_feature,
        view_transform,
    )
    from plottter.osm.types import MapFeature
    from plottter.models.canvas import Canvas

    # A fixture feature set spanning an asymmetric geographic region so that
    # neither axis trivially cancels.
    features = [
        MapFeature(
            tags={},
            coords=[
                (48.85, 2.35),   # Paris-ish
                (48.90, 2.40),
                (48.80, 2.30),
                (48.95, 2.28),
            ],
            is_area=False,
        ),
        MapFeature(
            tags={},
            coords=[
                (48.87, 2.36),
                (48.83, 2.42),
            ],
            is_area=False,
        ),
    ]

    canvas = Canvas(width_mm=200.0, height_mm=150.0, margin_mm=10.0)

    # Ground-truth transform from fit_transform.
    ft = fit_transform(features, canvas)
    fit_coords = [project_feature(f, ft) for f in features]

    # Transform built from the default_map_view.
    view = default_map_view(features, canvas)
    assert "center_lat" in view
    assert "center_lon" in view
    assert "scale" in view

    vt = view_transform(view["center_lat"], view["center_lon"], view["scale"], canvas)
    view_coords = [project_feature(f, vt) for f in features]

    # Every projected coordinate must agree within 1e-6 mm.
    for feat_idx, (fc, vc) in enumerate(zip(fit_coords, view_coords)):
        assert len(fc) == len(vc), (
            f"feature {feat_idx}: point count mismatch {len(fc)} vs {len(vc)}"
        )
        for pt_idx, ((fx, fy), (vx, vy)) in enumerate(zip(fc, vc)):
            assert abs(fx - vx) < 1e-6, (
                f"feature {feat_idx} point {pt_idx}: x mismatch "
                f"fit={fx:.9f} view={vx:.9f} diff={abs(fx - vx):.2e}"
            )
            assert abs(fy - vy) < 1e-6, (
                f"feature {feat_idx} point {pt_idx}: y mismatch "
                f"fit={fy:.9f} view={vy:.9f} diff={abs(fy - vy):.2e}"
            )
