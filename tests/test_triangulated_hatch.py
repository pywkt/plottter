"""Tests for TriangulatedHatchGenerator — edge-aware seed point placement."""

from __future__ import annotations

import numpy as np

from plottter.generators.triangulated_hatch import (
    TriangulatedHatchGenerator,
    _compute_angle_map,
    _discard_outside_triangles,
    _edge_aware_seeds,
    _triangulate_and_sample,
)
from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def make_gray_with_edges() -> np.ndarray:
    """100x100 grayscale image with a strong black rectangle in the centre — lots of edges."""
    img = np.full((100, 100), 200, dtype=np.uint8)
    img[30:70, 30:70] = 0  # dark square → strong Canny edges on the border
    return img


def make_uniform_gray() -> np.ndarray:
    """100x100 uniform mid-gray — almost no Canny edges."""
    return np.full((100, 100), 128, dtype=np.uint8)


def img_rect_for_canvas(canvas: Canvas) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = canvas.drawing_area()
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Tests for _edge_aware_seeds
# ---------------------------------------------------------------------------


def test_seeds_within_bounds() -> None:
    """All generated seed points must lie within the image rect."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(0)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=200, edge_weight=0.7, rng=rng)

    x1, y1, x2, y2 = img_rect
    tol = 1e-9
    assert np.all(seeds[:, 0] >= x1 - tol), "Some x coords are below x1"
    assert np.all(seeds[:, 0] <= x2 + tol), "Some x coords are above x2"
    assert np.all(seeds[:, 1] >= y1 - tol), "Some y coords are below y1"
    assert np.all(seeds[:, 1] <= y2 + tol), "Some y coords are above y2"


def test_corners_always_included() -> None:
    """The 4 corners of the image rect must always be present in the seeds."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(1)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=50, edge_weight=0.5, rng=rng)

    x1, y1, x2, y2 = img_rect
    expected_corners = [
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
    ]
    seed_set = set(map(tuple, seeds.tolist()))
    for corner in expected_corners:
        assert corner in seed_set, f"Corner {corner} missing from seeds"


def test_approximately_num_points_generated() -> None:
    """The returned array should have exactly num_points + 4 corners entries."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(2)

    num_points = 300
    seeds = _edge_aware_seeds(gray, img_rect, num_points=num_points, edge_weight=0.7, rng=rng)

    # 4 corners + num_points interior seeds
    assert seeds.shape == (num_points + 4, 2)


def test_edge_weight_1_more_points_near_edges() -> None:
    """With edge_weight=1, more seed points should cluster near edges than with edge_weight=0."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()

    num_points = 500

    # edge_weight=1: attracted to edges
    rng1 = np.random.default_rng(10)
    seeds_edge = _edge_aware_seeds(gray, img_rect, num_points=num_points, edge_weight=1.0, rng=rng1)

    # edge_weight=0: uniform
    rng2 = np.random.default_rng(10)
    seeds_uniform = _edge_aware_seeds(gray, img_rect, num_points=num_points, edge_weight=0.0, rng=rng2)

    # The edges of the dark square in the image are around rows/cols 30 and 70 (in pixel space).
    # Convert to mm to define "edge zone" and count how many points fall in it.
    import cv2
    edges_map = cv2.Canny(gray, 50, 150)
    edge_density = cv2.GaussianBlur(edges_map.astype(np.float32), (0, 0), 3)

    x1, y1, x2, y2 = img_rect
    h, w = gray.shape[:2]

    def count_near_edges(seeds: np.ndarray, threshold: float = 10.0) -> int:
        """Count seeds at pixels where edge_density > threshold."""
        count = 0
        for x_mm, y_mm in seeds[4:]:  # skip corners
            px = int((x_mm - x1) / (x2 - x1) * (w - 1))
            py = int((y_mm - y1) / (y2 - y1) * (h - 1))
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            if edge_density[py, px] > threshold:
                count += 1
        return count

    near_edge_count_edge = count_near_edges(seeds_edge)
    near_edge_count_uniform = count_near_edges(seeds_uniform)

    # Edge-attracted sampling should produce more points near edges than uniform
    assert near_edge_count_edge > near_edge_count_uniform, (
        f"Expected more points near edges with edge_weight=1 ({near_edge_count_edge}) "
        f"vs edge_weight=0 ({near_edge_count_uniform})"
    )


def test_seeds_ndarray_shape() -> None:
    """Seeds must be returned as a 2D numpy array with shape (N, 2)."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_uniform_gray()
    rng = np.random.default_rng(3)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=100, edge_weight=0.3, rng=rng)

    assert isinstance(seeds, np.ndarray)
    assert seeds.ndim == 2
    assert seeds.shape[1] == 2


# ---------------------------------------------------------------------------
# Tests for generate()
# ---------------------------------------------------------------------------


def test_generate_returns_list() -> None:
    """generate() must return a list (even if empty for the scaffold stage)."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()
    gray = make_gray_with_edges()

    # Wrap in 3-channel RGB as the generator expects _source_image to potentially be RGB
    rgb = np.stack([gray, gray, gray], axis=-1)

    params = {
        "_source_image": rgb,
        "num_points": 100,
        "edge_weight": 0.7,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_radius": 0.0,
        "invert": False,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }

    result = gen.generate(params, canvas)
    assert isinstance(result, list)


def test_generate_no_source_image() -> None:
    """generate() must return [] when _source_image is not provided."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()
    result = gen.generate({}, canvas)
    assert result == []


def test_generator_registered() -> None:
    """TriangulatedHatchGenerator must appear in the GENERATORS registry."""
    from plottter.generators import GENERATORS

    assert "Triangulated Hatching" in GENERATORS
    assert GENERATORS["Triangulated Hatching"] is TriangulatedHatchGenerator


def test_generator_category() -> None:
    """Generator must have category='image'."""
    gen = TriangulatedHatchGenerator()
    assert gen.category == "image"


def test_get_parameters_includes_required() -> None:
    """get_parameters() must include num_points, edge_weight, and standard image params."""
    gen = TriangulatedHatchGenerator()
    param_names = {p.name for p in gen.get_parameters()}

    assert "num_points" in param_names
    assert "edge_weight" in param_names
    assert "brightness" in param_names
    assert "contrast" in param_names
    assert "blur_radius" in param_names
    assert "invert" in param_names
    assert "x_offset_mm" in param_names
    assert "y_offset_mm" in param_names


# ---------------------------------------------------------------------------
# Tests for _triangulate_and_sample
# ---------------------------------------------------------------------------


def test_triangulation_covers_seed_points() -> None:
    """Triangulation must produce triangles whose vertices come from the seed points."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(5)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=50, edge_weight=0.5, rng=rng)
    triangles = _triangulate_and_sample(seeds, gray, img_rect)

    assert len(triangles) > 0, "Expected at least one triangle"

    # Every vertex must be one of the original seeds (within float tolerance)
    seed_set = {(round(x, 9), round(y, 9)) for x, y in seeds.tolist()}
    for verts_mm, _ in triangles:
        for vx, vy in verts_mm:
            assert (round(vx, 9), round(vy, 9)) in seed_set, (
                f"Triangle vertex ({vx}, {vy}) is not a seed point"
            )


def test_triangulation_brightness_range() -> None:
    """Each triangle must have brightness in [0, 255]."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(6)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=100, edge_weight=0.7, rng=rng)
    triangles = _triangulate_and_sample(seeds, gray, img_rect)

    assert len(triangles) > 0
    for verts_mm, brightness in triangles:
        assert 0.0 <= brightness <= 255.0, f"Brightness {brightness} out of [0, 255]"


def test_triangulation_returns_correct_structure() -> None:
    """Each element must be (list-of-3-pairs, float)."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_uniform_gray()
    rng = np.random.default_rng(7)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=30, edge_weight=0.0, rng=rng)
    triangles = _triangulate_and_sample(seeds, gray, img_rect)

    for verts_mm, brightness in triangles:
        assert len(verts_mm) == 3
        for pt in verts_mm:
            assert len(pt) == 2
        assert isinstance(brightness, float)


def test_triangulation_centroid_within_image() -> None:
    """Centroids of all triangles (before discarding) must map to valid pixel coords."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(8)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=80, edge_weight=0.5, rng=rng)
    # Seeds include corners so all centroids that we later keep are within rect
    triangles = _discard_outside_triangles(
        _triangulate_and_sample(seeds, gray, img_rect), img_rect
    )

    img_x1, img_y1, img_x2, img_y2 = img_rect
    for verts_mm, _ in triangles:
        cx = (verts_mm[0][0] + verts_mm[1][0] + verts_mm[2][0]) / 3.0
        cy = (verts_mm[0][1] + verts_mm[1][1] + verts_mm[2][1]) / 3.0
        assert img_x1 <= cx <= img_x2, f"Centroid x {cx} outside [{img_x1}, {img_x2}]"
        assert img_y1 <= cy <= img_y2, f"Centroid y {cy} outside [{img_y1}, {img_y2}]"


# ---------------------------------------------------------------------------
# Tests for _discard_outside_triangles
# ---------------------------------------------------------------------------


def test_discard_outside_removes_exterior_triangles() -> None:
    """Triangles with centroids outside the rect must be removed."""
    img_rect = (0.0, 0.0, 100.0, 100.0)

    inside = ([( 10.0,  10.0), ( 20.0,  10.0), ( 15.0,  20.0)], 128.0)   # centroid inside
    outside = ([(110.0, 110.0), (120.0, 110.0), (115.0, 120.0)], 50.0)    # centroid outside

    result = _discard_outside_triangles([inside, outside], img_rect)
    assert len(result) == 1
    assert result[0][1] == 128.0


def test_discard_outside_keeps_boundary_triangles() -> None:
    """Triangles whose centroid is exactly on the boundary edge should be kept."""
    img_rect = (0.0, 0.0, 100.0, 100.0)

    # centroid exactly at (0, 0)
    on_boundary = ([(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0)], 200.0)  # cx=0, cy=0

    result = _discard_outside_triangles([on_boundary], img_rect)
    assert len(result) == 1


def test_discard_outside_all_inside() -> None:
    """When all triangles are inside, nothing should be removed."""
    img_rect = (0.0, 0.0, 100.0, 100.0)

    triangles = [
        ([(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)], 100.0),
        ([(50.0, 50.0), (60.0, 50.0), (55.0, 60.0)], 200.0),
    ]

    result = _discard_outside_triangles(triangles, img_rect)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests for hatching behaviour
# ---------------------------------------------------------------------------


def _make_params(extra: dict | None = None) -> dict:
    """Build a minimal params dict for the generator."""
    gray = make_gray_with_edges()
    rgb = np.stack([gray, gray, gray], axis=-1)
    base = {
        "_source_image": rgb,
        "num_points": 80,
        "edge_weight": 0.5,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_radius": 0.0,
        "invert": False,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "min_density": 0.0,
        "max_density": 6.0,
        "angle_mode": "Fixed",
        "fixed_angle_deg": 45.0,
        "cross_hatch": False,
        "cross_hatch_threshold": 0.3,
        "draw_edges": False,
    }
    if extra:
        base.update(extra)
    return base


def test_generate_returns_polylines() -> None:
    """generate() must return a non-empty list of polylines with hatching enabled."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()
    result = gen.generate(_make_params(), canvas)
    assert isinstance(result, list)
    assert len(result) > 0, "Expected hatching polylines in result"
    for poly in result:
        assert len(poly) >= 2, "Each polyline must have at least 2 points"


def test_dark_triangles_more_lines_than_bright() -> None:
    """Dark image areas should produce more hatch lines than bright areas."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    # All-dark image (brightness≈0 → max density)
    dark_img = np.zeros((100, 100), dtype=np.uint8)
    dark_rgb = np.stack([dark_img, dark_img, dark_img], axis=-1)

    # All-bright image (brightness≈255 → min density = 0 → no lines)
    bright_img = np.full((100, 100), 255, dtype=np.uint8)
    bright_rgb = np.stack([bright_img, bright_img, bright_img], axis=-1)

    dark_params = _make_params({"_source_image": dark_rgb, "min_density": 0.0, "max_density": 6.0})
    bright_params = _make_params({"_source_image": bright_rgb, "min_density": 0.0, "max_density": 6.0})

    dark_result = gen.generate(dark_params, canvas)
    bright_result = gen.generate(bright_params, canvas)

    assert len(dark_result) > len(bright_result), (
        f"Expected more lines for dark image ({len(dark_result)}) vs bright ({len(bright_result)})"
    )


def test_min_density_zero_bright_areas_no_lines() -> None:
    """When min_density=0, fully bright triangles should produce no hatch lines."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    # All-white image: brightness = 255, density = 0 + (1-1)*max = 0 → skip
    white_img = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white_img, white_img, white_img], axis=-1)

    result = gen.generate(_make_params({"_source_image": white_rgb, "min_density": 0.0}), canvas)
    assert result == [], f"Expected no lines for all-white image with min_density=0, got {len(result)}"


def test_fixed_angle_mode() -> None:
    """In 'Fixed' mode, the angle_mode param is respected without crashing."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    result = gen.generate(_make_params({"angle_mode": "Fixed", "fixed_angle_deg": 30.0}), canvas)
    assert isinstance(result, list)
    assert len(result) > 0


def test_edge_flow_angle_mode() -> None:
    """'Edge Flow' mode must produce polylines without errors."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    result = gen.generate(_make_params({"angle_mode": "Edge Flow"}), canvas)
    assert isinstance(result, list)
    assert len(result) > 0


def test_gradient_angle_mode() -> None:
    """'Gradient' mode must produce polylines without errors."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    result = gen.generate(_make_params({"angle_mode": "Gradient"}), canvas)
    assert isinstance(result, list)
    assert len(result) > 0


def test_cross_hatch_produces_more_lines() -> None:
    """Cross-hatch enabled should produce at least as many lines as single-hatch."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    # Dark image so cross_hatch threshold is crossed
    dark_img = np.zeros((100, 100), dtype=np.uint8)
    dark_rgb = np.stack([dark_img, dark_img, dark_img], axis=-1)

    single = gen.generate(_make_params({
        "_source_image": dark_rgb,
        "cross_hatch": False,
        "cross_hatch_threshold": 0.5,
    }), canvas)
    crossed = gen.generate(_make_params({
        "_source_image": dark_rgb,
        "cross_hatch": True,
        "cross_hatch_threshold": 0.5,
    }), canvas)

    assert len(crossed) >= len(single), (
        f"Cross-hatch ({len(crossed)}) should produce >= lines vs single ({len(single)})"
    )


def test_cross_hatch_skipped_for_bright_triangles() -> None:
    """Cross-hatch must not be applied to bright triangles (brightness ≥ threshold)."""
    # For a uniform gray image right at the threshold, cross-hatch should not appear.
    # We compare bright (255) single vs cross-hatch — both should produce 0 lines
    # when min_density=0 and cross_hatch_threshold=0.3 (brightness/255=1.0 ≥ 0.3).
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    white_img = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white_img, white_img, white_img], axis=-1)

    result = gen.generate(_make_params({
        "_source_image": white_rgb,
        "min_density": 0.0,
        "cross_hatch": True,
        "cross_hatch_threshold": 0.3,
    }), canvas)
    assert result == [], "Bright-only image with min_density=0 should yield no lines"


def test_compute_angle_map_shape() -> None:
    """_compute_angle_map must return an array matching the input shape."""
    gray = make_gray_with_edges()
    for mode in ("Edge Flow", "Gradient"):
        angle_map = _compute_angle_map(gray, mode)
        assert angle_map.shape == gray.shape, f"Shape mismatch for mode={mode}"


def test_compute_angle_map_edge_flow_vs_gradient_differ() -> None:
    """Edge Flow and Gradient angle maps should not be identical for a non-uniform image."""
    gray = make_gray_with_edges()
    ef = _compute_angle_map(gray, "Edge Flow")
    gr = _compute_angle_map(gray, "Gradient")
    # They differ by 90° everywhere, so they should not be equal
    assert not np.allclose(ef, gr), "Edge Flow and Gradient angle maps must differ"


def test_get_parameters_includes_hatching_params() -> None:
    """get_parameters() must include all new hatching parameters."""
    gen = TriangulatedHatchGenerator()
    param_names = {p.name for p in gen.get_parameters()}

    for required in (
        "min_density",
        "max_density",
        "angle_mode",
        "fixed_angle_deg",
        "cross_hatch",
        "cross_hatch_threshold",
        "draw_edges",
    ):
        assert required in param_names, f"Missing parameter: {required}"


# ---------------------------------------------------------------------------
# Tests for draw_edges, coordinate offset
# ---------------------------------------------------------------------------


def test_draw_edges_produces_more_polylines() -> None:
    """draw_edges=True must produce more polylines than draw_edges=False."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    without_edges = gen.generate(_make_params({"draw_edges": False}), canvas)
    with_edges = gen.generate(_make_params({"draw_edges": True}), canvas)

    assert len(with_edges) > len(without_edges), (
        f"Expected more polylines with edges ({len(with_edges)}) vs without ({len(without_edges)})"
    )


def test_draw_edges_no_duplicates() -> None:
    """When draw_edges=True, no two edge polylines should be identical (deduplication).

    Uses an all-white source so min_density=0 produces zero hatch lines,
    leaving only the deduplicated edge polylines in the output.
    """
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    white_img = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white_img, white_img, white_img], axis=-1)

    # min_density=0 + all-white → no hatch lines; only draw_edges polylines remain
    result = gen.generate(
        _make_params({"_source_image": white_rgb, "draw_edges": True, "min_density": 0.0}),
        canvas,
    )

    assert len(result) > 0, "Expected edge polylines in result"

    # Each edge polyline must have exactly 2 points
    for poly in result:
        assert len(poly) == 2, f"Edge polyline has {len(poly)} points, expected 2"

    # Deduplicate undirected
    def edge_key(poly: list) -> frozenset:
        r0 = (round(poly[0][0], 4), round(poly[0][1], 4))
        r1 = (round(poly[1][0], 4), round(poly[1][1], 4))
        return frozenset((r0, r1))

    keys = [edge_key(p) for p in result]
    assert len(keys) == len(set(keys)), (
        f"Duplicate edges found: {len(keys)} total, {len(set(keys))} unique"
    )


def test_offset_shifts_all_polylines() -> None:
    """x_offset_mm and y_offset_mm must shift every point in every polyline (hatch and edge)."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    dx, dy = 10.0, 5.0
    base = gen.generate(_make_params({"draw_edges": True, "x_offset_mm": 0.0, "y_offset_mm": 0.0}), canvas)
    shifted = gen.generate(_make_params({"draw_edges": True, "x_offset_mm": dx, "y_offset_mm": dy}), canvas)

    assert len(base) == len(shifted), "Offset must not change number of polylines"

    for poly_b, poly_s in zip(base, shifted):
        assert len(poly_b) == len(poly_s)
        for (bx, by), (sx, sy) in zip(poly_b, poly_s):
            assert abs(sx - bx - dx) < 1e-9, f"X shift mismatch: {sx} - {bx} != {dx}"
            assert abs(sy - by - dy) < 1e-9, f"Y shift mismatch: {sy} - {by} != {dy}"


def test_draw_edges_offset_applied() -> None:
    """draw_edges polylines must also be shifted by x/y offset."""
    canvas = make_canvas()
    gen = TriangulatedHatchGenerator()

    dx, dy = 7.5, -3.0
    base = gen.generate(_make_params({"draw_edges": True, "x_offset_mm": 0.0, "y_offset_mm": 0.0}), canvas)
    shifted = gen.generate(_make_params({"draw_edges": True, "x_offset_mm": dx, "y_offset_mm": dy}), canvas)

    assert len(base) == len(shifted)
    for poly_b, poly_s in zip(base, shifted):
        for (bx, by), (sx, sy) in zip(poly_b, poly_s):
            assert abs(sx - bx - dx) < 1e-9
            assert abs(sy - by - dy) < 1e-9
