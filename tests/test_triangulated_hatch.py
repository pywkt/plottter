"""Tests for MosaicHatchGenerator — Delaunay-triangulated and Voronoi-tessellated hatching."""

from __future__ import annotations

import numpy as np

from plottter.generators.triangulated_hatch import (
    MosaicHatchGenerator,
    _compute_angle_map,
    _discard_outside_triangles,
    _edge_aware_seeds,
    _hexagon_cells,
    _quadtree_cells,
    _rectangle_cells,
    _triangulate_and_sample,
    _voronoi_cells,
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
    gen = MosaicHatchGenerator()
    gray = make_gray_with_edges()

    # Wrap in 3-channel RGB as the generator expects _source_image to potentially be RGB
    rgb = np.stack([gray, gray, gray], axis=-1)

    params = {
        "_source_image": rgb,
        "mesh_type": "Triangles",
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
    gen = MosaicHatchGenerator()
    result = gen.generate({}, canvas)
    assert result == []


def test_generator_registered() -> None:
    """MosaicHatchGenerator must appear in the GENERATORS registry under 'Mosaic Hatching'."""
    from plottter.generators import GENERATORS

    assert "Mosaic Hatching" in GENERATORS
    assert GENERATORS["Mosaic Hatching"] is MosaicHatchGenerator


def test_old_name_not_registered() -> None:
    """'Triangulated Hatching' must no longer be in the GENERATORS registry."""
    from plottter.generators import GENERATORS

    assert "Triangulated Hatching" not in GENERATORS


def test_generator_category() -> None:
    """Generator must have category='image'."""
    gen = MosaicHatchGenerator()
    assert gen.category == "image"


def test_get_parameters_includes_required() -> None:
    """get_parameters() must include mesh_type, num_points, edge_weight, and standard image params."""
    gen = MosaicHatchGenerator()
    param_names = {p.name for p in gen.get_parameters()}

    assert "mesh_type" in param_names
    assert "num_points" in param_names
    assert "edge_weight" in param_names
    assert "brightness" in param_names
    assert "contrast" in param_names
    assert "blur_radius" in param_names
    assert "invert" in param_names
    assert "x_offset_mm" in param_names
    assert "y_offset_mm" in param_names


def test_mesh_type_is_first_parameter() -> None:
    """mesh_type must be the first parameter returned by get_parameters()."""
    gen = MosaicHatchGenerator()
    params = gen.get_parameters()
    assert params[0].name == "mesh_type"


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
# Tests for _voronoi_cells
# ---------------------------------------------------------------------------


def test_voronoi_cells_returns_list() -> None:
    """_voronoi_cells must return a list of (verts_mm, brightness) tuples."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(20)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=100, edge_weight=0.5, rng=rng)
    cells = _voronoi_cells(seeds, gray, img_rect)

    assert isinstance(cells, list)
    assert len(cells) > 0, "Expected at least one Voronoi cell"


def test_voronoi_cells_polygon_structure() -> None:
    """Each Voronoi cell must be (list-of-at-least-3-pairs, float)."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_uniform_gray()
    rng = np.random.default_rng(21)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=50, edge_weight=0.0, rng=rng)
    cells = _voronoi_cells(seeds, gray, img_rect)

    assert len(cells) > 0
    for verts_mm, brightness in cells:
        assert len(verts_mm) >= 3, f"Cell has only {len(verts_mm)} vertices"
        for pt in verts_mm:
            assert len(pt) == 2
        assert isinstance(brightness, float)
        assert 0.0 <= brightness <= 255.0


def test_voronoi_cells_clipped_to_image_rect() -> None:
    """All Voronoi cell vertices must lie within (or on) the image rect."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(22)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=80, edge_weight=0.5, rng=rng)
    cells = _voronoi_cells(seeds, gray, img_rect)

    x1, y1, x2, y2 = img_rect
    tol = 1e-6
    for verts_mm, _ in cells:
        for vx, vy in verts_mm:
            assert x1 - tol <= vx <= x2 + tol, f"Cell vertex x={vx} outside [{x1}, {x2}]"
            assert y1 - tol <= vy <= y2 + tol, f"Cell vertex y={vy} outside [{y1}, {y2}]"


def test_voronoi_cells_brightness_range() -> None:
    """Each Voronoi cell must have brightness in [0, 255]."""
    canvas = make_canvas()
    img_rect = img_rect_for_canvas(canvas)
    gray = make_gray_with_edges()
    rng = np.random.default_rng(23)

    seeds = _edge_aware_seeds(gray, img_rect, num_points=80, edge_weight=0.5, rng=rng)
    cells = _voronoi_cells(seeds, gray, img_rect)

    assert len(cells) > 0
    for _, brightness in cells:
        assert 0.0 <= brightness <= 255.0, f"Brightness {brightness} out of [0, 255]"


# ---------------------------------------------------------------------------
# Tests for hatching behaviour
# ---------------------------------------------------------------------------


def _make_params(extra: dict | None = None) -> dict:
    """Build a minimal params dict for the generator."""
    gray = make_gray_with_edges()
    rgb = np.stack([gray, gray, gray], axis=-1)
    base = {
        "_source_image": rgb,
        "mesh_type": "Triangles",
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
    gen = MosaicHatchGenerator()
    result = gen.generate(_make_params(), canvas)
    assert isinstance(result, list)
    assert len(result) > 0, "Expected hatching polylines in result"
    for poly in result:
        assert len(poly) >= 2, "Each polyline must have at least 2 points"


def test_dark_triangles_more_lines_than_bright() -> None:
    """Dark image areas should produce more hatch lines than bright areas."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

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
    gen = MosaicHatchGenerator()

    # All-white image: brightness = 255, density = 0 + (1-1)*max = 0 → skip
    white_img = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white_img, white_img, white_img], axis=-1)

    result = gen.generate(_make_params({"_source_image": white_rgb, "min_density": 0.0}), canvas)
    assert result == [], f"Expected no lines for all-white image with min_density=0, got {len(result)}"


def test_fixed_angle_mode() -> None:
    """In 'Fixed' mode, the angle_mode param is respected without crashing."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    result = gen.generate(_make_params({"angle_mode": "Fixed", "fixed_angle_deg": 30.0}), canvas)
    assert isinstance(result, list)
    assert len(result) > 0


def test_edge_flow_angle_mode() -> None:
    """'Edge Flow' mode must produce polylines without errors."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    result = gen.generate(_make_params({"angle_mode": "Edge Flow"}), canvas)
    assert isinstance(result, list)
    assert len(result) > 0


def test_gradient_angle_mode() -> None:
    """'Gradient' mode must produce polylines without errors."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    result = gen.generate(_make_params({"angle_mode": "Gradient"}), canvas)
    assert isinstance(result, list)
    assert len(result) > 0


def test_cross_hatch_produces_more_lines() -> None:
    """Cross-hatch enabled should produce at least as many lines as single-hatch."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

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
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

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
    """get_parameters() must include all hatching parameters."""
    gen = MosaicHatchGenerator()
    param_names = {p.name for p in gen.get_parameters()}

    for required in (
        "mesh_type",
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
# Tests for Voronoi mode via generate()
# ---------------------------------------------------------------------------


def test_voronoi_mode_produces_polylines() -> None:
    """Voronoi mesh_type must produce hatching polylines."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    result = gen.generate(_make_params({"mesh_type": "Voronoi", "max_density": 6.0}), canvas)
    assert isinstance(result, list)
    assert len(result) > 0, "Expected polylines from Voronoi mode"
    for poly in result:
        assert len(poly) >= 2


def test_voronoi_mode_bright_no_lines() -> None:
    """Voronoi mode with all-white image and min_density=0 should produce no lines."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    white_img = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white_img, white_img, white_img], axis=-1)

    result = gen.generate(_make_params({
        "_source_image": white_rgb,
        "mesh_type": "Voronoi",
        "min_density": 0.0,
    }), canvas)
    assert result == [], f"Expected no lines for all-white image in Voronoi mode, got {len(result)}"


def test_voronoi_mode_draw_edges() -> None:
    """Voronoi mode with draw_edges=True must produce more polylines than without."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    without = gen.generate(_make_params({"mesh_type": "Voronoi", "draw_edges": False}), canvas)
    with_edges = gen.generate(_make_params({"mesh_type": "Voronoi", "draw_edges": True}), canvas)

    assert len(with_edges) > len(without), (
        f"Expected more polylines with draw_edges ({len(with_edges)}) vs without ({len(without)})"
    )


def test_triangles_and_voronoi_same_input_different_output() -> None:
    """Triangles and Voronoi modes should produce different output for the same image/params."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    tri_result = gen.generate(_make_params({"mesh_type": "Triangles"}), canvas)
    vor_result = gen.generate(_make_params({"mesh_type": "Voronoi"}), canvas)

    # Both should have output
    assert len(tri_result) > 0
    assert len(vor_result) > 0

    # They should not be identical (different tessellations → different lines)
    assert tri_result != vor_result


# ---------------------------------------------------------------------------
# Tests for draw_edges, coordinate offset
# ---------------------------------------------------------------------------


def test_draw_edges_produces_more_polylines() -> None:
    """draw_edges=True must produce more polylines than draw_edges=False."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

    without_edges = gen.generate(_make_params({"draw_edges": False}), canvas)
    with_edges = gen.generate(_make_params({"draw_edges": True}), canvas)

    assert len(with_edges) > len(without_edges), (
        f"Expected more polylines with edges ({len(with_edges)}) vs without ({len(without_edges)})"
    )


def test_draw_edges_no_duplicates() -> None:
    """When draw_edges=True, no two edge polylines should be identical (deduplication)."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()

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
    gen = MosaicHatchGenerator()

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
    gen = MosaicHatchGenerator()

    dx, dy = 7.5, -3.0
    base = gen.generate(_make_params({"draw_edges": True, "x_offset_mm": 0.0, "y_offset_mm": 0.0}), canvas)
    shifted = gen.generate(_make_params({"draw_edges": True, "x_offset_mm": dx, "y_offset_mm": dy}), canvas)

    assert len(base) == len(shifted)
    for poly_b, poly_s in zip(base, shifted):
        for (bx, by), (sx, sy) in zip(poly_b, poly_s):
            assert abs(sx - bx - dx) < 1e-9
            assert abs(sy - by - dy) < 1e-9


# ---------------------------------------------------------------------------
# Tests for presets and fit mode
# ---------------------------------------------------------------------------


def test_all_presets_generate_valid_output() -> None:
    """Every preset must produce a non-empty list of valid polylines."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    gray = make_gray_with_edges()
    rgb = np.stack([gray, gray, gray], axis=-1)

    presets = gen.get_presets()
    assert len(presets) > 0, "Expected at least one preset"

    named = {p.name: p for p in presets}
    required = [
        "Pen & Ink", "Cross-Hatched Portrait", "Geometric Mesh", "Dense Illustration",
        "Minimal Sketch", "Voronoi Portrait", "Geometric Grid", "Honeycomb",
    ]
    for name in required:
        assert name in named, f"Missing required preset: {name!r}"

    for preset in presets:
        params = dict(preset.params)
        params["_source_image"] = rgb
        # Use small num_points to keep test fast (only relevant for Triangles/Voronoi)
        if "num_points" in params:
            params["num_points"] = min(params["num_points"], 200)
        result = gen.generate(params, canvas)
        assert isinstance(result, list), f"Preset {preset.name!r}: generate() must return list"
        assert len(result) > 0, f"Preset {preset.name!r}: expected non-empty output"
        for poly in result:
            assert len(poly) >= 2, f"Preset {preset.name!r}: polyline has fewer than 2 points"


def test_voronoi_portrait_preset_exists() -> None:
    """'Voronoi Portrait' preset must exist and use Voronoi mesh type."""
    gen = MosaicHatchGenerator()
    presets = {p.name: p for p in gen.get_presets()}

    assert "Voronoi Portrait" in presets
    assert presets["Voronoi Portrait"].params["mesh_type"] == "Voronoi"


def test_presets_include_mesh_type() -> None:
    """All presets must include the mesh_type parameter."""
    gen = MosaicHatchGenerator()
    for preset in gen.get_presets():
        assert "mesh_type" in preset.params, (
            f"Preset {preset.name!r} is missing 'mesh_type' parameter"
        )


def test_fit_mode_output_within_image_rect() -> None:
    """Output polyline points must lie within the image rect for 'fit' mode."""
    from plottter.generators._helpers import compute_image_rect

    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    gray = make_gray_with_edges()
    rgb = np.stack([gray, gray, gray], axis=-1)

    params = _make_params({
        "_source_image": rgb,
        "num_points": 100,
        "draw_edges": True,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "image_fit_mode": "fit",
    })
    result = gen.generate(params, canvas)

    # Compute the expected img_rect that the generator will use
    draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    img_h, img_w = gray.shape[:2]
    x1, y1, x2, y2 = compute_image_rect(
        "fit", img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2
    )
    # For a 100×100 square image on portrait A4, the fit rect is narrower in
    # height than the full drawing area — confirming this test is non-trivial
    assert y2 < draw_y2, "Fit rect should be shorter than drawing area for this image"

    tol = 0.01
    assert len(result) > 0, "Expected non-empty output"
    for poly in result:
        for px, py in poly:
            assert x1 - tol <= px <= x2 + tol, f"Point x={px} outside fit rect [{x1}, {x2}]"
            assert y1 - tol <= py <= y2 + tol, f"Point y={py} outside fit rect [{y1}, {y2}]"


# ---------------------------------------------------------------------------
# Tests for _rectangle_cells
# ---------------------------------------------------------------------------


def test_rectangle_cells_returns_list() -> None:
    """_rectangle_cells must return a list of (verts, brightness) tuples."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_uniform_gray()
    cells = _rectangle_cells(gray, img_rect, cell_size=10.0)
    assert isinstance(cells, list)
    assert len(cells) > 0


def test_rectangle_cells_are_quads() -> None:
    """Each rectangle cell must have exactly 4 vertices."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()
    cells = _rectangle_cells(gray, img_rect, cell_size=10.0)
    assert len(cells) > 0
    for verts, brightness in cells:
        assert len(verts) == 4, f"Rectangle cell has {len(verts)} vertices, expected 4"
        for pt in verts:
            assert len(pt) == 2


def test_rectangle_cells_axis_aligned() -> None:
    """Rectangle cell edges must be axis-aligned (horizontal and vertical)."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_uniform_gray()
    cells = _rectangle_cells(gray, img_rect, cell_size=10.0)
    tol = 1e-9
    for verts, _ in cells:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        # Exactly 2 distinct x values and 2 distinct y values
        unique_xs = {round(x, 9) for x in xs}
        unique_ys = {round(y, 9) for y in ys}
        assert len(unique_xs) == 2, f"Expected 2 unique x coords, got {len(unique_xs)}: {unique_xs}"
        assert len(unique_ys) == 2, f"Expected 2 unique y coords, got {len(unique_ys)}: {unique_ys}"


def test_rectangle_cells_cover_image_rect() -> None:
    """The union of all rectangle cells must cover the entire image rect."""
    img_rect = (0.0, 0.0, 30.0, 30.0)
    gray = make_uniform_gray()
    cells = _rectangle_cells(gray, img_rect, cell_size=10.0)

    # Collect all x and y extents
    x_covered_min = min(min(v[0] for v in verts) for verts, _ in cells)
    y_covered_min = min(min(v[1] for v in verts) for verts, _ in cells)
    x_covered_max = max(max(v[0] for v in verts) for verts, _ in cells)
    y_covered_max = max(max(v[1] for v in verts) for verts, _ in cells)

    x1, y1, x2, y2 = img_rect
    tol = 1e-9
    assert x_covered_min <= x1 + tol
    assert y_covered_min <= y1 + tol
    assert x_covered_max >= x2 - tol
    assert y_covered_max >= y2 - tol


def test_rectangle_cells_brightness_range() -> None:
    """Each rectangle cell must have brightness in [0, 255]."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()
    cells = _rectangle_cells(gray, img_rect, cell_size=8.0)
    for _, brightness in cells:
        assert 0.0 <= brightness <= 255.0


def test_rectangle_cell_size_controls_count() -> None:
    """Larger cell_size should produce fewer cells than smaller cell_size."""
    img_rect = (0.0, 0.0, 100.0, 100.0)
    gray = make_uniform_gray()
    cells_small = _rectangle_cells(gray, img_rect, cell_size=5.0)
    cells_large = _rectangle_cells(gray, img_rect, cell_size=20.0)
    assert len(cells_small) > len(cells_large), (
        f"Smaller cells ({len(cells_small)}) should outnumber larger cells ({len(cells_large)})"
    )


def test_rectangle_cells_dimensions_match_cell_size() -> None:
    """Interior (non-edge) cells must have width and height equal to cell_size."""
    img_rect = (0.0, 0.0, 40.0, 40.0)
    gray = make_uniform_gray()
    cell_size = 10.0
    cells = _rectangle_cells(gray, img_rect, cell_size=cell_size)

    tol = 1e-9
    full_cells_count = 0
    for verts, _ in cells:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if abs(width - cell_size) < tol and abs(height - cell_size) < tol:
            full_cells_count += 1

    # For a 40×40 image with 10mm cells: 4×4 = 16 full cells (all of them)
    assert full_cells_count == 16, f"Expected 16 full cells, got {full_cells_count}"


# ---------------------------------------------------------------------------
# Tests for _hexagon_cells
# ---------------------------------------------------------------------------


def test_hexagon_cells_returns_list() -> None:
    """_hexagon_cells must return a non-empty list."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_uniform_gray()
    cells = _hexagon_cells(gray, img_rect, cell_size=10.0)
    assert isinstance(cells, list)
    assert len(cells) > 0


def test_hexagon_cells_polygon_structure() -> None:
    """Each hexagon cell must have at least 3 vertices and valid brightness."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_uniform_gray()
    cells = _hexagon_cells(gray, img_rect, cell_size=10.0)
    assert len(cells) > 0
    for verts, brightness in cells:
        assert len(verts) >= 3, f"Hexagon cell has only {len(verts)} vertices"
        for pt in verts:
            assert len(pt) == 2
        assert 0.0 <= brightness <= 255.0


def test_hexagon_interior_cells_are_hexagons() -> None:
    """Interior hexagon cells (not clipped by image boundary) must have exactly 6 vertices."""
    # Large image rect so interior hexagons are not clipped
    img_rect = (0.0, 0.0, 200.0, 200.0)
    gray = np.full((100, 100), 128, dtype=np.uint8)
    cells = _hexagon_cells(gray, img_rect, cell_size=10.0)

    six_vertex_count = sum(1 for verts, _ in cells if len(verts) == 6)
    assert six_vertex_count > 0, "Expected some interior (unclipped) hexagons with 6 vertices"
    # Majority of cells should be 6-vertex hexagons
    assert six_vertex_count > len(cells) // 2, (
        f"Expected majority of cells to be hexagons, got {six_vertex_count}/{len(cells)}"
    )


def test_hexagon_cells_clipped_to_image_rect() -> None:
    """All hexagon cell vertices must lie within (or on) the image rect."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()
    cells = _hexagon_cells(gray, img_rect, cell_size=8.0)

    x1, y1, x2, y2 = img_rect
    tol = 1e-6
    for verts, _ in cells:
        for vx, vy in verts:
            assert x1 - tol <= vx <= x2 + tol, f"Hex vertex x={vx} outside [{x1}, {x2}]"
            assert y1 - tol <= vy <= y2 + tol, f"Hex vertex y={vy} outside [{y1}, {y2}]"


def test_hexagon_cell_size_controls_count() -> None:
    """Larger cell_size produces fewer hexagons."""
    img_rect = (0.0, 0.0, 100.0, 100.0)
    gray = make_uniform_gray()
    cells_small = _hexagon_cells(gray, img_rect, cell_size=5.0)
    cells_large = _hexagon_cells(gray, img_rect, cell_size=20.0)
    assert len(cells_small) > len(cells_large), (
        f"Smaller hexagons ({len(cells_small)}) should outnumber larger ({len(cells_large)})"
    )


def test_hexagon_cells_cover_image_rect() -> None:
    """Hexagon cells must collectively cover the entire image rect."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_uniform_gray()
    cells = _hexagon_cells(gray, img_rect, cell_size=8.0)

    # Check: the 4 corners of the image rect are inside at least one cell polygon
    from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPoly

    x1, y1, x2, y2 = img_rect
    test_points = [
        ((x1 + x2) / 2, (y1 + y2) / 2),  # centre
        (x1 + 1.0, y1 + 1.0),              # near top-left
        (x2 - 1.0, y2 - 1.0),              # near bottom-right
    ]

    for tx, ty in test_points:
        pt = ShapelyPoint(tx, ty)
        covered = any(ShapelyPoly(verts).contains(pt) or ShapelyPoly(verts).boundary.contains(pt)
                      for verts, _ in cells)
        assert covered, f"Point ({tx}, {ty}) is not covered by any hexagon cell"


# ---------------------------------------------------------------------------
# Tests for Rectangles and Hexagons via generate()
# ---------------------------------------------------------------------------


def _make_grid_params(mesh_type: str, cell_size: float = 5.0, extra: dict | None = None) -> dict:
    gray = make_gray_with_edges()
    rgb = np.stack([gray, gray, gray], axis=-1)
    params: dict = {
        "_source_image": rgb,
        "mesh_type": mesh_type,
        "cell_size_mm": cell_size,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_radius": 0.0,
        "invert": False,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "min_density": 0.0,
        "max_density": 4.0,
        "angle_mode": "Fixed",
        "fixed_angle_deg": 45.0,
        "cross_hatch": False,
        "cross_hatch_threshold": 0.3,
        "draw_edges": False,
    }
    if extra:
        params.update(extra)
    return params


def test_rectangles_generate_produces_polylines() -> None:
    """Rectangles mode must produce hatching polylines for a dark image."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    dark = np.zeros((100, 100), dtype=np.uint8)
    dark_rgb = np.stack([dark, dark, dark], axis=-1)
    params = _make_grid_params("Rectangles", cell_size=8.0)
    params["_source_image"] = dark_rgb
    params["max_density"] = 4.0

    result = gen.generate(params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0, "Expected hatch lines for dark image with Rectangles"
    for poly in result:
        assert len(poly) >= 2


def test_rectangles_bright_image_no_lines() -> None:
    """Rectangles + white image + min_density=0 must produce no lines."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    white = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white, white, white], axis=-1)

    result = gen.generate(_make_grid_params("Rectangles", extra={"_source_image": white_rgb}), canvas)
    assert result == [], f"Expected no lines for all-white image with Rectangles, got {len(result)}"


def test_rectangles_draw_edges() -> None:
    """Rectangles with draw_edges=True must produce edge polylines (2-point segments)."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    white = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white, white, white], axis=-1)

    result = gen.generate(
        _make_grid_params("Rectangles", extra={"_source_image": white_rgb, "draw_edges": True}),
        canvas,
    )
    assert len(result) > 0, "Expected edge polylines from Rectangles with draw_edges=True"
    for poly in result:
        assert len(poly) == 2, f"Edge polyline has {len(poly)} points, expected 2"


def test_hexagons_generate_produces_polylines() -> None:
    """Hexagons mode must produce hatching polylines for a dark image."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    dark = np.zeros((100, 100), dtype=np.uint8)
    dark_rgb = np.stack([dark, dark, dark], axis=-1)
    params = _make_grid_params("Hexagons", cell_size=10.0)
    params["_source_image"] = dark_rgb
    params["max_density"] = 4.0

    result = gen.generate(params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0, "Expected hatch lines for dark image with Hexagons"
    for poly in result:
        assert len(poly) >= 2


def test_hexagons_bright_image_no_lines() -> None:
    """Hexagons + white image + min_density=0 must produce no lines."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    white = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white, white, white], axis=-1)

    result = gen.generate(_make_grid_params("Hexagons", extra={"_source_image": white_rgb}), canvas)
    assert result == [], f"Expected no lines for all-white image with Hexagons, got {len(result)}"


def test_hexagons_draw_edges() -> None:
    """Hexagons with draw_edges=True must produce edge polylines."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    white = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white, white, white], axis=-1)

    result = gen.generate(
        _make_grid_params("Hexagons", cell_size=10.0, extra={"_source_image": white_rgb, "draw_edges": True}),
        canvas,
    )
    assert len(result) > 0, "Expected edge polylines from Hexagons with draw_edges=True"
    for poly in result:
        assert len(poly) == 2, f"Edge polyline has {len(poly)} points, expected 2"


def test_cell_size_mm_parameter_exists() -> None:
    """get_parameters() must include a cell_size_mm parameter."""
    gen = MosaicHatchGenerator()
    param_names = {p.name for p in gen.get_parameters()}
    assert "cell_size_mm" in param_names


def test_mesh_type_choices_include_rectangles_and_hexagons() -> None:
    """mesh_type ChoiceParam must include 'Rectangles' and 'Hexagons'."""
    gen = MosaicHatchGenerator()
    mesh_param = next(p for p in gen.get_parameters() if p.name == "mesh_type")
    assert hasattr(mesh_param, "choices")
    assert "Rectangles" in mesh_param.choices
    assert "Hexagons" in mesh_param.choices


def test_geometric_grid_preset_exists() -> None:
    """'Geometric Grid' preset must exist and use Rectangles mesh type."""
    gen = MosaicHatchGenerator()
    presets = {p.name: p for p in gen.get_presets()}
    assert "Geometric Grid" in presets
    assert presets["Geometric Grid"].params["mesh_type"] == "Rectangles"
    assert presets["Geometric Grid"].params["draw_edges"] is True


def test_honeycomb_preset_exists() -> None:
    """'Honeycomb' preset must exist and use Hexagons mesh type."""
    gen = MosaicHatchGenerator()
    presets = {p.name: p for p in gen.get_presets()}
    assert "Honeycomb" in presets
    assert presets["Honeycomb"].params["mesh_type"] == "Hexagons"
    assert presets["Honeycomb"].params["draw_edges"] is True


def test_num_points_and_edge_weight_hidden_for_grid_modes() -> None:
    """num_points and edge_weight must have visible_when restricting them to Triangles/Voronoi."""
    gen = MosaicHatchGenerator()
    params_dict = {p.name: p for p in gen.get_parameters()}

    num_p = params_dict["num_points"]
    edge_p = params_dict["edge_weight"]

    assert num_p.visible_when is not None, "num_points must have visible_when set"
    assert edge_p.visible_when is not None, "edge_weight must have visible_when set"

    assert "mesh_type" in num_p.visible_when
    assert "Triangles" in num_p.visible_when["mesh_type"]
    assert "Voronoi" in num_p.visible_when["mesh_type"]
    assert "Rectangles" not in num_p.visible_when["mesh_type"]
    assert "Hexagons" not in num_p.visible_when["mesh_type"]


def test_cell_size_mm_visible_for_grid_modes_only() -> None:
    """cell_size_mm must have visible_when restricting it to Rectangles/Hexagons."""
    gen = MosaicHatchGenerator()
    params_dict = {p.name: p for p in gen.get_parameters()}

    cell_p = params_dict["cell_size_mm"]
    assert cell_p.visible_when is not None, "cell_size_mm must have visible_when set"
    assert "mesh_type" in cell_p.visible_when
    assert "Rectangles" in cell_p.visible_when["mesh_type"]
    assert "Hexagons" in cell_p.visible_when["mesh_type"]
    assert "Triangles" not in cell_p.visible_when["mesh_type"]
    assert "Voronoi" not in cell_p.visible_when["mesh_type"]


# ---------------------------------------------------------------------------
# Tests for _quadtree_cells
# ---------------------------------------------------------------------------


def test_quadtree_cells_returns_list() -> None:
    """_quadtree_cells must return a non-empty list."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()
    cells = _quadtree_cells(gray, img_rect, max_depth=3)
    assert isinstance(cells, list)
    assert len(cells) > 0


def test_quadtree_cells_are_rectangles() -> None:
    """All leaf cells must be valid axis-aligned rectangles (4 vertices, 2 unique x, 2 unique y)."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()
    cells = _quadtree_cells(gray, img_rect, max_depth=4)
    assert len(cells) > 0
    for verts, brightness in cells:
        assert len(verts) == 4, f"Quadtree cell has {len(verts)} vertices, expected 4"
        for pt in verts:
            assert len(pt) == 2
        xs = {round(v[0], 9) for v in verts}
        ys = {round(v[1], 9) for v in verts}
        assert len(xs) == 2, f"Expected 2 unique x coords, got {len(xs)}"
        assert len(ys) == 2, f"Expected 2 unique y coords, got {len(ys)}"
        assert 0.0 <= brightness <= 255.0


def test_quadtree_high_contrast_more_cells_than_uniform() -> None:
    """High-contrast image should produce more cells than uniform gray at same depth."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray_contrast = make_gray_with_edges()
    gray_uniform = make_uniform_gray()

    cells_contrast = _quadtree_cells(gray_contrast, img_rect, max_depth=4)
    cells_uniform = _quadtree_cells(gray_uniform, img_rect, max_depth=4)

    assert len(cells_contrast) > len(cells_uniform), (
        f"High-contrast image ({len(cells_contrast)} cells) should produce more cells "
        f"than uniform ({len(cells_uniform)} cells)"
    )


def test_quadtree_uniform_minimal_subdivision() -> None:
    """Uniform gray image should produce only 1 cell (no subdivision needed)."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_uniform_gray()
    cells = _quadtree_cells(gray, img_rect, max_depth=6)
    # Uniform image has zero contrast — no subdivision at any depth
    assert len(cells) == 1, f"Expected 1 cell for uniform image, got {len(cells)}"


def test_quadtree_depth_limits_cell_count() -> None:
    """Higher max_depth should produce more cells for a high-contrast image."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()

    cells_shallow = _quadtree_cells(gray, img_rect, max_depth=2)
    cells_deep = _quadtree_cells(gray, img_rect, max_depth=5)

    assert len(cells_deep) >= len(cells_shallow), (
        f"Deeper quadtree ({len(cells_deep)}) should have >= cells vs shallow ({len(cells_shallow)})"
    )


def test_quadtree_depth1_max_cells() -> None:
    """At max_depth=1, a cell can subdivide at most once giving at most 4 cells."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()
    cells = _quadtree_cells(gray, img_rect, max_depth=1)
    # With depth=1, root can split into 4 quadrants, none of which can split further
    assert len(cells) <= 4, f"At max_depth=1, expected ≤4 cells, got {len(cells)}"


def test_quadtree_cells_cover_image_rect() -> None:
    """All quadtree cells must be contained within the image rect."""
    img_rect = (5.0, 5.0, 55.0, 55.0)
    gray = make_gray_with_edges()
    cells = _quadtree_cells(gray, img_rect, max_depth=3)

    x1, y1, x2, y2 = img_rect
    tol = 1e-9
    for verts, _ in cells:
        for vx, vy in verts:
            assert x1 - tol <= vx <= x2 + tol, f"Vertex x={vx} outside rect [{x1}, {x2}]"
            assert y1 - tol <= vy <= y2 + tol, f"Vertex y={vy} outside rect [{y1}, {y2}]"


def test_quadtree_brightness_range() -> None:
    """Each quadtree cell brightness must be in [0, 255]."""
    img_rect = (0.0, 0.0, 50.0, 50.0)
    gray = make_gray_with_edges()
    cells = _quadtree_cells(gray, img_rect, max_depth=4)
    for _, brightness in cells:
        assert 0.0 <= brightness <= 255.0, f"Brightness {brightness} out of [0, 255]"


# ---------------------------------------------------------------------------
# Tests for Quadtree mode via generate()
# ---------------------------------------------------------------------------


def _make_quadtree_params(extra: dict | None = None) -> dict:
    """Build a minimal params dict for Quadtree mode."""
    gray = make_gray_with_edges()
    rgb = np.stack([gray, gray, gray], axis=-1)
    base: dict = {
        "_source_image": rgb,
        "mesh_type": "Quadtree",
        "quadtree_depth": 4,
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


def test_quadtree_mode_produces_polylines() -> None:
    """Quadtree mesh_type must produce hatching polylines for a dark image."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    dark = np.zeros((100, 100), dtype=np.uint8)
    dark_rgb = np.stack([dark, dark, dark], axis=-1)
    params = _make_quadtree_params({"_source_image": dark_rgb})
    result = gen.generate(params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0, "Expected hatch lines for dark image with Quadtree"
    for poly in result:
        assert len(poly) >= 2


def test_quadtree_mode_bright_no_lines() -> None:
    """Quadtree mode with all-white image and min_density=0 should produce no lines."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    white = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white, white, white], axis=-1)
    result = gen.generate(_make_quadtree_params({"_source_image": white_rgb, "min_density": 0.0}), canvas)
    assert result == [], f"Expected no lines for all-white image in Quadtree mode, got {len(result)}"


def test_quadtree_draw_edges() -> None:
    """Quadtree with draw_edges=True must produce edge polylines (2-point segments)."""
    canvas = make_canvas()
    gen = MosaicHatchGenerator()
    white = np.full((100, 100), 255, dtype=np.uint8)
    white_rgb = np.stack([white, white, white], axis=-1)
    result = gen.generate(_make_quadtree_params({
        "_source_image": white_rgb,
        "draw_edges": True,
        "min_density": 0.0,
    }), canvas)
    assert len(result) > 0, "Expected edge polylines from Quadtree with draw_edges=True"
    for poly in result:
        assert len(poly) == 2, f"Edge polyline has {len(poly)} points, expected 2"


def test_mesh_type_choices_include_quadtree() -> None:
    """mesh_type ChoiceParam must include 'Quadtree'."""
    gen = MosaicHatchGenerator()
    mesh_param = next(p for p in gen.get_parameters() if p.name == "mesh_type")
    assert hasattr(mesh_param, "choices")
    assert "Quadtree" in mesh_param.choices


def test_quadtree_depth_param_exists_and_visible_when() -> None:
    """quadtree_depth param must exist and be visible only for Quadtree mode."""
    gen = MosaicHatchGenerator()
    params_dict = {p.name: p for p in gen.get_parameters()}
    assert "quadtree_depth" in params_dict, "quadtree_depth parameter must exist"
    qd = params_dict["quadtree_depth"]
    assert qd.visible_when is not None
    assert "mesh_type" in qd.visible_when
    assert "Quadtree" in qd.visible_when["mesh_type"]
    assert "Triangles" not in qd.visible_when["mesh_type"]
    assert "Voronoi" not in qd.visible_when["mesh_type"]


def test_num_points_and_edge_weight_hidden_for_quadtree() -> None:
    """num_points and edge_weight must not be visible when mesh_type='Quadtree'."""
    gen = MosaicHatchGenerator()
    params_dict = {p.name: p for p in gen.get_parameters()}
    assert "Quadtree" not in params_dict["num_points"].visible_when["mesh_type"]
    assert "Quadtree" not in params_dict["edge_weight"].visible_when["mesh_type"]


def test_adaptive_detail_preset_exists() -> None:
    """'Adaptive Detail' preset must exist and use Quadtree mesh type with depth 5."""
    gen = MosaicHatchGenerator()
    presets = {p.name: p for p in gen.get_presets()}
    assert "Adaptive Detail" in presets, "Missing 'Adaptive Detail' preset"
    preset = presets["Adaptive Detail"]
    assert preset.params["mesh_type"] == "Quadtree"
    assert preset.params["quadtree_depth"] == 5
    assert preset.params["max_density"] == 8.0
    assert preset.params["cross_hatch"] is True
    assert preset.params["angle_mode"] == "Edge Flow"
