"""Tests for TriangulatedHatchGenerator — edge-aware seed point placement."""

from __future__ import annotations

import numpy as np

from plottter.generators.triangulated_hatch import (
    TriangulatedHatchGenerator,
    _edge_aware_seeds,
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
