"""Tests for _pointillist_core.py — pure-numeric helpers.

Covers:
- Mitchell sampler: all dots inside disc mask; nearest-neighbour distances
  within 0.5×–2× of expected spacing; determinism for fixed (mask, n, seed).
- render_dots: correct polyline counts and lengths per style.
- image_to_canvas_mm: corner-pixel dots map to expected mm coords.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators._pointillist_core import (
    image_to_canvas_mm,
    mitchell_sample,
    render_dots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disc_mask(H: int = 100, W: int = 100) -> np.ndarray:
    """Binary disc mask centred at (H/2, W/2) with radius ~ min(H, W)/2 - 5."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cy, cx = H / 2.0, W / 2.0
    r = min(H, W) / 2.0 - 5.0
    ys, xs = np.ogrid[:H, :W]
    inside = (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2
    mask[inside] = 255
    return mask


def _nn_distances(pts: np.ndarray) -> np.ndarray:
    """For each point in pts (n,2), return distance to nearest other point."""
    if len(pts) < 2:
        return np.array([])
    dists = []
    for i, p in enumerate(pts):
        others = np.delete(pts, i, axis=0)
        d = np.sqrt(((others - p) ** 2).sum(axis=1))
        dists.append(d.min())
    return np.array(dists)


# ---------------------------------------------------------------------------
# mitchell_sample tests
# ---------------------------------------------------------------------------

class TestMitchellSample:

    def test_all_dots_inside_disc(self):
        H, W = 100, 100
        mask = _disc_mask(H, W)
        cy, cx = H / 2.0, W / 2.0
        r = min(H, W) / 2.0 - 5.0

        pts = mitchell_sample(mask, n=50, seed=42)
        assert pts.shape[1] == 2

        for row_coord, col_coord in pts:
            dist = math.sqrt((row_coord - cy) ** 2 + (col_coord - cx) ** 2)
            assert dist <= r + 1.0, (
                f"Dot ({row_coord},{col_coord}) is outside the disc (dist={dist:.2f}, r={r:.2f})"
            )

    def test_all_dots_are_mask_255(self):
        mask = _disc_mask()
        pts = mitchell_sample(mask, n=40, seed=0)
        for r, c in pts:
            assert mask[int(r), int(c)] == 255, f"Dot at ({r},{c}) is not inside mask"

    def test_returns_correct_count(self):
        mask = _disc_mask()
        pts = mitchell_sample(mask, n=30, seed=7)
        assert pts.shape == (30, 2)

    def test_count_capped_at_white_pixels(self):
        """Requesting more dots than white pixels returns at most #white pixels."""
        tiny = np.zeros((5, 5), dtype=np.uint8)
        tiny[2, 2] = 255  # only 1 white pixel
        pts = mitchell_sample(tiny, n=100, seed=0)
        assert len(pts) == 1

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        pts = mitchell_sample(mask, n=10, seed=0)
        assert pts.shape == (0, 2)

    def test_determinism(self):
        mask = _disc_mask()
        a = mitchell_sample(mask, n=60, seed=123)
        b = mitchell_sample(mask, n=60, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        mask = _disc_mask()
        a = mitchell_sample(mask, n=30, seed=1)
        b = mitchell_sample(mask, n=30, seed=2)
        # Very unlikely to be identical for different seeds.
        assert not np.array_equal(a, b)

    def test_nn_distances_roughly_uniform(self):
        """Nearest-neighbour distances should be within 0.5×–2× of expected."""
        H, W = 200, 200
        mask = np.full((H, W), 255, dtype=np.uint8)  # full coverage
        n = 100
        pts = mitchell_sample(mask, n=n, seed=0).astype(float)

        # Expected spacing ≈ sqrt(area / n)  in pixels.
        expected_spacing = math.sqrt(H * W / n)

        nn = _nn_distances(pts)
        # Allow generous bounds: Mitchell isn't Poisson-disk, just best-candidate.
        assert nn.mean() > expected_spacing * 0.5, (
            f"Mean NN distance {nn.mean():.2f} too small (expected > {expected_spacing * 0.5:.2f})"
        )
        assert nn.mean() < expected_spacing * 2.0, (
            f"Mean NN distance {nn.mean():.2f} too large (expected < {expected_spacing * 2.0:.2f})"
        )

    def test_n_zero_returns_empty(self):
        mask = _disc_mask()
        pts = mitchell_sample(mask, n=0, seed=0)
        assert pts.shape == (0, 2)


# ---------------------------------------------------------------------------
# render_dots tests
# ---------------------------------------------------------------------------

class TestRenderDots:

    def _make_coords(self, n: int = 5) -> np.ndarray:
        coords = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            coords[i] = [float(i) * 2.0, float(i) * 3.0]
        return coords

    # --- point style ---

    def test_point_count(self):
        coords = self._make_coords(7)
        polys = render_dots(coords, style="point", size_mm=0.5)
        assert len(polys) == 7

    def test_point_polyline_length(self):
        coords = self._make_coords(4)
        for poly in render_dots(coords, style="point", size_mm=0.5):
            assert len(poly) == 2, f"Expected len 2 for point, got {len(poly)}"

    def test_point_tiny_displacement(self):
        """The two points should differ only in x by 0.01."""
        coords = np.array([[10.0, 20.0]])
        poly = render_dots(coords, style="point", size_mm=0.5)[0]
        x0, y0 = poly[0]
        x1, y1 = poly[1]
        assert abs(x1 - x0 - 0.01) < 1e-9
        assert abs(y1 - y0) < 1e-9

    # --- cross style ---

    def test_cross_count(self):
        coords = self._make_coords(6)
        polys = render_dots(coords, style="cross", size_mm=1.0)
        assert len(polys) == 12  # 2 per dot

    def test_cross_polyline_length(self):
        coords = self._make_coords(3)
        for poly in render_dots(coords, style="cross", size_mm=1.0):
            assert len(poly) == 2

    def test_cross_arm_length(self):
        """Arms should extend r = size_mm * 0.5 either side of centre."""
        coords = np.array([[5.0, 7.0]])
        size_mm = 2.0
        r = size_mm * 0.5
        polys = render_dots(coords, style="cross", size_mm=size_mm)
        assert len(polys) == 2
        # Horizontal arm
        hx0, hy0 = polys[0][0]
        hx1, hy1 = polys[0][1]
        assert abs(hx0 - (5.0 - r)) < 1e-9
        assert abs(hx1 - (5.0 + r)) < 1e-9
        assert abs(hy0 - 7.0) < 1e-9 and abs(hy1 - 7.0) < 1e-9
        # Vertical arm
        vx0, vy0 = polys[1][0]
        vx1, vy1 = polys[1][1]
        assert abs(vy0 - (7.0 - r)) < 1e-9
        assert abs(vy1 - (7.0 + r)) < 1e-9
        assert abs(vx0 - 5.0) < 1e-9 and abs(vx1 - 5.0) < 1e-9

    # --- circle style ---

    def test_circle_count(self):
        coords = self._make_coords(4)
        polys = render_dots(coords, style="circle", size_mm=1.0)
        assert len(polys) == 4  # 1 per dot

    def test_circle_vertex_count(self):
        """Circle = 12 vertices + closing point = 13 total."""
        coords = self._make_coords(2)
        for poly in render_dots(coords, style="circle", size_mm=1.0):
            assert len(poly) == 13, f"Expected 13 vertices, got {len(poly)}"

    def test_circle_closes(self):
        """First and last vertex should be the same."""
        coords = np.array([[0.0, 0.0]])
        poly = render_dots(coords, style="circle", size_mm=1.0)[0]
        assert abs(poly[0][0] - poly[-1][0]) < 1e-9
        assert abs(poly[0][1] - poly[-1][1]) < 1e-9

    def test_circle_radius(self):
        """All vertices should lie on radius = size_mm * 0.5 from centre."""
        cx, cy = 3.0, 4.0
        coords = np.array([[cx, cy]])
        size_mm = 2.0
        r = size_mm * 0.5
        poly = render_dots(coords, style="circle", size_mm=size_mm)[0]
        for x, y in poly:
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            assert abs(dist - r) < 1e-9, f"Vertex ({x},{y}) not on circle (dist={dist:.6f}, r={r})"

    # --- general contract ---

    def test_all_polylines_len_ge_2(self):
        coords = self._make_coords(10)
        for style in ("point", "cross", "circle"):
            for poly in render_dots(coords, style=style, size_mm=0.5):
                assert len(poly) >= 2, f"Style={style!r}: polyline len {len(poly)} < 2"

    def test_empty_coords_returns_empty(self):
        coords = np.empty((0, 2))
        for style in ("point", "cross", "circle"):
            polys = render_dots(coords, style=style, size_mm=0.5)
            assert polys == [], f"Style={style!r}: expected [] for empty coords"

    def test_unknown_style_raises(self):
        coords = self._make_coords(1)
        with pytest.raises(ValueError, match="Unknown dot style"):
            render_dots(coords, style="zigzag", size_mm=0.5)


# ---------------------------------------------------------------------------
# image_to_canvas_mm tests
# ---------------------------------------------------------------------------

class TestImageToCanvasMm:

    def test_top_left_corner(self):
        """Pixel (0, 0) should map to (left, top) of drawing area."""
        rc = np.array([[0, 0]])
        H, W = 100, 100
        left, top, right, bottom = 10.0, 20.0, 110.0, 120.0
        xy = image_to_canvas_mm(rc, (H, W), (left, top, right, bottom))
        assert xy.shape == (1, 2)
        assert abs(xy[0, 0] - left) < 1e-9
        assert abs(xy[0, 1] - top) < 1e-9

    def test_bottom_right_corner(self):
        """Pixel (H, W) should map exactly to (right, bottom)."""
        H, W = 100, 100
        left, top, right, bottom = 10.0, 20.0, 110.0, 120.0
        rc = np.array([[H, W]])
        xy = image_to_canvas_mm(rc, (H, W), (left, top, right, bottom))
        assert abs(xy[0, 0] - right) < 1e-9
        assert abs(xy[0, 1] - bottom) < 1e-9

    def test_centre_pixel(self):
        """Pixel (H/2, W/2) maps to centre of drawing area."""
        H, W = 100, 100
        left, top, right, bottom = 0.0, 0.0, 100.0, 100.0
        rc = np.array([[H // 2, W // 2]])
        xy = image_to_canvas_mm(rc, (H, W), (left, top, right, bottom))
        assert abs(xy[0, 0] - 50.0) < 1e-9
        assert abs(xy[0, 1] - 50.0) < 1e-9

    def test_known_100x100_drawing_area(self):
        """Corner-pixel checks for a 100×100 mm canvas with no margin."""
        H, W = 200, 200
        drawing_area = (0.0, 0.0, 100.0, 100.0)

        corners = np.array([
            [0,   0],    # top-left
            [0,   W],    # top-right
            [H,   0],    # bottom-left
            [H,   W],    # bottom-right
        ])
        expected = np.array([
            [0.0,   0.0],
            [100.0, 0.0],
            [0.0,   100.0],
            [100.0, 100.0],
        ])
        xy = image_to_canvas_mm(corners, (H, W), drawing_area)
        np.testing.assert_allclose(xy, expected, atol=1e-9)

    def test_empty_input(self):
        rc = np.empty((0, 2), dtype=np.int64)
        xy = image_to_canvas_mm(rc, (100, 100), (0.0, 0.0, 100.0, 100.0))
        assert xy.shape == (0, 2)

    def test_non_square_image(self):
        H, W = 50, 200
        left, top, right, bottom = 5.0, 10.0, 55.0, 60.0  # 50mm wide, 50mm tall
        rc = np.array([[H, W]])  # bottom-right corner
        xy = image_to_canvas_mm(rc, (H, W), (left, top, right, bottom))
        assert abs(xy[0, 0] - right) < 1e-9
        assert abs(xy[0, 1] - bottom) < 1e-9

    def test_output_shape(self):
        rc = np.array([[0, 0], [10, 20], [30, 40]])
        xy = image_to_canvas_mm(rc, (100, 100), (0.0, 0.0, 100.0, 100.0))
        assert xy.shape == (3, 2)
