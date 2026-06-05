"""Tests for generators/contour/_subpixel.py — extract_subpixel_contours."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.contour._subpixel import extract_subpixel_contours


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filled_circle(size: int = 64, radius: float | None = None) -> np.ndarray:
    """Return a uint8 image with a filled white circle on black background."""
    if radius is None:
        radius = size / 4.0
    img = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size / 2.0, size / 2.0
    ys, xs = np.ogrid[:size, :size]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    img[mask] = 255
    return img


def _make_edge_crossing_stripe(size: int = 64) -> np.ndarray:
    """Return a uint8 image with a horizontal white stripe that exits both sides."""
    img = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    img[mid - 4 : mid + 4, :] = 255
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractSubpixelContours:

    def test_filled_circle_one_closed_contour(self):
        """A filled circle produces exactly one closed contour."""
        img = _make_filled_circle(size=64, radius=14.0)
        contours = extract_subpixel_contours(img, level=127.0, min_points=5)
        assert len(contours) == 1, (
            f"Expected 1 contour from filled circle, got {len(contours)}"
        )
        pts, is_closed = contours[0]
        assert is_closed, "Contour of interior circle should be closed"
        assert pts.shape[1] == 2

    def test_edge_crossing_shape_is_open(self):
        """A stripe that exits the image edge yields an is_closed=False contour."""
        img = _make_edge_crossing_stripe(size=64)
        contours = extract_subpixel_contours(img, level=127.0, min_points=3)
        assert contours, "Should find at least one contour"
        # All contours should be open (they exit the image boundary)
        assert all(not is_closed for _, is_closed in contours), (
            "Edge-crossing contours must not be marked closed"
        )

    def test_min_points_filters_short_contours(self):
        """min_points discards contours with fewer vertices."""
        img = _make_filled_circle(size=64, radius=14.0)
        # With a very high min_points threshold nothing should survive
        contours_none = extract_subpixel_contours(img, level=127.0, min_points=100_000)
        assert contours_none == [], (
            "All contours should be filtered when min_points is very high"
        )
        # With min_points=1 at least the circle contour survives
        contours_all = extract_subpixel_contours(img, level=127.0, min_points=1)
        assert len(contours_all) >= 1

    def test_supersample_2_preserves_original_pixel_space(self):
        """With supersample=2 the returned coordinates are in original pixel space.

        The circle's estimated radius from the contour must be close to the
        known pixel radius (within a generous but meaningful tolerance).
        """
        size = 64
        known_radius = 14.0
        img = _make_filled_circle(size=size, radius=known_radius)

        contours_ss1 = extract_subpixel_contours(img, level=127.0, min_points=5, supersample=1)
        contours_ss2 = extract_subpixel_contours(img, level=127.0, min_points=5, supersample=2)

        assert len(contours_ss1) == 1
        assert len(contours_ss2) == 1

        pts_ss1, _ = contours_ss1[0]
        pts_ss2, _ = contours_ss2[0]

        cx, cy = size / 2.0, size / 2.0

        def _mean_radius(pts: np.ndarray) -> float:
            return float(np.mean(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)))

        r1 = _mean_radius(pts_ss1)
        r2 = _mean_radius(pts_ss2)

        # Both should be close to the known pixel radius
        assert abs(r1 - known_radius) < 2.0, (
            f"supersample=1 radius {r1:.2f} too far from expected {known_radius}"
        )
        assert abs(r2 - known_radius) < 2.0, (
            f"supersample=2 radius {r2:.2f} too far from expected {known_radius}"
        )

        # The max coordinate in original pixel space must stay within image bounds
        assert pts_ss2[:, 0].max() < size, "x coords must be in original pixel space"
        assert pts_ss2[:, 1].max() < size, "y coords must be in original pixel space"

    def test_output_dtype_and_shape(self):
        """pts_xy must be a float (N, 2) array."""
        img = _make_filled_circle(size=32, radius=8.0)
        contours = extract_subpixel_contours(img, level=127.0, min_points=3)
        assert contours
        for pts, _ in contours:
            assert pts.ndim == 2
            assert pts.shape[1] == 2
            assert np.issubdtype(pts.dtype, np.floating)

    def test_xy_ordering(self):
        """pts[:, 0] is x (col) and pts[:, 1] is y (row) — not raw (row, col)."""
        # Build an off-centre circle to the right so x centroid > y centroid
        size = 64
        img = np.zeros((size, size), dtype=np.uint8)
        # Circle centred at col=48, row=16 (right side, upper area)
        cx_true, cy_true = 48, 16
        radius = 8
        ys, xs = np.ogrid[:size, :size]
        mask = (xs - cx_true) ** 2 + (ys - cy_true) ** 2 <= radius ** 2
        img[mask] = 255
        contours = extract_subpixel_contours(img, level=127.0, min_points=3)
        assert contours
        pts, _ = contours[0]
        centroid_x = pts[:, 0].mean()
        centroid_y = pts[:, 1].mean()
        # x centroid should be near col 48, y centroid near row 16
        assert centroid_x > centroid_y, (
            f"Expected x > y in (col>row) centroid, got x={centroid_x:.1f} y={centroid_y:.1f}"
        )


# ---------------------------------------------------------------------------
# Helpers for hierarchy tests
# ---------------------------------------------------------------------------

def _square_ring(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Return a closed rectangular ring as (N, 2) float array (x, y)."""
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0],  # closing vertex
        ],
        dtype=float,
    )


def _circle_ring(cx: float, cy: float, r: float, n: int = 64) -> np.ndarray:
    """Return a closed circular ring as (N, 2) float array (x, y)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles)))
    pts = np.vstack([pts, pts[0]])  # close
    return pts


# ---------------------------------------------------------------------------
# Tests for build_contour_hierarchy
# ---------------------------------------------------------------------------

class TestBuildContourHierarchy:

    def test_empty_input(self):
        """Empty input returns empty list."""
        from plottter.generators.contour._subpixel import build_contour_hierarchy

        result = build_contour_hierarchy([])
        assert result == []

    def test_single_ring_no_holes(self):
        """A single ring produces one (outer, []) pair."""
        from plottter.generators.contour._subpixel import build_contour_hierarchy

        outer = _square_ring(0, 0, 10, 10)
        result = build_contour_hierarchy([outer])
        assert len(result) == 1
        outer_ring, holes = result[0]
        assert holes == []
        assert outer_ring.shape[1] == 2

    def test_square_with_hole(self):
        """Outer square + inner hole → one (outer, [hole]) pair."""
        from plottter.generators.contour._subpixel import build_contour_hierarchy

        outer = _square_ring(0, 0, 100, 100)
        hole = _square_ring(20, 20, 80, 80)
        result = build_contour_hierarchy([outer, hole])

        assert len(result) == 1, f"Expected 1 (outer,[hole]) pair, got {len(result)}"
        outer_ring, holes = result[0]
        assert len(holes) == 1, f"Expected 1 hole, got {len(holes)}"

        # The outer ring should have the larger area
        from shapely.geometry import Polygon
        outer_area = Polygon(outer_ring).area
        hole_area = Polygon(holes[0]).area
        assert outer_area > hole_area

    def test_nested_donut_disc_is_own_outer(self):
        """Outer square → hole square → disc: disc is its own even-depth outer.

        Containment forest:
          outer_square (depth 0, outer)
          └── hole_square (depth 1, hole of outer_square)
              └── disc (depth 2, outer — its own pair with no holes)

        Result: [(outer_square, [hole_square]), (disc, [])]
        """
        from plottter.generators.contour._subpixel import build_contour_hierarchy

        outer = _square_ring(0, 0, 100, 100)
        hole = _square_ring(20, 20, 80, 80)
        # disc centred at (50, 50) with radius 10, well inside the hole region
        disc = _circle_ring(50, 50, 10)
        result = build_contour_hierarchy([outer, hole, disc])

        assert len(result) == 2, (
            f"Expected 2 (outer,[holes]) pairs (outer_square and disc), got {len(result)}"
        )

        # Sort by area so we can identify which is which
        from shapely.geometry import Polygon
        by_area = sorted(result, key=lambda t: Polygon(t[0]).area, reverse=True)

        big_outer_ring, big_holes = by_area[0]
        small_outer_ring, small_holes = by_area[1]

        # The large outer should have exactly one hole
        assert len(big_holes) == 1, (
            f"Outer square should have 1 hole, got {len(big_holes)}"
        )
        # The nested disc should have no holes
        assert small_holes == [], (
            f"Nested disc should have no holes, got {small_holes}"
        )

        # The disc area should be close to pi * 10^2
        import math
        disc_area = Polygon(small_outer_ring).area
        expected = math.pi * 10 ** 2
        assert abs(disc_area - expected) / expected < 0.05, (
            f"Disc area {disc_area:.1f} not close to expected {expected:.1f}"
        )

    def test_disjoint_shapes_independent_pairs(self):
        """Two disjoint squares → two independent (outer, []) pairs."""
        from plottter.generators.contour._subpixel import build_contour_hierarchy

        left = _square_ring(0, 0, 20, 20)
        right = _square_ring(40, 0, 60, 20)
        result = build_contour_hierarchy([left, right])

        assert len(result) == 2, f"Expected 2 pairs for disjoint shapes, got {len(result)}"
        for outer_ring, holes in result:
            assert holes == [], f"Disjoint shapes should have no holes, got {holes}"

    def test_input_order_independence(self):
        """Result is the same regardless of input list order (outer and hole)."""
        from plottter.generators.contour._subpixel import build_contour_hierarchy

        outer = _square_ring(0, 0, 100, 100)
        hole = _square_ring(20, 20, 80, 80)

        result_fwd = build_contour_hierarchy([outer, hole])
        result_rev = build_contour_hierarchy([hole, outer])

        assert len(result_fwd) == 1
        assert len(result_rev) == 1
        assert len(result_fwd[0][1]) == 1
        assert len(result_rev[0][1]) == 1

    def test_degenerate_ring_skipped(self):
        """A collinear (zero-area) ring is silently skipped."""
        from plottter.generators.contour._subpixel import build_contour_hierarchy

        good = _square_ring(0, 0, 10, 10)
        degenerate = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=float)
        result = build_contour_hierarchy([good, degenerate])

        # Only the good ring should survive
        assert len(result) == 1
        _, holes = result[0]
        assert holes == []


# ---------------------------------------------------------------------------
# Diagonal smoothness regression
# ---------------------------------------------------------------------------

class TestDiagonalSmoothness:
    """Sub-pixel tracing must produce dramatically fewer staircase corners
    than cv2.findContours on the same rasterized circle image.

    The test uses a Gaussian-blurred circle to represent an anti-aliased source
    (scanned line art, photographed drawings).  Marching squares on the gray
    image interpolates the true smooth boundary; cv2.findContours on the
    binarized image is quantized to pixel-center positions and produces sharp
    direction changes (~45°) at pixel-grid transitions — the staircase artifact.
    """

    @staticmethod
    def _count_staircase_corners(pts: np.ndarray, angle_thresh_deg: float = 20.0) -> int:
        """Count vertices where the incoming/outgoing segment directions differ
        by more than *angle_thresh_deg*.  These are the 'staircase corners' that
        appear in pixel-quantized contours at grid transitions."""
        count = 0
        for i in range(len(pts) - 2):
            ab = pts[i + 1] - pts[i]
            bc = pts[i + 2] - pts[i + 1]
            la = float(np.linalg.norm(ab))
            lb = float(np.linalg.norm(bc))
            if la < 1e-9 or lb < 1e-9:
                continue
            cos_t = float(np.clip(np.dot(ab, bc) / (la * lb), -1.0, 1.0))
            if math.degrees(math.acos(cos_t)) > angle_thresh_deg:
                count += 1
        return count

    def test_circle_subpixel_fewer_staircase_corners_than_findcontours(self):
        """Sub-pixel traced circle on an anti-aliased gray image has dramatically
        fewer staircase corners than cv2.findContours on the binarized image.

        Rasterize a circle with Gaussian blur (representing a real scanned or
        photographed line), then compare:
        - Sub-pixel (marching squares on gray): smooth sub-pixel interpolation.
        - cv2.findContours on binary: quantized pixel-center tracing with
          direction jumps (~45° per pixel-grid step transition) that are the
          mathematical signature of the staircase artifact.

        Count vertices where the incoming and outgoing segment directions differ
        by > 20°.  Sub-pixel must yield dramatically fewer such corners (< 20% of
        cv2's count), proving the staircase is gone for anti-aliased sources.
        """
        from scipy.ndimage import gaussian_filter
        import cv2

        size = 128
        radius = 40.0
        cx, cy = size / 2.0, size / 2.0

        # Rasterize a circle, then blur to create an anti-aliased source image
        # (models a real scanned or photographed line-art stroke)
        img = np.zeros((size, size), dtype=np.uint8)
        ys, xs = np.ogrid[:size, :size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        img[mask] = 255
        gray = gaussian_filter(img.astype(float), sigma=3.0).astype(np.uint8)

        # -- Sub-pixel path: marching squares on the gray image (new behaviour) --
        contours_sp = extract_subpixel_contours(gray, level=127.0, min_points=5)
        assert contours_sp, "Should find at least one contour"
        pts_sp, _ = contours_sp[0]

        # -- cv2 path: binarize first, then findContours (old behaviour) --
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv_contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        assert cv_contours, "cv2 should also find a contour"
        cv_pts = max(cv_contours, key=len).reshape(-1, 2).astype(float)

        corners_sp = self._count_staircase_corners(pts_sp)
        corners_cv = self._count_staircase_corners(cv_pts)

        # Sub-pixel must produce dramatically fewer staircase corners
        assert corners_sp < corners_cv * 0.2, (
            f"Sub-pixel trace has {corners_sp} staircase corners (direction change > 20°); "
            f"cv2 has {corners_cv}. Expected sub-pixel to be dramatically smoother "
            "(< 20% of cv2 corners), proving the staircase is eliminated."
        )


# ---------------------------------------------------------------------------
# Tests for _trace_line_art
# ---------------------------------------------------------------------------

class TestTraceLineArt:
    """Tests for the _trace_line_art function (open/closed behaviour)."""

    # Shared canvas params (1:1 pixel-to-mm mapping for easy assertions)
    SIZE = 64
    DRAW_X1, DRAW_Y1 = 0.0, 0.0
    DRAW_X2, DRAW_Y2 = float(SIZE), float(SIZE)

    def _trace(self, img: np.ndarray, **kwargs) -> list:
        from plottter.generators.contour._line_art import _trace_line_art
        defaults = dict(
            threshold=127,
            img_w=self.SIZE,
            img_h=self.SIZE,
            draw_x1=self.DRAW_X1,
            draw_y1=self.DRAW_Y1,
            draw_x2=self.DRAW_X2,
            draw_y2=self.DRAW_Y2,
            simplify_tol=0.0,
            min_length=3,
            smooth_iterations=0,
        )
        defaults.update(kwargs)
        return _trace_line_art(img, **defaults)

    def test_interior_circle_contour_is_closed(self):
        """A filled circle inside the image produces a closed contour."""
        size = self.SIZE
        img = np.zeros((size, size), dtype=np.uint8)
        cy, cx = size / 2.0, size / 2.0
        ys, xs = np.ogrid[:size, :size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= (size / 4.0) ** 2
        # Dark circle on white background → ink is dark → invert for gray
        # _trace_line_art traces pixels *darker* than threshold as ink.
        # Make the circle pixels dark (0) and background light (255).
        img[:] = 255
        img[mask] = 0

        polylines = self._trace(img)
        assert polylines, "Should produce at least one polyline for interior circle"

        # At least one polyline should be closed (first point == last point)
        closed_count = sum(
            1 for poly in polylines if len(poly) >= 2 and poly[0] == poly[-1]
        )
        assert closed_count >= 1, (
            f"Expected at least one closed polyline for interior circle, "
            f"got 0 closed out of {len(polylines)}"
        )

    def test_border_crossing_stripe_not_force_closed(self):
        """A horizontal stripe crossing the image border yields open (not force-closed)
        polylines — no spurious chord from last to first point."""
        size = self.SIZE
        img = np.zeros((size, size), dtype=np.uint8)
        # White background, dark horizontal stripe across full width (border-touching)
        img[:] = 255
        mid = size // 2
        img[mid - 4 : mid + 4, :] = 0  # dark stripe from col 0 to col size-1

        polylines = self._trace(img)
        assert polylines, "Should produce at least one polyline for border-crossing stripe"

        # None of the polylines should be "force-closed" by a spurious chord.
        # A closed polyline has poly[0] == poly[-1].  For a stripe that exits
        # both sides, the correct result is open polylines.
        # We check that at least one polyline is open (not closed).
        open_count = sum(
            1 for poly in polylines if len(poly) >= 2 and poly[0] != poly[-1]
        )
        assert open_count >= 1, (
            f"Expected at least one open polyline for border-crossing stripe, "
            f"but all {len(polylines)} polylines are closed (possible spurious chord bug)"
        )


# ---------------------------------------------------------------------------
# Tests for fills — spec §7
# ---------------------------------------------------------------------------

class TestFillsStillWork:
    """§7: Concentric and Hatching fills on a simple shape produce non-empty
    output; every fill line lies within the (now smooth) outer ring's bounds.
    """

    SIZE = 64
    DRAW_X1, DRAW_Y1 = 0.0, 0.0
    DRAW_X2, DRAW_Y2 = float(SIZE), float(SIZE)

    @staticmethod
    def _make_circle_image(size: int = 64, radius: float = 16.0) -> np.ndarray:
        """Dark-ink circle on white background."""
        img = np.full((size, size), 255, dtype=np.uint8)
        cy, cx = size / 2.0, size / 2.0
        ys, xs = np.ogrid[:size, :size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        img[mask] = 0
        return img

    def _extract(self, img: np.ndarray, **kwargs):
        from plottter.generators.contour._isolines import _extract_contours_with_hierarchy
        defaults = dict(
            threshold=127,
            img_w=self.SIZE,
            img_h=self.SIZE,
            draw_x1=self.DRAW_X1,
            draw_y1=self.DRAW_Y1,
            draw_x2=self.DRAW_X2,
            draw_y2=self.DRAW_Y2,
            simplify_tol=0.0,
            min_length=3,
            smooth_iterations=0,
        )
        defaults.update(kwargs)
        return _extract_contours_with_hierarchy(img, **defaults)

    def test_extract_produces_pairs(self):
        """_extract_contours_with_hierarchy returns at least one (outer, holes) pair."""
        img = self._make_circle_image()
        pairs = self._extract(img)
        assert pairs, "Should produce at least one (outer, holes) pair for a filled circle"
        outer, holes = pairs[0]
        assert len(outer) >= 3, "Outer ring must have at least 3 vertices"

    def test_border_crossing_region_is_kept_and_fillable(self):
        """A region that runs off the image edge must still be returned as a
        closed, fillable (outer, holes) pair — close_border closes it against
        the canvas boundary instead of dropping it (regression: edge content
        was silently dropped after the cv2->marching-squares rewire)."""
        from plottter.generators.contour._fills import _fill_polygon_concentric

        size = self.SIZE
        img = np.full((size, size), 255, dtype=np.uint8)
        # Dark vertical band touching the top AND bottom edges (crosses border),
        # plus a fully-interior dark square that does NOT touch any edge.
        img[:, size // 2 - 8 : size // 2 + 8] = 0
        img[8:24, 4:20] = 0

        pairs = self._extract(img)
        # Both regions must survive: the interior square AND the edge band.
        assert len(pairs) >= 2, (
            f"Edge-crossing band was dropped: expected >=2 filled regions, got {len(pairs)}"
        )
        # The edge band is the widest region; its concentric fill must be non-empty.
        widest = max(pairs, key=lambda pr: len(pr[0]))
        outer, holes = widest
        assert _fill_polygon_concentric(outer, holes, spacing_mm=2.0), (
            "Edge-crossing region should be fillable once closed against the border"
        )

    def test_hatch_fill_non_empty(self):
        """Hatching fill on a circle produces non-empty output."""
        from plottter.generators.contour._fills import _fill_polygon_hatch

        img = self._make_circle_image()
        pairs = self._extract(img)
        assert pairs, "Need at least one pair to test hatching"

        outer, holes = pairs[0]
        fill_lines = _fill_polygon_hatch(outer, holes, angle_deg=45.0, spacing_mm=2.0)
        assert fill_lines, "Hatching fill should produce non-empty output for a circle"

    def test_hatch_fill_lines_within_bounds(self):
        """All hatch fill lines lie within the outer ring's bounding box."""
        from plottter.generators.contour._fills import _fill_polygon_hatch
        from shapely.geometry import Point, Polygon as ShapelyPolygon

        img = self._make_circle_image()
        pairs = self._extract(img)
        assert pairs

        outer, holes = pairs[0]
        fill_lines = _fill_polygon_hatch(outer, holes, angle_deg=0.0, spacing_mm=2.0)
        assert fill_lines

        # Build the outer polygon in mm space for containment checking
        outer_poly = ShapelyPolygon([(p[0], p[1]) for p in outer])
        expanded = outer_poly.buffer(0.6)  # slight tolerance for clipping artefacts

        for line in fill_lines:
            for pt in line:
                assert expanded.contains(Point(pt[0], pt[1])), (
                    f"Fill point {pt} is outside the (buffered) outer ring"
                )

    def test_concentric_fill_non_empty(self):
        """Concentric fill on a circle produces non-empty output."""
        from plottter.generators.contour._fills import _fill_polygon_concentric

        img = self._make_circle_image()
        pairs = self._extract(img)
        assert pairs, "Need at least one pair to test concentric fill"

        outer, holes = pairs[0]
        fill_lines = _fill_polygon_concentric(outer, holes, spacing_mm=2.0)
        assert fill_lines, "Concentric fill should produce non-empty output for a circle"

    def test_concentric_fill_lines_within_bounds(self):
        """All concentric fill rings lie within the outer ring's bounding box."""
        from plottter.generators.contour._fills import _fill_polygon_concentric
        from shapely.geometry import Point, Polygon as ShapelyPolygon

        img = self._make_circle_image()
        pairs = self._extract(img)
        assert pairs

        outer, holes = pairs[0]
        fill_lines = _fill_polygon_concentric(outer, holes, spacing_mm=2.0)
        assert fill_lines

        outer_poly = ShapelyPolygon([(p[0], p[1]) for p in outer])
        expanded = outer_poly.buffer(0.6)

        for line in fill_lines:
            for pt in line:
                assert expanded.contains(Point(pt[0], pt[1])), (
                    f"Concentric fill point {pt} is outside the (buffered) outer ring"
                )
