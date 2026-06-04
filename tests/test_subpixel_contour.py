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
