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
